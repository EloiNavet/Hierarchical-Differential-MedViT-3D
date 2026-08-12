import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb as w
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import _MAX_UINT32, dir_path, normalize_seed, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class EnsembleMLP(nn.Module):
    """Small MLP that combines ViT + SVM probability vectors.

    Architecture: input_dim → hidden[0] → hidden[1] → num_classes
    with ReLU activations and Dropout between hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            )
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLP ensembles from paired ViT+SVM predictions.",
    )

    parser.add_argument(
        "--dataset-dir",
        type=dir_path,
        required=True,
        help="Directory containing paired_*.csv files from create_vit_svm_mlp_dataset.py.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Output directory for MLP checkpoints.",
    )
    parser.add_argument(
        "--runname",
        type=str,
        default="mlp-ensemble",
        help="Experiment name (used for W&B and folder naming).",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="MLP_Ensemble",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="W&B mode (default: disabled).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        nargs="?",
        help="Train only for a specific fold (default: all folds).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Maximum training epochs (default: 500).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Gaussian noise σ for data augmentation on probabilities (default: 0.05).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience (default: 100).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        choices=["loss", "mcc", "bacc", "macro-f1"],
        help="Metric to monitor for early stopping and best model selection (default: loss).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for internal MLP validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="42",
        help="Random seed (set to 'none' or 'false' to disable).",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropy loss.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="64,32",
        help="Comma-separated hidden layer dimensions (default: '64,32').",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (default: 0.2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size (default: 128).",
    )

    return parser.parse_args()


def load_paired_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load a paired prediction CSV and extract features + labels.

    Returns
    -------
    features : np.ndarray, shape (N, 2 * num_classes)
        Concatenated ViT and SVM probabilities.
    labels : np.ndarray, shape (N,)
        Integer class labels.
    diseases : list[str]
        Disease names (order matches label encoding).
    metadata : dict
        Traceability info from the CSV.
    """
    df = pd.read_csv(csv_path)

    # Discover disease names from column prefixes
    vit_cols = sorted([c for c in df.columns if c.startswith("vit_pred_")])
    svm_cols = sorted([c for c in df.columns if c.startswith("svm_pred_")])

    if not vit_cols or not svm_cols:
        raise ValueError(
            f"No vit_pred_* or svm_pred_* columns found in {csv_path.name}"
        )

    diseases = [c.replace("vit_pred_", "") for c in vit_cols]

    # Verify SVM columns match
    svm_diseases = [c.replace("svm_pred_", "") for c in svm_cols]
    if diseases != svm_diseases:
        raise ValueError(
            f"Disease mismatch between ViT and SVM columns: {diseases} vs {svm_diseases}"
        )

    # Extract features: [vit_probs | svm_probs]
    vit_probs = df[vit_cols].values.astype(np.float32)
    svm_probs = df[svm_cols].values.astype(np.float32)
    features = np.concatenate([vit_probs, svm_probs], axis=1)

    # Extract labels
    labels = df["Diagnosis"].map({d: i for i, d in enumerate(diseases)}).values
    invalid = np.isnan(labels.astype(float))
    if invalid.any():
        unknown = df.loc[invalid, "Diagnosis"].unique().tolist()
        logger.warning(
            f"Dropping {invalid.sum()} samples with unknown diagnoses: {unknown}"
        )
        features = features[~invalid]
        labels = labels[~invalid].astype(int)
    else:
        labels = labels.astype(int)

    # Metadata for traceability
    meta = {
        "vit_checkpoint": df["_vit_checkpoint"].iloc[0]
        if "_vit_checkpoint" in df.columns
        else csv_path.stem,
        "svm_checkpoint": df["_svm_checkpoint"].iloc[0]
        if "_svm_checkpoint" in df.columns
        else "",
        "svm_scaler": df["_svm_scaler"].iloc[0] if "_svm_scaler" in df.columns else "",
        "fold": int(df["_fold"].iloc[0]) if "_fold" in df.columns else -1,
    }

    return features, labels, diseases, meta


def add_probability_noise(
    features: torch.Tensor,
    num_classes: int,
    noise_std: float,
) -> torch.Tensor:
    """Add Gaussian noise to probability sub-vectors in logit space.

    Parameters
    ----------
    features : torch.Tensor, shape (B, 2 * num_classes)
    num_classes : int
    noise_std : float

    Returns
    -------
    torch.Tensor, shape (B, 2 * num_classes)
    """
    if noise_std <= 0:
        return features

    noisy = features.clone()

    # Convert probabilities to logits before adding noise.
    # Adding noise directly to probs and applying softmax would squash
    # the distribution towards uniform (e.g. [1, 0] -> [0.73, 0.27]).

    # Add noise to ViT probabilities
    vit_probs = noisy[:, :num_classes]
    vit_logits = torch.log(vit_probs + 1e-7)
    vit_logits = vit_logits + torch.randn_like(vit_logits) * noise_std
    noisy[:, :num_classes] = F.softmax(vit_logits, dim=1)

    # Add noise to SVM probabilities
    svm_probs = noisy[:, num_classes:]
    svm_logits = torch.log(svm_probs + 1e-7)
    svm_logits = svm_logits + torch.randn_like(svm_logits) * noise_std
    noisy[:, num_classes:] = F.softmax(svm_logits, dim=1)

    return noisy


def train_single_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    diseases: list[str],
    meta: dict,
    args: argparse.Namespace,
    save_dir: Path,
    seed: int | None,
) -> dict:
    """Train a single MLP and save the best checkpoint.

    Returns
    -------
    dict
        Training summary (best val loss, best val accuracy, etc.).
    """
    num_classes = len(diseases)
    input_dim = features.shape[1]
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stratified split
    split_seed = seed if seed is not None else 42
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=args.val_split,
        stratify=labels,
        random_state=split_seed,
    )

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Class weights
    weights = None
    if args.use_class_weights:
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)  # avoid division by zero
        weights = torch.from_numpy(1.0 / class_counts).to(device)
        weights = weights / weights.sum() * num_classes  # normalize

    # Model, optimizer, loss
    model = EnsembleMLP(input_dim, num_classes, hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_bacc = 0.0
    best_val_mcc = -1.0
    best_val_f1 = 0.0
    best_metric_val = float("inf") if args.metric == "loss" else -float("inf")
    best_state = None
    patience_counter = 0

    vit_ckpt_name = meta["vit_checkpoint"]
    # e.g. model_7buw1ylh_7_best0.pt → mlp_model_7buw1ylh_7_best0.pt
    if vit_ckpt_name.endswith(".pt"):
        mlp_name = f"mlp_{vit_ckpt_name}"
    else:
        mlp_name = f"mlp_{vit_ckpt_name}.pt"
    save_path = save_dir / mlp_name

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            # Data augmentation: add noise to probabilities
            batch_x_aug = add_probability_noise(batch_x, num_classes, args.noise_std)

            logits = model(batch_x_aug)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_y.size(0)
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        with torch.inference_mode():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(1)
            val_acc = (val_preds == y_val_t).float().mean().item()

            # Balanced accuracy
            val_preds_np = val_preds.cpu().numpy()
            y_val_np = y_val_t.cpu().numpy()
            per_class_acc = []
            for c in range(num_classes):
                mask = y_val_np == c
                if mask.sum() > 0:
                    per_class_acc.append((val_preds_np[mask] == c).mean())
            val_bacc = np.mean(per_class_acc) if per_class_acc else 0.0

            # MCC and Macro-F1
            val_mcc = matthews_corrcoef(y_val_np, val_preds_np)
            val_f1 = f1_score(y_val_np, val_preds_np, average="macro")

        # W&B logging
        if w.run is not None and w.run.settings.mode != "disabled":
            w.log(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/bacc": val_bacc,
                    "val/mcc": val_mcc,
                    "val/macro-f1": val_f1,
                    "epoch": epoch,
                }
            )

        # Early stopping check
        current_metric_val = {
            "loss": val_loss,
            "mcc": val_mcc,
            "bacc": val_bacc,
            "macro-f1": val_f1,
        }[args.metric]

        is_best = (
            current_metric_val < best_metric_val
            if args.metric == "loss"
            else current_metric_val > best_metric_val
        )

        if is_best:
            best_metric_val = current_metric_val
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_bacc = val_bacc
            best_val_mcc = val_mcc
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    # Save best checkpoint
    checkpoint = {
        "mlp_model": best_state,
        "diseases": diseases,
        "mlp_config": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": list(hidden_dims),
            "dropout": args.dropout,
        },
        "fold": meta["fold"],
        "vit_checkpoint_name": meta["vit_checkpoint"],
        "svm_checkpoint_name": meta["svm_checkpoint"],
        "svm_scaler_name": meta["svm_scaler"],
        "train_metrics": {"loss": train_loss, "acc": train_acc},
        "val_metrics": {
            "loss": best_val_loss,
            "acc": best_val_acc,
            "bacc": best_val_bacc,
            "mcc": best_val_mcc,
            "macro-f1": best_val_f1,
        },
        "best_metric_used": args.metric,
    }
    torch.save(checkpoint, save_path)
    logger.info(
        f"  Saved {save_path.name} "
        f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}, val_bacc={best_val_bacc:.4f}, "
        f"val_mcc={best_val_mcc:.4f}, val_macro-f1={best_val_f1:.4f})"
    )

    return {
        "mlp_path": str(save_path),
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "val_bacc": best_val_bacc,
        "val_mcc": best_val_mcc,
        "val_macro-f1": best_val_f1,
        "epochs_trained": epoch + 1,
    }


def main() -> None:
    args = get_args()

    # Seed
    seed = normalize_seed(args.seed)
    if seed is not None:
        seed = int(seed) % _MAX_UINT32
        seed_everything(seed)

    save_dir = Path(args.save_dir) / args.runname
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover paired CSVs
    dataset_dir = Path(args.dataset_dir)
    csv_files = sorted(dataset_dir.glob("paired_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No paired_*.csv files found in {dataset_dir}")

    # Filter by fold if requested
    if args.fold is not None:
        fold_pattern = re.compile(r"_(\d+)_best")
        csv_files = [
            f
            for f in csv_files
            if (m := fold_pattern.search(f.stem)) and int(m.group(1)) == args.fold
        ]
        if not csv_files:
            raise FileNotFoundError(
                f"No paired CSVs found for fold {args.fold} in {dataset_dir}"
            )

    logger.info(f"Found {len(csv_files)} paired CSVs to process")
    logger.info(f"Saving MLPs to {save_dir}")

    # W&B init (single run for all MLPs)
    w.init(
        project=args.project_name,
        name=args.runname,
        mode=args.wandb_mode,
        config={
            "epochs": args.epochs,
            "lr": args.lr,
            "noise_std": args.noise_std,
            "patience": args.patience,
            "val_split": args.val_split,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "use_class_weights": args.use_class_weights,
            "seed": seed,
            "metric": args.metric,
        },
    )

    results = []
    for csv_path in csv_files:
        logger.info(f"\n--- {csv_path.name} ---")

        features, labels, diseases, meta = load_paired_csv(csv_path)
        logger.info(
            f"  {len(labels)} samples, {len(diseases)} classes, "
            f"input_dim={features.shape[1]}"
        )

        # Show class distribution
        for i, d in enumerate(diseases):
            count = (labels == i).sum()
            logger.info(f"    {d}: {count} ({count / len(labels) * 100:.1f}%)")

        summary = train_single_mlp(
            features,
            labels,
            diseases,
            meta,
            args,
            save_dir,
            seed,
        )
        results.append(summary)

    # Summary
    logger.info("\n===== Training Summary =====")
    metric_key = f"val_{args.metric}"
    val_metrics = [r[metric_key] for r in results]
    logger.info(
        f"Mean {metric_key}: {np.mean(val_metrics):.4f} ± {np.std(val_metrics):.4f}"
    )
    logger.info(
        f"Best {metric_key}: {max(val_metrics) if args.metric != 'loss' else min(val_metrics):.4f}"
    )
    logger.info(
        f"Worst {metric_key}: {min(val_metrics) if args.metric != 'loss' else max(val_metrics):.4f}"
    )

    w.finish()
    logger.info("Done!")


if __name__ == "__main__":
    main()
