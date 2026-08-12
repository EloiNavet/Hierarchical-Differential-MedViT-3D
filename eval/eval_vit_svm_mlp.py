"""Evaluate trained MLP ensemble checkpoints on test ID/OD splits.

Supports two modes:

1. **End-to-end** inference: loads ViT + SVM + MLP checkpoints, runs all
   three models on preprocessed data, and produces prediction CSVs.
2. **From pre-computed CSVs**: loads existing ViT and SVM prediction CSVs,
   feeds them through the MLP, and produces the combined prediction CSVs.

Output CSVs use the same format as existing evaluation scripts, so they
are directly compatible with ``compute_metrics_plot_violin_csv.py``.

Example (end-to-end)
--------------------
python eval/eval_vit_svm_mlp.py \
    --mode end-to-end \
    --training-csv-dir /data/.../10_fold_CV/ \
    --vit-intermediate-dir /data/.../intermediate/transformer/7classes/ \
    --svm-intermediate-dir /data/.../intermediate/svm/7classes/ \
    --vit-dir /data/.../saved_models/transformers/7classes/runname/ \
    --svm-dir /data/.../saved_models/svm/7classes/runname/ \
    --mlp-checkpoints /data/.../saved_models/mlp_ensemble/runname/mlp_*.pt \
    --eval-csv /data/.../test_OD.csv \
    --output-dir /data/.../results/mlp_ensemble/ \
    --cuda-device 0

Example (from CSVs)
-------------------
python eval/eval_vit_svm_mlp.py \
    --mode from-csvs \
    --vit-predictions-dir /data/.../results/transformer/ \
    --svm-predictions-dir /data/.../results/svm/ \
    --mlp-checkpoints /data/.../saved_models/mlp_ensemble/runname/mlp_*.pt \
    --output-dir /data/.../results/mlp_ensemble/
"""

import argparse
import logging
import pickle
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb as w
import yaml
from torch import nn
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import (
    compute_bootstrap_metrics,
    get_train_val_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP model (must match train/train_vit_svm_mlp.py)
# ---------------------------------------------------------------------------


class EnsembleMLP(nn.Module):
    """Small MLP that combines ViT + SVM probability vectors."""

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


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MLP ensemble checkpoints.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["end-to-end", "from-csvs"],
        required=True,
        help="Evaluation mode: 'end-to-end' (load ViT+SVM+MLP, run inference) "
        "or 'from-csvs' (load pre-computed prediction CSVs).",
    )
    parser.add_argument(
        "--mlp-checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="One or more MLP checkpoint paths (glob supported).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for prediction CSVs.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run evaluation even if output files exist.",
    )

    # --- End-to-end mode arguments ---
    e2e = parser.add_argument_group("End-to-end mode")
    e2e.add_argument(
        "--training-csv-dir",
        type=str,
        default=None,
        help="Directory containing k-fold CSV files (required for end-to-end).",
    )
    e2e.add_argument(
        "--vit-intermediate-dir",
        type=str,
        default=None,
        help="Preprocessed ViT data directory (required for end-to-end).",
    )
    e2e.add_argument(
        "--svm-intermediate-dir",
        type=str,
        default=None,
        help="Preprocessed SVM data directory (required for end-to-end).",
    )
    e2e.add_argument(
        "--vit-dir",
        type=str,
        default=None,
        help="Directory containing ViT .pt checkpoints (required for end-to-end).",
    )
    e2e.add_argument(
        "--svm-dir",
        type=str,
        default=None,
        help="Directory containing SVM .pkl + scaler .pkl (required for end-to-end).",
    )
    e2e.add_argument(
        "--eval-csv",
        type=str,
        default=None,
        help="CSV file for OOD evaluation (optional).",
    )
    e2e.add_argument(
        "--cuda-device",
        type=str,
        default="0",
        help="CUDA device index (default: 0).",
    )
    e2e.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for ViT inference (default: 8).",
    )

    # --- From-CSVs mode arguments ---
    csv_grp = parser.add_argument_group("From-CSVs mode")
    csv_grp.add_argument(
        "--vit-predictions-dir",
        type=str,
        default=None,
        help="Directory with ViT prediction CSVs (required for from-csvs).",
    )
    csv_grp.add_argument(
        "--svm-predictions-dir",
        type=str,
        default=None,
        help="Directory with SVM prediction CSVs (required for from-csvs).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# MLP loading
# ---------------------------------------------------------------------------


def load_mlp(checkpoint_path: Path, device: torch.device) -> tuple[EnsembleMLP, dict]:
    """Load an MLP checkpoint and return the model + metadata."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ckpt["mlp_config"]
    model = EnsembleMLP(
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["mlp_model"])
    model.eval()

    return model, ckpt


def discover_mlp_checkpoints(paths: list[str]) -> list[Path]:
    """Expand glob patterns and return sorted list of MLP checkpoint paths."""
    all_paths: list[Path] = []
    for p in paths:
        expanded = glob(p)
        if expanded:
            all_paths.extend(Path(x) for x in expanded)
        else:
            candidate = Path(p)
            if candidate.exists():
                all_paths.append(candidate)

    if not all_paths:
        raise FileNotFoundError(f"No MLP checkpoints found: {paths}")

    return sorted(all_paths)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def attach_predictions(
    metadata: pd.DataFrame,
    predictions: np.ndarray,
    diseases: list[str],
) -> pd.DataFrame:
    """Append pred_* columns to metadata."""
    df = metadata.reset_index(drop=True).copy()
    for i, disease in enumerate(diseases):
        df[f"pred_{disease}"] = predictions[:, i]
    return df


def log_bootstrap_metrics(
    split_name: str,
    preds: np.ndarray,
    targets: np.ndarray,
    diseases: list[str],
) -> dict | None:
    """Compute and log bootstrap metrics. Returns the bootstrap dict."""
    if preds.size == 0 or targets.size == 0:
        logger.info(f"=== {split_name} === (no data)")
        return None

    labels = targets if targets.ndim == 1 else targets.argmax(axis=1)
    bootstrap = compute_bootstrap_metrics(labels, preds)

    acc = bootstrap["accuracy"]
    bacc = bootstrap["balanced_accuracy"]
    auc = bootstrap["roc_auc"]
    mcc = bootstrap.get("mcc", {})

    logger.info(f"=== {split_name} ===")
    logger.info(
        f"Accuracy: {acc['mean'] * 100:.2f}% [{acc['lower'] * 100:.2f} - {acc['upper'] * 100:.2f}] | "
        f"Balanced Accuracy: {bacc['mean'] * 100:.2f}% [{bacc['lower'] * 100:.2f} - {bacc['upper'] * 100:.2f}]"
    )
    logger.info(
        f"ROC-AUC: {auc['mean'] * 100:.2f}% [{auc['lower'] * 100:.2f} - {auc['upper'] * 100:.2f}]"
    )
    if mcc:
        logger.info(
            f"MCC: {mcc['mean'] * 100:.2f}% [{mcc['lower'] * 100:.2f} - {mcc['upper'] * 100:.2f}]"
        )

    # Per-class F1
    for i, disease in enumerate(diseases):
        f1 = bootstrap.get("f1", {}).get(i, {})
        if f1:
            logger.info(
                f"  F1 {disease}: {f1['mean'] * 100:.2f}% "
                f"[{f1['lower'] * 100:.2f} - {f1['upper'] * 100:.2f}]"
            )

    return bootstrap


def run_mlp_inference(
    model: EnsembleMLP,
    vit_probs: np.ndarray,
    svm_probs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Concatenate ViT+SVM probs, feed through MLP, return softmax probabilities."""
    features = np.concatenate([vit_probs, svm_probs], axis=1).astype(np.float32)
    x = torch.from_numpy(features).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Config loading (same as create_vit_svm_mlp_dataset.py)
# ---------------------------------------------------------------------------


def load_vit_config(model_path: Path) -> dict:
    """Load ViT config from checkpoint directory or W&B run dir."""
    for name in ("config.yaml", "config-defaults.yaml"):
        candidate = model_path.parent / name
        if candidate.exists():
            with open(candidate) as f:
                raw = yaml.safe_load(f)
            return _flatten_config(raw)

    parts = model_path.stem.split("_")
    run_id = None
    if len(parts) >= 2 and re.fullmatch(r"[a-z0-9]{8}", parts[1]):
        run_id = parts[1]

    if run_id is None:
        raise FileNotFoundError(f"Cannot find config for {model_path.name}")

    wandb_parent = model_path.parent / "wandb"
    candidates = list(wandb_parent.glob(f"run-*-{run_id}")) + list(
        wandb_parent.glob(f"offline-run-*-{run_id}")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No W&B directory for run {run_id} under {wandb_parent}"
        )

    wandb_dir = min(candidates, key=lambda p: p.stem)
    for config_name in ("files/config.yaml", "files/config-defaults.yaml"):
        cfg_path = wandb_dir / config_name
        if cfg_path.exists():
            with open(cfg_path) as f:
                raw = yaml.safe_load(f)
            return _flatten_config(raw)

    raise FileNotFoundError(f"No config YAML found in {wandb_dir}")


def _flatten_config(raw: dict) -> dict:
    return {
        k: (v["value"] if isinstance(v, dict) and "value" in v else v)
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------------
# End-to-end mode
# ---------------------------------------------------------------------------


def evaluate_end_to_end(
    args: argparse.Namespace,
    mlp_checkpoints: list[Path],
) -> None:
    """Run full ViT + SVM + MLP inference on test data."""
    from monai.transforms import Compose, NormalizeIntensity, Resize
    from torch.utils.data import DataLoader

    from dataset.dataset import NormalDataset
    from dataset.preprocessing import DataPrepa, DataPrepaSVM, load_svm_features
    from eval.eval_transformer import build_model as build_vit_model
    from utils import ChannelSelectiveTransform, MultiChannelResize

    # Validate required args
    required = [
        "training_csv_dir",
        "vit_intermediate_dir",
        "svm_intermediate_dir",
        "vit_dir",
        "svm_dir",
    ]
    for attr in required:
        if getattr(args, attr, None) is None:
            raise ValueError(
                f"--{attr.replace('_', '-')} is required for end-to-end mode"
            )

    cuda_device = int(args.cuda_device)
    if torch.cuda.is_available() and cuda_device < torch.cuda.device_count():
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vit_preprocess_dir = Path(args.vit_intermediate_dir) / "train"
    vit_preprocess_od_dir = Path(args.vit_intermediate_dir) / "testset"
    svm_preprocess_dir = Path(args.svm_intermediate_dir) / "train"
    svm_preprocess_od_dir = Path(args.svm_intermediate_dir) / "testset"

    # Cache for ViT/SVM inference results to avoid re-running for same model
    vit_cache: dict[str, dict[str, np.ndarray]] = {}  # vit_ckpt_name → {split → probs}
    vit_disease_order: dict[
        str, list[str]
    ] = {}  # vit_ckpt_name → config DISEASES order
    svm_cache: dict[int, dict[str, np.ndarray]] = {}  # fold → {split → probs}

    for mlp_path in mlp_checkpoints:
        mlp_model, ckpt = load_mlp(mlp_path, device)
        diseases = ckpt["diseases"]
        fold = ckpt["fold"]
        vit_ckpt_name = ckpt["vit_checkpoint_name"]
        svm_ckpt_name = ckpt["svm_checkpoint_name"]
        svm_scaler_name = ckpt.get("svm_scaler_name", "")

        # Derive output name: mlp_model_{id}_{fold}_best{n} → prediction_mlp_model_{id}_{fold}_best{n}_{id|od}.csv
        # Strip mlp_ prefix and .pt extension from MLP checkpoint name
        vit_stem = vit_ckpt_name.replace(".pt", "")
        output_base = f"prediction_mlp_{vit_stem}"
        output_id = output_dir / f"{output_base}_id.csv"
        output_od = output_dir / f"{output_base}_od.csv"

        if (
            not args.force_eval
            and output_id.exists()
            and (args.eval_csv is None or output_od.exists())
        ):
            logger.info(
                f"  {output_id.name} already exists, skipping (use --force-eval)"
            )
            continue

        logger.info(f"\n===== {mlp_path.name} (fold {fold}) =====")

        # Get data splits
        config_data = load_vit_config(Path(args.vit_dir) / vit_ckpt_name)
        kfold = config_data.get("KFOLD", 10)
        split = tuple(config_data.get("SPLIT", [7, 2, 1]))

        _, _, meta_test, meta_all = get_train_val_test(
            Path(args.training_csv_dir),
            fold,
            kfold,
            split=split,
        )
        # Filter to known diseases (safety — avoids load_svm_features crash)
        meta_test = meta_test[meta_test["Diagnosis"].isin(diseases)].reset_index(
            drop=True
        )
        logger.info(f"  Test (ID): {len(meta_test)} samples")

        # --- ViT inference ---
        if vit_ckpt_name not in vit_cache:
            vit_ckpt_path = Path(args.vit_dir) / vit_ckpt_name
            if not vit_ckpt_path.exists():
                raise FileNotFoundError(f"ViT checkpoint not found: {vit_ckpt_path}")

            vit_config = load_vit_config(vit_ckpt_path)

            # Ensure preprocessed
            vit_preprocess_dir.mkdir(parents=True, exist_ok=True)
            data_prepa = DataPrepa(
                meta_all,
                has_seg=vit_config.get("IN_CHANNELS", 1) > 1,
                preprocess_data_dir=vit_preprocess_dir,
                device=device,
            )
            data_prepa.preprocess_data(
                crop=tuple(vit_config["IMG_SIZE"]),
                downsample=None,
                tqdm_kwargs={"desc": "ViT preprocess", "dynamic_ncols": True},
            )

            # Build model through wandb config
            run = w.init(mode="disabled", settings=w.Settings(allow_val_change=True))
            run.config.update(vit_config, allow_val_change=True)

            try:
                vit_model = build_vit_model(device)
                checkpoint = torch.load(
                    vit_ckpt_path, map_location=device, weights_only=False
                )
                vit_model.load_state_dict(checkpoint["model"], strict=False)
                vit_model.eval()

                def _run_vit(
                    metadata,
                    preprocess_d,
                    desc,
                    *,
                    vit_config=vit_config,
                    diseases=diseases,
                    vit_model=vit_model,
                ):
                    is_multimodal = vit_config.get("IN_CHANNELS", 1) > 1
                    target_size = tuple(
                        vit_config.get("RESHAPE_SIZE") or vit_config["IMG_SIZE"]
                    )
                    if is_multimodal:
                        transforms = Compose(
                            [
                                MultiChannelResize(target_size),
                                ChannelSelectiveTransform(
                                    NormalizeIntensity(nonzero=True), channels=[0]
                                ),
                            ]
                        )
                    else:
                        transforms = Compose(
                            [
                                Resize(target_size),
                                NormalizeIntensity(nonzero=True, channel_wise=True),
                            ]
                        )

                    dataset = NormalDataset(
                        preprocess_d,
                        metadata.reset_index(drop=True),
                        device="cpu",
                        diseases=diseases,
                        transform=transforms,
                    )
                    loader = DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=int(vit_config.get("NUM_WORKERS", 4)),
                        pin_memory=True,
                    )

                    all_probs = []
                    with torch.inference_mode():
                        for inputs, _ in tqdm(
                            loader,
                            desc=desc,
                            dynamic_ncols=True,
                            bar_format="{l_bar}{bar:20}{r_bar}",
                        ):
                            inputs = inputs.to(device, non_blocking=True)
                            if vit_config.get("ARCHITECTURE", "") != "MedViT":
                                inputs = inputs.to(memory_format=torch.channels_last_3d)
                            logits = vit_model(inputs)
                            probs = torch.softmax(logits, dim=1)
                            all_probs.append(probs.cpu().numpy())
                    return np.concatenate(all_probs, axis=0)

                vit_cache[vit_ckpt_name] = {}
                # Store the ViT's disease order for later reordering
                vit_disease_order[vit_ckpt_name] = vit_config["DISEASES"]

                vit_cache[vit_ckpt_name]["id"] = _run_vit(
                    meta_test,
                    vit_preprocess_dir,
                    f"ViT ID ({vit_ckpt_name})",
                )

                # OOD
                if args.eval_csv:
                    meta_od = pd.read_csv(args.eval_csv)
                    meta_od = meta_od[meta_od["Diagnosis"].isin(diseases)].reset_index(
                        drop=True
                    )
                    vit_preprocess_od_dir.mkdir(parents=True, exist_ok=True)
                    data_prepa_od = DataPrepa(
                        meta_od,
                        has_seg=vit_config.get("IN_CHANNELS", 1) > 1,
                        preprocess_data_dir=vit_preprocess_od_dir,
                        device=device,
                    )
                    data_prepa_od.preprocess_data(
                        crop=tuple(vit_config["IMG_SIZE"]),
                        downsample=None,
                        tqdm_kwargs={
                            "desc": "ViT OD preprocess",
                            "dynamic_ncols": True,
                        },
                    )
                    vit_cache[vit_ckpt_name]["od"] = _run_vit(
                        meta_od,
                        vit_preprocess_od_dir,
                        f"ViT OD ({vit_ckpt_name})",
                    )
            finally:
                w.finish()
                del vit_model
                torch.cuda.empty_cache()

        # Reorder ViT probs from ViT config order to MLP's expected order (alphabetical)
        vit_config_diseases = vit_disease_order[vit_ckpt_name]
        reorder_vit = [vit_config_diseases.index(d) for d in diseases]
        vit_probs_id = vit_cache[vit_ckpt_name]["id"][:, reorder_vit]

        # --- SVM inference ---
        if fold not in svm_cache:
            svm_path = Path(args.svm_dir) / svm_ckpt_name
            scaler_path = Path(args.svm_dir) / svm_scaler_name
            if not svm_path.exists():
                raise FileNotFoundError(f"SVM checkpoint not found: {svm_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"SVM scaler not found: {scaler_path}")

            with open(svm_path, "rb") as f:
                classifier = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # Ensure SVM preprocessed
            svm_preprocess_dir.mkdir(parents=True, exist_ok=True)
            preparer_svm = DataPrepaSVM(meta_all, svm_preprocess_dir, device="cpu")
            preparer_svm.preprocess_data(n_jobs=-1, verbose=0)

            # SVM was trained with sorted diseases — use sorted order for features
            # predict_proba returns columns in sorted order, which matches
            # the MLP's expected order (alphabetical = sorted)
            svm_diseases_sorted = sorted(diseases)
            svm_cache[fold] = {}

            X_id, _ = load_svm_features(
                svm_preprocess_dir, meta_test, svm_diseases_sorted
            )
            svm_cache[fold]["id"] = classifier.predict_proba(scaler.transform(X_id))

            if args.eval_csv:
                meta_od = pd.read_csv(args.eval_csv)
                meta_od = meta_od[meta_od["Diagnosis"].isin(diseases)].reset_index(
                    drop=True
                )
                svm_preprocess_od_dir.mkdir(parents=True, exist_ok=True)
                preparer_svm_od = DataPrepaSVM(
                    meta_od, svm_preprocess_od_dir, device="cpu"
                )
                preparer_svm_od.preprocess_data(n_jobs=-1, verbose=0)
                X_od, _ = load_svm_features(
                    svm_preprocess_od_dir, meta_od, svm_diseases_sorted
                )
                svm_cache[fold]["od"] = classifier.predict_proba(scaler.transform(X_od))

        svm_probs_id = svm_cache[fold]["id"]

        # --- MLP inference (ID) ---
        mlp_probs_id = run_mlp_inference(mlp_model, vit_probs_id, svm_probs_id, device)

        # Ground truth labels
        label_map = {d: i for i, d in enumerate(diseases)}
        gt_id = meta_test["Diagnosis"].map(label_map).values

        # Log metrics
        log_bootstrap_metrics("Test (ID)", mlp_probs_id, gt_id, diseases)

        # Save prediction CSV
        id_df = attach_predictions(meta_test, mlp_probs_id, diseases)
        id_df.to_csv(output_id, index=False)
        logger.info(f"  Saved {output_id.name}")

        # --- OOD ---
        if (
            args.eval_csv
            and "od" in vit_cache.get(vit_ckpt_name, {})
            and "od" in svm_cache.get(fold, {})
        ):
            meta_od = pd.read_csv(args.eval_csv)
            meta_od = meta_od[meta_od["Diagnosis"].isin(diseases)].reset_index(
                drop=True
            )
            # Reorder ViT OD probs to MLP's expected order (alphabetical)
            vit_probs_od = vit_cache[vit_ckpt_name]["od"][:, reorder_vit]
            svm_probs_od = svm_cache[fold]["od"]

            mlp_probs_od = run_mlp_inference(
                mlp_model, vit_probs_od, svm_probs_od, device
            )

            gt_od = meta_od["Diagnosis"].map(label_map).values
            log_bootstrap_metrics("Test (OD)", mlp_probs_od, gt_od, diseases)

            od_df = attach_predictions(meta_od, mlp_probs_od, diseases)
            od_df.to_csv(output_od, index=False)
            logger.info(f"  Saved {output_od.name}")

        del mlp_model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# From-CSVs mode
# ---------------------------------------------------------------------------

_VIT_CSV_PATTERN = re.compile(
    r"prediction_model_([a-z0-9]+)_(\d+)_best(\d+)_(id|od)\.csv"
)
_SVM_CSV_PATTERN = re.compile(r"prediction_svm_([a-z0-9]+)_(\d+)_(id|od)\.csv")


def find_matching_csv(
    directory: Path,
    pattern: re.Pattern,
    fold: int,
    split: str,
    extra_match: str | None = None,
) -> Path | None:
    """Find a CSV in directory matching the pattern, fold, and split."""
    for f in directory.iterdir():
        if f.suffix != ".csv":
            continue
        m = pattern.match(f.name)
        if m:
            csv_fold = int(m.group(2))
            csv_split = m.group(m.lastindex)
            if (
                csv_fold == fold
                and csv_split == split
                and (extra_match is None or extra_match in f.name)
            ):
                return f
    return None


def evaluate_from_csvs(
    args: argparse.Namespace,
    mlp_checkpoints: list[Path],
) -> None:
    """Evaluate by loading pre-computed ViT + SVM prediction CSVs."""
    if args.vit_predictions_dir is None or args.svm_predictions_dir is None:
        raise ValueError(
            "--vit-predictions-dir and --svm-predictions-dir are required for from-csvs mode"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vit_dir = Path(args.vit_predictions_dir)
    svm_dir = Path(args.svm_predictions_dir)

    for mlp_path in mlp_checkpoints:
        mlp_model, ckpt = load_mlp(mlp_path, device)
        diseases = ckpt["diseases"]
        fold = ckpt["fold"]
        vit_ckpt_name = ckpt["vit_checkpoint_name"]

        vit_stem = vit_ckpt_name.replace(".pt", "")
        output_base = f"prediction_mlp_{vit_stem}"

        logger.info(f"\n===== {mlp_path.name} (fold {fold}) =====")

        for split_suffix in ("id", "od"):
            output_csv = output_dir / f"{output_base}_{split_suffix}.csv"

            if not args.force_eval and output_csv.exists():
                logger.info(f"  {output_csv.name} already exists, skipping")
                continue

            # Find matching ViT CSV
            vit_csv = None
            for f in vit_dir.iterdir():
                if (
                    f.suffix == ".csv"
                    and vit_stem in f.name
                    and f.name.endswith(f"_{split_suffix}.csv")
                ):
                    vit_csv = f
                    break

            if vit_csv is None:
                # Try with prediction_ prefix
                expected_name = f"prediction_{vit_stem}_{split_suffix}.csv"
                candidate = vit_dir / expected_name
                if candidate.exists():
                    vit_csv = candidate

            if vit_csv is None:
                logger.warning(
                    f"  No ViT CSV found for {vit_stem}_{split_suffix} in {vit_dir}"
                )
                continue

            # Find matching SVM CSV
            svm_csv = find_matching_csv(svm_dir, _SVM_CSV_PATTERN, fold, split_suffix)
            if svm_csv is None:
                logger.warning(
                    f"  No SVM CSV found for fold {fold}_{split_suffix} in {svm_dir}"
                )
                continue

            logger.info(f"  ViT CSV: {vit_csv.name}")
            logger.info(f"  SVM CSV: {svm_csv.name}")

            # Load CSVs
            vit_df = pd.read_csv(vit_csv)
            svm_df = pd.read_csv(svm_csv)

            # Merge on Subject to align predictions
            pred_cols_vit = [f"pred_{d}" for d in diseases]
            pred_cols_svm = [f"pred_{d}" for d in diseases]

            # Verify columns exist
            missing_vit = [c for c in pred_cols_vit if c not in vit_df.columns]
            missing_svm = [c for c in pred_cols_svm if c not in svm_df.columns]
            if missing_vit:
                logger.error(f"  Missing ViT columns: {missing_vit}")
                continue
            if missing_svm:
                logger.error(f"  Missing SVM columns: {missing_svm}")
                continue

            # Merge to align subjects
            svm_rename = {c: f"svm_{c}" for c in pred_cols_svm}
            svm_subset = svm_df[["Subject"] + pred_cols_svm].rename(columns=svm_rename)

            merged = vit_df.merge(svm_subset, on="Subject", how="inner")

            if len(merged) < len(vit_df):
                logger.warning(
                    f"  Merge dropped {len(vit_df) - len(merged)} subjects "
                    f"(ViT: {len(vit_df)}, SVM: {len(svm_df)}, merged: {len(merged)})"
                )

            if merged.empty:
                logger.error("  No matching subjects after merge, skipping")
                continue

            # Extract probability matrices
            vit_probs = merged[pred_cols_vit].values.astype(np.float32)
            svm_probs = merged[[f"svm_{c}" for c in pred_cols_svm]].values.astype(
                np.float32
            )

            # MLP inference
            mlp_probs = run_mlp_inference(mlp_model, vit_probs, svm_probs, device)

            # Metrics
            if "Diagnosis" in merged.columns:
                label_map = {d: i for i, d in enumerate(diseases)}
                gt = merged["Diagnosis"].map(label_map).values
                valid = ~np.isnan(gt.astype(float))
                if valid.all():
                    split_name = "Test (ID)" if split_suffix == "id" else "Test (OD)"
                    log_bootstrap_metrics(
                        split_name, mlp_probs, gt.astype(int), diseases
                    )

            # Save: use original metadata columns (drop SVM merge columns)
            drop_cols = [c for c in merged.columns if c.startswith("svm_pred_")]
            out_df = merged.drop(columns=drop_cols + pred_cols_vit, errors="ignore")
            out_df = attach_predictions(out_df, mlp_probs, diseases)
            out_df.to_csv(output_csv, index=False)
            logger.info(f"  Saved {output_csv.name} ({len(out_df)} rows)")

        del mlp_model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = get_args()

    mlp_checkpoints = discover_mlp_checkpoints(args.mlp_checkpoints)
    logger.info(f"Found {len(mlp_checkpoints)} MLP checkpoints to evaluate")

    if args.mode == "end-to-end":
        evaluate_end_to_end(args, mlp_checkpoints)
    else:
        evaluate_from_csvs(args, mlp_checkpoints)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
