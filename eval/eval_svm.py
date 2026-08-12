import argparse
import logging
import os
import pickle
import re
import sys
from glob import glob
from pathlib import Path

import pandas as pd
import wandb as w

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset.preprocessing import DataPrepaSVM, load_svm_features
from utils import compute_bootstrap_metrics, dir_path, file_path, get_train_val_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.CRITICAL)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SVM Model")
    parser.add_argument(
        "--training-csv-dir",
        type=dir_path,
        required=True,
        help="Original training CSV directory (for fold split).",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=dir_path,
        required=True,
        help="Directory containing preprocessed data.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory or glob pattern for model files (e.g., /path/to/models/ or /path/to/svm_*.pkl).",
    )
    parser.add_argument(
        "--eval-csv",
        type=file_path,
        default=None,
        help="Optional CSV for out-of-domain evaluation.",
    )
    parser.add_argument(
        "--project-name", type=str, default="project", help="Wandb project."
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
    )
    return parser.parse_args()


def find_model_pairs(models_dir: str) -> list[tuple[Path, Path, str, int]]:
    """Find matching svm and scaler checkpoint pairs.

    Returns:
        List of tuples: (model_path, scaler_path, run_id, fold)
    """
    # Handle both directory and glob pattern inputs
    models_path = Path(models_dir)
    if models_path.is_dir():
        svm_files = list(models_path.glob("svm_*.pkl"))
    else:
        # It's a glob pattern
        svm_files = [Path(f) for f in glob(models_dir)]
        if not svm_files:
            # Try as directory with wildcard
            svm_files = [Path(f) for f in glob(str(models_path / "svm_*.pkl"))]

    if not svm_files:
        raise FileNotFoundError(
            f"No SVM model files found matching pattern: {models_dir}"
        )

    pairs = []
    for svm_path in svm_files:
        # Extract run_id and fold from filename: svm_{run_id}_{fold}.pkl
        match = re.match(r"svm_([a-z0-9]+)_(\d+)\.pkl", svm_path.name)
        if not match:
            logger.warning(
                f"Skipping file with unexpected name format: {svm_path.name}"
            )
            continue

        run_id = match.group(1)
        fold = int(match.group(2))

        # Find corresponding scaler file
        scaler_path = svm_path.parent / f"scaler_{run_id}_{fold}.pkl"
        if not scaler_path.exists():
            logger.warning(f"Scaler not found for {svm_path.name}: {scaler_path}")
            continue

        pairs.append((svm_path, scaler_path, run_id, fold))

    if not pairs:
        raise FileNotFoundError(f"No matching model/scaler pairs found in {models_dir}")

    # Sort by fold for consistent ordering
    pairs.sort(key=lambda x: x[3])
    return pairs


def log_metrics_table(
    bootstrap_data: dict, split_name: str, diseases: list[str]
) -> dict:
    """Format bootstrap metrics into a table row."""

    def fmt(val: float) -> float:
        return round(val * 100, 2)

    row = {
        "Split": split_name,
        "Accuracy": fmt(bootstrap_data["accuracy"]["mean"]),
        "BACC": fmt(bootstrap_data["balanced_accuracy"]["mean"]),
        "ROC-AUC": fmt(bootstrap_data["roc_auc"]["mean"]),
        "PR-AUC": fmt(bootstrap_data["pr_auc"]["mean"]),
        "MCC": fmt(bootstrap_data["mcc"]["mean"]),
    }

    row["ECE"] = round(bootstrap_data["ece"]["mean"], 4)
    row["MCE"] = round(bootstrap_data["mce"]["mean"], 4)
    row["Brier"] = round(bootstrap_data["brier_score"]["mean"], 4)

    # Per-class F1 scores
    for cls_idx, cls_name in enumerate(diseases):
        f1_stats = bootstrap_data.get("f1", {}).get(cls_idx, {})
        if f1_stats:
            row[f"F1:{cls_name}"] = fmt(f1_stats["mean"])

    return row


def main():
    args = get_args()

    if (
        args.wandb_mode == "online"
        and not os.getenv("WANDB_API_KEY")
        and os.path.exists("wandb.key")
    ):
        with open("wandb.key", "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    model_pairs = find_model_pairs(args.models_dir)
    logger.info(f"Found {len(model_pairs)} models to evaluate")

    _, _, _, metadata_all = get_train_val_test(Path(args.training_csv_dir), 0, kfold=10)
    diseases = sorted(metadata_all.Diagnosis.unique().tolist())

    preprocess_dir = Path(args.intermediate_dir) / "train"
    preprocess_od_dir = Path(args.intermediate_dir) / "testset"

    # Collect all test subjects from all folds
    all_test_subjects = []
    for _, _, _, fold in model_pairs:
        _, _, meta_test_id, _ = get_train_val_test(
            Path(args.training_csv_dir), fold, kfold=10
        )
        all_test_subjects.append(meta_test_id)

    # Concatenate and deduplicate
    all_test_metadata = pd.concat(all_test_subjects, ignore_index=True)
    all_test_metadata = all_test_metadata.drop_duplicates(subset=["Subject"])

    # Preprocess all test data once
    logger.info(f"Preprocessing {len(all_test_metadata)} test subjects (ID)")
    preparer_id = DataPrepaSVM(all_test_metadata, preprocess_dir, device="cpu")
    preparer_id.preprocess_data(n_jobs=-1, verbose=0)

    # Preprocess OD data if provided
    if args.eval_csv:
        meta_od_all = pd.read_csv(args.eval_csv)
        logger.info(f"Preprocessing {len(meta_od_all)} test subjects (OD)")
        preparer_od = DataPrepaSVM(meta_od_all, preprocess_od_dir, device="cpu")
        preparer_od.preprocess_data(n_jobs=-1, verbose=0)

    for model_path, scaler_path, run_id, fold in model_pairs:
        logger.info(f"===== Evaluating: fold {fold} ({run_id}) =====")

        _, _, meta_test_id, _ = get_train_val_test(
            Path(args.training_csv_dir), fold, kfold=10
        )

        # Set wandb dir to the same location as training
        wandb_dir = model_path.parent / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

        # Resume the existing run from training
        w.init(
            project=args.project_name,
            id=run_id,
            resume="allow",
            mode=args.wandb_mode,
            dir=str(wandb_dir),
        )

        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        def run_evaluation(
            metadata: pd.DataFrame,
            data_root: Path,
            split_name: str,
            split_suffix: str,
            *,
            classifier=classifier,
            scaler=scaler,
            model_path=model_path,
            run_id=run_id,
            fold=fold,
        ) -> dict | None:
            logger.info(f"=== {split_name} ===")

            if metadata.empty:
                logger.warning(f"Metadata for {split_name} is empty.")
                return

            # Filter metadata to only include diagnoses present in training
            metadata_filtered = metadata[metadata["Diagnosis"].isin(diseases)].copy()

            if len(metadata_filtered) < len(metadata):
                excluded_diagnoses = set(metadata["Diagnosis"]) - set(diseases)
                logger.warning(
                    f"{split_name}: Excluding {len(metadata) - len(metadata_filtered)} samples with diagnoses not in training set: {excluded_diagnoses}"
                )

            if metadata_filtered.empty:
                logger.warning(f"No valid samples for {split_name} after filtering.")
                return

            X, Y_true = load_svm_features(data_root, metadata_filtered, diseases)

            if len(X) == 0:
                logger.warning(f"No features loaded for {split_name}")
                return

            X = scaler.transform(X)
            y_probs = classifier.predict_proba(X)

            bootstrap_metrics = compute_bootstrap_metrics(Y_true, y_probs)

            def fmt_pct(value: float) -> float:
                return round(value * 100, 2)

            logger.info(
                f"Accuracy: {fmt_pct(bootstrap_metrics['accuracy']['mean'])}±[{fmt_pct(bootstrap_metrics['accuracy']['lower'])}-{fmt_pct(bootstrap_metrics['accuracy']['upper'])}]"
            )
            logger.info(
                f"BACC: {fmt_pct(bootstrap_metrics['balanced_accuracy']['mean'])}±[{fmt_pct(bootstrap_metrics['balanced_accuracy']['lower'])}-{fmt_pct(bootstrap_metrics['balanced_accuracy']['upper'])}]"
            )
            logger.info(
                f"ROC-AUC: {fmt_pct(bootstrap_metrics['roc_auc']['mean'])}±[{fmt_pct(bootstrap_metrics['roc_auc']['lower'])}-{fmt_pct(bootstrap_metrics['roc_auc']['upper'])}]"
            )
            logger.info(
                f"PR-AUC: {fmt_pct(bootstrap_metrics['pr_auc']['mean'])}±[{fmt_pct(bootstrap_metrics['pr_auc']['lower'])}-{fmt_pct(bootstrap_metrics['pr_auc']['upper'])}]"
            )
            logger.info(
                f"MCC: {fmt_pct(bootstrap_metrics['mcc']['mean'])}±[{fmt_pct(bootstrap_metrics['mcc']['lower'])}-{fmt_pct(bootstrap_metrics['mcc']['upper'])}]"
            )
            logger.info(
                f"ECE: {fmt_pct(bootstrap_metrics['ece']['mean'])}±[{fmt_pct(bootstrap_metrics['ece']['lower'])}-{fmt_pct(bootstrap_metrics['ece']['upper'])}]"
            )
            logger.info(
                f"Brier: {fmt_pct(bootstrap_metrics['brier_score']['mean'])}±[{fmt_pct(bootstrap_metrics['brier_score']['lower'])}-{fmt_pct(bootstrap_metrics['brier_score']['upper'])}]"
            )
            logger.info(
                f"F1: {fmt_pct(bootstrap_metrics['macro_f1']['mean'])}±[{fmt_pct(bootstrap_metrics['macro_f1']['lower'])}-{fmt_pct(bootstrap_metrics['macro_f1']['upper'])}]"
            )

            df = metadata_filtered.copy()
            for i, disease in enumerate(diseases):
                df[f"pred_{disease}"] = y_probs[:, i]

            save_csv_path = (
                model_path.parent / f"prediction_svm_{run_id}_{fold}_{split_suffix}.csv"
            )
            df.to_csv(save_csv_path, index=False)

            artifact_name = f"preds_{split_name.replace(' ', '_').replace('(', '').replace(')', '')}_fold{fold}"
            artifact = w.Artifact(artifact_name, type="predictions")
            artifact.add_file(str(save_csv_path))
            w.log_artifact(artifact)

            return log_metrics_table(
                bootstrap_data=bootstrap_metrics,
                split_name=split_name,
                diseases=diseases,
            )

        table_rows = []

        # Evaluate test set (ID)
        row_id = run_evaluation(meta_test_id, preprocess_dir, "Test (ID)", "id")
        if row_id:
            table_rows.append(row_id)

        # Evaluate OD test set if provided
        if args.eval_csv:
            meta_od = pd.read_csv(args.eval_csv)
            row_od = run_evaluation(meta_od, preprocess_od_dir, "Test (OD)", "od")
            if row_od:
                table_rows.append(row_od)

        if table_rows:
            columns = list(table_rows[0].keys())
            data = [list(r.values()) for r in table_rows]
            table = w.Table(columns=columns, data=data)
            w.log({"evaluation_metrics": table})

        w.finish()

    logger.info(f"Completed evaluation of {len(model_pairs)} models")


if __name__ == "__main__":
    main()

# Example usage:
# python eval/eval_svm.py \
#   --training-csv-dir ./path/to/10fold_CV/ \
#   --intermediate-dir ./path/to/preprocessed/ \
#   --models-dir ./path/to/saved_models/svm/my-experiment/ \
#   --eval-csv ./path/to/test_set.csv \
#   --wandb-mode online
