"""Generate paired ViT + SVM prediction datasets for MLP ensemble training.

For each fold × ViT best model, runs ViT and SVM inference on the *validation*
set and saves paired probability vectors as CSVs.  These CSVs are later
consumed by ``train/train_vit_svm_mlp.py``.

Example
-------
python dataset/create_vit_svm_mlp_dataset.py \
    --training-csv-dir ./data/data/.../10_fold_CV/ \
    --vit-intermediate-dir ./data/intermediate/transformer/7classes/ \
    --svm-intermediate-dir ./data/intermediate/svm/7classes/ \
    --vit-checkpoints ./data/saved_models/transformers/.../model_*_best*.pt \
    --svm-dir ./data/saved_models/svm/7classes/ \
    --output-dir ./data/paired_datasets/7classes/ \
    --cuda-device 0
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
import wandb as w
import yaml
from tqdm.auto import tqdm

# Add project root to sys.path and remove script directory to avoid shadowing
# the 'dataset' package with the local 'dataset.py' module.
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)
script_dir = str(Path(__file__).resolve().parent)
if script_dir in sys.path:
    sys.path.remove(script_dir)

from dataset.preprocessing import DataPrepa, DataPrepaSVM, load_svm_features
from utils import dir_path, get_train_val_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired ViT+SVM prediction datasets for MLP training.",
    )

    parser.add_argument(
        "--training-csv-dir",
        type=dir_path,
        required=True,
        help="Directory containing k-fold CSV files (fold_0.csv … fold_9.csv).",
    )
    parser.add_argument(
        "--vit-intermediate-dir",
        type=str,
        required=True,
        help="Directory for preprocessed ViT tensors (3D volumes).",
    )
    parser.add_argument(
        "--svm-intermediate-dir",
        type=str,
        required=True,
        help="Directory for preprocessed SVM features (133-dim region volumes).",
    )
    parser.add_argument(
        "--vit-checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="ViT checkpoint files (glob or list, e.g. /path/model_*_best*.pt).",
    )
    parser.add_argument(
        "--svm-dir",
        type=str,
        required=True,
        help="Directory containing svm_*.pkl and scaler_*.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for paired prediction CSVs.",
    )
    parser.add_argument(
        "--cuda-device",
        type=str,
        default="0",
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for ViT inference (default: 8).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing paired CSVs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery helpers
# ---------------------------------------------------------------------------

_VIT_PATTERN = re.compile(r"model_([a-z0-9]+)_(\d+)_best(\d+)\.pt")


def discover_vit_checkpoints(paths: list[str]) -> dict[int, list[Path]]:
    """Group ViT checkpoint paths by fold.

    Returns
    -------
    dict[int, list[Path]]
        Mapping from fold → sorted list of checkpoint paths.
    """
    # Expand globs
    all_paths: list[Path] = []
    for p in paths:
        expanded = glob(p)
        if expanded:
            all_paths.extend(Path(x) for x in expanded)
        else:
            candidate = Path(p)
            if candidate.exists():
                all_paths.append(candidate)

    by_fold: dict[int, list[Path]] = {}
    for cp in all_paths:
        m = _VIT_PATTERN.match(cp.name)
        if m:
            fold = int(m.group(2))
            by_fold.setdefault(fold, []).append(cp)
        else:
            logger.warning(f"Skipping ViT checkpoint with unexpected name: {cp.name}")

    # Sort each fold's checkpoints by best index
    for fold, checkpoints in by_fold.items():
        checkpoints.sort(key=lambda p: int(_VIT_PATTERN.match(p.name).group(3)))

    return by_fold


_SVM_PATTERN = re.compile(r"svm_([a-z0-9]+)_(\d+)\.pkl")


def discover_svm_models(svm_dir: str) -> dict[int, tuple[Path, Path, str]]:
    """Find SVM model + scaler pairs grouped by fold.

    Returns
    -------
    dict[int, tuple[Path, Path, str]]
        Mapping from fold → (svm_path, scaler_path, run_id).
    """
    svm_path = Path(svm_dir)
    svm_files = list(svm_path.glob("svm_*.pkl"))
    if not svm_files:
        raise FileNotFoundError(f"No SVM model files found in {svm_dir}")

    by_fold: dict[int, tuple[Path, Path, str]] = {}
    for sp in svm_files:
        m = _SVM_PATTERN.match(sp.name)
        if not m:
            continue
        run_id = m.group(1)
        fold = int(m.group(2))
        scaler_path = sp.parent / f"scaler_{run_id}_{fold}.pkl"
        if not scaler_path.exists():
            logger.warning(f"Scaler not found for {sp.name}, skipping")
            continue
        by_fold[fold] = (sp, scaler_path, run_id)

    if not by_fold:
        raise FileNotFoundError(f"No valid SVM model/scaler pairs in {svm_dir}")

    return by_fold


# ---------------------------------------------------------------------------
# Config loading (same logic as eval_transformer.py)
# ---------------------------------------------------------------------------


def load_vit_config(model_path: Path) -> dict:
    """Load ViT config from checkpoint directory or W&B run dir."""
    # 1) Fast path: config alongside checkpoint
    for name in ("config.yaml", "config-defaults.yaml"):
        candidate = model_path.parent / name
        if candidate.exists():
            with open(candidate) as f:
                raw = yaml.safe_load(f)
            return _flatten_config(raw)

    # 2) W&B directory
    parts = model_path.stem.split("_")
    run_id = None
    if len(parts) >= 2 and re.fullmatch(r"[a-z0-9]{8}", parts[1]):
        run_id = parts[1]

    if run_id is None:
        raise FileNotFoundError(
            f"Cannot find config for {model_path.name}: no local config.yaml "
            f"and unable to infer W&B run id."
        )

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
    """Flatten W&B-style config ``{key: {value: x}}`` to ``{key: x}``."""
    return {
        k: (v["value"] if isinstance(v, dict) and "value" in v else v)
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------------
# ViT inference (adapted from eval_transformer.py)
# ---------------------------------------------------------------------------


def run_vit_inference(
    model_path: Path,
    config: dict,
    preprocess_dir: Path,
    metadata: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run ViT inference on *metadata* and return probability matrix (N, C)."""
    from monai.transforms import Compose, NormalizeIntensity, Resize
    from torch.utils.data import DataLoader

    from dataset.dataset import NormalDataset
    from utils import ChannelSelectiveTransform, MultiChannelResize

    # Temporarily inject config into wandb so build_model() works
    run = w.init(mode="disabled", settings=w.Settings(allow_val_change=True))
    run.config.update(config, allow_val_change=True)

    try:
        from eval.eval_transformer import build_model

        model = build_model(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()

        # Build eval transforms
        is_multimodal = config.get("IN_CHANNELS", 1) > 1
        target_size = tuple(config.get("RESHAPE_SIZE") or config["IMG_SIZE"])

        if is_multimodal:
            eval_transforms = Compose(
                [
                    MultiChannelResize(target_size),
                    ChannelSelectiveTransform(
                        NormalizeIntensity(nonzero=True), channels=[0]
                    ),
                ]
            )
        else:
            eval_transforms = Compose(
                [
                    Resize(target_size),
                    NormalizeIntensity(nonzero=True, channel_wise=True),
                ]
            )

        dataset = NormalDataset(
            preprocess_dir,
            metadata.reset_index(drop=True),
            device="cpu",
            diseases=config["DISEASES"],
            transform=eval_transforms,
        )

        num_workers = int(config.get("NUM_WORKERS", 4))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        all_probs = []
        with torch.inference_mode():
            for inputs, _ in tqdm(
                loader,
                desc=f"ViT inference ({model_path.stem})",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:20}{r_bar}",
            ):
                inputs = inputs.to(device, non_blocking=True)
                if config.get("ARCHITECTURE", "") != "MedViT":
                    inputs = inputs.to(memory_format=torch.channels_last_3d)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)
    finally:
        w.finish()


# ---------------------------------------------------------------------------
# SVM inference
# ---------------------------------------------------------------------------


def run_svm_inference(
    svm_path: Path,
    scaler_path: Path,
    preprocess_dir: Path,
    metadata: pd.DataFrame,
    diseases: list[str],
) -> np.ndarray:
    """Run SVM inference and return probability matrix (N, C).

    The SVM was trained with alphabetically sorted disease labels, so
    ``predict_proba`` returns columns in sorted order.  This function
    reorders the output to match the caller's ``diseases`` order.
    """
    with open(svm_path, "rb") as f:
        classifier = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # SVM was trained with sorted diseases — use the same order for features
    svm_diseases = sorted(diseases)
    X, _ = load_svm_features(preprocess_dir, metadata, svm_diseases)
    X = scaler.transform(X)
    probs = classifier.predict_proba(X)  # columns in svm_diseases (sorted) order

    # Validate class count
    if probs.shape[1] != len(diseases):
        raise ValueError(
            f"SVM predicts {probs.shape[1]} classes but expected {len(diseases)}. "
            f"SVM classes: {list(classifier.classes_)}, diseases: {diseases}"
        )

    # Reorder columns from sorted order to the caller's disease order
    reorder_idx = [svm_diseases.index(d) for d in diseases]
    return probs[:, reorder_idx]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    args = get_args()

    cuda_device = int(args.cuda_device)
    if torch.cuda.is_available() and cuda_device < torch.cuda.device_count():
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover checkpoints
    vit_by_fold = discover_vit_checkpoints(args.vit_checkpoints)
    svm_by_fold = discover_svm_models(args.svm_dir)

    logger.info(
        f"Found ViT checkpoints for folds {sorted(vit_by_fold.keys())} "
        f"({sum(len(v) for v in vit_by_fold.values())} total)"
    )
    logger.info(f"Found SVM models for folds {sorted(svm_by_fold.keys())}")

    folds = sorted(set(vit_by_fold.keys()) & set(svm_by_fold.keys()))
    if not folds:
        raise RuntimeError("No overlapping folds between ViT and SVM checkpoints")
    logger.info(f"Processing folds: {folds}")

    # Load config from the first ViT checkpoint to get diseases and split info
    first_vit = next(iter(vit_by_fold.values()))[0]
    config = load_vit_config(first_vit)
    diseases = config["DISEASES"]
    kfold = config.get("KFOLD", 10)
    split = tuple(config.get("SPLIT", [7, 2, 1]))

    logger.info(f"Diseases: {diseases}")
    logger.info(f"k-fold: {kfold}, split: {split}")

    vit_preprocess_dir = Path(args.vit_intermediate_dir) / "train"
    svm_preprocess_dir = Path(args.svm_intermediate_dir) / "train"

    for fold in folds:
        logger.info(f"\n===== Fold {fold} =====")

        # Get validation metadata
        _, metadata_val, _, metadata_all = get_train_val_test(
            Path(args.training_csv_dir),
            fold,
            kfold,
            split=split,
        )
        logger.info(f"Validation set: {len(metadata_val)} samples")

        # Ensure preprocessed data exists
        vit_preprocess_dir.mkdir(parents=True, exist_ok=True)
        svm_preprocess_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess for ViT (idempotent — skips existing)
        data_prepa = DataPrepa(
            metadata_all,
            has_seg=config.get("IN_CHANNELS", 1) > 1,
            preprocess_data_dir=vit_preprocess_dir,
            device=device,
        )
        data_prepa.preprocess_data(
            crop=tuple(config["IMG_SIZE"]),
            downsample=None,
            tqdm_kwargs={
                "desc": f"ViT preprocess (fold {fold})",
                "dynamic_ncols": True,
            },
        )

        # Preprocess for SVM (idempotent)
        preparer_svm = DataPrepaSVM(metadata_all, svm_preprocess_dir, device="cpu")
        preparer_svm.preprocess_data(n_jobs=-1, verbose=0)

        # Filter validation metadata to known diseases (avoid crash in load_svm_features)
        metadata_val_filtered = metadata_val[
            metadata_val["Diagnosis"].isin(diseases)
        ].reset_index(drop=True)
        if len(metadata_val_filtered) < len(metadata_val):
            excluded = set(metadata_val["Diagnosis"]) - set(diseases)
            logger.warning(
                f"Excluding {len(metadata_val) - len(metadata_val_filtered)} "
                f"samples with unknown diagnoses: {excluded}"
            )

        # SVM inference on validation set (same for all ViT best models in this fold)
        svm_path, scaler_path, _svm_run_id = svm_by_fold[fold]
        svm_probs = run_svm_inference(
            svm_path,
            scaler_path,
            svm_preprocess_dir,
            metadata_val_filtered,
            diseases,
        )
        logger.info(f"SVM inference done: {svm_probs.shape}")

        # ViT inference for each best model
        vit_checkpoints = vit_by_fold[fold]
        for vit_cp in vit_checkpoints:
            vit_stem = vit_cp.stem  # e.g. model_7buw1ylh_7_best0
            out_csv = output_dir / f"paired_{vit_stem}.csv"

            if out_csv.exists() and not args.force:
                logger.info(f"  {out_csv.name} already exists, skipping")
                continue

            logger.info(f"  Processing {vit_cp.name} ...")

            # Load this checkpoint's specific config (in case different models
            # in different folds have slightly different configs)
            vit_config = load_vit_config(vit_cp)

            vit_probs = run_vit_inference(
                vit_cp,
                vit_config,
                vit_preprocess_dir,
                metadata_val_filtered,
                device,
                args.batch_size,
            )
            logger.info(f"    ViT probs shape: {vit_probs.shape}")

            # Build paired DataFrame
            df = metadata_val_filtered.reset_index(drop=True).copy()

            # ViT prediction columns
            for i, disease in enumerate(diseases):
                df[f"vit_pred_{disease}"] = vit_probs[:, i]

            # SVM prediction columns
            for i, disease in enumerate(diseases):
                df[f"svm_pred_{disease}"] = svm_probs[:, i]

            # Save metadata for later traceability
            df.attrs["vit_checkpoint"] = vit_cp.name
            df.attrs["svm_checkpoint"] = svm_path.name
            df.attrs["svm_scaler"] = scaler_path.name
            df.attrs["fold"] = fold

            # Store traceability info as extra columns that will be dropped
            # during MLP training but useful for debugging
            df["_vit_checkpoint"] = vit_cp.name
            df["_svm_checkpoint"] = svm_path.name
            df["_svm_scaler"] = scaler_path.name
            df["_fold"] = fold

            df.to_csv(out_csv, index=False)
            logger.info(f"    Saved {out_csv.name} ({len(df)} rows)")

        # Free GPU memory between folds
        torch.cuda.empty_cache()

    logger.info("\nDone! All paired datasets saved.")


if __name__ == "__main__":
    main()
