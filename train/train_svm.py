import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import wandb as w
from optuna.samplers import TPESampler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset.preprocessing import LABELS_SLANT, DataPrepaSVM, load_svm_features
from utils import dir_path, get_train_val_test

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("wandb").setLevel(logging.CRITICAL)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVM for MRI Classification")
    parser.add_argument(
        "--training-csv-dir",
        type=dir_path,
        required=True,
        help="Directory containing k-fold CSV files.",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=dir_path,
        required=True,
        help="Directory to save intermediate data.",
    )
    parser.add_argument(
        "--save-dir", type=dir_path, required=True, help="Path to save models."
    )
    parser.add_argument(
        "--project-name", type=str, default="project", help="Wandb project name."
    )
    parser.add_argument("--runname", type=str, help="Wandb run name.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        help='Fold to train on. Use "all" to train on all 10 folds.',
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for hyperparameter optimization.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mcc",
        choices=[
            "bacc",
            "acc",
            "roc_auc",
            "mcc",
            "f1",
            "pr_auc",
            "precision",
            "recall",
            "neg_log_loss",
        ],
        help="Metric to optimize during hyperparameter search.",
    )
    return parser.parse_args()


def compute_metric(
    Y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    metric: str,
) -> float:
    """Compute the specified metric for evaluation."""
    if metric == "bacc":
        return balanced_accuracy_score(Y_true, y_pred)
    elif metric == "acc":
        return accuracy_score(Y_true, y_pred)
    elif metric == "roc_auc":
        return roc_auc_score(Y_true, y_pred_prob, multi_class="ovr", average="macro")
    elif metric == "mcc":
        return matthews_corrcoef(Y_true, y_pred)
    elif metric == "f1":
        return f1_score(Y_true, y_pred, average="macro")
    elif metric == "pr_auc":
        n_classes = y_pred_prob.shape[1]
        Y_bin = label_binarize(Y_true, classes=range(n_classes))
        return average_precision_score(Y_bin, y_pred_prob, average="macro")
    elif metric == "precision":
        return precision_score(Y_true, y_pred, average="macro", zero_division=0)
    elif metric == "recall":
        return recall_score(Y_true, y_pred, average="macro", zero_division=0)
    elif metric == "neg_log_loss":
        return -log_loss(Y_true, y_pred_prob, labels=list(range(y_pred_prob.shape[1])))
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def compute_feature_importance(
    classifier: SVC,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    kernel: str,
    n_repeats: int = 10,
) -> np.ndarray:
    """
    Compute feature importance for any kernel type.

    For linear kernels, uses absolute coefficients.
    For non-linear kernels (rbf, poly), uses permutation importance.

    Parameters
    ----------
    classifier : SVC
        Trained SVM classifier
    X_val : np.ndarray
        Validation features for permutation importance
    Y_val : np.ndarray
        Validation labels
    kernel : str
        Kernel type ('linear', 'rbf', 'poly')
    n_repeats : int
        Number of permutations for importance calculation

    Returns
    -------
    np.ndarray
        Feature importance scores
    """
    if kernel == "linear":
        importances = np.abs(classifier.coef_).mean(axis=0)
    else:
        perm_importance = permutation_importance(
            classifier,
            X_val,
            Y_val,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
        )
        importances = perm_importance.importances_mean

    return importances


def train_svm_optuna(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    model_save_dir: Path,
    fold: int,
    run_id: str,
    n_trials: int = 100,
    metric: Literal[
        "bacc",
        "acc",
        "roc_auc",
        "mcc",
        "f1",
        "pr_auc",
        "precision",
        "recall",
    ] = "bacc",
) -> float:
    """
    Train SVM using Optuna Bayesian optimization (TPE sampler).

    This is much faster than grid search and often finds better hyperparameters.
    Uses Tree-structured Parzen Estimator (TPE) for intelligent search.

    Parameters
    ----------
    X_train, Y_train : np.ndarray
        Training data
    X_val, Y_val : np.ndarray
        Validation data
    model_save_dir : Path
        Directory to save the best model
    fold : int
        Fold number
    run_id : str
        W&B run ID
    n_trials : int
        Number of Optuna trials
    metric : str
        Metric to optimize

    Returns
    -------
    float
        Best validation metric score
    """

    best_score = -float("inf")
    best_classifier = None
    best_params = None

    # Store metric curves for visualization
    metric_history = {"linear": [], "rbf": [], "poly": []}
    C_history = {"linear": [], "rbf": [], "poly": []}
    gamma_history = {"rbf": [], "poly": []}

    def objective(trial):
        nonlocal best_score, best_classifier, best_params

        # Suggest hyperparameters
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        C = trial.suggest_float("C", 1e-4, 1, log=True)

        svm_params = {
            "C": C,
            "kernel": kernel,
            "decision_function_shape": "ovr",
            "random_state": 42,
            "probability": True,
        }

        # Add gamma for rbf and poly kernels
        if kernel in ["rbf", "poly"]:
            gamma = trial.suggest_float("gamma", 1e-5, 1e-1, log=True)
            svm_params["gamma"] = gamma
            gamma_history[kernel].append(gamma)

        # Add degree for poly kernel
        if kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5)
            svm_params["degree"] = degree

        # Train classifier
        classifier = SVC(**svm_params)
        classifier.fit(X_train, Y_train)

        y_pred_prob = classifier.predict_proba(X_val)
        y_pred = np.argmax(y_pred_prob, axis=1)
        score = compute_metric(Y_val, y_pred, y_pred_prob, metric)

        # Store for visualization
        metric_history[kernel].append(score)
        C_history[kernel].append(C)

        # Update best model
        if score > best_score:
            best_score = score
            best_classifier = classifier
            best_params = svm_params.copy()

        return score

    # Create Optuna study with TPE sampler
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
    )

    logger.info(f"Optimizing fold {fold} ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best {metric}: {best_score:.4f} | {best_params}")

    for kernel in ["linear", "rbf", "poly"]:
        if metric_history[kernel]:
            data = [
                [C_history[kernel][i], metric_history[kernel][i]]
                for i in range(len(C_history[kernel]))
            ]
            table = w.Table(data=data, columns=["C", metric])
            w.log({f"{kernel}_{metric}_optimization": table})

    model_filename = f"svm_{run_id}_{fold}.pkl"
    save_path = model_save_dir / model_filename
    with open(save_path, "wb") as f:
        pickle.dump(best_classifier, f)

    importances = compute_feature_importance(
        best_classifier,
        X_val,
        Y_val,
        best_params["kernel"],
        n_repeats=10,
    )

    # Map feature indices to actual brain region names from LABELS_SLANT
    feature_names = list(LABELS_SLANT.values())

    # Sort by importance descending (most important first)
    sorted_indices = np.argsort(importances)[::-1]
    importance_data = [
        [feature_names[i].strip(), float(importances[i])] for i in sorted_indices
    ]

    w.log(
        {
            f"feature_importance_fold_{fold}": w.Table(
                data=importance_data, columns=["feature", "importance"]
            )
        }
    )

    return best_score


def train_single_fold(
    args: argparse.Namespace,
    fold: int,
    diseases: list[str],
    preprocess_dir: Path,
    model_save_dir: Path,
) -> float:
    """Train SVM on a single fold."""
    run_name = f"{args.runname}_{fold}"

    wandb_dir = model_save_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    w.init(
        project=args.project_name,
        name=run_name,
        mode=args.wandb_mode,
        config={**vars(args), "fold": fold},
        dir=str(wandb_dir),
    )

    # Get train/val splits for this fold
    meta_train, meta_val, _, _ = get_train_val_test(
        Path(args.training_csv_dir), fold, kfold=10
    )

    logger.info(f"Fold {fold}: Train={len(meta_train)}, Val={len(meta_val)}")

    X_train, Y_train = load_svm_features(preprocess_dir, meta_train, diseases)
    X_val, Y_val = load_svm_features(preprocess_dir, meta_val, diseases)

    logger.info(
        f"Features: {X_train.shape[1]}D, Train={X_train.shape[0]}, Val={X_val.shape[0]}"
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    scaler_path = model_save_dir / f"scaler_{w.run.id}_{fold}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    val_metric = train_svm_optuna(
        X_train,
        Y_train,
        X_val,
        Y_val,
        model_save_dir,
        fold,
        w.run.id,
        n_trials=args.n_trials,
        metric=args.metric,
    )

    w.finish()
    return val_metric


def main():
    args = get_args()

    if (
        args.wandb_mode == "online"
        and not os.getenv("WANDB_API_KEY")
        and os.path.exists("wandb.key")
    ):
        with open("wandb.key", "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

    # Infer DISEASES from metadata
    _, _, _, metadata_all = get_train_val_test(Path(args.training_csv_dir), 0, kfold=10)
    diseases = sorted(metadata_all.Diagnosis.unique().tolist())

    model_save_dir = Path(args.save_dir) / args.runname
    model_save_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

    preprocess_dir = Path(args.intermediate_dir) / "train"
    preprocess_dir.mkdir(parents=True, exist_ok=True, mode=0o777)

    logger.info("Preprocessing data...")
    data_prepa = DataPrepaSVM(metadata_all, preprocess_dir, device="cpu")
    data_prepa.preprocess_data(n_jobs=-1, verbose=0)

    # Determine which folds to train
    if args.fold.lower() == "all":
        folds_to_train = list(range(10))
        logger.info("Training on all 10 folds")
    else:
        try:
            fold_num = int(args.fold)
            if fold_num < 0 or fold_num > 9:
                raise ValueError("Fold must be between 0 and 9 or 'all'")
            folds_to_train = [fold_num]
        except ValueError:
            logger.error(f"Invalid fold value: {args.fold}. Must be 0-9 or 'all'")
            raise

    val_metrics = []
    for fold in folds_to_train:
        val_metric = train_single_fold(
            args, fold, diseases, preprocess_dir, model_save_dir
        )
        val_metrics.append(val_metric)

    # Summary statistics if training on all folds
    if len(folds_to_train) > 1:
        logger.info(f"{'=' * 60}")
        logger.info("Training Summary")
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Mean validation metric: {np.mean(val_metrics):.4f} ± {np.std(val_metrics):.4f}"
        )
        logger.info(
            f"Best fold: {np.argmax(val_metrics)} (metric: {np.max(val_metrics):.4f})"
        )
        logger.info(
            f"Worst fold: {np.argmin(val_metrics)} (metric: {np.min(val_metrics):.4f})"
        )
        logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()

# Example usage:
# python train/train_svm.py \
#   --training-csv-dir ./path/to/10fold_CV/ \
#   --save-dir ./path/to/saved_models/svm/ \
#   --intermediate-dir ./path/to/preprocessed/ \
#   --runname my-experiment \
#   --fold all \
#   --n-trials 100 \
#   --metric bacc \
#   --wandb-mode online
