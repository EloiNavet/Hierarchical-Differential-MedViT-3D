"""Bootstrap confidence intervals for classification metrics."""

import os
import warnings
from collections.abc import Iterable

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm


def _compute_ece(y_true, y_pred_probs, correct_mask=None, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures calibration by binning predictions by confidence level
    and computing the weighted average of |accuracy - confidence| per bin.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (N,)
    y_pred_probs : np.ndarray
        Prediction probabilities (N, n_classes)
    correct_mask : np.ndarray, optional
        Boolean mask indicating correct predictions. If None, computed from argmax.
    n_bins : int, default=15
        Number of bins for calibration

    Returns
    -------
    float
        ECE value in range [0, 1], lower is better
    """
    if correct_mask is None:
        y_pred = np.argmax(y_pred_probs, axis=1)
        correct_mask = y_pred == y_true

    # Get confidence (maximum predicted probability)
    confidences = np.max(y_pred_probs, axis=1)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct_mask[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def _compute_mce(y_true, y_pred_probs, correct_mask=None, n_bins=15):
    """
    Compute Maximum Calibration Error (MCE).

    MCE measures the worst-case calibration error across all bins.
    It is the maximum absolute difference between accuracy and confidence
    in any bin. This is useful for identifying the most problematic
    confidence regions, which is critical in medical applications.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (N,)
    y_pred_probs : np.ndarray
        Prediction probabilities (N, n_classes)
    correct_mask : np.ndarray, optional
        Boolean mask indicating correct predictions. If None, computed from argmax.
    n_bins : int, default=15
        Number of bins for calibration

    Returns
    -------
    float
        MCE value in range [0, 1], lower is better
    """
    if correct_mask is None:
        y_pred = np.argmax(y_pred_probs, axis=1)
        correct_mask = y_pred == y_true

    # Get confidence (maximum predicted probability)
    confidences = np.max(y_pred_probs, axis=1)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    max_calibration_error = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct_mask[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_calibration_error = max(max_calibration_error, calibration_error)

    return float(max_calibration_error)


def _compute_brier(y_true, y_pred_probs, n_classes, sample_mask=None):
    """
    Compute Brier score for multi-class classification.

    Brier score measures the mean squared error between predicted probabilities
    and true labels (one-hot encoded). Returns both overall and per-class scores.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (N,)
    y_pred_probs : np.ndarray
        Prediction probabilities (N, n_classes)
    n_classes : int
        Number of classes
    sample_mask : np.ndarray, optional
        Boolean mask to select subset of samples (e.g., for top-k filtering)

    Returns
    -------
    tuple
        (brier_overall, brier_per_class_dict)
        - brier_overall: float, mean squared error across all classes and samples
        - brier_per_class_dict: dict mapping class_idx to float, per-class Brier scores
    """
    if sample_mask is not None:
        y_true = y_true[sample_mask]
        y_pred_probs = y_pred_probs[sample_mask]

    if len(y_true) == 0:
        # No samples to evaluate
        return np.nan, {c: np.nan for c in range(n_classes)}

    # Create one-hot encoded labels
    y_true_onehot = np.zeros((len(y_true), n_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Overall Brier score: mean squared error over all classes and samples
    brier_overall = float(np.mean((y_pred_probs - y_true_onehot) ** 2))

    # Per-class Brier score: for each class, MSE of probability predictions
    brier_per_class = {}
    for c in range(n_classes):
        brier_per_class[c] = float(
            np.mean((y_pred_probs[:, c] - y_true_onehot[:, c]) ** 2)
        )

    return brier_overall, brier_per_class


def _compute_softmax_entropy(y_pred_probs, sample_mask=None):
    """
    Compute mean softmax entropy of predictions.

    Entropy measures the uncertainty in predictions. Higher entropy indicates
    more uniform probability distributions (higher uncertainty), while lower
    entropy indicates more peaked distributions (higher confidence).

    Entropy = -sum(p_i * log(p_i)) for i in classes

    Parameters
    ----------
    y_pred_probs : np.ndarray
        Prediction probabilities (N, n_classes)
    sample_mask : np.ndarray, optional
        Boolean mask to select subset of samples

    Returns
    -------
    float
        Mean entropy across all samples
    """
    if sample_mask is not None:
        y_pred_probs = y_pred_probs[sample_mask]

    if len(y_pred_probs) == 0:
        return np.nan

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(y_pred_probs * np.log(y_pred_probs + eps), axis=1)

    return float(np.mean(entropy))


def _compute_gini(y_pred_probs, sample_mask=None):
    """
    Compute mean Gini impurity index of predictions.

    Gini index measures the probability of misclassifying a randomly chosen element
    if it were randomly labeled according to the distribution of labels in the subset.

    Gini = 1 - sum(p_i^2) for i in classes

    Lower Gini indicates more confident predictions (one class dominates),
    while higher Gini indicates more uncertain predictions (uniform distribution).

    Parameters
    ----------
    y_pred_probs : np.ndarray
        Prediction probabilities (N, n_classes)
    sample_mask : np.ndarray, optional
        Boolean mask to select subset of samples

    Returns
    -------
    float
        Mean Gini index across all samples
    """
    if sample_mask is not None:
        y_pred_probs = y_pred_probs[sample_mask]

    if len(y_pred_probs) == 0:
        return np.nan

    # Compute Gini: 1 - sum(p^2)
    gini = 1.0 - np.sum(y_pred_probs**2, axis=1)

    return float(np.mean(gini))


def _compute_renyi_entropy(y_pred_probs, alpha=2.0, sample_mask=None):
    """
    Compute mean Rényi entropy of predictions.

    Rényi entropy is a generalization of Shannon entropy. For alpha=2,
    it is also known as collision entropy.

    Rényi entropy (alpha=2) = -log(sum(p_i^2)) for i in classes

    This is related to the Gini index but in log space, providing another
    measure of prediction uncertainty.

    Parameters
    ----------
    y_pred_probs : np.ndarray
        Prediction probabilities (N, n_classes)
    alpha : float, default=2.0
        Order of Rényi entropy (alpha > 0, alpha != 1)
    sample_mask : np.ndarray, optional
        Boolean mask to select subset of samples

    Returns
    -------
    float
        Mean Rényi entropy across all samples
    """
    if sample_mask is not None:
        y_pred_probs = y_pred_probs[sample_mask]

    if len(y_pred_probs) == 0:
        return np.nan

    if alpha <= 0 or alpha == 1:
        raise ValueError("Alpha must be positive and different from 1")

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    # Compute Rényi entropy: (1/(1-alpha)) * log(sum(p^alpha))
    sum_p_alpha = np.sum(y_pred_probs**alpha, axis=1)
    renyi = (1.0 / (1.0 - alpha)) * np.log(sum_p_alpha + eps)

    return float(np.mean(renyi))


def compute_bootstrap_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_state: int = 42,
    n_jobs: int = -1,
    top_n: int | Iterable[int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute bootstrap confidence intervals for classification metrics using parallel processing.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth labels, of shape (N,)
    y_pred_probs : numpy.ndarray
        Model prediction probabilities, of shape (N, n_classes)
    n_bootstrap : int, default=10000
        Number of bootstrap iterations
    confidence : float, default=0.95
        Confidence level for intervals
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of jobs to run in parallel (-1 means using all processors)

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary containing mean, lower, and upper bounds for each metric
    """

    np.random.seed(random_state)
    n_samples = len(y_true)

    # y_true must be int type
    y_true = y_true.astype(int)

    # Convert y_pred_probs to class predictions for metrics that need them
    y_pred = np.argmax(y_pred_probs, axis=1)  # (N,)

    # Get number of classes from prediction probabilities shape
    # This handles OD evaluation where not all training classes may be present
    n_classes = y_pred_probs.shape[1]

    # Get unique classes actually present in y_true
    classes = np.unique(y_true)

    # Normalize and validate top_n(s)
    top_ns: list[int] = []
    if top_n is not None:
        if isinstance(top_n, int):
            top_ns = [top_n]
        else:  # iterable
            top_ns = sorted({int(k) for k in top_n})
        for k in top_ns:
            if k < 1:
                raise ValueError("All top-n values must be >=1")
            if k > n_classes:
                raise ValueError(
                    f"top-n value {k} cannot exceed number of classes ({n_classes})"
                )

    # Define function for single bootstrap iteration
    def _bootstrap_iteration(iteration_idx):
        # Seed this iteration deterministically based on the global seed and iteration index
        # This ensures reproducibility even with parallel processing
        iteration_seed = (random_state + iteration_idx) % (2**32)
        np.random.seed(iteration_seed)

        # Generate bootstrap indices with retry logic for rare classes
        # Only ensure classes that are actually present in the data appear in bootstrap
        all_present_classes_in_sample = False
        max_attempts = 100  # Prevent infinite loops
        attempt = 0

        while not all_present_classes_in_sample and attempt < max_attempts:
            # Generate bootstrap indices
            indices = np.random.choice(n_samples, n_samples, replace=True)

            # Get bootstrap samples
            bs_y_true = y_true[indices]

            # Check if all classes that are present in y_true are also in bootstrap sample
            if len(np.unique(bs_y_true)) == len(classes):
                all_present_classes_in_sample = True
            else:
                attempt += 1

        if not all_present_classes_in_sample:
            # If we couldn't get all present classes after max attempts, use stratified sampling
            # to ensure all classes that exist in y_true are represented
            strat_indices = []
            for c in classes:
                class_indices = np.where(y_true == c)[0]
                class_sample = np.random.choice(
                    class_indices,
                    max(1, int(len(class_indices) * n_samples / len(y_true))),
                    replace=True,
                )
                strat_indices.extend(class_sample)

            # Shuffle the indices
            np.random.shuffle(strat_indices)
            # Trim or extend to match original sample size
            if len(strat_indices) > n_samples:
                strat_indices = strat_indices[:n_samples]
            else:
                # Append random samples to reach n_samples
                additional = np.random.choice(
                    range(n_samples), n_samples - len(strat_indices), replace=True
                )
                strat_indices.extend(additional)

            indices = strat_indices

        bs_y_true = y_true[indices]
        bs_y_pred = y_pred[indices]
        bs_y_pred_proba = y_pred_probs[indices]

        if n_classes == 2:
            # We pass the probabilities of the positive class (column 1)
            roc_auc_val = roc_auc_score(bs_y_true, bs_y_pred_proba[:, 1])
            pr_auc_val = average_precision_score(bs_y_true, bs_y_pred_proba[:, 1])
        else:
            # For multiclass, use label binarization to handle OD data
            # where not all training classes may be present
            bs_y_true_bin = label_binarize(bs_y_true, classes=range(n_classes))

            # Suppress expected warnings for OD evaluation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*No positive class found.*")
                warnings.filterwarnings("ignore", message=".*Only one class.*")
                roc_auc_val = roc_auc_score(
                    bs_y_true_bin,
                    bs_y_pred_proba,
                    average="macro",
                    multi_class="ovr",
                )
                pr_auc_val = average_precision_score(
                    bs_y_true_bin, bs_y_pred_proba, average="macro"
                )

        # Calculate metrics
        # Suppress expected warnings for OD evaluation with missing classes
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*y_pred contains classes not in y_true.*"
            )
            metrics = {
                "accuracy": accuracy_score(bs_y_true, bs_y_pred),
                "balanced_accuracy": balanced_accuracy_score(bs_y_true, bs_y_pred),
                "confusion_matrix": confusion_matrix(
                    bs_y_true, bs_y_pred, labels=range(n_classes)
                ),
                "classification_report": classification_report(
                    bs_y_true,
                    bs_y_pred,
                    labels=range(n_classes),
                    output_dict=True,
                    zero_division=0,
                ),
                "roc_auc": roc_auc_val,
                "pr_auc": pr_auc_val,
                "mcc": matthews_corrcoef(bs_y_true, bs_y_pred),
                "precision": {},
                "recall": {},
                "f1": {},
            }

        # Compute ECE (Expected Calibration Error)
        metrics["ece"] = _compute_ece(bs_y_true, bs_y_pred_proba)

        # Compute MCE (Maximum Calibration Error)
        metrics["mce"] = _compute_mce(bs_y_true, bs_y_pred_proba)

        # Compute Brier score (overall and per-class)
        brier_overall, brier_per_class = _compute_brier(
            bs_y_true, bs_y_pred_proba, n_classes
        )
        metrics["brier_score"] = brier_overall
        metrics["brier_class"] = brier_per_class

        # Compute uncertainty metrics
        metrics["softmax_entropy"] = _compute_softmax_entropy(bs_y_pred_proba)
        metrics["gini"] = _compute_gini(bs_y_pred_proba)
        metrics["renyi"] = _compute_renyi_entropy(bs_y_pred_proba, alpha=2.0)

        if top_ns:
            sorted_inds = np.argsort(bs_y_pred_proba, axis=1)
            for k in top_ns:
                topk_preds = sorted_inds[:, -k:]
                topk_correct = (topk_preds == bs_y_true[:, None]).any(axis=1)
                metrics[f"top_{k}_accuracy"] = float(np.mean(topk_correct))

                # Compute Top-k ECE (using top-k correctness definition)
                metrics[f"top_{k}_ece"] = _compute_ece(
                    bs_y_true, bs_y_pred_proba, correct_mask=topk_correct
                )

                # Compute Top-k MCE (using top-k correctness definition)
                metrics[f"top_{k}_mce"] = _compute_mce(
                    bs_y_true, bs_y_pred_proba, correct_mask=topk_correct
                )

                # Compute Top-k Brier score (only on samples where true class is in top-k)
                brier_topk_overall, brier_topk_per_class = _compute_brier(
                    bs_y_true, bs_y_pred_proba, n_classes, sample_mask=topk_correct
                )
                metrics[f"top_{k}_brier_score"] = brier_topk_overall
                metrics[f"top_{k}_brier_class"] = brier_topk_per_class

                # Compute Top-k uncertainty metrics (on top-k subset)
                metrics[f"top_{k}_softmax_entropy"] = _compute_softmax_entropy(
                    bs_y_pred_proba, sample_mask=topk_correct
                )
                metrics[f"top_{k}_gini"] = _compute_gini(
                    bs_y_pred_proba, sample_mask=topk_correct
                )
                metrics[f"top_{k}_renyi"] = _compute_renyi_entropy(
                    bs_y_pred_proba, alpha=2.0, sample_mask=topk_correct
                )

                # Compute Top-k balanced accuracy: mean per-class recall when allowing k guesses
                class_recalls = []
                # Also collect per-class precision/recall counts for top-k F1
                topk_precision_dict = {}
                topk_recall_dict = {}
                topk_f1_dict = {}
                for c in classes:
                    class_mask = bs_y_true == c
                    if np.any(class_mask):
                        class_correct = topk_correct[class_mask]
                        recall_c = class_correct.mean()
                        class_recalls.append(recall_c)

                        # Precision: among samples where class c appears in top-k, fraction actually class c
                        appears_mask = (topk_preds == c).any(axis=1)
                        tp_c = np.logical_and(appears_mask, bs_y_true == c).sum()
                        pred_pos_c = appears_mask.sum()
                        precision_c = tp_c / pred_pos_c if pred_pos_c > 0 else 0.0
                        topk_precision_dict[c] = precision_c
                        topk_recall_dict[c] = recall_c
                        if precision_c + recall_c > 0:
                            topk_f1_dict[c] = (
                                2 * precision_c * recall_c / (precision_c + recall_c)
                            )
                        else:
                            topk_f1_dict[c] = 0.0
                    else:
                        topk_precision_dict[c] = 0.0
                        topk_recall_dict[c] = 0.0
                        topk_f1_dict[c] = 0.0

                if class_recalls:
                    metrics[f"top_{k}_balanced_accuracy"] = float(
                        np.mean(class_recalls)
                    )
                metrics[f"top_{k}_precision"] = topk_precision_dict
                metrics[f"top_{k}_recall"] = topk_recall_dict
                metrics[f"top_{k}_f1"] = topk_f1_dict

        # Store class metrics
        for class_idx in range(n_classes):
            metrics["precision"][class_idx] = metrics["classification_report"][
                str(class_idx)
            ]["precision"]
            metrics["recall"][class_idx] = metrics["classification_report"][
                str(class_idx)
            ]["recall"]
            metrics["f1"][class_idx] = metrics["classification_report"][str(class_idx)][
                "f1-score"
            ]

        # Compute macro-averaged metrics (mean across classes)
        metrics["macro_f1"] = np.mean([metrics["f1"][c] for c in range(n_classes)])
        metrics["macro_prec"] = np.mean(
            [metrics["precision"][c] for c in range(n_classes)]
        )
        metrics["macro_sens"] = np.mean(
            [metrics["recall"][c] for c in range(n_classes)]
        )

        # Compute macro-averaged top-k metrics if applicable
        if top_ns:
            for k in top_ns:
                if f"top_{k}_f1" in metrics:
                    metrics[f"top_{k}_macro_f1"] = np.mean(
                        [metrics[f"top_{k}_f1"][c] for c in range(n_classes)]
                    )
                if f"top_{k}_precision" in metrics:
                    metrics[f"top_{k}_macro_prec"] = np.mean(
                        [metrics[f"top_{k}_precision"][c] for c in range(n_classes)]
                    )
                if f"top_{k}_recall" in metrics:
                    metrics[f"top_{k}_macro_sens"] = np.mean(
                        [metrics[f"top_{k}_recall"][c] for c in range(n_classes)]
                    )

        return metrics

    # Run bootstrap iterations in parallel with progress bar
    results_list = Parallel(
        n_jobs=n_jobs,
        backend="threading"
        if (os.environ.get("SLURM_JOB_ID") is not None)
        else "loky",  # should work only on clusters
    )(
        delayed(_bootstrap_iteration)(i)
        for i in tqdm(
            range(n_bootstrap),
            desc="Bootstrapping",
            leave=False,
            disable=(os.environ.get("SLURM_JOB_ID") is not None),
        )
    )

    metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "roc_auc": [],
        "pr_auc": [],
        "mcc": [],
        "ece": [],
        "mce": [],
        "brier_score": [],
        "softmax_entropy": [],
        "gini": [],
        "renyi": [],
        "precision": {class_idx: [] for class_idx in range(n_classes)},
        "recall": {class_idx: [] for class_idx in range(n_classes)},
        "f1": {class_idx: [] for class_idx in range(n_classes)},
        "brier_class": {class_idx: [] for class_idx in range(n_classes)},
        "macro_f1": [],
        "macro_prec": [],
        "macro_sens": [],
        "confusion_matrix": [],
        "classification_report": [],
    }
    if top_ns:
        for k in top_ns:
            metrics[f"top_{k}_accuracy"] = []
            metrics[f"top_{k}_balanced_accuracy"] = []
            metrics[f"top_{k}_ece"] = []
            metrics[f"top_{k}_mce"] = []
            metrics[f"top_{k}_brier_score"] = []
            metrics[f"top_{k}_softmax_entropy"] = []
            metrics[f"top_{k}_gini"] = []
            metrics[f"top_{k}_renyi"] = []
            metrics[f"top_{k}_precision"] = {
                class_idx: [] for class_idx in range(n_classes)
            }
            metrics[f"top_{k}_recall"] = {
                class_idx: [] for class_idx in range(n_classes)
            }
            metrics[f"top_{k}_f1"] = {class_idx: [] for class_idx in range(n_classes)}
            metrics[f"top_{k}_brier_class"] = {
                class_idx: [] for class_idx in range(n_classes)
            }
            metrics[f"top_{k}_macro_f1"] = []
            metrics[f"top_{k}_macro_prec"] = []
            metrics[f"top_{k}_macro_sens"] = []

    for result in results_list:
        for metric_name, value in result.items():
            if (
                metric_name
                in [
                    "accuracy",
                    "balanced_accuracy",
                    "roc_auc",
                    "pr_auc",
                    "mcc",
                    "ece",
                    "mce",
                    "brier_score",
                    "softmax_entropy",
                    "gini",
                    "renyi",
                    "macro_f1",
                    "macro_prec",
                    "macro_sens",
                ]
                or metric_name == "confusion_matrix"
                or metric_name == "classification_report"
            ):
                metrics[metric_name].append(value)
            elif metric_name in ["precision", "recall", "f1", "brier_class"]:
                for class_idx, class_value in value.items():
                    metrics[metric_name][class_idx].append(class_value)
            elif top_ns and any(
                metric_name
                in (
                    f"top_{k}_accuracy",
                    f"top_{k}_balanced_accuracy",
                    f"top_{k}_ece",
                    f"top_{k}_mce",
                    f"top_{k}_brier_score",
                    f"top_{k}_softmax_entropy",
                    f"top_{k}_gini",
                    f"top_{k}_renyi",
                    f"top_{k}_macro_f1",
                    f"top_{k}_macro_prec",
                    f"top_{k}_macro_sens",
                )
                for k in top_ns
            ):
                metrics[metric_name].append(value)
            elif top_ns and any(
                metric_name
                in (
                    f"top_{k}_precision",
                    f"top_{k}_recall",
                    f"top_{k}_f1",
                    f"top_{k}_brier_class",
                )
                for k in top_ns
            ):
                for class_idx, class_value in value.items():
                    metrics[metric_name][class_idx].append(class_value)
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    final_results = {}

    def calculate_ci(values):
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            lower = np.percentile(valid_values, 100 * alpha)
            upper = np.percentile(valid_values, 100 * (1 - alpha))
            mean = np.mean(valid_values)
        else:
            lower, upper, mean = (np.nan,) * 3
        return mean, lower, upper

    # Process overall metrics
    overall_metric_names = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "mcc",
        "ece",
        "mce",
        "brier_score",
        "softmax_entropy",
        "gini",
        "renyi",
        "macro_f1",
        "macro_prec",
        "macro_sens",
    ]
    for k in top_ns:
        overall_metric_names.append(f"top_{k}_accuracy")
        overall_metric_names.append(f"top_{k}_balanced_accuracy")
        overall_metric_names.append(f"top_{k}_ece")
        overall_metric_names.append(f"top_{k}_mce")
        overall_metric_names.append(f"top_{k}_brier_score")
        overall_metric_names.append(f"top_{k}_softmax_entropy")
        overall_metric_names.append(f"top_{k}_gini")
        overall_metric_names.append(f"top_{k}_renyi")
        overall_metric_names.append(f"top_{k}_macro_f1")
        overall_metric_names.append(f"top_{k}_macro_prec")
        overall_metric_names.append(f"top_{k}_macro_sens")

    for metric_name in overall_metric_names:
        values = np.array(metrics[metric_name])
        mean, lower, upper = calculate_ci(values)
        final_results[metric_name] = {
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "samples": values,
        }

    # Process class-specific metrics
    class_metric_roots = ["precision", "recall", "f1", "brier_class"]
    for k in top_ns:
        class_metric_roots.extend(
            [
                f"top_{k}_precision",
                f"top_{k}_recall",
                f"top_{k}_f1",
                f"top_{k}_brier_class",
            ]
        )

    for metric_name in class_metric_roots:
        final_results[metric_name] = {}
        for class_idx in range(n_classes):
            values = np.array(metrics[metric_name][class_idx])
            mean, lower, upper = calculate_ci(values)
            final_results[metric_name][class_idx] = {
                "mean": mean,
                "lower": lower,
                "upper": upper,
                "samples": values,
            }

    # Process confusion matrix
    confusion_matrices = np.array(metrics["confusion_matrix"])
    mean_cm = np.mean(confusion_matrices, axis=0)
    lower_cm = np.percentile(confusion_matrices, 100 * alpha, axis=0)
    upper_cm = np.percentile(confusion_matrices, 100 * (1 - alpha), axis=0)

    final_results["confusion_matrix"] = {
        "mean": mean_cm,
        "lower": lower_cm,
        "upper": upper_cm,
    }

    # Process classification report
    # Initialize dictionaries to hold aggregated values across bootstrap iterations
    classification_report_metrics = {}

    # Get all keys from the first report to initialize the structure
    sample_report = metrics["classification_report"][0]
    for key in sample_report:
        if isinstance(sample_report[key], dict):
            classification_report_metrics[key] = {
                metric: [] for metric in sample_report[key]
            }
        else:
            classification_report_metrics[key] = []

    # Collect values across all bootstrap iterations
    for report in metrics["classification_report"]:
        for key, value in report.items():
            if isinstance(value, dict):
                for metric, score in value.items():
                    classification_report_metrics[key][metric].append(score)
            else:
                classification_report_metrics[key].append(value)

    # Calculate statistics for the classification report
    final_results["classification_report"] = {}
    for key, values in classification_report_metrics.items():
        if isinstance(values, dict):
            final_results["classification_report"][key] = {}
            for metric, scores in values.items():
                scores_array = np.array(scores)
                valid_scores = scores_array[~np.isnan(scores_array)]
                final_results["classification_report"][key][metric] = {
                    "mean": np.mean(valid_scores),
                    "lower": np.percentile(valid_scores, 100 * alpha),
                    "upper": np.percentile(valid_scores, 100 * (1 - alpha)),
                }
        else:
            scores_array = np.array(values)
            valid_scores = scores_array[~np.isnan(scores_array)]
            final_results["classification_report"][key] = {
                "mean": np.mean(valid_scores),
                "lower": np.percentile(valid_scores, 100 * alpha),
                "upper": np.percentile(valid_scores, 100 * (1 - alpha)),
            }

    return final_results
