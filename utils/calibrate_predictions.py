import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from .bootstrap_metric import _compute_ece
from .calibration import IsotonicCalibration, PlattScaling, TemperatureScaling


def find_prediction_files(
    model_dir: Path, pattern: str = "*_best0_*.csv"
) -> list[Path]:
    """Find all prediction CSV files matching pattern."""
    return sorted(model_dir.glob(pattern))


def extract_fold_from_filename(filepath: Path) -> int:
    """Extract fold number from filename.

    Supports:
    - 'prediction_model_59zznxlz_8_best0_id.csv'  (transformer)
    - 'prediction_svm_2d9tuewc_6_id.csv'           (SVM / no best index)
    """
    import re

    # Transformer-style: _<fold>_best<n>_
    match = re.search(r"_(\d+)_best\d+_", filepath.name)
    if match:
        return int(match.group(1))
    # SVM-style: _<fold>_(id|od).csv
    match = re.search(r"_(\d+)_(id|od)\.csv$", filepath.name)
    if match:
        return int(match.group(1))
    return -1


def extract_best_index_from_filename(filepath: Path) -> int:
    """Extract best index from filename like 'prediction_model_59zznxlz_8_best0_id.csv'.

    Returns 0 when no best index is present (e.g. SVM files).
    """
    import re

    match = re.search(r"_best(\d+)_", filepath.name)
    if match:
        return int(match.group(1))
    return 0


def load_predictions(csv_path: Path) -> tuple:
    """Load predictions and extract classes, logits, labels."""
    df = pd.read_csv(csv_path)

    # Find prediction columns
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError(f"No prediction columns in {csv_path}")

    # Extract class names
    classes = [c.replace("pred_", "").replace("_ensemble", "") for c in pred_cols]

    # Extract logits/probs
    logits = df[pred_cols].values
    logits = softmax(logits, axis=1)

    # Extract labels
    labels = df["Diagnosis"].map({d: i for i, d in enumerate(classes)}).values

    return df, classes, pred_cols, logits, labels


def calibrate_fold(
    calibrator: object,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    method: str,
) -> tuple:
    """Apply a fitted calibrator and return calibrated probs + metrics."""

    # ECE avant
    test_probs_uncalib = softmax(test_logits, axis=1)
    ece_before = _compute_ece(
        test_labels, test_probs_uncalib, correct_mask=None, n_bins=15
    )

    # Apply transformation
    if method == "isotonic":
        # Isotonic (as defined in calibration.py) expects probabilities
        test_probs_calib = calibrator.transform(test_probs_uncalib)
    else:
        # TempScale and Platt expect logits
        test_probs_calib = calibrator.transform(test_logits)

    # ECE after calibration
    ece_after = _compute_ece(
        test_labels, test_probs_calib, correct_mask=None, n_bins=15
    )

    # Get temperature (returns None if not applicable)
    temperature = getattr(calibrator, "temperature", None)

    return test_probs_calib, ece_before, ece_after, temperature


def main():
    parser = argparse.ArgumentParser(
        description="Apply calibration to model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use fold 0 as validation, calibrate all others
  python visualizations/results/calibrate_predictions.py \\
      --model-dir /data/models/medvit/ \\
      --output-dir /data/models/medvit_calibrated/ \\
      --val-fold 0 \\
      --method temperature
  
  # Process all folds and create ensemble
  python visualizations/results/calibrate_predictions.py \\
      --model-dir /data/models/medvit/ \\
      --output-dir /data/models/medvit_calibrated/ \\
      --val-fold all \\
      --method temperature
        """,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing prediction CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save calibrated predictions",
    )
    parser.add_argument(
        "--val-fold",
        type=str,
        required=True,
        help="Fold to use as validation set for fitting calibrator, or 'all' to process all folds",
    )
    parser.add_argument(
        "--method",
        choices=["temperature", "platt", "isotonic"],
        default="temperature",
        help="Calibration method",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_best*_*.csv",
        help="Glob pattern for prediction files (default: *_best*_*.csv to match all best models)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress"
    )

    args = parser.parse_args()

    # Handle "all" mode
    if args.val_fold.lower() == "all":
        return process_all_folds(args)

    # Convert to int for single fold mode
    try:
        val_fold = int(args.val_fold)
    except ValueError:
        print(f"Error: --val-fold must be an integer or 'all', got '{args.val_fold}'")
        return 1

    # Original single-fold processing
    return process_single_fold(args, val_fold)


def process_single_fold(args, val_fold: int):
    """Process calibration for a single validation fold."""
    # Find all prediction files
    all_files = find_prediction_files(args.model_dir, args.pattern)

    if not all_files:
        print(
            f"\tNo prediction files found in {args.model_dir} matching '{args.pattern}'"
        )
        return 1

    print(f"Found {len(all_files)} prediction files")

    # Separate by domain (ID vs OD)
    id_files = [f for f in all_files if "_id.csv" in f.name]
    od_files = [f for f in all_files if "_od.csv" in f.name]

    print(f"\t- {len(id_files)} ID domain files")
    print(f"\t- {len(od_files)} OD domain files")

    # Group by (fold, best_index)
    id_by_fold_best = {}  # (fold, best_idx) -> file
    for f in id_files:
        fold = extract_fold_from_filename(f)
        best_idx = extract_best_index_from_filename(f)
        id_by_fold_best[(fold, best_idx)] = f

    od_by_fold_best = {}  # (fold, best_idx) -> file
    for f in od_files:
        fold = extract_fold_from_filename(f)
        best_idx = extract_best_index_from_filename(f)
        od_by_fold_best[(fold, best_idx)] = f

    # Get available folds and best indices
    available_folds = sorted({k[0] for k in id_by_fold_best})
    available_best_indices = sorted({k[1] for k in id_by_fold_best})
    print(f"\nAvailable folds: {available_folds}")
    print(f"Available best indices: {available_best_indices}")

    # Check validation fold exists
    if val_fold not in available_folds:
        print(f"\tValidation fold {val_fold} not found in ID files")
        print(f"\t Available folds: {available_folds}")
        return 1

    print(f"\nUsing fold {val_fold} as validation set")
    print(f"Calibration method: {args.method}")

    # Load validation data (ID domain) - use best0 for fitting the calibrator
    val_best_idx = 0  # Use best0 for calibrator fitting
    if (val_fold, val_best_idx) not in id_by_fold_best:
        # Fallback to first available best index for this fold
        val_best_idx = min(k[1] for k in id_by_fold_best if k[0] == val_fold)
    val_file = id_by_fold_best[(val_fold, val_best_idx)]
    print(f"\nLoading validation data: {val_file.name}")
    val_df, val_classes, _, val_logits, val_labels = load_predictions(val_file)
    print(f"\tValidation samples: {len(val_df)}")
    print(f"\tClasses: {val_classes}")

    print(
        f"\nFitting calibrator '{args.method}' on validation data (fold {val_fold}, best{val_best_idx})..."
    )

    if args.method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.fit(val_logits, val_labels, verbose=args.verbose)
    elif args.method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(val_logits, val_labels, verbose=args.verbose)
    elif args.method == "isotonic":
        calibrator = IsotonicCalibration()
        # Isotonic (as defined in calibration.py) expects probabilities
        val_probs = softmax(val_logits, axis=1)
        calibrator.fit(val_probs, val_labels, verbose=args.verbose)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print("Calibrator fitted.")

    # Results tracking
    results = []

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process ID domain
    print("\n" + "=" * 80)
    print("CALIBRATING ID DOMAIN")
    print("=" * 80)

    for (fold, best_idx), test_file in sorted(id_by_fold_best.items()):
        if fold == val_fold:
            print(f"\nFold {fold} best{best_idx}: SKIPPED (validation set)")
            continue

        print(f"\nFold {fold} best{best_idx}: {test_file.name}")

        # Load test data
        test_df, test_classes, test_pred_cols, test_logits, test_labels = (
            load_predictions(test_file)
        )

        if test_classes != val_classes:
            print(
                f"\t\tWARNING: Class mismatch! Val: {val_classes}, Test: {test_classes}"
            )

        # Apply calibration
        test_probs_calib, ece_before, ece_after, temperature = calibrate_fold(
            calibrator,  # Pass the fitted calibrator
            test_logits,
            test_labels,
            method=args.method,
        )
        print(f"\tSamples: {len(test_df)}")
        print(f"\tECE before: {ece_before:.4f}")
        print(f"\tECE after:  {ece_after:.4f}")
        print(
            f"\tImprovement: {(ece_before - ece_after):.4f} ({(1 - ece_after / max(ece_before, 1e-10)) * 100:.1f}% reduction)"
        )
        if temperature is not None:
            print(f"\tTemperature: {temperature:.4f}")

        # Save calibrated predictions
        output_df = test_df.copy()
        for i, col in enumerate(test_pred_cols):
            output_df[col] = test_probs_calib[:, i]

        output_path = args.output_dir / test_file.name
        output_df.to_csv(output_path, index=False)
        print(f"\tSaved: {output_path.name}")

        results.append(
            {
                "domain": "ID",
                "fold": fold,
                "best_idx": best_idx,
                "filename": test_file.name,
                "n_samples": len(test_df),
                "ece_before": ece_before,
                "ece_after": ece_after,
                "improvement": ece_before - ece_after,
                "temperature": temperature,
            }
        )

    # Process OD domain
    print("\n" + "=" * 80)
    print("CALIBRATING OD DOMAIN")
    print("=" * 80)

    for (fold, best_idx), test_file in sorted(od_by_fold_best.items()):
        if fold == val_fold:
            print(f"\nFold {fold} best{best_idx}: SKIPPED (validation set)")
            continue

        print(f"\nFold {fold} best{best_idx}: {test_file.name}")

        # Load test data
        test_df, test_classes, test_pred_cols, test_logits, test_labels = (
            load_predictions(test_file)
        )

        # Apply calibration
        test_probs_calib, ece_before, ece_after, temperature = calibrate_fold(
            calibrator, test_logits, test_labels, method=args.method
        )

        print(f"\tSamples: {len(test_df)}")
        print(f"\tECE before: {ece_before:.4f}")
        print(f"\tECE after:  {ece_after:.4f}")
        print(
            f"\tImprovement: {(ece_before - ece_after):.4f} ({(1 - ece_after / max(ece_before, 1e-10)) * 100:.1f}% reduction)"
        )
        if temperature is not None:
            print(f"\tTemperature: {temperature:.4f}")

        # Save calibrated predictions
        output_df = test_df.copy()
        for i, col in enumerate(test_pred_cols):
            output_df[col] = test_probs_calib[:, i]

        output_path = args.output_dir / test_file.name
        output_df.to_csv(output_path, index=False)
        print(f"\tSaved: {output_path.name}")

        results.append(
            {
                "domain": "OD",
                "fold": fold,
                "best_idx": best_idx,
                "filename": test_file.name,
                "n_samples": len(test_df),
                "ece_before": ece_before,
                "ece_after": ece_after,
                "improvement": ece_before - ece_after,
                "temperature": temperature,
            }
        )

    # Generate summary report
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)

    df_results = pd.DataFrame(results)

    # Overall statistics
    print("\nOverall Statistics:")
    for domain in ["ID", "OD"]:
        df_dom = df_results[df_results["domain"] == domain]
        if len(df_dom) == 0:
            continue
        print(f"\n{domain} Domain:")
        print(f"\tFolds processed: {len(df_dom)}")
        print(f"\tTotal samples: {df_dom['n_samples'].sum()}")
        print(
            f"\tECE before: {df_dom['ece_before'].mean():.4f} ± {df_dom['ece_before'].std():.4f}"
        )
        print(
            f"\tECE after:  {df_dom['ece_after'].mean():.4f} ± {df_dom['ece_after'].std():.4f}"
        )
        print(f"\tAvg improvement: {df_dom['improvement'].mean():.4f}")
        if args.method == "temperature" and "temperature" in df_results.columns:
            temps = df_dom["temperature"].dropna()
            if len(temps) > 0:
                print(f"\tAvg temperature: {temps.mean():.4f} ± {temps.std():.4f}")

    # Save results table
    results_csv = args.output_dir / "calibration_results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")

    # Generate report text
    report_path = args.output_dir / "CALIBRATION_REPORT.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CALIBRATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model directory: {args.model_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Validation fold: {val_fold}\n")
        f.write(f"Calibration method: {args.method}\n")
        f.write(f"Files processed: {len(results)}\n\n")

        f.write("SUMMARY BY DOMAIN\n")
        f.write("-" * 80 + "\n")
        for domain in ["ID", "OD"]:
            df_dom = df_results[df_results["domain"] == domain]
            if len(df_dom) == 0:
                continue
            f.write(f"\n{domain} Domain:\n")
            f.write(f"\tFolds: {len(df_dom)}\n")
            f.write(f"\tSamples: {df_dom['n_samples'].sum()}\n")
            f.write(
                f"\tECE before: {df_dom['ece_before'].mean():.4f} ± {df_dom['ece_before'].std():.4f}\n"
            )
            f.write(
                f"\tECE after:  {df_dom['ece_after'].mean():.4f} ± {df_dom['ece_after'].std():.4f}\n"
            )
            f.write(f"\tImprovement: {df_dom['improvement'].mean():.4f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n\n")
        f.write(df_results.to_string(index=False))

    print(f"Report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"\nCalibrated predictions saved to: {args.output_dir}")

    return 0


def process_all_folds(args):
    """Process all folds 0-9, creating individual calibrated outputs and final ensemble."""
    print("=" * 80)
    print("PROCESSING ALL FOLDS MODE")
    print("=" * 80)
    print("Will calibrate folds 0-9, using each as validation in turn")
    print(f"Method: {args.method}")
    print()

    # Find all prediction files once
    all_files = find_prediction_files(args.model_dir, args.pattern)
    if not all_files:
        print(
            f"\tNo prediction files found in {args.model_dir} matching '{args.pattern}'"
        )
        return 1

    # Separate by domain
    id_files = [f for f in all_files if "_id.csv" in f.name]
    od_files = [f for f in all_files if "_od.csv" in f.name]

    print(f"Found {len(all_files)} prediction files")
    print(f"\t- {len(id_files)} ID domain files")
    print(f"\t- {len(od_files)} OD domain files")

    # Group by (fold, best_idx)
    id_by_fold_best = {}  # (fold, best_idx) -> file
    for f in id_files:
        fold = extract_fold_from_filename(f)
        best_idx = extract_best_index_from_filename(f)
        id_by_fold_best[(fold, best_idx)] = f

    od_by_fold_best = {}  # (fold, best_idx) -> file
    for f in od_files:
        fold = extract_fold_from_filename(f)
        best_idx = extract_best_index_from_filename(f)
        od_by_fold_best[(fold, best_idx)] = f

    # Get available folds and best indices
    available_folds = sorted({k[0] for k in id_by_fold_best})
    available_best_indices = sorted({k[1] for k in id_by_fold_best})
    print(f"\nAvailable folds: {available_folds}")
    print(f"Available best indices: {available_best_indices}")

    # Storage for all calibrated predictions per (fold, best_idx)
    calibrated_predictions_id = {}  # (fold, best_idx) -> list of dataframes
    calibrated_predictions_od = {}  # (fold, best_idx) -> list of dataframes

    all_results = []

    # Process each fold as validation
    for val_fold in range(10):
        if val_fold not in available_folds:
            print(f"\n{'=' * 80}")
            print(f"SKIPPING FOLD {val_fold} (not found in data)")
            print(f"{'=' * 80}")
            continue

        print(f"\n{'=' * 80}")
        print(f"PROCESSING WITH VAL_FOLD = {val_fold}")
        print(f"{'=' * 80}")

        # Create subfolder for this validation fold
        fold_output_dir = args.output_dir / f"val_fold_{val_fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Load validation data - use best0 for fitting the calibrator
        val_best_idx = 0
        if (val_fold, val_best_idx) not in id_by_fold_best:
            val_best_idx = min(k[1] for k in id_by_fold_best if k[0] == val_fold)
        val_file = id_by_fold_best[(val_fold, val_best_idx)]
        val_df, _, _, val_logits, val_labels = load_predictions(val_file)

        print(
            f"\nValidation data: fold {val_fold} best{val_best_idx}, {len(val_df)} samples"
        )

        # Fit calibrator
        if args.method == "temperature":
            calibrator = TemperatureScaling()
            calibrator.fit(val_logits, val_labels, verbose=False)
        elif args.method == "platt":
            calibrator = PlattScaling()
            calibrator.fit(val_logits, val_labels, verbose=False)
        elif args.method == "isotonic":
            calibrator = IsotonicCalibration()
            val_probs = softmax(val_logits, axis=1)
            calibrator.fit(val_probs, val_labels, verbose=False)

        # Calibrate all folds - ID domain (all best indices)
        # Note: we also calibrate val_fold itself using its own calibrator
        for (test_fold, test_best_idx), test_file in sorted(id_by_fold_best.items()):
            test_df, _, test_pred_cols, test_logits, test_labels = load_predictions(
                test_file
            )
            test_probs_calib, ece_before, ece_after, temperature = calibrate_fold(
                calibrator, test_logits, test_labels, method=args.method
            )

            # Store calibrated predictions
            output_df = test_df.copy()
            for i, col in enumerate(test_pred_cols):
                output_df[col] = test_probs_calib[:, i]

            # Save to subfolder
            output_path = fold_output_dir / test_file.name
            output_df.to_csv(output_path, index=False)

            # Store for final ensemble (using (test_fold, test_best_idx) as key)
            key = (test_fold, test_best_idx)
            if key not in calibrated_predictions_id:
                calibrated_predictions_id[key] = []
            calibrated_predictions_id[key].append(output_df)

            all_results.append(
                {
                    "val_fold": val_fold,
                    "test_fold": test_fold,
                    "best_idx": test_best_idx,
                    "domain": "ID",
                    "n_samples": len(test_df),
                    "ece_before": ece_before,
                    "ece_after": ece_after,
                    "temperature": temperature,
                }
            )

        # Calibrate all folds - OD domain (all best indices)
        # Note: we also calibrate val_fold itself using its own calibrator
        for (test_fold, test_best_idx), test_file in sorted(od_by_fold_best.items()):
            test_df, _test_classes, test_pred_cols, test_logits, test_labels = (
                load_predictions(test_file)
            )
            test_probs_calib, ece_before, ece_after, temperature = calibrate_fold(
                calibrator, test_logits, test_labels, method=args.method
            )

            output_df = test_df.copy()
            for i, col in enumerate(test_pred_cols):
                output_df[col] = test_probs_calib[:, i]

            output_path = fold_output_dir / test_file.name
            output_df.to_csv(output_path, index=False)

            key = (test_fold, test_best_idx)
            if key not in calibrated_predictions_od:
                calibrated_predictions_od[key] = []
            calibrated_predictions_od[key].append(output_df)

            all_results.append(
                {
                    "val_fold": val_fold,
                    "test_fold": test_fold,
                    "best_idx": test_best_idx,
                    "domain": "OD",
                    "n_samples": len(test_df),
                    "ece_before": ece_before,
                    "ece_after": ece_after,
                    "temperature": temperature,
                }
            )

    # Create final ensemble by averaging all 10 calibrations for each (fold, best_idx)
    print(f"\n{'=' * 80}")
    print("CREATING ENSEMBLE PREDICTIONS")
    print(f"{'=' * 80}")

    final_dir = args.output_dir.parent / f"{args.output_dir.name}_all"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Ensemble ID domain
    for (test_fold, test_best_idx), calib_dfs in sorted(
        calibrated_predictions_id.items()
    ):
        print(
            f"\nEnsembling fold {test_fold} best{test_best_idx} (ID): {len(calib_dfs)} calibrations"
        )

        # Get base dataframe structure from first calibration
        base_df = calib_dfs[0].copy()
        pred_cols = [c for c in base_df.columns if c.startswith("pred_")]

        # Average prediction columns across all calibrations
        pred_array = np.zeros((len(base_df), len(pred_cols)))
        for df in calib_dfs:
            pred_array += df[pred_cols].values
        pred_array /= len(calib_dfs)

        # Create output dataframe
        ensemble_df = base_df.copy()
        for i, col in enumerate(pred_cols):
            ensemble_df[col] = pred_array[:, i]

        # Save with original filename pattern
        original_file = id_by_fold_best[(test_fold, test_best_idx)]
        output_path = final_dir / original_file.name
        ensemble_df.to_csv(output_path, index=False)
        print(f"\tSaved: {output_path.name}")

    # Ensemble OD domain
    for (test_fold, test_best_idx), calib_dfs in sorted(
        calibrated_predictions_od.items()
    ):
        print(
            f"\nEnsembling fold {test_fold} best{test_best_idx} (OD): {len(calib_dfs)} calibrations"
        )

        base_df = calib_dfs[0].copy()
        pred_cols = [c for c in base_df.columns if c.startswith("pred_")]

        pred_array = np.zeros((len(base_df), len(pred_cols)))
        for df in calib_dfs:
            pred_array += df[pred_cols].values
        pred_array /= len(calib_dfs)

        ensemble_df = base_df.copy()
        for i, col in enumerate(pred_cols):
            ensemble_df[col] = pred_array[:, i]

        original_file = od_by_fold_best[(test_fold, test_best_idx)]
        output_path = final_dir / original_file.name
        ensemble_df.to_csv(output_path, index=False)
        print(f"\tSaved: {output_path.name}")

    # Save comprehensive results
    df_results = pd.DataFrame(all_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "all_folds_calibration_results.csv"
    df_results.to_csv(results_csv, index=False)

    # Generate summary report
    report_path = final_dir / "CALIBRATION_REPORT.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ALL FOLDS CALIBRATION + ENSEMBLE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model directory: {args.model_dir}\n")
        f.write(f"Output directory: {final_dir}\n")
        f.write(f"Calibration method: {args.method}\n")
        f.write(f"Folds processed: {available_folds}\n")
        f.write(f"Ensemble size: {len(available_folds)} calibrations per fold\n\n")

        f.write("SUMMARY BY DOMAIN\n")
        f.write("-" * 80 + "\n")
        for domain in ["ID", "OD"]:
            df_dom = df_results[df_results["domain"] == domain]
            if len(df_dom) == 0:
                continue
            f.write(f"\n{domain} Domain:\n")
            f.write(f"\tCalibrations: {len(df_dom)}\n")
            f.write(
                f"\tECE before: {df_dom['ece_before'].mean():.4f} ± {df_dom['ece_before'].std():.4f}\n"
            )
            f.write(
                f"\tECE after:  {df_dom['ece_after'].mean():.4f} ± {df_dom['ece_after'].std():.4f}\n"
            )
            if args.method == "temperature":
                temps = df_dom["temperature"].dropna()
                if len(temps) > 0:
                    f.write(
                        f"\tAvg temperature: {temps.mean():.4f} ± {temps.std():.4f}\n"
                    )

    print(f"\n{'=' * 80}")
    print("ALL FOLDS CALIBRATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nEnsemble predictions saved to: {final_dir}")
    print(f"Individual fold calibrations saved to: {args.output_dir}/val_fold_*/")


if __name__ == "__main__":
    sys.exit(main())

# Example usage:
# python -m utils.calibrate_predictions \
#   --model-dir ./path/to/saved_models/my-experiment/ \
#   --output-dir ./path/to/saved_models/my-experiment-calibrated/ \
#   --val-fold all \
#   --method temperature
