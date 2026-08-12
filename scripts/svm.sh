#!/bin/bash

run_with_echo() {
    echo ""
    echo "----------------------------------------------------------------"
    echo -e "\033[1;34m[EXECUTING]\033[0m" # Blue text header
    echo "$@" | sed 's/ --/ \\\n  --/g'   # Pretty print with newlines for readability
    echo "----------------------------------------------------------------"
    
    # Execute the command passed as arguments
    "$@"
    
    # Check status
    local status=$?
    if [ $status -ne 0 ]; then
        echo -e "\033[1;31mError: Command failed with exit code $status\033[0m"
        exit $status
    fi
}

# 1. Argument Parsing
# ------------------------------------------------------------------------------
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options (ALL REQUIRED):"
    echo "  --training-csv-dir DIR    Path to training CV folds"
    echo "  --save-dir DIR            Directory to save SVM models"
    echo "  --intermediate-dir DIR    Path to extracted Transformer features"
    echo "  --eval-csv PATH           Path to external test CSV"
    echo "  --runname NAME            Name of the experiment/run"
    echo "  --metric METRIC           Optimization metric"
    echo "  --wandb-mode MODE         WandB mode (online/offline/disabled)"
    echo "  --project-name NAME       WandB project name"
    echo "  --n-trials INT            Number of Optuna trials"
    echo "  --fold STR                Fold to process (int or 'all')"
    echo "  -h, --help                Show this help message"
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --training-csv-dir) TRAINING_CSV_DIR="$2"; shift ;;
        --save-dir)         SAVE_DIR="$2"; shift ;;
        --intermediate-dir) INTERMEDIATE_DIR="$2"; shift ;;
        --eval-csv)         EVAL_CSV="$2"; shift ;;
        --runname)          RUNNAME="$2"; shift ;;
        --metric)           METRIC="$2"; shift ;;
        --wandb-mode)       WANDB_MODE="$2"; shift ;;
        --project-name)     PROJECT_NAME="$2"; shift ;;
        --n-trials)         N_TRIALS="$2"; shift ;;
        --fold)             FOLD="$2"; shift ;;
        -h|--help)          show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help ;;
    esac
    shift
done

# 2. Safety Check: Validation
# ------------------------------------------------------------------------------
# Since defaults are removed, we MUST ensure variables are set.
if [[ -z "$TRAINING_CSV_DIR" || -z "$SAVE_DIR" || -z "$INTERMEDIATE_DIR" || \
      -z "$EVAL_CSV" || -z "$RUNNAME" || -z "$METRIC" || \
      -z "$WANDB_MODE" || -z "$PROJECT_NAME" || -z "$N_TRIALS" || -z "$FOLD" ]]; then
    echo "Error: Missing required arguments."
    echo "Please run '$0 --help' to see all required flags."
    exit 1
fi

# 3. Environment Setup
# ------------------------------------------------------------------------------
# Activate your Python environment before running this script.
# Example: conda activate myenv

echo "----------------------------------------------------------------"
echo "Starting Pipeline: $RUNNAME"
echo "Fold: $FOLD | Metric: $METRIC | Trials: $N_TRIALS"
echo "----------------------------------------------------------------"

# 4. Execution: Training
# ------------------------------------------------------------------------------
echo "[1/2] Running SVM Training/Optimization..."
run_with_echo python train/train_svm.py \
  --training-csv-dir "$TRAINING_CSV_DIR" \
  --save-dir "$SAVE_DIR" \
  --intermediate-dir "$INTERMEDIATE_DIR" \
  --runname "$RUNNAME" \
  --fold "$FOLD" \
  --n-trials "$N_TRIALS" \
  --metric "$METRIC" \
  --wandb-mode "$WANDB_MODE" \
  --project-name "$PROJECT_NAME"

# Check return code of the python script
if [ $? -ne 0 ]; then
    echo "Error: Training failed. Aborting evaluation."
    exit 1
fi

# 5. Execution: Evaluation
# ------------------------------------------------------------------------------
echo "[2/2] Running Evaluation on External Set..."
run_with_echo python eval/eval_svm.py \
  --training-csv-dir "$TRAINING_CSV_DIR" \
  --intermediate-dir "$INTERMEDIATE_DIR" \
  --models-dir "$SAVE_DIR/$RUNNAME/" \
  --eval-csv "$EVAL_CSV" \
  --wandb-mode "$WANDB_MODE" \
  --project-name "$PROJECT_NAME"

echo "Pipeline finished successfully."

# ==============================================================================
# Example Usage
# ==============================================================================
# ./scripts/svm.sh \
#   --training-csv-dir "./path/to/10fold_CV/" \
#   --save-dir "./path/to/saved_models/svm" \
#   --intermediate-dir "./path/to/preprocessed/" \
#   --runname "my-experiment" \
#   --metric "neg_log_loss" \
#   --eval-csv "./path/to/test_set.csv" \
#   --wandb-mode "online" \
#   --project-name "project" \
#   --n-trials 200 \
#   --fold "all"