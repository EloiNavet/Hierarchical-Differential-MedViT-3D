#!/bin/bash

# Help message
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Train and evaluate Transformer models"
    echo
    echo "Required options:"
    echo "  --training-csv-dir DIR     Directory containing 10 fold csv files"
    echo "  --save-dir DIR             Directory to save models"
    echo "  --intermediate-dir DIR     Directory for intermediate files"
    echo "  --runname NAME             Name of the run"
    echo "  --eval-csv FILE           Path to evaluation csv"
    echo
    echo "Optional options:"
    echo "  --wandb-mode MODE         WandB mode (online, offline, disabled)"
    echo "  --project-name NAME       WandB project name"
    echo "  --cuda-devices DEVICES    CUDA devices to use (e.g., '0,1' or '2', default: '0,1')"
    echo "  --fold NUMBER             Specific fold number to train (optional)"
    echo "  --config FILE             Path to configuration file (optional)"
    echo "  --checkpoint FILE         Path to checkpoint file (optional)"
    echo "  --seed VALUE              Override training seed (use 'none' to disable determinism)"
    echo "  -h, --help               Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --training-csv-dir)
        TRAINING_CSV_DIR="$2"
        shift 2
        ;;
    --save-dir)
        SAVE_DIR="$2"
        shift 2
        ;;
    --intermediate-dir)
        INTERMEDIATE_DIR="$2"
        shift 2
        ;;
    --runname)
        RUNNAME="$2"
        shift 2
        ;;
    --eval-csv)
        EVAL_CSV="$2"
        shift 2
        ;;
    --wandb-mode)
        WANDB_MODE="$2"
        shift 2
        ;;
    --project-name)
        PROJECT_NAME="$2"
        shift 2
        ;;
    --cuda-devices)
        CUDA_DEVICES="$2"
        shift 2
        ;;
    --fold)
        FOLD="$2"
        shift 2
        ;;
    --config)
        CONFIG="$2"
        shift 2
        ;;
    --checkpoint)
        CHECKPOINT="$2"
        shift 2
        ;;
    --seed)
        SEED="$2"
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
done

# Check required arguments
if [ -z "$TRAINING_CSV_DIR" ] || [ -z "$SAVE_DIR" ] || [ -z "$INTERMEDIATE_DIR" ] || [ -z "$RUNNAME" ] || [ -z "$EVAL_CSV" ]; then
    echo "Error: Missing required arguments"
    usage
    exit 1
fi

# Set default values for optional arguments
WANDB_MODE=${WANDB_MODE:-"disabled"}
PROJECT_NAME=${PROJECT_NAME:-"project"}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1"}

# Handle seed overrides
SEED_PARAM=""
if [ ! -z "$SEED" ]; then
    SEED_LOWER=$(echo "$SEED" | tr '[:upper:]' '[:lower:]')
    if [[ "$SEED_LOWER" == "none" || "$SEED_LOWER" == "false" || "$SEED_LOWER" == "disable" || "$SEED_LOWER" == "disabled" ]]; then
        unset PYTHONHASHSEED
        SEED_PARAM="--seed none"
    else
        export PYTHONHASHSEED="$SEED"
        SEED_PARAM="--seed $SEED"
    fi
fi

# Calculate number of GPUs by counting comma-separated values
NUM_GPUS=$(echo $CUDA_DEVICES | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

# Create model save directory
MODEL_SAVE_DIR="$SAVE_DIR/$RUNNAME"

echo "Starting training with $NUM_GPUS GPU(s)..."

set -x


# Choose appropriate training command based on number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
    # Build command arguments string for display only
    CMD_ARGS="--save-dir \"$SAVE_DIR\" --training-csv-dir \"$TRAINING_CSV_DIR\" --intermediate-dir \"$INTERMEDIATE_DIR\" --runname \"$RUNNAME\" --wandb-mode \"$WANDB_MODE\" --project-name \"$PROJECT_NAME\""

    # Add optional fold, config, and checkpoint arguments to display string if provided
    if [ ! -z "$FOLD" ]; then
        CMD_ARGS="$CMD_ARGS --fold $FOLD"
    fi
    if [ ! -z "$CONFIG" ]; then
        CMD_ARGS="$CMD_ARGS --config $CONFIG"
    fi
    if [ ! -z "$CHECKPOINT" ]; then
        CMD_ARGS="$CMD_ARGS --checkpoint $CHECKPOINT"
    fi
    if [ ! -z "$SEED_PARAM" ]; then
        CMD_ARGS="$CMD_ARGS $SEED_PARAM"
    fi

    # Single GPU - run without distributed training
    LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=4 \
        python train/train_transformer.py \
        --save-dir "$SAVE_DIR" \
        --training-csv-dir "$TRAINING_CSV_DIR" \
        --intermediate-dir "$INTERMEDIATE_DIR" \
        --runname "$RUNNAME" \
        --wandb-mode "$WANDB_MODE" \
        --project-name "$PROJECT_NAME" \
        ${FOLD:+--fold $FOLD} \
        ${CONFIG:+--config $CONFIG} \
        ${CHECKPOINT:+--checkpoint $CHECKPOINT} \
        ${SEED_PARAM}
else
    # Build command arguments string for display only
    CMD_ARGS="--save-dir \"$SAVE_DIR\" --training-csv-dir \"$TRAINING_CSV_DIR\" --intermediate-dir \"$INTERMEDIATE_DIR\" --runname \"$RUNNAME\" --wandb-mode \"$WANDB_MODE\" --project-name \"$PROJECT_NAME\""

    # Add optional fold, config, and checkpoint arguments to display string if provided
    if [ ! -z "$FOLD" ]; then
        CMD_ARGS="$CMD_ARGS --fold $FOLD"
    fi
    if [ ! -z "$CONFIG" ]; then
        CMD_ARGS="$CMD_ARGS --config $CONFIG"
    fi
    if [ ! -z "$CHECKPOINT" ]; then
        CMD_ARGS="$CMD_ARGS --checkpoint $CHECKPOINT"
    fi
    if [ ! -z "$SEED_PARAM" ]; then
        CMD_ARGS="$CMD_ARGS $SEED_PARAM"
    fi

    # NCCL environment variables for better reliability
    # - NCCL_ASYNC_ERROR_HANDLING: Enable async error handling to detect hangs faster
    # - NCCL_IB_DISABLE: Disable InfiniBand (not needed for single-node multi-GPU)
    # - TORCH_NCCL_BLOCKING_WAIT: Use blocking wait for more reliable synchronization
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_IB_DISABLE=1
    export TORCH_NCCL_BLOCKING_WAIT=1

    # Multi-GPU - use distributed training with torchrun
    LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=4 torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train/train_transformer.py \
        --save-dir "$SAVE_DIR" \
        --training-csv-dir "$TRAINING_CSV_DIR" \
        --intermediate-dir "$INTERMEDIATE_DIR" \
        --runname "$RUNNAME" \
        --wandb-mode "$WANDB_MODE" \
        --project-name "$PROJECT_NAME" \
        ${FOLD:+--fold $FOLD} \
        ${CONFIG:+--config $CONFIG} \
        ${CHECKPOINT:+--checkpoint $CHECKPOINT} \
        ${SEED_PARAM}
fi

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

# Extract the CUDA devices into an array
IFS=',' read -ra CUDA_DEV_ARR <<<"$CUDA_DEVICES"
NUM_GPUS=${#CUDA_DEV_ARR[@]}

# Find all checkpoint files matching the pattern (could be improved)
CHECKPOINTS_ARR=()
while IFS= read -r line; do
    CHECKPOINTS_ARR+=("$line")
done < <(find "$MODEL_SAVE_DIR" -type f -name "model*best*.pt")

NUM_CHECKPOINTS=${#CHECKPOINTS_ARR[@]}

# Check if any checkpoints were found
if [ $NUM_CHECKPOINTS -eq 0 ]; then
    echo "Error: No checkpoints matching 'model*best*.pt' were found in $MODEL_SAVE_DIR"
    exit 1
fi

echo "Starting evaluation..."

# Track background evaluation processes so we can clean them up on Ctrl+C/termination
PIDS=()

cleanup() {
    local signal=${1:-TERM}
    if [ ${#PIDS[@]} -eq 0 ]; then
        return
    fi
    echo "Received $signal, terminating evaluation workers..." >&2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -s "$signal" -- "-$pid" 2>/dev/null || kill -s "$signal" "$pid" 2>/dev/null
        fi
    done
    wait
    PIDS=()
}

trap 'cleanup INT; exit 130' INT
trap 'cleanup TERM; exit 143' TERM

# Split checkpoints among GPUs and launch parallel evaluations
for ((i = 0; i < $NUM_GPUS; i++)); do
    GPU_ID=${CUDA_DEV_ARR[$i]}
    # Calculate start and end indices for this GPU's checkpoints
    START=$(((NUM_CHECKPOINTS * i) / NUM_GPUS))
    END=$(((NUM_CHECKPOINTS * (i + 1)) / NUM_GPUS))
    # Build the list of checkpoints for this GPU
    GPU_CHECKPOINTS=("${CHECKPOINTS_ARR[@]:$START:$((END - START))}")
    if [ ${#GPU_CHECKPOINTS[@]} -eq 0 ]; then
        continue
    fi
    # Build checkpoints argument
    CHECKPOINTS_ARG=""
    for ckpt in "${GPU_CHECKPOINTS[@]}"; do
        CHECKPOINTS_ARG="$CHECKPOINTS_ARG $ckpt"
    done

    # Only log evaluation to W&B if training was configured to use W&B.
    EVAL_WANDB_ARGS=()
    if [ "$WANDB_MODE" != "disabled" ]; then
        EVAL_WANDB_ARGS=(--project-name "$PROJECT_NAME" --log-to-wandb)
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python eval/eval_transformer.py \
        --eval-csv "$EVAL_CSV" \
        --training-csv-dir "$TRAINING_CSV_DIR" \
        --intermediate-dir "$INTERMEDIATE_DIR" \
        --cuda-device 0 \
        "${EVAL_WANDB_ARGS[@]}" \
        --checkpoints $CHECKPOINTS_ARG &
    PIDS+=($!)
done

# Wait for all evaluations to finish
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=1
done
trap - INT TERM
PIDS=()

if [ $FAIL -ne 0 ]; then
    echo "One or more evaluations failed."
    exit 1
fi

# Print eval results (ie the file in $SAVE_DIR/$RUNNAME/results_model_[a-zA-Z0-9]{8}_[fold].txt)
echo "Evaluation results:"
find "$MODEL_SAVE_DIR" -type f -name "results_model_*.txt" -exec cat {} \;

echo "Successfully completed training and evaluation!"

# Example usage:
# ./scripts/transformer.sh \
#     --training-csv-dir ./path/to/10fold_CV/ \
#     --save-dir ./path/to/saved_models/ \
#     --intermediate-dir ./path/to/preprocessed/ \
#     --runname my-experiment \
#     --eval-csv ./path/to/test_set.csv \
#     --wandb-mode disabled \
#     --project-name project \
#     --cuda-devices 0,1
#
# With config and checkpoint:
# ./scripts/transformer.sh \
#     --training-csv-dir ./path/to/10fold_CV/ \
#     --save-dir ./path/to/saved_models/ \
#     --intermediate-dir ./path/to/preprocessed/ \
#     --runname my-experiment \
#     --eval-csv ./path/to/test_set.csv \
#     --wandb-mode online \
#     --project-name project \
#     --cuda-devices 0,1 \
#     --config configs/medvit-rope-diff_attn-hierarchical_loss.yaml \
#     --fold 0
