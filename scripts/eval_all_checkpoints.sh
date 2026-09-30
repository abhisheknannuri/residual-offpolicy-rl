#!/usr/bin/env bash
# Usage:
#   To run for a specific folder:
#     ./eval_all_checkpoints.sh <path_to_checkpoints_dir> [output_dir_name]
#
#   To run for multiple pre-configured folders:
#     Configure the CHECKPOINT_PAIRS array below and run:
#     ./eval_all_checkpoints.sh

set -euo pipefail

# --- CONFIGURATION FOR MULTIPLE EVALUATIONS ---
# Define pairs in the format: "path_to_checkpoints_dir output_dir_name"
# You can add as many pairs as you want. Order of execution is preserved.
CHECKPOINT_PAIRS=(
    "/home/qte9489/Research/LEROBOT/lerobot/outputs/train/policy_bc/checkpoints/ policy_bc_50_episode_train"
)
# ----------------------------------------------

# Determine if inputs are passed via command line
if [ "$#" -ge 1 ]; then
    CHECKPOINT_BASE_DIR=$1
    OUTPUT_NAME=${2:-"batch_eval_$(date +%Y%m%d_%H%M%S)"}
    RUN_PAIRS=("$CHECKPOINT_BASE_DIR $OUTPUT_NAME")
else
    if [ ${#CHECKPOINT_PAIRS[@]} -eq 0 ]; then
        echo "Usage: $0 <path_to_checkpoints_dir> [output_dir_name]"
        echo "Or configure the CHECKPOINT_PAIRS array inside the script and run without arguments."
        exit 1
    fi
    RUN_PAIRS=("${CHECKPOINT_PAIRS[@]}")
fi

for PAIR in "${RUN_PAIRS[@]}"; do
    # Split the pair into checkpoint dir and output name
    read -r CHECKPOINT_BASE_DIR OUTPUT_NAME <<< "$PAIR"
    
    # Resolve base dir to absolute path if it exists
    if [ -d "$CHECKPOINT_BASE_DIR" ]; then
        CHECKPOINT_BASE_DIR=$(realpath "$CHECKPOINT_BASE_DIR")
    fi
    
    OUTPUT_DIR="outputs/eval/${OUTPUT_NAME}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "=========================================================="
    echo "Starting batch evaluation for checkpoints in: $CHECKPOINT_BASE_DIR"
    echo "Results will be saved to: $OUTPUT_DIR"
    echo "=========================================================="
    
    # Find all directories in the base dir that consist only of numbers (e.g., 005000, 010000)
    # and sort them numerically
    CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" 2>/dev/null | grep -E '^[0-9]+$' | sort -n || true)
    
    if [ -z "$CHECKPOINTS" ]; then
        echo "No numeric checkpoint directories found in $CHECKPOINT_BASE_DIR. Skipping."
        continue
    fi
    
    # Run evaluation for each checkpoint
    for CKPT in $CHECKPOINTS; do
        CKPT_PATH="${CHECKPOINT_BASE_DIR}/${CKPT}"
        echo "----------------------------------------------------------"
        echo "Evaluating Checkpoint: $CKPT"
        echo "Path: $CKPT_PATH"
        echo "----------------------------------------------------------"
        
        # Check if pretrained_model or policy directories exist under the checkpoint folder
        if [ -d "${CKPT_PATH}/pretrained_model" ]; then
            EVAL_PATH="${CKPT_PATH}/pretrained_model"
        elif [ -d "${CKPT_PATH}/policy" ]; then
            EVAL_PATH="${CKPT_PATH}/policy"
        else
            EVAL_PATH="${CKPT_PATH}"
        fi
        
        python scripts/eval_policy_can.py \
            --checkpoint_dir "$EVAL_PATH" \
            --output_dir "$OUTPUT_DIR/ckpt_$CKPT" \
            --seed "none" \
            --eval_num_envs 8 \
            --eval_camera_size 84 \
            --eval_horizon 200
        
        echo "Finished evaluating checkpoint $CKPT."
    done
    
    echo "Batch evaluation complete for $CHECKPOINT_BASE_DIR. Results are in $OUTPUT_DIR"
done
