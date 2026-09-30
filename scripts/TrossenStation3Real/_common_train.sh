#!/usr/bin/env bash
# Shared settings for the Station3 Trial1 ACT training scripts. Sourced, not run.
#
# These run on the SERVER (TVD29ServerRack), where `uv run` is the established
# way to launch lerobot-train (see training_prompts.md). The "never uv" rule is
# about the LAPTOP repo venv, not this. Override TRAIN_CMD if you prefer the
# venv binary directly:
#   TRAIN_CMD="/path/to/lerobot/.venv/bin/lerobot-train" ./train_act_abs.sh
LEROBOT_DIR="${LEROBOT_DIR:-/home/qte9489/Research/LEROBOT/lerobot}"
DATA_DIR="${DATA_DIR:-/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Station3_Trial1}"
TRAIN_CMD="${TRAIN_CMD:-uv run lerobot-train}"

# ACT hyper-parameters - identical to the Trial1 runs (training_prompts.md).
ACT_CONFIG="${ACT_CONFIG:-act_256_20260805_150824_and_153134_153727_PickAndInsertCube_override.yaml}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STEPS="${STEPS:-75000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
N_ACTION_STEPS="${N_ACTION_STEPS:-20}"
LR="${LR:-1e-4}"
WANDB_PROJECT="${WANDB_PROJECT:-Trossen_PickAndInsertCube_Real}"

echo "lerobot : ${LEROBOT_DIR}"
echo "data    : ${DATA_DIR}"
echo "steps   : ${STEPS}  batch ${BATCH_SIZE}  chunk ${CHUNK_SIZE}/${N_ACTION_STEPS}  lr ${LR}"
