#!/usr/bin/env bash
# ACT BC - DELTA joint actions (joints 0-5 delta, gripper absolute).
#
# DATASET picks the v3 dataset under DATA_DIR; everything else is shared
# (_common_train.sh). Trial1 is the default; the iterative-training datasets
# use the same command with only the dataset changed (see README.md):
#   DATASET=PickCubeAndInsert_TS3_Iter1_deltajoint ./train_act_deltajoint.sh
#   DATASET=PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint ./train_act_deltajoint.sh
# USE_VAE=true trains ACT with its CVAE (Trial1 used false); "_vae" is appended
# to the output dir and job name so the runs don't collide.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common_train.sh"
DATASET="${DATASET:-PickCubeAndInsert_Station3_Trial1_deltajoint}"
USE_VAE="${USE_VAE:-false}"
RUN="${DATASET}$([ "${USE_VAE}" = true ] && echo _vae || true)"
NOTES="${NOTES:-ACT BC Station3 ${DATASET} Delta Joint Action, use_vae=${USE_VAE}, 256x256}"
echo "dataset : ${DATASET}  use_vae ${USE_VAE}"
cd "${LEROBOT_DIR}"
${TRAIN_CMD} \
  --config_path "${ACT_CONFIG}" \
  --dataset.repo_id="${DATA_DIR}/${DATASET}/" \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir="outputs/train/ACT_BC_${RUN}" \
  --batch_size=${BATCH_SIZE} \
  --num_workers=${NUM_WORKERS} \
  --steps=${STEPS} \
  --save_checkpoint=true \
  --save_freq=${SAVE_FREQ} \
  --eval_freq=0 \
  --log_freq=${LOG_FREQ} \
  --wandb.enable=true \
  --job_name="act_bc_${RUN}" \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.notes="${NOTES}" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=${CHUNK_SIZE} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --policy.use_vae=${USE_VAE} \
  --policy.optimizer_lr=${LR}
