#!/usr/bin/env bash
# ACT BC - ABSOLUTE joint actions. Station3 Trial1 (178 eps / 55177 frames).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common_train.sh"
cd "${LEROBOT_DIR}"
${TRAIN_CMD} \
  --config_path "${ACT_CONFIG}" \
  --dataset.repo_id="${DATA_DIR}/PickCubeAndInsert_Station3_Trial1/" \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/ACT_BC_PickCubeAndInsert_Station3_Trial1 \
  --batch_size=${BATCH_SIZE} \
  --num_workers=${NUM_WORKERS} \
  --steps=${STEPS} \
  --save_checkpoint=true \
  --save_freq=${SAVE_FREQ} \
  --eval_freq=0 \
  --log_freq=${LOG_FREQ} \
  --wandb.enable=true \
  --job_name=act_bc_pickcubeandinsert_station3_abs \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.notes="ACT BC Station3 Trial1 (5 sessions merged, 178 eps) Abs Joint Action, 256x256" \
  --wandb.disable_artifact=true \
  --policy.chunk_size=${CHUNK_SIZE} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --policy.use_vae=false \
  --policy.optimizer_lr=${LR}
