#!/usr/bin/env bash
# ============================================================================
# train_residual_rl_square_stage_sparse.sh — Residual RL for Square Stage Task
# ============================================================================
# Fine-tunes a frozen BC base policy with a small learned residual action
# correction using TD3 (Twin Delayed DDPG) with off-policy data, starting from
# the placement stage of the NutAssemblySquare task (STAGE_RESET=1).
# ============================================================================
set -euo pipefail

# Export environment variable to reset environment directly at the placement stage
export STAGE_RESET=1

# ── Base BC Policy ───────────────────────────────────────────────────────────
# Path to the pretrained Behavior Cloning checkpoint (ACT model)
BASE_LOCAL_PATH="/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/act_bc_suarenut/checkpoints/045000/pretrained_model"
BASE_WANDB_ID="robomimic-square-bc/dzbkdpwp"
BASE_WT_TYPE="latest"
BASE_WT_VERSION="latest"

# ── Hydra Config ─────────────────────────────────────────────────────────────
CONFIG_NAME="residual_td3_square_config"

# ── Offline Dataset (Sliced Stage Dataset) ───────────────────────────────────
OFFLINE_DATASET="nutsquare256_v21_stage"
OFFLINE_DATA_ROOT="/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil/NutSquare/nutsquare256_v21_stage"
OFFLINE_EPISODES=250
IMAGE_SIZE=84

# ── Exploration Noise & Action Scales ───────────────────────────────────────
# ACTION_SCALE: Maximum magnitude of the residual action correction (e.g. 0.2 means ±20% range)
ACTION_SCALE=${ACTION_SCALE:-0.2}

# STDDEV_MAX/MIN: Exploration noise standard deviation (truncated Gaussian noise added during training)
STDDEV_MAX=${STDDEV_MAX:-0.025}
STDDEV_MIN=${STDDEV_MIN:-0.025}
STDDEV_STEP=${STDDEV_STEP:-200000}
STDDEV_CLIP=${STDDEV_CLIP:-0.3}

# ── Core RL Hyperparameters ──────────────────────────────────────────────────
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-300000}
N_STEP=${N_STEP:-3}
GAMMA=${GAMMA:-0.995}
UTD=${UTD:-4}
ACTOR_UPDATES_PER_ITER=1
UPDATE_EVERY_N_STEPS=1
BATCH_SIZE=${BATCH_SIZE:-256}
BUFFER_SIZE=${BUFFER_SIZE:-80000}
PREFETCH=4

# ── Warmup & Learning Rates ─────────────────────────────────────────────────
LEARNING_STARTS=${LEARNING_STARTS:-10000} # Reduced from 10K since we start close to the goal
CRITIC_WARMUP=${CRITIC_WARMUP:-10000}
ACTOR_LR_WARMUP_STEPS=${ACTOR_LR_WARMUP_STEPS:-0}

USE_BASE_POLICY_FOR_WARMUP="true"
RANDOM_ACTION_NOISE_SCALE=0.2

ACTOR_LR=1e-6
CRITIC_LR=1e-4
CRITIC_TARGET_TAU=0.005
ACTOR_LAST_LAYER_INIT=0.0
ACTION_L2_REG=0.0
CRITIC_GRAD_CLIP=100.0
ACTOR_GRAD_CLIP=100.0
BC_LOSS_COEF=0.0
FREEZE_ENCODER="true"
TARGET_ACTION_NOISE="true"

# ── Critic Head Config ───────────────────────────────────────────────────────
NUM_Q_HEADS=10
MIN_Q_HEADS=2
POLICY_GRADIENT_TYPE="ensemble_mean"
CRITIC_LOSS_TYPE="mse"
V_MIN=0.0
V_MAX=1.0

# ── Normalization Safeguards ────────────────────────────────────────────────
MIN_ACTION_RANGE=0.1
MIN_STATE_STD=0.1

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-8}
EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
EVAL_FIRST=${EVAL_FIRST:-"true"}
SAVE_VIDEO=${SAVE_VIDEO:-"true"}

# ── Environment ──────────────────────────────────────────────────────────────
HEADLESS="true"
NUM_ENVS=1
VIDEO_KEY="observation.images.agentview"
RL_CAMERA="[observation.images.agentview,observation.images.robot0_eye_in_hand]"
REWARD_SHAPING="false" # Use sparse reward

# ── Checkpointing & Logging ──────────────────────────────────────────────────
SAVE_FREQ=${SAVE_FREQ:-10000}
NO_CLEANUP="true"
WANDB_PROJECT=${WANDB_PROJECT:-"robomimic-square-residual-td3"}
WANDB_MODE=${WANDB_MODE:-"online"}
DEBUG=${DEBUG:-"false"}
TORCH_DETERMINISTIC="false"

# ── Setup parameters from Env/Args if provided ──
WANDB_NAME=${WANDB_NAME:-"resfit_square_stage"}
WANDB_GROUP=${WANDB_GROUP:-"resfit_square_stage"}
SEED=${SEED:-""}
RESUME_CKPT=${RESUME_CKPT:-""}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Residual RL Training — TD3 (NutAssemblySquare Stage Reset)   "
echo "║  Base policy local path: ${BASE_LOCAL_PATH}                  "
echo "║  Offline dataset: ${OFFLINE_DATASET}                          "
echo "║  Steps: ${TOTAL_TIMESTEPS}  |  UTD: ${UTD}  |  Buffer: ${BUFFER_SIZE}"
echo "║  Action scale: ${ACTION_SCALE}  |  Actor LR: ${ACTOR_LR}   "
echo "║  Offline episodes: ${OFFLINE_EPISODES}                      "
echo "║  Exploration Stddev: ${STDDEV_MAX} (clip: ${STDDEV_CLIP})    "
echo "║  WandB: ${WANDB_PROJECT}/${WANDB_NAME}                      "
echo "╚══════════════════════════════════════════════════════════════╝"

# Build the command with all overrides
CMD=(
    /home/qte9489/personal_abhi/temp/residual-offpolicy-rl/.venv/bin/python resfit/rl_finetuning/scripts/train_residual_td3.py
    --config-name="${CONFIG_NAME}"

    # ── Base policy ──
    base_policy.wandb_id="${BASE_WANDB_ID}"
    base_policy.wt_type="${BASE_WT_TYPE}"
    base_policy.wt_version="${BASE_WT_VERSION}"
    base_policy.local_path="${BASE_LOCAL_PATH}"

    # ── Algorithm ──
    algo.total_timesteps="${TOTAL_TIMESTEPS}"
    algo.n_step="${N_STEP}"
    algo.gamma="${GAMMA}"
    algo.num_updates_per_iteration="${UTD}"
    algo.actor_updates_per_iteration="${ACTOR_UPDATES_PER_ITER}"
    algo.update_every_n_steps="${UPDATE_EVERY_N_STEPS}"
    algo.batch_size="${BATCH_SIZE}"
    algo.buffer_size="${BUFFER_SIZE}"
    algo.learning_starts="${LEARNING_STARTS}"
    algo.offline_fraction="${OFFLINE_FRACTION:-0.5}"
    algo.sampling_strategy="${SAMPLING:-uniform}"
    algo.prefetch_batches="${PREFETCH}"
    algo.random_action_noise_scale="${RANDOM_ACTION_NOISE_SCALE}"
    algo.use_base_policy_for_warmup="${USE_BASE_POLICY_FOR_WARMUP}"
    algo.progressive_clipping_steps=0

    # ── Exploration noise schedule ──
    algo.stddev_max="${STDDEV_MAX}"
    algo.stddev_min="${STDDEV_MIN}"
    algo.stddev_step="${STDDEV_STEP}"
    algo.critic_warmup_steps="${CRITIC_WARMUP}"
    algo.actor_lr_warmup_steps="${ACTOR_LR_WARMUP_STEPS}"

    # ── Actor config ──
    agent.actor_lr="${ACTOR_LR}"
    agent.critic_lr="${CRITIC_LR}"
    agent.critic_target_tau="${CRICIT_TARGET_TAU:-$CRITIC_TARGET_TAU}"
    agent.stddev_clip="${STDDEV_CLIP}"
    agent.actor.action_scale="${ACTION_SCALE}"
    agent.actor.actor_last_layer_init_scale="${ACTOR_LAST_LAYER_INIT}"
    agent.actor.action_l2_reg_weight="${ACTION_L2_REG}"
    agent.critic_grad_clip_norm="${CRITIC_GRAD_CLIP}"
    agent.actor_grad_clip_norm="${ACTOR_GRAD_CLIP}"
    agent.bc_loss_coef="${BC_LOSS_COEF}"
    agent.freeze_encoder="${FREEZE_ENCODER}"
    agent.target_action_noise="${TARGET_ACTION_NOISE}"

    # ── Critic config ──
    agent.critic.num_q="${NUM_Q_HEADS}"
    agent.critic.min_q_heads="${MIN_Q_HEADS}"
    agent.critic.policy_gradient_type="${POLICY_GRADIENT_TYPE}"
    agent.critic.loss.type="${CRITIC_LOSS_TYPE}"
    agent.critic.loss.v_min="${V_MIN}"
    agent.critic.loss.v_max="${V_MAX}"

    # ── Offline data ──
    offline_data.name="${OFFLINE_DATASET}"
    offline_data.root="${OFFLINE_DATA_ROOT}"
    offline_data.num_episodes="${OFFLINE_EPISODES}"
    offline_data.min_action_range="${MIN_ACTION_RANGE}"
    offline_data.min_state_std="${MIN_STATE_STD}"
    offline_data.image_size="${IMAGE_SIZE}"

    # ── Evaluation ──
    eval_num_envs="${EVAL_NUM_ENVS}"
    eval_num_episodes="${EVAL_NUM_EPISODES}"
    eval_interval_every_steps="${EVAL_INTERVAL}"
    eval_first="${EVAL_FIRST}"
    save_video="${SAVE_VIDEO}"

    # ── Environment ──
    headless="${HEADLESS}"
    num_envs="${NUM_ENVS}"
    video_key="${VIDEO_KEY}"
    "rl_camera=${RL_CAMERA}"
    reward_shaping="${REWARD_SHAPING}"

    # ── Checkpointing ──
    save_freq="${SAVE_FREQ}"
    no_cleanup="${NO_CLEANUP}"

    # ── WandB ──
    wandb.project="${WANDB_PROJECT}"
    wandb.mode="${WANDB_MODE}"

    # ── Debug ──
    debug="${DEBUG}"
    torch_deterministic="${TORCH_DETERMINISTIC}"
)

# Add optional overrides only if set
[[ -n "${WANDB_NAME}" ]]   && CMD+=(wandb.name="${WANDB_NAME}")
[[ -n "${WANDB_GROUP}" ]]  && CMD+=(wandb.group="${WANDB_GROUP}")
[[ -n "${WANDB_ENTITY:-}" ]] && CMD+=(wandb.entity="${WANDB_ENTITY}")
[[ -n "${SEED}" ]]         && CMD+=(seed="${SEED}")
[[ -n "${RESUME_CKPT}" ]]  && CMD+=(resume_ckpt="${RESUME_CKPT}")

"${CMD[@]}"

echo ""
echo "✓ Residual RL training complete."
echo "  Check WandB: ${WANDB_PROJECT}"
