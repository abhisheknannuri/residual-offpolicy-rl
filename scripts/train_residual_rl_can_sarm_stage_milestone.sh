#!/usr/bin/env bash
# ============================================================================
# train_residual_rl_can_sarm_stage_milestone.sh
#   Residual RL (TD3/RLPD) for Can with a STAGE-AWARE *MILESTONE* reward
#   (ratchet one-time stage payouts + terminal success bonus — NOT PBRS).
# ============================================================================
#
# This is the ablation counterpart to train_residual_rl_can_sarm_stage_pbrs.sh.
# Everything (base policy, hyper-params, offline dataset, MSE critic,
# terminated-only bootstrap) is IDENTICAL — only the reward differs:
#   * PBRS script     : r = r_sparse + gamma*phi(s') - phi(s)   (policy-invariant)
#   * THIS script      : ratchet milestone payouts on first monotonic stage entry
#                        + a dominant terminal success bonus     (NOT invariant)
#
# Goal: test whether a non-PBRS, stage-aware DENSE/DISCRETE reward changes
# sample efficiency vs. sparse / PBRS in the residual-RL setting.
#
# IMPORTANT: relabel the offline dataset with the SAME milestone params first:
#   python scripts/label_offline_pbrs.py --reward_mode milestone \
#       --milestone_payouts 0.0 0.3 0.3 0.4 --milestone_success_bonus 1.0 \
#       --dataset_root <v21> --server_url http://127.0.0.1:8003 \
#       --hysteresis_k 4 --query_every_k 4 --output ./milestone_sarm_progress.parquet
# The RL config-guard hard-errors if the parquet params differ from below.
# ============================================================================
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. BASE BC POLICY                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
BASE_WANDB_ID="resfit-robomimic-can-bc/pzqj1tmd"
BASE_WT_TYPE="latest"
BASE_WT_VERSION="latest"
# Local ACT checkpoint (takes priority over W&B when non-empty).
BASE_LOCAL_PATH="/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_rabc/checkpoints/050000/pretrained_model"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. HYDRA CONFIG                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
CONFIG_NAME="residual_td3_can_config"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3-4. ACTION / STATE NORMALIZATION                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ACTION_SCALE=0.2
MIN_ACTION_RANGE=0.1
MIN_STATE_STD=0.1

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. CORE RL HYPERPARAMETERS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
TOTAL_TIMESTEPS=30000
N_STEP=3
GAMMA=0.995
UTD=4

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. EXPLORATION NOISE                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
STDDEV_MAX=0.025
STDDEV_MIN=0.025
STDDEV_STEP=200000
STDDEV_CLIP=0.3

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. WARMUP PHASES                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
LEARNING_STARTS=5000
USE_BASE_POLICY_FOR_WARMUP="true"
RANDOM_ACTION_NOISE_SCALE=0.2
CRITIC_WARMUP=5000

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. REPLAY BUFFER & OFFLINE DATA                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
OFFLINE_DATASET="sarm-can-v21"
BUFFER_SIZE=40000
BATCH_SIZE=128
SAMPLING="uniform"
OFFLINE_EPISODES=50
OFFLINE_FRACTION=0.5

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9. LEARNING RATES                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ACTOR_LR=1e-6
CRITIC_LR=1e-4
CRITIC_TARGET_TAU=0.005

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  10. ACTOR NETWORK CONFIG                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ACTOR_LAST_LAYER_INIT=0.0
PROGRESSIVE_CLIPPING=0

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  11. CRITIC NETWORK CONFIG                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
NUM_Q_HEADS=10
MIN_Q_HEADS=2
POLICY_GRADIENT_TYPE="ensemble_mean"
# MSE critic ignores v_min/v_max -> the milestone return (up to 2.0) is NOT clamped.
# If you switch to c51/hl_gauss, set V_MAX >= the max milestone return.
CRITIC_LOSS_TYPE="mse"
V_MIN=0.0
V_MAX=80.0

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  12. EVALUATION                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
EVAL_NUM_ENVS=4
EVAL_NUM_EPISODES=10
EVAL_INTERVAL=5000
EVAL_FIRST="true"
SAVE_VIDEO="true"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  13. CHECKPOINTING                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
SAVE_FREQ=5000
NO_CLEANUP="true"
RESUME_CKPT=""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  14. ENVIRONMENT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
HEADLESS="true"
RL_CAMERA='[observation.images.agentview,observation.images.robot0_eye_in_hand]'
VIDEO_KEY="observation.images.agentview"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  15. REWARD SHAPING (simulator dense reward — off; we use the model)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
REWARD_SHAPING="false"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  16. IMAGE RESOLUTION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
IMAGE_SIZE=84

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  17. WANDB                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
WANDB_PROJECT="robomimic-can-residual-sarm-stage-milestone"
WANDB_NAME=""
WANDB_GROUP=""
WANDB_ENTITY=""
WANDB_MODE="online"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  18. SEED & DEBUG                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
SEED="42"
TORCH_DETERMINISTIC="false"
DEBUG="false"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  19. ADVANCED                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
NUM_ENVS=1
PREFETCH=4
ACTOR_UPDATES_PER_ITER=1
UPDATE_EVERY_N_STEPS=1
CRITIC_GRAD_CLIP=1.0
ACTOR_GRAD_CLIP=1.0
ACTOR_LR_WARMUP_STEPS=0
BC_LOSS_COEF=0.0
ACTION_L2_REG=0.0
FREEZE_ENCODER="false"
TARGET_ACTION_NOISE="true"

# ╔══════════════════════════════════════════════════════════════╗
# ║  20. STAGE-AWARE EXTERNAL REWARD MODEL — MILESTONE MODE                ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Start the SARM/TCC reward server FIRST on ${REWARD_SERVER_URL}. Enabling this
# also sets algo.terminated_only_bootstrap=true (truncation must bootstrap, only
# true success terminates the bootstrap — required for milestone AND PBRS).
#
REWARD_MODEL_ENABLED="true"
REWARD_MODEL_BACKEND="sarm"                  # "sarm" | "tcc"
REWARD_SERVER_URL="http://127.0.0.1:8003"
REWARD_TASK_PROMPT="pick up the can and place it in the bin"
REWARD_QUERY_EVERY_K=4                        # query every k env steps (20Hz -> 5Hz)
REWARD_HYSTERESIS_K=4                         # consecutive confs to advance a stage
REWARD_CONF_THRESHOLD=0.80                    # min stage confidence for an upgrade
REWARD_POTENTIALS="[0.0,0.05,0.1,0.15]"       # unused by milestone; sets num_stages=4
REWARD_KEEP_SPARSE_TERM="true"                # (PBRS-only; ignored in milestone mode)
REWARD_IMAGE_VFLIP="false"
TERMINATED_ONLY_BOOTSTRAP="true"

# ── MILESTONE reward design ────────────────────────────────────────────────
# Ratchet: pay MILESTONE_PAYOUTS[k] once on the first monotonic entry into stage
# k (skipped stages summed); MILESTONE_SUCCESS_BONUS on TRUE simulator success.
# STRONG dense test -> total return 2.0 (0.3+0.3+0.4 + 1.0). MSE critic doesn't
# clamp, so 2.0 is fine. For a [0,1]-safe variant (e.g. if using c51 v_max=1.0)
# use MILESTONE_PAYOUTS="[0.0,0.1,0.1,0.1]" and MILESTONE_SUCCESS_BONUS="0.7".
REWARD_MODE="milestone"
MILESTONE_PAYOUTS="[0.0,0.3,0.3,0.4]"
MILESTONE_SUCCESS_BONUS="1.0"

# Reward-model DEBUG: randomly record annotated TRAINING rollouts.
SAVE_TRAINING_ROLLOUTS="true"
TRAINING_ROLLOUT_PROB="0.2"

# Offline milestone reward (MUST match the labeler's --reward_mode/--milestone_*).
OFFLINE_REWARD_PARQUET="./milestone_sarm_progress.parquet"
OFFLINE_REWARD_COLUMN="reward_milestone"

# Local offline dataset (LeRobot v2.1) + key remap.
OFFLINE_ROOT="/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil/PickPlaceCan/SARM-robosuite-can-mh-stages_v21"
REMAP_PRESET="sarm_v21"

# ============================================================================
# BUILD AND RUN
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Residual RL — TD3 for Can (STAGE-AWARE MILESTONE reward)     "
echo "║  Base policy: ${BASE_WANDB_ID} (${BASE_WT_TYPE})            "
echo "║  Steps: ${TOTAL_TIMESTEPS}  |  UTD: ${UTD}  |  γ: ${GAMMA} "
echo "║  Milestone payouts: ${MILESTONE_PAYOUTS}  |  success: ${MILESTONE_SUCCESS_BONUS}"
echo "║  Buffer: ${BUFFER_SIZE}  |  Offline eps: ${OFFLINE_EPISODES}"
echo "║  WandB: ${WANDB_PROJECT}                                     "
echo "╚══════════════════════════════════════════════════════════════╝"

CMD=(
    python resfit/rl_finetuning/scripts/train_residual_td3.py
    --config-name="${CONFIG_NAME}"

    # ── Base policy ──
    base_policy.wandb_id="${BASE_WANDB_ID}"
    base_policy.wt_type="${BASE_WT_TYPE}"
    base_policy.wt_version="${BASE_WT_VERSION}"

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
    algo.offline_fraction="${OFFLINE_FRACTION}"
    algo.sampling_strategy="${SAMPLING}"
    algo.prefetch_batches="${PREFETCH}"
    algo.random_action_noise_scale="${RANDOM_ACTION_NOISE_SCALE}"
    algo.use_base_policy_for_warmup="${USE_BASE_POLICY_FOR_WARMUP}"
    algo.progressive_clipping_steps="${PROGRESSIVE_CLIPPING}"

    # ── Exploration noise schedule ──
    algo.stddev_max="${STDDEV_MAX}"
    algo.stddev_min="${STDDEV_MIN}"
    algo.stddev_step="${STDDEV_STEP}"
    algo.critic_warmup_steps="${CRITIC_WARMUP}"
    algo.actor_lr_warmup_steps="${ACTOR_LR_WARMUP_STEPS}"

    # ── Actor config ──
    agent.actor_lr="${ACTOR_LR}"
    agent.critic_lr="${CRITIC_LR}"
    agent.critic_target_tau="${CRITIC_TARGET_TAU}"
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
    offline_data.num_episodes="${OFFLINE_EPISODES}"
    offline_data.min_action_range="${MIN_ACTION_RANGE}"
    offline_data.min_state_std="${MIN_STATE_STD}"

    # ── Evaluation ──
    eval_num_envs="${EVAL_NUM_ENVS}"
    eval_num_episodes="${EVAL_NUM_EPISODES}"
    eval_interval_every_steps="${EVAL_INTERVAL}"
    eval_first="${EVAL_FIRST}"
    save_video="${SAVE_VIDEO}"
    save_training_rollouts="${SAVE_TRAINING_ROLLOUTS}"
    training_rollout_prob="${TRAINING_ROLLOUT_PROB}"

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

# Optional overrides
[[ -n "${WANDB_NAME}" ]]   && CMD+=(wandb.name="${WANDB_NAME}")
[[ -n "${WANDB_GROUP}" ]]  && CMD+=(wandb.group="${WANDB_GROUP}")
[[ -n "${WANDB_ENTITY}" ]] && CMD+=(wandb.entity="${WANDB_ENTITY}")
[[ -n "${SEED}" ]]         && CMD+=(seed="${SEED}")
[[ -n "${RESUME_CKPT}" ]]  && CMD+=(resume_ckpt="${RESUME_CKPT}")
[[ -n "${IMAGE_SIZE}" ]]   && CMD+=(offline_data.image_size="${IMAGE_SIZE}")

# Local base-policy checkpoint (takes priority over W&B)
[[ -n "${BASE_LOCAL_PATH}" ]] && CMD+=(base_policy.local_path="${BASE_LOCAL_PATH}")

# Local offline dataset + key remap
[[ -n "${OFFLINE_ROOT}" ]] && CMD+=(offline_data.root="${OFFLINE_ROOT}" offline_data.remap_preset="${REMAP_PRESET}")

# Offline milestone reward for the offline buffer
[[ -n "${OFFLINE_REWARD_PARQUET}" ]] && CMD+=(offline_data.reward_parquet="${OFFLINE_REWARD_PARQUET}" offline_data.reward_column="${OFFLINE_REWARD_COLUMN}")

# Stage-aware external reward model (MILESTONE). Requires the reward server running.
if [[ "${REWARD_MODEL_ENABLED}" == "true" ]]; then
    CMD+=(
        reward_model.enabled=true
        reward_model.backend="${REWARD_MODEL_BACKEND}"
        reward_model.server_url="${REWARD_SERVER_URL}"
        reward_model.task_prompt="${REWARD_TASK_PROMPT}"
        reward_model.query_every_k="${REWARD_QUERY_EVERY_K}"
        reward_model.hysteresis_k="${REWARD_HYSTERESIS_K}"
        reward_model.conf_threshold="${REWARD_CONF_THRESHOLD}"
        "reward_model.potentials=${REWARD_POTENTIALS}"
        reward_model.keep_sparse_term="${REWARD_KEEP_SPARSE_TERM}"
        reward_model.image_vflip="${REWARD_IMAGE_VFLIP}"
        reward_model.reward_mode="${REWARD_MODE}"
        "reward_model.milestone_payouts=${MILESTONE_PAYOUTS}"
        reward_model.milestone_success_bonus="${MILESTONE_SUCCESS_BONUS}"
        algo.terminated_only_bootstrap="${TERMINATED_ONLY_BOOTSTRAP}"
    )
fi

"${CMD[@]}"

echo ""
echo "✓ Residual RL (milestone) training complete."
echo "  Check WandB: ${WANDB_PROJECT}"
