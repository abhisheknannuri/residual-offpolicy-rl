#!/usr/bin/env bash
# =============================================================================
# _run_rl.sh - assembles and runs the training command.
#
# Sourced (not executed) by every mode script, after it has set its overrides.
# Never run directly.
#
# One place builds the command, so the five modes cannot drift apart on a flag
# that neither of us would notice - which is the failure this whole folder is
# arranged to prevent.
#
# Set DIST=1 (see agentlace/_common_dist.sh) to run the distributed entrypoint
# instead; it is the same config plus role/transport arguments.
# =============================================================================
set -euo pipefail

cd "${REPO_ROOT}"

if [[ "${DIST:-0}" == "1" ]]; then
    ENTRY="resfit/rl_finetuning/scripts/train_residual_td3_distributed.py"
    CONFIG_NAME="residual_td3_trossen_real_dist_config"
else
    ENTRY="resfit/rl_finetuning/scripts/train_residual_td3.py"
    CONFIG_NAME="residual_td3_trossen_real_config"
fi

echo "┌──────────────────────────────────────────────────────────────"
echo "│ MODE            ${MODE_NAME}"
echo "│ ${MODE_DESC}"
echo "├──────────────────────────────────────────────────────────────"
echo "│ entry           ${ENTRY}"
echo "│ python          ${PYTHON_BIN}"
[[ "${DIST:-0}" == "1" ]] && echo "│ role            ${ROLE}  (learner ${LEARNER_IP}:${LEARNER_PORT})"
echo "│ station         ${STATION_CONFIG_NAME}   policy_server ${POLICY_SERVER_URL}"
echo "│ dataset         ${OFFLINE_DATASET}  (${OFFLINE_EPISODES} episodes)"
echo "├── hashed (must be identical across every script here) ────────"
echo "│ learning_starts ${LEARNING_STARTS}      real_max_steps ${REAL_MAX_STEPS}"
echo "│ n_step ${N_STEP}  gamma ${GAMMA}  batch ${BATCH_SIZE}  offline_frac ${OFFLINE_FRACTION}"
echo "│ buffer_size ${BUFFER_SIZE}  sampling ${SAMPLING}  wandb.name '${WANDB_NAME}'"
echo "├── this mode ──────────────────────────────────────────────────"
echo "│ offline_pretrain_only ${OFFLINE_PRETRAIN_ONLY}   train_offline_rl ${DO_OFFLINE_RL} (${OFFLINE_RL_STEPS} steps)"
echo "│ critic_warmup_steps   ${CRITIC_WARMUP}   total_timesteps ${TOTAL_TIMESTEPS}"
echo "│ resume_ckpt           ${RESUME_CKPT:-<none>}"
echo "└──────────────────────────────────────────────────────────────"

CMD=(
    "${PYTHON_BIN}" "${ENTRY}"
    --config-name="${CONFIG_NAME}"
    real_hardware=true
    station_config_name="${STATION_CONFIG_NAME}"
    policy_server_url="${POLICY_SERVER_URL}"
    real_action_space="${REAL_ACTION_SPACE}"
    real_max_steps="${REAL_MAX_STEPS}"
    real_settle_time_s="${REAL_SETTLE_TIME_S}"
    enable_intervention="${ENABLE_INTERVENTION}"
    algo.total_timesteps="${TOTAL_TIMESTEPS}"
    algo.n_step="${N_STEP}"
    algo.gamma="${GAMMA}"
    algo.num_updates_per_iteration="${UTD}"
    algo.actor_updates_per_iteration="${ACTOR_UPDATES_PER_ITER}"
    algo.update_every_n_steps="${UPDATE_EVERY_N_STEPS}"
    algo.batch_size="${BATCH_SIZE}"
    algo.buffer_size="${BUFFER_SIZE}"
    algo.learning_starts="${LEARNING_STARTS}"
    algo.online_warmup_save_freq="${ONLINE_WARMUP_SAVE_FREQ}"
    algo.offline_fraction="${OFFLINE_FRACTION}"
    algo.sampling_strategy="${SAMPLING}"
    algo.prefetch_batches="${PREFETCH}"
    algo.random_action_noise_scale="${RANDOM_ACTION_NOISE_SCALE}"
    algo.use_base_policy_for_warmup="${USE_BASE_POLICY_FOR_WARMUP}"
    algo.progressive_clipping_steps="${PROGRESSIVE_CLIPPING}"
    algo.stddev_max="${STDDEV_MAX}"
    algo.stddev_min="${STDDEV_MIN}"
    algo.stddev_step="${STDDEV_STEP}"
    algo.critic_warmup_steps="${CRITIC_WARMUP}"
    algo.actor_lr_warmup_steps="${ACTOR_LR_WARMUP_STEPS}"
    algo.offline_pretrain_only="${OFFLINE_PRETRAIN_ONLY}"
    algo.train_offline_rl="${DO_OFFLINE_RL}"
    algo.offline_rl_steps="${OFFLINE_RL_STEPS}"
    algo.offline_save_freq="${OFFLINE_RL_SAVE_FREQ}"
    algo.critic_warmup_save_freq="${CRITIC_WARMUP_SAVE_FREQ}"
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
    agent.critic.num_q="${NUM_Q_HEADS}"
    agent.critic.min_q_heads="${MIN_Q_HEADS}"
    agent.critic.policy_gradient_type="${POLICY_GRADIENT_TYPE}"
    agent.critic.loss.type="${CRITIC_LOSS_TYPE}"
    agent.critic.loss.v_min="${V_MIN}"
    agent.critic.loss.v_max="${V_MAX}"
    offline_data.name="${OFFLINE_DATASET}"
    offline_data.root="${OFFLINE_DATASET_ROOT}"
    offline_data.num_episodes="${OFFLINE_EPISODES}"
    offline_data.use_base_policy_for_base_actions="${USE_BASE_POLICY_FOR_BASE_ACTIONS}"
    offline_data.min_action_range="${MIN_ACTION_RANGE}"
    offline_data.min_state_std="${MIN_STATE_STD}"
    offline_data.image_size="${IMAGE_SIZE}"
    eval_num_envs="${EVAL_NUM_ENVS}"
    eval_num_episodes="${EVAL_NUM_EPISODES}"
    eval_interval_every_steps="${EVAL_INTERVAL}"
    eval_first="${EVAL_FIRST}"
    save_video="${SAVE_VIDEO}"
    num_envs="${NUM_ENVS}"
    "rl_camera=${RL_CAMERA}"
    reward_shaping="${REWARD_SHAPING}"
    save_freq="${SAVE_FREQ}"
    no_cleanup="${NO_CLEANUP}"
    seed="${SEED}"
    torch_deterministic="${TORCH_DETERMINISTIC}"
    debug="${DEBUG}"
    wandb.project="${WANDB_PROJECT}"
    wandb.mode="${WANDB_MODE}"
)

# wandb.name is deliberately only passed when non-empty: an empty string is NOT
# the same as unset for the cache hash (None vs ""), and the Station-1 caches on
# disk were built with it unset.
[[ -n "${WANDB_NAME}"  ]] && CMD+=(wandb.name="${WANDB_NAME}")
[[ -n "${WANDB_GROUP}" ]] && CMD+=(wandb.group="${WANDB_GROUP}")
[[ -n "${WANDB_ENTITY}" ]] && CMD+=(wandb.entity="${WANDB_ENTITY}")
[[ -n "${RESUME_CKPT}" ]] && CMD+=(resume_ckpt="${RESUME_CKPT}")
[[ -n "${REAL_LOG_FILE}" ]] && CMD+=(real_log_file="${REAL_LOG_FILE}")

if [[ "${DIST:-0}" == "1" ]]; then
    CMD+=(
        role="${ROLE}"
        dist.ip="${LEARNER_IP}"
        dist.port="${LEARNER_PORT}"
        dist.broadcast_port="${LEARNER_BROADCAST_PORT}"
        dist.steps_per_update="${STEPS_PER_UPDATE}"
        dist.target_utd="${TARGET_UTD}"
    )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

exec "${CMD[@]}" "$@"
