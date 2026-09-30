#!/usr/bin/env bash
# =============================================================================
# GENERATED COPY - distributed (actor/learner) variant
# =============================================================================
# Source of truth for ALL hyper-parameters: ../train_residual_rl_offline_pretrain.sh
# This file is a byte-for-byte copy of that script except for:
#   1. sources ./common_dist.sh  (role parsing, IP/ports, PYTHON_BIN)
#   2. entrypoint  -> train_residual_td3_distributed.py
#   3. CONFIG_NAME -> ..._dist_config
#   4. CMD gains   -> role= / dist.* arguments
#
# Run BOTH roles from this same file so the two nodes cannot disagree on any
# hyper-parameter (a silent actor/learner config mismatch is the single most
# dangerous failure mode of this architecture):
#
#     ./train_residual_rl_offline_pretrain.sh learner     # on the GPU server
#     ./train_residual_rl_offline_pretrain.sh actor       # on the laptop  (LEARNER_IP=<server> ./train_residual_rl_offline_pretrain.sh actor)
#
# Regenerate after editing the original:  ./_regenerate.sh
# =============================================================================

# ── Distributed role / transport (see common_dist.sh) ────────────────────────
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common_dist.sh
source "${_HERE}/common_dist.sh"

# This phase touches no environment at all - it is learner-only.
if [[ "${ROLE}" == "actor" ]]; then
    echo "ERROR: this script runs offline_pretrain_only (no env, no robot) - run it with: learner" >&2
    exit 1
fi
ROLE="learner"

# ============================================================================
# train_residual_rl_offline_pretrain.sh — Offline-only pretraining: critic
#   warmup and/or the offline TD3-BC phase, using ONLY the offline/online
#   replay-buffer CACHES already on disk. NO robot, follower server, policy
#   server, leader arm, or pedal needed - this never calls get_envs() at all.
# ============================================================================
#
# Derived from train_residual_rl_real.sh (same CONFIG_NAME, same algorithm/
# network hyperparameters) with THREE differences: `algo.offline_pretrain_only`
# is set (skips get_envs()/env.reset() entirely, then exits right after the
# critic-warmup/offline-RL phase - never reaches the online training loop,
# which genuinely needs a live env), `critic_warmup_save_freq` is new
# (critic-warmup previously saved no checkpoints at all - see
# scripts/TrossenStation1Real/BUFFER_POPULATION_ARCHITECTURE.md), and
# `train_offline_rl`/`offline_rl_steps` are exposed as top-level toggles below.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ PRE-FLIGHT CHECKLIST                                                   │
# │                                                                         │
# │ 1. Nothing to start. No follower server, no policy_server.py, no       │
# │    leader arm, no pedal - none of that is ever touched by this script. │
# │                                                                         │
# │ 2. The offline buffer cache (OFFLINE_DATASET_ROOT's hash) AND the       │
# │    online buffer cache (a hash of algo.*/offline_data.* fields) must    │
# │    ALREADY exist on disk, matching every relevant config value below   │
# │    exactly (same as whatever real-hardware run originally produced     │
# │    them) - if either doesn't match, this FAILS LOUDLY with a clear      │
# │    RuntimeError naming which cache and which fields to check, rather    │
# │    than silently trying to connect to hardware that isn't there.        │
# │                                                                         │
# │ 3. Pick ONE experiment per run via DO_OFFLINE_RL below:                │
# │      false -> critic-warmup only (critic_warmup_steps updates,          │
# │                checkpointed every critic_warmup_save_freq steps)        │
# │      true  -> offline TD3-BC phase (offline_rl_steps updates, actor    │
# │                AND critic, checkpointed every offline_save_freq steps - │
# │                this ALSO runs the critic-only updates internally, so    │
# │                the separate critic-warmup phase is skipped when this    │
# │                is true - they are mutually exclusive by design, not a   │
# │                bug; see train_residual_td3.py's own `not _did_offline_rl`│
# │                guard)                                                   │
# │    Run this script twice (flip DO_OFFLINE_RL between runs) if you want  │
# │    checkpoints from BOTH experiments to compare.                        │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   bash scripts/TrossenStation1Real/train_residual_rl_offline_pretrain.sh
#
# Output:
#   - Run directory: run_<timestamp>_resfit__<params>__seed<N>/
#     ├── models/   (checkpoints: latest/, policy_step_N/, final/ -
#     │              NO "best/" for real hardware, see §12 below)
#     └── outputs/
# ============================================================================
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. BASE BC POLICY                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# NOT loaded locally for real hardware - the frozen base ACT policy is served
# remotely by policy_server.py (separate process/venv, see checklist above)
# and queried over HTTP by TrossenResidualEnv. base_policy.wandb_id is left
# at the config's placeholder default and is unused (train_residual_td3.py
# skips the entire local-load block when real_hardware=true).

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. HYDRA CONFIG                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# residual_td3_trossen_real_config (resfit/rl_finetuning/config/residual_td3.py)
# - the real-hardware task config: task="trossen_real", real_hardware=True,
# num_envs=eval_num_envs=1 by default.
CONFIG_NAME="residual_td3_trossen_real_dist_config"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. REAL HARDWARE CONNECTION                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# station_config_name: matches trossen_real/configs/<name>.yaml AND whatever
# --config was passed to follower_single_server.py above.
STATION_CONFIG_NAME="trossen_station1_single"
#
# policy_server_url: where policy_server.py (step 2 of the checklist) is listening.
POLICY_SERVER_URL="http://127.0.0.1:5070"
#
# real_action_space: "delta_joint" (per-frame joint delta vs. observation.state
# at the SAME tick, matches trossen_real/scripts/convert_to_delta_joint_dataset.py)
# or "absolute" - MUST match how the base policy checkpoint was trained.
REAL_ACTION_SPACE="delta_joint"
#
# real_max_steps: per-episode step cap (truncation) - there's no MuJoCo horizon
# to read on real hardware. At REAL_ACTION_SPACE=delta_joint / 20Hz, 150 steps
# ≈ 7.5s of autonomous+residual rollout per episode before a forced truncation
# (reset pedal can still end an episode earlier than this).
REAL_MAX_STEPS=400
#
# real_settle_time_s: seconds to sleep after the staged-position reset move
# (+ leader sync) completes, before resetting the policy queue / taking the
# first obs - lets the arm fully stop vibrating, and gives you a moment to
# release any pedal still held from the previous episode (reset() logs a
# warning if the reward pedal is still down after this window elapses).
REAL_SETTLE_TIME_S=2.0
#
# enable_intervention: see the pre-flight checklist above. Required=true means
# you WILL be asked to have the leader physically connected before this can
# start - there's no silent "training worked anyway without a leader" mode.
ENABLE_INTERVENTION="true"
#
# real_log_file: optional JSONL diagnostic log path (one line per env.step() -
# reward/terminated/truncated/intervened/executed action/raw follower state/
# achieved Hz, everything EXCEPT images). Purely for offline inspection -
# training itself never reads this file, only the replay buffer. Empty = off.
REAL_LOG_FILE="./RealTrainLogs/real_env_diagnostic_log28Aug2026_1600.jsonl"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. ACTION NORMALIZATION                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Same ActionScaler math as sim (see train_residual_rl_can_sparse.sh's §3 for
# the full derivation) - fit from the REAL dataset's action stats (7D: 6 joint
# deltas + gripper). START CONSERVATIVE on real hardware and increase only
# after verifying stability - these defaults have NOT been tuned against your
# specific checkpoint/dataset yet.
#
# action_scale: residual can adjust ±action_scale in NORMALIZED [-1,1] space,
# per-dimension. Smaller for the gripper (7th) since its own range is already
# tiny/binary-ish (see PickAndInsertCube.md).
ACTION_SCALE="[0.15,0.15,0.15,0.05,0.05,0.05,0.1]"
MIN_ACTION_RANGE=0.1

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. STATE NORMALIZATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
MIN_STATE_STD=0.1

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. CORE RL HYPERPARAMETERS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# total_timesteps: MUCH smaller than sim - this is REAL wall-clock time.
# At 20Hz, 20000 steps ≈ 1000s (~17 min) of env-stepping alone, before adding
# gradient-update compute time on top (see the timing discussion in
# docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md). Start small, scale up once
# you've verified a short run works end-to-end.
TOTAL_TIMESTEPS=75000
#
# N-step returns - same guidance as sim (sparse reward benefits from N>1).
N_STEP=3
#
# Discount factor. real_max_steps=150 is much shorter than sim's 100-500 step
# horizons - 0.99 is a reasonable default here (0.99^150 ≈ 0.22).
GAMMA=0.9975
#
# Updates-to-data ratio (UTD). NOTE: each gradient-update burst runs
# SYNCHRONOUSLY between env.step() calls on real hardware - a high UTD here
# directly slows down the real control loop's effective Hz (see the doc's
# timing section). Start lower than sim's typical UTD=4 if you observe the
# achieved control rate dropping noticeably below control.frequency_hz.
UTD=4

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. EXPLORATION NOISE                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Same TruncatedNormal mechanism as sim (see can/lift scripts' §6 for the full
# explanation). Kept proportionally small vs. ACTION_SCALE above - real
# hardware, be conservative.
STDDEV_MAX="[0.015,0.015,0.015,0.03,0.03,0.03,0.0125]"
STDDEV_MIN="[0.015,0.015,0.015,0.03,0.03,0.03,0.0125]"
STDDEV_STEP=15000
STDDEV_CLIP=0.3

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. WARMUP PHASES                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# learning_starts: random-action warmup, filling the ONLINE buffer via live
# env.step() calls BEFORE any gradient updates. At 20Hz, 2000 steps ≈ 100s.
LEARNING_STARTS=10000
#
# use_base_policy_for_warmup: MUST be "true" for real hardware -
# train_residual_td3.py asserts this at startup (the "pure random minus base"
# warmup mode can imply arbitrarily large corrections - unsafe on real
# hardware). Do NOT set this to "false".
USE_BASE_POLICY_FOR_WARMUP="true"
#
# random_action_noise_scale: uniform noise around the base policy during
# warmup (used as: env gets base_policy(obs) + noise). Match ACTION_SCALE's
# shape/spirit - keep conservative.
RANDOM_ACTION_NOISE_SCALE="[0.15,0.15,0.15,0.05,0.05,0.05,0.1]"
#
# critic_warmup_steps: critic-only updates (no actor). No live robot involved
# in this script at all, so this is purely compute-bound - default bumped up
# from the online script's 10000 to 20000 for a more thorough offline pass.
# Only actually runs when DO_OFFLINE_RL="false" below (mutually exclusive with
# the offline-RL phase - see the pre-flight note at the top of this file).
CRITIC_WARMUP=20000
#
# critic_warmup_save_freq: NEW field (algo.critic_warmup_save_freq) - previously
# critic-warmup saved no checkpoints at all. Saves to
# <run_dir>/models/critic_warmup_step_<N>/checkpoint.pt every this many steps,
# plus a final one at <run_dir>/models/critic_warmup_final/checkpoint.pt.
CRITIC_WARMUP_SAVE_FREQ=5000

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6b. OFFLINE-ONLY PRETRAINING TOGGLES (this script's own, not in the     ║
# ║       live-robot .sh)                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# offline_pretrain_only: ALWAYS true in this script - skips get_envs()/
# env.reset() entirely and exits right after critic-warmup/offline-RL phase,
# before the online loop. This is what makes running away from the lab safe.
OFFLINE_PRETRAIN_ONLY="true"
#
# do_offline_rl: false = critic-warmup only (CRITIC_WARMUP steps, see above).
# true = offline TD3-BC phase instead (OFFLINE_RL_STEPS below, actor+critic) -
# mutually exclusive with critic-warmup by design (train_residual_td3.py's own
# `not _did_offline_rl` guard skips critic-warmup when this is true). Run this
# script TWICE, flipping this between runs, if you want checkpoints from BOTH.
DO_OFFLINE_RL="true"
#
# offline_rl_steps: only used when DO_OFFLINE_RL="true". Actor+critic updates
# (TD3-BC style, BC-blended actor loss - see INTERVENTION_CLIPPING_AND_SYNC.md
# for why this is a DIFFERENT actor-update path than the online one, which has
# no BC loss at all). Checkpointed every OFFLINE_RL_SAVE_FREQ steps.
OFFLINE_RL_STEPS=75000
OFFLINE_RL_SAVE_FREQ=5000

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9. REPLAY BUFFER & OFFLINE DATA                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# offline_dataset: the LOCAL v2.1 delta-joint dataset, REWARD-LABELED variant
# (has a `next.reward` column added on top of the BC-training dataset - same
# demos otherwise). repo_id is just a label when root is given directly
# (matches replay_episode.py's convention).
OFFLINE_DATASET="PickAndInsertCube_Station1_merged_deltajoint_old"
OFFLINE_DATASET_ROOT="/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint_old"
#
# use_base_policy_for_base_actions: true (default, matches your intent) -
# queries the REMOTE policy_server.py (via env.policy) for each offline
# dataset frame's base action, using FULL-RESOLUTION images, then resizes to
# IMAGE_SIZE (§14) before storing - see docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md
# for the full "offline buffer image-resolution / OOM fix" section. false =
# GT-as-base (residual target always 0 for demo data) - simpler/faster to
# populate but less consistent with online training.
USE_BASE_POLICY_FOR_BASE_ACTIONS="true"
#
# buffer_size: ONLINE replay buffer capacity (transitions). Real data
# collection is slow (20Hz) - you'll never fill anywhere near sim-scale
# buffers in a real session, so this just needs to comfortably exceed
# TOTAL_TIMESTEPS.
BUFFER_SIZE=70000
BATCH_SIZE=256
SAMPLING="uniform"
#
# offline_episodes: how many demo episodes to load. MUST be set (not empty) -
# train_residual_td3.py asserts cfg.offline_data.num_episodes is not None
# regardless of real_hardware. ${OFFLINE_DATASET} has 136 episodes total
# (verified directly via meta/info.json + per-episode parquet inspection,
# NOT the 161-episode count from a DIFFERENT, older dataset variant -
# `..._delta_rewardLabled_v21_old` - that this one was derived from) - set to
# 136 to use all of them, or lower (as here) for a quicker run/smoke test.
OFFLINE_EPISODES=120
#
# offline_fraction: fraction of each training batch from the offline buffer.
OFFLINE_FRACTION=0.5

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  10. LEARNING RATES                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ACTOR_LR=1e-6
CRITIC_LR=1e-4
CRITIC_TARGET_TAU=0.005

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  11. ACTOR / CRITIC NETWORK CONFIG                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
ACTOR_LAST_LAYER_INIT=0.0
PROGRESSIVE_CLIPPING=0
NUM_Q_HEADS=10
MIN_Q_HEADS=2
POLICY_GRADIENT_TYPE="ensemble_mean"
CRITIC_LOSS_TYPE="mse"
V_MIN=0.0
V_MAX=1.0

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  12. EVALUATION                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# NO periodic evaluation loop runs for real hardware (train_residual_td3.py
# skips BOTH eval blocks entirely when real_hardware=true - run_dexmg_evaluation
# is sim/MuJoCo-specific). There is therefore no "best/" checkpoint either -
# rely purely on SAVE_FREQ (§13) periodic checkpointing, and evaluate saved
# checkpoints manually/offline afterward. eval_num_envs is forced to 1
# (asserted in train_residual_td3.py) regardless of what's set here.
EVAL_NUM_ENVS=1
EVAL_NUM_EPISODES=1
EVAL_INTERVAL=999999999
EVAL_FIRST="false"
SAVE_VIDEO="false"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  13. CHECKPOINTING                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# save_freq: THE primary safety net for real hardware (no eval-triggered
# "best" checkpoint exists here) - keep this frequent.
SAVE_FREQ=2500
NO_CLEANUP="true"
RESUME_CKPT=""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  14. IMAGE RESOLUTION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Drives BOTH the offline buffer (resized post-query in _populate_offline_buffer())
# AND the online buffer (TrossenResidualEnv's rl_image_size, wired from this
# SAME field in get_envs()) - they cannot drift out of sync. The base-policy
# query itself (offline AND online) always uses the camera's FULL native
# resolution regardless of this value - only what gets STORED is resized.
IMAGE_SIZE=84

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  15. CAMERAS                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
RL_CAMERA='[observation.images.cam_left_wrist,observation.images.cam_right_wrist]'

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  16. REWARD                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ONLINE reward (live robot rollout): ALWAYS sparse binary from the reward foot
# pedal (TrossenResidualEnv.step()'s EpisodeBoundaryMonitor.get_frame_reward(),
# same PedalListener.consume_reward_latched() as before - held continuously
# from success onset to reset/release gives reward=1 on EVERY transition in
# between, not just the last one). This flag has NO effect on it whatsoever -
# `get_envs()`'s real_hardware branch returns a `TrossenResidualEnv` directly
# and never reaches the sim-only `create_vectorized_env(reward_shaping=...)`
# call that this flag also feeds; `TrossenResidualEnv.__init__` doesn't even
# accept a reward_shaping parameter. Verified by reading the code directly,
# not assumed.
#
# NEW (this session): releasing the reward pedal after holding it now ALSO
# ends the episode online (mirrors teleop/infer's "Stop Recording"/"Stop
# Inference" pedal behavior) - previously only the reset pedal did this. See
# `trossen_real/human_intervention/PEDAL_BEHAVIOR.md`.
#
# OFFLINE reward (this flag, `_populate_offline_buffer()` only): true = read
# `next.reward` per-frame; false = fall back to `next.done` per-frame instead.
# VERIFIED DIRECTLY (parquet inspection, not assumed) on THIS dataset
# (${OFFLINE_DATASET}, 136 total episodes, ${OFFLINE_EPISODES} of them loaded
# here): `next.done` is byte-identical to `next.reward>0.5` in ALL 136
# episodes - both columns mark the SAME "held from success onset to episode
# end" region (mean 41 frames/episode), not just a single terminal frame.
# CONSEQUENCE: this flag is currently a NO-OP for training behavior either
# way - both settings read numerically identical per-frame values. Set to
# "false" per your call (this is a sparse-reward run, not a dense/shaped
# one - see below), but be aware it will not change what gets loaded unless
# `next.done` is ever re-labeled to be a genuine single-frame terminal marker
# independent of `next.reward`.
#
# Terminology note: neither setting of this flag produces genuinely DENSE/
# shaped reward in the RL sense (a signal informative throughout the WHOLE
# trajectory) - both are sparse, binary, and zero for the ~86% of each
# episode before the success window. Actual dense/shaped reward in this
# codebase is the separate, unrelated PBRS `reward_model` system
# (`cfg.reward_model.enabled`, requires `algo.terminated_only_bootstrap=true`
# - not used by this script). `algo.terminated_only_bootstrap` defaults to
# `false` and is left at that default here - correct for this sparse-reward
# run, since it means `done` (terminated OR truncated) is used uniformly for
# the n-step bootstrap cutoff, with no need to distinguish a real
# reset/success termination from a plain REAL_MAX_STEPS timeout.
REWARD_SHAPING="false"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  17. WANDB                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
WANDB_PROJECT=${WANDB_PROJECT:-"trossen-station1-pick-and-insert-residual-td3"}
WANDB_NAME=${WANDB_NAME:-""}
WANDB_GROUP=${WANDB_GROUP:-""}
WANDB_ENTITY=${WANDB_ENTITY:-""}
WANDB_MODE=${WANDB_MODE:-"online"}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  18. SEED & DEBUG                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
SEED="42"
TORCH_DETERMINISTIC="false"
DEBUG="false"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  19. ADVANCED / RARELY CHANGED                                        ║
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

# ============================================================================
# BUILD AND RUN
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Real-Hardware Residual RL Training — Trossen PickAndInsertCube "
echo "║  Station: ${STATION_CONFIG_NAME}  |  Policy server: ${POLICY_SERVER_URL}"
echo "║  Intervention enabled: ${ENABLE_INTERVENTION}                "
echo "║  Steps: ${TOTAL_TIMESTEPS}  |  UTD: ${UTD}  |  γ: ${GAMMA}   "
echo "║  Action scale: ${ACTION_SCALE}                                "
echo "║  WandB: ${WANDB_PROJECT}                                      "
echo "╚══════════════════════════════════════════════════════════════╝"

CMD=(
    "${PYTHON_BIN}" resfit/rl_finetuning/scripts/train_residual_td3_distributed.py
    --config-name="${CONFIG_NAME}"

    # ── Distributed actor/learner (agentlace) ──
    role="${ROLE}"
    dist.ip="${LEARNER_IP}"
    dist.port="${LEARNER_PORT}"
    dist.broadcast_port="${LEARNER_BROADCAST_PORT}"
    dist.steps_per_update="${STEPS_PER_UPDATE}"
    dist.target_utd="${TARGET_UTD}"

    # ── Real hardware ──
    real_hardware=true
    station_config_name="${STATION_CONFIG_NAME}"
    policy_server_url="${POLICY_SERVER_URL}"
    real_action_space="${REAL_ACTION_SPACE}"
    real_max_steps="${REAL_MAX_STEPS}"
    real_settle_time_s="${REAL_SETTLE_TIME_S}"
    enable_intervention="${ENABLE_INTERVENTION}"

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
    algo.critic_warmup_save_freq="${CRITIC_WARMUP_SAVE_FREQ}"
    algo.actor_lr_warmup_steps="${ACTOR_LR_WARMUP_STEPS}"

    # ── Offline-only pretraining (this script's reason for existing) ──
    algo.offline_pretrain_only="${OFFLINE_PRETRAIN_ONLY}"
    algo.train_offline_rl="${DO_OFFLINE_RL}"
    algo.offline_rl_steps="${OFFLINE_RL_STEPS}"
    algo.offline_save_freq="${OFFLINE_RL_SAVE_FREQ}"

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
    offline_data.root="${OFFLINE_DATASET_ROOT}"
    offline_data.use_base_policy_for_base_actions="${USE_BASE_POLICY_FOR_BASE_ACTIONS}"
    offline_data.min_action_range="${MIN_ACTION_RANGE}"
    offline_data.min_state_std="${MIN_STATE_STD}"
    offline_data.image_size="${IMAGE_SIZE}"

    # ── Evaluation (inert for real hardware - see §12) ──
    eval_num_envs="${EVAL_NUM_ENVS}"
    eval_num_episodes="${EVAL_NUM_EPISODES}"
    eval_interval_every_steps="${EVAL_INTERVAL}"
    eval_first="${EVAL_FIRST}"
    save_video="${SAVE_VIDEO}"

    # ── Environment ──
    num_envs="${NUM_ENVS}"
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

[[ -n "${WANDB_NAME}" ]]     && CMD+=(wandb.name="${WANDB_NAME}")
[[ -n "${WANDB_GROUP}" ]]    && CMD+=(wandb.group="${WANDB_GROUP}")
[[ -n "${WANDB_ENTITY}" ]]   && CMD+=(wandb.entity="${WANDB_ENTITY}")
[[ -n "${SEED}" ]]           && CMD+=(seed="${SEED}")
[[ -n "${RESUME_CKPT}" ]]    && CMD+=(resume_ckpt="${RESUME_CKPT}")
[[ -n "${OFFLINE_EPISODES}" ]] && CMD+=(offline_data.num_episodes="${OFFLINE_EPISODES}")
[[ -n "${REAL_LOG_FILE}" ]]  && CMD+=(real_log_file="${REAL_LOG_FILE}")

"${CMD[@]}"

echo ""
echo "✓ Real-hardware residual RL training complete."
echo "  Check WandB: ${WANDB_PROJECT}"
