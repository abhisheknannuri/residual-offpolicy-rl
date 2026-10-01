#!/usr/bin/env bash
# =============================================================================
# _common_rl.sh - every setting shared by the Station-3 residual-RL scripts.
#
# Sourced by all of them. Never run directly.
#
# WHY THIS FILE EXISTS
# --------------------
# Both replay-buffer caches are addressed by a hash of their config
# (docs/real/BUFFER_CACHES.md). Change one hashed value in one script and that
# script silently builds its OWN buffer instead of reusing the one you stood at
# the robot to collect. Station-1's scripts already have this bug: four of them
# use algo.learning_starts=10000 and one uses 15000, so they do not share an
# online buffer.
#
# The hashed values therefore live HERE, once, and are marked `readonly` - a
# mode script that tries to override one gets a bash error instead of a quietly
# different cache.
# =============================================================================

# --- repo + interpreter ------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# Use the venv interpreter explicitly. Bare `python` resolves to the conda env
# on this laptop, which is NOT the environment this repo is set up in.
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

# =============================================================================
# ██ HASHED - identical in every script, or buffers stop being shared ██
# =============================================================================
# Every value below appears in the offline and/or online cache key. See
# docs/real/BUFFER_CACHES.md section 2 for which goes where.

# wandb.name is in BOTH hashes. Leaving it empty (-> cfg.wandb.name=None) is what
# the Station-1 runs did, and W&B still separates runs because the display name is
# "{wandb.name}__{timestamp}__{hyperparams}__seed{N}". Do not set it per script.
readonly WANDB_NAME=""

readonly N_STEP=3
readonly GAMMA=0.9975
readonly BATCH_SIZE=256
readonly OFFLINE_FRACTION=0.5          # with BATCH_SIZE, sets both batch sizes -> both hashes
readonly BUFFER_SIZE=70000
readonly SAMPLING="uniform"
readonly MIN_ACTION_RANGE=0.1
readonly MIN_STATE_STD=0.1
readonly REWARD_SHAPING="false"

# learning_starts is the online warm-up size and is hashed as "size".
readonly LEARNING_STARTS=15000

# real_max_steps is hashed as "horizon" (it is what the training env's max_steps
# is set from, in BOTH the offline_pretrain_only and normal branches).
readonly REAL_MAX_STEPS=350

# Station-3 offline dataset. MUST be the v2.1 copy (the "_old" suffix): the
# vendored lerobot is CODEBASE_VERSION "v2.1" and cannot read the v3.0 sibling.
readonly OFFLINE_DATASET="PickCubeAndInsert_Station3_Trial1_deltajoint_old"
readonly OFFLINE_DATASET_ROOT="${REPO_ROOT}/trossen_real/datasets/${OFFLINE_DATASET}"
readonly OFFLINE_EPISODES=178          # the whole dataset (178 episodes / 55,177 frames)
readonly USE_BASE_POLICY_FOR_BASE_ACTIONS="true"

readonly RL_CAMERA='[observation.images.cam_left_wrist,observation.images.cam_right_wrist]'
readonly IMAGE_SIZE=84

readonly RANDOM_ACTION_NOISE_SCALE="[0.15,0.15,0.15,0.05,0.05,0.05,0.1]"
# =============================================================================
# ██ end of hashed block ██
# =============================================================================

# --- station / hardware ------------------------------------------------------
STATION_CONFIG_NAME="${STATION_CONFIG_NAME:-trossen_station3_single}"
POLICY_SERVER_URL="${POLICY_SERVER_URL:-http://127.0.0.1:5070}"
REAL_ACTION_SPACE="delta_joint"
REAL_SETTLE_TIME_S=0.5
ENABLE_INTERVENTION="${ENABLE_INTERVENTION:-true}"

# --- agent / optimisation (not hashed, but keep them aligned anyway) ----------
ACTION_SCALE="[0.15,0.15,0.15,0.05,0.05,0.05,0.1]"
ACTOR_LR=1e-06
CRITIC_LR=1e-06
CRITIC_TARGET_TAU=0.005
STDDEV_CLIP=0.3
STDDEV_MAX=0.3
STDDEV_MIN=0.1
STDDEV_STEP=100000
ACTOR_LAST_LAYER_INIT=0.01
ACTION_L2_REG=0.0
CRITIC_GRAD_CLIP=10.0
ACTOR_GRAD_CLIP=10.0
BC_LOSS_COEF=0.0
FREEZE_ENCODER="false"
TARGET_ACTION_NOISE=0.1
NUM_Q_HEADS=2
MIN_Q_HEADS=2
POLICY_GRADIENT_TYPE="min"
CRITIC_LOSS_TYPE="mse"
V_MIN=0.0
V_MAX=1.0
ACTOR_LR_WARMUP_STEPS=0
PROGRESSIVE_CLIPPING=0
USE_BASE_POLICY_FOR_WARMUP="true"      # must be true on real hardware

# --- schedule ----------------------------------------------------------------
TOTAL_TIMESTEPS=75000
UTD=4
ACTOR_UPDATES_PER_ITER=1
UPDATE_EVERY_N_STEPS=1
PREFETCH=3
# Periodic online-buffer checkpoint during warm-up, in transitions. This is what
# makes a dropped robot connection survivable - without it the buffer is only
# written once, at the very end of collection.
ONLINE_WARMUP_SAVE_FREQ=1000

# --- eval / logging ----------------------------------------------------------
EVAL_NUM_ENVS=1
EVAL_NUM_EPISODES=0                    # no automated eval on real hardware
EVAL_INTERVAL=1000000
EVAL_FIRST="false"
SAVE_VIDEO="false"
NUM_ENVS=1
SAVE_FREQ=5000
SEED=42
TORCH_DETERMINISTIC="false"
DEBUG="false"
WANDB_PROJECT="${WANDB_PROJECT:-trossen-station3-pick-and-insert-residual-td3}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# no_cleanup MUST stay true. With it false, train_residual_td3.py runs
# `shutil.rmtree(run_cache_dir)` on success - deleting the checkpoints AND
# online_buffer_final, which is the only copy of everything collected during
# online RL (see docs/real/BUFFER_CACHES.md section 6).
NO_CLEANUP="true"

# --- mode defaults (each script overrides what it needs) ---------------------
OFFLINE_PRETRAIN_ONLY="false"
DO_OFFLINE_RL="false"
OFFLINE_RL_STEPS=0
OFFLINE_RL_SAVE_FREQ=5000
CRITIC_WARMUP_SAVE_FREQ=5000
CRITIC_WARMUP=0
RESUME_CKPT=""
REAL_LOG_FILE=""
