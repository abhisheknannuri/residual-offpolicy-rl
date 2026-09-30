#!/usr/bin/env bash
# ============================================================================
# train_residual_rl_lift.sh — Residual RL fine-tuning (TD3) for Lift (Panda)
# ============================================================================
#
# Fine-tunes a frozen BC base policy with a small learned residual action
# correction using TD3 (Twin Delayed DDPG) with off-policy data.
#
# Task:   Lift  (robosuite, single Franka Panda, 7-DOF + gripper)
# Robot:  Panda — action is 7D: Δpos(3) + Δrot_axis_angle(3) + gripper(1)
# State:  9D — eef_pos(3) + eef_quat(4) + gripper_qpos(2)
#         (no joint_pos, no velocities, no torques, no object state)
# Images: 84×84 pixels, 2 cameras: agentview + robot0_eye_in_hand
#         (hardcoded default in dexmg.py, NOT configurable via CLI)
# Horizon: 100 steps  (hardcoded in dexmg.py for "Lift" task)
# Reward:  SPARSE binary — 0 every step, 1.0 on first successful lift
#          (robosuite.make() NOT passed reward_shaping → defaults to False)
#          Episode terminates immediately on success (reward == 1.0)
#          Dense reward available via reward_shaping=True (see v_min/v_max section)
# FPS:     20 Hz control_freq
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ HOW THE PIPELINE WORKS                                                  │
# │                                                                         │
# │ 1. Load frozen BC base policy from WandB (your trained ACT model)       │
# │ 2. Load offline dataset (same demos used for BC) → compute action/state │
# │    normalization stats → fill offline replay buffer                     │
# │ 3. Create train env (1) + eval envs (N parallel)                       │
# │ 4. Warmup: collect random transitions into online replay buffer         │
# │ 5. Critic warmup: train critic only (no actor) for stability            │
# │ 6. Main loop: collect with base+residual, train critic+actor from       │
# │    mixed online/offline batches                                         │
# │                                                                         │
# │ At each env step:                                                       │
# │   base_action = frozen_BC_policy(obs)     # in original action space    │
# │   base_naction = ActionScaler.scale(base_action)  # → [-1, 1]          │
# │   residual_naction = actor(obs, base_naction)  # small correction       │
# │   combined_naction = base_naction + residual_naction  # in [-1, 1]      │
# │   env_action = ActionScaler.unscale(combined_naction) # back to raw     │
# │   obs, reward = env.step(env_action)                                    │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Prerequisites:
#   - A trained BC policy uploaded to WandB (from train_bc_lift.sh)
#   - Set BASE_WANDB_ID below to "project/run_id" from your BC WandB run
#
# Usage:
#   # Inside container:
#   bash scripts/train_residual_rl_lift.sh
#
#   # From host:
#   docker compose run --rm --name qte9489-resfit-rl train \
#       bash scripts/train_residual_rl_lift.sh
#
# Output:
#   - Run directory: run_<timestamp>_resfit__<params>__seed<N>/
#     ├── models/         (checkpoints: best/, latest/, policy_step_N/, final/)
#     └── outputs/        (eval videos, Q-value plots)
# ============================================================================
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. BASE BC POLICY                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# The frozen BC policy that the residual actor corrects on top of.
# Find this in WandB → your BC training run → copy "project/run_id".
#
BASE_WANDB_ID="resfit-robomimic-can-bc/pzqj1tmd"
#
# Which checkpoint to load from that WandB run:
#   "best"   — highest eval success rate during BC training (recommended)
#   "latest" — last saved checkpoint
#   "final"  — end of training
#   "policy_step" — specific checkpoint at step N (e.g. 25000)
#
BASE_WT_TYPE="latest"   # ← step number of the checkpoint to load (recommended: best checkpoint from BC training)
#
# Which version of the WandB artifact to use:
#   "latest" — most recent upload (default)
#   "v0", "v1", etc. — specific version
#
BASE_WT_VERSION="v9"

# Local checkpoint directory for the base ACT policy. When set (non-empty), the
# base policy is loaded from this local path and the W&B download above is
# SKIPPED entirely (local takes priority). Use this for ACT checkpoints trained
# in another LeRobot repo that are not on W&B.
# Example (LeRobot pretrained_model dir):
#   BASE_LOCAL_PATH="/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/GeneralistRewardModels/lerobot/outputs/train/policy_rabc/checkpoints/050000/pretrained_model"
BASE_LOCAL_PATH=${BASE_LOCAL_PATH:-""}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. HYDRA CONFIG                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Hydra config name registered in resfit/rl_finetuning/config/residual_td3.py
# via cs.store(). The config is auto-loaded by the training script — no
# separate registration step is needed. Hydra discovers it at import time
# because train_residual_td3.py imports residual_td3.py which calls cs.store().
#
# Available configs:
#   residual_td3_cube_lift_config    — Lift (Panda, single arm)      ← THIS ONE
#   residual_td3_can_config          — Can (Panda, single arm)
#   residual_td3_square_config       — Square (Panda, single arm)
#   residual_td3_coffee_config       — TwoArmCoffee (bimanual)
#   residual_td3_box_clean_config    — TwoArmBoxCleanup (bimanual)
#   residual_td3_two_arm_cansort_config — TwoArmCanSort (bimanual)
#   residual_td3_dexmg_config        — Base config (must override task=)
#
CONFIG_NAME="residual_td3_can_config"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. ACTION NORMALIZATION                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ HOW ACTION NORMALIZATION WORKS                                          │
# │                                                                         │
# │ 1. Load the OFFLINE DATASET (same LeRobot dataset as BC training)       │
# │ 2. Extract dataset.meta.stats["action"] → {min, max} per dimension     │
# │    For Lift (7D): min/max of Δpos, Δrot, gripper from all 300 demos    │
# │ 3. Compute midpoint and half-range:                                     │
# │      mid = (min + max) / 2                                              │
# │      half_range = (max - min) / 2                                       │
# │ 4. Clamp half_range to at least `min_action_range / 2` per dim         │
# │    (prevents blow-up if some dim has near-zero variance)                │
# │ 5. Expand the range by `action_scale`:                                  │
# │      expanded_half = half_range * (1 + action_scale)                    │
# │    This gives the residual room to push beyond the demo distribution   │
# │ 6. Final limits:                                                        │
# │      action_min = mid - expanded_half                                   │
# │      action_max = mid + expanded_half                                   │
# │                                                                         │
# │ scale(a)   = 2 * (a - action_min) / (action_max - action_min) - 1     │
# │ unscale(n) = action_min + (n + 1) * (action_max - action_min) / 2     │
# │                                                                         │
# │ EXAMPLE (Lift, action_scale=0.1):                                       │
# │   If demo Δpos_x range is [-0.05, 0.05]:                               │
# │     mid = 0.0, half_range = 0.05                                        │
# │     expanded = 0.05 * 1.1 = 0.055                                       │
# │     limits: [-0.055, 0.055]                                             │
# │   The base policy outputs ~[-0.05, 0.05] which maps to ~[-0.91, 0.91]  │
# │   The residual can push it to the full [-1, 1] = [-0.055, 0.055]       │
# └─────────────────────────────────────────────────────────────────────────┘
#
# action_scale: how much extra room beyond the demo action range.
#   The actor network outputs:  residual = network_output * action_scale
#   So action_scale = 0.1 means residual can adjust ±0.1 in NORMALIZED [-1,1] space
#   i.e., ±10% of the full normalized action range.
#
#   0.05 = very conservative, tiny corrections only
#   0.1  = standard for easy tasks like Lift (recommended)
#   0.2  = more freedom for harder tasks
#   0.3  = aggressive, risk of destabilizing the base policy
#
# NOTE: To use a 7D action scale (e.g., different scales for translation vs rotation),
# you can pass a list in bash. Ensure there are NO SPACES in the list!
# Example:
# ACTION_SCALE="[0.116,0.116,0.116,0.024,0.024,0.024,0.1]"
# if you update this action scale then makes ure you also update the RANDOM_ACTION_NOISE_SCALE=0.2
# and STDDEV_MAX="[0.0145,0.0145,0.0145,0.003,0.003,0.003,0.0125]"
# and STDDEV_MIN="[0.0145,0.0145,0.0145,0.003,0.003,0.003,0.0125]"
#
ACTION_SCALE=${ACTION_SCALE:-0.2}
# ACTION_SCALE="[0.15,0.15,0.15,0.05,0.05,0.05,0.1]"
#
# min_action_range: minimum range per action dimension (prevents div-by-zero
# if some dimension has near-zero variance in the dataset).
# Default 0.1 is safe. Only change if you know what you're doing.
#
MIN_ACTION_RANGE=0.1

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. STATE NORMALIZATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ HOW STATE NORMALIZATION WORKS                                           │
# │                                                                         │
# │ 1. From dataset.meta.stats["observation.state"] → {mean, std} per dim   │
# │ 2. Clamp std to at least `min_state_std` per dim (prevents blow-up)     │
# │ 3. At runtime: normalized_state = (state - mean) / std                  │
# │                                                                         │
# │ For Lift (9D state): eef_pos(3) + eef_quat(4) + gripper_qpos(2)         │
# │                                                                         │
# │ Both the critic and actor receive standardized states.                  │
# │ The base_action is normalized to [-1,1] via ActionScaler and passed     │
# │ as a separate obs key "observation.base_action" to the residual actor.  │
# └─────────────────────────────────────────────────────────────────────────┘
#
MIN_STATE_STD=0.1

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. CORE RL HYPERPARAMETERS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Total environment steps to train for.
# Lift horizon=100, so 300K steps ≈ 3000 episodes of interaction.
# 300K is the default. 500K for thorough training. 100K for quick tests.
TOTAL_TIMESTEPS=200000

# N-step returns — how many steps of actual reward to use before bootstrapping.
# With sparse reward (only r=1 at success), higher N helps propagate reward signal.
#   1 = standard 1-step TD (slow reward propagation with sparse reward)
#   3 = default for RLPD (good balance)
#   5 = recommended for Lift with sparse reward (faster propagation)
#   10 = aggressive, more variance but faster signal
N_STEP=3

# Discount factor γ — how much to value future rewards.
# With horizon=100 and γ=0.995: γ^100 ≈ 0.61 (reward at horizon still valued at 61%)
# With horizon=100 and γ=0.99:  γ^100 ≈ 0.37
#   0.99   = standard (shorter effective horizon)
#   0.995  = recommended for Lift (long-horizon sparse reward)
#   0.999  = very long horizon
GAMMA=0.995

# Updates-to-data ratio (UTD) — how many gradient updates per environment step.
# Higher = more sample efficient but slower wall-clock per step.
#   1  = on-policy ratio
#   4  = standard (recommended)
#   8  = more aggressive, can be unstable
#  16  = RED-Q style, very sample efficient
UTD=4

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. EXPLORATION NOISE                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ HOW EXPLORATION NOISE WORKS                                             │
# │                                                                         │
# │ During rollouts (not eval), the actor predicts:                         │
# │   residual_mean = actor_network(obs) * action_scale                    │
# │ Then noise is sampled from TruncatedNormal:                             │
# │   residual_action ~ TruncatedNormal(residual_mean, stddev)             │
# │   clipped to [mean - stddev_clip, mean + stddev_clip]                  │
# │                                                                         │
# │ The stddev follows a LINEAR SCHEDULE:                                   │
# │   stddev(step) = lerp(stddev_max, stddev_min, step / stddev_step)     │
# │   i.e., starts at stddev_max and linearly decays to stddev_min         │
# │                                                                         │
# │ If stddev_max == stddev_min → constant noise (no annealing)            │
# │                                                                         │
# │ IMPORTANT: stddev is in NORMALIZED action space [-1, 1].               │
# │ So stddev=0.05 means ±5% of the normalized range as noise std.        │
# │ With action_scale=0.1, the actor mean is already small (~0.1),         │
# │ so stddev=0.05 is actually significant relative to the mean!           │
# │                                                                         │
# │ During eval: stddev=0 (deterministic, no noise)                        │
# └─────────────────────────────────────────────────────────────────────────┘
#
# stddev_max: exploration noise at the START of training
# stddev_min: exploration noise at the END of training (after stddev_step steps)
# stddev_step: over how many steps to anneal from max to min
#
#   action_scale=0.1, stddev=0.05 → noise is ~50% of the residual range
#   action_scale=0.1, stddev=0.025 → noise is ~25% of the residual range
#   action_scale=0.2, stddev=0.05 → noise is ~25% of the residual range
#
STDDEV_MAX=${STDDEV_MAX:-0.025}
STDDEV_MIN=${STDDEV_MIN:-0.025}
# STDDEV_MAX=${STDDEV_MAX:-0.025}
# STDDEV_MAX="[0.015,0.015,0.015,0.003,0.003,0.003,0.0125]"
# STDDEV_MIN=${STDDEV_MIN:-0.025}
# STDDEV_MIN="[0.015,0.015,0.015,0.003,0.003,0.003,0.0125]"
STDDEV_STEP=${STDDEV_STEP:-200000}
#
# stddev_clip: hard clip on the TruncatedNormal distribution.
# Actions are clipped to [mean - stddev_clip, mean + stddev_clip].
# Default 0.3. Only change if you want tighter/looser exploration bounds.
STDDEV_CLIP=${STDDEV_CLIP:-0.3}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. WARMUP PHASES                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# There are TWO separate warmup phases:
#
# Phase 1: RANDOM ACTION WARMUP (learning_starts)
# ────────────────────────────────────────────────
# Before any learning, fill the online replay buffer with transitions
# collected using random actions (or base_policy + noise).
# This gives the critic initial data to learn from.
#
LEARNING_STARTS=10000
#
# use_base_policy_for_warmup controls WHAT actions are used during Phase 1:
#
#   true (default):
#     residual = uniform_noise * random_action_noise_scale
#     env gets: base_policy(obs) + noise
#     → stays close to demo distribution, safer exploration
#
#   false:
#     pure_random = uniform * random_action_noise_scale
#     residual = pure_random - base_action  (cancels out base policy)
#     env gets: pure random actions
#     → more diverse but might collect useless transitions
#
USE_BASE_POLICY_FOR_WARMUP="true"
#
# random_action_noise_scale: scale of the uniform noise during warmup
#   With use_base_policy_for_warmup=true:
#     0.2 = small noise around base policy (recommended for residual)
#     1.0 = full [-1, 1] noise added to base policy (very noisy)
#   With use_base_policy_for_warmup=false:
#     1.0 = full uniform random actions in [-1, 1]
#
RANDOM_ACTION_NOISE_SCALE=0.2
# RANDOM_ACTION_NOISE_SCALE="[0.15,0.15,0.15,0.05,0.05,0.05,0.1]"
#
# Phase 2: CRITIC WARMUP (critic_warmup_steps)
# ─────────────────────────────────────────────
# After filling the buffer, train the CRITIC ONLY (no actor gradient updates)
# for this many update steps. This lets the Q-function stabilize before the
# actor starts using Q-gradients for policy improvement.
#
# 0     = no critic warmup (start training actor immediately)
# 10000 = recommended (lets critic converge before actor depends on it)
#
CRITIC_WARMUP=0

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. REPLAY BUFFER & OFFLINE DATA                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# offline_dataset: HuggingFace dataset ID for the offline demo buffer.
# This should be the SAME dataset used for BC training.
# Default (from Hydra config): ankile/robomimic-mh-lift-image (84×84)
# If you trained BC on a different dataset (e.g. your own 256×256),
# override it here so RL uses matching demos.
#
OFFLINE_DATASET="poolvarine/robomimic-mh-can-image-sparse"
# OFFLINE_DATASET="ankile/robomimic-mh-can-image"   # ← ankile's 84×84 original
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ TWO REPLAY BUFFERS                                                      │
# │                                                                         │
# │ OFFLINE buffer: filled once at start from the demo dataset              │
# │   - Contains transitions from human demonstrations                      │
# │   - Actions are re-labeled: what would the base policy have predicted?  │
# │     (use_base_policy_for_base_actions=true in OfflineDataConfig)        │
# │   - Size = num_episodes * avg_episode_length transitions                │
# │   - For Lift with 50 eps: ~50 * 100 = 5000 transitions                 │
# │                                                                         │
# │ ONLINE buffer: filled during training from env interaction              │
# │   - Contains transitions from base_policy + residual_policy             │
# │   - Size = buffer_size (configurable)                                   │
# │   - Initially filled during warmup phase (learning_starts transitions)  │
# │                                                                         │
# │ MIXING: each training batch is split:                                   │
# │   offline_fraction of batch from offline buffer                         │
# │   (1 - offline_fraction) from online buffer                             │
# │   Default: 0.5 = half offline, half online                              │
# └─────────────────────────────────────────────────────────────────────────┘
#
# buffer_size: capacity of the ONLINE replay buffer (in transitions)
# Lift horizon=100, so 80K ≈ 800 episodes worth.
# Should be large enough that old transitions don't get overwritten too fast.
#   40000  = tight (for limited RAM)
#   80000  = standard (recommended)
#   200000 = large (for >32GB RAM servers)
#
BUFFER_SIZE=100000
#
# batch_size: number of transitions per gradient update
# Each batch = offline_fraction from offline_rb + rest from online_rb.
BATCH_SIZE=256
#
# sampling_strategy: how to sample from replay buffers
#   "uniform"             — equal probability for all transitions
#   "prioritized_replay"  — sample high-TD-error transitions more often
#                           (alpha controls how much, beta controls IS correction)
SAMPLING="uniform"
#
# offline_episodes: how many demo episodes to load into the offline buffer
# The Lift MH dataset has 300 episodes. Using fewer = faster loading.
#   50  = recommended by task-specific config (small but sufficient for Lift)
#   100 = more demo data for stabler training
#   300 = all available (slower loading, more conservative training)
OFFLINE_EPISODES=300
#
# offline_fraction: what fraction of each training batch comes from offline data
#   0.5 = half offline, half online (standard RLPD)
#   0.0 = pure online RL (no demo mixing)
#   0.8 = mostly offline (very conservative, slow improvement)
OFFLINE_FRACTION=0.5

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9. LEARNING RATES                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# actor_lr: learning rate for the residual actor network
# CRITICAL: must be very small for residual RL! Too high → base policy
# corrections become too large → instability.
#   1e-7 = ultra conservative
#   1e-6 = standard for residual RL (recommended)
#   1e-5 = aggressive (risk of instability)
#
ACTOR_LR=1e-6
#
# critic_lr: learning rate for the Q-function (critic)
# Can be much higher than actor since critic doesn't directly affect actions.
#   1e-4 = standard (recommended)
#   3e-4 = faster critic convergence
#
CRITIC_LR=1e-4
#
# critic_target_tau: soft update rate for the target critic network
# target_params = tau * online_params + (1 - tau) * target_params
#   0.005 = standard for TD3 (recommended)
#   0.01  = faster target updates
CRITIC_TARGET_TAU=0.005

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  10. ACTOR NETWORK CONFIG                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# actor_last_layer_init_scale: initialization scale for the actor's LAST linear layer.
# For residual RL, start near zero so initial residual ≈ 0 (pure base policy).
#   0.0   = zero init → residual starts at exactly 0 (recommended for residual)
#   1e-3  = near-zero init
#   null  = PyTorch default init (NOT recommended for residual)
#
ACTOR_LAST_LAYER_INIT=0.0
#
# progressive_clipping_steps: linearly ramp up the residual action magnitude
# from 0 to full over this many steps. Prevents sudden large residuals.
#   0      = disabled (residual at full scale from step 1)
#   50000  = ramp up over 50K steps
#   100000 = slow ramp for stability
#
PROGRESSIVE_CLIPPING=0

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  11. CRITIC NETWORK CONFIG                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# num_q: number of Q-function heads in the critic ensemble
# TD3 uses 2 (min of 2 for target). RED-Q styles use 10+ for better estimates.
#   2  = standard TD3
#   10 = default in this codebase (RED-Q style ensemble, recommended)
#
NUM_Q_HEADS=10
#
# min_q_heads: how many Q-heads to take min over for target computation
# Standard TD3 uses 2 (clipped double Q-learning).
MIN_Q_HEADS=2
#
# policy_gradient_type: how to compute the actor's policy gradient from critic
#   "ensemble_mean"   — mean over all Q-heads (RED-Q style, less conservative)
#   "min_random_pair"  — min of 2 random heads (standard RED-Q)
#   "q1"              — just use first Q-head (standard TD3)
POLICY_GRADIENT_TYPE="ensemble_mean"
#
# Critic loss type:
#   "mse"       — standard MSE loss
#   "hl_gauss"  — Huber-like Gaussian (HL-Gauss) distributional RL
#   "c51"       — Categorical distributional RL
CRITIC_LOSS_TYPE="mse"
#
# v_min / v_max: value range for distributional critic (hl_gauss / c51 only). Since CRITIC_LOSS_TYPE is "mse" (not distributional), these are not used in this config. If you switch to a distributional loss, set these according to the expected return range:
# For sparse binary reward with γ=0.995 and horizon=100:
#   max possible return = γ^0 * 1 = 1.0 (success at last step)
#   v_min=0.0, v_max=1.0 covers the full range
#
# For DENSE reward (reward_shaping=True in robosuite, NOT currently enabled):
#   Per-step reward is in [0, 1.0] (robosuite normalizes by reward_scale/2.25)
#   Components: reaching [0,1] + grasping {0, 0.25} + lifting {0, 1} = max 2.25
#   Normalized: max 1.0 per step
#   With γ=0.995 and horizon=100: max return ≈ Σ γ^t * 1.0 for t=0..99 ≈ 63.5
#   Realistically a good policy might see returns of 20-40.
#   Safe range for dense:  v_min=0.0, v_max=70.0
#   To enable dense reward: edit dexmg.py → add reward_shaping=True to
#   robosuite.make() call in DexMimicGenEnv.__init__() around line 214.
#   Also set reward_scale=1.0 in robosuite.make() to keep the [0, 2.25] → [0, 1.0]
#   normalization (this is already the default).
V_MIN=0.0
# V_MAX=1.0
V_MAX=1.0

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  12. EVALUATION                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# eval_num_envs: parallel environments for evaluation
# Each eval env runs in a separate process (AsyncVectorEnv).
# More = faster eval but more RAM/VRAM. Capped by CPU count.
#   4  = safe for any machine
#   10 = good for 16+ core machines (recommended)
#   20 = for high-core-count servers
EVAL_NUM_ENVS=10
#
# eval_num_episodes: total episodes per evaluation round
#   20 = quick sanity check
#   50 = standard (good statistical estimate, default)
EVAL_NUM_EPISODES=50
#
# eval_interval: evaluate every N environment steps
#   5000  = frequent eval (for debugging / short runs)
#   10000 = standard (recommended)
EVAL_INTERVAL=5000
#
EVAL_HORIZON=500
#
# eval_first: run an evaluation at step 0 (before any training)
# Useful to measure base policy performance as a baseline.
EVAL_FIRST="true"
#
# save_video: record evaluation rollout videos and upload to WandB
SAVE_VIDEO="true"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  13. CHECKPOINTING                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# save_freq: save a checkpoint every N environment steps
# Setting -1 disables periodic saving (only best model is saved on eval).
SAVE_FREQ=5000
#
# no_cleanup: keep all local checkpoints after training finishes
# true  = keep everything (useful if WandB upload might fail)
# false = auto-clean old checkpoints to save disk
NO_CLEANUP="false"
#
# resume_ckpt: path to a checkpoint.pt to resume training from
# Leave empty for fresh training. Set to a path to resume.
# Example: "run_2026-.../models/latest/checkpoint.pt"
RESUME_CKPT=""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  14. ENVIRONMENT                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# headless: true = EGL offscreen rendering (server, fast)
#           false = GLFW on-screen MuJoCo viewer (for visualization)
HEADLESS="true"
#
# rl_camera: which camera images the critic/actor see
# For Lift (single Panda): agentview + wrist camera
# These must match the cameras in the dataset.
RL_CAMERA='[observation.images.agentview,observation.images.robot0_eye_in_hand]'
#
# video_key: which camera to use for eval video recording
VIDEO_KEY="observation.images.agentview"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  15. REWARD SHAPING                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Controls whether the env returns dense shaped rewards per step or only a
# sparse binary reward (1.0) on task completion.
#
# IMPORTANT: This must be CONSISTENT between the env AND the offline dataset.
# If your dataset was converted with dense rewards (HDF5 has "rewards" key
# and you used the updated conversion script), set reward_shaping=true.
# If your dataset has no rewards (e.g. ankile's pre-built datasets), use false.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  Sparse (false):                                                        │
# │    - Env reward: 0 every step, 1.0 on successful lift                  │
# │    - Offline reward: 0 or 1 (from done flag)                           │
# │    - Q-value range: [0, 1]                                             │
# │                                                                         │
# │  Dense (true):                                                          │
# │    - Env reward: [0, 0.556] per step (reaching+grasping), 1.0 on lift  │
# │    - Offline reward: read from dataset's next.reward field              │
# │    - Q-value range: [0, ~65]                                           │
# │    - Requires dataset converted with rewards (--shaped flag in HDF5)   │
# └─────────────────────────────────────────────────────────────────────────┘
#
REWARD_SHAPING="false"                 # ← for sparse reward (default, recommended for Lift)
# REWARD_SHAPING="true"                  # ← for dense reward training

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  16. IMAGE RESOLUTION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ Environment images:                                                     │
# │   Camera images from the simulator are 84×84 pixels.                    │
# │   Hardcoded default in DexMimicGenEnv / create_vectorized_env.          │
# │                                                                         │
# │ Offline dataset images:                                                 │
# │   If your HuggingFace dataset has a DIFFERENT resolution (e.g. 256×256) │
# │   set IMAGE_SIZE below to resize them to match the env (84×84).         │
# │   This must also match the resolution the BC policy was trained on.     │
# │                                                                         │
# │ Why it matters:                                                         │
# │   - The RL critic's ViT expects 84×84 (PatchEmbed2 num_patch=81)       │
# │   - Online (env) images are 84×84; offline images must match            │
# │   - The BC base policy also runs on offline images during buffer fill   │
# │   - Mismatched resolutions will crash on batch concatenation            │
# └─────────────────────────────────────────────────────────────────────────┘
#
# IMAGE_SIZE: resize dataset images to this square size before storing in
# the offline replay buffer.  Leave empty to use native dataset resolution.
# Set to 84 if your dataset is 256×256 but the env/BC use 84×84.
#
# IMAGE_SIZE=""
IMAGE_SIZE=84                           # ← uncomment if dataset is not 84×84

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  16. IMAGE RESOLUTION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  17. WANDB                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
WANDB_PROJECT=${WANDB_PROJECT:-"robomimic-can-residual-td3-ablation-studies"}
WANDB_NAME=${WANDB_NAME:-"sparse_with_25k_offlineRL_300epi_02resScale_0025std_ratio025_Run1"}               # empty = auto-generated name with params + seed
WANDB_GROUP=${WANDB_GROUP:-""}
WANDB_ENTITY=${WANDB_ENTITY:-""}             # empty = default entity
WANDB_MODE=${WANDB_MODE:-"online"}         # "online", "offline", "disabled"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  18. SEED & DEBUG                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# seed: random seed for reproducibility
# If empty → auto-generated random seed (logged in run name)
# Seeds: python random, np.random, torch, torch.cuda
# NOTE: robosuite env seeding is incomplete — see Notes.md for details.
SEED="42"
#
# torch_deterministic: set torch.backends.cudnn.deterministic
# true = slower but more reproducible. false = default (faster).
TORCH_DETERMINISTIC="false"
#
# debug: enables verbose logging and SyncVectorEnv (easier debugging)
DEBUG="false"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  19. ADVANCED / RARELY CHANGED                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# num_envs: training environments (must be 1 due to n-step implementation)
NUM_ENVS=1
#
# prefetch_batches: background batch prefetching for GPU pipeline
#   0 = disabled, 4 = recommended, 8 = for fast GPUs
PREFETCH=4
#
# actor_updates_per_iteration: actor updates per UTD cycle
# TD3 delays actor updates relative to critic. Default 1 = update actor every UTD cycle.
ACTOR_UPDATES_PER_ITER=1
#
# update_every_n_steps: how many env steps between each training update cycle
UPDATE_EVERY_N_STEPS=1
#
# Gradient clipping norms
CRITIC_GRAD_CLIP=1.0
ACTOR_GRAD_CLIP=1.0
#
# LR warmup for actor (0 = disabled)
ACTOR_LR_WARMUP_STEPS=0
#
# BC loss coefficient (regularize actor toward base policy behavior)
# 0.0 = pure RL (default). >0 = mixed RL+BC objective.
BC_LOSS_COEF=0.0
#
# L2 regularization on action magnitude (penalizes large residuals)
# 0.0 = disabled (default). Small values like 0.001 can help stability.
ACTION_L2_REG=0.0
#
# Encoder freezing: freeze the vision backbone (no gradient updates)
# Can speed up training if base policy's visual features are good enough.
FREEZE_ENCODER="false"
#
# Target action noise (TD3 policy smoothing)
TARGET_ACTION_NOISE="true"

# ╔══════════════════════════════════════════════════════════════╗
# ║  20. STAGE-AWARE EXTERNAL REWARD MODEL (SARM / TCC via HTTP)           ║
# ╚══════════════════════════════════════════════════════════════╝
#
# When enabled, the training env reward is replaced by a Potential-Based Reward
# Shaping (PBRS) signal driven by an external reward model served over HTTP:
#   r = r_sparse + gamma * phi(stage') - phi(stage)
# Start the reward server FIRST (SARM: opensarm/reward_server.py, TCC:
# tcc_torch/reward_server.py) on ${REWARD_SERVER_URL}. Training aborts at startup
# if the server is unreachable (require_server defaults true).
#
# NOTE: enabling this also sets algo.terminated_only_bootstrap=true so the PBRS
# potentials telescope correctly under n-step returns.
#
REWARD_MODEL_ENABLED="false"                 # ← set true to use the reward model
REWARD_MODEL_BACKEND="sarm"                  # "sarm" | "tcc"
REWARD_SERVER_URL="http://127.0.0.1:8001"
REWARD_TASK_PROMPT="pick up the can and place it in the bin"
REWARD_QUERY_EVERY_K=5                        # query every k env steps (20Hz -> 4Hz)
REWARD_HYSTERESIS_K=5                         # consecutive confs to advance a stage
REWARD_CONF_THRESHOLD=0.80                    # min stage confidence for an upgrade
REWARD_POTENTIALS="[0.0,0.05,0.1,0.15]"       # per-stage potentials (4 stages)
REWARD_KEEP_SPARSE_TERM="true"                # keep sparse success reward in PBRS
REWARD_IMAGE_VFLIP="false"                    # vertically flip frames if needed
TERMINATED_ONLY_BOOTSTRAP="true"              # required for PBRS (see note above)

# Offline PBRS reward labeling (optional). Point at a sidecar parquet produced by
# scripts/label_offline_pbrs.py to feed matching PBRS rewards into the OFFLINE
# buffer (read like next.reward, keyed by global frame index). Leave empty to use
# the dataset's own sparse/dense reward. Generic: works for any labeled column.
OFFLINE_REWARD_PARQUET=""
OFFLINE_REWARD_COLUMN="reward_pbrs"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  21. OFFLINE RL (TD3-BC)                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
TRAIN_OFFLINE_RL="true"
OFFLINE_RL_STEPS=25000
OFFLINE_SAVE_FREQ=5000
OFFLINE_EVAL_INTERVAL=5000
OFFLINE_RL_LOAD_CKPT="latest"
OFFLINE_RL_SAMPLING_METHOD="fixed_ratio"
OFFLINE_RL_OFFLINE_RATIO=0.25


# ============================================================================
# BUILD AND RUN
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Residual RL Training — TD3 for Lift (Panda)                "
echo "║  Base policy: ${BASE_WANDB_ID} (${BASE_WT_TYPE})            "
echo "║  Steps: ${TOTAL_TIMESTEPS}  |  UTD: ${UTD}  |  γ: ${GAMMA} "
echo "║  Action scale: ${ACTION_SCALE}  |  stddev: ${STDDEV_MAX}→${STDDEV_MIN}"
echo "║  Buffer: ${BUFFER_SIZE}  |  Offline eps: ${OFFLINE_EPISODES}"
echo "║  Actor LR: ${ACTOR_LR}  |  Critic LR: ${CRITIC_LR}         "
echo "║  WandB: ${WANDB_PROJECT}                                     "
echo "╚══════════════════════════════════════════════════════════════╝"

# Build the command with all overrides
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

    # ── Offline RL Phase (TD3-BC) ──
    algo.train_offline_rl="${TRAIN_OFFLINE_RL}"
    algo.offline_rl_steps="${OFFLINE_RL_STEPS}"
    algo.offline_save_freq="${OFFLINE_SAVE_FREQ}"
    algo.offline_eval_interval_every_steps="${OFFLINE_EVAL_INTERVAL}"
    algo.offline_rl_load_ckpt="${OFFLINE_RL_LOAD_CKPT}"
    algo.offline_rl_sampling_method="${OFFLINE_RL_SAMPLING_METHOD}"
    algo.offline_rl_offline_ratio="${OFFLINE_RL_OFFLINE_RATIO}"

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
    eval_horizon="${EVAL_HORIZON}"
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
[[ -n "${WANDB_ENTITY}" ]] && CMD+=(wandb.entity="${WANDB_ENTITY}")
[[ -n "${SEED}" ]]         && CMD+=(seed="${SEED}")
[[ -n "${RESUME_CKPT}" ]]  && CMD+=(resume_ckpt="${RESUME_CKPT}")
[[ -n "${IMAGE_SIZE}" ]]   && CMD+=(offline_data.image_size="${IMAGE_SIZE}")

# Local base-policy checkpoint (takes priority over W&B)
[[ -n "${BASE_LOCAL_PATH}" ]] && CMD+=(base_policy.local_path="${BASE_LOCAL_PATH}")

# Offline sidecar reward (PBRS / any labeled column) for the offline buffer
[[ -n "${OFFLINE_REWARD_PARQUET}" ]] && CMD+=(offline_data.reward_parquet="${OFFLINE_REWARD_PARQUET}" offline_data.reward_column="${OFFLINE_REWARD_COLUMN}")

# Stage-aware external reward model (PBRS). Requires the reward server running.
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
        algo.terminated_only_bootstrap="${TERMINATED_ONLY_BOOTSTRAP}"
    )
fi

"${CMD[@]}"

echo ""
echo "✓ Residual RL training complete."
echo "  Check WandB: ${WANDB_PROJECT}"
