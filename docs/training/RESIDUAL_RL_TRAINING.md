# Residual RL Training — Complete Deep Dive

This document explains every aspect of the Residual RL (Reinforcement Learning) fine-tuning pipeline in this project. It follows the same structure as [../policies/BC_POLICY_TRAINING.md](../policies/BC_POLICY_TRAINING.md) — tracing from the training command through every file, function, parameter, and design decision.

**Pre-requisite**: You must have a trained BC policy (see [../policies/BC_POLICY_TRAINING.md](../policies/BC_POLICY_TRAINING.md)) before running residual RL training. The BC policy is frozen and used as a "base" — the RL agent learns a small correction on top.

---

## Table of Contents

1. [What is Residual RL Training?](#1-what-is-residual-rl-training)
2. [The Training Command — Every Parameter Explained](#2-the-training-command--every-parameter-explained)
3. [Complete Execution Flow](#3-complete-execution-flow)
4. [Base Policy Loading](#4-base-policy-loading)
5. [Action & State Normalization](#5-action--state-normalization)
6. [Environment Wrapper — BasePolicyVecEnvWrapper](#6-environment-wrapper--basepolicyvecenvwrapper)
7. [The QAgent — Vision Encoder, Actor, Critic](#7-the-qagent--vision-encoder-actor-critic)
8. [Replay Buffers — Online and Offline](#8-replay-buffers--online-and-offline)
9. [The Offline Buffer — Converting Demonstrations to Transitions](#9-the-offline-buffer--converting-demonstrations-to-transitions)
10. [Warmup Phase — Random Exploration](#10-warmup-phase--random-exploration)
11. [Critic Warmup Phase](#11-critic-warmup-phase)
12. [The Main Training Loop — Step by Step](#12-the-main-training-loop--step-by-step)
13. [Critic Update — How Q-Values Are Learned](#13-critic-update--how-q-values-are-learned)
14. [Actor Update — How the Residual Policy Improves](#14-actor-update--how-the-residual-policy-improves)
15. [Evaluation During Training](#15-evaluation-during-training)
16. [Logging and Checkpointing](#16-logging-and-checkpointing)
17. [File Reference Map](#17-file-reference-map)
18. [Extra Flags: `headless=false`, `eval_num_envs`, and `--no_cleanup`](#18-extra-flags-headlessfalse-eval_num_envs-and---no_cleanup)
19. [Reward Flow, Critic Loss Types, and v_min/v_max](#19-reward-flow-critic-loss-types-and-v_minv_max)

**Concept Deep-Dive Documents** (linked inline):
- [../algorithms/TD3_ALGORITHM.md](../algorithms/TD3_ALGORITHM.md) — Twin Delayed DDPG with math + examples
- [../algorithms/NSTEP_RETURNS.md](../algorithms/NSTEP_RETURNS.md) — Multi-step TD learning with numerical walkthrough
- [../algorithms/REPLAY_BUFFERS.md](../algorithms/REPLAY_BUFFERS.md) — Prioritized replay, online/offline mixing, prefetching
- [../algorithms/RESIDUAL_LEARNING.md](../algorithms/RESIDUAL_LEARNING.md) — Why residual RL works, zero initialization, action scaling
- [../algorithms/ACTION_NORMALIZATION.md](../algorithms/ACTION_NORMALIZATION.md) — ActionScaler + StateStandardizer with worked examples
- [CHECKPOINTING_AND_RESUME.md](CHECKPOINTING_AND_RESUME.md) — Checkpointing, resume, RL vision encoder, buffer lifecycle, memory budget
- [../rewards/REWARD_AND_SUCCESS.md](../rewards/REWARD_AND_SUCCESS.md) — Reward signal, success criteria for all 12 tasks, n-step returns, eval metrics

---

## 1. What is Residual RL Training?

Residual RL is the **second stage** of the ResFiT pipeline. Instead of training an RL policy from scratch, you:

1. **Freeze** the BC policy trained in stage 1 (it never changes again)
2. **Add a small residual network** that outputs a correction $a_{\text{res}}$
3. **Combine** them: $a_{\text{final}} = a_{\text{base}} + a_{\text{res}}$
4. **Train only the residual** using TD3 (an off-policy actor-critic RL algorithm)

The key insight: the BC policy already knows roughly what to do. The residual only needs to fix mistakes — so it's initialized to output **zero** and trained with a **small action scale** (±0.2 in normalized space). This makes learning dramatically easier than training RL from scratch.

> **Clarification: "initialized to output zero" means weight initialization, NOT a runtime override.** The last layer's weights and biases are set to exactly 0.0 via `nn.init.normal_(weight, mean=0, std=0)`. Since the network ends with `Linear → Tanh` and `tanh(0) = 0`, the output is zero at initialization. From the very first training step, the actor is free to learn nonzero outputs — there is no "override phase" that holds the residual at zero for some steps and then releases it. The `action_scale` keeps outputs small, and the extremely low learning rate (`1e-6`) ensures the residual drifts away from zero gradually.

See [../algorithms/RESIDUAL_LEARNING.md](../algorithms/RESIDUAL_LEARNING.md) for the full mathematical motivation, zero-initialization trick, and worked examples.

### Algorithm: TD3 + RLPD Recipe (Hybrid)

The core algorithm is **TD3 (Twin Delayed DDPG)** — an off-policy actor-critic method with three key improvements over DDPG:
1. **Twin critics** — two Q-networks to combat overestimation bias
2. **Delayed actor updates** — actor updated less frequently than critic
3. **Target policy smoothing** — noise added to target actions

On top of TD3, this codebase layers **RLPD's (Reinforcement Learning with Prior Data) recipe**:
- **50/50 symmetric sampling** from demo buffer + online buffer
- **LayerNorm** in critic networks for stability

And further additions from other works:
- **10 Q-heads** (RED-Q style ensemble) instead of just 2, takes the **min over 2 random heads** for targets, and uses **ensemble mean** for actor gradients
- **N-step returns** (from IBRL/CQN) for better credit assignment over long horizons

So it's accurate to call it "**Residual TD3 with RLPD-style training recipe**" — the paper refers to the overall approach as RLPD, but the underlying RL algorithm is TD3.

See [../algorithms/TD3_ALGORITHM.md](../algorithms/TD3_ALGORITHM.md) for the complete algorithm with math, pseudocode, and examples.

---

## 2. The Training Command — Every Parameter Explained

Here is the full training command from the README:

```bash
python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name residual_td3_box_clean_config \
    base_policy.wandb_id=dexmg-box-clean-bc/<run_id> \
    algo.prefetch_batches=4 \
    algo.n_step=5 \
    algo.gamma=0.995 \
    algo.learning_starts=10_000 \
    algo.critic_warmup_steps=10_000 \
    algo.total_timesteps=500_000 \
    algo.num_updates_per_iteration=4 \
    algo.buffer_size=300_000 \
    algo.stddev_max=0.025 \
    algo.stddev_min=0.025 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=1e-6
```

### Config System: Hydra

Unlike the BC training script (which uses argparse), the RL training uses **Hydra** — a configuration framework. The `--config-name` selects a registered dataclass config (e.g., `ResidualTD3BoxCleanConfig`), and every dot-separated override modifies a nested field.

**Config file**: `resfit/rl_finetuning/config/residual_td3.py`
**Parent config**: `resfit/rl_finetuning/config/rlpd.py`

### Parameter Reference Table

#### Base Policy

| Parameter | Value | Description |
|---|---|---|
| `base_policy.wandb_id` | `dexmg-box-clean-bc/<run_id>` | WandB run path to download the frozen BC policy. Format: `project/run_id`. |
| `base_policy.wt_type` | `best` (default) | Which checkpoint: `best` (highest eval success) or `last`. |
| `base_policy.wt_version` | `latest` (default) | WandB artifact version. |

#### Algorithm Hyperparameters

| Parameter | Default | Set to | Description |
|---|---|---|---|
| `algo.total_timesteps` | 500,000 | 500,000 | Total environment steps (not gradient updates). |
| `algo.batch_size` | 256 | 256 | Transitions per gradient update. Split 50/50 online/offline. |
| `algo.buffer_size` | 200,000 | 300,000 | Online replay buffer capacity. |
| `algo.learning_starts` | 10,000 | 10,000 | Collect this many transitions before training starts (warmup). |
| `algo.gamma` | 0.99 | 0.995 | Discount factor. Higher value → agent cares more about future rewards. |
| `algo.n_step` | 3 | 5 | N-step return horizon. See [../algorithms/NSTEP_RETURNS.md](../algorithms/NSTEP_RETURNS.md). |
| `algo.num_updates_per_iteration` | 4 | 4 | Gradient updates per env step (UTD ratio). Critic updated all 4 times; actor updated once. |
| `algo.actor_updates_per_iteration` | 1 | 1 | Actor updates per iteration cycle. With UTD=4, actor updates every 4th step. |
| `algo.update_every_n_steps` | 1 | 1 | Env steps between gradient update cycles. |
| `algo.offline_fraction` | 0.5 | 0.5 | Fraction of batch from offline buffer. batch_size × 0.5 = 128 online + 128 offline. |
| `algo.critic_warmup_steps` | 10,000 | 10,000 | Critic-only gradient updates before actor training begins. |
| `algo.prefetch_batches` | 4 | 4 | Background threads pre-load this many batches from the replay buffer. |
| `algo.random_action_noise_scale` | 0.2 | 0.2 | Noise scale during warmup exploration (±0.2 in normalized action space). |
| `algo.use_base_policy_for_warmup` | `true` | `true` | During warmup: base_action + noise (true) vs pure random (false). |
| `algo.progressive_clipping_steps` | 0 | 0 | If >0, linearly ramps residual clip from 0→action_scale over this many steps. |

#### Exploration Noise (Standard Deviation Schedule)

| Parameter | Default | Set to | Description |
|---|---|---|---|
| `algo.stddev_max` | 0.05 | 0.025 | Starting exploration noise std. |
| `algo.stddev_min` | 0.05 | 0.025 | Final exploration noise std. |
| `algo.stddev_step` | 300,000 | 300,000 | Steps over which noise linearly decays from max to min. |

The schedule is generated as `linear(0.025,0.025,300000)` — since max=min, the noise is **constant at 0.025** throughout training. This is very small because the base policy already provides good actions.

#### Agent Network Configuration

| Parameter | Default | Set to | Description |
|---|---|---|---|
| `agent.actor_lr` | 1e-6 | 1e-6 | Actor learning rate. Extremely low to prevent the residual from destabilizing the base policy. |
| `agent.critic_lr` | 1e-4 | 1e-4 | Critic learning rate. 100× higher than actor — the critic learns faster. |
| `agent.critic_target_tau` | 0.005 | 0.005 | Polyak averaging coefficient for target network soft updates. |
| `agent.actor.action_scale` | 0.1 | 0.2 | Residual output capped to ±0.2 in normalized action space (Tanh × 0.2). |
| `agent.actor.actor_last_layer_init_scale` | 0.0 | 0.0 | Last layer weights initialized to 0 → residual starts outputting exactly zero. |
| `agent.critic.num_q` | 10 | 10 | Number of Q-heads in the critic ensemble (RED-Q style). |
| `agent.critic.policy_gradient_type` | `ensemble_mean` | `ensemble_mean` | Actor loss uses mean of all Q-heads. |
| `agent.critic.min_q_heads` | 2 | 2 | Target Q computed as min over 2 random heads from the ensemble. |
| `agent.critic_grad_clip_norm` | 1.0 | 1.0 | Max gradient norm for critic + encoder. |
| `agent.actor_grad_clip_norm` | 1.0 | 1.0 | Max gradient norm for actor. |

#### Offline Data Configuration

| Parameter | Default | Description |
|---|---|---|
| `offline_data.name` | task-dependent | HuggingFace dataset ID (e.g., `ankile/dexmg-two-arm-box-cleanup`). |
| `offline_data.num_episodes` | task-dependent | Number of episodes to use (e.g., 1000). |
| `offline_data.use_base_policy_for_base_actions` | `true` | Use base policy to generate base actions in offline data (more realistic). |
| `offline_data.min_action_range` | 0.1 | Minimum per-dimension action range for normalization safety. |
| `offline_data.min_state_std` | 0.1 | Minimum per-dimension state std for normalization safety. |

#### Evaluation

| Parameter | Default | Description |
|---|---|---|
| `eval_interval_every_steps` | 10,000 | Evaluate every N environment steps. |
| `eval_num_envs` | 4 | Parallel evaluation environments (capped at CPU count - 1). Reduce on memory-constrained systems — each eval env spawns a full MuJoCo process with all cameras. On a 32 GB laptop, 4 is safe; 8 may OOM-kill the process. |
| `eval_num_episodes` | 50 | Episodes per evaluation round. |
| `eval_first` | `true` | Run evaluation at step 0 (before any training). |
| `save_video` | `true` | Record evaluation videos to WandB. |

---

## 3. Complete Execution Flow

Here is the entire training pipeline, start to finish:

```
Entry point: python -m resfit.rl_finetuning.scripts.train_residual_td3
       │
       ▼
1. Thread Limiting (OMP_NUM_THREADS=1, etc.)
       │
       ▼
2. Hydra config loading → ResidualTD3CoffeeConfig
       │
       ▼
3. MuJoCo GL backend selection
   headless=True  → MUJOCO_GL=egl   (offscreen GPU, default)
   headless=False → MUJOCO_GL=glfw  (on-screen viewer window)
       │
       ▼
4. Download & load frozen BC base policy from WandB
       │
       ▼
5. Load dataset → compute ActionScaler & StateStandardizer from stats
       │
       ▼
6. Create training env + eval env (wrapped in BasePolicyVecEnvWrapper)
   ⚠ If headless=False, env creation opens a MuJoCo viewer window here.
   The window appears but stays blank/frozen until env.step() is called.
       │
       ▼
7. Create QAgent (ViT encoders, Actor, Critic, target networks, optimizers)
       │
       ▼
8. Create online replay buffer (TensorDictPrioritizedReplayBuffer + MultiStepTransform)
   Create offline replay buffer (same type)
       │
       ▼
9. ★ SLOW STEP ★ Populate offline buffer from dataset demonstrations
   Iterates through ALL 326K frames, decoding video frames via torchcodec
   and converting pairs into (s, a, r, s') transitions → ~321K transitions.
   Takes ~30+ minutes on first run. Cached to disk for reuse.
   (or loads from cache in seconds if available)
       │
       ▼
10. Warmup: collect 10,000 random transitions into online buffer
    (base_action + uniform noise as residual)
    🖥 If headless=False, MuJoCo viewer starts animating here
    (render_viewer() is called after each env.step() in the training loop)
       │
       ▼
11. ★ wandb.init() happens HERE ★
    The wandb run URL is printed to the terminal AFTER steps 9 and 10.
    Until this point there is no wandb logging.
       │
       ▼
12. Critic warmup: 10,000 critic-only gradient updates
    (no actor training — learn to predict Q-values first)
       │
       ▼
13. ┌─── Main Training Loop (500,000 env steps) ───┐
    │                                               │
    │  a. Agent selects residual action             │
    │  b. Env executes base + residual              │
    │     🖥 render_viewer() updates on-screen view │
    │  c. Transition added to online buffer         │
    │  d. Every 10K steps: evaluate & log to wandb  │
    │  e. 4 gradient updates per env step:          │
    │     - 4 critic updates                        │
    │     - 1 actor update (every 4th)              │
    │                                               │
    └───────────────────────────────────────────────┘
       │
       ▼
14. Cleanup (delete local cache, final WandB sync)
```

> **Why haven't I seen a wandb URL yet?** Because `wandb.init()` is at step 11, AFTER the offline buffer population (step 9) and warmup (step 10). If the training is still in step 9 (the progress bar showing 71%), wandb hasn't initialized yet. The URL will appear after the `"Launching run with the following config:"` print statement.

> **Why is the offline buffer loading slow and seemingly "bigger" than BC training data?**
> During BC training, the DataLoader reads frames on-the-fly — each batch only decodes a small number of video frames.
> During RL offline buffer population, the script iterates through **every single frame** in the dataset sequentially (326,707 frames) and decodes the video for each one via torchcodec.
> For each consecutive pair of frames within an episode, it creates a transition `(s_t, a_t, r_t, s_{t+1})` and stores it in a replay buffer in RAM.
> This means **every video frame gets decoded exactly once** upfront, unlike BC where decoding happens lazily per batch.
> The ~321K transitions (with 3 camera images each stored as uint8) consume several GB of RAM.
> The good news: this is **cached to disk** after the first run. On subsequent runs with the same config, it loads from cache in seconds.

### Thread Limiting (Lines 8–17)

Before any import, the script caps BLAS/OpenMP threads:

```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
```

**Why?** Each PyTorch operation spawns $N$ threads by default. With many operations per step, these threads contend with each other, causing severe slowdown. Setting to 1 forces single-threaded math which, counterintuitively, is **faster** for RL training where operations are small and frequent.

---

## 4. Base Policy Loading

**File**: `resfit/rl_finetuning/scripts/train_residual_td3.py` (lines 220–240)

```python
policy_dir, _ = download_policy_from_wandb(
    cfg.base_policy.wandb_id,        # "dexmg-box-clean-bc/<run_id>"
    step=cfg.base_policy.wt_type,    # "best"
    artifact_version=cfg.base_policy.wt_version,  # "latest"
)
base_policy: ACTPolicy = load_policy(policy_dir)
base_policy.to(device)
base_policy.eval()
```

This downloads the BC checkpoint from WandB as an artifact, loads it as an `ACTPolicy`, moves it to GPU, and sets it to eval mode. The base policy is **never trained** — it only runs inference.

Two separate copies are loaded:
- `base_policy` — used in the **training** environment wrapper
- `eval_base_policy` — used in the **evaluation** environment wrapper

This prevents any state contamination between training and evaluation (the ACT policy has internal hidden states that get reset between episodes).

---

## 5. Action & State Normalization

**File**: `resfit/rl_finetuning/utils/normalization.py`

All actions and states are normalized before being used by the RL agent. This is critical because raw action dimensions can have vastly different ranges (e.g., joint angles in radians vs gripper 0/1).

See [../algorithms/ACTION_NORMALIZATION.md](../algorithms/ACTION_NORMALIZATION.md) for the complete mathematical treatment with worked examples.

### ActionScaler — Min-Max to [-1, 1]

Created from dataset statistics:

```python
action_scaler = ActionScaler.from_dataset_stats(
    action_stats=dataset.meta.stats["action"],  # has 'min' and 'max' per dim
    action_scale=cfg.agent.actor.action_scale,   # 0.2 — expands range by 20%
    min_range_per_dim=cfg.offline_data.min_action_range,  # 0.1 — safety floor
    device=device,
)
```

**Scaling formula** (per dimension $d$):

$$a_{\text{scaled}}^{(d)} = 2 \cdot \frac{a^{(d)} - a_{\min}^{(d)}}{a_{\max}^{(d)} - a_{\min}^{(d)}} - 1$$

where $a_{\min}, a_{\max}$ are expanded by `(1 + action_scale)` from the dataset min/max. The expansion ensures there's room for the residual to push beyond the demonstrated range.

> **Clarification: `action_scale` does NOT overwrite dataset statistics.** The dataset's `min` and `max` per action dimension define the center and shape of the normalization range. The `action_scale=0.2` passed from the command line tells the `ActionScaler` to **expand** that range by 20% (multiply the half-range by 1.2). This creates headroom so that when the residual adds ±0.2 to a base action near the boundary, the combined action can still be unscaled back to a valid raw action. The same `action_scale` value is also used inside the Actor to bound the residual output (`Tanh × 0.2`). They are **separate mechanisms connected by the same config value** — one bounds the residual, the other expands the normalization range to accommodate it.

### StateStandardizer — Mean-Std (Z-Score)

```python
state_standardizer = StateStandardizer.from_dataset_stats(
    state_stats=dataset.meta.stats["observation.state"],  # has 'mean' and 'std'
    min_std=cfg.offline_data.min_state_std,  # 0.1 — prevents division blow-up
    device=device,
)
```

**Formula**:

$$s_{\text{std}} = \frac{s - \mu_s}{\max(\sigma_s, 0.1)}$$

---

## 6. Environment Wrapper — BasePolicyVecEnvWrapper

**File**: `resfit/rl_finetuning/wrappers/residual_env_wrapper.py`

This wrapper is the **key architectural choice** of residual RL. It sits between the RL agent and the actual environment:

```
RL Agent (Actor)
    │ outputs residual action a_res ∈ [-0.2, 0.2]
    ▼
BasePolicyVecEnvWrapper
    │ 1. Combines: a_combined = a_base + a_res
    │ 2. Unscales: a_raw = unscale(a_combined)
    │ 3. Steps the real env with a_raw
    │ 4. Gets next observation from env
    │ 5. Runs frozen base policy: a_base_next = base_policy(next_obs)
    │ 6. Augments next_obs with a_base_next and standardized state
    ▼
Returns: next_obs (augmented), reward, terminated, truncated, info
```

### The reset() Method

```python
def reset(self):
    obs, info = self.vec_env.reset()                 # Reset real env
    self.base_policy.reset()                          # Reset ACT hidden states
    base_action = self.base_policy.select_action(obs) # Get first base action
    scaled_base = self.action_scaler.scale(base_action)
    obs = self._augment_obs(obs, scaled_base)         # Add base_action to obs
    return obs, info
```

### The step() Method

```python
def step(self, residual_naction):
    # 1. Combine base + residual
    combined = self.current_base_naction + residual_naction
    
    # 2. Unscale to raw action space
    raw_action = self.action_scaler.unscale(combined)
    
    # 3. Step real environment
    next_obs, reward, terminated, truncated, info = self.vec_env.step(raw_action)
    
    # 4. Store combined action for replay buffer
    info["scaled_action"] = combined.clone()
    
    # 5. Get next base action from frozen BC
    next_base_action = self.base_policy.select_action(next_obs)
    next_base_scaled = self.action_scaler.scale(next_base_action)
    
    # 6. Handle episode resets (re-run base policy on new episode obs)
    for i where terminated[i] or truncated[i]:
        # Reset base policy, get fresh base action for new episode
        ...
    
    # 7. Augment observation
    next_obs = self._augment_obs(next_obs, next_base_scaled)
    
    return next_obs, reward, terminated, truncated, info
```

### What `_augment_obs()` Does

Adds two things to the observation:
1. `observation.base_action` — the scaled base action from the frozen BC policy
2. Standardizes `observation.state` using the StateStandardizer

The RL agent's observation space thus contains:
- Camera images (one per camera, e.g., 3 cameras → 3 image tensors)
- `observation.state` — standardized proprioceptive state (joint positions, velocities, etc.)
- `observation.base_action` — what the frozen BC policy would do (in [-1, 1] normalized space)

### Full Environment Wrapper Stack

The RL agent never talks to robosuite directly. There are **four layers** of wrapping, each with a specific responsibility:

```
RL Agent (QAgent.actor)
    │  outputs residual_naction ∈ [-0.2, 0.2]  (torch tensor)
    ▼
Layer 4: BasePolicyVecEnvWrapper          ← resfit/rl_finetuning/wrappers/residual_env_wrapper.py
    │  • Combines base_action + residual → unscales to raw
    │  • Steps layer 3, then runs frozen base policy on next_obs
    │  • Augments obs with base_action + standardized state
    │  Methods: step(), reset(), render(), render_viewer(), close()
    ▼
Layer 3: VectorizedEnvWrapper             ← resfit/dexmg/environments/dexmg.py
    │  • Converts numpy ↔ torch tensors (observations, rewards, dones)
    │  • Thin adapter over Gymnasium's vectorized API
    │  Methods: step(), reset(), render(), render_viewer(), close()
    ▼
Layer 2: gymnasium.vector.SyncVectorEnv   ← Gymnasium built-in  (headless=False)
     or  gymnasium.vector.AsyncVectorEnv           〃              (headless=True)
    │  • Manages N parallel sub-environments
    │  • SyncVectorEnv: same process, sequential stepping
    │  • AsyncVectorEnv: separate processes (context="spawn"), shared memory
    │  Training always uses num_envs=1; eval uses eval_num_envs (default 4)
    ▼
Layer 1: RobosuiteGymWrapper              ← resfit/dexmg/environments/dexmg.py
    │  • Wraps the raw robosuite env with Gymnasium API
    │  • Converts robosuite obs dict → {observation.state, observation.images.*}
    │  • Handles episode termination on success (reward=1) for shortcircuiting
    │  Methods: step(), reset(), render() [offscreen frame], render_viewer() [GLFW]
    ▼
robosuite Environment (MuJoCo)
```

**Key design decisions:**

| Concern | Where handled | Notes |
|---|---|---|
| Action composition (base + residual) | Layer 4 | `BasePolicyVecEnvWrapper.step()` |
| Action unscaling ([-1,1] → raw) | Layer 4 | Uses `ActionScaler.unscale()` |
| Base policy inference | Layer 4 | Frozen ACT, runs on `next_obs` |
| Numpy ↔ Torch conversion | Layer 3 | `VectorizedEnvWrapper` |
| Parallelism (multi-env) | Layer 2 | `SyncVectorEnv` / `AsyncVectorEnv` |
| Observation processing | Layer 1 | `RobosuiteGymWrapper._process_obs()` |
| On-screen viewer | Layer 1 | `render_viewer()` → `self.env.render()` |
| Video frame capture | Layer 1 | `render()` → `sim.render()` (offscreen) |

**Rendering path (two separate flows):**

```
Video recording (eval):        On-screen viewer (training, headless=False):
  evaluate_dexmg.py              train_residual_td3.py
    → env.render()                 → env.render_viewer()
    → Layer 4 passthrough          → Layer 4 passthrough
    → Layer 3 → vec_env.render()   → Layer 3 → iterate SyncVectorEnv.envs[]
    → Layer 2 → each sub-env       → Layer 1 → self.env.render() (GLFW)
    → Layer 1 → sim.render()
      (offscreen RGB frame)          (on-screen window refresh)
```

---

## 7. The QAgent — Vision Encoder, Actor, Critic

**File**: `resfit/rl_finetuning/off_policy/rl/q_agent.py`

The QAgent contains all trainable networks:

```
                    Observation
                        │
     ┌──────────────────┼──────────────────┐
     │                  │                  │
 Camera 1          Camera 2          Camera 3
     │                  │                  │
 ViT Encoder 1    ViT Encoder 2    ViT Encoder 3
     │                  │                  │
 [81, 128]         [81, 128]         [81, 128]
     │                  │                  │
     └────── concat along patches ─────────┘
                        │
                   [243, 128]
                        │
           ┌────────────┼────────────┐
           │                         │
        Actor                     Critic
    (residual MLP)          (Q-value ensemble)
           │                         │
     a_res ∈ [-0.2, 0.2]    Q(s, a) × 10 heads
```

### Vision Encoder: MinViT

**File**: `resfit/rl_finetuning/off_policy/networks/min_vit.py`

Each camera gets its own Vision Transformer (ViT) encoder:

1. **Patch embedding** (`PatchEmbed2`): Two convolutions that convert an 84×84×3 image into 81 patches of dimension 128
   - `Conv2d(3, 128, kernel_size=8, stride=4)` → intermediate feature map
   - `GroupNorm(128, 128)` + `ReLU`
   - `Conv2d(128, 128, kernel_size=3, stride=2)` → 9×9 = 81 patches
   
2. **Positional embedding**: Learnable positional embeddings added to patches

3. **Transformer layer** (depth=1 by default): Standard pre-norm transformer with:
   - Multi-head self-attention (4 heads, embed_dim=128)
   - Feed-forward network (128 → 512 → 128)
   - Layer normalization + residual connections

4. **Output**: `[B, 81, 128]` — 81 patch tokens of dimension 128

With 3 cameras, the concatenated output is `[B, 243, 128]` (patches concatenated along dim=1).

**Data augmentation**: `RandomShiftsAug(pad=4)` — pads the image by 4 pixels on each side, then randomly crops back to original size. Applied to images during training but NOT during evaluation.

### Actor Network

**File**: `resfit/rl_finetuning/off_policy/rl/actor.py`

```
Visual features [B, 243, 128]   State [B, prop_dim]   Base action [B, action_dim]
        │                              │                       │
        ▼                              │                       │
  Linear(243×128, 128)                 │                       │
  LayerNorm(128)                       │                       │
  Dropout(0)                           │                       │
  ReLU                                 │                       │
        │                              │                       │
        └──────── concatenate ─────────┼───────────────────────┘
                        │
                 [B, 128 + prop_dim + action_dim]
                        │
              Linear(in, 1024) → LN → Dropout → ReLU
              Linear(1024, 1024) → LN → Dropout → ReLU
              Linear(1024, action_dim) → Tanh
                        │
                   mu ∈ [-1, 1]
                        │
                   × action_scale (0.2)
                        │
                scaled_mu ∈ [-0.2, 0.2]
                        │
           TruncatedNormal(scaled_mu, std=0.025)
                        │
                  a_res ~ sample()
```

**Critical detail — zero initialization**: The last `Linear` layer's weights and biases are initialized to 0.0 (`actor_last_layer_init_scale=0.0`). This means the actor initially outputs exactly `mu = [0, 0, ..., 0]`, so `a_res = 0` — the residual does **nothing** at the start. Training gradually pushes the residual away from zero only where the base policy needs correction.

### Critic Network (Q-Value Ensemble)

**File**: `resfit/rl_finetuning/off_policy/rl/critic.py`

The critic estimates "how good is action $a$ in state $s$" with an ensemble of 10 independent Q-heads:

```
Visual features [B, 243, 128]   State [B, prop_dim]   Action [B, action_dim]
        │                              │                       │
        ▼                              │                       │
  SpatialEmbQEnsemble                  │                       │
  (shared trunk)                       │                       │
        │                              │                       │
     Spatial attention                 │                       │
     (like Actor's SpatialEmb          │                       │
      but fuses patches + state)       │                       │
        │                              │                       │
        └──────── concatenate ─────────┼───────────────────────┘
                        │
              [B, spatial_emb + action_dim]
                        │
          ┌────────────────────────────────┐
          │    10 independent MLP heads     │
          │   (implemented via PyTorch vmap │
          │    for efficiency)              │
          │                                │
          │   Each: Linear(in, 1024)→LN→ReLU│
          │         Linear(1024, 1024)→LN→ReLU│
          │         Linear(1024, 1)         │
          └────────────────────────────────┘
                        │
               Q values: [10, B, 1]
```

The 10 heads share the spatial trunk but have **completely independent MLP weights** — they're ensembled using PyTorch's `vmap` + `functional_call` + `stack_module_state` for efficiency (this runs all 10 heads in a single batched operation rather than 10 sequential forward passes).

**How the ensemble is used**:
- **Target Q-value** (for TD target): Min over 2 random heads from the 10
- **Actor gradient** (for policy improvement): Mean over all 10 heads

See [../algorithms/TD3_ALGORITHM.md](../algorithms/TD3_ALGORITHM.md) for why this ensemble structure helps.

---

## 8. Replay Buffers — Online and Offline

**File**: `resfit/rl_finetuning/scripts/train_residual_td3.py` (lines 395–510)

Two separate replay buffers store transitions for training:

| Buffer | Source | Size | Purpose |
|---|---|---|---|
| **Online** | Agent + environment interaction | 300,000 | Real RL experience |
| **Offline** | Dataset demonstrations (converted) | ~num_episodes × avg_length | Stabilizing signal |

Every training batch is 50% online, 50% offline (`offline_fraction=0.5`).

Both use `TensorDictPrioritizedReplayBuffer` from TorchRL, with a `MultiStepTransform` that automatically computes n-step returns (see [../algorithms/NSTEP_RETURNS.md](../algorithms/NSTEP_RETURNS.md)).

See [../algorithms/REPLAY_BUFFERS.md](../algorithms/REPLAY_BUFFERS.md) for the full deep dive on prioritized replay, caching, and prefetching.

### Transition Format (TensorDict)

Each transition stored in the buffer:

```python
TensorDict({
    "obs": {
        "observation.images.agentview":        [3, 84, 84]  (uint8)
        "observation.images.robot0_eye_in_hand": [3, 84, 84]  (uint8)
        "observation.images.robot1_eye_in_hand": [3, 84, 84]  (uint8)
        "observation.state":                   [prop_dim]    (float32, standardized)
        "observation.base_action":             [action_dim]  (float32, scaled [-1,1])
    },
    "action":    [action_dim]     (float32, combined base+residual, scaled [-1,1])
    "next": {
        "obs": { ... same structure ... },
        "done":   bool,
        "reward": float32
    },
    "_priority": float32          (10.0 initially — high priority for new samples)
})
```

**Key detail**: Images are stored as **uint8** (0–255) to save memory. They're converted to float32 and divided by 255 when sampled for training.

---

## 9. The Offline Buffer — Converting Demonstrations to Transitions

**File**: `resfit/rl_finetuning/scripts/train_residual_td3.py`, `_populate_offline_buffer()` (lines 516–645)

The offline buffer converts the demonstration dataset (sequences of frames) into $(s, a, r, s')$ transitions. There are two modes:

### Mode 1: GT-as-Base (`use_base_policy_for_base_actions=False`)

- `base_action` = ground-truth action from the dataset
- `target_action` = same ground-truth action
- Residual target = GT - GT = **0** (teaches the residual to do nothing)

### Mode 2: Base-Policy-as-Base (`use_base_policy_for_base_actions=True`) ← Default

- `base_action` = what the frozen BC policy would predict for this observation
- `target_action` = ground-truth action from the dataset
- Residual target = GT - BC prediction ≈ the correction the residual should learn

**Mode 2 is more realistic** because at test time the residual will receive actual BC predictions (not GT actions), so the offline data distribution matches the online distribution better.

#### Why does Mode 2 run base policy inference on every dataset frame?

In Mode 2, for each observation in the dataset, the frozen BC policy is called via `base_policy.select_action(obs)` to generate the `observation.base_action` field. This is needed because:

1. **The residual actor takes `base_action` as input** — it needs to see what the base policy would suggest in order to decide how to correct it. So each transition in the offline buffer must have the base policy's actual prediction for that observation, not the GT action.
2. **The actions in the dataset are NOT replaced** — the ground-truth (GT) action from the dataset remains as the `action` field of the transition (= the combined target action). Only the `observation.base_action` field is populated by base policy inference.
3. **ACT's action chunking mitigates the cost** — ACT's `select_action()` maintains an internal action queue. On the first call, it runs one forward pass and generates `n_action_steps` actions (e.g., 10), then pops one. For the next 9 calls, it just pops from the queue without running the model. So the actual number of ACT forward passes is ~320k / `n_action_steps`, not 320k. However, the per-frame overhead (GPU data transfer for images, queue management) still adds up.
4. **The main bottleneck is video frame decoding** — even in Mode 1 (no base policy inference), decoding 320k video frames via torchcodec is slow. Mode 2 adds base policy inference on top of that.

**If dataset loading hangs or is too slow on a laptop**, see the tips below.

### Reward Structure

The reward is **sparse binary**: `reward = 1.0` if the episode terminates successfully (task completed), `reward = 0.0` otherwise. There are no intermediate rewards.

### Why is this step so slow?

The offline buffer population iterates through **every frame** in the dataset (326,707 frames for TwoArmCoffee) sequentially via a `DataLoader(batch_size=1, shuffle=False)`. For each frame, it:

1. **Decodes one video frame per camera** via torchcodec (the bottleneck — seeks into mp4 files)
2. Extracts and normalizes the state and action
3. Pairs consecutive frames into `(s_t, a_t, r_t, s_{t+1})` transitions
4. Stores the transition in the replay buffer (images as uint8 in CPU memory)

This is fundamentally different from BC training, where frames are decoded lazily in random-access fashion per mini-batch. Here, **every single frame** must be decoded upfront. The result is ~321K transitions (326K frames minus one per episode boundary), each containing 3 camera images, state, and action.

At ~160 it/s (mostly bound by torchcodec video seeking), this takes **~30 minutes** on first run.

Additionally, with `use_base_policy_for_base_actions=True` (the default), **base policy inference runs on every frame** — this moves image tensors to GPU and runs the ACT model forward pass (mitigated somewhat by action chunking, but still significant). On a laptop without a powerful GPU, this combined load (video decoding + ACT inference + RAM for ~321K image transitions) can cause the system to become unresponsive.

### Laptop-Friendly Tips

If the dataset population step causes your laptop to hang or run out of memory:

1. **Reduce the number of episodes** — add `offline_data.num_episodes=100` to your command. This cuts ~320k frames to ~32k, dramatically reducing time and memory.
2. **Use GT-as-base mode** — add `offline_data.use_base_policy_for_base_actions=false`. This skips base policy inference entirely (no ACT forward passes), using GT actions as both base and target. Faster to populate, trades some distribution matching quality.
3. **Skip offline data entirely** — add `algo.offline_fraction=0.0`. Creates a minimal unused offline buffer. You lose the RLPD-style 50/50 demo mixing benefit.
4. **Let it cache** — if you can get through it once, subsequent runs with the same config load from `~/.cache/residual_rl/offline_buffers/<hash>/` in seconds.

Recommended for laptops: **Option 1** (`offline_data.num_episodes=100`) — keeps the RLPD recipe intact with minimal episodes.

### Caching

Buffer population is expensive (requires decoding every video frame and optionally running the base policy on each). The result is **cached to disk** using a SHA-1 hash of the metadata (dataset name, episodes, image keys, n_step, gamma, sampling_strategy, torchrl version, etc.). On subsequent runs with the same configuration, the buffer is loaded from cache in seconds instead of being recomputed. The cache can also be uploaded to / downloaded from a HuggingFace Hub repository.

The cache directory is at `~/.cache/residual_rl/offline_buffers/<hash>/`.

> **Tip**: If the population crashes partway through (e.g., KeyboardInterrupt at 71%), no cache is saved — you have to start over. Let it finish!

---

## 10. Warmup Phase — Random Exploration

**File**: `resfit/rl_finetuning/scripts/train_residual_td3.py` (lines 730–795)

Before any training happens, the online buffer must be filled with at least `learning_starts = 10,000` transitions. During warmup, the agent does NOT use its actor network — instead, it explores with noise:

### With `use_base_policy_for_warmup=True` (default):

```python
rand_actions = (torch.rand(num_envs, action_dim) * 2 - 1) * random_action_noise_scale
# rand_actions ∈ [-0.2, 0.2]  (random_action_noise_scale=0.2)
```

The environment wrapper adds this to the base action: `executed = base_action + rand_actions`

This means warmup exploration is **centered on the BC policy** — the agent follows the BC policy with small random perturbations. This produces trajectories similar to what the BC policy would generate, providing a much better initial dataset than purely random actions.

### With `use_base_policy_for_warmup=False`:

```python
pure_random = (torch.rand(num_envs, action_dim) * 2 - 1) * random_action_noise_scale
rand_actions = pure_random - base_action
# After env combines: executed = base_action + (pure_random - base_action) = pure_random
```

This cancels out the base action entirely, resulting in pure random exploration. Not recommended for complex tasks.

---

## 11. Critic Warmup Phase

**File**: `resfit/rl_finetuning/scripts/train_residual_td3.py`, `_run_critic_warmup()` (lines 856–930)

After the online buffer is filled but **before** actor training starts, the critic runs `critic_warmup_steps = 10,000` gradient updates with `update_actor=False`.

**Why?** If the actor starts training immediately, it optimizes against a random, untrained critic. The actor would learn a garbage policy based on meaningless Q-value estimates. By pre-training the critic, the Q-values become somewhat meaningful before the actor starts "listening" to them.

Each warmup step:
1. Sample 128 online + 128 offline transitions
2. Compute TD target: $y = r + \gamma \cdot Q_{\text{target}}(s', a')$
3. Update critic weights: minimize $(Q(s,a) - y)^2$
4. Soft-update target critic: $\theta' \leftarrow \tau \theta + (1-\tau) \theta'$
5. Do **not** update actor

---

## 12. The Main Training Loop — Step by Step

**File**: `resfit/rl_finetuning/scripts/train_residual_td3.py` (lines 942–1080)

For each of 500,000 environment steps:

### Step (1): Collect Action + Env Step

```python
stddev = schedule("linear(0.025,0.025,300000)", global_step)  # = 0.025
action = agent.act(obs, eval_mode=False, stddev=stddev)       # residual action
next_obs, reward, terminated, truncated, info = env.step(action)
# env internally does: combined = base_action + action
```

### Step (2): Add to Online Buffer

```python
combined_action = info["scaled_action"]  # The actually executed base+residual
_add_transitions_to_buffer(obs, next_obs, combined_action, reward, done, ...)
```

**Important**: The action stored in the buffer is the **combined** action (base + residual), NOT just the residual. This is what the critic needs to evaluate.

**Terminal observation handling**: When `done=True`, the environment auto-resets. The "next observation" from the env is from the NEW episode. But for the Q-value update, we need the final observation of the FINISHED episode. This is stored in `info["final_obs"]` and substituted during buffer insertion.

### Step (3): Periodic Evaluation

Every 10,000 steps, run `run_dexmg_evaluation()` with 50 episodes across 8 parallel environments. Records success rate, mean return, and optionally videos with Q-value overlays.

### Step (4): Gradient Updates (UTD = 4)

```python
actor_update_cadence = num_updates_per_iteration // actor_updates_per_iteration
# = 4 // 1 = 4  →  actor updated every 4th critic update

for i in range(4):  # 4 gradient updates per env step
    batch = sample_online(128) + sample_offline(128)  # 256 total
    
    update_actor = ((i + 1) % 4 == 0)  # True only on i=3
    
    metrics = agent.update(batch, stddev, update_actor)
```

This means the critic gets 4× more gradient updates than the actor — a key TD3 design choice. The critic needs to be ahead of the actor so the actor has accurate Q-value gradients to follow.

---

## 13. Critic Update — How Q-Values Are Learned

**File**: `resfit/rl_finetuning/off_policy/rl/q_agent.py`, `update_critic()` (lines 302–440)

See [../algorithms/TD3_ALGORITHM.md](../algorithms/TD3_ALGORITHM.md) for the full mathematical derivation.

### Step-by-Step Walkthrough

**1. Encode observations with augmentation**

```python
obs["feat"] = self._encode(obs, augment=True)        # training: augment with RandomShiftsAug
next_obs["feat"] = self._encode(next_obs, augment=True)  # no_grad context
```

**2. Compute target Q-value**

```python
# Get target actor's prediction for next state
next_residual = actor_target(next_obs, stddev, clip=0.3)  # Clipped noise for smoothing

# Combine with base action
next_action = clamp(next_obs["base_action"] + next_residual, -1, 1)

# Min over 2 random heads from 10-head ensemble
target_Q = critic_target.q_value(next_obs["feat"], next_state, next_action)

# TD target (n-step already handled by MultiStepTransform)
y = reward + gamma^n * (1 - done) * target_Q
```

**3. Compute critic loss**

```python
# All 10 heads predict Q for current (s, a)
Q_all = critic(obs["feat"], obs_state, action)  # shape [10, B, 1]

# MSE loss: mean across heads and batch
critic_loss = mean((Q_all - y)^2)
```

**4. Update weights**

```python
encoder_opt.zero_grad()
critic_opt.zero_grad()
critic_loss.backward()
clip_grad_norm_(encoders, 1.0)    # Gradient clipping
clip_grad_norm_(critic, 1.0)
encoder_opt.step()
critic_opt.step()
```

**5. Soft-update target critic**

```python
# θ' ← τ·θ + (1-τ)·θ'   where τ = 0.005
soft_update_params(critic, critic_target, tau=0.005)
```

### The Discount Computation

The `effective_discount` in the update combines the γ schedule with terminal state handling:

```python
effective_discount = batch["gamma"] * batch["nonterminal"]
```

- `batch["gamma"]` comes from the `MultiStepTransform`: it's $\gamma^n$ where $n$ is the actual number of steps (may be < `n_step` if the episode ended early)
- `batch["nonterminal"]` is `False` if the n-step lookahead reaches a terminal state, zeroing out the bootstrap

---

## 14. Actor Update — How the Residual Policy Improves

**File**: `resfit/rl_finetuning/off_policy/rl/q_agent.py`, `update_actor()` (lines 465–510) and `_compute_actor_loss()` (lines 442–462)

**Key insight**: The actor does NOT backprop through the encoder:

```python
obs["feat"] = obs["feat"].detach()  # Stop gradients to encoder
```

This means the encoder is trained **only** by the critic loss. The actor receives frozen visual features and learns to map them to good actions.

### Actor Loss Computation

```python
# 1. Actor predicts residual
residual = actor(obs, stddev=0.0)  # Note: no exploration noise during update

# 2. Combine with base action
combined = clamp(obs["base_action"] + residual, -1, 1)

# 3. Evaluate with critic (mean of all 10 heads)
Q = critic.q_value_for_policy(obs["feat"], obs_state, combined)

# 4. Actor loss = -Q (maximize Q-value)
actor_loss_base = -Q.mean()

# 5. Optional L2 regularization on residual magnitude
l2_penalty = action_l2_reg_weight * mean(sum(residual^2, dim=-1))

# 6. Total loss
actor_loss = actor_loss_base + l2_penalty
```

**Intuition**: The actor asks "which residual correction would make the critic happiest?" and follows the gradient of Q with respect to the action.

### Actor Update Steps

```python
actor_opt.zero_grad()
actor_loss.backward()
clip_grad_norm_(actor, 1.0)
actor_opt.step()
soft_update_params(actor, actor_target, tau=0.005)
```

---

## 15. Evaluation During Training

**File**: `resfit/rl_finetuning/utils/evaluate_dexmg.py`

Every 10,000 steps, `run_dexmg_evaluation()` runs 50 episodes across 8 parallel environments:

1. **Agent in eval mode** — no exploration noise (`stddev=0.0`), uses distribution mean
2. **Q-value tracking** — at each step, the critic predicts Q-value for the selected action, stored per episode
3. **Frame annotation** — if `save_video=True`, each frame is annotated with env index, episode number, step, SUCCESS/FAIL, and predicted Q-value
4. **Success detection** — `reward == 1.0` at terminal step indicates success

### Metrics logged to WandB

| Metric | Description |
|---|---|
| `eval/success_rate` | Fraction of episodes that succeeded |
| `eval/mean_return` | Mean undiscounted return across episodes |
| `eval/mean_successful_episode_length` | Average steps for successful episodes |
| `eval/video` | Annotated evaluation video |
| `value/q_trajectories` | Q-value trajectories plot (success vs failure) |

---

## 16. Logging and Checkpointing

### Training Metrics (every 100 steps)

| Metric | Description |
|---|---|
| `training/SPS` | Steps per second |
| `training/global_step` | Current environment step |
| `training/actor_lr` | Current actor learning rate |
| `train/critic_loss` | Critic MSE loss |
| `train/critic_qt` | Mean target Q-value |
| `train/actor_loss_base` | Actor policy gradient loss |
| `train/residual_l1_magnitude` | Mean |residual action| — shows how much correction is applied |
| `train/residual_l2_magnitude` | Mean residual² — energy of corrections |
| `train/actor_grad_norm` | Actor gradient norm (before clipping) |
| `train/critic_grad_norm` | Critic gradient norm (before clipping) |
| `data/batch_R` | Mean reward in the training batch |
| `data/batch_terminal_R` | Mean reward for terminal transitions only |
| `data/terminal_share` | Fraction of batch that are terminal transitions |
| `buffer/online_size` | Current online buffer occupancy |

### Timing Breakdown

| Metric | Description |
|---|---|
| `timing/env_step_percentage` | % of time spent on environment interaction |
| `timing/gradient_update_percentage` | % of time spent on gradient computation |
| `timing/batch_sampling_percentage` | % of time spent sampling from replay buffer |
| `timing/evaluation_percentage` | % of time spent on evaluation |

### Histograms

| Metric | Description |
|---|---|
| `histograms/residual_actions` | Distribution of residual action values |
| `histograms/critic_qt` | Distribution of target Q-values |

### Checkpointing

Checkpoints are saved automatically at three trigger points via `save_checkpoint()` in `resfit/rl_finetuning/utils/checkpoint.py`:

| Trigger | Directory | WandB artifact |
|---|---|---|
| Every `save_freq` steps (default: 10,000) | `models/policy_step_<N>/checkpoint.pt` | `run_<id>_model_step_<N>` |
| Every `save_freq` steps (same) | `models/latest/checkpoint.pt` (overwritten) | `run_<id>_latest` |
| New best eval success rate | `models/best/checkpoint.pt` | `run_<id>_best` |
| End of training | `models/final/checkpoint.pt` | `run_<id>_final` |

Each checkpoint contains: agent weights (`state_dict`), all three optimizer states (actor, critic, encoder AdamW), LR scheduler states, `global_step`, `success_rate`, `actor_updates` counter, and the full Hydra config. This is everything needed for a perfect resume.

### Resuming from a Checkpoint

If training crashes or is interrupted, resume from the latest checkpoint:

```bash
python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name residual_td3_coffee_config \
    base_policy.wandb_id=dexmg-bc/<run_id> \
    base_policy.wt_type=best \
    offline_data.num_episodes=100 \
    seed=<same_seed> \
    resume_ckpt=<run_dir>/models/latest \
    wandb.continue_run_id=<wandb_run_id>
```

| Parameter | Description |
|---|---|
| `resume_ckpt` | Path to a checkpoint directory or `.pt` file. Restores weights, optimizers, and `global_step`. |
| `wandb.continue_run_id` | WandB run ID (8-char code from the URL) to continue logging to the same dashboard. |

On resume:
- **Agent weights + optimizers** are restored from the checkpoint
- **Critic warmup is skipped** (detected via `global_step > 0` — already warmed up)
- **Online buffer** restarts from the cached 10K warmup entries (transitions since warmup are lost, but this has minimal impact because the agent weights carry all the learned knowledge)
- **Offline buffer** loads from cache as usual (no loss)

See [CHECKPOINTING_AND_RESUME.md](CHECKPOINTING_AND_RESUME.md) for the full deep-dive: what gets saved, what gets restored, what doesn't survive, and the exact resume flow.

---

## 17. File Reference Map

| File | Lines | Purpose |
|---|---|---|
| `resfit/rl_finetuning/scripts/train_residual_td3.py` | ~1370 | Main training script — orchestrates everything |
| `resfit/rl_finetuning/config/residual_td3.py` | ~293 | Residual TD3 Hydra configs (inherits from RLPD). Includes `resume_ckpt` param. |
| `resfit/rl_finetuning/config/rlpd.py` | 601 | Parent configs: QAgentConfig, ActorConfig, CriticConfig, etc. |
| `resfit/rl_finetuning/utils/checkpoint.py` | ~190 | `save_checkpoint()` and `load_checkpoint()` — agent weights, optimizers, counters |
| `resfit/rl_finetuning/wrappers/residual_env_wrapper.py` | 200 | BasePolicyVecEnvWrapper — combines base+residual actions |
| `resfit/rl_finetuning/off_policy/rl/q_agent.py` | 664 | QAgent — encoder, actor, critic, update methods |
| `resfit/rl_finetuning/off_policy/rl/actor.py` | 200 | Actor network (MLP with Tanh + action_scale) |
| `resfit/rl_finetuning/off_policy/rl/critic.py` | 598 | Critic ensemble (SpatialEmb + vmap MLP heads) |
| `resfit/rl_finetuning/off_policy/networks/encoder.py` | 45 | VitEncoder wrapper |
| `resfit/rl_finetuning/off_policy/networks/min_vit.py` | 170 | MinViT — patch embed, transformer, positional embeddings |
| `resfit/rl_finetuning/off_policy/common_utils/utils.py` | 200 | TruncatedNormal, soft_update_params, RandomShiftsAug, schedule |
| `resfit/rl_finetuning/utils/normalization.py` | 268 | ActionScaler + StateStandardizer |
| `resfit/rl_finetuning/utils/rb_transforms.py` | 358 | MultiStepTransform for n-step returns in replay buffer |
| `resfit/rl_finetuning/utils/evaluate_dexmg.py` | 337 | Evaluation loop with Q-value tracking + video annotation |

---

## 18. Extra Flags: `headless=false`, `eval_num_envs`, and `--no_cleanup`

### On-Screen MuJoCo Viewer (`headless=false`)

By default, training runs headless (`headless=true`) using EGL offscreen rendering — no display needed. Setting `headless=false` enables the on-screen MuJoCo viewer window:

**What it changes:**

| Setting | `headless=true` (default) | `headless=false` |
|---|---|---|
| `MUJOCO_GL` | `egl` (offscreen GPU) | `glfw` (on-screen) |
| `has_renderer` in robosuite | `False` | `True` |
| Vectorization | `AsyncVectorEnv` (fast, multiprocess) | `SyncVectorEnv` (required — viewer must be in main process) |
| Viewer window | None | Opens one MuJoCo window per training env |

**Architecture: `render()` vs `render_viewer()`**

The rendering path is split into two separate methods, following the same pattern as SERL:

| Method | Purpose | Where called | Performance cost |
|---|---|---|---|
| `render()` | Captures an **offscreen RGB frame** for video recording | `evaluate_dexmg.py` during evaluation | Minimal (MuJoCo `sim.render()` offscreen) |
| `render_viewer()` | Updates the **on-screen GLFW window** | Main training loop, after `env.step()` | High (GLFW swap, vsync, display) |

`render_viewer()` is **not called inside `step()`**. It lives in the main training loop as an explicit call after `env.step()`. When `headless=True`, `render_viewer()` is a no-op. This design ensures that:
1. `step()` is always fast — no hidden rendering overhead
2. Video recording (`render()`) works in both headless and non-headless modes
3. On-screen visualization is opt-in and controlled from the top level

The call chain for `render_viewer()` is:
```
train_residual_td3.py  →  env.render_viewer()
  →  BasePolicyVecEnvWrapper.render_viewer()
    →  VectorizedEnvWrapper.render_viewer()
      →  iterates SyncVectorEnv.envs[]
        →  RobosuiteGymWrapper.render_viewer()
          →  self.env.render()  (robosuite's GLFW refresh)
```

**When the window appears:**

The MuJoCo viewer window is created when the robosuite environment is constructed (step 6 in the execution flow). However, it appears blank/frozen until `render_viewer()` is first called, which happens during:

- **Warmup phase (step 10)**: The viewer starts animating here. After each `env.step()`, `render_viewer()` refreshes the on-screen window. You'll see the robot performing random exploratory motions (base policy + noise).

- **Main training loop (step 13)**: The viewer continues animating as the agent interacts with the environment. You'll see the residual policy gradually improve.

**What to look for in the terminal** to know the viewer is about to animate:
```
On-screen rendering enabled (MUJOCO_GL=glfw). A viewer window will open.
...
Warm-up: filling online buffer with 10000 random steps…
← Window starts animating here
```

**Performance impact**: `SyncVectorEnv` + on-screen rendering is significantly slower than async headless rendering. Use `headless=false` only for debugging/visualization, not for full training runs.

**Requirement**: Needs a display server (`$DISPLAY` set). Won't work over SSH without X11 forwarding.

### OOM During Evaluation (`eval_num_envs`)

**Symptom**: Training runs fine for 10,000 steps, then gets "Killed" by the OS when the first evaluation starts. The terminal shows `BrokenPipeError` from `AsyncVectorEnv` workers.

**Root cause**: Evaluation spawns `eval_num_envs` parallel environments using `AsyncVectorEnv` with `context="spawn"`. Each child process gets a full Python interpreter + MuJoCo instance + 3 camera framebuffers. On a 32 GB laptop where the training process already uses ~8 GB, spawning 8 eval workers can push total memory past the system limit, triggering the Linux OOM killer.

**Fix**: Reduce `eval_num_envs` to match your available RAM:
```bash
# 32 GB RAM laptop → 4 eval envs is safe
eval_num_envs=4

# 16 GB RAM → 2 eval envs
eval_num_envs=2
```

The default is `eval_num_envs=4`. If you still hit OOM, reduce further.

### Preventing Cleanup (`--no_cleanup`) — BC Training Only

The BC training script (`train_bc_dexmg.py`) has a `--no_cleanup` flag:

```bash
python resfit/lerobot/scripts/train_bc_dexmg.py --no_cleanup ...
```

By default, after BC training completes, the script deletes the local run directory via `shutil.rmtree()`. This is fine if all checkpoints were uploaded to wandb, but if the internet drops during training, the last few checkpoints may fail to upload — and then they're gone forever.

Passing `--no_cleanup` skips the deletion so local checkpoints are preserved. This is a safety net — always use it if you want to keep local copies of checkpoints.

> **Note**: This flag only applies to BC training. The RL training script (`train_residual_td3.py`) has its own cache management.

---

## 19. Reward Flow, Critic Loss Types, and v_min/v_max

### Reward Flow: Environment → Buffer → Critic

The reward is **never hardcoded**. It flows directly from the environment through the buffer into the Bellman target:

```
robosuite env.step()          →  reward (whatever robosuite returns)
    ↓
BasePolicyVecEnvWrapper.step() →  passes reward through untouched
    ↓
online_rb.add(TensorDict{"next": {"reward": reward[i]}})   # stored as-is
    ↓
MultiStepTransform             →  r + γr' + γ²r'' + ...    (n-step sum)
    ↓
batch[("next", "reward")]       →  used in Bellman backup
```

Ref: [train_residual_td3.py](resfit/rl_finetuning/scripts/train_residual_td3.py#L198) stores `reward[i]` directly from `env.step()`. The [q_agent.py](resfit/rl_finetuning/off_policy/rl/q_agent.py#L633) `update()` method reads `reward = batch[("next", "reward")]`.

If you enable dense reward (`reward_shaping=True` in `dexmg.py`), the per-step values (e.g. 0.73) flow through the same path — no code changes needed.

### The Bellman Target Formula

```python
# q_agent.py lines 334-337
target_all = self.critic_target.q_value(next_obs["feat"], next_obs["observation.state"], next_action)
target_q_min = target_all.squeeze(-1)   # min over random 2 of 10 heads → [B]
target_q = (reward + (discount * target_q_min)).detach()
```

This is the standard Bellman equation $y = r + \gamma^n Q_{\text{target}}(s', a')$ — works for **any** reward structure.

### The One Sparse-Specific Guard

```python
# q_agent.py line 338-339
if self.cfg.clip_q_target_to_reward_range:
    target_q = torch.clamp(target_q, min=0, max=1)   # hardcoded {0, 1}
```

This clamps Q-targets to [0, 1]. **Default is `False`** ([rlpd.py](resfit/rl_finetuning/config/rlpd.py#L126)), so it's disabled. If you ever enable it with dense reward, you'd need to change the clamp range or leave it off.

### Three Critic Loss Types

The critic loss type is set by `agent.critic.loss.type` (default `"mse"`):

| Type | `v_min`/`v_max` needed? | Output dim | How it works |
|---|---|---|---|
| `"mse"` | **No** | 1 (scalar Q) | Standard MSE: $\|Q(s,a) - y\|^2$ |
| `"hl_gauss"` | **Yes** | `n_bins` (51) | HL-Gauss distributional: predicts a histogram over value range |
| `"c51"` | **Yes** | `n_bins` (51) | C51 distributional: predicts categorical distribution over atoms |

### What v_min/v_max Are and Where They're Used

`v_min` and `v_max` define the **range of possible Q-values** that distributional critics can represent. They are only used when `loss.type` is `"hl_gauss"` or `"c51"` — **not used with MSE**.

**Config** ([rlpd.py](resfit/rl_finetuning/config/rlpd.py#L29-L30)):
```python
@dataclass
class CriticLossCfg:
    type: str = "mse"
    n_bins: int = 51
    v_min: float = 0.0
    v_max: float = 1.0
```

**HL-Gauss** ([critic.py](resfit/rl_finetuning/off_policy/rl/critic.py#L23-L24)): Creates `n_bins` bins spanning `[v_min, v_max]`. The target Q-value is converted to a soft Gaussian distribution over these bins. The critic predicts logits over bins, and the loss is cross-entropy against the Gaussian target.

```python
# critic.py — HLGaussLoss.__init__
bin_edges = torch.linspace(min_value, max_value, num_bins + 1)  # 52 edges → 51 bins
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2              # 51 centers
```

**C51** ([critic.py](resfit/rl_finetuning/off_policy/rl/critic.py#L159-L170)): Creates `n_bins` atoms spanning `[v_min, v_max]`. The critic predicts a probability distribution over these atoms. Q-value = expected value = $\sum p_i \cdot z_i$.

```python
# critic.py — C51Loss.__init__
support = torch.linspace(v_min, v_max, num_atoms)   # 51 atoms
delta_z = (v_max - v_min) / (num_atoms - 1)          # spacing between atoms
```

During target computation, C51 projects the Bellman-shifted atoms back onto the support and **clamps to [v_min, v_max]** ([critic.py](resfit/rl_finetuning/off_policy/rl/critic.py#L182)):
```python
target_support = torch.clamp(target_support, self.v_min, self.v_max)
```

### How to Set v_min/v_max

| Reward type | v_min | v_max | Reasoning |
|---|---|---|---|
| **Sparse** (default, 0/1) | 0.0 | 1.0 | Max return is 1.0 (success at current step) |
| **Dense** (reward_shaping=True) | 0.0 | 70.0 | Per-step reward ∈ [0, 1.0], γ=0.995, horizon=100 → max ≈ 63.5 |

If `v_max` is too small, the distributional critic **cannot represent** Q-values above it — all probability mass piles up at the boundary. If too large, resolution between bins decreases (same `n_bins` spread over a wider range). The default `v_min=0, v_max=1` is tuned for sparse reward.

### Summary: What to Change for Dense Reward

| What | Change? | Details |
|---|---|---|
| `dexmg.py` robosuite.make() | **Yes** | Add `reward_shaping=True` |
| Bellman formula in q_agent.py | **No** | Already general |
| `clip_q_target_to_reward_range` | **No** | Already `False` by default |
| `v_min`/`v_max` | **Only if using hl_gauss/c51** | Set `v_max=70.0` (or higher) |
| MSE critic loss | **No** | Works with any reward range |

---

**Concept deep-dive documents:**
- [../algorithms/TD3_ALGORITHM.md](../algorithms/TD3_ALGORITHM.md) — Full TD3 algorithm with math, pseudocode, and worked example
- [../algorithms/NSTEP_RETURNS.md](../algorithms/NSTEP_RETURNS.md) — Multi-step returns with numerical walkthrough
- [../algorithms/REPLAY_BUFFERS.md](../algorithms/REPLAY_BUFFERS.md) — Online/offline mixing, PER, prefetching, caching
- [../algorithms/RESIDUAL_LEARNING.md](../algorithms/RESIDUAL_LEARNING.md) — Why residual RL, zero init, action scaling math
- [../algorithms/ACTION_NORMALIZATION.md](../algorithms/ACTION_NORMALIZATION.md) — ActionScaler and StateStandardizer formulas + examples
- [CHECKPOINTING_AND_RESUME.md](CHECKPOINTING_AND_RESUME.md) — Checkpointing, resume, RL vision encoder, buffer lifecycle, memory budget
- [../rewards/REWARD_AND_SUCCESS.md](../rewards/REWARD_AND_SUCCESS.md) — Reward signal, success criteria for all 12 tasks, n-step returns, eval metrics
