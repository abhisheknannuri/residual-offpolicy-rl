# Residual Learning — Deep Dive with Math and Examples

This document explains **why** residual RL works, the **zero initialization** trick, **action scaling** math, and how the residual formulation makes RL training dramatically easier for robotics tasks.

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)

---

## Table of Contents

1. [The Problem: Why Is RL Hard for Robotics?](#1-the-problem-why-is-rl-hard-for-robotics)
2. [The Idea: Don't Train From Scratch](#2-the-idea-dont-train-from-scratch)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [Zero Initialization: Starting From Nothing](#4-zero-initialization-starting-from-nothing)
5. [Action Scaling: Constraining the Correction](#5-action-scaling-constraining-the-correction)
6. [Numerical Example: End-to-End Walkthrough](#6-numerical-example-end-to-end-walkthrough)
7. [Why the Residual Learning Rate Is So Low](#7-why-the-residual-learning-rate-is-so-low)
8. [Comparison: RL From Scratch vs Residual RL](#8-comparison-rl-from-scratch-vs-residual-rl)
9. [What the RL Agent Sees: Observations](#9-what-the-rl-agent-sees-observations)
10. [Reward Structure](#10-reward-structure)
11. [Image Resolution](#11-image-resolution)
12. [State Composition and How to Change It](#12-state-composition-and-how-to-change-it)
13. [Changing Image Resolution](#13-changing-image-resolution)
14. [Adding Custom Gym Wrappers](#14-adding-custom-gym-wrappers)

---

## 1. The Problem: Why Is RL Hard for Robotics?

Standard RL for robotic manipulation faces several compounding difficulties:

### Sparse Rewards

Most manipulation tasks have binary success/failure rewards:

$$r_t = \begin{cases} 1 & \text{if task completed at step } t \\ 0 & \text{otherwise} \end{cases}$$

In a 300-step episode, the agent gets **zero reward** for 299 steps and then either 1 or 0 at the end. To learn, the agent must:
1. Discover a successful trajectory by chance (extremely unlikely with random exploration)
2. Propagate the reward backward through 300+ Q-value estimates

### High-Dimensional Continuous Actions

A dual-arm robot has 14 action dimensions (7 per arm: 6 DoF + gripper). Random exploration in $[-1, 1]^{14}$ has probability approximately 0 of producing any useful behavior.

### Long Horizons

With 300-step episodes and $\gamma = 0.995$, the effective horizon is:

$$H_{\text{eff}} = \frac{1}{1 - \gamma} = \frac{1}{0.005} = 200 \text{ steps}$$

The agent must plan 200 steps ahead coherently.

### The Practical Result

Training RL from scratch on a task like `TwoArmBoxCleanup` (two arms, 14 action dims, 300-step horizon, sparse reward) would require **millions** of environment steps and might never converge at all.

---

## 2. The Idea: Don't Train From Scratch

We already have a BC policy that gets ~60-70% success rate. Instead of starting from zero, **build on top of it**:

```
Without residual:                  With residual:
                                   
RL agent must learn:               RL agent must learn:
  "Insert peg into hole"             "Adjust by 2mm to the left"
  
Difficulty: ★★★★★                  Difficulty: ★★☆☆☆
```

The base policy handles the gross motor plan. The residual handles fine adjustments — corrections that are small in magnitude but critical for success.

---

## 3. Mathematical Formulation

### Standard RL

The policy directly outputs the action:

$$a_t = \pi_\theta(s_t) + \epsilon_t$$

where $\pi_\theta$ is the learned policy and $\epsilon_t$ is exploration noise.

### Residual RL

The action is composed of two parts:

$$a_t = \underbrace{a_t^{\text{base}}}_{\text{frozen BC}} + \underbrace{a_t^{\text{res}}}_{\text{learned residual}}$$

where:
- $a_t^{\text{base}} = \pi_{\text{BC}}(s_t)$ — output of the frozen (non-trainable) BC policy
- $a_t^{\text{res}} = f_\theta(s_t, a_t^{\text{base}})$ — output of the trainable residual network

The residual network takes both the state $s_t$ AND the base action $a_t^{\text{base}}$ as input. This is important — the residual can see what the base policy suggests and decide how to correct it.

### Policy Gradient

The actor loss is:

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{s \sim \mathcal{B}} \left[ Q_\phi(s, \text{clip}(a^{\text{base}} + a^{\text{res}}, -1, 1)) \right]$$

The gradient flows through:

$$\nabla_\theta \mathcal{L} = -\nabla_a Q \cdot \nabla_\theta a^{\text{res}}$$

Notice: the gradient only flows through $a^{\text{res}}$ (the residual network). The base policy is frozen — no gradients flow through it.

### Clamping to [-1, 1]

The combined action is clamped to the valid normalized action range:

$$a_t = \text{clip}(a_t^{\text{base}} + a_t^{\text{res}}, -1, 1)$$

If $a^{\text{base}} = 0.9$ and $a^{\text{res}} = 0.3$, the raw sum is 1.2, which clips to 1.0. This means the gradient of the clip function is zero in this region — the residual can't push beyond the action limits.

---

## 4. Zero Initialization: Starting From Nothing

### The Trick

The last layer of the residual actor has its weights and biases initialized to **exactly zero**:

```python
# In actor.py, _initialize_weights():
if cfg.actor_last_layer_init_scale is not None:  # = 0.0
    # Find the last Linear layer before Tanh
    utils.initialize_layer_weights(
        final_layer,
        distribution="normal",    # N(0, scale²)
        scale=0.0,                # scale = 0.0 → all zeros
    )
```

### What This Means

At the start of training, the MLP's last layer computes:

$$\text{output} = W_{\text{last}} \cdot h + b_{\text{last}} = \mathbf{0} \cdot h + \mathbf{0} = \mathbf{0}$$

Then through Tanh and action_scale:

$$a^{\text{res}} = \text{Tanh}(\mathbf{0}) \times 0.2 = 0 \times 0.2 = \mathbf{0}$$

So the combined action is:

$$a_t = a_t^{\text{base}} + \mathbf{0} = a_t^{\text{base}}$$

**The agent starts by doing exactly what the BC policy would do.** This is crucial:
- Performance starts at the BC baseline (~60-70% success), not zero
- The agent doesn't need to "rediscover" basic motor skills
- Gradient updates gradually push the residual away from zero only where corrections help

### Important: This Is Weight Initialization, NOT a Runtime Override

A common misconception is that the residual is "overridden to zero" for some number of initial steps and then "switched on". **That is not what happens.** The zero output is a consequence of the weight initialization — `nn.init.normal_(weight, mean=0, std=0)` sets all weights and biases to exactly 0. Since the MLP ends with `Linear → Tanh`, and `tanh(0) = 0`, the output is zero at initialization.

From the very first training step (step 0), the residual actor is **free to learn nonzero outputs**. There is no override, no schedule, no switching. The first gradient update that flows through the actor creates the first nonzero weights, and these are targeted — they push the residual in the direction that improves Q-values. The small `action_scale` (e.g., 0.2) keeps the outputs bounded, so the residual moves away from zero **gradually and only where corrections help**.

### Why Not Small Random Instead of Zero?

Small random initialization (e.g., `scale=1e-3`) would add tiny random perturbations to every action at every timestep. While individual perturbations are small, their cumulative effect over a 300-step episode can be significant — the trajectory diverges from the BC policy's trajectory.

Zero initialization guarantees the **exact same trajectory** as the BC policy at step 0. The first gradient update creates the first non-zero weights, and these weights are targeted — they push the residual in the direction that improves Q-value.

---

## 5. Action Scaling: Constraining the Correction

### The Problem

Without constraints, the residual could learn to output large actions that completely override the base policy. This defeats the purpose of residual learning and makes training unstable.

### The Solution: Tanh × action_scale

The actor architecture enforces a hard bound on the residual:

```
MLP output → Tanh → × action_scale
                          ↓
              Tanh ∈ [-1, 1]
              × 0.2
              → output ∈ [-0.2, 0.2]
```

With `action_scale = 0.2`, the residual is constrained to:

$$a^{\text{res}} \in [-0.2, 0.2]^d$$

in the **normalized** action space (where the full range is [-1, 1]).

### How the Output Range Works (No Division Involved)

The actor's output range is **not** [-1, 1] divided by something. The flow is:

1. The MLP produces raw logits (any real number)
2. `Tanh` squashes them to **[-1, +1]**
3. **Multiply** by `action_scale`: `scaled_mu = tanh(logits) × 0.2` → **[-0.2, +0.2]**

```python
# In actor.py forward():
mu = self.policy(policy_input)       # Tanh output → [-1, 1]
scaled_mu = mu * self.cfg.action_scale  # × 0.2 → [-0.2, 0.2]
```

This is a direct multiplication, not a division. The Tanh already bounds the output to [-1, 1], and the `action_scale` simply shrinks that range.

During **evaluation** (`eval_mode=True`), the action is exactly `scaled_mu` — strictly bounded to [-0.2, +0.2].

During **training**, `TruncatedNormal(scaled_mu, std=0.025)` samples around the mean with exploration noise, so actual sampled actions can slightly exceed ±0.2 depending on `std`, but the distribution is centered within that range.

### What ±0.2 Means in Practice

Consider a single action dimension — say, the x-position of a robot's end effector:

| Value | Meaning |
|---|---|
| Raw action range | [-0.05, 0.05] m/step (from dataset min/max) |
| Normalized range | [-1, 1] |
| Base action | e.g., 0.3 (normalized) |
| Residual range | [-0.2, 0.2] (normalized) |
| Combined range | [0.1, 0.5] (normalized) |
| In raw units | residual = ±0.2 × 0.05 m = ±0.01 m = ±1 cm |

The residual can adjust the end-effector position by at most ±1 cm per step. This is enough for fine positioning corrections but prevents wild, destructive movements.

### action_scale and ActionScaler Interaction

The `action_scale` parameter appears in **two** different places with different effects:

1. **Actor's `action_scale` (0.2)**: Multiplies the Tanh output → constrains residual magnitude
2. **ActionScaler's `action_scale` (0.2)**: Expands the normalization range by `(1 + 0.2) = 1.2×` → leaves room for the residual to push beyond the demonstrated action range

The second one is important: if the normalization range exactly matches the dataset's min/max, then any action at the boundary (e.g., normalized = 1.0) plus a positive residual would be clipped. Expanding the range by 20% ensures there's "headroom" for corrections.

**They are separate mechanisms connected by the same config value.** The command-line arg `agent.actor.action_scale=0.2` sets the actor's output bound. This same value is also passed to `ActionScaler.from_dataset_stats(action_scale=cfg.agent.actor.action_scale)` to expand the normalization range. The dataset statistics (min/max) define the *center and shape* of the normalization — `action_scale` does NOT overwrite those stats. It just tells the scaler "allow 20% extra headroom beyond what the dataset contained" so combined actions (base ± residual) can still be unmapped to valid raw actions.

---

## 6. Numerical Example: End-to-End Walkthrough

Let's trace through one complete step with concrete numbers.

### Setup

- Task: TwoArmBoxCleanup
- Action dimension: 14 (7 per arm)
- action_scale: 0.2
- We focus on one dimension: `joint_3_velocity` of robot arm 1

### Dataset Statistics for `joint_3_velocity`

```
action_min[3] = -0.083  (raw units, from dataset)
action_max[3] = 0.071   (raw units)
```

### ActionScaler Initialization

```
mid = (-0.083 + 0.071) / 2 = -0.006
half_range = (0.071 - (-0.083)) / 2 = 0.077
expanded_half_range = 0.077 × (1 + 0.2) = 0.0924

final_min = -0.006 - 0.0924 = -0.0984
final_max = -0.006 + 0.0924 = 0.0864
```

### At Timestep t

**1. Base policy outputs raw action:**

$$a_{\text{base}}^{\text{raw}} = 0.025 \text{ (in raw units)}$$

**2. Scale to [-1, 1]:**

$$a_{\text{base}}^{\text{norm}} = 2 \times \frac{0.025 - (-0.0984)}{0.0864 - (-0.0984)} - 1 = 2 \times \frac{0.1234}{0.1848} - 1 = 2 \times 0.668 - 1 = 0.335$$

**3. Residual actor outputs correction:**

The actor MLP produces (before Tanh): $z = 0.15$

$$a_{\text{res}}^{\text{norm}} = \tanh(0.15) \times 0.2 = 0.149 \times 0.2 = 0.0298$$

**4. Combine:**

$$a_{\text{combined}}^{\text{norm}} = \text{clip}(0.335 + 0.0298, -1, 1) = 0.365$$

**5. Unscale to raw action:**

$$a_{\text{combined}}^{\text{raw}} = -0.0984 + \frac{0.365 + 1}{2} \times 0.1848 = -0.0984 + 0.6825 \times 0.1848 = -0.0984 + 0.1262 = 0.0278$$

**6. Effect of the residual:**

$$\Delta a^{\text{raw}} = 0.0278 - 0.025 = 0.0028 \text{ (raw units)}$$

The residual changed the joint velocity by 0.0028 units — a small but potentially critical adjustment.

---

## 7. Why the Residual Learning Rate Is So Low

The actor learning rate is `1e-6` — **100× lower** than the critic's `1e-4`. Here's why:

### Risk of Catastrophic Forgetting

The base policy represents thousands of gradient steps of BC training. If the residual changes too quickly, it can:
1. Push the combined action far from the demonstrated distribution
2. Create trajectories the critic has never seen → garbage Q-estimates
3. The actor follows garbage gradients → policy collapses → success rate drops to 0%

### Conservative Updates

With `lr = 1e-6` and gradient clipping at norm 1.0:

$$\Delta \theta_{\text{actor}} = -\text{lr} \times \nabla_\theta \mathcal{L} \approx 10^{-6} \times \text{direction}$$

This means each update changes the actor weights by approximately one millionth of the gradient magnitude. Over 500,000 steps with 1 actor update per step = 125,000 actor updates (UTD=4, actor every 4th), the cumulative change is small enough to stay near the BC policy's behavior while gradually improving.

### Learning Rate Hierarchy

```
critic_lr  = 1e-4    (critic must be ahead of actor)
encoder_lr = 1e-4    (same as critic — encoder trained by critic loss)
actor_lr   = 1e-6    (very conservative — actor changes slowly)
```

This 100× ratio ensures the critic's Q-estimates stabilize before the actor tries to exploit them. If the actor moved faster than the critic, it would chase phantom Q-values.

---

## 8. Comparison: RL From Scratch vs Residual RL

### Exploration Efficiency

| Aspect | RL From Scratch | Residual RL |
|---|---|---|
| Initial success rate | ~0% | ~60-70% (BC baseline) |
| Exploration strategy | Random in $[-1,1]^{14}$ | BC policy ± 0.2 |
| Steps to first success | ~100,000+ | ~300 (BC does it) |
| Useful experience in buffer | <1% early on | ~60%+ from the start |

### Learning Stability

| Aspect | RL From Scratch | Residual RL |
|---|---|---|
| Q-value scale | Unknown, drifts | Anchored near 0–1 (sparse reward) |
| Gradient signal | Very noisy | Stable (50% offline demos) |
| Policy collapse risk | High | Low (constrained by action_scale) |
| Recovery from bad updates | Very difficult | Automatic (residual can return to ~0) |

### Computational Cost

| Aspect | RL From Scratch | Residual RL |
|---|---|---|
| Typical timesteps needed | 2M–10M | 200K–500K |
| Wall-clock time | Days | Hours |
| GPU memory (extra) | — | Base policy forward pass (small overhead) |

### When Residual RL Can Fail

1. **Base policy is terrible** (<10% success) — not enough good trajectories to build on
2. **Task requires fundamentally different strategy** — residual can only adjust, not replan
3. **action_scale too small** — the residual literally can't make the needed correction
4. **action_scale too large** — the residual overpowers the base, reverting to scratch RL

The 0.2 action_scale is a sweet spot: large enough for meaningful corrections, small enough to keep the policy well-behaved.

---

## 9. What the RL Agent Sees: Observations

The RL agent (both critic and actor) receives **images + state + base action** at every step. This is different from BC training, which sees images + state only.

### Observation Structure (per step, Lift task)

| Key | Shape | Source | Description |
|---|---|---|---|
| `observation.images.agentview` | `(1, 3, 84, 84)` | MuJoCo render | Third-person workspace camera |
| `observation.images.robot0_eye_in_hand` | `(1, 3, 84, 84)` | MuJoCo render | Wrist-mounted camera |
| `observation.state` | `(1, 9)` | Robot sensors → z-score standardized | `eef_pos(3) + eef_quat(4) + gripper_qpos(2)` |
| `observation.base_action` | `(1, 7)` | Frozen BC policy → ActionScaler | Normalized base action in [-1, 1] |

### How Observations Flow Through the Pipeline

```
MuJoCo env → raw obs {images, state}
                ↓
BasePolicyVecEnvWrapper._augment_obs():
  1. base_action = frozen_BC_policy.select_action(raw_obs)
  2. base_naction = ActionScaler.scale(base_action)   → [-1, 1]
  3. obs["observation.base_action"] = base_naction
  4. obs["observation.state"] = StateStandardizer.standardize(state)   → z-score
                ↓
augmented_obs → Actor and Critic networks
```

### What the Actor Network Sees

The actor receives:
- **Images** through per-camera `VitEncoder` (MinViT) → feature embeddings per camera
- **Standardized state** (z-scored: zero mean, unit std)
- **Base normalized action** → so the residual knows what the BC policy suggests

The actor outputs: `residual_naction = tanh(MLP(features)) * action_scale`

### What the Critic Network Sees

The critic receives the same observations as the actor PLUS the combined action (base + residual) to estimate Q-values.

### State Standardization (RL)

Unlike BC which uses per-feature MEAN_STD normalization, the RL pipeline uses a dedicated `StateStandardizer`:

$$s_{\text{norm}} = \frac{s - \mu}{\max(\sigma, 0.1)}$$

Stats ($\mu$, $\sigma$) come from the **offline dataset** (`dataset.meta.stats["observation.state"]`). The `min_std=0.1` floor prevents division by near-zero standard deviation.

---

## 10. Reward Structure

### Sparse Reward (Default — What This Codebase Uses)

`robosuite.make()` in `dexmg.py` does NOT pass `reward_shaping`, so robosuite defaults to `reward_shaping=False` (sparse).

$$r_t = \begin{cases} 1.0 & \text{if } \texttt{\_check\_success()} \text{ is True} \\ 0.0 & \text{otherwise} \end{cases}$$

**Episode terminates immediately on first success.** In `dexmg.py` `step()`:

```python
success = reward == 1.0
terminated_scalar = bool(success)
```

So with a 100-step horizon, the agent gets 0 for every step until either:
- Cube is lifted → reward = 1.0 → episode terminates immediately
- 100 steps elapse → episode truncates with reward = 0.0

**For N-step returns with sparse reward:** The reward signal only appears in the last step of successful episodes. With `n_step=5` and `γ=0.995`, a success at step $t$ yields:

$$G_t = 0 + 0\gamma + 0\gamma^2 + 0\gamma^3 + 1.0 \cdot \gamma^4 = 0.98$$

Higher `n_step` helps propagate this reward back through more Q-value estimates.

**v_min/v_max for distributional critic (sparse reward):** With only 0/1 rewards, the maximum possible return is 1.0 (success at the current step with no discount). So `v_min=0.0, v_max=1.0` is correct.

### Dense Reward (Available via Code Change)

If you add `reward_shaping=True` to `robosuite.make()` in `dexmg.py` (around line 214), the Lift reward becomes:

| Component | Range | Description |
|---|---|---|
| Reaching | `[0.0, 1.0]` | Tanh of distance to cube |
| Grasping | `{0.0, 0.25}` | Binary — gripper contacts cube or not |
| Lifting | `{0.0, 1.0}` | Binary — cube above threshold height or not |
| **Total** | **[0.0, 2.25]** | Sum of components |

Robosuite normalizes by `reward_scale / 2.25` (default `reward_scale=1.0`) → per-step reward in `[0.0, 1.0]`.

**v_min/v_max for distributional critic (dense reward):** With per-step reward ∈ [0, 1.0], γ=0.995, horizon=100:

$$V_{\max} = \sum_{t=0}^{99} \gamma^t \cdot 1.0 = \frac{1 - 0.995^{100}}{1 - 0.995} \approx 63.5$$

Realistically a good policy solving in ~50 steps sees returns around 20–40. Safe range: `v_min=0.0, v_max=70.0`.

**To enable dense reward:** Edit `dexmg.py`, add to `robosuite.make()` call in `DexMimicGenEnv.__init__()`:
```python
env_kwargs = {
    ...
    "reward_shaping": True,    # ← add this
}
```
Then update `v_min`/`v_max` and potentially increase `n_step` since the reward signal is denser.

---

## 11. Image Resolution

### Current: 84×84 Everywhere

| Stage | Resolution | Configurable? |
|---|---|---|
| Dataset (ankile/robomimic-mh-lift-image) | 84×84 | Fixed at conversion time |
| BC training | Whatever is in dataset (84×84) | No CLI flag for training size |
| BC eval rollouts | 84×84 | Yes: `--eval_camera_size 84` |
| RL training (env rendering) | 84×84 | **No** — hardcoded default in `dexmg.py` |
| RL critic/actor (MinViT patches) | 84×84 (**hardcoded**) | **No** — `num_patch=81` in `min_vit.py` |

### Why 84×84?

- Standard benchmark resolution for MuJoCo-based RL (from Atari/DM Control tradition)
- Small enough for fast rendering + training (fit many envs in VRAM)
- Sufficient for tabletop manipulation tasks (objects are large relative to pixels)

### Cameras Per Task

| Task | Cameras | Count |
|---|---|---|
| Lift, Can, Square | `agentview` + `robot0_eye_in_hand` | 2 |
| TwoArmCoffee, Pouring, CanSort | `agentview` + `robot0_eye_in_left_hand` + `robot0_eye_in_right_hand` | 3 |
| Transport | `agentview` + `robot0_eye_in_hand` + `robot1_eye_in_hand` + `shouldercamera0` + `shouldercamera1` | 5 |

---

## 12. State Composition and How to Change It

### Current State (Lift Task): 9D

| Key | Dims | Content |
|---|---|---|
| `robot0_eef_pos` | 3 | End-effector XYZ position (world frame) |
| `robot0_eef_quat` | 4 | End-effector quaternion orientation |
| `robot0_gripper_qpos` | 2 | Gripper finger joint positions |
| **Total** | **9** | |

**What's NOT included:** joint positions (7D), joint velocities (7D), gripper velocities (2D), EE velocities (6D), object state (14D).

### All Available Keys for Lift (Panda)

| Key | Dims | Description |
|---|---|---|
| `robot0_eef_pos` | 3 | EE position |
| `robot0_eef_quat` | 4 | EE orientation |
| `robot0_joint_pos` | 7 | 7 revolute joint angles |
| `robot0_joint_vel` | 7 | Joint angular velocities |
| `robot0_gripper_qpos` | 2 | Gripper finger positions |
| `robot0_gripper_qvel` | 2 | Gripper finger velocities |
| `robot0_eef_vel_lin` | 3 | EE linear velocity |
| `robot0_eef_vel_ang` | 3 | EE angular velocity |
| `object` | 14 | Cube pos(3) + quat(4) + vel(7) |

### Example: 16D State (adding joint_pos)

To increase from 9D to 16D (`eef_pos + eef_quat + joint_pos + gripper_qpos`):

#### Step 1: Create a New Dataset

The state composition is **baked into the dataset at conversion time**. You cannot change it at training time. Edit `get_expected_low_dim_keys()` in two files then re-convert:

**File 1:** [resfit/lerobot/dataset/convert_robomimic_to_lerobot.py](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py) ~line 268

```python
panda_low_dim_keys = [
    "robot0_eef_pos",          # 3D
    "robot0_eef_quat",         # 4D
    "robot0_joint_pos",        # 7D  ← ADD THIS
    "robot0_gripper_qpos",     # 2D
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
]
```

**File 2:** [resfit/dexmg/environments/dexmg.py](resfit/dexmg/environments/dexmg.py) ~line 460 — `_get_expected_low_dim_keys()`

```python
panda_low_dim_keys_single = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_joint_pos",        # ← ADD THIS (must match dataset)
    "robot0_gripper_qpos",
]
```

**Both files MUST have the same keys in the same order.** If they don't match, the dataset state and env state will have different dimensions — causing runtime crashes or silent bugs.

#### Step 2: Re-convert the HDF5 Dataset

```bash
# Get the raw HDF5 (if you don't have it already)
cd deps/robomimic
python robomimic/scripts/download_datasets.py --tasks lift --dataset_types mh --hdf5_types raw

# Render with cameras
python robomimic/scripts/dataset_states_to_obs.py \
    --dataset datasets/lift/mh/demo_v15.hdf5 \
    --output_name image_custom.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 84 --camera_width 84

# Convert to LeRobot format
cd ../..
python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
    --dataset deps/robomimic/datasets/lift/mh/image_custom.hdf5 \
    --output_dir ~/lerobot_datasets/lift-mh-16d \
    --max_episodes 300
```

#### Step 3: Retrain Everything

1. **Retrain BC** with `--dataset ~/lerobot_datasets/lift-mh-16d` — the ACT policy auto-detects the new 16D state shape from `dataset.meta.features["observation.state"].shape`
2. **Retrain RL** — the RL agent auto-detects `lowdim_dim` from `env.observation_space["observation.state"].shape[1]` and `StateStandardizer` reads new stats from the dataset

**Nothing else needs to change in code.** The ACT model creates `nn.Linear(state_dim, dim_model)` dynamically. The RL critic/actor creates `nn.Linear(lowdim_dim, hidden_dim)` dynamically. Both auto-adapt.

### Example: 25D State (adding velocities)

Same process but with more keys:

```python
panda_low_dim_keys_single = [
    "robot0_eef_pos",          # 3
    "robot0_eef_quat",         # 4
    "robot0_joint_pos",        # 7
    "robot0_joint_vel",        # 7
    "robot0_gripper_qpos",     # 2
    "robot0_gripper_qvel",     # 2
]
# Total: 25D
```

**Caveat:** The HDF5 must contain these keys. The raw robomimic HDF5 has all of them. But if you downloaded a pre-processed file, some keys may be missing. Check with:

```python
import h5py
f = h5py.File("image.hdf5", "r")
print(list(f["data/demo_0/obs"].keys()))
```

---

## 13. Changing Image Resolution

Changing from 84×84 requires coordinated changes across the entire pipeline.

### What Must Change

| Component | File | Change Required |
|---|---|---|
| Dataset | Re-convert HDF5 with new `--camera_height/width` | New resolution in HDF5 |
| BC training | No code change (auto-detects from dataset) | Just pass new dataset |
| BC eval | `--eval_camera_size <new_size>` | CLI flag |
| RL env rendering | [dexmg.py](resfit/dexmg/environments/dexmg.py) ~line 92 | Change `camera_size=84` default or pass explicitly |
| RL vision backbone | [min_vit.py](resfit/rl_finetuning/off_policy/networks/min_vit.py) line 35 | **MUST update `num_patch`** |

### MinViT Patch Count Math

`PatchEmbed2` has two conv layers:
- Conv1: `kernel=8, stride=4` → output spatial: `(H - 8) / 4 + 1`
- Conv2: `kernel=3, stride=2` → output spatial: `(prev - 3) / 2 + 1`
- `num_patch = spatial_h × spatial_w`

| Input Size | Conv1 Output | Conv2 Output | num_patch | Status |
|---|---|---|---|---|
| 84×84 | 20×20 | 9×9 | **81** | Current default |
| 96×96 | 23×23 | 11×11 | **121** | Requires code change |
| 128×128 | 31×31 | 15×15 | **225** | Requires code change |
| 64×64 | 15×15 | 7×7 | **49** | Requires code change |

To change, edit `PatchEmbed2` in [min_vit.py](resfit/rl_finetuning/off_policy/networks/min_vit.py):
```python
# self.num_patch = 81  # for 84x84
self.num_patch = 121   # for 96x96
```

There is also `PatchEmbed1` with `kernel=8, stride=8`:
```python
# PatchEmbed1: num_patch = 144 for 96x96
```
But `PatchEmbed1` is unused — only `PatchEmbed2` (via `embed_style="embed2"`) is used in practice.

### BC Side: ACT Uses ResNet (Variable Input Size)

The ACT backbone is ResNet18, which is fully convolutional and handles any spatial resolution. The output feature map size changes:
- 84×84 → layer4 output 3×3 → 9 spatial tokens per camera
- 128×128 → layer4 output 4×4 → 16 spatial tokens per camera

The ACT transformer encoder handles variable sequence lengths, so **no ACT code changes are needed** — just retrain with the new dataset.

### Step-by-Step: Switching to 128×128

```bash
# 1. Re-render HDF5 at 128×128
python robomimic/scripts/dataset_states_to_obs.py \
    --dataset datasets/lift/mh/demo_v15.hdf5 \
    --output_name image_128.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 128 --camera_width 128

# 2. Convert to LeRobot format
python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
    --dataset deps/robomimic/datasets/lift/mh/image_128.hdf5 \
    --output_dir ~/lerobot_datasets/lift-mh-128
```

Then edit `min_vit.py`:
```python
self.num_patch = 225  # for 128x128
```

And either:
- Edit `dexmg.py` default: `camera_size: int = 128`
- Or pass `camera_size=128` in `create_vectorized_env()` call in `train_residual_td3.py`

Retrain BC first, then RL.

---

## 14. Adding Custom Gym Wrappers

### The RL Environment Pipeline

```
robosuite.make()
    ↓
RobosuiteGymWrapper          ← single env, Gymnasium API
    ↓
[INSERT PER-ENV WRAPPERS HERE]  ← Option A
    ↓
AsyncVectorEnv / SyncVectorEnv   ← vectorization (N copies)
    ↓
VectorizedEnvWrapper          ← numpy → torch tensor conversion
    ↓
[INSERT POST-VEC WRAPPERS HERE]  ← Option B
    ↓
BasePolicyVecEnvWrapper       ← adds base policy + residual combining
    ↓
Training loop
```

### Option A: Per-Environment Wrapper (Before Vectorization)

Best for: reward shaping, observation filtering, action transformations, curriculum.

Edit `make_dexmimicgen_env()` in [dexmg.py](resfit/dexmg/environments/dexmg.py) ~line 643:

```python
def _make():
    env = RobosuiteGymWrapper(
        env_name=env_name,
        camera_size=camera_size,
        render_size=render_size,
        image_keys=image_keys,
        headless=headless,
    )
    # ─── Insert your wrapper here ───
    env = YourCustomWrapper(env)
    # ────────────────────────────────
    return env
```

**Example — Dense reward shaping wrapper:**

```python
import gymnasium as gym
import numpy as np

class DenseRewardWrapper(gym.Wrapper):
    """Add a small reward bonus for reaching toward the cube."""
    
    def __init__(self, env, reach_bonus_scale=0.1):
        super().__init__(env)
        self.reach_bonus_scale = reach_bonus_scale
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add distance-based bonus (smaller distance → larger bonus)
        if "robot0_eef_pos" in info and "cube_pos" in info:
            dist = np.linalg.norm(info["robot0_eef_pos"] - info["cube_pos"])
            reach_bonus = self.reach_bonus_scale * max(0, 1.0 - dist)
            reward += reach_bonus
        return obs, reward, terminated, truncated, info
```

**Important:** Per-env wrappers run in **forked subprocess** workers (AsyncVectorEnv). They must be picklable and cannot reference GPU tensors or shared state.

### Option B: Post-Vectorization Wrapper

Best for: batched reward modifications, logging, episode statistics, frame stacking.

Edit `get_envs()` in [train_residual_td3.py](resfit/rl_finetuning/scripts/train_residual_td3.py) ~line 296:

```python
def get_envs(...):
    vec_env = create_vectorized_env(
        env_name=env_name, num_envs=num_envs, device=device,
        video_key=video_key, debug=debug, headless=cfg.headless,
    )
    # ─── Insert post-vec wrapper here ───
    vec_env = YourVecEnvWrapper(vec_env)
    # ────────────────────────────────────
    return BasePolicyVecEnvWrapper(
        vec_env=vec_env, base_policy=base_policy,
        action_scaler=action_scaler, state_standardizer=state_standardizer,
    )
```

**Example — Episode return logger:**

```python
class EpisodeReturnTracker:
    """Track per-episode returns for debugging."""
    
    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.episode_returns = torch.zeros(vec_env.num_envs)
    
    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        self.episode_returns.zero_()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.vec_env.step(action)
        self.episode_returns += reward.squeeze()
        done = terminated | truncated
        if done.any():
            for i in torch.where(done)[0]:
                print(f"Env {i}: episode return = {self.episode_returns[i]:.3f}")
                self.episode_returns[i] = 0.0
        return obs, reward, terminated, truncated, info
    
    # Delegate everything else
    def __getattr__(self, name):
        return getattr(self.vec_env, name)
```

### Existing Wrapper to Follow

The canonical example is `BasePolicyVecEnvWrapper` in [resfit/rl_finetuning/wrappers/residual_env_wrapper.py](resfit/rl_finetuning/wrappers/residual_env_wrapper.py). Key patterns:

1. Stores `self.vec_env` and delegates via `__getattr__`
2. Has `observation_space` and `action_space` properties
3. `reset()` returns `(obs_dict, info)`
4. `step()` returns `(obs_dict, reward, terminated, truncated, info)`
5. Modifies observations in `_augment_obs()` — copies dict, modifies tensors

### Things to Watch Out For

1. **Observation space must stay consistent.** If your wrapper adds/removes observation keys, update `observation_space` accordingly, otherwise the replay buffer will crash.
2. **Don't modify images in-place.** The `VectorizedEnvWrapper` creates torch tensors that may share memory. Always `.clone()` before modifying.
3. **Per-env wrappers must be picklable** (AsyncVectorEnv forks). No lambdas, no GPU tensors, no file handles.
4. **Reward modifications affect Q-value scale.** If you add dense reward, update `v_min`/`v_max` for distributional critics.
5. **N-step returns.** Reward shaping wrappers interact with n-step return calculation. Make sure your modified rewards are compatible with the n-step buffer logic.

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)
