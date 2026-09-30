# Reward Signal & Success Criteria — Every Task Explained

This document explains how the reward and success signals work in the residual RL pipeline — from the physics-level success checks inside each simulated task, through the environment wrapper, into the replay buffer, and finally into the critic's TD target.

**Key takeaway**: Every task uses **sparse binary reward** (0.0 or 1.0). Reward = 1.0 only when a task-specific `_check_success()` method returns True. There is no dense reward shaping (stubs exist but do nothing for the tasks used in this project).

---

## Table of Contents

1. [How Reward Flows Through the System](#1-how-reward-flows-through-the-system)
2. [Success Detection in the Training Loop](#2-success-detection-in-the-training-loop)
3. [Task-by-Task Success Criteria](#3-task-by-task-success-criteria)
   - [TwoArmCoffee](#twarmcoffee)
   - [TwoArmBoxCleanup](#twoarmboxcleanup)
   - [TwoArmCanSortRandom](#twoarmcansortrandom)
   - [TwoArmThreading](#twoarmthreading)
   - [TwoArmThreePieceAssembly](#twoarmthreepieceassembly)
   - [TwoArmTransport](#twoarmtransport)
   - [TwoArmLiftTray](#twoarmlifttray)
   - [TwoArmDrawerCleanup](#twoarmdrawercleanup)
   - [TwoArmPouring](#twoarmpouring)
   - [PickPlaceCan (Can)](#pickplacecan-can)
   - [NutAssemblySquare (Square)](#nutassemblysquare-square)
4. [How Reward Enters the Replay Buffer](#4-how-reward-enters-the-replay-buffer)
5. [N-Step Return Transform](#5-n-step-return-transform)
6. [How the Critic Uses the Reward](#6-how-the-critic-uses-the-reward)
7. [Evaluation: How success_rate Is Computed](#7-evaluation-how-success_rate-is-computed)
8. [Real Robot Experiments](#8-real-robot-experiments)
9. [Reward Shaping: What Exists and What's Used](#9-reward-shaping-what-exists-and-whats-used)

---

## 1. How Reward Flows Through the System

```
robosuite / DexMimicGen environment
    │
    │  env.step(action)
    │  internally calls: self.reward(action)
    │    → calls _check_success()
    │    → returns 1.0 if success, 0.0 otherwise
    │
    ▼
RobosuiteGymWrapper.step()                    ← resfit/dexmg/environments/dexmg.py L309
    │  obs, reward, done, info = self.env.step(action)
    │  reward_scalar = float(reward)           ← raw 0.0 or 1.0
    │  success = reward == 1.0                 ← success iff reward is exactly 1.0
    │  terminated = bool(success)              ← episode terminates ON success
    │  truncated = bool(done)                  ← robosuite done = horizon timeout
    │
    ▼
VectorizedEnvWrapper.step()                    ← converts to torch tensors
    │
    ▼
BasePolicyVecEnvWrapper.step()                 ← combines base+residual, passes reward through
    │
    ▼
Training loop                                  ← resfit/rl_finetuning/scripts/train_residual_td3.py
    │  next_obs, reward, terminated, truncated, info = env.step(action)
    │  reward goes into replay buffer UNCHANGED
    │
    ▼
MultiStepTransform (at sampling time)           ← resfit/rl_finetuning/utils/rb_transforms.py
    │  Computes n-step discounted return: R_n = Σ γ^i × r_{t+i}
    │
    ▼
Critic update                                  ← resfit/rl_finetuning/off_policy/rl/q_agent.py
    │  target_q = R_n + γ^n × Q_target(s_{t+n}, a_{t+n})
    │  critic_loss = MSE(Q(s,a), target_q)
```

### The reward is sparse binary

Every task in this codebase follows the same pattern:

```python
# In every DexMimicGen environment's reward() method:
def reward(self, action=None):
    reward = 0.0
    if self._check_success():
        reward = 1.0
    if self.reward_shaping:
        pass          # ← stub, does nothing
    if self.reward_scale is not None:
        reward *= self.reward_scale  # default 1.0
    return reward
```

The agent sees:
- **0.0** for every step where the task is not yet complete
- **1.0** for the single step where `_check_success()` first returns True
- The episode then terminates immediately (the wrapper sets `terminated=True` on success)

This means the agent must learn entirely from the difference between "episode ended with reward 1.0" vs "episode timed out with reward 0.0".

---

## 2. Success Detection in the Training Loop

The `RobosuiteGymWrapper` at `resfit/dexmg/environments/dexmg.py` lines 309–332 converts robosuite's 4-value return into Gymnasium's 5-value return:

```python
obs, reward, done, info = self.env.step(action)    # robosuite: 4 values
reward_scalar = float(reward)                       # 0.0 or 1.0

success = reward == 1.0
terminated_scalar = bool(success)   # episode ENDS because task succeeded
truncated_scalar = bool(done)       # episode ENDS because horizon reached (timeout)
```

**Important distinction:**

| Condition | `terminated` | `truncated` | `reward` | Meaning |
|---|---|---|---|---|
| Task succeeded (mid-episode) | `True` | `False` | `1.0` | Agent solved the task — episode ends early |
| Horizon reached, no success | `False` | `True` | `0.0` | Agent ran out of time — 400 steps elapsed |
| Success at exact horizon step | `True` | `True` | `1.0` | Rare: succeeded on the very last allowed step |

This distinction matters for the critic: `terminated=True` means there's no future reward (Q-target uses `nonterminal=False`), while `truncated=True` means the agent *might* have succeeded if given more time (Q-target still uses `nonterminal=True` for bootstrapping — controlled by the n-step transform's `done` handling).

---

## 3. Task-by-Task Success Criteria

Every task has a `_check_success()` method that returns `True` or `False` based on the **current physics state** (object positions, joint angles, contact forces). The check runs at every simulation step. Here are all the tasks configured in this codebase:

---

### TwoArmCoffee

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_coffee.py` (lines 327–375)

**What the robot must do**: Pick up a coffee pod, open the coffee machine lid, place the pod in the holder, and close the lid.

**Success condition** — both must be true simultaneously:

| Check | Condition | Threshold |
|---|---|---|
| **Lid closed** | Coffee machine hinge angle < tolerance | hinge_angle < **15°** (0.2618 rad) |
| **Pod in holder** | Pod center horizontally within holder bounds | XY distance < `pod_holder_radius - pod_radius` |
| | Pod vertically between holder floor and lid | `holder_bottom ≤ pod_z ≤ lid_bottom` |

```python
# Lid check (line 335):
hinge_tolerance = 15.0 * np.pi / 180.0
lid_check = self.sim.data.qpos[self.hinge_qpos_addr] < hinge_tolerance

# Pod check (line 341):
r_diff = self.pod_holder_size[0] - self.pod_size[0]
pod_horz_ok = np.linalg.norm(pod_pos[:2] - holder_pos[:2]) <= r_diff
pod_vert_ok = (pod_z - pod_radius >= holder_bottom) and (pod_z + pod_radius <= lid_bottom)

success = lid_check and pod_horz_ok and pod_vert_ok
```

**Plain English**: The pod must be sitting inside the holder (not floating or tilted out), and the lid must be closed within 15 degrees of shut.

---

### TwoArmBoxCleanup

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_box_cleanup.py` (lines 269–304)

**What the robot must do**: Pick up a lid and place it on top of a box.

**Success condition**:

| Check | Condition | Threshold |
|---|---|---|
| **XY alignment** | Lid center vs box center in XY plane | distance < **0.03 m** (3 cm) |
| **Z alignment** | Lid height vs box height | Z distance < **1.01 × (box_height + lid_height)** |

```python
xy_check = np.linalg.norm(box_pos[:2] - lid_pos[:2]) < 0.03
z_check = np.abs(box_pos[2] - lid_pos[2]) < 1.01 * (box_height + lid_height)
success = xy_check and z_check
```

**Plain English**: The lid must be centered on the box within 3 cm horizontally and at the correct height (sitting on top, not floating above or beside it).

---

### TwoArmCanSortRandom

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_can_sort.py` (lines 248–285)

**What the robot must do**: Pick up a can and place it in the correct colored bin (red or blue, randomly selected each episode).

**Success condition**:

| Check | Condition | Threshold |
|---|---|---|
| **X position** | Can center vs target bin center | < `bin_width / 2` |
| **Y position** | Can center vs target bin center | < `bin_depth / 2` |
| **Z position** | Can height above bin base | < `can_height / 2` |

The target bin color is randomized at episode start (`self.is_red`). Only the correct bin counts.

**Plain English**: The can must be inside the correct colored bin — within the bin's XY boundary and sitting near the bottom (not balanced on the rim).

---

### TwoArmThreading

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_threading.py` (lines 205–230)

**What the robot must do**: Thread a needle through a small ring mounted on a tripod.

**Success condition**:

| Check | Condition | Threshold |
|---|---|---|
| **Needle-ring distance** | 3D Euclidean distance from needle tip to ring centroid | < **0.012 m** (12 mm) |

```python
ring_pos = average of all ring geom positions
needle_pos = sim.data.geom_xpos["needle_obj_needle"]
success = np.linalg.norm(needle_pos - ring_pos) < 0.012
```

**Plain English**: The needle tip must be within 12 mm of the center of the ring. This is extremely precise — the tightest tolerance of any task.

---

### TwoArmThreePieceAssembly

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_three_piece_assembly.py` (lines 338–413)

**What the robot must do**: Assemble two pieces onto a base in sequence — piece 1 first, then piece 2 on top.

**Success condition** (both pieces assembled):

| Check | Condition | Threshold |
|---|---|---|
| **Piece 1 placed** | XY distance to base | < **0.02 m** (2 cm) |
| | Robot not grasping piece 1 | gripper open / no contact |
| **Piece 2 placed** | Piece 1 must already be assembled | (above passes) |
| | XY distance to piece 1 | < **0.02 m** (2 cm) |
| | Z height correct | within **0.02 m** of `base_z + 4 × piece_2_size` |
| | Robot not grasping piece 2 | gripper open / no contact |

**Plain English**: Both pieces must be stacked on the base (each within 2 cm XY), piece 2 at the correct height, and the robot must have released both pieces.

---

### TwoArmTransport

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_transport.py` (lines 361–373)

**What the robot must do**: Transport a payload to a target bin and trash to a trash bin.

**Success condition**:

| Check | Condition | Method |
|---|---|---|
| **Payload in target bin** | Physics contact between payload and target bin base geoms | `SU.check_contact()` |
| **Trash in trash bin** | Physics contact between trash and trash bin base geoms | `SU.check_contact()` |

**Plain English**: Both objects must be in their correct bins, detected via MuJoCo contact simulation (not distance). This is a contact-based check rather than position-based.

---

### TwoArmLiftTray

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_lift_tray.py` (lines 357–385)

**What the robot must do**: Lift a tray (pot) with two objects on it above the table.

**Success condition**:

| Check | Condition | Threshold |
|---|---|---|
| **Tray lifted** | Pot bottom height above table | > **0.10 m** (10 cm) |
| **Object 1 lifted** | Object 1 height above table | > **0.10 m** (10 cm) |
| **Object 2 lifted** | Object 2 height above table | > **0.10 m** (10 cm) |
| **Objects on tray** | Both objects in contact with pot base geom | `check_contact()` |

**Plain English**: The tray must be lifted at least 10 cm above the table with both objects still sitting on it (contact maintained). If the objects fall off during the lift, it's not a success.

**Note**: This is the only DexMimicGen task with actual reward shaping implementation (dense reward for reaching handles, grasping, lifting, tilt direction). However, the default config uses `reward_shaping=False`, so it still gets sparse 0/1 reward.

---

### TwoArmDrawerCleanup

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_drawer_cleanup.py` (lines 336–354)

**What the robot must do**: Open a drawer, place an object inside, and close the drawer.

**Success condition**:

| Check | Condition | Threshold |
|---|---|---|
| **Object in drawer** | Object in contact with drawer bottom geom | `check_contact()` |
| **Drawer closed** | Drawer joint position near zero | qpos > **-0.01** (where 0.0 = fully closed, -0.135 = fully open) |

**Plain English**: The object must be inside the drawer (touching the drawer bottom), and the drawer must be closed (joint within 1% of fully closed position).

---

### TwoArmPouring

**File**: `deps/dexmimicgen/dexmimicgen/environments/two_arm_pouring.py` (lines 363–396)

**What the robot must do**: Pour a ball from a cup into a bowl, with the bowl sitting upright on a pad.

**Success condition** (all must be true):

| Check | Condition | Threshold |
|---|---|---|
| **Ball in bowl** | Ball in contact with bowl | `check_contact()` |
| **Ball centered** | Ball XY distance to bowl center | < **0.10 m** (10 cm) |
| **Bowl on pad** | Bowl in contact with pad | `check_contact()` |
| **Bowl upright** | Bowl z-axis alignment to vertical | `1.0 - z_axis[2]` < **0.05** (~18° tilt) |
| **Bowl centered on pad** | Bowl XY distance to pad center | < **0.06 m** (6 cm) |

**Plain English**: The ball must be inside the bowl (contact + within 10 cm XY), and the bowl must be sitting upright on the pad (contact, within 6 cm XY of pad center, tilted less than ~18° from vertical).

---

### PickPlaceCan (Can)

**File**: `deps/robosuite/robosuite/environments/manipulation/pick_place.py` (lines 733–762)

**What the robot must do**: Pick up a can and place it in the correct bin quadrant of a divided container.

**Success condition**:

| Check | Condition | Threshold |
|---|---|---|
| **Can in bin** | Can XY within bin quadrant boundaries | Dynamic (computed from bin center + bin_size) |
| | Can Z above bin base | < **0.10 m** above bin base |
| **Gripper released** | Gripper distance from can | > **~0.042 m** (4.2 cm) |

The gripper release check uses: `1 - tanh(10 × distance) < 0.6`, which solves to `distance > atanh(0.4)/10 ≈ 0.042 m`.

**Plain English**: The can must be inside its target bin quadrant and the robot must have released it (gripper at least 4.2 cm away).

**Note**: This robosuite task has **full dense reward shaping** (4 stages: reach → grasp → lift → hover), but only when `reward_shaping=True`. The residual RL configs use sparse reward.

---

### NutAssemblySquare (Square)

**File**: `deps/robosuite/robosuite/environments/manipulation/nut_assembly.py` (lines 619–643)

**What the robot must do**: Pick up a square nut and place it on the correct peg.

**Success condition**:

| Check | Condition | Threshold |
|---|---|---|
| **Nut near peg XY** | Nut X distance to peg center | < **0.03 m** (3 cm) |
| | Nut Y distance to peg center | < **0.03 m** (3 cm) |
| **Nut at table height** | Nut Z position | < **table_offset[2] + 0.05 m** |
| **Gripper released** | Gripper distance from nut | > **~0.042 m** (4.2 cm) |

**Plain English**: The square nut must be around its peg (within 3 cm XY, near table height) and the robot must have released it.

**Note**: Like PickPlaceCan, this has full dense reward shaping available but is used with sparse reward in the residual RL configs.

---

## 4. How Reward Enters the Replay Buffer

### Online buffer (RL interactions)

At `train_residual_td3.py` line ~193, each transition is stored:

```python
td = TensorDict({
    "obs": obs_dict,
    "action": scaled_action,
    "next": {
        "reward": reward[i],         # ← raw 0.0 or 1.0, UNCHANGED
        "done": done_flag,
        "obs": next_obs_dict,
        "original_reward": reward[i], # kept for logging
    },
})
online_rb.add(td)
```

No clipping, no scaling, no shaping. The raw sparse reward goes straight into the buffer.

### Offline buffer (demonstrations)

At `train_residual_td3.py` line ~620, demo transitions use:

```python
"reward": torch.tensor(float(done_flag), dtype=torch.float32)
```

Since all demonstrations are successful episodes, the last frame of each episode has `done=True` → `reward=1.0`. All other frames get `reward=0.0`. This matches what the environment would return if the demo were replayed live.

---

## 5. N-Step Return Transform

The `MultiStepTransform` (at `resfit/rl_finetuning/utils/rb_transforms.py` lines 217–293) is applied **at sampling time**, not at storage time. When a batch is sampled from the replay buffer:

For each transition at time $t$, it computes:

$$R_n = \sum_{i=0}^{n-1} \gamma^i \cdot r_{t+i}$$

With default `n_step=5`, `gamma=0.995`:

**Example — success at step $t+3$** (reward = [0, 0, 0, 1, x]):
$$R_5 = 0 + 0 + 0 + 0.995^3 \cdot 1.0 = 0.985$$

The transform also writes:
- `"gamma"` = $\gamma^{n_\text{actual}}$ where $n_\text{actual}$ may be < $n$ if the episode ended
- `"nonterminal"` = 0 if the episode terminated within the $n$ steps (no bootstrapping)
- `"next/obs"` updated to point to the observation $n_\text{actual}$ steps ahead

**Why this matters for sparse reward**: Without n-step returns, only the single transition where `reward=1.0` carries signal. With `n_step=5`, the 4 transitions *before* success also get nonzero reward through the discounted sum. This makes credit assignment much easier — the critic learns "being in states that lead to success within 5 steps is valuable".

---

## 6. How the Critic Uses the Reward

At `resfit/rl_finetuning/off_policy/rl/q_agent.py` line 336:

```python
# discount = γ^n × nonterminal (0 if episode ended, γ^n otherwise)
discount = batch["gamma"] * batch["nonterminal"]

# Bellman target
target_q = reward + discount * target_q_min

# Optional clamping (disabled by default)
if self.cfg.clip_q_target_to_reward_range:
    target_q = torch.clamp(target_q, min=0, max=1)  # since rewards ∈ {0, 1}
```

The `target_q_min` is the minimum Q-value from 2 randomly selected heads out of the 10-head ensemble (RED-Q style), evaluated on the next observation with the target actor + smoothing noise.

### What the Q-values represent intuitively

Since rewards are 0/1 and $\gamma = 0.995$:

$$Q(s, a) \approx \gamma^k \cdot P(\text{success within horizon} \mid s, a)$$

where $k$ is the expected number of steps until success. A Q-value of 0.8 roughly means "~80% chance of success, with some discounting for the number of steps remaining". Q-values near 0 mean "almost certainly going to time out". Q-values near 1 mean "about to succeed".

---

## 7. Evaluation: How success_rate Is Computed

**File**: `resfit/rl_finetuning/utils/evaluate_dexmg.py` (lines 174–269)

Every `eval_interval_every_steps` (default 10,000) steps, evaluation runs `eval_num_episodes` (default 50) episodes across `eval_num_envs` parallel environments:

```python
# For each episode that finishes:
is_success = bool(reward[env_idx].item() == 1.0)   # line 213
successes.append(is_success)

# After all episodes complete:
success_rate = float(np.mean(successes))             # line 269
```

**How it works**:
1. The agent runs in eval mode (no exploration noise, `stddev=0.0`)
2. Each episode runs until `terminated` or `truncated`
3. Success = the reward on the step that triggered `done` was 1.0
4. `success_rate` = fraction of episodes where this happened

**Why checking `reward == 1.0` at done time is sufficient**: When the task succeeds, two things happen simultaneously:
- The environment's `reward()` returns 1.0
- The wrapper sets `terminated=True`

So at the exact step the episode ends with `terminated=True`, the reward must be 1.0 for it to count as success. If the episode ends with `truncated=True` (timeout), the reward on that step is 0.0.

### Metrics logged to WandB

| Metric | Formula |
|---|---|
| `eval/success_rate` | `mean(successes)` — fraction of episodes that succeeded |
| `eval/mean_return` | `mean(episode_returns)` — mean undiscounted sum of rewards per episode (equals 1.0 for success, 0.0 for failure, since reward is sparse) |
| `eval/mean_successful_episode_length` | Mean number of steps for episodes that succeeded (shorter = better — the agent solved it faster) |

---

## 8. Real Robot Experiments

**The codebase is simulation-only.** There is no real-robot reward or success computation code in this repository.

The ResFiT paper reports real-robot results (e.g., 64% success rate on physical TwoArmCoffee), but those were evaluated via:
- **Manual human evaluation** — a person watches the robot attempt the task and judges success/failure
- **Blind A/B testing** — evaluators don't know which policy (BC vs ResFiT) is running

This is standard practice for sim-to-real transfer research — reward functions are physics-simulation-specific (they read MuJoCo joint angles and body positions), so they can't run on a physical robot. Real-world success is determined by human judgment.

---

## 9. Reward Shaping: What Exists and What's Used

### DexMimicGen tasks: stubs only

Every DexMimicGen task has this pattern:

```python
def reward(self, action=None):
    reward = 0.0
    if self._check_success():
        reward = 1.0
    if self.reward_shaping:
        pass              # ← does nothing, no dense reward implemented
    return reward
```

**Exception**: `TwoArmLiftTray` has actual reward shaping code (tilt direction, lift height, reach distance, grasp detection), but `reward_shaping` defaults to `False` in the configs used by this project.

### Robosuite tasks: implemented but unused

The standard robosuite tasks (`PickPlaceCan`, `NutAssemblySquare`) have **full 4-stage dense reward shaping**:

1. **Reach** (0–0.1): `tanh(distance to object)` — reward for moving gripper toward object
2. **Grasp** (0.35): Binary — reward for successfully grasping the object
3. **Lift** (0.35–0.5): Reward for lifting the object above a threshold height
4. **Hover** (0.5–0.7): Reward for moving the lifted object near the target bin/peg

These are gated sequentially (can't get lift reward without grasp, etc.). The total potential reward with shaping is ~0.7, with the remaining 0.3 reserved for final placement success.

**However**, the residual RL configs in this project use these tasks with `reward_shaping=False`, so only the sparse 0/1 success reward is active.

### Why sparse reward works here

Normally, sparse reward is extremely difficult for RL — the agent gets no gradient signal until it accidentally succeeds, which may never happen in a 400-step episode with 24 action dimensions.

Residual RL makes sparse reward tractable because:
1. **The frozen BC policy already gets close** — it knows roughly what to do from imitation learning
2. **The residual only needs small corrections** — bounded to ±0.2 in normalized space
3. **Offline demonstrations provide reward signal** — the 50/50 online/offline mixing ensures every batch contains successful transitions from demos
4. **N-step returns propagate signal** — reward from the success step spreads backwards through $n$ preceding transitions

Without the BC base policy and offline demonstrations, sparse reward with these tasks would be essentially impossible to learn.

---

### Quick Reference: All Success Thresholds

| Task | Key Threshold(s) | Type |
|---|---|---|
| TwoArmCoffee | Lid < 15°, pod within holder radius | Angle + position |
| TwoArmBoxCleanup | XY < 3 cm, Z within 1.01× stacked height | Position |
| TwoArmCanSortRandom | Within bin XY bounds, Z < can_height/2 | Position |
| TwoArmThreading | Needle-ring distance < **12 mm** | Position (tightest) |
| TwoArmThreePieceAssembly | Each piece within 2 cm XY, correct Z, released | Position + contact |
| TwoArmTransport | Payload + trash touching correct bin bases | Contact only |
| TwoArmLiftTray | Tray + objects > 10 cm above table, objects on tray | Height + contact |
| TwoArmDrawerCleanup | Object touching drawer bottom, drawer joint > -0.01 | Contact + joint |
| TwoArmPouring | Ball in bowl (contact + 10 cm XY), bowl upright on pad (<18° tilt, <6 cm XY) | Contact + position + orientation |
| PickPlaceCan | Can in bin quadrant, Z < 10 cm above base, gripper > 4.2 cm away | Position + release |
| NutAssemblySquare | Nut within 3 cm XY of peg, Z < table+5 cm, gripper > 4.2 cm away | Position + release |

---

**Related documents:**
- [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md) — Full training pipeline walkthrough
- [../algorithms/NSTEP_RETURNS.md](../algorithms/NSTEP_RETURNS.md) — Multi-step TD learning with numerical walkthrough
- [../training/CHECKPOINTING_AND_RESUME.md](../training/CHECKPOINTING_AND_RESUME.md) — What gets saved, buffer lifecycle, memory budget
- [../algorithms/TD3_ALGORITHM.md](../algorithms/TD3_ALGORITHM.md) — Critic update with Bellman targets
