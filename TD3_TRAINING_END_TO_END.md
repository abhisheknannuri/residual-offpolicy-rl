# TD3 Residual RL Training — Complete End-to-End Pipeline

This document explains **everything** that happens from the moment you run `train_residual_rl_lift.sh` to the point where the critic and actor are being updated, in the exact order the code executes. No assumptions.

---

## Table of Contents

1. [Phase 0: Load BC Policy & Dataset](#phase-0-load-bc-policy--dataset)
2. [Phase 1: Build Normalization From Dataset](#phase-1-build-normalization-from-dataset)
3. [Phase 2: Create Replay Buffers (With N-Step Transform)](#phase-2-create-replay-buffers-with-n-step-transform)
4. [Phase 3: Populate Offline Buffer From Dataset](#phase-3-populate-offline-buffer-from-dataset)
5. [Phase 4: How The N-Step Transform Processes Transitions](#phase-4-how-the-n-step-transform-processes-transitions)
6. [Phase 5: Warmup — Fill Online Buffer](#phase-5-warmup--fill-online-buffer)
7. [Phase 6: Critic Warmup (No Actor Updates)](#phase-6-critic-warmup-no-actor-updates)
8. [Phase 7: Main Training Loop (Step By Step)](#phase-7-main-training-loop-step-by-step)
9. [Phase 8: The Gradient Update (UTD, Critic, Actor)](#phase-8-the-gradient-update-utd-critic-actor)
10. [The Done Flag Problem You Asked About](#the-done-flag-problem-you-asked-about)
11. [Complete Numerical Example](#complete-numerical-example)

---

## Phase 0: Load BC Policy & Dataset

**File: `resfit/rl_finetuning/scripts/train_residual_td3.py`, lines 237–259**

```python
# Download the frozen BC policy from WandB
policy_dir, _ = download_policy_from_wandb(
    cfg.base_policy.wandb_id,
    step=cfg.base_policy.wt_type,
    artifact_version=cfg.base_policy.wt_version,
)

base_policy: ACTPolicy = load_policy(policy_dir)
base_policy.to(device)
base_policy.eval()
```

Then the HuggingFace dataset:

```python
dataset = LeRobotDataset(cfg.offline_data.name, image_transforms=offline_image_transforms)
```

**What comes back from the dataset:**

The dataset `poolvarine/robomimic-mh-lift-image-dense` has these keys per sample:
```
['action', 'next.done', 'next.reward', 'observation.state', 'timestamp',
 'frame_index', 'episode_index', 'index', 'task_index']
+ image keys like 'observation.images.agentview', 'observation.images.robot0_eye_in_hand'
```

There is NO discount, NO Q-value, NO nonterminal in the raw dataset. Those are computed later by the replay buffer transform.

---

## Phase 1: Build Normalization From Dataset

**File: `resfit/rl_finetuning/scripts/train_residual_td3.py`, lines 265–279**

```python
# Create action scaler from dataset statistics
action_scaler = ActionScaler.from_dataset_stats(
    action_stats=dataset.meta.stats["action"],
    action_scale=cfg.agent.actor.action_scale,
    min_range_per_dim=cfg.offline_data.min_action_range,
    device=device,
)

# Create state standardizer from dataset statistics
state_standardizer = StateStandardizer.from_dataset_stats(
    state_stats=dataset.meta.stats["observation.state"],
    min_std=cfg.offline_data.min_state_std,
    device=device,
)
```

**What these do:**

- `ActionScaler`: computes `action_min` and `action_max` per dimension from the dataset's min/max stats. Then `scale(a)` maps raw actions to `[-1, 1]` and `unscale(n)` maps back.
- `StateStandardizer`: computes mean and std per state dimension. Then `standardize(s) = (s - mean) / std`.

These are used for ALL data that enters the system — both offline dataset transitions AND online environment transitions.

---

## Phase 2: Create Replay Buffers (With N-Step Transform)

**File: `resfit/rl_finetuning/scripts/train_residual_td3.py`, lines 419–441**

```python
online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))  # 256 * 0.5 = 128
offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)        # 256 * 0.5 = 128

# ONLINE buffer — transitions from environment interaction
online_rb = TensorDictPrioritizedReplayBuffer(
    storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
    alpha=alpha,  # 0.0 for uniform sampling
    beta=beta,    # 0.0 for uniform sampling
    priority_key="_priority",
    transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
    batch_size=online_batch_size,
)

# OFFLINE buffer — transitions from demonstration dataset
offline_rb = TensorDictPrioritizedReplayBuffer(
    storage=LazyTensorStorage(max_size=max_offline_transitions, device="cpu"),
    alpha=alpha,
    beta=beta,
    priority_key="_priority",
    transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
    batch_size=max(offline_batch_size, 1),
)
```

**CRITICAL:** Both buffers have `MultiStepTransform` attached. This transform runs when transitions are **added** to the buffer (not when sampled). It converts raw 1-step transitions into n-step transitions.

With `N_STEP=3` and `GAMMA=0.995`:
- The transform accumulates 3 consecutive transitions
- Computes the discounted reward sum: $R^{(3)} = r_t + 0.995 \cdot r_{t+1} + 0.990 \cdot r_{t+2}$
- Replaces `next.obs` with the observation 3 steps ahead
- Sets `batch["gamma"] = 0.995^3 = 0.985`
- Sets `batch["nonterminal"]` based on whether the episode ended within those 3 steps

---

## Phase 3: Populate Offline Buffer From Dataset

**File: `resfit/rl_finetuning/scripts/train_residual_td3.py`, lines 547–662**

This is the function `_populate_offline_buffer()`. It iterates through the HuggingFace dataset frame-by-frame and creates transitions:

```python
for sample in tqdm(loader, desc="Processing offline dataset"):
    ep_idx = int(sample["episode_index"].item())
    
    # 1. Scale the ground-truth action to [-1, 1]
    _gt_action = sample["action"].float().squeeze(0)
    gt_action_scaled = action_scaler.scale(_gt_action)
    done_flag = bool(sample["next.done"].item())
    
    # 2. Set base_action = GT action (in default mode)
    base_action_scaled = gt_action_scaled
    
    # 3. Build the observation dict
    curr_obs = {
        "observation.state": state_standardizer.standardize(sample["observation.state"].float().squeeze(0)),
        "observation.base_action": base_action_scaled,
    }
    for k in image_keys:
        curr_obs[k] = sample[k].squeeze(0)
    
    # 4. Get reward (dense from dataset if available, else sparse from done)
    if reward_shaping and "next.reward" in sample:
        step_reward = float(sample["next.reward"].item())
    else:
        step_reward = float(done_flag)  # 1.0 on done, 0.0 otherwise
    
    # 5. Pair with PREVIOUS frame to form a transition
    if ep_idx in episode_cache:
        prev_obs = episode_cache[ep_idx]["obs"]
        prev_action_scaled = episode_cache[ep_idx]["action"]
        prev_reward = episode_cache[ep_idx]["reward"]
        
        transition = TensorDict({
            "obs": TensorDict(prev_obs, batch_size=[]),
            "action": prev_action_scaled,
            "next": TensorDict({
                "obs": TensorDict(curr_obs, batch_size=[]),
                "done": torch.tensor(done_flag, dtype=torch.bool),
                "reward": torch.tensor(prev_reward, dtype=torch.float32),
            }, batch_size=[]),
            "_priority": torch.tensor(10.0, dtype=torch.float32),
        }, batch_size=[]).unsqueeze(0)
        
        rb.add(transition)  # ← THIS triggers the MultiStepTransform
    
    # 6. Cache current frame for next iteration
    episode_cache[ep_idx] = {
        "obs": curr_obs,
        "action": gt_action_scaled,
        "reward": step_reward,
        "done": done_flag,
    }
```

### What Each Transition Looks Like Before N-Step Transform

```
{
  "obs":    { state, base_action, images }   ← observation at time t
  "action": scaled_action_at_t               ← what action was taken at t
  "next": {
    "obs":    { state, base_action, images } ← observation at time t+1
    "done":   True/False                     ← did episode end at t+1?
    "reward": float                          ← reward received at t (prev step's reward)
  }
}
```

### Important Detail: The Reward Lag

Notice that `prev_reward` is used as the reward in the transition, NOT `step_reward`. The reward stored is the reward for the **previous** step's action. This is because the dataset gives `next.reward` which is the reward received AFTER taking the current action and arriving at the next state.

So the transition `(s_t, a_t, r_t, s_{t+1})` has:
- `obs` = state at time t
- `action` = action taken at time t  
- `next.reward` = reward for taking action at time t (from previous cache entry)
- `next.obs` = state at time t+1
- `next.done` = whether t+1 is terminal

---

## Phase 4: How The N-Step Transform Processes Transitions

**File: `resfit/rl_finetuning/utils/rb_transforms.py`**

When `rb.add(transition)` is called, the `MultiStepTransform._inv_call()` fires:

```python
def _inv_call(self, tensordict):
    total_cat = self._append_tensordict(tensordict)  # Append to internal buffer
    if total_cat.shape[-1] >= self.n_steps:          # Wait until we have n_steps transitions
        out = _multi_step_func(total_cat, ...)       # Compute n-step returns
        return out[..., -self.n_steps]               # Return the processed transition
    return None  # Not enough transitions yet, hold in buffer
```

### The Internal Buffer

The transform keeps an internal buffer of the last `n_steps` transitions. Each new transition is concatenated with this buffer. Only when we have at least `n_steps` transitions does the transform emit a processed transition.

```python
def _append_tensordict(self, data):
    if self._buffer is None:
        total_cat = data
        self._buffer = data[..., -self.n_steps :].copy()
    else:
        total_cat = torch.cat([self._buffer, data], -1)
        self._buffer = total_cat[..., -self.n_steps :].copy()
    return total_cat
```

### The Core N-Step Computation

**File: `resfit/rl_finetuning/utils/rb_transforms.py`, lines 220–295**

```python
def _multi_step_func(tensordict, *, done_key, done_keys, reward_keys, mask_key, n_steps, gamma):
    n_steps = n_steps - 1  # Convert from "look ahead n" to "index offset n-1"
    done = tensordict.get(("next", done_key))
    
    # --- Step A: Sum rewards with gamma decay ---
    # Uses a 1D convolution with filter [1, γ, γ², ..., γ^(n-1)]
    # This computes R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... for each t
    summed_reward, time_to_obs = _get_reward(gamma, reward, done, n_steps)
    
    # --- Step B: Gather the future observation ---
    # Instead of next_obs being s_{t+1}, replace it with s_{t+n}
    idx_to_gather = torch.arange(T) + time_to_obs
    tensordict_gather = tensordict.get("next").gather(-1, idx_to_gather)
    tensordict.get("next").update(tensordict_gather)
    
    # --- Step C: Set the gamma field ---
    # γ^(actual_steps) — may be less than γ^n if episode ended early
    tensordict.set("gamma", gamma ** (time_to_obs + 1))
    
    # --- Step D: Set the nonterminal field ---
    future_done = done.gather(-1, idx_to_gather).squeeze(-1)
    nonterminal = (time_to_obs == n_steps) & (~future_done)
    tensordict.set("nonterminal", nonterminal)
```

### What `nonterminal` Means

```python
nonterminal = (time_to_obs == n_steps) & (~future_done)
```

`nonterminal = True` only when BOTH:
1. We got the full n-step lookahead (episode didn't end early)
2. The state we jumped to is NOT terminal

If `nonterminal = False`, it means:
- The episode ended somewhere within the n-step window
- So we should NOT bootstrap (no Q-target for the next state)
- The target becomes just the summed reward: $y = R^{(n)}$

### What A Processed Transition Looks Like After N-Step Transform

```
{
  "obs":         { state, base_action, images }    ← observation at time t (UNCHANGED)
  "action":      scaled_action_at_t                ← action taken at t (UNCHANGED)
  "next": {
    "obs":       { state, base_action, images }    ← observation at time t+n (CHANGED!)
    "reward":    R^(n) = Σ γ^k * r_{t+k}          ← n-step discounted reward (CHANGED!)
    "done":      ...                               ← (not used directly anymore)
  }
  "gamma":       γ^n or γ^k if episode ended      ← NEW FIELD
  "nonterminal": True/False                        ← NEW FIELD (this is the 1-done)
}
```

---

## Phase 5: Warmup — Fill Online Buffer

**File: `resfit/rl_finetuning/scripts/train_residual_td3.py`, lines 760–830**

Before any learning happens, the code collects `learning_starts` (default 10,000) transitions from the environment using the base policy + noise:

```python
while len(online_rb) < cfg.algo.learning_starts:
    if cfg.algo.use_base_policy_for_warmup:
        # residual = uniform noise * scale (base policy still runs underneath)
        rand_actions = (torch.rand((num_envs, action_dim)) * 2 - 1) * random_action_noise_scale
    
    next_obs, reward, terminated, truncated, info = env.step(rand_actions)
    done = terminated | truncated
    
    # Store the COMBINED action (base + residual) that was actually executed
    combined_action = info["scaled_action"]
    _add_transitions_to_buffer(obs=obs, next_obs=next_obs, actions=combined_action, 
                                reward=reward, done=done, ...)
    obs = next_obs
```

**Key:** The environment wrapper (`BasePolicyVecEnvWrapper`) always runs the frozen BC policy internally. Whatever you pass as the "action" to `env.step()` is treated as the **residual**. The wrapper does:
```
combined = base_policy(obs) + residual
env_action = unscale(combined)
```

So during warmup, the environment executes `base_policy + noise`, not pure random.

Each transition added here also goes through `MultiStepTransform` — same n-step processing as the offline buffer.

---

## Phase 6: Critic Warmup (No Actor Updates)

**File: `resfit/rl_finetuning/scripts/train_residual_td3.py`, lines 940–1000**

After the online buffer is filled, the critic is trained alone for `critic_warmup_steps` (default 10,000) gradient updates:

```python
for i in range(cfg.algo.critic_warmup_steps):
    # Sample mixed batch: 128 from online + 128 from offline = 256 total
    online_batch = online_rb.sample(online_batch_size)     # 128 transitions
    offline_batch = offline_rb.sample(offline_batch_size)   # 128 transitions
    batch = torch.cat([online_batch, offline_batch], dim=0) # 256 transitions
    
    # ONLY update critic, NOT actor
    metrics = agent.update(batch, stddev=0.0, update_actor=False)
```

**Why:** The critic needs decent Q-value estimates before the actor starts using Q-gradients. If the actor starts improving before the critic is calibrated, it follows garbage gradients.

---

## Phase 7: Main Training Loop (Step By Step)

**File: `resfit/rl_finetuning/scripts/train_residual_td3.py`, lines 1000–1250**

After all warmup phases, the main loop runs. **Each iteration of this loop does exactly this:**

### Step 1: Collect ONE Environment Step

```python
while global_step <= cfg.algo.total_timesteps:
    # (1) Agent picks an action using current actor + exploration noise
    with torch.no_grad():
        stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
        action = agent.act(obs, eval_mode=False, stddev=stddev)
    
    # (2) Environment executes: base_policy(obs) + action, returns next state
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated
    
    # (3) Store transition in online buffer (goes through n-step transform)
    combined_action = info["scaled_action"]
    _add_transitions_to_buffer(obs=obs, next_obs=next_obs, actions=combined_action,
                                reward=reward, done=done, ...)
    obs = next_obs
    global_step += 1
```

**This is 1 environment step.** Not a full episode. The loop does 1 step, then does gradient updates, then does 1 more step, etc.

### Step 2: Do UTD Gradient Updates

```python
    # (4) UTD gradient updates (default: 4 updates per env step)
    i = 0
    actor_update_cadence = num_updates_per_iteration // actor_updates_per_iteration  # 4 // 1 = 4
    
    while i < cfg.algo.num_updates_per_iteration:  # 4 times
        # Sample 128 online + 128 offline = 256 transitions
        online_batch = online_rb.sample(online_batch_size)
        offline_batch = offline_rb.sample(offline_batch_size)
        batch = torch.cat([online_batch, offline_batch], dim=0)
        
        # Update actor only on the last of every actor_update_cadence iterations
        update_actor = (i + 1) % actor_update_cadence == 0  # True when i=3
        
        metrics = agent.update(batch, stddev, update_actor)
        i += 1
```

**So for UTD=4 and actor_updates_per_iteration=1:**
- Critic is updated **4 times** per env step
- Actor is updated **1 time** per env step (on the 4th critic update)
- Each update uses a fresh random batch of 256 transitions (128 online + 128 offline)

---

## Phase 8: The Gradient Update (UTD, Critic, Actor)

**File: `resfit/rl_finetuning/off_policy/rl/q_agent.py`, lines 630–680**

When `agent.update(batch, stddev, update_actor)` is called:

### 8.1 Unpack The Batch

```python
obs = batch["obs"]                          # Current observations (from time t)
action = batch["action"]                    # Action taken at time t
reward = batch[("next", "reward")]          # N-step summed reward: R^(n) = Σ γ^k * r_{t+k}
discount = batch["gamma"]                   # γ^n (or less if episode ended early)
next_nonterminal = batch["nonterminal"]     # Can we bootstrap? (the 1-done equivalent)
next_obs = batch[("next", "obs")]           # Observation at time t+n (NOT t+1!)
```

### 8.2 Compute Effective Discount (The 1-done)

```python
# THIS IS WHERE (1 - done) HAPPENS:
effective_discount = discount * next_nonterminal
```

When `nonterminal = False` (episode ended within n steps):
- `effective_discount = γ^n * 0 = 0`
- The Q-target will be just the reward (no bootstrapping)

When `nonterminal = True` (episode still running at t+n):
- `effective_discount = γ^n * 1 = γ^n`
- We bootstrap from Q(s_{t+n}, a_{t+n})

### 8.3 Encode Observations

```python
obs["feat"] = self._encode(obs, augment=True)       # Run images through ViT encoder
next_obs["feat"] = self._encode(next_obs, augment=True)  # Same for next obs
```

### 8.4 Compute Bellman Target (Same For All Loss Types)

```python
with torch.no_grad():
    # Actor target predicts next action
    next_residual_action = self._act_default(obs=next_obs, use_target=True, stddev=stddev)
    next_action = torch.clamp(next_obs["observation.base_action"] + next_residual_action, -1.0, 1.0)
    
    # Target critic predicts Q(s_{t+n}, a_{t+n})
    target_all = self.critic_target.q_value(next_obs["feat"], next_obs["observation.state"], next_action)
    target_q_min = target_all.squeeze(-1)  # Min over random 2 heads
    
    # THE BELLMAN BACKUP:
    target_q = (reward + (effective_discount * target_q_min)).detach()
```

**In math:**

$$y = R^{(n)} + \gamma^n \cdot \mathbf{1}_{\text{nonterminal}} \cdot \min_{j \in \text{random 2 heads}} Q^{(j)}_{\text{target}}(s_{t+n}, a_{t+n})$$

### 8.5 Compute Critic Loss (MSE branch)

```python
# All 10 critic heads predict Q for this (s, a)
q_all = self.critic(obs["feat"], obs["observation.state"], action).squeeze(-1)  # [10, 256]

# Mean absolute TD error across heads
td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [256]

# Square and average over batch
critic_loss = (td_errors**2).mean()  # scalar
```

### 8.6 Backprop Critic

```python
self.encoder_opt.zero_grad()
self.critic_opt.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(self.encoders.parameters(), critic_grad_clip_norm)
torch.nn.utils.clip_grad_norm_(self.critic.parameters(), critic_grad_clip_norm)
self.encoder_opt.step()
self.critic_opt.step()
```

### 8.7 Soft Update Target Critic

After EVERY critic update:
```python
utils.soft_update_params(self.critic, self.critic_target, tau=0.005)
# target_params = 0.005 * online_params + 0.995 * target_params
```

### 8.8 Update Actor (Only On The 4th Iteration)

```python
if update_actor:
    obs["feat"] = obs["feat"].detach()  # Don't backprop into encoder from actor
    
    # Actor predicts residual
    residual = actor(obs, stddev=0.0)
    
    # Combine with base
    combined = clamp(obs["base_action"] + residual, -1, 1)
    
    # Actor loss = negative Q (maximize Q)
    Q = critic.q_value_for_policy(obs["feat"], obs["state"], combined)  # Mean of all 10 heads
    actor_loss = -Q.mean()
    
    actor_loss.backward()
    clip_grad_norm_(actor.parameters(), actor_grad_clip_norm)
    actor_opt.step()
    
    # Soft update target actor
    soft_update_params(self.actor, self.actor_target, tau=0.005)
```

---

## The Done Flag Problem You Asked About

### The Question

You see multiple `done=True` flags near the end of each episode in the dataset (e.g., the last 4-6 frames all have `next.done=True`). You asked: does this break the critic?

### The Answer: It Does NOT Break The Critic

Here's why, step by step:

### How Transitions Are Built From The Dataset

Look at `_populate_offline_buffer()` again:

```python
if ep_idx in episode_cache:
    prev_obs = episode_cache[ep_idx]["obs"]
    prev_action_scaled = episode_cache[ep_idx]["action"]
    prev_reward = episode_cache[ep_idx]["reward"]
    transition = TensorDict({
        "obs": prev_obs,
        "action": prev_action_scaled,
        "next": {
            "obs": curr_obs,
            "done": torch.tensor(done_flag, dtype=torch.bool),  # ← THIS done
            "reward": torch.tensor(prev_reward, dtype=torch.float32),
        },
    })
    rb.add(transition)

episode_cache[ep_idx] = {"obs": curr_obs, "action": gt_action_scaled, "reward": step_reward, "done": done_flag}
```

The `done` in the transition is `sample["next.done"]` of the CURRENT frame (the one being processed right now), NOT the previous frame.

So if an episode has frames 0, 1, 2, ..., 97, 98, 99 and frames 95-99 all have `next.done=True`:

| Transition | obs_t | action_t | next.obs | next.done | next.reward |
|---|---|---|---|---|---|
| t=93→94 | s_93 | a_93 | s_94 | False | r_93 |
| t=94→95 | s_94 | a_94 | s_95 | **True** | r_94 |
| t=95→96 | s_95 | a_95 | s_96 | **True** | r_95 |
| t=96→97 | s_96 | a_96 | s_97 | **True** | r_96 |
| t=97→98 | s_97 | a_97 | s_98 | **True** | r_97 |
| t=98→99 | s_98 | a_98 | s_99 | **True** | r_98 |

### What The N-Step Transform Does With These

For N_STEP=3 and γ=0.995, consider transition starting at t=93:

**Input stream (done flags):**
```
t:    93    94    95    96    97    98    99
done: False True  True  True  True  True  True
```

The `_get_reward` function uses done flags to **separate trajectories**. When it sees `done=True` at t=94, it knows the episode boundary is there. The n-step lookahead STOPS at the boundary.

So for transition at t=93:
- It can only look 1 step ahead (episode ends at t=94)
- `time_to_obs = 1` (not the full n_steps=2)
- `gamma = γ^(1+1) = γ^2 = 0.990`
- `nonterminal = (time_to_obs == n_steps) & (~future_done) = (1 == 2) & ... = False`

**Result:** `nonterminal = False` → no bootstrapping → target is just the reward

For transition at t=94 (where done=True at this step already):
- The reward sum is just r_94 (the next step is already terminal)
- `nonterminal = False`

**This is CORRECT behavior.** Once the episode is done, we don't bootstrap. The critic target is just the reward.

### But What About The REPEATED Dones?

The multiple `done=True` flags (at t=95, 96, 97, 98, 99) create transitions that look like they're all terminal. The n-step transform handles this by:

1. Using `done_cumsum` to split into separate "trajectories"
2. Each transition with `done=True` in its future gets `nonterminal=False`
3. The reward sums are computed correctly within trajectory boundaries

**So for those trailing transitions (t=95 onwards), they all get:**
- `nonterminal = False` (no bootstrapping)
- Target = just the summed reward within the episode
- The critic learns: "at these states, the value is just the remaining reward"

### Why It Does NOT Break The Critic

The concern was: "if done=True appears 5 times, does the critic see conflicting targets?"

No, because:
1. Each transition is independent — the critic just sees `(state, action) → target_q`
2. All those terminal transitions correctly have `nonterminal=False`
3. Their targets are just the immediate rewards (no bootstrapping corruption)
4. The critic learns that those states near the end have predictable value = remaining reward

The only scenario that would "break" the critic is if `nonterminal=True` was set incorrectly for a terminal state — then the critic would bootstrap from a state that doesn't exist in the same episode. But the n-step transform prevents this.

---

## Complete Numerical Example — Full 10-Step Episode

### Setup
- Episode length: 10 transitions (indices 0 through 9)
- n_steps = 3, γ = 0.995
- Dense rewards: `[0, 0.1, 0.2, 0.3, 0.4, 0.55, 1, 1, 1, 1]`
- Done flags: `[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]`
- (reward[t] is the reward received AT step t, done[t] means the state reached at t is terminal)

The transitions in the buffer BEFORE n-step transform:

| Index | obs | action | next.obs | next.reward | next.done |
|-------|-----|--------|----------|-------------|-----------|
| 0 | s₀ | a₀ | s₁ | r₀ = 0.0 | False |
| 1 | s₁ | a₁ | s₂ | r₁ = 0.1 | False |
| 2 | s₂ | a₂ | s₃ | r₂ = 0.2 | False |
| 3 | s₃ | a₃ | s₄ | r₃ = 0.3 | False |
| 4 | s₄ | a₄ | s₅ | r₄ = 0.4 | False |
| 5 | s₅ | a₅ | s₆ | r₅ = 0.55 | False |
| 6 | s₆ | a₆ | s₇ | r₆ = 1.0 | **True** |
| 7 | s₇ | a₇ | s₈ | r₇ = 1.0 | **True** |
| 8 | s₈ | a₈ | s₉ | r₈ = 1.0 | **True** |
| 9 | s₉ | a₉ | s₁₀ | r₉ = 1.0 | **True** |

### Step 1: How `_get_reward` Splits Into Trajectories

The function first computes `done_cumsum`:
```
done:         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
cumsum:       [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
shifted:      [0, 0, 0, 0, 0, 0, 0, 1, 2, 3]   (prepend 0, drop last)
```

This creates `num_traj = max(shifted) + 1 = 4` trajectory masks:
```
traj_0 (shifted==0): [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  ← main trajectory (steps 0-6)
traj_1 (shifted==1): [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  ← step 7 alone
traj_2 (shifted==2): [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  ← step 8 alone
traj_3 (shifted==3): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  ← step 9 alone
```

**Key insight:** Each `done=True` boundary starts a NEW trajectory for subsequent transitions. Steps 7, 8, 9 each become their own 1-step "trajectory" because the done at step 6 already ended the main trajectory.

### Step 2: Reward Convolution Per Trajectory

The convolution filter for n_steps=3 is: `[1, γ, γ²] = [1, 0.995, 0.990025]`

**Trajectory 0** (the main one, steps 0-6):
```
rewards_traj0: [0, 0.1, 0.2, 0.3, 0.4, 0.55, 1, 0, 0, 0] (padded with 2 zeros on right)
                                                             → [0, 0.1, 0.2, 0.3, 0.4, 0.55, 1, 0, 0, 0, 0, 0]
```

Convolution at each position (sliding window of [1, 0.995, 0.990]):
```
pos 0: 0×1 + 0.1×0.995 + 0.2×0.990 = 0 + 0.0995 + 0.198 = 0.2975
pos 1: 0.1×1 + 0.2×0.995 + 0.3×0.990 = 0.1 + 0.199 + 0.297 = 0.596
pos 2: 0.2×1 + 0.3×0.995 + 0.4×0.990 = 0.2 + 0.2985 + 0.396 = 0.8945
pos 3: 0.3×1 + 0.4×0.995 + 0.55×0.990 = 0.3 + 0.398 + 0.5445 = 1.2425
pos 4: 0.4×1 + 0.55×0.995 + 1.0×0.990 = 0.4 + 0.54725 + 0.990 = 1.93725
pos 5: 0.55×1 + 1.0×0.995 + 0×0.990 = 0.55 + 0.995 + 0 = 1.545
pos 6: 1.0×1 + 0×0.995 + 0×0.990 = 1.0
```

**WAIT** — the padding zeros don't just mean "no reward". The trajectory ENDS at the done boundary. So for pos 5 and 6, the convolution only sums rewards that are WITHIN trajectory 0. Since traj_0 mask covers steps 0-6, rewards at steps 7+ are zeroed by the mask.

So the summed rewards for traj_0 = `[0.2975, 0.596, 0.8945, 1.2425, 1.93725, 1.545, 1.0, 0, 0, 0]`

**Trajectory 1** (step 7 only):
```
rewards_traj1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```
Convolution: only pos 7 has a nonzero value.
```
pos 7: 1.0×1 + 0×0.995 + 0×0.990 = 1.0
```
So summed_rewards_traj1 = `[0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0]`

**Trajectory 2** (step 8 only):
```
pos 8: 1.0×1 = 1.0
```

**Trajectory 3** (step 9 only):
```
pos 9: 1.0×1 = 1.0
```

**Final summed reward** (sum across all trajectories):
```
summed_reward = [0.2975, 0.596, 0.8945, 1.2425, 1.93725, 1.545, 1.0, 1.0, 1.0, 1.0]
```

### Step 3: Compute `time_to_obs`

`time_to_obs` tells us: "how many steps forward does the next_obs jump?"

The formula:
```python
time_to_obs = traj_ids.flip(-1).cumsum(-1).clamp_max(max_steps + 1).flip(-1) * traj_ids
time_to_obs = time_to_obs.sum(0) - 1
```

For traj_0 (mask = `[1,1,1,1,1,1,1,0,0,0]`):
```
flip:           [0,0,0,1,1,1,1,1,1,1]
cumsum:         [0,0,0,1,2,3,3,3,3,3]  (clamp_max = max_steps+1 = 3)
flip back:      [3,3,3,3,3,3,1,0,0,0]  WAIT let me redo this carefully
```

Actually let me redo. `max_steps = n_steps - 1 = 2`. So `clamp_max = max_steps + 1 = 3`.

```
traj_0 mask:    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
flip:           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
cumsum:         [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
clamp_max(3):   [0, 0, 0, 1, 2, 3, 3, 3, 3, 3]
flip back:      [3, 3, 3, 3, 3, 3, 3, 0, 0, 0]  
× traj_0 mask:  [3, 3, 3, 3, 3, 3, 3, 0, 0, 0]
```

Hmm wait, let me reconsider. The traj mask is `[1,1,1,1,1,1,1,0,0,0]` — steps 0 through 6.

```
flip mask:       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
cumsum:          [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
clamp_max(3):    [0, 0, 0, 1, 2, 3, 3, 3, 3, 3]
flip back:       [3, 3, 3, 3, 3, 3, 1, 0, 0, 0]
× mask:          [3, 3, 3, 3, 3, 3, 1, 0, 0, 0]
```

Wait I'm confusing myself. Let me be very careful:

Mask for traj_0: positions are `[T₀, T₁, T₂, T₃, T₄, T₅, T₆, T₇, T₈, T₉]` = `[1,1,1,1,1,1,1,0,0,0]`

`.flip(-1)` reverses the last dimension:
```
[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

`.cumsum(-1)`:
```
[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
```

`.clamp_max(max_steps + 1)` where max_steps = n_steps - 1 = 2, so clamp to 3:
```
[0, 0, 0, 1, 2, 3, 3, 3, 3, 3]
```

`.flip(-1)` reverses again:
```
[3, 3, 3, 3, 3, 2, 1, 0, 0, 0]
```

`× traj_0_mask`:
```
[3, 3, 3, 3, 3, 2, 1, 0, 0, 0]
```

For traj_1 (mask = `[0,0,0,0,0,0,0,1,0,0]`):
```
flip:           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
cumsum:         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
clamp_max(3):   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
flip:           [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
× mask:         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

For traj_2 (mask = `[0,0,0,0,0,0,0,0,1,0]`):
```
× mask:         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
```

For traj_3 (mask = `[0,0,0,0,0,0,0,0,0,1]`):
```
× mask:         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

**Sum across trajectories:**
```
[3, 3, 3, 3, 3, 2, 1, 1, 1, 1]
```

**Subtract 1:**
```
time_to_obs = [2, 2, 2, 2, 2, 1, 0, 0, 0, 0]
```

### Step 4: Compute `idx_to_gather`

```python
idx_to_gather = arange(T) + time_to_obs
```
```
arange(10):   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
time_to_obs:  [2, 2, 2, 2, 2, 1, 0, 0, 0, 0]
idx_to_gather:[2, 3, 4, 5, 6, 6, 6, 7, 8, 9]
```

This means:
- Transition 0's `next.obs` will be replaced with `next.obs[2]` = s₃
- Transition 1's `next.obs` will be replaced with `next.obs[3]` = s₄
- Transition 2's `next.obs` → s₅
- Transition 3's `next.obs` → s₆
- Transition 4's `next.obs` → s₇
- Transition 5's `next.obs` → s₇ (only 1 step ahead because done at 6)
- Transition 6's `next.obs` → s₇ (stays same — 0 steps ahead, already at boundary)
- Transition 7's `next.obs` → s₈ (stays same)
- Transition 8's `next.obs` → s₉ (stays same)
- Transition 9's `next.obs` → s₁₀ (stays same)

### Step 5: Compute `gamma` Field

```python
tensordict.set("gamma", gamma ** (time_to_obs + 1))
```
```
time_to_obs + 1: [3, 3, 3, 3, 3, 2, 1, 1, 1, 1]
gamma^(t+1):     [γ³, γ³, γ³, γ³, γ³, γ², γ¹, γ¹, γ¹, γ¹]
                 [0.985, 0.985, 0.985, 0.985, 0.985, 0.990, 0.995, 0.995, 0.995, 0.995]
```

### Step 6: Compute `nonterminal`

```python
future_done = done.gather(-1, idx_to_gather)
nonterminal = (time_to_obs == n_steps) & (~future_done)
```

Remember: `n_steps` inside the function is `n_steps - 1 = 2` (it was decremented at the start).

```
done:           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
idx_to_gather:  [2, 3, 4, 5, 6, 6, 6, 7, 8, 9]
future_done:    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  (done at those gathered positions)

time_to_obs:    [2, 2, 2, 2, 2, 1, 0, 0, 0, 0]
time_to_obs==2: [T, T, T, T, T, F, F, F, F, F]
~future_done:   [T, T, T, T, F, F, F, F, F, F]

nonterminal:    [T, T, T, T, F, F, F, F, F, F]
```

### Final Result: All 10 Processed Transitions

| Idx | obs | action | next.obs | next.reward (summed) | gamma | nonterminal |
|-----|-----|--------|----------|---------------------|-------|-------------|
| 0 | s₀ | a₀ | **s₃** | 0 + 0.995×0.1 + 0.990×0.2 = **0.2975** | 0.995³ = **0.985** | **True** |
| 1 | s₁ | a₁ | **s₄** | 0.1 + 0.995×0.2 + 0.990×0.3 = **0.596** | **0.985** | **True** |
| 2 | s₂ | a₂ | **s₅** | 0.2 + 0.995×0.3 + 0.990×0.4 = **0.8945** | **0.985** | **True** |
| 3 | s₃ | a₃ | **s₆** | 0.3 + 0.995×0.4 + 0.990×0.55 = **1.2425** | **0.985** | **True** |
| 4 | s₄ | a₄ | **s₇** | 0.4 + 0.995×0.55 + 0.990×1.0 = **1.937** | **0.985** | **False** ❌ |
| 5 | s₅ | a₅ | **s₇** | 0.55 + 0.995×1.0 = **1.545** | 0.995² = **0.990** | **False** ❌ |
| 6 | s₆ | a₆ | **s₇** | **1.0** | 0.995¹ = **0.995** | **False** ❌ |
| 7 | s₇ | a₇ | **s₈** | **1.0** | **0.995** | **False** ❌ |
| 8 | s₈ | a₈ | **s₉** | **1.0** | **0.995** | **False** ❌ |
| 9 | s₉ | a₉ | **s₁₀** | **1.0** | **0.995** | **False** ❌ |

### What The Critic Sees At Training Time

**For transition 0 (nonterminal=True — bootstrapping happens):**
```python
effective_discount = gamma * nonterminal = 0.985 * 1.0 = 0.985
target_q = 0.2975 + 0.985 * Q_target(s₃, π(s₃))
```
The critic bootstraps from the value of s₃.

**For transition 3 (nonterminal=True — bootstrapping happens):**
```python
effective_discount = 0.985 * 1.0 = 0.985
target_q = 1.2425 + 0.985 * Q_target(s₆, π(s₆))
```

**For transition 4 (nonterminal=False — no bootstrapping):**
```python
effective_discount = 0.985 * 0.0 = 0.0
target_q = 1.937 + 0.0 * Q_target(s₇, π(s₇)) = 1.937
```
The episode ended within the 3-step window (done=True at step 6).
The critic learns: "from s₄ taking a₄, the total discounted return is 1.937"

**For transition 5 (nonterminal=False — truncated lookahead):**
```python
effective_discount = 0.990 * 0.0 = 0.0
target_q = 1.545
```
Could only look 2 steps ahead instead of 3 (done at step 6 cut it short).
Note gamma=0.990=γ² because only 2 steps of lookahead were used.

**For transitions 6-9 (nonterminal=False — these are AFTER the first done):**
```python
effective_discount = 0.995 * 0.0 = 0.0
target_q = 1.0  (just the immediate reward, no future)
```
These are all trivially terminal — the episode already ended.

### Why `time_to_obs` Varies Near The Done Boundary

- **Transitions 0-4:** `time_to_obs = 2` (full 3-step lookahead), gamma = γ³
- **Transition 5:** `time_to_obs = 1` (only 2 steps until done), gamma = γ²  
- **Transition 6:** `time_to_obs = 0` (already AT the done), gamma = γ¹
- **Transitions 7-9:** `time_to_obs = 0` (each is its own 1-step trajectory), gamma = γ¹

The gamma ADAPTS to how far ahead we actually looked. Near the end of an episode, we can't look the full n steps, so gamma shrinks accordingly.

---

## Summary: The Complete Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STARTUP                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Download frozen BC policy from WandB                                      │
│ 2. Download HuggingFace dataset                                             │
│ 3. Compute ActionScaler (min/max from dataset stats)                        │
│ 4. Compute StateStandardizer (mean/std from dataset stats)                  │
│ 5. Create online_rb + offline_rb (both with MultiStepTransform(n=3, γ=.995))│
│ 6. Populate offline_rb: iterate dataset → normalize → pair frames →         │
│    create (s,a,r,s',done) transitions → rb.add() triggers n-step transform │
│ 7. Create environment (wrapped with base_policy)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ WARMUP PHASE 1: Fill Online Buffer (10,000 transitions)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Loop 10,000 times:                                                           │
│   residual_noise = uniform(-0.2, 0.2)                                       │
│   env executes: base_policy(obs) + residual_noise                           │
│   store transition in online_rb (n-step transform applied)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ WARMUP PHASE 2: Critic-Only Training (10,000 gradient updates)              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Loop 10,000 times:                                                           │
│   batch = 128 from online_rb + 128 from offline_rb                          │
│   update critic only (update_actor=False)                                   │
│   soft update target critic                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ MAIN TRAINING LOOP (300,000 env steps)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ For each env step:                                                           │
│                                                                              │
│   ┌──── (A) Collect 1 env step ────┐                                       │
│   │ residual = actor(obs) + noise   │                                       │
│   │ env does: base_policy + residual│                                       │
│   │ get (next_obs, reward, done)    │                                       │
│   │ store in online_rb              │                                       │
│   └─────────────────────────────────┘                                       │
│                                                                              │
│   ┌──── (B) 4 gradient updates (UTD=4) ────┐                               │
│   │                                          │                               │
│   │ Iteration 1: critic update only          │                               │
│   │   batch = sample 128 online + 128 offline│                               │
│   │   compute target_q (Bellman backup)      │                               │
│   │   critic_loss = MSE(Q, target_q)         │                               │
│   │   backprop critic + encoder              │                               │
│   │   soft update target critic (τ=0.005)    │                               │
│   │                                          │                               │
│   │ Iteration 2: critic update only          │                               │
│   │   (same as above, new random batch)      │                               │
│   │                                          │                               │
│   │ Iteration 3: critic update only          │                               │
│   │   (same as above, new random batch)      │                               │
│   │                                          │                               │
│   │ Iteration 4: critic + actor update       │                               │
│   │   critic update (same as above)          │                               │
│   │   actor_loss = -mean(Q(s, base+residual))│                               │
│   │   backprop actor only                    │                               │
│   │   soft update target actor (τ=0.005)     │                               │
│   └──────────────────────────────────────────┘                               │
│                                                                              │
│   global_step += 1                                                           │
│   Every 5000 steps: run evaluation (50 episodes, no noise)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Your Questions Answered Directly

### Q1: "For critic warmup do we only use offline buffer, or 50-50?"

**50-50.** The critic warmup uses the SAME batching as main training:

```python
# From _run_critic_warmup(), line ~950:
online_batch = online_rb.sample(online_batch_size)     # 128 online
offline_batch = offline_rb.sample(offline_batch_size)   # 128 offline
batch = torch.cat([online_batch, offline_batch], dim=0) # 256 total
```

The online buffer was ALREADY filled during Warmup Phase 1 (the random exploration phase with base_policy + noise). So by the time critic warmup starts, BOTH buffers are populated.

Timeline:
1. Fill online buffer with 10,000 transitions (base_policy + random noise)
2. THEN do 10,000 critic-only gradient updates sampling 128 from each buffer

---

### Q2: "What do you mean by fresh random batch? Are we doing epochs? Does a transition get seen only once?"

**NO epochs. Transitions get seen MULTIPLE times.**

There are no epochs here. This is NOT supervised learning. The replay buffer is a pool of transitions that we sample from **randomly with replacement**.

- The online buffer has ~10,000 transitions initially (grows to 80,000 over training)
- The offline buffer has ~5,000 transitions (fixed, never grows)
- Each gradient update samples 128 from each buffer **randomly** (uniform or prioritized)
- The same transition CAN be sampled again in the next batch, or 10 batches later, or never again

**Example with numbers:**
- You do 300,000 env steps × 4 updates per step = 1,200,000 gradient updates
- Each update uses 128 online transitions
- Total online samples drawn: 1,200,000 × 128 = 153,600,000 samples
- Online buffer holds 80,000 transitions max
- So on average, each transition is sampled **~1,920 times** over training

This is completely standard in off-policy RL. The whole point of a replay buffer is to reuse data many times. That's what "off-policy" means — you learn from old data, not just the latest transition.

---

### Q3: "How does the n-step transform work for the ONLINE buffer? We only do 1 env step — we don't have the future yet!"

**Excellent question. The answer: the transform BUFFERS transitions internally and only emits them AFTER it has n_steps worth of data.**

Here's what happens step by step:

```python
# Main loop iteration 1: env step produces transition T₀
online_rb.add(T₀)
# Inside MultiStepTransform._inv_call():
#   self._buffer = [T₀]
#   total_cat.shape[-1] = 1 < n_steps(3) → return None
#   NOTHING is actually stored in the buffer yet!

# Main loop iteration 2: env step produces transition T₁  
online_rb.add(T₁)
# Inside:
#   self._buffer = [T₀, T₁]
#   total_cat.shape[-1] = 2 < 3 → return None
#   STILL nothing stored!

# Main loop iteration 3: env step produces transition T₂
online_rb.add(T₂)
# Inside:
#   total_cat = [T₀, T₁, T₂]
#   total_cat.shape[-1] = 3 >= 3 → PROCESS!
#   out = _multi_step_func([T₀, T₁, T₂], ...)
#   Returns the FIRST transition (T₀) with:
#     - next.obs replaced by T₂'s next.obs (= s₃, which is 3 steps ahead of s₀)
#     - next.reward = r₀ + γ*r₁ + γ²*r₂
#     - gamma = γ³
#     - nonterminal = True/False based on done flags
#   self._buffer = [T₁, T₂]  (keeps last n_steps=3 for next time)
#   → This processed T₀ is ACTUALLY stored in the replay buffer

# Main loop iteration 4: env step produces transition T₃
online_rb.add(T₃)
# Inside:
#   total_cat = [T₁, T₂, T₃]  (buffer + new)
#   Processes T₁ with 3-step lookahead to T₃'s next.obs
#   self._buffer = [T₂, T₃]
```

**So there's a DELAY of (n_steps - 1) = 2 env steps before a transition actually appears in the replay buffer.** The transform waits until it has enough future information.

This is why the code asserts `num_envs == 1`:
```python
assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"
```

With multiple envs, the transitions from different environments would get interleaved and the n-step buffer would mix transitions from different episodes/environments incorrectly.

**What happens at episode boundaries (done=True)?**

When `done=True` arrives, the n-step transform handles it correctly:
- It sees the done flag in the transition sequence
- The reward sum stops at the done boundary (uses the trajectory splitting we showed above)
- The `nonterminal` is set to False
- The `next.obs` points to the terminal state (not beyond it)

So even though we only advance 1 step at a time, the internal buffer accumulates the future steps we need.

---

### Q4: "The done flag explanation was unclear — explain with the full example"

See the [Complete Numerical Example](#complete-numerical-example--full-10-step-episode) above. I rewrote it with your exact numbers (10 steps, dense rewards, done at step 6 onwards) and traced every single computation.

The key takeaways:
1. The FIRST `done=True` (at step 6) creates a trajectory boundary
2. Steps BEFORE the boundary (0-3) get full 3-step lookahead with `nonterminal=True`
3. Steps NEAR the boundary (4-5) get truncated lookahead with `nonterminal=False` (because future_done is True at the gathered position)
4. Steps AT and AFTER the boundary (6-9) each become their own tiny "trajectory", get `nonterminal=False`, and their target is just the immediate reward
5. The repeated `done=True` flags don't cause problems — they just create more single-step trajectories that all correctly get `nonterminal=False`

---

## Key Numbers For Your Lift Config

| What | Value | Source |
|---|---|---|
| Total env steps | 300,000 | `TOTAL_TIMESTEPS` |
| N-step | 3 | `N_STEP` |
| Discount γ | 0.995 | `GAMMA` |
| UTD (updates per env step) | 4 | `UTD` |
| Actor updates per env step | 1 | `ACTOR_UPDATES_PER_ITER` |
| Batch size | 256 | `BATCH_SIZE` |
| Online portion | 128 | 256 × 0.5 |
| Offline portion | 128 | 256 × 0.5 |
| Critic heads | 10 | `NUM_Q_HEADS` |
| Min heads for target | 2 | `MIN_Q_HEADS` |
| Policy gradient | mean of all 10 heads | `POLICY_GRADIENT_TYPE` |
| Learning starts (warmup transitions) | 10,000 | `LEARNING_STARTS` |
| Critic warmup (gradient updates) | 10,000 | `CRITIC_WARMUP` |
| Target soft update τ | 0.005 | `CRITIC_TARGET_TAU` |
| Actor LR | 1e-6 | `ACTOR_LR` |
| Critic LR | 1e-4 | `CRITIC_LR` |
| Horizon | 100 | hardcoded in dexmg.py |
| Reward | Dense (0-0.55 per step, 1.0 on success) | `REWARD_SHAPING=true` |

### Total Gradient Updates Over Training

- Critic updates: 300,000 × 4 = **1,200,000**
- Actor updates: 300,000 × 1 = **300,000**
- Plus 10,000 critic-only warmup updates
- Total critic: **1,210,000** updates
- Total actor: **300,000** updates

### Total Transitions In Buffers

- Online buffer capacity: 80,000 transitions
- Online buffer initially filled: 10,000 transitions (warmup)
- Offline buffer: ~50 episodes × ~100 steps = ~4,950 transitions
- After 80,000 env steps, online buffer is full and starts overwriting oldest

---

## File Reference

| Step | File | Lines |
|---|---|---|
| Load BC policy | `train_residual_td3.py` | 237–259 |
| Load dataset | `train_residual_td3.py` | 263 |
| Action scaler | `train_residual_td3.py` | 266–270 |
| State standardizer | `train_residual_td3.py` | 273–277 |
| Create replay buffers | `train_residual_td3.py` | 419–441, 527 |
| Populate offline buffer | `train_residual_td3.py` | 547–662 |
| N-step transform logic | `rb_transforms.py` | 188–295 |
| Warmup collection | `train_residual_td3.py` | 760–830 |
| Critic warmup | `train_residual_td3.py` | 940–1000 |
| Main loop: env step | `train_residual_td3.py` | 1010–1065 |
| Main loop: gradient updates | `train_residual_td3.py` | 1180–1250 |
| Bellman target | `q_agent.py` | 313–339 |
| MSE critic loss | `q_agent.py` | 380–392 |
| Actor loss | `q_agent.py` | 442–462 |
| Effective discount | `q_agent.py` | 640 |
