# Replay Buffers — Deep Dive

This document explains the replay buffer architecture used in residual RL training: online vs offline buffers, prioritized experience replay (PER), n-step transforms, caching to disk, and prefetching.

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)

---

## Table of Contents

1. [Two Buffers: Online and Offline](#1-two-buffers-online-and-offline)
2. [Buffer Implementation: TensorDictPrioritizedReplayBuffer](#2-buffer-implementation-tensordictprioritizedreplaybuffer)
3. [Online/Offline Mixing](#3-onlineoffline-mixing)
4. [Transition Storage Format](#4-transition-storage-format)
5. [Prioritized Experience Replay (PER)](#5-prioritized-experience-replay-per)
6. [Buffer Caching — Disk and HuggingFace Hub](#6-buffer-caching--disk-and-huggingface-hub)
7. [Prefetching — Background Thread Optimization](#7-prefetching--background-thread-optimization)
8. [Memory Optimization: uint8 Images](#8-memory-optimization-uint8-images)

---

## 1. Two Buffers: Online and Offline

The training uses two separate replay buffers that serve complementary purposes:

### Online Buffer

| Property | Value |
|---|---|
| **Source** | Agent interacting with the environment |
| **Size** | 300,000 transitions |
| **Content** | Real RL experience: obs, combined action, reward, next obs, done |
| **Updates** | Continuously filled during training |
| **Purpose** | Learn from actual environment outcomes |

### Offline Buffer

| Property | Value |
|---|---|
| **Source** | Demonstration dataset (converted to transitions) |
| **Size** | ~num_episodes × avg_episode_length transitions |
| **Content** | Expert demonstrations with base policy actions |
| **Updates** | Populated once before training, never modified |
| **Purpose** | Stabilizing signal — prevents catastrophic forgetting of demonstrated behavior |

### Why Two Buffers?

Pure online RL is sample-inefficient and unstable — the agent needs thousands of episodes to learn even basic skills. By mixing 50% offline demonstration data, the training benefits from:

1. **Better Q-value initialization**: Demonstration transitions show successful trajectories with reward=1.0 at the end, giving the critic meaningful targets from the start
2. **Distribution anchoring**: The offline data keeps the actor grounded near demonstrated behavior
3. **Faster convergence**: The critic learns from both successful demonstrations and the agent's own experience simultaneously

This approach is inspired by **RLPD** (Reinforcement Learning with Prior Data) — hence the config class inheritance from `RLPDAlgoConfig`.

---

## 2. Buffer Implementation: TensorDictPrioritizedReplayBuffer

Both buffers use TorchRL's `TensorDictPrioritizedReplayBuffer`:

```python
online_rb = TensorDictPrioritizedReplayBuffer(
    storage=LazyTensorStorage(max_size=300_000, device="cpu"),
    alpha=0.0,           # 0 = uniform sampling (PER disabled by default)
    beta=0.0,            # 0 = no importance sampling correction
    eps=1e-6,            # Small epsilon to prevent zero priorities
    priority_key="_priority",
    transform=MultiStepTransform(n_steps=5, gamma=0.995),
    pin_memory=True,     # Enable pinned memory for faster CPU→GPU transfer
    prefetch=4,          # Pre-load 4 batches in background threads
    batch_size=128,      # Online portion of batch (256 × 0.5)
)
```

### Key Components

| Component | Purpose |
|---|---|
| `LazyTensorStorage` | Lazily allocates memory — adapts to the first transition's structure |
| `alpha` | PER exponent: 0 = uniform, 0.6 = moderate prioritization |
| `beta` | Importance sampling correction: 0 = none, 0.4 = partial |
| `MultiStepTransform` | Converts 1-step transitions to n-step returns (see [NSTEP_RETURNS.md](NSTEP_RETURNS.md)) |
| `pin_memory` | Locks memory pages for faster async CPU→GPU transfer |
| `prefetch` | Number of batches to prepare in background threads |

---

## 3. Online/Offline Mixing

Each training batch is split 50/50:

```python
offline_fraction = 0.5
batch_size = 256

online_batch_size = 256 * (1 - 0.5) = 128
offline_batch_size = 256 * 0.5 = 128

# During training:
online_batch = online_rb.sample(128)    # 128 from online
offline_batch = offline_rb.sample(128)   # 128 from offline
batch = torch.cat([online_batch, offline_batch], dim=0)  # 256 total
```

The concatenation means the gradient update sees a mix of:
- **128 transitions**: agent's own experience (possibly suboptimal, but real)
- **128 transitions**: expert demonstrations (high quality, but potentially off-distribution)

The critic learns from both, and the actor benefits from a more stable loss landscape.

---

## 4. Transition Storage Format

Each transition is stored as a `TensorDict`:

```python
TensorDict({
    # Current observation
    "obs": TensorDict({
        "observation.images.agentview":        [3, 84, 84]   uint8
        "observation.images.robot0_eye_in_hand": [3, 84, 84] uint8
        "observation.images.robot1_eye_in_hand": [3, 84, 84] uint8
        "observation.state":                   [prop_dim]     float32 (standardized)
        "observation.base_action":             [action_dim]   float32 (scaled)
    }),
    
    # Executed action (base + residual combined, scaled to [-1,1])
    "action": [action_dim]   float32
    
    # Next step (modified by MultiStepTransform to be n-step)
    "next": TensorDict({
        "obs": { ... same structure as above ... },
        "done":   bool,
        "reward": float32   # After n-step: cumulative discounted reward
    }),
    
    # Priority for PER (initially high for new samples)
    "_priority": float32    # 10.0 for new transitions
    
    # Added by MultiStepTransform:
    "gamma":       float32  # γ^n or γ^k if episode ended early
    "nonterminal": bool     # Whether to bootstrap from next obs
})
```

### Terminal Observation Handling

When a `done` flag is True, the environment auto-resets and returns the **new** episode's first observation. But the replay buffer needs the **final** observation from the completed episode.

The wrapper stores this in `info["final_obs"]`, and the `_add_transitions_to_buffer` helper substitutes it:

```python
if done[i] and "final_obs" in info and info["final_obs"][i] is not None:
    next_obs_i = info["final_obs"][i]   # Final obs of completed episode
else:
    next_obs_i = next_obs[i]             # Normal next obs
```

---

## 5. Prioritized Experience Replay (PER)

### Default: Uniform Sampling

By default, `sampling_strategy="uniform"` which sets `alpha=0, beta=0`. This means all transitions are sampled with equal probability — standard FIFO replay buffer behavior with random sampling.

### Optional: PER (`sampling_strategy="prioritized_replay"`)

When enabled (`alpha=0.6, beta=0.4`):

**Idea**: Transitions with higher TD error contain more learning signal — the critic was most "surprised" by them. Sample these more often.

**Priority computation**:

$$p_i = |\delta_i| + \epsilon$$

where $\delta_i = Q(s_i, a_i) - y_i$ is the TD error and $\epsilon = 10^{-6}$ prevents zero priority.

**Sampling probability**:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

- $\alpha = 0$: uniform sampling (all probabilities equal)
- $\alpha = 1$: fully prioritized (highest TD error transitions sampled most)
- $\alpha = 0.6$: moderate prioritization

**Importance sampling correction**:

Prioritized sampling introduces bias (some transitions are over-represented). Correct with weights:

$$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta$$

- $\beta = 0$: no correction (biased but faster learning)
- $\beta = 1$: full correction (unbiased but slower)
- $\beta = 0.4$: partial correction

The weights are applied to the TD error in the critic loss:

```python
if importance_weights is not None:
    weighted_td_errors = td_errors**2 * importance_weights
    critic_loss = weighted_td_errors.mean()
```

### Priority Update Cycle

```
1. New transition added to buffer → priority = 10.0 (high initial priority)
2. Batch sampled from buffer → transitions come with their current priorities
3. Critic computes TD errors for the batch
4. TD errors written back to the batch as updated priorities:
   batch["_priority"] = |T errors|
5. Buffer's priority tree updated:
   online_rb.update_tensordict_priority(batch)
```

---

## 6. Buffer Caching — Disk and HuggingFace Hub

Both buffer populations (offline demonstrations + warmup exploration) are expensive — they require running the base policy and/or environment for thousands of steps. The codebase implements a multi-level caching system:

### Cache Key: SHA-1 Hash of Metadata

```python
cache_meta = {
    "task": "TwoArmBoxCleanup",
    "dataset_name": "ankile/dexmg-two-arm-box-cleanup",
    "num_episodes": 1000,
    "image_keys": ["observation.images.agentview", ...],
    "n_step": 5,
    "gamma": 0.995,
    "base_policy_wandb_id": "dexmg-box-clean-bc/abc123",
    "torchrl_version": "0.7.1",
    "tensordict_version": "0.7.1",
    ...
}
cache_hash = sha1(json.dumps(cache_meta, sort_keys=True))[:8]
# e.g., "a3f2b1c9"
```

Any change in any metadata field produces a different hash → different cache.

### Cache Hierarchy

```
1. Check local disk: ~/.cache/offline_buffer_cache/<hash>/
   ├── Found → load directly (seconds)
   └── Not found ↓

2. Check HuggingFace Hub: HF_OFFLINE_BUFFER_REPO/<hash>.tar.gz
   ├── Found → download + extract to local → load
   └── Not found ↓

3. Populate from scratch:
   └── Process dataset → fill buffer → save to local → upload to HF Hub
```

### What Gets Cached

The cache stores the replay buffer's internal state — the `LazyTensorStorage` contents and the sampler state. Loading from cache restores the buffer to exactly the state it was in when saved, including all transitions and their priorities.

```python
# Save
optimized_replay_buffer_dumps(offline_rb, cache_dir)

# Load
offline_rb.sampler._empty()  # Reset sampler first
optimized_replay_buffer_loads(offline_rb, cache_dir)
```

---

## 7. Prefetching — Background Thread Optimization

```python
prefetch=4  # Number of batches to pre-load
```

When `prefetch=4`, the replay buffer uses background threads to prepare the next 4 batches while the current batch is being used for gradient computation. The data flow:

```
Time ─────────────────────────────────►

GPU:   [train on batch 1] [train on batch 2] [train on batch 3] [train on batch 4]
                │               │                  │                   │
CPU:   [prepare batch 2] [prepare batch 3] [prepare batch 4] [prepare batch 5]
       [prepare batch 3] [prepare batch 4] [prepare batch 5] [prepare batch 6]
       [prepare batch 4] [prepare batch 5] [prepare batch 6] [prepare batch 7]
       [prepare batch 5] [prepare batch 6] [prepare batch 7] [prepare batch 8]
```

Benefits:
- **No waiting**: The GPU never idles waiting for data
- **Pipeline overlap**: CPU data preparation runs concurrently with GPU computation
- **Pin memory**: `pin_memory=True` allocates page-locked memory for faster async CPU→GPU transfer

With UTD=4 (4 gradient updates per env step), prefetching is especially beneficial — all 4 batches can be pre-loaded during the environment step.

---

## 8. Memory Optimization: uint8 Images

Images are stored as **uint8** (8-bit unsigned integers, 0–255) rather than float32 in the replay buffer:

```python
# During buffer insertion:
to_uint8(curr_obs_i, image_keys)  # Converts float32 images to uint8

# During training (in q_agent._encode):
if data.dtype == torch.uint8:
    data = data.float().div_(255.0)  # Convert back to float32 in [0, 1]
```

### Memory Savings

For a single image observation (3 cameras, 84×84×3):

| Format | Per image | Per transition (3 cameras) | 300K buffer |
|---|---|---|---|
| float32 | 84×84×3×4 = 84.7 KB | 254 KB | 72.3 GB |
| uint8 | 84×84×3×1 = 21.2 KB | 63.5 KB | 18.1 GB |

Each transition stores **two** observations (current + next), so the savings are doubled. With uint8: ~36 GB for 300K transitions vs ~145 GB with float32.

The conversion is lossless for pixel values (which are inherently 0–255 integers).

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)
