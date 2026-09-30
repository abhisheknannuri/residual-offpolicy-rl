# Checkpointing, Resume & Buffer Lifecycle

This document covers checkpointing and resume for the **Residual RL training** stage (`train_residual_td3.py`). For BC policy training checkpointing, see [../policies/BC_POLICY_TRAINING.md](../policies/BC_POLICY_TRAINING.md).

---

## Table of Contents

1. [What Gets Checkpointed](#1-what-gets-checkpointed)
2. [Checkpoint Schedule — When & Where](#2-checkpoint-schedule--when--where)
3. [Resuming a Failed Run](#3-resuming-a-failed-run)
4. [What Resume Restores vs What It Doesn't](#4-what-resume-restores-vs-what-it-doesnt)
5. [Buffer Lifecycle — What Survives a Crash](#5-buffer-lifecycle--what-survives-a-crash)
6. [The RL Vision Encoder — Separate from BC](#6-the-rl-vision-encoder--separate-from-bc)
7. [Memory Budget: `buffer_size` Must Match Your RAM](#7-memory-budget-buffer_size-must-match-your-ram)
8. [Known Crash: Matplotlib/Tkinter Thread Conflict](#8-known-crash-matplotlibtkinter-thread-conflict)

---

## 1. What Gets Checkpointed

Every checkpoint saved by `save_checkpoint()` (`resfit/rl_finetuning/utils/checkpoint.py`) contains:

```python
checkpoint_data = {
    "agent_state_dict": agent.state_dict(),    # ALL trainable weights (encoder, actor, critic, targets)
    "global_step": global_step,                # Current environment step
    "optimizer_state_dict": {
        "actor_opt":   agent.actor_opt.state_dict(),    # Actor AdamW state (momentum, variance)
        "critic_opt":  agent.critic_opt.state_dict(),   # Critic AdamW state
        "encoder_opt": agent.encoder_opt.state_dict(),  # Encoder AdamW state
    },
    "scheduler_state_dict": { ... },           # LR scheduler states (if any active)
    "config": OmegaConf.to_container(config),  # Full Hydra config snapshot
    "success_rate": float,                     # Best eval success rate at save time
    "actor_updates": int,                      # Actor update counter (for LR warmup bookkeeping)
}
```

### What `agent.state_dict()` contains

The `QAgent(nn.Module)` checkpoint captures every component:

| Component | What it is | Trainable? | Count |
|---|---|---|---|
| `encoders` | Per-camera MinViT vision encoders | ✅ Yes (default) | 3 (one per camera) |
| `actor` | Residual action MLP | ✅ Yes | 1 |
| `critic` | Q-value ensemble with SpatialEmb | ✅ Yes | 1 (contains 10 Q-heads) |
| `critic_target` | EMA copy of critic | ❌ No (soft-updated) | 1 |
| `actor_target` | EMA copy of actor | ❌ No (soft-updated) | 1 |
| `aug` | RandomShiftsAug (pad=4) | ❌ No params | 1 |

The three AdamW optimizers each have their own momentum/adaptive-second-moment state. These **must** be restored for training to continue smoothly — without them, Adam effectively restarts from cold, causing the learning rate to spike much higher than intended during the initial bias-correction phase.

### What WandB receives

In addition to local saves, model artifacts are pushed to WandB:
- **Timestamped artifact** per `save_freq` steps (e.g., `run_<id>_model_step_10000`)
- **Latest artifact** (overwritten each save — for easy resume download)
- **Best artifact** (overwritten when a new best success rate is achieved)

---

## 2. Checkpoint Schedule — When & Where

### Automatic saves

Checkpoints are saved at three trigger points:

| Trigger | What's saved | Directory | WandB artifact |
|---|---|---|---|
| **Every `save_freq` steps** (default: 10,000) | Full checkpoint | `models/policy_step_<N>/checkpoint.pt` | `run_<id>_model_step_<N>` |
| **Every `save_freq` steps** (same) | Latest (overwritten) | `models/latest/checkpoint.pt` | `run_<id>_latest` (alias: `latest`) |
| **New best eval success rate** | Best model | `models/best/checkpoint.pt` | `run_<id>_best` (alias: `best`) |
| **End of training** (final step) | Final model | `models/final/checkpoint.pt` | `run_<id>_final` (aliases: `final`, `latest`) |

### Directory structure of a run

```
run_2026-02-22_21-00-24_resfit__...__seed1987747100/
├── models/
│   ├── policy_step_10000/
│   │   └── checkpoint.pt        ← Historical snapshot
│   ├── policy_step_20000/
│   │   └── checkpoint.pt
│   ├── ...
│   ├── policy_step_90000/
│   │   └── checkpoint.pt
│   ├── latest/
│   │   └── checkpoint.pt        ← Always the most recent save (for resume)
│   ├── best/
│   │   └── checkpoint.pt        ← Highest eval success rate so far
│   └── final/                   ← Only exists if training completed normally
│       └── checkpoint.pt
└── outputs/
    └── ...                      ← Hydra config dumps, logs
```

### Config parameter

| Parameter | Default | Description |
|---|---|---|
| `save_freq` | 10,000 | Save a checkpoint every N environment steps. Set to 0 to disable. |
| `no_cleanup` | `false` | When `true`, skip deleting the run directory after training finishes (preserve local checkpoints). |

---

## 3. Resuming a Failed Run

### The `resume_ckpt` parameter

If training crashes or is interrupted, you can resume from the latest (or any) checkpoint using the `resume_ckpt` Hydra override:

```bash
python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name residual_td3_coffee_config \
    base_policy.wandb_id=dexmg-bc/<run_id> \
    base_policy.wt_type=best \
    offline_data.num_episodes=100 \
    seed=<same_seed> \
    resume_ckpt=<path_to_checkpoint> \
    wandb.continue_run_id=<wandb_run_id>
```

### What `resume_ckpt` accepts

The parameter accepts either:
- A **directory** containing `checkpoint.pt` (e.g., `run_.../models/latest`) — the script automatically appends `/checkpoint.pt`
- A **direct path** to a `.pt` file (e.g., `run_.../models/policy_step_50000/checkpoint.pt`)

### Concrete example — resuming after a crash at step 90,000

```bash
python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name residual_td3_coffee_config \
    base_policy.wandb_id=dexmg-bc/kpid6s4r \
    base_policy.wt_type=best \
    offline_data.num_episodes=100 \
    seed=1987747100 \
    resume_ckpt=run_2026-02-22_21-00-24_resfit__2026-02-22_21-00-22__TwoArmCoffee_n5_utd4_buf80000_off100ep_lr1e-06__seed1987747100/models/latest \
    wandb.continue_run_id=1wrldnus
```

This will:
1. Re-create the entire pipeline (base policy download, env setup, agent, buffers)
2. Load the checkpoint → restore all weights, optimizers, counters
3. Resume training from step 90,001 instead of step 0
4. Continue logging to the same WandB dashboard

### WandB continuation with `wandb.continue_run_id`

When resuming, set `wandb.continue_run_id` to the crashed run's ID so metrics append to the existing WandB dashboard rather than creating a new run. Without this, you'll get a separate WandB run that starts logging from step 90,000 — which works but creates a confusing split in your dashboard.

The WandB run ID is the 8-character code in the URL: `https://wandb.ai/.../runs/1wrldnus` → `1wrldnus`.

You can find it in the local wandb directory too:
```bash
ls wandb/  # → run-20260222_210023-1wrldnus  (the last part after the dash)
```

### Config parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `resume_ckpt` | `str \| None` | `None` | Path to checkpoint file or directory. When set, agent weights, optimizers, and global_step are restored. |
| `wandb.continue_run_id` | `str \| None` | `None` | WandB run ID to continue logging to. Calls `wandb.init(id=..., resume="allow")`. |

---

## 4. What Resume Restores vs What It Doesn't

### Full restoration (from checkpoint)

| State | Restored? | How |
|---|---|---|
| Agent weights (encoder, actor, critic, targets) | ✅ | `agent.load_state_dict()` |
| Actor optimizer (AdamW momentum + variance) | ✅ | `actor_opt.load_state_dict()` |
| Critic optimizer | ✅ | `critic_opt.load_state_dict()` |
| Encoder optimizer | ✅ | `encoder_opt.load_state_dict()` |
| LR scheduler states | ✅ | If schedulers present in checkpoint |
| `global_step` counter | ✅ | From checkpoint — training loop continues from this step |
| `best_eval_success_rate` | ✅ | From checkpoint `success_rate` field |
| `actor_updates` counter | ✅ | From checkpoint — needed for actor LR warmup schedule |

### NOT restored (re-created from scratch)

| State | What happens on resume | Impact |
|---|---|---|
| Online replay buffer | Re-populated via warmup cache (10K entries) + new collection | Buffer starts with only warmup data; re-fills as training progresses. Transitions collected between the last warmup cache and the crash are lost. |
| Offline replay buffer | Loaded from `offline_buffer_cache/` | ✅ No impact — always loads from cache. |
| Environment RNG state | New random seed from config | Episode boundaries shift slightly. Minimal impact on training. |
| Critic warmup | **Skipped** on resume | The critic was already warmed up before the crash. The script detects `global_step > 0` and skips warmup. |

### Why the online buffer loss is acceptable

When training crashes at step 90,000 with `buffer_size=80,000`, the online buffer held the most recent 80K transitions (steps 10,001–90,000). On resume, this data is lost — the buffer restarts from the cached 10K warmup entries and re-fills during continued training.

This means steps 90,001–100,001 will train with a smaller, mostly-warmup buffer. However:
- The **agent weights** (which encode 90K steps of learning) are fully restored — this is what matters most
- The buffer re-fills quickly (within 80K steps it's back to full capacity)
- The offline buffer (demonstrations) is intact, so 50% of every training batch is still high-quality data
- In practice, the policy continues improving without any visible regression

### Resume flow diagram

```
Resume command (resume_ckpt=.../latest)
    │
    ├─ 1. Download base policy from WandB    ← Same as fresh run
    ├─ 2. Create environments                ← Same as fresh run
    ├─ 3. Build QAgent (random weights)      ← Same as fresh run
    ├─ 4. Populate offline buffer (cached)   ← Loads from offline_buffer_cache/
    ├─ 5. Populate online warmup (cached)    ← Loads from online_buffer_cache/
    ├─ 6. Init WandB (continue_run_id)       ← Appends to existing dashboard
    │
    ├─ 7. ★ load_checkpoint() ★
    │      ├─ Overwrite agent weights         global_step = 90,000
    │      ├─ Overwrite optimizer states      best_eval_success_rate = 0.44
    │      └─ Restore counters                actor_updates = 0
    │
    ├─ 8. Skip critic warmup                 ← Detected: global_step > 0
    │
    └─ 9. Training loop starts at step 90,001
           └─ while global_step <= 500,000: ...
```

---

## 5. Buffer Lifecycle — What Survives a Crash

### Two buffers, two caching strategies

| Buffer | Capacity | Cached when? | Survives crash? |
|---|---|---|---|
| **Offline** (demo transitions) | `num_episodes × avg_ep_len` (e.g., 80,531) | After first population from dataset | ✅ Yes — cached to `offline_buffer_cache/` |
| **Online** (RL interactions) | `buffer_size` (e.g., 80,000) | After warmup only (first 10K steps) | ⚠️ Only warmup data survives |

### Timeline of what gets cached and saved

```
Step 0: Training starts
   │
   ├─ [CACHED] Offline buffer populated from dataset → offline_buffer_cache/<hash>/
   │            (or loaded from cache if already exists — takes seconds)
   │
   ├─ Online buffer warmup: collect 10K random transitions
   │
   ├─ [CACHED] Online buffer (10K entries) → online_buffer_cache/<hash>/
   │            (or loaded from cache if already exists — takes seconds)
   │
   ├─ Critic warmup: 10K critic-only gradient updates
   │
   ├─ Training loop starts (step 10,001 → 500,000)
   │   │
   │   ├─ Step 10,000: [SAVED] checkpoint → models/policy_step_10000/ + models/latest/
   │   ├─ Step 20,000: [SAVED] checkpoint → models/policy_step_20000/ + models/latest/
   │   ├─ ...each eval may update → models/best/
   │   ├─ Step 90,000: [SAVED] checkpoint → models/policy_step_90000/ + models/latest/
   │   │
   │   └─ ★ CRASH ★
   │       ↑ Online buffer data since warmup → LOST
   │       ↑ Agent weights → SAFE in models/latest/checkpoint.pt
   │       ↑ Resume: run with resume_ckpt=.../models/latest
   │
   └─ (If training completes normally: models/final/ saved, run dir optionally cleaned up)
```

### The FIFO ring buffer behavior

The online buffer is a `TensorDictPrioritizedReplayBuffer` backed by `LazyTensorStorage(max_size=buffer_size)`. It works as a circular ring buffer:

```
buffer_size = 80,000

Step 10,001:  entry stored at index 10,001  (warmup filled 0–9,999)
Step 20,000:  entry stored at index 20,000
...
Step 89,999:  entry stored at index 79,999  ← buffer is NOW FULL (80K entries)
Step 90,000:  entry stored at index 0       ← OVERWRITES the oldest entry
Step 90,001:  entry stored at index 1       ← OVERWRITES second oldest
...
```

Old data doesn't accumulate — it gets overwritten. The buffer always holds the **most recent `buffer_size` transitions**.

### What the cache hash depends on

The online buffer cache key is a SHA1 hash of:
```python
{
    "task": "TwoArmCoffee",
    "image_keys": ["observation.images.agentview", ...],
    "n_step": 5,
    "gamma": 0.995,
    "buffer_size": 80000,     # ← THIS IS IN THE HASH
    "batch_size": 64,
    "random_action_noise_scale": 0.2,
    ...
}
```

**Important**: `buffer_size` is part of the hash. Changing `buffer_size` invalidates the cache and warmup re-runs (which is correct — the old cache has the wrong capacity).

---

## 6. The RL Vision Encoder — Separate from BC

### Two completely independent vision systems

This is one of the most surprising architectural decisions. The system has **two separate vision encoders** running simultaneously:

```
┌──────────────────────────────────────────────────────────────────┐
│                    FROZEN BC POLICY (ACT)                        │
│                                                                  │
│  Observation ─→ [ACT's Vision Encoder] ─→ ACT Transformer ─→ a_base │
│                  (frozen, never updated)                         │
│                  ResNet18 or ViT (from LeRobot)                  │
│                  Runs INSIDE BasePolicyVecEnvWrapper              │
│                  Processes ALL cameras in the same architecture   │
└──────────────────────────────────────────────────────────────────┘
                              │
                         a_base added to obs
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RL AGENT (QAgent)                              │
│                                                                  │
│  obs (images + state + a_base) ─→ [MinViT Encoders] ─→ Actor ─→ a_res │
│                                    (trained from scratch!)       │
│                                    One MinViT per camera         │
│                                    3 cameras = 3 separate MinViTs│
│                                                                  │
│                                 ─→ [MinViT Encoders] ─→ Critic  │
│                                    (shared with actor)           │
└──────────────────────────────────────────────────────────────────┘
```

### Why separate encoders?

The BC policy's encoder is part of the frozen ACT model — it's optimized for imitation learning and produces features for the ACT transformer's cross-attention mechanism. The RL agent needs features optimized for Q-value estimation and policy gradients, which have very different objectives. Training RL gradients through the BC encoder would also break the frozen policy (the whole point of residual RL is that the base policy never changes).

### The MinViT architecture (RL agent's encoder)

Each camera gets its own `VitEncoder` → `MinVit`:

```
Input: [B, 3, 84, 84] RGB image (uint8 → float32, normalized to [-0.5, 0.5])
                │
                ▼
PatchEmbed2 (two-stage CNN patchifier):
  Conv2d(3, 128, kernel_size=8, stride=4) → GroupNorm → ReLU
  Conv2d(128, 128, kernel_size=3, stride=2)
  Output: [B, 81, 128]  (81 patches of 128 dimensions)
                │
                ▼
+ Learned positional embeddings [1, 81, 128]
                │
                ▼
TransformerLayer × 1 (default depth=1):
  Pre-LayerNorm → MultiHeadAttention(4 heads, flash attention) → dropout
  Pre-LayerNorm → FFN(128 → 512 → 128, GELU) → dropout
                │
                ▼
Final LayerNorm
  Output: [B, 81, 128]  (81 patches × 128 dim = 10,368 per camera)
```

**Default config** (from `resfit/rl_finetuning/config/rlpd.py` lines 14–22):

| Parameter | Value | Meaning |
|---|---|---|
| `embed_style` | `"embed2"` | Two-stage CNN patchifier |
| `embed_dim` | 128 | Patch embedding dimension |
| `num_heads` | 4 | Attention heads per transformer layer |
| `depth` | 1 | Number of transformer layers |
| `embed_norm` | `True` | GroupNorm in patchifier |

### How multiple cameras are processed

**NOT shared** — each camera has its own MinViT with independent weights (though identical architecture):

```python
# q_agent.py lines 155–172
def _build_encoders(self, obs_shape):
    encoders = nn.ModuleList()
    for _ in self.rl_cameras:           # e.g., 3 cameras
        enc = VitEncoder(obs_shape, self.cfg.vit)
        encoders.append(enc)            # 3 separate VitEncoder instances
    return encoders
```

At inference time, images are encoded in a loop and features concatenated:

```python
# q_agent.py lines 213–237
def _encode(self, obs, augment):
    feats = []
    for cam_idx, cam_name in enumerate(self.rl_cameras):
        data = obs[cam_name]
        if augment:
            data = self.aug(data)        # RandomShiftsAug(pad=4)
        feat_cam = self.encoders[cam_idx].forward(data, flatten=False)
        feats.append(feat_cam)
    return torch.cat(feats, dim=1)       # [B, 81*3, 128] = [B, 243, 128]
```

Total representation: `81 patches × 3 cameras = 243 patches × 128 dim = 31,104` feature dimensions.

### Is the RL encoder frozen or trained?

**Trained by default** (`freeze_encoder: bool = False` in config). The encoder has its own AdamW optimizer (`encoder_opt`) and receives gradients from:

1. **Critic loss** — backward through Q(encoder(obs), action) at every critic update
2. **Actor loss** — backward through actor(encoder(obs)) every 4th update (if `bc_backprop_encoder=True`)

The encoder is the most expensive part of the RL agent (3 separate ViTs), which is why the default `depth=1` keeps it lightweight.

### Comparison: BC encoder vs RL encoder

| Property | BC (Frozen ACT) Encoder | RL (QAgent) Encoder |
|---|---|---|
| Architecture | ResNet18 or ViT (from LeRobot ACTPolicy) | MinViT (custom, lightweight) |
| Weights | Frozen — never updated during RL | Trained from scratch during RL |
| Camera handling | Depends on ACT config (often shared encoder) | One separate MinViT per camera |
| Purpose | Produce features for action chunking transformer | Produce features for Q-value estimation + policy gradient |
| Input resolution | As configured in BC training (e.g., 84×84) | 84×84 |
| Output | Fed into ACT decoder → `a_base` | Fed into Critic MLP + Actor MLP → `a_residual` |
| Where it runs | Inside `BasePolicyVecEnvWrapper` (env layer) | Inside `QAgent` (RL agent) |

### Total inference cost per env step

Every environment step requires **both** vision systems to run:

```
1. QAgent._encode(obs)           → 3× MinViT forward   → a_residual
2. BasePolicyVecEnvWrapper
   → base_policy.select_action() → ACT encoder forward  → a_base
3. Combined: a_final = a_base + a_residual
```

This is computationally expensive: 4 encoder forward passes per step (3 MinViT + 1 ACT). This is the main reason training is GPU-bound rather than CPU-bound on modern GPUs.

---

## 7. Memory Budget: `buffer_size` Must Match Your RAM

### The pre-allocation trap

`LazyTensorStorage(max_size=buffer_size)` allocates torch tensors of shape `[buffer_size, ...]` on first write. For 3 cameras at 84×84, each entry stores:

| Field | Shape | Dtype | Bytes per entry |
|---|---|---|---|
| `obs/observation.images.*` (3 cameras) | `[3, 84, 84]` × 3 | uint8 | 63,504 |
| `next/obs/observation.images.*` (3 cameras) | `[3, 84, 84]` × 3 | uint8 | 63,504 |
| `obs/observation.state` | `[36]` | float32 | 144 |
| `next/obs/observation.state` | `[36]` | float32 | 144 |
| `obs/observation.base_action` | `[24]` | float32 | 96 |
| `next/obs/observation.base_action` | `[24]` | float32 | 96 |
| `action` | `[24]` | float32 | 96 |
| Other (reward, done, priority, gamma, index) | various | various | ~100 |
| **Total per entry** | | | **~127 KB** |

### Memory table for different `buffer_size` values

| `buffer_size` | Online buffer RAM | + Offline (9.6 GB) + System (~8 GB) | Total | Fits 32 GB? |
|---|---|---|---|---|
| **50,000** | 5.9 GB | 23.5 GB | **23.5 GB** | ✅ Comfortable |
| **80,000** | 9.5 GB | 27.1 GB | **27.1 GB** | ✅ Tight but OK |
| 100,000 | 11.9 GB | 29.5 GB | **29.5 GB** | ⚠️ Borderline |
| 150,000 | 17.8 GB | 35.4 GB | **35.4 GB** | ❌ OOM |
| **200,000** (default) | 23.8 GB | 41.4 GB | **41.4 GB** | ❌ Way over |

> **The default `buffer_size=200,000` is designed for machines with 64+ GB RAM.**
> On a 32 GB laptop with 3 cameras, use `algo.buffer_size=80_000`.

### Why loading from cache makes it worse

On a fresh run, `LazyTensorStorage` uses Linux's lazy page allocation — physical RAM is consumed only as entries are written. But loading from cache (`.loads()`) reads every memmap byte, forcing **all** pages into physical RAM at once:

```
Fresh run:  virtual = 24 GB, physical grows gradually 0 → 24 GB over 200K steps
Cache load: virtual = 24 GB, physical = 24 GB IMMEDIATELY
```

If your previous run used `buffer_size=200,000`, the cache has 200K-capacity memmaps. Loading that cache on a 32 GB machine will instantly OOM. **Delete the stale cache** before re-running with a smaller `buffer_size`:

```bash
rm -rf online_buffer_cache/
```

The offline buffer cache is fine — its capacity is fixed by the dataset size, not a configurable parameter.

---

## 8. Known Crash: Matplotlib/Tkinter Thread Conflict

### Symptom

Training crashes with:
```
RuntimeError: main thread is not in main loop
Tcl_AsyncDelete: async handler deleted by the wrong thread
Aborted (core dumped)
```

Followed by `BrokenPipeError` from `AsyncVectorEnv` worker pipes.

### Root cause

The evaluation utility `resfit/rl_finetuning/utils/evaluate_dexmg.py` imports `matplotlib.pyplot` at module level to generate Q-value diagnostic plots during evaluation. When matplotlib is imported, it auto-selects a GUI backend — on Linux with a display server this defaults to **TkAgg**, which imports `tkinter` and initializes the **Tcl event loop** in the main process.

The problem: `AsyncVectorEnv` uses `multiprocessing` with `context="spawn"` to run environment workers. When the main process's Tcl state collides with worker process cleanup (garbage collection of GUI-related objects in a non-main thread), Tcl aborts with the thread mismatch error. This kills the main process, which then breaks the pipes to all worker processes.

The import chain is:
```
train_residual_td3.py
  → from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
    → import matplotlib.pyplot as plt              ← triggers backend selection
      → matplotlib auto-selects TkAgg backend
        → import tkinter                           ← Tcl event loop initialized
          → Tcl state exists in main process
            → AsyncVectorEnv spawns workers
              → Worker cleanup triggers Tcl_AsyncDelete from wrong thread
                → CRASH
```

### Fix

`evaluate_dexmg.py` now forces the non-interactive `Agg` backend before importing pyplot:

```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — prevents Tkinter/Tcl thread conflicts
import matplotlib.pyplot as plt
```

The `Agg` backend renders to in-memory buffers and files without any GUI toolkit — Tkinter is never imported, and the Tcl event loop is never created. This is the correct choice for headless training since we only need matplotlib to save Q-value plots as PNG files, not to display them interactively.

---

**Related documents:**
- [RESIDUAL_RL_TRAINING.md](RESIDUAL_RL_TRAINING.md) — Full training pipeline walkthrough
- [../algorithms/REPLAY_BUFFERS.md](../algorithms/REPLAY_BUFFERS.md) — Prioritized replay, online/offline mixing, prefetching
- [../algorithms/RESIDUAL_LEARNING.md](../algorithms/RESIDUAL_LEARNING.md) — Why residual RL works, zero initialization, action scaling
