# BC Policy Training — Complete Deep Dive

This document explains every aspect of the BC (Behavior Cloning) training pipeline in this project, starting from the command in the README and tracing through every file, function, parameter, and design decision.

---

## Table of Contents

1. [What is BC Training?](#1-what-is-bc-training)
2. [The Training Command — Every Parameter Explained](#2-the-training-command--every-parameter-explained)
3. [Gradient Clipping — What It Actually Is](#3-gradient-clipping--what-it-actually-is)
4. [Output Directory — Where Checkpoints Go](#4-output-directory--where-checkpoints-go)
5. [Complete Execution Flow](#5-complete-execution-flow)
6. [Dataset Loading Pipeline](#6-dataset-loading-pipeline)
7. [Delta Timestamps — What They Are and Why They Exist](#7-delta-timestamps--what-they-are-and-why-they-exist)
8. [Image Transforms (Data Augmentation)](#8-image-transforms-data-augmentation)
9. [Normalization — Formulas and What Gets Normalized](#9-normalization--formulas-and-what-gets-normalized)
10. [Policy Creation Flow](#10-policy-creation-flow)
11. [The Training Loop](#11-the-training-loop)
12. [Evaluation During Training](#12-evaluation-during-training)
13. [Checkpoint Saving Strategy](#13-checkpoint-saving-strategy)
14. [Simulation Environments — MuJoCo, XMLs, Control Loop](#14-simulation-environments--mujoco-xmls-control-loop)
15. [ACT vs Diffusion — Comparison](#15-act-vs-diffusion--comparison)
16. [What the ACT Policy Sees: Complete Input Breakdown](#16-what-the-act-policy-sees-complete-input-breakdown)
17. [Configuring State Composition (Increasing State Dimensions)](#17-configuring-state-composition-increasing-state-dimensions)
18. [Configuring Image Resolution](#18-configuring-image-resolution)
19. [File Reference Map](#19-file-reference-map)

---

## 1. What is BC Training?

BC (Behavior Cloning) is supervised learning on expert demonstrations. The model learns: **"given this observation, produce this action."** It is the first stage of the ResFiT pipeline:

1. **Train BC policy** (this document) — produces a base policy from demonstrations
2. **Freeze BC policy** — lock it as a frozen "base" in the residual RL stage
3. **Train residual RL policy** — learns a small correction $a_{\text{residual}}$ on top: $a_{\text{final}} = a_{\text{BC}} + a_{\text{residual}}$

The BC policy trains purely offline — it reads observations and actions from a recorded dataset and never interacts with the environment during training. Optional evaluation rollouts run periodically to monitor performance but do not contribute to learning.

---

## 2. The Training Command — Every Parameter Explained

### Required Parameters

| Parameter | Default | Description |
|---|---|---|
| `--dataset` | *required* | HuggingFace Hub repo ID (e.g. `ankile/dexmg-two-arm-coffee`). Auto-downloads to `~/.cache/huggingface/lerobot/` on first run. |
| `--policy` | `diffusion` | Architecture: `act` or `diffusion`. Others (`pi0`, `pi0fast`, `tdmpc`, `vqbet`) raise `NotImplementedError`. |

### Training Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--steps` | `100000` | Total gradient updates. Not epochs — the dataset is cycled infinitely. |
| `--batch_size` | `64` | Samples per gradient step. Each sample = 1 timestep (obs + images + action chunk). |
| `--grad_clip_norm` | `10.0` | Maximum L2 norm for gradient clipping (see [Section 3](#3-gradient-clipping--what-it-actually-is)). |
| `--num_workers` | `4` | DataLoader CPU worker processes for video decoding. |
| `--seed` | `None` | Random seed (torch + numpy + python). |
| `--device` | auto-detect | `cuda` if GPU available, else `cpu`. |

### Logging & Checkpoints

| Parameter | Default | Description |
|---|---|---|
| `--output_dir` | `outputs/train_hf` | Unused — overridden by a timestamped directory (see [Section 4](#4-output-directory--where-checkpoints-go)). |
| `--log_freq` | `100` | Print + WandB log interval (steps). |
| `--save_freq` | `10000` | Checkpoint save interval (steps). |

### WandB

| Parameter | Default | Description |
|---|---|---|
| `--wandb_enable` | `False` | Flag to enable Weights & Biases logging. |
| `--wandb_project` | `None` | W&B project name. Required when `--wandb_enable` is set. |
| `--wandb_entity` | `None` | W&B team/entity. Uses default account if omitted. |

### Resume

| Parameter | Default | Description |
|---|---|---|
| `--resume_ckpt` | `None` | Path to a local checkpoint directory. Loads model + optimizer + step count. |
| `--resume_run_id` | `None` | WandB run ID. Downloads the `latest` artifact from WandB. |

### Evaluation Rollouts

| Parameter | Default | Description |
|---|---|---|
| `--rollout_freq` | `None` | Run eval every N steps. Disabled when not set. |
| `--eval_env` | `None` | Which simulation to evaluate in (must match the dataset task). |
| `--eval_num_envs` | `5` | Parallel simulation instances. Each is a separate OS process with its own MuJoCo. |
| `--eval_num_episodes` | `20` | Total episodes per evaluation round. |
| `--eval_camera_size` | `84` | Image resolution fed to the policy during eval. Must match dataset. |
| `--eval_render_size` | `None` | Resolution for video recording (separate from policy input). Higher = nicer videos. |
| `--eval_video_key` | `None` | Camera for video recording (e.g. `observation.images.frontview`). |
| `--debug` | `False` | Use `SyncVectorEnv` (single-process) instead of `AsyncVectorEnv` (multi-process). |

### Policy Overrides

| Parameter | Default | Description |
|---|---|---|
| `--policy_kwargs` | `None` | Override policy hyperparameters. JSON or `key=val,key=val`. |
| `--policy_cameras` | `None` | Filter to only use specific cameras from dataset. |
| `--disable_proprioceptive_obs` | `False` | Remove `observation.state` — vision-only training. |

### Supported Environments

All environments below use **MuJoCo** as the physics engine, through robosuite. Not Gazebo, not PyBullet, not Isaac — MuJoCo only.

| Category | Environments | Robot |
|---|---|---|
| DexMimicGen (bimanual humanoid) | TwoArmCoffee, TwoArmPouring, TwoArmCanSortRandom | GR1FixedLowerBody |
| DexMimicGen (bimanual panda) | TwoArmThreading, TwoArmThreePieceAssembly, TwoArmTransport | Panda × 2 |
| DexMimicGen (dexterous hands) | TwoArmLiftTray, TwoArmBoxCleanup, TwoArmDrawerCleanup | PandaDexRH + PandaDexLH |
| Robomimic (single arm) | Lift, Can, Square, Transport | Panda |
| MimicGen (single arm) | Threading | Panda |

---

## 3. Gradient Clipping — What It Actually Is

**Gradient clipping is NOT layer normalization.** They are completely different operations at different stages of the pipeline. For a full numerical walkthrough with exact arithmetic on a concrete MLP, see [../algorithms/LAYERNORM_VS_GRADCLIP.md](../algorithms/LAYERNORM_VS_GRADCLIP.md).

| Property | Layer Normalization | Gradient Clipping (`grad_clip_norm`) |
|---|---|---|
| **When** | Forward pass (input → output) | Backward pass (loss → gradients → weights) |
| **Operates on** | Activations (data between layers) | Gradients (proposed weight updates) |
| **Formula** | $\hat{z}_d = \frac{z_d - \mu}{\sigma + \epsilon}$ | $g_{\text{clip}} = g \cdot \min\!\left(1, \;\frac{\text{max\_norm}}{\|g\|_2}\right)$ |
| **Scope** | Local (one token, one layer, 512 dims) | Global (all ~34M parameters as one vector) |
| **Purpose** | Keep activations well-scaled for stable learning | Prevent catastrophic weight updates from exploding gradients |
| **During inference?** | Yes — it's part of the model architecture | No — there are no gradients at inference |

### How gradient clipping works

After `loss.backward()` computes gradients for all ~34M parameters, gradient clipping measures the **total L2 norm** of all gradients as one flat vector:

$$\|g\|_2 = \sqrt{\sum_{p \in \text{all params}} \sum_{i} g_{p,i}^2}$$

If $\|g\|_2 > 10.0$, every gradient is scaled by the same factor $\alpha = 10.0 / \|g\|_2$ so the total norm becomes exactly 10.0. **The update direction is preserved** — only the step size shrinks.

The code, at [train_bc_dexmg.py line 774](resfit/lerobot/scripts/train_bc_dexmg.py#L774):

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
```

### What this does NOT do

- Does NOT limit parameter values or predicted actions to any range
- Does NOT normalize activations during the forward pass
- Does NOT affect inference — only training
- The value 10.0 is a threshold for the **total gradient vector norm**, not a per-element clamp

The value 10.0 is fairly permissive — it rarely activates during normal training and mainly protects against rare catastrophic gradient spikes from bad batches.

---

## 4. Output Directory — Where Checkpoints Go

When you run the command **without** explicitly setting `--output_dir`, the `--output_dir` argument defaults to `"outputs/train_hf"`, but **this value is never actually used**. The actual output directory is created at [train_bc_dexmg.py lines 76-77 and 420-422](resfit/lerobot/scripts/train_bc_dexmg.py#L76-L77):

```python
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_cache_dir = _CACHE_ROOT / f"bc_run_{timestamp}_{Path(cfg.dataset).name}_{cfg.policy}"
```

Then at [line 672](resfit/lerobot/scripts/train_bc_dexmg.py#L672):

```python
output_dir = run_cache_dir
```

### Where exactly is it?

- If `CACHE_DIR` environment variable is NOT set → `_CACHE_ROOT` = current working directory (`.`)
- If `CACHE_DIR` is set → that path

So running from `~/personal_abhi/temp/residual-offpolicy-rl/` produces:

```
~/personal_abhi/temp/residual-offpolicy-rl/bc_run_2026-02-18_14-30-00_dexmg-two-arm-coffee_act/
```

### Directory structure

```
bc_run_<timestamp>_<dataset>_<policy>/
├── policy_step_10000/policy/       # Model-only checkpoint at step 10K
│   ├── config.json
│   └── model.safetensors
├── policy_step_20000/policy/       # Model-only checkpoint at step 20K
│   └── ...
├── latest/                         # Full training state (overwritten each save)
│   ├── policy/config.json
│   ├── policy/model.safetensors
│   └── trainer_state.pt            # Optimizer state dict + step count
├── best/                           # Full state of best success rate
│   └── ...
├── best_step_15000/policy/         # Model-only best checkpoint
│   └── ...
└── eval_twoardcoffee/<run_start_time>/
    └── eval_step_5000_<timestamp>.mp4  # Evaluation videos
```

### Cleanup warning

**After training finishes, the entire directory is deleted** ([line 923](resfit/lerobot/scripts/train_bc_dexmg.py#L923)):

```python
if run_cache_dir.exists():
    shutil.rmtree(run_cache_dir)
```

This assumes all checkpoints were uploaded to WandB. **If WandB is disabled, all local files are lost.** To keep local checkpoints without WandB, comment out the cleanup code or set `CACHE_DIR` and separately back up the files.

---

## 5. Complete Execution Flow

### Phase 1: Setup (lines 408-430)

Parse CLI → create timestamped output directory under `CACHE_DIR` (default: current directory) → set random seed.

### Phase 2: Dataset + Policy Config (lines 432-560)

1. **Fetch metadata** from HuggingFace Hub (`LeRobotDatasetMetadata`) — downloads only `meta/info.json`, not the actual data
2. **Build policy config** (`ACTConfig` or `DiffusionConfig`) with CLI overrides
3. **Hardcode** `chunk_size=20` and `n_action_steps=20` (matching 20fps environments)
4. **Filter cameras** if `--policy_cameras` specified
5. **Remove** `observation.state` if `--disable_proprioceptive_obs`
6. **Resolve delta_timestamps** — convert policy delta indices to seconds using dataset fps
7. **Create image transforms** — photometric augmentations (brightness, contrast, saturation, hue, sharpness)
8. **Create `LeRobotDataset`** — downloads actual data (parquets + videos) to `~/.cache/huggingface/lerobot/`
9. **Create `DataLoader`** with shuffle, pin_memory, persistent_workers → wrap in `cycle()` for infinite iteration

### Phase 3: Policy + Optimizer (lines 565-590)

1. `make_policy(policy_cfg, ds_meta)` → instantiate `ACTPolicy` with ResNet18 backbone
2. Create `AdamW` optimizer with two parameter groups (backbone lr=1e-5, everything else lr=1e-4)
3. Set policy to training mode

### Phase 4: WandB + Resume (lines 590-665)

Initialize WandB run (if enabled) → optionally load checkpoint from local path or WandB artifact.

### Phase 5: Create Eval Environments (lines 670-730)

If `--rollout_freq` and `--eval_env` set → create `create_vectorized_env()` → N parallel MuJoCo simulations.

### Phase 6: Training Loop (lines 735-835)

For each step: load batch → move to GPU → forward → loss.backward → clip gradients → optimizer.step → periodic logging/checkpointing/evaluation.

### Phase 7: Cleanup (lines 835-930)

Close eval environments → finish WandB run → delete local run directory.

---

## 6. Dataset Loading Pipeline

### Source: HuggingFace Hub

The `--dataset` parameter is a HuggingFace Hub repo ID. `LeRobotDataset` handles downloading:

```python
dataset = LeRobotDataset(
    "ankile/dexmg-two-arm-coffee",     # HuggingFace Hub repo ID
    delta_timestamps=delta_timestamps,  # Which timesteps to load per sample
    download_videos=True,               # Download MP4 video files
    image_transforms=image_transforms,  # Augmentation pipeline
)
```

Files are cached at `~/.cache/huggingface/lerobot/ankile/dexmg-two-arm-coffee/`.

### Using custom / local data

`LeRobotDataset` expects a HuggingFace Hub repo ID. To use custom local data:
1. **Recommended:** Convert your data to LeRobot v2.x format and upload to HuggingFace Hub using `lerobot` tools
2. **Alternative:** Modify `LeRobotDataset` to accept a local path (non-trivial, requires changes to download logic)

The dataset must follow the LeRobot v2.x specification: parquet files + MP4 videos + `meta/info.json`.

### How `__getitem__` builds a training sample

From [lerobot_dataset.py line 702](deps/lerobot/lerobot/common/datasets/lerobot_dataset.py#L702):

1. DataLoader picks a random index `idx` — one specific frame from one specific episode
2. Reads the base row from the HuggingFace dataset (memory-mapped parquet — fast)
3. If `delta_indices` are set: computes query indices for temporal neighbors (the next 20 action frames)
4. **Clamps** indices to episode boundaries and creates padding mask
5. **Decodes video frames** using `torchcodec` for the requested timestamps — this is the CPU bottleneck
6. Applies image transforms (augmentation)
7. Returns the assembled dictionary

### Padding at episode boundaries

From [lerobot_dataset.py `_get_query_indices`](deps/lerobot/lerobot/common/datasets/lerobot_dataset.py#L643):

```python
query_indices[key] = [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
padding[f"{key}_is_pad"] = [(idx + delta < ep_start) or (idx + delta >= ep_end) for delta in delta_idx]
```

**Example:** Episode has 100 frames (index 0-99), `chunk_size=20`, sampling frame 95:
- Requested action indices: 95, 96, 97, 98, 99, 100, 101, ..., 114
- Clamped indices: 95, 96, 97, 98, **99, 99, 99, ..., 99** (last valid frame repeated)
- `action_is_pad`: `[F, F, F, F, F, T, T, ..., T]` — 5 real, 15 padded

The training loss uses this mask to **zero out the loss on padded actions**: `l1_loss * ~action_is_pad`.

### How many samples exist?

**Every single frame** in every episode is a valid sample index. With 1014 episodes averaging ~322 frames = **326,707 total samples**.

The chunk size does NOT reduce the sample count. You don't split 1000 frames into 50 chunks of 20. Instead, each of the 1000 frames is a sample, and each sample fetches a sliding window of 20 future actions starting from that frame.

### DataLoader workers (num_workers=4)

These 4 workers are for **data loading** — not evaluation environments. Each worker:
1. Picks a sample index from the shuffled queue
2. Reads parquet row (fast, memory-mapped)
3. Decodes video frames via `torchcodec` (CPU-heavy — the bottleneck)
4. Applies image transforms
5. Returns via shared memory to the main process

While the GPU trains on one batch, workers are decoding the next batch in parallel. More workers = faster data pipeline = less time the GPU sits idle waiting for data.

### What "dataset exhausted" means and how cycling works

With `shuffle=True`, the DataLoader shuffles all 326,707 indices at the start, then serves them in random order. When all indices have been served once (one "epoch"), the underlying iterator raises `StopIteration`. The `cycle()` wrapper catches this and creates a fresh iterator:

```python
def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)  # Re-shuffles and restarts
```

Within one pass: each sample is used exactly once (no duplicates within an epoch). Across passes: repeated sampling with different shuffle orders + different random augmentations. With batch_size=256 and 326,707 samples, one full pass ≈ 1,276 steps. Training for 200,000 steps ≈ 157 full passes through the dataset.

---

## 7. Delta Timestamps — What They Are and Why They Exist

### The problem they solve

A training sample needs **multiple timesteps** of data — not just one frame. ACT needs the current observation plus 20 future actions. Diffusion needs 2 past observations plus 16 actions. The delta timestamp system tells the dataset loader **which timesteps to fetch relative to the sampled frame**.

### How they work

Datasets store **absolute timestamps** in parquet — frame 0 at 0.0s, frame 1 at 0.05s, frame 2 at 0.1s (at 20fps). Delta timestamps are **relative offsets from the sampled frame**, expressed in seconds.

For ACT (`chunk_size=20`, 20fps):

```python
# ACTConfig properties:
observation_delta_indices = None           # Only current frame
action_delta_indices = [0, 1, 2, ..., 19]  # Next 20 frames
```

`resolve_delta_timestamps` ([deps/lerobot/lerobot/common/datasets/factory.py](deps/lerobot/lerobot/common/datasets/factory.py)) converts:

$$\text{delta\_timestamp}_i = \frac{\text{delta\_index}_i}{\text{fps}} = \frac{i}{20} \text{ seconds}$$

Result: `action` delta_timestamps = `[0.0, 0.05, 0.1, ..., 0.95]` seconds

These timestamps are then converted to frame index offsets inside `LeRobotDataset`:

$$\text{delta\_index}_i = \text{round}(\text{delta\_timestamp}_i \times \text{fps}) = i$$

The dataset uses these offsets to fetch the right frames from parquet and decode the right frames from video.

### Are actions delta (relative) or absolute?

**It depends on the robot and environment.** The action type is determined by the robosuite controller configuration loaded per robot, NOT set in the policy code:

| Environment | Robot(s) | Controller | Action Mode |
|---|---|---|---|
| `PickPlaceCan`, `NutAssemblySquare`, `Lift`, `Threading` | Panda (single-arm) | `OSC_POSE` with `input_type: "delta"` | **Delta** EE pose (6-DoF) + gripper |
| `TwoArmBoxCleanup`, `TwoArmDrawerCleanup`, `TwoArmLiftTray` | PandaDex (dual dexterous) | `OSC_POSE` with `input_type: "delta"` | **Delta** EE pose per arm + hand joints |
| `TwoArmCoffee`, `TwoArmCanSortRandom`, `TwoArmPouring` | GR1 Humanoid | `WHOLE_BODY_MINK_IK` with `ik_input_type: "absolute"` | **Absolute** EE pose + hand joints |

**For Panda tasks (delta mode):** Each action is a *change* relative to the current end-effector pose. An action of `[0.01, 0, 0, ...]` means "move 0.01m in x from where you are now." The controller (`OSC_POSE`) computes `goal = current_pos + scale(delta)`.

**For GR1 tasks (absolute mode):** Each action is a *target* end-effector pose relative to the robot base frame. An action of `[0.5, 0.3, 0.8, ...]` means "move to position (0.5, 0.3, 0.8)." The IK solver computes joint targets to achieve this pose.

**The policy doesn't know or care** — it just predicts numbers that match the training data distribution. The controller stack in robosuite interprets them correctly based on the JSON config (see `deps/robosuite/robosuite/controllers/config/robots/`).

**Important**: Each action in the chunk is independently interpretable. They are NOT cumulative deltas on top of previous chunk actions.

### Auto-detection from dataset

The fps comes from the dataset metadata, but `chunk_size=20` and `n_action_steps=20` are **hardcoded** in the training script, not auto-detected. If you used a 10fps dataset, you would need to manually change these values.

---

## 8. Image Transforms (Data Augmentation)

### Which augmentations are applied?

From [deps/lerobot/lerobot/common/datasets/transforms.py](deps/lerobot/lerobot/common/datasets/transforms.py):

**Only photometric (color) augmentations** — no spatial transforms (no cropping, no flipping, no resizing):

| Transform | Parameter Range | Operation |
|---|---|---|
| Brightness | factor $\in [0.8, 1.2]$ | $\text{pixel} \times \text{factor}$ |
| Contrast | factor $\in [0.8, 1.2]$ | $(\text{pixel} - \mu) \times \text{factor} + \mu$ |
| Saturation | factor $\in [0.5, 1.5]$ | Interpolate between grayscale and original |
| Hue | shift $\in [-0.05, 0.05]$ | Rotate hue channel in HSV space |
| Sharpness | factor $\in [0.5, 1.5]$ | Blend between blurred and sharpened versions |

Applied via `RandomSubsetApply`: on each image, randomly selects **up to 3 of the 5** transforms and applies them sequentially with random parameters sampled from the ranges above.

### Why no cropping?

ACT accepts images at the exact resolution provided — 84×84 in this dataset. There is no random crop in the data pipeline because the training image size must match the evaluation image size. If the training images were cropped, the ResNet feature map dimensions would change.

**Exception:** Diffusion Policy applies its own cropping _inside the model_ via `DiffusionRgbEncoder` (random crop during training, center crop during eval). But this is a model-internal operation, not a data transform. Diffusion's default `crop_shape=(240,240)` assumes ~256×256 inputs and would need adjustment for 84×84.

### Does augmentation create extra data?

**No.** The dataset size remains 326,707 samples. Augmentation applies random perturbations **on-the-fly** each time a sample is loaded. So in 157 epochs, the model sees each frame ~157 times with slightly different color variations. This teaches color invariance without inflating the dataset.

### Does the model ever see the original unaugmented images?

During training, every image gets a random augmentation (or none — if 0 transforms happen to be selected, the image passes through unchanged). During evaluation, `image_transforms` is **not applied**. The eval environments produce clean, unaugmented observations directly from MuJoCo rendering.

The augmentations are mild (±20% brightness, ±5% hue) — small enough that a model trained on augmented data generalizes seamlessly to clean images. The model does NOT need to see the "original" version to work at eval time; the slight color shifts just make it more robust to lighting variation.

---

## 9. Normalization — Formulas and What Gets Normalized

### MEAN_STD normalization (used by ACT for all feature types)

**Forward (normalize):**

$$x_{\text{norm}} = \frac{x - \mu}{\sigma + 10^{-8}}$$

**Inverse (unnormalize):**

$$x = x_{\text{norm}} \cdot \sigma + \mu$$

Where $\mu$ and $\sigma$ are precomputed from the training dataset and stored as non-trainable model buffers.

### MIN_MAX normalization (used by Diffusion for state and action)

**Forward (normalize):**

$$x_{\text{norm}} = \frac{2(x - x_{\min})}{x_{\max} - x_{\min} + 10^{-8}} - 1$$

Maps to $[-1, 1]$.

**Inverse (unnormalize):**

$$x = \frac{x_{\text{norm}} + 1}{2} \cdot (x_{\max} - x_{\min}) + x_{\min}$$

### What gets normalized, and per-feature details

**ACT defaults** (from `ACTConfig.normalization_mapping`):

| Feature | Type | Normalization | Stats shape | Typical effect |
|---|---|---|---|---|
| Images | VISUAL | MEAN_STD | $(C, 1, 1)$ — per-channel | Images enter as float $[0, 1]$ (already /255). Per-channel $\mu$ and $\sigma$ are **computed from your dataset** (NOT fixed ImageNet constants). They depend on the scene (table color, lighting, objects). Normalized range: **unbounded** z-scores, roughly $[-3, +4]$. See [VISION_BACKBONE.md](VISION_BACKBONE.md) for details. |
| Robot state | STATE | MEAN_STD | $(D_\text{state},)$ — per-dimension | Each state component (ee position, quaternion, gripper angle) independently centered and scaled to roughly unit variance. Output range: **unbounded** z-scores, roughly $[-3, +3]$. |
| Action | ACTION | MEAN_STD | $(D_\text{action},)$ — per-dimension | Action targets normalized to z-scores. Output range: **unbounded**, roughly $[-3, +3]$. Model predicts in normalized space. Predictions unnormalized before sending to environment. |

**Diffusion defaults** (from `DiffusionConfig.normalization_mapping`):

| Feature | Type | Normalization |
|---|---|---|
| Images | VISUAL | MEAN_STD |
| Robot state | STATE | MIN_MAX → $[-1, 1]$ |
| Action | ACTION | MIN_MAX → $[-1, 1]$ |

### When normalization happens

**Training (forward pass):**
```
batch = normalize_inputs(batch)    # images + state: MEAN_STD
batch = normalize_targets(batch)   # actions: MEAN_STD
loss = L1(predicted, target)       # Loss in normalized space
```

**Inference (select_action):**
```
batch = normalize_inputs(batch)    # images + state: MEAN_STD
actions_norm = model(batch)        # Predict in normalized space
actions = unnormalize(actions_norm) # Convert back to physical units
```

The statistics ($\mu$, $\sigma$, $x_{\min}$, $x_{\max}$) are automatically computed from the `LeRobotDataset` and stored as `nn.Buffer` inside the `Normalize`/`Unnormalize` modules. They are saved/loaded as part of the model checkpoint.

---

## 10. Policy Creation Flow

### How `make_policy()` works

From [resfit/lerobot/policies/factory.py](resfit/lerobot/policies/factory.py):

```python
def make_policy(cfg, ds_meta):
    # 1. Convert dataset features to policy features
    features = dataset_to_policy_features(ds_meta.features)
    
    # 2. Split into inputs vs outputs
    cfg.output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    cfg.input_features  = {k: f for k, f in features.items() if k not in cfg.output_features}
    
    # 3. Instantiate policy class
    policy = ACTPolicy(config=cfg)  # or DiffusionPolicy
    policy.to(cfg.device)
```

### Feature properties on PreTrainedConfig

From [resfit/lerobot/configs/policies.py](resfit/lerobot/configs/policies.py):

```python
@property
def robot_state_feature(self):  # First input with type == STATE
@property
def image_features(self):       # All inputs with type == VISUAL
@property
def action_feature(self):       # First output with type == ACTION
```

For the TwoArmCoffee dataset:
- `robot_state_feature`: `observation.state` → shape $(36,)$ (6 EE states × 2 arms: pos, quat, gripper)
- `image_features`: 3 cameras each $(3, 84, 84)$: agentview, left hand, right hand
- `action_feature`: `action` → shape $(24,)$ (pos + rot + hand joints per arm)

### Vision backbone: ResNet18

ACT uses ResNet18 (from torchvision) initialized with ImageNet-pretrained weights. The final fully-connected classification layer is stripped — only the convolutional feature extractor is kept. **See [VISION_BACKBONE.md](VISION_BACKBONE.md) for a complete deep dive** with worked numerical examples, multi-camera handling, and ACT vs Diffusion comparison.

**ResNet18 architecture for 84×84 input:**

| Stage | Output shape | Details |
|---|---|---|
| Input | $(B, 3, 84, 84)$ | RGB image, float $[0,1]$, then MEAN_STD normalized |
| Conv1 (7×7, stride 2) + BN + ReLU + MaxPool(3×3, stride 2) | $(B, 64, 21, 21)$ | Initial feature extraction |
| Layer1 (2× BasicBlock) | $(B, 64, 21, 21)$ | Stride 1 |
| Layer2 (2× BasicBlock) | $(B, 128, 11, 11)$ | Stride 2 downsample |
| Layer3 (2× BasicBlock) | $(B, 256, 6, 6)$ | Stride 2 downsample |
| Layer4 (2× BasicBlock) | $(B, 512, 3, 3)$ | Stride 2 downsample |

The model extracts `layer4` output → a $(B, 512, 3, 3)$ feature map per camera, yielding $3 \times 3 = 9$ spatial tokens per camera, each of dimension 512.

These are projected and flattened into transformer encoder tokens. With 3 cameras: $3 \times 9 = 27$ image tokens plus 1 latent token plus 1 state token = 29 total encoder input tokens.

**ResNet18 accepts any spatial resolution** — it's fully convolutional. The 84×84 input works fine; it just produces smaller feature maps than 224×224 would.

**Why not ViT?** The ACT backbone is hardcoded to ResNet:
```python
if not self.vision_backbone.startswith("resnet"):
    raise ValueError(...)
```
Options: `resnet18`, `resnet34`, `resnet50`. No ViT support. The reason is practical: ResNet is faster and uses less memory for small images (84×84). ViT requires larger images (224×224+) and more data to work well.

### Two learning rate groups

From [modeling_act.py `get_optim_params()`](resfit/lerobot/policies/act/modeling_act.py#L79):

```python
return [
    {"params": [non-backbone params], "lr": 1e-4},     # Transformer + heads
    {"params": [backbone params],     "lr": 1e-5},     # ResNet18 (pretrained)
]
```

The ResNet backbone gets 10× lower learning rate because it starts with ImageNet-pretrained weights — fine-tuning too fast would destroy useful low-level visual features.

---

## 11. The Training Loop

From [train_bc_dexmg.py lines 735-780](resfit/lerobot/scripts/train_bc_dexmg.py#L735-L780):

```python
while step < cfg.steps:
    batch = next(dl_iter)                                            # 1. Load batch
    for key, val in batch.items():
        batch[key] = val.to(device, non_blocking=True)               # 2. GPU transfer
    loss, _ = policy.forward(batch)                                  # 3. Forward pass
    loss.backward()                                                  # 4. Backward pass
    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm) # 5. Clip gradients
    optimizer.step()                                                 # 6. Update weights
    optimizer.zero_grad(set_to_none=True)                            # 7. Clear gradients
    step += 1
```

### Batch contents

| Key | Shape | Content |
|---|---|---|
| `observation.state` | $(B, 36)$ | Robot state |
| `observation.images.agentview` | $(B, 3, 84, 84)$ | Camera 1 |
| `observation.images.robot0_eye_in_left_hand` | $(B, 3, 84, 84)$ | Camera 2 |
| `observation.images.robot0_eye_in_right_hand` | $(B, 3, 84, 84)$ | Camera 3 |
| `action` | $(B, 20, 24)$ | Ground-truth action chunk (20 steps × 24-dim) |
| `action_is_pad` | $(B, 20)$ | Boolean mask: True for padded boundary actions |

### Steps vs epochs

| Metric | Value |
|---|---|
| Total samples | 326,707 |
| Batch size | 256 |
| Steps per epoch | $\lfloor 326707 / 256 \rfloor = 1276$ |
| Total steps | 200,000 |
| Total epochs | $\approx 157$ |

---

## 12. Evaluation During Training

### How evaluation is toggled

Evaluation runs **only** when both `--rollout_freq` and `--eval_env` are set. The condition at [line 842](resfit/lerobot/scripts/train_bc_dexmg.py#L842):

```python
if (cfg.rollout_freq is not None and cfg.eval_env is not None
    and step % cfg.rollout_freq == 0 and step != start_step):
```

If either is `None`, evaluation never runs. The eval code is completely optional.

### How the environment is selected

The `--eval_env` name is matched against three hardcoded lists at [lines 678-700](resfit/lerobot/scripts/train_bc_dexmg.py#L678-L700):

```python
dexmimicgen_envs = ["TwoArmCoffee", "TwoArmThreading", ...]
robomimic_envs = ["Lift", "Can", "Square", "Transport"]
mimicgen_envs = ["Threading"]
```

If the name is in one of these lists, `create_vectorized_env()` builds the corresponding robosuite environment. If not, a `ValueError` is raised. The environment name maps to a specific robot, controller, and camera configuration via the `ENV_ROBOTS` dict and helper functions in [dexmg.py](resfit/dexmg/environments/dexmg.py).

### Simulation-only — no real-world eval

This codebase evaluates **only in MuJoCo simulation**. The eval loop uses `gymnasium.vector.AsyncVectorEnv` with MuJoCo instances. There is no real-world evaluation capability.

For real-world deployment of the ResFiT pipeline:
1. Train BC with `--save_freq` set to save periodic checkpoints
2. All checkpoints upload to WandB as artifacts
3. Download candidate checkpoints → deploy each on physical robot → manually measure success rate
4. Pick the best-performing checkpoint as the frozen BC base for residual RL

### The evaluation loop

From [`_run_rollouts()` at line 272](resfit/lerobot/scripts/train_bc_dexmg.py#L272):

```
1. Switch policy to eval mode (dropout disabled)
2. Reset all parallel environments
3. Loop:
   a. policy.select_action(obs)  → returns one action per env
   b. env.step(action)           → advance MuJoCo by one control step
   c. Capture rendered frames
   d. When done: record success/fail, annotate video, reset
4. success_rate = successes / total_episodes
5. Save annotated MP4 video
6. Switch back to train mode
```

### Can I see the evaluation live?

**No live MuJoCo window by default.** All environments run headless (`has_renderer=False`) with offscreen EGL rendering. Frames are rendered to GPU memory, captured as numpy arrays, and written to MP4 files.

To see a live visualization:
1. Set `has_renderer=True` in `RobosuiteGymWrapper.__init__` at [dexmg.py line 196](resfit/dexmg/environments/dexmg.py#L196)
2. Use `--debug` flag (forces single-process `SyncVectorEnv`)
3. This requires a display (X11/Wayland) — won't work over SSH without X forwarding
4. You'd see one MuJoCo window (not one per env)

The standard workflow: review the MP4 videos saved to disk or logged to WandB.

### Memory cost per eval environment

| Resource | Per environment |
|---|---|
| CPU RAM | ~200-500 MB (MuJoCo state) |
| GPU VRAM | ~100-300 MB (EGL offscreen rendering) |
| CPU cores | 1 (in async mode) |

`--eval_num_envs 16` → ~16 cores + ~2-5 GB VRAM on top of training memory. This is why the RTX 4090 Mobile (16GB) ran out of memory with `batch_size=256 + eval_num_envs=16`.

---

## 13. Checkpoint Saving Strategy

### Three checkpoint types

**1. Periodic model-only** (every `--save_freq` steps):
```
policy_step_10000/policy/
├── config.json          # Policy hyperparameters (JSON)
└── model.safetensors    # Weights only (no optimizer)
```

**2. Latest full state** (overwritten each save):
```
latest/
├── policy/config.json
├── policy/model.safetensors
└── trainer_state.pt     # {step: int, optimizer: state_dict}
```

**3. Best model** (on new best eval success rate):
```
best/                    # Full state (overwritten)
best_step_15000/policy/  # Model-only with step number
```

### WandB artifacts

| Name | Type | When |
|---|---|---|
| `run_{id}_model_step_{N}` | Model only | Every `save_freq` |
| `run_{id}_latest` | Full state | Every `save_freq` (overwrites) |
| `run_{id}_best` | Full state | New best success rate |

### File formats

| File | Format | Content |
|---|---|---|
| `config.json` | JSON | Human-readable policy hyperparameters |
| `model.safetensors` | SafeTensors | Weights — safe, fast, no arbitrary code execution |
| `trainer_state.pt` | PyTorch pickle | Optimizer momentum buffers + step counter |

---

## 14. Simulation Environments — MuJoCo, XMLs, Control Loop

### Physics engine

All environments use **MuJoCo** via the robosuite framework. Not Gazebo, not PyBullet, not Isaac Sim.

### Robot and object XML files

Robot bodies, grippers, arenas, and task objects are defined as MuJoCo MJCF XML files. Robosuite assembles them programmatically at runtime.

| Category | Path | Files |
|---|---|---|
| Robot bodies | `deps/robosuite/robosuite/models/assets/robots/` | `panda/robot.xml`, `gr1/robot.xml`, `ur5e/robot.xml`, etc. (12 robots) |
| Grippers | `deps/robosuite/robosuite/models/assets/grippers/` | `panda_gripper.xml`, `robotiq_85_gripper.xml`, `inspire_right_hand.xml`, etc. |
| Arenas | `deps/robosuite/robosuite/models/assets/arenas/` | `table_arena.xml`, `bins_arena.xml`, `empty_arena.xml` |
| Mounts/bases | `deps/robosuite/robosuite/models/assets/bases/` | `rethink_mount.xml`, `omron_mobile_base.xml` |
| Task objects (DexMG) | `deps/dexmimicgen/dexmimicgen/models/assets/objects/` | `coffee_base.xml`, `coffee_body.xml`, `coffee_lid.xml`, `coffee_pod.xml`, `cabinet.xml`, etc. |

### Control loop: 20 Hz (hardcoded)

The control frequency is set in the environment wrapper at [dexmg.py line 200](resfit/dexmg/environments/dexmg.py#L200):

```python
env_kwargs = {
    "control_freq": 20,   # 20 Hz — hardcoded, not auto-detected
}
```

The training script also hardcodes `chunk_size=20` and `n_action_steps=20` to match. The dataset was also recorded at 20fps. These three values (dataset fps, control_freq, chunk_size) are all manually kept in sync — there is no automatic detection.

### What 20 Hz means

Every 50ms (1/20 second):
1. MuJoCo reads sensor state → produces observations (images + robot state)
2. Policy predicts action (or pops from action queue for ACT)
3. Action sent to MuJoCo controller
4. MuJoCo advances physics by 50ms (internally may use multiple substeps at ~2ms each)

### Episode horizon

Each task has a fixed maximum episode length ([dexmg.py lines 147-160](resfit/dexmg/environments/dexmg.py#L147-L160)):

| Task | Horizon (steps) | Time at 20Hz |
|---|---|---|
| Lift | 100 | 5s |
| PickPlaceCan | 200 | 10s |
| NutAssemblySquare | 300 | 15s |
| TwoArmThreading | 300 | 15s |
| TwoArmBoxCleanup | 300 | 15s |
| TwoArmCoffee | 400 | 20s |
| TwoArmPouring | 400 | 20s |
| TwoArmCanSortRandom | 400 | 20s |
| TwoArmThreePieceAssembly | 500 | 25s |
| Threading | 500 | 25s |
| TwoArmLiftTray | 650 | 32.5s |
| TwoArmTransport | 800 | 40s |

Episodes end early if the task succeeds (reward = 1.0).

---

## 15. ACT vs Diffusion — Comparison

Both predict **chunks of future actions** from observations, but use fundamentally different generation methods. Full architectural details are in the separate [ACT_ARCHITECTURE.md](ACT_ARCHITECTURE.md) document.

### Core differences

| Aspect | ACT | Diffusion Policy |
|---|---|---|
| **Observation history** | 1 frame (current only) | 2 frames (current + previous) |
| **Prediction horizon** | `chunk_size` = 20 | `horizon` = 16 |
| **Actions executed** | `n_action_steps` = 20 (all) | `n_action_steps` = 8 (first half) |
| **Generation** | Single forward pass → L1 regression | Iterative denoising (100 DDPM steps) |
| **Loss** | $\mathcal{L} = \text{L1}(\hat{a}, a) + \lambda \cdot D_\text{KL}$ (if VAE) | $\mathcal{L} = \text{MSE}(\hat{\epsilon}, \epsilon)$ |
| **Vision encoder** | ResNet18 → 2D feature map → transformer tokens | ResNet18 → SpatialSoftmax → 32 keypoints → FC(64) |
| **Core architecture** | Transformer encoder-decoder (DETR-style) | 1D Conditional U-Net with FiLM conditioning |
| **State normalization** | MEAN_STD | MIN_MAX → $[-1, 1]$ |
| **Action normalization** | MEAN_STD | MIN_MAX → $[-1, 1]$ |
| **Inference speed** | Fast (1 forward pass per 20 steps) | Slow (100 denoising passes per 8 steps) |

### How Diffusion Policy predicts actions

Diffusion frames action prediction as a **denoising diffusion process**:

**Training:**
1. Ground-truth action trajectory: $a_0 \in \mathbb{R}^{H \times D_a}$
2. Sample noise: $\epsilon \sim \mathcal{N}(0, I)$
3. Sample diffusion timestep: $t \sim \text{Uniform}\{1, ..., 100\}$
4. Add noise: $a_t = \sqrt{\bar{\alpha}_t} \cdot a_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$
5. U-Net predicts the added noise: $\hat{\epsilon} = f_\theta(a_t, t, \text{obs})$
6. Loss: $\mathcal{L} = \|\hat{\epsilon} - \epsilon\|_2^2$

**Inference:**
1. Start from pure noise: $a_T \sim \mathcal{N}(0, I)$
2. Iteratively denoise for $T=100$ steps using the trained U-Net
3. Final $a_0$ is the predicted trajectory
4. Execute first 8 actions, discard the rest, then re-predict

### How batching works for both policies

**Identical data loading.** Every frame is a valid sample. The DataLoader randomly picks frames from any episode. When frame $t$ is picked:

| What's loaded | ACT | Diffusion |
|---|---|---|
| Observations | Frame $t$ only | Frames $t-1$ and $t$ |
| Actions | Frames $t$ through $t+19$ | Frames $t-1$ through $t+14$ |

Batches mix samples from different episodes, different timesteps, even different phases of the task — completely random. No sequential ordering within a batch. The temporal structure comes from the delta_timestamps, not from batch ordering.

### Can you swap ResNet for ViT?

In this codebase, **no**. ACT hardcodes:
```python
if not self.vision_backbone.startswith("resnet"):
    raise ValueError(...)
```
Options: `resnet18`, `resnet34`, `resnet50`. To use ViT requires rewriting the backbone integration code.

---

## 16. What the ACT Policy Sees: Complete Input Breakdown

### Answer: Both Images AND State (By Default)

The ACT model receives **images + proprioceptive state** at every forward pass. State is optional and can be disabled with `--disable_proprioceptive_obs`.

### Exact Inputs During Training

A training batch from the DataLoader contains:

| Key | Shape (Lift, batch=128) | Type | Used by ACT? |
|---|---|---|---|
| `observation.state` | `(128, 9)` | STATE | Yes — projected to transformer token |
| `observation.images.agentview` | `(128, 3, 84, 84)` | VISUAL | Yes — through ResNet backbone |
| `observation.images.robot0_eye_in_hand` | `(128, 3, 84, 84)` | VISUAL | Yes — through ResNet backbone |
| `action` | `(128, 20, 7)` | ACTION | Yes — ground-truth for L1 loss |
| `action_is_pad` | `(128, 20)` | Mask | Yes — zeros out loss on padded boundary actions |

### How ACT Processes Inputs

1. **Images → ResNet18 backbone → feature map → transformer tokens**
   - Each camera image → ResNet18 → `layer4` output: `(B, 512, 3, 3)` for 84×84 input
   - Projected via `encoder_img_feat_input_proj` (Conv2d) to `(B, dim_model, 3, 3)`
   - Flattened to 9 spatial tokens per camera
   - With 2 cameras (Lift): 18 image tokens

2. **State → linear projection → 1 transformer token**
   - `observation.state` (9D for Lift) → `encoder_robot_state_input_proj` → `(B, dim_model)`
   - Added as a single token to the encoder

3. **Latent → 1 transformer token**
   - During training: VAE encoder compresses the action into a latent z → `encoder_latent_input_proj(z)` → 1 token
   - During inference: zeros (no VAE) → `encoder_latent_input_proj(zeros)` → 1 token

4. **Total encoder input for Lift**: 18 image tokens + 1 state token + 1 latent token = **20 tokens**

5. **Decoder**: 20 learned query tokens → cross-attend to encoder output → predict action chunk `(B, 20, 7)`

### How Feature Detection Works (Auto-Sizing)

The policy does NOT hardcode any dimensions. Everything is auto-detected from the dataset:

```
Dataset metadata → features["observation.state"].shape = (9,)
                                    ↓
dataset_to_policy_features() → PolicyFeature(type=STATE, shape=(9,))
                                    ↓
ACTConfig.input_features["observation.state"] = PolicyFeature(shape=(9,))
                                    ↓
ACT.__init__() → self.encoder_robot_state_input_proj = nn.Linear(9, dim_model)
```

If you change to a 16D state dataset, the `nn.Linear(16, dim_model)` is created automatically. No code changes needed.

### Vision-Only Mode (`--disable_proprioceptive_obs`)

When you pass `--disable_proprioceptive_obs` to `train_bc_dexmg.py`:

1. At [line ~530](resfit/lerobot/scripts/train_bc_dexmg.py#L530): `observation.state` is removed from `ds_meta.features`
2. `make_policy()` → `dataset_to_policy_features()` finds no STATE feature → `robot_state_feature = None`
3. All `if self.config.robot_state_feature:` branches in ACT are skipped
4. **The model is vision-only** — 18 image tokens + 1 latent token = 19 tokens, no state input

This is useful for testing if the policy relies on proprioception or can work from vision alone.

### During Inference (Eval Rollouts)

`select_action()` receives the same keys from the environment:
- `observation.state` → from `RobosuiteGymWrapper` (robot sensor data, same keys as dataset)
- `observation.images.*` → from MuJoCo offscreen rendering at `camera_size=84`

Same normalization is applied. The model returns a single action (or pops from the action chunk queue).

---

## 17. Configuring State Composition (Increasing State Dimensions)

### State Is Baked Into the Dataset

You CANNOT change state composition at training time via CLI flags. The state vector is determined during dataset conversion and must be consistent between:

1. **Dataset** — `get_expected_low_dim_keys()` in [convert_robomimic_to_lerobot.py](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py) ~line 268
2. **Environment** — `_get_expected_low_dim_keys()` in [dexmg.py](resfit/dexmg/environments/dexmg.py) ~line 460

If these don't match, BC eval rollouts and RL training will crash or silently use wrong state dimensions.

### Current State Per Task

| Task | Robot | State Dim | Keys |
|---|---|---|---|
| Lift, Can, Square | Panda (single) | 9D | eef_pos(3) + eef_quat(4) + gripper_qpos(2) |
| Threading | Panda (single) | 9D | Same as above |
| TwoArmThreading, BoxCleanup, etc. | Panda (dual) | 18D | robot0 + robot1: eef_pos(3) + eef_quat(4) + gripper_qpos(2) each |
| TwoArmCoffee, Pouring, CanSort | GR1 Humanoid | 26D | right + left: eef_pos(3) + eef_quat(4) + gripper_qpos(6) each |

### Step-by-Step: Upgrade Lift to 16D

See the detailed recipe in [../algorithms/RESIDUAL_LEARNING.md](../algorithms/RESIDUAL_LEARNING.md#12-state-composition-and-how-to-change-it) — the process is identical for BC and RL.

**Summary:**
1. Edit `get_expected_low_dim_keys()` in the conversion script — add `"robot0_joint_pos"` (7D)
2. Edit `_get_expected_low_dim_keys()` in `dexmg.py` — add the same key
3. Re-render HDF5 with `dataset_states_to_obs.py` (raw HDF5 has all keys)
4. Re-convert to LeRobot format
5. Retrain BC with new dataset — ACT auto-detects 16D from `dataset.meta.features["observation.state"].shape`
6. Retrain RL — also auto-detects from env observation space

### What Auto-Adapts vs What Breaks

| Component | Auto-adapts? | Details |
|---|---|---|
| ACT `encoder_robot_state_input_proj` | **Yes** — `nn.Linear(state_dim, dim_model)` | Reads shape from dataset metadata |
| ACT normalization stats (μ, σ) | **Yes** — computed per-dim from new dataset | New dims get their own stats |
| RL `StateStandardizer` | **Yes** — reads from `dataset.meta.stats` | Auto-adapts to new dim count |
| RL critic/actor state input | **Yes** — reads `lowdim_dim` from env obs space | Auto-adapts |
| Old BC checkpoint | **No** — trained on 9D, cannot load into 16D model | Must retrain from scratch |
| Old RL checkpoint | **No** — network dimensions mismatch | Must retrain from scratch |

### Example: 25D State (Lift with Velocities)

```python
# In both conversion script and dexmg.py:
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

**Caveat:** Velocities can be noisier than positions. Check if the added signal actually helps by comparing eval success rates.

---

## 18. Configuring Image Resolution

### Current Resolution: 84×84

All images are 84×84 throughout the pipeline — in the dataset, during BC training, during BC eval, and during RL training. This is NOT a single config flag — it's set independently at each stage.

### Where Resolution Is Determined

| Stage | Where It's Set | How to Change |
|---|---|---|
| **Dataset** | `dataset_states_to_obs.py --camera_height 84 --camera_width 84` | Re-render HDF5 with new size |
| **BC training images** | From dataset (no resize) | Change dataset |
| **BC eval rollouts** | `--eval_camera_size 84` on CLI | Change CLI arg |
| **RL env rendering** | `camera_size=84` default in `dexmg.py` | Edit code or pass parameter |

### Does the ACT Model Care About Image Size?

**No.** The ResNet18 backbone is fully convolutional — it accepts any spatial resolution. The output feature map scales:

| Input | ResNet layer4 Output | Spatial Tokens (per camera) |
|---|---|---|
| 84×84 | 3×3 | 9 |
| 96×96 | 3×3 | 9 |
| 128×128 | 4×4 | 16 |
| 224×224 | 7×7 | 49 |

The transformer encoder handles variable sequence lengths, so **no ACT code changes are needed** for different resolutions — just retrain with the new dataset.

### Step-by-Step: Switching to 128×128

```bash
# 1. Re-render HDF5 at 128×128
cd deps/robomimic
python robomimic/scripts/dataset_states_to_obs.py \
    --dataset datasets/lift/mh/demo_v15.hdf5 \
    --output_name image_128.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 128 --camera_width 128

# 2. Convert to LeRobot format
cd ../..
python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
    --dataset deps/robomimic/datasets/lift/mh/image_128.hdf5 \
    --output_dir ~/lerobot_datasets/lift-mh-128

# 3. Train BC
bash scripts/train_bc_lift.sh   # ← change DATASET to the new path
# Also set: EVAL_CAMERA_SIZE=128

# 4. For RL (if proceeding to residual RL), edit dexmg.py or pass camera_size=128
```

### Tradeoffs

| Resolution | BC training speed | RL VRAM per env | Visual detail | Practical advice |
|---|---|---|---|---|
| 64×64 | Fastest | ~50 MB | Low — may lose small details | Only for quick prototyping |
| **84×84** | Fast | ~100 MB | **Good enough for tabletop** | **Default — recommended** |
| 128×128 | Medium | ~200 MB | Better for small objects | Worthwhile if 84 isn't enough |
| 224×224 | Slow | ~500 MB | Maximum detail | Overkill for most robosuite tasks |

### Warning: BC and RL Must Use the Same Resolution

If BC was trained on 84×84 images and RL renders at 128×128, the frozen BC policy will receive images at a different resolution than it was trained on. ResNet doesn't crash (it's resolution-agnostic) but the **feature statistics will be off** — the BC base actions will be lower quality, degrading residual RL performance.

**Always match:** dataset resolution = BC eval resolution = RL env resolution.

---

## 19. File Reference Map

### Core training

| File | Description |
|---|---|
| [resfit/lerobot/scripts/train_bc_dexmg.py](resfit/lerobot/scripts/train_bc_dexmg.py) | Main training script — CLI, loop, eval, checkpoints |
| [resfit/lerobot/policies/factory.py](resfit/lerobot/policies/factory.py) | `make_policy()`, `make_policy_config()` |
| [resfit/lerobot/configs/policies.py](resfit/lerobot/configs/policies.py) | `PreTrainedConfig` base class |
| [resfit/lerobot/policies/pretrained.py](resfit/lerobot/policies/pretrained.py) | `PreTrainedPolicy` base (save/load) |
| [resfit/lerobot/utils/load_policy.py](resfit/lerobot/utils/load_policy.py) | Checkpoint save/load/download |

### ACT policy

| File | Description |
|---|---|
| [resfit/lerobot/policies/act/configuration_act.py](resfit/lerobot/policies/act/configuration_act.py) | `ACTConfig` — all hyperparameters |
| [resfit/lerobot/policies/act/modeling_act.py](resfit/lerobot/policies/act/modeling_act.py) | `ACTPolicy`, `ACT`, encoder/decoder layers |

### Diffusion policy

| File | Description |
|---|---|
| [resfit/lerobot/policies/diffusion/configuration_diffusion.py](resfit/lerobot/policies/diffusion/configuration_diffusion.py) | `DiffusionConfig` — all hyperparameters |
| [resfit/lerobot/policies/diffusion/modeling_diffusion.py](resfit/lerobot/policies/diffusion/modeling_diffusion.py) | `DiffusionPolicy`, U-Net, RGB encoder |

### Environment

| File | Description |
|---|---|
| [resfit/dexmg/environments/dexmg.py](resfit/dexmg/environments/dexmg.py) | `RobosuiteGymWrapper`, `VectorizedEnvWrapper`, `create_vectorized_env()` |

### Dataset + transforms + normalization (deps/)

| File | Description |
|---|---|
| [deps/lerobot/lerobot/common/datasets/lerobot_dataset.py](deps/lerobot/lerobot/common/datasets/lerobot_dataset.py) | `LeRobotDataset` — loading, `__getitem__`, video decode |
| [deps/lerobot/lerobot/common/datasets/factory.py](deps/lerobot/lerobot/common/datasets/factory.py) | `resolve_delta_timestamps()` |
| [deps/lerobot/lerobot/common/datasets/transforms.py](deps/lerobot/lerobot/common/datasets/transforms.py) | `ImageTransforms`, `ImageTransformsConfig` |
| [deps/lerobot/lerobot/common/policies/normalize.py](deps/lerobot/lerobot/common/policies/normalize.py) | `Normalize`, `Unnormalize` — MEAN_STD / MIN_MAX |

### MuJoCo assets

| Path | Content |
|---|---|
| `deps/robosuite/robosuite/models/assets/robots/` | Robot MJCF XMLs (Panda, GR1, UR5e, etc.) |
| `deps/robosuite/robosuite/models/assets/grippers/` | Gripper XMLs |
| `deps/robosuite/robosuite/models/assets/arenas/` | Arena/table XMLs |
| `deps/dexmimicgen/dexmimicgen/models/assets/objects/` | Task objects (coffee, cabinet, etc.) |
