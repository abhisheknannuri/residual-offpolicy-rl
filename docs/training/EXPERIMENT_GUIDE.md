# Experiment Configuration Guide

How to configure this codebase for different Robosuite environments, including new tasks like `Lift` with Franka (Panda). Covers both **BC training** and **Residual RL fine-tuning**, reward switching, and custom reward models.

---

## Table of Contents

1. [Supported Environments](#1-supported-environments)
2. [Configuring BC Training for a New Task](#2-configuring-bc-training-for-a-new-task)
3. [Configuring Residual RL for a New Task](#3-configuring-residual-rl-for-a-new-task)
4. [Full Worked Example: Lift with Panda](#4-full-worked-example-lift-with-panda)
5. [Switching from Sparse to Dense Reward](#5-switching-from-sparse-to-dense-reward)
6. [Using a Custom Reward Model](#6-using-a-custom-reward-model)
7. [Quick Reference: What to Change per Task](#7-quick-reference-what-to-change-per-task)

---

## 1. Supported Environments

All supported environments are registered in [resfit/dexmg/environments/dexmg.py](resfit/dexmg/environments/dexmg.py#L37-L62) in the `ENV_ROBOTS` dictionary:

| Environment Name | Robot(s) | Type | Horizon |
|---|---|---|---|
| `Lift` | `Panda` | Single-arm | 100 |
| `PickPlaceCan` (alias `Can`) | `Panda` | Single-arm | 200 |
| `NutAssemblySquare` (alias `Square`) | `Panda` | Single-arm | 300 |
| `Threading` | `Panda` | Single-arm (MimicGen) | 500 |
| `TwoArmThreading` | `Panda`, `Panda` | Dual-arm | 300 |
| `TwoArmThreePieceAssembly` | `Panda`, `Panda` | Dual-arm | 500 |
| `TwoArmTransport` | `Panda`, `Panda` | Dual-arm | 800 |
| `TwoArmLiftTray` | `PandaDexRH`, `PandaDexLH` | Dual Dexterous | 650 |
| `TwoArmBoxCleanup` | `PandaDexRH`, `PandaDexLH` | Dual Dexterous | 300 |
| `TwoArmDrawerCleanup` | `PandaDexRH`, `PandaDexLH` | Dual Dexterous | 1000 |
| `TwoArmCoffee` | `GR1FixedLowerBody` | Humanoid | 400 |
| `TwoArmPouring` | `GR1FixedLowerBody` | Humanoid | 400 |
| `TwoArmCanSortRandom` | `GR1FixedLowerBody` | Humanoid | 400 |

**Aliases** (mapped at [dexmg.py](resfit/dexmg/environments/dexmg.py#L104-L108)):
- `"Can"` → `"PickPlaceCan"`
- `"Square"` → `"NutAssemblySquare"`
- `"Transport"` → `"TwoArmTransport"`

Horizons are defined at [dexmg.py](resfit/dexmg/environments/dexmg.py#L130-L143). Any environment not in that dict defaults to 1000.

---

## 2. Configuring BC Training for a New Task

The BC training script is [scripts/train_bc_act.sh](scripts/train_bc_act.sh), which calls [resfit/lerobot/scripts/train_bc_dexmg.py](resfit/lerobot/scripts/train_bc_dexmg.py).

### What you MUST change in `train_bc_act.sh`

| Variable | What it does | Example for Lift |
|---|---|---|
| `DATASET` | HuggingFace dataset ID for demonstrations | `"ankile/robomimic-mh-lift-image"` or your own |
| `EVAL_ENV` | Robosuite environment name for rollout eval | `"Lift"` |
| `EVAL_VIDEO_KEY` | Camera key for video recording | `"observation.images.agentview"` |
| `WANDB_PROJECT` | WandB project name | `"robomimic-lift-bc"` |

### What you MAY want to change

| Variable | Default | When to change |
|---|---|---|
| `STEPS` | `200000` | Simpler tasks (Lift) may converge with `100000` |
| `BATCH_SIZE` | `128` | Increase to `256` on 48GB GPU |
| `ROLLOUT_FREQ` | `5000` | Lower for more frequent eval |
| `EVAL_NUM_EPISODES` | `100` | `20` for quick sanity checks |
| `EVAL_NUM_ENVS` | `4` | Increase to `8-16` on bigger GPU |

### How the dataset / camera / state keys are resolved

The BC script auto-detects everything from the dataset metadata:
- **Camera keys**: Inferred from dataset features (keys matching `observation.images.*`)
- **State keys**: Inferred from dataset features (keys matching `observation.state`)
- **Action space**: Inferred from dataset features

You do NOT need to manually specify cameras or state dimensions for BC. The policy config is built by [resfit/lerobot/scripts/train_bc_dexmg.py](resfit/lerobot/scripts/train_bc_dexmg.py#L288-L295) using the dataset metadata.

### Providing your own dataset

If you have a local LeRobot-format dataset instead of a HuggingFace one, just point `DATASET` to the local path. The `LeRobotDataset` class in LeRobot supports both.

For converting Robomimic HDF5 datasets to LeRobot format, see [resfit/lerobot/dataset/convert_robomimic_to_lerobot.py](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py).

---

## 3. Configuring Residual RL for a New Task

The RL training script is [scripts/train_residual_rl.sh](scripts/train_residual_rl.sh), which calls [resfit/rl_finetuning/scripts/train_residual_td3.py](resfit/rl_finetuning/scripts/train_residual_td3.py) via Hydra configs.

### Two approaches: use an existing Hydra config OR override via CLI

#### Approach A: Reuse a generic config + CLI overrides (quick, no code changes)

Use `--config-name=residual_td3_dexmg_config` (the generic base config) and override everything from the shell script:

```bash
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name="residual_td3_dexmg_config" \
    task="Lift" \
    rl_camera='["observation.images.agentview","observation.images.robot0_eye_in_hand"]' \
    video_key="observation.images.agentview" \
    base_policy.wandb_id="YOUR_BC_PROJECT/YOUR_BC_RUN_ID" \
    base_policy.wt_type="best" \
    offline_data.name="ankile/robomimic-mh-lift-image" \
    offline_data.num_episodes=300 \
    wandb.project="robomimic-lift-residual-td3" \
    algo.total_timesteps=300000 \
    # ... other overrides
```

#### Approach B: Add a new Hydra config dataclass (recommended for repeated use)

Add a new config class in [resfit/rl_finetuning/config/residual_td3.py](resfit/rl_finetuning/config/residual_td3.py) and register it with Hydra. Here's the pattern:

**Step 1** — Add a new dataclass in [residual_td3.py](resfit/rl_finetuning/config/residual_td3.py) (before the `# Register with Hydra` section at [line 285](resfit/rl_finetuning/config/residual_td3.py#L285)):

```python
@dataclass
class ResidualTD3LiftConfig(ResidualTD3DexmgConfig):
    task: str = "Lift"

    rl_camera: list[str] = field(
        default_factory=lambda: [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
        ]
    )

    algo: ResidualTD3AlgoConfig = field(
        default_factory=lambda: ResidualTD3AlgoConfig(
            total_timesteps=300_000,
        )
    )

    wandb: WandBConfig = field(
        default_factory=lambda: WandBConfig(project="robomimic-lift-residual-td3")
    )

    offline_data: OfflineDataConfig = field(
        default_factory=lambda: OfflineDataConfig(
            name="ankile/robomimic-mh-lift-image",  # or your dataset
            num_episodes=300,
        )
    )

    base_policy: BasePolicyConfig = field(
        default_factory=lambda: BasePolicyConfig(
            wandb_id="YOUR_BC_PROJECT/YOUR_BC_RUN_ID",
            wt_type="best",
            wt_version="latest",
        )
    )
```

**Step 2** — Register with Hydra at [line 287](resfit/rl_finetuning/config/residual_td3.py#L287):

```python
cs.store(name="residual_td3_lift_config", node=ResidualTD3LiftConfig)
```

**Step 3** — Use it in the shell script:

```bash
CONFIG_NAME="residual_td3_lift_config"
```

### Existing Hydra configs (already registered)

These are defined in [resfit/rl_finetuning/config/residual_td3.py](resfit/rl_finetuning/config/residual_td3.py#L287-L292):

| Config Name | Task | Dataset |
|---|---|---|
| `residual_td3_dexmg_config` | `Can` (generic base) | `ankile/robomimic-mh-can-image` |
| `residual_td3_can_config` | `Can` | `ankile/robomimic-mh-can-image` |
| `residual_td3_square_config` | `Square` | `ankile/robomimic-mh-square-image` |
| `residual_td3_box_clean_config` | `TwoArmBoxCleanup` | `ankile/dexmg-two-arm-box-cleanup` |
| `residual_td3_coffee_config` | `TwoArmCoffee` | `ankile/dexmg-two-arm-coffee` |
| `residual_td3_two_arm_cansort_config` | `TwoArmCanSortRandom` | `ankile/dexmg-two-arm-can-sort-random` |

### What you MUST change in `train_residual_rl.sh`

| Variable | What it does | Where it maps |
|---|---|---|
| `BASE_WANDB_ID` | Your trained BC policy from Step 1 | `base_policy.wandb_id` |
| `CONFIG_NAME` | Hydra config name (see table above) | `--config-name` |
| `WANDB_PROJECT` | WandB project for RL runs | `wandb.project` |

### What you MAY want to change

| Variable | Default | Notes |
|---|---|---|
| `TOTAL_TIMESTEPS` | `500000` | `300000` for simpler tasks like Lift |
| `OFFLINE_EPISODES` | `1000` | Match your dataset size |
| `ACTION_SCALE` | `0.2` | How much residual correction is allowed |
| `ACTOR_LR` | `1e-6` | Keep very low to avoid destabilizing base policy |
| `BUFFER_SIZE` | `80000` | Increase to `200000` on bigger machines |

### How cameras are resolved for RL

Unlike BC, the RL config **requires explicit camera specification** via `rl_camera`. This is the list of observation keys fed to the critic/actor networks.

Camera keys for each task type are defined in [dexmg.py](resfit/dexmg/environments/dexmg.py#L384-L425):

| Task Type | Camera Keys |
|---|---|
| Single-arm Panda (Lift, Can, Square, Threading) | `agentview_image`, `robot0_eye_in_hand_image` |
| Dual-arm Panda (TwoArmThreading, etc.) | `agentview_image`, `robot0_eye_in_hand_image`, `robot1_eye_in_hand_image` |
| TwoArmTransport | `shouldercamera0_image`, `shouldercamera1_image` |
| Humanoid (Coffee, Pouring) | `agentview_image`, `robot0_eye_in_left_hand_image`, `robot0_eye_in_right_hand_image` |
| TwoArmCanSortRandom | `frontview_image`, `robot0_eye_in_left_hand_image`, `robot0_eye_in_right_hand_image` |

In the RL config, you reference these as `observation.images.<name>` (without `_image` suffix):
```python
rl_camera: list[str] = field(
    default_factory=lambda: [
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
    ]
)
```

### How low-dimensional state keys are resolved

Automatically determined by `_get_expected_low_dim_keys()` at [dexmg.py](resfit/dexmg/environments/dexmg.py#L441-L470):

| Task Type | State Keys |
|---|---|
| Single-arm Panda | `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` |
| Dual-arm Panda | Above + `robot1_eef_pos`, `robot1_eef_quat`, `robot1_gripper_qpos` |
| Humanoid | `robot0_right_eef_pos/quat`, `robot0_right_gripper_qpos`, `robot0_left_eef_pos/quat`, `robot0_left_gripper_qpos` |

These are resolved automatically from the environment name — no manual config needed.

---

## 4. Full Worked Example: Lift with Panda

### Step 1: Get/create a dataset

You need a LeRobot-format dataset with demonstrations. Options:
- Use an existing HuggingFace dataset: `ankile/robomimic-mh-lift-image` (if it exists)
- Convert a Robomimic HDF5: `python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`
- Collect your own demonstrations

### Step 2: BC Training

Create/edit `scripts/train_bc_lift.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASET="ankile/robomimic-mh-lift-image"   # Your HF dataset
POLICY="act"
STEPS=100000                                # Lift is simpler, 100K may suffice
BATCH_SIZE=128
EVAL_ENV="Lift"                             # Must match ENV_ROBOTS key
ROLLOUT_FREQ=5000
EVAL_NUM_ENVS=4
EVAL_NUM_EPISODES=100
EVAL_VIDEO_KEY="observation.images.agentview"
EVAL_RENDER_SIZE=224
WANDB_PROJECT="robomimic-lift-bc"
WANDB_ENABLE="--wandb_enable"
NO_CLEANUP="--no_cleanup"

python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset "${DATASET}" \
    --policy "${POLICY}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --wandb_project "${WANDB_PROJECT}" \
    --eval_env "${EVAL_ENV}" \
    --rollout_freq "${ROLLOUT_FREQ}" \
    --eval_video_key "${EVAL_VIDEO_KEY}" \
    --eval_render_size "${EVAL_RENDER_SIZE}" \
    --eval_num_envs "${EVAL_NUM_ENVS}" \
    --eval_num_episodes "${EVAL_NUM_EPISODES}" \
    ${WANDB_ENABLE} \
    ${NO_CLEANUP}
```

Run it:
```bash
bash scripts/train_bc_lift.sh
```

After training, note the WandB `project/run_id` (e.g., `robomimic-lift-bc/abc12345`).

### Step 3: Residual RL Fine-tuning

**Option A: No code changes (CLI overrides only)**

Edit `scripts/train_residual_rl.sh`:

```bash
BASE_WANDB_ID="robomimic-lift-bc/abc12345"    # From Step 2
BASE_WT_TYPE="best"
CONFIG_NAME="residual_td3_dexmg_config"       # Generic base config
TOTAL_TIMESTEPS=300000
OFFLINE_EPISODES=300                           # Match your dataset
WANDB_PROJECT="robomimic-lift-residual-td3"
```

And add the task override to the python command:

```bash
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name="${CONFIG_NAME}" \
    task="Lift" \
    base_policy.wandb_id="${BASE_WANDB_ID}" \
    # ... rest of overrides
```

**Option B: Add a proper Hydra config** (see [Section 3, Approach B](#approach-b-add-a-new-hydra-config-dataclass-recommended-for-repeated-use))

---

## 5. Switching from Sparse to Dense Reward

### Current reward system

ALL tasks in this codebase use **sparse binary reward** (0.0 or 1.0). The reward flows through:

1. `robosuite_env.reward(action)` — calls `_check_success()`, returns 1.0 if success
2. [RobosuiteGymWrapper.step()](resfit/dexmg/environments/dexmg.py#L309-L332) — passes reward through unchanged
3. [BasePolicyVecEnvWrapper.step()](resfit/rl_finetuning/wrappers/residual_env_wrapper.py#L142) — passes reward through unchanged
4. Training loop replay buffer — stores reward as-is

### How robosuite dense reward works

Many robosuite tasks have built-in dense reward shaping (reach → grasp → lift → hover stages). It's controlled by the `reward_shaping` parameter passed at environment creation time. By default, robosuite sets `reward_shaping=False`.

Tasks with actual dense reward implementations:
- **Lift**: Full 4-stage dense reward (reach, grasp, lift, hover)
- **PickPlaceCan**: Full 4-stage dense reward
- **NutAssemblySquare**: Full dense reward
- **TwoArmLiftTray**: Dense reward for tilt direction, lift height, reach, grasp

### How to enable dense reward

**Modify** [resfit/dexmg/environments/dexmg.py](resfit/dexmg/environments/dexmg.py#L187-L198) where `robosuite.make()` is called. Add `reward_shaping=True` to the `env_kwargs`:

```python
# In RobosuiteGymWrapper.__init__(), around line 187
env_kwargs = {
    "env_name": env_name,
    "robots": robots,
    # ... existing kwargs ...
    "reward_shaping": True,     # ← ADD THIS LINE
    "reward_scale": 1.0,        # ← optional, controls reward magnitude
}
```

### Important: Update the termination logic

When using dense reward, `reward == 1.0` no longer reliably indicates success. You need to modify the step function at [dexmg.py line 319](resfit/dexmg/environments/dexmg.py#L319):

```python
# BEFORE (sparse reward):
success = reward == 1.0

# AFTER (dense reward) — use robosuite's actual success check:
success = self.env._check_success()
```

The full modified `step()` method at [dexmg.py lines 302-332](resfit/dexmg/environments/dexmg.py#L302-L332) would look like:

```python
def step(self, action):
    if hasattr(action, "cpu"):
        action = action.cpu().numpy()
    if action.ndim > 1:
        action = action[0]

    obs, reward, done, info = self.env.step(action)
    self.episode_steps += 1

    processed_obs = self._process_obs(obs)
    reward_scalar = float(reward)

    # Use _check_success() for termination (works with both sparse and dense)
    success = self.env._check_success()
    terminated_scalar = bool(success)
    truncated_scalar = bool(done)

    if terminated_scalar or truncated_scalar:
        info = {
            **info,
            "success": success,
            "episode_steps": self.episode_steps,
        }
        self.episode_steps = 0

    return processed_obs, reward_scalar, terminated_scalar, truncated_scalar, info
```

### Important: Adjust the critic's value range

With dense reward, the critic should model a wider value range. Adjust `v_max` in the critic loss config at [resfit/rl_finetuning/config/rlpd.py](resfit/rl_finetuning/config/rlpd.py#L29-L34):

```python
@dataclass
class CriticLossCfg:
    v_min: float = 0.0
    v_max: float = 1.0    # ← change to match your reward range, e.g. 5.0 for dense
```

Or override via Hydra CLI:
```bash
agent.critic.loss.v_max=5.0
```

### Dense reward clips and gamma

Dense rewards are typically in [0, 1] per step, so cumulative return scales with episode length. With `gamma=0.995` and horizon 100 (Lift), max return ≈ 100. Consider adjusting `gamma` or `v_max` accordingly.

---

## 6. Using a Custom Reward Model

If you have a learned reward model (e.g., from preference learning, inverse RL, etc.), the cleanest place to plug it in is the `step()` method of `RobosuiteGymWrapper`.

### Where to modify

[resfit/dexmg/environments/dexmg.py](resfit/dexmg/environments/dexmg.py#L302-L332), in the `step()` method of `RobosuiteGymWrapper`:

```python
def step(self, action):
    if hasattr(action, "cpu"):
        action = action.cpu().numpy()
    if action.ndim > 1:
        action = action[0]

    obs, reward, done, info = self.env.step(action)
    self.episode_steps += 1
    processed_obs = self._process_obs(obs)

    # ── Custom reward model ──────────────────────────────────────────
    # Replace or augment the environment reward with your model
    # Option 1: Replace entirely
    reward_scalar = float(self.reward_model(obs, action))

    # Option 2: Combine with environment reward
    # env_reward = float(reward)
    # custom_reward = float(self.reward_model(obs, action))
    # reward_scalar = env_reward + custom_reward_weight * custom_reward

    # ── Success detection (keep env's check for termination) ─────────
    success = self.env._check_success()
    terminated_scalar = bool(success)
    truncated_scalar = bool(done)

    if terminated_scalar or truncated_scalar:
        info = {**info, "success": success, "episode_steps": self.episode_steps}
        self.episode_steps = 0

    return processed_obs, reward_scalar, terminated_scalar, truncated_scalar, info
```

### How to pass the reward model to the wrapper

Modify the `__init__` of `RobosuiteGymWrapper` at [dexmg.py line 85](resfit/dexmg/environments/dexmg.py#L85) to accept and store it:

```python
def __init__(self, env_name, num_envs=1, ..., reward_model=None):
    # ... existing init code ...
    self.reward_model = reward_model
```

Then pass it through the factory function `make_dexmimicgen_env` at [dexmg.py line 559](resfit/dexmg/environments/dexmg.py#L559):

```python
def make_dexmimicgen_env(env_name, ..., reward_model=None):
    def _make():
        return RobosuiteGymWrapper(
            env_name=env_name,
            ...,
            reward_model=reward_model,
        )
    return _make
```

And `create_vectorized_env` at [dexmg.py line 681](resfit/dexmg/environments/dexmg.py#L681):

```python
def create_vectorized_env(env_name, ..., reward_model=None):
    env_fns = []
    for env_id in range(num_envs):
        env_fns.append(make_dexmimicgen_env(
            env_name, ..., reward_model=reward_model
        ))
    # ...
```

Finally, pass it from the training script where `create_vectorized_env` is called at [train_residual_td3.py line 232](resfit/rl_finetuning/scripts/train_residual_td3.py#L232):

```python
vec_env = create_vectorized_env(
    env_name=env_name,
    ...,
    reward_model=my_reward_model,
)
```

### Important considerations for custom reward models

1. **Keep the model on CPU** if used inside the env wrapper (env runs on CPU for MuJoCo). Or `.cpu()` outputs before returning.
2. **Normalize your reward** to a reasonable range. The default critic is calibrated for [0, 1] (`v_min=0.0`, `v_max=1.0` in [rlpd.py](resfit/rl_finetuning/config/rlpd.py#L29-L34)). Adjust `v_max` via Hydra override if your reward range differs.
3. **Keep success-based termination** (`self.env._check_success()`) even with custom reward — this controls episode ending and success rate metrics.

---

## 7. Quick Reference: What to Change per Task

### Checklist for any new Robosuite task

1. **Is the task in `ENV_ROBOTS`?** ([dexmg.py line 37](resfit/dexmg/environments/dexmg.py#L37))
   - Yes → proceed
   - No → add it: `"YourTask": ["Panda"]` (or whatever robot)

2. **Does it need DexMimicGen import?** ([dexmg.py line 160](resfit/dexmg/environments/dexmg.py#L160))
   - Only if it's a DexMimicGen task. Standard robosuite tasks (Lift, Can, Square) don't need this.

3. **Does it have a horizon override?** ([dexmg.py line 130](resfit/dexmg/environments/dexmg.py#L130))
   - Add it to the horizon dict if the default (1000) isn't right.

4. **Are camera keys correct?** ([dexmg.py line 384](resfit/dexmg/environments/dexmg.py#L384))
   - `_get_expected_image_keys()` needs to return the right cameras. Single-arm Panda tasks are already handled.

5. **Are low-dim state keys correct?** ([dexmg.py line 441](resfit/dexmg/environments/dexmg.py#L441))
   - `_get_expected_low_dim_keys()` needs to return the right state keys. Single-arm Panda tasks are already handled.

6. **Do you have a dataset?**
   - HuggingFace dataset ID or local LeRobot dataset path.

### Changes by file

| File | What | When |
|---|---|---|
| [resfit/dexmg/environments/dexmg.py](resfit/dexmg/environments/dexmg.py) | `ENV_ROBOTS`, horizons, cameras, state keys, `reward_shaping` | Adding a new task or changing reward |
| [scripts/train_bc_act.sh](scripts/train_bc_act.sh) | `DATASET`, `EVAL_ENV`, `WANDB_PROJECT` | BC training for new task |
| [resfit/rl_finetuning/config/residual_td3.py](resfit/rl_finetuning/config/residual_td3.py) | New config dataclass + `cs.store()` | RL training for new task (Approach B) |
| [scripts/train_residual_rl.sh](scripts/train_residual_rl.sh) | `BASE_WANDB_ID`, `CONFIG_NAME`, `WANDB_PROJECT`, task overrides | RL training for new task |
| [resfit/rl_finetuning/wrappers/residual_env_wrapper.py](resfit/rl_finetuning/wrappers/residual_env_wrapper.py) | (usually no changes needed) | Only if custom obs augmentation |

### For Lift specifically

`Lift` is already in `ENV_ROBOTS` → `["Panda"]`, has horizon `100`, and camera/state key functions already handle single-arm Panda tasks. All you need is:

1. A dataset (e.g., `ankile/robomimic-mh-lift-image`)
2. Set `EVAL_ENV="Lift"` and `DATASET="..."` in BC script
3. After BC training, use the WandB run ID as `BASE_WANDB_ID` in RL script
4. Set `CONFIG_NAME="residual_td3_dexmg_config"` with `task="Lift"` override, or create `ResidualTD3LiftConfig`
