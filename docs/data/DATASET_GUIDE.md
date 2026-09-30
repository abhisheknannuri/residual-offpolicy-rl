# Dataset Guide — From Raw Demos to Training

Everything about getting, creating, converting, and using datasets in this codebase. Covers Robomimic benchmark tasks (Lift, Can, Square) and DexMimicGen tasks (TwoArmCoffee, etc.).

---

## Table of Contents

1. [How Datasets Work in This Codebase](#1-how-datasets-work-in-this-codebase)
2. [Pre-Converted Datasets on HuggingFace (Ready to Use)](#2-pre-converted-datasets-on-huggingface-ready-to-use)
3. [What's Inside a Dataset](#3-whats-inside-a-dataset)
4. [Getting Robomimic HDF5 Files](#4-getting-robomimic-hdf5-files)
5. [Generating Image Observations from Raw HDF5](#5-generating-image-observations-from-raw-hdf5)
6. [Converting HDF5 to LeRobot Format](#6-converting-hdf5-to-lerobot-format)
7. [Mixing MH + PH (or Any Subsets)](#7-mixing-mh--ph-or-any-subsets)
8. [Using a Local Dataset for Training](#8-using-a-local-dataset-for-training)
9. [Camera Selection — Can I Use Only 1 or 2 Cameras?](#9-camera-selection--can-i-use-only-1-or-2-cameras)
10. [State Observations — What's Included and Can I Change It?](#10-state-observations--whats-included-and-can-i-change-it)
11. [Action Space by Task](#11-action-space-by-task)
12. [Collecting Your Own Demonstrations](#12-collecting-your-own-demonstrations)
13. [End-to-End Example: Lift from Scratch](#13-end-to-end-example-lift-from-scratch)

---

## 1. How Datasets Work in This Codebase

The codebase uses **LeRobot format** datasets. LeRobot is HuggingFace's robotics dataset library. A LeRobot dataset is a directory with:

```
my-dataset/
├── meta/
│   ├── info.json          # Metadata: fps, total_episodes, features, shapes
│   ├── episodes.jsonl     # Per-episode info (length, task)
│   └── tasks.jsonl        # Task descriptions
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # Tabular data (actions, states, done flags)
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.agentview/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.robot0_eye_in_hand/
            ├── episode_000000.mp4
            └── ...
```

**Two ways datasets are loaded:**

1. **HuggingFace Hub ID** (e.g., `ankile/robomimic-mh-lift-image`) — auto-downloaded to `~/.cache/huggingface/lerobot/` on first use
2. **Local path** — just point to the directory

The loading happens at [resfit/lerobot/scripts/train_bc_dexmg.py](resfit/lerobot/scripts/train_bc_dexmg.py#L560-L565):
```python
dataset = LeRobotDataset(
    cfg.dataset,             # ← HF repo ID or local path
    delta_timestamps=delta_timestamps,
    download_videos=True,
    image_transforms=image_transforms,
)
```

For RL, the dataset is also loaded for the offline replay buffer at [resfit/rl_finetuning/scripts/train_residual_td3.py](resfit/rl_finetuning/scripts/train_residual_td3.py#L226):
```python
dataset = LeRobotDataset(cfg.offline_data.name)
```

---

## 2. Pre-Converted Datasets on HuggingFace (Ready to Use)

The codebase author (ankile) has already converted many datasets. Use these directly — no conversion needed.

### Robomimic Benchmark Tasks (Single-arm Panda)

| Task | Dataset Type | HuggingFace ID | Episodes | Frames | Cameras |
|---|---|---|---|---|---|
| **Lift** | PH (proficient-human) | `ankile/robomimic-ph-lift-image` | 200 | ~9,670 | agentview, robot0_eye_in_hand |
| **Lift** | MH (multi-human) | `ankile/robomimic-mh-lift-image` | 300 | ~31,127 | agentview, robot0_eye_in_hand |
| **Can** | PH | `ankile/robomimic-ph-can-image` | 200 | ~23,200 | agentview, robot0_eye_in_hand |
| **Can** | MH | `ankile/robomimic-mh-can-image` | 300 | ~62,756 | agentview, robot0_eye_in_hand |
| **Square** | PH | `ankile/robomimic-ph-square-image` | 200 | ~30,200 | agentview, robot0_eye_in_hand |
| **Square** | MH | `ankile/robomimic-mh-square-image` | 300 | ~80,700 | agentview, robot0_eye_in_hand |
| **Transport** | MH | `ankile/robomimic-mh-transport-image` | 300 | ~196,000 | shouldercamera0, shouldercamera1 |
| **Transport** | PH | `ankile/robomimic-ph-transport-image` | 200 | ~93,800 | shouldercamera0, shouldercamera1 |
| **Lift** | MG (machine-generated) | `ankile/robomimic-mg-lift-image` | ~1,500 | ~1,500 | agentview, robot0_eye_in_hand |

### DexMimicGen Tasks (Humanoid / Multi-arm)

| Task | HuggingFace ID | Episodes | Cameras |
|---|---|---|---|
| **TwoArmCoffee** | `ankile/dexmg-two-arm-coffee` | 1,000 | agentview, robot0_eye_in_left_hand, robot0_eye_in_right_hand |
| **TwoArmBoxCleanup** | `ankile/dexmg-two-arm-box-cleanup` | 1,000 | agentview, robot0_eye_in_hand, robot1_eye_in_hand |
| **TwoArmCanSortRandom** | `ankile/dexmg-two-arm-can-sort-random` | 1,000 | frontview, robot0_eye_in_left_hand, robot0_eye_in_right_hand |
| **TwoArmThreading** | `ankile/dexmg-two-arm-threading` | 1,000 | agentview, robot0_eye_in_hand, robot1_eye_in_hand |
| **TwoArmPouring** | `ankile/dexmg-two-arm-pouring` | 1,000 | agentview, robot0_eye_in_left_hand, robot0_eye_in_right_hand |
| **Threading** (MimicGen) | `ankile/mimicgen-d0-threading` | ~300 | agentview, robot0_eye_in_hand |

### What PH vs MH vs MG Means

- **PH (Proficient-Human)**: 200 demos from 1 skilled operator. Clean, high-quality data. Good baseline.
- **MH (Multi-Human)**: 300 demos from 6 operators of varying skill (2 "worse", 2 "okay", 2 "better"). More diverse, mixed quality. **This is the standard benchmark dataset** for testing algorithms that handle suboptimal data.
- **MG (Machine-Generated)**: Demos from RL policy checkpoints at various training stages. Mixed quality, larger quantity.

**For this codebase's residual RL approach, MH is the most commonly used type** (it's what the configs reference). The whole point of the residual RL is to improve a BC policy trained on mixed-quality data.

### Full list of datasets

Browse all 655+ datasets: https://huggingface.co/ankile/datasets

Search with filters: https://huggingface.co/ankile/datasets?search=robomimic

---

## 3. What's Inside a Dataset

Taking `ankile/robomimic-mh-lift-image` as an example, the `meta/info.json` tells you everything:

```json
{
    "total_episodes": 300,
    "total_frames": 31127,
    "fps": 20,
    "features": {
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["action_0", ..., "action_6"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [9],
            "names": [
                "robot0_eef_pos_0", "robot0_eef_pos_1", "robot0_eef_pos_2",
                "robot0_eef_quat_0", "robot0_eef_quat_1", "robot0_eef_quat_2", "robot0_eef_quat_3",
                "robot0_gripper_qpos_0", "robot0_gripper_qpos_1"
            ]
        },
        "observation.images.agentview": {
            "dtype": "video",
            "shape": [84, 84, 3]
        },
        "observation.images.robot0_eye_in_hand": {
            "dtype": "video",
            "shape": [84, 84, 3]
        },
        "next.done": {
            "dtype": "bool",
            "shape": [1]
        }
    }
}
```

**Key things to note:**
- **Images are 84x84x3** — this is the standard camera resolution for these tasks
- **State is 9D** for single-arm Panda: 3 (eef pos) + 4 (eef quat) + 2 (gripper qpos)
- **Actions are 7D** for single-arm Panda: 3 (delta pos) + 3 (delta rot axis-angle) + 1 (gripper)
- **FPS is 20** — the control frequency of the environment

---

## 4. Getting Robomimic HDF5 Files

If you need to create your own dataset (custom cameras, different episodes, merge MH+PH, etc.), the pipeline is:

**Raw HDF5 → Image HDF5 → LeRobot format**

### Step 1: Download the raw HDF5

Robomimic provides raw state-only HDF5 files that contain MuJoCo simulator states but NO observations. You render observations from these states.

**Option A: Use the robomimic download script**

```bash
# Install robomimic first
pip install robomimic

# Download Lift MH raw dataset
cd deps/robomimic   # or wherever robomimic is installed
python robomimic/scripts/download_datasets.py \
    --tasks lift \
    --dataset_types mh \
    --hdf5_types raw

# Download Lift PH raw dataset
python robomimic/scripts/download_datasets.py \
    --tasks lift \
    --dataset_types ph \
    --hdf5_types raw

# Download pre-extracted image datasets (skip Step 2)
python robomimic/scripts/download_datasets.py \
    --tasks lift \
    --dataset_types mh \
    --hdf5_types image
```

Available `--tasks`: `lift`, `can`, `square`, `transport`, `tool_hang`
Available `--dataset_types`: `ph`, `mh`, `mg`, `paired`
Available `--hdf5_types`: `raw`, `low_dim`, `image`

**Option B: Download directly from HuggingFace**

All raw HDF5 files are at: https://huggingface.co/datasets/amandlek/robomimic/tree/main/v1.5

The structure is: `v1.5/{task}/{dataset_type}/{hdf5_type}.hdf5`

Example paths:
- `v1.5/lift/ph/image.hdf5` — Lift PH with images already rendered
- `v1.5/lift/mh/demo_v15.hdf5` — Lift MH raw (states only, need to render)
- `v1.5/can/mh/image.hdf5` — Can MH with images

**Option C: Download pre-extracted image datasets directly**

If you just want the image HDF5 without doing the observation extraction yourself:
```bash
python robomimic/scripts/download_datasets.py \
    --tasks lift \
    --dataset_types mh \
    --hdf5_types image
```

This gives you the ready-to-convert `image.hdf5`.

---

## 5. Generating Image Observations from Raw HDF5

If you downloaded raw HDF5 (states only), you need to render camera observations. This is done with robomimic's `dataset_states_to_obs.py`:

```bash
python robomimic/scripts/dataset_states_to_obs.py \
    --dataset /path/to/demo_v15.hdf5 \
    --output_name image.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 84 \
    --camera_width 84
```

**Parameters explained:**

| Parameter | What it does |
|---|---|
| `--dataset` | Path to the raw HDF5 file |
| `--output_name` | Name of the output file (created in same directory) |
| `--done_mode 2` | Done flag on task success (standard for this codebase) |
| `--camera_names` | Which cameras to render. **This is where you choose your cameras** |
| `--camera_height/width` | Resolution. **84x84 is standard** for this codebase |
| `--compress` | (optional) Lossless compression, saves 5x storage |
| `--exclude-next-obs` | (optional) Skip next-obs, saves storage, fine for BC |

### Available camera names by robot type

**Single-arm Panda tasks (Lift, Can, Square):**
- `agentview` — third-person view of the workspace
- `robot0_eye_in_hand` — wrist camera on the robot's gripper
- `frontview` — front-facing camera
- `sideview` — side view
- `birdview` — top-down view

**Dual-arm Panda tasks (Transport, Threading):**
- `agentview`
- `robot0_eye_in_hand` — wrist camera on robot 0
- `robot1_eye_in_hand` — wrist camera on robot 1
- `shouldercamera0` — shoulder-mounted camera on robot 0
- `shouldercamera1` — shoulder-mounted camera on robot 1

**You can choose ANY cameras you want.** The conversion script will include whatever you put in `--camera_names`. The existing datasets use 2-3 cameras, but you could use 1 or 5 — it's your choice.

### Example: Lift with only agentview (1 camera)

```bash
python robomimic/scripts/dataset_states_to_obs.py \
    --dataset datasets/lift/mh/demo_v15.hdf5 \
    --output_name image_agentview_only.hdf5 \
    --done_mode 2 \
    --camera_names agentview \
    --camera_height 84 \
    --camera_width 84
```

### Example: Can with 3 cameras + higher resolution

```bash
python robomimic/scripts/dataset_states_to_obs.py \
    --dataset datasets/can/mh/demo_v15.hdf5 \
    --output_name image_3cam_128.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand frontview \
    --camera_height 128 \
    --camera_width 128
```

The resulting HDF5 will have image observations under `obs/{camera_name}_image` for each camera you specified.

---

## 6. Converting HDF5 to LeRobot Format

Once you have an image HDF5 file, convert it to LeRobot format using the script at [resfit/lerobot/dataset/convert_robomimic_to_lerobot.py](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py):

```bash
python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
    --dataset /path/to/image.hdf5 \
    --output_dir /path/to/output/my-lift-dataset \
    --repo_id "your-hf-username/my-lift-mh-image"   # optional: to upload to HF Hub
```

### All conversion options

| Argument | What it does |
|---|---|
| `--dataset` | Path to the Robomimic image HDF5 file |
| `--output_dir` | Where to write the LeRobot dataset |
| `--repo_id` | (optional) HuggingFace repo ID — if set, uploads after conversion |
| `--filter_key` | (optional) Use only a subset of demos (e.g., `train` from mask/train) |
| `--max_episodes` | (optional) Convert only the first N episodes |
| `--exclude-episodes` | (optional) Skip specific episodes (1-indexed) |
| `--train_ratio` | (optional) Train/test split ratio (default 1.0 = all train) |

### What the conversion does automatically

The script at [convert_robomimic_to_lerobot.py](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L100-L140) auto-detects:

1. **Environment name** from the HDF5 metadata (`data.attrs["env_args"]`)
2. **Expected image keys** from the env name — see `get_expected_image_keys()` at [line 203](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L203). It filters to only include cameras that match the expected keys for that task type.
3. **Expected state keys** from the env name — see `get_expected_low_dim_keys()` at [line 253](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L253). It selects the right proprioceptive observations and concatenates them.
4. **Action naming** from action dimension — see `get_action_names()` at [line 291](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L291).

**Important**: If you used custom/non-standard cameras during observation extraction (Step 5), you may need to update the `get_expected_image_keys()` function at [line 203](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L203) to include your camera names. Otherwise the converter will filter them out.

### Example: Convert Lift MH to LeRobot

```bash
python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
    --dataset datasets/lift/mh/image.hdf5 \
    --output_dir ~/lerobot_datasets/lift-mh-image
```

This creates a local LeRobot dataset at `~/lerobot_datasets/lift-mh-image` that you can use directly.

---

## 7. Mixing MH + PH (or Any Subsets)

There's no built-in "MH+PH mix" dataset on HuggingFace. To create one, you have two options:

### Option A: Merge at HDF5 level, then convert

Use `h5py` to concatenate two HDF5 files before converting:

```python
import h5py
import shutil

# Start with MH as base
shutil.copy("datasets/lift/mh/image.hdf5", "datasets/lift/mh_ph/image.hdf5")

with h5py.File("datasets/lift/ph/image.hdf5", "r") as ph, \
     h5py.File("datasets/lift/mh_ph/image.hdf5", "a") as merged:

    mh_demos = list(merged["data"].keys())
    next_idx = max(int(d.replace("demo_", "")) for d in mh_demos) + 1

    for demo_key in ph["data"].keys():
        new_key = f"demo_{next_idx}"
        ph.copy(f"data/{demo_key}", merged["data"], name=new_key)
        next_idx += 1

    print(f"Merged dataset has {len(merged['data'].keys())} demos")
```

Then convert the merged HDF5:
```bash
python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
    --dataset datasets/lift/mh_ph/image.hdf5 \
    --output_dir ~/lerobot_datasets/lift-mh-ph-image
```

### Option B: Convert separately and merge at LeRobot level

Convert MH and PH separately, then use LeRobot's dataset tools to concatenate. This is more complex and not directly supported by the conversion script.

### Option C: Just use MH (recommended)

For the residual RL pipeline, **MH alone is usually what you want**. It has 300 diverse demos (vs PH's 200 clean ones), and the mixed quality is exactly the challenging scenario that residual RL is designed to improve on. Adding PH episodes would actually make the baseline BC policy better, potentially leaving less room for RL to improve.

---

## 8. Using a Local Dataset for Training

### BC Training

In [scripts/train_bc_act.sh](scripts/train_bc_act.sh), the `DATASET` variable accepts both HuggingFace IDs and local paths:

```bash
# HuggingFace dataset (auto-downloads)
DATASET="ankile/robomimic-mh-lift-image"

# Local dataset (no download)
DATASET="/home/user/lerobot_datasets/lift-mh-image"
```

The path is passed to `LeRobotDataset()` at [train_bc_dexmg.py line 560](resfit/lerobot/scripts/train_bc_dexmg.py#L560). LeRobot checks if the string is a local path first; if not, it treats it as an HF repo ID.

### RL Training

In [scripts/train_residual_rl.sh](scripts/train_residual_rl.sh) or the Hydra config, the offline dataset name works the same way:

```bash
# HuggingFace
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_dexmg_config \
    offline_data.name="ankile/robomimic-mh-lift-image" \
    ...

# Local path
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_dexmg_config \
    offline_data.name="/home/user/lerobot_datasets/lift-mh-image" \
    ...
```

### Configuring episode count

For RL, the `offline_data.num_episodes` parameter controls how many episodes from the dataset are loaded into the offline replay buffer. This is set in:

- **Hydra config**: [resfit/rl_finetuning/config/residual_td3.py](resfit/rl_finetuning/config/residual_td3.py#L15) — `OfflineDataConfig.num_episodes`
- **Shell script override**: `offline_data.num_episodes=300`

If your dataset has 300 episodes and you set `num_episodes=100`, only the first 100 will be used.

---

## 9. Camera Selection — Can I Use Only 1 or 2 Cameras?

**Yes, absolutely.** Cameras are independently configurable at each stage.

### During dataset creation (HDF5 → LeRobot)

You choose cameras when extracting observations in Step 5:
```bash
# 1 camera only
--camera_names agentview

# 2 cameras
--camera_names agentview robot0_eye_in_hand

# 3 cameras
--camera_names agentview robot0_eye_in_hand frontview
```

The LeRobot dataset will only contain the cameras you specified.

### During BC training — `--policy_cameras`

Even if your dataset has 3 cameras, you can train the BC policy on a subset using `--policy_cameras` at [train_bc_dexmg.py line 207](resfit/lerobot/scripts/train_bc_dexmg.py#L207):

```bash
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ankile/robomimic-mh-can-image \
    --policy act \
    --policy_cameras agentview         # ← only use agentview for the policy
    ...
```

Or use 2 out of 3:
```bash
    --policy_cameras agentview robot0_eye_in_hand
```

If you don't specify `--policy_cameras`, **all cameras in the dataset** will be used. The filtering logic is at [train_bc_dexmg.py line 499](resfit/lerobot/scripts/train_bc_dexmg.py#L499-L524).

### During RL training — `rl_camera`

The RL training uses `rl_camera` (in the Hydra config) to specify which cameras the critic/actor networks observe. This MUST be a subset of cameras that exist in the dataset.

In the shell script:
```bash
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_dexmg_config \
    rl_camera='["observation.images.agentview"]' \
    ...
```

Or in a Hydra config dataclass (at [residual_td3.py](resfit/rl_finetuning/config/residual_td3.py)):
```python
rl_camera: list[str] = field(
    default_factory=lambda: [
        "observation.images.agentview",
    ]
)
```

### Consistency requirement

**The RL `rl_camera` must use the same cameras (or a subset) that the BC policy was trained on.** The base BC policy and the RL policy share the same vision encoder, so they need to observe the same camera views.

---

## 10. State Observations — What's Included and Can I Change It?

### What's included by default

The state vector is automatically assembled from expected low-dim keys. This happens in both the conversion script ([convert_robomimic_to_lerobot.py line 253](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L253)) and the environment wrapper ([dexmg.py line 441](resfit/dexmg/environments/dexmg.py#L441)).

**Single-arm Panda (Lift, Can, Square):**

| Key | Dimensions | Description |
|---|---|---|
| `robot0_eef_pos` | 3 | End-effector XYZ position |
| `robot0_eef_quat` | 4 | End-effector quaternion orientation |
| `robot0_gripper_qpos` | 2 | Gripper joint positions |
| **Total** | **9** | |

**Dual-arm Panda (Transport, Threading):**

| Key | Dimensions | Description |
|---|---|---|
| `robot0_eef_pos` | 3 | Robot 0 end-effector position |
| `robot0_eef_quat` | 4 | Robot 0 quaternion |
| `robot0_gripper_qpos` | 2 | Robot 0 gripper |
| `robot1_eef_pos` | 3 | Robot 1 end-effector position |
| `robot1_eef_quat` | 4 | Robot 1 quaternion |
| `robot1_gripper_qpos` | 2 | Robot 1 gripper |
| **Total** | **18** | |

**Humanoid (Coffee, Pouring, CanSort):**

| Key | Dimensions | Description |
|---|---|---|
| `robot0_right_eef_pos` | 3 | Right hand position |
| `robot0_right_eef_quat` | 4 | Right hand quaternion |
| `robot0_right_gripper_qpos` | ~6 | Right hand finger joints |
| `robot0_left_eef_pos` | 3 | Left hand position |
| `robot0_left_eef_quat` | 4 | Left hand quaternion |
| `robot0_left_gripper_qpos` | ~6 | Left hand finger joints |

### Can I change what's in the state?

**Yes, but you need to change it in two places:**

1. **Conversion script** — `get_expected_low_dim_keys()` at [convert_robomimic_to_lerobot.py line 253](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L253)
2. **Environment wrapper** — `_get_expected_low_dim_keys()` at [dexmg.py line 441](resfit/dexmg/environments/dexmg.py#L441)

**Both must return the same keys in the same order**, otherwise the dataset state and the environment state will have different dimensions and training will fail.

For example, to add object position to the Lift state:
```python
# In both files, for the single-arm Panda section:
panda_low_dim_keys_single = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",              # ← add this (object XYZ position for Lift)
]
```

**Available keys in Robomimic HDF5** (varies by task, check with `h5ls -r your_file.hdf5 | grep obs`):
- `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos` — standard robot state
- `robot0_joint_pos`, `robot0_joint_vel` — full joint state
- `object` — object positions (task-specific)
- `robot0_gripper_qvel` — gripper velocities

### Can I disable state (vision-only)?

Yes, for BC use `--disable_proprioceptive_obs`:
```bash
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --disable_proprioceptive_obs \
    ...
```

This removes `observation.state` from the policy inputs entirely. See [train_bc_dexmg.py line 530](resfit/lerobot/scripts/train_bc_dexmg.py#L530).

---

## 11. Action Space by Task

Actions are determined by the robot + controller configuration. The conversion script names them at [convert_robomimic_to_lerobot.py line 291](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L291).

| Task | Action Dim | Components |
|---|---|---|
| **Lift, Can, Square, Threading** | 7 | 3 (Δpos xyz) + 3 (Δrot axis-angle) + 1 (gripper) |
| **Transport** | 14 | 7 per arm (Δpos + Δrot + gripper) × 2 arms |
| **TwoArmThreading** | 14 | Same as Transport |
| **TwoArmCoffee, Pouring, CanSort** | 24 | Per arm: 3 (Δpos) + 3 (Δrot) + 6 (finger joints) × 2 |
| **TwoArmBoxCleanup, LiftTray** | 24 | Per arm: 3 (Δpos) + 3 (Δrot) + 6 (finger joints) × 2 |

**You don't choose the action space** — it's determined by the robot(s) and controller configuration in the environment. The dataset and environment must agree.

---

## 12. Collecting Your Own Demonstrations

If no existing dataset fits your needs, you can collect demos from scratch:

### Using robosuite's teleoperation

```bash
# Collect demonstrations with keyboard teleoperation
python robosuite/scripts/collect_human_demonstrations.py \
    --environment Lift \
    --robots Panda \
    --directory /path/to/save \
    --device keyboard     # or "spacemouse" for SpaceMouse
```

This creates a raw `demo.hdf5`. Then follow Steps 4-6 above to convert it.

### Using a scripted policy in robosuite

You can also write a script that runs a policy in robosuite and saves trajectories. Robomimic provides utilities for this — see robomimic docs on dataset collection.

---

## 13. End-to-End Example: Lift from Scratch

### Using the existing HF dataset (fastest path)

```bash
# ──────── BC Training ────────
# Just use the pre-converted dataset. No conversion needed.
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ankile/robomimic-mh-lift-image \
    --policy act \
    --steps 100000 \
    --batch_size 128 \
    --eval_env Lift \
    --rollout_freq 5000 \
    --eval_video_key observation.images.agentview \
    --eval_render_size 224 \
    --eval_num_envs 4 \
    --eval_num_episodes 100 \
    --wandb_project robomimic-lift-bc \
    --wandb_enable \
    --no_cleanup

# After training → note wandb project/run_id (e.g., robomimic-lift-bc/abc12345)

# ──────── Residual RL ────────
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_dexmg_config \
    task=Lift \
    base_policy.wandb_id="robomimic-lift-bc/abc12345" \
    base_policy.wt_type=best \
    offline_data.name="ankile/robomimic-mh-lift-image" \
    offline_data.num_episodes=300 \
    algo.total_timesteps=300000 \
    wandb.project="robomimic-lift-residual-td3" \
    headless=true \
    eval_num_envs=4
```

### Creating your own dataset with custom cameras

```bash
# 1. Download raw HDF5
cd deps/robomimic
python robomimic/scripts/download_datasets.py \
    --tasks lift --dataset_types mh --hdf5_types raw

# 2. Extract images with your chosen cameras
python robomimic/scripts/dataset_states_to_obs.py \
    --dataset ../datasets/lift/mh/demo_v15.hdf5 \
    --output_name image_custom.hdf5 \
    --done_mode 2 \
    --camera_names agentview frontview \
    --camera_height 84 --camera_width 84

# 3. Convert to LeRobot format
cd ../..   # back to repo root
python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py \
    --dataset deps/robomimic/datasets/lift/mh/image_custom.hdf5 \
    --output_dir ~/lerobot_datasets/lift-mh-custom-2cam

# NOTE: if you used non-standard cameras (like "frontview"), you need to edit
# get_expected_image_keys() in convert_robomimic_to_lerobot.py to include them.
# See line 203: add "frontview_image" to the panda_image_keys list.

# 4. Train BC with local dataset
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ~/lerobot_datasets/lift-mh-custom-2cam \
    --policy act \
    --eval_env Lift \
    ...

# 5. For RL, also update rl_camera to match your cameras
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-name=residual_td3_dexmg_config \
    task=Lift \
    offline_data.name="~/lerobot_datasets/lift-mh-custom-2cam" \
    rl_camera='["observation.images.agentview","observation.images.frontview"]' \
    ...
```

### Using custom cameras: what to edit in the converter

If you use cameras not in the default `get_expected_image_keys()` lists, the converter will filter them out. Edit [convert_robomimic_to_lerobot.py line 222](resfit/lerobot/dataset/convert_robomimic_to_lerobot.py#L222) to add your cameras:

```python
# For a single-arm Panda task with frontview added:
panda_image_keys = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "frontview_image",              # ← add custom cameras here
]
```

Also update the environment wrapper at [dexmg.py line 393](resfit/dexmg/environments/dexmg.py#L393) so the env produces the same cameras during rollout:

```python
panda_image_keys_single = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "frontview_image",              # ← same addition here
]
```

Both the dataset and the environment must agree on which cameras exist, otherwise evaluation rollouts will crash.
