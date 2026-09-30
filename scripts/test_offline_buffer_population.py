"""
test_offline_buffer_population.py

Stand-alone test that mirrors the EXACT dataset loading and offline buffer
population logic from train_residual_td3.py.

Run with:
    .venv/bin/python scripts/test_offline_buffer_population.py

Steps:
  1. Loads the local dataset from disk (no HuggingFace needed)
  2. Builds ActionScaler + StateStandardizer from dataset stats
  3. Creates the offline TensorDictPrioritizedReplayBuffer
  4. Populates it frame-by-frame (same logic as _populate_offline_buffer)
  5. Samples a batch and prints diagnostics (shapes, value ranges)

Exit code 0 = all good. Any crash = fix it before running training.
"""

import sys
from pathlib import Path

# make sure the repo root is on sys.path so resfit imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tqdm import tqdm

from resfit.rl_finetuning.datasets.remapped_lerobot import RemappedLeRobotDataset
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.dtype import to_uint8

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — edit to match your actual training .sh / config
# ═══════════════════════════════════════════════════════════════════════════

# Full path to the dataset directory (the one containing meta/, data/, videos/)
DATASET_PATH = Path(
    "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl"
    "/trossen_real/datasets"
    "/20260805_150824_and_153134_153727_20260807_110711_PickAndInsertCube_delta_rewardLabled_v21_old"
)

# Camera keys used during RL training (match rl_camera= in your .sh)
IMAGE_KEYS = [
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

# ── Image resize ────────────────────────────────────────────────────────────
# CRITICAL: 256×256 × 2 cams × 161 episodes × ~390 frames ≈ 17 GB RAM → OOM.
# The training script uses offline_data.image_size to resize before storing.
# Set IMAGE_SIZE to match whatever your RL training .sh sets IMAGE_SIZE= to.
# The RL critic ViT was designed for 84×84 in the original sim runs.
# For real hardware you likely also resize — check your training .sh / config.
# Set to None to skip resizing (only safe for a few episodes).
IMAGE_SIZE = 84   # resize to 84×84 before storing in buffer (matches training)

# Normalization — copy from your .sh
ACTION_SCALE     = 0.2   # ACTION_SCALE
MIN_ACTION_RANGE = 0.1   # MIN_ACTION_RANGE
MIN_STATE_STD    = 0.1   # MIN_STATE_STD

# Use next.reward from the dataset? (True because this dataset has reward labels)
USE_DATASET_REWARD = True

# ── Episode limit ────────────────────────────────────────────────────────────
# For a quick sanity test, load only the first N episodes.
# Set to None to process ALL 161 episodes (same as training with num_episodes=None).
# With IMAGE_SIZE=84, all 161 episodes fit in ~1.5 GB — safe to set None.
# With IMAGE_SIZE=None (256×256), keep this at ≤10 to avoid OOM.
NUM_EPISODES = None   # None = all episodes (safe when IMAGE_SIZE=84)

# N-step / gamma — match your config
N_STEP = 3
GAMMA  = 0.99

# Batch size for the final sample test
TEST_BATCH_SIZE = 32

# ═══════════════════════════════════════════════════════════════════════════


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── STEP 1: Load dataset ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Loading dataset")
    print("="*60)
    print(f"  Path: {DATASET_PATH}")
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"

    # NOTE: root = full path to the dataset folder (not the parent).
    # LeRobotDatasetMetadata uses root as-is when root is given — it does NOT
    # append repo_id. So we pass the full dataset directory as root.
    # key_map={} because this dataset already uses standard feature names.
    dataset = RemappedLeRobotDataset(
        repo_id=DATASET_PATH.name,
        root=str(DATASET_PATH),
        key_map={},
    )

    print(f"  Total frames   : {dataset.meta.total_frames}")
    print(f"  Total episodes : {dataset.meta.total_episodes}")
    print(f"  Features       : {list(dataset.meta.features.keys())}")
    print(f"  Stats keys     : {list(dataset.meta.stats.keys())}")

    assert "action"            in dataset.meta.stats,   "FAIL: 'action' stats missing"
    assert "observation.state" in dataset.meta.stats,   "FAIL: 'observation.state' stats missing"
    assert "next.reward"       in dataset.meta.features,"FAIL: 'next.reward' feature missing — set USE_DATASET_REWARD=False"

    # ── STEP 2: Normalization ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Building ActionScaler + StateStandardizer")
    print("="*60)

    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["action"],
        action_scale=ACTION_SCALE,
        min_range_per_dim=MIN_ACTION_RANGE,
        device=device,
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset.meta.stats["observation.state"],
        min_std=MIN_STATE_STD,
        device=device,
    )
    print(f"  action_min = {action_scaler._limits.min.cpu().tolist()}")
    print(f"  action_max = {action_scaler._limits.max.cpu().tolist()}")
    print(f"  state mean = {state_standardizer._mean.cpu().tolist()}")
    print(f"  state std  = {state_standardizer._std.cpu().tolist()}")

    # ── STEP 3: Create replay buffer ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — Creating offline replay buffer")
    print("="*60)

    num_eps = NUM_EPISODES if NUM_EPISODES is not None else dataset.meta.total_episodes
    total_frames = sum(
        dataset.meta.episodes[ep_idx]["length"]
        for ep_idx in range(min(num_eps, dataset.meta.total_episodes))
    )
    max_transitions = max(0, total_frames - num_eps)
    print(f"  Episodes       : {num_eps}")
    print(f"  Total frames   : {total_frames}")
    print(f"  Max transitions: {max_transitions}")
    print(f"  Image size     : {IMAGE_SIZE}x{IMAGE_SIZE} (None = native 256x256)")

    # RAM estimate
    if IMAGE_SIZE is not None:
        img_bytes = IMAGE_SIZE * IMAGE_SIZE * 3
    else:
        img_bytes = 256 * 256 * 3
    est_gb = max_transitions * len(IMAGE_KEYS) * 2 * img_bytes / 1e9
    print(f"  Est. RAM for images: {est_gb:.1f} GB")

    offline_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max_transitions, device="cpu"),
        alpha=0.0,
        beta=0.0,
        eps=1e-6,
        priority_key="_priority",
        transform=MultiStepTransform(
            n_steps=N_STEP,
            gamma=GAMMA,
            use_terminated_for_bootstrap=True,
        ),
        pin_memory=True,
        prefetch=3,
        batch_size=TEST_BATCH_SIZE,
    )
    print(f"  Buffer created (capacity {max_transitions})")

    # ── STEP 4: Populate buffer ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — Populating buffer (mirrors _populate_offline_buffer in training)")
    print("="*60)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    episode_cache: dict = {}
    transitions = 0
    reward_sum  = 0.0
    reward_min  = float("inf")
    reward_max  = float("-inf")
    done_count  = 0

    for sample in tqdm(loader, desc="Processing frames"):
        ep_idx = int(sample["episode_index"].item())
        if NUM_EPISODES is not None and ep_idx >= NUM_EPISODES:
            break

        # actions
        gt_action        = sample["action"].float().squeeze(0)              # (7,)
        gt_action_scaled = action_scaler.scale(gt_action.to(device)).cpu() # -> [-1, 1]
        base_action_scaled = gt_action_scaled   # GT-as-base mode

        done_flag = bool(sample["next.done"].item())

        # state
        state_raw = sample["observation.state"].float().squeeze(0)
        state_std = state_standardizer.standardize(state_raw.to(device)).cpu()

        curr_obs = {
            "observation.state":       state_std,
            "observation.base_action": base_action_scaled,
        }
        for k in IMAGE_KEYS:
            img = sample[k].squeeze(0)   # (C, H, W) float from video backend
            if IMAGE_SIZE is not None:
                # Same as training: T.Resize applied as offline_image_transforms
                img = T.functional.resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
            curr_obs[k] = img
        to_uint8(curr_obs, IMAGE_KEYS)

        # reward
        if USE_DATASET_REWARD and "next.reward" in sample:
            step_reward = float(sample["next.reward"].item())
        else:
            step_reward = float(done_flag)

        if done_flag:
            done_count += 1

        # build transition from (prev, curr)
        if ep_idx in episode_cache:
            prev = episode_cache[ep_idx]
            transition = TensorDict(
                {
                    "obs": TensorDict(prev["obs"], batch_size=[]),
                    "action": prev["action"],
                    "next": TensorDict(
                        {
                            "obs":        TensorDict(curr_obs, batch_size=[]),
                            "done":       torch.tensor(prev["done"],   dtype=torch.bool),
                            "terminated": torch.tensor(prev["done"],   dtype=torch.bool),
                            "reward":     torch.tensor(prev["reward"], dtype=torch.float32),
                        },
                        batch_size=[],
                    ),
                    "_priority": torch.tensor(10.0, dtype=torch.float32),
                },
                batch_size=[],
            ).unsqueeze(0)

            offline_rb.add(transition)
            transitions += 1
            r = prev["reward"]
            reward_sum += r
            reward_min  = min(reward_min, r)
            reward_max  = max(reward_max, r)

        episode_cache[ep_idx] = {
            "obs":    curr_obs,
            "action": gt_action_scaled,
            "reward": step_reward,
            "done":   done_flag,
        }

    print(f"\n  Transitions added : {transitions}")
    print(f"  Buffer size now   : {len(offline_rb)}")
    print(f"  Episodes with done: {done_count}")
    print(f"  Reward min/max    : {reward_min:.4f} / {reward_max:.4f}")
    print(f"  Reward mean       : {reward_sum / max(transitions, 1):.6f}")

    # ── STEP 5: Sample + sanity check ───────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — Sampling from buffer & sanity checks")
    print("="*60)

    if len(offline_rb) < TEST_BATCH_SIZE:
        print(f"  WARNING: only {len(offline_rb)} transitions, can't sample {TEST_BATCH_SIZE}")
        return

    batch = offline_rb.sample(TEST_BATCH_SIZE)
    print(f"  Batch shape         : {batch.shape}")
    print(f"  Top-level keys      : {sorted(batch.keys())}")
    print(f"  obs keys            : {sorted(batch['obs'].keys())}")
    print(f"  next keys           : {sorted(batch['next'].keys())}")
    print(f"  next.obs keys       : {sorted(batch['next']['obs'].keys())}")

    print("\n  Per-key shapes & value ranges:")
    def _show(name, t):
        t = t.float()
        print(f"    {name:50s}  shape={tuple(t.shape)}"
              f"  min={t.min().item():.4f}  max={t.max().item():.4f}"
              f"  mean={t.mean().item():.4f}")

    _show("action",                      batch["action"])
    _show("next.reward",                 batch["next"]["reward"])
    _show("next.done",                   batch["next"]["done"].float())
    _show("obs.observation.state",       batch["obs"]["observation.state"])
    _show("obs.observation.base_action", batch["obs"]["observation.base_action"])
    for k in IMAGE_KEYS:
        _show(f"obs.{k}",      batch["obs"][k].float())
        _show(f"next.obs.{k}", batch["next"]["obs"][k].float())

    # assertions
    assert "action" in batch,               "FAIL: 'action' missing"
    assert "next" in batch,                 "FAIL: 'next' missing"
    assert "reward" in batch["next"],       "FAIL: 'next.reward' missing"
    assert "obs" in batch["next"],          "FAIL: 'next.obs' missing"
    assert "observation.state" in batch["obs"], "FAIL: 'obs.observation.state' missing"
    for k in IMAGE_KEYS:
        assert k in batch["obs"],           f"FAIL: '{k}' missing from obs"
        assert k in batch["next"]["obs"],   f"FAIL: '{k}' missing from next.obs"

    print("\n" + "="*60)
    print("✅  ALL CHECKS PASSED — buffer loading is working correctly!")
    print("="*60)


if __name__ == "__main__":
    main()
