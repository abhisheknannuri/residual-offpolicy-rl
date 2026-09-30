#!/usr/bin/env python3
"""
Collect rollout data from a trained policy in the Robomimic/DexMimicGen simulator
and save it as a LeRobot v2.1 dataset.

Two modes:
  - evaluate_and_save: Run exactly N episodes, save only successful ones.
  - collect_n_successful: Run until N successful episodes are collected.

The output dataset is in LeRobot v2.1 format (per-episode parquets + per-episode videos).
It can then be converted to v3 using tools/convert_v21_to_v3.py.

Usage:
    python scripts/collect_rollouts.py \
        --checkpoint_dir /path/to/checkpoints/050000 \
        --output_dataset_dir /path/to/output_dataset \
        --eval_num_episodes 50 \
        --mode collect_n_successful \
        --eval_camera_size 256
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Set multiprocessing start method for CUDA compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add deps/lerobot to path for LeRobotDataset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "lerobot"))

import numpy as np
import torch

from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper, create_vectorized_env
from resfit.lerobot.policies.pretrained import PreTrainedPolicy
from resfit.lerobot.utils.load_policy import load_policy

# Import LeRobotDataset from deps/lerobot (v2.1 API)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _obs_to_frame(obs: dict, action: torch.Tensor, reward: torch.Tensor,
                   env_idx: int, camera_keys: list) -> dict:
    """
    Convert a single environment's observation + action + reward into a frame dict
    suitable for LeRobotDataset.add_frame().

    The environment returns images as float32 tensors in (C, H, W) format with values [0, 1].
    LeRobotDataset expects numpy arrays — images in (H, W, C) uint8 format.
    """
    frame = {}

    # Proprioceptive state: (num_envs, state_dim) -> (state_dim,) float32
    state = obs["observation.state"][env_idx].detach().cpu().numpy().astype(np.float32)
    frame["observation.state"] = state

    # Action: (num_envs, action_dim) -> (action_dim,) float32
    act = action[env_idx].detach().cpu().numpy().astype(np.float32)
    frame["action"] = act

    # Reward: scalar -> (1,) float32
    rew = np.array([reward[env_idx].item()], dtype=np.float32)
    frame["reward"] = rew

    # Camera images: (num_envs, C, H, W) float32 [0,1] -> (H, W, C) uint8
    for cam_key in camera_keys:
        img_chw = obs[cam_key][env_idx].detach().cpu().numpy()  # (C, H, W) float32
        img_hwc = np.transpose(img_chw, (1, 2, 0))              # (H, W, C)
        img_uint8 = (img_hwc * 255.0).clip(0, 255).astype(np.uint8)
        frame[cam_key] = img_uint8

    return frame


def _collect_rollouts(
    *,
    policy: PreTrainedPolicy,
    env: VectorizedEnvWrapper,
    dataset: LeRobotDataset,
    num_episodes: int,
    mode: str,
    task_label: str,
    camera_keys: list,
    seed: int = 42,
    policy_obs_input_camsize: int | None = None,
):
    """
    Run rollouts and write successful episodes to the dataset.

    Args:
        policy: The loaded policy in eval mode.
        env: Vectorized environment wrapper.
        dataset: LeRobotDataset instance (created via .create()).
        num_episodes: Target number of episodes (meaning depends on mode).
        mode: "evaluate_and_save" or "collect_n_successful".
        task_label: Task string for dataset (e.g. "pick and place can").
        camera_keys: List of camera observation keys.
        seed: Random seed.

    Returns:
        dict with metrics.
    """
    policy_was_training = policy.training
    policy.eval()

    num_parallel_envs = env.num_envs
    env_name = getattr(env, "env_name", "Unknown")

    saved_episodes = 0
    done_episodes = 0
    total_steps = 0
    episode_successes = []
    episode_lengths_all = []

    start_time = time.perf_counter()

    logger.info(f"Starting rollout collection in '{env_name}' environment")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Target episodes: {num_episodes}")
    logger.info(f"  Parallel envs: {num_parallel_envs}")
    logger.info(f"  Camera keys: {camera_keys}")

    # Per-environment episode buffers
    # Each buffer is a list of (frame_dict, task_label) tuples
    episode_buffers: list[list] = [[] for _ in range(num_parallel_envs)]
    episode_steps = [0] * num_parallel_envs

    obs, _ = env.reset(seed=seed)

    # Build policy-input resizer if policy_obs_input_camsize differs from env camera size
    policy_resizer = None
    if policy_obs_input_camsize is not None:
        from torchvision.transforms import v2
        policy_resizer = v2.Resize(
            [policy_obs_input_camsize, policy_obs_input_camsize]
        )
        logger.info(
            f"  Policy input resizer: downscaling env images from env camera size to "
            f"{policy_obs_input_camsize}x{policy_obs_input_camsize} for policy inference "
            f"(using torchvision v2.Resize, matching training pipeline)"
        )

    def _should_continue():
        if mode == "evaluate_and_save":
            return done_episodes < num_episodes
        elif mode == "collect_n_successful":
            return saved_episodes < num_episodes
        return False

    while _should_continue():
        with torch.inference_mode():
            if policy_resizer is not None:
                # Resize images for policy inference (matching training transform)
                # but keep original obs for dataset saving
                policy_obs = {}
                for k, v in obs.items():
                    if k in camera_keys:
                        policy_obs[k] = policy_resizer(v)
                    else:
                        policy_obs[k] = v
                action = policy.select_action(policy_obs)
            else:
                action = policy.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Append current (obs, action, reward) to each env's buffer
        # Images saved at original env camera resolution (no resize for dataset)
        for env_idx in range(num_parallel_envs):
            frame = _obs_to_frame(obs, action, reward, env_idx, camera_keys)
            episode_buffers[env_idx].append(frame)
            episode_steps[env_idx] += 1

        total_steps += num_parallel_envs
        done = terminated | truncated

        if any(done):
            terminated_envs = torch.where(done)[0]
            # Determine success from reward (reward == 1.0 means success in robosuite)
            success_envs = torch.where(reward == 1.0)[0]

            # Reset policy state for terminated envs
            policy.reset(env_ids=terminated_envs)

            for env_idx in terminated_envs:
                env_idx_int = env_idx.item()
                is_success = env_idx in success_envs

                done_episodes += 1
                episode_successes.append(bool(is_success))
                episode_lengths_all.append(episode_steps[env_idx_int])

                should_save = False
                if mode == "evaluate_and_save" and done_episodes <= num_episodes:
                    should_save = True
                elif mode == "collect_n_successful" and is_success and saved_episodes < num_episodes:
                    should_save = True

                if should_save:
                    # Write buffered frames to dataset
                    for frame_dict in episode_buffers[env_idx_int]:
                        dataset.add_frame(frame_dict, task=task_label)
                    dataset.save_episode()
                    saved_episodes += 1
                    status_icon = "✅" if is_success else "⚠️"
                    logger.info(
                        f"  {status_icon} Saved episode {saved_episodes} (success={is_success}) "
                        f"(eval'd {done_episodes}, len={episode_steps[env_idx_int]})"
                    )
                else:
                    status = "FAIL" if not is_success else "SKIPPED (target reached)"
                    logger.debug(
                        f"  ❌ {status} episode "
                        f"(eval'd {done_episodes}, len={episode_steps[env_idx_int]})"
                    )

                # Clear buffer for this environment
                episode_buffers[env_idx_int] = []
                episode_steps[env_idx_int] = 0

        # Advance observation
        obs = next_obs

        if total_steps % 2000 == 0:
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"  Progress: saved={saved_episodes}, eval'd={done_episodes}, "
                f"steps={total_steps}, FPS={total_steps / elapsed:.1f}"
            )

    elapsed = time.perf_counter() - start_time
    success_rate = sum(episode_successes) / len(episode_successes) if episode_successes else 0.0

    if policy_was_training:
        policy.train()

    logger.info(f"\n{'='*60}")
    logger.info(f"Collection Complete!")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Episodes evaluated: {done_episodes}")
    logger.info(f"  Episodes saved: {saved_episodes}")
    logger.info(f"  Success rate: {success_rate * 100:.1f}%")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Elapsed: {elapsed:.1f}s")
    logger.info(f"  FPS: {total_steps / elapsed:.1f}")
    logger.info(f"{'='*60}\n")

    return {
        "saved_episodes": saved_episodes,
        "done_episodes": done_episodes,
        "success_rate": success_rate,
        "episode_successes": episode_successes,
        "episode_lengths": episode_lengths_all,
        "total_steps": total_steps,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect rollout data from a policy and save as LeRobot v2.1 dataset."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to checkpoint directory (containing config.json and weights)."
    )
    parser.add_argument(
        "--output_dataset_dir", type=str, required=True,
        help="Output directory for the new LeRobot v2.1 dataset."
    )
    parser.add_argument(
        "--repo_id", type=str, default="lerobot/pick_place_can_rollouts",
        help="Dataset repo ID (used as dataset identifier, not pushed to hub)."
    )
    parser.add_argument(
        "--mode", type=str, default="collect_n_successful",
        choices=["evaluate_and_save", "collect_n_successful"],
        help="Collection mode. evaluate_and_save: run N episodes, save only successes. "
             "collect_n_successful: run until N successful episodes are saved."
    )
    parser.add_argument(
        "--task_label", type=str, default="pick and place can",
        help="Task label string for the dataset."
    )
    parser.add_argument(
        "--eval_env", type=str, default="Can",
        help="Environment to evaluate in."
    )
    parser.add_argument(
        "--eval_num_envs", type=int, default=8,
        help="Number of parallel environments."
    )
    parser.add_argument(
        "--eval_num_episodes", type=int, default=50,
        help="Target number of episodes (meaning depends on --mode)."
    )
    parser.add_argument(
        "--eval_camera_size", type=int, default=256,
        help="Camera size for environment rendering and dataset recording. "
             "Images are saved to the dataset at this resolution."
    )
    parser.add_argument(
        "--policy_obs_input_camsize", type=int, default=None,
        help="If set, resizes the environment camera images to this size before passing to the policy "
             "for inference, using torchvision v2.Resize (matching the training pipeline's transform). "
             "The original --eval_camera_size images are still saved to the dataset. "
             "Example: --eval_camera_size 256 --policy_obs_input_camsize 84"
    )
    parser.add_argument(
        "--eval_render_size", type=int, default=256,
        help="Render size for video."
    )
    parser.add_argument(
        "--eval_video_key", type=str, default="observation.images.agentview",
        help="Video key for rendering."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use."
    )
    # parser.add_argument(
    #     "--seed", type=int, default=42,
    #     help="Random seed for deterministic evaluation."
    # )
    parser.add_argument(
        "--seed", type=lambda x: int(x) if str(x).lower() != 'none' else None, default=None,
        help="Random seed for deterministic evaluation. Pass 'none' for random seeds."
    )
    parser.add_argument(
        "--eval_horizon", type=int, default=None,
        help="Episode step limit (horizon)."
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    # Set seed globally
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"Set global random seed to {args.seed}")

    device_str = args.device
    logger.info(f"Using device: {device_str}")

    # ── Load policy ──────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory {checkpoint_dir} does not exist.")
        sys.exit(1)

    # Resolve actual policy dir containing config.json
    if (checkpoint_dir / "config.json").exists():
        policy_dir = checkpoint_dir
    elif (checkpoint_dir / "policy" / "config.json").exists():
        policy_dir = checkpoint_dir / "policy"
    elif (checkpoint_dir / "pretrained_model" / "config.json").exists():
        policy_dir = checkpoint_dir / "pretrained_model"
    else:
        logger.error(
            f"Could not find config.json in {checkpoint_dir} or its "
            "'policy'/'pretrained_model' subdirectories."
        )
        sys.exit(1)

    logger.info(f"Loading policy from {policy_dir}...")
    try:
        policy = load_policy(policy_dir)
        policy.to(device_str)
        policy.eval()
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        sys.exit(1)

    # ── Determine camera keys from the policy config ─────────
    camera_keys = [
        k for k, ft in policy.config.input_features.items()
        if k.startswith("observation.images.")
    ]
    if not camera_keys:
        logger.warning("No camera keys found in policy config. Using defaults.")
        camera_keys = [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
        ]
    logger.info(f"Camera keys: {camera_keys}")

    # ── Create environment ───────────────────────────────────
    camera_size = args.eval_camera_size
    logger.info(f"Using camera size: {camera_size}x{camera_size}")

    logger.info(f"Creating environment {args.eval_env}...")
    try:
        eval_env = create_vectorized_env(
            env_name=args.eval_env,
            num_envs=args.eval_num_envs,
            device=device_str,
            camera_size=camera_size,
            render_size=args.eval_render_size,
            video_key=args.eval_video_key,
            debug=False,
            horizon=args.eval_horizon,
        )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        sys.exit(1)

    # ── Create LeRobot v2.1 dataset ──────────────────────────
    output_dir = Path(args.output_dataset_dir)
    if output_dir.exists():
        logger.warning(f"Output directory already exists: {output_dir}")
        logger.warning("Removing it to start fresh...")
        import shutil
        shutil.rmtree(output_dir)

    logger.info(f"Creating LeRobot v2.1 dataset at: {output_dir}")

    # Define features matching the SARM v3/v2.1 schema
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": None,
        },
        "reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }

    # Add camera features
    for cam_key in camera_keys:
        features[cam_key] = {
            "dtype": "video",
            "shape": (camera_size, camera_size, 3),
            "names": ["height", "width", "channels"],
        }

    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=20,
            features=features,
            root=str(output_dir),
            robot_type="panda",
            use_videos=True,
            image_writer_threads=4,
        )
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        sys.exit(1)

    logger.info(f"Dataset created with features: {list(features.keys())}")

    # ── Run rollout collection ───────────────────────────────
    logger.info("Starting rollout collection...")
    try:
        metrics = _collect_rollouts(
            policy=policy,
            env=eval_env,
            dataset=dataset,
            num_episodes=args.eval_num_episodes,
            mode=args.mode,
            task_label=args.task_label,
            camera_keys=camera_keys,
            seed=args.seed,
            policy_obs_input_camsize=args.policy_obs_input_camsize,
        )
    except Exception as e:
        logger.error(f"Rollout collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Standardize Video Encodings ──────────────────────────
    logger.info("Standardizing collected videos to strictly match expert dataset encoding (H264, 20fps)...")
    video_dir = output_dir / "videos"
    if video_dir.exists():
        import subprocess
        for p in video_dir.rglob("*.mp4"):
            if not p.is_file(): continue
            tmp_p = str(p) + ".tmp.mp4"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error", "-i", str(p), 
                    "-an", "-r", "20", "-video_track_timescale", "10240", 
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", tmp_p
                ], check=True)
                os.rename(tmp_p, str(p))
            except Exception as e:
                logger.error(f"Failed to standardize video {p.name}: {e}")
        logger.info("Video standardization complete.")
    
    # ── Save summary ─────────────────────────────────────────
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "output_dataset_dir": str(output_dir),
        "mode": args.mode,
        "target_episodes": args.eval_num_episodes,
        "saved_episodes": metrics["saved_episodes"],
        "done_episodes": metrics["done_episodes"],
        "success_rate": metrics["success_rate"],
        "episode_successes": metrics["episode_successes"],
        "episode_lengths": metrics["episode_lengths"],
        "total_steps": metrics["total_steps"],
        "elapsed_s": metrics["elapsed_s"],
        "camera_size": camera_size,
        "eval_env": args.eval_env,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = output_dir / "collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary saved to: {summary_path}")

    # ── Final output ─────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset saved to: {output_dir}")
    logger.info(f"  Format: LeRobot v2.1")
    logger.info(f"  Episodes: {metrics['saved_episodes']}")
    logger.info(f"  Success rate: {metrics['success_rate']*100:.1f}%")
    logger.info(f"  Next step: Convert to v3 using tools/convert_v21_to_v3.py")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
