#!/usr/bin/env python3
"""
Evaluate a trained policy checkpoint in the Robomimic/DexMimicGen simulator.
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
# This must be done before any other multiprocessing operations
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add deps/lerobot to path just in case
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deps" / "lerobot"))

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper, create_vectorized_env
from resfit.lerobot.policies.pretrained import PreTrainedPolicy
from resfit.lerobot.utils.load_policy import load_policy

logger = logging.getLogger(__name__)

def _annotate_frame(
    frame: np.ndarray,
    env_idx: int,
    episode_num: int,
    total_episodes: int,
    episode_step: int,
    is_success: bool,
    font=None,
) -> np.ndarray:
    """Annotate a single frame with episode information."""
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    episode_text = f"Env {env_idx + 1} | Episode {episode_num}/{total_episodes}"
    step_text = f"Step {episode_step}"
    status_text = "SUCCESS" if is_success else "FAIL"
    status_color = (0, 255, 0) if is_success else (255, 0, 0)

    y_offset = 10
    draw.text((10, y_offset), episode_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), step_text, fill=(255, 255, 255), font=font)
    y_offset += 15
    draw.text((10, y_offset), status_text, fill=status_color, font=font)

    return np.array(pil_img)


def _run_rollouts(
    *,
    policy: PreTrainedPolicy,
    env: VectorizedEnvWrapper,
    save_dir: Path,
    step: int,
    num_episodes: int,
    run_start_time: str,
    seed: int = 42,
    eval_camera_size: int = 84,
    env_cam_render_size: int = 256,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    policy_was_training = policy.training
    policy.eval()

    num_parallel_envs = env.num_envs
    env_name = getattr(env, "env_name", "Unknown")

    successes = 0
    done_episodes = 0
    total_steps = 0
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []

    start_time = time.perf_counter()

    logger.info(f"Running rollouts with environment: {env_name}")
    logger.info(f"Starting evaluation: {num_episodes} episodes using {num_parallel_envs} parallel environments")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_dir = save_dir / f"eval_{env_name.lower()}" / run_start_time
    parent_dir.mkdir(parents=True, exist_ok=True)
    video_path = parent_dir / f"eval_step_{step}_{now}.mp4"

    video_writer = imageio.get_writer(video_path.as_posix(), fps=20)

    camera_keys = [
        k for k, ft in policy.config.input_features.items()
        if k.startswith("observation.images.")
    ]
    if not camera_keys:
        camera_keys = [
            "observation.images.agentview",
            "observation.images.robot0_eye_in_hand",
        ]

    policy_resizer = None
    if env_cam_render_size != eval_camera_size:
        from torchvision.transforms import v2
        policy_resizer = v2.Resize([eval_camera_size, eval_camera_size])
        logger.info(f"Policy input resizer: downscaling env images from {env_cam_render_size} to {eval_camera_size} for policy inference")

    obs, _ = env.reset(seed=seed)
    episode_frames = [[] for _ in range(num_parallel_envs)]
    episode_steps = [0] * num_parallel_envs

    while done_episodes < num_episodes:
        with torch.inference_mode():
            if policy_resizer is not None:
                policy_obs = {}
                for k, v in obs.items():
                    if k in camera_keys:
                        policy_obs[k] = policy_resizer(v)
                    else:
                        policy_obs[k] = v
                action = policy.select_action(policy_obs)
            else:
                action = policy.select_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        frames = env.render()
        for env_idx in range(num_parallel_envs):
            episode_frames[env_idx].append(frames[env_idx])
            episode_steps[env_idx] += 1

        total_steps += num_parallel_envs
        done = terminated | truncated

        if any(done):
            terminated_envs = torch.where(done)[0]
            success_envs = torch.where(reward == 1.0)[0]

            policy.reset(env_ids=terminated_envs)

            for env_idx in terminated_envs:
                env_idx_int = env_idx.item()
                is_success = env_idx in success_envs
                done_episodes += 1
                successes += int(is_success)
                episode_lengths.append(episode_steps[env_idx_int])
                episode_successes.append(bool(is_success))

                for step_idx, frame in enumerate(episode_frames[env_idx_int]):
                    annotated_frame = _annotate_frame(
                        frame=frame,
                        env_idx=env_idx_int,
                        episode_num=done_episodes,
                        total_episodes=num_episodes,
                        episode_step=step_idx + 1,
                        is_success=is_success,
                        font=font,
                    )
                    video_writer.append_data(annotated_frame)

                episode_frames[env_idx_int] = []
                episode_steps[env_idx_int] = 0

        if total_steps % 1_000 == 0:
            logger.info(
                f"Total steps: {total_steps}, done episodes: {done_episodes}, successes: {successes}, "
                f"FPS: {total_steps / (time.perf_counter() - start_time):.1f}"
            )

    video_writer.close()

    success_rate = successes / done_episodes if done_episodes > 0 else 0.0

    if policy_was_training:
        policy.train()

    total_elapsed_time = time.perf_counter() - start_time
    final_fps = total_steps / total_elapsed_time if total_elapsed_time > 0 else 0.0
    episodes_per_sec = done_episodes / total_elapsed_time if total_elapsed_time > 0 else 0.0

    logger.info(f"Evaluation completed: {done_episodes} episodes, {successes} successes ({success_rate * 100:.1f}%)")
    logger.info(f"Performance: {total_steps} total steps in {total_elapsed_time:.1f}s")
    logger.info(f"Average FPS: {final_fps:.1f} frames/sec | Episodes/sec: {episodes_per_sec:.2f}")
    logger.info(
        f"Parallel efficiency: {num_parallel_envs} environments,"
        f" {final_fps / num_parallel_envs:.1f} frames/sec per environment"
    )
    logger.info(f"Video saved with annotated frames: {video_path}")

    mean_episode_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    successful_episode_lengths = [
        length for length, is_success in zip(episode_lengths, episode_successes) if is_success
    ]
    mean_successful_episode_length = (
        float(np.mean(successful_episode_lengths)) if successful_episode_lengths else 0.0
    )

    return {
        "success_rate": success_rate,
        "video_path": video_path,
        "final_fps": final_fps,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
        "mean_episode_length": mean_episode_length,
        "mean_successful_episode_length": mean_successful_episode_length,
        "eval_dir": parent_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a loaded policy in DexMimicGen/Robomimic environments.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory (containing config.json and weights). e.g. path/to/checkpoints/005000/pretrained_model")
    parser.add_argument("--eval_env", type=str, default="Square", help="Environment to evaluate in")
    parser.add_argument("--eval_num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--eval_num_episodes", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--eval_camera_size", type=int, default=84, help="Camera size for observations (what policy expects)")
    parser.add_argument("--env_cam_render_size", type=int, default=256, help="Camera size for environment rendering")
    parser.add_argument("--eval_render_size", type=int, default=224, help="Render size for video")
    parser.add_argument("--eval_video_key", type=str, default="observation.images.agentview", help="Video key for rendering")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="outputs/eval", help="Output directory for videos and logs")
    parser.add_argument(
        "--seed", type=lambda x: int(x) if str(x).lower() != 'none' else None, default=None,
        help="Random seed for deterministic evaluation. Pass 'none' for random seeds."
    )
    parser.add_argument("--eval_horizon", type=int, default=None, help="Episode step limit (horizon)")
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

    # Set seed globally
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info(f"Set global random seed to {args.seed}")
    
    device_str = args.device
    logger.info(f"Using device: {device_str}")
    
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
        logger.error(f"Could not find config.json in {checkpoint_dir} or its 'policy'/'pretrained_model' subdirectories.")
        sys.exit(1)

    logger.info(f"Loading policy from {policy_dir}...")
    try:
        policy = load_policy(policy_dir)
        policy.to(device_str)
        policy.eval()
    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        sys.exit(1)
    
    # Use camera size directly from command-line arguments
    camera_size = args.env_cam_render_size
    logger.info(f"Using environment camera size: {camera_size}x{camera_size}, policy expects: {args.eval_camera_size}x{args.eval_camera_size}")

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
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    logger.info("Starting rollouts...")
    try:
        metrics = _run_rollouts(
            policy=policy,
            env=eval_env,
            save_dir=output_dir,
            step=0,
            num_episodes=args.eval_num_episodes,
            run_start_time=run_start_time,
            seed=args.seed,
            eval_camera_size=args.eval_camera_size,
            env_cam_render_size=args.env_cam_render_size,
        )
    except Exception as e:
        logger.error(f"Rollout execution failed: {e}")
        sys.exit(1)
    
    logger.info("="*50)
    logger.info("Evaluation Complete!")
    logger.info(f"Environment: {args.eval_env}")
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info(f"Success Rate: {metrics['success_rate']*100:.2f}%")
    logger.info(f"Video saved to: {metrics['video_path']}")
    logger.info("="*50)

    # Dump JSON Summary
    summary = {
        "success_rate": metrics["success_rate"],
        "mean_episode_length": metrics["mean_episode_length"],
        "mean_successful_episode_length": metrics["mean_successful_episode_length"],
        "episode_lengths": metrics["episode_lengths"],
        "episode_successes": metrics["episode_successes"],
        "checkpoint_dir": str(checkpoint_dir),
        "args": vars(args),
    }
    
    summary_path = metrics["eval_dir"] / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()
