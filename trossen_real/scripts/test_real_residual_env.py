# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Manual/interactive REAL-HARDWARE smoke test for `TrossenResidualEnv`
(`trossen_real/rl/real_residual_env.py`) - exercises `reset()`/`step()`/
`close()` mechanics, the reward-pedal latch, reset-pedal episode
termination, and (optionally) leader-intervention takeover, WITHOUT any RL
actor or training loop involved.

The residual action is ALWAYS zero here, so the executed action is always
exactly the base ACT policy's own prediction - i.e. plain autonomous
behavior already validated by `live_infer_deploy.py`. This isolates testing
to the NEW code (reward/reset/intervention plumbing, settle timing, obs
augmentation, `info["scaled_action"]` contract) without the added risk/
variance of an untrained or partially-trained residual actor.

Uses the REAL calibrated `ActionScaler`/`StateStandardizer` (fit from the
actual training dataset's stats, same as `train_residual_td3.py`) - NOT a
placeholder/identity scaler - because with a wrong scaler, unscaling a
real-unit base action could clamp it to the wrong range and command a bad
joint target. Since residual=0 always, this script's real-world behavior
should be indistinguishable from `live_infer_deploy.py` except for the new
pedal/reward/reset instrumentation on top.

SAFETY: this drives the real robot autonomously. Stay at the robot, able to
hit an E-stop / kill the process. Start with `--max-steps` small (e.g. 20).

Run (needs, in order):
  1. `follower_single_server.py` already running + connected for this station.
  2. The training repo's `policy_server.py` already running (separate venv).
  3. From THIS repo (resfit venv):
       python -m trossen_real.scripts.test_real_residual_env \\
           --config trossen_station2_single \\
           --policy-server-url http://127.0.0.1:5070 \\
           --dataset-repo-id <repo_id> --dataset-root <root> \\
           --action-space delta_joint \\
           --max-steps 20 --num-episodes 2 \\
           [--enable-intervention]

Suggested manual test sequence (do these one at a time, rerunning as needed):
  1. No pedal presses at all - confirm behavior matches plain autonomous
     rollout: reward always 0.0, terminated only via --max-steps truncation.
  2. Press+HOLD the REWARD pedal for several consecutive printed steps, then
     release - confirm reward=1.0 on EVERY step while held, not just one.
  3. Press+release the REWARD pedal as fast as possible, timed to land
     inside a single control tick - confirm that step's printed reward is
     still 1.0 (this is the press-inside-one-tick latch fix; see
     REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md).
  4. Press the RESET pedal - confirm the CURRENT episode ends
     (terminated=True printed on that step), then a fresh reset (staged
     position + configured settle time) happens automatically before the
     next episode's steps start printing.
  5. Hold the REWARD pedal THROUGH a reset-pedal press (i.e. don't release
     reward before pressing reset) - confirm the WARNING about "reward pedal
     still held at episode start" prints, and use it as a cue to release
     the pedal during the settle window before the new episode's first step.
  6. (only if --enable-intervention) Hold the INTERVENTION pedal, backdrive
     the leader - confirm `intervened=True` prints and the follower tracks
     the leader smoothly; release - confirm a smooth resume (no jerk) and
     `intervened=False`.
  7. Unplug the foot pedal mid-run - confirm the "lost its device
     connection" error prints repeatedly. Do NOT continue training in this
     state - reconnect first.
"""

from __future__ import annotations

import argparse

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from trossen_real.cameras.camera_manager import CameraManager
from trossen_real.config import load_station_config
from trossen_real.human_intervention.pedal_listener import PedalListener
from trossen_real.inference.policy_client import PolicyClient, check_camera_health
from trossen_real.leader.trossen_leader_single import TrossenSingleLeader
from trossen_real.rl.real_residual_env import TrossenResidualEnv
from trossen_real.teleop.follower_client import FollowerClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Station config name")
    parser.add_argument("--policy-server-url", default="http://127.0.0.1:5070")
    parser.add_argument("--dataset-repo-id", required=True, help="Real delta-joint dataset repo_id (for action/state stats)")
    parser.add_argument("--dataset-root", default=None, help="Local dataset root, if not using the HF cache default")
    parser.add_argument("--action-space", choices=["absolute", "delta_joint"], default="delta_joint")
    parser.add_argument("--action-scale", type=float, default=0.2, help="Must match training's cfg.agent.actor.action_scale")
    parser.add_argument("--min-action-range", type=float, default=0.1)
    parser.add_argument("--min-state-std", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=20, help="Per-episode step cap (truncation)")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--settle-time-s", type=float, default=2.0)
    parser.add_argument("--enable-intervention", action="store_true")
    args = parser.parse_args()

    station_config = load_station_config(args.config)
    if station_config.arm_mode != "single":
        raise ValueError(f"Only arm_mode: single is supported (got '{station_config.arm_mode}').")
    if station_config.control.command_space != "joint":
        raise ValueError(f"Only command_space: joint is supported (got '{station_config.control.command_space}').")

    follower = FollowerClient(station_config.follower_server_url)
    if follower.check():
        print(f"Follower already connected at {station_config.follower_server_url}.")
    else:
        print(f"Connecting to follower at {station_config.follower_server_url}...")
        follower.connect()

    policy = PolicyClient(args.policy_server_url)
    policy_health = policy.health()
    if not policy_health.get("loaded"):
        raise RuntimeError(f"policy_server.py at {args.policy_server_url} has no policy loaded - {policy_health}")
    image_keys = policy_health["image_keys"]
    print(f"Policy loaded from {policy_health['checkpoint']} - image_keys={image_keys}")

    cameras = CameraManager(station_config)
    cameras.start()
    problems = check_camera_health(cameras, image_keys)
    if problems:
        raise RuntimeError("Refusing to start - camera health check failed: " + "; ".join(problems))

    leader = None
    pedal = None
    if args.enable_intervention:
        leader = TrossenSingleLeader(station_config.leader_ips["single"])
        leader.connect()
        pedal = PedalListener()
        print("Intervention ENABLED - leader + pedal connected.")
    else:
        print("Intervention disabled (--enable-intervention not passed) - reward/reset pedal will NOT be "
              "active either (pedal is only constructed when --enable-intervention is set, matching "
              "train_residual_td3.py's exact gating). Pass --enable-intervention to test reward/reset too.")

    print(f"Loading dataset stats from {args.dataset_repo_id} (root={args.dataset_root})...")
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    action_scaler = ActionScaler.from_dataset_stats(
        dataset.meta.stats["action"], action_scale=args.action_scale,
        min_range_per_dim=args.min_action_range, device="cpu",
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        dataset.meta.stats["observation.state"], min_std=args.min_state_std, device="cpu",
    )

    env = TrossenResidualEnv(
        follower=follower,
        cameras=cameras,
        policy=policy,
        control=station_config.control,
        reset_cfg=station_config.reset["single"],
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        image_keys=image_keys,
        camera_resolution=station_config.cameras.resolution,
        action_dim=7,
        action_space=args.action_space,
        leader=leader,
        pedal=pedal,
        max_steps=args.max_steps,
        settle_time_s=args.settle_time_s,
    )

    try:
        for ep in range(args.num_episodes):
            print(f"\n=== Episode {ep} - resetting (staged position + {args.settle_time_s}s settle) ===")
            obs, _ = env.reset()
            zero_residual = torch.zeros((1, 7), dtype=torch.float32)
            step = 0
            while True:
                next_obs, reward, terminated, truncated, info = env.step(zero_residual)
                print(
                    f"  step={step:3d} reward={reward.item():.1f} terminated={bool(terminated.item())} "
                    f"truncated={bool(truncated.item())} intervened={info['intervened']}"
                )
                obs = next_obs
                step += 1
                if bool(terminated.item()) or bool(truncated.item()):
                    print(f"  Episode {ep} ended after {step} steps (terminated={bool(terminated.item())}, "
                          f"truncated={bool(truncated.item())}).")
                    break
    finally:
        print("Closing env (parking arms, disconnecting)...")
        env.close()


if __name__ == "__main__":
    main()
