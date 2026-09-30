# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Deploy a trained ACT checkpoint LIVE against the real robot.

Two-process architecture (see `policy_server.py` in the training repo for
the full explanation of why): this script owns the robot arm connection
(`FollowerClient`, talking to an already-running `follower_single_server.py`)
and the cameras (`CameraManager`, direct RealSense/mock capture) - both live
in THIS workspace's `trossen_real` package. The POLICY itself (ACT model +
its exact checkpoint/processor classes) runs in a SEPARATE process, in the
training repo's own venv (`policy_server.py`), since that repo's `lerobot`
package version is what the checkpoint actually needs, and this workspace's
venv doesn't have it (nor does the training repo's venv have `pyrealsense2`
for cameras) - no cross-venv Python imports anywhere, just plain HTTP
between the two processes, exactly like the follower/teleop split already
in this repo.

ONLY SUPPORTS `command_space: joint` checkpoints (matches the actual
PickAndInsertCube checkpoint this was built for) - action = absolute
6-joint + gripper target, sent directly via `/move_to_joint_positions`. An
`command_space: ee` checkpoint would need different handling entirely (delta
pose action applied against the current pose, 9D sim-state observation via
axis-angle->quaternion conversion) - not implemented here.

Every control tick (paced at the station config's `control.frequency_hz`):
  1. `follower.get_state()` - one HTTP call, returns pose+q+gripper_pos+dq+
     efforts+accelerations together (see `follower_single_server.py`'s
     `/getstate` - doesn't branch on command_space, so all 6 non-image
     observation keys the checkpoint needs are always available for free).
  2. `cameras.get_all_latest()` - latest frame per configured camera.
  3. POST the observation to the policy server's `/predict` - internally
     calls `policy.select_action()` (the standard ACT receding-horizon
     deployment API: manages the "predict a chunk_size-length chunk, only
     actually execute n_action_steps of it before replanning" bookkeeping
     itself, only actually looking at the observation you POST once every
     `n_action_steps` calls - the rest are cheap queue-pops). Response is
     already unnormalized (real units, ready to send directly).
  4. `follower.move_to_joint_positions(action, blocking=False, ...)` - same
     non-blocking execution style as live teleop / `replay_episode.py`
     (matches how this checkpoint's own training data was recorded).
  5. Sleep to hold the target frequency, repeat.

RESET BEFORE INFERENCE (default, `--skip-reset` to disable): every
recorded training episode started from the SAME post-reset staged position
(`/api/start_recording` always runs `follower.reset()` first - see
`app.py::_do_reset()`/`follower_single.py::reset()`). If inference starts
from whatever ARBITRARY pose the arm happens to be resting at instead, the
first observation is badly out-of-distribution vs. everything the policy
was trained on - confirmed on real hardware to produce a violent, jerky
first move as the model tries to reconcile an unfamiliar start state. This
script now calls the SAME staged reset used at data-collection time before
starting inference, then waits `--settle-time-s` (default 2.0s) for the
arm to fully stop moving/oscillating before taking the first observation -
both configurable via CLI.

SAFETY: this drives the real robot completely autonomously, continuously,
with no human in the loop. Stay at the robot able to hit an E-stop / kill
the process. Consider `--max-steps` for a short first look.

Run (needs, in order):
  1. `follower_single_server.py` already running + connected for this
     station (see trossen_real/README.md §8).
  2. The training repo's `policy_server.py` already running (separate venv):
       cd <training_repo>
       uv run python custom_scripts/policy_server.py \\
           --checkpoint outputs/train/policy_bc_PickAndInsertCube/checkpoints/050000/pretrained_model \\
           --port 5070
  3. Then, from THIS repo (inside the resfit venv):
       python -m trossen_real.scripts.live_infer_deploy \\
           --config trossen_station2_single \\
           --policy-server-url http://127.0.0.1:5070 \\
           --settle-time-s 2.0 \\
           --max-steps 100
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from trossen_real.cameras.camera_manager import CameraManager
from trossen_real.config import load_station_config
from trossen_real.inference.policy_client import (
    PolicyClient,
    TickLogger,
    build_observation,
    check_action_space_mismatch,
    check_camera_health,
    reconstruct_absolute_action,
)
from trossen_real.teleop.follower_client import FollowerClient


def _check_camera_health(cameras: CameraManager, image_keys: list[str]) -> None:
    """Refuse to start inference if a camera the policy actually needs is
    mock or unhealthy - see `check_camera_health()` for the shared logic
    (also used by the inference web UI's status panel)."""
    problems = check_camera_health(cameras, image_keys)
    if problems:
        raise RuntimeError(
            "Refusing to start inference - the policy needs real, healthy camera feeds and: " + "; ".join(problems)
        )
    print(f"Camera health check passed for all policy-required cameras: {[k.removeprefix('observation.images.') for k in image_keys]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Station config name (should match how the checkpoint was trained)")
    parser.add_argument("--policy-server-url", default="http://127.0.0.1:5070", help="Where policy_server.py (training repo) is listening")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many control ticks (default: run until Ctrl+C)")
    parser.add_argument(
        "--settle-time-s", type=float, default=2.0,
        help="Seconds to wait AFTER the pre-inference reset move completes, before taking the first observation - "
             "lets any residual motion/oscillation fully stop (default 2.0s).",
    )
    parser.add_argument(
        "--skip-reset", action="store_true",
        help="Skip the pre-inference staged reset (NOT recommended - training episodes always started from this "
             "same reset position; starting from an arbitrary pose instead is badly out-of-distribution and was "
             "observed to cause a violent first move on real hardware).",
    )
    parser.add_argument(
        "--action-space", choices=["absolute", "delta_joint"], default="absolute",
        help="'absolute' for the original checkpoint (policy output IS the joint target). 'delta_joint' for a "
             "checkpoint trained on trossen_real/scripts/convert_to_delta_joint_dataset.py output (policy output "
             "is a per-frame joint delta vs. the SAME tick's observation.state, gripper still absolute) - see "
             "PickAndInsertCube.md and reconstruct_absolute_action() for exactly how this is turned back into an "
             "absolute target. STRONGLY RECOMMENDED to also pass --n-action-steps 1 to policy_server.py (or "
             "otherwise ensure it re-infers every tick) when using delta_joint - see the doc for why.",
    )
    parser.add_argument(
        "--force-action-space", action="store_true",
        help="Proceed even if policy_server.py's heuristic guess (from the checkpoint's train_config.json) "
             "disagrees with --action-space. NOT recommended - see check_action_space_mismatch()'s docstring "
             "for why getting this wrong is dangerous in either direction. Only use once you've manually "
             "confirmed which action space this specific checkpoint actually was trained with.",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Path to a JSONL file to append one diagnostic record per tick to (raw normalized model output, "
             "real-units prediction, reconstructed/clipped action, follower state, timing) - for OFFLINE "
             "inspection (e.g. plotting predicted vs. actual gripper over time), not used for control. Default: "
             "no logging. Parent directories are created automatically.",
    )
    args = parser.parse_args()

    config = load_station_config(args.config)
    if config.control.command_space != "joint":
        raise ValueError(
            f"live_infer_deploy.py only supports command_space: joint checkpoints, but '{args.config}' has "
            f"command_space: '{config.control.command_space}'. See the module docstring for why."
        )

    follower = FollowerClient(config.follower_server_url)
    if not follower.check():
        raise RuntimeError(
            f"follower_single_server.py not reachable/connected at {config.follower_server_url} - "
            "start it and POST /connect_to_robot first."
        )

    policy = PolicyClient(args.policy_server_url)
    policy_health = policy.health()
    if not policy_health.get("loaded"):
        raise RuntimeError(f"policy_server.py at {args.policy_server_url} has no policy loaded - {policy_health}")
    image_keys = policy_health["image_keys"]
    chunk_size = policy_health["chunk_size"]
    n_action_steps = policy_health["n_action_steps"]
    print(f"Policy loaded from {policy_health['checkpoint']} (device={policy_health['device']}, "
          f"chunk_size={chunk_size}, n_action_steps={n_action_steps})")
    print(f"Policy expects image keys: {image_keys}")
    print(f"action_space={args.action_space}")
    mismatch = check_action_space_mismatch(args.action_space, policy_health)
    if mismatch:
        if args.force_action_space:
            print(f"WARNING (proceeding anyway, --force-action-space given): {mismatch}")
        else:
            raise RuntimeError(f"{mismatch} Pass --force-action-space to proceed anyway (not recommended).")
    if args.action_space == "delta_joint" and n_action_steps != 1:
        print(
            f"WARNING: action_space=delta_joint but policy_server reports n_action_steps={n_action_steps} "
            "(not 1). Per-frame delta reconstruction is only exactly correct for the FIRST step of a freshly-"
            "inferred chunk - steps beyond that reuse a state reference that no longer matches what the model "
            "conditioned on. See PickAndInsertCube.md. Strongly consider re-starting policy_server.py with "
            "n_action_steps=1."
        )

    cameras = CameraManager(config)
    cameras.start()
    print(f"Cameras started: {cameras.camera_names}")
    # Give the camera background threads a moment to produce their first real frames.
    time.sleep(1.0)
    _check_camera_health(cameras, image_keys)

    if args.skip_reset:
        print("--skip-reset given: NOT moving to the staged reset position - starting inference from whatever "
              "pose the arm is currently at. Not recommended (see module docstring).")
    else:
        # Same staged reset `_do_reset()`/data collection always ran before every
        # episode - critical for matching the distribution of start states the
        # policy was actually trained on (see module docstring's RESET BEFORE
        # INFERENCE section for why skipping this caused a violent first move).
        print("Moving to the staged reset position (same as data-collection time)...")
        follower.reset()
        print(f"Reset move complete - settling for {args.settle_time_s:.1f}s before the first observation...")
        time.sleep(args.settle_time_s)

    print("Resetting policy's internal action-chunk queue (fresh rollout)...")
    policy.reset()

    tick_logger = None
    if args.log_file:
        tick_logger = TickLogger(args.log_file)
        print(f"Logging one diagnostic record per tick to: {args.log_file}")

    period_s = 1.0 / config.control.frequency_hz
    step = 0
    try:
        while args.max_steps is None or step < args.max_steps:
            t_start = time.time()

            follower_state = follower.get_state()
            images = cameras.get_all_latest()
            obs = build_observation(follower_state, images, image_keys)

            result = policy.predict_full(obs)  # {"action": real units, "action_normalized": diagnostic-only}
            predicted = result["action"]
            action = reconstruct_absolute_action(
                args.action_space, predicted, follower_state,
                gripper_bounds=(config.control.gripper_closed, config.control.gripper_open),
            )
            follower.move_to_joint_positions(action, blocking=False, min_time_to_move=config.control.goal_time_s)

            elapsed = time.time() - t_start
            time.sleep(max(0.0, period_s - elapsed))

            step += 1
            achieved_hz = 1.0 / (time.time() - t_start) if (time.time() - t_start) > 0 else float("inf")
            print(f"step {step:5d}: action={np.round(action, 4)}  freq={achieved_hz:5.1f}Hz (target {config.control.frequency_hz}Hz)")

            if tick_logger is not None:
                tick_logger.log(
                    step=step,
                    action_space=args.action_space,
                    follower_state_q=follower_state["q"],
                    follower_state_gripper=follower_state["gripper_pos"],
                    policy_action_normalized=result["action_normalized"],
                    policy_action_real=predicted,
                    reconstructed_action=action,
                    achieved_hz=achieved_hz,
                )
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        cameras.stop()
        if tick_logger is not None:
            tick_logger.close()


if __name__ == "__main__":
    main()
