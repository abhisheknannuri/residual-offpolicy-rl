# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Replay one recorded episode's actions against the REAL follower, and
verify how closely the robot's actual motion tracks what was recorded.

Supports BOTH `control.command_space` modes (auto-detected from `--config`,
must match whatever station config was active when the episode was
recorded):

  - `ee` (cartesian EE-pose control): `observation.state` is the 9D
    sim-convention `eef_pos(3)+eef_quat(4)+gripper_qpos(2)`, `action` is a
    7D DELTA pose axis-angle(6) + absolute gripper(1) - replay computes
    `target = follower_current_pose + action[:6]` each step (see
    `dataset_recorder.py::_to_sim_state()`/`_sim_state_to_pose7()` below).
  - `joint` (raw joint-space control, matches Trossen's own official
    `lerobot_trossen` reference): `observation.state` AND `action` are both
    the raw 7D joint vector (6 arm joints (rad) + gripper (m)) - `action`
    is already the exact ABSOLUTE joint goal that was sent that tick (see
    `control_loop.py::_tick()`'s joint branch), so replay just sends
    `target = action` directly each step, no delta math, no state needed
    to compute the target (only used for the tracking-error comparison).

ALSO supports `command_space: joint` datasets that went through
`scripts/convert_to_delta_joint_dataset.py` (`--action-space delta_joint`,
see `trossen_real/datasets/PickAndInsertCube.md`'s "Delta-joint action
 preprocessing" section) - there, `action[:6]` is a PER-FRAME DELTA
(`action[t]-state[t]` at that SAME frame, computed once at preprocessing
time) rather than an absolute target, and `action[6]` (gripper) is still
absolute either way. Auto-detected via the presence of that script's
`DELTA_JOINT_TRANSFORM.md` provenance file in the dataset root (override
with `--action-space` if needed). Target reconstruction there is
`target[:6] = action[:6] + observation.state[:6]` using THIS SAME FRAME's
recorded state (not a live follower reading) - this is even more exact
than live-inference reconstruction (`reconstruct_absolute_action()` in
`trossen_real/inference/policy_client.py`, which has to use a live/current
state reading instead), since replay has the EXACT historical state the
delta was originally computed against.

No separate "reset pose" needs to be saved anywhere: `/api/start_recording`
already runs the follower's staged/randomized reset BEFORE recording
starts (see `app.py::api_start_recording()`), so frame 0's
`observation.state` of every episode already IS the post-reset pose
(including wherever `randomize_ee` landed that specific episode) - this
script reads it back and moves there. IMPORTANT: it also reruns the SAME
safe staged-position reset first (`follower.reset()`, i.e.
`follower_single.py::reset()`'s Stage 1 - joint-space move to
`config.reset[side].staged_joint_positions`, unconditionally, regardless
of `command_space`) before moving to that recorded frame-0 pose - exactly
mirroring what every real recorded episode did. Without this (`--skip-
staging-reset` to disable, not recommended), the script would interpolate
DIRECTLY from whatever arbitrary pose the arm currently happens to be in
straight to frame 0's pose, with no known-safe intermediate waypoint - a
real gap if the arm was left in a very different pose since the last run.

Sequence:
  1. Load one episode from a recorded session (`LeRobotDataset(...,
     episodes=[N])`).
  2. Convert frame 0's `observation.state` back to the SDK's native 7D
     format (EE pose7 or joint7, depending on `command_space`) and move
     the follower there directly (slow - matches the reset convention of
     giving a big jump plenty of time, default 3.0s, not the fast per-tick
     teleop time).
  3. Replay each frame's `action` one at a time - `ee` mode: `target =
     follower_current_pose + action[:6]`, gripper = `action[6]` (clipped);
     mode-agnostic w.r.t. whether the episode was recorded with
     `control_mode: delta` or `absolute` (both record a delta either way -
     see README §4's `control.control_mode` note). `joint` mode: `target =
     action` directly (already absolute, clipped only on the gripper
     component). Moves are NON-blocking by default (`--blocking` to
     override) - matches the live teleop control loop's own execution mode
     exactly (both run non-blocking at the configured `frequency_hz` with
     overlapping moves), which also matches how this action sequence will
     actually be consumed by a deployed policy. `--blocking` was tried (in
     `ee` mode) and made tracking fidelity WORSE, not better - see the
     flag's help text / README §5a for why.
  4. After each step, compares the follower's ACTUAL resulting pose/joints
     against the dataset's recorded `observation.state` for the NEXT frame
     (i.e. what a perfectly-tracking follower would have reached),
     reporting position/orientation/gripper error (`ee`) or per-joint error
     (`joint`) - this is the "did the robot do the same motion"
     verification. Some steady, bounded tracking error is EXPECTED for
     this control scheme (the same real-time tracking lag exists during
     live teleop too) - it is NOT necessarily a bug.

`--action-source state-delta` (HYPOTHESIS TEST, `ee` mode only - doesn't
apply to `joint` mode, which already records the absolute target directly):
the recorded `action` field is the LEADER's raw tick-to-tick command,
which is NOT necessarily what the follower actually achieved by the next
tick (non-blocking moves overlap, goal_time_s > tick period). If that
mismatch is the real source of tracking error, then re-deriving the delta
from the recorded observation.state trajectory itself (`state[i+1] -
state[i]`, converted back out of quaternion into axis-angle) - i.e. what
the follower's pose ACTUALLY changed by, tick to tick, during the original
recording - and replaying THAT (ideally with `--blocking` so each step
actually completes it) should track near-perfectly, since it's driving the
follower through the exact absolute waypoints it visited before, not the
leader's un-achieved intent. Falls back to `recorded` (the default)
otherwise.

Timing: `min_time_to_move` for every per-tick move uses
`config.control.goal_time_s`, which is DERIVED as `goal_time_multiplier /
frequency_hz` (see `ControlConfig.goal_time_s` in `config.py`) - NOT a
hardcoded constant, so this automatically matches whatever multiplier/rate
the station config specifies (tune `goal_time_multiplier` there, not here).

Config resolution: `--config` is now OPTIONAL. `dataset_recorder.py::
start_session()` copies the EXACT station config YAML used at recording
time into the session dir itself (`<dataset>/station_config.yaml`) - this
script ALWAYS prefers that saved copy when present, since it's the ground
truth for how the episode was actually recorded, independent of whatever
the original config file says NOW (it may have been retuned since) or
whatever name someone happens to pass. `--config` is only actually USED as
a fallback for datasets recorded before this feature existed (no saved
copy) - if a saved copy exists AND `--config` is also given, the saved
copy still wins, but a mismatch on anything that would change how replay
behaves (`command_space`, `control_mode`, `frequency_hz`,
`goal_time_multiplier`) is loudly warned about (usually means `--config`
is stale/wrong, not a real problem with the dataset).

Talks to an already-running `follower_single_server.py` over HTTP (the same
`FollowerClient` used by `app.py`/`control_loop.py`) - start that first,
and make sure NOTHING else (no `app.py` teleop session) is also driving the
same follower at the same time.

SAFETY: this moves the real robot through an entire recorded episode
automatically, at speed, with no human in the loop pressing anything. Stay
at the robot able to hit an E-stop / kill the process. Try a short episode
first, and consider `--max-steps` to only replay the first few frames
while you're getting a feel for it.

Run (from the repo root, inside the resfit venv)::

    python -m trossen_real.scripts.replay_episode \\
        --dataset trossen_real/datasets/<session_dir> \\
        --episode 0 \\
        --config trossen_station2_single
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation

from trossen_real.config import StationConfig, load_station_config
from trossen_real.teleop.dataset_recorder import SAVED_CONFIG_FILENAME
from trossen_real.teleop.follower_client import FollowerClient

# Written by `scripts/convert_to_delta_joint_dataset.py` into the OUTPUT dataset's
# root - its presence is used to auto-detect `--action-space delta_joint` datasets
# (see `_resolve_action_space()`).
DELTA_JOINT_PROVENANCE_FILENAME = "DELTA_JOINT_TRANSFORM.md"


def _sim_state_to_pose7(sim_state: np.ndarray) -> np.ndarray:
    """Inverse of `dataset_recorder.py::_to_sim_state()` for one side (`command_space: ee` only).

    9D sim-convention (eef_pos(3) + eef_quat(4) + gripper_qpos(2)) -> 7D SDK
    native format (x, y, z, ax, ay, az, gripper). `gripper_qpos_0` and
    `gripper_qpos_1` are always identical (see `_to_sim_state()`), either
    one is used directly.
    """
    sim_state = np.asarray(sim_state, dtype=np.float32)
    pos = sim_state[:3]
    quat = sim_state[3:7]  # [x, y, z, w] - scipy/robosuite convention
    gripper = sim_state[7]
    axis_angle = Rotation.from_quat(quat).as_rotvec()
    return np.concatenate([pos, axis_angle, [gripper]]).astype(np.float32)


def _dataset_state_to_native(config: StationConfig, state: np.ndarray) -> np.ndarray:
    """Convert one frame's recorded `observation.state` to the SDK's native
    7D format for the active `command_space` - EE pose7
    (`_sim_state_to_pose7()`, `ee` mode) or the raw joint7 vector, passed
    through as-is (`joint` mode, no conversion - matches
    `dataset_recorder.py::add_frame()`'s symmetric skip of `_to_sim_state()`).
    """
    if config.control.command_space == "joint":
        return np.asarray(state, dtype=np.float32)
    return _sim_state_to_pose7(state)


def _pose_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    """(position_error_m, orientation_error_rad, gripper_error_m) between two 7D EE poses (`ee` mode)."""
    pos_err = float(np.linalg.norm(actual[:3] - expected[:3]))
    rel = Rotation.from_rotvec(actual[3:6]).inv() * Rotation.from_rotvec(expected[3:6])
    ori_err = float(np.linalg.norm(rel.as_rotvec()))
    gripper_err = float(abs(actual[6] - expected[6]))
    return pos_err, ori_err, gripper_err


def _joint_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """(joint_error_rad [L2 norm over the 6 arm joints], gripper_error_m) between two 7D joint vectors (`joint` mode)."""
    joint_err = float(np.linalg.norm(actual[:6] - expected[:6]))
    gripper_err = float(abs(actual[6] - expected[6]))
    return joint_err, gripper_err


def _resolve_action_space(dataset_path: Path, action_space_arg: str | None) -> str:
    """'absolute' (default, matches every dataset recorded before the delta-joint
    preprocessing existed) or 'delta_joint' (`action[:6]` is a per-frame delta,
    see module docstring). Auto-detected via `DELTA_JOINT_PROVENANCE_FILENAME`'s
    presence in the dataset root (written by `convert_to_delta_joint_dataset.py`) -
    `--action-space` overrides the auto-detection if explicitly given."""
    detected = "delta_joint" if (dataset_path / DELTA_JOINT_PROVENANCE_FILENAME).exists() else "absolute"
    if action_space_arg is None:
        print(f"Auto-detected action_space='{detected}' "
              f"({'found' if detected == 'delta_joint' else 'no'} {DELTA_JOINT_PROVENANCE_FILENAME} in dataset root).")
        return detected
    if action_space_arg != detected:
        print(f"WARNING: --action-space '{action_space_arg}' was given but auto-detection says '{detected}' - "
              f"using '{action_space_arg}' as explicitly requested, but double check this is really what you want.")
    return action_space_arg


# Config fields that actually change HOW replay behaves - a mismatch here
# between the dataset's saved config and an explicitly-passed `--config`
# means replay would do something different from how the data was
# recorded, which is exactly the "modified and forgot" failure mode this
# saved-copy mechanism exists to catch.
_REPLAY_RELEVANT_FIELDS = ("command_space", "control_mode", "frequency_hz", "goal_time_multiplier")


def _resolve_config(dataset_path: Path, config_arg: str | None) -> StationConfig:
    """Resolve the station config to replay with - ALWAYS prefers the exact
    config saved into the dataset at recording time (`dataset_recorder.py::
    start_session()` copies it to `<dataset>/station_config.yaml`) over a
    separately-passed `--config` name/path, since the latter can silently
    drift out of sync (the original file gets retuned later, or the wrong
    name is passed) while the saved copy is ground truth for how THIS
    episode was actually recorded.

    `--config` is only actually used as a fallback when no saved copy
    exists (datasets recorded before this feature existed). If a saved copy
    DOES exist and `--config` is ALSO given, the saved copy still wins, but
    a mismatch on anything that would change replay behavior is loudly
    warned about.
    """
    saved_path = dataset_path / SAVED_CONFIG_FILENAME
    if saved_path.exists():
        config = load_station_config(str(saved_path))
        print(f"Using the station config saved inside the dataset at recording time: {saved_path}")
        if config_arg is not None:
            try:
                provided = load_station_config(config_arg)
            except Exception as exc:
                print(f"WARNING: --config '{config_arg}' was also given but failed to load ({exc}) - ignoring it, using the saved copy.")
                return config
            mismatches = [
                f"{field}: saved={getattr(config.control, field)!r} vs --config={getattr(provided.control, field)!r}"
                for field in _REPLAY_RELEVANT_FIELDS
                if getattr(config.control, field) != getattr(provided.control, field)
            ]
            if mismatches:
                print(
                    f"WARNING: --config '{config_arg}' was also given and DISAGREES with the dataset's own saved "
                    f"config on fields that change replay behavior - using the SAVED copy (ground truth), NOT "
                    f"--config. Mismatches: {'; '.join(mismatches)}. If this is unexpected, --config is likely "
                    f"stale (edited/retuned since this dataset was recorded)."
                )
            else:
                print(f"(--config '{config_arg}' was also given and matches the saved copy on all replay-relevant fields.)")
        return config

    if config_arg is None:
        raise FileNotFoundError(
            f"No saved station config found at {saved_path} (this dataset predates the auto-saved-config "
            "feature) - pass --config explicitly to specify which station config to replay with."
        )
    print(
        f"WARNING: no saved station config found at {saved_path} (this dataset predates the auto-saved-config "
        f"feature) - trusting --config '{config_arg}' as-is. Double check it actually matches what was used to "
        "record this episode."
    )
    return load_station_config(config_arg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to the session directory (contains meta/, data/, videos/)")
    parser.add_argument("--episode", type=int, required=True, help="Episode index within that session to replay")
    parser.add_argument(
        "--config", default=None,
        help=(
            "Station config name/path - OPTIONAL. By default this script uses the exact config saved inside "
            "the dataset at recording time (<dataset>/station_config.yaml, see dataset_recorder.py::"
            "start_session()), which is always ground truth. Only needed as a fallback for datasets recorded "
            "before this feature existed (no saved copy) - if a saved copy DOES exist, it always wins over this."
        ),
    )
    parser.add_argument(
        "--goal-time-to-start", type=float, default=3.0,
        help="Time given to reach the episode's starting pose (default 3.0s, matches the reset convention for big jumps)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Only replay the first N actions (for a quick/safe first look)")
    parser.add_argument(
        "--blocking", action="store_true",
        help=(
            "Wait for each move to fully complete before sending the next (default: off, matches the live "
            "teleop control loop's non-blocking overlapping-move behavior). NOT recommended for `ee` mode with "
            "--action-source recorded (the default) - confirmed on real hardware to cause runaway divergence "
            "and a joint-limit fault, since the recorded delta reflects the leader's raw motion under "
            "non-blocking damping; forcing full completion every step overshoots what the follower actually "
            "covered during recording. Intended to be paired with --action-source state-delta instead (`ee` "
            "mode), to test whether blocking + true-achieved deltas gives near-perfect tracking. For `joint` "
            "mode, the recorded action is already the exact absolute goal that was sent (no leader/follower "
            "mismatch), so --blocking there doesn't have that same overshoot risk - untested on real hardware."
        ),
    )
    parser.add_argument(
        "--action-source", choices=["recorded", "state-delta"], default="recorded",
        help=(
            "`ee` mode only (ignored/invalid for `joint` mode, which already records the absolute target "
            "directly - there's no delta to re-derive). 'recorded' (default): use each frame's stored `action` "
            "field (the leader's raw commanded delta). 'state-delta': ignore `action` and instead compute the "
            "delta from consecutive recorded `observation.state` frames (state[i+1] - state[i]) - i.e. what the "
            "follower's pose actually changed by during recording. Hypothesis test for whether the action/state "
            "mismatch (inherent to non-blocking recording) is the real source of replay tracking error - pair "
            "with --blocking."
        ),
    )
    parser.add_argument(
        "--action-space", choices=["absolute", "delta_joint"], default=None,
        help=(
            "`joint` mode only. 'absolute' (matches every dataset recorded before the delta-joint preprocessing "
            "existed): `action[:6]` already IS the absolute joint target. 'delta_joint' (see "
            "scripts/convert_to_delta_joint_dataset.py / PickAndInsertCube.md): `action[:6]` is a per-frame delta "
            "vs. that SAME frame's observation.state - reconstructed as target=action[:6]+state[:6], gripper "
            "(action[6]) unaffected either way. Default: auto-detect from the dataset (see "
            "_resolve_action_space()) - only pass this to override the auto-detection."
        ),
    )
    parser.add_argument(
        "--skip-staging-reset", action="store_true",
        help=(
            "Skip the safe staged-position reset (follower.reset(), i.e. `follower_single.py::reset()`'s "
            "Stage 1 - joint-space move to config.reset[side].staged_joint_positions) that would otherwise "
            "ALWAYS run before jumping to episode frame 0's recorded pose. NOT recommended: every real recorded "
            "episode started this same way (`/api/start_recording` always resets first, see `app.py`) - without "
            "it, this script interpolates DIRECTLY from whatever arbitrary pose the arm currently happens to be "
            "in straight to frame 0's pose, which (unlike the staged reset) has no known-safe intermediate "
            "waypoint and could sweep through an unexpected/unsafe configuration if the current pose is far "
            "from frame 0's."
        ),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    config = _resolve_config(dataset_path, args.config)
    command_space = config.control.command_space
    use_state_delta = args.action_source == "state-delta"
    if command_space == "joint" and use_state_delta:
        raise ValueError(
            "--action-source state-delta only applies to command_space: ee - `joint` mode's recorded `action` "
            "is already the exact absolute goal that was sent each tick, there's no leader-vs-follower delta "
            "mismatch to re-derive from consecutive states."
        )
    action_space = _resolve_action_space(dataset_path, args.action_space)
    if action_space == "delta_joint" and command_space != "joint":
        raise ValueError(
            f"--action-space delta_joint only applies to command_space: joint datasets, but this station config "
            f"has command_space: '{command_space}'."
        )
    print(f"Station command_space: '{command_space}', control_mode: '{config.control.control_mode}', "
          f"action_space: '{action_space}', goal_time_s: {config.control.goal_time_s:.4f}s "
          f"(goal_time_multiplier={config.control.goal_time_multiplier} / frequency_hz={config.control.frequency_hz})")

    dataset = LeRobotDataset(repo_id=dataset_path.name, root=str(dataset_path), episodes=[args.episode])
    n_steps = len(dataset) if args.max_steps is None else min(args.max_steps, len(dataset))
    print(f"Loaded episode {args.episode} from {dataset_path}: {len(dataset)} frames total, replaying {n_steps}.")

    follower = FollowerClient(config.follower_server_url)
    if not follower.check():
        raise RuntimeError(
            f"follower_single_server.py not reachable/connected at {config.follower_server_url} - "
            "start it and POST /connect_to_robot first."
        )

    if args.skip_staging_reset:
        print("--skip-staging-reset given: NOT running the safe staged-position reset first - moving directly "
              "from the arm's current pose to episode frame 0. Not recommended (see the flag's help text).")
    else:
        # Mirrors `/api/start_recording`'s own behavior (`app.py`) - every real
        # recorded episode ALWAYS started with this same staged reset
        # (`follower_single.py::reset()`'s Stage 1: joint-space move to a
        # fixed, known-safe `staged_joint_positions`, regardless of
        # `command_space` or wherever the arm currently is) BEFORE landing on
        # the specific (possibly randomized) start pose. Running it here too,
        # unconditionally, closes the gap where this script used to jump
        # directly from an arbitrary current pose straight to frame 0's
        # pose with no known-safe intermediate waypoint. No `pose_to_reach`
        # passed - the follow-up move below (to frame 0's EXACT recorded
        # state, gripper included) already handles final precision; this
        # call only needs to get the arm to a safe, repeatable starting
        # point first.
        print("Running the safe staged-position reset first (follower.reset()) before moving to episode frame 0...")
        follower.reset()
        time.sleep(0.2)

    first_state = dataset[0]["observation.state"].numpy()
    start_native = _dataset_state_to_native(config, first_state)
    if command_space == "joint":
        print(f"Moving to episode start joint positions (joint_0..joint_5,gripper): {start_native}")
        follower.move_to_joint_positions(start_native, blocking=True, min_time_to_move=args.goal_time_to_start)
    else:
        print(f"Moving to episode start pose (x,y,z,ax,ay,az,gripper): {start_native}")
        follower.move_to_ee_pose(start_native, blocking=True, min_time_to_move=args.goal_time_to_start)
    time.sleep(0.2)

    period_s = 1.0 / dataset.fps
    pos_errors: list[float] = []
    ori_errors: list[float] = []
    joint_errors: list[float] = []
    gripper_errors: list[float] = []
    tick_durations: list[float] = []

    # One `get_state()` read before the loop starts, then each iteration
    # reuses the READ IT DOES ANYWAY at the end (for the tracking-error
    # comparison) as the next iteration's "current pose/joints" - one HTTP
    # call per step, not two. An earlier version did a SECOND, unaccounted
    # get_state() call after the sleep (for the comparison only), which
    # silently corrupted the loop's pacing: that call's latency wasn't
    # included in `elapsed`/`sleep`, so the ACTUAL achieved step period
    # was `period_s + that call's round-trip time`, not `period_s` -
    # exactly the kind of timing drift that can produce a steady-state
    # tracking offset over many steps (see README §4a's note on how
    # delta-mode tracking lag compounds when the commanded rate doesn't
    # match the rate the follower is actually being sampled/driven at).
    current = follower.get_state()
    if command_space == "joint":
        current_native = np.asarray(current["q"], dtype=np.float32)
    else:
        current_native = np.concatenate(
            [np.asarray(current["pose"], dtype=np.float32), [float(current["gripper_pos"])]]
        ).astype(np.float32)

    if use_state_delta:
        # No next-frame state exists for the very last frame, so there's
        # one fewer usable transition than for --action-source recorded.
        n_steps = min(n_steps, len(dataset) - 1)
        print(
            "Using --action-source state-delta: deriving each step's delta from "
            "recorded observation.state[i+1] - observation.state[i] instead of the recorded action."
        )

    for i in range(n_steps):
        t_start = time.time()
        action = dataset[i]["action"].numpy()

        if command_space == "joint":
            if action_space == "delta_joint":
                # `action[:6]` is a PER-FRAME DELTA vs. THIS SAME FRAME's
                # recorded observation.state (see
                # scripts/convert_to_delta_joint_dataset.py /
                # PickAndInsertCube.md) - reconstruct the absolute target
                # using the RECORDED state (not a live follower reading -
                # replay has the exact historical state the delta was
                # originally computed against, which is even more exact
                # than live-inference reconstruction has to be).
                recorded_state = dataset[i]["observation.state"].numpy()
                target_joints = action[:6] + recorded_state[:6]
            else:
                # `action` is already the exact ABSOLUTE joint goal that was
                # sent that tick (see `control_loop.py::_tick()`'s joint
                # branch / README's `control.command_space` note) - no delta
                # math, target = action directly.
                target_joints = action[:6]
            # Gripper is absolute either way (delta-joint conversion never
            # touches it - see PickAndInsertCube.md §3) - just re-clip
            # (matches `goal_joint()`'s own clip - the arm joints were
            # already safety-clamped at record time).
            gripper_target = float(np.clip(action[6], config.control.gripper_closed, config.control.gripper_open))
            target = np.concatenate([target_joints, [gripper_target]]).astype(np.float32)
        else:
            if use_state_delta:
                recorded_pose7 = _sim_state_to_pose7(dataset[i]["observation.state"].numpy())
                recorded_next_pose7 = _sim_state_to_pose7(dataset[i + 1]["observation.state"].numpy())
                action6 = recorded_next_pose7[:6] - recorded_pose7[:6]
                gripper_recorded = recorded_next_pose7[6]
            else:
                action6 = action[:6]
                gripper_recorded = action[6]
            target_pose = current_native[:6] + action6
            gripper_target = float(np.clip(gripper_recorded, config.control.gripper_closed, config.control.gripper_open))
            target = np.concatenate([target_pose, [gripper_target]]).astype(np.float32)

        # NON-blocking by default (matches the live teleop control loop's
        # own execution mode exactly, and how this action sequence will
        # actually be consumed by a deployed policy - both run non-blocking
        # at the configured frequency_hz with overlapping moves, hil-serl's
        # own design).
        #
        # `--blocking` with `command_space: ee` + `--action-source recorded`
        # (the default) was tried and made things WORSE, not better - the
        # recorded `action` in delta mode is the LEADER's raw tick-to-tick
        # delta, not what the follower actually achieved; the ORIGINAL
        # recording's follower systematically under-travels each delta in
        # real time (goal_time_s > the tick period means moves overlap and
        # never fully complete before the next one arrives - this is the
        # same "lag" that's normal during live teleop too). Forcing
        # `blocking=True` there makes EVERY step fully complete its full
        # recorded delta - MORE distance per step than the original
        # follower ever actually covered - and that overshoot compounds
        # with nothing to correct it, producing runaway divergence (and,
        # confirmed live, eventually a real joint-limit fault) instead of
        # the steady, bounded tracking gap non-blocking replay produces.
        #
        # `--blocking` with `--action-source state-delta` (`ee` mode) is a
        # DIFFERENT situation: the delta being replayed there is the
        # follower's own achieved state-to-state motion, not the leader's
        # un-achieved intent, so forcing full completion doesn't have that
        # same overshoot mechanism - this is the combination worth testing.
        # `command_space: joint`'s recorded action is likewise already the
        # exact absolute goal sent (not the leader's raw intent), so the
        # same overshoot mechanism doesn't obviously apply there either -
        # but this hasn't been tested against real hardware yet.
        if command_space == "joint":
            follower.move_to_joint_positions(target, blocking=args.blocking, min_time_to_move=config.control.goal_time_s)
        else:
            follower.move_to_ee_pose(target, blocking=args.blocking, min_time_to_move=config.control.goal_time_s)

        elapsed = time.time() - t_start
        time.sleep(max(0.0, period_s - elapsed))

        # Single get_state() for BOTH this step's tracking-error comparison
        # AND next step's delta baseline.
        next_state = follower.get_state()
        if command_space == "joint":
            current_native = np.asarray(next_state["q"], dtype=np.float32)
        else:
            current_native = np.concatenate(
                [np.asarray(next_state["pose"], dtype=np.float32), [float(next_state["gripper_pos"])]]
            ).astype(np.float32)

        tick_duration = time.time() - t_start
        tick_durations.append(tick_duration)
        achieved_hz = 1.0 / tick_duration if tick_duration > 0 else float("inf")

        if i + 1 < len(dataset):
            recorded_next_native = _dataset_state_to_native(config, dataset[i + 1]["observation.state"].numpy())
            if command_space == "joint":
                joint_err, grip_err = _joint_error(current_native, recorded_next_native)
                joint_errors.append(joint_err)
                gripper_errors.append(grip_err)
                print(
                    f"step {i:4d}: joint_err={np.degrees(joint_err):6.2f}deg  gripper_err={grip_err * 1000:5.1f}mm  "
                    f"freq={achieved_hz:5.1f}Hz (target {dataset.fps}Hz)"
                )
            else:
                pos_err, ori_err, grip_err = _pose_error(current_native, recorded_next_native)
                pos_errors.append(pos_err)
                ori_errors.append(ori_err)
                gripper_errors.append(grip_err)
                print(
                    f"step {i:4d}: pos_err={pos_err * 1000:6.1f}mm  "
                    f"ori_err={np.degrees(ori_err):5.2f}deg  gripper_err={grip_err * 1000:5.1f}mm  "
                    f"freq={achieved_hz:5.1f}Hz (target {dataset.fps}Hz)"
                )

    mean_hz = 1.0 / np.mean(tick_durations) if tick_durations else float("nan")
    if command_space == "joint" and joint_errors:
        print(
            f"\nMean tracking error over {len(joint_errors)} steps: "
            f"joint={np.degrees(np.mean(joint_errors)):.2f}deg  "
            f"gripper={np.mean(gripper_errors) * 1000:.1f}mm"
        )
        print(f"Max tracking error: joint={np.degrees(np.max(joint_errors)):.2f}deg")
        print(f"Average control frequency during replay: {mean_hz:.2f}Hz (target: {dataset.fps}Hz)")
    elif pos_errors:
        print(
            f"\nMean tracking error over {len(pos_errors)} steps: "
            f"pos={np.mean(pos_errors) * 1000:.1f}mm  "
            f"ori={np.degrees(np.mean(ori_errors)):.2f}deg  "
            f"gripper={np.mean(gripper_errors) * 1000:.1f}mm"
        )
        print(
            f"Max tracking error: "
            f"pos={np.max(pos_errors) * 1000:.1f}mm  ori={np.degrees(np.max(ori_errors)):.2f}deg"
        )
        print(f"Average control frequency during replay: {mean_hz:.2f}Hz (target: {dataset.fps}Hz)")


if __name__ == "__main__":
    main()

