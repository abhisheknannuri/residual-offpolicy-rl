# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Follower-only Trossen arm control engine - single arm.

Owned exclusively by `follower_single_server.py`. Never talks to a leader -
in this split architecture the leader is a separate class
(`leader/trossen_leader_single.py`) used directly by the teleop
orchestrator (`teleop/app.py`), potentially in a different process/machine
entirely.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from trossen_real.arm_driver import HAS_TROSSEN_SDK, MockArm, RealArm, make_arm
from trossen_real.config import StationConfig

logger = logging.getLogger(__name__)


class TrossenFollowerSingle:
    """Owns the follower arm for one station and exposes teleop/RL primitives.

    Backs `follower_single_server.py`'s hil-serl-shaped HTTP routes
    (`/pose`, `/reset`, `/move_gripper`, `/getstate`, ...).
    """

    def __init__(self, config: StationConfig) -> None:
        self.config = config
        self.sides = config.sides
        self._follower: dict[str, RealArm | MockArm] = {}
        self.connected = False
        # Guards every arm-commanding call (goal_ee/reset/set_gripper*) so
        # concurrent HTTP requests never issue conflicting commands to the
        # same arm at the same time.
        self._cmd_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        try:
            for side in self.sides:
                follower = make_arm(self.config.follower_ips[side], role="follower")
                follower.connect()
                self._follower[side] = follower
        except Exception:
            # Don't leave a half-connected arm dangling (e.g. left arm connects
            # fine, right arm's configure() fails) - tear down whatever did
            # connect so a retried connect() starts from a clean slate.
            logger.exception("connect() failed partway through; disconnecting any arms that did connect.")
            self.disconnect()
            raise
        self.connected = True

    def disconnect(self, skip_staged_position: bool = False) -> None:
        """Move to the staged position, then an all-zero "sleep" position, then
        release the connection (mirrors hil-serl's
        `windowxai_follower_server.py::disconnect()`, which does the same
        staged->sleep sequence before `driver.cleanup()`).

        Each stage (mode switch, staged move, sleep move) is attempted and
        error-handled INDEPENDENTLY - if e.g. the staged move fails partway
        (transient comms error, driver fault), we still attempt the
        sleep-pose move as a best-effort fallback, rather than leaving the
        arm sitting at some random intermediate pose and cutting power to
        it immediately. Only if the driver itself is unresponsive (both
        moves fail) do we give up and just release the connection - safer
        motion isn't possible if the driver can't be commanded at all, and
        there's nothing more we can do about that case.

        `skip_staged_position`: default False preserves the staged->sleep
        sequence above exactly as always. If True, none of that motion is
        commanded at all - the arm is released from wherever it currently
        is (e.g. mid-episode, already at/near a safe pose from other means,
        or when you specifically want a fast disconnect without waiting on
        the staged/sleep move's `goal_time_s`). Callers opting into this are
        responsible for the arm being in a position that's safe to just cut
        connection from - this bypasses the "always leave it at a known-safe
        sleep pose" guarantee the default behavior provides.

        Safe to call multiple times (e.g. once via an explicit API route,
        again via the atexit/signal-handler backstop on process shutdown) -
        the arm dict is cleared after the first pass, so a repeat call is a
        no-op. Note: `_cleanup()`/the signal handlers in
        `follower_single_server.py` always call this with no arguments
        (i.e. `skip_staged_position=False`) - an unexpected process exit
        still gets the safe staged/sleep move, regardless of what a prior
        explicit `/disconnect_from_robot` call requested.
        """
        with self._cmd_lock:
            for side in self.sides:
                if side not in self._follower:
                    continue
                follower = self._follower[side]

                if skip_staged_position:
                    logger.info(
                        "disconnect(skip_staged_position=True) for follower (%s): releasing the "
                        "connection immediately from the current position, no staged/sleep move.", side
                    )
                    follower.disconnect()
                    continue

                reset_cfg = self.config.reset.get(side)
                goal_time = reset_cfg.goal_time_s if reset_cfg else 2.0
                staged = reset_cfg.staged_joint_positions if reset_cfg else None
                sleep_pose = [0.0] * 7

                try:
                    follower.set_mode_position()
                except Exception:
                    logger.exception(
                        "Error switching follower (%s) to position mode before disconnect; "
                        "still attempting the staged/sleep moves.", side
                    )

                if staged is not None:
                    try:
                        follower.set_joint_positions(staged, goal_time=goal_time, blocking=True)
                    except Exception:
                        logger.exception(
                            "Error moving follower (%s) to staged position before disconnect; "
                            "still attempting the sleep-pose move.", side
                        )

                try:
                    follower.set_joint_positions(sleep_pose, goal_time=goal_time, blocking=True)
                except Exception:
                    logger.exception(
                        "Error moving follower (%s) to sleep position before disconnect - driver may be "
                        "unresponsive; releasing the connection anyway.", side
                    )

                follower.disconnect()

        self._follower.clear()
        self.connected = False

    def health(self) -> dict[str, bool]:
        return {side: self._follower[side].is_connected() if side in self._follower else False for side in self.sides}

    def using_mock_hardware(self) -> bool:
        return not HAS_TROSSEN_SDK

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def get_follower_state(self) -> dict[str, np.ndarray]:
        """Per side: 7D = absolute pose (6) + absolute gripper (1)."""
        states = {}
        for side in self.sides:
            arm = self._follower[side]
            pose = arm.get_cartesian_positions()
            gripper = arm.get_gripper_position()
            states[side] = np.concatenate([pose, [gripper]]).astype(np.float32)
        return states

    def get_follower_joint_state(self, side: str) -> dict[str, np.ndarray]:
        """q/dq/efforts/accelerations (7D each: 6 arm joints + gripper) for one follower side."""
        arm = self._follower[side]
        return {
            "q": arm.get_joint_positions(),
            "dq": arm.get_joint_velocities(),
            "efforts": arm.get_joint_efforts(),
            "accelerations": arm.get_joint_accelerations(),
        }

    def _clip_xyz(self, side: str, xyz: np.ndarray) -> np.ndarray:
        bounds = self.config.ee_bounds[side]
        return np.clip(xyz, bounds.xyz_min, bounds.xyz_max)

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------
    def goal_ee(
        self,
        side: str,
        pose7: np.ndarray,
        goal_time: float | None = None,
        blocking: bool = False,
    ) -> None:
        """Command an ABSOLUTE target [x,y,z,ax,ay,az,gripper] for one follower side.

        This is the RL-gym-wrapper-facing entry point (mirrors
        `windowxai_follower_server.py`'s `/pose` semantics: the caller computes
        the absolute target, this just clips + sends it). `goal_time`/`blocking`
        default to the station's regular per-tick `control.goal_time_s` if not
        given, but can be overridden per call (matches `/pose`'s optional
        `min_time_to_move`/`blocking` body fields).
        """
        pose7 = np.asarray(pose7, dtype=np.float32)
        target_pose = pose7[:6].copy()
        target_pose[:3] = self._clip_xyz(side, target_pose[:3])
        goal_time = self.config.control.goal_time_s if goal_time is None else goal_time
        gripper_target = float(
            np.clip(pose7[6], self.config.control.gripper_closed, self.config.control.gripper_open)
        )
        with self._cmd_lock:
            follower = self._follower[side]
            follower.set_cartesian_positions(target_pose, goal_time=goal_time, blocking=blocking)
            follower.set_gripper_position(gripper_target, goal_time=goal_time, blocking=blocking)

    def goal_joint(
        self,
        side: str,
        joint7: np.ndarray,
        goal_time: float | None = None,
        blocking: bool = False,
    ) -> None:
        """Command an ABSOLUTE target [joint_0..joint_5, gripper] for one follower side.

        `command_space: joint` counterpart to `goal_ee()` above - matches
        Trossen's own `lerobot_trossen` reference (`widowxai_follower.py::
        send_action()`): the caller (control loop / a future joint-space RL
        gym-wrapper) computes the absolute joint target, this just clips the
        gripper component + sends it via a SINGLE `set_all_positions()` call
        (unlike `goal_ee()`, which needs two separate SDK calls - cartesian
        then gripper - since the trossen_arm SDK has no combined cartesian+
        gripper call, but joint space naturally covers both in one array).
        The caller is responsible for any safety clamp on the 6 arm joints
        themselves (see `control_loop.py`'s `joint_max_relative_target`) -
        this method only clips the gripper, mirroring `goal_ee()`.
        """
        joint7 = np.asarray(joint7, dtype=np.float32)
        goal_time = self.config.control.goal_time_s if goal_time is None else goal_time
        gripper_target = float(
            np.clip(joint7[6], self.config.control.gripper_closed, self.config.control.gripper_open)
        )
        target = np.concatenate([joint7[:6], [gripper_target]]).astype(np.float32)
        with self._cmd_lock:
            follower = self._follower[side]
            follower.set_joint_positions(target.tolist(), goal_time=goal_time, blocking=blocking)

    # ------------------------------------------------------------------
    # Reset / gripper
    # ------------------------------------------------------------------
    def reset(
        self,
        rng: np.random.Generator | None = None,
        pose_to_reach: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Two-stage follower reset, per side.

        Stage 1 (always): joint-space move to `staged_joint_positions`
        (gripper commanded open via that array's 7th element - no separate
        gripper move needed). Stage 2: if `pose_to_reach[side]` is given,
        move there directly (mirrors hil-serl's `reset(pose_to_reach=...)`);
        otherwise, only if `randomize_ee`, read back the EE pose stage 1
        landed on, sample a random offset around it, and move there - always
        via cartesian EE-pose control (never joint control).
        """
        rng = rng or np.random.default_rng()
        pose_to_reach = pose_to_reach or {}
        with self._cmd_lock:
            for side in self.sides:
                reset_cfg = self.config.reset[side]
                goal_time = reset_cfg.goal_time_s
                follower = self._follower[side]

                # Stage 1: joint-space move to the known-safe staged configuration.
                # `staged_joint_positions`'s 7th element IS the gripper target,
                # so `set_joint_positions` (-> `driver.set_all_positions()`)
                # already moves the gripper in the same call - no separate
                # gripper move needed here (that used to double the time this
                # stage took, for nothing).
                follower.set_mode_position()
                follower.set_joint_positions(reset_cfg.staged_joint_positions, goal_time=goal_time, blocking=True)

                staged_pose = follower.get_cartesian_positions()

                if side in pose_to_reach:
                    # Explicit override (hil-serl's `pose_to_reach` param) - go
                    # straight there via cartesian control, skip randomization.
                    target_pose = np.asarray(pose_to_reach[side], dtype=np.float32)[:6].copy()
                    target_pose[:3] = self._clip_xyz(side, target_pose[:3])
                    follower.set_cartesian_positions(target_pose, goal_time=goal_time, blocking=True)
                    follower.set_gripper_position(self.config.control.gripper_open, goal_time=goal_time, blocking=True)
                elif reset_cfg.randomize_ee:
                    # Stage 2 (optional): EE-space randomization around the staged pose.
                    xyz_offset = np.asarray(reset_cfg.xyz_offset, dtype=np.float32)
                    xyz = staged_pose[:3] + rng.uniform(-xyz_offset, xyz_offset).astype(np.float32)
                    xyz = self._clip_xyz(side, xyz)

                    if reset_cfg.randomize_angles:
                        angle_offset = np.asarray(reset_cfg.angle_offset, dtype=np.float32)
                        angles = staged_pose[3:6] + rng.uniform(-angle_offset, angle_offset).astype(np.float32)
                    else:
                        angles = staged_pose[3:6]

                    target_pose = np.concatenate([xyz, angles]).astype(np.float32)
                    follower.set_cartesian_positions(target_pose, goal_time=goal_time, blocking=True)
                    follower.set_gripper_position(self.config.control.gripper_open, goal_time=goal_time, blocking=True)

    def set_gripper(self, side: str, open_: bool) -> None:
        target = self.config.control.gripper_open if open_ else self.config.control.gripper_closed
        with self._cmd_lock:
            follower = self._follower[side]
            follower.set_gripper_position(target, goal_time=self.config.control.goal_time_s, blocking=False)

    def set_gripper_position(
        self,
        side: str,
        position: float,
        goal_time: float | None = None,
        blocking: bool = False,
    ) -> None:
        """Command an arbitrary continuous gripper position (not just open/closed).

        Mirrors `windowxai_follower_server.py`'s `/move_gripper` route. Always
        clipped to `[gripper_closed, gripper_open]` for safety.
        """
        goal_time = self.config.control.goal_time_s if goal_time is None else goal_time
        target = float(np.clip(position, self.config.control.gripper_closed, self.config.control.gripper_open))
        with self._cmd_lock:
            follower = self._follower[side]
            follower.set_gripper_position(target, goal_time=goal_time, blocking=blocking)

    def is_gripper_open(self, side: str) -> bool:
        pos = self._follower[side].get_gripper_position()
        midpoint = (self.config.control.gripper_open + self.config.control.gripper_closed) / 2.0
        return pos >= midpoint

