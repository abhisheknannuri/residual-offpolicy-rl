# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Low-level `trossen_arm` SDK wrapper for one physical arm (leader or follower).

Shared by `follower/follower_single.py` (follower engine) and
`leader/trossen_leader_single.py` (direct leader read) - both need the same
connect/mode-switch/cleanup/mock-fallback behavior for a single arm, just
used differently (follower: position mode + commanded; leader: freedrive +
read-only).

When the `trossen_arm` SDK is not importable (e.g. developing away from the
robot), every arm is replaced by an in-process `MockArm` so the rest of the
stack can still be exercised end-to-end with synthetic data.

Real-arm calls below are reconstructed from the reference implementation at
hil-serl's `trossen_arm_mujoco/trossen_real/robot_hardware.py` and
`serl_robot_infra/robot_servers/windowxai_follower_server.py`, and verified
directly against the installed `trossen_arm` SDK's own docstrings and
against real hardware.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    import trossen_arm

    HAS_TROSSEN_SDK = True
except ImportError:
    trossen_arm = None
    HAS_TROSSEN_SDK = False

# Reasonable free-floating default pose for the mock arm: [x, y, z, ax, ay, az] (m, rad).
_MOCK_DEFAULT_POSE = np.array([0.25, 0.0, 0.20, 0.0, 0.0, 0.0], dtype=np.float32)


class ArmError(RuntimeError):
    """Raised when an arm cannot be connected to or commanded."""


class MockArm:
    """Stand-in for a `trossen_arm.TrossenArmDriver` when the SDK/hardware is unavailable.

    Free-floats to whatever pose it's last commanded to (no real dynamics) -
    enough to exercise the control loop, recording, and UI end to end.
    """

    def __init__(self, ip: str, role: str) -> None:
        self.ip = ip
        self.role = role
        self._pose = _MOCK_DEFAULT_POSE.copy()
        self._gripper = 0.044
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        logger.warning("[MockArm] %s (%s): trossen_arm SDK unavailable, using mock.", self.ip, self.role)

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_mode_position(self) -> None:
        pass

    def set_mode_freedrive(self) -> None:
        pass

    def get_cartesian_positions(self) -> np.ndarray:
        return self._pose.copy()

    def get_gripper_position(self) -> float:
        return self._gripper

    def get_joint_positions(self) -> np.ndarray:
        """6 arm joints (always 0 in mock) + gripper."""
        return np.array([0.0] * 6 + [self._gripper], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def get_joint_efforts(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def get_joint_accelerations(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def set_cartesian_positions(self, pose: np.ndarray, goal_time: float = 0.1, blocking: bool = False) -> None:
        self._pose = np.asarray(pose, dtype=np.float32).copy()

    def set_gripper_position(self, pos: float, goal_time: float = 0.1, blocking: bool = False) -> None:
        self._gripper = float(pos)

    def set_joint_positions(self, joint_positions: list[float], goal_time: float = 2.0, blocking: bool = False) -> None:
        """joint_positions: 6 joints (rad) + gripper (m). Mock just snaps its EE pose to the mock default."""
        self._pose = _MOCK_DEFAULT_POSE.copy()
        self._gripper = float(joint_positions[6]) if len(joint_positions) > 6 else self._gripper


class RealArm:
    """Thin wrapper around `trossen_arm.TrossenArmDriver` for one physical arm."""

    # `configure()`'s own default is 20.0s - way too long to block a UI
    # request on for a simple typo'd/wrong IP (observed: a bad IP config
    # makes /api/connect hang the Flask worker thread for 20s, or 40s+ with
    # the retry below, and since /api/connect holds `state.lock` the whole
    # time, it also blocks the shutdown signal handler from acquiring that
    # same lock - "can't cancel, can't even Ctrl+C out" on real hardware).
    # 5s is still generous for a real, reachable controller (observed
    # successful connects complete in well under 1s).
    CONNECT_TIMEOUT_S = 5.0

    def __init__(self, ip: str, role: str) -> None:
        self.ip = ip
        self.role = role  # "leader" | "follower"
        self._driver = trossen_arm.TrossenArmDriver()
        self._connected = False

    def connect(self) -> None:
        ee_type = (
            trossen_arm.StandardEndEffector.wxai_v0_leader
            if self.role == "leader"
            else trossen_arm.StandardEndEffector.wxai_v0_follower
        )
        # The controller's TCP server reliably drops the FIRST connection
        # attempt after being idle for a while ("... due to Connection reset
        # by peer") - a fast, distinctive failure, then accepts the very
        # next attempt. Observed consistently on both leader and follower.
        # Retry ONLY on that specific message - a genuine timeout (wrong IP,
        # unreachable host, powered-off controller) is a completely
        # different failure mode where retrying just doubles the wait for
        # no benefit; let that raise immediately instead.
        try:
            self._driver.configure(trossen_arm.Model.wxai_v0, ee_type, self.ip, False, timeout=self.CONNECT_TIMEOUT_S)
        except Exception as exc:
            if "Connection reset by peer" not in str(exc):
                raise
            logger.warning(
                "First connect attempt to %s (%s) failed (known transient "
                "'Connection reset by peer' hiccup) - retrying once.", self.ip, self.role
            )
            time.sleep(0.5)
            self._driver.configure(trossen_arm.Model.wxai_v0, ee_type, self.ip, False, timeout=self.CONNECT_TIMEOUT_S)
        time.sleep(0.1)
        if self.role == "leader":
            self.set_mode_freedrive()
        else:
            self.set_mode_position()
        self._connected = True

    def disconnect(self) -> None:
        """Release the underlying `TrossenArmDriver` connection.

        Safe to call more than once (e.g. once via an explicit API route,
        again via the atexit/signal-handler backstop on process shutdown).
        """
        if not self._connected:
            return
        try:
            self._driver.cleanup()
        except Exception:
            logger.exception("Error while cleaning up arm driver for %s (%s); ignoring.", self.ip, self.role)
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_mode_position(self) -> None:
        self._driver.set_all_modes(trossen_arm.Mode.position)

    def set_mode_freedrive(self) -> None:
        """Gravity-compensated freedrive so a human can backdrive the arm."""
        self._driver.set_all_modes(trossen_arm.Mode.external_effort)
        self._driver.set_all_external_efforts([0.0] * 7, 0.0, False)

    def get_cartesian_positions(self) -> np.ndarray:
        return np.asarray(self._driver.get_cartesian_positions(), dtype=np.float32)

    def get_gripper_position(self) -> float:
        return float(self._driver.get_gripper_position())

    def get_joint_positions(self) -> np.ndarray:
        """6 arm joints (rad) + gripper (m), via `driver.get_all_positions()`."""
        return np.asarray(self._driver.get_all_positions(), dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        return np.asarray(self._driver.get_all_velocities(), dtype=np.float32)

    def get_joint_efforts(self) -> np.ndarray:
        return np.asarray(self._driver.get_all_efforts(), dtype=np.float32)

    def get_joint_accelerations(self) -> np.ndarray:
        return np.asarray(self._driver.get_all_accelerations(), dtype=np.float32)

    def set_cartesian_positions(self, pose: np.ndarray, goal_time: float = 0.1, blocking: bool = False) -> None:
        self._driver.set_cartesian_positions(
            goal_positions=trossen_arm.ArrayDouble6(list(pose[:6])),
            interpolation_space=trossen_arm.InterpolationSpace.cartesian,
            goal_time=goal_time,
            blocking=blocking,
        )

    def set_gripper_position(self, pos: float, goal_time: float = 0.1, blocking: bool = False) -> None:
        self._driver.set_gripper_position(float(pos), goal_time=goal_time, blocking=blocking)

    def set_joint_positions(self, joint_positions: list[float], goal_time: float = 2.0, blocking: bool = False) -> None:
        """joint_positions: 6 joints (rad) + gripper (m). Used only for the stage-1 reset move."""
        self._driver.set_all_positions(list(joint_positions), goal_time=goal_time, blocking=blocking)


def make_arm(ip: str, role: str) -> RealArm | MockArm:
    if HAS_TROSSEN_SDK:
        return RealArm(ip, role)
    return MockArm(ip, role)
