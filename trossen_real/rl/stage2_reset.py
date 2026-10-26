# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Reset into SARM stage 2 ("cube above the hole") for stage-2 residual RL.

`Stage2ResetFollower` is a `FollowerClient` whose `reset()` replaces the normal
staged reset (which starts at stage 0, cube on the table):

  1. joint move (blocking) to `STAGE2_JOINTS`, gripper open
  2. (no leader move here: see "Leader" below)
  3. wait until the operator has put the cube between the fingers and typed
     0 + Enter
  4. close the gripper on the cube, wait `GRIP_SETTLE_S`
  5. randomize the start pose a little (`mode`):
       "joint" - uniform noise on `JOINT_NOISE_INDICES` (+/- `JOINT_NOISE`),
                 clipped to `JOINT_MIN`/`JOINT_MAX` (robosuite-style
                 initialization noise, which is on joints)
       "ee"    - read the EE pose, add uniform +/- `EE_XYZ_OFFSET` (m) and
                 +/- `EE_ANGLE_OFFSET` (rad), move there in EE control; the
                 follower server clips xyz to the station's ee_bounds. Then
                 everything after is joint control again.
       "none"  - stay at `STAGE2_JOINTS`
     The gripper stays closed on the cube in every mode.

Leader (intervention): the leader is synced once, AFTER the reset, by whoever
called it (infer app `InferenceLoop.reset_and_sync`, RL
`TrossenResidualEnv.reset`), to `leader_sync_joints()`: the follower's joints
with the gripper at the COMMANDED `GRIPPER_CUBE_CLOSE`, not the measured value.
With the cube held the follower's gripper reads the cube's width (~0.0115) while
0.004 is what squeezes it; syncing the leader to the reading would make the
follower loosen its grip to 0.0115 on the first intervened tick (the follower
copies the leader's gripper), and the cube could slip. Synced to 0.004 the
leader's (empty) gripper closes fully and the follower keeps squeezing.

Selected with `reset.single.mode` in the station config (`make_follower`).
Test it with `scripts/TrossenStation3Real/RL/test_stage2_reset.py`.

The constants are hard-coded on purpose (no config params yet); where they come
from is noted next to each. See `scripts/TrossenStation3Real/RL/STAGE2_RL.md`.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from trossen_real.teleop.follower_client import FollowerClient

logger = logging.getLogger(__name__)

# ---- hard-coded from the stage-2 dataset -----------------------------------------
# Source: trossen_real/datasets/iterativeTraining/TrossenStation3/RL_Stage2/
#         PickCubeAndInsert_TS3_Stage2_RL_deltajoint_v21, frame 0 of all 99
#         episodes (= where SARM first detected stage 2). Joint order:
#         0 base, 1 shoulder, 2 elbow, 3 wrist pitch, 4 wrist yaw, 5 wrist roll.
#
# Stage-2 start pose: per-joint MEDIAN of observation.state at those frames
# (joints rad, gripper m - commanded open here; it is closed on the cube later).
STAGE2_JOINTS = [-0.2657, 1.6978, 1.3430, -1.1645, -0.0032, -0.2020, 0.040]
GRIPPER_OPEN = 0.040
# The commanded gripper at those frames is 0.004 (= control.gripper_closed) while
# the reading is ~0.0115 (the cube's width): command 0.004 to hold the cube.
GRIPPER_CUBE_CLOSE = 0.004
GRIP_SETTLE_S = 2.0
MOVE_TIME_S = 3.0  # = reset.goal_time_s

# "joint" mode: small uniform noise on base / shoulder / elbow / wrist pitch only.
# Wrist yaw/roll are left alone: they set the cube's orientation to the square
# hole. Starting values, deliberately small (robosuite's default init noise is
# gaussian sigma 0.02 rad on every joint): tune with test_stage2_reset.py.
JOINT_NOISE_INDICES = [0, 1, 2, 3]
JOINT_NOISE = [0.05, 0.03, 0.03, 0.03]  # rad, +/-
# Hard limits after noise: the 10th / 90th percentile of the same frames.
JOINT_MIN = [-0.3482, 1.5755, 1.1426, -1.2219, -0.0293, -0.3028]
JOINT_MAX = [0.0605, 1.8336, 1.4870, -1.0904, 0.0185, 0.0444]

# "ee" mode: offsets around the EE pose reached at STAGE2_JOINTS. At those
# frames the EE spread (std) is x 0.026, y 0.050, z 0.011 m and ~0.03-0.04 rad
# per angle; these offsets stay well inside that. xyz is also clipped to the
# station's ee_bounds by the follower server.
EE_XYZ_OFFSET = [0.015, 0.015, 0.010]  # m, +/- x, y, z
EE_ANGLE_OFFSET = [0.03, 0.03, 0.03]  # rad, +/- ax, ay, az


class Stage2ResetFollower(FollowerClient):
    def __init__(self, base_url: str, mode: str = "joint", seed: int | None = None,
                 confirm=input, request_timeout_s: float = 10.0) -> None:
        super().__init__(base_url, request_timeout_s=request_timeout_s)
        if mode not in ("joint", "ee", "none"):
            raise ValueError(f"mode must be 'joint', 'ee' or 'none', got {mode!r}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.confirm = confirm  # injectable for tests
        self.last_target: dict | None = None

    # FollowerClient.reset(pose_to_reach=None) signature, so TrossenResidualEnv can call it unchanged.
    def reset(self, pose_to_reach=None) -> None:  # noqa: ARG002
        staged = np.asarray(STAGE2_JOINTS, dtype=np.float32).copy()
        staged[6] = GRIPPER_OPEN
        logger.info("stage-2 reset: moving to %s", np.round(staged, 4).tolist())
        self.move_to_joint_positions(staged, blocking=True, min_time_to_move=MOVE_TIME_S)

        while True:
            answer = self.confirm("Put the cube between the gripper fingers, then type 0 + Enter: ")
            if answer.strip() == "0":
                break
        self.move_gripper(GRIPPER_CUBE_CLOSE, blocking=True, min_time_to_move=1.0)
        time.sleep(GRIP_SETTLE_S)

        target = self.sample_target()
        self.last_target = target
        if target["space"] == "joint":
            self.move_to_joint_positions(target["joints"], blocking=True, min_time_to_move=MOVE_TIME_S)
        elif target["space"] == "ee":
            self.move_to_ee_pose(target["pose7"], blocking=True, min_time_to_move=MOVE_TIME_S)
        logger.info("stage-2 reset done (%s): %s", self.mode, {k: v for k, v in target.items() if k != "space"})

    def leader_sync_joints(self, follower_q) -> list[float]:
        """Where to sync the leader after this reset: the follower's joints, gripper
        at the commanded GRIPPER_CUBE_CLOSE (see module docstring, "Leader")."""
        q = [float(x) for x in follower_q]
        q[6] = GRIPPER_CUBE_CLOSE
        return q

    def sample_target(self) -> dict:
        """The randomized start target for the current mode (no motion)."""
        if self.mode == "joint":
            q = np.asarray(STAGE2_JOINTS, dtype=np.float64).copy()
            noise = self.rng.uniform(-np.asarray(JOINT_NOISE), np.asarray(JOINT_NOISE))
            q[list(JOINT_NOISE_INDICES)] += noise
            q[:6] = np.clip(q[:6], JOINT_MIN, JOINT_MAX)
            q[6] = GRIPPER_CUBE_CLOSE
            return {"space": "joint", "joints": q.astype(np.float32).tolist(),
                    "noise": dict(zip(JOINT_NOISE_INDICES, np.round(noise, 4).tolist()))}
        if self.mode == "ee":
            pose = np.asarray(self.get_state()["pose"], dtype=np.float64)[:6]
            dxyz = self.rng.uniform(-np.asarray(EE_XYZ_OFFSET), np.asarray(EE_XYZ_OFFSET))
            dang = self.rng.uniform(-np.asarray(EE_ANGLE_OFFSET), np.asarray(EE_ANGLE_OFFSET))
            target = np.concatenate([pose[:3] + dxyz, pose[3:6] + dang, [GRIPPER_CUBE_CLOSE]])
            return {"space": "ee", "pose7": target.astype(np.float32).tolist(),
                    "dxyz": np.round(dxyz, 4).tolist(), "dangle": np.round(dang, 4).tolist()}
        return {"space": "none"}


class CubeConfirm:
    """`confirm` for Stage2ResetFollower when the reset runs inside a server
    (the infer app): `__call__` blocks the reset until `confirm()` is called
    (the UI's "Cube placed" button), or raises if `cancel()` is called (Stop)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._cancelled = threading.Event()
        self.waiting = False
        self.prompt = ""

    def __call__(self, prompt: str) -> str:
        self._event.clear()
        self._cancelled.clear()
        self.prompt, self.waiting = prompt, True
        logger.info("stage-2 reset: waiting for the cube (%s)", prompt)
        try:
            while not self._event.wait(0.2):
                if self._cancelled.is_set():
                    raise RuntimeError("stage-2 reset cancelled while waiting for the cube")
            return "0"
        finally:
            self.waiting = False

    def confirm(self) -> bool:
        """True if a reset was waiting (and is now released)."""
        if not self.waiting:
            return False
        self._event.set()
        return True

    def cancel(self) -> None:
        self._cancelled.set()


def make_follower(base_url: str, reset_mode: str = "normal", confirm=input) -> FollowerClient:
    """The follower client for a station's `reset.<side>.mode`: a plain
    FollowerClient for "normal", a Stage2ResetFollower for "stage2_*"."""
    if reset_mode == "normal":
        return FollowerClient(base_url)
    if not reset_mode.startswith("stage2_"):
        raise ValueError(f"unknown reset mode {reset_mode!r}")
    return Stage2ResetFollower(base_url, mode=reset_mode.removeprefix("stage2_"), confirm=confirm)


def leader_sync_joints(follower, follower_q) -> list[float]:
    """Leader sync target after `follower.reset()`: the follower's own choice if
    it has one (Stage2ResetFollower: commanded gripper on the cube), else its
    measured joints (normal reset: gripper open, measured == commanded)."""
    fn = getattr(follower, "leader_sync_joints", None)
    return fn(follower_q) if fn is not None else [float(x) for x in follower_q]
