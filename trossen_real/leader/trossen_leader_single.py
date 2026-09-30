# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Direct (non-HTTP) leader arm control - single arm.

Mirrors hil-serl's `serl_robot_infra/trossen_env/teleoperator/widowxai_leader.py`:
the leader is read directly via the `trossen_arm` SDK (gravity-compensated
freedrive, so a human can backdrive it), NOT through a Flask server - only
the process doing teleop data collection needs the leader's pose, so there's
no HTTP hop for it.

Additionally supports dynamic mode-switching (freedrive <-> position mode) so
the leader can track an autonomous policy during non-intervention periods,
and instantly yield to human teleop when an intervention is triggered.

Reuses the same `RealArm`/`MockArm` wrapper as `arm_driver.py` so
connect/freedrive/cleanup behavior (and the mock fallback when the SDK/
hardware is unavailable) stays identical and isn't duplicated.
"""

from __future__ import annotations

import logging

import numpy as np

from trossen_real.arm_driver import MockArm, RealArm, make_arm

logger = logging.getLogger(__name__)


class TrossenSingleLeader:
    """One leader arm, read directly via the SDK. Not a server - just a class.
    
    Can operate in freedrive (for human teleop) or position mode (to mirror a policy).
    """

    def __init__(self, ip: str, start_in_freedrive: bool = True) -> None:
        self.ip = ip
        self.start_in_freedrive = start_in_freedrive
        self._arm: RealArm | MockArm | None = None
        self._prev_pose: np.ndarray | None = None
        self._current_mode: str | None = None

    def connect(self) -> None:
        self._arm = make_arm(self.ip, role="leader")
        self._arm.connect()
        # The wrapper connect() initializes it in freedrive for the leader role
        self._current_mode = "freedrive"
        
        if not self.start_in_freedrive:
            self.set_position_mode()
            
        self._prev_pose = self._arm.get_cartesian_positions()

    def disconnect(self, park_waypoints: list[list[float]] | None = None, goal_time: float = 3.0) -> None:
        """Disconnect the leader, optionally parking it through a sequence of
        known joint configurations first.

        If `park_waypoints` is given (e.g. `[staged_joint_positions,
        sleep_pose]`, mirroring the follower's own staged->sleep disconnect
        sequence), briefly leaves freedrive to move through each waypoint in
        order (position-mode, blocking, joint-space) before releasing the
        connection. Joint-space (not cartesian) is used deliberately - the
        leader could be resting anywhere freedrive left it, and a
        cartesian/IK-based move over a large, arbitrary starting
        configuration can behave unpredictably; joint-space moves are
        deterministic regardless of the starting configuration (matches how
        the follower's own disconnect()/reset() do their big moves).

        This WILL fight the operator's hand if they're still holding the
        leader at disconnect time (leaving freedrive means the arm stops
        being backdrivable for the duration of these moves), so only pass
        waypoints here if that tradeoff is acceptable for your setup. Best
        effort: if a move fails partway (e.g. transient comms error), this
        is logged and the remaining waypoints/connection release are still
        attempted rather than getting stuck.
        """
        if self._arm is None:
            return
        if park_waypoints:
            try:
                self.set_position_mode()
            except Exception:
                logger.exception("Error switching leader to position mode before parking; disconnecting anyway.")
            for waypoint in park_waypoints:
                try:
                    self._arm.set_joint_positions(waypoint, goal_time=goal_time, blocking=True)
                except Exception:
                    logger.exception("Error moving leader to a park waypoint; continuing with remaining waypoints.")
        self._arm.disconnect()
        self._arm = None

    def is_connected(self) -> bool:
        return self._arm is not None and self._arm.is_connected()

    def set_freedrive_mode(self) -> None:
        """Switch the leader to freedrive so a human can backdrive it."""
        if self._arm is None or self._current_mode == "freedrive":
            return
        self._arm.set_mode_freedrive()
        self._current_mode = "freedrive"

    def set_position_mode(self) -> None:
        """Switch the leader to position mode so it can track automated commands."""
        if self._arm is None or self._current_mode == "position":
            return
        self._arm.set_mode_position()
        self._current_mode = "position"

    def track_joints(self, joint_positions: list[float], goal_time: float, blocking: bool = False) -> None:
        """Send a joint command to the leader without forcing mode switches.
        
        Assumes the caller has already put the arm in position mode via 
        `set_position_mode()`. Intended for high-frequency mirroring during
        autonomous rollouts.
        """
        if self._arm is None:
            return
        self._arm.set_joint_positions(list(joint_positions), goal_time=goal_time, blocking=blocking)

    def get_cartesian_positions(self) -> np.ndarray:
        return self._arm.get_cartesian_positions()

    def get_gripper_position(self) -> float:
        return self._arm.get_gripper_position()

    def get_joint_positions(self) -> np.ndarray:
        """7D = 6 arm joints (rad) + gripper carriage (m) - `command_space: joint`'s
        raw-mirroring counterpart to `get_cartesian_positions()`/`get_action()`
        above. Unlike `get_action()`, this is NOT a delta - matches Trossen's
        own `widowxai_leader.py::get_action()`, which reads and forwards the
        leader's raw absolute joint positions directly every tick."""
        return self._arm.get_joint_positions()

    def get_action(self) -> np.ndarray:
        """7D = delta pose (6, since the last call) + absolute gripper (1).

        Mirrors `widowxai_leader.py::get_action()`: gripper is sent as an
        absolute position, not a delta.
        """
        pose = self._arm.get_cartesian_positions()
        gripper = self._arm.get_gripper_position()
        delta = pose - self._prev_pose
        self._prev_pose = pose
        return np.concatenate([delta, [gripper]]).astype(np.float32)

    def rebaseline(self) -> None:
        """Re-sync the delta-action reference to the current pose (call this
        after a reset, so teleop doesn't jump on the next tick)."""
        self._prev_pose = self._arm.get_cartesian_positions()

    def sync_to_joints(
        self, joint_positions: list[float], goal_time: float = 2.0, end_in_freedrive: bool = True
    ) -> None:
        """Briefly leave freedrive to move the leader to match a target JOINT
        configuration (not a cartesian pose).

        Used by the teleop app right after a follower reset, so the human
        operator starts each episode with the leader physically aligned to
        wherever the follower actually ended up (rather than left wherever
        freedrive last let it drift/rest) - mirrors what the old
        `sync_leader_on_reset` follower-engine feature did, relocated here
        since only this class (not the follower engine) has a handle on the
        leader in the split architecture.

        Takes the follower's own joint positions (not its cartesian pose) as
        the target - since both arms are the same model/mounting, matching
        joint angles directly gives a matching EE pose, and is deterministic
        regardless of how different the leader's current configuration is
        from the target (a cartesian/IK-based move doesn't have that
        guarantee, and was observed on real hardware to sometimes end up
        near a degenerate "home"-looking configuration instead of the
        intended pose when the jump was large).

        `end_in_freedrive` (default `True`, matches every caller before this
        param existed - teleop): whether to switch BACK to freedrive once
        the move completes, so a human can immediately grab/backdrive the
        leader (teleop's use case - the operator starts each episode able
        to move it right away). Pass `False` for callers that want the
        leader to STAY in position mode after syncing (e.g. autonomous
        inference mirroring a policy, where there's no reason to ever visit
        freedrive at all unless a human explicitly takes over) - motors
        actively holding position mode is also generally preferable when
        nobody needs to backdrive the arm, since freedrive was observed to
        let joints slowly drift/sag under gravity when left unattended
        (fine during teleop, where the operator is right there to catch it;
        not fine for an autonomous inference session with nobody holding
        it).

        Does NOT call `rebaseline()` itself - the caller should do that
        right after, so the next `get_action()` delta is computed from this
        synced pose.
        """
        self.set_position_mode()
        self._arm.set_joint_positions(list(joint_positions), goal_time=goal_time, blocking=True)
        if end_in_freedrive:
            self.set_freedrive_mode()

