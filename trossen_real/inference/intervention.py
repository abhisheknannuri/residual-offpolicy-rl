# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Shared human-intervention coordination helper - used by BOTH the
inference web UI's control loop (`infer_app/infer_loop.py`) and (later)
the real-hardware residual-RL gym wrapper (`trossen_real/rl/
real_residual_env.py`), so the pedal-check / leader-mode-switch /
policy-queue-reset logic lives in exactly one place instead of being
duplicated between the two.

Design (see `trossen_real/human_intervention/INTERVENTION_PLAN.md` for the
full rationale):

  - The ACT policy is called EVERY tick regardless of intervention state -
    never skipped, never specially paused. While intervened, its output is
    simply not applied to the follower (the human's leader-arm readback is
    sent instead). This is deliberately different from (and simpler than)
    an "explicitly flush the queue on takeover" design.
  - The one place staleness CAN still matter: a checkpoint using
    `n_action_steps > 1` (not the case for today's delta_joint checkpoints,
    which require `n_action_steps=1` for a different reason - see
    `policy_client.py::reconstruct_absolute_action()` - but could apply to
    some future absolute-action checkpoint). In that case the policy
    server's internal action-chunk queue would keep serving cached actions
    computed from BEFORE/DURING the intervention once control resumes,
    causing a jerk back toward a stale plan. The fix: call `policy.reset()`
    exactly once, on the intervention RELEASE edge (True -> False),
    BEFORE the next `predict_full()` call - this forces a genuine fresh
    forward pass reflecting the human-corrected position, regardless of
    `n_action_steps`. No reset is needed on the takeover edge, since the
    policy's output is discarded throughout intervention anyway.
"""

from __future__ import annotations

import logging

import numpy as np

from trossen_real.human_intervention.pedal_listener import PedalListener
from trossen_real.inference.policy_client import PolicyClient
from trossen_real.leader.trossen_leader_single import TrossenSingleLeader

logger = logging.getLogger(__name__)


class InterventionManager:
    """Coordinates one leader arm + one foot pedal + one policy-server
    connection for human-intervention takeover during an autonomous
    rollout. Not itself a control loop - callers still own the actual
    tick loop and call `tick()` once per control tick, at the START of the
    tick (before reading observations / calling the policy), then use
    `get_leader_action()`/`mirror_to_leader()` as appropriate afterward.
    """

    def __init__(self, leader: TrossenSingleLeader, pedal: PedalListener, policy: PolicyClient) -> None:
        self.leader = leader
        self.pedal = pedal
        self.policy = policy
        self._prev_intervened = False

    def tick(self) -> bool:
        """Call once per control tick, BEFORE building the observation /
        calling `policy.predict_full()` for this tick. Returns whether
        intervention is currently active.

        Side effects (every tick, cheap/no-op if already in that state):
          - Switches the leader to freedrive (intervened) or position mode
            (not intervened) - `TrossenSingleLeader`'s own `_current_mode`
            tracking makes repeated calls in the same mode a no-op, so
            calling this unconditionally every tick is safe and avoids
            duplicating edge-detection here.
          - On the intervention RELEASE edge (True -> False) ONLY: calls
            `policy.reset()` to force a fresh forward pass on the very
            next `predict_full()` call (see module docstring for why).
        """
        intervened_now = self.pedal.is_intervention()

        if self._prev_intervened and not intervened_now:
            logger.info("Intervention released - resetting policy's action-chunk queue for a fresh forward pass.")
            self.policy.reset()

        if intervened_now:
            self.leader.set_freedrive_mode()
        else:
            self.leader.set_position_mode()

        self._prev_intervened = intervened_now
        return intervened_now

    def sync_to_follower(self, follower_q, goal_time: float) -> None:
        """Align the leader to the follower's CURRENT joint positions -
        call this once, right after `follower.reset()` completes and
        BEFORE the tick loop starts. Without this, the leader is left
        wherever it happened to be resting (e.g. from a previous session's
        park pose, or wherever freedrive last drifted to) while the
        follower jumps to its fresh reset pose - the very first autonomous
        tick would then try to mirror the policy's action (a small
        delta/step relative to the follower's NEW position) onto a leader
        that's potentially far away, commanding an unintended large, fast
        jump that can fault the arm (joint velocity/limit errors). Mirrors
        `teleop/app.py::api_reset()`'s identical `sync_to_joints()` +
        `rebaseline()` sequence, but with `end_in_freedrive=False`.

        Blocking (matches `sync_to_joints()`'s own blocking big-jump move)
        - takes `goal_time` seconds.

        `leader.sync_to_joints()` is shared with the TELEOP app, which
        wants it to end in freedrive (`end_in_freedrive=True`, its default)
        so the human can immediately grab/backdrive the leader. That's the
        WRONG end state for inference - there's no reason to ever visit
        freedrive here unless a human is actually taking over, and
        freedrive was observed to let the arm slowly drift/sag under
        gravity when nobody's hand is on it to catch it (fine during
        teleop, where the operator is right there; not fine for an
        unattended autonomous rollout) - so this passes
        `end_in_freedrive=False`, staying in position mode directly with no
        freedrive dip at all. Still calls `tick()` right after (cheap,
        reads the CURRENT pedal state) purely to correctly handle the edge
        case where the operator already had the pedal held down at reset
        time - that should still leave the leader in freedrive, matching
        reality.
        """
        self.leader.sync_to_joints(follower_q, goal_time=goal_time, end_in_freedrive=False)
        self.leader.rebaseline()
        self.tick()

    def get_leader_action(self) -> np.ndarray:
        """7D absolute (6 joints + gripper) - the leader's raw current
        joint positions, to be sent DIRECTLY to the follower while
        intervened (no delta math - `command_space: joint` is already
        absolute on both sides)."""
        return self.leader.get_joint_positions()

    def mirror_to_leader(self, action: np.ndarray, goal_time: float) -> None:
        """Drive the leader to track `action` (the same target being sent
        to the follower) - only call this while NOT intervened, so a
        human resting their hand on the leader "rides along" with the
        autonomous policy and can feel/preempt what it's about to do.
        Non-blocking (matches the follower's own non-blocking tracking)."""
        self.leader.track_joints(list(action), goal_time=goal_time, blocking=False)
