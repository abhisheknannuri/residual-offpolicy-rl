# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Pedal-driven episode-boundary control, shared by BOTH the teleop app
(`trossen_real/teleop/control_loop.py`) and the inference app
(`trossen_real/infer_app/infer_loop.py`) so this behavior can't drift apart
between the two. See `PEDAL_BEHAVIOR.md` (this directory) for the full
per-app/per-pedal behavior reference.

Deliberately separate from `InterventionManager` (`trossen_real/inference/
intervention.py`): intervention needs a leader arm and only ever applies to
the infer app; episode-boundary control needs neither a leader nor a policy
and applies to both apps - a `PedalListener` used here does NOT require
`enable_intervention`/a leader connection at all.

Behavior (confirmed design, see `REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md`
for the DIFFERENT semantics the real-hardware RL training env uses instead -
this module is NOT used there and does not change it):
  - Reset pedal: a single press ends the current episode immediately - reuses
    `PedalListener.consume_reset()`'s existing edge-triggered "fires once per
    press, however brief" semantics as-is.
  - Reward pedal: `next.reward` is 1.0 for every frame while held (via
    `PedalListener.consume_reward_latched()`, which also catches a
    press-then-release that happens entirely within one control tick) - AND,
    on the pedal's held->released edge, if it was held at any point during
    the current episode, that ALSO ends the current episode (mirrors the
    "Stop Recording"/"Stop Inference" button - the operator releasing the
    reward pedal after marking success is treated as "this episode is done").
  - Either trigger only ENDS the current episode (equivalent to a manual
    Stop Recording/Stop Inference click) - it never auto-starts a new one;
    the operator clicks Start Recording/Start Inference again for the next
    episode.
"""

from __future__ import annotations

import logging

from trossen_real.human_intervention.pedal_listener import PedalListener

logger = logging.getLogger(__name__)


class EpisodeBoundaryMonitor:
    """Wraps one `PedalListener` and turns its raw reset/reward signals into
    episode-control events. Not itself a control loop - callers call
    `start_episode()` once when a new episode's recording begins,
    `get_frame_reward()` once per recorded frame, and `check_episode_end()`
    once per tick (order relative to `get_frame_reward()` doesn't matter -
    both read/consume independent pedal state)."""

    def __init__(self, pedal: PedalListener) -> None:
        self.pedal = pedal
        self._reward_seen_this_episode = False
        self._prev_reward_pressed = False

    def start_episode(self) -> None:
        """Call once at the start of each recorded episode.

        Drains any STALE pedal state left over from before this episode's
        recording began, before doing anything else:
          - `consume_reset()` is edge-triggered but STICKY - a press sets
            an internal flag that stays `True` until something actually
            calls `consume_reset()`, however long that takes. Between two
            episodes (after one ends, before the next one's `start_episode()`
            call), `check_episode_end()` isn't being called by anyone -
            a reset-pedal tap during that idle window (deliberate or
            accidental) would otherwise sit latched and get misattributed
            to the BRAND NEW episode's very first tick, ending it after
            just 1 step. Discarding it here (and logging a warning if one
            was actually pending) makes only presses that happen AFTER
            this call count for the episode about to start.
          - `consume_reward_latched()` is discarded the same way (mirrors
            `real_residual_env.py::reset()`'s identical discard) - does NOT
            by itself protect against the pedal being held CONTINUOUSLY
            through the boundary, hence the separate `is_pressed` check
            below to at least warn about that case.
        """
        if self.pedal.consume_reset():
            logger.warning(
                "Reset pedal had a stale, unconsumed press from before this episode started recording - "
                "discarding it so it doesn't end this brand new episode after its very first tick."
            )
        if self.pedal.consume_reward_latched():
            logger.warning(
                "Reward pedal had a stale, unconsumed press from before this episode started recording - "
                "discarding it (does not protect against the pedal being held continuously through the "
                "boundary - release it before this episode's first real step if so)."
            )
        self._reward_seen_this_episode = False
        self._prev_reward_pressed = self.pedal.get_reward() == 1

    def get_frame_reward(self) -> float:
        """Call once per recorded frame - returns 1.0/0.0 for this frame's
        `next.reward`, and remembers whether the reward pedal was ever
        pressed during the current episode (see `check_episode_end()`)."""
        latched = self.pedal.consume_reward_latched()
        if latched:
            self._reward_seen_this_episode = True
        return 1.0 if latched else 0.0

    def check_episode_end(self) -> tuple[bool, str | None]:
        """Call once per tick - returns `(True, reason)` if the current
        episode should end right now, else `(False, None)`. `reason` is
        `"reset_pedal"` or `"reward_release"` (for UI/log display)."""
        if self.pedal.consume_reset():
            return True, "reset_pedal"

        current_pressed = self.pedal.get_reward() == 1
        released_edge = self._prev_reward_pressed and not current_pressed
        self._prev_reward_pressed = current_pressed
        if released_edge and self._reward_seen_this_episode:
            return True, "reward_release"
        return False, None
