"""Speed limit and flip filter for a continuous gripper width.

Every tick the policy (ACT + residual) or the human (leader) asks for an
absolute gripper width; this decides what is actually sent to the follower.
Two rules:

1. Flips: the command only moves toward the policy's request once the last
   `confirm_ticks` requests ALL lie on the same side of the current command
   (all wider, or all narrower). It then moves toward the closest of them, the
   width all of them agree on. A request that flips open/closed from tick to
   tick never passes, whatever its size, and the rule is symmetric, so flipping
   does not ratchet the gripper in either direction. A steady change starts
   `confirm_ticks - 1` ticks late and then follows with that lag.
2. Speed: the command changes by at most `max_step` per tick (m), whoever asks.
   Clipping the target to the gripper range does not stop the motor from
   overshooting it: a full stroke in one 0.125 s move overshot 40 mm to 44.015
   mm and faulted the arm (RL run 2026-11-05 14:31).

While the human intervenes only the speed limit applies (the leader does not
flip), and the policy's recent requests are forgotten: after the hand-back the
policy must agree `confirm_ticks` ticks in a row before the gripper moves away
from where the human left it, and the hand-back itself moves at most
`max_step` per tick.

Defaults: max_step 6 mm, just above the fastest measured gripper motion in the
ACT Iter1 demos (5.9 mm/tick) and in the thin ACT's live runs (5.2 mm/tick), so
nothing demonstrated is slowed; confirm_ticks 3, as GripperController's.
`confirm_ticks=1` turns rule 1 off (speed limit only).

Numpy-free, no robot imports. Used by the RL env (trossen_real/rl/
real_residual_env.py); tests: trossen_real/inference/tests/test_gripper_filter.py.
"""

from __future__ import annotations

import collections
import math


class GripperFilter:
    def __init__(self, max_step=0.006, confirm_ticks=3, limits=(0.001, 0.040)):
        if max_step <= 0 or confirm_ticks < 1:
            raise ValueError(f"need max_step > 0 and confirm_ticks >= 1, got {max_step}, {confirm_ticks}")
        self.max_step, self.confirm_ticks = max_step, confirm_ticks
        self.lo, self.hi = limits
        self.reset(self.hi)

    def _clip(self, w):
        return min(max(w, self.lo), self.hi)

    def reset(self, command):
        """New episode: start from the width the gripper is commanded to now."""
        self.command = self._clip(command)
        self.recent = collections.deque(maxlen=self.confirm_ticks)

    def step(self, width, human=False):
        """One tick: the requested width -> the width to send."""
        if human:
            target = self._clip(width) if math.isfinite(width) else self.command
            self.recent.clear()
        elif not math.isfinite(width):
            target = self.command
            self.recent.clear()
        else:
            self.recent.append(self._clip(width))
            target = self.command
            if len(self.recent) == self.confirm_ticks:
                if all(r > self.command for r in self.recent):
                    target = min(self.recent)
                elif all(r < self.command for r in self.recent):
                    target = max(self.recent)
        step = max(-self.max_step, min(self.max_step, target - self.command))
        self.command = self._clip(self.command + step)
        return self.command
