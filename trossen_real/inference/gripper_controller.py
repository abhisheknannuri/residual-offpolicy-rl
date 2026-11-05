"""Open/closed gripper: two thresholds and a ramp.

`GripperController` runs on the robot (infer app, RL env); `binarize` makes the
dataset labels with the same thresholds, so the two cannot differ. Tested and
used by temp/gripper/ (sim_gripper.py, real_gripper_test.py,
binarize_gripper_dataset.py) and temp/dataset_prep/. Numpy-free, no robot imports.

The policy (ACT + residual) gives an absolute width every tick. We read it as a
vote:
    width > open_above   -> open
    width < close_below  -> closed
    in between           -> keep the current state
The state changes only after the same vote `confirm_ticks` ticks in a row, so
one-tick spikes and NaN do nothing. While the human intervenes, the target is
the leader's width instead. Either way the command moves toward the target by at
most `max_step` per tick: no snaps, also not when the human lets go.

Thresholds 10 / 16 mm: in the ACT Iter1 data every width held for 15+ frames is
either <= 10 mm (closed) or >= 20 mm (open; the recovery episodes release at only
20-29 mm), so the band sits in that gap.

`limits` is the hardware range the command is clipped to (default [closed,
open]); give the station's gripper_closed/gripper_open. The human's width is
only clipped to `limits`, never to `open`, so a fully opened leader opens the
follower fully too: it follows the leader's absolute width, it only lags it by
max_step per tick, it never gets stuck short of it.
"""

from __future__ import annotations

import math


def binarize(widths, closed=0.004, open=0.040, close_below=0.010, open_above=0.016, min_frames=3):
    """Dataset labels: each width -> closed or open, with the controller's thresholds.

    Offline, so no confirm_ticks and no ramp: the label switches on the frame the
    width leaves the band. The first frame starts from whichever side of the
    band's middle it is on. A label that lasts fewer than `min_frames` frames (an
    operator twitch) keeps the label before it: the controller would ignore it
    anyway, and as a label it only teaches ACT noise.
    """
    out, is_open = [], None
    for w in widths:
        if is_open is None:
            is_open = w > (close_below + open_above) / 2
        elif w > open_above:
            is_open = True
        elif w < close_below:
            is_open = False
        out.append(open if is_open else closed)
    k = 1
    while k < len(out):
        if out[k] != out[k - 1]:
            n = 1
            while k + n < len(out) and out[k + n] == out[k]:
                n += 1
            if n < min_frames:
                out[k:k + n] = [out[k - 1]] * n
            k += n
        else:
            k += 1
    return out


def looks_binarized(action_stats, dim=6, ratio=0.98):
    """Was this checkpoint trained on binarized gripper labels? From its saved
    action stats (min/max/mean/std, as the policy server reports them).

    On [min, max] a std can be at most sqrt((max - mean) * (mean - min)), and it
    is that only when every value is min or max (Bhatia-Davis). ACT Iter1: the
    binarized gripper 1.00 of that bound, the recorded one 0.91. Returns None
    when the stats are missing.
    """
    try:
        lo, hi = float(action_stats["min"][dim]), float(action_stats["max"][dim])
        mean, std = float(action_stats["mean"][dim]), float(action_stats["std"][dim])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    bound = math.sqrt(max((hi - mean) * (mean - lo), 0.0))
    return bool(bound > 0 and std / bound >= ratio)


class GripperController:
    def __init__(self, closed=0.004, open=0.040, close_below=0.010, open_above=0.016,
                 confirm_ticks=3, max_step=0.003, limits=None):
        self.closed, self.open = closed, open
        self.close_below, self.open_above = close_below, open_above
        self.confirm_ticks, self.max_step = confirm_ticks, max_step
        self.lo, self.hi = limits if limits is not None else (closed, open)
        self.reset(closed)

    def reset(self, command):
        """New episode: start from the width the gripper is commanded to now."""
        self.command = min(max(command, self.lo), self.hi)
        self.is_open = self.command > (self.close_below + self.open_above) / 2
        self.count = 0

    def step(self, policy_width, human_width=None):
        """One tick. Returns the width to send to the follower."""
        if human_width is not None:  # intervention: follow the leader
            target = human_width
            self.is_open = human_width > (self.close_below + self.open_above) / 2
            self.count = 0
        else:
            w = policy_width if math.isfinite(policy_width) else None
            want = (True if w is not None and w > self.open_above
                    else False if w is not None and w < self.close_below
                    else self.is_open)
            self.count = self.count + 1 if want != self.is_open else 0
            if self.count >= self.confirm_ticks:
                self.is_open, self.count = want, 0
            target = self.open if self.is_open else self.closed

        step = max(-self.max_step, min(self.max_step, target - self.command))
        self.command = min(max(self.command + step, self.lo), self.hi)
        return self.command


def gripper_from_config(control, requested_mode=None, action_stats=None):
    """The session's gripper from the station config's `control` (ControlConfig).

    -> (GripperController or None for "raw", info dict for the UI/logs).
    The mode is control.gripper.mode unless `requested_mode` (e.g. the infer
    app's Connect dropdown) overrides it. Raises ValueError for an unknown mode
    and for "raw" with a checkpoint trained on binarized gripper labels (its
    output jumps between the two widths: a full stroke in one tick when sent
    raw). "binary" with a checkpoint trained on recorded widths only warns: it
    works, but changes that checkpoint's behaviour.
    """
    from trossen_real.config import GRIPPER_MODES

    g = control.gripper
    mode = (requested_mode or "").strip() or g.mode
    if mode not in GRIPPER_MODES:
        raise ValueError(f"gripper mode must be one of {GRIPPER_MODES}, got '{mode}'")
    binarized = looks_binarized(action_stats) if action_stats else None
    warning = None
    if binarized is None:
        warning = ("the policy server reports no action_stats (policy_server.py older than this check?), "
                   "so the gripper mode cannot be checked against the checkpoint")
    elif mode == "raw" and binarized:
        raise ValueError("this checkpoint was trained on binarized gripper labels: in raw mode its gripper would "
                         "jump between the two widths in one tick. Use gripper mode 'binary'.")
    elif mode == "binary" and not binarized:
        warning = ("this checkpoint was trained on recorded gripper widths, not binarized labels: binary mode "
                   "works but changes its behaviour (partial widths become open/closed)")
    info = dict(mode=mode, source="connect" if (requested_mode or "").strip() else "station config",
                checkpoint_binarized=binarized, warning=warning)
    if mode == "raw":
        return None, info
    params = dict(closed=g.closed, open=g.open, close_below=g.close_below, open_above=g.open_above,
                  confirm_ticks=g.confirm_ticks, max_step=g.max_step)
    return GripperController(**params, limits=(control.gripper_closed, control.gripper_open)), {**info, **params}
