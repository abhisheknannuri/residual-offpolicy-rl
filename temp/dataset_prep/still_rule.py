"""Which frames of an episode are "still", and which of those to drop.

One rule, used by still_explorer.py (the HTML viewer mirrors it in JS, see
STILL_RULE_JS there) and by the thinning script, so they cannot disagree.

    speed[t, j]  = |q[t+1, j] - q[t, j]|            measured joints (observation.state), rad/frame
                                                     (the last frame reuses the one before it)
    still[t]     = speed[t, j] < joint_thr[j] for all 6 joints
                   and |g[t+1] - g[t]| < grip_thr     measured gripper, m/frame
    run          = consecutive still frames, only runs >= min_run count
    in a run     keep the first keep_head and the last keep_tail frames, drop the rest
    protected    frames within +-protect of a gripper label switch are never dropped

Measured joints, not the action: the delta action of a holding arm is mostly the
gravity/tracking offset (~0.013 rad shoulder), so it would look like motion.
Per joint, not the norm: a slow single-joint alignment move must not hide
behind five still joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StillRule:
    joint_thr: list = field(default_factory=lambda: [0.002] * 6)  # rad/frame (x20 = rad/s)
    grip_thr: float = 0.0005  # m/frame
    min_run: int = 20  # frames (20 Hz): 1 s
    keep_head: int = 10
    keep_tail: int = 5
    protect: int = 10


def speeds(q):
    """q: (T, 7) measured joints + gripper -> (T, 7) |forward difference| per frame."""
    d = np.abs(np.diff(q, axis=0))
    return np.vstack([d, d[-1:]]) if len(d) else np.zeros_like(q)


def classify(sp, switches, rule: StillRule):
    """sp: (T, 7) speeds, switches: frame indices of gripper label switches.

    Returns per-frame codes: 0 moving, 1 still (not in a long run), 2 kept (in a
    run: head/tail), 3 dropped, 4 protected (in a run, near a switch)."""
    T = len(sp)
    still = np.all(sp[:, :6] < np.asarray(rule.joint_thr), axis=1) & (sp[:, 6] < rule.grip_thr)
    prot = np.zeros(T, bool)
    for s in switches:
        prot[max(0, s - rule.protect): s + rule.protect + 1] = True
    code = np.where(still, 1, 0)
    t = 0
    while t < T:
        if not still[t]:
            t += 1
            continue
        e = t
        while e + 1 < T and still[e + 1]:
            e += 1
        if e - t + 1 >= rule.min_run:
            for k in range(t, e + 1):
                if k - t < rule.keep_head or e - k < rule.keep_tail:
                    code[k] = 2
                else:
                    code[k] = 4 if prot[k] else 3
        t = e + 1
    return code
