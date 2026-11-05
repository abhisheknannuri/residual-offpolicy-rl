"""GripperFilter (trossen_real/inference/gripper_filter.py) and its use in the RL env.

Why: in RL run 2026-11-05 the residual's gripper output flipped between its
limits from tick to tick (gripper target 1 <-> 21 mm, up to 34 mm measured in
one tick), and one full-stroke open overshot 40 mm to 44.015 mm and faulted the
arm. The filter must (1) never change the command by more than max_step per
tick, from any source, (2) ignore flipping of any size without drifting, (3)
still pass every steady request, (4) hand back from the human smoothly, and the
env must store exactly what it sent.

Widths in mm in the scenario tables, metres in the code. Station range 1-40 mm.

Run:  .venv/bin/python -m pytest trossen_real/inference/tests/test_gripper_filter.py -v
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from trossen_real.config import GripperFilterConfig
from trossen_real.inference.gripper_filter import GripperFilter
from trossen_real.rl.real_residual_env import TrossenResidualEnv

MM = 0.001
LIMITS = (0.001, 0.040)
STEP = 0.006
EPS = 1e-9


def make(command_mm, confirm=3, open_gap=None):
    f = GripperFilter(max_step=STEP, confirm_ticks=confirm, limits=LIMITS,
                      open_gap=None if open_gap is None else open_gap * MM)
    f.reset(command_mm * MM)
    return f


def run(f, requests_mm, human=None, measured_mm=None):
    """Feed requests (mm), optionally with the measured finger width (mm);
    returns the sent commands (mm)."""
    human = human or [False] * len(requests_mm)
    measured_mm = measured_mm or [None] * len(requests_mm)
    return [round(f.step(r * MM, human=h, measured=None if m is None else m * MM) / MM, 6)
            for r, h, m in zip(requests_mm, human, measured_mm)]


def max_jump(cmds_mm, start_mm):
    seq = [start_mm] + list(cmds_mm)
    return max(abs(b - a) for a, b in zip(seq, seq[1:]))


# ----------------------------------------------------------------------------
# Speed limit
# ----------------------------------------------------------------------------

def test_full_open_stroke_is_spread_over_ticks():
    """ACT releases: 4 -> 40 mm in one tick (today's fault). Speed limit only."""
    f = make(4, confirm=1)
    out = run(f, [40] * 10)
    assert out[:6] == pytest.approx([10, 16, 22, 28, 34, 40])
    assert out[6:] == pytest.approx([40] * 4)
    assert max(out) <= 40 + EPS


def test_full_close_stroke_is_spread_over_ticks():
    f = make(40, confirm=1)
    out = run(f, [1] * 10)
    assert out[:7] == pytest.approx([34, 28, 22, 16, 10, 4, 1])


def test_small_changes_pass_unchanged():
    """Below max_step the speed limit does nothing (demos: <= 5.9 mm/tick)."""
    f = make(20, confirm=1)
    assert run(f, [25.9, 21, 21.5]) == pytest.approx([25.9, 21, 21.5])


def test_requests_outside_the_range_are_clipped():
    f = make(38, confirm=1)
    assert run(f, [60, 60]) == pytest.approx([40, 40])
    f = make(3, confirm=1)
    assert run(f, [-5, -5]) == pytest.approx([1, 1])
    f = make(0)  # reset clips too
    assert f.command == pytest.approx(1 * MM)


# ----------------------------------------------------------------------------
# Flips
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("center, amp", [
    (4, 8.6),    # holding, ACT 4 mm, residual at scale 0.15 (+-8.6 mm)
    (4, 17.0),   # holding, residual at scale 0.3 (+-17 mm): today's 1 <-> 21 mm
    (22, 18.0),  # mid-range, full 4 <-> 40 mm flipping
    (35, 17.0),  # open, ACT 40 mm, residual clipped at 40
])
@pytest.mark.parametrize("first", [+1, -1])
def test_flipping_every_tick_never_moves_the_gripper(center, amp, first):
    start = min(max(center, 1), 40)
    f = make(start)
    req = [center + first * amp * (1 if i % 2 == 0 else -1) for i in range(200)]
    out = run(f, req)
    assert out == pytest.approx([start] * 200)


@pytest.mark.parametrize("pattern", [[+1, +1, -1, -1], [+1, -1, -1], [+1, +1, -1]])
def test_flipping_shorter_than_confirm_ticks_never_moves(pattern):
    """Any pattern without 3 same-side ticks in a row is held."""
    f = make(20)
    req = [20 + 10 * pattern[i % len(pattern)] for i in range(120)]
    assert run(f, req) == pytest.approx([20] * 120)


def test_flipping_after_a_motion_does_not_ratchet():
    """The gripper just opened (last motion: opening). Flipping must not keep
    opening it: the rule compares against the command, not the last direction."""
    f = make(4)
    run(f, [20] * 8)            # opens to 20 mm
    assert f.command == pytest.approx(20 * MM)
    out = run(f, [20 + 15 * (1 if i % 2 else -1) for i in range(100)])
    assert out == pytest.approx([20] * 100)


def test_symmetric_noise_does_not_drift():
    """Exploration noise (std ~1.2 mm, scale 0.3 x stddev 0.02) around a steady
    request: the command stays near it, no one-sided drift."""
    rng = random.Random(0)
    f = make(22)
    out = run(f, [22 + rng.gauss(0, 1.2) for _ in range(5000)])
    assert abs(np.mean(out) - 22) < 0.3
    assert max(abs(o - 22) for o in out) < 4


# ----------------------------------------------------------------------------
# Steady requests still pass
# ----------------------------------------------------------------------------

def test_steady_request_moves_on_the_confirm_tick():
    f = make(4)
    out = run(f, [40] * 9)
    assert out[:2] == pytest.approx([4, 4])                   # 2 ticks: not yet confirmed
    assert out[2:] == pytest.approx([10, 16, 22, 28, 34, 40, 40])


def test_three_same_side_requests_move_to_the_closest():
    """All three wider than the command, different sizes: it goes to the one
    all three agree on (the smallest), not further."""
    f = make(10)
    out = run(f, [30, 13, 14])
    assert out == pytest.approx([10, 10, 13])


def test_smooth_ramp_is_followed_with_a_two_tick_lag():
    """ACT closing smoothly (2 mm/tick, like the demos): follows, 2 ticks late."""
    f = make(40)
    req = [40 - 2 * (i + 1) for i in range(15)]  # 38, 36, ..., 10
    out = run(f, req)
    assert out[:2] == pytest.approx([40, 40])
    assert out[2:] == pytest.approx(req[:-2])


def test_release_while_residual_flips():
    """Holding at 4 mm, ACT switches to 40 mm, residual keeps flipping +-17 mm:
    every request is >= 23 mm, so the release goes through at <= 6 mm/tick."""
    f = make(4)
    req = [min(40, 40 + (17 if i % 2 else -17)) for i in range(20)]
    out = run(f, req)
    assert max_jump(out, 4) <= 6 + 1e-6
    assert out[-1] >= 23 - 1e-6
    assert max(out) <= 40 + EPS


def test_holding_cube_with_flipping_residual_keeps_the_grip():
    """ACT holds at 4 mm, residual +-8.6 mm (scale 0.15): requests 12.6 / 1 mm."""
    f = make(4)
    out = run(f, [12.6 if i % 2 else max(1, 4 - 8.6) for i in range(100)])
    assert out == pytest.approx([4] * 100)


def test_nan_holds_and_restarts_the_confirmation():
    f = make(10)
    out = run(f, [20, 20, float("nan"), 20, 20, 20])
    assert out == pytest.approx([10, 10, 10, 10, 10, 16])


def test_confirm_ticks_1_is_speed_limit_only():
    f = make(20, confirm=1)
    out = run(f, [30, 10, 30, 10])
    assert out == pytest.approx([26, 20, 26, 20])


# ----------------------------------------------------------------------------
# Human intervention and the hand-back
# ----------------------------------------------------------------------------

def test_human_is_speed_limited_but_not_delayed():
    f = make(4)
    out = run(f, [30] * 6, human=[True] * 6)
    assert out == pytest.approx([10, 16, 22, 28, 30, 30])


def test_hand_back_holds_until_the_policy_agrees_then_ramps():
    """Human leaves the gripper at 30 mm, the policy wants 4 mm."""
    f = make(30)
    run(f, [30] * 3, human=[True] * 3)
    out = run(f, [4] * 8)
    assert out == pytest.approx([30, 30, 24, 18, 12, 6, 4, 4])


def test_hand_back_ignores_the_policy_history_from_before_the_intervention():
    f = make(20)
    run(f, [35, 35])                     # two policy votes for wider...
    run(f, [20], human=[True])           # ...human takes over...
    out = run(f, [35])                   # ...one more vote must not complete them
    assert out == pytest.approx([20])


# ----------------------------------------------------------------------------
# Opening cap: the command may not run more than open_gap ahead of the fingers
# ----------------------------------------------------------------------------

def test_opening_cap_holds_the_command_near_lagging_fingers():
    """Fingers stuck at 12 mm, request 40: the command stops at 12 + 9 = 21."""
    f = make(10, confirm=1, open_gap=9)
    assert run(f, [40] * 5, measured_mm=[12] * 5) == pytest.approx([16, 21, 21, 21, 21])


def test_opening_cap_follows_the_fingers_as_they_open():
    f = make(10, confirm=1, open_gap=9)
    assert run(f, [40] * 6, measured_mm=[10, 12, 16, 22, 28, 33]) == pytest.approx([16, 21, 25, 31, 37, 40])


def test_the_16_22_fault_sequence_is_capped():
    """2026-11-05 16:22: request 40 mm, fingers at 22.0 / 22.2 / 22.3 / 24.0 /
    26.6 mm; the commands sent were 22.4, 22.4, 28.4, 34.4, 40.0 (the last one
    ~12 mm ahead of the fingers). With the cap: never more than 9 mm ahead."""
    f = make(22.4, confirm=1, open_gap=9)
    meas = [21.95, 22.15, 22.25, 24.04, 26.60]
    out = run(f, [40] * 5, measured_mm=meas)
    assert all(o - m <= 9 + 1e-6 for o, m in zip(out, meas))
    assert out == pytest.approx([28.4, 31.15, 31.25, 33.04, 35.6])


def test_opening_cap_never_pulls_the_command_back():
    """Command already 15 mm ahead (cap just switched on): it holds, it does
    not drop toward the fingers."""
    f = make(30, confirm=1, open_gap=9)
    assert run(f, [40] * 3, measured_mm=[15] * 3) == pytest.approx([30, 30, 30])


def test_closing_is_never_capped():
    f = make(30, confirm=1, open_gap=9)
    assert run(f, [4] * 6, measured_mm=[30] * 6) == pytest.approx([24, 18, 12, 6, 4, 4])


def test_holding_the_cube_keeps_the_grip_and_can_still_open():
    """Command 4 mm, fingers blocked by the cube at 11.4 mm: the 7.4 mm grip
    gap stays; a steady request to open passes (it is behind the fingers)."""
    f = make(4, open_gap=9)
    assert run(f, [4] * 5, measured_mm=[11.4] * 5) == pytest.approx([4] * 5)
    assert run(f, [16] * 4, measured_mm=[11.4] * 4) == pytest.approx([4, 4, 10, 16])


def test_opening_cap_applies_to_the_human_too():
    f = make(10, open_gap=9)
    assert run(f, [40] * 3, human=[True] * 3, measured_mm=[10] * 3) == pytest.approx([16, 19, 19])


def test_no_measurement_means_no_cap():
    f = make(10, confirm=1, open_gap=9)
    assert run(f, [40] * 3, measured_mm=[None, float("nan"), None]) == pytest.approx([16, 22, 28])


# ----------------------------------------------------------------------------
# Invariants under random input
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_random_requests_respect_speed_range_and_opening_cap(seed):
    rng = random.Random(seed)
    gap = rng.choice([None, 3, 9])
    f = make(rng.uniform(1, 40), confirm=rng.choice([1, 2, 3, 5]), open_gap=gap)
    prev, fingers = f.command, f.command
    for _ in range(2000):
        kind = rng.random()
        w = (float("nan") if kind < 0.02 else rng.uniform(-20, 70) * MM if kind < 0.5
             else rng.choice([1, 4, 40, 60]) * MM)
        c = f.step(w, human=rng.random() < 0.1, measured=fingers)
        assert abs(c - prev) <= STEP + EPS
        assert LIMITS[0] - EPS <= c <= LIMITS[1] + EPS
        if gap is not None and c > prev + EPS:
            assert c <= max(prev, fingers + gap * MM) + EPS
        fingers += rng.uniform(0.2, 1.0) * (c - fingers)  # lagging fingers
        prev = c


def test_bad_parameters_are_refused():
    with pytest.raises(ValueError):
        GripperFilter(max_step=0, confirm_ticks=3)
    with pytest.raises(ValueError):
        GripperFilter(max_step=0.006, confirm_ticks=0)
    with pytest.raises(ValueError):
        GripperFilter(max_step=0.006, confirm_ticks=3, open_gap=0)


# ----------------------------------------------------------------------------
# In the RL env, on fake hardware
# ----------------------------------------------------------------------------

# Stage-2 RL dataset action range (delta_joint), action_scale 0.1 / 0.3 (gripper).
ACTION_MIN = [-0.0782, -0.1546, -0.1436, -0.1213, -0.071, -0.0973, 0.004]
ACTION_MAX = [0.0488, 0.0999, 0.1183, 0.0736, 0.0782, 0.1331, 0.04]
ACTION_SCALE = [0.1] * 6 + [0.3]
Q = [-0.27, 2.08, 1.50, -0.84, 0.006, -0.29, 0.0114]


class _Follower:
    def __init__(self, q, commanded_gripper=None, track=False):
        self.q = np.asarray(q, dtype=np.float32)
        self.commanded_gripper = commanded_gripper
        self.track = track
        self.sent = []

    def get_state(self):
        return {"q": self.q.tolist(), "pose": [0.0] * 6, "gripper_pos": float(self.q[6]),
                "dq": [0.0] * 7, "efforts": [0.0] * 7, "accelerations": [0.0] * 7}

    def move_to_joint_positions(self, arr, blocking=False, min_time_to_move=None):
        self.sent.append(np.asarray(arr, dtype=np.float32))
        if self.track:  # fingers reach the sent width by the next read (no cube)
            self.q[6] = self.sent[-1][6]

    def reset(self):
        pass

    def leader_sync_joints(self, q):  # like Stage2ResetFollower: commanded close on the cube
        q = [float(x) for x in q]
        if self.commanded_gripper is not None:
            q[6] = self.commanded_gripper
        return q


class _Cameras:
    def get_all_latest(self):
        return {"cam": np.zeros((8, 8, 3), dtype=np.uint8)}


class _Policy:
    """ACT: returns `gripper` (settable) as its gripper width, zero joint deltas."""

    def __init__(self, gripper):
        self.gripper = gripper

    def predict(self, obs):
        return np.asarray([0.0] * 6 + [self.gripper], dtype=np.float32)

    def reset(self):
        pass


class _Intervention:
    def __init__(self, script):
        self.script, self.i, self.mirrored = list(script), 0, []

    def tick(self):
        v = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return v

    def mirror_to_leader(self, action, goal_time):
        self.mirrored.append(float(action[6]))


class _Leader:
    def __init__(self, gripper):
        self.gripper = gripper

    def get_joint_positions(self):
        return np.asarray(Q[:6] + [self.gripper], dtype=np.float32)


def _env(act_gripper, filter_cfg=GripperFilterConfig(), commanded_gripper=0.004, track=True):
    scaler = ActionScaler(torch.tensor(ACTION_MIN), torch.tensor(ACTION_MAX), ACTION_SCALE, min_range_per_dim=0.1)
    control = SimpleNamespace(gripper_closed=LIMITS[0], gripper_open=LIMITS[1], goal_time_s=0.05,
                              frequency_hz=100000, gripper_filter=filter_cfg)
    env = TrossenResidualEnv(
        follower=_Follower(Q, commanded_gripper, track), cameras=_Cameras(), policy=_Policy(act_gripper), control=control,
        reset_cfg=None, action_scaler=scaler,
        state_standardizer=StateStandardizer(torch.zeros(7), torch.ones(7)),
        image_keys=["observation.images.cam"], camera_resolution=(8, 8),
        action_space="delta_joint", policy_image_encoding="raw", settle_time_s=0.0,
    )
    env.reset()
    return env, scaler


def _residual(grip_n):
    return torch.tensor([[0.0] * 6 + [grip_n]])


def test_env_reset_starts_the_filter_at_the_commanded_width():
    """Stage-2 reset: the gripper is commanded to 4 mm on the cube (measured
    11.4 mm). The filter must start from the command, not the measurement."""
    env, _ = _env(0.004)
    assert env.gripper_filter.command == pytest.approx(0.004)


def test_env_flipping_residual_does_not_move_the_gripper_and_stores_what_was_sent():
    env, scaler = _env(0.004)
    for i in range(60):
        _, _, _, _, info = env.step(_residual(0.3 if i % 2 else -0.3))
        sent = env.follower.sent[-1]
        stored_real = scaler.unscale(info["scaled_action"]).squeeze(0).numpy()
        assert sent[6] == pytest.approx(0.004, abs=1e-6)
        assert stored_real[6] == pytest.approx(sent[6], abs=1e-6)


def test_env_act_release_is_speed_limited():
    """ACT jumps 4 -> 40 mm with the residual at +0.3: sent width <= 6 mm/tick."""
    env, scaler = _env(0.004)
    env.step(_residual(0.0))
    env.policy.gripper = 0.040
    widths = [0.004]
    for _ in range(12):
        _, _, _, _, info = env.step(_residual(0.3))
        widths.append(float(env.follower.sent[-1][6]))
        assert scaler.unscale(info["scaled_action"]).squeeze(0).numpy()[6] == pytest.approx(widths[-1], abs=1e-6)
    assert max(abs(b - a) for a, b in zip(widths, widths[1:])) <= STEP + 1e-6
    assert widths[-1] == pytest.approx(0.040, abs=1e-6)
    assert max(widths) <= 0.040 + 1e-6


def test_env_hand_back_from_the_human_is_smooth_and_mirrored():
    """Human holds the gripper at 30 mm, lets go; ACT wants 4 mm. The leader
    mirror gets the filtered width, never a jump."""
    env, _ = _env(0.004)
    env.leader = _Leader(0.030)
    env.intervention = _Intervention([True] * 6 + [False] * 10)
    sent = []
    for _ in range(16):
        env.step(_residual(0.0))
        sent.append(round(float(env.follower.sent[-1][6]) / MM, 4))
    assert sent[:6] == pytest.approx([10, 16, 22, 28, 30, 30])   # human, speed limited
    assert sent[6:] == pytest.approx([30, 30, 24, 18, 12, 6, 4, 4, 4, 4])
    assert [round(m / MM, 4) for m in env.intervention.mirrored] == pytest.approx(sent[6:])


def test_env_opening_cap_uses_the_measured_fingers():
    """Fingers stuck at the cube width (11.4 mm, track=False): ACT opens to 40 mm,
    the command stops 9 mm ahead of the fingers."""
    env, _ = _env(0.004, track=False)
    env.step(_residual(0.0))
    env.policy.gripper = 0.040
    for _ in range(10):
        env.step(_residual(0.0))
    assert env.follower.sent[-1][6] == pytest.approx(0.0114 + 0.009, abs=1e-6)


def test_env_without_gripper_filter_config_sends_the_request_unchanged():
    """Control objects without `gripper_filter` (older fakes) keep the old behaviour."""
    env, _ = _env(0.004, filter_cfg=None)
    assert env.gripper_filter is None
    env.policy.gripper = 0.040
    env.step(_residual(0.0))  # still the base action queried before the change (4 mm)
    env.step(_residual(0.0))
    assert env.follower.sent[-1][6] == pytest.approx(0.040, abs=1e-6)  # 4 -> 40 mm in one tick


def test_station_config_has_the_filter():
    from trossen_real.config import load_station_config

    f = load_station_config("trossen_station3_single").control.gripper_filter
    assert (f.max_step, f.confirm_ticks, f.open_gap) == (0.006, 3, 0.009)
