"""The replay buffer must store the action the robot EXECUTED.

Regression for the gripper mismatch: on autonomous ticks the buffer stored
`clamp(base + residual, -1, 1)`, but the robot clips the gripper to its
physical range [gripper_closed, gripper_open] (normalized ~+-0.327 for the
Station-3 stats). 39% of autonomous rows in run_2026-10-09_11-00-05 held a
gripper value the robot never executed.

Runs a real `TrossenResidualEnv.step()` against fake hardware, with the
Station-3 ActionScaler stats from that run's checkpoint.

Run:  <python> -m pytest trossen_real/inference/tests/test_executed_action_storage.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from trossen_real.rl.real_residual_env import TrossenResidualEnv

# Station-3 dataset action stats (delta_joint) + action_scale, from
# run_2026-10-09_11-00-05 checkpoint deployment.normalization.
ACTION_MIN = [-0.06867, -0.17433, -0.16899, -0.12932, -0.06065, -0.23308, 0.0040]
ACTION_MAX = [0.07858, 0.10910, 0.12207, 0.11292, 0.06752, 0.19493, 0.0400]
ACTION_SCALE = [0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.1]
GRIPPER = (0.004, 0.04)
GRIP_N = 0.32727  # scale(0.04) for these stats


def _scaler() -> ActionScaler:
    return ActionScaler(torch.tensor(ACTION_MIN), torch.tensor(ACTION_MAX), ACTION_SCALE, min_range_per_dim=0.1)


class _Follower:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=np.float32)
        self.sent = None

    def get_state(self):
        return {"q": self.q.tolist(), "pose": [0.0] * 6, "gripper_pos": float(self.q[6]),
                "dq": [0.0] * 7, "efforts": [0.0] * 7, "accelerations": [0.0] * 7}

    def move_to_joint_positions(self, arr, blocking=False, min_time_to_move=None):
        self.sent = np.asarray(arr, dtype=np.float32)


class _Cameras:
    def get_all_latest(self):
        return {"cam": np.zeros((8, 8, 3), dtype=np.uint8)}


class _Policy:
    def __init__(self, action_real):
        self.action_real = np.asarray(action_real, dtype=np.float32)

    def predict(self, obs):
        return self.action_real

    def reset(self):
        pass


def _env(base_real, q):
    scaler = _scaler()
    control = SimpleNamespace(gripper_closed=GRIPPER[0], gripper_open=GRIPPER[1],
                              goal_time_s=0.05, frequency_hz=100000)
    env = TrossenResidualEnv(
        follower=_Follower(q), cameras=_Cameras(), policy=_Policy(base_real), control=control,
        reset_cfg=None, action_scaler=scaler,
        state_standardizer=StateStandardizer(torch.zeros(7), torch.ones(7)),
        image_keys=["observation.images.cam"], camera_resolution=(8, 8),
        action_space="delta_joint", policy_image_encoding="raw",
    )
    env._last_base_naction = env._query_base_action(env.follower.get_state(), {"cam": np.zeros((8, 8, 3), np.uint8)})
    return env, scaler


def test_normalized_bounds():
    low, high = _scaler().normalized_bounds(GRIPPER)
    np.testing.assert_allclose(low[:6].numpy(), -1.0)
    np.testing.assert_allclose(high[:6].numpy(), 1.0)
    assert low[6].item() == pytest.approx(-GRIP_N, abs=1e-4)
    assert high[6].item() == pytest.approx(GRIP_N, abs=1e-4)
    # Without gripper bounds it is the plain +-1 box.
    low, high = _scaler().normalized_bounds()
    np.testing.assert_allclose(low.numpy(), -1.0)
    np.testing.assert_allclose(high.numpy(), 1.0)


@pytest.mark.parametrize("grip_real, grip_residual, expected_exec", [
    (0.0400, +0.10, 0.040),   # fully open, residual pushes further open
    (0.0040, -0.10, 0.004),   # fully closed, residual pushes further closed
    (0.0415, 0.00, 0.040),    # ACT itself overshoots open
    (0.0200, +0.05, None),    # mid-range: nothing clipped
])
def test_stored_action_matches_executed(grip_real, grip_residual, expected_exec):
    q = [-0.27, 2.08, 1.50, -0.84, 0.006, -0.29, 0.02]
    base_real = [0.003, 0.010, -0.013, -0.001, -0.004, 0.003, grip_real]
    env, scaler = _env(base_real, q)
    residual = torch.tensor([[0.05, -0.02, 0.10, 0.03, -0.04, 0.05, grip_residual]])

    _, _, _, _, info = env.step(residual)
    stored = info["scaled_action"]
    executed = env.follower.sent

    # Gripper: stored value lies inside the physical range and IS what was executed.
    assert abs(stored[0, 6].item()) <= GRIP_N + 1e-4
    stored_real = scaler.unscale(stored).squeeze(0).numpy()
    assert stored_real[6] == pytest.approx(executed[6], abs=1e-6)
    if expected_exec is not None:
        assert executed[6] == pytest.approx(expected_exec, abs=1e-6)

    # Joints: stored delta + q_before == executed absolute target, and unchanged vs base + residual.
    np.testing.assert_allclose(stored_real[:6] + np.asarray(q[:6], np.float32), executed[:6], atol=1e-5)
    base_n = scaler.scale(torch.tensor([base_real]))
    np.testing.assert_allclose(stored[0, :6].numpy(), (base_n + residual)[0, :6].numpy(), atol=1e-5)

    # What the actor proposed is the same combined value (now bounded identically).
    np.testing.assert_allclose(info["actor_proposed_action"].numpy(), stored.numpy(), atol=1e-5)


def test_agent_combine_matches_env_bounds():
    base = torch.tensor([[0.95, -0.95, 0.0, 0.0, 0.0, 0.0, 0.32]])
    residual = torch.tensor([[0.10, -0.10, 0.05, 0.0, 0.0, 0.0, 0.10]])
    unbounded = SimpleNamespace(_act_low=None, _act_high=None)
    np.testing.assert_allclose(QAgent._combine(unbounded, base, residual).numpy(),
                               torch.clamp(base + residual, -1, 1).numpy())

    low, high = _scaler().normalized_bounds(GRIPPER)
    bounded = SimpleNamespace(_act_low=low, _act_high=high)
    out = QAgent._combine(bounded, base, residual)
    assert out[0, 0].item() == pytest.approx(1.0) and out[0, 1].item() == pytest.approx(-1.0)
    assert out[0, 6].item() == pytest.approx(GRIP_N, abs=1e-4)
