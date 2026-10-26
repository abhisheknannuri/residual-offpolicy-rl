"""TrossenResidualEnv + SARM success: rewards / episode ends (fake hardware only).

Rules (real_residual_env.py step()): the pedals win; otherwise a SARM success gives
reward 1 and ends the episode when the station config's stage_detector flags say so.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from trossen_real.inference.stage_detector import StageStatus
from trossen_real.inference.tests.test_executed_action_storage import _env

Q = [0.0, 1.5, 1.5, -1.5, 0.0, 0.0, 0.02]


class FakeDetector:
    def __init__(self, success, label=True, end=True):
        self.cfg = SimpleNamespace(label_reward_on_success=label, end_episode_on_success=end)
        self.success = success
        self.pushed, self.episodes = [], 0

    def start_episode(self):
        self.episodes += 1

    def push(self, tick, images, q):
        self.pushed.append(tick)

    def status(self):
        return StageStatus(enabled=True, stage=4 if self.success else 3, conf=0.99, success=self.success)

    def close(self):
        pass


class FakeMonitor:
    def __init__(self, reward=0.0, end=(False, None)):
        self.reward, self.end = reward, end

    def get_frame_reward(self):
        return self.reward

    def check_episode_end(self):
        return self.end

    def start_episode(self):
        pass


def make(detector, monitor=None):
    env, _ = _env([0.0] * 6 + [0.02], Q)
    env.settle_time_s = 0.0
    env.follower.reset = lambda *a, **k: None
    env.stage_detector = detector
    env.episode_monitor = monitor
    logged = []
    env.tick_logger = type("L", (), {"log": lambda self, **kw: logged.append(kw), "close": lambda self: None})()
    return env, logged


def step(env):
    _, reward, terminated, _, info = env.step(torch.zeros(1, 7))
    return float(reward.item()), bool(terminated.item()), bool(info["buffer_done"].item())


def test_sarm_success_gives_reward_and_ends_episode():
    det = FakeDetector(success=True)
    env, logged = make(det)
    assert step(env) == (1.0, True, True)
    assert logged[-1]["stop_reason"] == "sarm_success"
    assert det.episodes == 1  # the terminal step reset the env -> new SARM session


def test_no_success_no_reward():
    env, logged = make(FakeDetector(success=False))
    assert step(env) == (0.0, False, False)
    assert logged[-1]["sarm_stage"] == 3


def test_reward_pedal_held_wins():
    # pedal held: reward 1 but the episode continues until the release, SARM does not end it
    env, logged = make(FakeDetector(success=True), FakeMonitor(reward=1.0))
    assert step(env) == (1.0, False, True)
    assert logged[-1]["stop_reason"] is None


def test_reset_pedal_wins():
    env, logged = make(FakeDetector(success=True), FakeMonitor(end=(True, "reset_pedal")))
    assert step(env) == (0.0, True, True)
    assert logged[-1]["stop_reason"] == "reset_pedal"


@pytest.mark.parametrize("label,end,expected", [
    (False, False, (0.0, False, False)),
    (True, False, (1.0, False, True)),   # reward only; buffer done (reward 1), episode goes on
    (False, True, (0.0, True, True)),
])
def test_flags(label, end, expected):
    env, _ = make(FakeDetector(success=True, label=label, end=end))
    assert step(env) == expected


def test_frames_pushed_every_tick():
    det = FakeDetector(success=False)
    env, _ = make(det)
    env.reset()
    for _ in range(3):
        step(env)
    assert det.pushed == [0, 1, 2, 3]
