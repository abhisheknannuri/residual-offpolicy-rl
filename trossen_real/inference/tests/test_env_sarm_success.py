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


# ---- operator confirmation (stage_detector.confirm_success) ------------------------

class FakePedal:
    """Stale presses before the prompt, then the operator's answer after `delay` polls."""

    def __init__(self, answer, delay=3, stale_reset=True, stale_reward=True):
        self.answer, self.delay = answer, delay
        self.stale = {"reset": stale_reset, "reward": stale_reward}
        self.polls = 0

    def consume_reset(self):
        if self.stale["reset"]:
            self.stale["reset"] = False
            return True
        self.polls += 1
        return self.answer == "reset" and self.polls > self.delay

    def consume_reward_latched(self):
        if self.stale["reward"]:
            self.stale["reward"] = False
            return True
        return self.answer == "reward" and self.polls > self.delay

    def is_healthy(self):
        return True

    def get_reward(self):  # read by the env's reset() after the episode ends
        return 0


def make_confirming(answer):
    det = FakeDetector(success=True)
    det.cfg.confirm_success = True
    env, logged = make(det)
    env.pedal = FakePedal(answer)
    return env, logged


def test_confirm_keep_with_reward_pedal():
    env, logged = make_confirming("reward")
    assert step(env) == (1.0, True, True)
    assert logged[-1]["stop_reason"] == "sarm_success_confirmed"


def test_confirm_override_with_reset_pedal():
    env, logged = make_confirming("reset")
    assert step(env) == (0.0, True, True)  # failure: reward 0, episode ends
    assert logged[-1]["stop_reason"] == "sarm_success_overridden"


def test_stale_presses_before_the_prompt_are_ignored():
    # both pedals have a press pending from before the prompt; the answer comes later
    env, logged = make_confirming("reset")
    step(env)
    assert env.pedal.polls > env.pedal.delay  # it waited for the new press
    assert logged[-1]["stop_reason"] == "sarm_success_overridden"


def test_confirm_needs_a_pedal():
    from trossen_real.inference.tests.test_executed_action_storage import _Cameras, _Follower, _Policy, _scaler
    from trossen_real.inference.tests.test_executed_action_storage import TrossenResidualEnv, StateStandardizer
    det = FakeDetector(success=True)
    det.cfg.confirm_success = True
    with pytest.raises(ValueError, match="pedal"):
        TrossenResidualEnv(
            follower=_Follower(Q), cameras=_Cameras(), policy=_Policy([0.0] * 7), control=SimpleNamespace(
                gripper_closed=0.004, gripper_open=0.04, goal_time_s=0.05, frequency_hz=1000),
            reset_cfg=None, action_scaler=_scaler(), state_standardizer=StateStandardizer(torch.zeros(7), torch.ones(7)),
            image_keys=["observation.images.cam"], camera_resolution=(8, 8), stage_detector=det,
        )


# ---- pedal state must not leak into the next episode --------------------------------

from trossen_real.human_intervention.episode_boundary import EpisodeBoundaryMonitor  # noqa: E402
from trossen_real.human_intervention.pedal_listener import PedalListener  # noqa: E402


class HeldPedal(PedalListener):
    """The real PedalListener latch logic without a device. The reward pedal is held
    (pressed to confirm) and released after `release_after` reads of it."""

    def __init__(self, release_after=5):
        self.reward_pressed, self._reward_latch = True, True
        self.reset_pressed, self._reset_triggered = False, False
        self.intervention_pressed = False
        self._dummy_mode, self._dev = False, object()
        self.reads, self.release_after = 0, release_after

    def get_reward(self):
        self.reads += 1
        if self.reads > self.release_after:
            self.reward_pressed = False
        return super().get_reward()


class OneSuccessDetector(FakeDetector):
    def start_episode(self):
        super().start_episode()
        self.success = False  # the next episode has no success


class ConfirmThenHoldPedal(HeldPedal):
    """Pedal up when the SARM prompt appears; the operator presses the reward pedal
    during the prompt and keeps it down through the reset, releasing it after
    `release_after` reads."""

    def __init__(self, release_after=5):
        super().__init__(release_after)
        self.reward_pressed = self._reward_latch = False
        self.polls = 0

    def consume_reward_latched(self):
        self.polls += 1
        if self.polls == 3 and self.reads <= self.release_after:  # pressed during the prompt
            self.reward_pressed = self._reward_latch = True
        return super().consume_reward_latched()


def test_confirm_press_does_not_leak_into_next_episode():
    det = OneSuccessDetector(success=True)
    det.cfg.confirm_success = True
    env, logged = make(det)
    env.pedal = ConfirmThenHoldPedal(release_after=5)
    env.episode_monitor = EpisodeBoundaryMonitor(env.pedal)
    env.episode_monitor.start_episode()
    assert step(env) == (1.0, True, True)  # confirmed via the prompt, episode ends
    assert logged[-1]["stop_reason"] == "sarm_success_confirmed"
    # New episode: the press that confirmed must not give reward or end it.
    for _ in range(3):
        assert step(env) == (0.0, False, False)
    assert not env.pedal.reward_pressed


def test_reset_waits_for_held_pedal_then_starts_clean():
    det = FakeDetector(success=False)
    env, logged = make(det)
    env.pedal = HeldPedal(release_after=10)
    env.episode_monitor = EpisodeBoundaryMonitor(env.pedal)
    env.reset()
    assert env.pedal.reads > 10  # it waited for the release
    assert step(env) == (0.0, False, False)
