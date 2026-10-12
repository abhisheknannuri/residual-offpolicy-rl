"""TrossenResidualEnv must hold control.frequency_hz over the WHOLE loop.

Regression: step() slept until period_s right after the move command, then read
the next obs (q_after, cameras, base policy), and the caller ran agent.act() -
both outside the budget. run_2026-10-12_18-47-57 ran at 15.7 Hz instead of 20 Hz,
while infer_app / teleop hold 19.9 Hz. step() now subtracts the measured
overhead from its sleep, so consecutive step() starts are period_s apart.

Run:  <python> -m pytest trossen_real/inference/tests/test_env_loop_pacing.py -v
"""
from __future__ import annotations

import time

import numpy as np
import torch

from trossen_real.inference.tests.test_executed_action_storage import _Cameras, _env

POST_SLEEP_S = 0.008  # base-policy query after the pacing sleep
CALLER_S = 0.007      # agent.act() + buffer add between step() calls
PERIOD_S = 0.05


def _slow_env(max_steps=None):
    env, _ = _env([0.0] * 6 + [0.02], [0.0, 1.5, 1.5, -1.5, 0.0, 0.0, 0.02])
    env.control.frequency_hz = 20
    env.max_steps = max_steps
    env.settle_time_s = 0.0
    env.follower.reset = lambda *a, **k: None
    predict = env.policy.predict

    def slow_predict(obs):
        time.sleep(POST_SLEEP_S)
        return predict(obs)

    env.policy.predict = slow_predict
    return env


def _run(env, n):
    starts = []
    for _ in range(n):
        starts.append(time.perf_counter())
        env.step(torch.zeros(1, 7))
        time.sleep(CALLER_S)
    return np.diff(starts)


def test_loop_period_matches_frequency():
    env = _slow_env()
    dt = _run(env, 40)[5:]  # skip warm-up while the overhead estimate fills
    assert abs(np.median(dt) - PERIOD_S) < 0.003, f"median tick {1000 * np.median(dt):.1f} ms"
    # Old behaviour would be ~PERIOD + POST_SLEEP + CALLER = 65 ms.
    assert np.median(dt) < PERIOD_S + 0.5 * (POST_SLEEP_S + CALLER_S)


def test_reset_is_not_counted_as_overhead():
    env = _slow_env(max_steps=10)
    env.settle_time_s = 0.3  # a long reset in the middle of the run
    dt = _run(env, 30)
    # The tick after the reset is long (the reset itself), but the overhead estimate
    # must stay ~15 ms, not absorb the 300 ms reset, so pacing recovers immediately.
    assert all(o < 0.05 for o in env._tick_overhead_s), list(env._tick_overhead_s)
    steady = np.concatenate([dt[3:9], dt[13:19], dt[23:29]])
    assert abs(np.median(steady) - PERIOD_S) < 0.003


def test_achieved_hz_is_full_loop_rate():
    env = _slow_env()
    logged = []
    env.tick_logger = type("L", (), {"log": lambda self, **kw: logged.append(kw), "close": lambda self: None})()
    _run(env, 25)
    assert logged[0]["achieved_hz"] is None  # first tick after construction/reset
    hz = np.median([r["achieved_hz"] for r in logged[5:]])
    assert abs(hz - 1 / PERIOD_S) < 1.5, hz
