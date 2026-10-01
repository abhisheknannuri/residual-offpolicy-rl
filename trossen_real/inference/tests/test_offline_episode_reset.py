# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""The offline buffer build must reset the base policy at every episode boundary.

Why it matters: `policy_server.py`'s `/predict` serves from
`ACTPolicy.select_action()`'s internal queue and only runs the model when that
queue is empty, so with `n_action_steps=k` only one call in `k` looks at the
image you sent. The dataset is streamed with `shuffle=False` and episodes follow
one another, so without a reset a chunk planned at the end of one episode
supplies base actions for the first frames of the next - labels computed from a
different episode's images.

Online this cannot happen: `TrossenResidualEnv.reset()` resets the policy each
episode. These tests pin the offline side to the same behaviour.

The population loop is a closure inside `train_residual_td3.py` and importing
that module pulls in the whole training stack, so rather than execute it these
tests assert on the loop's source: that the reset exists, is keyed on the
episode index, and sits before the base-policy query.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TRAINER = Path(__file__).resolve().parents[3] / "resfit/rl_finetuning/scripts/train_residual_td3.py"


@pytest.fixture(scope="module")
def loop_src() -> str:
    src = TRAINER.read_text()
    start = src.index('for sample in tqdm(loader, desc="Processing offline dataset")')
    end = src.index("# Save buffer to disk for future runs", start)
    return src[start:end]


def test_reset_is_keyed_on_the_episode_index(loop_src):
    assert "if ep_idx != prev_ep_idx:" in loop_src, (
        "the reset must fire on an episode CHANGE, not every frame - resetting "
        "per frame would make every base action a fresh inference, which is a "
        "different training signal from what the policy does online"
    )
    assert "real_policy_client.reset()" in loop_src
    assert "base_policy.reset()" in loop_src, "the sim/in-process path needs it too"


def test_reset_happens_before_the_base_action_query(loop_src):
    """Order is the whole point: resetting after the query would leave the first
    frame of each episode served from the previous episode's chunk."""
    reset_at = loop_src.index("if ep_idx != prev_ep_idx:")
    query_at = loop_src.index("real_policy_client.predict(")
    assert reset_at < query_at, "reset must precede the first predict() of the episode"


def test_prev_ep_idx_is_initialised_and_advanced(loop_src):
    src = TRAINER.read_text()
    assert "prev_ep_idx: int | None = None" in src, "must start as None so episode 0 also resets"
    assert re.search(r"prev_ep_idx\s*=\s*ep_idx", loop_src), "must advance, or it resets every frame"


def test_reset_simulation_matches_expected_call_pattern():
    """Re-implement the guard exactly as written and check it fires once per
    episode boundary over a realistic frame sequence - including episode 0."""
    episodes = [0] * 5 + [1] * 3 + [2] * 4          # 12 frames, 3 episodes
    calls, prev = [], None
    for i, ep in enumerate(episodes):
        if ep != prev:
            calls.append(i)
            prev = ep
    assert calls == [0, 5, 8], f"expected a reset at each boundary incl. frame 0, got {calls}"
    assert len(calls) == len(set(episodes))


def test_guard_does_not_fire_within_an_episode():
    episodes = [7] * 50
    calls, prev = [], None
    for i, ep in enumerate(episodes):
        if ep != prev:
            calls.append(i)
            prev = ep
    assert calls == [0], "a single episode must reset exactly once, at its first frame"
