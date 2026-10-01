# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Chunked fetching must stay correct across intervention and episode resets.

The risk with a client-side action queue is a stale plan outliving the moment it
stopped being valid. Two moments matter on real hardware:

* **Episode reset.** `TrossenResidualEnv.reset()` calls `policy.reset()`
  (real_residual_env.py:233). The next query must come from a fresh chunk.
* **Intervention release.** `InterventionManager.tick()` calls `policy.reset()`
  on the True -> False edge ONLY (intervention.py:82), so that the first
  autonomous action after the human lets go is a fresh forward pass rather than
  the continuation of a plan made before they took over.

Both work only because the env hands `InterventionManager` the SAME policy
object it queries (real_residual_env.py:147) - so wrapping that object in
`ChunkedPolicyClient` puts one queue behind both reset paths. These tests pin
that, and pin that the manager does NOT reset on the press edge or mid-hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from trossen_real.inference.intervention import InterventionManager
from trossen_real.inference.policy_client import ChunkedPolicyClient


class _FakeClient:
    """Stands in for PolicyClient. Each chunk is tagged so a stale action is
    identifiable by value, not just by count."""

    def __init__(self, n=5):
        self.n = n
        self.chunk_id = 0
        self.chunk_calls = 0
        self.resets = 0

    def predict_chunk(self, obs, n_steps=None):
        self.chunk_calls += 1
        self.chunk_id += 1
        return [{"action": np.full(7, float(self.chunk_id), dtype=np.float32),
                 "action_normalized": None} for _ in range(self.n)]

    def reset(self):
        self.resets += 1

    def health(self):
        return {"loaded": True, "supports_predict_chunk": True}

    def close(self):
        pass


class _FakePedal:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def is_intervention(self):
        v = self.script[self.i]
        self.i += 1
        return v


class _FakeLeader:
    def set_freedrive_mode(self):
        pass

    def set_position_mode(self):
        pass


OBS = {"observation.state": [0.0] * 7}


def test_one_round_trip_serves_a_whole_chunk():
    c = _FakeClient(n=5)
    p = ChunkedPolicyClient(c, 5)
    vals = [float(p.predict_full(OBS)["action"][0]) for _ in range(5)]
    assert c.chunk_calls == 1
    assert vals == [1.0] * 5
    p.predict_full(OBS)                      # 6th query exhausts the chunk
    assert c.chunk_calls == 2


def test_reset_drops_the_queue_so_the_next_action_is_fresh():
    c = _FakeClient(n=5)
    p = ChunkedPolicyClient(c, 5)
    p.predict_full(OBS)
    assert p.buffered == 4
    p.reset()
    assert p.buffered == 0
    assert float(p.predict_full(OBS)["action"][0]) == 2.0, "must come from a NEW chunk"
    assert c.resets == 1, "the server's own queue must be reset too"


def test_intervention_release_resets_the_chunk():
    """press, hold, release -> exactly one reset, on the release edge."""
    c = _FakeClient(n=5)
    p = ChunkedPolicyClient(c, 5)
    mgr = InterventionManager(_FakeLeader(), _FakePedal([False, True, True, False, False]), p)

    mgr.tick()                                  # idle
    assert float(p.predict_full(OBS)["action"][0]) == 1.0
    mgr.tick()                                  # press edge - must NOT reset
    assert c.resets == 0, "resetting on the PRESS edge would discard a valid plan"
    mgr.tick()                                  # held - still no reset
    assert c.resets == 0
    p.predict_full(OBS)                         # queried while intervened, as the env does
    mgr.tick()                                  # RELEASE edge -> reset
    assert c.resets == 1
    assert p.buffered == 0
    assert float(p.predict_full(OBS)["action"][0]) == 2.0, "first autonomous action must be fresh"
    mgr.tick()                                  # still idle - no further reset
    assert c.resets == 1


def test_queue_is_consumed_while_intervened_matching_server_side_behaviour():
    """The env queries the base policy every step even while intervened
    (real_residual_env.py:333), and the server's own queue would also advance.
    Client-side chunking must behave the same, not freeze."""
    c = _FakeClient(n=5)
    p = ChunkedPolicyClient(c, 5)
    mgr = InterventionManager(_FakeLeader(), _FakePedal([True] * 4), p)
    for _ in range(4):
        mgr.tick()
        p.predict_full(OBS)
    assert p.buffered == 1, "the chunk must advance during intervention, as it does server-side"
    assert c.resets == 0


def test_manager_holds_the_same_object_the_env_queries():
    """If these were two different clients, a release-edge reset would clear a
    queue nobody reads and leave the real one stale."""
    c = _FakeClient()
    p = ChunkedPolicyClient(c, 5)
    mgr = InterventionManager(_FakeLeader(), _FakePedal([False]), p)
    assert mgr.policy is p
