# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""`ChunkedPolicyClient` must change the network pattern and NOTHING else.

The whole argument for chunked fetching is that `/predict` already served
actions out of a server-side queue, so moving that queue to the client cannot
change what the robot does. That is a claim about equivalence, so it is tested
against a stand-in server that mimics `ACTPolicy.select_action()`'s queue
exactly (`modeling_act.py:127-133`): refill with
`predict_action_chunk(...)[:, :n_action_steps]`, then `popleft()`.

Also covers the two things that would quietly break a live run: keep-alive not
actually reusing the socket, and `reset()` leaving a stale plan in the buffer
(which would make the robot replay pre-intervention actions after a human let go
of the leader arm).
"""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from trossen_real.inference.policy_client import ChunkedPolicyClient, PolicyClient

N_ACTION_STEPS = 15
CHUNK_SIZE = 20
ACTION_DIM = 7


def _chunk_for(call_index: int) -> np.ndarray:
    """Deterministic stand-in for the model: same input -> same chunk."""
    return np.random.default_rng(call_index).normal(size=(CHUNK_SIZE, ACTION_DIM)).round(6)


class _FakePolicyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    queue: deque = deque()
    model_calls = 0
    stats = {"predict": 0, "predict_chunk": 0, "model_runs": 0, "connections": 0}

    def setup(self):
        type(self).stats["connections"] += 1
        super().setup()

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send({"loaded": True, "n_action_steps": N_ACTION_STEPS,
                    "supports_predict_chunk": True, "image_keys": []})

    def do_POST(self):  # noqa: N802
        cls = type(self)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/reset":
            cls.queue.clear()
            return self._send({"status": "ok"})
        if self.path == "/predict":
            cls.stats["predict"] += 1
            if not cls.queue:                                   # select_action()'s refill
                cls.stats["model_runs"] += 1
                cls.queue.extend(_chunk_for(cls.model_calls)[:N_ACTION_STEPS])
                cls.model_calls += 1
            action = cls.queue.popleft()
            return self._send({"action": action.tolist(), "action_normalized": action.tolist()})
        if self.path == "/predict_chunk":
            cls.stats["predict_chunk"] += 1
            cls.stats["model_runs"] += 1
            n = body.get("n_steps") or N_ACTION_STEPS
            chunk = _chunk_for(cls.model_calls)[:n]
            cls.model_calls += 1
            cls.queue.clear()
            return self._send({"actions": chunk.tolist(),
                               "actions_normalized": chunk.tolist(), "n": len(chunk)})
        return self._send({"error": "unknown"}, 404)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakePolicyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _reset_fake():
    _FakePolicyHandler.queue.clear()
    _FakePolicyHandler.model_calls = 0
    _FakePolicyHandler.stats = {"predict": 0, "predict_chunk": 0, "model_runs": 0, "connections": 0}


def _run(client, ticks):
    obs = {"observation.state": [0.0] * 7}
    return [client.predict_full(obs)["action"] for _ in range(ticks)]


def test_chunked_and_per_tick_produce_identical_actions(server):
    ticks = 3 * N_ACTION_STEPS

    _reset_fake()
    per_tick = PolicyClient(server)
    per_tick.reset()
    seq_a = _run(per_tick, ticks)
    round_trips_a = _FakePolicyHandler.stats["predict"]
    model_runs_a = _FakePolicyHandler.stats["model_runs"]
    per_tick.close()

    _reset_fake()
    chunked = ChunkedPolicyClient(PolicyClient(server), N_ACTION_STEPS)
    chunked.reset()
    seq_b = _run(chunked, ticks)
    round_trips_b = _FakePolicyHandler.stats["predict_chunk"]
    model_runs_b = _FakePolicyHandler.stats["model_runs"]
    chunked.close()

    assert all(np.array_equal(a, b) for a, b in zip(seq_a, seq_b))
    assert round_trips_a == ticks                     # one per control tick
    assert round_trips_b == ticks // N_ACTION_STEPS   # one per chunk
    # The model ran the same number of times either way - which is the point:
    # the extra round trips were never buying extra inference.
    assert model_runs_a == model_runs_b == ticks // N_ACTION_STEPS


def test_keep_alive_reuses_one_connection(server):
    _reset_fake()
    client = PolicyClient(server)
    _run(client, 30)
    client.close()
    assert _FakePolicyHandler.stats["connections"] == 1


def test_reset_drops_the_buffered_chunk(server):
    _reset_fake()
    chunked = ChunkedPolicyClient(PolicyClient(server), N_ACTION_STEPS)
    chunked.predict_full({"observation.state": [0.0] * 7})
    assert chunked.buffered == N_ACTION_STEPS - 1
    chunked.reset()
    assert chunked.buffered == 0
    chunked.close()


def test_n_steps_none_uses_server_default(server):
    _reset_fake()
    chunked = ChunkedPolicyClient(PolicyClient(server), None)
    chunked.predict_full({"observation.state": [0.0] * 7})
    assert chunked.buffered == N_ACTION_STEPS - 1
    chunked.close()


def test_empty_chunk_is_an_error_not_a_silent_hang(server, monkeypatch):
    chunked = ChunkedPolicyClient(PolicyClient(server), N_ACTION_STEPS)
    monkeypatch.setattr(chunked.client, "predict_chunk", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="empty chunk"):
        chunked.predict_full({"observation.state": [0.0] * 7})
    chunked.close()
