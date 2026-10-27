"""ResidualPolicyClient with residual_active=False passes the base action through
(no RL network call). Built without a checkpoint: only what the gated path uses."""

from __future__ import annotations

import numpy as np

from trossen_real.inference.residual_client import ResidualPolicyClient


class _Base:
    def predict_full(self, obs):
        return {"action": np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 0.004]), "chunk_index": 3}


class _NoRL:
    def act(self, *a, **k):  # must not be called while gated off
        raise AssertionError("RL network called while the residual is gated off")


def test_gated_off_returns_base_action():
    c = object.__new__(ResidualPolicyClient)
    c._torch = None
    c.base, c.agent, c._log, c.ticks = _Base(), _NoRL(), None, 0
    c.residual_active = False
    out = c.predict_full({"observation.state": [0.0] * 7})
    np.testing.assert_allclose(out["action"], _Base().predict_full({})["action"])
    assert out["residual_active"] is False and out["residual_abs_max"] == 0.0
    assert out["chunk_index"] == 3  # base result fields kept
    assert c.ticks == 1
