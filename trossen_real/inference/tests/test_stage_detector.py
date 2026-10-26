"""StageDetector against a fake reward server (no network)."""

from __future__ import annotations

import io
import time

import numpy as np
import pytest

from trossen_real.config import StageDetectorConfig
from trossen_real.inference import stage_detector as sdm

CAMS = ["observation.images.cam_specialist", "observation.images.cam_left_wrist"]
NAMES = ["approaching", "grasped", "above", "inserting", "inserted"]


class FakeResp:
    def __init__(self, payload, code=200):
        self._p, self.status_code = payload, code

    def json(self):
        return self._p


class FakeSession:
    """Answers /health, then pops (stage, conf) from `script` per /predict_stage."""

    def __init__(self, script):
        self.script = list(script)
        self.posts = []

    def get(self, url, timeout=None):
        return FakeResp({"camera_names": CAMS, "stage_names": NAMES, "num_stages": 5, "buffer_cap": 128})

    def post(self, url, data=None, files=None, timeout=None):
        frames = np.load(io.BytesIO(files["frames"][1].read()))
        self.posts.append((dict(data), frames.shape))
        stage, conf = self.script.pop(0)
        return FakeResp({"raw_stages": [stage], "raw_confs": [conf]})

    def close(self):
        pass


def make(monkeypatch, script, query_every=3, **cfg):
    fake = FakeSession(script)
    monkeypatch.setattr(sdm.requests, "Session", lambda: fake)
    det = sdm.StageDetector(StageDetectorConfig(enabled=True, **cfg), query_every=query_every)
    det.start_episode()
    return det, fake


def run_ticks(det, n, start=0):
    img = np.zeros((4, 4, 3), np.uint8)
    for t in range(start, start + n):
        det.push(t, {"cam_specialist": img, "cam_left_wrist": img}, np.zeros(7))
        time.sleep(0.01)
    deadline = time.time() + 2
    while det._batch is not None and time.time() < deadline:
        time.sleep(0.01)


def test_query_cadence_and_payload(monkeypatch):
    det, fake = make(monkeypatch, [(0, 0.99)] * 4, query_every=3)
    run_ticks(det, 9)
    assert len(fake.posts) == 3
    data, shape = fake.posts[0]
    assert data["reset"] == "1" and data["num_anchors"] == "1"
    assert shape == (3, 2, 4, 4, 3)  # 3 ticks x 2 cameras, all frames since the last query
    assert fake.posts[1][0]["reset"] == "0"
    det.close()


def test_success_needs_consecutive_confident_queries_and_is_live(monkeypatch):
    script = [(3, 0.99), (4, 0.99), (4, 0.80), (4, 0.95), (4, 0.95), (3, 0.99)]
    det, _ = make(monkeypatch, script, query_every=1, consecutive=2, min_conf=0.85, success_stage=4)
    seen = []
    img = np.zeros((4, 4, 3), np.uint8)
    for t in range(len(script)):
        det.push(t, {"cam_specialist": img, "cam_left_wrist": img}, np.zeros(7))
        deadline = time.time() + 2
        while det.status().queries < t + 1 and time.time() < deadline:
            time.sleep(0.005)
        seen.append(det.status().success)
    # 4@.99 once -> no; 4@.80 breaks it (conf); 4@.95 x2 -> success; 3 -> cleared
    assert seen == [False, False, False, False, True, False]
    det.close()


def test_new_episode_drops_stale_results(monkeypatch):
    det, _ = make(monkeypatch, [(4, 0.99)] * 10, query_every=1)
    run_ticks(det, 3)
    assert det.status().queries > 0
    det.start_episode()
    st = det.status()
    assert st.stage is None and st.queries == 0 and not st.success
    det.close()


def test_bad_stage_index_rejected(monkeypatch):
    with pytest.raises(ValueError):
        make(monkeypatch, [], success_stage=7)


def test_server_error_reported_not_raised(monkeypatch):
    det, fake = make(monkeypatch, [], query_every=1)

    def boom(*a, **k):
        raise ConnectionError("down")

    fake.post = boom
    run_ticks(det, 2)
    st = det.status()
    assert st.error and "unreachable" in st.error and st.stage is None
    det.close()
