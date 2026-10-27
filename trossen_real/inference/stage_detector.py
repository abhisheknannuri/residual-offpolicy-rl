# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""SARM stage and success detection for the inference app.

Talks to opensarm's `reward_server.py` (multi-camera, 2026-10-22). The control
loop calls `push()` every tick with that tick's camera frames and joint state;
that only appends to a list. Every `query_every` ticks (default: the ACT chunk
size) a background thread sends the frames collected since the last query and
asks the server to score only the newest one (`num_anchors=1`). All frames have
to be sent, not just the newest: the server builds each window from the
episode's past frames (n_obs_steps samples, frame_gap apart), exactly as in
training. The control loop never waits on the network: it reads the latest
result with `status()`.

Detection rule (StageDetectorConfig): a stage counts as detected when the last
`consecutive` queries all predicted it with confidence > `min_conf`. It is
live: the next query that breaks the rule clears it. Success = the
`success_stage` is detected.

If the server is unreachable or errors, the error is reported in `status()`
and detection simply stops updating; inference is never interrupted.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import requests

from trossen_real.config import StageDetectorConfig

logger = logging.getLogger(__name__)

IMAGE_PREFIX = "observation.images."


@dataclass
class StageStatus:
    enabled: bool = False
    stage: int | None = None  # latest predicted stage, None before the first result
    stage_name: str | None = None
    conf: float | None = None
    stage_tick: int | None = None  # control tick the latest prediction is for
    detected_stage: int | None = None  # stage meeting the consecutive/min_conf rule, else None
    success: bool = False
    target_active: bool = False
    # Residual-RL gate (residual_from_stage): None when not configured, else False until
    # residual_consecutive queries in a row said stage >= residual_from_stage with
    # conf > residual_min_conf, then True for the rest of the episode.
    residual_active: bool | None = None
    queries: int = 0
    last_latency_s: float | None = None
    error: str | None = None
    history: list[dict] = field(default_factory=list)  # recent queries, newest last


def _npy(arr: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    buf.seek(0)
    return buf


class StageDetector:
    HISTORY = 20

    def __init__(self, cfg: StageDetectorConfig, query_every: int) -> None:
        self.cfg = cfg
        self.query_every = max(1, int(query_every))
        self.url = cfg.server_url.rstrip("/")
        self._session = requests.Session()
        health = self._session.get(f"{self.url}/health", timeout=cfg.request_timeout_s).json()
        if "camera_names" not in health or "stage_names" not in health:
            raise RuntimeError(
                f"reward server at {self.url} does not report camera_names/stage_names "
                "(needs the multi-camera reward_server.py)"
            )
        self.server_health = health
        self.stage_names: list[str] = list(health["stage_names"])
        # SARM camera keys -> CameraManager names
        self.camera_names = [c[len(IMAGE_PREFIX):] if c.startswith(IMAGE_PREFIX) else c
                             for c in health["camera_names"]]
        for name, s in (("success_stage", cfg.success_stage), ("target_stage", cfg.target_stage),
                        ("residual_from_stage", cfg.residual_from_stage)):
            if s is not None and not 0 <= s < len(self.stage_names):
                raise ValueError(f"stage_detector.{name}={s} but the model has stages 0..{len(self.stage_names) - 1}")

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending_frames: list[np.ndarray] = []  # each [num_cameras, H, W, 3] uint8
        self._pending_states: list[np.ndarray] = []
        self._pending_last_tick = -1
        self._batch: tuple | None = None  # handed to the worker
        self._session_id = ""
        self._first = True
        self._recent: list[tuple[int, float]] = []  # (stage, conf) of the last `consecutive` queries
        self._gate_recent: list[bool] = []  # residual gate: did the last residual_consecutive queries pass?
        self._status = self._fresh_status()
        self._stop = False
        self._worker = threading.Thread(target=self._run, daemon=True, name="stage-detector")
        self._worker.start()

    # ---- control-loop side (must never block) --------------------------------
    def start_episode(self) -> None:
        with self._lock:
            self._session_id = f"infer-{uuid.uuid4().hex[:8]}"
            self._first = True
            self._pending_frames.clear()
            self._pending_states.clear()
            self._pending_last_tick = -1
            self._batch = None
            self._recent.clear()
            self._gate_recent.clear()
            self._status = self._fresh_status()

    def push(self, tick: int, images: dict[str, np.ndarray], joint_state) -> None:
        """Add one control tick. Missing cameras skip the frame (and are reported)."""
        try:
            frame = np.stack([images[c] for c in self.camera_names])
        except KeyError as exc:
            with self._lock:
                self._status.error = f"camera {exc} not available for the stage detector"
            return
        with self._lock:
            self._pending_frames.append(frame)
            self._pending_states.append(np.asarray(joint_state, dtype=np.float32))
            self._pending_last_tick = tick
            # Hand over a batch once enough ticks have piled up and the worker is idle.
            # If it is still busy the frames keep accumulating and go with the next request.
            if self._batch is None and len(self._pending_frames) >= self.query_every:
                self._batch = (
                    self._session_id, self._first, np.stack(self._pending_frames),
                    np.stack(self._pending_states), self._pending_last_tick,
                )
                self._first = False
                self._pending_frames.clear()
                self._pending_states.clear()
                self._cv.notify()

    def _fresh_status(self) -> StageStatus:
        return StageStatus(enabled=True,
                           residual_active=None if self.cfg.residual_from_stage is None else False)

    def status(self) -> StageStatus:
        with self._lock:
            s = self._status
            return StageStatus(**{**s.__dict__, "history": list(s.history)})

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._cv.notify()
        self._worker.join(timeout=2.0)
        self._session.close()

    # ---- worker -----------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._lock:
                while self._batch is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                batch = self._batch
                session_id, first, frames, states, tick = batch
            result, error, latency = None, None, None
            try:
                t0 = time.perf_counter()
                r = self._session.post(
                    f"{self.url}/predict_stage",
                    data={"session_id": session_id, "reset": "1" if first else "0", "num_anchors": "1"},
                    files={"frames": ("frames.npy", _npy(frames), "application/octet-stream"),
                           "states": ("states.npy", _npy(states), "application/octet-stream")},
                    timeout=self.cfg.request_timeout_s,
                )
                latency = time.perf_counter() - t0
                out = r.json()
                if r.status_code != 200:
                    error = f"reward server: {out.get('error', out)}"
                else:
                    result = (int(out["raw_stages"][-1]), float(out["raw_confs"][-1]))
            except Exception as exc:  # noqa: BLE001 - reported, never raised into the control loop
                error = f"reward server unreachable: {exc}"
            with self._lock:
                if self._batch is batch:  # start_episode() may have replaced it meanwhile
                    self._batch = None
                if session_id != self._session_id:
                    continue  # a new episode started while this request was in flight
                self._apply(result, error, latency, tick)

    def _apply(self, result, error, latency, tick) -> None:
        st = self._status
        st.last_latency_s = latency
        st.error = error
        if result is None:
            return
        stage, conf = result
        k = max(1, self.cfg.consecutive)
        self._recent = (self._recent + [(stage, conf)])[-k:]
        detected = None
        if len(self._recent) == k and all(s == stage and c > self.cfg.min_conf for s, c in self._recent):
            detected = stage
        st.stage, st.conf, st.stage_tick = stage, conf, tick
        st.stage_name = self.stage_names[stage]
        st.detected_stage = detected
        st.success = detected == self.cfg.success_stage
        st.target_active = self.cfg.target_stage is not None and detected == self.cfg.target_stage
        if self.cfg.residual_from_stage is not None and not st.residual_active:
            k_res = max(1, self.cfg.residual_consecutive)
            ok = stage >= self.cfg.residual_from_stage and conf > self.cfg.residual_min_conf
            self._gate_recent = (self._gate_recent + [ok])[-k_res:]
            if len(self._gate_recent) == k_res and all(self._gate_recent):
                st.residual_active = True  # latched until start_episode()
        st.queries += 1
        st.history = (st.history + [{"tick": tick, "stage": stage, "conf": round(conf, 3),
                                     "success": st.success}])[-self.HISTORY:]


def make_stage_detector(station_config, policy_chunk_steps: int, policy_health: dict) -> StageDetector | None:
    """The StageDetector for a station config's `stage_detector` block, or None if
    it is disabled. Used by the RL trainers (the infer app has the same logic
    inline in /api/connect). Raises if it is enabled but unusable, so a run never
    silently goes without SARM success detection.

    query_every: the block's value, else the ACT chunk size (`policy_chunk_steps`
    if > 0, else the policy server's n_action_steps).
    """
    sd_cfg = station_config.stage_detector
    if not sd_cfg.enabled:
        return None
    if sd_cfg.query_every:
        query_every = int(sd_cfg.query_every)
    elif policy_chunk_steps and policy_chunk_steps > 0:
        query_every = int(policy_chunk_steps)
    else:
        query_every = int(policy_health.get("n_action_steps") or 0)
    if query_every <= 0:
        raise ValueError("stage_detector.query_every is not set and the ACT chunk size is unknown - "
                         "set stage_detector.query_every in the station config")
    try:
        det = StageDetector(sd_cfg, query_every)
    except Exception as exc:
        raise RuntimeError(f"stage_detector.enabled but the SARM reward server at {sd_cfg.server_url} "
                           f"is not usable - is reward_server.py running? ({exc})") from exc
    real_cams = {d.name for d in station_config.cameras.devices if d.serial}
    missing = [c for c in det.camera_names if c not in real_cams]
    if missing:
        det.close()
        raise ValueError(f"the SARM model needs cameras {missing}, which are not real cameras (with a "
                         f"serial) in the station config")
    logger.info("SARM stage detector ON: %s, query every %d ticks, success = stage %d in %d queries "
                "with conf > %.2f, label_reward=%s end_episode=%s", sd_cfg.server_url, query_every,
                sd_cfg.success_stage, sd_cfg.consecutive, sd_cfg.min_conf,
                sd_cfg.label_reward_on_success, sd_cfg.end_episode_on_success)
    return det
