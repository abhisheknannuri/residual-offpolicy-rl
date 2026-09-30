# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Synchronous HTTP client for the external stage-aware reward server.

The client is intentionally model-agnostic: it ships a rolling buffer of raw
frames + states to the server as ``.npy`` blobs, and the server performs all
model-specific windowing (SARM temporal sampling / TCC context), inference, and
server-side stage-transition gating (hysteresis + confidence). This keeps the
RL side identical whether the reward model is SARM or TCC.

Wire protocol (multipart/form-data over HTTP):

    GET  /health
        -> {"status": "ok", "model_type": "sarm"|"tcc", "num_stages": int, ...}

    POST /reset
        form: {"session_id": str}
        -> {"status": "ok"}

    POST /predict_stage
        files: frames (.npy uint8 [W, H, W_img, 3]),
               states (.npy float32 [W, state_dim])
        form:  session_id, task, num_anchors, reset ("0"/"1"),
               hysteresis_k, conf_threshold, monotonic ("0"/"1")
        -> {"gated_stage": int, "stage_conf": float,
            "raw_stages": [int, ...], "raw_confs": [float, ...],
            "taus": [float, ...], "num_anchors": int,
            "server_latency_s": float}
"""

from __future__ import annotations

import io
import time

import numpy as np
import requests


def _numpy_to_npy_bytes(arr: np.ndarray) -> io.BytesIO:
    """Serialize a numpy array to in-memory ``.npy`` bytes (Robometer-style)."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    buf.seek(0)
    return buf


class RewardModelClient:
    """Blocking client for the stage-aware reward server."""

    def __init__(
        self,
        server_url: str,
        *,
        task_prompt: str = "",
        request_timeout_s: float = 20.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.task_prompt = task_prompt
        self.request_timeout_s = float(request_timeout_s)

    # ------------------------------------------------------------------
    # Health / lifecycle
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """Return the server ``/health`` payload (raises on failure)."""
        r = requests.get(f"{self.server_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def check(self) -> bool:
        """Return True iff the server is reachable and reports status ok."""
        try:
            info = self.health()
            return str(info.get("status", "")).lower() == "ok"
        except Exception:
            return False

    def reset(self, session_id: str) -> None:
        """Clear the server-side hysteresis state for ``session_id``."""
        r = requests.post(
            f"{self.server_url}/reset",
            data={"session_id": session_id},
            timeout=self.request_timeout_s,
        )
        r.raise_for_status()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_stage(
        self,
        frames: np.ndarray,
        states: np.ndarray,
        *,
        session_id: str,
        num_anchors: int,
        reset: bool = False,
        hysteresis_k: int = 5,
        conf_threshold: float = 0.80,
        monotonic: bool = True,
        task: str | None = None,
    ) -> dict:
        """Score the ``num_anchors`` most-recent frames and return the gated stage.

        Args:
            frames: uint8 array ``[W, H, W_img, 3]`` (chronological, most-recent last).
            states: float32 array ``[W, state_dim]`` aligned with ``frames``.
            session_id: per-env session key for server-side hysteresis.
            num_anchors: number of most-recent frames to score in this request.
            reset: if True, the server clears hysteresis state before scoring.
        Returns:
            The parsed JSON response with an extra ``client_latency_s`` field.
        """
        frames = np.ascontiguousarray(frames)
        states = np.ascontiguousarray(states.astype(np.float32, copy=False))
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)

        files = {
            "frames": ("frames.npy", _numpy_to_npy_bytes(frames), "application/octet-stream"),
            "states": ("states.npy", _numpy_to_npy_bytes(states), "application/octet-stream"),
        }
        data = {
            "session_id": session_id,
            "task": task if task is not None else self.task_prompt,
            "num_anchors": str(int(num_anchors)),
            "reset": "1" if reset else "0",
            "hysteresis_k": str(int(hysteresis_k)),
            "conf_threshold": f"{float(conf_threshold):.6f}",
            "monotonic": "1" if monotonic else "0",
        }

        t0 = time.perf_counter()
        r = requests.post(
            f"{self.server_url}/predict_stage",
            files=files,
            data=data,
            timeout=self.request_timeout_s,
        )
        client_latency = time.perf_counter() - t0

        if r.status_code >= 400:
            try:
                err = r.json().get("error", r.text)
            except Exception:
                err = r.text
            raise RuntimeError(f"reward server /predict_stage failed ({r.status_code}): {err}")

        out = r.json()
        out["client_latency_s"] = client_latency
        return out
