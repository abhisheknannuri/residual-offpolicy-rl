# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Shared HTTP client for `policy_server.py` (the training repo's process)
and the observation-building helpers around it - used by BOTH the CLI
deploy script (`scripts/live_infer_deploy.py`) and the inference web UI
(`infer_app/`), so there's exactly one place that defines "what an
observation looks like" and "how images get encoded for the wire".

See `policy_server.py` (training repo, `custom_scripts/policy_server.py`)
for the server side of this exact same contract.
"""

from __future__ import annotations

import base64
import json
import time
import zlib
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests

from trossen_real.call_stats import track_call
from trossen_real.cameras.camera_manager import CameraManager


class TickLogger:
    """Appends one JSON object per line (JSONL) per inference tick - for
    OFFLINE inspection/plotting (e.g. gripper predicted vs. actual over
    time), not for control. Used by both `scripts/live_infer_deploy.py` and
    `infer_app/infer_loop.py` so the log schema is identical either way.

    Deliberately dumb/robust: one line per call, flushed immediately (so a
    crash/Ctrl+C mid-rollout doesn't lose already-written ticks), never
    raises (a logging failure should never take down a live control loop -
    logs a warning to stderr once and disables itself instead).
    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a")
        self._disabled = False

    def log(self, **fields) -> None:
        if self._disabled:
            return
        record = {"wall_time": time.time(), **fields}

        def _default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return str(o)

        try:
            self._file.write(json.dumps(record, default=_default) + "\n")
            self._file.flush()
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "TickLogger failed to write to %s - disabling further logging for this run.", self.log_path
            )
            self._disabled = True

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class PolicyClient:
    """Thin HTTP client for `policy_server.py`.

    Uses a single `requests.Session` for the client's whole lifetime, so the
    TCP connection is opened once and reused (HTTP keep-alive) instead of being
    rebuilt per call. With the server far away - especially behind an SSH
    tunnel, where a new TCP connection also means a new tunnelled channel -
    the handshake was costing more than the payload.
    """

    def __init__(self, base_url: str, request_timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        # Short tag for the live call_stats view - e.g. "policy:5090".
        self._stats_client = f"policy:{self.base_url.rsplit(':', 1)[-1]}"
        self._session = requests.Session()
        # One connection is all a single control loop needs; capping the pool
        # makes an accidental connection leak show up as a warning rather than
        # as silent socket growth.
        self._session.mount(
            "http://", requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=2)
        )

    def close(self) -> None:
        self._session.close()

    def health(self) -> dict:
        with track_call(self._stats_client, "/health"):
            r = self._session.get(f"{self.base_url}/health", timeout=self.request_timeout_s)
            r.raise_for_status()
        return r.json()

    def reset(self) -> None:
        with track_call(self._stats_client, "/reset"):
            r = self._session.post(f"{self.base_url}/reset", timeout=self.request_timeout_s)
            r.raise_for_status()

    def predict(self, obs: dict) -> np.ndarray:
        return self.predict_full(obs)["action"]

    def predict_full(self, obs: dict, raw_images: dict | None = None) -> dict:
        """Same as `predict()` but returns the FULL `/predict` response as
        numpy arrays - `{"action": ..., "action_normalized": ...}` - for
        callers that want to log/inspect the model's raw (still-normalized)

        `raw_images`: accepted and IGNORED here. It exists so the control loop
        can hand the undecoded camera frames to any wrapper that needs real
        pixels (`ResidualPolicyClient` does) without that wrapper having to
        decode the wire payload back. Adding it as an ignored parameter keeps
        one call signature across every policy client.

        output alongside the real-units one it actually acts on (see
        `policy_server.py::_predict()`'s docstring for why `action_normalized`
        is diagnostic-only, never meant to be used for control)."""
        with track_call(self._stats_client, "/predict"):
            r = self._session.post(f"{self.base_url}/predict", json=obs, timeout=self.request_timeout_s)
            r.raise_for_status()
            payload = r.json()
            if "error" in payload:
                raise RuntimeError(f"policy_server /predict error: {payload['error']}")
        return {
            "action": np.asarray(payload["action"], dtype=np.float32),
            "action_normalized": np.asarray(payload["action_normalized"], dtype=np.float32)
            if "action_normalized" in payload else None,
        }

    def predict_chunk(self, obs: dict, n_steps: int | None = None) -> list[dict]:
        """One round trip -> the whole action chunk, as a list of the same
        dicts `predict_full()` returns one at a time.

        See `ChunkedPolicyClient` for why this exists. `n_steps` defaults to
        the server's configured `n_action_steps`.
        """
        body = dict(obs)
        if n_steps is not None:
            body["n_steps"] = int(n_steps)
        with track_call(self._stats_client, "/predict_chunk"):
            r = self._session.post(f"{self.base_url}/predict_chunk", json=body,
                                   timeout=self.request_timeout_s)
            r.raise_for_status()
            payload = r.json()
            if "error" in payload:
                raise RuntimeError(f"policy_server /predict_chunk error: {payload['error']}")
        actions = payload["actions"]
        normalized = payload.get("actions_normalized") or [None] * len(actions)
        return [
            {
                "action": np.asarray(a, dtype=np.float32),
                "action_normalized": np.asarray(n, dtype=np.float32) if n is not None else None,
            }
            for a, n in zip(actions, normalized)
        ]


class ChunkedPolicyClient:
    """A `PolicyClient` that fetches a whole action chunk per round trip.

    Drop-in for `PolicyClient` everywhere the control loop touches it
    (`predict_full`/`reset`/`health`), so `infer_loop.py` needs no changes.

    **This changes the network pattern, not the actions.** The server already
    chunks: `/predict` calls `ACTPolicy.select_action()`, which runs the model
    only when its internal queue is empty and pops a cached step otherwise (see
    `modeling_act.py::select_action`), so with `n_action_steps=15` the model ran
    once per 15 requests and the other 14 requests uploaded ~512 KB of images
    just to trigger a `deque.popleft()` on the far side. This moves that queue
    to the client. The actions are bit-identical because `select_action()`'s
    refill is exactly `predict_action_chunk(batch)[:, :n_action_steps]`, which
    is what `/predict_chunk` returns.

    It is NOT prefetching/real-time chunking: the next chunk is requested only
    once the previous one is exhausted, from a fresh observation, exactly as the
    server's own queue behaved. So the policy never sees an observation staler
    than it did before.

    Cost: the control loop pauses for one round trip every `n_steps` ticks.
    That pause is the reason the round trip itself has to be cheap (keep-alive,
    and a payload that isn't half a megabyte of base64).
    """

    def __init__(self, client: PolicyClient, n_steps: int | None = None) -> None:
        self.client = client
        self.n_steps = n_steps
        self._queue: deque[dict] = deque()

    # -- PolicyClient interface ------------------------------------------
    def health(self) -> dict:
        return self.client.health()

    def reset(self) -> None:
        # Order matters: drop the buffered plan FIRST. These actions were
        # planned against a pre-reset observation, and a reset happens exactly
        # when that plan stopped being valid (end of run, or a human taking
        # over via the leader arm - see InterventionManager).
        self._queue.clear()
        self.client.reset()

    def predict_full(self, obs: dict, raw_images: dict | None = None) -> dict:
        if not self._queue:
            chunk = self.client.predict_chunk(obs, self.n_steps)
            if not chunk:
                raise RuntimeError("policy_server /predict_chunk returned an empty chunk")
            self._queue.extend(chunk)
        return self._queue.popleft()

    def predict(self, obs: dict) -> np.ndarray:
        return self.predict_full(obs)["action"]

    def close(self) -> None:
        self.client.close()

    # -- introspection ----------------------------------------------------
    @property
    def buffered(self) -> int:
        """Actions left in the current chunk; 0 means the next tick pays a round trip."""
        return len(self._queue)


IMAGE_ENCODINGS = ("raw", "zlib", "jpeg")


def encode_image(frame: np.ndarray, encoding: str = "raw", jpeg_quality: int = 95) -> dict:
    """(H,W,3) uint8 RGB numpy array -> the JSON-safe payload `policy_server.py`'s
    `_decode_image()` expects.

    `encoding` trades wire size against fidelity. Measured on real 256x256
    wrist-cam frames (2 images per observation, base64 included):

        raw    512 KB   lossless   - the original behaviour, and the default
        zlib   157 KB   lossless   - 3.3x smaller, bit-identical pixels
        jpeg    34 KB   lossy      - 15x smaller at quality 95

    On a remote policy server this payload IS the control-loop budget: at 20 Hz
    the raw form needs 84 Mbit/s sustained, and every tick waits for it.

    On `jpeg` being lossy: the LeRobot dataset stores its images as AV1 video at
    crf=30, so the policy was TRAINED on lossily-compressed frames. Measured
    against the same real frames, JPEG q95 distorts less than that AV1 pass does
    (mean abs error 0.79 vs 1.39 on 0-255; PSNR 45.8 dB vs 43.1 dB) - so it does
    not push inference further from the training distribution than training
    already was. Below about q85 that stops being true; don't go there without
    re-measuring.
    """
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    payload = {"shape": list(frame.shape), "dtype": "uint8"}
    if encoding == "raw":
        blob = frame.tobytes()
    elif encoding == "zlib":
        # Level 1: on 256x256 camera frames levels 6 and 1 land within a few
        # percent of each other in size, but level 6 costs ~3.5 ms per image
        # against a 50 ms tick budget. Compression here is about the link, not
        # about the last byte.
        blob = zlib.compress(frame.tobytes(), 1)
        payload["encoding"] = "zlib"
    elif encoding == "jpeg":
        # cv2 works in BGR; the server flips it back. Keeping the flip on both
        # sides (rather than encoding RGB as if it were BGR) means the JPEG is a
        # correct image on its own and can be eyeballed when debugging.
        ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed to JPEG-encode a camera frame")
        blob = buf.tobytes()
        payload["encoding"] = "jpeg"
    else:
        raise ValueError(f"encoding must be one of {IMAGE_ENCODINGS}, got {encoding!r}")
    payload["data_b64"] = base64.b64encode(blob).decode("ascii")
    return payload


def build_observation(follower_state: dict, images: dict[str, np.ndarray], image_keys: list[str],
                      image_encoding: str = "raw", jpeg_quality: int = 95) -> dict:
    """Build the exact JSON body `policy_server.py`'s `/predict` expects.

    All 6 non-image keys come from ONE `/getstate` call, regardless of
    `command_space` (`follower_single_server.py`'s `/getstate` doesn't
    branch on it) - `observation.state` here is the raw 7D joint vector
    (q), matching how `command_space: joint` recorded it (no axis-angle/
    quaternion conversion, see `dataset_recorder.py::add_frame()`'s
    symmetric skip of `_to_sim_state()` in joint mode).

    Raises a clear `RuntimeError` (not a confusing KeyError deep inside
    `policy_server.py`) if a camera the policy actually needs has no frame
    available - e.g. a mock/disconnected slot.
    """
    q = list(follower_state["q"])
    pose = list(follower_state["pose"])
    gripper = float(follower_state["gripper_pos"])
    obs = {
        "observation.state": q,
        "observation.joint_pos_raw": q,
        "observation.ee_pose_raw": pose + [gripper],
        "observation.velocity": list(follower_state["dq"]),
        "observation.effort": list(follower_state["efforts"]),
        "observation.acceleration": list(follower_state["accelerations"]),
    }
    for key in image_keys:
        cam_name = key.removeprefix("observation.images.")
        frame = images.get(cam_name)
        if frame is None:
            raise RuntimeError(
                f"Policy needs camera '{cam_name}' but no frame is available for it "
                "(check the station config's cameras.devices)."
            )
        obs[key] = encode_image(frame, image_encoding, jpeg_quality)
    return obs


def reconstruct_absolute_action(
    action_space: str,
    predicted: np.ndarray,
    follower_state: dict,
    gripper_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Turn the policy server's raw output into an ABSOLUTE joint target
    ready to send via `move_to_joint_positions()`.

    `action_space`:
      - `"absolute"` (default, matches the original non-delta checkpoint):
        `predicted` already IS the absolute target - passed through as-is.
      - `"delta_joint"` (see `scripts/convert_to_delta_joint_dataset.py` /
        `PickAndInsertCube.md`): `predicted[:6]` is a JOINT DELTA relative
        to `observation.state` at the SAME tick the policy conditioned on -
        exactly mirroring how the training data was constructed
        (`action[t]-state[t]`, per-frame, NOT a chunk-start reference - see
        the doc for why that distinction matters here). Reconstruction is
        therefore a single add using the FRESHLY MEASURED current state:
        `absolute[:6] = predicted[:6] + follower_state["q"][:6]`. Gripper
        (`predicted[6]`) was never delta-converted at training time (kept
        absolute, see `convert_to_delta_joint_dataset.py`'s gripper
        exclusion) - passed through as-is, exactly like the `"absolute"`
        case.

        CORRECTED 2026-09-02 (a previous version of this comment wrongly
        claimed this was only exactly correct at chunk position 0, requiring
        `n_action_steps=1`): this reconstruction is actually correct at
        EVERY chunk position, including cached ones popped several ticks
        after the chunk was inferred - because `follower_state` here is
        whatever the CALLER passes in, and `infer_loop.py` always fetches
        it FRESH at the current tick (confirmed by reading that code)
        before calling this function. The training target for chunk
        position `k` was `action[t+k] - state[t+k]` (each row's OWN
        matching state, not a chunk-start anchor - see
        `convert_to_delta_joint_dataset.py`), so `predicted[k] +
        real_state_at_this_tick` reconstructs it correctly regardless of
        how many ticks have passed since the chunk was inferred. There is
        no `n_action_steps=1` requirement for this checkpoint type -
        `n_action_steps` is purely a compute/reactivity tradeoff (fewer
        forward passes vs. less responsive to real-time disturbances), same
        as for an `"absolute"` checkpoint.

    `gripper_bounds`: if given, `(gripper_closed, gripper_open)` physical
    limits (e.g. `config.control.gripper_closed`/`.gripper_open`) - the
    gripper dim is ALWAYS clipped to this range regardless of
    `action_space` (a policy's raw regression output has no hard guarantee
    of staying in-range, same principle as clipping any network output at
    the denormalization boundary - mirrors the exact clip already applied
    to the LEADER's gripper command in `control_loop.py::_tick()`). Pass
    `None` to skip (not recommended for real hardware).
    """
    predicted = np.asarray(predicted, dtype=np.float32)
    if action_space == "absolute":
        absolute = predicted.copy()
    elif action_space == "delta_joint":
        q = np.asarray(follower_state["q"], dtype=np.float32)
        absolute = predicted.copy()
        absolute[:6] = predicted[:6] + q[:6]
    else:
        raise ValueError(f"Unknown action_space '{action_space}' - expected 'absolute' or 'delta_joint'.")

    if gripper_bounds is not None:
        gripper_closed, gripper_open = gripper_bounds
        absolute[6] = np.clip(absolute[6], min(gripper_closed, gripper_open), max(gripper_closed, gripper_open))
    return absolute


def check_action_space_mismatch(action_space: str, policy_health: dict) -> str | None:
    """Cross-checks the caller's chosen `action_space` against
    `policy_health["likely_action_space"]` - a best-effort HEURISTIC
    `policy_server.py` computes from the checkpoint's own `train_config.json`
    (`dataset.repo_id` containing "delta", case-insensitive - matches this
    project's dataset naming convention, see
    `scripts/convert_to_delta_joint_dataset.py`).

    Returns a human-readable warning string if they disagree, else `None`.
    `None` is also returned if the heuristic itself is `None` (unknown -
    e.g. an older checkpoint with no `train_config.json`, or a checkpoint
    trained outside this project's naming convention) - there's nothing
    useful to compare against in that case, so it's silently skipped rather
    than treated as a mismatch.

    IMPORTANT: this is a SAFETY NET, not a substitute for the caller
    actually knowing which checkpoint they loaded - getting `action_space`
    wrong is dangerous in BOTH directions: treating a delta_joint
    checkpoint's small (~0.01-0.03 rad) output as an absolute target would
    command a sudden large jump toward near-zero joint angles; treating an
    absolute checkpoint's output as a delta would add it on top of the
    current state, roughly doubling the intended target. The heuristic can
    be wrong (e.g. a checkpoint/dataset renamed to not contain "delta") -
    callers should warn loudly, not silently trust either the heuristic or
    the user's selection blindly.
    """
    likely = policy_health.get("likely_action_space")
    if likely is None or likely == action_space:
        return None
    return (
        f"action_space mismatch: you selected '{action_space}', but the checkpoint's train_config.json "
        f"dataset.repo_id suggests this checkpoint was actually trained with action_space='{likely}' "
        f"(heuristic based on 'delta' appearing in the dataset path - checkpoint: "
        f"{policy_health.get('checkpoint')!r}). Double-check before proceeding - this is dangerous to get "
        f"wrong in either direction (see check_action_space_mismatch()'s docstring)."
    )


def check_camera_health(cameras: CameraManager, image_keys: list[str]) -> list[str]:
    """Return a list of human-readable PROBLEMS (empty = all good) for every
    camera the policy actually needs (`image_keys`, e.g.
    `['observation.images.cam_right_wrist', ...]`) that is either not
    configured, currently a MOCK feed, or unhealthy (no fresh frame
    recently - see `CameraManager.health()`, which reflects ACTUAL frame
    freshness, not just whether the camera object was constructed).

    Used by both `scripts/live_infer_deploy.py` (raises if non-empty) and
    the inference web UI (shown directly in the status panel) - silently
    feeding the policy synthetic/stale frames would make its predictions
    meaningless with no indication anything was wrong, so this check is
    deliberately loud and blocking rather than a warning buried in a log.
    """
    health = cameras.health()
    problems = []
    for key in image_keys:
        cam_name = key.removeprefix("observation.images.")
        if cam_name not in health:
            problems.append(f"'{cam_name}' is not even configured in this station's cameras.devices")
        elif cameras.is_mock(cam_name):
            problems.append(f"'{cam_name}' is a MOCK feed (no real serial connected, or it fell back to mock)")
        elif not health[cam_name]:
            problems.append(f"'{cam_name}' is unhealthy (no fresh frame recently - see get_camera_status())")
    return problems
