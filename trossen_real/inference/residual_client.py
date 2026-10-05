# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""`ResidualPolicyClient` - a drop-in `PolicyClient` that adds an RL residual.

WHAT IT IS
----------
A wrapper that sits where `PolicyClient`/`ChunkedPolicyClient` sit, so the
control loop in `infer_loop.py` needs NO changes at all. It exposes the same
`predict_full` / `predict` / `reset` / `health` / `close`, and returns an action
in the same real units.

Per tick it:

    1. asks the inner client for the BASE BC action (full-resolution images,
       chunked if the inner client is a ChunkedPolicyClient)
    2. normalises it with the checkpoint's OWN ActionScaler
    3. resizes the frames to the RL encoder's size (84) - AFTER the base query,
       exactly as training did
    4. runs the residual actor, and the critic if `collect_q`
    5. combined = clamp(base_n + residual, -1, 1), then unscale to real units
    6. logs base / residual / combined / Q to JSONL, then returns

WHY IT DECODES THE IMAGES RATHER THAN TAKING THEM RAW
-----------------------------------------------------
`infer_loop` hands `predict_full()` an observation whose images are already
base64-encoded for the wire. Taking raw frames instead would mean changing the
loop's signature. Decoding here costs ~1-2 ms per tick and keeps the patch to
"construct a different object", which is the whole point of a drop-in.

WHY THE COMPOSITION LIVES HERE
------------------------------
`clamp(base + residual)` then unscale is ResFiT's rule. Keeping it in this class
means the control loop never learns any method's arithmetic, and a different
residual method is a different class rather than an `if` in the loop.

REQUIRES A HERMETIC CHECKPOINT
------------------------------
The `deployment` block (normalization + base-policy identity) must be present.
A checkpoint without it is refused rather than run against guessed action
scaling - and the base policy's `weights_sha256` is verified against the live
server, because a residual against the wrong base fails silently.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ResidualUnavailable(RuntimeError):
    """Missing deps, missing checkpoint, or a checkpoint without deployment metadata."""


class BasePolicyMismatch(RuntimeError):
    """The server is serving a different ACT checkpoint than this residual expects."""


def decode_image(obj: dict[str, Any]) -> np.ndarray:
    """Exact inverse of `policy_client.encode_image`. Returns HWC uint8 RGB.

    Mirrors that function key for key. Two details it is easy to get wrong, and
    both were: the field is `encoding`, not `codec`; and the jpeg path flips
    channels on BOTH sides (`frame[:, :, ::-1]` going out), which `policy_server
    ._decode_image` also does, so the stored JPEG is a correct image on its own.
    Reversing only one side yields a silently blue-swapped frame fed to the
    encoder - a bug that would look like a bad policy.
    """
    raw = base64.b64decode(obj["data_b64"])
    enc = obj.get("encoding")          # NOT "codec"
    if enc == "zlib":
        raw = zlib.decompress(raw)
    elif enc == "jpeg":
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("jpeg decode failed")
        return np.ascontiguousarray(arr[:, :, ::-1])     # undo the outgoing flip
    return np.frombuffer(raw, np.uint8).reshape(tuple(obj["shape"]))


class ResidualTickLog:
    """One JSONL line per tick: base, residual, combined, Q.

    This is the artifact that answers "is the RL doing anything, or emitting
    zeros" after the fact, so it records the pieces separately rather than only
    the action that was sent.

    Writes on a background thread: a logging stall must never delay a command.
    """

    def __init__(self, path: str | Path, queue_maxsize: int = 2000) -> None:
        import queue

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self.dropped = 0
        self.written = 0
        self._t = threading.Thread(target=self._run, name="residual-log", daemon=True)
        self._t.start()

    def write(self, rec: dict[str, Any]) -> None:
        import queue as _q
        try:
            self._q.put_nowait(rec)
        except _q.Full:
            self.dropped += 1

    def _run(self) -> None:
        import queue as _q
        while not self._stop.is_set():
            try:
                rec = self._q.get(timeout=0.2)
            except _q.Empty:
                continue
            try:
                self._f.write(json.dumps(rec) + "\n")
                self.written += 1
                if self.written % 20 == 0:
                    self._f.flush()
            except Exception:
                logger.exception("residual tick log write failed; disabling")
                return

    def close(self) -> None:
        deadline = time.monotonic() + 5.0
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        self._stop.set()
        self._t.join(timeout=2.0)
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


class ResidualPolicyClient:
    """Drop-in for `PolicyClient`: base BC action + learned residual."""

    def __init__(
        self,
        base_client: Any,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        collect_q: bool = True,
        base_policy_assert: str = "strict",
        log_path: str | Path | None = None,
        residual_scale: float = 1.0,
    ) -> None:
        self.base = base_client
        self.checkpoint_path = Path(checkpoint_path)
        self.collect_q = collect_q
        # A global dial on the residual, for bringing it up carefully on real
        # hardware: 0.0 is pure BC, 1.0 is the policy as trained. Anything in
        # between is NOT what was trained, so it is for commissioning only.
        self.residual_scale = float(residual_scale)
        self._log = ResidualTickLog(log_path) if log_path else None

        self.ticks = 0
        self._used_raw = 0
        self._used_decoded = 0
        self.meta: dict[str, Any] = {}
        self._load(device, base_policy_assert)

    # -- loading -----------------------------------------------------------
    def _load(self, device: str | None, assert_mode: str) -> None:
        try:
            import torch
            from omegaconf import OmegaConf

            from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
            from resfit.rl_finetuning.utils.normalization import (
                ActionScaler,
                StateStandardizer,
            )
        except ImportError as exc:
            raise ResidualUnavailable(
                f"residual inference needs torch + resfit ({exc})"
            ) from exc

        self._torch = torch
        if not self.checkpoint_path.exists():
            raise ResidualUnavailable(f"checkpoint not found: {self.checkpoint_path}")

        ck = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        dep = ck.get("deployment")
        if not dep:
            raise ResidualUnavailable(
                f"{self.checkpoint_path} has no 'deployment' block - its "
                "normalization and base-policy identity are unknown. Retrain; "
                "do not run a residual against guessed action scaling."
            )

        cfg = OmegaConf.create(ck["config"])
        # QAgent puts itself on cfg.agent.device, so inputs must follow it.
        self.device = device or str(cfg.agent.device)
        cfg.agent.device = self.device

        norm = dep["normalization"]
        self.scaler = ActionScaler.from_state_dict(norm["action_scaler"], device=self.device)
        self.standardizer = StateStandardizer.from_state_dict(
            norm["state_standardizer"], device=self.device)

        obs = dep.get("obs") or {}
        self.image_keys: list[str] = list(obs.get("image_keys") or cfg.rl_camera)
        self.rl_image_size = int(obs.get("rl_image_size") or cfg.offline_data.image_size or 84)
        self.n = int(obs.get("action_dim") or 7)

        # Verify the base BEFORE anything can move.
        bp = dep.get("base_policy") or {}
        self.expected_base_sha = bp.get("weights_sha256")
        health = self.base.health()
        self._assert_base(health, bp, assert_mode)

        self.agent = QAgent(
            obs_shape=(3, self.rl_image_size, self.rl_image_size),
            prop_shape=(self.n,), action_dim=self.n,
            rl_cameras=self.image_keys, cfg=cfg.agent, residual_actor=True,
        )
        missing, unexpected = self.agent.load_state_dict(ck["agent_state_dict"], strict=False)
        if missing or unexpected:
            raise ResidualUnavailable(
                f"checkpoint/agent mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected. First missing: {missing[:3]}")
        self.agent.train(False)
        self.agent.to(self.device)

        self.meta = {
            "checkpoint": str(self.checkpoint_path),
            "actor_updates": ck.get("actor_updates"),
            "global_step": ck.get("global_step"),
            "device": self.device,
            "rl_image_size": self.rl_image_size,
            "image_keys": self.image_keys,
            "action_scale": list(cfg.agent.actor.action_scale),
            "expected_base_sha256": self.expected_base_sha,
            "server_base_sha256": health.get("weights_sha256"),
            "base_checkpoint": bp.get("server_checkpoint_path"),
            "dataset": (dep.get("provenance") or {}).get("dataset_name"),
            "collect_q": self.collect_q,
            "residual_scale": self.residual_scale,
        }
        # WARM UP before the control loop ever calls us.
        #
        # Measured: the first forward pass costs ~126 ms on CUDA (kernel
        # autotuning and lazy module init), against a 50 ms tick budget. Paying
        # that on tick 1 of a live episode is a guaranteed missed deadline with
        # the arm already moving. Paying it here, at load, costs nothing.
        self._warmup()

        logger.info("residual policy loaded: %s", json.dumps(self.meta, default=str))
        if self._log:
            self._log.write({"event": "load", "t": time.time(), **self.meta})

    def _warmup(self, n: int = 3) -> None:
        torch = self._torch
        t0 = time.perf_counter()
        dummy = {k: torch.zeros(1, 3, self.rl_image_size, self.rl_image_size,
                                dtype=torch.uint8, device=self.device)
                 for k in self.image_keys}
        dummy["observation.state"] = torch.zeros(1, self.n, device=self.device)
        dummy["observation.base_action"] = torch.zeros(1, self.n, device=self.device)
        with torch.no_grad():
            for _ in range(n):
                self.agent.act(dict(dummy), eval_mode=True, stddev=0.0, cpu=False)
                if self.collect_q:
                    f = self.agent._encode(dict(dummy), augment=False)
                    self.agent.critic(f, dummy["observation.state"],
                                      dummy["observation.base_action"])
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        self.meta["warmup_ms"] = round(ms, 1)
        logger.info("residual warmup: %.0f ms over %d passes", ms, n)

    def _assert_base(self, health: dict, bp: dict, mode: str) -> None:
        if mode == "off":
            return
        want, got = bp.get("weights_sha256"), health.get("weights_sha256")
        if want and got and want == got:
            return
        msg = (
            "BASE POLICY MISMATCH - this residual was trained against a different "
            "ACT checkpoint. A residual is a correction to ONE base policy and is "
            f"meaningless against another.\n"
            f"  expects weights_sha256 = {want}\n"
            f"    (trained against {bp.get('server_checkpoint_path')})\n"
            f"  server reports         = {got}\n"
            f"    (now serving {health.get('checkpoint')})"
        )
        if got is None:
            msg += ("\n  The server reports no weights_sha256 - it predates the hash. "
                    "Update policy_server.py, or pass base_policy_assert='off'.")
        if mode == "warn":
            logger.warning(msg)
            return
        raise BasePolicyMismatch(msg)

    # -- PolicyClient interface --------------------------------------------
    def health(self) -> dict:
        h = dict(self.base.health())
        h["residual"] = self.meta
        return h

    def reset(self) -> None:
        """Episode boundary / intervention release.

        Forwards to the base client, which drops any chunk queue - those
        actions were planned against a pre-reset observation. The residual
        actor is stateless.
        """
        self.base.reset()

    def predict(self, obs: dict, raw_images: dict | None = None) -> np.ndarray:
        return self.predict_full(obs, raw_images)["action"]

    def predict_full(self, obs: dict, raw_images: dict | None = None) -> dict:
        """`raw_images`: the undecoded camera frames, when the caller has them.

        Strongly preferred over decoding the wire payload. Training fed the RL
        encoder a clean frame resized to 84; decoding a jpeg back would add
        artifacts training never saw. Measured on real wrist-cam frames that
        mismatch is small - 0.67/255 mean at 84x84, below the 1.39 the dataset's
        own AV1 encoding already contributes - but it is free to avoid, and
        skipping the decode saves time on the control path too.
        """
        torch = self._torch
        t0 = time.perf_counter()

        base_res = self.base.predict_full(obs)
        base_real = np.asarray(base_res["action"], dtype=np.float64).reshape(-1)
        t_base = time.perf_counter()

        with torch.no_grad():
            base_t = torch.as_tensor(base_real, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            base_n = self.scaler.scale(base_t)

            rl_obs = self._rl_obs(obs, base_n, raw_images)
            residual = self.agent.act(rl_obs, eval_mode=True, stddev=0.0, cpu=False)
            if self.residual_scale != 1.0:
                residual = residual * self.residual_scale

            combined_n = torch.clamp(base_n + residual, -1.0, 1.0)
            combined_real = self.scaler.unscale(combined_n)

            qinfo: dict[str, Any] = {}
            if self.collect_q:
                feat = self.agent._encode(rl_obs, augment=False)
                qv = self.agent.critic(feat, rl_obs["observation.state"], combined_n)
                qa = qv.detach().cpu().numpy().ravel()
                qinfo = {"q_mean": float(qa.mean()), "q_std": float(qa.std()),
                         "q_min": float(qa.min()), "q_max": float(qa.max()),
                         "q_heads": [float(v) for v in qa]}

        action = combined_real.squeeze(0).cpu().numpy().astype(np.float64)
        res_np = residual.squeeze(0).cpu().numpy().astype(np.float64)
        base_n_np = base_n.squeeze(0).cpu().numpy().astype(np.float64)
        # How much of the residual the [-1,1] clamp threw away. Non-zero here
        # means the base was already near the edge of the action box.
        unclamped = base_n_np + res_np
        clipped = float(np.abs(unclamped - np.clip(unclamped, -1, 1)).max())

        self.ticks += 1
        t_end = time.perf_counter()
        if self._log:
            self._log.write({
                "event": "tick", "i": self.ticks, "t": time.time(),
                "base_action": base_real.tolist(),
                "base_naction": base_n_np.tolist(),
                "residual": res_np.tolist(),
                "combined_naction": combined_n.squeeze(0).cpu().numpy().tolist(),
                "action": action.tolist(),
                "residual_abs_max": float(np.abs(res_np).max()),
                "clipped_amount": clipped,
                "dt_base_ms": round((t_base - t0) * 1000, 3),
                "dt_rl_ms": round((t_end - t_base) * 1000, 3),
                **qinfo,
            })

        out = dict(base_res)
        out.update({
            "action": action,
            "base_action": base_real,
            "residual": res_np,
            "residual_abs_max": float(np.abs(res_np).max()),
            "clipped_amount": clipped,
            "image_source": "raw" if raw_images is not None else "decoded",
            **qinfo,
        })
        return out

    @property
    def image_source_counts(self) -> dict[str, int]:
        """How many frames came in clean vs had to be decoded back.

        Decoded frames carry jpeg artifacts the RL encoder never saw in
        training. A nonzero `decoded` count means something is calling
        predict_full() without raw_images.
        """
        return {"raw": self._used_raw, "decoded": self._used_decoded}

    def _rl_obs(self, obs: dict, base_n: Any, raw_images: dict | None = None) -> dict:
        """Resize to the RL size and standardize. The base policy saw full res.

        Prefers `raw_images` (clean pixels, as training had) and falls back to
        decoding the wire payload only when the caller did not supply them.
        """
        torch = self._torch
        import torch.nn.functional as F

        out: dict[str, Any] = {}
        for key in self.image_keys:
            cam = key.removeprefix("observation.images.")
            if raw_images is not None and cam in raw_images:
                img = np.asarray(raw_images[cam])
                self._used_raw += 1
            else:
                raw = obs.get(key)
                if raw is None:
                    raise KeyError(f"residual needs image '{key}' in the observation")
                img = decode_image(raw) if isinstance(raw, dict) else np.asarray(raw)
                self._used_decoded += 1
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
            t = t.unsqueeze(0).to(self.device).float()
            if t.shape[-1] != self.rl_image_size or t.shape[-2] != self.rl_image_size:
                t = F.interpolate(t, size=(self.rl_image_size, self.rl_image_size),
                                  mode="bilinear", align_corners=False)
            out[key] = t.round().clamp(0, 255).to(torch.uint8)

        q = torch.as_tensor(np.asarray(obs["observation.state"], dtype=np.float32),
                            device=self.device).reshape(1, -1)
        out["observation.state"] = self.standardizer.standardize(q)
        out["observation.base_action"] = base_n
        return out

    def close(self) -> None:
        if self._log:
            self._log.close()
        try:
            self.base.close()
        except Exception:
            pass
        self.agent = None
        if self._torch is not None and self._torch.cuda.is_available():
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass
