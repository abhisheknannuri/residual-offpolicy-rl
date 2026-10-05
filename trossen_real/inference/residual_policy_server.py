# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Residual RL policy server - the same shape as `policy_server.py`.

    GET  /health            -> checkpoint identity, base-policy hash, obs spec
    POST /reset             -> forwards to the base policy (drops its chunk queue)
    POST /predict           -> {"action", "base_action", "residual", "q_*"}
    GET  /stats             -> tick count, latency, residual magnitude so far

WHEN TO USE THIS RATHER THAN THE IN-PROCESS CLIENT
--------------------------------------------------
`ResidualPolicyClient` in-process costs ~3.9 ms/tick and is the better default.
This server exists for the cases where that is not possible:

  * the machine driving the robot has no GPU or no torch/resfit;
  * you want to swap residual checkpoints without restarting the infer app;
  * you want the RL on a different box from the arm.

The cost is honest and worth stating: the residual CANNOT be chunked, because
it needs the current observation AND the current base action every tick. So
every tick is a round trip carrying images. At 84x84 that is small (~5 KB jpeg
for two cameras), but it is still a network hop inside the control loop, where
the in-process path has none.

NOTE ON THE BASE POLICY: this server owns the base client too, so the images
arrive here once and the base query happens server-side. That keeps the wire
payload to one set of frames per tick rather than two.

!! USE raw OR zlib, NOT jpeg, WHEN THE RL IS BEHIND THIS SERVER !!

In-process, the residual client is handed the UNENCODED frames, so the encoder
sees exactly what training fed it. Over this server there are no raw frames to
hand over - only the wire payload - so the images must be decoded back, and a
jpeg round trip is then baked into the RL's input.

That is not a cosmetic difference. Measured on real wrist-cam frames, jpeg q95
changes the residual by up to 0.0199, which is 39.7% of the smallest
action_scale and roughly 96% of the residual's own mean magnitude. The pixel
error looks negligible (0.67/255 at 84x84); the OUTPUT error is not. The policy
amplifies it.

zlib is lossless and ~2.9x smaller than raw, and with the base policy chunked
the frames only cross the wire once per n_action_steps anyway, so the bandwidth
argument for jpeg mostly evaporates. This server refuses jpeg by default for
that reason; --allow-lossy overrides it if you are deliberately measuring the
effect.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LossyInput(RuntimeError):
    """jpeg frames reached the RL encoder. See the module docstring."""


_warned_lossy = [False]


def _check_lossless(body: dict) -> None:
    """Refuse jpeg input unless explicitly allowed.

    Loud rather than silent because the damage is invisible: the arm keeps
    moving, the numbers look plausible, and the residual is simply wrong by
    about its own magnitude.
    """
    lossy = [k for k, v in body.items()
             if k.startswith("observation.images.")
             and isinstance(v, dict) and v.get("encoding") == "jpeg"]
    if not lossy:
        return
    if state.allow_lossy:
        if not _warned_lossy[0]:
            _warned_lossy[0] = True
            logger.warning(
                "jpeg frames are reaching the RL encoder (%s). Training used clean "
                "frames; measured effect is ~40%% of the smallest action_scale. "
                "Proceeding because --allow-lossy was passed.", ", ".join(lossy))
        return
    raise LossyInput(
        f"jpeg-encoded frames ({', '.join(lossy)}) would reach the RL encoder, which "
        f"training never saw. Measured effect: up to 39.7% of the smallest "
        f"action_scale. Send image_encoding='zlib' (lossless, ~2.9x smaller than raw) "
        f"or 'raw', or pass --allow-lossy to this server if you are deliberately "
        f"measuring the degradation."
    )


class _State:
    client: Any = None
    started: float = 0.0
    ticks: int = 0
    dt_sum: float = 0.0
    dt_max: float = 0.0
    residual_abs_max: float = 0.0
    q_sum: float = 0.0
    allow_lossy: bool = False


state = _State()


def _handler_cls():
    class Handler(BaseHTTPRequestHandler):
        # Keep-alive: a fresh TCP connection per tick turned a ~50 ms call into
        # ~475 ms in the base policy server. The same applies here.
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet; the control loop logs its own timing
            pass

        def _send(self, obj: dict, status: int = 200) -> None:
            blob = json.dumps(obj, default=_json_default).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            c = state.client
            if self.path == "/health":
                if c is None:
                    self._send({"loaded": False}, 503)
                    return
                self._send({"loaded": True, "kind": "residual_rl", **c.meta})
            elif self.path == "/stats":
                n = max(state.ticks, 1)
                self._send({
                    "ticks": state.ticks,
                    "uptime_s": round(time.time() - state.started, 1),
                    "dt_mean_ms": round(state.dt_sum / n, 3),
                    "dt_max_ms": round(state.dt_max, 3),
                    "residual_abs_max": round(state.residual_abs_max, 6),
                    "q_mean": round(state.q_sum / n, 5),
                })
            else:
                self._send({"error": f"unknown GET {self.path}"}, 404)

        def do_POST(self):
            c = state.client
            if c is None:
                self._send({"error": "no residual policy loaded"}, 503)
                return
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)

            if self.path == "/reset":
                c.reset()
                self._send({"status": "ok"})
                return
            if self.path != "/predict":
                self._send({"error": f"unknown POST {self.path}"}, 404)
                return

            t0 = time.perf_counter()
            try:
                body = json.loads(raw)
                _check_lossless(body)
                out = c.predict_full(body)
            except LossyInput as exc:
                self._send({"error": str(exc)}, 400)
                return
            except Exception as exc:
                logger.exception("predict failed")
                self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            dt = (time.perf_counter() - t0) * 1000.0

            state.ticks += 1
            state.dt_sum += dt
            state.dt_max = max(state.dt_max, dt)
            state.residual_abs_max = max(state.residual_abs_max,
                                         float(out.get("residual_abs_max", 0.0)))
            state.q_sum += float(out.get("q_mean", 0.0))
            out["server_dt_ms"] = round(dt, 3)
            self._send(out)

    return Handler


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="residual_policy_server",
        description="Serve a residual RL checkpoint on top of a base BC policy server.")
    ap.add_argument("--checkpoint", required=True, help="ResFiT checkpoint.pt")
    ap.add_argument("--base-url", default="http://127.0.0.1:5070",
                    help="the ACT policy server this residual was trained against")
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default: this endpoint moves a robot")
    ap.add_argument("--port", type=int, default=5080)
    ap.add_argument("--device", default=None, help="default: whatever the checkpoint says")
    ap.add_argument("--chunk-steps", type=int, default=-1,
                    help="base policy chunking: 0 per tick, -1 the server's n_action_steps")
    ap.add_argument("--no-q", action="store_true", help="skip the critic (saves ~1.1 ms)")
    ap.add_argument("--assert-base", default="strict", choices=("strict", "warn", "off"))
    ap.add_argument("--log", default=None, help="per-tick JSONL path")
    ap.add_argument("--residual-scale", type=float, default=1.0,
                    help="commissioning dial: 0.0 = pure BC, 1.0 = as trained")
    ap.add_argument("--allow-lossy", action="store_true",
                    help="accept jpeg frames for the RL encoder (see the module "
                         "docstring - this measurably changes the residual)")
    args = ap.parse_args()
    state.allow_lossy = args.allow_lossy

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from trossen_real.inference.policy_client import ChunkedPolicyClient, PolicyClient
    from trossen_real.inference.residual_client import ResidualPolicyClient

    base: Any = PolicyClient(args.base_url)
    if args.chunk_steps:
        base = ChunkedPolicyClient(base, None if args.chunk_steps < 0 else args.chunk_steps)

    state.client = ResidualPolicyClient(
        base, args.checkpoint, device=args.device, collect_q=not args.no_q,
        base_policy_assert=args.assert_base, log_path=args.log,
        residual_scale=args.residual_scale,
    )
    state.started = time.time()

    if args.host not in ("127.0.0.1", "localhost"):
        logger.warning("binding to %s exposes an endpoint that MOVES A ROBOT", args.host)

    srv = ThreadingHTTPServer((args.host, args.port), _handler_cls())
    logger.info("residual policy server on http://%s:%d  (base: %s)",
                args.host, args.port, args.base_url)
    logger.info("  checkpoint: %s", args.checkpoint)
    logger.info("  residual_scale=%s collect_q=%s allow_lossy=%s",
                args.residual_scale, not args.no_q, args.allow_lossy)
    if not args.allow_lossy:
        logger.info("  jpeg input will be REFUSED - send image_encoding='zlib' or 'raw'")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        srv.server_close()
        if state.client is not None:
            state.client.close()


if __name__ == "__main__":
    main()
