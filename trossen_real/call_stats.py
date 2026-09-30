# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Lightweight, dependency-free, real-time call tracking for the two shared
HTTP clients every app in this repo talks through:
`trossen_real.teleop.follower_client.FollowerClient` (-> `follower_single_server.py`)
and `trossen_real.inference.policy_client.PolicyClient` (-> `policy_server.py`).

Both classes are constructed identically by the teleop app, the infer web
app, `scripts/live_infer_deploy.py`, AND residual RL training
(`train_residual_td3.py` / `trossen_real/rl/real_residual_env.py`) - see each
file's own imports. Instrumenting the two client classes ONCE, here, means
every one of those callers gets live call/latency tracking for free, with
zero changes to any of their own code.

Tracked per (client, endpoint) pair: calls attempted, calls that completed
successfully (2xx), calls that failed (broken out by reason: timeout /
connection_error / http_error:<code> / other:<ExceptionClassName>), and a
bounded rolling window of latencies (wall-clock seconds from just before the
request is sent to just after it returns or raises) for a live avg/p50/p95/max.

This is deliberately NOT a monitoring framework - just a couple of dicts
behind one lock, updated with plain arithmetic on the hot path (no I/O). The
only I/O this module ever does is the optional background printer thread
below, and even that is a single `logging` call every few seconds (NOT a
raw `print()` - see `_logger` below - so it goes through the SAME
`logging.getLogger(__name__)` pipeline as everything else, which means it's
captured into `logs/<app_name>/<date>/<time>/app.log` too, once an app has
called `trossen_real.log_setup.setup_logging(...)` - see that module).

Usage (automatic - nothing to import elsewhere):
    Set the environment variable TROSSEN_CALL_STATS=1 (default: ON - see
    _DEFAULT_ENABLED below) before launching teleop/app.py, infer_app/app.py,
    scripts/live_infer_deploy.py, or train_residual_rl_real.sh. A one-block
    summary logs every TROSSEN_CALL_STATS_INTERVAL_S seconds (default 5) -
    console AND (if the app called setup_logging()) its log file:

        [call_stats] 14:32:07
          follower:5095   /getstate                    calls=812   ok=812  avg=3ms p50=2ms p95=6ms max=14ms
          follower:5095   /move_to_joint_positions      calls=812   ok=810  FAILED=2 (timeout=2)  avg=4ms p50=3ms p95=9ms max=210ms
          policy:5090     /predict                      calls=812   ok=812  avg=41ms p50=38ms p95=71ms max=340ms

    Disable entirely with TROSSEN_CALL_STATS=0. To read it programmatically
    instead of printing (e.g. to surface in the infer app's /api/health),
    call `get_registry().snapshot()`.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import threading
import time
from collections.abc import Iterator

import requests

_DEFAULT_ENABLED = True  # explicit user request: visible "every time", no per-app opt-in needed
_DEFAULT_INTERVAL_S = 5.0

# Own module logger, forced to INFO regardless of whatever level an app's
# setup_logging(..., level=...) call chose for the root logger (e.g.
# train_residual_td3.py uses level=logging.WARNING to keep its console
# quiet) - call-stats summaries should always show up (in the console AND
# in logs/<app>/.../app.log, once setup_logging() has been called - see
# trossen_real/log_setup.py), never silently suppressed by an app's own
# unrelated verbosity choice. A logger's OWN explicit level (set here) takes
# precedence over its ancestors' (root's) level when Python logging decides
# whether to even build/emit a record - the root's HANDLERS still receive
# and process it normally either way.
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_LATENCY_WINDOW = 200  # rolling window size per endpoint, plenty for live avg/p50/p95/max


class _EndpointStats:
    __slots__ = ("attempted", "ok", "failed_by_reason", "latencies")

    def __init__(self) -> None:
        self.attempted = 0
        self.ok = 0
        self.failed_by_reason: dict[str, int] = collections.defaultdict(int)
        self.latencies: collections.deque[float] = collections.deque(maxlen=_LATENCY_WINDOW)


class CallStatsRegistry:
    """Process-wide singleton (see `get_registry()`) - every `PolicyClient`/
    `FollowerClient` instance in ONE process reports into the same place,
    regardless of which app constructed it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[tuple[str, str], _EndpointStats] = {}
        self._printer_thread: threading.Thread | None = None
        self._stop_printer = threading.Event()

    def record(self, client: str, endpoint: str, latency_s: float, outcome: str) -> None:
        """`outcome`: "ok", or a failure-reason string such as "timeout",
        "connection_error", "http_error:500", "other:JSONDecodeError"."""
        key = (client, endpoint)
        with self._lock:
            stats = self._stats.get(key)
            if stats is None:
                stats = self._stats[key] = _EndpointStats()
            stats.attempted += 1
            stats.latencies.append(latency_s)
            if outcome == "ok":
                stats.ok += 1
            else:
                stats.failed_by_reason[outcome] += 1

    def snapshot(self) -> dict[tuple[str, str], dict]:
        """Point-in-time copy, safe to hold onto / serialize (e.g. for an
        /api/health endpoint) without touching the live registry again."""
        with self._lock:
            out: dict[tuple[str, str], dict] = {}
            for key, s in self._stats.items():
                lat = sorted(s.latencies)
                n = len(lat)
                out[key] = {
                    "attempted": s.attempted,
                    "ok": s.ok,
                    "failed": s.attempted - s.ok,
                    "failed_by_reason": dict(s.failed_by_reason),
                    "avg_ms": (sum(lat) / n * 1000.0) if n else None,
                    "p50_ms": (lat[n // 2] * 1000.0) if n else None,
                    "p95_ms": (lat[int(n * 0.95)] * 1000.0) if n else None,
                    "max_ms": (lat[-1] * 1000.0) if n else None,
                    "window": n,
                }
            return out

    def format_snapshot(self) -> str:
        snap = self.snapshot()
        if not snap:
            return "[call_stats] no calls recorded yet"
        lines = [f"[call_stats] {time.strftime('%H:%M:%S')}"]
        for (client, endpoint), s in sorted(snap.items()):
            fail_str = ""
            if s["failed"]:
                reasons = ", ".join(f"{k}={v}" for k, v in s["failed_by_reason"].items())
                fail_str = f"  FAILED={s['failed']} ({reasons})"
            lat_str = ""
            if s["avg_ms"] is not None:
                lat_str = (
                    f"  avg={s['avg_ms']:.0f}ms p50={s['p50_ms']:.0f}ms "
                    f"p95={s['p95_ms']:.0f}ms max={s['max_ms']:.0f}ms"
                )
            lines.append(
                f"  {client:<15} {endpoint:<28} calls={s['attempted']:<6} ok={s['ok']}{fail_str}{lat_str}"
            )
        return "\n".join(lines)

    def start_background_printer(self, interval_s: float = _DEFAULT_INTERVAL_S) -> None:
        """Idempotent - a second call is a no-op if a printer is already
        running, so constructing several PolicyClient/FollowerClient
        instances in the same process never spawns more than one thread."""
        if self._printer_thread is not None:
            return

        def _loop() -> None:
            while not self._stop_printer.wait(interval_s):
                _logger.info(self.format_snapshot())

        self._printer_thread = threading.Thread(target=_loop, daemon=True, name="call-stats-printer")
        self._printer_thread.start()

    def stop_background_printer(self) -> None:
        if self._printer_thread is not None:
            self._stop_printer.set()
            self._printer_thread.join(timeout=2.0)
            self._printer_thread = None
            self._stop_printer.clear()


_registry = CallStatsRegistry()


def get_registry() -> CallStatsRegistry:
    return _registry


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "")


@contextlib.contextmanager
def track_call(client: str, endpoint: str) -> Iterator[None]:
    """Wrap one outgoing HTTP call: `with track_call("follower:5095", "/getstate"): ...`

    Classifies `requests` exceptions into stable reason buckets; anything not
    a `requests` exception (e.g. a `RuntimeError` raised by the caller after
    inspecting the JSON body, like `PolicyClient.predict_full()`'s
    `"error" in payload` check) is still counted as a failure, bucketed by
    its class name, and always re-raised unchanged - this module only
    observes calls, it never changes control-loop behavior or swallows
    anything.
    """
    start = time.perf_counter()
    try:
        yield
    except requests.exceptions.Timeout:
        _registry.record(client, endpoint, time.perf_counter() - start, "timeout")
        raise
    except requests.exceptions.ConnectionError:
        _registry.record(client, endpoint, time.perf_counter() - start, "connection_error")
        raise
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        _registry.record(client, endpoint, time.perf_counter() - start, f"http_error:{code}")
        raise
    except Exception as exc:
        _registry.record(client, endpoint, time.perf_counter() - start, f"other:{type(exc).__name__}")
        raise
    else:
        _registry.record(client, endpoint, time.perf_counter() - start, "ok")


# Auto-start the background printer at import time unless explicitly disabled
# (TROSSEN_CALL_STATS=0) - this is the one thing that makes stats show up
# "every time" in whichever terminal is running teleop/infer_app/RL training,
# with nothing extra to remember to pass. Interval is overridable via
# TROSSEN_CALL_STATS_INTERVAL_S (seconds, default 5).
if _env_flag("TROSSEN_CALL_STATS", default=_DEFAULT_ENABLED):
    _registry.start_background_printer(
        interval_s=float(os.environ.get("TROSSEN_CALL_STATS_INTERVAL_S", str(_DEFAULT_INTERVAL_S)))
    )
