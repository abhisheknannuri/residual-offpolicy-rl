# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Standardized logging setup, shared by every app in this repo (infer app,
teleop app, follower server, online RL training - anything that already
calls `logging.getLogger(__name__)` and `logger.info(...)`/`logger.warning(
...)` etc., which is essentially every module here).

The problem this solves: every app's console output (`logger.info(...)`
etc.) has only ever gone to the terminal - nothing was ever written to disk
automatically. Debugging something that happened 20 minutes ago in a
terminal you've since scrolled past (or that crashed and closed) meant the
information was just gone.

The fix: ONE function, `setup_logging(app_name)`, called once near the top
of each app's entrypoint. It attaches a `FileHandler` to the ROOT logger, so
EVERY existing `logging.getLogger(__name__)` instance across the whole
codebase (there's no way to enumerate/change every individual call site,
nor should there be) gets captured automatically via normal Python logging
propagation - no per-call-site changes anywhere else. Console output keeps
working exactly as before (same format), nothing currently visible in your
terminal disappears.

Folder layout (LOCAL machine time, deliberately not UTC - easier to
eyeball against wall-clock/your own memory of when something happened):

    <repo_root>/logs/<app_name>/<YYYY-MM-DD>/<HH-MM-SS>/app.log
    <repo_root>/logs/<app_name>/<YYYY-MM-DD>/<HH-MM-SS>/console_output.log

`app.log` has only formatted `logging` output. `console_output.log` (only
written when `capture_print=True`, the default - see that parameter below)
is a complete transcript of everything that appeared in the terminal,
`logging` lines AND raw `print()` output included, in the order it happened.

`<repo_root>` is resolved from THIS FILE's own location on disk
(`trossen_real/log_setup.py`'s parent's parent), never from the current
working directory - so it doesn't matter which directory you actually ran
your command from, every app's logs land under the same
`<repo_root>/logs/` regardless.

This is INTENTIONALLY separate from, and does not touch, the existing
per-tick diagnostic JSONL logs (`infer_app`'s `TickLogger` /
`--log-dir logs` default, `REAL_LOG_FILE` in the RL training `.sh`
scripts) - those already write exactly what a caller configures them to,
under paths the caller controls; this module only concerns itself with
ordinary Python `logging` output, which previously went nowhere but the
terminal.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# trossen_real/log_setup.py -> parent is trossen_real/, parent.parent is the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO_ROOT / "logs"

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class _Tee:
    """File-like object duplicating writes to two underlying streams -
    used to mirror sys.stdout/sys.stderr into a plain text file IN ADDITION
    to writing normally, so raw `print()` calls (this app's own, or any
    library's - Flask/Werkzeug's request log, a stray warnings.warn(), etc.)
    get captured too, without touching a single print() call site anywhere.

    Deliberately transparent for anything that introspects the stream
    (isatty/fileno/encoding all delegate to the REAL first stream, not this
    wrapper) - code that branches on "is this a real terminal" (e.g. a
    progress-bar library deciding whether to use carriage-return in-place
    updates) sees exactly what it would without this wrapper in place.
    """

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            n = s.write(data)
        return n

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty()

    def fileno(self) -> int:
        return self._streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", "utf-8")


def setup_logging(
    app_name: str, level: int = logging.INFO, also_console: bool = True, capture_print: bool = True
) -> Path:
    """Call once, as early as possible in an app's entrypoint (before most
    other imports/setup happens, so as much as possible gets captured) -
    safe to call after some earlier `logging.basicConfig(...)` already ran
    (this replaces any bare `StreamHandler` that installed with a
    same-format one of its own, rather than doubling up console output).

    `app_name`: a short, stable label for this app - e.g. "infer_app",
    "teleop", "follower_server", "online_rl_real". Becomes the first path
    component under `logs/`, so pick something that reads well as a folder
    name and stays consistent across runs of the same app (so you can
    `ls logs/infer_app/` and see a clean history).

    `capture_print`: also redirects `sys.stdout`/`sys.stderr` so raw
    `print()` calls (not just `logging` output) land in `console_output.log`
    in the same run folder - see `_Tee` above for why this is safe to leave
    on by default. Set to False for a script where you specifically want
    ZERO risk of any stdout/stderr behavior change - e.g.
    `train_residual_td3.py` leaves this off: it's the one script in this
    repo relying heavily on tqdm progress bars across many print()-heavy
    loops, and while `_Tee` is designed to be transparent, that script's
    "never touch anything you don't have to" bar is deliberately higher
    than everywhere else. For that script, only `logging` output (which is
    substantial - print()-heavy files still emit real logger.info/warning/
    error calls at the important events) is captured via the FileHandler
    above regardless of this flag.

    Returns the run's own log directory - callers that ALSO write their own
    per-run artifacts (e.g. a summary file, a copy of the config used) are
    welcome to put them alongside `app.log` in this same folder, though
    that's entirely optional; this function's own job is only to make sure
    logging (and, unless disabled, print()) output is captured,
    unconditionally, everywhere.
    """
    now = datetime.now()  # local machine time, deliberately not UTC.
    run_dir = LOG_ROOT / app_name / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # capture_print BEFORE building the console logging handler below, so
    # that handler picks up the already-Tee'd sys.stdout - console_output.log
    # then ends up a complete transcript (formatted log lines AND raw
    # print() output, in order), not just the print() half of it.
    if capture_print:
        console_log_path = run_dir / "console_output.log"
        console_log_file = open(console_log_path, "a")  # noqa: SIM115 - intentionally kept open for process lifetime
        if not isinstance(sys.stdout, _Tee):
            sys.stdout = _Tee(sys.stdout, console_log_file)
        if not isinstance(sys.stderr, _Tee):
            sys.stderr = _Tee(sys.stderr, console_log_file)

    fmt = logging.Formatter(_LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)

    log_file = run_dir / "app.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if also_console:
        # Drop any plain StreamHandler a prior logging.basicConfig(...) call
        # already installed on the root logger (avoids duplicate console
        # lines) and replace it with one in the same format, so console
        # output is unaffected either way.
        for h in list(root.handlers):
            if type(h) is logging.StreamHandler and h is not file_handler:
                root.removeHandler(h)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)

    logging.getLogger(__name__).info(
        "Logging to %s (console output: %s, raw print() also captured: %s)", log_file, also_console, capture_print
    )
    return run_dir
