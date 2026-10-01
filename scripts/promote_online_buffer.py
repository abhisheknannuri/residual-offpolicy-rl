# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Carry an online-RL session's collected experience forward into the next run.

THE PROBLEM
-----------
`train_residual_td3.py` keeps ONE online replay buffer in memory for the whole
run. It is never cleared. But it writes that buffer to TWO different places,
in two different phases:

    warm-up collection   -> online_buffer_cache/<hash>/       (lines 1493, 1500)
    online-RL training   -> run_<...>/online_buffer_final/    (lines 2187, 2440)

and on startup it only ever LOADS `online_buffer_cache/<hash>/` (line 881).

Both warm-up writes sit inside `if len(online_rb) < learning_starts:`, which
stops running the moment warm-up finishes. So everything online RL collects
after that lands only in the run folder, and the next run does not see it - it
loads the warm-up-sized buffer and the session's experience sits unused on disk.

This script promotes a run's buffer into the hashed cache so the next run
resumes from everything.

WHY THIS REPLACES RATHER THAN APPENDS
-------------------------------------
"Merge the extra data in" sounds right but would corrupt the buffer, for two
independent reasons:

1. The run's dump is NOT a delta. It is the same in-memory buffer, which was
   loaded FROM the cache at startup and then extended - so it already contains
   everything the cache has. Appending would store the warm-up transitions twice
   and silently bias sampling toward them.
2. The data on disk is already POST-`MultiStepTransform`. That transform is an
   *inverse* (write-time) transform, so pushing dumped rows back through
   `extend()` would apply the n-step collapse a second time. See
   BUFFER_POPULATION_ARCHITECTURE.md.

So the correct operation is: replace the cache with the run's (larger) buffer.

THE BUFFER IS CIRCULAR - READ THIS BEFORE A LONG SESSION
--------------------------------------------------------
The storage is `LazyTensorStorage(max_size=algo.buffer_size)`, which is circular.
Once the buffer holds `algo.buffer_size` transitions (70,000 in the Station-1
configs), every new transition **overwrites the oldest one**.

The warm-up / population data is written FIRST, so it is evicted FIRST. After an
online-RL session long enough to push the buffer past `buffer_size`, the
original population transitions are already gone from the in-memory buffer - and
therefore from `online_buffer_final` too. Promoting then makes the cache match
what training actually had, which also means the cache no longer holds that
original data.

This is the circular buffer doing what it was configured to do, not something
this script causes - the eviction already happened in memory, during training.
But it has two practical consequences:

  * Do not treat the promoted cache as an archive of everything ever collected.
    It is a window of the most recent `buffer_size` transitions.
  * If you want to keep the original population buffer, keep the
    `<dest>.bak-<timestamp>` folder this script leaves behind (it is the
    pre-promotion cache), or raise `algo.buffer_size` - though note that
    `buffer_size` is itself in the online cache hash, so changing it starts a
    new, empty cache.

Below `buffer_size` nothing is evicted and the run's buffer is a strict superset
of the cache, so promoting is lossless.

USAGE
-----
    # see what would happen (default - nothing is written)
    .venv/bin/python scripts/promote_online_buffer.py \
        --from run_2026-10-01_.../online_buffer_final --to c3aeae2b

    # do it (the existing cache is moved aside, never deleted)
    .venv/bin/python scripts/promote_online_buffer.py \
        --from run_2026-10-01_... --to c3aeae2b --apply

`--from` accepts either the buffer directory itself or the run folder (in which
case `online_buffer_final` is preferred, falling back to `online_buffer_latest`).
`--to` accepts either the 8-character hash or a full path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ONLINE_CACHE_ROOT = REPO_ROOT / "online_buffer_cache"

# Files torchrl's ReplayBuffer.dumps() always writes. Used to tell a real buffer
# dump from a directory that merely looks like one.
REQUIRED = ("storage", "writer", "sampler", "buffer_metadata.json")


def _resolve_source(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if (p / "storage").is_dir():
        return p
    # A run folder: prefer the final dump, fall back to the periodic one.
    for name in ("online_buffer_final", "online_buffer_latest"):
        if (p / name / "storage").is_dir():
            return p / name
    raise SystemExit(
        f"ERROR: no buffer dump found at {p}\n"
        f"       expected either a dump directory (containing storage/) or a run\n"
        f"       folder containing online_buffer_final/ or online_buffer_latest/"
    )


def _resolve_dest(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute() and not p.exists():
        # Bare hash, e.g. "c3aeae2b"
        candidate = ONLINE_CACHE_ROOT / raw
        if candidate.exists():
            return candidate
        p = REPO_ROOT / raw
    return p


def _check_dump(p: Path, what: str) -> None:
    missing = [f for f in REQUIRED if not (p / f).exists()]
    if missing:
        raise SystemExit(f"ERROR: {what} at {p} is not a replay-buffer dump (missing: {missing})")


def _buffer_len(p: Path) -> tuple[int, int, int | None]:
    """(valid transitions, write cursor, max_size) read from the dump's metadata."""
    sm = json.loads((p / "storage" / "storage_metadata.json").read_text())
    wm = json.loads((p / "writer" / "metadata.json").read_text())
    max_size = None
    meta = p / "storage" / "meta.json"
    if meta.exists():
        fields = json.loads(meta.read_text())
        for v in fields.values():
            if isinstance(v, dict) and "shape" in v and v["shape"]:
                max_size = v["shape"][0]
                break
    return int(sm.get("len", -1)), int(wm.get("cursor", -1)), max_size


def _dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True,
                    help="run folder, or the buffer dump directory inside it")
    ap.add_argument("--to", dest="dst", required=True,
                    help="online_buffer_cache hash (e.g. c3aeae2b) or a full path")
    ap.add_argument("--apply", action="store_true",
                    help="actually do it (default: dry run, nothing is written)")
    ap.add_argument("--force", action="store_true",
                    help="proceed even when the source has FEWER transitions than the "
                         "destination (which would lose data)")
    args = ap.parse_args(argv)

    src = _resolve_source(args.src)
    dst = _resolve_dest(args.dst)

    _check_dump(src, "source")
    if not dst.exists():
        raise SystemExit(
            f"ERROR: destination {dst} does not exist.\n"
            f"       Promoting into a cache that was never created would give the next run\n"
            f"       a buffer with no user_metadata.json describing its config hash.\n"
            f"       Run the normal warm-up once first, or pass the correct hash.\n"
            f"       Available: {sorted(p.name for p in ONLINE_CACHE_ROOT.glob('*')) if ONLINE_CACHE_ROOT.exists() else 'none'}"
        )
    _check_dump(dst, "destination")

    s_len, s_cur, s_max = _buffer_len(src)
    d_len, d_cur, d_max = _buffer_len(dst)

    print(f"source      {src}")
    print(f"            {s_len:,} transitions (cursor {s_cur:,}, max_size {s_max:,})"
          if s_max else f"            {s_len:,} transitions (cursor {s_cur:,})")
    print(f"            {_dir_size_gb(src):.2f} GB")
    print(f"destination {dst}")
    print(f"            {d_len:,} transitions (cursor {d_cur:,}, max_size {d_max:,})"
          if d_max else f"            {d_len:,} transitions (cursor {d_cur:,})")
    print(f"            {_dir_size_gb(dst):.2f} GB")
    print()

    if s_max is not None and d_max is not None and s_max != d_max:
        print(f"WARNING: different max_size ({s_max:,} vs {d_max:,}) - these buffers were built")
        print("         with different algo.buffer_size, so they are NOT the same cache entry.")
        print("         Promoting across them will make the next run's hash not match. Check --to.")
        print()

    if s_len < d_len and not args.force:
        raise SystemExit(
            f"REFUSING: the source has FEWER transitions ({s_len:,}) than the destination "
            f"({d_len:,}).\n"
            f"          Replacing would lose {d_len - s_len:,} transitions. This usually means\n"
            f"          --from points at the wrong run, or at online_buffer_latest from early\n"
            f"          in a run. Pass --force if you are sure."
        )

    gained = s_len - d_len
    print(f"=> destination would go {d_len:,} -> {s_len:,} transitions "
          f"({gained:+,})")

    if not args.apply:
        print("\n(dry run - nothing written. Re-run with --apply.)")
        return 0

    # The cache's user_metadata.json records the config hash this directory
    # belongs to. The run dump has no such file, so it must be carried over or
    # the next run cannot tell what this cache is.
    user_meta = dst / "user_metadata.json"
    saved_meta = user_meta.read_text() if user_meta.exists() else None
    if saved_meta is None:
        print("WARNING: destination has no user_metadata.json; the promoted cache will have none.")

    backup = dst.with_name(dst.name + f".bak-{datetime.now():%Y%m%d_%H%M%S}")
    print(f"\nmoving existing cache aside -> {backup}")
    shutil.move(str(dst), str(backup))
    try:
        print(f"copying {src} -> {dst}  ({_dir_size_gb(src):.2f} GB, this can take a while)")
        shutil.copytree(src, dst)
        if saved_meta is not None:
            (dst / "user_metadata.json").write_text(saved_meta)
            print("restored the cache's user_metadata.json (its config-hash record)")
    except Exception:
        print("\nCOPY FAILED - restoring the original cache", file=sys.stderr)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(backup), str(dst))
        raise

    n_len, _, _ = _buffer_len(dst)
    print(f"\ndone: {dst} now holds {n_len:,} transitions")
    print(f"the previous cache is kept at {backup} - delete it once the next run has loaded cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
