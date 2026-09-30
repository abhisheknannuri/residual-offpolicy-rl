# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""One-off retroactive fix for v2.1 datasets collected BEFORE the 2026-09-01
`next.done`/`next.reward` sync fix (see `infer_loop.py`/`control_loop.py`/
`dataset_recorder.py`): widens `next.done` to match `next.reward` wherever
the reward pedal was actually held, WITHOUT touching `next.reward` itself
and WITHOUT ever un-setting an existing `next.done=True`.

Exact transform, per frame: `next.done = next.done OR (next.reward == 1)`.
One direction only - this never infers/changes `next.reward` from
`next.done`, and never removes a `next.done=True` that's already there
(e.g. one already correctly set by a reset-pedal/timeout/manual-stop
terminal frame).

Writes a NEW dataset (full copy of the input, then only the affected
parquet files are rewritten) - the original is never modified. Refuses to
run if the output path already exists, matching this repo's other dataset
tools (`merge_v21_datasets.py`, `filter_episodes.py`).

Does NOT recompute `meta/stats.json`/`meta/episodes_stats.jsonl` - those
carry per-column min/max/mean/std, which is meaningless for a boolean
column like `next.done` and isn't consumed by ACT/RL training either way
(next.done is never a model input); recomputing them for every other
column just to leave next.done's untouched wasn't worth the added
complexity for a data-correctness fix. Flagged here explicitly rather than
silently skipped.

Usage:
    .venv/bin/python -m trossen_real.scripts.fix_next_done_sync \\
        --input-root /path/to/some_session \\
        --output-root /path/to/some_session_donefixed

    # Preview what would change, without writing anything:
    .venv/bin/python -m trossen_real.scripts.fix_next_done_sync \\
        --input-root /path/to/some_session --output-root /tmp/unused --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _find_episode_files(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "data").rglob("episode_*.parquet"))


def _scalar(x):
    """Extract a plain Python/numpy scalar from a parquet cell, whether it's
    stored as a bare scalar OR wrapped in a length-1 list/array - checked
    directly against real data (NOT `meta/info.json`'s declared "shape",
    which says [1] for both next.done/next.reward but does NOT match what's
    actually in the parquet: confirmed both are bare scalars there, e.g.
    `np.False_`/`np.float32(0.0)`, not `array([False])`. info.json can't be
    trusted for this - it's static and untied to the real per-cell format."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x).reshape(-1)[0]
    return x


def _like(original_cell, value):
    """Repackage `value` (a plain bool) into WHATEVER container shape
    `original_cell` actually had - bare scalar, or a length-1 list/array -
    so the column's per-cell representation/dtype never changes, only the
    boolean value does. This is what the previous version of this function
    got wrong: it unconditionally reshaped into (n, 1) arrays and wrote
    back a list of length-1 arrays, silently turning a plain `dtype=bool`
    column into an `dtype=object` column of arrays - confirmed by directly
    inspecting a "_donefixed" output's parquet (not by trusting info.json,
    which still claimed shape=[1]/dtype=bool the whole time, unaffected by
    the actual corruption)."""
    if isinstance(original_cell, np.ndarray):
        return np.array([value], dtype=original_cell.dtype)
    if isinstance(original_cell, list):
        return [value]
    if isinstance(original_cell, tuple):
        return (value,)
    return bool(value)


def fix_dataset(input_root: Path, output_root: Path | None, dry_run: bool) -> list[dict]:
    """Returns one dict per episode: {path, n_frames, n_changed}."""
    episode_files = _find_episode_files(output_root if not dry_run else input_root)
    if not episode_files:
        raise RuntimeError(f"No episode_*.parquet files found under "
                            f"{(output_root if not dry_run else input_root)}/data/")

    reports = []
    for path in episode_files:
        df = pd.read_parquet(path)
        if "next.reward" not in df.columns or "next.done" not in df.columns:
            reports.append({"path": path, "n_frames": len(df), "n_changed": 0,
                             "note": "missing next.reward/next.done - skipped"})
            continue

        reward_col = df["next.reward"].tolist()
        done_col = df["next.done"].tolist()
        new_done_col = []
        n_changed = 0
        for reward_cell, done_cell in zip(reward_col, done_col):
            reward_val = _scalar(reward_cell)
            done_val = bool(_scalar(done_cell))
            new_val = done_val or (reward_val == 1)
            if new_val != done_val:
                n_changed += 1
            new_done_col.append(_like(done_cell, new_val))

        if n_changed > 0 and not dry_run:
            df["next.done"] = new_done_col
            df.to_parquet(path, index=False)

        reports.append({"path": path, "n_frames": len(df), "n_changed": n_changed, "note": ""})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", required=True, help="v2.1 dataset to fix (never modified).")
    parser.add_argument("--output-root", required=True, help="Output path for the corrected copy.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would change (reading --input-root directly), write nothing.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if args.dry_run:
        reports = fix_dataset(input_root, None, dry_run=True)
    else:
        if output_root.exists():
            raise FileExistsError(f"--output-root {output_root} already exists - refusing to overwrite.")
        print(f"Copying {input_root} -> {output_root} ...")
        shutil.copytree(input_root, output_root)
        reports = fix_dataset(input_root, output_root, dry_run=False)

    total_frames = sum(r["n_frames"] for r in reports)
    total_changed = sum(r["n_changed"] for r in reports)
    for r in reports:
        if r["n_changed"] > 0 or r["note"]:
            print(f"  {r['path'].name}: {r['n_frames']} frames, {r['n_changed']} next.done flipped to True"
                  + (f" ({r['note']})" if r["note"] else ""))
    print(f"\n{len(reports)} episodes, {total_frames} frames total, {total_changed} frames had next.done "
          f"flipped False->True (next.reward was 1, next.done wasn't set yet).")
    if args.dry_run:
        print("--dry-run: nothing written.")
    else:
        print(f"Corrected copy written to {output_root}")


if __name__ == "__main__":
    main()
