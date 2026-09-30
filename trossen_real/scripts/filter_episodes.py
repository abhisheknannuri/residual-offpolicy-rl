# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Filter "bad"/aborted episodes out of a v2.1 dataset collected via the
infer app (`trossen_real/infer_app/`) or teleop app with pedal-driven
episode-boundary control enabled, before merging it into a clean training
set (e.g. via `merge_v21_datasets.py`, the RECAP-paper-style iterative
training workflow this repo uses).

Classification rule (confirmed 2026-09-01 - see
`trossen_real/human_intervention/episode_boundary.py`/`PEDAL_BEHAVIOR.md`
for what these pedals actually mean, and `infer_loop.py`/`control_loop.py`/
`dataset_recorder.py` for how `next.done`/`next.reward` actually get
written now):

  DISCARD an episode if EITHER of these holds:
    (a) `next.reward` is never 1 anywhere in the episode - the reward pedal
        was never pressed at all (episode ended by reset pedal, timeout, or
        a manual stop with no success ever marked).
    (b) `next.done` is True on EXACTLY one frame (the last one) and that
        frame's `next.reward` is not 1 - the reset pedal was pressed
        (`next.done` gets forced True on the terminal frame regardless of
        outcome - see `DatasetRecorder.stop_recording()` - but the reward
        pedal was never held, so this episode was intentionally aborted).
  KEEP otherwise - i.e. the reward pedal was held at some point (marking
  success), which (per the current, correct recording behavior) also means
  `next.done` is True on every one of those same frames, not just the very
  last one.

  Under the CURRENT (fixed 2026-09-01) recording behavior these two
  conditions mostly overlap - (a) alone already covers the reset-pedal case,
  since a reset-pedal-only episode never sees reward=1 anywhere. (b) is kept
  as an explicit, separate check anyway (not silently folded into (a)) so
  this still correctly classifies data from OLDER labeling conventions where
  next.done might not perfectly track next.reward frame-for-frame.

Data collected BEFORE the 2026-09-01 fix may have `next.done` missing
entirely or not matching this convention - such an episode is kept with a
note rather than silently guessed at (see `next.reward`-only check in (a),
which still works correctly regardless of `next.done`'s state, since it
never looks at `next.done` when reward WAS marked).

This script does not reimplement the actual v2.1 reindexing/video-copying/
metadata-rewriting logic - it classifies episodes, then delegates the
"write a new v2.1 dataset containing exactly these episode indices,
renumbered" part to `merge_v21_datasets.py` (called with a single
--dataset PATH::EP1,EP2,... spec - a single dataset with an episode
selection is a fully supported, tested use of that script, not a hack).
One source of truth for the tricky reindexing bookkeeping, instead of two
scripts that could drift apart.

Usage:
    .venv/bin/python -m trossen_real.scripts.filter_episodes \\
        --dataset /path/to/infer_dataset/some_session \\
        --output /path/to/infer_dataset/some_session_filtered

    # See what would happen without writing anything:
    .venv/bin/python -m trossen_real.scripts.filter_episodes \\
        --dataset /path/to/some_session --output /tmp/unused --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The N-way-merge-capable version of this script (see merge_v21_datasets.py's
# own module docstring for the "2 or more --dataset flags, each optionally
# suffixed with ::EP1,EP2,..." interface this script relies on).
DEFAULT_MERGE_SCRIPT = Path(
    "/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/"
    "DatasetUtil/tools/merge_v21_datasets.py"
)


def _find_episode_files(dataset_dir: Path) -> list[tuple[int, Path]]:
    """[(episode_index, parquet_path), ...], sorted by episode_index."""
    out = []
    for parquet in (dataset_dir / "data").rglob("episode_*.parquet"):
        ep_idx = int(parquet.stem.replace("episode_", ""))
        out.append((ep_idx, parquet))
    return sorted(out)


def classify_episodes(dataset_dir: Path) -> list[dict]:
    """Returns one dict per episode: {episode_index, n_frames, any_reward,
    done_true_count, last_reward, keep, note}, sorted by episode_index."""
    episodes = _find_episode_files(dataset_dir)
    if not episodes:
        raise RuntimeError(f"No episode_*.parquet files found under {dataset_dir}/data/")

    results = []
    for ep_idx, path in episodes:
        df = pd.read_parquet(path, columns=None)
        n_frames = len(df)

        if "next.reward" not in df.columns:
            # Not a pedal-reward-labeled dataset at all - nothing to filter on,
            # keep it and say so plainly rather than silently guessing.
            results.append({
                "episode_index": ep_idx, "n_frames": n_frames, "any_reward": None,
                "done_true_count": None, "last_reward": None, "keep": True,
                "note": "no next.reward column - not pedal-labeled, keeping by default",
            })
            continue

        reward = df["next.reward"].to_numpy()
        any_reward = bool((reward == 1).any())
        last_reward = float(reward[-1])

        done_true_count = None
        note = ""
        if "next.done" in df.columns:
            done = np.asarray(df["next.done"].tolist()).reshape(n_frames, -1).any(axis=1)
            done_true_idx = np.flatnonzero(done)
            done_true_count = int(len(done_true_idx))
            reset_pedal_pattern = (
                done_true_count == 1 and done_true_idx[0] == n_frames - 1 and last_reward != 1
            )
        else:
            reset_pedal_pattern = False
            note = "no next.done column - only checking next.reward"

        discard = (not any_reward) or reset_pedal_pattern
        if discard and not any_reward and reset_pedal_pattern:
            note = "reward pedal never held AND matches reset-pedal terminal pattern"
        elif discard and not any_reward:
            note = "reward pedal never held anywhere in this episode"
        elif discard and reset_pedal_pattern:
            note = "next.done True only on last frame with next.reward!=1 there - reset-pedal abort pattern"

        results.append({
            "episode_index": ep_idx, "n_frames": n_frames, "any_reward": any_reward,
            "done_true_count": done_true_count, "last_reward": last_reward,
            "keep": not discard, "note": note,
        })
    return results


def print_report(results: list[dict]) -> None:
    kept = [r for r in results if r["keep"]]
    discarded = [r for r in results if not r["keep"]]
    print(f"{'ep':>5}  {'frames':>7}  {'any_reward':>10}  {'done_true_n':>11}  {'last_reward':>11}  decision  note")
    for r in results:
        decision = "KEEP" if r["keep"] else "DISCARD"
        print(
            f"{r['episode_index']:>5}  {r['n_frames']:>7}  {str(r['any_reward']):>10}  "
            f"{str(r['done_true_count']):>11}  {str(r['last_reward']):>11}  {decision:<8}  {r['note']}"
        )
    print()
    print(
        f"{len(kept)} kept, {len(discarded)} discarded, {len(results)} total episodes "
        f"({sum(r['n_frames'] for r in kept)} frames kept, "
        f"{sum(r['n_frames'] for r in discarded)} frames discarded)"
    )
    if discarded:
        print(f"Discarded episode indices: {[r['episode_index'] for r in discarded]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="v2.1 dataset directory to filter.")
    parser.add_argument("--output", required=True, help="Output directory for the filtered, re-indexed dataset.")
    parser.add_argument(
        "--merge-script", default=str(DEFAULT_MERGE_SCRIPT),
        help=f"Path to merge_v21_datasets.py (default: {DEFAULT_MERGE_SCRIPT}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print the classification, write nothing.")
    parser.add_argument(
        "--exclude", default=None,
        help="Comma-separated episode indices to force-DISCARD regardless of the automatic "
             "classification above (e.g. an episode where the reward pedal was genuinely held "
             "but the task actually failed/was messed up - not detectable from next.reward/"
             "next.done alone, confirmed by manually reviewing the episode's video). Has no "
             "effect on episodes the classifier already discards on its own.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    results = classify_episodes(dataset_dir)

    exclude_set = {int(e) for e in args.exclude.split(",") if e.strip()} if args.exclude else set()
    for r in results:
        if r["episode_index"] in exclude_set and r["keep"]:
            r["keep"] = False
            r["note"] = f"MANUAL OVERRIDE (--exclude) - was otherwise: {r['note'] or 'KEEP'}"

    print_report(results)

    kept_indices = [r["episode_index"] for r in results if r["keep"]]
    if not kept_indices:
        print("\nERROR: every episode was discarded - nothing to write. Not calling the merge script.")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: not writing anything.")
        return

    merge_script = Path(args.merge_script)
    if not merge_script.exists():
        print(f"\nERROR: merge script not found at {merge_script} (pass --merge-script to point at it).")
        sys.exit(1)

    ep_spec = f"{dataset_dir}::{','.join(str(i) for i in kept_indices)}"
    cmd = [sys.executable, str(merge_script), "--dataset", ep_spec, "--output", args.output]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
