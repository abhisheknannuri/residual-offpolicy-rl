# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""One-off port of manually-annotated `next.reward`/`next.done` labels from a
TRIMMED delta-joint dataset back onto the UNTRIMMED absolute-joint dataset it
was derived from (`convert_to_delta_joint_dataset.py`).

Why this is needed: the delta-joint dataset had both start- and end-trim
applied (the old, since-disabled heuristic - see convert_to_delta_joint_dataset.py's
2026-09-01 comments), so its frame indices do NOT line up 1:1 with the
abs-joint source dataset's indices per episode, and no provenance file
survived for this particular dataset. So this script re-derives the mapping
from real data instead of trusting any index/order metadata:

  For each delta episode, `observation.state` is a byte-exact passthrough
  column (never recomputed - only `action` is transformed to a delta).
  Trimming only removes frames from the START and END of an episode, never
  the middle. So the delta episode's full `observation.state` sequence must
  appear as an EXACT contiguous block inside some abs-joint episode's
  `observation.state` sequence. Finding that block (by exact float32 array
  equality) recovers the (abs_episode_index, start_offset) mapping with no
  ambiguity - verified empirically: all 136/136 delta episodes matched a
  unique block, in their own same-numbered abs episode.

Verified empirically before writing anything (see investigation in the
2026-09-01 session):
  - ALL 136/136 episodes: the delta episode's FIRST frame has next.reward==0
    -> the start-trim never cut into an annotated reward-hold region, so the
       abs-joint frames BEFORE the matched block are left at the untouched
       default (reward=0, done=False) - no synthesized signal there.
  - ALL 136/136 episodes: the delta episode's LAST frame has next.reward==1
    -> the end-trim uniformly cut INTO an active reward-hold region (the
       annotator's labeling stopped only because the trimmed video ran out,
       not because the hold ended). So the abs-joint frames AFTER the
       matched block (which exist for real, untrimmed, in the abs dataset)
       get the hold carried forward: reward=1 AND done=True on every one of
       them, consistent with this dataset's already-established wide
       next.done invariant (next.done==True on every frame where
       next.reward==1, not just the last one - see fix_next_done_sync.py).

Per frame in the output:
  - index < offset               (front-trimmed, unannotated): reward=0, done=False (unchanged from source)
  - offset <= index < offset+n   (matched to delta frame j=index-offset): copied verbatim from delta[j]
  - index >= offset+n            (end-trimmed, hold carried forward): reward=1, done=True

Writes a NEW dataset (full copy of --abs-root, then only next.reward/next.done
columns are rewritten in the matched episodes) - the source abs-joint dataset
is never modified. Refuses to run if --output-root already exists.

Usage:
    .venv/bin/python -m trossen_real.scripts.annotate_reward_from_delta \\
        --abs-root trossen_real/datasets/PickAndInsertCube_Station1_merged_old \\
        --delta-root trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint_old \\
        --output-root trossen_real/datasets/PickAndInsertCube_Station1_merged_annotated

    # Preview the episode-index/offset mapping and per-episode counts only:
    .venv/bin/python -m trossen_real.scripts.annotate_reward_from_delta \\
        --abs-root ... --delta-root ... --output-root /tmp/unused --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _scalar(x):
    """Extract a plain scalar from a parquet cell regardless of whether it's
    stored as a bare scalar or wrapped in a length-1 list/array - checked
    against real data, never assumed from info.json (see fix_next_done_sync.py
    for why info.json's declared shape can't be trusted for this)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x).reshape(-1)[0]
    return x


def _like(original_cell, value):
    """Repackage `value` into whatever container shape `original_cell`
    actually had (bare scalar / length-1 list / length-1 ndarray / tuple),
    so the column's per-cell representation/dtype never changes - only the
    value does. Mirrors fix_next_done_sync.py's `_like()`."""
    if isinstance(original_cell, np.ndarray):
        return np.array([value], dtype=original_cell.dtype)
    if isinstance(original_cell, list):
        return [value]
    if isinstance(original_cell, tuple):
        return (value,)
    return type(original_cell)(value) if not isinstance(original_cell, bool) else bool(value)


def _find_episode_files(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "data").rglob("episode_*.parquet"))


def _episode_index_of(df: pd.DataFrame, path: Path) -> int:
    if "episode_index" in df.columns:
        return int(_scalar(df["episode_index"].iloc[0]))
    return int(path.stem.split("_")[-1])


def _state_block(df: pd.DataFrame) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df["observation.state"].tolist()])


def find_block_offset(abs_states: np.ndarray, delta_states: np.ndarray) -> int | None:
    """Return the start offset where abs_states[offset:offset+n] exactly
    equals delta_states, or None if no such offset exists."""
    n = len(delta_states)
    if len(abs_states) < n:
        return None
    for offset in range(0, len(abs_states) - n + 1):
        if np.array_equal(abs_states[offset:offset + n], delta_states):
            return offset
    return None


def build_mapping(abs_root: Path, delta_root: Path) -> dict[int, dict]:
    """Returns {abs_episode_index: {"delta_path": Path, "offset": int, "n": int}}."""
    abs_files = {_episode_index_of(pd.read_parquet(f, columns=["episode_index"]) if False else pd.read_parquet(f), f): f
                 for f in _find_episode_files(abs_root)}
    delta_files = {_episode_index_of(pd.read_parquet(f), f): f for f in _find_episode_files(delta_root)}

    abs_states_cache: dict[int, np.ndarray] = {}

    def get_abs_states(ep: int) -> np.ndarray:
        if ep not in abs_states_cache:
            abs_states_cache[ep] = _state_block(pd.read_parquet(abs_files[ep]))
        return abs_states_cache[ep]

    mapping = {}
    for d_ep, d_path in sorted(delta_files.items()):
        d_df = pd.read_parquet(d_path)
        d_states = _state_block(d_df)
        candidates = [d_ep] + [e for e in abs_files if e != d_ep]
        matched_ep = None
        offset = None
        for a_ep in candidates:
            offset = find_block_offset(get_abs_states(a_ep), d_states)
            if offset is not None:
                matched_ep = a_ep
                break
        if matched_ep is None:
            raise RuntimeError(f"delta episode {d_ep} ({d_path.name}) has no matching block in any abs episode - "
                                f"refusing to guess. Investigate before proceeding.")
        mapping[matched_ep] = {"delta_path": d_path, "offset": offset, "n": len(d_states)}
    return mapping


def ensure_reward_column(output_root: Path) -> bool:
    """The source `_old` dataset predates the reward-pedal pipeline entirely
    and has NO `next.reward` column in its schema (only `next.done`, always
    False). Add it: update `meta/info.json`'s features dict (mirroring the
    exact `{"dtype": "float32", "shape": [1], "names": None}` spec already
    used by the deltajoint dataset that HAS this column for real), then add
    a `next.reward` column (bare float32 scalar per cell, matching
    `next.done`'s existing bare-scalar convention in this dataset - checked
    directly against a real cell, not assumed) to every episode parquet,
    defaulted to 0.0. Returns True if the column was actually added (False
    if it already existed - script is then a no-op here, safe to re-run)."""
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    added = False
    if "next.reward" not in info["features"]:
        info["features"]["next.reward"] = {"dtype": "float32", "shape": [1], "names": None}
        info_path.write_text(json.dumps(info, indent=4))
        added = True

    for f in _find_episode_files(output_root):
        df = pd.read_parquet(f)
        if "next.reward" not in df.columns:
            df["next.reward"] = np.float32(0.0)
            df.to_parquet(f, index=False)
            added = True
    return added


def _episode_stats_block(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": [float(arr.min())], "max": [float(arr.max())],
        "mean": [float(arr.mean())], "std": [float(arr.std())],
        "count": [int(len(arr))],
    }


def recompute_reward_done_stats(output_root: Path) -> None:
    """Recompute `next.reward`/`next.done` stats (min/max/mean/std/count) in
    both `meta/episodes_stats.jsonl` (per-episode) and `meta/stats.json`
    (global) from the actual, now-annotated parquet data - mirrors the exact
    stats shape already used by every other boolean/float column in this
    dataset family (see PickAndInsertCube_Station1_merged_deltajoint_old for
    the reference format). next.done's stats also get refreshed since its
    values changed too (it was all-False -> all_zero stats before)."""
    ep_files_by_index: dict[int, Path] = {}
    for f in _find_episode_files(output_root):
        df = pd.read_parquet(f)
        ep_idx = _episode_index_of(df, f)
        ep_files_by_index[ep_idx] = f

    stats_path = output_root / "meta" / "episodes_stats.jsonl"
    lines = stats_path.read_text().splitlines()
    new_lines = []
    all_reward: list[float] = []
    all_done: list[float] = []
    for line in lines:
        rec = json.loads(line)
        ep_idx = int(rec["episode_index"])
        df = pd.read_parquet(ep_files_by_index[ep_idx])
        reward_vals = [float(_scalar(x)) for x in df["next.reward"].tolist()]
        done_vals = [float(bool(_scalar(x))) for x in df["next.done"].tolist()]
        rec["stats"]["next.reward"] = _episode_stats_block(reward_vals)
        done_block = _episode_stats_block(done_vals)
        done_block["min"] = [bool(done_block["min"][0])]
        done_block["max"] = [bool(done_block["max"][0])]
        rec["stats"]["next.done"] = done_block
        all_reward.extend(reward_vals)
        all_done.extend(done_vals)
        new_lines.append(json.dumps(rec))
    stats_path.write_text("\n".join(new_lines) + "\n")

    global_stats_path = output_root / "meta" / "stats.json"
    global_stats = json.loads(global_stats_path.read_text())
    global_stats["next.reward"] = _episode_stats_block(all_reward)
    done_block = _episode_stats_block(all_done)
    done_block["min"] = [bool(done_block["min"][0])]
    done_block["max"] = [bool(done_block["max"][0])]
    global_stats["next.done"] = done_block
    global_stats_path.write_text(json.dumps(global_stats, indent=4))


def annotate_episode(abs_path: Path, delta_path: Path, offset: int, n: int) -> dict:
    abs_df = pd.read_parquet(abs_path)
    delta_df = pd.read_parquet(delta_path)
    abs_len = len(abs_df)

    if "next.reward" not in abs_df.columns or "next.done" not in abs_df.columns:
        raise RuntimeError(f"{abs_path} missing next.reward/next.done columns - cannot annotate.")

    d_reward = [_scalar(x) for x in delta_df["next.reward"].tolist()]
    d_done = [_scalar(x) for x in delta_df["next.done"].tolist()]
    assert len(d_reward) == n, f"{delta_path}: expected {n} frames, got {len(d_reward)}"
    # Verified before calling this: d_reward[0] == 0.0 (front-trim safe) and
    # d_reward[-1] == 1.0 (end-trim cut into an active hold) for every episode.
    assert float(d_reward[0]) == 0.0, f"{delta_path}: first frame reward != 0, re-verify assumption"
    assert float(d_reward[-1]) == 1.0, f"{delta_path}: last frame reward != 1, re-verify assumption"

    new_reward_col = list(abs_df["next.reward"].tolist())
    new_done_col = list(abs_df["next.done"].tolist())
    n_carried_forward = 0
    for i in range(abs_len):
        orig_reward_cell = new_reward_col[i]
        orig_done_cell = new_done_col[i]
        if i < offset:
            continue  # front-trimmed dead time: leave as-is (already reward=0/done=False)
        elif i < offset + n:
            j = i - offset
            new_reward_col[i] = _like(orig_reward_cell, float(d_reward[j]))
            new_done_col[i] = _like(orig_done_cell, bool(d_done[j]))
        else:
            new_reward_col[i] = _like(orig_reward_cell, 1.0)
            new_done_col[i] = _like(orig_done_cell, True)
            n_carried_forward += 1

    abs_df["next.reward"] = new_reward_col
    abs_df["next.done"] = new_done_col
    abs_df.to_parquet(abs_path, index=False)

    n_reward1 = sum(1 for x in new_reward_col if float(_scalar(x)) == 1.0)
    n_done1 = sum(1 for x in new_done_col if bool(_scalar(x)))
    return {"abs_len": abs_len, "offset": offset, "n_matched": n, "n_carried_forward": n_carried_forward,
            "n_reward1": n_reward1, "n_done1": n_done1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--abs-root", required=True)
    parser.add_argument("--delta-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Report the mapping only; write nothing.")
    args = parser.parse_args()

    abs_root = Path(args.abs_root)
    delta_root = Path(args.delta_root)
    output_root = Path(args.output_root)

    print(f"Building content-based episode mapping ({abs_root.name} <-> {delta_root.name}) ...")
    mapping = build_mapping(abs_root, delta_root)
    print(f"Matched {len(mapping)} episodes.")

    if args.dry_run:
        for ep in sorted(mapping):
            m = mapping[ep]
            print(f"  abs_ep={ep:3d}  delta={m['delta_path'].name}  offset={m['offset']:4d}  n_matched={m['n']:4d}")
        print("--dry-run: nothing written.")
        return

    if output_root.exists():
        raise FileExistsError(f"--output-root {output_root} already exists - refusing to overwrite.")
    print(f"Copying {abs_root} -> {output_root} ...")
    shutil.copytree(abs_root, output_root)

    added = ensure_reward_column(output_root)
    print(f"next.reward column {'added (source dataset predates the reward pipeline)' if added else 'already present'}.")

    abs_files = {_episode_index_of(pd.read_parquet(f), f): f for f in _find_episode_files(output_root)}

    total_reward1 = 0
    total_done1 = 0
    total_carried = 0
    for ep in sorted(mapping):
        m = mapping[ep]
        r = annotate_episode(abs_files[ep], m["delta_path"], m["offset"], m["n"])
        total_reward1 += r["n_reward1"]
        total_done1 += r["n_done1"]
        total_carried += r["n_carried_forward"]
        print(f"  episode {ep:3d}: {r['abs_len']} frames, matched [{r['offset']}:{r['offset']+r['n_matched']}], "
              f"carried-forward {r['n_carried_forward']} trailing frames, "
              f"reward==1: {r['n_reward1']}, done==True: {r['n_done1']}")

    print("Recomputing next.reward/next.done stats (meta/episodes_stats.jsonl + meta/stats.json) ...")
    recompute_reward_done_stats(output_root)

    print(f"\nDone. {len(mapping)} episodes annotated, {total_reward1} total next.reward==1 frames, "
          f"{total_done1} total next.done==True frames ({total_carried} of those from end-trim carry-forward).")
    print(f"Annotated copy written to {output_root}")


if __name__ == "__main__":
    main()
