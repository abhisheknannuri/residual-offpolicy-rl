# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Aggregate eval runs into the two markdown tables used in
`trossen_real/EVAL/ACT/TrossenStation3/Notes.md`, so they stop being filled in
by hand. Offline and read-only: reads `<eval_root>/index.jsonl` and nothing else.

    .venv/bin/python -m trossen_real.scripts.eval_report --eval-root eval_runs
    .venv/bin/python -m trossen_real.scripts.eval_report --eval-root eval_runs --format csv
    .venv/bin/python -m trossen_real.scripts.eval_report --eval-root eval_runs \\
        --checkpoint ACT_BC_..._055000            # just one pose sheet

Two column conventions, deliberately different, both matching the existing sheet:

* **Pose sheet** `S0..S3`: ticks ONLY the furthest sub-stage reached, because
  that is what the hand-filled sheet does (pose 01 there has just `S2` ticked,
  pose 02 just `S3`). Reading a row: "got as far as S2".
* **Checkpoint summary** `Sub-Stage k Done`: CUMULATIVE count of poses that
  reached *at least* stage k, which is the meaningful aggregate (and is
  monotonically non-increasing across k, an easy sanity check).

Assisted runs (operator intervened) are EXCLUDED from the headline success rate
by default and reported separately - an assisted run does not measure the policy.
Pass `--include-assisted` to fold them in.

Only `status == "saved"` rows count. Discarded and unreviewed runs stay in
index.jsonl forever (nothing is ever deleted) but never affect the numbers.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from trossen_real.infer_app.eval_writer import read_index

TICK = "- [X]"
BLANK = "- [ ]"


def _fmt_median(values: list[int | float]) -> str:
    if not values:
        return "-"
    med = statistics.median(values)
    return f"{med:.0f}" if float(med).is_integer() else f"{med:.1f}"


def _group(rows: list[dict], *, include_assisted: bool) -> dict[str, dict[int, dict]]:
    """checkpoint -> pose_index -> row (latest saved run wins for a given pose)."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        if r.get("status") != "saved":
            continue
        if r.get("assisted") and not include_assisted:
            continue
        ckpt = r.get("checkpoint_label") or "(unlabelled)"
        pose = r.get("pose_index")
        if pose is None:
            continue
        prev = out[ckpt].get(pose)
        # A pose re-run supersedes the earlier attempt.
        if prev is None or (r.get("started_at") or 0) >= (prev.get("started_at") or 0):
            out[ckpt][pose] = r
    return out


def render_checkpoint_summary(rows: list[dict], n_stages: int, n_poses: int,
                              *, include_assisted: bool) -> str:
    grouped = _group(rows, include_assisted=include_assisted)
    assisted_by_ckpt: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("status") == "saved" and r.get("assisted"):
            assisted_by_ckpt[r.get("checkpoint_label") or "(unlabelled)"] += 1

    header = (
        "| Checkpoint | Eval Done (" + str(n_poses) + " Poses) | "
        + " | ".join(f"Sub-Stage {k} Done" for k in range(n_stages))
        + f" | Full Success (x/{n_poses}) | Median Steps | Timeout (x/{n_poses}) | Notes |"
    )
    sep = "| " + " | ".join(["---"] * (n_stages + 6)) + " |"
    lines = [header, sep]

    for ckpt in sorted(grouped):
        poses = grouped[ckpt]
        done = len(poses)
        stage_counts = [
            sum(1 for r in poses.values()
                if r.get("furthest_stage") is not None and r["furthest_stage"] >= k)
            for k in range(n_stages)
        ]
        success = sum(1 for r in poses.values() if r.get("furthest_stage") == n_stages - 1)
        timeouts = sum(1 for r in poses.values() if r.get("timed_out"))
        steps = [r["steps"] for r in poses.values() if r.get("steps") is not None]
        note_bits = []
        if assisted_by_ckpt.get(ckpt):
            note_bits.append(f"{assisted_by_ckpt[ckpt]} assisted"
                             + ("" if include_assisted else " (excluded)"))
        if done < n_poses:
            note_bits.append(f"{n_poses - done} pose(s) pending")
        lines.append(
            f"| {ckpt} | {TICK if done >= n_poses else f'{done}/{n_poses}'} | "
            + " | ".join(str(c) for c in stage_counts)
            + f" | {success}/{n_poses} | {_fmt_median(steps)} | {timeouts}/{n_poses} | "
            + ("; ".join(note_bits) or "-") + " |"
        )
    if len(lines) == 2:
        lines.append(f"| _(no saved runs)_ |{' |' * (n_stages + 5)} |")
    return "\n".join(lines)


def render_pose_sheet(rows: list[dict], checkpoint: str, n_stages: int, n_poses: int,
                      *, include_assisted: bool) -> str:
    grouped = _group(rows, include_assisted=include_assisted).get(checkpoint, {})
    header = ("| Pose # | " + " | ".join(f"S{k}" for k in range(n_stages))
              + " | Steps | Result (Pass/Fail) | Failure Mode / Note |")
    sep = "| " + " | ".join(["---"] * (n_stages + 4)) + " |"
    lines = [f"- Checkpoint Under Test: `{checkpoint}`", "", header, sep]
    for pose in range(1, n_poses + 1):
        r = grouped.get(pose)
        if r is None:
            lines.append(f"| {pose:02d} | " + " | ".join([BLANK] * n_stages) + " |  |  |  |")
            continue
        fs = r.get("furthest_stage")
        # Only the furthest stage is ticked - matches the hand-filled sheet.
        ticks = [TICK if fs == k else BLANK for k in range(n_stages)]
        passed = fs == n_stages - 1
        note = r.get("failure_note") or ""
        if r.get("assisted"):
            seg = r.get("intervened_during_stage")
            note = (f"ASSISTED ({r.get('total_intervened_steps')} steps"
                    + (f", stage {seg}" if seg is not None else "") + ")"
                    + (f" - {note}" if note else ""))
        lines.append(
            f"| {pose:02d} | " + " | ".join(ticks)
            + f" | {r.get('steps', '')} | {'Pass' if passed else 'Fail'} | {note} |"
        )
    return "\n".join(lines)


def write_csv(rows: list[dict], out: Path) -> int:
    saved = [r for r in rows if r.get("status") == "saved"]
    cols = ["run_id", "checkpoint_label", "pose_index", "furthest_stage", "success",
            "steps", "max_steps", "stop_reason", "timed_out", "assisted",
            "total_intervened_steps", "intervened_during_stage", "failure_note",
            "operator", "started_at", "eval_config_hash"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in saved:
            w.writerow(r)
    return len(saved)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", default="eval_runs", help="directory containing index.jsonl")
    ap.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    ap.add_argument("--out", default=None, help="write to this file instead of stdout (csv: required target)")
    ap.add_argument("--checkpoint", default=None, help="render the pose sheet for this checkpoint only")
    ap.add_argument("--n-stages", type=int, default=4)
    ap.add_argument("--n-poses", type=int, default=20)
    ap.add_argument("--include-assisted", action="store_true",
                    help="count operator-assisted runs in the headline rate (default: excluded)")
    args = ap.parse_args(argv)

    index_path = Path(args.eval_root) / "index.jsonl"
    rows = read_index(index_path)
    if not rows:
        print(f"No eval runs found in {index_path}", file=sys.stderr)
        return 1

    if args.format == "csv":
        out = Path(args.out or (Path(args.eval_root) / "eval_runs.csv"))
        n = write_csv(rows, out)
        print(f"wrote {n} saved runs -> {out}")
        return 0

    parts = ["##### Checkpoint Summary (Quick Tracking)", "",
             render_checkpoint_summary(rows, args.n_stages, args.n_poses,
                                       include_assisted=args.include_assisted), ""]
    checkpoints = ([args.checkpoint] if args.checkpoint
                   else sorted(_group(rows, include_assisted=args.include_assisted)))
    for ckpt in checkpoints:
        parts += ["##### Pose-Wise Sheet (Use Per Checkpoint)", "",
                  render_pose_sheet(rows, ckpt, args.n_stages, args.n_poses,
                                    include_assisted=args.include_assisted), ""]
    text = "\n".join(parts)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
