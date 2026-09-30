# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Two plots for an eval campaign. Deliberately simple.

Reads the collected archive (`<dest>/runs/**/summary.json`), not the live
`eval_runs/`, so a figure can always be regenerated from the frozen record.

    .venv/bin/python -m trossen_real.scripts.plot_eval_results \
        --runs trossen_real/EVAL/ACT/TrossenStation3/runs \
        --out  trossen_real/EVAL/ACT/TrossenStation3/analysis \
        --exclude-checkpoint 055000

1. `success_rate_vs_checkpoint.png` - full-task (S3) success rate per checkpoint.
   Red text under a bar = how many of those runs a human had to touch.
2. `substage_success_rate.png` - CUMULATIVE sub-stage rate: reaching S3 counts as
   having done S2 and S1 too, so the bars are non-increasing left to right within
   a checkpoint.

Assisted runs (a human touched the leader arm) are not autonomous successes, so
they count as failures while the denominator stays at 20 - equal denominators
matter because the poses are fixed across checkpoints. Discarded runs are ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STAGE_NAMES = ["S0 approach", "S1 grasp & lift", "S2 transport", "S3 insert (success)"]
STAGE_COLORS = ["#C62828", "#EF6C00", "#F9A825", "#2E7D32"]


def load(runs_root: Path, exclude: set[str]) -> dict[str, dict[int, dict]]:
    """ckpt_step -> pose_index -> summary (latest saved run wins)."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in sorted(runs_root.glob("ckpt_*/pose_*/run_*/summary.json")):
        s = json.loads(f.read_text())
        if s.get("status") != "saved":
            continue
        step = s["checkpoint_label"].rsplit("/", 1)[-1]
        if step in exclude:
            continue
        prev = out[step].get(s["pose"]["index"])
        if prev is None or s["started_at"] >= prev["started_at"]:
            out[step][s["pose"]["index"]] = s
    return out


def _stat(runs: list[dict]) -> dict:
    n = len(runs)
    stages = [r["annotation"]["furthest_stage"] for r in runs]
    assisted = [r["intervention"]["assisted"] for r in runs]
    succ = sum(1 for r, a in zip(runs, assisted)
               if r["annotation"]["success"] and not a)
    return dict(n=n, succ=succ, p=succ / n,
                assisted=sum(assisted),
                timeout=sum(r["run"]["timed_out"] for r in runs),
                med_steps=float(np.median([r["run"]["steps"] for r in runs])),
                # cumulative: furthest == 3 also counts as having done 1 and 2
                cum=[sum(1 for s in stages if s >= k) for k in range(4)])


def fig_success(steps, stats, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(steps))
    pct = [s["p"] * 100 for s in stats]
    ax.bar(x, pct, width=0.6, color="#2E7D32")
    for xi, s, v in zip(x, stats, pct):
        ax.text(xi, v + 1.5, f"{s['succ']}/{s['n']}  ({v:.0f}%)", ha="center",
                fontsize=10, fontweight="bold")
        if s["assisted"]:
            ax.text(xi, -4.0, f"{s['assisted']} assisted", ha="center", va="center",
                    fontsize=8, color="#B71C1C")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s) // 1000}k" for s in steps])
    ax.set_xlabel("ACT checkpoint (training steps)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(-7, max(max(pct) + 12, 40))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.005, "20 fixed cube poses per checkpoint; assisted runs count as failures",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_substage(steps, stats, out: Path, title: str):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(steps))
    w = 0.26
    for j, k in enumerate((1, 2, 3)):
        vals = [s["cum"][k] / s["n"] * 100 for s in stats]
        pos = x + (j - 1) * w
        ax.bar(pos, vals, width=w, color=STAGE_COLORS[k], label=STAGE_NAMES[k])
        for xi, v, s in zip(pos, vals, stats):
            ax.text(xi, v + 1.5, f"{s['cum'][k]}", ha="center", fontsize=8,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s) // 1000}k" for s in steps])
    ax.set_xlabel("ACT checkpoint (training steps)")
    ax.set_ylabel("% of 20 poses that completed the sub-stage")
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, ncol=3, loc="upper center", framealpha=0.95)
    fig.text(0.5, 0.005,
             "cumulative: a run that inserted (S3) also counts for S2 and S1",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-checkpoint", nargs="*", default=[],
                    help="checkpoint step strings to drop, e.g. 055000")
    ap.add_argument("--title", default="ACT BC - PickCubeAndInsert - Trossen Station 3 (delta-joint)")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    per_pose = load(Path(args.runs), set(args.exclude_checkpoint))
    if not per_pose:
        print("no saved runs found"); return 1
    steps = sorted(per_pose, key=int)
    stats = [_stat(list(per_pose[s].values())) for s in steps]

    with (out / "eval_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["checkpoint_step", "n_poses", "success", "success_rate_pct",
                    "did_s1", "did_s2", "did_s3", "assisted", "timeout",
                    "median_steps"])
        for st, s in zip(steps, stats):
            w.writerow([int(st), s["n"], s["succ"], f"{s['p'] * 100:.0f}",
                        s["cum"][1], s["cum"][2], s["cum"][3],
                        s["assisted"], s["timeout"], f"{s['med_steps']:.0f}"])

    fig_success(steps, stats, out / "success_rate_vs_checkpoint.png",
                f"{args.title}\nfull-task success (S3 insert)")
    fig_substage(steps, stats, out / "substage_success_rate.png",
                 f"{args.title}\nsub-stage success rate (cumulative)")

    print(f"{len(steps)} checkpoints x {stats[0]['n']} poses -> {out}")
    for st, s in zip(steps, stats):
        print(f"  {int(st)//1000:>3}k  succ {s['succ']:>2}/{s['n']} ({s['p']*100:3.0f}%)  "
              f"S1 {s['cum'][1]:>2}  S2 {s['cum'][2]:>2}  S3 {s['cum'][3]:>2}  "
              f"assisted {s['assisted']}  timeout {s['timeout']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
