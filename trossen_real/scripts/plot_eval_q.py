#!/usr/bin/env python
"""Q-value plots for a day of residual-RL evaluations.

WHY THIS EXISTS
---------------
The eval runs under `eval_runs/<date>/` record what the operator SAW - pose,
stage reached, success - but not what the critic THOUGHT. The Q values live in
a separate per-connect log under `logs/residual/`, written by
`ResidualTickLog`, and nothing joined the two.

This joins them on wall-clock time and draws the plots that answer one
question: **is the offline critic learning anything about this task?**

A critic that has learned something assigns higher Q to the trajectories that
succeeded. One that has not produces the same values either way - which looks
exactly like a healthy training curve on the learner side, and is only visible
here.

HOW THE JOIN WORKS
------------------
`summary.json` has `started_at` / `ended_at`; every residual tick has `t`. A
tick belongs to the run whose window contains it. Verified exact on the
2026-10-05 set: every run's matched tick count equals its recorded step count,
for all 41 runs.

Runs are grouped by checkpoint, because comparing Q across checkpoints is
meaningless - a critic's absolute scale is arbitrary and drifts with training.

THE PLOT IS RESFIT'S, NOT A NEW ONE
-----------------------------------
`_create_q_trajectory_plots` from `resfit/rl_finetuning/utils/evaluate_dexmg.py`
is copied verbatim into `_resfit_q_plot.py` - extracted with `ast`, verified
byte-identical, and checked again on every run. It is nested inside
`run_dexmg_evaluation()` so it cannot be imported.

That means these figures are the SAME figures the training-time evaluation
produces, with the same file names, the same JSON payload, and the same W&B
key: `value/q_trajectories`. Nothing here draws its own plot.

USAGE
-----
    .venv/bin/python -m trossen_real.scripts.plot_eval_q
    .venv/bin/python -m trossen_real.scripts.plot_eval_q --date 2026-10-05
    .venv/bin/python -m trossen_real.scripts.plot_eval_q --wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trossen_real.scripts._resfit_q_plot import _create_q_trajectory_plots, verify

logger = logging.getLogger(__name__)

@dataclass
class EvalRun:
    """One evaluation run, joined to its Q trajectory."""

    run_dir: Path
    run_id: str
    pose: int
    checkpoint: str
    t0: float
    t1: float
    steps: int
    stage: int | None
    success: bool
    stop_reason: str
    q_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    q_std: np.ndarray = field(default_factory=lambda: np.array([]))
    q_min: np.ndarray = field(default_factory=lambda: np.array([]))
    q_max: np.ndarray = field(default_factory=lambda: np.array([]))
    residual_abs_max: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_ticks(self) -> int:
        return len(self.q_mean)

    @property
    def label(self) -> str:
        return f"pose_{self.pose:02d}"


def load_runs(eval_dir: Path) -> list[EvalRun]:
    runs: list[EvalRun] = []
    for summary_path in sorted(eval_dir.rglob("summary.json")):
        d = json.loads(summary_path.read_text())
        if d.get("status") == "discarded":
            continue
        ann = d.get("annotation") or {}
        runs.append(EvalRun(
            run_dir=summary_path.parent,
            run_id=d.get("run_id", summary_path.parent.name),
            pose=int(d["pose"]["index"]),
            # Last path component: the checkpoint directories are long, and
            # the step is the only part that differs within a day.
            checkpoint=str(d["checkpoint_label"]).rstrip("/").split("/")[-1],
            t0=float(d["started_at"]),
            t1=float(d["ended_at"]),
            steps=int(d["run"]["steps"]),
            stage=ann.get("furthest_stage"),
            success=bool(ann.get("success", False)),
            stop_reason=str(d["run"].get("stop_reason", "")),
        ))
    return runs


def load_ticks(residual_dir: Path) -> list[dict[str, Any]]:
    """Every residual tick that carries a Q value, from every log."""
    ticks: list[dict[str, Any]] = []
    for path in sorted(residual_dir.glob("*.jsonl")):
        for line in path.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "load" or "q_mean" not in rec:
                continue
            ticks.append(rec)
    ticks.sort(key=lambda r: r["t"])
    return ticks


def join(runs: list[EvalRun], ticks: list[dict[str, Any]]) -> None:
    """Attach each run's Q trajectory, by wall-clock window."""
    times = np.array([t["t"] for t in ticks])
    for run in runs:
        lo = int(np.searchsorted(times, run.t0, "left"))
        hi = int(np.searchsorted(times, run.t1, "right"))
        window = ticks[lo:hi]
        run.q_mean = np.array([t["q_mean"] for t in window], dtype=float)
        run.q_std = np.array([t.get("q_std", np.nan) for t in window], dtype=float)
        run.q_min = np.array([t.get("q_min", np.nan) for t in window], dtype=float)
        run.q_max = np.array([t.get("q_max", np.nan) for t in window], dtype=float)
        run.residual_abs_max = np.array(
            [t.get("residual_abs_max", np.nan) for t in window], dtype=float)


def plot_checkpoint(runs: list[EvalRun], checkpoint: str, out_dir: Path,
                    global_step: int | None) -> dict[str, Any]:
    """Call resfit's own plotter, and save its JSON the way resfit does.

    No second plot, no reinterpretation. `_create_q_trajectory_plots` is copied
    verbatim in `_resfit_q_plot.py` (verified byte-identical), so the figure is
    the same one the training-time evaluation produces and logs to W&B as
    `value/q_trajectories`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step_tag = global_step if global_step is not None else "NA"
    plot_path = out_dir / f"eval_q_trajectories_{checkpoint}_step_{step_tag}.png"
    data_path = out_dir / f"eval_q_trajectories_{checkpoint}_step_{step_tag}.json"

    trajectories = [r.q_mean.tolist() for r in runs]
    lengths = [r.n_ticks for r in runs]
    successes = [r.success for r in runs]

    _create_q_trajectory_plots(
        trajectories=trajectories,
        episode_lengths=lengths,
        successes=successes,
        output_path=plot_path,
        global_step=global_step,
    )

    # resfit's dump_data shape, so anything that reads one reads the other.
    # `pose` is the only addition - these are real evaluation poses, not
    # simulator episodes, and without it a trajectory cannot be traced back.
    data_path.write_text(json.dumps({
        "global_step": global_step,
        "episodes": [
            {
                "episode_idx": i + 1,
                "success": successes[i],
                "length": lengths[i],
                "q_trajectory": trajectories[i],
                "pose": runs[i].pose,
                "run_id": runs[i].run_id,
                "furthest_stage": runs[i].stage,
            }
            for i in range(len(successes))
        ],
    }, indent=2))

    return {"checkpoint": checkpoint, "global_step": global_step,
            "n_runs": len(runs), "n_success": sum(successes),
            "n_failure": len(runs) - sum(successes),
            "total_ticks": int(sum(lengths)),
            "plot": str(plot_path), "data": str(data_path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="plot_eval_q",
        description="Join eval runs to their residual Q logs and plot.")
    ap.add_argument("--repo", default=".", help="repo root")
    ap.add_argument("--date", default=None,
                    help="eval_runs subdirectory (default: the newest)")
    ap.add_argument("--eval-dir", default=None, help="an explicit eval directory")
    ap.add_argument("--residual-dir", default=None,
                    help="default: <repo>/logs/residual")
    ap.add_argument("--wandb", action="store_true",
                    help="also log the figures to W&B under value/")
    ap.add_argument("--wandb-project", default="resfit-eval-q")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = Path(args.repo).expanduser().resolve()

    if args.eval_dir:
        eval_dir = Path(args.eval_dir).expanduser().resolve()
    else:
        root = repo / "eval_runs"
        if args.date:
            eval_dir = root / args.date
        else:
            dates = sorted(p for p in root.iterdir() if p.is_dir())
            if not dates:
                print(f"no eval runs under {root}", file=sys.stderr)
                return 1
            eval_dir = dates[-1]
    residual_dir = Path(args.residual_dir) if args.residual_dir else repo / "logs" / "residual"

    print(f"eval runs      {eval_dir}")
    print(f"residual logs  {residual_dir}\n")

    runs = load_runs(eval_dir)
    if not runs:
        print(f"no summary.json under {eval_dir}", file=sys.stderr)
        return 1
    ticks = load_ticks(residual_dir)
    join(runs, ticks)

    missing = [r for r in runs if r.n_ticks == 0]
    mismatched = [r for r in runs if r.n_ticks and r.n_ticks != r.steps]
    print(f"{len(runs)} runs, {len(ticks)} Q ticks available")
    print(f"  joined        {len(runs) - len(missing)}/{len(runs)}")
    if missing:
        print(f"  NO Q DATA     {', '.join(r.label for r in missing)}")
    if mismatched:
        # Worth seeing: the join is by time window, so a mismatch means the
        # window caught ticks from something else, or the run overlapped
        # another connect.
        print(f"  tick/step mismatch in {len(mismatched)} run(s): "
              + ", ".join(f"{r.label}({r.n_ticks}!={r.steps})" for r in mismatched[:5]))
    else:
        print("  tick count == step count for every joined run")

    by_ckpt: dict[str, list[EvalRun]] = {}
    for r in runs:
        if r.n_ticks:
            by_ckpt.setdefault(r.checkpoint, []).append(r)

    if not verify(repo):
        print("\n  WARNING: the copy of _create_q_trajectory_plots no longer "
              "matches resfit. Re-extract it.", file=sys.stderr)

    all_stats = []
    for ckpt, group in sorted(by_ckpt.items()):
        # The checkpoint directories are named ..._offline_step_90000, so the
        # trailing number is the global step resfit's plot title expects.
        m = re.search(r"(\d+)$", ckpt)
        step = int(m.group(1)) if m else None

        stats = plot_checkpoint(group, ckpt, eval_dir / ckpt, step)
        all_stats.append(stats)

        print(f"\n=== {ckpt} ===")
        print(f"  {stats['n_runs']} runs ({stats['n_success']} success, "
              f"{stats['n_failure']} failure), {stats['total_ticks']} ticks")
        print(f"  -> {stats['plot']}")
        print(f"  -> {stats['data']}")

    if args.wandb:
        _to_wandb(args.wandb_project, eval_dir, all_stats)
    return 0


def _to_wandb(project: str, eval_dir: Path, all_stats: list[dict[str, Any]]) -> None:
    """Log exactly as resfit does: the image under `value/q_trajectories`.

    `evaluate_dexmg.py` sets `metrics["value/q_trajectories"] = wandb.Image(...)`
    and `wandb.save()`s the JSON alongside it. Same key, same two artefacts, so
    these eval plots land where the training-time ones already are.
    """
    try:
        import wandb
    except ImportError:
        print("\nwandb is not installed; skipping upload", file=sys.stderr)
        return
    for stats in all_stats:
        ckpt = stats["checkpoint"]
        run = wandb.init(project=project, name=f"{eval_dir.name}_{ckpt}",
                         config=stats, reinit=True)
        run.log({"value/q_trajectories": wandb.Image(stats["plot"])},
                step=stats["global_step"])
        run.save(stats["data"], base_path=str(eval_dir))
        run.finish()
        print(f"  logged {ckpt} -> value/q_trajectories")


if __name__ == "__main__":
    raise SystemExit(main())
