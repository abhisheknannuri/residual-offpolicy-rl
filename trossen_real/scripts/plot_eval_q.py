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

PLOT STYLE follows `resfit/rl_finetuning/utils/evaluate_dexmg.py`
(`_create_q_trajectory_plots`, logged to W&B as `value/q_trajectories`):
green for success, red for failure, plus a distribution across episode
progress.

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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")                      # no display on the rig
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

logger = logging.getLogger(__name__)

SUCCESS_COLOR = "#2e7d32"
FAILURE_COLOR = "#c62828"


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


def _normalised(traj: np.ndarray, n: int = 100) -> np.ndarray:
    """Resample a trajectory onto a common 0-100% progress axis."""
    if len(traj) < 2:
        return np.full(n, traj[0] if len(traj) else np.nan)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(traj)), traj)


def plot_checkpoint(runs: list[EvalRun], checkpoint: str, out: Path) -> dict[str, Any]:
    """The six-panel figure for one checkpoint. Returns the summary stats."""
    succ = [r for r in runs if r.success]
    fail = [r for r in runs if not r.success]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"Residual-RL critic diagnostics - {checkpoint}\n"
        f"{len(runs)} runs: {len(succ)} success, {len(fail)} failure",
        fontsize=14, fontweight="bold")
    ax = axes.ravel()

    # -- 1. every Q trajectory, green/red --------------------------------
    for i, r in enumerate(fail):
        ax[0].plot(r.q_mean, color=FAILURE_COLOR, alpha=0.45, linewidth=1,
                   label="Failure" if i == 0 else "")
    for i, r in enumerate(succ):
        ax[0].plot(r.q_mean, color=SUCCESS_COLOR, alpha=0.85, linewidth=1.6,
                   label="Success" if i == 0 else "")
    ax[0].set_xlabel("Episode step")
    ax[0].set_ylabel("Q (mean over critic heads)")
    ax[0].set_title("Q trajectories")
    ax[0].grid(alpha=0.3)
    if runs:
        ax[0].legend(loc="best")

    # -- 2. mean +/- 1 sd band on a common progress axis -----------------
    for group, color, name in ((succ, SUCCESS_COLOR, "Success"),
                               (fail, FAILURE_COLOR, "Failure")):
        if not group:
            continue
        stack = np.vstack([_normalised(r.q_mean) for r in group])
        m, s = stack.mean(0), stack.std(0)
        x = np.linspace(0, 100, stack.shape[1])
        ax[1].plot(x, m, color=color, linewidth=2, label=f"{name} (n={len(group)})")
        ax[1].fill_between(x, m - s, m + s, color=color, alpha=0.18)
    ax[1].set_xlabel("Episode progress (%)")
    ax[1].set_ylabel("Q")
    ax[1].set_title("Mean Q +/- 1 sd, time-normalised\n"
                    "A separation here is the critic discriminating")
    ax[1].grid(alpha=0.3)
    ax[1].legend(loc="best")

    # -- 3. distribution at progress points (the reference box plot) -----
    points = [0.25, 0.5, 0.75, 1.0]
    pos, data, colors = [], [], []
    for j, p in enumerate(points):
        for k, (group, color) in enumerate(((succ, SUCCESS_COLOR),
                                            (fail, FAILURE_COLOR))):
            vals = [r.q_mean[min(int(p * len(r.q_mean)), len(r.q_mean) - 1)]
                    for r in group if r.n_ticks]
            if vals:
                data.append(vals)
                pos.append(j * 3 + k)
                colors.append(color)
    if data:
        bp = ax[2].boxplot(data, positions=pos, widths=0.8, patch_artist=True,
                           medianprops={"color": "black"})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
    ax[2].set_xticks([j * 3 + 0.5 for j in range(len(points))])
    ax[2].set_xticklabels([f"{int(p * 100)}%" for p in points])
    ax[2].set_xlabel("Episode progress")
    ax[2].set_ylabel("Q")
    ax[2].set_title("Q distribution by progress (green=success, red=failure)")
    ax[2].grid(alpha=0.3, axis="y")

    # -- 4. terminal Q, the sharpest single diagnostic -------------------
    term_s = [float(r.q_mean[-1]) for r in succ if r.n_ticks]
    term_f = [float(r.q_mean[-1]) for r in fail if r.n_ticks]
    bins = 20
    if term_s or term_f:
        lo = min(term_s + term_f)
        hi = max(term_s + term_f)
        edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins
        if term_f:
            ax[3].hist(term_f, bins=edges, color=FAILURE_COLOR, alpha=0.6,
                       label=f"Failure (n={len(term_f)})")
        if term_s:
            ax[3].hist(term_s, bins=edges, color=SUCCESS_COLOR, alpha=0.75,
                       label=f"Success (n={len(term_s)})")
    ax[3].set_xlabel("Q at the final step")
    ax[3].set_ylabel("Runs")
    ax[3].axvline(0.99, color="black", linestyle=":", linewidth=1.2)
    ax[3].text(0.99, ax[3].get_ylim()[1] * 0.96, " Q=0.99", fontsize=8, va="top")
    ax[3].set_title("Terminal Q distribution\n"
                    "Overlapping histograms = the critic cannot tell them apart")
    ax[3].grid(alpha=0.3, axis="y")
    if term_s or term_f:
        ax[3].legend(loc="best")

    # -- 5. mean Q per run against the stage actually reached ------------
    stages = sorted({r.stage for r in runs if r.stage is not None})
    for st in stages:
        group = [r for r in runs if r.stage == st and r.n_ticks]
        if not group:
            continue
        xs = np.full(len(group), st, dtype=float)
        xs += np.random.default_rng(0).uniform(-0.12, 0.12, len(group))
        ys = [float(r.q_mean.mean()) for r in group]
        cs = [SUCCESS_COLOR if r.success else FAILURE_COLOR for r in group]
        ax[4].scatter(xs, ys, c=cs, alpha=0.8, s=45, edgecolors="none")
    ax[4].set_xlabel("Furthest stage reached")
    ax[4].set_ylabel("Mean Q over the run")
    ax[4].set_title("Mean Q vs stage reached\n"
                    "An upward trend means Q tracks real progress")
    ax[4].set_xticks(stages)
    ax[4].grid(alpha=0.3)

    # -- 6. critic ensemble disagreement --------------------------------
    for group, color, name in ((succ, SUCCESS_COLOR, "Success"),
                               (fail, FAILURE_COLOR, "Failure")):
        vals = [float(np.nanmean(r.q_std)) for r in group
                if r.n_ticks and not np.all(np.isnan(r.q_std))]
        if vals:
            ax[5].hist(vals, bins=15, color=color, alpha=0.6,
                       label=f"{name} (n={len(vals)})")
    ax[5].set_xlabel("Mean sd across critic heads")
    ax[5].set_ylabel("Runs")
    ax[5].set_title("Ensemble disagreement\n"
                    "High sd = out-of-distribution states the critic is unsure about")
    ax[5].grid(alpha=0.3, axis="y")
    if ax[5].get_legend_handles_labels()[0]:
        ax[5].legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return _stats(checkpoint, runs, succ, fail, term_s, term_f)


def _stats(checkpoint: str, runs: list[EvalRun], succ: list[EvalRun],
           fail: list[EvalRun], term_s: list[float],
           term_f: list[float]) -> dict[str, Any]:
    """The numbers behind the figure, so a verdict is not eyeballed."""
    def arr(xs):
        return np.array(xs, dtype=float) if xs else np.array([np.nan])

    mean_s = arr([float(r.q_mean.mean()) for r in succ if r.n_ticks])
    mean_f = arr([float(r.q_mean.mean()) for r in fail if r.n_ticks])

    # Separation in units of the pooled spread. Cohen's d: how far apart the
    # two groups are relative to how wide they each are. |d| < 0.2 is
    # negligible, > 0.8 is large.
    pooled = np.sqrt((np.nanvar(mean_s) + np.nanvar(mean_f)) / 2)
    d = float((np.nanmean(mean_s) - np.nanmean(mean_f)) / pooled) if pooled > 0 else float("nan")

    # AUC via the Mann-Whitney relation: the probability that a randomly
    # chosen success scores above a randomly chosen failure. 0.5 is chance,
    # which is exactly what an uninformative critic produces.
    auc = float("nan")
    if len(term_s) and len(term_f):
        wins = sum((a > b) + 0.5 * (a == b) for a in term_s for b in term_f)
        auc = wins / (len(term_s) * len(term_f))

    # Over-optimism. Q saturates at ~1.0 (sparse 0/1 reward, discounted), so
    # "Q reached 0.99" means the critic became confident the task was about to
    # be solved. Counting how often that happened on runs that then FAILED is
    # the sharpest single number here: a critic can discriminate at the final
    # step and still be confidently wrong throughout.
    conf_fail = sum(1 for r in fail if r.n_ticks and float(r.q_mean.max()) >= 0.99)
    conf_succ = sum(1 for r in succ if r.n_ticks and float(r.q_mean.max()) >= 0.99)

    return {
        "checkpoint": checkpoint,
        "n_runs": len(runs), "n_success": len(succ), "n_failure": len(fail),
        "failures_reaching_q99": conf_fail,
        "frac_failures_reaching_q99": conf_fail / len(fail) if fail else float("nan"),
        "successes_reaching_q99": conf_succ,
        "q_mean_success": float(np.nanmean(mean_s)),
        "q_mean_failure": float(np.nanmean(mean_f)),
        "q_terminal_success": float(np.nanmean(arr(term_s))),
        "q_terminal_failure": float(np.nanmean(arr(term_f))),
        "cohens_d_mean_q": d,
        "auc_terminal_q": auc,
        "total_ticks": int(sum(r.n_ticks for r in runs)),
    }


def verdict(s: dict[str, Any]) -> list[str]:
    """Say what the numbers mean, rather than leaving it to a glance."""
    out = []
    auc, d = s["auc_terminal_q"], s["cohens_d_mean_q"]
    if s["n_success"] == 0:
        return ["  NO SUCCESSFUL RUNS - nothing to separate against. The plots "
                "show the Q range but cannot say whether the critic discriminates."]
    if s["n_success"] < 5:
        out.append(f"  CAUTION: only {s['n_success']} successful run(s). Every "
                   f"number below is noisy at that sample size.")
    if not np.isnan(auc):
        if auc >= 0.75:
            out.append(f"  AUC {auc:.2f} - terminal Q separates success from failure well.")
        elif auc >= 0.6:
            out.append(f"  AUC {auc:.2f} - weak but non-trivial separation.")
        elif auc >= 0.4:
            out.append(f"  AUC {auc:.2f} - AT CHANCE. The critic's terminal Q does "
                       f"not distinguish a success from a failure.")
        else:
            out.append(f"  AUC {auc:.2f} - INVERTED: failures score HIGHER than "
                       f"successes. Worse than uninformative.")
    if not np.isnan(d):
        out.append(f"  Cohen's d {d:+.2f} on mean Q "
                   f"({'negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'}).")

    frac = s.get("frac_failures_reaching_q99", float("nan"))
    if not np.isnan(frac):
        n, tot = s["failures_reaching_q99"], s["n_failure"]
        if frac >= 0.3:
            out.append(f"  OVER-OPTIMISTIC: {n}/{tot} failures ({frac:.0%}) still "
                       f"reached Q >= 0.99 - the critic became confident the task "
                       f"was about to be solved on runs that did not solve it.")
        elif frac > 0:
            out.append(f"  {n}/{tot} failures ({frac:.0%}) reached Q >= 0.99.")
        else:
            out.append("  No failure reached Q >= 0.99 - the critic reserves high "
                       "confidence for runs that succeed.")
    return out


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

    all_stats = []
    for ckpt, group in sorted(by_ckpt.items()):
        out_dir = eval_dir / ckpt
        png = out_dir / "q_diagnostics.png"
        stats = plot_checkpoint(group, ckpt, png)
        all_stats.append(stats)

        (out_dir / "q_stats.json").write_text(json.dumps(stats, indent=2))
        # The raw joined series, so any other plot can be made without
        # re-doing the time-window join.
        (out_dir / "q_trajectories.json").write_text(json.dumps({
            "checkpoint": ckpt,
            "runs": [{"run_id": r.run_id, "pose": r.pose, "success": r.success,
                      "stage": r.stage, "steps": r.steps,
                      "stop_reason": r.stop_reason,
                      "q_mean": [round(v, 6) for v in r.q_mean.tolist()]}
                     for r in group],
        }, indent=2))

        print(f"\n=== {ckpt} ===")
        print(f"  {stats['n_runs']} runs ({stats['n_success']} success), "
              f"{stats['total_ticks']} ticks")
        print(f"  mean Q      success {stats['q_mean_success']:+.3f}   "
              f"failure {stats['q_mean_failure']:+.3f}")
        print(f"  terminal Q  success {stats['q_terminal_success']:+.3f}   "
              f"failure {stats['q_terminal_failure']:+.3f}")
        print(f"  failures reaching Q>=0.99: {stats['failures_reaching_q99']}"
              f"/{stats['n_failure']}")
        for line in verdict(stats):
            print(line)
        print(f"  -> {png}")

    if args.wandb:
        _to_wandb(args.wandb_project, eval_dir, by_ckpt, all_stats)
    return 0


def _to_wandb(project: str, eval_dir: Path, by_ckpt: dict[str, list[EvalRun]],
              all_stats: list[dict[str, Any]]) -> None:
    try:
        import wandb
    except ImportError:
        print("\nwandb is not installed; skipping upload", file=sys.stderr)
        return
    for stats in all_stats:
        ckpt = stats["checkpoint"]
        run = wandb.init(project=project, name=f"{eval_dir.name}_{ckpt}",
                         config=stats, reinit=True)
        # `value/` to match evaluate_dexmg.py, which logs its Q plot as
        # `value/q_trajectories`.
        run.log({"value/q_diagnostics": wandb.Image(str(eval_dir / ckpt / "q_diagnostics.png")),
                 **{f"value/{k}": v for k, v in stats.items()
                    if isinstance(v, (int, float))}})
        run.finish()
    print("\nlogged to W&B")


if __name__ == "__main__":
    raise SystemExit(main())
