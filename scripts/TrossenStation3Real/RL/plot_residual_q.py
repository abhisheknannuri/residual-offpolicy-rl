#!/usr/bin/env python3
"""Q values of residual RL rollouts in the infer app, per checkpoint.

Reads the infer app's residual tick logs (logs/residual/*.jsonl, written when a
residual checkpoint is loaded; `collect_q` on). Keeps only the ticks where the
residual actually ran (`residual_active` not False, i.e. from the SARM stage gate
to the end of the episode), splits them into episodes (the residual turning off
again = the next episode's gated start, or a pause > --gap-s = a reset), and
groups sessions by the checkpoint they loaded.

Writes to --out:
  q_<checkpoint>.png   one per checkpoint: Q (mean over critic heads) vs seconds
                       since the residual switched on, one thin line per episode,
                       the per-second median in bold
  q_compare.png        the median curves of all checkpoints on one axis
  summary.csv          per checkpoint: episodes, ticks, Q median / first / last
                       second, mean |residual|, share of ticks per dim with
                       residual > 0 (near 0 or 1 = a constant push)

Usage (repo venv):
    .venv/bin/python scripts/TrossenStation3Real/RL/plot_residual_q.py \\
        --run run_2026-10-26_15-11-58 --out RealTrainLogs/q_plots_stage2
--run keeps only sessions whose checkpoint path contains that text.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots in fixed order,
# muted ink for the per-episode lines, text/grid tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT2, GRID, EPISODE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b2"
FPS = 20.0


def load_sessions(pattern: str, run: str | None) -> dict[str, list[list[dict]]]:
    """checkpoint label -> list of episodes (each a list of residual-on ticks)."""
    by_ckpt: dict[str, list[list[dict]]] = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        if os.path.getsize(f) == 0:
            continue
        ckpt, ep, last_t = None, [], None
        rows = [json.loads(line) for line in open(f) if line.strip()]
        for r in rows:
            if r.get("event") == "load":
                ckpt = r.get("checkpoint", "?")
                continue
            if r.get("event") != "tick" or ckpt is None:
                continue
            on = r.get("residual_active") is not False and "q_mean" in r
            gap = last_t is not None and r["t"] - last_t > ARGS.gap_s
            if (not on or gap) and ep:
                by_ckpt[ckpt].append(ep)
                ep = []
            if on:
                ep.append(r)
            last_t = r["t"]
        if ep and ckpt is not None:
            by_ckpt[ckpt].append(ep)
    if run:
        by_ckpt = {k: v for k, v in by_ckpt.items() if run in k}
    return by_ckpt


def label(ckpt: str) -> str:
    m = re.search(r"/models/([^/]+)/", ckpt)
    return m.group(1) if m else Path(ckpt).parent.name


def step_of(lbl: str) -> int:
    m = re.search(r"(\d+)$", lbl)
    return int(m.group(1)) if m else 0


def median_curve(episodes: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    """Median Q per second since the residual switched on (seconds with >= 2 episodes)."""
    secs = defaultdict(list)
    for ep in episodes:
        for k, r in enumerate(ep):
            secs[int(k / FPS)].append(r["q_mean"])
    xs = sorted(s for s, v in secs.items() if len(v) >= 2)
    return np.array(xs, float) + 0.5, np.array([np.median(secs[s]) for s in xs])


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.set_xlabel("seconds since the residual switched on", color=TEXT2, fontsize=9)
    ax.set_ylabel("Q (mean over critic heads)", color=TEXT2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT2, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_sessions(ARGS.logs, ARGS.run)
    if not data:
        raise SystemExit("no residual sessions with Q values matched")
    ckpts = sorted(data, key=lambda c: step_of(label(c)))
    rows = []
    for c in ckpts:
        eps, lbl = data[c], label(c)
        q = np.concatenate([[r["q_mean"] for r in ep] for ep in eps])
        res = np.concatenate([[r["residual"] for r in ep] for ep in eps])
        first = np.concatenate([[r["q_mean"] for r in ep[: int(FPS)]] for ep in eps])
        last = np.concatenate([[r["q_mean"] for r in ep[-int(FPS):]] for ep in eps])
        rows.append({"checkpoint": lbl, "episodes": len(eps), "ticks": len(q),
                     "q_median": round(float(np.median(q)), 4),
                     "q_first_second": round(float(np.median(first)), 4),
                     "q_last_second": round(float(np.median(last)), 4),
                     "mean_abs_residual": " ".join(f"{v:.3f}" for v in np.abs(res).mean(0)),
                     "share_positive": " ".join(f"{v:.2f}" for v in (res > 0).mean(0))})

        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
        for ep in eps:
            ax.plot(np.arange(len(ep)) / FPS, [r["q_mean"] for r in ep], color=EPISODE, linewidth=0.8)
        xs, med = median_curve(eps)
        if len(xs):
            ax.plot(xs, med, color=SERIES[0], linewidth=2, label="median per second")
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="best")
        style(ax, f"{lbl}: Q while the residual ran ({len(eps)} episodes, thin = one episode)")
        fig.tight_layout()
        fig.savefig(out / f"q_{lbl}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for i, c in enumerate(ckpts[: len(SERIES)]):
        xs, med = median_curve(data[c])
        if not len(xs):
            continue
        color = SERIES[i]
        ax.plot(xs, med, color=color, linewidth=2, label=label(c))
        ax.annotate(label(c), (xs[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=TEXT2, va="center")
    style(ax, "Median Q per second while the residual ran, per checkpoint")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT2, loc="best")
    fig.tight_layout()
    fig.savefig(out / "q_compare.png", dpi=130)
    plt.close(fig)
    if len(ckpts) > len(SERIES):
        print(f"q_compare.png shows the first {len(SERIES)} checkpoints only")

    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="logs/residual/*.jsonl")
    ap.add_argument("--run", default=None, help="Keep checkpoints whose path contains this text.")
    ap.add_argument("--gap-s", type=float, default=1.0, help="A pause longer than this starts a new episode.")
    ap.add_argument("--out", default="RealTrainLogs/q_plots")
    ARGS = ap.parse_args()
    main()
