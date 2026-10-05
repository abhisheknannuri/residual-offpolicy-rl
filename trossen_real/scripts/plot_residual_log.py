# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Read a residual tick log and answer: is the RL doing anything?

    .venv/bin/python -m trossen_real.scripts.plot_residual_log \
        --log logs/residual/<run>.jsonl --out logs/residual/<run>_analysis

Four plots, each chosen because it answers a question you would otherwise have
to guess at:

  1. residual magnitude per joint over time, against the action_scale bound.
     The bound matters: a residual pinned at its limit is being clipped by its
     own configuration, not choosing that value.
  2. base vs final action per joint. If these overlay exactly, the RL is doing
     nothing regardless of what the residual array says.
  3. Q over time with the min/max band across the critic heads. The BAND is the
     interesting part for offline RL - wide spread means the critics disagree
     about states the policy is actually visiting.
  4. a summary panel: per-joint residual as a fraction of bound, and how much
     the [-1,1] clamp threw away.

Prints a verdict too, so the headline does not depend on reading a chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    ticks: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "load":
            meta = rec
        elif rec.get("event") == "tick":
            ticks.append(rec)
    if not ticks:
        raise SystemExit(f"no tick records in {path}")
    return meta, ticks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: alongside the log)")
    ap.add_argument("--joint-names", default=None,
                    help="comma separated; default joint_0..joint_N")
    args = ap.parse_args()

    log = Path(args.log)
    out = Path(args.out) if args.out else log.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    meta, ticks = load(log)

    R = np.array([t["residual"] for t in ticks])
    B = np.array([t["base_action"] for t in ticks])
    A = np.array([t["action"] for t in ticks])
    clipped = np.array([t.get("clipped_amount", 0.0) for t in ticks])
    n = R.shape[1]
    names = (args.joint_names.split(",") if args.joint_names
             else meta.get("joint_names") or [f"joint_{i}" for i in range(n)])
    bound = np.asarray(meta.get("action_scale") or [np.nan] * n, dtype=float)

    have_q = "q_mean" in ticks[0]
    if have_q:
        qm = np.array([t["q_mean"] for t in ticks])
        qh = np.array([t["q_heads"] for t in ticks])

    # ---------------- verdict, printed ----------------
    frac = np.abs(R).max(0) / bound if np.isfinite(bound).all() else np.full(n, np.nan)
    print(f"=== {log.name}: {len(ticks)} ticks ===")
    if meta:
        print(f"  checkpoint      {meta.get('checkpoint')}")
        print(f"  actor_updates   {meta.get('actor_updates')}  device={meta.get('device')}")
        print(f"  base sha        {str(meta.get('expected_base_sha256'))[:16]}... "
              f"verified={meta.get('expected_base_sha256') == meta.get('server_base_sha256')}")
    print()
    all_zero = bool(np.all(R == 0))
    varies = bool(R.std(0).max() > 1e-9)
    print(f"  RESIDUAL ALL ZERO?      {all_zero}")
    print(f"  varies with observation? {varies}  (max per-joint std {R.std(0).max():.6f})")
    print(f"  residual abs max        {np.abs(R).max():.6f}")
    print(f"  final action != base?   {not np.allclose(A, B)}")
    print(f"  clamp threw away (max)  {clipped.max():.6f}")
    print()
    print(f"  {'joint':<12}{'|res| mean':>12}{'|res| max':>12}{'bound':>10}{'frac of bound':>15}")
    for i, nm in enumerate(names[:n]):
        print(f"  {nm:<12}{np.abs(R[:, i]).mean():>12.5f}{np.abs(R[:, i]).max():>12.5f}"
              f"{bound[i]:>10.3f}{frac[i]:>15.1%}")
    if have_q:
        print()
        print(f"  Q mean {qm.mean():.4f}  range [{qm.min():.4f}, {qm.max():.4f}]")
        print(f"  ensemble spread (std across {qh.shape[1]} heads), mean {qh.std(1).mean():.4f}")

    saturated = [names[i] for i in range(n) if np.isfinite(frac[i]) and frac[i] > 0.9]
    if all_zero:
        print("\n  VERDICT: the residual is identically zero - the RL is contributing nothing.")
    elif not varies:
        print("\n  VERDICT: the residual is a CONSTANT offset - it is not responding to "
              "observations.")
    elif saturated:
        print(f"\n  VERDICT: the residual is active, but {', '.join(saturated)} reach >90% of "
              f"the action_scale bound.\n           The bound is likely limiting the policy "
              f"rather than the policy choosing that value.")
    else:
        print("\n  VERDICT: the residual is active and observation-dependent, within its bound.")

    # ---------------- plots ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n  (matplotlib not installed; numbers above only)")
        return

    t = np.arange(len(ticks))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, nm in enumerate(names[:n]):
        ax.plot(t, R[:, i], lw=1.1, label=nm)
    if np.isfinite(bound).all():
        for i in range(n):
            ax.axhline(bound[i], color="k", lw=.4, alpha=.25)
            ax.axhline(-bound[i], color="k", lw=.4, alpha=.25)
    ax.set_title("Residual per joint (thin grey lines = action_scale bound)")
    ax.set_xlabel("tick"); ax.set_ylabel("residual (normalised units)")
    ax.legend(ncol=4, fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "residual.png", dpi=150); plt.close(fig)

    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.1 * rows), squeeze=False)
    for i, nm in enumerate(names[:n]):
        a = axes[i // 2][i % 2]
        a.plot(t, B[:, i], lw=1.0, label="base BC")
        a.plot(t, A[:, i], lw=1.0, label="base + residual")
        a.set_title(nm, fontsize=9); a.grid(alpha=.2); a.tick_params(labelsize=7)
        if i == 0:
            a.legend(fontsize=7)
    for j in range(n, rows * 2):
        axes[j // 2][j % 2].axis("off")
    fig.suptitle("Base vs final action - overlapping lines mean the RL changed nothing",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out / "base_vs_final.png", dpi=150); plt.close(fig)

    if have_q:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(t, qh.min(1), qh.max(1), alpha=.25,
                        label="min-max across critic heads")
        ax.plot(t, qm, lw=1.3, label="Q mean")
        ax.set_title("Critic Q over time - the BAND is the ensemble disagreement")
        ax.set_xlabel("tick"); ax.set_ylabel("Q"); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(out / "q_values.png", dpi=150); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(n)
    a1.bar(x, frac * 100, color=["#f85149" if f > .9 else "#58a6ff" for f in frac])
    a1.axhline(100, color="k", lw=.8, ls="--")
    a1.set_xticks(x); a1.set_xticklabels(names[:n], rotation=45, ha="right", fontsize=8)
    a1.set_ylabel("% of action_scale"); a1.set_title("Peak residual vs its bound")
    a1.grid(alpha=.2, axis="y")
    a2.plot(t, clipped, lw=1.1, color="#d29922")
    a2.set_title("Amount the [-1,1] clamp discarded"); a2.set_xlabel("tick")
    a2.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "summary.png", dpi=150); plt.close(fig)

    print(f"\n  wrote {len(list(out.glob('*.png')))} plots to {out}")


if __name__ == "__main__":
    main()
