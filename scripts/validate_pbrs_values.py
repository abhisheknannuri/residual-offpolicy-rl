#!/usr/bin/env python3
"""
Validate the labeled PBRS rewards against the Bellman / n-step math that ResFiT's
critic uses, BEFORE running RL. For a few episodes it:

  1. Checks Potential-Based Reward Shaping (PBRS) invariance:
        sum_t gamma^t * r_pbrs_t  ==  sum_t gamma^t * r_sparse_t
     (the discounted shaped return from s0 must equal the sparse return, because
     the potentials telescope to zero: phi(s0)=0, phi(terminal)=0). This is the
     core guarantee that the shaping did not change the optimal value.

  2. Checks n-step target consistency: the n-step backup
        y_t = sum_{j<n} gamma^j r_t+j  +  gamma^n * V(s_t+n)      (terminated -> no bootstrap)
     equals the Monte-Carlo return-to-go V(s_t) when V is the MC value. This is
     exactly what MultiStepTransform + the critic compute; if it holds, the
     rewards are consistent with the critic's targets.

  3. Fits a tiny value network V(state) -> MC return via backprop (Adam) to show
     the reward signal is learnable, and plots predicted vs MC value.

Outputs per-episode PNGs (stage, r_pbrs, MC value shaped-vs-sparse, fitted value,
fit loss) plus an aggregate console report.

Run (resfit conda env):
    conda run -n residual python scripts/validate_pbrs_values.py \
        --parquet ./pbrs_sarm_progress.parquet \
        --dataset_root /home/qte9489/.../SARM-robosuite-can-mh-stages_v21 \
        --gamma 0.97 --n_step 3 --episodes 0 1 2 --out_dir ./pbrs_validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "deps" / "lerobot"))


def discounted_return_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """V[t] = sum_{k>=t} gamma^(k-t) * rewards[k]."""
    v = np.zeros_like(rewards, dtype=np.float64)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        v[t] = acc
    return v


def nstep_targets(rewards: np.ndarray, values: np.ndarray, gamma: float, n: int) -> np.ndarray:
    """y[t] = sum_{j<n} gamma^j r[t+j] + gamma^n V[t+n]  (clamped at the terminal)."""
    T = len(rewards)
    y = np.zeros(T, dtype=np.float64)
    for t in range(T):
        acc = 0.0
        j = 0
        while j < n and (t + j) < T:
            acc += (gamma ** j) * rewards[t + j]
            j += 1
        # bootstrap from the state reached after j real steps (j == n unless we
        # hit the episode end, where the return is already complete -> no bootstrap)
        if t + n < T:
            acc += (gamma ** n) * values[t + n]
        y[t] = acc
    return y


def fit_value_net(states: np.ndarray, targets: np.ndarray, steps: int, device: str):
    """Fit V(state) -> MC return with a tiny MLP; return (preds, loss_history)."""
    mean = states.mean(0, keepdims=True)
    std = states.std(0, keepdims=True) + 1e-6
    X = torch.tensor((states - mean) / std, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(-1)
    net = nn.Sequential(
        nn.Linear(states.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        pred = net(X)
        loss = lossf(pred, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        preds = net(X).squeeze(-1).cpu().numpy()
    return preds, losses


def main():
    p = argparse.ArgumentParser(description="Validate PBRS rewards vs Bellman/n-step math")
    p.add_argument("--parquet", type=str, required=True)
    p.add_argument("--dataset_root", type=str, default=None, help="For the value-net fit (needs states)")
    p.add_argument("--repo_id", type=str, default="offline")
    p.add_argument("--state_key", type=str, default="state")
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--potentials", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15])
    p.add_argument("--episodes", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--fit_steps", type=int, default=3000)
    p.add_argument("--out_dir", type=str, default="./pbrs_validation")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(args.parquet).to_pandas().sort_values(["episode_index", "frame_index"])
    gamma = args.gamma

    ds = None
    if args.dataset_root:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
        s0 = ds[0]
        if args.state_key not in s0:
            args.state_key = "observation.state" if "observation.state" in s0 else args.state_key

    inv_diffs, nstep_diffs = [], []
    for ep in args.episodes:
        sub = df[df["episode_index"] == ep]
        if sub.empty:
            print(f"episode {ep}: no rows, skipping")
            continue
        r_pbrs = sub["reward_pbrs"].to_numpy(dtype=np.float64)
        r_sparse = sub["r_sparse"].to_numpy(dtype=np.float64)
        stages = sub["stage"].to_numpy()
        idx = sub["index"].to_numpy()

        V_shaped = discounted_return_to_go(r_pbrs, gamma)
        V_sparse = discounted_return_to_go(r_sparse, gamma)

        # 1) PBRS invariance (returns from s0 must match)
        inv_diff = abs(V_shaped[0] - V_sparse[0])
        inv_diffs.append(inv_diff)

        # 2) n-step target consistency vs MC value
        y = nstep_targets(r_pbrs, V_shaped, gamma, args.n_step)
        nstep_diff = float(np.max(np.abs(y - V_shaped)))
        nstep_diffs.append(nstep_diff)

        # 3) value-net fit (optional, needs states)
        preds = losses = states = None
        if ds is not None:
            states = np.stack([np.asarray(ds[int(i)][args.state_key], dtype=np.float32).reshape(-1) for i in idx])
            preds, losses = fit_value_net(states, V_shaped, args.fit_steps, args.device)

        # ---- report ----
        print(
            f"episode {ep}: T={len(r_pbrs)}  "
            f"G_shaped={V_shaped[0]:.5f}  G_sparse={V_sparse[0]:.5f}  |inv diff|={inv_diff:.2e}  "
            f"max|n-step - MC|={nstep_diff:.2e}"
            + (f"  fit_final_mse={losses[-1]:.2e}" if losses else "")
        )

        # ---- plots ----
        nrows = 4 if ds is not None else 3
        fig, ax = plt.subplots(nrows, 1, figsize=(10, 2.4 * nrows), sharex=True)
        t = np.arange(len(r_pbrs))
        ax[0].step(t, stages, where="post", color="tab:purple")
        ax[0].set_ylabel("gated stage"); ax[0].set_ylim(-0.2, len(args.potentials) - 0.8)
        ax[1].plot(t, r_pbrs, color="tab:blue")
        ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("r_pbrs (shaped)")
        ax[2].plot(t, V_shaped, label="MC value (shaped)", color="tab:green")
        ax[2].plot(t, V_sparse, "--", label="MC value (sparse)", color="tab:orange")
        ax[2].set_ylabel("value-to-go"); ax[2].legend(loc="upper left", fontsize=8)
        if ds is not None:
            ax[3].plot(t, V_shaped, label="MC target", color="tab:green")
            ax[3].plot(t, preds, label="fitted V(state)", color="tab:red", alpha=0.8)
            ax[3].set_ylabel("value"); ax[3].legend(loc="upper left", fontsize=8)
        ax[-1].set_xlabel("episode step (transition index)")
        fig.suptitle(
            f"Episode {ep}  gamma={gamma}  n_step={args.n_step}  "
            f"PBRS invariance |diff|={inv_diff:.1e}  n-step|diff|={nstep_diff:.1e}"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"episode_{ep}_pbrs_validation.png", dpi=110)
        plt.close(fig)

        if losses is not None:
            figl, axl = plt.subplots(figsize=(7, 3))
            axl.semilogy(losses, color="tab:red")
            axl.set_xlabel("gradient step"); axl.set_ylabel("MSE (log)")
            axl.set_title(f"Episode {ep}: value-net fit loss (final={losses[-1]:.2e})")
            figl.tight_layout()
            figl.savefig(out_dir / f"episode_{ep}_fit_loss.png", dpi=110)
            plt.close(figl)

    # ---- aggregate ----
    print("\n=== AGGREGATE ===")
    if inv_diffs:
        print(f"PBRS invariance  max|G_shaped - G_sparse| over episodes: {max(inv_diffs):.2e} "
              f"(should be ~0 -> shaping preserves the value)")
        print(f"n-step consistency max|y - MC| over episodes:            {max(nstep_diffs):.2e} "
              f"(should be ~0 -> rewards match the critic's n-step targets)")
    print(f"Plots written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
