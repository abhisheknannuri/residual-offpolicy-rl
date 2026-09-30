#!/usr/bin/env python3
"""
evaluate_checkpoints.py

Evaluates every checkpoint saved by a critic-warmup and/or offline-RL run
(train_residual_rl_offline_pretrain.sh) against a FIXED set of sampled real
episodes (some from the offline buffer, some from the online buffer), and
plots how the critic's Q-value estimate (and, for checkpoints where the actor
was actually trained, the actor's predicted action vs. the ground-truth stored
action) evolve across checkpoints.

EVERY TIMESTEP of every sampled episode is evaluated, start to end - the full
span, no sub-sampling of frames or timesteps anywhere in this script.

Q-values are computed with EXACTLY ONE call to `agent.critic.q_value()` per
state/action pair - the SAME single call, same random 2-of-10 (`min_q_heads`-
of-`num_q`) head draw, that `run_dexmg_evaluation()` uses for the real Q-value
plots wandb sees during actual training (`resfit/rl_finetuning/utils/
evaluate_dexmg.py:275-287` - `agent.act()` for the action, `agent._encode()`
for the features, one `agent.critic.q_value(feat, prop, action)` call). No
averaging over repeated draws, no smoothing - an earlier version of this
script called `q_value()` multiple times per pair and averaged the results to
produce a less noisy-looking plot; that was WRONG, because it doesn't match
what training's own evaluation reports, and has been removed. Whatever
randomness `q_value()`'s own head-subset draw has is reported as-is, exactly
like it is everywhere else in this codebase.

Each checkpoint's own saved config (`checkpoint.pt`'s `"config"` field) is used
to reconstruct the exact QAgent architecture that checkpoint was trained
with - no hyperparameters are hardcoded/assumed here.

Episode boundaries are recovered from `next.done` runs of True (see
scripts/TrossenStation1Real/BUFFER_POPULATION_ARCHITECTURE.md for why this
column is a WIDE, multi-frame-per-episode region rather than a single frame) -
an episode boundary is the last True in a contiguous run (the next entry, if
any, starts a new episode with done=False). Episodes without a genuine
trailing True (i.e. truncated before ever reaching one) are excluded. Once an
episode is chosen, its FULL span (every frame from the episode's first frame
to its last) is evaluated - there is no sub-sampling of frames within it.

REAL COMMAND, evaluating the critic-warmup checkpoints from the run on 2026-08-30
(the exact command used to produce this run's own eval_plots/ - reuse verbatim,
just swap --models_dir for a different run's models/ folder, e.g. once an
offline-RL run also exists - discovery of critic_warmup_* vs offline_step_*
checkpoints under --models_dir is automatic, so this is the SAME command for
both kinds of runs, not a different one per phase):

    python scripts/TrossenStation1Real/evaluate_checkpoints.py \\
        --models_dir "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/run_2026-08-30_17-45-02_2026-08-30_17-45-02__trossen_real_n3_utd4_buf70000_off120ep_lr1e-06__seed42/models" \\
        --offline_buffer_cache "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/offline_buffer_cache/b7bfa8d3" \\
        --online_buffer_cache "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/online_buffer_cache/c3aeae2b"

Full flag set (all shown with their defaults - append any of these to override):
    python scripts/TrossenStation1Real/evaluate_checkpoints.py \\
        --models_dir "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/run_2026-08-30_17-45-02_2026-08-30_17-45-02__trossen_real_n3_utd4_buf70000_off120ep_lr1e-06__seed42/models" \\
        --offline_buffer_cache "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/offline_buffer_cache/b7bfa8d3" \\
        --online_buffer_cache "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/online_buffer_cache/c3aeae2b" \\
        --num_offline_episodes 5 --num_online_episodes 5 \\
        --seed 42 --device cuda

Output (written to <models_dir>/../eval_plots/ by default - i.e. for the exact
command above, into ".../run_2026-08-30_17-45-02_..._seed42/eval_plots/"):
    - eval_q_trajectories_<checkpoint_label>.png (+ matching .json)  <- THE plot:
      one per checkpoint, EXACTLY reproducing evaluate_dexmg.py's own
      _create_q_trajectory_plots() (the real code behind wandb's `value/q_trajectories`
      during actual training) - top panel: every sampled episode's Q(GT action)
      trajectory overlaid, green=success/red=failure; bottom panel: box plot of
      Q-values at 25/50/75/100% episode progress.
    - critic_q_vs_checkpoint.png, actor_vs_gt_mse_vs_checkpoint.png,
      q_actor_vs_checkpoint.png: bonus trend-across-checkpoints summaries this
      script adds on top (not part of the original training-time plots) - useful
      for seeing the aggregate direction at a glance across many checkpoints.
    - metrics.json  (every raw per-checkpoint, per-episode number, for further analysis)


python scripts/TrossenStation1Real/evaluate_checkpoints.py \
        --models_dir "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/run_2026-08-30_17-45-02_2026-08-30_17-45-02__trossen_real_n3_utd4_buf70000_off120ep_lr1e-06__seed42/models" \
        --offline_buffer_cache "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/offline_buffer_cache/b7bfa8d3" \
        --online_buffer_cache "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/online_buffer_cache/c3aeae2b" \
        --num_offline_episodes 5 --num_online_episodes 5 \
        --seed 42 --device cuda
        
        



python scripts/TrossenStation1Real/evaluate_checkpoints.py \
        --models_dir "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/run_2026-08-30_19-00-23_2026-08-30_19-00-21__trossen_real_n3_utd4_buf70000_off120ep_lr1e-06__seed42/models/" \
        --offline_buffer_cache "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/offline_buffer_cache/b7bfa8d3" \
        --online_buffer_cache "/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/online_buffer_cache/c3aeae2b" \
        --num_offline_episodes 5 --num_online_episodes 5 \
        --seed 42 --device cuda

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_replay_buffer import load_valid  # noqa: E402
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent  # noqa: E402

# Fixed 2-color categorical palette (offline vs online) - not cycled, assigned
# by identity, distinct hue+lightness (validated informally: blue vs orange,
# a colorblind-safe pair with good separation in both CVD and normal vision).
COLOR_OFFLINE = "#1f6feb"   # blue
COLOR_ONLINE = "#e8730a"    # orange
COLOR_REWARD = "#8a8f98"    # muted gray (reference series, not a primary one)


# ============================================================================
# Episode extraction
# ============================================================================

def find_episode_spans(done: torch.Tensor) -> list[tuple[int, int]]:
    """Returns [(start, end), ...] inclusive index pairs, one per episode that
    ends in a genuine done=True run. `done` is the buffer's stored (possibly
    WIDE) next.done column, 1-D bool tensor in temporal order."""
    done_np = done.cpu().numpy().astype(bool)
    n = len(done_np)
    spans = []
    start = 0
    i = 0
    while i < n:
        if done_np[i]:
            # walk to the end of this contiguous True run
            j = i
            while j + 1 < n and done_np[j + 1]:
                j += 1
            spans.append((start, j))
            start = j + 1
            i = j + 1
        else:
            i += 1
    return spans


def sample_episodes(cache_dir: str, n: int, seed: int, source: str) -> list[dict]:
    """Loads the buffer at cache_dir, finds all episode spans, samples `n` of
    them (fixed seed - same episodes reused across every checkpoint), and
    returns a list of dicts with the raw per-timestep tensors each episode
    needs for evaluation (kept on CPU; moved to device per-checkpoint)."""
    td, cursor, raw_len = load_valid(cache_dir)
    spans = find_episode_spans(td["next", "done"])
    if len(spans) == 0:
        raise RuntimeError(f"No complete episodes (next.done run) found in {cache_dir}")
    rng = np.random.default_rng(seed)
    n = min(n, len(spans))
    chosen_idx = rng.choice(len(spans), size=n, replace=False)
    episodes = []
    for idx in sorted(chosen_idx.tolist()):
        s, e = spans[idx]
        ep_td = td[s : e + 1]
        episodes.append(
            {
                "source": source,
                "span": (s, e),
                "length": e - s + 1,
                "obs": {k: ep_td["obs", k].clone() for k in ep_td["obs"].keys()},
                "action": ep_td["action"].clone(),
                "reward": ep_td["next", "original_reward"].clone(),
                "done": ep_td["next", "done"].clone(),
            }
        )
    print(f"  {source}: {len(spans)} complete episodes found in cache, sampled {n} "
          f"(lengths: {[e['length'] for e in episodes]})")
    return episodes


# ============================================================================
# Agent reconstruction from a checkpoint's own saved config
# ============================================================================

def build_agent_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[QAgent, dict]:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" not in raw or raw["config"] is None:
        raise RuntimeError(f"{ckpt_path} has no saved 'config' - cannot reconstruct the agent architecture.")
    cfg = OmegaConf.create(raw["config"])

    # Same derivation as train_residual_td3.py's offline_pretrain_only branch -
    # no env object involved, matches what TrossenResidualEnv itself would compute.
    action_dim = 7
    lowdim_dim = action_dim
    img_c = 3
    img_h = img_w = int(cfg.offline_data.image_size) if cfg.offline_data.image_size is not None else 256
    image_keys = [cfg.rl_camera] if isinstance(cfg.rl_camera, str) else list(cfg.rl_camera)

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    ).to(device)
    agent.load_state_dict(raw["agent_state_dict"])
    agent.eval()
    return agent, {
        "config": cfg,
        "image_keys": image_keys,
        "global_step": raw.get("global_step", 0),
        "actor_updates": raw.get("actor_updates", 0),
    }


# ============================================================================
# Per-episode evaluation
# ============================================================================

def evaluate_episode(agent: QAgent, image_keys: list[str], episode: dict, device: torch.device) -> dict:
    obs = {}
    for k in image_keys:
        img = episode["obs"][k].to(device)
        obs[k] = img.float().div_(255.0) if img.dtype == torch.uint8 else img.float()
    obs["observation.state"] = episode["obs"]["observation.state"].to(device).float()
    obs["observation.base_action"] = episode["obs"]["observation.base_action"].to(device).float()
    gt_action = episode["action"].to(device).float()

    with torch.no_grad():
        # Exactly evaluate_dexmg.py:275-287 (run_dexmg_evaluation - the real code that
        # produces the wandb Q-plots during actual training) - agent.act() for the
        # action, agent._encode() for feat, ONE single agent.critic.q_value() call per
        # state/action pair. No averaging, no repeated sampling - a single q_value()
        # call IS what training-time evaluation reports, exactly as-is, randomness
        # included.
        feat = agent._encode(obs, augment=False)  # noqa: SLF001 - internal, but this is an eval/inspection tool
        prop = obs["observation.state"]

        pred_residual = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)
        pred_combined = torch.clamp(obs["observation.base_action"] + pred_residual, -1.0, 1.0)

        q_gt = agent.critic.q_value(feat, prop, gt_action).squeeze(-1).cpu().numpy()
        q_actor = agent.critic.q_value(feat, prop, pred_combined).squeeze(-1).cpu().numpy()
        actor_residual_np = pred_residual.cpu().numpy()
        actor_combined_np = pred_combined.cpu().numpy()
        gt_action_np = gt_action.cpu().numpy()
        per_step_mse = ((actor_combined_np - gt_action_np) ** 2).mean(axis=-1)

    return {
        "source": episode["source"],
        "span": episode["span"],
        "length": episode["length"],
        "q_gt": q_gt.tolist(),
        "q_actor": q_actor.tolist(),
        "reward": episode["reward"].cpu().numpy().tolist(),
        "actor_vs_gt_mse": per_step_mse.tolist(),
        "actor_residual_abs_mean": float(np.abs(actor_residual_np).mean()),
        "mean_q_gt": float(q_gt.mean()),
        "mean_q_actor": float(q_actor.mean()),
        "mean_actor_vs_gt_mse": float(per_step_mse.mean()),
    }


# ============================================================================
# Checkpoint discovery / ordering
# ============================================================================

def discover_checkpoints(models_dir: Path) -> list[tuple[str, int, Path]]:
    """Returns [(label, sort_key, checkpoint.pt path), ...] sorted for plotting.
    sort_key groups critic-warmup checkpoints before offline-RL ones (they're
    mutually exclusive per run in practice, but this stays sane if both exist),
    then orders by step number within each group; "*_final"/"*_best" sort last
    within their group."""
    entries = []
    for ckpt_path in sorted(models_dir.rglob("checkpoint.pt")):
        label = ckpt_path.parent.name
        m = re.search(r"_step_(\d+)$", label)
        if m:
            step = int(m.group(1))
        elif label.endswith("_final"):
            step = 10**9
        elif label.endswith("_best"):
            step = 10**9 + 1
        else:
            step = -1
        phase_rank = 0 if label.startswith("critic_warmup") else 1
        entries.append((label, phase_rank * 10**10 + step, ckpt_path))
    entries.sort(key=lambda x: x[1])
    return entries


# ============================================================================
# Plotting (matplotlib, static PNGs - see module docstring)
# ============================================================================

def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_metric_vs_checkpoint(results: list[dict], key: str, ylabel: str, title: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    offline_vals = [r[f"offline_{key}"] for r in results]
    online_vals = [r[f"online_{key}"] for r in results]

    import textwrap
    fig_width = max(7.5, 1.1 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    _style_axes(ax)
    ax.plot(x, offline_vals, marker="o", color=COLOR_OFFLINE, linewidth=2, markersize=6, label="offline episodes")
    ax.plot(x, online_vals, marker="o", color=COLOR_ONLINE, linewidth=2, markersize=6, label="online episodes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    wrap_width = max(40, int(fig_width * 8))
    ax.set_title("\n".join(textwrap.wrap(title, wrap_width)), fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def create_q_trajectory_plot_like_training(episode_results: list[dict], checkpoint_label: str, out_path: Path,
                                            global_step: int | None = None) -> None:
    """Reproduces `_create_q_trajectory_plots()` from `evaluate_dexmg.py:91-150`
    EXACTLY - same 2-panel figure, same colors, same labels, same box-plot progress
    points (25/50/75/100%) - this is the actual plot real training's periodic
    evaluation logs to wandb as `value/q_trajectories`, not a different chart
    invented for this script. `q_gt` (Q of the action ACTUALLY taken/recorded at
    each buffer timestep) is used as the trajectory - the direct analog of the
    original's `q_pred` (Q of whatever action was actually taken that live-rollout
    step). "success" is derived as "did this episode's raw reward ever hit 1.0"
    (the closest available substitute for the original's env-reported `info["success"]`,
    which real hardware's TrossenResidualEnv/replay buffer doesn't carry)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectories = [ep["q_gt"] for ep in episode_results]
    successes = [bool(np.max(ep["reward"]) >= 1.0) for ep in episode_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    successful_trajs = [traj for traj, s in zip(trajectories, successes) if s]
    failed_trajs = [traj for traj, s in zip(trajectories, successes) if not s]

    for i, traj in enumerate(successful_trajs):
        steps = list(range(len(traj)))
        ax1.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label="Success" if i == 0 else "")
    for i, traj in enumerate(failed_trajs):
        steps = list(range(len(traj)))
        ax1.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label="Failure" if i == 0 else "")

    ax1.set_xlabel("Episode Step")
    ax1.set_ylabel("Q-Value")
    ax1.set_title(f"Q-Value Trajectories Over Time (checkpoint '{checkpoint_label}', step {global_step or 'N/A'})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    progress_points = [0.25, 0.5, 0.75, 1.0]
    q_values_at_progress = {f"{int(p * 100)}%": [] for p in progress_points}
    for traj in trajectories:
        traj_len = len(traj)
        for p in progress_points:
            step_idx = min(int(p * traj_len), traj_len - 1)
            if step_idx < len(traj):
                q_values_at_progress[f"{int(p * 100)}%"].append(traj[step_idx])

    box_data = [q_values_at_progress[f"{int(p * 100)}%"] for p in progress_points]
    box_labels = [f"{int(p * 100)}%" for p in progress_points]
    ax2.boxplot(box_data, tick_labels=box_labels)
    ax2.set_xlabel("Episode Progress")
    ax2.set_ylabel("Q-Value")
    ax2.set_title("Q-Value Distribution at Different Episode Progress Points")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    data_path = out_path.with_suffix(".json")
    dump_data = {
        "checkpoint": checkpoint_label,
        "global_step": global_step,
        "episodes": [
            {
                "episode_idx": i + 1,
                "source": ep["source"],
                "span": ep["span"],
                "success": successes[i],
                "length": ep["length"],
                "q_trajectory": ep["q_gt"],
            }
            for i, ep in enumerate(episode_results)
        ],
    }
    with open(data_path, "w") as f:
        json.dump(dump_data, f, indent=2)


def plot_all_checkpoints_q_trajectories(all_raw: dict, checkpoint_order: list[str], out_path: Path) -> None:
    """ONE combined figure, one row per checkpoint (in training order, top to
    bottom), each row reproducing the SAME top-panel trajectory-overlay style as
    create_q_trajectory_plot_like_training() (green=success/red=failure) - lets
    you compare every checkpoint's Q-shape at a glance instead of opening each
    per-checkpoint PNG separately. All rows share the same y-axis so the shapes
    are directly, visually comparable across checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(checkpoint_order)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=False, sharey=True)
    if n == 1:
        axes = [axes]

    # Shared y-limits across every row, computed from every trajectory in every
    # checkpoint, so a flatter/lower line in an early checkpoint is visibly
    # different from a taller one later - not an artifact of independent scaling.
    all_vals = [v for label in checkpoint_order
                for ep in (all_raw[label]["offline_episodes"] + all_raw[label]["online_episodes"])
                for v in ep["q_gt"]]
    y_lo, y_hi = (min(all_vals), max(all_vals)) if all_vals else (0.0, 1.0)
    pad = 0.05 * (y_hi - y_lo if y_hi > y_lo else 1.0)

    for ax, label in zip(axes, checkpoint_order):
        episode_results = all_raw[label]["offline_episodes"] + all_raw[label]["online_episodes"]
        trajectories = [ep["q_gt"] for ep in episode_results]
        successes = [bool(np.max(ep["reward"]) >= 1.0) for ep in episode_results]

        for i, (traj, s) in enumerate(zip(trajectories, successes)):
            if s:
                ax.plot(traj, "g-", alpha=0.6, linewidth=1)
            else:
                ax.plot(traj, "r-", alpha=0.6, linewidth=1)

        n_succ = sum(successes)
        ax.set_ylabel("Q-Value", fontsize=9)
        ax.set_title(f"'{label}'  ({n_succ}/{len(successes)} sampled episodes classified success)", fontsize=10)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode Step")
    fig.suptitle("Q-Value Trajectories Across ALL Checkpoints (green=success, red=failure - same episodes, same y-scale, every row)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_trajectory_qvalues(episode_results: list[dict], title: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(episode_results)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    for i, ep in enumerate(episode_results):
        ax = axes[i // ncols][i % ncols]
        _style_axes(ax)
        t = np.arange(ep["length"])
        color = COLOR_OFFLINE if ep["source"] == "offline" else COLOR_ONLINE
        ax.plot(t, ep["q_gt"], color=color, linewidth=1.8, label="Q(GT action)")
        ax.plot(t, ep["reward"], color=COLOR_REWARD, linewidth=1.2, linestyle="--", label="raw reward")
        ax.set_title(f"{ep['source']} ep @ idx {ep['span'][0]}", fontsize=9)
        ax.set_xlabel("timestep", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("value", fontsize=8)
        if i == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models_dir", required=True, type=Path)
    ap.add_argument("--offline_buffer_cache", required=True, type=Path)
    ap.add_argument("--online_buffer_cache", required=True, type=Path)
    ap.add_argument("--num_offline_episodes", type=int, default=5)
    ap.add_argument("--num_online_episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42, help="episode sampling seed - SAME episodes reused across every checkpoint")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir", type=Path, default=None,
                     help="default: <models_dir>/../eval_plots (i.e. inside the run folder, sibling to models/)")
    args = ap.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir or (args.models_dir.parent / "eval_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling episodes (seed={args.seed}, reused for every checkpoint)...")
    offline_episodes = sample_episodes(str(args.offline_buffer_cache), args.num_offline_episodes, args.seed, "offline")
    online_episodes = sample_episodes(str(args.online_buffer_cache), args.num_online_episodes, args.seed, "online")

    checkpoints = discover_checkpoints(args.models_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoint.pt found anywhere under {args.models_dir}")
    print(f"\nFound {len(checkpoints)} checkpoints: {[c[0] for c in checkpoints]}")

    per_checkpoint_results = []
    all_raw = {}
    for label, _sort_key, ckpt_path in checkpoints:
        print(f"\nEvaluating checkpoint: {label} ({ckpt_path})")
        agent, meta = build_agent_from_checkpoint(ckpt_path, device)
        image_keys = meta["image_keys"]

        offline_ep_results = [evaluate_episode(agent, image_keys, ep, device) for ep in offline_episodes]
        online_ep_results = [evaluate_episode(agent, image_keys, ep, device) for ep in online_episodes]

        def agg(results, key):
            return float(np.mean([r[key] for r in results]))

        row = {
            "label": label,
            "global_step": meta["global_step"],
            "actor_updates": meta["actor_updates"],
            "offline_mean_q_gt": agg(offline_ep_results, "mean_q_gt"),
            "online_mean_q_gt": agg(online_ep_results, "mean_q_gt"),
            "offline_mean_q_actor": agg(offline_ep_results, "mean_q_actor"),
            "online_mean_q_actor": agg(online_ep_results, "mean_q_actor"),
            "offline_mean_actor_vs_gt_mse": agg(offline_ep_results, "mean_actor_vs_gt_mse"),
            "online_mean_actor_vs_gt_mse": agg(online_ep_results, "mean_actor_vs_gt_mse"),
        }
        per_checkpoint_results.append(row)
        all_raw[label] = {"offline_episodes": offline_ep_results, "online_episodes": online_ep_results, "meta": {
            "global_step": meta["global_step"], "actor_updates": meta["actor_updates"],
        }}
        print(f"  mean_q_gt: offline={row['offline_mean_q_gt']:.4f} online={row['online_mean_q_gt']:.4f} | "
              f"mean_q_actor: offline={row['offline_mean_q_actor']:.4f} online={row['online_mean_q_actor']:.4f} | "
              f"actor_vs_gt_mse: offline={row['offline_mean_actor_vs_gt_mse']:.5f} online={row['online_mean_actor_vs_gt_mse']:.5f}")

        # THE plot: reproduces evaluate_dexmg.py's own _create_q_trajectory_plots()
        # exactly - one per checkpoint, same as real training logs one per periodic
        # eval call. This is "the predicted Q value plots" from the original script.
        create_q_trajectory_plot_like_training(
            offline_ep_results + online_ep_results,
            checkpoint_label=label,
            out_path=output_dir / f"eval_q_trajectories_{label}.png",
            global_step=meta["global_step"],
        )

    print("\nPlotting...")

    plot_metric_vs_checkpoint(
        per_checkpoint_results,
        key="mean_q_gt", ylabel="mean Q(GT action)",
        title="Critic value estimate for the ACTUAL executed action, across checkpoints",
        out_path=output_dir / "critic_q_vs_checkpoint.png",
    )
    plot_metric_vs_checkpoint(
        per_checkpoint_results, key="mean_actor_vs_gt_mse", ylabel="mean squared error",
        title="Actor predicted action vs. GT/stored action, across checkpoints",
        out_path=output_dir / "actor_vs_gt_mse_vs_checkpoint.png",
    )
    plot_metric_vs_checkpoint(
        per_checkpoint_results, key="mean_q_actor", ylabel="mean Q(actor's own predicted action)",
        title="Critic value estimate for the ACTOR'S OWN proposal, across checkpoints",
        out_path=output_dir / "q_actor_vs_checkpoint.png",
    )

    latest_label = checkpoints[-1][0]
    plot_trajectory_qvalues(
        all_raw[latest_label]["offline_episodes"] + all_raw[latest_label]["online_episodes"],
        title=f"Per-timestep Q(GT action) vs. raw reward - checkpoint '{latest_label}'",
        out_path=output_dir / "trajectory_qvalues_latest_checkpoint.png",
    )

    # THE combined comparison view - every checkpoint's trajectories in one figure,
    # same y-scale, training order top to bottom.
    plot_all_checkpoints_q_trajectories(
        all_raw, checkpoint_order=[label for label, _sk, _p in checkpoints],
        out_path=output_dir / "eval_q_trajectories_ALL_CHECKPOINTS_combined.png",
    )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"per_checkpoint_summary": per_checkpoint_results, "raw": all_raw}, f, indent=2)
    print(f"\nSaved raw metrics to {metrics_path}")
    print(f"All outputs in {output_dir}")


if __name__ == "__main__":
    main()
