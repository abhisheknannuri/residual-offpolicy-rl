"""`_create_q_trajectory_plots`, copied VERBATIM from resfit.

Source: `resfit/rl_finetuning/utils/evaluate_dexmg.py`, where it is nested
inside `run_dexmg_evaluation()` and therefore cannot be imported. It is the
function whose output is logged to W&B as `value/q_trajectories`.

The body below is byte-identical to the original after dedenting - extracted
with `ast`, not retyped. `verify()` re-extracts from the source and compares,
so if the original changes this copy stops being silently stale:

    .venv/bin/python -m trossen_real.scripts._resfit_q_plot

DO NOT EDIT THE FUNCTION. The whole point is that the plots produced here are
the same plots the training-time evaluation produces.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SOURCE = "resfit/rl_finetuning/utils/evaluate_dexmg.py"
FUNC = "_create_q_trajectory_plots"


# --- BEGIN VERBATIM COPY -------------------------------------------------
def _create_q_trajectory_plots(
    trajectories: list[list[float]],
    episode_lengths: list[int],
    successes: list[bool],
    output_path: Path,
    global_step: int | None = None,
) -> None:
    """Create Q-value trajectory plots for all episodes."""
    if not trajectories:
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: All Q-trajectories over time
    # Separate successful and failed episodes
    successful_trajs = [traj for i, traj in enumerate(trajectories) if successes[i]]
    failed_trajs = [traj for i, traj in enumerate(trajectories) if not successes[i]]

    # Plot all trajectories with different colors for success/failure
    for i, traj in enumerate(successful_trajs):
        steps = list(range(len(traj)))
        ax1.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label="Success" if i == 0 else "")

    for i, traj in enumerate(failed_trajs):
        steps = list(range(len(traj)))
        ax1.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label="Failure" if i == 0 else "")

    ax1.set_xlabel("Episode Step")
    ax1.set_ylabel("Q-Value")
    ax1.set_title(f"Q-Value Trajectories Over Time (Step {global_step or 'N/A'})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Q-value distribution at different episode progress points
    progress_points = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100% of episode
    q_values_at_progress = {f"{int(p * 100)}%": [] for p in progress_points}

    for traj in trajectories:
        traj_len = len(traj)
        for p in progress_points:
            step_idx = min(int(p * traj_len), traj_len - 1)
            if step_idx < len(traj):
                q_values_at_progress[f"{int(p * 100)}%"].append(traj[step_idx])

    # Create box plot
    box_data = [q_values_at_progress[f"{int(p * 100)}%"] for p in progress_points]
    box_labels = [f"{int(p * 100)}%" for p in progress_points]

    ax2.boxplot(box_data, labels=box_labels)
    ax2.set_xlabel("Episode Progress")
    ax2.set_ylabel("Q-Value")
    ax2.set_title("Q-Value Distribution at Different Episode Progress Points")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved Q-trajectory plots to: {output_path}")
# --- END VERBATIM COPY ---------------------------------------------------


def extract_from_source(repo: str | Path = ".") -> str:
    """Re-extract the original, dedented, for comparison."""
    import ast

    src = (Path(repo) / SOURCE).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == FUNC:
            lines = src.splitlines(keepends=True)
            return textwrap.dedent("".join(lines[node.lineno - 1: node.end_lineno]))
    raise LookupError(f"{FUNC} not found in {SOURCE}")


def copied_text() -> str:
    """This file's copy, sliced out between the markers."""
    text = Path(__file__).read_text()
    start = text.index("# --- BEGIN VERBATIM COPY") 
    start = text.index("\n", start) + 1
    end = text.index("# --- END VERBATIM COPY")
    return text[start:end]


def verify(repo: str | Path = ".") -> bool:
    return copied_text().strip() == extract_from_source(repo).strip()


if __name__ == "__main__":
    import sys

    ok = verify()
    print(("MATCHES" if ok else "DIFFERS FROM") + f" {SOURCE}::{FUNC}")
    sys.exit(0 if ok else 1)
