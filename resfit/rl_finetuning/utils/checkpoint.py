# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Utilities for loading and saving model checkpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
import wandb

from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


def _download_from_wandb(checkpoint_spec: str) -> tuple[Path, dict]:
    """Download checkpoint from W&B.

    Args:
        checkpoint_spec: W&B specification in format:
            - "entity/project/runs/run_id/files/path/to/checkpoint.pt"
            - "run_id" (uses current project/entity from wandb.run)

    Returns:
        (checkpoint_path, wandb_config): Path to downloaded file and W&B run config

    Raises:
        ImportError: If wandb is not available
        ValueError: If checkpoint_spec format is invalid
    """
    # Parse the checkpoint specification
    # Extract entity/project/runs/run_id from the full path
    parts = checkpoint_spec.split("/")

    entity = parts[0]
    project = parts[1]
    run_id = parts[3]

    # Extract the file path within the run (everything after "files/")
    files_idx = parts.index("files")
    file_path = "/".join(parts[files_idx + 1 :])

    # Create API instance
    api = wandb.Api()

    # Construct run path
    run_path = f"{entity}/{project}/{run_id}"

    print(f"Downloading checkpoint from W&B run: {run_path}")
    print(f"File path: {file_path}")

    run = api.run(run_path)

    # Get the W&B config
    wandb_config = dict(run.config)

    # Create temporary directory for download
    temp_dir = Path(tempfile.mkdtemp(prefix="wandb_checkpoint_"))

    # Download the file
    downloaded_file = run.file(file_path).download(root=str(temp_dir), replace=True)
    checkpoint_path = Path(downloaded_file.name)

    print(f"✅ Downloaded checkpoint to: {checkpoint_path}")
    return checkpoint_path, wandb_config


def save_checkpoint(
    agent: QAgent,
    checkpoint_path: str | Path,
    global_step: int,
    config: Any = None,
    success_rate: float | None = None,
    deployment: dict | None = None,
    **extra_data: Any,
) -> None:
    """Save a QAgent checkpoint.

    Args:
        agent: The QAgent to save
        checkpoint_path: Path where to save the checkpoint
        global_step: Current training step
        config: Training configuration (optional)
        success_rate: Success rate when checkpoint was saved (optional)
        deployment: the block from `utils/deployment.py::build_deployment_meta()` -
            normalization stats, base BC policy identity, obs spec and provenance.
            This is what makes a checkpoint deployable on its own: with it,
            inference needs no dataset and can verify it is running against the
            base policy it was actually trained on. Built ONCE at startup and
            passed to every save in the run.
        **extra_data: Additional data to include in checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        "agent_state_dict": agent.state_dict(),
        "global_step": global_step,
        **extra_data,
    }

    # Save optimizer and scheduler states
    optimizer_state_dict = {
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "encoder_opt": agent.encoder_opt.state_dict(),
    }
    scheduler_state_dict = {}
    if hasattr(agent, "actor_scheduler") and agent.actor_scheduler is not None:
        scheduler_state_dict["actor_scheduler"] = agent.actor_scheduler.state_dict()
    if hasattr(agent, "critic_scheduler") and agent.critic_scheduler is not None:
        scheduler_state_dict["critic_scheduler"] = agent.critic_scheduler.state_dict()
    if hasattr(agent, "encoder_scheduler") and agent.encoder_scheduler is not None:
        scheduler_state_dict["encoder_scheduler"] = agent.encoder_scheduler.state_dict()

    checkpoint_data["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict:  # Only add if there are schedulers
        checkpoint_data["scheduler_state_dict"] = scheduler_state_dict

    if config is not None:
        # Convert OmegaConf to plain dict to avoid PyTorch 2.6+ loading issues
        try:
            from omegaconf import OmegaConf

            if hasattr(config, "_metadata"):  # Check if it's an OmegaConf object
                checkpoint_data["config"] = OmegaConf.to_container(config, resolve=True)
            else:
                checkpoint_data["config"] = config
        except ImportError:
            checkpoint_data["config"] = config
    if success_rate is not None:
        checkpoint_data["success_rate"] = success_rate
    if deployment is not None:
        checkpoint_data["deployment"] = deployment

    torch.save(checkpoint_data, checkpoint_path)
    print(f"💾 Saved checkpoint to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    agent: QAgent,
    device: torch.device | str = "cpu",
) -> dict:
    """Load a QAgent checkpoint and restore agent + optimizer states.

    Args:
        checkpoint_path: Path to the checkpoint .pt file.
        agent: The QAgent whose weights and optimizers will be restored.
        device: Device to map tensors onto.

    Returns:
        A dict with auxiliary info from the checkpoint:
        ``{"global_step": int, "success_rate": float | None, "config": ... }``
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore agent weights
    agent.load_state_dict(ckpt["agent_state_dict"])

    # Restore optimizer states
    opt_states = ckpt.get("optimizer_state_dict", {})
    if "actor_opt" in opt_states:
        agent.actor_opt.load_state_dict(opt_states["actor_opt"])
    if "critic_opt" in opt_states:
        agent.critic_opt.load_state_dict(opt_states["critic_opt"])
    if "encoder_opt" in opt_states:
        agent.encoder_opt.load_state_dict(opt_states["encoder_opt"])

    # Restore scheduler states (if any)
    sched_states = ckpt.get("scheduler_state_dict", {})
    if "actor_scheduler" in sched_states and hasattr(agent, "actor_scheduler") and agent.actor_scheduler is not None:
        agent.actor_scheduler.load_state_dict(sched_states["actor_scheduler"])
    if "critic_scheduler" in sched_states and hasattr(agent, "critic_scheduler") and agent.critic_scheduler is not None:
        agent.critic_scheduler.load_state_dict(sched_states["critic_scheduler"])
    if "encoder_scheduler" in sched_states and hasattr(agent, "encoder_scheduler") and agent.encoder_scheduler is not None:
        agent.encoder_scheduler.load_state_dict(sched_states["encoder_scheduler"])

    global_step = ckpt.get("global_step", 0)
    success_rate = ckpt.get("success_rate")
    config = ckpt.get("config")

    print(f"✅ Restored checkpoint: global_step={global_step}, success_rate={success_rate}")

    return {
        "global_step": global_step,
        "success_rate": success_rate,
        "config": config,
        "actor_updates": ckpt.get("actor_updates", 0),
    }
