# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import json
from pathlib import Path

import torch
import wandb

from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from resfit.lerobot.policies.pretrained import PreTrainedPolicy


def download_policy_from_wandb(
    run_id: str,
    *,
    step: str | None = None,
    artifact_version: str = "latest",
) -> tuple[Path, str]:
    """Download a policy checkpoint logged on W&B and return its folder.

    The policy is expected to have been created with the training utilities in
    `train_hf.py` and therefore to contain a `config.json` in the root of the
    downloaded artifact.
    """
    api = wandb.Api()
    project, id_ = run_id.split("/")

    if step is None or str(step).lower() == "latest":
        artifact_name = f"run_{id_}_latest:{artifact_version}"
        checkpoint_step = "latest"
    elif str(step).lower() == "best":
        artifact_name = f"run_{id_}_best:{artifact_version}"
        checkpoint_step = "best"
    else:
        artifact_name = f"run_{id_}_model_step_{step}:{artifact_version}"
        checkpoint_step = str(step)

    artifact_path = f"{project}/{artifact_name}"
    artifact = api.artifact(artifact_path)

    art_dir = Path(artifact.download())
    
    # Locate where config.json exists in the downloaded artifact
    if (art_dir / "config.json").exists():
        policy_dir = art_dir
    elif (art_dir / "policy" / "config.json").exists():
        policy_dir = art_dir / "policy"
    elif (art_dir / "pretrained_model" / "config.json").exists():
        policy_dir = art_dir / "pretrained_model"
    else:
        candidates = list(art_dir.glob("**/config.json"))
        if candidates:
            policy_dir = candidates[0].parent
        else:
            raise FileNotFoundError(f"config.json not found inside downloaded artifact: {art_dir}")

    return policy_dir, checkpoint_step


def resolve_local_policy_dir(local_path: str | Path) -> Path:
    """Resolve a local checkpoint directory to the folder containing `config.json`.

    Mirrors the artifact-layout resolution used by ``download_policy_from_wandb`` so a
    locally trained checkpoint (e.g. a LeRobot ``.../checkpoints/050000/pretrained_model``
    folder) can be loaded without W&B. Accepts the policy dir itself or a parent that
    holds a ``policy/`` or ``pretrained_model/`` subfolder.
    """
    root = Path(local_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"base_policy.local_path does not exist: {root}")

    if (root / "config.json").exists():
        return root
    if (root / "policy" / "config.json").exists():
        return root / "policy"
    if (root / "pretrained_model" / "config.json").exists():
        return root / "pretrained_model"

    candidates = sorted(root.glob("**/config.json"))
    if candidates:
        return candidates[0].parent
    raise FileNotFoundError(
        f"config.json not found in local base policy path (or its policy/ | pretrained_model/ "
        f"subdirs): {root}"
    )


def load_policy(policy_dir: Path) -> PreTrainedPolicy:
    """Infer policy type (diffusion / act) from `config.json` and load weights."""

    config_path = policy_dir / "config.json"
    with config_path.open() as f:
        cfg_dict = json.load(f)

    policy_name_field = str(cfg_dict.get("type", "")).lower()

    if "diffusion" in policy_name_field:
        allowed_keys = {
            "n_obs_steps",
            "horizon",
            "n_action_steps",
            "normalization_mapping",
            "drop_n_last_frames",
            "vision_backbone",
            "crop_shape",
            "crop_is_random",
            "pretrained_backbone_weights",
            "use_group_norm",
            "spatial_softmax_num_keypoints",
            "use_separate_rgb_encoder_per_camera",
            "down_dims",
            "kernel_size",
            "n_groups",
            "diffusion_step_embed_dim",
            "use_film_scale_modulation",
            "noise_scheduler_type",
            "num_train_timesteps",
            "beta_schedule",
            "beta_start",
            "beta_end",
            "prediction_type",
            "clip_sample",
            "clip_sample_range",
            "num_inference_steps",
            "do_mask_loss_for_padding",
            "optimizer_lr",
            "optimizer_betas",
            "optimizer_eps",
            "optimizer_weight_decay",
            "scheduler_name",
            "scheduler_warmup_steps",
            "input_features",
            "output_features",
            "device",
            "use_amp",
            "type",
        }
    else:
        # Filter keys to match ResFit's ACTConfig / PreTrainedConfig fields exactly
        allowed_keys = {
            "n_obs_steps",
            "normalization_mapping",
            "input_features",
            "output_features",
            "device",
            "use_amp",
            "chunk_size",
            "n_action_steps",
            "vision_backbone",
            "pretrained_backbone_weights",
            "replace_final_stride_with_dilation",
            "pre_norm",
            "dim_model",
            "n_heads",
            "dim_feedforward",
            "feedforward_activation",
            "n_encoder_layers",
            "n_decoder_layers",
            "use_vae",
            "latent_dim",
            "n_vae_encoder_layers",
            "temporal_ensemble_coeff",
            "dropout",
            "kl_weight",
            "optimizer_lr",
            "optimizer_weight_decay",
            "optimizer_lr_backbone",
            "type",
        }

    # Filter input_features to only keep visual observations and consolidated observation.state
    if "input_features" in cfg_dict:
        cfg_dict["input_features"] = {
            k: v for k, v in cfg_dict["input_features"].items()
            if k.startswith("observation.images.") or k == "observation.state"
        }
    # Filter output_features to only keep consolidated action
    if "output_features" in cfg_dict:
        cfg_dict["output_features"] = {
            k: v for k, v in cfg_dict["output_features"].items()
            if k == "action"
        }

    cfg_dict = {k: v for k, v in cfg_dict.items() if k in allowed_keys}

    # Write filtered config back so standard from_pretrained loader succeeds
    with config_path.open("w") as f:
        json.dump(cfg_dict, f, indent=4)

    if "diffusion" in policy_name_field:
        policy = DiffusionPolicy.from_pretrained(policy_dir)
    elif "use_vae" in cfg_dict or policy_name_field == "act":
        policy = ACTPolicy.from_pretrained(policy_dir)
    else:
        raise ValueError(f"Unknown policy type: {policy_name_field}")

    # Load preprocessor normalization stats from v3 safetensors files
    import safetensors.torch

    norm_files = list(policy_dir.glob("*normalizer_processor.safetensors"))
    if norm_files:
        norm_stats = safetensors.torch.load_file(norm_files[0])
        for key in policy.normalize_inputs.features:
            buffer_name = "buffer_" + key.replace(".", "_")
            if hasattr(policy.normalize_inputs, buffer_name):
                buffer = getattr(policy.normalize_inputs, buffer_name)
                mean_key = f"{key}.mean"
                std_key = f"{key}.std"
                min_key = f"{key}.min"
                max_key = f"{key}.max"
                if mean_key in norm_stats and "mean" in buffer:
                    buffer["mean"].copy_(norm_stats[mean_key])
                if std_key in norm_stats and "std" in buffer:
                    buffer["std"].copy_(norm_stats[std_key])
                if min_key in norm_stats and "min" in buffer:
                    buffer["min"].copy_(norm_stats[min_key])
                if max_key in norm_stats and "max" in buffer:
                    buffer["max"].copy_(norm_stats[max_key])

    unnorm_files = list(policy_dir.glob("*unnormalizer_processor.safetensors"))
    if unnorm_files:
        unnorm_stats = safetensors.torch.load_file(unnorm_files[0])
        for key in policy.normalize_targets.features:
            buffer_name = "buffer_" + key.replace(".", "_")
            if hasattr(policy.normalize_targets, buffer_name):
                buffer = getattr(policy.normalize_targets, buffer_name)
                mean_key = f"{key}.mean"
                std_key = f"{key}.std"
                min_key = f"{key}.min"
                max_key = f"{key}.max"
                if mean_key in unnorm_stats and "mean" in buffer:
                    buffer["mean"].copy_(unnorm_stats[mean_key])
                if std_key in unnorm_stats and "std" in buffer:
                    buffer["std"].copy_(unnorm_stats[std_key])
                if min_key in unnorm_stats and "min" in buffer:
                    buffer["min"].copy_(unnorm_stats[min_key])
                if max_key in unnorm_stats and "max" in buffer:
                    buffer["max"].copy_(unnorm_stats[max_key])

            if hasattr(policy.unnormalize_outputs, buffer_name):
                buffer = getattr(policy.unnormalize_outputs, buffer_name)
                mean_key = f"{key}.mean"
                std_key = f"{key}.std"
                min_key = f"{key}.min"
                max_key = f"{key}.max"
                if mean_key in unnorm_stats and "mean" in buffer:
                    buffer["mean"].copy_(unnorm_stats[mean_key])
                if std_key in unnorm_stats and "std" in buffer:
                    buffer["std"].copy_(unnorm_stats[std_key])
                if min_key in unnorm_stats and "min" in buffer:
                    buffer["min"].copy_(unnorm_stats[min_key])
                if max_key in unnorm_stats and "max" in buffer:
                    buffer["max"].copy_(unnorm_stats[max_key])

    return policy

    raise ValueError(f"Unknown policy type: {policy_name_field}")


def save_checkpoint(ckpt_dir: Path, step: int, policy, optimizer) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save model weights + config
    policy.save_pretrained(ckpt_dir / "policy")
    # Save optimizer & misc state
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
        },
        ckpt_dir / "trainer_state.pt",
    )


def load_checkpoint(ckpt_dir: Path, policy, optimizer):
    state_pth = ckpt_dir / "trainer_state.pt"
    if not state_pth.exists():
        raise FileNotFoundError(state_pth)
    state = torch.load(state_pth, map_location="cpu")
    policy_loaded = policy.from_pretrained(ckpt_dir / "policy")
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], policy_loaded, optimizer
