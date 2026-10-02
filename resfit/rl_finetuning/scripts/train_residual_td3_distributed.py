# =============================================================================
# GENERATED FILE - DO NOT EDIT BY HAND.
#   template: resfit/rl_finetuning/off_policy/distributed/tools/train_residual_td3_distributed.template.py
#   build:    <python> resfit/rl_finetuning/off_policy/distributed/tools/build_entrypoint.py
# Regions between "verbatim from train_residual_td3.py:A-B" markers are exact copies
# of the original trainer, anchor-checked at build time.
# =============================================================================
"""Distributed actor/learner entrypoint for ResFiT residual TD3 (agentlace).

    role=single   -> calls train_residual_td3.main(cfg) unchanged (parity baseline)
    role=learner  -> GPU server: buffers, offline RL / critic warmup, TD3 updates,
                     checkpoints, wandb, weight publishing
    role=actor    -> laptop: env, cameras, pedal/leader, base-policy + residual
                     inference with exploration noise, n-step, transition outbox

Design: scripts/TrossenStation1Real/RESFIT_DISTRIBUTED_MIGRATION.md
Launch: scripts/TrossenStation1Real/agentlace/*.sh  (same script for both roles)
"""

from __future__ import annotations

# Importing the original trainer FIRST applies its BLAS/OpenMP thread caps (which must
# be set before numpy/torch load) and its MuJoCo env vars - exactly as when it runs as
# a script. It is also where role=single dispatches to.
import resfit.rl_finetuning.scripts.train_residual_td3 as _orig  # noqa: I001

import atexit
import hashlib
import json
import logging
import os
import pprint
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import tensordict
import torch
import torchrl
from hydra.core.config_store import ConfigStore
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import OmegaConf
from tensordict import TensorDict
from termcolor import colored
from torch.utils.data import DataLoader
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictPrioritizedReplayBuffer
from torchvision.transforms import v2 as T
from tqdm import tqdm

import wandb
from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.utils.load_policy import (
    download_policy_from_wandb,
    load_policy,
    resolve_local_policy_dir,
)
from resfit.rl_finetuning.config.residual_td3 import (
    ResidualTD3DexmgConfig,
    ResidualTD3TrossenRealConfig,
)
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.off_policy.distributed.comms import (
    PHASE_DONE,
    PHASE_INIT,
    PHASE_PRETRAINING,
    PHASE_TRAINING,
    PHASE_WARMUP,
    ActorComms,
    LearnerComms,
)
from resfit.rl_finetuning.off_policy.distributed.nstep_stream import NStepStream
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.utils.deployment import (
    base_policy_identity,
    build_deployment_meta,
    write_buffer_sidecar,
)
from resfit.rl_finetuning.utils.checkpoint import load_checkpoint, save_checkpoint
from resfit.rl_finetuning.utils.dtype import to_uint8
from resfit.rl_finetuning.utils.hugging_face import (
    _hf_download_buffer,
    _hf_upload_buffer,
    optimized_replay_buffer_dumps,
    optimized_replay_buffer_loads,
)
from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.train_rollout_recorder import TrainingRolloutRecorder
from resfit.rl_finetuning.wrappers.residual_env_wrapper import BasePolicyVecEnvWrapper

# Reused directly from the original module (imported, not copied).
TrainingTimer = _orig.TrainingTimer
_add_transitions_to_buffer = _orig._add_transitions_to_buffer
_extract_final_info_for_env = _orig._extract_final_info_for_env
OFFLINE_HF_REPO = _orig.OFFLINE_HF_REPO
ONLINE_HF_REPO = _orig.ONLINE_HF_REPO
_CACHE_ROOT = _orig._CACHE_ROOT
OFFLINE_CACHE_DIR = _orig.OFFLINE_CACHE_DIR
ONLINE_CACHE_DIR = _orig.ONLINE_CACHE_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================
@dataclass
class DistConfig:
    # learner address as seen from the actor (the learner binds all interfaces)
    ip: str = "localhost"
    port: int = 5588  # REQ/REP: transitions, stats, handshake
    broadcast_port: int = 5589  # PUB/SUB: weights
    # publish encoder+actor weights every N gradient updates (HIL-SERL: 50)
    steps_per_update: int = 50
    # and at least this often, so an actor that connected late is never starved
    republish_every_s: float = 10.0
    # null = learner free-runs. A number pins agent.update() calls per env step
    # (4 reproduces today's update_every_n_steps=1 x num_updates_per_iteration=4).
    target_utd: float | None = None
    # wire codec, identical on both nodes (enforced at connect). zlib/none need no install.
    codec: str = "zlib"
    push_interval_s: float = 1.0  # actor comms thread tick (push + heartbeat)
    push_chunk: int = 64  # max transitions per message (~85 KB each at 2x84x84 cams)
    push_max_chunks_per_tick: int = 8  # bounds one tick's push so heartbeats are never starved
    shutdown_wait_s: float = 30.0  # learner waits this long for the actor to acknowledge DONE
    timeout_ms: int = 5000
    outbox_capacity: int = 50_000
    idle_poll_s: float = 0.5  # actor sleep while the learner is loading / pretraining


@dataclass
class ResidualTD3TrossenRealDistConfig(ResidualTD3TrossenRealConfig):
    role: str = "single"
    dist: DistConfig = field(default_factory=DistConfig)


@dataclass
class ResidualTD3DexmgDistConfig(ResidualTD3DexmgConfig):
    role: str = "single"
    dist: DistConfig = field(default_factory=DistConfig)


cs = ConfigStore.instance()
cs.store(name="residual_td3_trossen_real_dist_config", node=ResidualTD3TrossenRealDistConfig)
cs.store(name="residual_td3_dexmg_dist_config", node=ResidualTD3DexmgDistConfig)


# Keys allowed to differ between the two nodes. Everything else must match exactly:
# a silently mismatched action_scale / n_step / image_size / normalization knob raises
# no error anywhere - it just trains a critic in one action space and acts in another.
_FINGERPRINT_EXCLUDE = {
    "role",
    "dist",  # ports/codec are enforced by agentlace's own TrainerConfig hash
    "seed",  # may be randomly drawn per process
    "actor_name",  # mutated inside main()
    "eval_num_envs",  # mutated inside main()
    "real_log_file",
    "resume_ckpt",  # only the learner loads checkpoints
    "policy_server_url",  # may legitimately differ (tunnels)
    "no_cleanup",
    "wandb",  # only the learner logs
    "headless",
    "save_video",
}


def config_fingerprint(cfg) -> tuple[str, dict]:
    container = OmegaConf.to_container(cfg, resolve=True)
    flat: dict[str, object] = {}

    def walk(prefix: str, v) -> None:
        if isinstance(v, dict):
            for k in sorted(v):
                walk(f"{prefix}.{k}", v[k])
        else:
            flat[prefix] = v

    for k in sorted(container):
        if k not in _FINGERPRINT_EXCLUDE:
            walk(str(k), container[k])
    digest = hashlib.sha1(json.dumps(flat, sort_keys=True, default=str).encode()).hexdigest()[:16]  # noqa: S324
    return digest, flat


def _verify_fingerprint(local_hash: str, local_flat: dict, init: dict) -> None:
    remote_hash = init["fingerprint"]
    if remote_hash == local_hash:
        print(colored(f"[actor] config fingerprint matches learner ({local_hash})", "green"))
        return
    remote_flat = init["fingerprint_flat"]
    keys = sorted(set(local_flat) | set(remote_flat))
    diffs = [
        f"  {k}: actor={local_flat.get(k, '<missing>')!r}  learner={remote_flat.get(k, '<missing>')!r}"
        for k in keys
        if local_flat.get(k, "<missing>") != remote_flat.get(k, "<missing>")
    ]
    raise RuntimeError(
        "Actor and learner configs differ - refusing to run (this would train the critic "
        "in one setting and act in another, with no error anywhere else):\n"
        + "\n".join(diffs[:40])
        + ("\n  ..." if len(diffs) > 40 else "")
        + "\nRun both roles from the SAME script in scripts/TrossenStation1Real/agentlace/."
    )


# =============================================================================
# Small adapters that let verbatim regions run unchanged
# =============================================================================
class _NStepSink:
    """Stands in for ``online_rb`` on the actor.

    ``_add_transitions_to_buffer(..., online_rb=sink)`` builds the exact same 1-step
    TensorDict it always has and calls ``.add(td)``; the sink applies the n-step
    transform (see nstep_stream.py for why it moved here) and queues the result.
    """

    def __init__(self, stream: NStepStream, outbox):
        self.stream = stream
        self.outbox = outbox

    def add(self, td):
        out = self.stream.push(td)
        if out is not None:
            self.outbox.insert(out)

    def __len__(self) -> int:
        return self.stream.n_emitted


def _to_wire(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        return v.item() if v.numel() == 1 else v.numpy()
    if isinstance(v, dict):
        return {k: _to_wire(x) for k, x in v.items()}
    return v


class _ForwardToLearner:
    """Stands in for the ``wandb`` module inside the actor's verbatim regions.

    wandb lives on the learner; actor-side ``wandb.log(d, step=s)`` calls are queued and
    delivered by the comms thread, then logged by the learner main loop at step ``s``.
    """

    run = None

    def __init__(self, comms: ActorComms):
        self._comms = comms

    def log(self, data, step=None, **_ignored):
        self._comms.log(int(step) if step is not None else 0, _to_wire(dict(data)))


class _LazyRemotePolicyEnv:
    """Stands in for ``env`` in the learner's offline-buffer region.

    That region only ever touches ``env.policy`` (a ``PolicyClient``), and only when the
    offline cache misses AND ``real_hardware`` AND ``use_base_policy_for_base_actions``.
    Connecting lazily means a cache hit never needs the policy server at all.
    """

    def __init__(self, url: str, chunk_steps: int = 0):
        self._url = url
        self._chunk_steps = int(chunk_steps)
        self._client = None

    @property
    def policy(self):
        if self._client is None:
            from trossen_real.inference.policy_client import ChunkedPolicyClient, PolicyClient

            client = PolicyClient(self._url)
            health = client.health()
            if not health.get("loaded"):
                raise RuntimeError(
                    f"[learner] offline buffer needs base actions from policy_server.py at {self._url}, "
                    f"but it has no policy loaded - {health}. If the server runs on the laptop, "
                    "SSH-forward its port to this machine."
                )
            print(f"[learner] querying base policy at {self._url} to populate the offline buffer")
            # Chunked fetching: one round trip per N queries instead of per query.
            # Only the NETWORK pattern changes - /predict already serves from
            # ACTPolicy.select_action()'s internal queue, so a client-side queue
            # returns the identical sequence. The offline build is one query per
            # dataset frame, so this is where it is felt most. The wrapper drops
            # its queue on reset(), which _populate_offline_buffer() calls at
            # every episode boundary.
            if self._chunk_steps:
                if not health.get("supports_predict_chunk"):
                    raise RuntimeError(
                        f"[learner] policy_chunk_steps={self._chunk_steps} but policy_server.py at "
                        f"{self._url} does not support /predict_chunk. Either it predates that "
                        "endpoint, or the checkpoint is one where chunking would change the actions "
                        "(temporal ensembling, or use_relative_actions=True) and the server refuses "
                        "it. Set policy_chunk_steps=0 to use per-query /predict."
                    )
                n_steps = None if self._chunk_steps < 0 else self._chunk_steps
                client = ChunkedPolicyClient(client, n_steps)
                print(
                    f"[learner] policy fetching: CHUNKED "
                    f"({'server n_action_steps' if n_steps is None else n_steps} actions per round "
                    f"trip; server reports n_action_steps={health.get('n_action_steps')})"
                )
            else:
                print("[learner] policy fetching: per-query /predict (policy_chunk_steps=0)")
            self._client = client
        return self._client


def _skip_eval_in_distributed(**_kwargs):
    print("[learner] periodic evaluation is skipped in distributed mode (no env on the learner)")
    return {}


def _check_actor_spec(actor_spec: dict | None, learner_spec: dict) -> bool:
    """Raise if the registered actor's env disagrees with the learner's buffers/networks."""
    if actor_spec is None:
        return False
    bad = [
        f"  {k}: actor={actor_spec.get(k)!r} learner={learner_spec[k]!r}"
        for k in learner_spec
        if actor_spec.get(k) != learner_spec[k]
    ]
    if bad:
        raise RuntimeError(
            "Actor env spec does not match what the learner built its networks/buffers for:\n" + "\n".join(bad)
        )
    return True


def _normalization_payload(dataset, action_scaler, state_standardizer, device) -> dict:
    def arr(x):
        return np.asarray(torch.as_tensor(x).detach().cpu().numpy(), dtype=np.float32)

    a = dataset.meta.stats["action"]
    s = dataset.meta.stats["observation.state"]
    action_stats = {"min": arr(a["min"]), "max": arr(a["max"])}
    state_stats = {"mean": arr(s["mean"]), "std": arr(s["std"])}
    pa = torch.linspace(-1.5, 1.5, action_stats["min"].shape[-1])
    ps = torch.linspace(-1.5, 1.5, state_stats["mean"].shape[-1])
    return {
        "action_stats": action_stats,
        "state_stats": state_stats,
        # The actor rebuilds both normalizers and must reproduce these outputs exactly.
        "probe": {
            "action_in": pa.numpy(),
            "action_out": action_scaler.scale(pa.to(device)).detach().cpu().numpy(),
            "state_in": ps.numpy(),
            "state_out": state_standardizer.standardize(ps.to(device)).detach().cpu().numpy(),
        },
    }


def _normalizers_from_payload(cfg, payload: dict, device):
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=payload["action_stats"],
        action_scale=cfg.agent.actor.action_scale,
        min_range_per_dim=cfg.offline_data.min_action_range,
        device=device,
    )
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=payload["state_stats"],
        min_std=cfg.offline_data.min_state_std,
        device=device,
    )
    probe = payload["probe"]
    got_a = action_scaler.scale(torch.as_tensor(probe["action_in"]).to(device)).detach().cpu().numpy()
    got_s = state_standardizer.standardize(torch.as_tensor(probe["state_in"]).to(device)).detach().cpu().numpy()
    if not (np.allclose(got_a, probe["action_out"], atol=1e-5) and np.allclose(got_s, probe["state_out"], atol=1e-5)):
        raise RuntimeError(
            "Normalization mismatch: the actor's ActionScaler/StateStandardizer, rebuilt from the "
            "learner's dataset stats, do not reproduce the learner's outputs. Refusing to run."
        )
    print(colored("[actor] normalization verified against learner probe", "green"))
    return action_scaler, state_standardizer


# =============================================================================
# Shared setup (verbatim regions)
# =============================================================================
def _common_setup(cfg):
    from trossen_real.log_setup import setup_logging

    setup_logging(
        "online_rl_real" if cfg.real_hardware else "online_rl_sim", level=logging.WARNING, capture_print=False
    )
    # ↓↓↓ verbatim from train_residual_td3.py:305-326
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # ------------------------------------------------------------------
    # Select MuJoCo rendering backend before any environment is created.
    #   headless=True  (default) -> EGL (off-screen GPU rendering)
    #   headless=False           -> GLFW (on-screen MuJoCo viewer window)
    # Not applicable to real hardware - no MuJoCo/rendering involved at all.
    # ------------------------------------------------------------------
    if cfg.real_hardware:
        pass
    elif not cfg.headless:
        os.environ["MUJOCO_GL"] = "glfw"
        logger.info("On-screen rendering enabled (MUJOCO_GL=glfw). A viewer window will open.")
    else:
        os.environ["MUJOCO_GL"] = "egl"

    # Enable performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # ↑↑↑ end verbatim train_residual_td3.py:305-326
    # ↓↓↓ verbatim from train_residual_td3.py:588-603
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    # Comprehensive seeding for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # CUDA seeding for multi-GPU reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    print(f"Set random seed to {cfg.seed}")
    # ↑↑↑ end verbatim train_residual_td3.py:588-603
    return device, device_str


def _load_base_policies(cfg, device):
    # ↓↓↓ verbatim from train_residual_td3.py:337-375
    base_policy: ACTPolicy | None = None
    eval_base_policy: ACTPolicy | None = None
    if cfg.real_hardware:
        cfg.actor_name = "residual_act"
        # NOTE: unlike an earlier draft of this, `offline_data.use_base_policy_for_base_actions=True`
        # IS supported for real hardware - `_populate_offline_buffer()` queries the REMOTE
        # `policy_server.py` (via `env.policy`, the same PolicyClient TrossenResidualEnv already
        # uses) instead of a local in-process ACTPolicy - see its docstring / the design doc's
        # "offline buffer population with base-policy base actions" section for the full reasoning
        # on why this needs FULL-RESOLUTION images for the query but a resized copy for storage.
    else:
        assert "base_policy" in cfg, "Base policy configuration is required"
        base_local_path = cfg.base_policy.get("local_path", None)
        if base_local_path:
            # Local checkpoint takes priority over W&B download.
            policy_dir = resolve_local_policy_dir(base_local_path)
            logger.info(f"Loading base policy from LOCAL checkpoint: {policy_dir} (W&B download skipped)")
        else:
            policy_dir, _ = download_policy_from_wandb(
                cfg.base_policy.wandb_id,
                step=cfg.base_policy.wt_type,
                artifact_version=cfg.base_policy.wt_version,
            )
            logger.info(f"Loading base policy from W&B artifact: {cfg.base_policy.wandb_id} -> {policy_dir}")

        base_policy = load_policy(policy_dir)
        base_policy.to(device)
        base_policy.eval()
        eval_base_policy = load_policy(policy_dir)
        eval_base_policy.to(device)
        eval_base_policy.eval()

        # Extract the configuration from base policy
        base_cfg = base_policy.config

        if isinstance(base_cfg, ACTConfig):
            cfg.actor_name = "residual_act"
        else:
            raise ValueError(f"Unknown base policy type: {type(base_cfg)}")
    # ↑↑↑ end verbatim train_residual_td3.py:337-375
    return base_policy, eval_base_policy


def _load_dataset_and_normalizers(cfg, device):
    # ↓↓↓ verbatim from train_residual_td3.py:377-437
    # Load dataset and get normalization functions early
    print("Loading dataset and setting up normalization...")
    offline_image_transforms = None
    if cfg.offline_data.image_size is not None:
        offline_image_transforms = T.Resize(
            (cfg.offline_data.image_size, cfg.offline_data.image_size),
            antialias=True,
        )
        print(f"Resizing offline dataset images to {cfg.offline_data.image_size}×{cfg.offline_data.image_size}")

    # Real hardware + use_base_policy_for_base_actions needs FULL-RESOLUTION images to query
    # policy_server.py with (same resolution live rollout sends) - resizing at dataset-LOAD time
    # here would throw that away before we ever see it. Instead, load at native resolution and
    # apply the resize AFTER the policy query, per-sample, inside `_populate_offline_buffer()`
    # (see its docstring) - this does NOT reintroduce the OOM `offline_image_transforms` avoids,
    # since only ONE native-res sample is ever in memory at a time (streamed via the DataLoader),
    # never the whole dataset.
    _query_base_actions_from_remote_policy = cfg.real_hardware and cfg.offline_data.use_base_policy_for_base_actions
    dataset_image_transforms = None if _query_base_actions_from_remote_policy else offline_image_transforms
    if _query_base_actions_from_remote_policy:
        print(
            "real_hardware + use_base_policy_for_base_actions: loading dataset at NATIVE resolution "
            f"(resize to {cfg.offline_data.image_size}x{cfg.offline_data.image_size} applied AFTER "
            "querying policy_server.py, per-sample, before storing into the buffer)."
        )
    _offline_root = cfg.offline_data.get("root", None)
    _remap_preset = cfg.offline_data.get("remap_preset", None)
    if _offline_root or _remap_preset:
        from resfit.rl_finetuning.datasets.remapped_lerobot import REMAP_PRESETS, RemappedLeRobotDataset

        _key_map = REMAP_PRESETS.get(_remap_preset, {}) if _remap_preset else {}
        if _remap_preset and _remap_preset not in REMAP_PRESETS:
            raise ValueError(
                f"Unknown offline_data.remap_preset '{_remap_preset}'. Known: {list(REMAP_PRESETS)}"
            )
        logger.info(
            f"Loading offline dataset from local root={_offline_root} with remap_preset={_remap_preset}"
        )
        dataset = RemappedLeRobotDataset(
            cfg.offline_data.name,
            root=_offline_root,
            key_map=_key_map,
            image_transforms=dataset_image_transforms,
        )
    else:
        dataset = LeRobotDataset(cfg.offline_data.name, image_transforms=dataset_image_transforms)

    # Create action scaler from dataset statistics
    action_scaler = ActionScaler.from_dataset_stats(
        action_stats=dataset.meta.stats["action"], # action stats for whole dataset.
        action_scale=cfg.agent.actor.action_scale, # from .sh file, it is set to 0.2 but default is 0.1
        min_range_per_dim=cfg.offline_data.min_action_range, # from .sh and teh default are both 0.1
        device=device,
    )
    # breakpoint()
    # Create state standardizer from dataset statistics
    state_standardizer = StateStandardizer.from_dataset_stats(
        state_stats=dataset.meta.stats["observation.state"],
        min_std=cfg.offline_data.min_state_std, # in .sh and default aer set to 0.1
        device=device,
    )
    # ↑↑↑ end verbatim train_residual_td3.py:377-437
    return (
        dataset,
        action_scaler,
        state_standardizer,
        offline_image_transforms,
        _query_base_actions_from_remote_policy,
    )


def _make_get_envs(cfg):
    # ↓↓↓ verbatim from train_residual_td3.py:439-583
    def get_envs(
        env_name: str,
        num_envs: int,
        base_policy: ACTPolicy | None,
        device: str,
        video_key: str,
        debug: bool,
        action_scaler: ActionScaler,
        state_standardizer: StateStandardizer,
        apply_reward_model: bool = False,
        horizon: int | None = None,
        enable_intervention: bool = False,
    ):
        assert action_scaler is not None, "action_scaler must be provided for consistent normalization"
        assert state_standardizer is not None, "state_standardizer must be provided for consistent normalization"

        if cfg.real_hardware:
            import time as _time

            from trossen_real.cameras.camera_manager import CameraManager
            from trossen_real.config import load_station_config
            from trossen_real.human_intervention.pedal_listener import PedalListener
            from trossen_real.inference.policy_client import ChunkedPolicyClient, PolicyClient
            from trossen_real.leader.trossen_leader_single import TrossenSingleLeader
            from trossen_real.rl.real_residual_env import TrossenResidualEnv
            from trossen_real.teleop.follower_client import FollowerClient

            assert num_envs == 1, "Real hardware only supports a single station (num_envs must be 1)"
            station_config = load_station_config(cfg.station_config_name)
            if station_config.arm_mode != "single":
                raise ValueError(f"Only arm_mode: single is supported for now (got '{station_config.arm_mode}').")
            if station_config.control.command_space != "joint":
                raise ValueError(
                    f"Real-hardware residual RL only supports command_space: joint (got "
                    f"'{station_config.control.command_space}')."
                )

            follower = FollowerClient(station_config.follower_server_url)
            follower.connect()

            policy_client = PolicyClient(cfg.policy_server_url)
            policy_health = policy_client.health()
            if not policy_health.get("loaded"):
                follower.disconnect()
                raise RuntimeError(f"policy_server.py at {cfg.policy_server_url} has no policy loaded - {policy_health}")

            # Chunked fetching for the live rollout: one round trip per N ticks
            # instead of per tick. Only the NETWORK pattern changes - /predict
            # already serves from ACTPolicy.select_action()'s internal queue, so
            # a client-side queue returns the identical sequence. The wrapper
            # drops its queue on reset(), which TrossenResidualEnv.reset() calls
            # per episode and InterventionManager calls on the intervention
            # release edge - the latter works because the env hands that manager
            # this SAME object (real_residual_env.py:147).
            if cfg.policy_chunk_steps:
                if not policy_health.get("supports_predict_chunk"):
                    follower.disconnect()
                    raise RuntimeError(
                        f"policy_chunk_steps={cfg.policy_chunk_steps} but policy_server.py at "
                        f"{cfg.policy_server_url} does not support /predict_chunk. Either it predates "
                        "that endpoint, or the checkpoint is one where chunking would change the "
                        "actions (temporal ensembling, or use_relative_actions=True) and the server "
                        "refuses it. Set policy_chunk_steps=0 to use per-query /predict."
                    )
                n_steps = None if cfg.policy_chunk_steps < 0 else int(cfg.policy_chunk_steps)
                policy_client = ChunkedPolicyClient(policy_client, n_steps)
                print(
                    f"[actor] policy fetching: CHUNKED "
                    f"({'server n_action_steps' if n_steps is None else n_steps} actions per round "
                    f"trip; server reports n_action_steps={policy_health.get('n_action_steps')})"
                )
            else:
                print("[actor] policy fetching: per-query /predict (policy_chunk_steps=0)")

            cameras = CameraManager(station_config)
            cameras.start()
            _time.sleep(1.0)  # let the camera threads produce their first real frames

            leader = None
            pedal = None
            if enable_intervention:
                try:
                    leader = TrossenSingleLeader(station_config.leader_ips["single"])
                    leader.connect()
                except Exception as exc:
                    cameras.stop()
                    follower.disconnect()
                    raise RuntimeError(
                        f"enable_intervention=true but could not connect the leader arm - {exc}"
                    ) from exc
                pedal = PedalListener()

            real_image_keys = [cfg.rl_camera] if isinstance(cfg.rl_camera, str) else list(cfg.rl_camera)
            reset_cfg = station_config.reset["single"]
            return TrossenResidualEnv(
                follower=follower,
                cameras=cameras,
                policy=policy_client,
                control=station_config.control,
                reset_cfg=reset_cfg,
                action_scaler=action_scaler,
                state_standardizer=state_standardizer,
                image_keys=real_image_keys,
                camera_resolution=station_config.cameras.resolution,
                action_dim=7,
                action_space=cfg.real_action_space,
                leader=leader,
                pedal=pedal,
                max_steps=horizon if horizon is not None else cfg.real_max_steps,
                settle_time_s=cfg.real_settle_time_s,
                # Reuse offline_data.image_size as the single source of truth for the RL-facing
                # image resolution - keeps the online buffer (stored here) and the offline buffer
                # (resized in _populate_offline_buffer()) consistent, and avoids building an
                # oversized network from the raw camera capture resolution. The base-policy query
                # (_query_base_action()) is unaffected - always sends the full-res frame.
                rl_image_size=cfg.offline_data.image_size,
                log_file=cfg.real_log_file,
                policy_image_encoding=cfg.policy_image_encoding,
                policy_jpeg_quality=cfg.policy_jpeg_quality,
            )

        # Build the stage-aware reward-model config (training env only). Passed as a
        # plain picklable dict so it survives AsyncVectorEnv spawn if ever used.
        reward_model_cfg = None
        force_sync = False
        if apply_reward_model and cfg.reward_model.enabled:
            pbrs_gamma = cfg.reward_model.pbrs_gamma
            if pbrs_gamma is None:
                pbrs_gamma = cfg.algo.gamma
            reward_model_cfg = {
                "enabled": True,
                "server_url": cfg.reward_model.server_url,
                "task_prompt": cfg.reward_model.task_prompt,
                "request_timeout_s": cfg.reward_model.request_timeout_s,
                "session_prefix": f"{cfg.reward_model.backend}-{cfg.task}",
                "potentials": list(cfg.reward_model.potentials),
                "pbrs_gamma": float(pbrs_gamma),
                "query_every_k": cfg.reward_model.query_every_k,
                "image_key": cfg.reward_model.image_key,
                "keep_sparse_term": cfg.reward_model.keep_sparse_term,
                "hysteresis_k": cfg.reward_model.hysteresis_k,
                "conf_threshold": cfg.reward_model.conf_threshold,
                "monotonic": cfg.reward_model.monotonic,
                "image_vflip": cfg.reward_model.get("image_vflip", False),
                "reward_mode": cfg.reward_model.get("reward_mode", "pbrs"),
                "milestone_payouts": list(cfg.reward_model.get("milestone_payouts", (0.0, 0.1, 0.1, 0.1))),
                "milestone_success_bonus": float(cfg.reward_model.get("milestone_success_bonus", 0.7)),
            }
            # Query the reward server synchronously per step -> keep env in-process.
            force_sync = True

        # Create the vectorized environment
        vec_env = create_vectorized_env(
            env_name=env_name,
            num_envs=num_envs,
            device=device,
            video_key=video_key,
            debug=debug,
            headless=cfg.headless,
            reward_shaping=cfg.reward_shaping,
            use_reward_manipulation_wrapper=cfg.use_reward_manipulation_wrapper,
            terminate_on_success=cfg.terminate_on_success,
            dense_success_bonus_scale=cfg.dense_success_bonus_scale,
            reward_model_cfg=reward_model_cfg,
            force_sync=force_sync,
            horizon=horizon,
        )

        # Wrap it with the base policy wrapper
        return BasePolicyVecEnvWrapper(
            vec_env=vec_env,
            base_policy=base_policy,
            action_scaler=action_scaler,
            state_standardizer=state_standardizer,
        )
    # ↑↑↑ end verbatim train_residual_td3.py:439-583
    return get_envs


# =============================================================================
# LEARNER
# =============================================================================
def learner_main(cfg):
    fp_hash, fp_flat = config_fingerprint(cfg)  # before main() mutates cfg
    comms = LearnerComms(
        cfg.dist,
        freeze_encoder=bool(cfg.agent.freeze_encoder),
        learning_starts=int(cfg.algo.learning_starts),
    )
    comms.start()  # serve immediately so the actor can connect while we load
    try:
        _learner_body(cfg, comms, fp_hash, fp_flat)
    finally:
        comms.set_phase(PHASE_DONE)
        if not comms.wait_for_actor_ack_done(float(cfg.dist.shutdown_wait_s)):
            print(colored("[learner] actor did not acknowledge shutdown in time; stopping anyway", "yellow"))
        comms.stop()


def _learner_body(cfg, comms: LearnerComms, fp_hash: str, fp_flat: dict):
    device, device_str = _common_setup(cfg)
    base_policy, eval_base_policy = _load_base_policies(cfg, device)
    (
        dataset,
        action_scaler,
        state_standardizer,
        offline_image_transforms,
        _query_base_actions_from_remote_policy,
    ) = _load_dataset_and_normalizers(cfg, device)

    comms.init_info = {
        "fingerprint": fp_hash,
        "fingerprint_flat": fp_flat,
        "normalization": _normalization_payload(dataset, action_scaler, state_standardizer, device),
    }

    assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"
    if cfg.real_hardware:
        assert cfg.eval_num_envs == 1, "Real hardware only supports a single station (eval_num_envs must be 1)"
        assert cfg.algo.use_base_policy_for_warmup, (
            "real_hardware=True requires algo.use_base_policy_for_warmup=true - the 'pure random minus "
            "base' warm-up mode can imply arbitrarily large corrections and is not safe on real hardware."
        )
        # Checked here rather than on the first frame: an invalid value would
        # otherwise surface only after the robot is connected and the dataset
        # loaded. Both roles validate, so a typo cannot reach only one side.
        from trossen_real.inference.policy_client import IMAGE_ENCODINGS

        assert cfg.policy_image_encoding in IMAGE_ENCODINGS, (
            f"policy_image_encoding must be one of {list(IMAGE_ENCODINGS)}, "
            f"got {cfg.policy_image_encoding!r}"
        )
        assert 1 <= cfg.policy_jpeg_quality <= 100, (
            f"policy_jpeg_quality must be in [1, 100], got {cfg.policy_jpeg_quality}"
        )

    # ---------------------------------------------------------------------
    # Deployment metadata - built ONCE here, embedded in every checkpoint this
    # run saves (critic warm-up, offline RL, online RL, resume alike).
    #
    # The /health query is deliberately unconditional for real hardware, even
    # when the offline buffer comes from cache and no base action is ever
    # fetched: what is being recorded is the base policy's IDENTITY, not its
    # output. A residual is a correction to one specific ACT checkpoint and is
    # meaningless against any other, so a checkpoint that cannot name its base
    # is not reproducible. Training refuses to start rather than produce one.
    # ---------------------------------------------------------------------
    _base_policy_identity = None
    if cfg.real_hardware:
        _base_policy_identity = base_policy_identity(cfg.policy_server_url)
        print(colored(
            f"[learner] base BC policy pinned: weights_sha256="
            f"{_base_policy_identity['weights_sha256'][:16]}... "
            f"({_base_policy_identity['server_checkpoint_path']})", "green"))
    _deployment_meta = build_deployment_meta(
        cfg, action_scaler, state_standardizer, _base_policy_identity
    )

    if cfg.reward_model.enabled and not cfg.algo.terminated_only_bootstrap:
        logger.warning(
            "reward_model.enabled=True but algo.terminated_only_bootstrap=False. "
            "PBRS potentials telescope correctly only with terminated-only bootstrap; "
            "set algo.terminated_only_bootstrap=true."
        )

    # ---- dims: no env on the learner --------------------------------------------------
    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    if cfg.real_hardware:
        # Same derivation the original uses for offline_pretrain_only (no env object):
        # TrossenResidualEnv's spaces are pure config.
        action_dim = 7
        lowdim_dim = action_dim
        img_c = 3
        img_h = img_w = int(cfg.offline_data.image_size) if cfg.offline_data.image_size is not None else 256
        horizon = cfg.real_max_steps
    else:
        spec = comms.wait_for_actor_spec()
        action_dim = int(spec["action_dim"])
        lowdim_dim = int(spec["lowdim_dim"])
        img_c, img_h, img_w = (int(x) for x in spec["img_shape"])
        horizon = int(spec["horizon"])
    learner_spec = {
        "action_dim": int(action_dim),
        "lowdim_dim": int(lowdim_dim),
        "img_shape": [int(img_c), int(img_h), int(img_w)],
        "horizon": int(horizon),
        "image_keys": list(image_keys),
    }
    print(f"[learner] spec: {learner_spec}")
    lowdim_keys = ["observation.state", "observation.base_action"]

    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )

    # ↓↓↓ verbatim from train_residual_td3.py:791-797
    # Set up actor learning rate warmup
    actor_updates = 0
    if cfg.algo.actor_lr_warmup_steps > 0:
        print(
            f"Actor LR warmup enabled: 0.0 -> {cfg.agent.actor_lr:.2e} "
            f"over {cfg.algo.actor_lr_warmup_steps} actor updates"
        )
    # ↑↑↑ end verbatim train_residual_td3.py:791-797

    # ---- replay buffers ---------------------------------------------------------------
    # ↓↓↓ verbatim from train_residual_td3.py:806-814
    alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0
    beta = cfg.algo.priority_beta if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0

    online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
    offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)

    if cfg.algo.offline_fraction == 0.0:
        print("Online-only training mode: offline_fraction=0.0")

    # ↑↑↑ end verbatim train_residual_td3.py:806-814
    # Same as the original's online_rb EXCEPT no `transform=MultiStepTransform(...)`:
    # transitions arrive already n-step processed by the actor (nstep_stream.py), and
    # the ResFiT transform cannot coexist with bulk extend(). Buffer contents, cache
    # hash and on-disk dumps are identical, so existing online caches load as-is.
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,
        priority_key="_priority",
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,
        batch_size=online_batch_size,
    )
# ↓↓↓ verbatim from train_residual_td3.py:831-934

    # ------------------------------------------------------------------
    # Caching layer for online replay buffer ----------------------------
    # ------------------------------------------------------------------
    online_cache_meta = {
        "task": cfg.task,
        "wandb_name": cfg.wandb.name,
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        # Schema marker: transitions now store next.terminated in addition to next.done.
        # Bumping this invalidates pre-existing caches that lack the key.
        "schema": "terminated_v1",
        "horizon": horizon,
        "size": cfg.algo.learning_starts,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "buffer_size": cfg.algo.buffer_size,
        "batch_size": online_batch_size,
        # Include random action noise scale to prevent mixing data from different noise levels
        "random_action_noise_scale": utils.to_native_list(cfg.algo.random_action_noise_scale),
        # Normalization parameters for consistency
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "normalized_actions": True,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if cfg.algo.sampling_strategy == "prioritized_replay":
        online_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        online_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(online_cache_meta)
    _online_meta_str = json.dumps(online_cache_meta, sort_keys=True)
    online_cache_hash = hashlib.sha1(_online_meta_str.encode()).hexdigest()[:8]  # noqa: S324
    # Base local path for the online buffer ------------------------------
    online_cache_dir = ONLINE_CACHE_DIR / online_cache_hash

    # Attempt to download/extract from HF every run (no-op if already cached)
    dl_dir = None
    if ONLINE_HF_REPO is not None:
        print(f"Attempting to download online buffer {online_cache_hash} from {ONLINE_HF_REPO}...")
        dl_dir = _hf_download_buffer(ONLINE_HF_REPO, online_cache_hash, ONLINE_CACHE_DIR)
    if dl_dir is not None:
        online_cache_dir = dl_dir

    loaded_online_from_cache = False
    if online_cache_dir.exists():
        print(f"{online_cache_dir} found on disk. Attempting to load...")
        online_rb.sampler._empty()
        optimized_replay_buffer_loads(online_rb, online_cache_dir)
        loaded_online_from_cache = True
        print(f"Loaded online buffer from cache at {online_cache_dir} (size={len(online_rb)})")

    # Offline data is required for normalization, but can be unused for training if offline_fraction=0
    assert cfg.offline_data is not None and cfg.offline_data.num_episodes is not None

    # Dataset and normalization already loaded above - use existing dataset

    # Use actual dataset metadata for precise buffer sizing
    if cfg.offline_data.num_episodes is not None:
        # Only use subset of episodes if specified
        total_frames = sum(
            dataset.meta.episodes[ep_idx]["length"]
            for ep_idx in range(min(cfg.offline_data.num_episodes, dataset.meta.total_episodes))
        )
        num_episodes = cfg.offline_data.num_episodes
    else:
        # Use entire dataset
        total_frames = dataset.meta.total_frames
        num_episodes = dataset.meta.total_episodes

    # Calculate transitions: each episode contributes (episode_length - 1) transitions
    estimated_transitions = max(0, total_frames - num_episodes)

    print("Dataset buffer sizing:")
    print(f"  Total frames to process: {total_frames}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Estimated transitions: {estimated_transitions}")

    # Calculate buffer size for simplified approach (1 transition per frame pair)
    max_offline_transitions = (
        estimated_transitions if cfg.algo.offline_fraction > 0.0 else 1
    )  # Minimum size for online-only mode
    if cfg.algo.offline_fraction > 0.0:
        print(f"Offline buffer sized for GT-as-base approach: {max_offline_transitions} transitions")
    else:
        print("Online-only mode: creating minimal offline buffer (unused)")

    offline_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=max_offline_transitions, device="cpu"),
        alpha=alpha,
        beta=beta,
        eps=1e-6,  # Small epsilon added to priorities to prevent zero values
        priority_key="_priority",
        transform=MultiStepTransform(
            n_steps=cfg.algo.n_step,
            gamma=cfg.algo.gamma,
            use_terminated_for_bootstrap=cfg.algo.terminated_only_bootstrap,
        ),
        pin_memory=True,
        prefetch=cfg.algo.prefetch_batches,  # Add prefetching
        batch_size=max(offline_batch_size, 1),  # Ensure batch_size is at least 1
    )
# ↑↑↑ end verbatim train_residual_td3.py:831-934

    comms.attach_online_buffer(online_rb)
    comms.online_size = len(online_rb)

    env = _LazyRemotePolicyEnv(cfg.policy_server_url, cfg.policy_chunk_steps)
    # ↓↓↓ verbatim from train_residual_td3.py:936-1347
    # Normalization functions already defined above - use them

    # ------------------------------------------------------------------
    # Convert offline dataset episodes into transitions and fill buffer
    # ------------------------------------------------------------------
    def _dataset_sample_to_json_obs(sample: dict, image_keys: list[str],
                                    image_encoding: str = "raw",
                                    jpeg_quality: int = 95) -> dict:
        """Build the JSON body `policy_server.py`'s `/predict` expects, directly
        from a LeRobotDataset sample's NATIVE-resolution image tensors - used by
        `_populate_offline_buffer()`'s real-hardware branch (no live
        `follower_state` dict available here, unlike `TrossenResidualEnv`'s
        online `build_observation()` call). The auxiliary joint_pos_raw/
        ee_pose_raw/velocity/effort/acceleration keys are zero-filled: ACT only
        ever reads `observation.state` (confirmed by reading `modeling_act.py`
        - these extra keys are schema-listed but completely inert/unused), so
        zero-filling them has NO effect on the predicted action."""
        from trossen_real.inference.policy_client import encode_image

        q = sample["observation.state"].float().squeeze(0).tolist()
        obs = {
            "observation.state": q,
            "observation.joint_pos_raw": q,
            "observation.ee_pose_raw": [0.0] * 7,
            "observation.velocity": [0.0] * 7,
            "observation.effort": [0.0] * 7,
            "observation.acceleration": [0.0] * 7,
        }
        for key in image_keys:
            chw = sample[key].float().squeeze(0)  # (3, H, W), native resolution, [0, 1]
            hwc_uint8 = (chw.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).numpy()
            obs[key] = encode_image(hwc_uint8, image_encoding, jpeg_quality)
        return obs

    def _populate_offline_buffer(
        dataset: LeRobotDataset,
        rb: ReplayBuffer,
        image_keys: list[str],
        num_episodes: int | None = None,
        use_base_policy_for_base_actions: bool = False,
        base_policy: ACTPolicy | None = None,
        reward_shaping: bool = False,
        offline_reward_map: dict[int, float] | None = None,
        real_hardware: bool = False,
        real_policy_client=None,
        post_resize=None,
        policy_image_encoding: str = "raw",
        policy_jpeg_quality: int = 95,
    ) -> int:
        """
        Iterates through *dataset* sequentially, converts consecutive frames
        into residual RL transitions and pushes them into *rb*.

        Two modes:
        1. GT-as-base (use_base_policy_for_base_actions=False):
           Uses GT actions as both the base action (in observations) and the target action
           (in transitions). Teaches residual policy to output zero: residual = GT - GT = 0

        2. Base-policy-as-base (use_base_policy_for_base_actions=True):
           Uses base policy to generate base actions and GT actions as targets.
           More consistent with online training: residual = GT - base_policy_action

           On real hardware (`real_hardware=True`), the base policy is NOT loaded
           in-process (`base_policy=None`) - instead `real_policy_client` (the SAME
           `PolicyClient`/`env.policy` used for live rollout) is queried over HTTP,
           using the dataset's NATIVE-resolution image tensors (`dataset` must have
           been loaded WITHOUT a resize transform in this case - see the dataset-
           loading section above). `post_resize` (e.g. `T.Resize((84, 84))`) is then
           applied to the images AFTER the query, right before they're stored into
           `rb` - this keeps the buffer's memory footprint small (the OOM risk is
           from the STORED buffer size, not from a single transient full-res frame
           during the query) while still using full-resolution images for the
           policy query itself, matching what live rollout sends.

        Returns the number of transitions added.
        """
        if use_base_policy_for_base_actions:
            if real_hardware:
                if real_policy_client is None:
                    raise ValueError("real_policy_client must be provided when real_hardware and use_base_policy_for_base_actions=True")
            elif base_policy is None:
                raise ValueError("base_policy must be provided when use_base_policy_for_base_actions=True")

        # Populate buffer from pre-loaded dataset
        print("Populating offline buffer from dataset...")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        episode_cache: dict[int, dict] = {}
        transitions = 0
        step_id = 0

        for sample in tqdm(loader, desc="Processing offline dataset"):
            ep_idx = int(sample["episode_index"].item())
            if num_episodes is not None and ep_idx == num_episodes:
                break

            # ------------------------------------------------------------------
            # Build observation and action directly for replay buffer ----------
            # ------------------------------------------------------------------
            # Extract data and keep on CPU (replay buffer uses CPU storage)
            _gt_action: torch.Tensor = sample["action"].float().squeeze(0)
            gt_action_scaled = action_scaler.scale(_gt_action)
            done_flag = bool(sample["next.done"].item())

            # Generate base action based on the selected mode
            if use_base_policy_for_base_actions:
                if real_hardware:
                    # Query the REMOTE policy_server.py (over HTTP, same PolicyClient
                    # live rollout uses) with the FULL-RESOLUTION image tensors - never
                    # touches a local in-process ACTPolicy (none is loaded for real
                    # hardware; the checkpoint's lerobot version may not even match
                    # this repo's vendored resfit.lerobot ACT implementation).
                    json_obs = _dataset_sample_to_json_obs(
                        sample, image_keys, policy_image_encoding, policy_jpeg_quality)
                    base_action_np = real_policy_client.predict(json_obs)  # (7,) real units
                    base_action = torch.from_numpy(np.asarray(base_action_np, dtype=np.float32)).unsqueeze(0)
                    base_action_scaled = action_scaler.scale(base_action.squeeze(0).cpu())
                else:
                    # Use base policy to generate base action from current observation
                    # Build raw observation first for base policy inference
                    raw_obs = {}
                    for k in sample:
                        if "observation" in k:
                            raw_obs[k] = sample[k].to(device)  # Keep batch dimension for base policy

                    # Get base action from base policy
                    with torch.no_grad():
                        base_action = base_policy.select_action(raw_obs)
                    base_action_scaled = action_scaler.scale(base_action.squeeze(0).cpu())
            else:
                # Use GT action as base action (original behavior)
                base_action_scaled = gt_action_scaled

            # Build observation dict directly in target format
            curr_obs = {
                "observation.state": state_standardizer.standardize(sample["observation.state"].float().squeeze(0)),
                "observation.base_action": base_action_scaled,
            }
            for k in image_keys:
                img = sample[k].squeeze(0)
                if post_resize is not None:
                    # Real hardware only: dataset was loaded at native resolution (needed for the
                    # policy query above) - resize down NOW, right before storage, to keep the
                    # buffer's memory footprint small (matches offline_data.image_size / the RL
                    # critic's expected input size, and the online buffer's resolution).
                    img = post_resize(img)
                curr_obs[k] = img

            # Convert images to uint8 for memory-efficient storage
            to_uint8(curr_obs, image_keys)

            # Determine the reward for this step.
            # Priority:
            #   1) external sidecar reward (offline_reward_map, keyed by global frame
            #      index) — treated like next.reward (source-frame convention);
            #   2) dense dataset reward (next.reward) when reward_shaping;
            #   3) sparse binary reward (1.0 on done).
            if offline_reward_map is not None:
                gidx = int(sample["index"].item())
                step_reward = float(offline_reward_map.get(gidx, 0.0))
            elif reward_shaping and "next.reward" in sample:
                step_reward = float(sample["next.reward"].item())
            else:
                # print(f"[DEBUG] !!!! USING SPARSE REWARD and rewrad is populated based on done flag = {done_flag}, offline_reward_map = {offline_reward_map}, reward_shaping = {reward_shaping}")
                step_reward = float(done_flag)

            # ------------------------------------------------------------------
            # If we already cached the *previous* frame for this episode we can
            # create transitions now.
            # ------------------------------------------------------------------
            if ep_idx in episode_cache:
                # Create transitions for each combination of prev and current variants
                prev_obs = episode_cache[ep_idx]["obs"]
                prev_action_scaled = episode_cache[ep_idx]["action"]
                prev_reward = episode_cache[ep_idx]["reward"]
                prev_done = episode_cache[ep_idx]["done"]
                transition = TensorDict(
                    {
                        "obs": TensorDict(prev_obs, batch_size=[]),
                        "action": prev_action_scaled,
                        # Always False - offline demo data has no intervention concept. Must be
                        # present here (not just in the online buffer's `_add_transitions_to_buffer()`)
                        # because `torch.cat([online_batch, offline_batch])` (main training loop)
                        # requires both TensorDicts to have an identical key set, or it raises a
                        # KeyError at every mixed-batch training step whenever offline_fraction > 0.
                        "intervened": torch.tensor(False, dtype=torch.bool),
                        "next": TensorDict(
                            {
                                "obs": TensorDict(curr_obs, batch_size=[]),
                                "done": torch.tensor(prev_done, dtype=torch.bool),
                                "terminated": torch.tensor(prev_done, dtype=torch.bool),
                                "reward": torch.tensor(prev_reward, dtype=torch.float32),
                            },
                            batch_size=[],
                        ),
                        "_priority": torch.tensor(10.0, dtype=torch.float32),  # High initial priority for new samples
                    },
                    batch_size=[],
                ).unsqueeze(0)

                rb.add(transition)
                transitions += 1

                step_id += 1
            else:
                step_id = 0

            # Cache current frame for pairing with the next one ---------------
            episode_cache[ep_idx] = {
                "obs": curr_obs,
                "action": gt_action_scaled,
                "reward": step_reward,
                "done": done_flag,
                "step_id": step_id,
            }

        # Log final statistics
        print(f"Added {transitions} transitions")

        return transitions

    # ------------------------------------------------------------------
    # Caching layer for offline replay buffer ---------------------------
    # ------------------------------------------------------------------
    # Optional external per-transition reward source (sidecar parquet). Loaded
    # once into a {global_frame_index: reward} map and injected into the offline
    # fill (treated like next.reward). Generic across sparse/PBRS/future keys.
    offline_reward_map: dict[int, float] | None = None
    if cfg.offline_data.reward_parquet:
        import pyarrow.parquet as _pq

        _rp = Path(cfg.offline_data.reward_parquet).expanduser()
        if not _rp.exists():
            raise FileNotFoundError(f"offline_data.reward_parquet not found: {_rp}")
        _pf = _pq.ParquetFile(_rp)
        # Validate that the offline PBRS labeling config matches the RL config so
        # the offline rewards telescope identically to the online shaping. A gamma
        # / potentials / keep_sparse_term / num_stages mismatch silently corrupts
        # the PBRS invariance, so those are hard errors; the server-gating knobs
        # only change stage *assignment* and are warnings.
        import json as _json

        _meta = _pf.schema_arrow.metadata or {}
        _lbl_raw = _meta.get(b"labeler_config", None)
        if _lbl_raw is not None:
            _lbl = _json.loads(_lbl_raw.decode())
            _rm = cfg.reward_model
            _mode = _rm.get("reward_mode", "pbrs")
            _critical = []
            _warn = []
            # A milestone parquet must never feed a PBRS run (or vice-versa).
            if str(_lbl.get("reward_mode", "pbrs")) != str(_mode):
                _critical.append(f"reward_mode: parquet={_lbl.get('reward_mode')} vs RL={_mode}")
            if _mode == "milestone":
                if [float(x) for x in _lbl.get("milestone_payouts", [])] != [
                    float(x) for x in _rm.get("milestone_payouts", ())
                ]:
                    _critical.append(
                        f"milestone_payouts: parquet={_lbl.get('milestone_payouts')} "
                        f"vs RL={list(_rm.get('milestone_payouts', ()))}"
                    )
                if abs(float(_lbl.get("milestone_success_bonus", 0.0)) - float(_rm.get("milestone_success_bonus", 0.7))) > 1e-6:
                    _critical.append(
                        f"milestone_success_bonus: parquet={_lbl.get('milestone_success_bonus')} "
                        f"vs RL={_rm.get('milestone_success_bonus', 0.7)}"
                    )
            else:
                _pbrs_gamma = _rm.pbrs_gamma if _rm.pbrs_gamma is not None else cfg.algo.gamma
                if abs(float(_lbl.get("gamma", _pbrs_gamma)) - float(_pbrs_gamma)) > 1e-6:
                    _critical.append(f"gamma: parquet={_lbl.get('gamma')} vs RL={_pbrs_gamma}")
                if [float(x) for x in _lbl.get("potentials", [])] != [float(x) for x in _rm.potentials]:
                    _critical.append(f"potentials: parquet={_lbl.get('potentials')} vs RL={list(_rm.potentials)}")
                if bool(_lbl.get("keep_sparse_term", _rm.keep_sparse_term)) != bool(_rm.keep_sparse_term):
                    _critical.append(
                        f"keep_sparse_term: parquet={_lbl.get('keep_sparse_term')} vs RL={_rm.keep_sparse_term}"
                    )
            if int(_lbl.get("num_stages", _rm.num_stages)) != int(_rm.num_stages):
                _critical.append(f"num_stages: parquet={_lbl.get('num_stages')} vs RL={_rm.num_stages}")
            for _k, _rl_v in (
                ("hysteresis_k", _rm.hysteresis_k),
                ("query_every_k", _rm.query_every_k),
                ("conf_threshold", _rm.conf_threshold),
                ("image_vflip", _rm.get("image_vflip", False)),
            ):
                if _k in _lbl and _lbl[_k] != type(_lbl[_k])(_rl_v):
                    _warn.append(f"{_k}: parquet={_lbl[_k]} vs RL={_rl_v}")
            if _critical:
                raise ValueError(
                    "Offline reward parquet was labeled with a config that is INCOMPATIBLE "
                    f"with this RL run (would corrupt the reward-model shaping):\n  - "
                    + "\n  - ".join(_critical)
                    + f"\nRe-label {_rp.name} with matching params (scripts/label_offline_pbrs.py) "
                    "or fix the RL config."
                )
            if _warn:
                print(
                    "[reward_model] WARNING: offline labeling config differs from RL "
                    "(stage assignment may differ between offline/online):\n  - "
                    + "\n  - ".join(_warn)
                )
            print(f"[reward_model] offline labeling config verified against RL config: {_rp.name}")
        else:
            print(
                f"[reward_model] WARNING: {_rp.name} has no 'labeler_config' metadata; "
                "cannot verify it matches the RL PBRS config (gamma/potentials/...)."
            )
        _tbl = _pf.read(columns=["index", cfg.offline_data.reward_column]).to_pydict()
        offline_reward_map = {
            int(i): float(r) for i, r in zip(_tbl["index"], _tbl[cfg.offline_data.reward_column])
        }
        print(
            f"Loaded {len(offline_reward_map)} offline rewards from {_rp} "
            f"(column='{cfg.offline_data.reward_column}')"
        )

    # Build a metadata dictionary that uniquely identifies the buffer
    offline_cache_meta = {
        "task": cfg.task,
        "wandb_name": cfg.wandb.name,
        "dataset_name": cfg.offline_data.name,
        "num_episodes": cfg.offline_data.num_episodes,
        "use_base_policy_for_base_actions": cfg.offline_data.use_base_policy_for_base_actions,
        "min_action_range": cfg.offline_data.min_action_range,
        "min_state_std": cfg.offline_data.min_state_std,
        "reward_shaping": cfg.reward_shaping,
        # External reward source (invalidates cache when it changes).
        "reward_parquet": cfg.offline_data.reward_parquet,
        "reward_column": cfg.offline_data.reward_column,
        # Local dataset + key-remap (invalidate cache when they change).
        "offline_root": cfg.offline_data.get("root", None),
        "remap_preset": cfg.offline_data.get("remap_preset", None),
        "image_keys": image_keys,
        "n_step": cfg.algo.n_step,
        "gamma": cfg.algo.gamma,
        # Schema marker: transitions now store next.terminated in addition to next.done.
        "schema": "terminated_v1",
        "base_policy_wandb_id": cfg.base_policy.wandb_id,
        "sampling_strategy": cfg.algo.sampling_strategy,
        "normalized_actions": True,
        "batch_size": offline_batch_size,
        # Library versions for compatibility
        "torchrl_version": torchrl.__version__,
        "tensordict_version": tensordict.__version__,
    }
    if cfg.algo.sampling_strategy == "prioritized_replay":
        offline_cache_meta["priority_alpha"] = cfg.algo.priority_alpha
        offline_cache_meta["priority_beta"] = cfg.algo.priority_beta

    pprint.pprint(offline_cache_meta)

    # Deterministically hash the metadata to create a short cache directory name
    meta_str = json.dumps(offline_cache_meta, sort_keys=True)
    cache_hash = hashlib.sha1(meta_str.encode()).hexdigest()[:8]  # noqa: S324

    # Base local path for this buffer ---------------------------------------
    cache_dir = OFFLINE_CACHE_DIR / cache_hash

    # Try to download/extract from the Hub (will no-op if file not there)
    downloaded_dir = None
    if OFFLINE_HF_REPO is not None:
        print(f"Attempting to download offline buffer {cache_hash} from {OFFLINE_HF_REPO}...")
        downloaded_dir = _hf_download_buffer(OFFLINE_HF_REPO, cache_hash, OFFLINE_CACHE_DIR)
    if downloaded_dir is not None:
        cache_dir = downloaded_dir  # use extracted location

    loaded_from_cache = False
    added = 0

    if cfg.algo.offline_fraction > 0.0:
        # Only populate offline buffer if we're using offline data
        if cache_dir.exists():
            print(f"{cache_dir} found on disk. Attempting to load...")
            offline_rb.sampler._empty()
            optimized_replay_buffer_loads(offline_rb, cache_dir)
            loaded_from_cache = True
            print(f"Loaded offline buffer from cache at {cache_dir} (size={len(offline_rb)})")

        if not loaded_from_cache:
            if cfg.algo.offline_pretrain_only:
                raise RuntimeError(
                    "offline_pretrain_only=True requires the offline buffer to already exist on "
                    f"disk at {cache_dir} (this config's cache hash didn't match anything there) - "
                    "populating it from scratch needs a live env (real_policy_client=env.policy), "
                    "which offline_pretrain_only deliberately never creates. Run the normal (non-"
                    "offline_pretrain_only) real-hardware training once first to populate this cache, "
                    "or double-check every offline_data.*/reward_shaping/n_step/gamma config value "
                    "matches the run that originally produced it."
                )
            added = _populate_offline_buffer(
                dataset=dataset,
                rb=offline_rb,
                image_keys=image_keys,
                num_episodes=cfg.offline_data.num_episodes,
                use_base_policy_for_base_actions=cfg.offline_data.use_base_policy_for_base_actions,
                base_policy=base_policy if cfg.offline_data.use_base_policy_for_base_actions else None,
                reward_shaping=cfg.reward_shaping,
                offline_reward_map=offline_reward_map,
                real_hardware=cfg.real_hardware,
                real_policy_client=env.policy if _query_base_actions_from_remote_policy else None,
                post_resize=offline_image_transforms if _query_base_actions_from_remote_policy else None,
                policy_image_encoding=cfg.policy_image_encoding,
                policy_jpeg_quality=cfg.policy_jpeg_quality,
            )

            print(f"Added {added} offline transitions to buffer (size={len(offline_rb)})")

            # Save buffer to disk for future runs + upload to Hub ----------------
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimized_replay_buffer_dumps(offline_rb, cache_dir)

            with open(cache_dir / "user_metadata.json", "w") as f:
                json.dump(offline_cache_meta, f, indent=2)
            write_buffer_sidecar(cache_dir, _deployment_meta)

            if OFFLINE_HF_REPO is not None:
                _hf_upload_buffer(OFFLINE_HF_REPO, cache_dir, cache_hash)
        else:
            added = len(offline_rb)
    else:
        print("Skipping offline buffer population for online-only training")
    # ↑↑↑ end verbatim train_residual_td3.py:936-1347

    # ---- warm-up gate: the ACTOR collects, the learner waits ------------------------
    actor_spec_checked = _check_actor_spec(comms.actor_spec, learner_spec)
    if len(online_rb) < cfg.algo.learning_starts:
        # ↓↓↓ verbatim from train_residual_td3.py:1363-1383
        if cfg.algo.offline_pretrain_only:
            raise RuntimeError(
                "offline_pretrain_only=True requires the online buffer to already exist on disk "
                f"at {online_cache_dir} with at least algo.learning_starts={cfg.algo.learning_starts} "
                f"transitions (this config's cache hash didn't match anything there, or it matched "
                f"but had fewer than {cfg.algo.learning_starts} transitions saved) - collecting more "
                "needs a live env, which offline_pretrain_only deliberately never creates. Double-"
                "check every algo.*/offline_data.* config value matches the run that originally "
                "produced this cache (task, image_keys, n_step, gamma, horizon, learning_starts, "
                "buffer_size, batch_size, random_action_noise_scale, min_action_range, min_state_std)."
            )
        _resumed_from_partial_cache = loaded_online_from_cache and len(online_rb) > 0
        if _resumed_from_partial_cache:
            print(
                f"Resuming warm-up: online buffer already has {len(online_rb)} transitions "
                f"loaded from {online_cache_dir}; collecting "
                f"{cfg.algo.learning_starts - len(online_rb)} more to reach "
                f"algo.learning_starts={cfg.algo.learning_starts}…"
            )
        else:
            print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - len(online_rb)} random steps…")
        # ↑↑↑ end verbatim train_residual_td3.py:1363-1383
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        next_log_threshold = (len(online_rb) // 1000 + 1) * 1000
        # ↓↓↓ verbatim from train_residual_td3.py:1405-1408
        _save_freq = cfg.algo.online_warmup_save_freq
        next_save_threshold = (
            ((len(online_rb) // _save_freq) + 1) * _save_freq if _save_freq and _save_freq > 0 else None
        )
        # ↑↑↑ end verbatim train_residual_td3.py:1405-1408
        comms.set_phase(PHASE_WARMUP)
        while len(online_rb) < cfg.algo.learning_starts:
            n_new = comms.drain()
            if not actor_spec_checked:
                actor_spec_checked = _check_actor_spec(comms.actor_spec, learner_spec)
            if n_new == 0:
                time.sleep(0.05)
                continue
            if len(online_rb) >= next_log_threshold:
                print(f"[Warm-up] {len(online_rb)} / {cfg.algo.learning_starts} transitions received from actor")
                next_log_threshold += 1000
        # ↓↓↓ verbatim from train_residual_td3.py:1488-1496 (re-indented -4)
        if next_save_threshold is not None and len(online_rb) >= next_save_threshold:
            print(
                f"[Warm-up] Checkpointing online buffer at {len(online_rb)} transitions "
                f"to {online_cache_dir} (resumable if collection is interrupted)…"
            )
            optimized_replay_buffer_dumps(online_rb, online_cache_dir)
            with open(online_cache_dir / "user_metadata.json", "w") as f:
                json.dump(online_cache_meta, f, indent=2)
            write_buffer_sidecar(online_cache_dir, _deployment_meta)
            next_save_threshold += _save_freq
        # ↑↑↑ end verbatim train_residual_td3.py:1488-1496 (re-indented -4)
        # ↓↓↓ verbatim from train_residual_td3.py:1499-1507
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        optimized_replay_buffer_dumps(online_rb, online_cache_dir)
        with open(online_cache_dir / "user_metadata.json", "w") as f:
            json.dump(online_cache_meta, f, indent=2)
        write_buffer_sidecar(online_cache_dir, _deployment_meta)
        if ONLINE_HF_REPO is not None:
            _hf_upload_buffer(ONLINE_HF_REPO, online_cache_dir, online_cache_hash)
        print(f"Warm-up done. Online buffer size = {len(online_rb)} transitions")

        loaded_online_from_cache = True  # treat as cached going forward
        # ↑↑↑ end verbatim train_residual_td3.py:1499-1507

    comms.set_phase(PHASE_PRETRAINING)

    # ↓↓↓ verbatim from train_residual_td3.py:1509-1572
    _hp_parts: list[str] = [
        cfg.task,  # e.g. "TwoArmBoxCleanup"
        f"n{cfg.algo.n_step}",  # n-step horizon
        f"utd{cfg.algo.num_updates_per_iteration}",  # updates-to-data ratio
        f"buf{cfg.algo.buffer_size}",  # replay buffer size
    ]

    # Offline dataset statistics (if any)
    if cfg.offline_data is not None and cfg.offline_data.num_episodes is not None and cfg.algo.offline_fraction > 0.0:
        _hp_parts.append(f"off{cfg.offline_data.num_episodes}ep")
    elif cfg.algo.offline_fraction == 0.0:
        _hp_parts.append("online_only")

    # Learning-rate, expressed in scientific notation for brevity (e.g. 1e-4 → 1e-04)
    _hp_parts.append(f"lr{cfg.agent.actor_lr:.0e}")

    # Additional flags ---------------------------------------------------------
    if cfg.agent.clip_q_target_to_reward_range:
        _hp_parts.append("clipT")

    hp_str = "_".join(_hp_parts)

    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{hp_str}__seed{cfg.seed}"

    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"

    _wandb_config = OmegaConf.to_container(cfg, resolve=True)
    # Remove notes from config if present
    assert isinstance(_wandb_config, dict)
    _wandb_config["wandb"].pop("notes", None)

    # Print a nice summary of the config
    print("Launching run with the following config:")
    pprint.pprint(_wandb_config)
    print("-" * 80)

    _did_offline_rl = getattr(cfg.algo, "train_offline_rl", False) and getattr(cfg.algo, "offline_rl_steps", 0) > 0
    if _did_offline_rl and cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=_wandb_config,
            name=f"{run_name}_offline",
            mode=cfg.wandb.mode if not cfg.debug else "disabled",
            notes=cfg.wandb.notes,
            group=cfg.wandb.group,
            tags=["offline"] + (getattr(cfg.wandb, "tags", []) or []),
        )

    # Log horizon to wandb summary
    if wandb.run is not None:
        wandb.summary["environment/horizon"] = horizon

    # Create a timestamped folder in CACHE_DIR for all outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_cache_dir = _CACHE_ROOT / f"run_{timestamp}_{run_name}"

    # Create subdirectories for models and outputs
    model_save_dir = run_cache_dir / "models"
    outputs_dir = run_cache_dir / "outputs"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ↑↑↑ end verbatim train_residual_td3.py:1509-1572
    obs = None  # original resets the env here; the actor resets when training starts
    # ↓↓↓ verbatim from train_residual_td3.py:1575-1605
    global_step = 0
    best_eval_success_rate = 0.0
    training_cum_time = 0.0
    episode_count = 0

    # ------------------------------------------------------------------
    # Resume from checkpoint (if requested) --------------------------------
    # ------------------------------------------------------------------
    if getattr(cfg, "resume_ckpt", None) is not None:
        resume_path = Path(cfg.resume_ckpt)
        # Accept either a direct .pt file or a directory containing checkpoint.pt
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoint.pt"
        ckpt_info = load_checkpoint(resume_path, agent, device=device)
        global_step = ckpt_info["global_step"]
        best_eval_success_rate = ckpt_info.get("success_rate") or 0.0
        # Restore actor_updates counter (for LR warmup bookkeeping)
        actor_updates = ckpt_info.get("actor_updates", 0)
        print(
            colored(
                f"Resumed from checkpoint: global_step={global_step}, "
                f"best_success_rate={best_eval_success_rate:.4f}, "
                f"actor_updates={actor_updates}",
                "cyan",
            )
        )

    train_start_time = time.time()

    # Initialize timing utility
    training_timer = TrainingTimer()
    # ↑↑↑ end verbatim train_residual_td3.py:1575-1605

    # ↓↓↓ verbatim from train_residual_td3.py:1607-1705
    def _run_critic_warmup(
        agent, online_rb, offline_rb, cfg, device, training_timer, online_batch_size, offline_batch_size,
        model_save_dir=None,
    ):
        """Run critic-only updates for warmup phase.

        `model_save_dir`: if given, saves a checkpoint every `cfg.algo.critic_warmup_save_freq`
        steps (and a final one at the end) to `model_save_dir/critic_warmup_step_<i>/checkpoint.pt`
        - previously this phase saved no checkpoints at all. Uses the SAME `save_checkpoint()`
        utility (and therefore the SAME on-disk format) as every other checkpoint in this script,
        so it's loadable via the existing `resume_ckpt` mechanism without any special-casing.
        """
        save_freq = int(getattr(cfg.algo, "critic_warmup_save_freq", 0) or 0)
        for i in range(cfg.algo.critic_warmup_steps):
            # Sample mixed online/offline batch
            with training_timer.time("batch_sampling"):
                # Sample batches from replay buffers
                online_batch = online_rb.sample(online_batch_size)
                online_batch = online_batch.to(device, non_blocking=True)

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training
                    offline_batch = offline_rb.sample(offline_batch_size)
                    offline_batch = offline_batch.to(device, non_blocking=True)
                    batch = torch.cat([online_batch, offline_batch], dim=0)
                else:
                    # Online-only training
                    batch = online_batch

            # Only update critic during warmup (update_actor=False)
            with training_timer.time("gradient_update"):
                stddev = utils.schedule(cfg.algo.stddev_schedule, i)
                metrics = agent.update(batch, stddev=stddev, update_actor=False, bc_batch=None, ref_agent=agent)

            # Update priorities for prioritized experience replay
            if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                # Update priorities in the batch for priority updates
                td_errors = metrics["_td_errors"]
                batch["_priority"] = td_errors

                if cfg.algo.offline_fraction > 0.0:
                    # Mixed online/offline training - update both buffers
                    online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                    # Update online buffer priorities
                    if online_batch_size_actual > 0:
                        online_batch_subset = batch[:online_batch_size_actual]
                        online_rb.update_tensordict_priority(online_batch_subset)

                    # Update offline buffer priorities
                    if online_batch_size_actual < len(batch):
                        offline_batch_subset = batch[online_batch_size_actual:]
                        offline_rb.update_tensordict_priority(offline_batch_subset)
                else:
                    # Online-only training - update only online buffer
                    online_rb.update_tensordict_priority(batch)

            # Progress logging
            if i % 100 == 0:
                print(
                    f"Critic warmup: {i} / {cfg.algo.critic_warmup_steps}, "
                    f"train/critic_qt={metrics['train/critic_qt']:.4f} "
                    f"train/critic_loss={metrics['train/critic_loss']:.4f}"
                )

            if wandb.run is not None:
                wandb_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
                wandb.log(wandb_metrics, step=i)

            # Checkpointing - every `critic_warmup_save_freq` steps (1-indexed count of
            # completed steps, matching the offline-RL phase's own convention).
            if model_save_dir is not None and save_freq > 0 and (i + 1) % save_freq == 0:
                step_dir = model_save_dir / f"critic_warmup_step_{i + 1}"
                step_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    agent=agent,
                    checkpoint_path=step_dir / "checkpoint.pt",
                    global_step=0,
                    config=cfg,
                    success_rate=0.0,
                    actor_updates=0,  # critic-only phase - the actor is never updated here
                    deployment=_deployment_meta,
                )
                print(colored(f"Critic-warmup checkpoint saved @ {step_dir} (step {i + 1})", "magenta"))

        # Always save a final checkpoint at the end of critic warmup, even if
        # critic_warmup_steps isn't a clean multiple of critic_warmup_save_freq.
        if model_save_dir is not None and cfg.algo.critic_warmup_steps > 0:
            final_dir = model_save_dir / "critic_warmup_final"
            final_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                agent=agent,
                checkpoint_path=final_dir / "checkpoint.pt",
                global_step=0,
                config=cfg,
                success_rate=0.0,
                actor_updates=0,
                deployment=_deployment_meta,
            )
            print(colored(f"Final critic-warmup checkpoint saved @ {final_dir}", "magenta"))

    # ↑↑↑ end verbatim train_residual_td3.py:1607-1705

    eval_env = None
    run_dexmg_evaluation = _skip_eval_in_distributed  # offline-RL region's sim-only eval hook
    # ↓↓↓ verbatim from train_residual_td3.py:1709-1947
    if getattr(cfg.algo, "train_offline_rl", False):
        from resfit.rl_finetuning.off_policy.rl.unified_buffer import EpochUnifiedDataset
        
        print(f"Starting Offline TD3-BC phase for {cfg.algo.offline_rl_steps} steps...")
        
        # Get the underlying tensordicts from the lazy storage
        try:
            td_offline = offline_rb._storage._storage[:len(offline_rb)] if offline_rb is not None and len(offline_rb) > 0 else None
            td_online = online_rb._storage._storage[:len(online_rb)] if online_rb is not None and len(online_rb) > 0 else None
        except Exception as e:
            print(f"Warning: Failed to extract direct tensordict storage: {e}. Attempting fallback.")
            td_offline = offline_rb._storage[:len(offline_rb)] if offline_rb is not None and len(offline_rb) > 0 else None
            td_online = online_rb._storage[:len(online_rb)] if online_rb is not None and len(online_rb) > 0 else None

        if td_offline is not None or td_online is not None:
            offline_sampling = getattr(cfg.algo, "offline_rl_sampling_method", "fixed_ratio")
            offline_ratio = getattr(cfg.algo, "offline_rl_offline_ratio", 0.5)
            print(f"Offline RL sampling: method={offline_sampling}, offline_ratio={offline_ratio}")
            unified_dataset = EpochUnifiedDataset(
                td_offline, td_online,
                sampling_method=offline_sampling,
                offline_ratio=offline_ratio,
            )
            
            # Policy delay parameter for TD3-BC (Actor updates every 2 critic updates)
            policy_delay = 2
            
            for i in range(1, cfg.algo.offline_rl_steps + 1):
                with training_timer.time("batch_sampling"):
                    batch = unified_dataset.sample(cfg.algo.batch_size)
                    batch = batch.to(device, non_blocking=True)
                
                with training_timer.time("gradient_update"):
                    # 1. Critic update (Happens EVERY step)
                    stddev = utils.schedule(cfg.algo.stddev_schedule, i)
                    metrics = agent.update(batch, stddev=stddev, update_actor=False, bc_batch=None, ref_agent=agent)
                    
                    # 2. Delayed Actor update (Happens every 2nd step)
                    if i % policy_delay == 0:
                        # Actor TD3-BC update
                        # For TD3-BC, the target action is the residual correction needed.
                        # As per Rule 6, the target residual in the buffer is computed depending on the mode:
                        # 
                        # Mode 1: GT-as-Base (use_base_policy_for_base_actions = False)
                        # We tell the RL agent that the base policy would have taken the perfect GT action.
                        # - Base Action = GT Action
                        # - Target Action = GT Action
                        # - The true target residual = GT - GT = 0.0. 
                        # This teaches the RL actor: "If the base policy is doing exactly what the human expert did, 
                        # don't interfere. Output a residual of exactly 0.0."
                        #
                        # Mode 2: Base-Policy-as-Base (use_base_policy_for_base_actions = True)
                        # We actually query the neural network.
                        # - Base Action = BC Policy Prediction (Imperfect)
                        # - Target Action = GT Action (Perfect)
                        # - The true target residual = GT - BC Prediction (Non-Zero). 
                        # This teaches the RL actor: "The base policy is making a mistake here. 
                        # You need to output this exact non-zero correction vector to fix it and match the human."
                        # 
                        # We pass the full GT action stored in the buffer. The QAgent will dynamically calculate the target residual.
                        target_action = batch["action"]
                        actor_metrics = agent.update_actor_offline(batch["obs"], target_action, cfg.algo.offline_rl_bc_alpha)
                        metrics.update(actor_metrics)
                    
                if i % cfg.log_freq == 0:
                    print(
                        f"Offline RL: {i} / {cfg.algo.offline_rl_steps}, "
                        f"critic_loss={metrics['train/critic_loss']:.4f} "
                        f"critic_qt={metrics['train/critic_qt']:.4f} "
                        f"actor_loss={metrics.get('train_offline/actor_loss_total', 0):.4f} "
                        f"bc_loss={metrics.get('train_offline/bc_loss', 0):.4f} "
                        f"lambda={metrics.get('train_offline/lambda', 0):.4f}"
                    )
                    # Print debug diagnostics when actor update happened (every 2nd step)
                    if "debug_offline/action_pred_abs_mean" in metrics:
                        print(
                            f"  [DEBUG] pred_res_abs={metrics['debug_offline/action_pred_abs_mean']:.4f} "
                            f"target_res_abs={metrics['debug_offline/target_residual_abs_mean']:.4f} "
                            f"pct_exceeds_scale={metrics['debug_offline/pct_target_exceeds_scale']:.1f}% "
                            f"q_actor={metrics['debug_offline/q_actor_combined']:.4f} "
                            f"q_gt={metrics['debug_offline/q_gt_action']:.4f} "
                            f"combined_vs_gt_mse={metrics['debug_offline/combined_vs_gt_mse']:.4f}"
                        )
                    
                    if wandb.run is not None:
                        # Filter out internal keys (starting with _) before logging to wandb
                        wandb_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
                        wandb.log(wandb_metrics, step=i)
                # Periodic Evaluation & Checkpointing (skipped for real hardware - rely on save_freq
                # checkpointing instead; no interleaved eval rollouts on the real station)
                if not cfg.real_hardware and i % cfg.algo.offline_eval_interval_every_steps == 0:
                    print(f"--- Running offline evaluation at step {i} ---")
                    eval_metrics = run_dexmg_evaluation(
                        env=eval_env,
                        agent=agent,
                        num_episodes=cfg.eval_num_episodes,
                        device=device,
                        global_step=i,
                        save_video=cfg.save_video,
                        save_q_plots=cfg.save_video,
                        run_name=run_name,
                        output_dir=outputs_dir,
                    )
                    
                    if wandb.run is not None:
                        wandb.log(eval_metrics, step=i)
                        
                    current_offline_success = eval_metrics.get("eval/success_rate", 0.0)
                    if current_offline_success > best_eval_success_rate:
                        print(colored(f"🎉 New best offline success rate: {current_offline_success:.4f} (prev: {best_eval_success_rate:.4f})", "green"))
                        best_eval_success_rate = current_offline_success
                        
                        best_offline_dir = model_save_dir / "offline_best"
                        if best_offline_dir.exists():
                            shutil.rmtree(best_offline_dir)
                        best_offline_dir.mkdir(parents=True, exist_ok=True)
                        save_checkpoint(
                            agent=agent,
                            checkpoint_path=best_offline_dir / "checkpoint.pt",
                            global_step=global_step,
                            config=cfg,
                            success_rate=current_offline_success,
                            actor_updates=i // policy_delay,
                            deployment=_deployment_meta,
                        )
                        if wandb.run is not None:
                            art_best = wandb.Artifact(name=f"run_{wandb.run.id}_offline_best", type="model")
                            art_best.add_dir(str(best_offline_dir))
                            wandb.log_artifact(art_best, aliases=["offline_best"])
                    
                if i % cfg.algo.offline_save_freq == 0:
                    offline_save_dir = model_save_dir / f"offline_step_{i}"
                    offline_save_dir.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(
                        agent=agent,
                        checkpoint_path=offline_save_dir / "checkpoint.pt",
                        global_step=global_step,
                        config=cfg,
                        success_rate=eval_metrics.get("eval/success_rate", 0.0) if 'eval_metrics' in locals() else 0.0,
                        actor_updates=i // policy_delay,
                        deployment=_deployment_meta,
                    )
                    print(colored(f"Offline Checkpoint saved @ {offline_save_dir}", "magenta"))
                    
                    if wandb.run is not None:
                        art_step = wandb.Artifact(
                            name=f"run_{wandb.run.id}_offline_model_step_{i}",
                            type="model",
                        )
                        art_step.add_dir(str(offline_save_dir))
                        wandb.log_artifact(art_step, aliases=["offline_latest"])
            
            del unified_dataset
            print("Offline TD3-BC phase completed.")
            
            if 'eval_metrics' in locals():
                best_eval_success_rate = eval_metrics.get("eval/success_rate", 0.0)
                print(f"Initialized best online success rate from final offline eval: {best_eval_success_rate:.4f}")
                
            # Load the specified offline checkpoint for online RL
            load_choice = str(getattr(cfg.algo, "offline_rl_load_ckpt", "best"))
            ckpt_to_load = None
            if load_choice == "best":
                best_offline_dir = model_save_dir / "offline_best"
                ckpt_to_load = best_offline_dir / "checkpoint.pt"
            elif load_choice == "latest":
                pass # The agent in memory is already the latest
            else:
                # specific step number
                ckpt_to_load = model_save_dir / f"offline_step_{load_choice}" / "checkpoint.pt"
            
            if ckpt_to_load and ckpt_to_load.exists():
                print(colored(f"Loading '{load_choice}' offline checkpoint from {ckpt_to_load} for online RL...", "green"))
                ckpt_data = torch.load(ckpt_to_load, map_location=device)
                agent.load_state_dict(ckpt_data["agent_state_dict"])
            elif load_choice != "latest":
                print(colored(f"Warning: Offline checkpoint '{load_choice}' not found at {ckpt_to_load}. Using latest memory weights.", "yellow"))
                
        else:
            print("Warning: Both offline and online buffers are empty. Skipping Offline RL phase.")

    if wandb.run is not None:
        wandb.finish()
        
    # Initialize the main online WandB run
    if cfg.wandb.mode != "disabled":
        wandb.init(
            id=cfg.wandb.continue_run_id,
            resume=None if cfg.wandb.continue_run_id is None else "allow",
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=_wandb_config,
            name=run_name,
            mode=cfg.wandb.mode if not cfg.debug else "disabled",
            notes=cfg.wandb.notes,
            group=cfg.wandb.group,
            tags=getattr(cfg.wandb, "tags", None),
        )
        
    if wandb.run is not None:
        wandb.summary["environment/horizon"] = horizon

    # ------------------------------------------------------------------
    # Critic warmup phase ----------------------------------------------
    # ------------------------------------------------------------------
    _is_resuming = getattr(cfg, "resume_ckpt", None) is not None and global_step > 0
    _did_offline_rl = getattr(cfg.algo, "train_offline_rl", False) and getattr(cfg.algo, "offline_rl_steps", 0) > 0
    if cfg.algo.critic_warmup_steps > 0 and not _is_resuming and not _did_offline_rl:
        print(f"Critic warmup: running {cfg.algo.critic_warmup_steps} critic-only updates...")
        _run_critic_warmup(
            agent=agent,
            online_rb=online_rb,
            offline_rb=offline_rb,
            cfg=cfg,
            device=device,
            training_timer=training_timer,
            online_batch_size=online_batch_size,
            offline_batch_size=offline_batch_size,
            model_save_dir=model_save_dir,
        )
        print("Critic warmup completed.")
    elif _is_resuming:
        print(f"Skipping critic warmup (resuming from step {global_step}).")

    # ------------------------------------------------------------------
    # offline_pretrain_only: stop here, before the online loop -----------
    # ------------------------------------------------------------------
    # Everything above this point (critic-warmup and/or the offline-RL phase) only ever
    # reads from the offline/online replay-buffer caches - no live env was ever created.
    # The `while global_step <= cfg.algo.total_timesteps:` loop below DOES need a live
    # env (env.step()/env.reset() throughout), so it must never be reached in this mode.
    if cfg.algo.offline_pretrain_only:
        if wandb.run is not None:
            wandb.finish()
        print(colored(
            "offline_pretrain_only=True: critic-warmup/offline-RL phase(s) complete - "
            f"stopping here (checkpoints under {model_save_dir}). Not entering the online "
            "training loop (no live env was created).",
            "green",
        ))
        return
    # ↑↑↑ end verbatim train_residual_td3.py:1709-1947

    # ================================================================================
    # Online phase - learner half of train_residual_td3.py:1969-2406
    # ================================================================================
    start_env_step = int(global_step)
    env_step = start_env_step
    comms.extra["start_env_step"] = start_env_step
    comms.extra["env_step"] = env_step
    if not actor_spec_checked:
        actor_spec_checked = _check_actor_spec(comms.actor_spec, learner_spec)

    target_utd = cfg.dist.target_utd
    metrics: dict = {}
    clip_factor = 1.0
    grad_step = 0
    updates_since_start = 0
    last_logged_step = env_step
    next_log = (env_step // cfg.log_freq + 1) * cfg.log_freq
    next_save = (env_step // cfg.save_freq + 1) * cfg.save_freq if cfg.save_freq > 0 else None
    last_stale_warn = time.monotonic()
    t_train_start = time.monotonic()

    comms.set_phase(PHASE_TRAINING)
    comms.publish(agent, grad_step)
    print(colored(f"[learner] initial weights published; online training from env_step={env_step}", "cyan"))

    while True:
        iter_start = time.time()
        comms.drain()

        for item in comms.pop_stats():
            step = max(int(item.get("env_step", 0)), last_logged_step)
            if wandb.run is not None:
                wandb.log(item["data"], step=step)
            last_logged_step = step

        # Env steps come ONLY from the actor's own counter (heartbeat). Counting received
        # transitions instead double-counts warm-up transitions still in flight when
        # training starts (found by e2e_localhost.py: it ended training in 3 s).
        env_step = max(env_step, comms.env_step_reported)
        global_step = env_step  # the verbatim logging/checkpoint regions read `global_step`
        comms.extra["env_step"] = env_step

        if env_step > cfg.algo.total_timesteps or comms.actor_done:
            break

        if comms.last_heartbeat_t and time.monotonic() - comms.last_heartbeat_t > 30:
            if time.monotonic() - last_stale_warn > 30:
                print(colored(
                    f"[learner] no actor heartbeat for {time.monotonic() - comms.last_heartbeat_t:.0f}s - "
                    "still training on the existing buffer",
                    "yellow",
                ))
                last_stale_warn = time.monotonic()

        if target_utd is not None and updates_since_start >= float(target_utd) * max(0, env_step - start_env_step):
            comms.maybe_publish(agent, grad_step, republish_every_s=float(cfg.dist.republish_every_s))
            time.sleep(0.005)
            continue

        stddev = utils.schedule(cfg.algo.stddev_schedule, env_step)
        if cfg.algo.progressive_clipping_steps > 0:
            clip_factor = min(1.0, env_step / cfg.algo.progressive_clipping_steps)

        if True:  # original gate `global_step % update_every_n_steps == 0` removed: learner free-runs
            # ↓↓↓ verbatim from train_residual_td3.py:2193-2276
            i = 0
            actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration
            # Normal training loop - critic is already warmed up
            while i < cfg.algo.num_updates_per_iteration:
                # --------------------------------------------------------------
                # Sample mixed online/offline batch
                # --------------------------------------------------------------
                with training_timer.time("batch_sampling"):
                    # Sample batches from replay buffers
                    online_batch = online_rb.sample(online_batch_size)
                    online_batch = online_batch.to(device, non_blocking=True)

                    if cfg.algo.offline_fraction > 0.0:
                        # Mixed online/offline training
                        offline_batch = offline_rb.sample(offline_batch_size)
                        offline_batch = offline_batch.to(device, non_blocking=True)
                        batch = torch.cat([online_batch, offline_batch], dim=0)
                    else:
                        # Online-only training
                        batch = online_batch

                # Update actor on the last iteration of each update cycle
                update_actor = (i + 1) % actor_update_cadence == 0

                # # Apply actor learning rate warmup
                # if update_actor:
                #     # Start with the base learning rate
                #     current_lr = cfg.agent.actor_lr
                    
                #     # Hack: reduce LR by 10x after 20k online steps
                #     if global_step >= 25000:
                #         current_lr = current_lr * 0.5
                        
                #     if cfg.algo.actor_lr_warmup_steps > 0:
                #         # Calculate current LR with linear warmup from 0 to target
                #         warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                #         current_lr = current_lr * warmup_progress
                        
                #     for param_group in agent.actor_opt.param_groups:
                #         param_group["lr"] = current_lr

                #     actor_updates += 1

                # Apply actor learning rate warmup
                if update_actor:
                    if cfg.algo.actor_lr_warmup_steps > 0:
                        # Calculate current LR with linear warmup from 0 to target
                        warmup_progress = min(1.0, actor_updates / cfg.algo.actor_lr_warmup_steps)
                        current_lr = cfg.agent.actor_lr * warmup_progress
                        for param_group in agent.actor_opt.param_groups:
                            param_group["lr"] = current_lr

                    actor_updates += 1

                with training_timer.time("gradient_update"):
                    metrics = agent.update(batch, stddev, update_actor, bc_batch=None, ref_agent=agent)

                # Update priorities for prioritized experience replay
                if cfg.algo.sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                    # Update priorities in the batch for priority updates
                    td_errors = metrics["_td_errors"]
                    batch["_priority"] = td_errors

                    if cfg.algo.offline_fraction > 0.0:
                        # Mixed online/offline training - update both buffers
                        online_batch_size_actual = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))

                        # Update online buffer priorities
                        if online_batch_size_actual > 0:
                            online_batch_subset = batch[:online_batch_size_actual]
                            online_rb.update_tensordict_priority(online_batch_subset)

                        # Update offline buffer priorities
                        if online_batch_size_actual < len(batch):
                            offline_batch_subset = batch[online_batch_size_actual:]
                            offline_rb.update_tensordict_priority(offline_batch_subset)
                    else:
                        # Online-only training - update only online buffer
                        online_rb.update_tensordict_priority(batch)

                metrics["data/batch_terminal_R"] = batch["next"]["reward"][~batch["nonterminal"]].mean()
                metrics["data/terminal_share"] = (~batch["nonterminal"]).float().mean()

                i += 1
            # ↑↑↑ end verbatim train_residual_td3.py:2193-2276

        grad_step += int(cfg.algo.num_updates_per_iteration)
        updates_since_start += int(cfg.algo.num_updates_per_iteration)
        comms.grad_step = grad_step
        comms.maybe_publish(agent, grad_step, republish_every_s=float(cfg.dist.republish_every_s))
        training_cum_time += time.time() - iter_start

        if next_save is not None and env_step >= next_save:
            next_save = (env_step // cfg.save_freq + 1) * cfg.save_freq
            # ↓↓↓ verbatim from train_residual_td3.py:2127-2187
            # 1) Save a timestamped checkpoint for history
            step_dir = model_save_dir / f"policy_step_{global_step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                agent=agent,
                checkpoint_path=step_dir / "checkpoint.pt",
                global_step=global_step,
                config=cfg,
                success_rate=best_eval_success_rate,
                actor_updates=actor_updates,
                deployment=_deployment_meta,
            )

            # 2) Overwrite the "latest" directory (for resume)
            latest_dir = model_save_dir / "latest"
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            latest_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                agent=agent,
                checkpoint_path=latest_dir / "checkpoint.pt",
                global_step=global_step,
                config=cfg,
                success_rate=best_eval_success_rate,
                actor_updates=actor_updates,
                deployment=_deployment_meta,
            )

            print(
                colored(
                    f"Checkpoint saved (step={global_step}, "
                    f"history @ {step_dir}, latest @ {latest_dir})",
                    "magenta",
                )
            )

            # 3) Push artifacts to WandB
            if wandb.run is not None:
                # Timestamped artifact (keeps history)
                art_step = wandb.Artifact(
                    name=f"run_{wandb.run.id}_model_step_{global_step}",
                    type="model",
                )
                art_step.add_dir(str(step_dir))
                wandb.log_artifact(art_step)

                # "latest" artifact (overwritten each time, for easy resume)
                art_latest = wandb.Artifact(
                    name=f"run_{wandb.run.id}_latest", type="model"
                )
                art_latest.add_dir(str(latest_dir))
                wandb.log_artifact(art_latest, aliases=["latest"])

            # 4) Real hardware only: dump the online buffer (incl. all real transitions
            # collected so far - autonomous AND human-intervened) so it survives a crash and
            # can be inspected mid-run without waiting for training to finish. Overwrites the
            # same "latest" dir each time (buffer growth can be large with images) - mirrors
            # the checkpoint "latest" pattern above. See `_add_transitions_to_buffer()`'s
            # `intervened` field for filtering intervened vs. autonomous transitions, and
            # `scripts/temp/inspect_offline_buffer.py` for inspection.
            if cfg.real_hardware:
                online_buffer_dir = run_cache_dir / "online_buffer_latest"
                optimized_replay_buffer_dumps(online_rb, online_buffer_dir)
            # ↑↑↑ end verbatim train_residual_td3.py:2127-2187

        if env_step >= next_log and metrics:
            next_log = (env_step // cfg.log_freq + 1) * cfg.log_freq
            _elapsed = max(1e-6, time.monotonic() - t_train_start)
            _env_delta = max(1, env_step - start_env_step)
            if wandb.run is not None:
                wandb.log(
                    {
                        "dist/grad_step": grad_step,
                        "dist/effective_utd": updates_since_start / _env_delta,
                        "dist/grad_updates_per_s": updates_since_start / _elapsed,
                        "dist/env_steps_per_s": (env_step - start_env_step) / _elapsed,
                        "dist/inbox_pending": comms.inbox.pending,
                        "dist/weights_published": comms.publisher.n_published,
                        "dist/actor_heartbeat_age_s": (
                            time.monotonic() - comms.last_heartbeat_t if comms.last_heartbeat_t else -1.0
                        ),
                    },
                    step=env_step,
                )
            # ↓↓↓ verbatim from train_residual_td3.py:2284-2406
            sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

            # Prepare base logging dict
            log_dict = {
                "training/SPS": sps,
                "training/global_step": global_step,
                "buffer/online_size": len(online_rb),
                "buffer/offline_size": len(offline_rb) if offline_rb else 0,
                "timing/training_total_time": time.time() - train_start_time,
                "timing/aggregate_steps_per_second": global_step / (time.time() - train_start_time),
                "training/actor_lr": agent.actor_opt.param_groups[0]["lr"],
            }

            # Add timing statistics
            timing_stats = training_timer.get_timing_stats()
            log_dict.update(timing_stats)

            # Add metrics, filtering out internal data
            filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
            log_dict.update(filtered_metrics)

            # Compute residual action statistics only when logging
            if "_actions" in metrics:
                actions = metrics["_actions"]
                # Compute L1/L2 magnitudes (only during logging to save computation)
                residual_l1_magnitude = torch.mean(torch.abs(actions)).item()
                residual_l2_magnitude = torch.mean(torch.square(actions)).item()

                log_dict["train/residual_l1_magnitude"] = residual_l1_magnitude
                log_dict["train/residual_l2_magnitude"] = residual_l2_magnitude
                log_dict["histograms/residual_actions"] = wandb.Histogram(actions.numpy().reshape(-1))

                # ── Per-dimension residual action stats (line plots) ──
                # Tracks how each joint's residual correction evolves over training.
                #   mean    → systematic bias for this joint (should be ~0 unless RL learned a correction)
                #   std     → how "active" RL is on this joint (growing = learning to correct more)
                #   abs_max → is this joint saturating the action_scale range?
                for d in range(actions.shape[-1]):
                    dim_actions = actions[:, d]
                    log_dict[f"residual_per_dim/dim{d}_mean"] = dim_actions.mean().item()
                    log_dict[f"residual_per_dim/dim{d}_std"] = dim_actions.std().item()
                    log_dict[f"residual_per_dim/dim{d}_abs_max"] = dim_actions.abs().max().item()

                # ── Overall residual range evolution (line plots) ──
                # Answers: "Is the actor using the full action_scale range or staying small?"
                abs_actions = actions.abs()
                log_dict["residual_range/abs_max"] = abs_actions.max().item()
                log_dict["residual_range/p95"] = abs_actions.quantile(0.95).item()
                log_dict["residual_range/p50"] = abs_actions.quantile(0.50).item()
                log_dict["residual_range/std"] = actions.std().item()

                # ── Residual RL Analysis: histograms ──
                # Shows the full picture of how residual RL modifies the base BC policy:
                #   base_action       = what BC policy outputs
                #   residual_actions   = what RL wants to add (already logged above)
                #   before_clamp       = base + residual (what RL *intended*)
                #   after_clamp        = clamp(base + residual, -1, 1) (what actually executes)
                # Comparing before_clamp vs after_clamp reveals how much RL gets thrown away.
                if "_base_actions" in metrics:
                    base_actions = metrics["_base_actions"]
                    log_dict["residual_analysis/base_action_avg_magnitude"] = torch.mean(torch.abs(base_actions)).item()
                    log_dict["histograms/bc_base_actions"] = wandb.Histogram(base_actions.numpy().reshape(-1))

                if "_combined_actions" in metrics:
                    combined_actions = metrics["_combined_actions"]
                    log_dict["histograms/after_clamp"] = wandb.Histogram(combined_actions.numpy().reshape(-1))

                if "_unclamped_actions" in metrics:
                    unclamped_actions = metrics["_unclamped_actions"]
                    log_dict["histograms/before_clamp"] = wandb.Histogram(unclamped_actions.numpy().reshape(-1))

            else:
                residual_l1_magnitude = None
                residual_l2_magnitude = None

            # Add Q values histogram when available
            if "_target_q" in metrics:
                target_q = metrics["_target_q"]
                log_dict["histograms/critic_qt"] = wandb.Histogram(target_q.numpy().reshape(-1))

            if cfg.algo.progressive_clipping_steps > 0:
                log_dict["training/progressive_clipping_factor"] = clip_factor

            wandb.log(log_dict, step=global_step)

            # Enhanced print statement with residual action magnitudes, gradient norms, and actor LR
            current_actor_lr = agent.actor_opt.param_groups[0]["lr"]

            if "train/actor_loss_base" in metrics:
                actor_loss_str = f"actor_loss_base={metrics['train/actor_loss_base']:.4f}"
                print_msg = (
                    f"[{global_step}] {actor_loss_str} "
                    f"critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e}"
                )
            else:
                # During critic warmup, actor might not be updated
                print_msg = (
                    f"[{global_step}] critic_loss={metrics['train/critic_loss']:.4f} "
                    f"actor_lr={current_actor_lr:.2e} (actor not updated)"
                )
            if residual_l1_magnitude is not None and residual_l2_magnitude is not None:
                print_msg += f" residual_l1={residual_l1_magnitude:.4f} residual_l2={residual_l2_magnitude:.4f}"

            # Add gradient norms to print statement
            if "train/actor_grad_norm" in metrics:
                print_msg += f" actor_grad_norm={metrics['train/actor_grad_norm']:.4f}"

            # Add L2 penalty if active
            if "train/actor_l2_penalty" in metrics:
                print_msg += f" l2_penalty={metrics['train/actor_l2_penalty']:.4f}"

            # Add timing percentages to print statement
            if timing_stats:
                env_pct = timing_stats.get("timing/env_step_percentage", 0)
                grad_pct = timing_stats.get("timing/gradient_update_percentage", 0)
                batch_pct = timing_stats.get("timing/batch_sampling_percentage", 0)
                eval_pct = timing_stats.get("timing/evaluation_percentage", 0)
                print_msg += (
                    f" | Time%: env={env_pct:.1f} grad={grad_pct:.1f} batch={batch_pct:.1f} eval={eval_pct:.1f}"
                )

            print(print_msg)
            # ↑↑↑ end verbatim train_residual_td3.py:2284-2406
            last_logged_step = max(last_logged_step, env_step)

    comms.set_phase(PHASE_DONE)
    # ↓↓↓ verbatim from train_residual_td3.py:2408-2457
    print(f"Training finished in {time.time() - train_start_time:.2f} seconds.")

    # Final checkpoint at end of training
    if cfg.save_freq > 0:
        final_dir = model_save_dir / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            agent=agent,
            checkpoint_path=final_dir / "checkpoint.pt",
            global_step=global_step,
            config=cfg,
            success_rate=best_eval_success_rate,
            actor_updates=actor_updates,
            deployment=_deployment_meta,
        )
        print(colored(f"Final checkpoint saved @ {final_dir}", "magenta"))

        if wandb.run is not None:
            art_final = wandb.Artifact(
                name=f"run_{wandb.run.id}_final", type="model"
            )
            art_final.add_dir(str(final_dir))
            wandb.log_artifact(art_final, aliases=["final", "latest"])

    # Real hardware only: final dump of the full online buffer (every autonomous AND
    # human-intervened transition collected this run) for post-hoc inspection, e.g. via
    # `scripts/temp/inspect_offline_buffer.py`. Unconditional on `save_freq` (unlike model
    # checkpoints) since this is the only place the run's real-world data ends up on disk at
    # all - must happen before the `no_cleanup`/`run_cache_dir` cleanup below.
    if cfg.real_hardware:
        online_buffer_dir = run_cache_dir / "online_buffer_final"
        optimized_replay_buffer_dumps(online_rb, online_buffer_dir)
        print(colored(f"Final online buffer dumped @ {online_buffer_dir} (size={len(online_rb)})", "magenta"))

    if wandb.run is not None:
        wandb.finish()

    # Clean up entire run directory after successful completion (videos/logs are saved to wandb)
    if cfg.no_cleanup:
        logger.info(
            colored(
                f"Skipping cleanup (no_cleanup=True). Checkpoints preserved at: {run_cache_dir}",
                "yellow",
            )
        )
    elif run_cache_dir.exists():
        print(f"Cleaning up run directory: {run_cache_dir}")
        shutil.rmtree(run_cache_dir)
        print("Run directory cleaned up successfully.")
    # ↑↑↑ end verbatim train_residual_td3.py:2408-2457


# =============================================================================
# ACTOR
# =============================================================================
def actor_main(cfg):
    if cfg.algo.offline_pretrain_only:
        raise RuntimeError("algo.offline_pretrain_only=True touches no environment - run it with role=learner only.")
    fp_hash, fp_flat = config_fingerprint(cfg)  # before main() mutates cfg
    device, device_str = _common_setup(cfg)
    comms = ActorComms(cfg.dist, freeze_encoder=bool(cfg.agent.freeze_encoder))
    try:
        _actor_body(cfg, comms, fp_hash, fp_flat, device, device_str)
    finally:
        print("[actor] flushing outbox and notifying learner...")
        comms.flush_and_stop(done=True)
        r = comms.receiver
        print(
            f"[actor] done. pushed={comms.n_pushed} push_failures={comms.n_push_failures} "
            f"weights_applied={r.n_applied} superseded_before_apply={r.n_dropped} "
            f"last_applied_grad_step={r.last_applied_grad_step}"
        )


def _actor_body(cfg, comms: ActorComms, fp_hash: str, fp_flat: dict, device, device_str):
    init = comms.get_init()
    _verify_fingerprint(fp_hash, fp_flat, init)
    action_scaler, state_standardizer = _normalizers_from_payload(cfg, init["normalization"], device)
    base_policy, eval_base_policy = _load_base_policies(cfg, device)

    # ↓↓↓ verbatim from train_residual_td3.py:608-643
    assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"
    if cfg.real_hardware:
        assert cfg.eval_num_envs == 1, "Real hardware only supports a single station (eval_num_envs must be 1)"
        assert cfg.algo.use_base_policy_for_warmup, (
            "real_hardware=True requires algo.use_base_policy_for_warmup=true - the 'pure random minus "
            "base' warm-up mode can imply arbitrarily large corrections and is not safe on real hardware."
        )
        # Checked here rather than on the first frame: an invalid value would
        # otherwise surface only after the robot is connected and the dataset
        # loaded. Both roles validate, so a typo cannot reach only one side.
        from trossen_real.inference.policy_client import IMAGE_ENCODINGS

        assert cfg.policy_image_encoding in IMAGE_ENCODINGS, (
            f"policy_image_encoding must be one of {list(IMAGE_ENCODINGS)}, "
            f"got {cfg.policy_image_encoding!r}"
        )
        assert 1 <= cfg.policy_jpeg_quality <= 100, (
            f"policy_jpeg_quality must be in [1, 100], got {cfg.policy_jpeg_quality}"
        )

    # If a stage-aware reward model is enabled, verify the server is reachable
    # BEFORE building envs / loading data, and enforce terminated-only bootstrap.
    if cfg.reward_model.enabled:
        if not cfg.algo.terminated_only_bootstrap:
            logger.warning(
                "reward_model.enabled=True but algo.terminated_only_bootstrap=False. "
                "PBRS potentials telescope correctly only with terminated-only bootstrap; "
                "set algo.terminated_only_bootstrap=true."
            )
        from resfit.rl_finetuning.reward_models.reward_client import RewardModelClient

        _rm_client = RewardModelClient(
            server_url=cfg.reward_model.server_url,
            task_prompt=cfg.reward_model.task_prompt,
            request_timeout_s=cfg.reward_model.request_timeout_s,
        )
        if _rm_client.check():
            logger.info(f"Reward server OK at {cfg.reward_model.server_url}: {_rm_client.health()}")
        elif cfg.reward_model.require_server:
            raise RuntimeError(
                f"Reward server not reachable at {cfg.reward_model.server_url}. Start the SARM/TCC "
                f"reward server first, or run a sim-reward script (sparse/dense) instead."
            )
        else:
            logger.warning(
                f"Reward server not reachable at {cfg.reward_model.server_url}; "
                f"continuing (require_server=False)."
            )
    # ↑↑↑ end verbatim train_residual_td3.py:608-643

    get_envs = _make_get_envs(cfg)
    env = get_envs(
        env_name=cfg.task,
        num_envs=cfg.num_envs,
        base_policy=base_policy,
        device=device_str,
        video_key=cfg.video_key,
        debug=cfg.debug,
        action_scaler=action_scaler,
        state_standardizer=state_standardizer,
        apply_reward_model=True,
        enable_intervention=cfg.enable_intervention,
    )
    eval_env = None  # distributed mode runs no interleaved eval rollouts
    # ↓↓↓ verbatim from train_residual_td3.py:700-732
    # Seed environments explicitly for reproducibility
    if hasattr(env, "seed"):
        env.seed(cfg.seed)
    if hasattr(eval_env, "seed"):
        eval_env.seed(cfg.seed + 1)  # Use different seed for eval env to avoid correlation

    if cfg.real_hardware:
        # Best-effort shutdown (park arms, disconnect, stop pedal) on normal exit,
        # uncaught exceptions, and Ctrl-C/SIGTERM - mirrors infer_app/app.py::_cleanup().
        _real_hw_closed = False

        def _cleanup_real_hardware():
            nonlocal _real_hw_closed
            if _real_hw_closed:
                return
            _real_hw_closed = True
            for _e in {id(env): env, id(eval_env): eval_env}.values():
                if _e is None:  # offline_pretrain_only - nothing was ever connected
                    continue
                try:
                    _e.close()
                except Exception:
                    logger.exception("Error while closing real-hardware env")

        atexit.register(_cleanup_real_hardware)

        def _signal_handler(signum, frame):
            logger.warning(f"Received signal {signum} - shutting down real hardware and exiting.")
            _cleanup_real_hardware()
            sys.exit(1)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    # ↑↑↑ end verbatim train_residual_td3.py:700-732

    if isinstance(cfg.rl_camera, str):
        image_keys: list[str] = [cfg.rl_camera]
    else:
        image_keys = list(cfg.rl_camera)
    lowdim_dim = env.observation_space["observation.state"].shape[1]
    img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]
    action_dim = env.action_space.shape[1]
    lowdim_keys = ["observation.state", "observation.base_action"]
    horizon = env.vec_env.metadata["horizon"]

    # Full QAgent (same construction as the learner, so load_state_dict keys match);
    # only encoders + actor are ever used or synced here.
    agent = QAgent(
        obs_shape=(img_c, img_h, img_w),
        prop_shape=(lowdim_dim,),
        action_dim=action_dim,
        rl_cameras=image_keys,
        cfg=cfg.agent,
        residual_actor=True,
    )

    comms.register(
        {
            "action_dim": int(action_dim),
            "lowdim_dim": int(lowdim_dim),
            "img_shape": [int(img_c), int(img_h), int(img_w)],
            "horizon": int(horizon),
            "image_keys": list(image_keys),
        }
    )
    comms.start()

    # One continuous n-step stream across warm-up and online collection - identical to
    # the single MultiStepTransform instance attached to online_rb in the original.
    nstep = NStepStream(
        n_steps=cfg.algo.n_step,
        gamma=cfg.algo.gamma,
        use_terminated_for_bootstrap=cfg.algo.terminated_only_bootstrap,
    )
    online_rb = _NStepSink(nstep, comms.outbox)  # name used by the verbatim regions
    wandb = _ForwardToLearner(comms)  # noqa: F841 - shadows the module inside verbatim regions
    training_timer = TrainingTimer()
    run_name = f"actor_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if cfg.wandb.name is not None:
        run_name = f"{cfg.wandb.name}__{run_name}"
    outputs_dir = _CACHE_ROOT / f"run_{run_name}" / "outputs"

    obs = None
    warmup_started = False
    reward_sum = 0
    episode_count = 0
    success_count = 0
    warm_collected = 0
    last_wait_msg = 0.0
    idle_poll = float(cfg.dist.idle_poll_s)

    while True:
        phase = comms.phase
        if phase == PHASE_DONE:
            print("[actor] learner reports done")
            break

        if phase in (PHASE_INIT, PHASE_PRETRAINING):
            if time.monotonic() - last_wait_msg > 30:
                print(f"[actor] idle - learner phase: {phase}")
                last_wait_msg = time.monotonic()
            time.sleep(idle_poll)
            continue

        if phase == PHASE_WARMUP:
            if not warmup_started:
                print(
                    f"[actor] warm-up: collecting base-policy+noise transitions until the learner "
                    f"has {cfg.algo.learning_starts} (it has {comms.status.get('online_size', 0)})"
                )
                obs, _ = env.reset()
                warmup_started = True
            while comms.phase == PHASE_WARMUP:
                # ↓↓↓ verbatim from train_residual_td3.py:1416-1469 (re-indented +4)
                if cfg.algo.use_base_policy_for_warmup:
                    # Use base policy action + noise (residual exploration)
                    # Since the environment wrapper always adds base_action to residual_action,
                    # we just need to provide the noise as the residual action
                    noise_scale = utils.to_native_list(cfg.algo.random_action_noise_scale)

                    if isinstance(noise_scale, (list, tuple)):
                        noise_scale = torch.tensor(noise_scale, device=device, dtype=torch.float32)
                    rand_actions = (
                        torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                    ) * noise_scale
                else:
                    # Pure uniform random actions - need to cancel out the base policy action
                    # Since env does: combined = base_action + residual_action
                    # To get pure random: residual_action = random - base_action
                    base_action = obs["observation.base_action"]  # Already normalized to [-1, 1]
                    noise_scale = cfg.algo.random_action_noise_scale
                    if isinstance(noise_scale, (list, tuple)):
                        noise_scale = torch.tensor(noise_scale, device=device, dtype=torch.float32)
                    pure_random = (
                        torch.rand((cfg.num_envs, action_dim), device=device) * 2 - 1
                    ) * noise_scale
                    rand_actions = pure_random - base_action

                next_obs, reward, terminated, truncated, info = env.step(rand_actions)
                done = terminated | truncated

                reward_sum += reward.sum().item()
                episode_count += done.float().sum().item()

                # Count actual successes from terminal info (works for sparse/dense/scaled rewards).
                for env_idx in range(cfg.num_envs):
                    if not bool(done[env_idx]):
                        continue
                    ep_info = _extract_final_info_for_env(info, env_idx)
                    if bool(ep_info.get("success", False)):
                        success_count += 1

                # Use the executed combined action returned by the environment
                combined_action = info["scaled_action"]
                _add_transitions_to_buffer(
                    obs=obs,
                    next_obs=next_obs,
                    actions=combined_action,
                    reward=reward,
                    done=done,
                    info=info,
                    device=device,
                    image_keys=image_keys,
                    lowdim_keys=lowdim_keys,
                    num_envs=cfg.num_envs,
                    online_rb=online_rb,
                    terminated=terminated,
                )
                # ↑↑↑ end verbatim train_residual_td3.py:1416-1469 (re-indented +4)
                warm_collected += cfg.num_envs
                if warm_collected % 1000 == 0:
                    success_rate = success_count / episode_count if episode_count > 0 else 0.0
                    print(
                        f"[Warm-up] actor collected {warm_collected} (learner has "
                        f"{comms.status.get('online_size', '?')} / {cfg.algo.learning_starts}), "
                        f"reward_sum={reward_sum:.2f}, success_rate={success_rate:.3f} "
                        f"({success_count}/{int(episode_count)})"
                    )
                # ↓↓↓ verbatim from train_residual_td3.py:1498-1498 (re-indented +4)
                obs = next_obs  # roll state
                # ↑↑↑ end verbatim train_residual_td3.py:1498-1498 (re-indented +4)
            continue

        if phase == PHASE_TRAINING:
            print("[actor] waiting for first weights from learner...")
            t_wait = time.monotonic()
            applied = None
            while applied is None and comms.phase == PHASE_TRAINING:
                applied = comms.receiver.apply_if_new(agent)
                if applied is None:
                    if time.monotonic() - t_wait > 15:
                        print("[actor] still waiting for weights (learner republishes periodically)...")
                        t_wait = time.monotonic()
                    time.sleep(0.1)
            if applied is None:
                continue
            print(colored(f"[actor] weights applied (learner grad_step={applied})", "cyan"))

            # A (re)started actor continues from the learner's env-step count, which is
            # the resumed checkpoint's global_step on a fresh run.
            global_step = max(
                int(comms.status.get("start_env_step", 0)), int(comms.status.get("env_step", 0))
            )
            obs = env.reset()[0]  # original: train_residual_td3.py:1573
            episode_count = 0  # original: train_residual_td3.py:1578
            # ↓↓↓ verbatim from train_residual_td3.py:1949-1967 (re-indented +8)
            # Reward-model debug: randomly record annotated TRAINING rollouts. Only for
            # single-env training (one episode timeline) and when the reward model drives
            # the reward (otherwise there are no stage/PBRS stats to inspect).
            rollout_recorder = TrainingRolloutRecorder(
                output_dir=outputs_dir,
                run_name=run_name,
                fps=int(getattr(env, "fps", 20)),
                sample_prob=float(getattr(cfg, "training_rollout_prob", 0.2)),
                seed=int(cfg.seed),
                num_stages=int(cfg.reward_model.num_stages),
                enabled=bool(getattr(cfg, "save_training_rollouts", False))
                and cfg.reward_model.enabled
                and int(cfg.num_envs) == 1,
            )
            if rollout_recorder.enabled:
                print(
                    f"Training-rollout recorder ON: sampling ~{rollout_recorder.sample_prob:.0%} of "
                    f"episodes -> {rollout_recorder.dir}"
                )
            # ↑↑↑ end verbatim train_residual_td3.py:1949-1967 (re-indented +8)

            while global_step <= cfg.algo.total_timesteps and comms.phase == PHASE_TRAINING:
                comms.receiver.apply_if_new(agent)  # between env steps only - never mid-act()
                # ↓↓↓ verbatim from train_residual_td3.py:1970-2060 (re-indented +8)
                iter_start = time.time()
                # ------------------------------------------------------------------
                # (1) Collect action + Environment step ---------------------------
                # ------------------------------------------------------------------
                with training_timer.time("env_step"):
                    with torch.no_grad(), utils.eval_mode(agent):
                        stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)
                        action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)

                    if cfg.algo.progressive_clipping_steps > 0:
                        clip_factor = min(1.0, global_step / cfg.algo.progressive_clipping_steps)
                        action = action * clip_factor

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated | truncated

                # Update on-screen MuJoCo viewer (no-op when headless=True; not applicable to real hardware)
                if hasattr(env, "render_viewer"):
                    env.render_viewer()

                # Stage-aware reward-model diagnostics (latency, stage, confidence) ---
                if cfg.reward_model.enabled and (global_step % 50 == 0):
                    rm_log = {}
                    for _key, _wandb_key in (
                        ("reward_model_server_latency_s", "reward_model/server_latency_s"),
                        ("reward_model_client_latency_s", "reward_model/client_latency_s"),
                        ("reward_model_stage", "reward_model/stage"),
                        ("reward_model_stage_conf", "reward_model/stage_conf"),
                        ("reward_model_r_shaped", "reward_model/r_shaped"),
                        ("reward_model_r_sparse", "reward_model/r_sparse"),
                    ):
                        _v = info.get(_key, None)
                        if _v is not None:
                            _arr = np.asarray(_v, dtype=np.float32).reshape(-1)
                            if _arr.size:
                                rm_log[_wandb_key] = float(_arr.mean())
                    if rm_log:
                        wandb.log(rm_log, step=global_step)

                # Reward-model debug: capture annotated training-rollout frames + stats.
                if rollout_recorder.enabled:
                    rollout_recorder.record_step(
                        env=env, info=info, global_step=global_step, done=bool(done.any())
                    )

                if done.any():
                    episode_count += done.float().sum().item()
                    # Extract episode information from final_info
                    final_info = info["final_info"]
                    episode_steps = final_info["episode_steps"]
                    episode_indices = final_info["_episode_steps"]

                    # Calculate discounted episode return
                    discount_factor = cfg.algo.gamma ** episode_steps[episode_indices]
                    episode_rewards = reward.cpu().numpy()[episode_indices]
                    episode_return = np.mean(discount_factor * episode_rewards)

                    if rollout_recorder.enabled:
                        # r_shaped on the success step is ~1.0 - phi(prev) (>0.5); use it
                        # as a robust per-episode success flag for the recorder.
                        _ep_success = bool((reward.detach().cpu().numpy()[episode_indices] > 0.5).any())
                        rollout_recorder.end_episode(global_step=global_step, success=_ep_success)

                    wandb.log(
                        {
                            "training/episode_return": episode_return,
                            "training/episode_steps": episode_steps,
                            "training/episode_count": episode_count,
                        },
                        step=global_step,
                    )

                # Add to online replay buffer --------------------------------------
                # Use the executed combined action returned by the environment
                combined_action = info["scaled_action"]
                _add_transitions_to_buffer(
                    obs=obs,
                    next_obs=next_obs,
                    actions=combined_action,
                    reward=reward,
                    done=done,
                    info=info,
                    device=device,
                    image_keys=image_keys,
                    lowdim_keys=lowdim_keys,
                    num_envs=cfg.num_envs,
                    online_rb=online_rb,
                    terminated=terminated,
                )

                obs = next_obs  # roll
                # ↑↑↑ end verbatim train_residual_td3.py:1970-2060 (re-indented +8)
                global_step += cfg.num_envs
                comms.set_env_step(global_step)
                if global_step % cfg.log_freq == 0:
                    wandb.log(
                        {
                            **{f"actor_{k}": v for k, v in training_timer.get_timing_stats().items()},
                            **{f"actor_{k}": v for k, v in comms.receiver.staleness_info.items()},
                            "actor/outbox_len": len(comms.outbox),
                            "actor/transitions_pushed": comms.n_pushed,
                            "actor/push_failures": comms.n_push_failures,
                        },
                        step=global_step,
                    )
            break


# =============================================================================
# Entry
# =============================================================================
def main(cfg):
    role = str(cfg.role)
    if role == "single":
        return _orig.main(cfg)
    if role == "learner":
        return learner_main(cfg)
    if role == "actor":
        return actor_main(cfg)
    raise ValueError(f"role must be one of single|actor|learner, got {role!r}")


@hydra.main(version_base=None, config_name="residual_td3_trossen_real_dist_config")
def hydra_entry(cfg):
    cfg_conf = OmegaConf.structured(cfg)
    main(cfg_conf)


if __name__ == "__main__":
    hydra_entry()
