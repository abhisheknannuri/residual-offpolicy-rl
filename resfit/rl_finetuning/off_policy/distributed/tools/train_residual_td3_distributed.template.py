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

    def __init__(self, url: str):
        self._url = url
        self._client = None

    @property
    def policy(self):
        if self._client is None:
            from trossen_real.inference.policy_client import PolicyClient

            client = PolicyClient(self._url)
            health = client.health()
            if not health.get("loaded"):
                raise RuntimeError(
                    f"[learner] offline buffer needs base actions from policy_server.py at {self._url}, "
                    f"but it has no policy loaded - {health}. If the server runs on the laptop, "
                    "SSH-forward its port to this machine."
                )
            print(f"[learner] querying base policy at {self._url} to populate the offline buffer")
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
    #@@SPLICE 305 326@@
    #@@SPLICE 588 603@@
    return device, device_str


def _load_base_policies(cfg, device):
    #@@SPLICE 337 375@@
    return base_policy, eval_base_policy


def _load_dataset_and_normalizers(cfg, device):
    #@@SPLICE 377 437@@
    return (
        dataset,
        action_scaler,
        state_standardizer,
        offline_image_transforms,
        _query_base_actions_from_remote_policy,
    )


def _make_get_envs(cfg):
    #@@SPLICE 439 583@@
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

    #@@SPLICE 791 797@@

    # ---- replay buffers ---------------------------------------------------------------
    #@@SPLICE 806 814@@
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
    #@@SPLICE 831 934@@

    comms.attach_online_buffer(online_rb)
    comms.online_size = len(online_rb)

    env = _LazyRemotePolicyEnv(cfg.policy_server_url)
    #@@SPLICE 936 1347@@

    # ---- warm-up gate: the ACTOR collects, the learner waits ------------------------
    actor_spec_checked = _check_actor_spec(comms.actor_spec, learner_spec)
    if len(online_rb) < cfg.algo.learning_starts:
        #@@SPLICE 1363 1383@@
        online_cache_dir.mkdir(parents=True, exist_ok=True)
        next_log_threshold = (len(online_rb) // 1000 + 1) * 1000
        #@@SPLICE 1405 1408@@
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
            #@@SPLICE 1488 1496 shift=-4@@
        #@@SPLICE 1499 1507@@

    comms.set_phase(PHASE_PRETRAINING)

    #@@SPLICE 1509 1572@@
    obs = None  # original resets the env here; the actor resets when training starts
    #@@SPLICE 1575 1605@@

    #@@SPLICE 1607 1705@@

    eval_env = None
    run_dexmg_evaluation = _skip_eval_in_distributed  # offline-RL region's sim-only eval hook
    #@@SPLICE 1709 1947@@

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
            #@@SPLICE 2193 2276@@

        grad_step += int(cfg.algo.num_updates_per_iteration)
        updates_since_start += int(cfg.algo.num_updates_per_iteration)
        comms.grad_step = grad_step
        comms.maybe_publish(agent, grad_step, republish_every_s=float(cfg.dist.republish_every_s))
        training_cum_time += time.time() - iter_start

        if next_save is not None and env_step >= next_save:
            next_save = (env_step // cfg.save_freq + 1) * cfg.save_freq
            #@@SPLICE 2127 2187@@

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
            #@@SPLICE 2284 2406@@
            last_logged_step = max(last_logged_step, env_step)

    comms.set_phase(PHASE_DONE)
    #@@SPLICE 2408 2457@@


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

    #@@SPLICE 608 643@@

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
    #@@SPLICE 700 732@@

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
                #@@SPLICE 1416 1469 shift=+4@@
                warm_collected += cfg.num_envs
                if warm_collected % 1000 == 0:
                    success_rate = success_count / episode_count if episode_count > 0 else 0.0
                    print(
                        f"[Warm-up] actor collected {warm_collected} (learner has "
                        f"{comms.status.get('online_size', '?')} / {cfg.algo.learning_starts}), "
                        f"reward_sum={reward_sum:.2f}, success_rate={success_rate:.3f} "
                        f"({success_count}/{int(episode_count)})"
                    )
                #@@SPLICE 1498 1498 shift=+4@@
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
            #@@SPLICE 1949 1967 shift=+8@@

            while global_step <= cfg.algo.total_timesteps and comms.phase == PHASE_TRAINING:
                comms.receiver.apply_if_new(agent)  # between env steps only - never mid-act()
                #@@SPLICE 1970 2060 shift=+8@@
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
