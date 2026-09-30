"""End-to-end localhost test: real learner process + real actor process over ZMQ.

Runs the GENERATED entrypoint (train_residual_td3_distributed.py) unmodified, with
only two things faked because they need hardware / a real dataset:
  - the env (a TrossenResidualEnv-shaped fake: same obs/action spaces, info keys)
  - the dataset (a LeRobotDataset-shaped fake: meta.stats, meta.episodes, samples)
Everything else is real: QAgent (ViT), torchrl buffers, offline buffer population
and caching, n-step on the actor, agentlace transport, phase machine, weight sync,
TD3 updates, checkpointing, wandb (offline mode).

All outputs (CACHE_DIR, wandb, logs) go to a temp workdir - nothing is written
into the repo.

Usage (no uv - call an interpreter directly):
    PYTHONPATH=. <python> resfit/rl_finetuning/off_policy/distributed/tests/e2e_localhost.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
PORT, BPORT = 5788, 5789  # not the defaults, so a real run on this machine can't collide

OVERRIDES = [
    "real_hardware=true",
    "offline_data.image_size=84",
    "offline_data.num_episodes=3",
    "offline_data.use_base_policy_for_base_actions=false",
    "algo.offline_fraction=0.5",
    "algo.n_step=3",
    "algo.gamma=0.99",
    "algo.terminated_only_bootstrap=true",
    "algo.learning_starts=60",
    "algo.buffer_size=2000",
    "algo.batch_size=16",
    "algo.num_updates_per_iteration=4",
    "algo.actor_updates_per_iteration=1",
    "algo.total_timesteps=160",
    "algo.critic_warmup_steps=3",
    "algo.critic_warmup_save_freq=0",
    "algo.online_warmup_save_freq=30",
    "algo.prefetch_batches=0",
    "algo.use_base_policy_for_warmup=true",
    "real_max_steps=25",
    "log_freq=40",
    "save_freq=80",
    "wandb.mode=offline",
    "wandb.project=resfit-dist-e2e",
    "no_cleanup=true",
    f"dist.port={PORT}",
    f"dist.broadcast_port={BPORT}",
    "dist.steps_per_update=8",
    "dist.push_interval_s=0.25",
    "dist.republish_every_s=2.0",
    "dist.idle_poll_s=0.1",
]


# -----------------------------------------------------------------------------
# fakes
# -----------------------------------------------------------------------------
def _fakes():
    import gymnasium as gym
    import numpy as np
    import torch

    CAMS = ["observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    A = 7

    class FakeMeta:
        def __init__(self, n_eps, ep_len):
            self.stats = {
                "action": {"min": np.full(A, -1.0, np.float32), "max": np.full(A, 1.0, np.float32)},
                "observation.state": {"mean": np.zeros(A, np.float32), "std": np.ones(A, np.float32)},
            }
            self.episodes = {i: {"length": ep_len} for i in range(n_eps)}
            self.total_episodes = n_eps
            self.total_frames = n_eps * ep_len

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, n_eps=3, ep_len=12):
            self.meta = FakeMeta(n_eps, ep_len)
            self.ep_len = ep_len
            self.n = n_eps * ep_len

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            ep, t = divmod(i, self.ep_len)
            g = torch.Generator().manual_seed(i)
            d = {
                "episode_index": torch.tensor(ep),
                "index": torch.tensor(i),
                "action": torch.rand(A, generator=g) * 2 - 1,
                "next.done": torch.tensor(t == self.ep_len - 1),
                "observation.state": torch.randn(A, generator=g),
            }
            for c in CAMS:
                d[c] = torch.rand(3, 84, 84, generator=g)
            return d

    class FakeTrossenEnv:
        """Mirrors TrossenResidualEnv's interface as used by the training loops."""

        fps = 50  # env.step() blocks at this rate, like TrossenResidualEnv's control loop

        def __init__(self, device, horizon):
            self.device = torch.device(device)
            self.horizon = horizon
            spaces = {
                "observation.state": gym.spaces.Box(-np.inf, np.inf, (1, A), np.float32),
                "observation.base_action": gym.spaces.Box(-1, 1, (1, A), np.float32),
            }
            for c in CAMS:
                spaces[c] = gym.spaces.Box(0, 1, (1, 3, 84, 84), np.float32)
            self.observation_space = gym.spaces.Dict(spaces)
            self.action_space = gym.spaces.Box(-1, 1, (1, A), np.float32)
            self.metadata = {"horizon": horizon}
            self.vec_env = self
            self.t = 0
            self.policy = None
            self.n_steps = 0

        def _obs(self):
            o = {
                "observation.state": torch.randn(1, A, device=self.device),
                "observation.base_action": torch.rand(1, A, device=self.device) * 2 - 1,
            }
            for c in CAMS:
                o[c] = torch.rand(1, 3, 84, 84, device=self.device)
            return o

        def reset(self, **_):
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            time.sleep(1.0 / self.fps)
            self.t += 1
            self.n_steps += 1
            done = self.t >= self.horizon
            base = torch.rand(1, A, device=self.device) * 2 - 1
            info = {"scaled_action": (base + action).clamp(-1, 1), "intervened": False}
            reward = torch.tensor([1.0 if done else 0.0], device=self.device)
            terminated = torch.tensor([done], device=self.device)
            truncated = torch.tensor([False], device=self.device)
            obs = self._obs()
            if done:
                info["final_info"] = {
                    "episode_steps": np.array([self.t]),
                    "_episode_steps": np.array([True]),
                    "success": np.array([True]),
                }
                info["_final_info"] = np.array([True])
                obs, _ = self.reset()
            return obs, reward, terminated, truncated, info

        def close(self):
            pass

    return FakeDataset, FakeTrossenEnv


# -----------------------------------------------------------------------------
# one role, in this process
# -----------------------------------------------------------------------------
def run_role(role: str, workdir: Path) -> None:
    os.environ["CACHE_DIR"] = str(workdir / "cache")  # read at import of the original module
    os.environ["WANDB_DIR"] = str(workdir / f"wandb_{role}")
    os.environ["WANDB_SILENT"] = "true"
    (workdir / f"wandb_{role}").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))

    import trossen_real.log_setup as log_setup

    log_setup.LOG_ROOT = workdir / "logs"  # keep the repo's logs/ untouched

    import resfit.rl_finetuning.scripts.train_residual_td3_distributed as D
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer

    FakeDataset, FakeTrossenEnv = _fakes()

    def fake_dataset_and_normalizers(cfg, device):
        ds = FakeDataset()
        a = ActionScaler.from_dataset_stats(
            action_stats=ds.meta.stats["action"],
            action_scale=cfg.agent.actor.action_scale,
            min_range_per_dim=cfg.offline_data.min_action_range,
            device=device,
        )
        s = StateStandardizer.from_dataset_stats(
            state_stats=ds.meta.stats["observation.state"], min_std=cfg.offline_data.min_state_std, device=device
        )
        return ds, a, s, None, False

    D._load_dataset_and_normalizers = fake_dataset_and_normalizers
    D._make_get_envs = lambda cfg: (lambda **kw: FakeTrossenEnv(kw["device"], cfg.real_max_steps))

    with initialize(version_base=None):
        cfg = compose(config_name="residual_td3_trossen_real_dist_config", overrides=OVERRIDES + [f"role={role}"])
    D.main(OmegaConf.structured(cfg))
    print(f"E2E_ROLE_EXIT_OK {role}", flush=True)


# -----------------------------------------------------------------------------
# orchestrator
# -----------------------------------------------------------------------------
def orchestrate() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="resfit_dist_e2e_"))
    print(f"workdir: {workdir}")
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1")
    logs = {r: open(workdir / f"{r}.log", "w") for r in ("learner", "actor")}
    procs = {}
    procs["learner"] = subprocess.Popen(
        [sys.executable, __file__, "learner", str(workdir)], stdout=logs["learner"], stderr=subprocess.STDOUT,
        env=env, cwd=workdir,
    )
    time.sleep(3)  # actor must also tolerate starting first; this just keeps logs readable
    procs["actor"] = subprocess.Popen(
        [sys.executable, __file__, "actor", str(workdir)], stdout=logs["actor"], stderr=subprocess.STDOUT,
        env=env, cwd=workdir,
    )
    deadline = time.time() + 600
    while time.time() < deadline and any(p.poll() is None for p in procs.values()):
        time.sleep(2)
    for r, p in procs.items():
        if p.poll() is None:
            print(f"TIMEOUT: killing {r}")
            p.kill()
    for f in logs.values():
        f.close()

    text = {r: (workdir / f"{r}.log").read_text() for r in procs}
    ckpts = sorted(str(p.relative_to(workdir)) for p in (workdir / "cache").rglob("checkpoint.pt"))

    import re as _re

    def _applied(t):
        m = _re.search(r"weights_applied=(\d+)", t)
        return int(m.group(1)) if m else 0

    def _last_grad(t):
        m = _re.search(r"last_applied_grad_step=(-?\d+)", t)
        return int(m.group(1)) if m else -1

    def has(role, s):
        return s in text[role]

    checks = [
        ("learner exited cleanly", procs["learner"].returncode == 0 and has("learner", "E2E_ROLE_EXIT_OK learner")),
        ("actor exited cleanly", procs["actor"].returncode == 0 and has("actor", "E2E_ROLE_EXIT_OK actor")),
        ("no Traceback in learner", "Traceback" not in text["learner"]),
        ("no Traceback in actor", "Traceback" not in text["actor"]),
        ("fingerprint verified", has("actor", "config fingerprint matches learner")),
        ("normalization verified", has("actor", "normalization verified against learner probe")),
        ("actor registered", has("learner", "actor registered")),
        ("offline buffer populated", has("learner", "offline transitions to buffer")),
        ("learner went through warmup", has("learner", "phase: init -> warmup")),
        ("warm-up buffer dumped", has("learner", "Warm-up done. Online buffer size")),
        ("critic warmup ran", has("learner", "Critic warmup completed.")),
        ("learner entered training", has("learner", "phase: pretraining -> training")),
        ("initial weights published", has("learner", "initial weights published")),
        ("actor applied weights", has("actor", "weights applied (learner grad_step=")),
        ("checkpoint saved", any("policy_step_" in c for c in ckpts)),
        ("final checkpoint saved", any("/final/" in c for c in ckpts)),
        ("learner logged metrics", has("learner", "critic_loss=")),
        ("weights synced repeatedly (not just initial)", _applied(text["actor"]) >= 3),
        ("actor saw a trained snapshot (grad_step > 0)", _last_grad(text["actor"]) > 0),
    ]
    width = max(len(n) for n, _ in checks)
    ok = True
    for name, passed in checks:
        ok &= bool(passed)
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<{width}}")
    print(f"\ncheckpoints: {ckpts}")
    print(f"logs: {workdir}/learner.log  {workdir}/actor.log")
    if not ok:
        for r in procs:
            print(f"\n===== tail {r}.log =====")
            print("\n".join(text[r].splitlines()[-40:]))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] in ("learner", "actor"):
        run_role(sys.argv[1], Path(sys.argv[2]))
    else:
        sys.exit(orchestrate())
