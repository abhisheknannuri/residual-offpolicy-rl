# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Potential-Based Reward Shaping (PBRS) gym wrapper driven by a reward server.

Applied as a per-(sub)env wrapper on the *training* environment only. Because it
lives at the single-env level it sees clean ``reset()`` / ``step()`` episode
boundaries (no vector-autoreset bookkeeping) and, under ``SyncVectorEnv``, runs
in-process so the synchronous HTTP calls to the reward server just work.

Reward transform (PBRS, policy-invariant):

    r_shaped = r_sparse + gamma * phi(s') - phi(s)

where ``phi`` is a small per-stage potential and the stage is the monotonic,
server-gated task stage. On a true termination ``phi(s') = 0`` (absorbing).
The server owns the frame history + windowing + hysteresis; this wrapper only
streams the new frames/states collected since the last query and applies the
returned gated stage.
"""

from __future__ import annotations

import logging

import numpy as np

from resfit.rl_finetuning.reward_models.reward_client import RewardModelClient

logger = logging.getLogger(__name__)


class StageAwarePBRSWrapper:
    """Replace the env reward with a stage-aware PBRS signal.

    Args:
        env: the wrapped single environment (returns a gymnasium 5-tuple).
        client: a connected :class:`RewardModelClient`.
        session_id: unique per-env session key for server-side hysteresis.
        potentials: per-stage potential values (index = stage).
        gamma: discount used in the PBRS shaping term.
        query_every_k: query the server every ``k`` env steps (batches the k
            skipped frames into one request).
        image_key: obs key holding the camera frame (``[C, H, W]`` float in [0, 1]).
        keep_sparse_term: keep the env's sparse reward as the PBRS base term.
        hysteresis_k / conf_threshold / monotonic: server-side gating params.
        image_vflip: vertically flip frames before sending (robosuite obs are
            often upside-down relative to the reward model's training frames).
        task_prompt: natural-language task string forwarded to the reward model.
    """

    def __init__(
        self,
        env,
        *,
        client: RewardModelClient,
        session_id: str,
        potentials,
        gamma: float,
        query_every_k: int = 5,
        image_key: str = "observation.images.agentview",
        state_key: str = "observation.state",
        keep_sparse_term: bool = True,
        hysteresis_k: int = 5,
        conf_threshold: float = 0.80,
        monotonic: bool = True,
        image_vflip: bool = False,
        task_prompt: str = "",
        reward_mode: str = "pbrs",
        milestone_payouts=None,
        milestone_success_bonus: float = 0.7,
    ) -> None:
        self.env = env
        self.client = client
        self.session_id = str(session_id)
        self.potentials = [float(p) for p in potentials]
        self.num_stages = len(self.potentials)
        self.gamma = float(gamma)
        self.query_every_k = max(1, int(query_every_k))
        self.image_key = image_key
        self.state_key = state_key
        self.keep_sparse_term = bool(keep_sparse_term)
        self.hysteresis_k = int(hysteresis_k)
        self.conf_threshold = float(conf_threshold)
        self.monotonic = bool(monotonic)
        self.image_vflip = bool(image_vflip)
        self.task_prompt = task_prompt

        # Reward mode: "pbrs" (default, policy-invariant) or "milestone" (ratchet
        # one-time stage payouts + terminal success bonus; NOT policy-invariant).
        self.reward_mode = str(reward_mode)
        self.milestone_payouts = (
            [float(p) for p in milestone_payouts]
            if milestone_payouts is not None
            else [0.0] * self.num_stages
        )
        self.milestone_success_bonus = float(milestone_success_bonus)
        self._last_paid_stage = 0

        self._pending_frames: list[np.ndarray] = []
        self._pending_states: list[np.ndarray] = []
        self._step_ctr = 0
        self._current_stage = 0
        self._raw_stage = 0
        self._stage_conf = 1.0
        self._phi_current = self.potentials[0]

    # ------------------------------------------------------------------
    # Frame / state extraction
    # ------------------------------------------------------------------
    def _extract(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        img = np.asarray(obs[self.image_key])
        # [C, H, W] float in [0, 1] -> [H, W, C] uint8
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if self.image_vflip:
            img = img[::-1].copy()
        state = np.asarray(obs[self.state_key], dtype=np.float32).reshape(-1)
        return img, state

    def _query(self, reset: bool) -> None:
        if not self._pending_frames:
            return
        frames = np.stack(self._pending_frames, axis=0)
        states = np.stack(self._pending_states, axis=0)
        num_anchors = frames.shape[0]
        try:
            out = self.client.predict_stage(
                frames,
                states,
                session_id=self.session_id,
                num_anchors=num_anchors,
                reset=reset,
                hysteresis_k=self.hysteresis_k,
                conf_threshold=self.conf_threshold,
                monotonic=self.monotonic,
                task=self.task_prompt,
            )
        except Exception as e:  # noqa: BLE001
            # The reward server is unreachable/failing mid-training. We MUST NOT
            # silently continue with a stale/garbage reward — that would corrupt
            # the whole RL run. Fail loudly and clearly so training stops; the
            # user can restart the reward server and resume RL from a checkpoint.
            raise RuntimeError(
                f"Reward server call failed mid-training ({type(e).__name__}: {e}). "
                f"Aborting RL: the stage-aware reward cannot be computed without the "
                f"server. Restart the reward server and resume training from the last "
                f"checkpoint."
            ) from e
        stage = int(out["gated_stage"])
        if self.monotonic:
            stage = max(self._current_stage, stage)
        stage = min(max(stage, 0), self.num_stages - 1)
        self._current_stage = stage
        # Raw (pre-hysteresis) argmax stage of the most-recent scored anchor, kept
        # only for debugging so we can see when the reward model *would* have
        # jumped stages before the server-side hysteresis gate smoothed it.
        _raw = out.get("raw_stages", None)
        if _raw:
            self._raw_stage = int(min(max(int(_raw[-1]), 0), self.num_stages - 1))
        self._stage_conf = float(out.get("stage_conf", self._stage_conf))
        self._last_server_latency = float(out.get("server_latency_s", 0.0))
        self._last_client_latency = float(out.get("client_latency_s", 0.0))
        self._pending_frames.clear()
        self._pending_states.clear()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._pending_frames.clear()
        self._pending_states.clear()
        self._step_ctr = 0
        self._current_stage = 0
        self._raw_stage = 0
        self._stage_conf = 1.0
        self._last_server_latency = 0.0
        self._last_client_latency = 0.0

        img, state = self._extract(obs)
        self._pending_frames.append(img)
        self._pending_states.append(state)
        self._query(reset=True)
        self._phi_current = self.potentials[self._current_stage]
        self._last_paid_stage = self._current_stage
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r_sparse = float(reward)

        img, state = self._extract(obs)
        self._pending_frames.append(img)
        self._pending_states.append(state)
        self._step_ctr += 1

        # Query cadence: every k steps and always flush on truncation. In milestone
        # mode we ALSO query on termination so the final stage is scored before the
        # success bonus is paid (matches the offline labeler, which flushes on the
        # terminal frame). PBRS skips the terminal query (phi'=0 there anyway).
        if (
            (self._step_ctr % self.query_every_k == 0)
            or bool(truncated)
            or (bool(terminated) and self.reward_mode == "milestone")
        ):
            self._query(reset=False)

        if self.reward_mode == "milestone":
            # Ratchet milestone reward (NOT potential-based). Pay once per newly-
            # entered stage (skipped stages summed); add the dominant terminal
            # success bonus gated on the TRUE simulator success flag (r_sparse > 0).
            r_out = 0.0
            if self._current_stage > self._last_paid_stage:
                for _s in range(self._last_paid_stage + 1, self._current_stage + 1):
                    r_out += self.milestone_payouts[_s]
                self._last_paid_stage = self._current_stage
            if bool(terminated) and r_sparse > 0.5:
                r_out += self.milestone_success_bonus
            r_out = float(r_out)
            r_shaped = r_out
        else:
            # PBRS (policy-invariant, telescopes under n-step) — unchanged.
            if bool(terminated):
                phi_next = 0.0
            else:
                phi_next = self.potentials[self._current_stage]
            r_shaped = (r_sparse if self.keep_sparse_term else 0.0) + self.gamma * phi_next - self._phi_current
            self._phi_current = phi_next
            r_out = float(r_shaped)

        info = dict(info)
        info["reward_model_stage"] = int(self._current_stage)
        info["reward_model_raw_stage"] = int(self._raw_stage)
        info["reward_model_stage_conf"] = float(self._stage_conf)
        info["reward_model_server_latency_s"] = float(getattr(self, "_last_server_latency", 0.0))
        info["reward_model_client_latency_s"] = float(getattr(self, "_last_client_latency", 0.0))
        info["reward_model_r_sparse"] = r_sparse
        info["reward_model_r_shaped"] = float(r_shaped)
        info["reward_model_reward"] = r_out
        info["reward_model_r_milestone"] = r_out if self.reward_mode == "milestone" else 0.0
        info["reward_model_last_paid_stage"] = int(self._last_paid_stage)

        return obs, r_out, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        if hasattr(self.env, "get_wrapper_attr"):
            return self.env.get_wrapper_attr(name)
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def set_wrapper_attr(self, name: str, value):
        if hasattr(self, name):
            setattr(self, name, value)
            return
        if hasattr(self.env, "set_wrapper_attr"):
            self.env.set_wrapper_attr(name, value)
            return
        setattr(self.env, name, value)

    def __getattr__(self, name):
        return getattr(self.env, name)
