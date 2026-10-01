# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Real-hardware, single-station counterpart to `BasePolicyVecEnvWrapper`
(`resfit/rl_finetuning/wrappers/residual_env_wrapper.py`) - drives the ACTUAL
Trossen follower arm during residual RL training/eval, instead of a
vectorized MuJoCo sim.

Deliberately mirrors `BasePolicyVecEnvWrapper`'s exact contract (same
`info["scaled_action"]` semantics, same `observation.base_action` obs
augmentation, same `(1, dim)`-shaped batched tensors even though there's
only ever ONE real station) so `train_residual_td3.py`'s existing
`_add_transitions_to_buffer()` / `agent.act()` / `agent.update()` /
checkpointing / wandb logging code paths all consume it with ZERO changes.

Design doc: `docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md`.

Optional human-intervention takeover (leader + foot pedal), reusing Phase
1's `InterventionManager` as-is:
  - When `pedal is None` (`enable_intervention=False`), `self.intervention`
    is None and every branch below behaves EXACTLY like a plain autonomous
    env - no leader involved at all, zero regression risk.
  - When enabled: the base policy is ALWAYS queried and the residual actor
    is ALWAYS evaluated by the caller (this class never skips anything) -
    only what gets EXECUTED and STORED for that step differs. While
    intervened, the leader's own absolute joint readback is sent to the
    follower (not `base + residual`), and the buffer's stored action for
    that transition is the human's ACTUAL combined action (converted to the
    same per-frame delta/normalized convention as every other transition) -
    not the residual the actor happened to propose. No separate buffer, no
    extra training-loop code: the critic learns `Q(s, a_human)` for free via
    standard TD learning, and the actor improves indirectly via the
    existing Q-maximization gradient-ascent step - see the design doc's
    Section 4 for the full reasoning.

Episode-boundary control (reset pedal / reward pedal), via the SAME shared
`EpisodeBoundaryMonitor` teleop/infer use (`trossen_real.human_intervention.
episode_boundary`, see `PEDAL_BEHAVIOR.md`) - independent of `pedal is None`
vs `intervention is None` above, only needs `pedal`:
  - Reset pedal: a single press ends the episode (unchanged from before -
    `EpisodeBoundaryMonitor` just wraps the same `consume_reset()` primitive).
  - Reward pedal: `reward=1.0` every step while held (unchanged); NEW - on
    the held->released edge, if held at some point this episode, ALSO ends
    the episode (mirrors "Stop Recording"/"Stop Inference" in teleop/infer -
    releasing the reward pedal after marking success now ends the episode
    here too, not just in the UI apps).
  - `EpisodeBoundaryMonitor.start_episode()` (called from `reset()`) drains
    any stale, unconsumed reset-pedal press or reward latch from BEFORE this
    episode began - fixes a real bug where a second reset-pedal press
    landing during the settle window truncated the NEXT episode to 1 step.
"""

from __future__ import annotations

import logging
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from trossen_real.cameras.camera_manager import CameraManager
from trossen_real.config import ControlConfig, ResetConfig
from trossen_real.human_intervention.episode_boundary import EpisodeBoundaryMonitor
from trossen_real.human_intervention.pedal_listener import PedalListener
from trossen_real.inference.intervention import InterventionManager
from trossen_real.inference.policy_client import (
    PolicyClient,
    TickLogger,
    build_observation,
    reconstruct_absolute_action,
)
from trossen_real.leader.trossen_leader_single import TrossenSingleLeader
from trossen_real.teleop.follower_client import FollowerClient

logger = logging.getLogger(__name__)


class TrossenResidualEnv:
    """Single real Trossen station, gym-`Env`-like (`reset()`/`step()`/`close()`),
    NOT vectorized - real residual RL only ever has one physical station.
    """

    def __init__(
        self,
        follower: FollowerClient,
        cameras: CameraManager,
        policy: PolicyClient,
        control: ControlConfig,
        reset_cfg: ResetConfig,
        action_scaler,
        state_standardizer,
        image_keys: list[str],
        camera_resolution: tuple[int, int],
        action_dim: int = 7,
        action_space: str = "delta_joint",
        leader: TrossenSingleLeader | None = None,
        pedal: PedalListener | None = None,
        max_steps: int | None = None,
        settle_time_s: float = 2.0,
        rl_image_size: int | None = None,
        log_file: str | None = None,
        policy_image_encoding: str = "raw",
        policy_jpeg_quality: int = 95,
    ) -> None:
        self.follower = follower
        self.cameras = cameras
        self.policy = policy
        self.control = control
        self.reset_cfg = reset_cfg
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.image_keys = image_keys
        self.action_space_name = action_space
        self.leader = leader
        self.pedal = pedal
        self.max_steps = max_steps
        self.settle_time_s = settle_time_s
        self.rl_image_size = rl_image_size
        # How frames are packed for policy_server.py. Only affects the wire
        # format of the base-policy query - the images STORED in the replay
        # buffer are the rl_image_size resizes and are untouched by this.
        self.policy_image_encoding = policy_image_encoding
        self.policy_jpeg_quality = policy_jpeg_quality
        # All of THIS env's own tensors (state, images, base-policy action, stored action)
        # are built on the SAME device as action_scaler/state_standardizer - matching
        # train_residual_td3.py's `device` (e.g. cuda) so they can be combined directly
        # with agent.act()'s output / the warm-up loop's torch.rand(..., device=device)
        # without a "tensors on different devices" crash. Sim's env doesn't need this
        # explicit handling since its obs are already built on `device` end-to-end; real
        # hardware builds everything from numpy (camera/robot readouts), which defaults
        # to CPU unless moved explicitly.
        self.device = getattr(action_scaler, "device", torch.device("cpu"))
        # Optional per-step JSONL diagnostic log (everything EXCEPT images - reward,
        # terminated/truncated, intervened flag, real-units executed action, normalized
        # stored/proposed actions, raw follower state, achieved Hz). NOT a substitute for
        # the replay buffer (training reads from the buffer, never this file) - purely for
        # offline inspection (e.g. "which steps were human corrections and what did I do").
        # Reuses Phase 1's exact TickLogger (same JSONL convention as live_infer_deploy.py/
        # infer_app) - self-disabling on write failure, never risks crashing the control loop.
        self.tick_logger = TickLogger(log_file) if log_file else None

        self.intervention: InterventionManager | None = None
        if leader is not None and pedal is not None:
            self.intervention = InterventionManager(leader, pedal, policy)

        # Episode-boundary control (reset pedal press, OR reward pedal held->released) -
        # the SAME shared class used by teleop/infer (`trossen_real.human_intervention.
        # episode_boundary.EpisodeBoundaryMonitor`), reused as-is here rather than
        # duplicating pedal-edge-detection logic a third time. Independent of
        # `self.intervention` above - only needs `pedal`, not `leader`. See
        # `PEDAL_BEHAVIOR.md` for the exact behavior and why `start_episode()`
        # draining stale pedal state matters (fixes a real bug: a reset-pedal press
        # landing during the ~settle_time_s window between episodes would otherwise
        # sit latched and truncate the NEXT episode to 1 step).
        self.episode_monitor: EpisodeBoundaryMonitor | None = None
        if pedal is not None:
            self.episode_monitor = EpisodeBoundaryMonitor(pedal)

        self._last_base_naction: torch.Tensor | None = None
        self._step_count = 0

        # ------------------------------------------------------------------
        # gym-style spaces, batched with a leading dim of 1 (matches the
        # vectorized sim wrapper's shape convention - `num_envs=1` here - so
        # the training script's `env.observation_space[...].shape[1]` /
        # `env.action_space.shape[1]` dimension-probing code works unchanged).
        # ------------------------------------------------------------------
        state_dim = action_dim  # command_space: joint -> state IS the raw 7D joint vector, same dim as action
        # RL-facing images are stored/network-sized at `rl_image_size` (e.g. 84) if given,
        # NOT the raw camera capture resolution - avoids an oversized ViT/CNN AND keeps the
        # online replay buffer's memory footprint consistent with the offline buffer's
        # `offline_data.image_size`. The base policy query (`_query_base_action()`) is
        # UNCHANGED - it always sends the full-resolution frame to policy_server.py.
        h, w = (self.rl_image_size, self.rl_image_size) if self.rl_image_size is not None else camera_resolution
        obs_spaces = {
            "observation.state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, state_dim), dtype=np.float32),
        }
        for key in image_keys:
            obs_spaces[key] = gym.spaces.Box(low=0.0, high=1.0, shape=(1, 3, h, w), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, action_dim), dtype=np.float32)

        self.metadata = {"horizon": max_steps}
        # Self-referential alias so `env.vec_env.metadata[...]` (the sim wrapper's exact
        # attribute path) works completely unchanged for real hardware too - no
        # `if cfg.real_hardware` branch needed in train_residual_td3.py for this.
        self.vec_env = self

    @property
    def fps(self) -> int:
        return self.control.frequency_hz

    # ------------------------------------------------------------------
    # Core gym-like API
    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        # Discard any stale reward latch OR reset-pedal press left over from BEFORE this
        # episode starts (e.g. the operator still had their foot down on the reward pedal
        # at the exact instant the reset pedal was pressed, OR pressed the reset pedal a
        # SECOND time during the settle_time_s window below, while the previous reset was
        # still in flight) - neither may leak into the new episode's first transition.
        # `EpisodeBoundaryMonitor.start_episode()` (shared with teleop/infer, see
        # PEDAL_BEHAVIOR.md) does both drains and logs a warning if either was actually
        # pending. Without draining the reset-pedal side specifically, a second press
        # landing during this settle window would otherwise sit latched, unconsumed, and
        # fire on the very first `step()` of the NEW episode - truncating it to 1 step.
        # This does not, by itself, protect against the pedal being CONTINUOUSLY held
        # straight through the reset (see the post-settle check below for that case).
        if self.episode_monitor is not None:
            self.episode_monitor.start_episode()

        self.follower.reset()
        if self.intervention is not None:
            follower_state = self.follower.get_state()
            self.intervention.sync_to_follower(follower_state["q"], goal_time=self.reset_cfg.goal_time_s)

        # Settle before taking the first observation / resetting the policy's
        # action-chunk queue - mirrors infer_app/infer_loop.py's proven
        # post-reset sequence exactly (reset move -> settle -> policy.reset()).
        # Without this, the very first obs of the episode can be captured while
        # the arm is still physically settling from the staged-position move.
        time.sleep(self.settle_time_s)

        if self.pedal is not None and self.pedal.get_reward():
            logger.warning(
                "Reward pedal is STILL held down at the start of a new episode (after the %.1fs settle "
                "window) - if not released, the first transition(s) of this new episode will be "
                "incorrectly marked reward=1. Release the reward pedal before the episode starts.",
                self.settle_time_s,
            )
        if self.pedal is not None and not self.pedal.is_healthy():
            logger.error(
                "Foot pedal has lost its device connection (dummy mode / disconnected) - reward, reset, "
                "and intervention signals are all STALE/frozen. Reconnect the pedal before continuing."
            )

        self.policy.reset()

        follower_state = self.follower.get_state()
        images = self.cameras.get_all_latest()
        raw_obs = self._build_actor_obs(follower_state, images)
        base_naction = self._query_base_action(follower_state, images)
        obs = self._augment_obs(raw_obs, base_naction)

        self._last_base_naction = base_naction
        self._step_count = 0
        return obs, {}

    def step(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        t_start = time.time()

        # Defensive: ensure the caller's residual (agent.act() output, or the warm-up
        # loop's torch.rand(..., device=device)) is on the SAME device as this env's own
        # tensors, regardless of what device the caller happened to build it on.
        residual_naction = residual_naction.to(self.device)

        # 1. Pedal check + leader mode switch + release-edge policy.reset() - BEFORE
        #    anything else, so a release-edge reset takes effect before predict() below.
        intervened_now = self.intervention.tick() if self.intervention is not None else False

        combined_naction = torch.clamp(self._last_base_naction + residual_naction, -1.0, 1.0)
        follower_state_before = self.follower.get_state()

        if intervened_now:
            # The leader's raw absolute joint readback must NOT be forwarded to the follower
            # unclamped: a human can move the leader arbitrarily fast/far while backdriving it
            # (freedrive has no software rate limit), and nothing downstream clamps the 6 arm
            # joints either - `move_to_joint_positions()` -> `goal_joint()`'s own docstring says
            # the CALLER is responsible for that ("only clips the gripper"), all the way down to
            # the raw SDK `set_all_positions(goal_time=...)` call, which will attempt whatever
            # delta it's given within `goal_time_s` regardless of magnitude. Clamp the arm-joint
            # delta to `action_scaler.limits` - the SAME real-unit bound `unscale()` already
            # confines every AUTONOMOUS combined action to (the real-unit image of the full
            # `[-1,1]` action-space box, not the narrower `agent.actor.action_scale` residual-only
            # band) - so intervention can never command a step the RL system doesn't already
            # consider "in range" elsewhere. This also makes `stored_naction` below (which
            # applies this exact same clamp via `action_scaler.scale()`) byte-identical to what's
            # actually executed: without this, a human delta bigger than `action_scaler.limits`
            # would get silently truncated for STORAGE while the FOLLOWER still executed the full,
            # larger move - corrupting the critic (it would associate the clamped/stored action
            # with a next-state that was actually reached via a bigger, unclamped move).
            leader_joints = np.asarray(self.leader.get_joint_positions(), dtype=np.float32)
            follower_q_before = np.asarray(follower_state_before["q"], dtype=np.float32)
            delta_min = self.action_scaler.limits.min[:6].detach().cpu().numpy()
            delta_max = self.action_scaler.limits.max[:6].detach().cpu().numpy()

            executed_action = np.empty(7, dtype=np.float32)
            if self.action_space_name == "delta_joint":
                raw_delta = leader_joints[:6] - follower_q_before[:6]
                clamped_delta = np.clip(raw_delta, delta_min, delta_max)
                executed_action[:6] = follower_q_before[:6] + clamped_delta
            else:  # "absolute" - action_scaler.limits already bounds the absolute joint target directly
                executed_action[:6] = np.clip(leader_joints[:6], delta_min, delta_max)
            # Mirror control_loop.py's exact teleop safety clip (control_loop.py:279) - the
            # leader's raw gripper reading has no hard guarantee of staying within the
            # FOLLOWER's configured range (leader hardware can report slightly past it,
            # e.g. during a fast human motion) - sending it through unclipped can fault the
            # follower arm ("Joint limit exceeded").
            executed_action[6] = np.clip(leader_joints[6], self.control.gripper_closed, self.control.gripper_open)

            # Rate-limited follower: while the pedal is held, the follower chases the leader's
            # position by at most `action_scaler.limits` per tick rather than snapping to it -
            # if the human moves fast, the follower visibly lags and catches up over the next
            # few ticks instead of lurching.
            stored_naction = self._to_stored_naction(executed_action, follower_state_before)
        else:
            env_action = self.action_scaler.unscale(combined_naction).squeeze(0).detach().cpu().numpy()
            executed_action = reconstruct_absolute_action(
                self.action_space_name, env_action, follower_state_before,
                gripper_bounds=(self.control.gripper_closed, self.control.gripper_open),
            )
            stored_naction = combined_naction
            if self.intervention is not None:
                self.intervention.mirror_to_leader(executed_action, self.control.goal_time_s)

        self.follower.move_to_joint_positions(
            executed_action, blocking=False, min_time_to_move=self.control.goal_time_s
        )

        # Hold the target control frequency (same tick-timing pattern as infer_app's InferenceLoop).
        period_s = 1.0 / max(self.control.frequency_hz, 1)
        elapsed = time.time() - t_start
        time.sleep(max(0.0, period_s - elapsed))

        follower_state_after = self.follower.get_state()
        images = self.cameras.get_all_latest()
        next_raw_obs = self._build_actor_obs(follower_state_after, images)
        next_base_naction = self._query_base_action(follower_state_after, images)
        next_obs = self._augment_obs(next_raw_obs, next_base_naction)
        self._last_base_naction = next_base_naction

        # Reward + episode-end come from the SAME shared `EpisodeBoundaryMonitor` teleop/infer
        # use (see PEDAL_BEHAVIOR.md) - `terminated` now fires on EITHER a reset-pedal press
        # (unchanged from before) OR the reward pedal's held->released edge (new). The two
        # calls read/consume independent pedal state, so their order doesn't matter for
        # correctness - kept in this order (reward, then episode-end) to match the previous
        # code's structure.
        if self.episode_monitor is not None:
            reward_value = self.episode_monitor.get_frame_reward()
            terminated_now, stop_reason = self.episode_monitor.check_episode_end()
        else:
            reward_value = 0.0
            terminated_now = False
            stop_reason = None

        reward = torch.tensor([reward_value], dtype=torch.float32, device=self.device)
        terminated = torch.tensor([terminated_now], dtype=torch.bool, device=self.device)
        self._step_count += 1
        truncated = torch.tensor(
            [bool(self.max_steps is not None and self._step_count >= self.max_steps)],
            dtype=torch.bool, device=self.device,
        )

        # Buffer-facing done override - WIDER than `terminated` above: also True whenever
        # reward hit the sparse ceiling (1.0) this step, even if the robot is NOT physically
        # resetting right now (e.g. still holding the reward pedal while backing away after a
        # successful insertion, before releasing it). Matches the offline dataset's own
        # verified convention exactly (`next.done == next.reward` in every episode of the
        # training dataset - see REWARD_AND_PEDAL_INVESTIGATION_REPORT.md) and fixes a real
        # Bellman-target-inflation risk: without this, `target_q = reward + gamma^n*Q(next)`
        # for a reward=1-but-not-yet-terminal transition has NO upper bound (this run does not
        # set agent.clip_q_target_to_reward_range, so nothing else bounds it either) - each
        # additional held-reward step compounds a bootstrapped estimate ON TOP of an already-
        # maximal reward. Deliberately kept SEPARATE from `terminated` above (which alone
        # drives whether we physically call `self.reset()` below, and what `info["final_obs"]`/
        # `info["final_info"]` get populated with) - `_add_transitions_to_buffer()`
        # (train_residual_td3.py) reads this via `info["buffer_done"]` to override ONLY the
        # transition's STORED `next.done`/`next.terminated`, leaving the publicly-returned
        # `terminated`/`truncated` (and therefore `done.any()`-driven logging/final_info
        # access in the training loop) untouched - widening THOSE instead would make the
        # robot physically reset the instant the pedal is first pressed, not on release, and
        # would crash the training loop's `info["final_info"]` access on any step where this
        # widened condition fired without a real physical reset.
        buffer_done = bool(terminated_now or bool(truncated.item()) or reward_value >= 1.0)

        if self.pedal is not None and not self.pedal.is_healthy():
            logger.error(
                "Foot pedal has lost its device connection mid-episode - reward/reset/intervention signals "
                "are STALE/frozen at their last known values. Stop and reconnect the pedal."
            )

        info = {
            "scaled_action": stored_naction,           # <-- what actually goes into the replay buffer
            "intervened": intervened_now,
            "actor_proposed_action": combined_naction,  # <-- diagnostic only, NEVER trained on directly
            # Consumed by `_add_transitions_to_buffer()` (train_residual_td3.py) to override ONLY the
            # replay buffer's STORED next.done/next.terminated - see the comment above `buffer_done`'s
            # computation for why this is separate from the publicly-returned `terminated`/`truncated`.
            "buffer_done": torch.tensor([buffer_done], dtype=torch.bool, device=self.device),
        }

        if self.tick_logger is not None:
            self.tick_logger.log(
                step=self._step_count,
                intervened=intervened_now,
                reward=reward.item(),
                terminated=terminated.item(),
                truncated=truncated.item(),
                buffer_done=buffer_done,
                stop_reason=stop_reason,  # "reset_pedal" | "reward_release" | None (still running / no pedal)
                executed_action=np.asarray(executed_action, dtype=np.float32),
                stored_naction=stored_naction.detach().cpu().numpy(),
                actor_proposed_action=combined_naction.detach().cpu().numpy(),
                residual_naction=residual_naction.detach().cpu().numpy(),
                follower_state_before=follower_state_before,
                follower_state_after=follower_state_after,
                achieved_hz=1.0 / max(time.time() - t_start, 1e-6),
            )

        # Gymnasium (vectorized) autoreset convention - matches sim's vec_env exactly, so
        # train_residual_td3.py's existing final_obs/final_info-handling code (already written
        # for sim) works UNCHANGED for real hardware too, with zero real_hardware branches:
        #   - `next_obs` returned on a terminal step IS the FRESH post-reset observation, not
        #     the terminal one (a single real station has no auto-reset otherwise, unlike a
        #     sim vec-env - this is what makes that true here too).
        #   - The TRUE terminal observation is stashed in `info["final_obs"]` (a length-1 list,
        #     matching sim's per-env indexing convention) for `_add_transitions_to_buffer()` to
        #     use instead when storing this transition's `next.obs`.
        #   - `info["final_info"]["episode_steps"]`/`["_episode_steps"]` matches the exact
        #     shape/indexing the main training loop already reads for episode-return logging.
        if bool(terminated.item() or truncated.item()):
            info["final_obs"] = [{k: v.squeeze(0) for k, v in next_obs.items()}]
            info["final_info"] = {
                "episode_steps": np.array([self._step_count]),
                "_episode_steps": np.array([0]),
            }
            next_obs, _ = self.reset()

        return next_obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.intervention is not None:
            try:
                self.leader.disconnect(
                    park_waypoints=[self.reset_cfg.staged_joint_positions, [0.0] * 7],
                    goal_time=self.reset_cfg.goal_time_s,
                )
            except Exception:
                pass
            try:
                self.pedal.stop()
            except Exception:
                pass
        try:
            self.follower.disconnect()
        except Exception:
            pass
        if self.tick_logger is not None:
            self.tick_logger.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_actor_obs(self, follower_state: dict, images: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """RL-facing obs (torch tensors, batch dim 1, on `self.device`) - NOT the
        JSON obs sent to the base policy server (see `_query_base_action()`/
        `build_observation()` for that separate representation, which always
        uses the FULL-resolution frame regardless of `self.rl_image_size`)."""
        q = torch.tensor(follower_state["q"], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 7)
        obs: dict[str, torch.Tensor] = {"observation.state": q}
        for key in self.image_keys:
            cam_name = key.removeprefix("observation.images.")
            frame = images.get(cam_name)
            if frame is None:
                raise RuntimeError(f"Policy needs camera '{cam_name}' but no frame is available for it.")
            # (H,W,3) uint8 -> (1,3,H,W) float32 [0,1], matching the LeRobot dataset image
            # convention `_add_transitions_to_buffer()`/`to_uint8()` already assume.
            chw = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device) / 255.0
            if self.rl_image_size is not None:
                # Downsize ONLY the RL-facing copy (stored in the online buffer) - the
                # base-policy query in `_query_base_action()` uses the original full-res
                # `images` dict directly, untouched by this resize.
                chw = F.interpolate(
                    chw.unsqueeze(0), size=(self.rl_image_size, self.rl_image_size),
                    mode="bilinear", align_corners=False, antialias=True,
                ).squeeze(0)
            obs[key] = chw.unsqueeze(0)
        return obs

    def _query_base_action(self, follower_state: dict, images: dict[str, np.ndarray]) -> torch.Tensor:
        json_obs = build_observation(follower_state, images, self.image_keys,
                                     self.policy_image_encoding, self.policy_jpeg_quality)
        base_action_np = self.policy.predict(json_obs)  # (7,) real units
        base_action = torch.from_numpy(np.asarray(base_action_np, dtype=np.float32)).unsqueeze(0).to(self.device)  # (1,7)
        return self.action_scaler.scale(base_action)

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_naction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Mirrors `BasePolicyVecEnvWrapper._augment_obs()` exactly."""
        augmented = dict(raw_obs)
        augmented["observation.base_action"] = base_naction
        augmented["observation.state"] = self.state_standardizer.standardize(augmented["observation.state"])
        return augmented

    def _to_stored_naction(self, executed_action: np.ndarray, follower_state_before: dict) -> torch.Tensor:
        """The human's REAL, physically-executed combined action, converted to
        the SAME per-frame delta/normalized convention every other transition
        uses (matches `convert_to_delta_joint_dataset.py`'s exact convention).
        Only clipped to [-1,1] (via `action_scaler.scale()`), NEVER to
        `action_scale` - see design doc Section 3 for why."""
        raw = np.asarray(executed_action, dtype=np.float32).copy()
        if self.action_space_name == "delta_joint":
            raw[:6] = raw[:6] - np.asarray(follower_state_before["q"][:6], dtype=np.float32)
        raw_t = torch.from_numpy(raw).unsqueeze(0).to(self.device)  # (1, 7)
        return self.action_scaler.scale(raw_t)
