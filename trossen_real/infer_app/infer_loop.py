# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Background inference loop, driven by the inference web UI's Start/Stop
Inference buttons - the inference-only counterpart to
`teleop/control_loop.py`, but much simpler: there's no recording, and no
"state-polling-only vs actively-driving" distinction (cameras already run
their own independent background threads regardless - see `CameraManager`
- so there's nothing useful to poll BEFORE inference actually starts).

Optionally supports human-intervention takeover (leader + foot pedal) via
`trossen_real.inference.intervention.InterventionManager` - see
`trossen_real/human_intervention/INTERVENTION_PLAN.md` for the full design.
When `intervention` is None (default), behavior is EXACTLY as before this
feature existed - the policy's action is always sent straight to the
follower, no leader involved.

Optionally also supports LeRobot dataset recording (`recorder`, a
`trossen_real.teleop.dataset_recorder.DatasetRecorder` shared as-is with the
teleop app) plus pedal-driven episode-boundary control (`episode_monitor`, a
`trossen_real.human_intervention.episode_boundary.EpisodeBoundaryMonitor`) -
see `PEDAL_BEHAVIOR.md`. Both are independent of `intervention`/the leader -
dataset recording and pedal episode control work with no leader connected at
all. When both are `None` (default), behavior is EXACTLY as before this
feature existed.

Sequence when `start()` is called (mirrors `scripts/live_infer_deploy.py`
exactly - see there for the full reasoning on why each step matters,
especially the pre-inference reset):
  1. (unless `skip_reset`) `follower.reset()` - the SAME staged reset used
     at data-collection time. If intervention is enabled, IMMEDIATELY
     followed by syncing the leader to the follower's resulting joint
     positions (`InterventionManager.sync_to_follower()`, same pattern as
     `teleop/app.py::api_reset()`) - without this, the leader is left
     wherever it was previously resting while the follower jumps to a
     fresh pose, so the very first autonomous tick's mirrored action would
     command the leader to jump a large, fast, UNINTENDED distance
     (observed on real hardware: faults the arm with a joint
     velocity/limit error). Then sleep `settle_time_s`.
  2. `policy.reset()` - clears the policy server's internal action-chunk queue.
  3. Loop at `control.frequency_hz`: (if intervention enabled) check pedal
     state and switch leader mode -> get_state + camera frames -> build
     observation -> POST to policy server (ALWAYS, even while intervened -
     its result is simply not applied to the follower in that case) ->
     send either the policy's action (autonomous) or the leader's current
     joint readback (intervened) -> publish a snapshot (step count, last
     action, achieved frequency, intervention state, any error) for
     `/api/health` to report -> sleep to hold the target rate. The SAME
     action value is sent to both follower and leader every autonomous
     tick (never re-derived from a fresh follower read in between) - no
     extra round-trip/delay between the two.
Stops when `stop()` is called, `max_steps` is reached, or an unhandled
exception occurs (surfaced via the snapshot's `error` field, NOT silently
swallowed - an inference session going wrong is exactly what the operator
needs to see immediately).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from trossen_real.cameras.camera_manager import CameraManager
from trossen_real.config import ControlConfig
from trossen_real.human_intervention.episode_boundary import EpisodeBoundaryMonitor
from trossen_real.infer_app.video_recorder import InferVideoRecorder
from trossen_real.inference.intervention import InterventionManager
from trossen_real.inference.policy_client import PolicyClient, TickLogger, build_observation, reconstruct_absolute_action
from trossen_real.teleop.dataset_recorder import DatasetRecorder
from trossen_real.teleop.follower_client import FollowerClient

logger = logging.getLogger(__name__)


@dataclass
class InferSnapshot:
    timestamp: float = field(default_factory=time.time)
    running: bool = False
    step: int = 0
    max_steps: int | None = None
    last_action: list[float] | None = None
    achieved_hz: float | None = None
    error: str | None = None
    settling: bool = False  # True during the post-reset settle sleep, before inference actually starts
    log_file: str | None = None  # path to this run's diagnostic JSONL log, if logging is enabled
    video_files: list[str] | None = None  # paths to recorded video files if video recording is enabled
    intervention_enabled: bool = False  # whether this session has a leader+pedal wired up at all
    intervening: bool = False  # live pedal state (only meaningful when intervention_enabled)
    dataset_recording: bool = False  # whether this run is writing frames into a DatasetRecorder episode
    num_episodes: int = 0  # DatasetRecorder.num_episodes, if dataset recording is enabled
    stop_reason: str | None = None  # why the run ended: "manual", "max_steps", "reset_pedal", "reward_release", or None while still running


class InferenceLoop:
    """Owns a single inference "run" at a time - not a permanently-running
    background thread like `ControlLoop`, since there's no idle-state
    polling need here (see module docstring)."""

    def __init__(
        self,
        follower: FollowerClient,
        cameras: CameraManager,
        policy: PolicyClient,
        control: ControlConfig,
        image_keys: list[str],
        action_space: str = "absolute",
        intervention: InterventionManager | None = None,
        leader_sync_goal_time_s: float = 3.0,
        station_name: str = "station",
        image_encoding: str = "raw",
        jpeg_quality: int = 95,
    ) -> None:
        self.follower = follower
        self.cameras = cameras
        self.policy = policy
        self.control = control
        self.image_keys = image_keys
        self.action_space = action_space
        self.intervention = intervention
        self.leader_sync_goal_time_s = leader_sync_goal_time_s
        self.station_name = station_name
        # How camera frames are packed for the wire - see encode_image(). Only
        # matters when the policy server is remote, where the payload, not the
        # GPU, is what sets the control rate.
        self.image_encoding = image_encoding
        self.jpeg_quality = jpeg_quality
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._lock = threading.Lock()
        self._snapshot = InferSnapshot(intervention_enabled=intervention is not None)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        settle_time_s: float,
        max_steps: int | None,
        skip_reset: bool = False,
        log_file: str | None = None,
        record_video: bool = True,
        video_dir: str = "infer_videos",
        recorder: DatasetRecorder | None = None,
        episode_monitor: EpisodeBoundaryMonitor | None = None,
        eval_hook=None,
    ) -> None:
        if self.is_running:
            raise RuntimeError("Inference is already running - stop it first.")
        self._stop_requested = False
        with self._lock:
            self._snapshot = InferSnapshot(
                running=True, max_steps=max_steps, settling=not skip_reset, log_file=log_file,
                intervention_enabled=self.intervention is not None,
                dataset_recording=recorder is not None,
                num_episodes=recorder.num_episodes if recorder is not None else 0,
            )
        self._thread = threading.Thread(
            target=self._run,
            args=(settle_time_s, max_steps, skip_reset, log_file, record_video, video_dir, recorder,
                  episode_monitor, eval_hook),
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_requested = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        with self._lock:
            self._snapshot.running = False

    def get_snapshot(self) -> InferSnapshot:
        with self._lock:
            return self._snapshot

    def reset_and_sync(self) -> None:
        """Staged follower reset, plus the leader re-sync when intervention is on.

        Extracted so the Reset button (`/api/reset`) runs the EXACT same sequence
        `start()` does - if these ever drifted apart, pressing Reset could leave
        the leader somewhere the first autonomous tick would then jump away from,
        which is the fault this sync exists to prevent (see
        `InterventionManager.sync_to_follower`).

        Blocking; takes roughly `reset.goal_time_s` seconds. Callers must ensure
        no run is in progress.
        """
        logger.info("Moving to the staged reset position...")
        self.follower.reset()
        if self.intervention is not None:
            logger.info("Syncing leader to the follower's post-reset position...")
            follower_state = self.follower.get_state()
            self.intervention.sync_to_follower(follower_state["q"], self.leader_sync_goal_time_s)

    def _run(
        self,
        settle_time_s: float,
        max_steps: int | None,
        skip_reset: bool,
        log_file: str | None,
        record_video: bool = True,
        video_dir: str = "infer_videos",
        recorder: DatasetRecorder | None = None,
        episode_monitor: EpisodeBoundaryMonitor | None = None,
        eval_hook=None,
    ) -> None:
        tick_logger = TickLogger(log_file) if log_file else None
        stop_reason: str | None = None
        # Defined BEFORE the try so the finally's eval_hook.on_end() can always
        # report a step count, even if the pre-loop reset/settle raises.
        step = 0
        video_recorder = None
        if record_video:
            try:
                video_recorder = InferVideoRecorder(
                    base_dir=video_dir,
                    station_name=self.station_name,
                    camera_names=self.cameras.camera_names,
                    fps=self.control.frequency_hz,
                )
            except Exception:
                logger.exception("Could not initialize InferVideoRecorder; continuing inference without video recording.")

        saved_video_files = list(video_recorder.saved_paths.values()) if video_recorder is not None else None

        try:
            if not skip_reset:
                self.reset_and_sync()  # same sequence the Reset button uses
                logger.info("Reset move complete - settling for %.1fs...", settle_time_s)
                time.sleep(settle_time_s)
            elif self.intervention is not None:
                logger.warning(
                    "skip_reset=True with intervention enabled - the leader is NOT being synced to the "
                    "follower's current position. If they've drifted apart, the first autonomous tick's "
                    "mirrored action could command a large, fast leader jump. Proceed only if you know "
                    "leader and follower are already closely aligned."
                )
            with self._lock:
                self._snapshot.settling = False
                self._snapshot.video_files = saved_video_files

            logger.info("Resetting policy's internal action-chunk queue (fresh rollout)...")
            self.policy.reset()

            if recorder is not None:
                recorder.start_recording()
            if episode_monitor is not None:
                episode_monitor.start_episode()
                if not episode_monitor.pedal.is_healthy():
                    logger.error(
                        "Foot pedal has lost its device connection (dummy mode / disconnected) - "
                        "reward/reset/intervention signals recorded for this run will be STALE/frozen. "
                        "Reconnect the pedal before continuing."
                    )

            period_s = 1.0 / max(self.control.frequency_hz, 1)
            while not self._stop_requested and (max_steps is None or step < max_steps):
                t_start = time.time()

                # Check pedal + switch leader mode BEFORE calling the policy this tick, so a
                # release-edge policy.reset() (see InterventionManager) takes effect before
                # the predict_full() call below.
                intervened_now = self.intervention.tick() if self.intervention is not None else False

                follower_state = self.follower.get_state()
                images = self.cameras.get_all_latest()

                if video_recorder is not None:
                    video_recorder.write_frames(images)

                obs = build_observation(follower_state, images, self.image_keys,
                                        self.image_encoding, self.jpeg_quality)

                # Always called, even while intervened - the result is simply not applied to
                # the follower in that case (see module docstring / InterventionManager).
                # `raw_images` is the UNENCODED frames. Every policy client
                # ignores it except a residual wrapper, which needs real pixels
                # and would otherwise have to decode the wire payload back -
                # giving the RL encoder a jpeg round-trip that training never
                # applied.
                result = self.policy.predict_full(obs, raw_images=images)  # {"action": real units, ...}
                predicted = result["action"]
                policy_action = reconstruct_absolute_action(
                    self.action_space, predicted, follower_state,
                    gripper_bounds=(self.control.gripper_closed, self.control.gripper_open),
                )

                if intervened_now:
                    action = self.intervention.get_leader_action()
                else:
                    action = policy_action
                    if self.intervention is not None:
                        self.intervention.mirror_to_leader(action, self.control.goal_time_s)

                self.follower.move_to_joint_positions(
                    action, blocking=False, min_time_to_move=self.control.goal_time_s
                )

                # Checked BEFORE add_frame() below, and kept as its own top-level step
                # (not nested under `if recorder is not None:`) so pedal-driven stop
                # still works when dataset recording is off.
                should_end, reason = (
                    episode_monitor.check_episode_end() if episode_monitor is not None else (False, None)
                )

                if recorder is not None:
                    # Same pre-action `follower_state` this tick already read (BEFORE
                    # `action` was computed/sent, see module docstring) - matches teleop's
                    # `control_loop.py::_tick()` "read obs, then act, then record with the
                    # PRE-action obs" convention exactly.
                    extra_obs = {
                        "joint_pos_raw": np.asarray(follower_state["q"], dtype=np.float32),
                        "ee_pose_raw": np.concatenate(
                            [follower_state["pose"], [follower_state["gripper_pos"]]]
                        ).astype(np.float32),
                        "velocity": np.asarray(follower_state["dq"], dtype=np.float32),
                        "effort": np.asarray(follower_state["efforts"], dtype=np.float32),
                        "acceleration": np.asarray(follower_state["accelerations"], dtype=np.float32),
                    }
                    frame_reward = episode_monitor.get_frame_reward() if episode_monitor is not None else 0.0
                    # next.done is driven PURELY by next.reward here - True for every
                    # frame while the reward pedal marks success (WIDE, not just a
                    # single terminal tick - verified against the real reference
                    # dataset: its done==True frames are EXACTLY its reward==1 frames,
                    # every episode). This does NOT end the episode by itself - holding
                    # the reward pedal keeps recording; only its release (or reset
                    # pedal, or max_steps, or a manual stop) actually ends it (see
                    # `should_end` above / EpisodeBoundaryMonitor). Whatever the reason
                    # recording eventually stops for, DatasetRecorder.stop_recording()
                    # unconditionally forces the LAST frame's next.done=True too, so a
                    # reset-pedal abort (reward never 1) or a plain timeout still gets a
                    # correct terminal flag without needing to be anticipated here.
                    recorder.add_frame(
                        action=np.asarray(action, dtype=np.float32),
                        state=follower_state["q"],
                        images=images,
                        done=bool(frame_reward >= 1.0),
                        extra_obs=extra_obs,
                        extra_features={
                            "observation.intervened": [intervened_now],
                            "observation.policy_action": np.asarray(policy_action, dtype=np.float32),
                            "next.reward": [frame_reward],
                        },
                    )

                if should_end:
                    logger.info("Episode-boundary pedal triggered (%s) - ending this run after this tick.", reason)
                    self._stop_requested = True
                    stop_reason = reason

                elapsed = time.time() - t_start
                time.sleep(max(0.0, period_s - elapsed))

                step += 1
                tick_duration = time.time() - t_start
                achieved_hz = (1.0 / tick_duration) if tick_duration > 0 else None
                with self._lock:
                    self._snapshot = InferSnapshot(
                        running=True,
                        step=step,
                        max_steps=max_steps,
                        last_action=np.asarray(action, dtype=np.float32).round(4).tolist(),
                        achieved_hz=achieved_hz,
                        settling=False,
                        log_file=log_file,
                        video_files=saved_video_files,
                        intervention_enabled=self.intervention is not None,
                        intervening=intervened_now,
                        dataset_recording=recorder is not None,
                        num_episodes=recorder.num_episodes if recorder is not None else 0,
                    )

                # Eval bookkeeping (opt-in). `EvalHook` swallows its own errors -
                # see eval_session.py - so this cannot interrupt the control loop.
                if eval_hook is not None:
                    eval_hook.on_tick(step=step, intervened=intervened_now, achieved_hz=achieved_hz)

                if tick_logger is not None:
                    tick_logger.log(
                        step=step,
                        action_space=self.action_space,
                        follower_state_q=follower_state["q"],
                        follower_state_gripper=follower_state["gripper_pos"],
                        policy_action_normalized=result["action_normalized"],
                        policy_action_real=predicted,
                        reconstructed_action=action,
                        achieved_hz=achieved_hz,
                        intervened=intervened_now,
                    )
            if stop_reason is None:
                # Loop ended without a pedal trigger - either max_steps was reached, or
                # stop() was called externally (the Stop Inference button).
                stop_reason = "max_steps" if (max_steps is not None and step >= max_steps) else "manual"
        except Exception as exc:
            logger.exception("Inference loop failed")
            with self._lock:
                self._snapshot.error = str(exc)
            stop_reason = stop_reason or "error"
        finally:
            with self._lock:
                self._snapshot.running = False
                self._snapshot.stop_reason = stop_reason
            if eval_hook is not None:
                eval_hook.on_end(stop_reason=stop_reason, steps=step)
            if tick_logger is not None:
                tick_logger.close()
            if video_recorder is not None:
                video_recorder.close()
            if recorder is not None and recorder.is_recording:
                try:
                    n_frames = recorder.stop_recording()
                    logger.info("Saved inference episode (%d frames) to %s", n_frames, recorder.session_dir)
                except Exception:
                    logger.exception("Error saving dataset episode; continuing shutdown anyway.")
