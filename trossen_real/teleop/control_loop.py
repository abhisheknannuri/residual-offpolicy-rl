# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Fixed-rate (default 20Hz) teleop control loop - single arm.

Runs continuously from connect to disconnect purely to keep a live snapshot
of the follower's state fresh for `/api/get_state`/`/api/health` (and to
feed recorded frames when recording) - but only ACTIVELY DRIVES the
follower from the leader's deltas while `teleop_active` is True (toggled by
`/api/start_session` "Initiate Teleop" / `/api/stop_session` "End
Session"). Outside that window (e.g. right after connecting, before
Initiate Teleop, or after End Session) the leader is never read and the
follower is never commanded - this is what lets manual controls (the
gripper button, `/api/reset`) actually stick instead of being overwritten
by the leader's continuously-streamed absolute gripper position on the
very next tick.

Each tick (state-polling always; drive+record only if `teleop_active`):
read the follower's current state over HTTP (`follower_single_server.py`)
FIRST -> read the leader (SDK, no HTTP) -> compute the new absolute target
client-side and send it back over HTTP (non-blocking) -> grab latest
camera frames -> optionally record a frame using the state READ AT THE TOP
of this tick (i.e. BEFORE this tick's action was computed/sent) -> publish
a snapshot for the Flask routes/UI to poll. This read-obs-then-act-then-
record-with-the-PRE-action-obs ordering exactly matches the official
`lerobot` `record_loop()` (`lerobot.scripts.lerobot_record`) - `obs =
robot.get_observation()` happens BEFORE `act = teleop.get_action()` and
`robot.send_action(...)`, and the frame written to the dataset uses that
same pre-action `obs`. The NEXT tick's observation is only read after a
full `precise_sleep(control_interval - dt_s)` - i.e. the follower gets
nearly a WHOLE tick period to settle toward last tick's goal before its
new state is captured, not just the leftover processing time.

Branches on `control.command_space`:
  - "ee" (default): mirrors hil-serl's own gym-env pattern (e.g.
    `TrossenWindowxAIEnv.step()`) - the client reads the follower's current
    cartesian pose, adds the leader's delta (or takes the leader's absolute
    pose directly, per `control_mode`), and sends the absolute EE target
    back to the follower's HTTP server; the server itself does not do any
    delta math.
  - "joint": mirrors Trossen's own official `lerobot_trossen` reference
    implementation instead - raw absolute joint-position mirroring (leader's
    joint positions -> follower's joint goal directly, safety-clamped by
    `joint_max_relative_target`), no delta math, no EE/quaternion conversion
    at all. Requires `control_mode: absolute` (enforced at config-load time).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from trossen_real.cameras.camera_manager import CameraManager
from trossen_real.config import ControlConfig, EEBounds
from trossen_real.human_intervention.episode_boundary import EpisodeBoundaryMonitor
from trossen_real.teleop.dataset_recorder import DatasetRecorder
from trossen_real.teleop.follower_client import FollowerClient
from trossen_real.leader.trossen_leader_single import TrossenSingleLeader

logger = logging.getLogger(__name__)


@dataclass
class TickSnapshot:
    timestamp: float
    state: np.ndarray | None = None  # 7D: pose(6)+gripper(1) ("ee") or joint_0..5+gripper ("joint")
    gripper_open: bool = False
    tick_count: int = 0


class ControlLoop:
    """Background thread driving the teleop tick at `frequency_hz`."""

    # Stop the loop (rather than retrying forever) after this many
    # consecutive tick failures (e.g. follower server unreachable) - the
    # error is then surfaced via `get_error()` for `/api/health` to report.
    MAX_CONSECUTIVE_FAILURES = 5

    # Rolling window (in ticks) used to compute the ACHIEVED control
    # frequency (`get_actual_frequency_hz()`) - distinct from the
    # CONFIGURED `frequency_hz`/`period_s` target. HTTP round-trips to the
    # follower server, camera reads, etc. can all make the real achieved
    # rate fall short of the target; surfacing the measured rate (via
    # `/api/health`, shown in the UI) makes that visible instead of
    # silently assuming the target rate was actually hit.
    FREQUENCY_WINDOW_TICKS = 40

    def __init__(
        self,
        leader: TrossenSingleLeader,
        follower: FollowerClient,
        cameras: CameraManager,
        recorder: DatasetRecorder,
        control: ControlConfig,
        ee_bounds: EEBounds,
        episode_monitor: EpisodeBoundaryMonitor | None = None,
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.cameras = cameras
        self.recorder = recorder
        self.control = control
        self.ee_bounds = ee_bounds
        # Optional pedal-driven episode-boundary control (reset pedal / reward-pedal
        # release ends the current episode, mirrors "Stop Recording") - independent of
        # the leader/intervention concept (teleop always drives from the leader
        # directly; this pedal doesn't need a leader at all). See PEDAL_BEHAVIOR.md.
        self.episode_monitor = episode_monitor
        self._last_auto_stop_reason: str | None = None
        self.period_s = 1.0 / max(control.frequency_hz, 1)
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        # Dedicated lock guarding `teleop_active` TOGETHER WITH the "read it
        # -> decide -> send the move command" sequence in `_tick()` (see
        # there) - NOT the same lock as `_lock` above (which just guards
        # unrelated snapshot/error/timing state) so toggling teleop_active
        # never has to wait on that unrelated bookkeeping, only on an
        # in-flight tick's own decision+send window.
        self._teleop_gate = threading.Lock()
        self._latest: TickSnapshot | None = None
        self._tick_count = 0
        self._consecutive_failures = 0
        self._error: str | None = None
        self.teleop_active = False
        self._tick_durations: deque[float] = deque(maxlen=self.FREQUENCY_WINDOW_TICKS)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._consecutive_failures = 0
        self._error = None
        self.teleop_active = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_teleop_active(self, active: bool) -> None:
        """Enable/disable the leader-driven follower commands (and recording).

        State polling (the snapshot used by `/api/get_state`/`/api/health`)
        keeps running either way - only the leader-read + follower-move +
        frame-recording step is gated by this.

        Uses `_teleop_gate` (see `_tick()`), NOT just a plain flag write -
        this call BLOCKS until any tick CURRENTLY in the middle of its own
        read-teleop_active-then-send-a-move critical section has fully
        finished (including the HTTP round-trip to the follower server),
        before actually flipping the flag and returning. This closes a real
        race that used to let a stale in-flight `/pose` command land AFTER
        a reset had already moved the follower to its staged position - the
        control loop tick would read `teleop_active=True`, release the
        lock, do a slow `get_state()`/leader-read/target-compute, and only
        THEN send `/pose`; if `set_teleop_active(False)` (e.g. from
        `_do_reset()`) happened to run in that window, the stale `/pose`
        would still fire off - sometimes racing with or landing right after
        the reset's own moves, making the follower briefly jump back toward
        wherever the leader was. Observed rarely on real hardware (small
        window, usually loses the race) - reported as "the follower
        sometimes goes back to its pre-reset position / seems to follow
        the leader for an instant right after Start Recording resets it."
        """
        with self._teleop_gate:
            self.teleop_active = active

    def get_error(self) -> str | None:
        """Non-None if the loop auto-stopped after too many consecutive tick
        failures (as opposed to being stopped deliberately via `stop()`)."""
        with self._lock:
            return self._error

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_actual_frequency_hz(self) -> float | None:
        """Measured control frequency over the last `FREQUENCY_WINDOW_TICKS`
        ticks (full tick duration, including the pacing sleep) - None if no
        ticks have completed yet. Compare against `control.frequency_hz`
        (the configured target) to see if the loop is actually keeping up.
        """
        with self._lock:
            durations = list(self._tick_durations)
        if not durations:
            return None
        mean_duration = sum(durations) / len(durations)
        return 1.0 / mean_duration if mean_duration > 0 else None

    def _loop(self) -> None:
        while self._running:
            t_start = time.time()
            try:
                self._tick()
                self._consecutive_failures = 0
            except Exception as exc:
                self._consecutive_failures += 1
                logger.exception(
                    "Control loop tick failed (%d/%d consecutive).",
                    self._consecutive_failures,
                    self.MAX_CONSECUTIVE_FAILURES,
                )
                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    with self._lock:
                        self._error = (
                            f"Control loop stopped after {self._consecutive_failures} "
                            f"consecutive tick failures: {exc}"
                        )
                    logger.error(
                        "Control loop stopping after %d consecutive tick failures - "
                        "check /api/health and the follower_single_server.py process.",
                        self._consecutive_failures,
                    )
                    self._running = False
                    break
            elapsed = time.time() - t_start
            time.sleep(max(0.0, self.period_s - elapsed))
            with self._lock:
                self._tick_durations.append(time.time() - t_start)

    def _tick(self) -> None:
        recorded_action = None
        goal_time = self.control.goal_time_s

        # Single `/getstate` HTTP call regardless of `command_space` - the
        # follower server ALWAYS returns pose+q+gripper_pos+dq+efforts+
        # accelerations together (it doesn't branch on `command_space`), so
        # both representations plus velocity/effort/acceleration are always
        # available here for free, at no extra network cost - used below to
        # build `extra_obs` for `recorder.add_frame()` (see
        # `dataset_recorder.py::AUX_OBS_KEYS`/`config.py::
        # build_lerobot_features()`), regardless of which one is the
        # PRIMARY control-space this tick. Reading this doesn't itself race
        # with a reset (it's just a read), so it happens outside
        # `_teleop_gate` - only the DECISION to move (and the move itself)
        # needs to be atomic with `teleop_active`, see below.
        follower_state = self.follower.get_state()
        current_joints = np.asarray(follower_state["q"], dtype=np.float32)
        current_pose = np.asarray(follower_state["pose"], dtype=np.float32)
        current_gripper = float(follower_state["gripper_pos"])
        ee_pose_raw = np.concatenate([current_pose, [current_gripper]]).astype(np.float32)
        velocity = np.asarray(follower_state["dq"], dtype=np.float32)
        effort = np.asarray(follower_state["efforts"], dtype=np.float32)
        acceleration = np.asarray(follower_state["accelerations"], dtype=np.float32)

        # Everything from here down to the move being SENT is one atomic
        # critical section w.r.t. `set_teleop_active()` (see there for the
        # race this closes) - reading `teleop_active`, reading the leader,
        # computing the target, and sending `/pose`/`/move_to_joint_positions`
        # must all happen (or not happen at all) before a concurrent
        # `set_teleop_active(False)` call is allowed to return.
        with self._teleop_gate:
            teleop_active = self.teleop_active

            if self.control.command_space == "joint":
                if teleop_active:
                    # Absolute joint mirroring - matches Trossen's own
                    # `lerobot_trossen` reference exactly
                    # (`widowxai_leader.py::get_action()` reads the leader's raw
                    # absolute joint positions, `widowxai_follower.py::
                    # send_action()` forwards them directly as the follower's
                    # goal). No delta math (`command_space: joint` requires
                    # `control_mode: absolute`, enforced in `load_station_config()`).
                    leader_joints = np.asarray(self.leader.get_joint_positions(), dtype=np.float32)
                    # Safety clamp on the 6 arm joints only, mirrors Trossen's
                    # own `max_relative_target`/`ensure_safe_goal_position()` -
                    # a runaway-jump guard, not a routine per-tick limiter
                    # (normal per-tick deltas during continuous teleop are tiny
                    # by construction once leader/follower joints are synced).
                    delta = np.clip(
                        leader_joints[:6] - current_joints[:6],
                        -self.control.joint_max_relative_target,
                        self.control.joint_max_relative_target,
                    )
                    target_joints = current_joints[:6] + delta
                    gripper_target = float(
                        np.clip(leader_joints[6], self.control.gripper_closed, self.control.gripper_open)
                    )
                    target = np.concatenate([target_joints, [gripper_target]]).astype(np.float32)
                    self.follower.move_to_joint_positions(target, blocking=False, min_time_to_move=goal_time)
                    # Recorded action = the EXACT (post-safety-clamp) absolute
                    # joint goal actually sent - matches Trossen's own
                    # `send_action()` return-value convention ("action can
                    # eventually be clipped... so action actually sent is saved
                    # in the dataset"), NOT the leader's raw un-clamped intent.
                    recorded_action = target

                state_7d = current_joints
            else:
                if teleop_active:
                    if self.control.control_mode == "absolute":
                        # Follower target = the leader's raw absolute pose, forwarded
                        # directly (no delta math) - see ControlConfig.control_mode
                        # for why (avoids delta mode's accumulating-lag behavior on
                        # fast leader motion, at the cost of requiring the leader and
                        # follower to share a sensible common frame).
                        leader_pose = np.asarray(self.leader.get_cartesian_positions(), dtype=np.float32)
                        leader_gripper = float(self.leader.get_gripper_position())
                        # Clip xyz the same way the follower server will, so the
                        # RECORDED delta matches what's actually achievable/achieved
                        # instead of an unclippable aspirational target.
                        target_pose = leader_pose.copy()
                        target_pose[:3] = np.clip(leader_pose[:3], self.ee_bounds.xyz_min, self.ee_bounds.xyz_max)
                        gripper_target = float(
                            np.clip(leader_gripper, self.control.gripper_closed, self.control.gripper_open)
                        )
                        # The dataset `action` is STILL a delta either way, matching
                        # this repo's action-space convention everywhere else - here
                        # it's "how far THIS command asks the follower to move from
                        # its current pose", i.e. computed from THIS tick's target
                        # and THIS tick's observed state (NOT the leader's own
                        # tick-to-tick delta, which is a different physical quantity
                        # once the follower isn't just mirroring leader deltas).
                        recorded_action = np.concatenate(
                            [target_pose - current_pose, [leader_gripper]]
                        ).astype(np.float32)
                        # Safety: unlike delta mode, this target has no reference to
                        # the follower's own position, so it can be an arbitrarily
                        # large jump (e.g. the first tick after enabling teleop, if
                        # the leader happens to be far from wherever the follower
                        # currently is) - never command that in just `goal_time_s`
                        # (tuned for small continuous deltas). Scale the time given
                        # by distance instead, capped at a safe max speed.
                        jump_dist = float(np.linalg.norm(target_pose[:3] - current_pose[:3]))
                        if self.control.absolute_mode_max_speed_m_s > 0:
                            goal_time = max(goal_time, jump_dist / self.control.absolute_mode_max_speed_m_s)
                    else:
                        leader_action = self.leader.get_action()  # 7D: delta pose(6) + abs gripper(1)
                        target_pose = current_pose + leader_action[:6]
                        gripper_target = float(
                            np.clip(leader_action[6], self.control.gripper_closed, self.control.gripper_open)
                        )
                        recorded_action = leader_action

                    target = np.concatenate([target_pose, [gripper_target]]).astype(np.float32)
                    self.follower.move_to_ee_pose(target, blocking=False, min_time_to_move=goal_time)

                state_7d = ee_pose_raw

        images = self.cameras.get_all_latest()
        if self.recorder.is_recording and recorded_action is not None:
            extra_obs = {
                "joint_pos_raw": current_joints,
                "ee_pose_raw": ee_pose_raw,
                "velocity": velocity,
                "effort": effort,
                "acceleration": acceleration,
            }
            frame_reward = self.episode_monitor.get_frame_reward() if self.episode_monitor is not None else 0.0
            extra_features = {"next.reward": [frame_reward]} if self.episode_monitor is not None else None

            should_end, reason = self.episode_monitor.check_episode_end() if self.episode_monitor is not None else (False, None)

            # next.done is driven PURELY by next.reward here - True for every frame
            # while the reward pedal marks success (WIDE, not just a single terminal
            # tick - verified against the real reference dataset: its done==True
            # frames are EXACTLY its reward==1 frames, every episode). This does NOT
            # end the episode by itself; only its release (or reset pedal, or a
            # manual Stop Recording click) does (see `should_end` below). Whatever the
            # reason recording eventually stops for, DatasetRecorder.stop_recording()
            # unconditionally forces the LAST frame's next.done=True too, so a
            # reset-pedal abort (reward never 1) still gets a correct terminal flag
            # without needing to be anticipated here.
            self.recorder.add_frame(
                recorded_action, state_7d, images, done=bool(frame_reward >= 1.0),
                extra_obs=extra_obs, extra_features=extra_features,
            )

            if should_end:
                logger.info("Episode-boundary pedal triggered (%s) - stopping this recording (mirrors Stop Recording).", reason)
                self._last_auto_stop_reason = reason
                try:
                    self.recorder.stop_recording()
                except RuntimeError:
                    logger.exception("Error auto-stopping recording from pedal trigger.")

        self._tick_count += 1
        midpoint = (self.control.gripper_open + self.control.gripper_closed) / 2.0
        snapshot = TickSnapshot(
            timestamp=time.time(),
            state=state_7d,
            gripper_open=current_gripper >= midpoint,
            tick_count=self._tick_count,
        )
        with self._lock:
            self._latest = snapshot

    def get_latest(self) -> TickSnapshot | None:
        with self._lock:
            return self._latest

    def get_last_auto_stop_reason(self) -> str | None:
        """Non-None if the MOST RECENT episode ended via a pedal trigger rather than a
        manual Stop Recording click - "reset_pedal" or "reward_release". Cleared by
        `start_episode()`'s caller (see `app.py::api_start_recording()`) at the start of
        the NEXT episode, not by this getter (so `/api/health` can poll it repeatedly
        without losing it)."""
        return self._last_auto_stop_reason

    def clear_last_auto_stop_reason(self) -> None:
        self._last_auto_stop_reason = None
