# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Teleop data-collection app: web UI + camera MJPEG streams.

One process per station. Serves the web UI (static/index.html + friends),
reads the leader directly (SDK, no HTTP - see
`leader/trossen_leader_single.py`), and talks to the follower over HTTP via
`follower_client.py` - **requires `follower_single_server.py` to already be
running** for this station (launch it first, in its own terminal). This app
never owns the follower connection itself; it's just one of two possible
HTTP clients of `follower_single_server.py` (the other being an RL
gym-wrapper, built separately).

Run from the repo root (inside the resfit venv), AFTER
`follower_single_server.py` is already running for the same station::

    python -m trossen_real.teleop.app --port 5050
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import sys
import threading
import time

import cv2
from flask import Flask, jsonify, request, send_from_directory

from trossen_real.cameras.camera_manager import CameraManager
from trossen_real.config import list_available_configs, load_station_config
from trossen_real.human_intervention.episode_boundary import EpisodeBoundaryMonitor
from trossen_real.human_intervention.pedal_listener import PedalListener
from trossen_real.teleop.control_loop import ControlLoop
from trossen_real.teleop.dataset_recorder import DatasetRecorder
from trossen_real.teleop.follower_client import FollowerClient
from trossen_real.leader.trossen_leader_single import TrossenSingleLeader
from trossen_real.log_setup import setup_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
setup_logging("teleop")  # see trossen_real/log_setup.py - also writes to logs/teleop/<date>/<time>/app.log
logger = logging.getLogger(__name__)

STATIC_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
# This UI's static/*.js/*.html get edited frequently during development, and
# a stale cached copy (browser silently reusing an old app.js after a fix)
# is exactly the kind of confusing "the fix isn't working" symptom that
# already happened once - disable static file caching entirely rather than
# rely on operators remembering to hard-refresh every time.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Extra LeRobot dataset feature added ONLY when enable_pedal_episode_control=true (see
# api_connect()) - unchecked, teleop's recorded dataset schema is byte-identical to
# before this feature existed. See PEDAL_BEHAVIOR.md.
PEDAL_EXTRA_LEROBOT_FEATURES = {"next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]}}


class AppState:
    """Holds everything for the currently-connected station (or nothing, if disconnected)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.config = None
        self.leader: TrossenSingleLeader | None = None
        self.follower: FollowerClient | None = None
        self.cameras: CameraManager | None = None
        self.recorder: DatasetRecorder | None = None
        self.control_loop: ControlLoop | None = None
        # Opt-in pedal-driven episode-boundary control (reset pedal / reward-pedal
        # release ends the current recording, mirrors Stop Recording) - independent of
        # the leader (teleop already drives from the leader for normal control; this
        # pedal doesn't add a second arm or an "intervention" concept). See
        # PEDAL_BEHAVIOR.md.
        self.pedal: PedalListener | None = None
        self.episode_monitor: EpisodeBoundaryMonitor | None = None
        self.pedal_episode_control_enabled: bool = False

    @property
    def connected(self) -> bool:
        return self.leader is not None


state = AppState()


def _require_connected():
    if not state.connected:
        return jsonify(error="Not connected. POST /api/connect first."), 409
    return None


# ----------------------------------------------------------------------
# UI routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


# ----------------------------------------------------------------------
# Connection lifecycle
# ----------------------------------------------------------------------
@app.route("/api/stations", methods=["GET"])
def api_stations():
    return jsonify(stations=list_available_configs())


@app.route("/api/connect", methods=["POST"])
def api_connect():
    body = request.get_json(force=True, silent=True) or {}
    config_name = body.get("config_name")
    enable_pedal_episode_control = bool(body.get("enable_pedal_episode_control", False))
    if not config_name:
        return jsonify(error="config_name is required"), 400

    with state.lock:
        if state.connected:
            return jsonify(error="Already connected. POST /api/disconnect first."), 409
        try:
            config = load_station_config(config_name)
            if config.arm_mode != "single":
                return jsonify(error=f"Only arm_mode: single is supported for now (got '{config.arm_mode}')."), 400

            leader = TrossenSingleLeader(config.leader_ips["single"])
            leader.connect()

            follower = FollowerClient(config.follower_server_url)
            try:
                follower.connect()
            except Exception as exc:
                leader.disconnect()
                raise RuntimeError(
                    f"Could not reach follower_single_server.py at {config.follower_server_url} - "
                    f"is it running? ({exc})"
                ) from exc

            cameras = CameraManager(config)
            cameras.start()

            # No leader needed for this - reset/reward pedal episode-boundary control is
            # completely independent of the (always-connected, for normal teleop) leader
            # above. See PEDAL_BEHAVIOR.md.
            pedal = None
            episode_monitor = None
            if enable_pedal_episode_control:
                pedal = PedalListener()
                episode_monitor = EpisodeBoundaryMonitor(pedal)

            recorder = DatasetRecorder(
                config,
                extra_features=PEDAL_EXTRA_LEROBOT_FEATURES if enable_pedal_episode_control else None,
            )
            control_loop = ControlLoop(
                leader, follower, cameras, recorder, config.control, config.ee_bounds["single"],
                episode_monitor=episode_monitor,
            )
            control_loop.start()
        except Exception as exc:
            logger.exception("Failed to connect station '%s'", config_name)
            return jsonify(error=str(exc)), 500

        state.config = config
        state.leader = leader
        state.follower = follower
        state.cameras = cameras
        state.recorder = recorder
        state.control_loop = control_loop
        state.pedal = pedal
        state.episode_monitor = episode_monitor
        state.pedal_episode_control_enabled = enable_pedal_episode_control

    return jsonify(
        station_name=config.station_name,
        cameras=cameras.camera_names,
        pedal_episode_control_enabled=enable_pedal_episode_control,
    )


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    with state.lock:
        if not state.connected:
            return jsonify(status="already disconnected")
        if state.recorder is not None and (state.recorder.is_recording or state.recorder.is_session_active):
            return jsonify(error="Stop recording/session before disconnecting."), 409

        state.control_loop.stop()
        state.cameras.stop()
        # Park the leader through the same staged->sleep joint waypoints the
        # follower uses before releasing it - leaving freedrive to do this
        # will fight the operator's hand if they're still holding the
        # leader, but avoids leaving it wherever freedrive last let it
        # rest/droop once the connection drops.
        reset_cfg = state.config.reset["single"]
        state.leader.disconnect(
            park_waypoints=[reset_cfg.staged_joint_positions, [0.0] * 7],
            goal_time=reset_cfg.goal_time_s,
        )
        try:
            state.follower.disconnect()
        except Exception:
            logger.exception("Error telling follower_single_server.py to disconnect; continuing anyway.")
        if state.pedal is not None:
            state.pedal.stop()

        state.config = None
        state.leader = None
        state.follower = None
        state.cameras = None
        state.recorder = None
        state.control_loop = None
        state.pedal = None
        state.episode_monitor = None
        state.pedal_episode_control_enabled = False

    return jsonify(status="disconnected")


@app.route("/api/health", methods=["GET"])
def api_health():
    if not state.connected:
        return jsonify(connected=False)

    snapshot = state.control_loop.get_latest()
    return jsonify(
        connected=True,
        station_name=state.config.station_name,
        leader=state.leader.is_connected(),
        follower=state.follower.check(),
        cameras=state.cameras.health(),
        # Richer per-camera diagnostics than the plain bool above (staleness
        # age, consecutive read failures, reconnect attempts) - for anything
        # that wants to actually MONITOR/ALERT on camera health (e.g. an RL
        # training loop polling this endpoint) rather than just show a
        # status dot. See `CameraManager.get_camera_status()`/`_reconnect()`
        # - a real camera auto-recovers from the common "Frame didn't arrive"
        # RealSense hiccup on its own, and NEVER silently fabricates mock
        # data for a real-serial slot even if reconnecting keeps failing.
        camera_details=state.cameras.get_camera_status(),
        session_active=state.recorder.is_session_active,
        recording=state.recorder.is_recording,
        num_episodes=state.recorder.num_episodes,
        tick_count=snapshot.tick_count if snapshot else 0,
        last_tick_age_s=(time.time() - snapshot.timestamp) if snapshot else None,
        control_loop_alive=state.control_loop.is_alive(),
        control_loop_error=state.control_loop.get_error(),
        teleop_active=state.control_loop.teleop_active,
        actual_frequency_hz=state.control_loop.get_actual_frequency_hz(),
        target_frequency_hz=state.config.control.frequency_hz,
        pedal_episode_control_enabled=state.pedal_episode_control_enabled,
        last_auto_stop_reason=state.control_loop.get_last_auto_stop_reason(),
    )


@app.route("/api/get_state", methods=["GET"])
def api_get_state():
    err = _require_connected()
    if err:
        return err
    snapshot = state.control_loop.get_latest()
    if snapshot is None:
        return jsonify(error="No control loop tick yet."), 503
    return jsonify(
        timestamp=snapshot.timestamp,
        state=snapshot.state.tolist(),
        gripper_open=snapshot.gripper_open,
    )


def _do_reset() -> None:
    """Shared reset sequence used by both `/api/reset` (manual button) and
    `/api/start_recording` (automatic per-episode reset, mirroring what a
    gym env's `reset()` does at the start of each episode).

    Temporarily disables the control loop's active leader->follower driving
    for the duration - otherwise its background thread keeps sending /pose
    commands (computed from the follower's PRE-reset state + leader deltas)
    concurrently with the reset's own staged/randomize moves. `set_teleop_active(False)`
    genuinely BLOCKS until any tick CURRENTLY in the middle of sending such
    a command has fully finished (see `ControlLoop._teleop_gate`) - a
    previous version of this only flipped a plain flag, which left a real
    (rare, timing-dependent) race: a tick could read `teleop_active=True`
    just before this call, then take a little while (network round-trip)
    to actually send its `/pose`, and that stale command could still land
    on the follower server AFTER (or racing with) THIS reset's own moves -
    observed on real hardware as the follower briefly jumping back toward
    its pre-reset pose / appearing to chase the leader for an instant right
    after a reset. Restores whatever `teleop_active` was before the call
    once done - a Reset clicked before "Initiate Teleop" leaves teleop
    inactive afterward; one triggered by `start_recording` (which only runs
    while a session/teleop is already active) leaves it active afterward.
    """
    was_active = state.control_loop.teleop_active
    state.control_loop.set_teleop_active(False)
    try:
        state.follower.reset()
        # Align the leader to wherever the follower actually ended up, so
        # the human operator starts each episode with leader/follower in
        # sync (rather than the leader left wherever freedrive drifted to).
        # Uses the follower's JOINT positions (not cartesian pose) - both
        # arms are the same model/mounting, so matching joint angles gives a
        # matching EE pose, and is deterministic regardless of how far the
        # leader's current configuration is from the target (a cartesian/IK
        # move over a big jump was observed to sometimes land near a
        # degenerate "home"-looking configuration instead).
        follower_state = state.follower.get_state()
        state.leader.sync_to_joints(
            follower_state["q"],
            goal_time=state.config.reset["single"].goal_time_s,
        )
        state.leader.rebaseline()
    finally:
        state.control_loop.set_teleop_active(was_active)


# ----------------------------------------------------------------------
# Reset / gripper (proxy to follower_single_server.py, for the UI's buttons)
# ----------------------------------------------------------------------
@app.route("/api/reset", methods=["POST"])
def api_reset():
    err = _require_connected()
    if err:
        return err
    try:
        _do_reset()
    except Exception as exc:
        logger.exception("Reset failed")
        return jsonify(error=str(exc)), 500
    return jsonify(status="ok")


@app.route("/api/gripper", methods=["POST"])
def api_gripper():
    err = _require_connected()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    control = state.config.control
    if "open" in body:
        target_open = bool(body["open"])
    else:
        # Toggle: flip whatever the gripper's current state is.
        midpoint = (control.gripper_open + control.gripper_closed) / 2.0
        target_open = state.follower.get_gripper() < midpoint
    target = control.gripper_open if target_open else control.gripper_closed
    # Discrete button click doing a (near-)full open/close stroke - use the
    # slower dedicated goal_time, NOT the fast per-tick teleop goal_time_s
    # (that's tuned for small continuous deltas and was observed to
    # overshoot the gripper's open limit on a full-stroke move).
    state.follower.move_gripper(target, blocking=True, min_time_to_move=control.gripper_button_goal_time_s)
    return jsonify(status="ok", open=target_open)


# ----------------------------------------------------------------------
# Dataset session / recording
# ----------------------------------------------------------------------
@app.route("/api/start_session", methods=["POST"])
def api_start_session():
    err = _require_connected()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    task_name = body.get("task_name", "")
    try:
        session_dir = state.recorder.start_session(task_name)
    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400
    # "Initiate Teleop": from here on, the control loop actively reads the
    # leader's deltas and drives the follower every tick (previously this
    # started unconditionally at /api/connect, which meant the leader's
    # continuously-streamed absolute gripper position silently overrode the
    # manual gripper button within one tick, no matter when it was clicked).
    # Rebaseline first so teleop doesn't jump from whatever delta
    # accumulated while the leader was being read but not yet acted upon.
    state.leader.rebaseline()
    state.control_loop.set_teleop_active(True)
    return jsonify(status="ok", session_dir=session_dir)


@app.route("/api/stop_session", methods=["POST"])
def api_stop_session():
    err = _require_connected()
    if err:
        return err
    # "End Session": stop actively driving the follower from the leader -
    # manual controls (gripper button, Reset) work normally again afterward.
    state.control_loop.set_teleop_active(False)
    try:
        state.recorder.stop_session()
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(status="ok")


@app.route("/api/start_recording", methods=["POST"])
def api_start_recording():
    err = _require_connected()
    if err:
        return err
    try:
        # Per-episode reset, mirroring a gym env's reset() at the start of
        # each episode: same follower staged/randomize move + leader joint
        # sync as the manual Reset button, done automatically here so every
        # recorded episode starts from a known, leader/follower-synced pose.
        _do_reset()
        state.recorder.start_recording()
        if state.episode_monitor is not None:
            state.episode_monitor.start_episode()
            state.control_loop.clear_last_auto_stop_reason()
            if not state.episode_monitor.pedal.is_healthy():
                logger.error(
                    "Foot pedal has lost its device connection (dummy mode / disconnected) - "
                    "reward/reset signals recorded for this episode will be STALE/frozen. "
                    "Reconnect the pedal before continuing."
                )
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        logger.exception("Pre-episode reset failed")
        return jsonify(error=str(exc)), 500
    return jsonify(status="ok")


@app.route("/api/stop_recording", methods=["POST"])
def api_stop_recording():
    err = _require_connected()
    if err:
        return err
    try:
        n_frames = state.recorder.stop_recording()
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(status="ok", frames=n_frames, num_episodes=state.recorder.num_episodes)


# ----------------------------------------------------------------------
# Camera MJPEG streaming
# ----------------------------------------------------------------------
def _mjpeg_generator(cam_name: str):
    boundary = b"--frame"
    while True:
        if not state.connected:
            time.sleep(0.1)
            continue
        frame = state.cameras.get_latest_frame(cam_name)
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                yield (
                    boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(buf)).encode() + b"\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
        time.sleep(1.0 / 15.0)  # ~15fps browser preview, independent of camera capture rate


@app.route("/video_feed/<cam_name>")
def video_feed(cam_name: str):
    if not state.connected or cam_name not in state.cameras.camera_names:
        return jsonify(error=f"Unknown or disconnected camera '{cam_name}'"), 404
    return app.response_class(
        _mjpeg_generator(cam_name), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def _cleanup(lock_timeout: float = 3.0) -> bool:
    """Best-effort graceful shutdown. Returns True if it fully ran, False if
    it had to bail out early.

    `state.lock` is held for the ENTIRE duration of `/api/connect` (which
    can itself block for several seconds inside the SDK's own connect
    timeout if an IP is wrong/unreachable) - waiting on it unboundedly here
    would mean a stuck connect attempt also blocks shutdown, making
    Ctrl+C/`kill` appear to do nothing until that attempt finishes or times
    out on its own. Bounded instead: if the lock can't be acquired quickly,
    there's nothing safe to clean up anyway (nothing this function touches
    was assigned to `state` yet if a connect is still in progress), so skip
    straight to letting the caller force-exit.
    """
    acquired = state.lock.acquire(timeout=lock_timeout)
    if not acquired:
        logger.warning(
            "Could not acquire state lock within %.1fs during shutdown (a connect attempt "
            "may still be in progress) - skipping graceful disconnect.", lock_timeout
        )
        return False
    try:
        if state.control_loop is not None:
            state.control_loop.stop()
        if state.cameras is not None:
            state.cameras.stop()
        if state.leader is not None:
            reset_cfg = state.config.reset["single"] if state.config is not None else None
            state.leader.disconnect(
                park_waypoints=[reset_cfg.staged_joint_positions, [0.0] * 7] if reset_cfg else None,
                goal_time=reset_cfg.goal_time_s if reset_cfg else 3.0,
            )
        if state.follower is not None:
            try:
                state.follower.disconnect()
            except Exception:
                logger.exception("Error telling follower_single_server.py to disconnect during shutdown; ignoring.")
        if state.pedal is not None:
            state.pedal.stop()
    finally:
        state.lock.release()
    return True


atexit.register(_cleanup)


def _signal_handler(signum, _frame):
    # Covers `kill <pid>` (SIGTERM, the default), terminal-panel/session kill
    # (often SIGHUP), and Ctrl+C (SIGINT) - none of these reliably run
    # `atexit` callbacks on their own (only a "clean" interpreter exit does),
    # so we do the same graceful shutdown explicitly here and then exit.
    # `_cleanup()` is idempotent, so this is safe even if atexit also fires.
    logger.info("Received signal %s, disconnecting and shutting down...", signum)
    if _cleanup():
        sys.exit(0)
    else:
        # Flask's request-handling threads aren't daemon threads, so a
        # stuck one (e.g. a connect attempt still waiting on the SDK's own
        # timeout) would keep the process alive even after sys.exit() -
        # force-terminate immediately instead of hanging forever.
        logger.warning("Forcing immediate process exit (a background request may still be running).")
        os._exit(1)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _signal_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()

