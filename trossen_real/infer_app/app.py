# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Standalone inference-only web UI: live camera feeds + status + Start/Stop
Inference - completely SEPARATE from the teleop/data-collection web UI
(`teleop/app.py`) - no leader, no dataset recording, no gripper button.
Deliberately not mixed with it (different app, different static/ folder) so
neither gets harder to read chasing the other's concerns.

Talks to the SAME `follower_single_server.py` (own the arm connection,
`FollowerClient`) and the SAME `CameraManager` (own the cameras) as the
teleop app - only run ONE of these two apps against a given
`follower_single_server.py` at a time (both fully own the connection while
active, same as teleop). Also talks to `policy_server.py` (the training
repo's separate process - see there for why it has to be a separate
process/venv) via `PolicyClient`.

Run (needs `follower_single_server.py` AND `policy_server.py` already
running):

    python -m trossen_real.infer_app.app --port 5080
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
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory

from trossen_real.cameras.camera_manager import CameraManager
from trossen_real.config import list_available_configs, load_station_config
from trossen_real.human_intervention.episode_boundary import EpisodeBoundaryMonitor
from trossen_real.human_intervention.pedal_listener import PedalListener
from trossen_real.infer_app.eval_config import (
    EvalConfigError,
    list_available_eval_configs,
    load_eval_config,
)
from trossen_real.infer_app.eval_session import EvalSession, EvalSessionError
from trossen_real.infer_app.eval_writer import EvalWriter
from trossen_real.infer_app.infer_loop import InferenceLoop
from trossen_real.inference.intervention import InterventionManager
from trossen_real.inference.policy_client import (
    IMAGE_ENCODINGS, ChunkedPolicyClient, PolicyClient, check_action_space_mismatch,
    check_camera_health,
)
from trossen_real.leader.trossen_leader_single import TrossenSingleLeader
from trossen_real.log_setup import setup_logging
from trossen_real.teleop.dataset_recorder import DatasetRecorder
from trossen_real.teleop.follower_client import FollowerClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
setup_logging("infer_app")  # see trossen_real/log_setup.py - also writes to logs/infer_app/<date>/<time>/app.log
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.config = None
        self.follower: FollowerClient | None = None
        self.cameras: CameraManager | None = None
        self.policy: PolicyClient | None = None
        self.policy_health: dict | None = None
        self.infer_loop: InferenceLoop | None = None
        # Human-intervention takeover (optional, opt-in via enable_intervention=true
        # in /api/connect) - see trossen_real/human_intervention/INTERVENTION_PLAN.md.
        self.leader: TrossenSingleLeader | None = None
        self.pedal: PedalListener | None = None
        self.intervention_enabled: bool = False
        # Dataset recording (opt-in via enable_dataset_recording=true in /api/connect) -
        # independent of intervention_enabled/leader; reset/reward pedal episode-boundary
        # control (EpisodeBoundaryMonitor) rides the SAME PedalListener as intervention
        # when both are enabled, but works with no leader at all when intervention is off.
        # See PEDAL_BEHAVIOR.md.
        self.dataset_recorder: DatasetRecorder | None = None
        self.episode_monitor: EpisodeBoundaryMonitor | None = None
        self.dataset_recording_enabled: bool = False
        # Evaluation mode (opt-in via eval_config_name=... in /api/connect) - see
        # EVAL_MODE_PLAN.md. None => this app behaves exactly as it always has.
        # Deliberately NOT stored on InferSnapshot, which is rebuilt every tick.
        self.eval: EvalSession | None = None
        self.eval_writer: EvalWriter | None = None
        # Small dedicated lock for eval run-state transitions. Deliberately NOT
        # `self.lock`: /api/health polls twice a second and must never block
        # behind a slow connect (camera startup takes ~1s).
        self.eval_sync_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self.follower is not None

    @property
    def eval_enabled(self) -> bool:
        return self.eval is not None


state = AppState()

# Base directory for auto-generated per-run diagnostic JSONL logs (see
# api_start_inference()) - overridable via --log-dir, defaults to "logs"
# (relative to wherever this process was started from).
LOG_DIR = Path("logs")

# Base directory for recorded inference videos (one .mp4 per camera per run,
# stored in date subfolder YYYY-MM-DD) - overridable via --video-dir.
# Actions fetched per /predict_chunk round trip. 0 = per-tick /predict (the
# original behaviour), -1 = the server's own n_action_steps. Env var so it can
# be set at launch without a UI change; a request body may override per run.
DEFAULT_POLICY_CHUNK_STEPS = int(os.environ.get("INFER_POLICY_CHUNK_STEPS", "0"))

# How camera frames are packed for the policy server: "raw" (unchanged, 512 KB
# per tick), "zlib" (lossless, 3.3x smaller) or "jpeg" (15x smaller at q95).
# See encode_image() for the measurements behind those numbers.
DEFAULT_IMAGE_ENCODING = os.environ.get("INFER_IMAGE_ENCODING", "raw")
DEFAULT_JPEG_QUALITY = int(os.environ.get("INFER_JPEG_QUALITY", "95"))

VIDEO_DIR = Path("infer_videos")

# Base directory for recorded inference LeRobot datasets (one dataset/session
# per connect->disconnect cycle with enable_dataset_recording=true, one episode
# per Start/Stop Inference cycle within it) - overridable via --dataset-dir.
DATASET_DIR = Path("infer_dataset")

# Base directory for evaluation artifacts (summary.json + timeline.jsonl per run,
# plus a flat index.jsonl) when eval mode is enabled - overridable via --eval-dir.
# Defaults to the eval config's own `output_root` when this stays None.
EVAL_DIR: Path | None = None

# Extra LeRobot dataset features (beyond build_lerobot_features()'s usual set) recorded
# ONLY by the infer app's DatasetRecorder - see PEDAL_BEHAVIOR.md and infer_loop.py's
# add_frame() call. Fixed single-arm shapes since this app only supports arm_mode: single
# (enforced in api_connect()) - action_dim_per_side is 7 (6 joints + gripper) either way.
INFER_EXTRA_LEROBOT_FEATURES = {
    "observation.intervened": {"dtype": "bool", "shape": (1,), "names": ["intervened"]},
    "observation.policy_action": {
        "dtype": "float32", "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"],
    },
    "next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
}


def _require_connected():
    if not state.connected:
        return jsonify(error="Not connected. POST /api/connect first."), 409
    return None


def _require_eval():
    if not state.connected:
        return jsonify(error="Not connected. POST /api/connect first."), 409
    if state.eval is None:
        return jsonify(error="Eval mode is not enabled - reconnect with eval_config_name set."), 409
    return None


def _station_name() -> str:
    return state.config.station_name if state.config is not None else ""


def _eval_policy_meta() -> dict:
    """Everything needed to reproduce a run later. `n_action_steps` matters most:
    it is an EVAL-time server flag that can differ from the trained value, so two
    runs are only comparable if it matches."""
    ph = state.policy_health or {}
    return {
        "checkpoint": ph.get("checkpoint"),
        "device": ph.get("device"),
        "chunk_size": ph.get("chunk_size"),
        "n_action_steps": ph.get("n_action_steps"),
        "likely_action_space": ph.get("likely_action_space"),
        "action_space": state.infer_loop.action_space if state.infer_loop is not None else None,
    }


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
    policy_server_url = body.get("policy_server_url")
    action_space = body.get("action_space", "absolute")
    enable_intervention = bool(body.get("enable_intervention", False))
    enable_dataset_recording = bool(body.get("enable_dataset_recording", False))
    # Eval mode: opt-in, and inert everywhere when eval_config_name is absent.
    eval_config_name = (body.get("eval_config_name") or "").strip() or None
    eval_checkpoint_label = (body.get("checkpoint_label") or "").strip()
    eval_operator = (body.get("operator") or "").strip()
    if not config_name:
        return jsonify(error="config_name is required"), 400
    if not policy_server_url:
        return jsonify(error="policy_server_url is required"), 400
    if action_space not in ("absolute", "delta_joint"):
        return jsonify(error="action_space must be 'absolute' or 'delta_joint'"), 400

    # Load/validate the eval config BEFORE touching hardware: it is pure file I/O,
    # and failing here must not leave a connected follower/cameras behind.
    eval_config = None
    if eval_config_name:
        try:
            eval_config = load_eval_config(eval_config_name)
        except EvalConfigError as exc:
            return jsonify(error=f"Eval config problem: {exc}"), 400

    with state.lock:
        if state.connected:
            return jsonify(error="Already connected. POST /api/disconnect first."), 409
        try:
            config = load_station_config(config_name)
            if config.arm_mode != "single":
                return jsonify(error=f"Only arm_mode: single is supported for now (got '{config.arm_mode}')."), 400
            if config.control.command_space != "joint":
                return jsonify(
                    error=f"This inference app only supports command_space: joint checkpoints, "
                          f"but '{config_name}' has command_space: '{config.control.command_space}'."
                ), 400

            follower = FollowerClient(config.follower_server_url)
            try:
                follower.connect()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not reach follower_single_server.py at {config.follower_server_url} - is it running? ({exc})"
                ) from exc

            policy = PolicyClient(policy_server_url)
            try:
                policy_health = policy.health()
            except Exception as exc:
                follower.disconnect()
                raise RuntimeError(
                    f"Could not reach policy_server.py at {policy_server_url} - is it running? ({exc})"
                ) from exc
            if not policy_health.get("loaded"):
                follower.disconnect()
                return jsonify(error=f"policy_server.py has no policy loaded - {policy_health}"), 400

            # Chunked fetching: one round trip per N ticks instead of per tick.
            # Same actions (see ChunkedPolicyClient) - this only decides where
            # the action queue lives. Off by default so behaviour does not
            # change for anyone who has not asked for it; set
            # INFER_POLICY_CHUNK_STEPS=15 (or pass policy_chunk_steps in the
            # request body) to turn it on. -1 means "whatever the server's
            # n_action_steps is".
            chunk_steps = int(body.get("policy_chunk_steps", DEFAULT_POLICY_CHUNK_STEPS))
            if chunk_steps:
                if not policy_health.get("supports_predict_chunk"):
                    follower.disconnect()
                    return jsonify(
                        error="policy_chunk_steps was requested but this policy_server does not "
                              "support /predict_chunk (old server, or a temporal-ensembling / "
                              "relative-actions checkpoint where chunking would change the actions)."
                    ), 400
                policy = ChunkedPolicyClient(policy, None if chunk_steps < 0 else chunk_steps)
                logger.info("Policy fetching: CHUNKED, %s actions per round trip (server n_action_steps=%s)",
                            "server default" if chunk_steps < 0 else chunk_steps,
                            policy_health.get("n_action_steps"))
            else:
                logger.info("Policy fetching: per-tick (one /predict round trip per control step)")

            # ---- OPTIONAL residual RL wrap -----------------------------------
            # Purely additive: with no `residual_checkpoint` in the request this
            # whole block is skipped and the app behaves exactly as before.
            #
            # ResidualPolicyClient is a drop-in for PolicyClient, so wrapping it
            # HERE means infer_loop.py needs no changes at all - it still just
            # calls policy.predict_full(obs) and gets an action in real units.
            residual_ckpt = (body.get("residual_checkpoint") or "").strip()
            if residual_ckpt:
                from trossen_real.inference.residual_client import (
                    BasePolicyMismatch,
                    ResidualPolicyClient,
                    ResidualUnavailable,
                )

                res_log = body.get("residual_log")
                if res_log is None:
                    res_log = str(LOG_DIR / "residual" /
                                  f"residual_{state.config.station_name}_"
                                  f"{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
                try:
                    policy = ResidualPolicyClient(
                        policy, residual_ckpt,
                        device=body.get("residual_device"),
                        collect_q=bool(body.get("collect_q", True)),
                        base_policy_assert=str(body.get("base_policy_assert", "strict")),
                        log_path=res_log,
                        residual_scale=float(body.get("residual_scale", 1.0)),
                    )
                except BasePolicyMismatch as exc:
                    follower.disconnect()
                    return jsonify(error=str(exc)), 409
                except (ResidualUnavailable, Exception) as exc:
                    follower.disconnect()
                    return jsonify(error=f"could not load residual policy: {exc}"), 400
                logger.info("Residual RL ACTIVE: %s", policy.meta)
                logger.info("Residual tick log: %s", res_log)

            image_encoding = str(body.get("image_encoding", DEFAULT_IMAGE_ENCODING))
            jpeg_quality = int(body.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
            if image_encoding not in IMAGE_ENCODINGS:
                follower.disconnect()
                return jsonify(error=f"image_encoding must be one of {list(IMAGE_ENCODINGS)}, "
                                     f"got '{image_encoding}'."), 400
            if image_encoding != "raw":
                supported = policy_health.get("supported_image_encodings") or ["raw"]
                if image_encoding not in supported:
                    follower.disconnect()
                    return jsonify(
                        error=f"image_encoding='{image_encoding}' was requested but this "
                              f"policy_server only decodes {supported}. Update policy_server.py "
                              f"on the machine running it."
                    ), 400
            logger.info("Camera frames -> policy server encoded as '%s'%s",
                        image_encoding,
                        f" (quality {jpeg_quality})" if image_encoding == "jpeg" else "")

            force_action_space = bool(body.get("force_action_space", False))
            mismatch = check_action_space_mismatch(action_space, policy_health)
            if mismatch and not force_action_space:
                follower.disconnect()
                return jsonify(error=f"{mismatch} Pass force_action_space=true to proceed anyway (not recommended)."), 400
            elif mismatch:
                logger.warning("Proceeding despite action_space mismatch (force_action_space=true): %s", mismatch)

            cameras = CameraManager(config)
            cameras.start()
            time.sleep(1.0)  # let the camera threads produce their first real frames

            # Human-intervention takeover is opt-in and REQUIRED to actually work when
            # requested - a leader that fails to connect fails this WHOLE request (no
            # silent degraded mode), per the design doc.
            leader = None
            pedal = None
            intervention_mgr = None
            # A single PedalListener instance is shared by intervention (if enabled) AND
            # dataset-recording's episode-boundary control (if enabled) - reset/reward
            # pedal control does NOT need a leader at all, only the intervention pedal
            # does. See PEDAL_BEHAVIOR.md.
            # Eval mode needs the pedals too: the reward pedal is how a success is
            # marked (and ends the run early instead of burning all 500 steps),
            # the reset pedal aborts a run that has obviously failed. Neither
            # needs a leader or dataset recording - see PEDAL_BEHAVIOR.md.
            if enable_intervention or enable_dataset_recording or eval_config is not None:
                pedal = PedalListener()
            if enable_intervention:
                try:
                    leader = TrossenSingleLeader(config.leader_ips["single"])
                    leader.connect()
                except Exception as exc:
                    cameras.stop()
                    follower.disconnect()
                    raise RuntimeError(
                        f"enable_intervention=true but could not connect the leader arm - {exc}"
                    ) from exc
                intervention_mgr = InterventionManager(leader, pedal, policy)

            dataset_recorder = None
            episode_monitor = None
            if enable_dataset_recording:
                dataset_recorder = DatasetRecorder(
                    config, save_root=DATASET_DIR, extra_features=INFER_EXTRA_LEROBOT_FEATURES,
                )
            # Independent of the recorder: pedal episode-boundary control is
            # useful whenever there is a human watching, which is exactly the
            # eval case. With no recorder the frame reward simply goes unused -
            # the pedal still ends the run and sets stop_reason.
            if enable_dataset_recording or eval_config is not None:
                episode_monitor = EpisodeBoundaryMonitor(pedal)

            infer_loop = InferenceLoop(
                follower, cameras, policy, config.control, policy_health["image_keys"], action_space=action_space,
                intervention=intervention_mgr,
                leader_sync_goal_time_s=config.reset["single"].goal_time_s,
                station_name=config.station_name,
                image_encoding=image_encoding,
                jpeg_quality=jpeg_quality,
            )

            state.config = config
            state.follower = follower
            state.cameras = cameras
            state.policy = policy
            state.policy_health = policy_health
            state.infer_loop = infer_loop
            state.leader = leader
            state.pedal = pedal
            state.intervention_enabled = enable_intervention
            state.dataset_recorder = dataset_recorder
            state.episode_monitor = episode_monitor
            state.dataset_recording_enabled = enable_dataset_recording

            if eval_config is not None:
                eval_root = str(EVAL_DIR) if EVAL_DIR is not None else eval_config.output_root
                state.eval = EvalSession(
                    eval_config, output_root=eval_root,
                    checkpoint_label=eval_checkpoint_label, operator=eval_operator,
                )
                state.eval_writer = EvalWriter(eval_root)
                # Resume: a 20-pose x 15-checkpoint campaign spans many restarts,
                # so pick up where this checkpoint left off instead of at pose 1.
                try:
                    state.eval.load_history(state.eval_writer.read_index())
                    if state.eval.history_saved_poses:
                        logger.info(
                            "Eval resume: %d pose(s) already saved for '%s' -> starting at pose %d",
                            len(state.eval.history_saved_poses), eval_checkpoint_label,
                            state.eval.pose_cursor,
                        )
                except Exception:
                    # A damaged index must not block a session - worst case the
                    # operator sets the pose cursor by hand.
                    logger.exception("Could not read eval history; starting with an empty one.")
                logger.info(
                    "Eval mode ON: config=%s hash=%s (%d stages, %d poses) -> %s",
                    eval_config.eval_name, eval_config.config_hash,
                    eval_config.n_stages, eval_config.n_poses, eval_root,
                )
        except Exception as exc:
            logger.exception("Connect failed")
            return jsonify(error=str(exc)), 500

    return jsonify(
        status="ok",
        station_name=config.station_name,
        cameras=cameras.camera_names,
        policy_health=policy_health,
        action_space=action_space,
        intervention_enabled=enable_intervention,
        dataset_recording_enabled=enable_dataset_recording,
        eval_mode=state.eval is not None,
        eval_config=(state.eval.config.to_dict() if state.eval is not None else None),
    )


def _sync_eval_run_finished() -> None:
    """Move a finished run into 'pending review' and attach its artifacts.

    Driven from /api/health as well as the Stop button, because a run can end on
    its OWN - max_steps, the reset pedal, or releasing the reward pedal - with no
    HTTP call involved at all (see `EpisodeBoundaryMonitor`). Cheap and
    idempotent, so polling it is fine.
    """
    if state.eval is None or state.eval.current is None or state.infer_loop is None:
        return
    if state.infer_loop.is_running:
        return
    with state.eval_sync_lock:
        run = state.eval.current
        if run is None:
            return
        snap = state.infer_loop.get_snapshot()
        run.videos = list(snap.video_files or [])
        run.tick_log = run.tick_log or snap.log_file
        run.stop_reason = run.stop_reason or snap.stop_reason
        state.eval.end_run()
        logger.info("Eval run %s awaiting review (pose %s, %d steps, stop_reason=%s)",
                    run.run_id, run.pose_index, run.step_count, run.stop_reason)


def _finalize_pending_eval_run(where: str) -> None:
    """Shutdown path: a run that actually happened is never silently dropped -
    it is written with status='unreviewed' so the operator can score it later.
    Best-effort: must never prevent the arm from being disconnected."""
    if state.eval is None:
        return
    try:
        run = state.eval.abandon_pending(reason=f"{where} with a review still pending")
        if run is not None and state.eval_writer is not None:
            state.eval_writer.write_run(
                run, state.eval.config, station=_station_name(), policy=_eval_policy_meta()
            )
            logger.warning("Eval run %s written as UNREVIEWED (%s)", run.run_id, where)
    except Exception:
        logger.exception("Could not finalize the pending eval run; continuing shutdown anyway.")


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    with state.lock:
        if state.infer_loop is not None and state.infer_loop.is_running:
            state.infer_loop.stop()
        _finalize_pending_eval_run("disconnected")
        if state.cameras is not None:
            state.cameras.stop()
        if state.leader is not None:
            reset_cfg = state.config.reset["single"]
            try:
                state.leader.disconnect(
                    park_waypoints=[reset_cfg.staged_joint_positions, [0.0] * 7],
                    goal_time=reset_cfg.goal_time_s,
                )
            except Exception:
                logger.exception("Error parking/disconnecting leader; continuing anyway.")
        if state.pedal is not None:
            state.pedal.stop()
        if state.follower is not None:
            try:
                state.follower.disconnect()
            except Exception:
                logger.exception("Error disconnecting follower; continuing anyway.")
        # infer_loop.stop() above already guarantees any in-progress episode was saved
        # (see InferenceLoop._run()'s finally block) - just finalize the dataset session.
        if state.dataset_recorder is not None and state.dataset_recorder.is_session_active:
            try:
                state.dataset_recorder.stop_session()
            except RuntimeError:
                logger.exception("Error finalizing dataset session; continuing anyway.")
        state.config = None
        state.follower = None
        state.cameras = None
        state.policy = None
        state.policy_health = None
        state.infer_loop = None
        state.leader = None
        state.pedal = None
        state.intervention_enabled = False
        state.dataset_recorder = None
        state.episode_monitor = None
        state.dataset_recording_enabled = False
        state.eval = None
        state.eval_writer = None
    return jsonify(status="ok")


@app.route("/api/health", methods=["GET"])
def api_health():
    if not state.connected:
        return jsonify(connected=False)

    camera_health = state.cameras.health()
    camera_details = state.cameras.get_camera_status()
    camera_problems = check_camera_health(state.cameras, state.policy_health["image_keys"])
    _sync_eval_run_finished()  # a run may have ended by itself since the last poll
    snapshot = state.infer_loop.get_snapshot()
    # Live eval state for the UI. Read-only here - all eval mutation happens in
    # the /api/eval/* endpoints under state.lock.
    eval_state = state.eval.to_dict() if state.eval is not None else {"eval_mode": False}

    return jsonify(
        connected=True,
        station_name=state.config.station_name,
        follower=state.follower.check(),
        cameras=camera_health,
        camera_details=camera_details,
        camera_problems=camera_problems,  # non-empty -> would be refused if you tried to start inference right now
        policy_health=state.policy_health,
        action_space=state.infer_loop.action_space,
        inferring=snapshot.running,
        settling=snapshot.settling,
        step=snapshot.step,
        max_steps=snapshot.max_steps,
        last_action=snapshot.last_action,
        achieved_hz=snapshot.achieved_hz,
        target_hz=state.config.control.frequency_hz,
        error=snapshot.error,
        log_file=snapshot.log_file,
        video_files=snapshot.video_files,
        intervention_enabled=state.intervention_enabled,
        intervening=(state.pedal.is_intervention() if state.pedal is not None else False),
        leader_connected=(state.leader.is_connected() if state.leader is not None else False),
        dataset_recording_enabled=state.dataset_recording_enabled,
        dataset_session_active=(state.dataset_recorder.is_session_active if state.dataset_recorder is not None else False),
        dataset_recording=snapshot.dataset_recording,
        num_episodes=(state.dataset_recorder.num_episodes if state.dataset_recorder is not None else 0),
        dataset_dir=(str(state.dataset_recorder.session_dir) if state.dataset_recorder is not None and state.dataset_recorder.session_dir else None),
        stop_reason=snapshot.stop_reason,
        **{"eval": eval_state},
    )


# ----------------------------------------------------------------------
# Inference control
# ----------------------------------------------------------------------
@app.route("/api/start_inference", methods=["POST"])
def api_start_inference():
    err = _require_connected()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    settle_time_s = float(body.get("settle_time_s", 2.0))
    max_steps = body.get("max_steps")
    max_steps = int(max_steps) if max_steps not in (None, "") else None
    skip_reset = bool(body.get("skip_reset", False))
    enable_logging = bool(body.get("log", True))
    record_video = bool(body.get("record_video", True))

    if state.infer_loop.is_running:
        return jsonify(error="Inference is already running."), 409

    problems = check_camera_health(state.cameras, state.policy_health["image_keys"])
    if problems:
        return jsonify(error="Refusing to start - camera problems: " + "; ".join(problems)), 400

    log_file = None
    if enable_logging:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = str(LOG_DIR / f"infer_{state.config.station_name}_{timestamp}.jsonl")

    # ---- eval mode: open a run BEFORE the loop starts -------------------------
    eval_hook = None
    if state.eval is not None:
        if max_steps is None:
            max_steps = state.eval.config.max_steps_default  # docs/archive/Notes.md: 500
        try:
            with state.lock:
                eval_body = body.get("eval") or {}
                run = state.eval.begin_run(
                    pose_index=eval_body.get("pose_index"),
                    checkpoint_label=(eval_body.get("checkpoint_label") or "").strip() or None,
                    max_steps=max_steps,
                    tags=eval_body.get("tags") or [],
                    run_notes=eval_body.get("run_notes", ""),
                    intervention_enabled=state.intervention_enabled,
                )
                eval_hook = state.eval.hook
        except (EvalSessionError, EvalConfigError) as exc:
            return jsonify(error=str(exc)), 409

    if state.dataset_recorder is not None and not state.dataset_recorder.is_session_active:
        # First Start Inference press since Connect - opens the ONE dataset session for
        # this whole connect->disconnect cycle (see PEDAL_BEHAVIOR.md / api_disconnect()).
        task_name = body.get("task_name", "")
        if not task_name or not task_name.strip():
            return jsonify(error="task_name is required to start dataset recording (first Start Inference of this connection only)."), 400
        try:
            state.dataset_recorder.start_session(task_name)
        except (ValueError, RuntimeError) as exc:
            return jsonify(error=str(exc)), 400

    try:
        state.infer_loop.start(
            settle_time_s=settle_time_s,
            max_steps=max_steps,
            skip_reset=skip_reset,
            log_file=log_file,
            record_video=record_video,
            video_dir=VIDEO_DIR,
            recorder=state.dataset_recorder,
            episode_monitor=state.episode_monitor,
            eval_hook=eval_hook,
        )
    except RuntimeError as exc:
        # The run never happened - drop it rather than leaving it pending, which
        # would block every future Start.
        if state.eval is not None:
            state.eval.abort_current()
        return jsonify(error=str(exc)), 409

    if state.eval is not None and state.eval.current is not None:
        run = state.eval.current
        run.tick_log = log_file
        if state.dataset_recorder is not None and state.dataset_recorder.is_session_active:
            run.dataset_dir = str(state.dataset_recorder.session_dir)
            # This run's episode is the one about to be recorded, i.e. the count
            # BEFORE it is saved (episodes are appended on stop).
            run.dataset_episode_index = state.dataset_recorder.num_episodes
    return jsonify(status="ok", log_file=log_file)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Send the follower to the staged reset pose, and - when intervention is
    enabled - re-sync the leader to it afterwards.

    Runs `InferenceLoop.reset_and_sync()`, the SAME sequence Start Inference
    performs, so the two can never drift apart. Blocking (~reset.goal_time_s).

    Refused while a run is in progress: resetting mid-rollout would fight the
    policy for control of the arm.
    """
    err = _require_connected()
    if err:
        return err
    if state.infer_loop.is_running:
        return jsonify(error="Stop inference before resetting the robot."), 409
    try:
        state.infer_loop.reset_and_sync()
    except Exception as exc:
        logger.exception("Reset failed")
        return jsonify(error=str(exc)), 500
    return jsonify(status="ok", leader_synced=state.leader is not None)


@app.route("/api/stop_inference", methods=["POST"])
def api_stop_inference():
    err = _require_connected()
    if err:
        return err
    state.infer_loop.stop()
    _sync_eval_run_finished()
    return jsonify(status="ok")


# ----------------------------------------------------------------------
# Evaluation mode (see EVAL_MODE_PLAN.md). Every route here is a no-op for a
# normal, non-eval inference session - eval is only active when /api/connect was
# given an eval_config_name.
# ----------------------------------------------------------------------
@app.route("/api/eval/configs", methods=["GET"])
def api_eval_configs():
    return jsonify(configs=list_available_eval_configs())


@app.route("/api/eval/config", methods=["GET"])
def api_eval_config():
    err = _require_eval()
    if err:
        return err
    return jsonify(config=state.eval.config.to_dict(), session=state.eval.to_dict())


@app.route("/api/eval/pose_cursor", methods=["POST"])
def api_eval_pose_cursor():
    err = _require_eval()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    try:
        with state.eval_sync_lock:
            state.eval.set_pose_cursor(int(body["pose_index"]))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=f"pose_index must be an integer ({exc})"), 400
    except EvalConfigError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(status="ok", session=state.eval.to_dict())


@app.route("/api/eval/annotate", methods=["POST"])
def api_eval_annotate():
    """Idempotent draft update on the run awaiting review. Safe to call repeatedly
    as the operator changes their mind before saving."""
    err = _require_eval()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    kwargs = {}
    if "furthest_stage" in body:
        kwargs["furthest_stage"] = body["furthest_stage"]
    if "failure_note" in body:
        kwargs["failure_note"] = str(body["failure_note"])
    if "intervened_during_stage" in body:
        kwargs["intervened_during_stage"] = body["intervened_during_stage"]
    if "operator" in body:
        kwargs["operator"] = str(body["operator"])
    try:
        with state.eval_sync_lock:
            state.eval.annotate(**kwargs)
    except (EvalSessionError, EvalConfigError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(status="ok", session=state.eval.to_dict())


@app.route("/api/eval/finalize", methods=["POST"])
def api_eval_finalize():
    """Save or discard the pending run. Discard is SOFT: the summary/timeline are
    still written with status='discarded' and every referenced artifact (video,
    tick log, dataset episode) is left untouched."""
    err = _require_eval()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    action = str(body.get("action", "")).lower()
    try:
        with state.eval_sync_lock:
            run = state.eval.finalize(action, reason=str(body.get("reason", "")))
            summary = state.eval_writer.write_run(
                run, state.eval.config, station=_station_name(), policy=_eval_policy_meta()
            )
    except (EvalSessionError, EvalConfigError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        logger.exception("Failed to write eval run")
        return jsonify(error=f"Could not write eval artifacts: {exc}"), 500
    return jsonify(status="ok", run_id=run.run_id, eval_dir=summary["artifacts"]["eval_dir"],
                   session=state.eval.to_dict())


# ----------------------------------------------------------------------
# Camera MJPEG streaming (same pattern as teleop/app.py)
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


# ----------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------
def _cleanup(lock_timeout: float = 3.0) -> bool:
    acquired = state.lock.acquire(timeout=lock_timeout)
    if not acquired:
        logger.warning("Could not acquire state lock within %.1fs during shutdown - skipping graceful disconnect.", lock_timeout)
        return False
    try:
        if state.infer_loop is not None and state.infer_loop.is_running:
            state.infer_loop.stop()
        # Before anything else is torn down: a run that happened must not be lost
        # just because the app was killed instead of reviewed.
        _sync_eval_run_finished()
        _finalize_pending_eval_run("app shut down")
        if state.cameras is not None:
            state.cameras.stop()
        if state.leader is not None:
            try:
                reset_cfg = state.config.reset["single"] if state.config is not None else None
                state.leader.disconnect(
                    park_waypoints=[reset_cfg.staged_joint_positions, [0.0] * 7] if reset_cfg else None,
                    goal_time=reset_cfg.goal_time_s if reset_cfg else 3.0,
                )
            except Exception:
                logger.exception("Error parking/disconnecting leader during shutdown; ignoring.")
        if state.pedal is not None:
            state.pedal.stop()
        if state.follower is not None:
            try:
                state.follower.disconnect()
            except Exception:
                logger.exception("Error disconnecting follower during shutdown; ignoring.")
        # Fallback finalize for the "kill the app instead of clicking Disconnect" case
        # (infer_loop.stop() above already guarantees any in-progress episode was saved).
        if state.dataset_recorder is not None and state.dataset_recorder.is_session_active:
            try:
                state.dataset_recorder.stop_session()
            except Exception:
                logger.exception("Error finalizing dataset session during shutdown; ignoring.")
    finally:
        state.lock.release()
    return True


atexit.register(_cleanup)


def _signal_handler(signum, _frame):
    logger.info("Received signal %s, disconnecting and shutting down...", signum)
    if not _cleanup():
        import os
        os._exit(1)
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _signal_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5080)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--log-dir", default="logs",
        help="Base directory for auto-generated per-run diagnostic JSONL logs (one file per Start Inference "
             "run, named infer_<station>_<timestamp>.jsonl - see api_start_inference()). Default: ./logs "
             "(relative to wherever this process is started from). Created automatically if missing.",
    )
    parser.add_argument(
        "--video-dir", default="infer_videos",
        help="Base directory for recorded inference MP4 video files (organized into date subfolders YYYY-MM-DD). "
             "Default: ./infer_videos (relative to wherever this process is started from). Created automatically if missing.",
    )
    parser.add_argument(
        "--dataset-dir", default="infer_dataset",
        help="Base directory for recorded inference LeRobot datasets, when enable_dataset_recording=true "
             "(one dataset/session folder per connect->disconnect cycle, named {timestamp}_{task_name} - see "
             "PEDAL_BEHAVIOR.md). Default: ./infer_dataset (relative to wherever this process is started from).",
    )
    parser.add_argument(
        "--eval-dir", default=None,
        help="Base directory for evaluation artifacts when eval mode is enabled (summary.json + "
             "timeline.jsonl per run, plus a flat index.jsonl aggregated by "
             "trossen_real/scripts/eval_report.py). Default: the eval config's own 'output_root' "
             "(eval_runs). See trossen_real/infer_app/EVAL_MODE_PLAN.md.",
    )
    args = parser.parse_args()

    global LOG_DIR, VIDEO_DIR, DATASET_DIR, EVAL_DIR
    LOG_DIR = Path(args.log_dir)
    VIDEO_DIR = Path(args.video_dir)
    DATASET_DIR = Path(args.dataset_dir)
    EVAL_DIR = Path(args.eval_dir) if args.eval_dir else None

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
