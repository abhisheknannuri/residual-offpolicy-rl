# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Standalone Trossen FOLLOWER robot HTTP server - single arm.

Runs in its own terminal, on its own port. Endpoints and behavior closely
mirror hil-serl's `serl_robot_infra/robot_servers/windowxai_follower_server.py`
(same route names/payload shapes), so a future gym-wrapper client can be
built the same way hil-serl's own gym envs talk to that server.

Internally reuses `TrossenFollowerSingle` (follower-only engine, never
connects a leader) so the already hardware-tested safety clipping /
two-stage reset / disconnect-to-sleep-position logic isn't duplicated -
this file is just a thin hil-serl-shaped route layer on top.

Only single-arm (`arm_mode: single`) station configs are supported for now.

Run (from the repo root, inside the resfit venv)::

    python -m trossen_real.follower.follower_single_server --config trossen_station2_single --port 5060

Endpoints (all POST unless noted, matching windowxai_follower_server.py):
    /connect_to_robot            -> connect the follower
    /disconnect_from_robot       -> disconnect the follower; body optional
                                     {"skip_staged_position": bool} (default
                                     False = staged->sleep move first, same
                                     as always; True = disconnect immediately
                                     from wherever the arm currently is)
    /configure                   -> re-run the safe staged-position reset
    /getpos                      -> {"pose": [x,y,z,ax,ay,az]}
    /getq                        -> {"q": [6 joints + gripper]}
    /get_gripper                 -> {"gripper": float}
    /getstate                    -> {"pose":[6],"q":[7],"gripper_pos":f,"dq":[7],"efforts":[7],"accelerations":[7]}
    /pose, /move_to_ee_pose      -> body {"arr":[7], "blocking"?, "min_time_to_move"?}
    /move_to_joint_positions     -> body {"arr":[7] (joint_0..joint_5,gripper), "blocking"?, "min_time_to_move"?}
                                     - our own addition, not in hil-serl (which is EE-only); matches
                                     Trossen's own `lerobot_trossen` reference's joint-space control.
    /move_gripper                -> body {"gripper_pos":f, "blocking"?, "min_time_to_move"?}
    /jointreset                  -> staged-position reset
    /clearerr                    -> staged-position reset (same as hil-serl: full reset, not a light error clear)
    /reset                       -> body optional {"pose_to_reach":[7]}
    /health (GET)                -> our own addition, not in hil-serl, kept for convenience


curl -X POST http://127.0.0.1:5060/connect_to_robot
curl -X POST http://127.0.0.1:5060/reset
curl -X POST http://127.0.0.1:5060/disconnect_from_robot

curl -X POST http://127.0.0.1:5060/move_gripper \
     -H "Content-Type: application/json" \
     -d '{"gripper_pos": 0.4, "blocking": true, "min_time_to_move": 4}'


========================= Left Follower Reset For Data Collection =======================

curl -X POST http://127.0.0.1:5096/connect_to_robot

curl -X POST http://127.0.0.1:5096/move_to_joint_positions \
-H "Content-Type: application/json" \
-d '{"arr":[0,0,0,0,-0.1,0,0.4], "blocking": true, "min_time_to_move": 3}'

Make sure you run teh hgetq api call and get something close to this
curl -X POST http://127.0.0.1:5096/getq 
{"q":[-0.004386968910694122,0.0009536888683214784,0.01735713705420494,0.009727626107633114,-0.09327077120542526,-0.004005493130534887,0.03994884341955185]}

curl -X POST http://127.0.0.1:5096/disconnect_from_robot \
-H "Content-Type: application/json" \
-d '{"skip_staged_position": true}'

curl -X POST http://127.0.0.1:5096/move_gripper \
     -H "Content-Type: application/json" \
     -d '{"gripper_pos": 0.42, "blocking": true, "min_time_to_move": 3}'
===========================================================================================

"""

from __future__ import annotations

import argparse
import atexit
import logging
import signal
import sys

import numpy as np
from flask import Flask, jsonify, request

from trossen_real.config import StationConfig, load_station_config
from trossen_real.follower.follower_single import TrossenFollowerSingle
from trossen_real.log_setup import setup_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

SIDE = "single"  # this server only ever operates on the single-arm station side


class ServerState:
    def __init__(self) -> None:
        self.config: StationConfig | None = None
        self.hardware: TrossenFollowerSingle | None = None

    @property
    def connected(self) -> bool:
        return self.hardware is not None and self.hardware.connected


state = ServerState()


def _require_connected():
    if not state.connected:
        return jsonify(error="Not connected. POST /connect_to_robot first."), 409
    return None


# ----------------------------------------------------------------------
# Connection lifecycle
# ----------------------------------------------------------------------
@app.route("/connect_to_robot", methods=["POST"])
def connect_to_robot():
    if state.connected:
        return jsonify(error="Already connected. POST /disconnect_from_robot first."), 409
    try:
        state.hardware = TrossenFollowerSingle(state.config)
        state.hardware.connect()
    except Exception as exc:
        logger.exception("Connect failed")
        return jsonify(error=str(exc)), 500
    return "Connected to Robot"


@app.route("/disconnect_from_robot", methods=["POST"])
def disconnect_from_robot():
    """Body optional: {"skip_staged_position": bool} - default False (omit
    entirely for identical behavior to before this param existed): move to
    the staged position then the sleep position before disconnecting, same
    as always. True: skip both moves, disconnect immediately from wherever
    the arm currently is - see TrossenFollowerSingle.disconnect()'s
    docstring for what you're opting out of."""
    body = request.get_json(force=True, silent=True) or {}
    skip_staged_position = bool(body.get("skip_staged_position", False))
    if state.hardware is not None:
        state.hardware.disconnect(skip_staged_position=skip_staged_position)
    return "Disconnected from Robot"


@app.route("/configure", methods=["POST"])
def configure():
    """Re-establish a known-safe state (mirrors hil-serl's `configure()`,
    which sets position mode + moves to the staged position)."""
    err = _require_connected()
    if err:
        return err
    state.hardware.reset()
    return "Configured Robot"


@app.route("/health", methods=["GET"])
def health():
    """Not part of hil-serl's route set, kept for convenience."""
    if not state.connected:
        return jsonify(connected=False)
    return jsonify(
        connected=True,
        station_name=state.config.station_name,
        arms=state.hardware.health(),
        using_mock_hardware=state.hardware.using_mock_hardware(),
    )


# ----------------------------------------------------------------------
# State getters
# ----------------------------------------------------------------------
@app.route("/getpos", methods=["POST"])
def getpos():
    err = _require_connected()
    if err:
        return err
    pose = state.hardware.get_follower_state()[SIDE][:6]
    return jsonify(pose=pose.tolist())


@app.route("/getq", methods=["POST"])
def getq():
    err = _require_connected()
    if err:
        return err
    q = state.hardware.get_follower_joint_state(SIDE)["q"]
    return jsonify(q=q.tolist())


@app.route("/get_gripper", methods=["POST"])
def get_gripper():
    err = _require_connected()
    if err:
        return err
    gripper = float(state.hardware.get_follower_state()[SIDE][6])
    return jsonify(gripper=gripper)


@app.route("/getstate", methods=["POST"])
def getstate():
    err = _require_connected()
    if err:
        return err
    state_7d = state.hardware.get_follower_state()[SIDE]
    joint_state = state.hardware.get_follower_joint_state(SIDE)
    return jsonify(
        pose=state_7d[:6].tolist(),
        q=joint_state["q"].tolist(),
        gripper_pos=float(state_7d[6]),
        dq=joint_state["dq"].tolist(),
        efforts=joint_state["efforts"].tolist(),
        accelerations=joint_state["accelerations"].tolist(),
    )


# ----------------------------------------------------------------------
# Motion commands
# ----------------------------------------------------------------------
def _pose_route():
    err = _require_connected()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    arr = body.get("arr")
    if arr is None or len(arr) != 7:
        return jsonify(error="Expected {'arr': [7 floats] (x,y,z,ax,ay,az,gripper)}"), 400
    blocking = bool(body.get("blocking", False))
    min_time_to_move = body.get("min_time_to_move")
    goal_time = float(min_time_to_move) if min_time_to_move is not None else None
    state.hardware.goal_ee(SIDE, np.asarray(arr, dtype=np.float32), goal_time=goal_time, blocking=blocking)
    return "Moved"


@app.route("/pose", methods=["POST"])
def pose():
    return _pose_route()


@app.route("/move_to_ee_pose", methods=["POST"])
def move_to_ee_pose():
    return _pose_route()


def _joint_route():
    err = _require_connected()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    arr = body.get("arr")
    if arr is None or len(arr) != 7:
        return jsonify(error="Expected {'arr': [7 floats] (joint_0..joint_5, gripper)}"), 400
    blocking = bool(body.get("blocking", False))
    min_time_to_move = body.get("min_time_to_move")
    goal_time = float(min_time_to_move) if min_time_to_move is not None else None
    state.hardware.goal_joint(SIDE, np.asarray(arr, dtype=np.float32), goal_time=goal_time, blocking=blocking)
    return "Moved"


@app.route("/move_to_joint_positions", methods=["POST"])
def move_to_joint_positions():
    return _joint_route()


"""
curl -X POST http://127.0.0.1:5060/move_gripper \
     -H "Content-Type: application/json" \
     -d '{"gripper_pos": 0.042, "blocking": true, "min_time_to_move": 3.0}'
"""

@app.route("/move_gripper", methods=["POST"])
def move_gripper():
    err = _require_connected()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    if "gripper_pos" not in body:
        return jsonify(error="Expected {'gripper_pos': float}"), 400
    gripper_pos = float(body["gripper_pos"])
    blocking = bool(body.get("blocking", False))
    min_time_to_move = body.get("min_time_to_move")
    goal_time = float(min_time_to_move) if min_time_to_move is not None else None
    state.hardware.set_gripper_position(SIDE, gripper_pos, goal_time=goal_time, blocking=blocking)
    return "Moved Gripper"


# ----------------------------------------------------------------------
# Reset / error recovery
# ----------------------------------------------------------------------
@app.route("/jointreset", methods=["POST"])
def jointreset():
    err = _require_connected()
    if err:
        return err
    state.hardware.reset()
    return "Reset Joint"


@app.route("/clearerr", methods=["POST"])
def clearerr():
    """Matches hil-serl: `/clearerr` runs a full reset, not a light error clear."""
    err = _require_connected()
    if err:
        return err
    state.hardware.reset()
    return "Clear"


@app.route("/reset", methods=["POST"])
def reset():
    err = _require_connected()
    if err:
        return err
    body = request.get_json(force=True, silent=True) or {}
    pose_to_reach = body.get("pose_to_reach")
    if pose_to_reach is not None:
        state.hardware.reset(pose_to_reach={SIDE: np.asarray(pose_to_reach, dtype=np.float32)})
    else:
        state.hardware.reset()
    return "Reset"


# ----------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------
def _cleanup():
    if state.hardware is not None:
        state.hardware.disconnect()


atexit.register(_cleanup)


def _signal_handler(signum, _frame):
    # `kill <pid>` (SIGTERM), terminal-panel/session kill (often SIGHUP), and
    # Ctrl+C (SIGINT) don't reliably trigger `atexit` on their own - handle
    # them explicitly so the follower arm is always released.
    logger.info("Received signal %s, disconnecting and shutting down...", signum)
    _cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _signal_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Station config name under trossen_real/configs/ (must be arm_mode: single)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5060)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--auto-connect", action="store_true", help="Connect to the follower immediately on startup instead of waiting for POST /connect_to_robot"
    )
    args = parser.parse_args()

    # Includes station config + port so multiple follower servers running in
    # parallel (e.g. two stations) get clearly separate log folders instead
    # of interleaving into one.
    setup_logging(f"follower_server_{args.config}_port{args.port}")

    config = load_station_config(args.config)
    if config.arm_mode != "single":
        raise ValueError(
            f"follower_single_server.py only supports arm_mode: single station configs for now "
            f"(got '{config.arm_mode}' from '{args.config}')."
        )
    state.config = config

    if args.auto_connect:
        state.hardware = TrossenFollowerSingle(config)
        state.hardware.connect()
        logger.info("Auto-connected to '%s'", config.station_name)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
