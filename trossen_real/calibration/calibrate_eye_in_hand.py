#!/usr/bin/env python3
"""Eye-in-hand calibration for a wrist-mounted RealSense on a Trossen arm.

Solves for **T_gripper_cam**: where the camera sits in the gripper/EE frame.

It talks to the station's existing `follower_single_server.py` over HTTP for arm
poses (no ROS 2, no interbotix - this repo drives the arm with the `trossen_arm`
SDK) and opens the RealSense directly by serial number from the station config.

    Board pose  (charuco + solvePnP)   ->  T_cam_target
    Arm pose    (follower /getstate)   ->  T_base_gripper
    Solver      (handeye.py, AX = XB)  ->  T_gripper_cam

Run everything with the repo venv's interpreter directly - never `uv`:

    # 0. one-time: print a board whose geometry you know exactly
    .venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand print-board --out board.png

    # 1. start the follower server for this station (its own terminal)
    .venv/bin/python -m trossen_real.follower.follower_single_server \
        --config trossen_station1_single --port 5095

    # 2. calibrate (nothing else may hold the camera - stop teleop/infer apps)
    .venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand run \
        --config trossen_station1_single --camera cam_right_wrist

    # 3. later: re-solve from saved captures, or check a saved result live
    .venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand solve   --captures <file.json>
    .venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand verify  --result  <file.json>

Moving the arm between captures (`--motion`):
  manual  (default) you move it, from anywhere - the teleop app, or curl:
              curl -X POST http://127.0.0.1:5095/move_to_joint_positions \
                   -H 'Content-Type: application/json' \
                   -d '{"arr":[0,0,0,0,0,0,0.02],"blocking":true,"min_time_to_move":3.0}'
            then press SPACE here to capture. The arm is stiff (the server has no
            freedrive route), so it cannot be pushed by hand.
  guided    this script commands a preset sweep of small joint offsets around
            wherever the arm starts, one at a time, and captures at each.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from trossen_real.calibration.charuco import (
    BoardDetector,
    BoardSpec,
    calibrate_intrinsics,
    intrinsics_from_realsense,
)
from trossen_real.calibration.handeye import (
    T_inv,
    calibrate_eye_in_hand,
    pose6_to_T,
    rotation_angle_deg,
)
from trossen_real.config import load_station_config
from trossen_real.teleop.follower_client import FollowerClient

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Joint offsets (rad) applied to the starting pose in --motion guided. Wrist
# joints dominate: rotating the wrist swings the camera through different
# rotation axes (which is what makes R_X observable) while keeping the board in
# view. Kept small and slow - the arm moves blind, with the board in front of it.
GUIDED_OFFSETS = [
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, +0.35],
    [0.00, 0.00, 0.00, 0.00, 0.00, -0.35],
    [0.00, 0.00, 0.00, 0.00, +0.25, 0.00],
    [0.00, 0.00, 0.00, 0.00, -0.25, 0.00],
    [0.00, 0.00, 0.00, +0.25, 0.00, +0.25],
    [0.00, 0.00, 0.00, -0.25, 0.00, -0.25],
    [0.00, -0.10, +0.10, 0.00, +0.20, +0.20],
    [0.00, +0.10, -0.10, 0.00, -0.20, -0.20],
    [+0.12, 0.00, 0.00, 0.00, +0.15, -0.20],
    [-0.12, 0.00, 0.00, 0.00, -0.15, +0.20],
    [+0.08, -0.08, +0.08, +0.20, +0.20, 0.00],
    [-0.08, +0.08, -0.08, -0.20, -0.20, 0.00],
    [0.00, 0.00, 0.00, +0.30, -0.15, +0.30],
    [0.00, 0.00, 0.00, -0.30, +0.15, -0.30],
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _camera_serial(station, name: str) -> str:
    names = [d.name for d in station.cameras.devices]
    for dev in station.cameras.devices:
        if dev.name == name:
            if not dev.serial:
                raise SystemExit(f"Camera '{name}' has no serial in the station config (mock slot). Cameras: {names}")
            return dev.serial
    raise SystemExit(f"Camera '{name}' not found in station config. Available: {names}")


def _open_camera(serial: str, width: int, height: int, fps: int):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    try:
        profile = pipeline.start(cfg)
    except RuntimeError as e:
        raise SystemExit(
            f"Could not open RealSense {serial} at {width}x{height}@{fps}: {e}\n"
            "Another process (teleop app / infer app / dataset recorder) probably holds it - "
            "stop it and retry. Or pick a supported profile with --width/--height/--fps."
        ) from e
    for _ in range(15):  # let auto-exposure settle
        pipeline.wait_for_frames(timeout_ms=2000)
    return pipeline, profile


def _grab(pipeline, timeout_ms: int = 2000) -> np.ndarray:
    frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError("no colour frame from the RealSense")
    return np.asanyarray(color.get_data())


def _rotation_axis_spread(T_list) -> float:
    """Max angle between the rotation axes of all pose pairs, in degrees.

    This is the coverage metric that matters: rotating only about one axis leaves
    the calibration under-determined *and the residuals still look perfect*, so
    the spread has to be checked directly rather than inferred from the fit.
    """
    axes = []
    for i in range(len(T_list)):
        for j in range(i + 1, len(T_list)):
            rv = Rotation.from_matrix(T_inv(T_list[j])[:3, :3] @ T_list[i][:3, :3]).as_rotvec()
            n = np.linalg.norm(rv)
            if np.degrees(n) > 5.0:
                axes.append(rv / n)
    if len(axes) < 2:
        return 0.0
    axes = np.array(axes)
    dots = np.clip(np.abs(axes @ axes.T), -1, 1)
    return float(np.degrees(np.arccos(dots.min())))


def _quality_report(res, captures) -> list[str]:
    """Plain-language warnings. Residuals alone do NOT prove a good calibration."""
    msgs = []
    spread = _rotation_axis_spread([pose6_to_T(c["pose6"]) for c in captures])
    msgs.append(f"rotation-axis spread: {spread:.0f} deg (want > 45; this is NOT visible in the residuals)")
    if spread < 45:
        msgs.append("  !! Too little rotation variety. The result can be badly wrong while every "
                    "residual looks small. Recapture with the wrist rotated about clearly different axes.")
    if res.target_spread_mm > 5:
        msgs.append(f"  !! board-in-base spread {res.target_spread_mm:.1f} mm is high (want < ~3 mm). "
                    "Causes: board moved during capture, wrong square size, blurry frames, or FK error.")
    if res.rot_residual_deg > 1.0:
        msgs.append(f"  !! AX=XB rotation residual {res.rot_residual_deg:.2f} deg is high (want < ~0.5).")
    reproj = np.array([c["reproj_px"] for c in captures])
    msgs.append(f"board reprojection: mean {reproj.mean():.2f} px, max {reproj.max():.2f} px (want < ~1)")
    dists = np.array([np.linalg.norm(np.array(c["T_cam_target"])[:3, 3]) for c in captures])
    msgs.append(f"board distance: {dists.min() * 100:.0f}-{dists.max() * 100:.0f} cm")
    return msgs


def _save(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _result_payload(res, captures, meta) -> dict:
    X = res.T_gripper_cam
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        **meta,
        "T_gripper_cam": X.tolist(),
        "translation_m": X[:3, 3].tolist(),
        "quaternion_xyzw": Rotation.from_matrix(X[:3, :3]).as_quat().tolist(),
        "rotvec_axis_angle": Rotation.from_matrix(X[:3, :3]).as_rotvec().tolist(),
        "rpy_deg_xyz_extrinsic": Rotation.from_matrix(X[:3, :3]).as_euler("xyz", degrees=True).tolist(),
        "T_base_target": res.T_base_target.tolist(),
        "diagnostics": {
            "method": res.method,
            "n_samples": res.n_samples,
            "n_pairs": res.n_pairs,
            "rot_residual_deg": res.rot_residual_deg,
            "trans_residual_mm": res.trans_residual_mm,
            "target_spread_mm": res.target_spread_mm,
            "target_spread_deg": res.target_spread_deg,
            "rotation_axis_spread_deg": _rotation_axis_spread([pose6_to_T(c["pose6"]) for c in captures]),
        },
        "captures": captures,  # keep raw data so `solve` can re-run without the robot
    }


def _solve_and_report(captures, meta, method, min_angle_deg, refine) -> dict:
    T_bg = [pose6_to_T(c["pose6"]) for c in captures]
    T_ct = [np.array(c["T_cam_target"]) for c in captures]
    res = calibrate_eye_in_hand(T_bg, T_ct, method=method, min_angle_deg=min_angle_deg, refine=refine)
    print("\n" + "=" * 78)
    print(res.summary())
    print("-" * 78)
    for line in _quality_report(res, captures):
        print("  " + line)
    # cross-check with the other closed form: a large disagreement means the data
    # is under-constrained, whatever the residuals say
    other = "tsai" if method == "park" else "park"
    try:
        alt = calibrate_eye_in_hand(T_bg, T_ct, method=other, min_angle_deg=min_angle_deg, refine=refine)
        d_mm = np.linalg.norm(alt.T_gripper_cam[:3, 3] - res.T_gripper_cam[:3, 3]) * 1000
        d_deg = rotation_angle_deg(alt.T_gripper_cam[:3, :3].T @ res.T_gripper_cam[:3, :3])
        flag = "" if (d_mm < 5 and d_deg < 1) else "   !! large - data is under-constrained"
        print(f"  {method} vs {other}: {d_mm:.2f} mm, {d_deg:.3f} deg apart{flag}")
    except ValueError:
        pass
    print("=" * 78 + "\n")
    return _result_payload(res, captures, meta)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def cmd_print_board(args) -> int:
    spec = BoardSpec(args.squares_x, args.squares_y, args.square_len / 1000, args.marker_len / 1000, args.dictionary)
    det = BoardDetector(spec, np.eye(3), np.zeros(5))
    img = det.render(px_per_m=args.dpi / 0.0254, margin_px=args.margin_px)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"wrote {out}  ({img.shape[1]}x{img.shape[0]} px at {args.dpi} dpi)")
    print(f"board: {spec.describe()}")
    print("\nPRINT AT 100% SCALE (no 'fit to page'), then MEASURE a square with calipers and pass the")
    print("measured value as --square-len. A 2% print scale error becomes a 2% distance bias that")
    print("reprojection error will NOT reveal. Tape it FLAT to a rigid surface; any bow adds error.")
    return 0


def cmd_solve(args) -> int:
    data = json.loads(Path(args.captures).read_text())
    captures = data["captures"] if "captures" in data else data
    meta = {k: v for k, v in data.items() if k not in ("captures", "diagnostics")}
    payload = _solve_and_report(captures, meta, args.method, args.min_angle_deg, not args.no_refine)
    out = Path(args.out) if args.out else Path(args.captures).with_name(Path(args.captures).stem + "_resolved.json")
    print(f"saved: {_save(out, payload)}")
    return 0


def cmd_verify(args) -> int:
    """Live check: how well does a saved calibration predict the board's place in the base frame?"""
    data = json.loads(Path(args.result).read_text())
    X = np.array(data["T_gripper_cam"])
    station = load_station_config(data.get("station", args.config))
    spec = BoardSpec(**data["board"])
    serial = data.get("camera_serial") or _camera_serial(station, data.get("camera", args.camera))

    client = FollowerClient(data.get("server_url") or station.follower_server_url)
    client.connect()
    pipeline, profile = _open_camera(serial, args.width, args.height, args.fps)
    K, dist, _ = intrinsics_from_realsense(profile)
    if "intrinsics" in data and args.use_saved_intrinsics:
        K, dist = np.array(data["intrinsics"]["K"]), np.array(data["intrinsics"]["dist"])
    det = BoardDetector(spec, K, dist, min_corners=args.min_corners)

    print("Move the arm around. Each detection re-estimates the board pose in the BASE frame;")
    print("with a good calibration it stays put. SPACE = sample, q = quit.\n")
    samples = []
    try:
        if args.motion == "replay":
            captures, raw_corners, raw_ids = _replay_capture(args, client, pipeline, det)
            if len(captures) < args.min_captures:
                raise SystemExit(
                    f"only {len(captures)} usable captures (need {args.min_captures}). "
                    "Re-record poses where the board is clearly visible."
                )
        while args.motion != "replay":
            img = _grab(pipeline)
            d = det.detect(img)
            view = det.draw(img, d)
            if samples:
                p = np.array(samples)
                spread = np.linalg.norm(p - p.mean(axis=0), axis=1).std() * 1000
                cv2.putText(view, f"n={len(samples)}  spread={spread:.1f} mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("verify (SPACE=sample, q=quit)", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and d.ok:
                T_bt = pose6_to_T(client.get_pose()) @ X @ d.T_cam_target
                samples.append(T_bt[:3, 3])
                print(f"  sample {len(samples)}: board at base xyz = "
                      f"{np.round(T_bt[:3, 3] * 1000, 1)} mm")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    if len(samples) >= 2:
        p = np.array(samples)
        print(f"\nboard position spread over {len(p)} views: {np.linalg.norm(p - p.mean(axis=0), axis=1).std() * 1000:.2f} mm")
        print("(this is the end-to-end accuracy you will actually get)")
    return 0


def _replay_capture(args, client, pipeline, det):
    """Phase 2: drive the arm to each recorded pose and capture there.

    Joint-space replay (not EE/IK), so the arm reproduces exactly the
    configuration you teleoped to. Moves are blocking and slow; the arm travels
    between viewpoints with the board in front of it, so keep the workspace
    clear and stay near the e-stop.
    """
    data = json.loads(Path(args.poses).read_text())
    poses = data["poses"]
    print(f"replaying {len(poses)} recorded poses from {args.poses}")
    print(f"  recorded on station '{data.get('station')}' | axis spread "
          f"{data.get('rotation_axis_spread_deg', float('nan')):.0f} deg")
    if data.get("station") and data["station"] != args.config:
        print(f"  !! recorded on station '{data['station']}' but running with '{args.config}'")
    if not args.yes:
        print(f"\nThe arm will move to each pose at {args.goal_time:.1f} s per move. Clear the workspace.")
        if input("Type 'go' to start: ").strip().lower() != "go":
            raise SystemExit("aborted")

    captures, raw_corners, raw_ids = [], [], []
    for i, p in enumerate(poses, 1):
        target = np.concatenate([np.array(p["q"][:6], dtype=float), [float(p.get("gripper", p["q"][6] if len(p["q"]) > 6 else 0.0))]])
        print(f"\n[{i}/{len(poses)}] moving to q(deg) = {np.round(np.degrees(target[:6]), 1)}")
        try:
            client.move_to_joint_positions(target, blocking=True, min_time_to_move=args.goal_time)
        except Exception as e:  # noqa: BLE001
            print(f"  move FAILED ({e}); skipping. If the arm faulted: POST /clearerr then /jointreset.")
            continue
        time.sleep(args.settle_s)

        best = None
        for _ in range(max(1, args.frames_per_pose)):  # a few frames, keep the sharpest
            d = det.detect(_grab(pipeline))
            if d.ok and (best is None or d.reproj_px < best.reproj_px):
                best = d
        if best is None:
            print("  no board detected here - skipped (recorded pose may not see the board)")
            continue
        if best.reproj_px > args.max_reproj_px:
            print(f"  reprojection {best.reproj_px:.2f} px > {args.max_reproj_px} - skipped (blurry?)")
            continue

        pose6 = np.asarray(client.get_pose(), dtype=float)  # read FK AFTER settling, never the commanded value
        captures.append({
            "pose6": pose6.tolist(),
            "q": [float(x) for x in client.get_state()["q"]],
            "T_cam_target": best.T_cam_target.tolist(),
            "n_corners": int(best.n_corners),
            "reproj_px": float(best.reproj_px),
            "replay_index": i - 1,
        })
        raw_corners.append(best.corners)
        raw_ids.append(best.ids)
        if args.save_images:
            path = RESULTS_DIR / "images" / f"{args.camera}_replay_{i:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), det.draw(_grab(pipeline), best))
        sp = _rotation_axis_spread([pose6_to_T(c["pose6"]) for c in captures]) if len(captures) > 2 else 0.0
        print(f"  captured ({len(captures)} total): {best.n_corners} corners, "
              f"reproj {best.reproj_px:.2f} px, axis spread {sp:.0f} deg")

    print(f"\nreplay done: {len(captures)}/{len(poses)} poses captured")
    return captures, raw_corners, raw_ids


def cmd_record_poses(args) -> int:
    """Phase 1: teleop the arm by hand and bookmark good viewpoints.

    Uses NO camera, so the teleop app can keep the RealSense open while you do
    this. Recording joint positions (not just the EE pose) means the replay can
    command exactly the same configuration back, with no IK involved.
    """
    station = load_station_config(args.config)
    server_url = args.server_url or station.follower_server_url
    client = FollowerClient(server_url)
    if not client.check():
        raise SystemExit(f"Follower server not reachable at {server_url}")
    client.connect()

    print(f"Recording poses from {server_url}\n")
    print("Teleop the arm (leader arm / the teleop app / curl) so the WRIST CAMERA sees the board.")
    print("For a good calibration the set of poses must include:")
    print("  - the wrist ROTATED about several different axes (roll, pitch, yaw), not just moved around")
    print("  - >= 30 deg of rotation between poses, and a few different distances/angles to the board")
    print("  - the board fully visible and reasonably large in view at EVERY pose\n")
    print("ENTER = record | u + ENTER = undo | q + ENTER = finish and save\n")

    poses: list[dict] = []
    while True:
        try:
            cmd = input(f"[{len(poses)} recorded] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd == "q":
            break
        if cmd == "u":
            if poses:
                poses.pop()
                print(f"  undone -> {len(poses)}")
            continue
        if cmd:
            continue
        st = client.get_state()
        entry = {
            "q": [float(x) for x in st["q"]],
            "gripper": float(st["gripper_pos"]),
            "pose6": [float(x) for x in st["pose"]],
        }
        poses.append(entry)
        spread = _rotation_axis_spread([pose6_to_T(p["pose6"]) for p in poses]) if len(poses) > 2 else 0.0
        print(f"  recorded {len(poses)}: q(deg) = {np.round(np.degrees(entry['q'][:6]), 1)}"
              + (f"   axis spread {spread:.0f} deg {'(good)' if spread > 45 else '(ROTATE MORE)'}"
                 if len(poses) > 2 else ""))

    if len(poses) < 3:
        print("nothing saved (need at least 3 poses)")
        return 1
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out) if args.out else RESULTS_DIR / f"poses_{args.config}_{stamp}.json"
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "station": args.config,
        "server_url": server_url,
        "rotation_axis_spread_deg": _rotation_axis_spread([pose6_to_T(p["pose6"]) for p in poses]),
        "poses": poses,
    }
    print(f"\nsaved {len(poses)} poses: {_save(out, payload)}")
    print(f"axis spread: {payload['rotation_axis_spread_deg']:.0f} deg"
          + ("" if payload["rotation_axis_spread_deg"] > 45 else "   !! low - consider recording more rotation"))
    print("\nNext: stop the teleop app (it holds the camera), then\n"
          f"  .venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand run \\\n"
          f"      --config {args.config} --camera {args.camera} --motion replay --poses {out}")
    return 0


def cmd_run(args) -> int:
    station = load_station_config(args.config)
    serial = args.serial or _camera_serial(station, args.camera)
    server_url = args.server_url or station.follower_server_url
    spec = BoardSpec(args.squares_x, args.squares_y, args.square_len / 1000, args.marker_len / 1000, args.dictionary)

    print(f"station     : {args.config}")
    print(f"camera      : {args.camera} (serial {serial}) @ {args.width}x{args.height}")
    print(f"follower    : {server_url}")
    print(f"board       : {spec.describe()}")
    print(f"motion      : {args.motion}\n")

    client = FollowerClient(server_url)
    if not client.check():
        raise SystemExit(
            f"Follower server not reachable at {server_url}. Start it first:\n"
            f"  .venv/bin/python -m trossen_real.follower.follower_single_server "
            f"--config {args.config} --port {server_url.rsplit(':', 1)[-1]}"
        )
    client.connect()
    pipeline, profile = _open_camera(serial, args.width, args.height, args.fps)
    K, dist, cam_meta = intrinsics_from_realsense(profile)
    print(f"intrinsics  : fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} ({cam_meta['source']})\n")
    det = BoardDetector(spec, K, dist, min_corners=args.min_corners)

    captures: list[dict] = []
    raw_corners: list = []
    raw_ids: list = []
    guided_idx = 0
    q_start = None
    gui = not args.no_gui

    print("SPACE capture | u undo | g next guided pose | c solve+save | q quit\n"
          "Aim for >= 12 captures with the wrist rotated about SEVERAL different axes,\n"
          "board filling a good part of the frame, at a few different distances.\n")
    try:
        if args.motion == "replay":
            captures, raw_corners, raw_ids = _replay_capture(args, client, pipeline, det)
            if len(captures) < args.min_captures:
                raise SystemExit(
                    f"only {len(captures)} usable captures (need {args.min_captures}). "
                    "Re-record poses where the board is clearly visible."
                )
        while args.motion != "replay":
            img = _grab(pipeline)
            d = det.detect(img)
            key = -1
            if gui:
                view = det.draw(img, d)
                status = (f"n={len(captures)}  {'OK ' + str(d.n_corners) + ' corners' if d.ok else d.reason}"
                          f"{f'  reproj={d.reproj_px:.2f}px' if d.ok else ''}")
                cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if d.ok else (0, 165, 255), 2)
                if len(captures) >= 3:
                    sp = _rotation_axis_spread([pose6_to_T(c["pose6"]) for c in captures])
                    cv2.putText(view, f"axis spread {sp:.0f} deg {'(good)' if sp > 45 else '(ROTATE MORE)'}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if sp > 45 else (0, 0, 255), 2)
                cv2.imshow("eye-in-hand capture", view)
                key = cv2.waitKey(1) & 0xFF
            else:
                print(f"\r n={len(captures)} {'OK' if d.ok else d.reason}          ", end="", flush=True)
                time.sleep(0.05)

            if args.motion == "guided" and (key == ord("g") or (args.auto and d.ok)):
                if q_start is None:
                    q_start = np.array(client.get_state()["q"], dtype=float)
                if guided_idx >= len(GUIDED_OFFSETS):
                    print("\nguided sweep finished - press c to solve")
                else:
                    off = np.array(GUIDED_OFFSETS[guided_idx], dtype=float)
                    target = q_start.copy()
                    target[:6] += off
                    print(f"\n[guided {guided_idx + 1}/{len(GUIDED_OFFSETS)}] joint offsets (deg): "
                          f"{np.round(np.degrees(off), 1)}")
                    client.move_to_joint_positions(
                        np.concatenate([target[:6], [client.get_gripper()]]),
                        blocking=True, min_time_to_move=args.goal_time,
                    )
                    time.sleep(args.settle_s)
                    guided_idx += 1
                    key = 32  # capture at the new pose

            if key == 32:  # SPACE
                img2 = _grab(pipeline)
                d2 = det.detect(img2)
                if not d2.ok:
                    print(f"\n  not captured: {d2.reason}")
                    continue
                if d2.reproj_px > args.max_reproj_px:
                    print(f"\n  not captured: reprojection {d2.reproj_px:.2f} px > {args.max_reproj_px}")
                    continue
                pose6 = np.asarray(client.get_pose(), dtype=float)
                captures.append({
                    "pose6": pose6.tolist(),
                    "q": [float(x) for x in client.get_state()["q"]],
                    "T_cam_target": d2.T_cam_target.tolist(),
                    "n_corners": int(d2.n_corners),
                    "reproj_px": float(d2.reproj_px),
                })
                raw_corners.append(d2.corners)
                raw_ids.append(d2.ids)
                if args.save_images:
                    p = RESULTS_DIR / "images" / f"{args.camera}_{len(captures):03d}.png"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(p), img2)
                sp = _rotation_axis_spread([pose6_to_T(c["pose6"]) for c in captures]) if len(captures) > 2 else 0
                print(f"\n  captured {len(captures)}: {d2.n_corners} corners, reproj {d2.reproj_px:.2f} px, "
                      f"axis spread {sp:.0f} deg")

            elif key == ord("u") and captures:
                captures.pop()
                raw_corners.pop()
                raw_ids.pop()
                print(f"\n  undone -> {len(captures)} captures")

            elif key in (ord("c"), 13):
                if len(captures) < args.min_captures:
                    print(f"\n  need at least {args.min_captures} captures, have {len(captures)}")
                    continue
                break

            elif key == ord("q"):
                print("\nquit without solving")
                return 1
    finally:
        pipeline.stop()
        if gui:
            cv2.destroyAllWindows()

    intr = {"K": K.tolist(), "dist": dist.tolist(), **cam_meta}
    if args.calibrate_intrinsics:
        try:
            rms, K2, dist2, n = calibrate_intrinsics(raw_corners, raw_ids, det.board, (args.width, args.height))
            print(f"intrinsics from {n} captured views: rms={rms:.3f} px, fx {K[0, 0]:.1f} -> {K2[0, 0]:.1f}")
            print("NOTE: the board poses above were solved with the FACTORY intrinsics. These refined "
                  "ones are recorded for reference only - re-estimating poses needs the original frames "
                  "(--save-images).")
            intr = {"K": K2.tolist(), "dist": dist2.tolist(), "source": "charuco_selfcalib", "rms_px": rms,
                    "factory_K": K.tolist(), "applies_to_poses": False}
        except ValueError as e:
            print(f"intrinsics self-calibration skipped: {e}")

    meta = {
        "station": args.config,
        "camera": args.camera,
        "camera_serial": serial,
        "server_url": server_url,
        "board": {
            "squares_x": spec.squares_x, "squares_y": spec.squares_y,
            "square_len_m": spec.square_len_m, "marker_len_m": spec.marker_len_m,
            "dictionary": spec.dictionary, "legacy_pattern": spec.legacy_pattern,
        },
        "intrinsics": intr,
        "frame_convention": "T_gripper_cam maps a point in CAMERA coords to GRIPPER/EE coords; "
                            "the arm's EE pose is (x,y,z,axis-angle) in the BASE frame.",
    }
    payload = _solve_and_report(captures, meta, args.method, args.min_angle_deg, not args.no_refine)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.config}_{args.camera}_{stamp}.json"
    print(f"saved: {_save(out, payload)}")
    print(f"       (contains the raw captures - re-solve any time with:  solve --captures {out})")
    return 0


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_board(p):
        p.add_argument("--squares-x", type=int, default=7, help="chessboard squares across")
        p.add_argument("--squares-y", type=int, default=5, help="chessboard squares down")
        p.add_argument("--square-len", type=float, default=40.0, help="square side in MM (measure the printout)")
        p.add_argument("--marker-len", type=float, default=30.0, help="aruco marker side in MM")
        p.add_argument("--dictionary", default="5x5_100")

    def add_cam(p):
        p.add_argument("--config", default="trossen_station1_single")
        p.add_argument("--camera", default="cam_right_wrist")
        p.add_argument("--serial", default=None, help="override the serial from the station config")
        p.add_argument("--server-url", default=None)
        p.add_argument("--width", type=int, default=1280)
        p.add_argument("--height", type=int, default=720)
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--min-corners", type=int, default=8)

    def add_solve(p):
        p.add_argument("--method", choices=["park", "tsai"], default="park")
        p.add_argument("--min-angle-deg", type=float, default=5.0)
        p.add_argument("--no-refine", action="store_true")
        p.add_argument("--out", default=None)

    p = sub.add_parser("print-board", help="render a printable charuco board")
    add_board(p)
    p.add_argument("--out", default="charuco_board.png")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--margin-px", type=int, default=60)
    p.set_defaults(func=cmd_print_board)

    p = sub.add_parser("run", help="capture poses and solve (needs robot + camera)")
    add_board(p)
    add_cam(p)
    add_solve(p)
    p.add_argument("--motion", choices=["manual", "guided", "replay"], default="manual")
    p.add_argument("--poses", default=None, help="replay: JSON from `record-poses`")
    p.add_argument("--frames-per-pose", type=int, default=3, help="replay: frames to try per pose, keep the sharpest")
    p.add_argument("--yes", action="store_true", help="replay: skip the 'the arm will move' confirmation")
    p.add_argument("--auto", action="store_true", help="guided: advance automatically instead of pressing g")
    p.add_argument("--goal-time", type=float, default=3.0, help="guided: seconds per move (slow = safe)")
    p.add_argument("--settle-s", type=float, default=1.0, help="guided: settle time before capturing")
    p.add_argument("--min-captures", type=int, default=10)
    p.add_argument("--max-reproj-px", type=float, default=1.5)
    p.add_argument("--calibrate-intrinsics", action="store_true")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--no-gui", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("record-poses", help="phase 1: teleop and bookmark viewpoints (no camera needed)")
    p.add_argument("--config", default="trossen_station1_single")
    p.add_argument("--camera", default="cam_right_wrist", help="only used to print the next command")
    p.add_argument("--server-url", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_record_poses)

    p = sub.add_parser("solve", help="re-solve from a saved captures/result file")
    add_solve(p)
    p.add_argument("--captures", required=True)
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("verify", help="live check of a saved calibration")
    add_cam(p)
    p.add_argument("--result", required=True)
    p.add_argument("--use-saved-intrinsics", action="store_true")
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    if getattr(args, "motion", None) == "replay" and not args.poses:
        ap.error("--motion replay needs --poses <file from `record-poses`>")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
