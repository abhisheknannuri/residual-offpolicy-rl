"""Whole-pipeline test with no robot and no camera.

Builds a synthetic scene with a KNOWN camera-in-gripper transform, renders what
the wrist camera would see at each pose, runs the real detector, writes the real
captures JSON, invokes the real `solve` subcommand, and checks the recovered
transform against ground truth.

This is what catches convention bugs end to end: pose6 (axis-angle) -> matrix,
T_cam_target direction, and the A/B pairing inside the solver all have to agree
or the recovered transform is wrong.

Run (repo .venv only - never uv):
    PYTHONPATH=. .venv/bin/python trossen_real/calibration/tests/test_pipeline_end_to_end.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from trossen_real.calibration.calibrate_eye_in_hand import main as cli_main
from trossen_real.calibration.charuco import BoardDetector
from trossen_real.calibration.handeye import T_inv, rotation_angle_deg, rt_to_T
from trossen_real.calibration.tests.test_charuco import DIST, K, SPEC, _T, render_view

# Ground truth: camera 4 cm forward / 3 cm up on the wrist, rotated off the EE axis.
X_TRUE = rt_to_T(Rotation.from_euler("xyz", [8.0, -18.0, 92.0], degrees=True).as_matrix(),
                 np.array([0.042, -0.015, 0.031]))
# Board lying on the table in front of the robot, in the BASE frame.
T_BASE_TARGET = rt_to_T(Rotation.from_euler("xyz", [180.0, 0.0, 25.0], degrees=True).as_matrix(),
                        np.array([0.42, 0.06, 0.02]))

# Viewpoints expressed as board-in-camera poses, so the board is always framed.
# Deliberately spans several rotation axes - that is what makes R_X observable.
VIEWS = [
    ([0, 0, 0], [-0.14, -0.10, 0.42]),
    ([22, 0, 0], [-0.14, -0.12, 0.40]),
    ([-20, 0, 0], [-0.14, -0.08, 0.44]),
    ([0, 24, 0], [-0.17, -0.10, 0.41]),
    ([0, -24, 0], [-0.11, -0.10, 0.43]),
    ([0, 0, 30], [-0.14, -0.10, 0.45]),
    ([0, 0, -30], [-0.14, -0.10, 0.39]),
    ([18, 18, 12], [-0.15, -0.11, 0.46]),
    ([-18, -18, -12], [-0.13, -0.09, 0.38]),
    ([25, -12, 20], [-0.16, -0.10, 0.47]),
    ([-25, 12, -20], [-0.12, -0.10, 0.36]),
    ([14, -20, -25], [-0.15, -0.09, 0.43]),
]


def build_captures(noise_mm: float = 0.0, seed: int = 0):
    """Render each viewpoint, detect it, and return capture dicts + truth."""
    rng = np.random.default_rng(seed)
    det = BoardDetector(SPEC, K, DIST)
    captures = []
    for rpy, t in VIEWS:
        T_ct_true = _T(rpy, t)
        d = det.detect(render_view(det, T_ct_true))
        assert d.ok, f"view {rpy} not detected: {d.reason}"
        # Robot pose implied by the fixed board: T_bt = T_bg · X · T_ct
        T_bg = T_BASE_TARGET @ T_inv(T_ct_true) @ T_inv(X_TRUE)
        if noise_mm:  # imperfect FK
            T_bg = T_bg.copy()
            T_bg[:3, 3] += rng.normal(scale=noise_mm / 1000, size=3)
        pose6 = np.concatenate([T_bg[:3, 3], Rotation.from_matrix(T_bg[:3, :3]).as_rotvec()])
        captures.append({
            "pose6": pose6.tolist(),
            "q": [0.0] * 7,
            "T_cam_target": d.T_cam_target.tolist(),  # DETECTED, not the truth
            "n_corners": int(d.n_corners),
            "reproj_px": float(d.reproj_px),
        })
    return captures


def _err(X):
    return (
        np.linalg.norm(X[:3, 3] - X_TRUE[:3, 3]) * 1000,
        rotation_angle_deg(X[:3, :3].T @ X_TRUE[:3, :3]),
    )


def test_solve_subcommand_recovers_known_transform():
    captures = build_captures()
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "captures.json"
        out = Path(d) / "result.json"
        src.write_text(json.dumps({"station": "synthetic", "camera": "cam_right_wrist", "captures": captures}))
        assert cli_main(["solve", "--captures", str(src), "--out", str(out)]) == 0
        res = json.loads(out.read_text())
    mm, deg = _err(np.array(res["T_gripper_cam"]))
    print(f"  recovered: {mm:.2f} mm, {deg:.3f} deg from truth")
    print(f"  diagnostics: {json.dumps(res['diagnostics'], indent=None)}")
    assert mm < 2.0 and deg < 0.5, f"{mm:.2f} mm / {deg:.2f} deg - a convention is probably inverted"
    assert res["diagnostics"]["rotation_axis_spread_deg"] > 45
    assert len(res["captures"]) == len(captures)  # raw data kept for re-solving


def test_survives_realistic_fk_noise():
    captures = build_captures(noise_mm=0.5, seed=4)
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "c.json"
        out = Path(d) / "r.json"
        src.write_text(json.dumps({"captures": captures}))
        cli_main(["solve", "--captures", str(src), "--out", str(out)])
        res = json.loads(out.read_text())
    mm, deg = _err(np.array(res["T_gripper_cam"]))
    print(f"  with 0.5 mm FK noise: {mm:.2f} mm, {deg:.3f} deg")
    assert mm < 6.0 and deg < 1.0


def test_board_pose_in_base_is_recovered():
    captures = build_captures()
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "c.json"
        out = Path(d) / "r.json"
        src.write_text(json.dumps({"captures": captures}))
        cli_main(["solve", "--captures", str(src), "--out", str(out)])
        res = json.loads(out.read_text())
    mm = np.linalg.norm(np.array(res["T_base_target"])[:3, 3] - T_BASE_TARGET[:3, 3]) * 1000
    print(f"  board in base off by {mm:.2f} mm (spread {res['diagnostics']['target_spread_mm']:.2f} mm)")
    assert mm < 3.0


def test_saved_fields_are_consistent_with_each_other():
    """quaternion / rotvec / rpy in the output must all describe the same rotation."""
    captures = build_captures()
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "c.json"
        out = Path(d) / "r.json"
        src.write_text(json.dumps({"captures": captures}))
        cli_main(["solve", "--captures", str(src), "--out", str(out)])
        res = json.loads(out.read_text())
    R = np.array(res["T_gripper_cam"])[:3, :3]
    for key, rot in (
        ("quaternion_xyzw", Rotation.from_quat(res["quaternion_xyzw"])),
        ("rotvec_axis_angle", Rotation.from_rotvec(res["rotvec_axis_angle"])),
        ("rpy_deg_xyz_extrinsic", Rotation.from_euler("xyz", res["rpy_deg_xyz_extrinsic"], degrees=True)),
    ):
        assert rotation_angle_deg(rot.as_matrix().T @ R) < 1e-6, key
    assert np.allclose(res["translation_m"], np.array(res["T_gripper_cam"])[:3, 3])


def test_wrong_pose_convention_is_detectable():
    """If the EE rotation were read as euler instead of axis-angle, the fit blows
    up - i.e. the diagnostics would catch that class of mistake."""
    captures = build_captures()
    for c in captures:  # corrupt: treat the rotvec as if it were euler xyz
        p = np.array(c["pose6"])
        c["pose6"] = np.concatenate([p[:3], Rotation.from_euler("xyz", p[3:]).as_rotvec()]).tolist()
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "c.json"
        out = Path(d) / "r.json"
        src.write_text(json.dumps({"captures": captures}))
        cli_main(["solve", "--captures", str(src), "--out", str(out)])
        res = json.loads(out.read_text())
    spread = res["diagnostics"]["target_spread_mm"]
    mm, _ = _err(np.array(res["T_gripper_cam"]))
    print(f"  corrupted convention -> {mm:.0f} mm error, board spread {spread:.1f} mm (flagged as bad)")
    assert spread > 5.0, "a wrong rotation convention must show up in the board spread"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            print(f"- {fn.__name__}")
            fn()
            print("  PASS")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
