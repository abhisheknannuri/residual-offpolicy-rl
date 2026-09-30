"""Validate the hand-eye solvers against synthetic ground truth (and OpenCV 4.x if present).

Run (repo .venv only - never uv):
    PYTHONPATH=. .venv/bin/python trossen_real/calibration/tests/test_handeye.py
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.spatial.transform import Rotation

from trossen_real.calibration.handeye import (
    T_inv,
    calibrate_eye_in_hand,
    make_pairs,
    rotation_angle_deg,
    rt_to_T,
    solve_park,
    solve_tsai,
)

RNG = np.random.default_rng(0)


def _rand_T(rng, max_deg=180.0, t_scale=0.3):
    rv = rng.normal(size=3)
    rv = rv / np.linalg.norm(rv) * np.radians(rng.uniform(20, max_deg))
    return rt_to_T(Rotation.from_rotvec(rv).as_matrix(), rng.normal(scale=t_scale, size=3))


def make_scene(n=14, pos_noise_m=0.0, rot_noise_deg=0.0, seed=0):
    """Ground-truth eye-in-hand scene: fixed board, moving gripper-mounted camera."""
    rng = np.random.default_rng(seed)
    X_true = rt_to_T(  # camera ~5 cm out, 3 cm up, rotated off the gripper axis
        Rotation.from_euler("xyz", [12.0, -25.0, 95.0], degrees=True).as_matrix(),
        np.array([0.05, -0.02, 0.03]),
    )
    T_bt_true = rt_to_T(Rotation.from_euler("xyz", [175.0, 3.0, 40.0], degrees=True).as_matrix(),
                        np.array([0.45, 0.05, 0.10]))
    T_bg, T_ct = [], []
    for _ in range(n):
        g = _rand_T(rng, max_deg=70.0, t_scale=0.08)
        g[:3, 3] += np.array([0.25, 0.0, 0.25])
        ct = T_inv(X_true) @ T_inv(g) @ T_bt_true  # board seen from the camera
        if pos_noise_m or rot_noise_deg:
            dr = Rotation.from_rotvec(rng.normal(scale=np.radians(rot_noise_deg), size=3)).as_matrix()
            ct = ct.copy()
            ct[:3, :3] = dr @ ct[:3, :3]
            ct[:3, 3] += rng.normal(scale=pos_noise_m, size=3)
        T_bg.append(g)
        T_ct.append(ct)
    return X_true, T_bt_true, T_bg, T_ct


def _err(X, X_true):
    return (
        np.linalg.norm(X[:3, 3] - X_true[:3, 3]) * 1000,
        rotation_angle_deg(X[:3, :3].T @ X_true[:3, :3]),
    )


def test_noise_free_recovers_ground_truth():
    X_true, _, T_bg, T_ct = make_scene(n=12)
    for method in ("park", "tsai"):
        res = calibrate_eye_in_hand(T_bg, T_ct, method=method, refine=False)
        mm, deg = _err(res.T_gripper_cam, X_true)
        assert mm < 1e-6 and deg < 1e-6, f"{method}: {mm:.2e} mm, {deg:.2e} deg"
        print(f"  {method:5s} exact: {mm:.2e} mm, {deg:.2e} deg")


def test_recovers_board_pose_in_base():
    X_true, T_bt_true, T_bg, T_ct = make_scene(n=12)
    res = calibrate_eye_in_hand(T_bg, T_ct, method="park")
    mm, deg = _err(res.T_base_target, T_bt_true)
    assert mm < 1e-6 and deg < 1e-6, (mm, deg)
    assert res.target_spread_mm < 1e-6


def test_realistic_noise_is_accurate_and_refine_helps():
    """~0.5 mm / 0.3 deg board-pose noise is a realistic charuco+solvePnP level."""
    X_true, _, T_bg, T_ct = make_scene(n=20, pos_noise_m=0.0005, rot_noise_deg=0.3, seed=3)
    raw = calibrate_eye_in_hand(T_bg, T_ct, method="park", refine=False)
    ref = calibrate_eye_in_hand(T_bg, T_ct, method="park", refine=True)
    mm_raw, deg_raw = _err(raw.T_gripper_cam, X_true)
    mm_ref, deg_ref = _err(ref.T_gripper_cam, X_true)
    print(f"  noisy park       : {mm_raw:5.2f} mm, {deg_raw:5.3f} deg")
    print(f"  noisy park+refine: {mm_ref:5.2f} mm, {deg_ref:5.3f} deg")
    assert mm_ref < 5.0 and deg_ref < 1.0, (mm_ref, deg_ref)
    assert ref.target_spread_mm < 10.0


def test_degenerate_data_is_rejected_not_silently_wrong():
    """Poses differing only by translation cannot determine R_X - must raise."""
    _, _, T_bg, T_ct = make_scene(n=8)
    R0 = T_bg[0][:3, :3]
    for k in range(len(T_bg)):  # same orientation everywhere, only translation varies
        T_bg[k] = rt_to_T(R0, T_bg[k][:3, 3])
    try:
        calibrate_eye_in_hand(T_bg, T_ct, method="park")
    except ValueError as e:
        assert "pair" in str(e).lower()
        print(f"  rejected as expected: {str(e).splitlines()[0][:70]}...")
        return
    raise AssertionError("degenerate (rotation-free) data must be rejected")


def test_single_rotation_axis_is_flagged_by_residuals():
    """Rotating about ONE axis leaves a DOF free; residuals stay small but the
    answer is wrong - this is the classic silent failure, so document it."""
    rng = np.random.default_rng(5)
    X_true = rt_to_T(Rotation.from_euler("xyz", [10.0, -20.0, 80.0], degrees=True).as_matrix(),
                     np.array([0.04, -0.01, 0.06]))
    T_bt = rt_to_T(Rotation.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix(),
                   np.array([0.40, 0.0, 0.05]))
    T_bg, T_ct = [], []
    for _ in range(10):
        g = rt_to_T(Rotation.from_euler("z", rng.uniform(-60, 60), degrees=True).as_matrix(),
                    np.array([0.25, 0.0, 0.25]) + rng.normal(scale=0.05, size=3))
        T_bg.append(g)
        T_ct.append(T_inv(X_true) @ T_inv(g) @ T_bt)
    res = calibrate_eye_in_hand(T_bg, T_ct, method="park")
    mm, _ = _err(res.T_gripper_cam, X_true)
    print(f"  single-axis rotation -> {mm:6.1f} mm error while residual is "
          f"{res.rot_residual_deg:.4f} deg (looks fine, is not)")
    assert res.rot_residual_deg < 0.1  # residuals do NOT catch this


def test_pair_filter_drops_small_rotations():
    _, _, T_bg, T_ct = make_scene(n=10)
    a_all, _, _ = make_pairs(T_bg, T_ct, min_angle_deg=0.0)
    a_big, _, _ = make_pairs(T_bg, T_ct, min_angle_deg=45.0)
    assert len(a_big) < len(a_all)


def test_matches_opencv_calibratehandeye_if_available():
    """Cross-check against cv2.calibrateHandEye. The repo .venv has OpenCV 5,
    which dropped it; this runs only where 4.x is importable."""
    try:
        import cv2

        if not hasattr(cv2, "calibrateHandEye"):
            print(f"  SKIP: OpenCV {cv2.__version__} has no calibrateHandEye (expected in .venv)")
            return
    except ImportError:
        print("  SKIP: no cv2")
        return
    X_true, _, T_bg, T_ct = make_scene(n=14, pos_noise_m=0.0003, rot_noise_deg=0.2, seed=7)
    R_g2b = [T[:3, :3] for T in T_bg]
    t_g2b = [T[:3, 3].reshape(3, 1) for T in T_bg]
    R_t2c = [T[:3, :3] for T in T_ct]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in T_ct]
    R_cv, t_cv = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK)
    X_cv = rt_to_T(R_cv, t_cv.ravel())
    X_ours = calibrate_eye_in_hand(T_bg, T_ct, method="park", refine=False).T_gripper_cam
    mm, deg = _err(X_ours, X_cv)
    print(f"  ours vs cv2.calibrateHandEye(PARK): {mm:.3f} mm, {deg:.4f} deg apart")
    print(f"  (vs truth) ours {_err(X_ours, X_true)[0]:.2f} mm | cv2 {_err(X_cv, X_true)[0]:.2f} mm")
    assert mm < 1.0 and deg < 0.1


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
