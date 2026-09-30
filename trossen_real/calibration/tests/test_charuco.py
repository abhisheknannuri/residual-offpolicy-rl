"""Board detection + pose tests, with no camera and no robot.

A synthetic camera view is built by rendering the board and warping it with the
homography implied by a KNOWN board pose, so the recovered pose can be compared
against ground truth. This is what pins down the T_cam_target convention that
the hand-eye solver depends on.

Run (repo .venv only - never uv):
    PYTHONPATH=. .venv/bin/python trossen_real/calibration/tests/test_charuco.py
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from trossen_real.calibration.charuco import BoardDetector, BoardSpec, calibrate_intrinsics

W, H = 1280, 720
K = np.array([[900.0, 0.0, W / 2], [0.0, 900.0, H / 2], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)
SPEC = BoardSpec(squares_x=7, squares_y=5, square_len_m=0.040, marker_len_m=0.030, dictionary="5x5_100")


def _T(rpy_deg, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


def render_view(det: BoardDetector, T_cam_target: np.ndarray, px_per_m: int = 3000) -> np.ndarray:
    """Photograph the board, synthetically, from a known pose.

    The board frame has its origin at the board's top-left corner, +x right,
    +y down, z = 0 in the board plane (verified from board.getChessboardCorners()),
    and the rendered image maps pixel (u, v) -> board (u / px_per_m, v / px_per_m).
    """
    board_img = det.render(px_per_m=px_per_m, margin_px=0)
    h_px, w_px = board_img.shape[:2]
    src = np.float32([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]])
    corners_m = np.float32(
        [[0, 0, 0], [w_px / px_per_m, 0, 0], [w_px / px_per_m, h_px / px_per_m, 0], [0, h_px / px_per_m, 0]]
    )
    rvec = cv2.Rodrigues(T_cam_target[:3, :3])[0]
    dst, _ = cv2.projectPoints(corners_m, rvec, T_cam_target[:3, 3], K, DIST)
    warped = cv2.warpPerspective(
        board_img, cv2.getPerspectiveTransform(src, dst.reshape(-1, 2).astype(np.float32)), (W, H),
        borderValue=128,
    )
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def test_pose_convention_is_cam_from_target():
    """estimate_pose must return T_cam_target (board -> camera), not its inverse."""
    det = BoardDetector(SPEC, K, DIST)
    for rpy, t in (([0, 0, 0], [-0.14, -0.10, 0.45]),
                   ([18, -22, 9], [-0.10, -0.12, 0.40]),
                   ([-25, 14, -30], [-0.16, -0.08, 0.55])):
        T_true = _T(rpy, t)
        d = det.detect(render_view(det, T_true))
        assert d.ok, d.reason
        mm = np.linalg.norm(d.T_cam_target[:3, 3] - T_true[:3, 3]) * 1000
        deg = np.degrees(np.linalg.norm(Rotation.from_matrix(d.T_cam_target[:3, :3].T @ T_true[:3, :3]).as_rotvec()))
        print(f"  rpy={rpy} -> {d.n_corners:2d} corners, {mm:5.2f} mm, {deg:5.3f} deg, reproj {d.reproj_px:.3f} px")
        assert mm < 3.0 and deg < 1.0, f"pose off by {mm:.2f} mm / {deg:.2f} deg - convention may be inverted"
        # the inverse must NOT also match, otherwise the test proves nothing
        assert np.linalg.norm(np.linalg.inv(d.T_cam_target)[:3, 3] - T_true[:3, 3]) * 1000 > 50


def test_board_in_front_of_camera_has_positive_z():
    det = BoardDetector(SPEC, K, DIST)
    d = det.detect(render_view(det, _T([10, 10, 0], [-0.14, -0.10, 0.5])))
    assert d.ok and d.T_cam_target[2, 3] > 0


def test_partial_view_still_solves_and_blank_is_rejected():
    det = BoardDetector(SPEC, K, DIST)
    img = render_view(det, _T([0, 0, 0], [-0.14, -0.10, 0.45]))
    img[:, : W // 3] = 128  # occlude a third of the board
    d = det.detect(img)
    print(f"  partial view: ok={d.ok} corners={d.n_corners} reproj={d.reproj_px:.3f}")
    assert d.ok and d.n_corners < SPEC.n_corners
    blank = det.detect(np.full((H, W, 3), 128, np.uint8))
    assert not blank.ok and "corners" in blank.reason


def test_too_far_away_is_rejected_not_guessed():
    det = BoardDetector(SPEC, K, DIST, min_corners=8)
    d = det.detect(render_view(det, _T([0, 0, 0], [-0.14, -0.10, 6.0])))
    print(f"  6 m away: ok={d.ok} corners={d.n_corners} ({d.reason})")
    assert not d.ok


def test_wrong_board_spec_detects_nothing():
    """A spec that disagrees with the printed board must fail loudly."""
    det = BoardDetector(SPEC, K, DIST)
    img = render_view(det, _T([0, 0, 0], [-0.14, -0.10, 0.45]))
    other = BoardDetector(BoardSpec(squares_x=5, squares_y=7, dictionary="4x4_50"), K, DIST)
    assert not other.detect(img).ok


def test_square_size_error_scales_translation():
    """A mis-measured square size biases distance proportionally - the reason to
    measure the PRINTED board rather than trusting the PDF."""
    det = BoardDetector(SPEC, K, DIST)
    img = render_view(det, _T([0, 0, 0], [-0.14, -0.10, 0.50]))
    wrong = BoardDetector(BoardSpec(square_len_m=0.044, marker_len_m=0.033), K, DIST)
    d = wrong.detect(img)
    assert d.ok
    ratio = d.T_cam_target[2, 3] / 0.50
    print(f"  +10% square size -> distance x{ratio:.3f} (reproj still {d.reproj_px:.3f} px: looks perfect, is wrong)")
    assert 1.05 < ratio < 1.15


def test_intrinsics_recovered_from_views():
    det = BoardDetector(SPEC, K, DIST)
    all_c, all_i = [], []
    for rpy, t in (([0, 0, 0], [-0.14, -0.10, 0.40]), ([20, 0, 0], [-0.14, -0.12, 0.42]),
                   ([-20, 0, 5], [-0.14, -0.08, 0.44]), ([0, 22, 0], [-0.16, -0.10, 0.41]),
                   ([0, -22, -5], [-0.12, -0.10, 0.43]), ([15, 15, 10], [-0.15, -0.11, 0.46]),
                   ([-15, -15, -10], [-0.13, -0.09, 0.39]), ([25, -10, 0], [-0.14, -0.10, 0.45])):
        d = det.detect(render_view(det, _T(rpy, t)))
        if d.ok:
            all_c.append(d.corners)
            all_i.append(d.ids)
    rms, K_est, dist_est, n = calibrate_intrinsics(all_c, all_i, det.board, (W, H))
    err = abs(K_est[0, 0] - K[0, 0]) / K[0, 0] * 100
    print(f"  intrinsics from {n} views: rms={rms:.3f} px, fx={K_est[0, 0]:.1f} (truth {K[0, 0]:.1f}, {err:.2f}% off)")
    assert err < 2.0


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
