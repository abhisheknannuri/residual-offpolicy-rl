"""Charuco board detection and pose estimation, OpenCV 4.x and 5.x compatible.

OpenCV 5 removed ``aruco.estimatePoseCharucoBoard`` and
``aruco.interpolateCornersCharuco`` (the repo ``.venv`` ships OpenCV 5.0.0, and
those are exactly the calls most hand-eye snippets on the internet use, so they
fail there with AttributeError). The supported path, used here, is::

    CharucoDetector.detectBoard() -> board.matchImagePoints() -> cv2.solvePnP()

Conventions
-----------
:meth:`BoardDetector.estimate_pose` returns **T_cam_target**: the board's pose
expressed in the camera frame, i.e. the matrix that maps a point given in board
coordinates into camera coordinates. That is the ``target2cam`` quantity the
hand-eye solver expects, and getting its direction backwards is the classic
silent failure in hand-eye calibration - hence
``tests/test_charuco.py::test_pose_convention_is_cam_from_target``, which
renders a board at a known pose and checks the recovered matrix against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Dictionary names accepted on the command line -> OpenCV constants.
DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "apriltag_36h11": getattr(cv2.aruco, "DICT_APRILTAG_36h11", cv2.aruco.DICT_6X6_250),
}


@dataclass
class BoardSpec:
    """Physical description of the printed board. MUST match the real one.

    squares_x/y are the number of chessboard SQUARES across and down (e.g. a
    7x5 board has 7 columns and 5 rows of squares). ``square_len_m`` is the side
    of one black/white square; ``marker_len_m`` is the side of the aruco marker
    printed inside the white squares (always smaller). Measure both with
    calipers on the actual printout - printer scaling is a very common source of
    a systematic scale error that shows up as a translation bias.
    """

    squares_x: int = 7
    squares_y: int = 5
    square_len_m: float = 0.040
    marker_len_m: float = 0.030
    dictionary: str = "5x5_100"
    legacy_pattern: bool = False  # True if the board was made by OpenCV < 4.6

    def __post_init__(self):
        if self.marker_len_m >= self.square_len_m:
            raise ValueError(
                f"marker_len_m ({self.marker_len_m}) must be smaller than square_len_m "
                f"({self.square_len_m}); they are the aruco marker and the chessboard square."
            )
        if self.dictionary not in DICTS:
            raise ValueError(f"dictionary must be one of {sorted(DICTS)}, got {self.dictionary!r}")

    @property
    def n_corners(self) -> int:
        """Interior chessboard corners - the points actually used for the pose."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    def describe(self) -> str:
        return (
            f"{self.squares_x}x{self.squares_y} squares @ {self.square_len_m * 1000:.1f} mm "
            f"(marker {self.marker_len_m * 1000:.1f} mm, dict {self.dictionary}), "
            f"{self.n_corners} interior corners, "
            f"printed size {self.squares_x * self.square_len_m * 1000:.0f} x "
            f"{self.squares_y * self.square_len_m * 1000:.0f} mm"
        )


@dataclass
class Detection:
    ok: bool
    n_corners: int
    corners: np.ndarray | None = None
    ids: np.ndarray | None = None
    T_cam_target: np.ndarray | None = None
    reproj_px: float = float("nan")
    reason: str = ""


class BoardDetector:
    def __init__(self, spec: BoardSpec, K: np.ndarray, dist: np.ndarray, *, min_corners: int = 8):
        self.spec = spec
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.min_corners = min_corners
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICTS[spec.dictionary])
        self.board = cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y), spec.square_len_m, spec.marker_len_m, self.dictionary
        )
        if hasattr(self.board, "setLegacyPattern"):
            self.board.setLegacyPattern(spec.legacy_pattern)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    # -- detection ---------------------------------------------------------
    def detect(self, image: np.ndarray) -> Detection:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if corners is None or ids is None or len(ids) < self.min_corners:
            n = 0 if ids is None else len(ids)
            return Detection(False, n, corners, ids, reason=f"only {n} corners (need {self.min_corners})")

        obj_pts, img_pts = self.board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            return Detection(False, len(ids), corners, ids, reason="matchImagePoints returned too few points")

        # Degenerate (near-collinear) corner sets give a wild pose that still
        # "succeeds"; reject on the spread of the board points actually seen.
        pts = obj_pts.reshape(-1, 3)[:, :2]
        if min(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) < 1.5 * self.spec.square_len_m:
            return Detection(False, len(ids), corners, ids,
                             reason="corners nearly collinear - show more of the board")

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return Detection(False, len(ids), corners, ids, reason="solvePnP failed")
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, self.K, self.dist, rvec, tvec)

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.dist)
        reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2, axis=1))))

        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec.ravel()
        return Detection(True, len(ids), corners, ids, T_cam_target=T, reproj_px=reproj)

    # -- drawing -----------------------------------------------------------
    def draw(self, image: np.ndarray, det: Detection) -> np.ndarray:
        out = image.copy()
        if det.corners is not None and det.ids is not None and len(det.ids):
            cv2.aruco.drawDetectedCornersCharuco(out, det.corners, det.ids, (0, 255, 0) if det.ok else (0, 165, 255))
        if det.ok and det.T_cam_target is not None:
            rvec = cv2.Rodrigues(det.T_cam_target[:3, :3])[0]
            cv2.drawFrameAxes(out, self.K, self.dist, rvec, det.T_cam_target[:3, 3], self.spec.square_len_m * 2)
        return out

    def render(self, px_per_m: int = 4000, margin_px: int = 40) -> np.ndarray:
        """Board image for printing, at a known physical scale."""
        w = int(self.spec.squares_x * self.spec.square_len_m * px_per_m)
        h = int(self.spec.squares_y * self.spec.square_len_m * px_per_m)
        return self.board.generateImage((w, h), marginSize=margin_px)


def intrinsics_from_realsense(profile) -> tuple[np.ndarray, np.ndarray, dict]:
    """Factory intrinsics for a started ``rs.pipeline`` colour stream.

    Good enough for hand-eye on a factory-calibrated D4xx: the residual lens
    error is typically well under the FK error of the arm. Use
    ``--calibrate-intrinsics`` to solve for them from the same captures instead.
    """
    import pyrealsense2 as rs  # noqa: F401  (import kept local; only needed with a camera)

    vs = profile.get_stream(rs.stream.color).as_video_stream_profile()
    i = vs.get_intrinsics()
    K = np.array([[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]])
    dist = np.asarray(i.coeffs, dtype=np.float64)
    meta = {"width": i.width, "height": i.height, "model": str(i.model), "source": "realsense_factory"}
    return K, dist, meta


def calibrate_intrinsics(all_corners, all_ids, board, image_size):
    """Intrinsics from the charuco corners already collected (OpenCV 4/5 safe)."""
    obj_pts, img_pts = [], []
    for corners, ids in zip(all_corners, all_ids):
        o, i = board.matchImagePoints(corners, ids)
        if o is not None and len(o) >= 6:
            obj_pts.append(o)
            img_pts.append(i)
    if len(obj_pts) < 6:
        raise ValueError(f"need >= 6 usable views for intrinsics, got {len(obj_pts)}")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)
    return float(rms), K, dist.ravel(), len(obj_pts)
