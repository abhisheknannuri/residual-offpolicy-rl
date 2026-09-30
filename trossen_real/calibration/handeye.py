"""Eye-in-hand (AX = XB) solvers, in pure numpy/scipy.

Why not ``cv2.calibrateHandEye``
--------------------------------
The only environment on this machine with ``pyrealsense2`` is the repo ``.venv``,
which ships OpenCV **5.0.0** - and that build exposes neither ``calibrateHandEye``
nor ``aruco.estimatePoseCharucoBoard`` (both present in 4.x). So the solver lives
here instead of being borrowed from OpenCV. ``tests/test_handeye.py``
cross-checks these implementations against ``cv2.calibrateHandEye`` from the
OpenCV 4.11 install in the miniforge ``residual`` env, and against synthetic
ground truth.

The problem
-----------
Eye-in-hand: the camera is bolted to the gripper, the board is fixed in the world.
We know per sample i:

    T_bg[i]  gripper pose in the robot base   (from the arm's forward kinematics)
    T_ct[i]  board pose in the camera frame   (from solvePnP on the charuco board)

and want the constant X = T_gc, the camera's pose in the gripper frame.

Because the board does not move, for any two samples i, j::

    T_bg[i] · X · T_ct[i] = T_bg[j] · X · T_ct[j]        ( = T_bt, constant)

Rearranged into the classic form, with

    A = T_bg[j]⁻¹ · T_bg[i]      (relative gripper motion)
    B = T_ct[j] · T_ct[i]⁻¹      (relative camera motion)

this is ``A · X = X · B``.

What actually matters for accuracy
----------------------------------
- **Rotate a lot, between many axes.** The rotation of X is only observable
  through rotation of A. Pure translations tell you nothing about R_X, and two
  poses that differ by a rotation about the *same* axis leave one degree of
  freedom unresolved. Aim for >= 3 clearly different rotation axes and >= 30 deg
  of rotation between poses.
- **Translation accuracy degrades with small rotations**: t_X is recovered from
  ``(R_A - I) t_X = R_X t_B - t_A``, and ``R_A - I`` is singular for zero
  rotation. ``min_angle_deg`` in :func:`calibrate_eye_in_hand` drops such pairs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------
def rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def T_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def pose6_to_T(pose6) -> np.ndarray:
    """(x, y, z, ax, ay, az) -> 4x4. The Trossen SDK's cartesian format is
    position + **axis-angle** (rotation vector), per trossen_real/config.py."""
    p = np.asarray(pose6, dtype=float).reshape(6)
    return rt_to_T(Rotation.from_rotvec(p[3:6]).as_matrix(), p[:3])


def T_to_pose6(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def _nearest_rotation(M: np.ndarray) -> np.ndarray:
    """Closest proper rotation to M (orthogonal Procrustes, det = +1)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# Rotation solvers  (AX = XB)
# ---------------------------------------------------------------------------
def _solve_translation(A_list, B_list, R_X: np.ndarray) -> np.ndarray:
    """Least squares on (R_A - I) t_X = R_X t_B - t_A, stacked over all pairs."""
    C, d = [], []
    for A, B in zip(A_list, B_list):
        C.append(A[:3, :3] - np.eye(3))
        d.append(R_X @ B[:3, 3] - A[:3, 3])
    t_X, *_ = np.linalg.lstsq(np.vstack(C), np.concatenate(d), rcond=None)
    return t_X


def solve_park(A_list, B_list) -> np.ndarray:
    """Park & Martin (1994), closed form on so(3).

    M = Σ βᵢ αᵢᵀ with α = log(R_A), β = log(R_B);  R_X = (Mᵀ M)^(-1/2) Mᵀ.
    """
    M = np.zeros((3, 3))
    for A, B in zip(A_list, B_list):
        alpha = Rotation.from_matrix(A[:3, :3]).as_rotvec()
        beta = Rotation.from_matrix(B[:3, :3]).as_rotvec()
        M += np.outer(beta, alpha)
    # (MᵀM)^(-1/2) via eigendecomposition of the symmetric MᵀM
    w, V = np.linalg.eigh(M.T @ M)
    w = np.clip(w, 1e-12, None)
    R_X = V @ np.diag(1.0 / np.sqrt(w)) @ V.T @ M.T
    R_X = _nearest_rotation(R_X)
    return rt_to_T(R_X, _solve_translation(A_list, B_list, R_X))


def solve_tsai(A_list, B_list) -> np.ndarray:
    """Tsai & Lenz (1989), linear least squares on modified Rodrigues vectors."""
    C, d = [], []
    for A, B in zip(A_list, B_list):
        rA = Rotation.from_matrix(A[:3, :3]).as_rotvec()
        rB = Rotation.from_matrix(B[:3, :3]).as_rotvec()
        thA, thB = np.linalg.norm(rA), np.linalg.norm(rB)
        if thA < 1e-9 or thB < 1e-9:
            continue
        # P = 2 sin(θ/2) · axis
        PA = 2 * np.sin(thA / 2) * (rA / thA)
        PB = 2 * np.sin(thB / 2) * (rB / thB)
        C.append(_skew(PA + PB))
        d.append(PB - PA)
    if not C:
        raise ValueError("every pair had ~zero rotation; the data cannot constrain R_X")
    P_X_prime, *_ = np.linalg.lstsq(np.vstack(C), np.concatenate(d), rcond=None)
    P_X = 2 * P_X_prime / np.sqrt(1 + float(P_X_prime @ P_X_prime))
    n2 = float(P_X @ P_X)
    R_X = (1 - n2 / 2) * np.eye(3) + 0.5 * (np.outer(P_X, P_X) + np.sqrt(max(4 - n2, 0.0)) * _skew(P_X))
    R_X = _nearest_rotation(R_X)
    return rt_to_T(R_X, _solve_translation(A_list, B_list, R_X))


def refine_nonlinear(X0: np.ndarray, A_list, B_list) -> np.ndarray:
    """Levenberg-Marquardt on the full AX - XB residual (rotation + translation).

    The closed forms above solve rotation first and translation second, so
    translation error cannot feed back into rotation. This refines both jointly.
    """

    def unpack(p):
        return rt_to_T(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6])

    def residuals(p):
        X = unpack(p)
        out = []
        for A, B in zip(A_list, B_list):
            E = A @ X - X @ B
            out.append(Rotation.from_matrix(_nearest_rotation(A[:3, :3] @ X[:3, :3])).as_rotvec()
                       - Rotation.from_matrix(_nearest_rotation(X[:3, :3] @ B[:3, :3])).as_rotvec())
            out.append(E[:3, 3])
        return np.concatenate(out)

    p0 = np.concatenate([Rotation.from_matrix(X0[:3, :3]).as_rotvec(), X0[:3, 3]])
    sol = least_squares(residuals, p0, method="lm", xtol=1e-12, ftol=1e-12)
    return unpack(sol.x)


SOLVERS = {"park": solve_park, "tsai": solve_tsai}


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
@dataclass
class HandEyeResult:
    T_gripper_cam: np.ndarray  # X: camera pose in the gripper frame
    T_base_target: np.ndarray  # board pose in the base frame (mean over samples)
    method: str
    n_samples: int
    n_pairs: int
    # Diagnostics - read these before trusting the result.
    rot_residual_deg: float = 0.0  # RMS of angle(R_A R_X  vs  R_X R_B) over pairs
    trans_residual_mm: float = 0.0  # RMS of |A X - X B| translation over pairs
    target_spread_mm: float = 0.0  # spread of the recovered board position (THE number)
    target_spread_deg: float = 0.0  # spread of the recovered board orientation
    per_sample_target_err_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary(self) -> str:
        t = self.T_gripper_cam[:3, 3]
        rpy = Rotation.from_matrix(self.T_gripper_cam[:3, :3]).as_euler("xyz", degrees=True)
        return (
            f"method={self.method}  samples={self.n_samples}  pairs={self.n_pairs}\n"
            f"  camera in gripper: xyz = [{t[0] * 1000:+8.2f}, {t[1] * 1000:+8.2f}, {t[2] * 1000:+8.2f}] mm\n"
            f"                     rpy = [{rpy[0]:+8.2f}, {rpy[1]:+8.2f}, {rpy[2]:+8.2f}] deg (xyz-extrinsic)\n"
            f"  AX=XB residual:    rot {self.rot_residual_deg:.3f} deg | trans {self.trans_residual_mm:.2f} mm\n"
            f"  board-in-base spread: {self.target_spread_mm:.2f} mm | {self.target_spread_deg:.3f} deg"
        )


def make_pairs(T_bg_list, T_ct_list, min_angle_deg: float = 5.0):
    """All sample pairs whose relative motion actually constrains the solution."""
    A_list, B_list, kept = [], [], []
    for i, j in itertools.combinations(range(len(T_bg_list)), 2):
        A = T_inv(T_bg_list[j]) @ T_bg_list[i]
        B = T_ct_list[j] @ T_inv(T_ct_list[i])
        if rotation_angle_deg(A[:3, :3]) < min_angle_deg:
            continue
        A_list.append(A)
        B_list.append(B)
        kept.append((i, j))
    return A_list, B_list, kept


def calibrate_eye_in_hand(
    T_base_gripper: list[np.ndarray],
    T_cam_target: list[np.ndarray],
    *,
    method: str = "park",
    min_angle_deg: float = 5.0,
    refine: bool = True,
) -> HandEyeResult:
    """Solve for the camera pose in the gripper frame.

    Args:
        T_base_gripper: 4x4 gripper-in-base poses (robot FK), one per capture.
        T_cam_target:   4x4 board-in-camera poses (solvePnP), one per capture.
        method:         "park" or "tsai" for the closed-form initialisation.
        min_angle_deg:  drop pose pairs whose relative rotation is smaller.
        refine:         run the joint nonlinear refinement afterwards.
    """
    if len(T_base_gripper) != len(T_cam_target):
        raise ValueError("pose lists must be the same length")
    if len(T_base_gripper) < 3:
        raise ValueError(f"need at least 3 captures, got {len(T_base_gripper)}")
    if method not in SOLVERS:
        raise ValueError(f"method must be one of {sorted(SOLVERS)}, got {method!r}")

    A_list, B_list, _ = make_pairs(T_base_gripper, T_cam_target, min_angle_deg)
    if len(A_list) < 2:
        raise ValueError(
            f"only {len(A_list)} usable pose pairs (need >= 2 with at least {min_angle_deg} deg "
            "of relative rotation). Capture poses with MORE rotation, about different axes."
        )

    X = SOLVERS[method](A_list, B_list)
    if refine:
        X = refine_nonlinear(X, A_list, B_list)

    # --- diagnostics -------------------------------------------------------
    rot_errs, trans_errs = [], []
    for A, B in zip(A_list, B_list):
        L, R = A @ X, X @ B
        rot_errs.append(rotation_angle_deg(L[:3, :3].T @ R[:3, :3]))
        trans_errs.append(np.linalg.norm(L[:3, 3] - R[:3, 3]))
    # The board is physically fixed, so every sample must place it in the same
    # spot in the base frame. The spread of that estimate is the honest accuracy
    # figure - it folds in FK error, detection noise and the solve itself.
    T_bt = [T_bg @ X @ T_ct for T_bg, T_ct in zip(T_base_gripper, T_cam_target)]
    positions = np.array([T[:3, 3] for T in T_bt])
    mean_pos = positions.mean(axis=0)
    per_sample_mm = np.linalg.norm(positions - mean_pos, axis=1) * 1000
    mean_R = _nearest_rotation(np.mean([T[:3, :3] for T in T_bt], axis=0))
    ang = [rotation_angle_deg(mean_R.T @ T[:3, :3]) for T in T_bt]

    return HandEyeResult(
        T_gripper_cam=X,
        T_base_target=rt_to_T(mean_R, mean_pos),
        method=method + ("+refine" if refine else ""),
        n_samples=len(T_base_gripper),
        n_pairs=len(A_list),
        rot_residual_deg=float(np.sqrt(np.mean(np.square(rot_errs)))),
        trans_residual_mm=float(np.sqrt(np.mean(np.square(trans_errs))) * 1000),
        target_spread_mm=float(per_sample_mm.std()),
        target_spread_deg=float(np.std(ang)),
        per_sample_target_err_mm=per_sample_mm,
    )
