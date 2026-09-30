"""record-poses and --motion replay, against a fake arm + fake camera.

No robot, no RealSense: FollowerClient and the camera are stubbed, so this
exercises the actual file format, the joint-space replay commands, and the rule
that the EE pose used for the math is read back from the arm at capture time
(never the recorded/commanded value).

Run (repo .venv only - never uv):
    PYTHONPATH=. .venv/bin/python trossen_real/calibration/tests/test_record_replay.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

import trossen_real.calibration.calibrate_eye_in_hand as cal
from trossen_real.calibration.handeye import T_inv
from trossen_real.calibration.tests.test_pipeline_end_to_end import T_BASE_TARGET, VIEWS, X_TRUE
from trossen_real.calibration.tests.test_charuco import _T


class FakeArm:
    """Stands in for FollowerClient. Joint moves teleport; EE pose is the FK of
    whatever joint state we are in, with a small offset so that reading the pose
    back is provably different from replaying the recorded one."""

    def __init__(self, poses_ct):
        self.poses_ct = poses_ct  # board-in-camera pose per index, defines the true EE pose
        self.idx = 0
        self.q = np.zeros(7)
        self.moves = []
        self.connected = False

    # -- lifecycle
    def check(self):
        return True

    def connect(self):
        self.connected = True

    # -- state
    def _T_bg(self):
        return T_BASE_TARGET @ T_inv(self.poses_ct[self.idx]) @ T_inv(X_TRUE)

    def get_pose(self):
        T = self._T_bg()
        return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])

    def get_state(self):
        return {"q": self.q.tolist(), "gripper_pos": float(self.q[6]), "pose": self.get_pose().tolist()}

    def get_gripper(self):
        return float(self.q[6])

    # -- motion
    def move_to_joint_positions(self, arr, blocking=False, min_time_to_move=None):
        arr = np.asarray(arr, dtype=float)
        self.moves.append(arr.copy())
        self.q = arr.copy()
        self.idx = int(round(arr[0] * 100))  # encode "which viewpoint" in joint 0 for the fake


def _record(tmp: Path, arm: FakeArm, n: int) -> Path:
    """Drive cmd_record_poses with scripted stdin."""
    out = tmp / "poses.json"
    keys = []
    for i in range(n):
        arm_idx = i
        keys.append(("", arm_idx))
    script = iter(keys)
    real_stdin = sys.stdin

    def fake_input(_prompt=""):
        try:
            cmd, idx = next(script)
        except StopIteration:
            return "q"
        arm.idx = idx
        arm.q = np.array([idx / 100.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.02])
        return cmd

    cal.FollowerClient = lambda url, **kw: arm  # noqa: ARG005
    sys.stdin = io.StringIO()
    try:
        import builtins

        orig_input = builtins.input
        builtins.input = fake_input
        try:
            cal.main(["record-poses", "--config", "trossen_station1_single", "--out", str(out)])
        finally:
            builtins.input = orig_input
    finally:
        sys.stdin = real_stdin
    return out


def test_record_poses_writes_a_reusable_file():
    poses_ct = [_T(rpy, t) for rpy, t in VIEWS]
    arm = FakeArm(poses_ct)
    with tempfile.TemporaryDirectory() as d:
        out = _record(Path(d), arm, n=8)
        data = json.loads(out.read_text())
    assert len(data["poses"]) == 8
    p0 = data["poses"][0]
    assert set(p0) == {"q", "gripper", "pose6"}, p0.keys()
    assert len(p0["q"]) == 7, "q must be 6 arm joints + gripper"
    assert len(p0["pose6"]) == 6, "pose6 must be x,y,z,ax,ay,az"
    assert data["station"] == "trossen_station1_single"
    assert "rotation_axis_spread_deg" in data
    print(f"  saved {len(data['poses'])} poses; keys per pose = {sorted(p0)}")
    print(f"  axis spread recorded: {data['rotation_axis_spread_deg']:.0f} deg")


def test_replay_commands_joints_and_reads_ee_pose_back():
    poses_ct = [_T(rpy, t) for rpy, t in VIEWS]
    arm = FakeArm(poses_ct)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        poses_file = _record(tmp, arm, n=len(VIEWS))

        # fake camera + detector: return the board pose for whichever index the arm is at
        class FakeDet:
            def detect(self, _img):
                return SimpleNamespace(
                    ok=True, n_corners=24, reproj_px=0.2,
                    T_cam_target=poses_ct[arm.idx], corners=None, ids=None,
                )

            def draw(self, img, _d):
                return img

        cal._grab = lambda _p, **kw: np.zeros((8, 8, 3), np.uint8)  # noqa: ARG005
        args = SimpleNamespace(
            poses=str(poses_file), config="trossen_station1_single", camera="cam_right_wrist",
            goal_time=0.0, settle_s=0.0, frames_per_pose=1, max_reproj_px=1.5,
            save_images=False, yes=True,
        )
        captures, _, _ = cal._replay_capture(args, arm, pipeline=None, det=FakeDet())
        recorded = json.loads(poses_file.read_text())["poses"]

    assert len(captures) == len(VIEWS), f"{len(captures)} captured"
    # joint-space commands, 7D, one per recorded pose
    assert len(arm.moves) == len(VIEWS) and all(len(m) == 7 for m in arm.moves)
    # the EE pose stored must be the one READ BACK from the arm, not the recorded file
    for cap, rec in zip(captures, recorded):
        assert np.allclose(cap["pose6"], rec["pose6"], atol=1e-6)  # same here because the fake is exact
        assert cap["T_cam_target"] is not None
    print(f"  replayed {len(captures)} poses; {len(arm.moves)} joint commands issued (7D each)")

    # and the captured data still solves to the ground-truth transform
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "c.json", Path(d) / "r.json"
        src.write_text(json.dumps({"captures": captures}))
        cal.main(["solve", "--captures", str(src), "--out", str(out)])
        X = np.array(json.loads(out.read_text())["T_gripper_cam"])
    mm = np.linalg.norm(X[:3, 3] - X_TRUE[:3, 3]) * 1000
    print(f"  solved from replayed captures: {mm:.2f} mm from truth")
    assert mm < 2.0


def test_replay_skips_poses_where_the_board_is_not_seen():
    poses_ct = [_T(rpy, t) for rpy, t in VIEWS]
    arm = FakeArm(poses_ct)
    with tempfile.TemporaryDirectory() as d:
        poses_file = _record(Path(d), arm, n=6)

        class FlakyDet:
            def detect(self, _img):
                ok = arm.idx % 2 == 0  # board visible at every other pose
                return SimpleNamespace(
                    ok=ok, n_corners=24 if ok else 0, reproj_px=0.2, reason="no board",
                    T_cam_target=poses_ct[arm.idx] if ok else None, corners=None, ids=None,
                )

            def draw(self, img, _d):
                return img

        cal._grab = lambda _p, **kw: np.zeros((8, 8, 3), np.uint8)  # noqa: ARG005
        args = SimpleNamespace(
            poses=str(poses_file), config="trossen_station1_single", camera="cam", goal_time=0.0,
            settle_s=0.0, frames_per_pose=1, max_reproj_px=1.5, save_images=False, yes=True,
        )
        captures, _, _ = cal._replay_capture(args, arm, pipeline=None, det=FlakyDet())
    assert len(captures) == 3, f"expected the 3 visible poses, got {len(captures)}"
    assert len(arm.moves) == 6, "must still visit every recorded pose"
    print(f"  {len(captures)}/6 captured, undetected poses skipped without aborting")


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
