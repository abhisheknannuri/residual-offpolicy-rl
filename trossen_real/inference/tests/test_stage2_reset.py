"""Stage2ResetFollower's reset sequence against a fake follower (no robot)."""

from __future__ import annotations

import numpy as np
import pytest

from trossen_real.rl import stage2_reset as s2


class Recorder(s2.Stage2ResetFollower):
    """Records the motion calls instead of sending HTTP requests."""

    def __init__(self, mode, answers, seed=0):
        it = iter(answers)
        super().__init__("http://unused", mode=mode, seed=seed, confirm=lambda _: next(it))
        self.calls = []

    def move_to_joint_positions(self, joint7, blocking=False, min_time_to_move=None):
        self.calls.append(("joint", np.asarray(joint7, dtype=np.float64), blocking))

    def move_to_ee_pose(self, pose7, blocking=False, min_time_to_move=None):
        self.calls.append(("ee", np.asarray(pose7, dtype=np.float64), blocking))

    def move_gripper(self, gripper_pos, blocking=False, min_time_to_move=None):
        self.calls.append(("gripper", gripper_pos, blocking))

    def get_state(self):
        return {"q": list(s2.STAGE2_JOINTS), "pose": [0.30, -0.08, 0.08, 0.02, 1.54, -0.03]}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(s2.time, "sleep", lambda _: None)


def test_joint_mode_sequence_and_limits():
    f = Recorder("joint", answers=["", "x", "0"])  # anything but "0" keeps waiting
    f.reset()
    kinds = [c[0] for c in f.calls]
    assert kinds == ["joint", "gripper", "joint"]
    staged = f.calls[0][1]
    assert np.allclose(staged[:6], s2.STAGE2_JOINTS[:6]) and staged[6] == pytest.approx(s2.GRIPPER_OPEN)
    assert f.calls[1][1] == s2.GRIPPER_CUBE_CLOSE
    final = f.calls[2][1]
    assert final[6] == pytest.approx(s2.GRIPPER_CUBE_CLOSE)  # still holding the cube
    assert np.all(final[:6] >= np.asarray(s2.JOINT_MIN) - 1e-6) and np.all(final[:6] <= np.asarray(s2.JOINT_MAX) + 1e-6)
    untouched = [i for i in range(6) if i not in s2.JOINT_NOISE_INDICES]
    assert np.allclose(final[untouched], np.asarray(s2.STAGE2_JOINTS)[untouched], atol=1e-6)
    assert all(c[2] for c in f.calls)  # every move blocking


def test_joint_noise_bounded():
    f = Recorder("joint", answers=[])
    for _ in range(500):
        t = f.sample_target()
        d = np.asarray(t["joints"][:6]) - np.asarray(s2.STAGE2_JOINTS[:6])
        assert np.all(np.abs(d[s2.JOINT_NOISE_INDICES]) <= np.asarray(s2.JOINT_NOISE) + 1e-6)


def test_ee_mode_offsets_and_gripper():
    f = Recorder("ee", answers=["0"])
    f.reset()
    assert [c[0] for c in f.calls] == ["joint", "gripper", "ee"]
    pose7 = f.calls[2][1]
    base = np.asarray(f.get_state()["pose"])
    assert np.all(np.abs(pose7[:3] - base[:3]) <= np.asarray(s2.EE_XYZ_OFFSET) + 1e-6)
    assert np.all(np.abs(pose7[3:6] - base[3:6]) <= np.asarray(s2.EE_ANGLE_OFFSET) + 1e-6)
    assert pose7[6] == pytest.approx(s2.GRIPPER_CUBE_CLOSE)


def test_none_mode_stays():
    f = Recorder("none", answers=["0"])
    f.reset()
    assert [c[0] for c in f.calls] == ["joint", "gripper"]


# ---- infer-app confirm (UI button) and follower factory ----------------------------
import threading  # noqa: E402
import time as _time  # noqa: E402

from trossen_real.teleop.follower_client import FollowerClient  # noqa: E402


def _wait_until(pred, timeout=2.0):
    end = _time.time() + timeout
    while not pred() and _time.time() < end:
        _time.sleep(0.01)
    return pred()


def test_cube_confirm_button_releases_reset():
    c = s2.CubeConfirm()
    assert c.confirm() is False  # nothing waiting: a stray click does nothing
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", c("place the cube")))
    t.start()
    assert _wait_until(lambda: c.waiting)
    assert c.confirm() is True
    t.join(2)
    assert out["v"] == "0" and not c.waiting


def test_cube_confirm_cancel_raises():
    c = s2.CubeConfirm()
    err = {}

    def run():
        try:
            c("place the cube")
        except RuntimeError as e:
            err["e"] = e

    t = threading.Thread(target=run)
    t.start()
    assert _wait_until(lambda: c.waiting)
    c.cancel()
    t.join(2)
    assert "cancelled" in str(err["e"]) and not c.waiting


def test_make_follower_modes():
    assert type(s2.make_follower("http://x", "normal")) is FollowerClient
    for mode in ("joint", "ee", "none"):
        f = s2.make_follower("http://x", f"stage2_{mode}")
        assert isinstance(f, s2.Stage2ResetFollower) and f.mode == mode
    with pytest.raises(ValueError):
        s2.make_follower("http://x", "bogus")


def test_leader_sync_uses_commanded_gripper_after_stage2_reset():
    q = [0.1, 1.6, 1.3, -1.1, 0.0, -0.2, 0.0115]  # gripper reads the cube width
    f = s2.make_follower("http://x", "stage2_joint")
    t = s2.leader_sync_joints(f, q)
    assert t[:6] == q[:6] and t[6] == s2.GRIPPER_CUBE_CLOSE


def test_leader_sync_normal_reset_unchanged():
    q = [0.0, 1.57, 1.57, -1.57, 0.0, 0.0, 0.04]
    assert s2.leader_sync_joints(s2.make_follower("http://x", "normal"), q) == q
