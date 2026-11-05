"""control.gripper (raw / binary) in the infer app: config, checkpoint check, loop.

No hardware, no network, no station config of a real station: the follower,
cameras, policy and intervention are fakes, the config is a temporary YAML.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from trossen_real.config import CONFIG_DIR, GripperModeConfig, load_station_config
from trossen_real.infer_app.infer_loop import InferenceLoop
from trossen_real.inference.gripper_controller import GripperController, gripper_from_config, looks_binarized

# action stats like the ACT Iter1 checkpoints' (gripper = dim 6)
BIN = {"min": [0] * 6 + [0.004], "max": [0] * 6 + [0.04], "mean": [0] * 6 + [0.02255], "std": [0] * 6 + [0.01799]}
RAW = {"min": [0] * 6 + [0.004], "max": [0] * 6 + [0.0406], "mean": [0] * 6 + [0.02064], "std": [0] * 6 + [0.0164]}


# ---- config -------------------------------------------------------------------------------
def _write_cfg(tmp_path, gripper):
    raw = yaml.safe_load((CONFIG_DIR / "trossen_station3_single.yaml").read_text())
    raw = copy.deepcopy(raw)
    if gripper is None:
        raw["control"].pop("gripper", None)
    else:
        raw["control"]["gripper"] = gripper
    p = tmp_path / "station.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_config_default_is_raw(tmp_path):
    assert load_station_config(str(_write_cfg(tmp_path, None))).control.gripper == GripperModeConfig()


def test_config_binary_block(tmp_path):
    g = load_station_config(str(_write_cfg(tmp_path, {"mode": "binary", "max_step": 0.005}))).control.gripper
    assert g.mode == "binary" and g.max_step == 0.005 and g.open == 0.04


@pytest.mark.parametrize("bad", [{"mode": "snap"}, {"close_below": 0.02, "open_above": 0.016}])
def test_config_rejects(tmp_path, bad):
    with pytest.raises(ValueError):
        load_station_config(str(_write_cfg(tmp_path, bad)))


# ---- checkpoint check ------------------------------------------------------------------------
def _control(mode="raw"):
    return SimpleNamespace(gripper=GripperModeConfig(mode=mode), gripper_closed=0.001, gripper_open=0.04,
                           frequency_hz=1000, goal_time_s=0.002)


def test_looks_binarized():
    assert looks_binarized(BIN) is True
    assert looks_binarized(RAW) is False
    assert looks_binarized({}) is None


def test_raw_with_binarized_checkpoint_is_refused():
    with pytest.raises(ValueError, match="binary"):
        gripper_from_config(_control("raw"), None, BIN)


def test_override_and_info():
    ctrl, info = gripper_from_config(_control("raw"), "binary", BIN)
    assert isinstance(ctrl, GripperController) and (ctrl.lo, ctrl.hi) == (0.001, 0.04)
    assert info["mode"] == "binary" and info["source"] == "connect" and info["warning"] is None


def test_binary_with_recorded_widths_warns():
    ctrl, info = gripper_from_config(_control("binary"), None, RAW)
    assert ctrl is not None and info["source"] == "station config" and "recorded" in info["warning"]


def test_raw_default_and_unknown_stats():
    ctrl, info = gripper_from_config(_control("raw"), "", None)
    assert ctrl is None and info["mode"] == "raw" and "cannot be checked" in info["warning"]
    with pytest.raises(ValueError):
        gripper_from_config(_control(), "snap", RAW)


# ---- the loop ---------------------------------------------------------------------------------
Q0 = [0.0, 1.5, 1.5, -1.5, 0.0, 0.0, 0.004]


class Follower:
    def __init__(self):
        self.sent = []

    def get_state(self):
        return {"q": list(Q0), "pose": [0.0] * 6, "gripper_pos": 0.004, "dq": [0.0] * 7, "efforts": [0.0] * 7,
                "accelerations": [0.0] * 7}

    def move_to_joint_positions(self, a, blocking=False, min_time_to_move=None):
        self.sent.append(np.asarray(a, dtype=float).copy())


class Cameras:
    camera_names = []

    def get_all_latest(self):
        return {}


class Policy:
    """Gripper width per tick from a script, arm joints = Q0."""

    def __init__(self, widths):
        self.widths, self.i = list(widths), 0

    def predict_full(self, obs, raw_images=None):
        w = self.widths[min(self.i, len(self.widths) - 1)]
        self.i += 1
        return {"action": np.array(Q0[:6] + [w], dtype=np.float32), "action_normalized": None}

    def reset(self):
        pass


class Intervention:
    """Intervened on the ticks in `on`, the leader's gripper at `leader_w`."""

    def __init__(self, on, leader_w):
        self.on, self.leader_w, self.k, self.mirrored = set(on), leader_w, -1, []

    def tick(self):
        self.k += 1
        return self.k in self.on

    def get_leader_action(self):
        return np.array(Q0[:6] + [self.leader_w])

    def mirror_to_leader(self, a, goal_time):
        self.mirrored.append(float(np.asarray(a)[6]))


def run(widths, mode, intervention=None):
    control = _control(mode)
    gripper, _ = gripper_from_config(control, None, BIN if mode == "binary" else RAW)
    follower = Follower()
    loop = InferenceLoop(follower, Cameras(), Policy(widths), control, image_keys=[], action_space="absolute",
                         intervention=intervention, gripper=gripper)
    loop.start(settle_time_s=0.0, max_steps=len(widths), skip_reset=True, record_video=False)
    loop._thread.join(10)
    assert not loop.get_snapshot().error, loop.get_snapshot().error
    return np.array([a[6] for a in follower.sent])


def test_raw_mode_sends_the_policy_width():
    w = [0.004] * 5 + [0.04] * 5
    assert np.allclose(run(w, "raw"), w)  # unchanged behaviour: a 36 mm jump in one tick


def test_binary_mode_confirms_then_ramps():
    g = run([0.004] * 5 + [0.04] * 20, "binary")
    assert np.all(np.abs(np.diff(g)) <= 0.003 + 1e-9)  # never more than max_step per tick
    assert np.allclose(g[:7], 0.004)  # the open vote needs 3 ticks (5, 6, 7)
    assert g[7] > 0.004 and abs(g[-1] - 0.04) < 1e-9


def test_binary_mode_ignores_a_one_tick_spike():
    g = run([0.004] * 5 + [0.04] + [0.004] * 10, "binary")
    assert np.allclose(g, 0.004)


def test_binary_intervention_hands_back_without_a_snap():
    iv = Intervention(on=range(5, 20), leader_w=0.03)  # human opens to 30 mm, ACT keeps saying closed
    g = run([0.004] * 40, "binary", iv)
    assert np.all(np.abs(np.diff(g)) <= 0.003 + 1e-9)
    assert abs(g[19] - 0.03) < 1e-9  # followed the leader
    assert g[20] > 0.02  # no snap on release
    assert abs(g[-1] - 0.004) < 1e-9  # then ACT's close, ramped
    assert np.allclose(iv.mirrored, g[[k for k in range(40) if k not in iv.on]])  # leader mirrors the command
