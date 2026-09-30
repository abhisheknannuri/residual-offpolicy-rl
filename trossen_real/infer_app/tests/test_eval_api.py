"""P1 tests: the Flask API with fake hardware but the REAL `InferenceLoop`.

Only the three external dependencies are faked (follower server, cameras, policy
server). The inference thread, snapshot, stop_reason logic and the eval hook all
run for real, so this covers the wiring that unit tests cannot.

The first test is the one that matters most: **with eval mode off, nothing
changes** - that is the promise made in EVAL_MODE_PLAN.md.

Run (repo .venv only - never uv):
    PYTHONPATH=. .venv/bin/python trossen_real/infer_app/tests/test_eval_api.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import trossen_real.infer_app.app as appmod
from trossen_real.config import load_station_config

REPO = Path(__file__).resolve().parents[3]
EVAL_CFG = str(REPO / "trossen_real/configs/eval/pick_cube_and_insert_station3.yaml")
STATION = "trossen_station3_single"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakeFollower:
    def __init__(self, *_a, **_k):
        self.moves = 0

    def connect(self): pass
    def disconnect(self): pass
    def check(self): return True
    def reset(self): pass

    def get_state(self):
        return {"q": [0.1] * 7, "pose": [0.0] * 6, "gripper_pos": 0.02,
                "dq": [0.0] * 7, "efforts": [0.0] * 7, "accelerations": [0.0] * 7}

    def move_to_joint_positions(self, *_a, **_k):
        self.moves += 1


class FakeCameras:
    def __init__(self, config, *_a, **_k):
        self.camera_names = [d.name for d in config.cameras.devices if d.serial] or ["cam_right_wrist"]

    def start(self): pass
    def stop(self): pass
    def health(self): return {n: True for n in self.camera_names}
    def is_mock(self, _n): return False

    def get_camera_status(self):
        return {n: {"healthy": True, "is_mock": False, "last_frame_age_s": 0.01, "reconnect_count": 0}
                for n in self.camera_names}

    def get_latest_frame(self, _n):
        return np.zeros((64, 64, 3), np.uint8)

    def get_all_latest(self):
        return {n: np.zeros((64, 64, 3), np.uint8) for n in self.camera_names}


class FakePolicy:
    def __init__(self, *_a, **_k):
        self.image_keys = None

    def health(self):
        cfg = load_station_config(STATION)
        keys = [f"observation.images.{d.name}" for d in cfg.cameras.devices if d.serial][:1] \
            or ["observation.images.cam_right_wrist"]
        return {"loaded": True, "checkpoint": "/ckpt/055000/pretrained_model", "device": "cuda:0",
                "chunk_size": 20, "n_action_steps": 12, "likely_action_space": "delta_joint",
                "image_keys": keys}

    def reset(self): pass

    def predict_full(self, _obs):
        return {"action": np.zeros(7, np.float32), "action_normalized": np.zeros(7, np.float32)}


def install_fakes(monkey: dict):
    monkey["FollowerClient"] = appmod.FollowerClient
    monkey["CameraManager"] = appmod.CameraManager
    monkey["PolicyClient"] = appmod.PolicyClient
    monkey["check_camera_health"] = appmod.check_camera_health
    appmod.FollowerClient = FakeFollower
    appmod.CameraManager = FakeCameras
    appmod.PolicyClient = FakePolicy
    appmod.check_camera_health = lambda *_a, **_k: []
    # the loop builds its own CameraManager? no - it receives ours. But the loop
    # module resolves PolicyClient/FollowerClient only for type hints.


def restore_fakes(monkey: dict):
    appmod.FollowerClient = monkey["FollowerClient"]
    appmod.CameraManager = monkey["CameraManager"]
    appmod.PolicyClient = monkey["PolicyClient"]
    appmod.check_camera_health = monkey["check_camera_health"]


def client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def connect(c, **extra):
    body = {"config_name": STATION, "policy_server_url": "http://fake:5070",
            "action_space": "delta_joint", "force_action_space": True}
    body.update(extra)
    return c.post("/api/connect", json=body)


def run_once(c, *, steps=6, eval_body=None, settle=0.0):
    body = {"settle_time_s": settle, "skip_reset": True, "record_video": False,
            "max_steps": steps, "log": False}
    if eval_body is not None:
        body["eval"] = eval_body
    r = c.post("/api/start_inference", json=body)
    if r.status_code != 200:
        return r
    for _ in range(200):  # wait for the loop to finish on its own (max_steps)
        time.sleep(0.05)
        if not c.get("/api/health").get_json().get("inferring"):
            break
    return r


# ---------------------------------------------------------------------------
# 1. REGRESSION: eval mode off behaves exactly as before
# ---------------------------------------------------------------------------
def test_eval_off_is_unchanged():
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            appmod.EVAL_DIR = Path(d) / "eval_runs"
            r = connect(c)
            assert r.status_code == 200, r.get_json()
            body = r.get_json()
            assert body["eval_mode"] is False and body["eval_config"] is None

            h = c.get("/api/health").get_json()
            assert h["connected"] is True
            assert h["eval"] == {"eval_mode": False}, h["eval"]
            # every pre-existing key still present
            for key in ("station_name", "follower", "cameras", "camera_details", "policy_health",
                        "action_space", "inferring", "step", "max_steps", "last_action",
                        "achieved_hz", "target_hz", "error", "log_file", "video_files",
                        "intervention_enabled", "intervening", "leader_connected",
                        "dataset_recording_enabled", "dataset_session_active", "dataset_recording",
                        "num_episodes", "dataset_dir", "stop_reason"):
                assert key in h, f"/api/health lost '{key}'"

            assert run_once(c, steps=5).status_code == 200
            h = c.get("/api/health").get_json()
            assert h["step"] == 5 and h["stop_reason"] == "max_steps"

            # eval endpoints refuse cleanly rather than exploding
            assert c.post("/api/eval/finalize", json={"action": "save"}).status_code == 409
            assert c.get("/api/eval/config").status_code == 409

            assert c.post("/api/disconnect").get_json()["status"] == "ok"
            assert not (Path(d) / "eval_runs").exists(), "eval mode off must write no eval artifacts"
        print("  connect/start/stop/health/disconnect unchanged; no eval files written")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


# ---------------------------------------------------------------------------
# 2. eval flow
# ---------------------------------------------------------------------------
def test_eval_full_flow_save():
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "eval_runs"
            appmod.EVAL_DIR = root
            r = connect(c, eval_config_name=EVAL_CFG, checkpoint_label="ckpt_055000", operator="abhi")
            assert r.status_code == 200, r.get_json()
            cfg = r.get_json()["eval_config"]
            assert cfg["n_stages"] == 4 and cfg["n_poses"] == 20

            h = c.get("/api/health").get_json()["eval"]
            assert h["eval_mode"] and h["pose_cursor"] == 1 and h["pending_review"] is None
            assert "Pose #01" in h["pose_hint"]

            assert run_once(c, steps=7).status_code == 200
            h = c.get("/api/health").get_json()["eval"]
            pend = h["pending_review"]
            assert pend is not None and pend["steps"] == 7 and pend["stop_reason"] == "max_steps"
            assert h["run_active"] is False

            # start is refused while a review is pending
            r2 = c.post("/api/start_inference", json={"skip_reset": True, "max_steps": 3})
            assert r2.status_code == 409 and "Save or Discard" in r2.get_json()["error"]

            assert c.post("/api/eval/annotate", json={"furthest_stage": 2,
                                                      "failure_note": "did not seat"}).status_code == 200
            fin = c.post("/api/eval/finalize", json={"action": "save"})
            assert fin.status_code == 200, fin.get_json()
            sess = fin.get_json()["session"]
            assert sess["pose_cursor"] == 2, "saving must advance the pose cursor"
            assert sess["saved_pose_indices"] == [1]

            summary = json.loads((Path(fin.get_json()["eval_dir"]) / "summary.json").read_text())
            assert summary["status"] == "saved"
            assert summary["annotation"]["furthest_stage"] == 2
            assert summary["annotation"]["success"] is False
            assert summary["policy"]["n_action_steps"] == 12, "eval-time n_action_steps must be recorded"
            assert summary["pose"]["index"] == 1 and summary["pose"]["seed"] == 42
            assert summary["run"]["timed_out"] is True
            rows = [json.loads(x) for x in (root / "index.jsonl").read_text().splitlines()]
            assert len(rows) == 1 and rows[0]["furthest_stage"] == 2
            print(f"  saved -> {Path(fin.get_json()['eval_dir']).name}, cursor now {sess['pose_cursor']}")

            c.post("/api/disconnect")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_eval_discard_keeps_everything_and_holds_cursor():
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "eval_runs"
            appmod.EVAL_DIR = root
            connect(c, eval_config_name=EVAL_CFG, checkpoint_label="ckpt_055000")
            run_once(c, steps=4)
            fin = c.post("/api/eval/finalize", json={"action": "discard", "reason": "cube fell off"})
            assert fin.status_code == 200
            sess = fin.get_json()["session"]
            assert sess["pose_cursor"] == 1, "discard must NOT advance the cursor"
            assert sess["saved_pose_indices"] == []
            s = json.loads((Path(fin.get_json()["eval_dir"]) / "summary.json").read_text())
            assert s["status"] == "discarded" and s["discard"]["reason"] == "cube fell off"
            assert (Path(fin.get_json()["eval_dir"]) / "timeline.jsonl").exists()
            print("  discarded run persisted in full; cursor held at 1")
            c.post("/api/disconnect")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_pose_cursor_and_explicit_pose_index():
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            appmod.EVAL_DIR = Path(d) / "eval_runs"
            connect(c, eval_config_name=EVAL_CFG, checkpoint_label="ck")
            assert c.post("/api/eval/pose_cursor", json={"pose_index": 12}).status_code == 200
            assert c.get("/api/health").get_json()["eval"]["pose_cursor"] == 12
            assert c.post("/api/eval/pose_cursor", json={"pose_index": 99}).status_code == 400
            run_once(c, steps=3, eval_body={"pose_index": 5})
            assert c.get("/api/health").get_json()["eval"]["pending_review"]["pose_index"] == 5
            c.post("/api/eval/finalize", json={"action": "save"})
            c.post("/api/disconnect")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_missing_checkpoint_label_refuses_to_start():
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            appmod.EVAL_DIR = Path(d) / "eval_runs"
            connect(c, eval_config_name=EVAL_CFG)  # no checkpoint_label
            r = c.post("/api/start_inference", json={"skip_reset": True, "max_steps": 3})
            assert r.status_code == 409 and "checkpoint_label" in r.get_json()["error"]
            # and the aborted run must not wedge the session
            assert c.get("/api/health").get_json()["eval"]["pending_review"] is None
            r = c.post("/api/start_inference",
                       json={"skip_reset": True, "max_steps": 3, "eval": {"checkpoint_label": "ck"}})
            assert r.status_code == 200
            c.post("/api/disconnect")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_annotation_validation():
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            appmod.EVAL_DIR = Path(d) / "eval_runs"
            connect(c, eval_config_name=EVAL_CFG, checkpoint_label="ck")
            # cannot annotate/finalize before a run has happened
            assert c.post("/api/eval/annotate", json={"furthest_stage": 1}).status_code == 400
            run_once(c, steps=3)
            assert c.post("/api/eval/annotate", json={"furthest_stage": 9}).status_code == 400
            assert c.post("/api/eval/annotate", json={"furthest_stage": None}).status_code == 200
            assert c.post("/api/eval/finalize", json={"action": "sideways"}).status_code == 400
            assert c.post("/api/eval/finalize", json={"action": "save"}).status_code == 200
            c.post("/api/disconnect")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_disconnect_with_pending_review_writes_unreviewed():
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "eval_runs"
            appmod.EVAL_DIR = root
            connect(c, eval_config_name=EVAL_CFG, checkpoint_label="ck")
            run_once(c, steps=4)
            c.post("/api/disconnect")  # never reviewed
            rows = [json.loads(x) for x in (root / "index.jsonl").read_text().splitlines()]
            assert len(rows) == 1 and rows[0]["status"] == "unreviewed"
            print("  unreviewed run preserved on disconnect, not lost")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_run_ending_by_itself_is_detected_by_polling():
    """max_steps ends the loop with no HTTP call - /api/health must still move
    the run into pending review."""
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            appmod.EVAL_DIR = Path(d) / "eval_runs"
            connect(c, eval_config_name=EVAL_CFG, checkpoint_label="ck")
            c.post("/api/start_inference",
                   json={"settle_time_s": 0.0, "skip_reset": True, "record_video": False,
                         "max_steps": 3, "log": False})
            for _ in range(200):
                time.sleep(0.05)
                ev = c.get("/api/health").get_json()["eval"]
                if ev["pending_review"] is not None:
                    break
            assert ev["pending_review"]["stop_reason"] == "max_steps"
            assert ev["pending_review"]["steps"] == 3
            print("  self-terminating run detected without any Stop call")
            c.post("/api/disconnect")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_eval_config_error_does_not_connect_hardware():
    m = {}
    install_fakes(m)
    try:
        c = client()
        r = connect(c, eval_config_name="no_such_eval_config")
        assert r.status_code == 400 and "Eval config problem" in r.get_json()["error"]
        assert c.get("/api/health").get_json()["connected"] is False, \
            "a bad eval config must not leave the follower/cameras connected"
        print("  bad eval config rejected before any hardware was touched")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_max_steps_defaults_to_eval_config():
    m = {}
    install_fakes(m)
    try:
        c = client()
        with tempfile.TemporaryDirectory() as d:
            appmod.EVAL_DIR = Path(d) / "eval_runs"
            connect(c, eval_config_name=EVAL_CFG, checkpoint_label="ck")
            c.post("/api/start_inference",
                   json={"settle_time_s": 0.0, "skip_reset": True, "record_video": False, "log": False})
            time.sleep(0.3)
            assert c.get("/api/health").get_json()["max_steps"] == 500, "should use the config's 500"
            c.post("/api/stop_inference")
            c.post("/api/eval/finalize", json={"action": "discard", "reason": "test"})
            c.post("/api/disconnect")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


# ---------------------------------------------------------------------------
# pedals in eval mode
# ---------------------------------------------------------------------------
def test_eval_mode_enables_reward_reset_pedals_without_dataset_recording():
    """Without this, every eval run burns all 500 steps because there is no way
    to say 'it worked, stop now' - the reward pedal only existed when dataset
    recording was on."""
    m = {}
    install_fakes(m)
    made = []

    class FakePedal:
        def __init__(self, *a, **k): made.append(self)
        def stop(self): pass
        def is_intervention(self): return False
        def is_healthy(self): return True

    real_pedal, real_monitor = appmod.PedalListener, appmod.EpisodeBoundaryMonitor
    appmod.PedalListener = FakePedal
    try:
        c = client()
        # eval ON, dataset recording OFF, intervention OFF
        r = connect(c, eval_config_name=EVAL_CFG, checkpoint_label="ck")
        assert r.status_code == 200, r.get_json()
        assert len(made) == 1, "eval mode must create a PedalListener"
        assert appmod.state.episode_monitor is not None, \
            "eval mode must create an EpisodeBoundaryMonitor (reward/reset pedals)"
        assert appmod.state.dataset_recorder is None, "no dataset recording was requested"
        print("  eval mode: pedals live, no dataset recorder")
        c.post("/api/disconnect", json={})
    finally:
        appmod.PedalListener, appmod.EpisodeBoundaryMonitor = real_pedal, real_monitor
        restore_fakes(m)
        c.post("/api/disconnect", json={})


def test_plain_inference_still_has_no_pedals():
    """Regression: a non-eval, non-recording session must be untouched."""
    m = {}
    install_fakes(m)
    try:
        c = client()
        connect(c)
        assert appmod.state.episode_monitor is None
        assert appmod.state.pedal is None, "plain inference must not grab the pedal device"
        print("  plain inference unchanged: no pedal, no episode monitor")
        c.post("/api/disconnect", json={})
    finally:
        restore_fakes(m)
        c.post("/api/disconnect", json={})


# ---------------------------------------------------------------------------
# Reset Robot button
# ---------------------------------------------------------------------------
def test_reset_runs_the_same_sequence_as_start():
    """The endpoint must call InferenceLoop.reset_and_sync(), i.e. follower.reset()
    plus the leader re-sync - not a bespoke copy that could drift from Start."""
    m = {}
    install_fakes(m)
    calls = []

    class RecordingFollower(FakeFollower):
        def reset(self):
            calls.append("follower.reset")

    appmod.FollowerClient = RecordingFollower
    try:
        c = client()
        connect(c)
        r = c.post("/api/reset", json={})
        assert r.status_code == 200, r.get_json()
        assert calls == ["follower.reset"], calls
        assert r.get_json()["leader_synced"] is False  # no leader in this session
        c.post("/api/disconnect")
        print("  /api/reset -> follower.reset() via reset_and_sync()")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_reset_refused_while_inference_runs():
    """Resetting mid-rollout would fight the policy for the arm."""
    m = {}
    install_fakes(m)
    try:
        c = client()
        connect(c)
        c.post("/api/start_inference", json={"settle_time_s": 0.0, "skip_reset": True,
                                             "record_video": False, "max_steps": 400, "log": False})
        time.sleep(0.2)
        r = c.post("/api/reset", json={})
        assert r.status_code == 409 and "Stop inference" in r.get_json()["error"]
        c.post("/api/stop_inference")
        r = c.post("/api/reset", json={})
        assert r.status_code == 200, "reset must work once the run has stopped"
        c.post("/api/disconnect")
        print("  refused during a run, allowed after Stop")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


def test_reset_requires_connection():
    m = {}
    install_fakes(m)
    try:
        c = client()
        assert c.post("/api/reset", json={}).status_code == 409
    finally:
        restore_fakes(m)


def test_reset_syncs_leader_when_intervention_is_on():
    """With intervention enabled the leader must be re-synced after the reset,
    or the first autonomous tick can command a large jump (see
    InterventionManager.sync_to_follower)."""
    m = {}
    install_fakes(m)
    synced = []

    class FakeIntervention:
        def sync_to_follower(self, q, goal_time):
            synced.append((list(q), goal_time))

        def tick(self):
            return False

    try:
        c = client()
        connect(c)
        # inject an intervention manager the way a leader-enabled connect would
        appmod.state.infer_loop.intervention = FakeIntervention()
        appmod.state.leader = object()
        r = c.post("/api/reset", json={})
        assert r.status_code == 200 and r.get_json()["leader_synced"] is True
        assert len(synced) == 1, "leader was not re-synced after the reset"
        assert synced[0][1] > 0, "sync must use the configured goal_time"
        print(f"  leader re-synced after reset (goal_time={synced[0][1]}s)")
        appmod.state.infer_loop.intervention = None
        appmod.state.leader = None
        c.post("/api/disconnect")
    finally:
        restore_fakes(m)
        c.post("/api/disconnect")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            print(f"- {fn.__name__}")
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
