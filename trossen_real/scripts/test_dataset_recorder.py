# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Dev-time sanity check for `DatasetRecorder` using synthetic data (no robot/camera needed).

Records a couple of short dummy episodes into a temp directory using the
single-arm station config, then reloads the result with `LeRobotDataset` to
confirm the meta/data/video files are structurally valid v2.1 output. Also
verifies that mock camera slots (`serial: ""`, UI-preview-only) are
EXCLUDED from the recorded dataset entirely - no feature, no video file, no
meta/stats for them - while a real (non-empty serial) camera slot IS
recorded, even when the control loop passes frames for all 4 configured
slots every tick (mixing mock + real, as it does with a real station).

Also round-trip-checks the axis-angle <-> quaternion conversion used by
`_to_sim_state()` (axis-angle -> quat -> axis-angle should recover the
original rotation, compared via rotation matrices to sidestep any harmless
sign/axis representation differences) across random rotations plus known
edge cases (zero, tiny, exactly pi, near pi).

Run from the repo root (inside the resfit venv)::

    python -m trossen_real.scripts.test_dataset_recorder
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from trossen_real.config import load_station_config
from trossen_real.teleop.dataset_recorder import DatasetRecorder


def _check_axis_angle_quat_roundtrip() -> None:
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(2000):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi, np.pi)
        cases.append(axis * angle)
    # Known edge cases: zero rotation, tiny rotation, exactly pi, near pi.
    cases += [
        np.array([0.0, 0.0, 0.0]),
        np.array([1e-8, 0.0, 0.0]),
        np.array([np.pi, 0.0, 0.0]),
        np.array([np.pi - 1e-6, 0.0, 0.0]),
    ]

    max_err = 0.0
    for aa in cases:
        quat = Rotation.from_rotvec(aa).as_quat()
        aa_back = Rotation.from_quat(quat).as_rotvec()
        # Compare via rotation matrices, not raw vectors - robust to any
        # harmless equivalent-representation differences.
        err = np.abs(Rotation.from_rotvec(aa).as_matrix() - Rotation.from_rotvec(aa_back).as_matrix()).max()
        max_err = max(max_err, err)

    print(f"Axis-angle <-> quat round-trip max rotation-matrix error over {len(cases)} cases: {max_err:.2e}")
    assert max_err < 1e-6, f"axis-angle <-> quat round-trip error too large: {max_err}"


def _dummy_frame_images(cam_names: list[str], resolution: tuple[int, int], t: int) -> dict[str, np.ndarray]:
    h, w = resolution
    images = {}
    for i, name in enumerate(cam_names):
        img = np.full((h, w, 3), fill_value=(t * 5 + i * 30) % 256, dtype=np.uint8)
        images[name] = img
    return images


def main() -> None:
    _check_axis_angle_quat_roundtrip()

    tmp_root = Path(tempfile.mkdtemp(prefix="trossen_real_test_"))
    print(f"Using temp dataset root: {tmp_root}")

    config = load_station_config("trossen_station1_single")
    config.dataset.save_root = str(tmp_root)

    # All devices in trossen_station1_single.yaml are mock (serial: "") by
    # default. Fake a real serial on the first one so this test actually
    # exercises the mock-vs-real split (mirrors a real station where some
    # slots have a real serial and others don't).
    real_cam_name = config.cameras.devices[0].name
    config.cameras.devices[0].serial = "fake_serial_for_test"
    mock_cam_names = [d.name for d in config.cameras.devices[1:4]]

    recorder = DatasetRecorder(config)
    session_dir = recorder.start_session("dummy_task_test")
    print(f"Session dir: {session_dir}")

    # Simulate the control loop, which passes frames for ALL configured
    # slots every tick (mock included, for the UI preview) regardless of
    # which ones are actually real.
    cam_names = [d.name for d in config.cameras.devices[:4]]
    action_dim = len(config.action_names)
    # add_frame() is given the RAW per-tick state (7D/side: x,y,z,ax,ay,az,
    # gripper - the SDK's native format), NOT the 9D sim-convention shape;
    # it converts internally via _to_sim_state() before writing.
    raw_state_dim = 7 * len(config.sides)
    sim_state_dim = len(config.state_names)  # 9D/side after conversion

    n_episodes = 2
    n_frames_per_episode = 10
    aux_dim = 7 * len(config.sides)
    for ep in range(n_episodes):
        recorder.start_recording()
        for t in range(n_frames_per_episode):
            action = np.random.uniform(-0.01, 0.01, size=action_dim).astype(np.float32)
            state = np.random.uniform(-1.0, 1.0, size=raw_state_dim).astype(np.float32)
            images = _dummy_frame_images(cam_names, config.cameras.resolution, t)
            done = t == n_frames_per_episode - 1
            extra_obs = {
                "joint_pos_raw": np.random.uniform(-1.0, 1.0, size=aux_dim).astype(np.float32),
                "ee_pose_raw": np.random.uniform(-1.0, 1.0, size=aux_dim).astype(np.float32),
                "velocity": np.random.uniform(-1.0, 1.0, size=aux_dim).astype(np.float32),
                "effort": np.random.uniform(-1.0, 1.0, size=aux_dim).astype(np.float32),
                "acceleration": np.random.uniform(-1.0, 1.0, size=aux_dim).astype(np.float32),
            }
            recorder.add_frame(action, state, images, done=done, extra_obs=extra_obs)
        n_frames = recorder.stop_recording()
        print(f"Episode {ep}: saved {n_frames} frames")
        assert n_frames == n_frames_per_episode

    # add_frame() must reject an incomplete extra_obs (missing a required
    # key) with a clear error, rather than letting it surface as a
    # confusing LeRobotDataset "missing features" error deep inside
    # dataset.add_frame().
    recorder.start_recording()
    try:
        recorder.add_frame(
            np.zeros(action_dim, dtype=np.float32),
            np.zeros(raw_state_dim, dtype=np.float32),
            images={},
            extra_obs={"joint_pos_raw": np.zeros(aux_dim, dtype=np.float32)},  # missing the other 4 keys
        )
        raise AssertionError("add_frame() should have raised on incomplete extra_obs")
    except ValueError as exc:
        print(f"Correctly rejected incomplete extra_obs: {exc}")
    recorder.stop_recording()

    recorder.stop_session()

    # Reload and sanity-check, mirroring DatasetUtil/tools/verify_dataset_ready.py.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    repo_id = Path(session_dir).name
    dataset = LeRobotDataset(repo_id=repo_id, root=session_dir)
    print(f"Reloaded dataset: num_episodes={dataset.num_episodes}, num_frames={dataset.num_frames}, fps={dataset.fps}")
    print(f"Feature keys: {list(dataset.features.keys())}")

    assert dataset.num_episodes == n_episodes
    assert dataset.num_frames == n_episodes * n_frames_per_episode
    assert dataset.fps == config.control.frequency_hz

    # The real camera IS in the schema and IS in every frame.
    assert f"observation.images.{real_cam_name}" in dataset.features
    sample = dataset[0]
    print(f"dataset[0] keys: {list(sample.keys())}")
    assert f"observation.images.{real_cam_name}" in sample

    # The mock cameras are NOT in the schema and NOT in any frame - despite
    # `add_frame()` above having been called with images for them every tick.
    for cam_name in mock_cam_names:
        assert f"observation.images.{cam_name}" not in dataset.features, (
            f"mock camera '{cam_name}' leaked into the recorded dataset schema"
        )
        assert f"observation.images.{cam_name}" not in sample, (
            f"mock camera '{cam_name}' leaked into a recorded frame"
        )

    # observation.state was converted from the raw 7D/side SDK format to the
    # 9D/side sim convention (eef_pos(3) + eef_quat(4) + gripper_qpos(2)) -
    # verify the shape, the recorded quaternion is a unit quaternion (sanity
    # check the axis-angle -> quat conversion actually ran), and the
    # duplicated gripper_qpos pair match each other.
    state_sample = sample["observation.state"].numpy()
    print(f"observation.state shape: {state_sample.shape}, sim_state_dim expected: {sim_state_dim}")
    assert state_sample.shape[0] == sim_state_dim
    quat = state_sample[3:7]
    quat_norm = float(np.linalg.norm(quat))
    print(f"quat: {quat}, norm: {quat_norm:.6f}")
    assert abs(quat_norm - 1.0) < 1e-4, f"recorded quaternion is not unit-norm: {quat_norm}"
    assert state_sample[7] == state_sample[8], "gripper_qpos_0/1 should be duplicated from the single gripper value"

    # The 5 auxiliary, command_space-independent observations (see
    # config.py::build_lerobot_features()) are always present, regardless
    # of command_space, with the raw 7D/side shape (not the 9D sim-state one).
    for key in ("joint_pos_raw", "ee_pose_raw", "velocity", "effort", "acceleration"):
        feature_key = f"observation.{key}"
        assert feature_key in dataset.features, f"missing auxiliary observation feature '{feature_key}'"
        assert feature_key in sample, f"missing auxiliary observation '{feature_key}' in a recorded frame"
        aux_sample = sample[feature_key].numpy()
        assert aux_sample.shape[0] == aux_dim, f"'{feature_key}' shape {aux_sample.shape} != expected ({aux_dim},)"
    print(f"Auxiliary observations present with shape ({aux_dim},): joint_pos_raw, ee_pose_raw, velocity, effort, acceleration")

    print("OK - dataset_recorder sanity check passed (mock cameras correctly excluded).")
    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
