# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""LeRobot v2.1 dataset writer for a teleop session.

One `LeRobotDataset` is created per "session" (task name set once, folder
named `{date}_{time}_{task_name}` under the station's `dataset.save_root`).
Each Start/Stop Recording press in the UI records one more episode into that
same dataset (standard v2.1 behavior: one parquet + one mp4 per camera per
episode).

Uses the repo's vendored `deps/lerobot` package (editable-installed as
`lerobot`) directly - same `LeRobotDataset.create/add_frame/save_episode`
API used in `resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation

from trossen_real.config import StationConfig, build_lerobot_features

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")

# Filename (at the session root, sibling to meta/data/videos) the exact
# station config YAML used at recording time is copied to - see
# `start_session()`. Lets `replay_episode.py` (and anything else that needs
# to know exactly how a dataset was recorded) load the config straight from
# the dataset itself instead of trusting a separately-passed `--config`
# name/path, which can silently drift out of sync if the original config
# file is edited (or the wrong station name is passed) after recording.
SAVED_CONFIG_FILENAME = "station_config.yaml"

# The 5 auxiliary, `command_space`-independent observation keys declared by
# `config.py::build_lerobot_features()` - `add_frame()`'s `extra_obs` dict
# must supply exactly these (see there for what each one means).
AUX_OBS_KEYS = ("joint_pos_raw", "ee_pose_raw", "velocity", "effort", "acceleration")

def _safe_task_name(task_name: str) -> str:
    return _SAFE_NAME_RE.sub("_", task_name.strip())


def _to_sim_state(config: StationConfig, raw_state: np.ndarray) -> np.ndarray:
    """Convert the raw per-tick state (7D/side: x,y,z,ax,ay,az,gripper - the
    SDK's/hil-serl's native format) to this repo's sim/robomimic
    `observation.state` convention (9D/side: `eef_pos`(3) + `eef_quat`(4) +
    `gripper_qpos`(2)).

    Why convert at all: hil-serl's own real-hardware WidowXAI gym env
    (`trossen_windowxai.py::_get_obs()`) does NOT convert - it uses the raw
    axis-angle pose directly as `tcp_pose`. But THIS repo's sim-trained
    BC/RL pipeline (`resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`,
    `docs/algorithms/RESIDUAL_LEARNING.md` §12) was built from robomimic/robosuite datasets
    whose `observation.state` is `robot0_eef_pos`(3) + `robot0_eef_quat`(4) +
    `robot0_gripper_qpos`(2) = 9D, with a quaternion orientation - a policy/
    normalizer trained on that expects exactly this shape and
    representation. Recording the SDK's raw 7D axis-angle format instead
    would silently break compatibility (wrong dims, and even if padded to
    the same size, axis-angle numbers fed where quaternion numbers are
    expected would corrupt normalization).

    Quaternion convention: scipy's `Rotation.as_quat()` default is
    `[x,y,z,w]` (scalar-last) - this MATCHES `robot0_eef_quat` in this
    repo's existing sim datasets exactly (robosuite's own `transform_utils`
    outputs xyzw too - see `scratch/stage_reset_summary.md` §B; do NOT
    confuse this with MuJoCo's own internal `qpos` convention, which is
    `[w,x,y,z]` and is unrelated to this observation feature).

    Gripper: our hardware exposes a single continuous gripper position
    (leader/follower have one carriage joint), not two independently
    actuated fingers like a Panda parallel gripper - duplicated into both
    `gripper_qpos` slots as the simplest faithful mapping (adjust here if a
    different convention turns out to matter for policy transfer).

    Actions are NOT touched by this conversion - the 7D delta-pose +
    absolute-gripper action format already matches the sim convention's own
    action space (`get_action_names()` in the same conversion script:
    Δpos(3) + Δrot axis-angle(3) + gripper), which itself already uses
    axis-angle (small deltas, no discontinuity risk), not quaternion.
    """
    raw_state = np.asarray(raw_state, dtype=np.float32)
    chunks = []
    for i in range(len(config.sides)):
        raw = raw_state[i * 7 : (i + 1) * 7]
        pos, axis_angle, gripper = raw[:3], raw[3:6], raw[6]
        quat = Rotation.from_rotvec(axis_angle).as_quat()  # [x, y, z, w]
        chunks.append(np.concatenate([pos, quat, [gripper, gripper]]))
    return np.concatenate(chunks).astype(np.float32)


class DatasetRecorder:
    """Session/episode lifecycle wrapper around `LeRobotDataset`.

    Shared as-is by the infer app (`trossen_real/infer_app/app.py`) - not
    duplicated - via two optional constructor knobs:
      - `save_root`: overrides `config.dataset.save_root` (lets the infer
        app point sessions at `infer_dataset/` instead of teleop's own
        datasets folder) - `None` (default) preserves teleop's exact
        original behavior.
      - `extra_features`: additional `build_lerobot_features()` entries
        (e.g. `observation.intervened`/`observation.policy_action`/
        `next.reward`) - `None` (default) preserves teleop's exact original
        schema. See `add_frame()`'s `extra_features` param for the matching
        per-frame values.
    """

    def __init__(
        self,
        config: StationConfig,
        save_root: str | Path | None = None,
        extra_features: dict | None = None,
    ) -> None:
        self.config = config
        self.save_root = save_root
        self._extra_lerobot_features = extra_features
        self._extra_feature_keys = set(extra_features.keys()) if extra_features else set()
        self.dataset: LeRobotDataset | None = None
        self.task_name: str | None = None
        self.session_dir: Path | None = None
        self._recording = False
        self._episode_frame_count = 0
        # Only cameras with a real serial are ever written to the dataset -
        # a mock (serial="") slot is UI-preview-only, never a real
        # observation, and isn't in the schema `build_lerobot_features()`
        # built either (see there for why).
        self._recorded_camera_names = {cam.name for cam in config.cameras.devices[:4] if cam.serial}

    @property
    def is_session_active(self) -> bool:
        return self.dataset is not None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def num_episodes(self) -> int:
        return self.dataset.num_episodes if self.dataset is not None else 0

    def start_session(self, task_name: str) -> str:
        if self.dataset is not None:
            raise RuntimeError("A recording session is already active; stop it before starting a new one.")
        if not task_name or not task_name.strip():
            raise ValueError("task_name must be non-empty.")

        self.task_name = task_name.strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{_safe_task_name(self.task_name)}"
        save_root = self.save_root if self.save_root is not None else self.config.dataset.save_root
        root = Path(save_root) / folder_name
        root.parent.mkdir(parents=True, exist_ok=True)

        features = build_lerobot_features(self.config, extra_features=self._extra_lerobot_features)
        self.dataset = LeRobotDataset.create(
            repo_id=folder_name,
            fps=self.config.control.frequency_hz,
            root=str(root),
            robot_type=f"trossen_{self.config.arm_mode}",
            features=features,
            use_videos=True,
            image_writer_threads=4,
        )
        self.session_dir = root

        # Copy the EXACT station config YAML used for this session into the
        # session root itself - this is the ground-truth record of how the
        # data was collected (command_space, control_mode, gripper bounds,
        # ee_bounds, etc.), independent of whatever the config FILE ON DISK
        # says later (it may get edited/tuned afterward for the next
        # session) or whatever name/path someone happens to pass to a
        # downstream tool like `replay_episode.py`.
        if self.config.config_path:
            try:
                shutil.copy2(self.config.config_path, root / SAVED_CONFIG_FILENAME)
            except OSError:
                logger.exception(
                    "Could not copy station config into session dir %s - continuing without it "
                    "(downstream tools will have to be told the config explicitly).", root
                )

        logger.info("Started teleop session '%s' at %s", folder_name, root)
        return str(root)

    def start_recording(self) -> None:
        if self.dataset is None:
            raise RuntimeError("No active session; call start_session() first.")
        if self._recording:
            return
        self._recording = True
        self._episode_frame_count = 0

    def add_frame(
        self,
        action: np.ndarray,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        done: bool = False,
        extra_obs: dict[str, np.ndarray] | None = None,
        extra_features: dict[str, np.ndarray] | None = None,
    ) -> None:
        """No-op if not currently recording (call unconditionally from the control loop).

        `command_space: ee` (default): `state` is the raw 7D/side
        (x,y,z,ax,ay,az,gripper) pose - converted to this repo's
        sim-convention 9D/side (`eef_pos`+`eef_quat`+`gripper_qpos`) via
        `_to_sim_state()` before being written.
        `command_space: joint`: `state` is the raw 7D/side joint vector (6
        arm joints + gripper) - written AS-IS, no conversion (matches
        Trossen's own `lerobot_trossen` reference convention).
        `action` is recorded as-is either way (already matches the relevant
        action convention for the active `command_space`).

        `extra_obs`: the 5 auxiliary, mode-independent observations (see
        `AUX_OBS_KEYS`/`config.py::build_lerobot_features()`) - REQUIRED
        (must contain exactly `AUX_OBS_KEYS`) since the schema declares them
        unconditionally for every session; raises `ValueError` if missing or
        incomplete, since that would otherwise surface as a much more
        confusing `LeRobotDataset` "missing features" error deep inside
        `dataset.add_frame()`.

        `images` may include mock-camera entries (for the UI preview) - only
        the real (non-mock) cameras configured with a serial are actually
        written to the dataset, matching the schema `build_lerobot_features()`
        declared at `start_session()` time.

        `extra_features`: values for whatever extra schema entries were
        passed as `extra_features` to `__init__()` (e.g.
        `{"observation.intervened": [...], "next.reward": [...]}`) - REQUIRED
        to supply exactly those keys every frame if any were declared at
        construction time (same "missing keys -> ValueError" contract as
        `extra_obs`/`AUX_OBS_KEYS` below); must be omitted/empty if none were
        declared. `None` (default) is correct for a `DatasetRecorder`
        constructed without `extra_features` (teleop's normal case).
        """
        if not self._recording or self.dataset is None:
            return
        missing = set(AUX_OBS_KEYS) - set((extra_obs or {}).keys())
        if missing:
            raise ValueError(
                f"add_frame()'s extra_obs is missing required keys {sorted(missing)} - must supply all of "
                f"{AUX_OBS_KEYS} every frame (see build_lerobot_features(), which declares them unconditionally)."
            )
        extra_missing = self._extra_feature_keys - set((extra_features or {}).keys())
        if extra_missing:
            raise ValueError(
                f"add_frame()'s extra_features is missing required keys {sorted(extra_missing)} - this "
                f"recorder was constructed with extra_features declaring {sorted(self._extra_feature_keys)}, "
                "which must all be supplied every frame."
            )
        if self.config.control.command_space == "joint":
            # No EE/quaternion conversion in joint mode - the raw 7D/side
            # joint vector (6 arm joints + gripper) IS the state, matching
            # Trossen's own `lerobot_trossen` reference convention exactly
            # (`<joint>.pos` observation features).
            obs_state = np.asarray(state, dtype=np.float32)
        else:
            obs_state = _to_sim_state(self.config, state)
        frame = {
            "action": np.asarray(action, dtype=np.float32),
            "observation.state": obs_state,
            "next.done": np.array([done], dtype=bool),
        }
        for key in AUX_OBS_KEYS:
            frame[f"observation.{key}"] = np.asarray(extra_obs[key], dtype=np.float32)
        for key, value in (extra_features or {}).items():
            # dtype comes from the feature spec given to build_lerobot_features() at
            # start_session() time (e.g. bool for observation.intervened, float32 for
            # next.reward) - NOT hardcoded, so a caller declaring a bool column doesn't
            # get it silently upcast to float32 here.
            spec_dtype = (self._extra_lerobot_features or {}).get(key, {}).get("dtype", "float32")
            frame[key] = np.asarray(value, dtype=spec_dtype)
        for name, img in images.items():
            if img is not None and name in self._recorded_camera_names:
                frame[f"observation.images.{name}"] = img
        self.dataset.add_frame(frame=frame, task=self.task_name)
        self._episode_frame_count += 1

    def stop_recording(self) -> int:
        """Save the current episode (if any frames were recorded) and return its length."""
        if self.dataset is None:
            raise RuntimeError("No active session.")
        self._recording = False
        n_frames = self._episode_frame_count
        self._episode_frame_count = 0
        if n_frames == 0:
            logger.warning("Stop recording called with zero frames buffered - nothing saved.")
            return 0
        # The LAST recorded frame is, by definition, this episode's terminal
        # frame - regardless of WHY recording is stopping right now (reward
        # pedal held->released, reset pedal pressed, max_steps exhausted, a
        # manual Stop button click, or an exception unwinding through a
        # `finally:` that calls this). Force next.done=True on it here,
        # unconditionally, rather than trying to detect "is this tick the
        # last one" inside the control loop for every possible stop reason -
        # straightforward for pedal-driven/max_steps ends (known in advance),
        # NOT reliably knowable in advance for an async button click (can
        # arrive at any point after a frame's already been written). This
        # guarantees the invariant for every stop path with no risk of
        # missing one. Per-frame next.done is otherwise driven purely by
        # next.reward (done wherever reward==1 - see callers' add_frame()) -
        # this is the ONE place that also accounts for "episode ended but
        # reward was never marked" (e.g. reset pedal, timeout with no
        # success shown).
        # `episode_buffer[key]` is a plain Python list up until save_episode()
        # stacks it into an array, so indexing/mutating the last entry here is
        # safe and takes effect in what gets written.
        self.dataset.episode_buffer["next.done"][-1] = np.array([True], dtype=bool)
        self.dataset.save_episode()
        logger.info("Saved episode %d (%d frames) to %s", self.dataset.num_episodes - 1, n_frames, self.session_dir)
        return n_frames

    def stop_session(self) -> None:
        if self._recording:
            raise RuntimeError("Stop recording before ending the session.")
        logger.info("Ended teleop session at %s (%d episodes)", self.session_dir, self.num_episodes)
        self.dataset = None
        self.task_name = None
        self.session_dir = None
