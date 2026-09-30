# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Station config schema (YAML -> dataclasses) for the Trossen teleop client.

One station config fully describes a single physical setup: which follower
and leader arm(s) to connect to (single or dual), which cameras to stream,
control-loop rate, safety EE bounds, and randomized-reset ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass
class ArmEndpoint:
    ip: str


@dataclass
class CameraDevice:
    name: str
    serial: str = ""  # empty -> synthetic mock feed for this slot
    # Optional [x_min, y_min, x_max, y_max] pixel box (top-left/bottom-right
    # corners), in coordinates of the camera's NATIVE capture resolution
    # (640x480 for RealSense - see `_RealSenseCamera._CAPTURE_RESOLUTION` in
    # camera_manager.py, NOT `CamerasConfig.resolution`, which is the final
    # target size after crop+resize). If set, the frame is cropped to this
    # box BEFORE being resized down to `CamerasConfig.resolution` - use this
    # to select a region of interest (e.g. to properly frame a task-relevant
    # area for a wrist camera) instead of squashing the whole native frame.
    # None (default): no crop, resize the whole native frame as before.
    roi: list[int] | None = None


@dataclass
class CamerasConfig:
    resolution: tuple[int, int] = (256, 256)
    fps: int = 30
    devices: list[CameraDevice] = field(default_factory=list)


@dataclass
class ControlConfig:
    frequency_hz: int = 20
    # "ee" (default): leader/follower driven in cartesian EE-pose space
    #   (x,y,z,ax,ay,az + gripper), via `set_cartesian_positions()` - our own
    #   scheme, built for sim/robomimic state-format compatibility
    #   (`observation.state` = eef_pos+eef_quat+gripper_qpos, see
    #   `state_dim_per_side`).
    # "joint": leader/follower driven in raw JOINT space (6 arm joints (rad)
    #   + gripper carriage (m)), via `set_all_positions()` - matches
    #   Trossen's OWN official `lerobot_trossen` reference implementation
    #   exactly (`widowxai_follower.py::send_action()`/
    #   `widowxai_leader.py::get_action()`): the leader's raw ABSOLUTE joint
    #   positions are read directly and forwarded as the follower's goal
    #   every tick (safety-clipped per `joint_max_relative_target`), no
    #   delta math, no axis-angle/quaternion conversion anywhere. Only
    #   "absolute" `control_mode` is valid with this - Trossen's own
    #   reference never does delta joint control (see `load_station_config()`
    #   validation). `observation.state`/`action` become the raw 7D joint
    #   vector per side (see `state_dim_per_side`/`action_names`) instead of
    #   the sim eef_pos+quat convention.
    command_space: str = "ee"
    # "delta": follower target = follower's CURRENT pose + the leader's pose
    #   delta since the last tick (matches hil-serl's gym-env pattern). Can
    #   lag behind fast leader motion: each new target is computed from
    #   wherever the follower CURRENTLY is (possibly still mid-move from the
    #   previous tick, not yet caught up), not from the previously INTENDED
    #   target - so tracking error from fast motion doesn't get corrected on
    #   the next tick, it compounds. ONLY valid with `command_space: ee`.
    # "absolute" (default): follower target = the leader's raw CURRENT
    #   absolute pose/joints, forwarded directly (no delta math) - avoids
    #   delta's accumulating lag, and matches Trossen's own reference
    #   implementation (`lerobot_trossen`), which ALWAYS mirrors the
    #   leader's absolute joint positions directly, never a delta. Only
    #   sensible if the leader and follower are mounted with a consistent,
    #   known relationship (ideally the same base frame/mirrored setup) so
    #   the leader's raw coordinates map onto a sensible part of the
    #   follower's reachable workspace - otherwise the follower will just
    #   sit clipped at its `ee_bounds`/`joint_max_relative_target` boundary.
    #   In "ee" `command_space`, the recorded dataset `action` is STILL a
    #   delta either way (see `control_loop.py::_tick()`); in "joint"
    #   `command_space`, the recorded `action` is the exact (post-clip)
    #   absolute joint goal actually sent, matching Trossen's own
    #   `send_action()` return value convention.
    control_mode: str = "absolute"
    # NOT 0.044 (the real hardware max): observed live overshooting to
    # 0.044262 during a fast (0.1s) gripper move faulted the arm ("Joint
    # limit exceeded: expected in range [-0.004000, 0.044000]", idles the
    # motor + drops the TCP connection). hil-serl's own `move_gripper()`
    # clips to exactly [0.009, 0.044] with no margin on this side and can
    # hit the same fault; we leave a small margin here instead. Adjust if
    # your gripper's actual overshoot behavior differs.
    gripper_open: float = 0.04
    # NOT 0.0: the real controller's hard lower limit is -0.004, so commanding
    # exactly 0.0 leaves zero margin - a tiny overshoot faults the arm ("Joint
    # limit exceeded", idles + drops the connection). hil-serl's own
    # `move_gripper()` avoids this the same way, clipping to [0.009, 0.044].
    gripper_closed: float = 0.009
    # `goal_time_s` (used for every regular per-tick move) is DERIVED, not a
    # fixed constant - `goal_time_multiplier / frequency_hz`, exactly
    # matching Trossen's own `lerobot_trossen` reference
    # (`WidowXAIFollowerConfig.min_time_to_move_multiplier`, their own
    # comment: "smaller multiplier -> faster but jerkier; larger -> smoother
    # but more lag", their own default is 3.0/30Hz=0.1s). Our previous fixed
    # `goal_time_s: 0.1` at `frequency_hz: 20` was already `2.0x` the tick
    # period (0.05s) - same ballpark/spirit, just not expressed as a ratio,
    # so it didn't automatically stay sensible if `frequency_hz` changed.
    # Default `2.0` here reproduces that exact previous behavior at 20Hz.
    goal_time_multiplier: float = 2.0
    # Slower, dedicated goal_time for discrete UI/API gripper commands (the
    # "Open/Close Gripper" button, `/api/gripper`) - NOT the fast per-tick
    # `goal_time_s`, which is tuned for small continuous teleop deltas
    # and was observed to overshoot the gripper's open limit when used for a
    # full open/close stroke triggered by a single button press.
    gripper_button_goal_time_s: float = 1.5
    # Safety cap for `control_mode: absolute` + `command_space: ee` ONLY
    # (ignored otherwise). `goal_time_s` assumes small continuous per-tick
    # moves - true during steady, continuous teleop (bounded by human hand
    # speed). But in absolute EE mode the follower target is the leader's
    # raw pose with NO reference to the follower's own position, so the
    # very first tick after enabling teleop (or after any period the
    # leader/follower weren't being tracked together) can be an arbitrarily
    # large jump if the leader happens to be far from wherever the follower
    # currently is. This caps how fast such a jump may be commanded:
    # `goal_time` for that tick becomes
    # `max(goal_time_s, jump_distance_m / absolute_mode_max_speed_m_s)`
    # instead of always just `goal_time_s`.
    absolute_mode_max_speed_m_s: float = 0.3
    # Safety cap for `command_space: joint` ONLY - per-tick per-joint clamp
    # (rad) on how far the follower's target may jump from its OWN current
    # joint position, applied to the 6 arm joints (gripper is clipped
    # separately via `gripper_open`/`gripper_closed`). Mirrors Trossen's own
    # `max_relative_target` safety net (`ensure_safe_goal_position()` in
    # their `widowxai_follower.py`, default 5.0 rad - deliberately loose,
    # meant as a runaway-jump guard rather than a routine per-tick limiter,
    # since normal per-tick deltas during continuous teleop are already tiny
    # by construction). Only matters on the first tick(s) after enabling
    # teleop if the leader/follower joints aren't already closely synced
    # (`sync_to_joints()` at reset should normally prevent this).
    joint_max_relative_target: float = 0.5

    @property
    def goal_time_s(self) -> float:
        return self.goal_time_multiplier / max(self.frequency_hz, 1)


@dataclass
class EEBounds:
    xyz_min: list[float]
    xyz_max: list[float]


@dataclass
class ResetConfig:
    """Two-stage reset, mirroring hil-serl's `windowxai_follower_server.py::reset()`.

    Stage 1 (always): joint-space move to a fixed, known-safe
    `staged_joint_positions` (6 joints + gripper, matches
    `driver.set_all_positions()`'s input shape) - gripper commanded open.

    Stage 2 (only if `randomize_ee` is true): read back the EE pose that
    stage 1 landed on, sample a random offset within +/- `xyz_offset` (and,
    if `randomize_angles` is true, +/- `angle_offset`), then move there via
    cartesian EE-pose control (never joint control) - clipped against the
    station's `ee_bounds` safety box either way.
    """

    staged_joint_positions: list[float]  # 6 joints (rad) + gripper (m)
    randomize_ee: bool = False
    xyz_offset: list[float] = field(default_factory=lambda: [0.025, 0.025, 0.025])
    randomize_angles: bool = False
    angle_offset: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    # Time (s) given to reach the staged position (stage 1) and, if
    # applicable, the randomized EE target (stage 2). Deliberately slower
    # than regular per-tick teleop/policy moves (`control.goal_time_s`) since
    # a reset can be a large jump - matches hil-serl's
    # `windowxai_follower_server.py::reset()`, which uses `goal_time=3.0`.
    goal_time_s: float = 3.0


@dataclass
class DatasetConfig:
    save_root: str = "trossen_real/datasets"


@dataclass
class StationConfig:
    station_name: str
    arm_mode: str  # "single" | "dual"
    follower_ips: dict[str, str]  # side -> ip
    leader_ips: dict[str, str]  # side -> ip
    cameras: CamerasConfig
    control: ControlConfig
    ee_bounds: dict[str, EEBounds]  # side -> bounds
    reset: dict[str, ResetConfig]  # side -> reset config
    dataset: DatasetConfig
    # Where `follower_single_server.py` (the standalone follower-only Flask server)
    # is reachable for this station. The teleop data-collection app talks to
    # it over HTTP instead of owning the follower connection itself.
    follower_server_url: str = "http://127.0.0.1:5060"
    config_path: str = ""

    @property
    def sides(self) -> list[str]:
        return ["single"] if self.arm_mode == "single" else ["left", "right"]

    @property
    def action_dim_per_side(self) -> int:
        return 7  # (dx,dy,dz,dax,day,daz OR joint_0..joint_5) + gripper

    @property
    def action_names(self) -> list[str]:
        if self.control.command_space == "joint":
            # Matches Trossen's own `lerobot_trossen` reference exactly -
            # the gripper carriage IS just another entry in this same 7D
            # vector (`left_carriage_joint` in their naming), not a
            # separately-treated quantity.
            base = [f"joint_{i}" for i in range(6)] + ["gripper"]
        else:
            base = ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]
        if self.arm_mode == "single":
            return base
        return [f"left_{n}" for n in base] + [f"right_{n}" for n in base]

    @property
    def state_dim_per_side(self) -> int:
        # `command_space: joint`: the raw 7D joint vector (6 arm joints +
        # gripper) is recorded AS-IS, matching Trossen's own reference
        # (`<joint>.pos` observation features) - no EE/quaternion conversion
        # applies since there's no cartesian control happening at all.
        if self.control.command_space == "joint":
            return 7
        # `command_space: ee` (default): NOT the same as action_dim_per_side
        # (7) - the recorded observation.state deliberately matches this
        # repo's sim/robomimic convention (robot0_eef_pos(3) +
        # robot0_eef_quat(4) + robot0_gripper_qpos(2) = 9), NOT the raw 7D
        # axis-angle+gripper the SDK/hil-serl use internally - see
        # `dataset_recorder.py::_to_sim_state()` for the conversion. This
        # lets a policy/normalizer trained on this repo's existing sim
        # datasets (see RESIDUAL_LEARNING.md/BC_POLICY_TRAINING.md) load
        # real-hardware data recorded here without a dimension/representation
        # mismatch.
        return 9

    @property
    def state_names(self) -> list[str]:
        if self.control.command_space == "joint":
            # Same physical quantity as the action in joint mode (just
            # read vs. commanded) - Trossen's own reference uses the exact
            # same `<joint>.pos` key for both observation and action.
            base = [f"joint_{i}" for i in range(6)] + ["gripper"]
            if self.arm_mode == "single":
                return base
            return [f"left_{n}" for n in base] + [f"right_{n}" for n in base]
        # Matches this repo's sim/robomimic Panda naming convention exactly
        # (see `get_expected_low_dim_keys()` in
        # `resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`) - "robot0_"/
        # "robot1_" prefixes for dual-arm, NOT "left_"/"right_" (that's only
        # used for `action_names` above, which is our own convention since
        # actions were never tied to the sim format the same way).
        prefix = "robot0_" if self.arm_mode == "single" else None
        base = [
            "eef_pos_0", "eef_pos_1", "eef_pos_2",
            "eef_quat_0", "eef_quat_1", "eef_quat_2", "eef_quat_3",
            "gripper_qpos_0", "gripper_qpos_1",
        ]
        if prefix is not None:
            return [f"{prefix}{n}" for n in base]
        return [f"robot0_{n}" for n in base] + [f"robot1_{n}" for n in base]


def list_available_configs() -> list[str]:
    """Return station config file names (without extension) under configs/."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def load_station_config(name_or_path: str) -> StationConfig:
    """Load a station config by file name (under configs/) or an explicit path."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Station config not found: {name_or_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    arm_mode = raw["arm_mode"]
    sides = ["single"] if arm_mode == "single" else ["left", "right"]

    follower_ips = {side: raw["robots"]["follower"][side]["ip"] for side in sides}
    leader_ips = {side: raw["robots"]["leader"][side]["ip"] for side in sides}

    cam_raw = raw["cameras"]
    cameras = CamerasConfig(
        resolution=tuple(cam_raw.get("resolution", [256, 256])),
        fps=int(cam_raw.get("fps", 30)),
        devices=[
            CameraDevice(
                name=d["name"],
                serial=str(d.get("serial", "")),
                roi=list(d["roi"]) if d.get("roi") is not None else None,
            )
            for d in cam_raw.get("devices", [])
        ],
    )
    for dev in cameras.devices:
        if dev.roi is not None:
            if len(dev.roi) != 4:
                raise ValueError(
                    f"cameras.devices['{dev.name}'].roi must be [x_min, y_min, x_max, y_max] (4 ints), "
                    f"got {dev.roi} in {path}"
                )
            x_min, y_min, x_max, y_max = dev.roi
            if not (x_min < x_max and y_min < y_max):
                raise ValueError(
                    f"cameras.devices['{dev.name}'].roi={dev.roi} in {path} is invalid - need "
                    f"x_min < x_max and y_min < y_max."
                )

    control = ControlConfig(**raw.get("control", {}))
    if control.control_mode not in ("delta", "absolute"):
        raise ValueError(
            f"control.control_mode must be 'delta' or 'absolute', got '{control.control_mode}' in {path}"
        )
    if control.command_space not in ("ee", "joint"):
        raise ValueError(
            f"control.command_space must be 'ee' or 'joint', got '{control.command_space}' in {path}"
        )
    if control.command_space == "joint" and control.control_mode != "absolute":
        # Trossen's own reference (`lerobot_trossen`) never does delta joint
        # control - the leader's raw absolute joint positions are always
        # mirrored directly (safety-clipped by `joint_max_relative_target`).
        # "delta" only exists as a concept for `command_space: ee` (see
        # `ControlConfig.control_mode`).
        raise ValueError(
            f"control.command_space: joint requires control.control_mode: absolute "
            f"(got control_mode: '{control.control_mode}' in {path}) - joint-space delta "
            "control isn't supported (Trossen's own reference always mirrors absolute joints)."
        )

    ee_bounds = {
        side: EEBounds(xyz_min=list(v["xyz_min"]), xyz_max=list(v["xyz_max"]))
        for side, v in raw["ee_bounds"].items()
    }

    reset = {side: ResetConfig(**v) for side, v in raw["reset"].items()}

    dataset = DatasetConfig(**raw.get("dataset", {}))

    return StationConfig(
        station_name=raw["station_name"],
        arm_mode=arm_mode,
        follower_ips=follower_ips,
        leader_ips=leader_ips,
        cameras=cameras,
        control=control,
        ee_bounds=ee_bounds,
        reset=reset,
        dataset=dataset,
        follower_server_url=str(raw.get("follower_server_url", "http://127.0.0.1:5060")),
        config_path=str(path),
    )


def _joint_space_aux_names(sides: list[str]) -> list[str]:
    """Names for the always-present, mode-independent 7D/side joint-space
    auxiliary observations (`observation.joint_pos_raw`/`velocity`/
    `effort`/`acceleration`) - same naming convention as `action_names`'
    `command_space: joint` branch (`joint_0..joint_5` + `gripper`)."""
    base = [f"joint_{i}" for i in range(6)] + ["gripper"]
    if len(sides) == 1:
        return base
    return [f"left_{n}" for n in base] + [f"right_{n}" for n in base]


def _ee_pose_raw_names(sides: list[str]) -> list[str]:
    """Names for the always-present, mode-independent 7D/side raw EE pose
    auxiliary observation (`observation.ee_pose_raw`) - the SDK's native
    axis-angle format (x,y,z,ax,ay,az,gripper), NOT the 9D sim-convention
    quaternion format used by `observation.state` in `command_space: ee`."""
    base = ["x", "y", "z", "ax", "ay", "az", "gripper"]
    if len(sides) == 1:
        return base
    return [f"left_{n}" for n in base] + [f"right_{n}" for n in base]


def build_lerobot_features(config: StationConfig, extra_features: dict | None = None) -> dict:
    """Build the `features` dict passed to `LeRobotDataset.create(...)`.

    Matches the convention used elsewhere in this repo
    (`resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`): image features
    are `dtype="video"` with shape (H, W, C) and names ["height","width","channel"].

    Only cameras with a real `serial` set are included - a camera slot with
    `serial: ""` renders a synthetic mock feed in the web UI (so the layout
    can be previewed/developed without hardware), but that's UI-only: it's
    never a real observation, so it's deliberately excluded from the
    recorded dataset's schema entirely (no `observation.images.<name>`
    feature, no video file, no meta/stats for it).

    `action` (7D/side: dx,dy,dz,dax,day,daz,gripper) and `observation.state`
    (9D/side: eef_pos(3) + eef_quat(4) + gripper_qpos(2), matching this
    repo's sim/robomimic convention) are deliberately DIFFERENT dimensions -
    see `state_dim_per_side` for why.

    Also always includes 5 auxiliary, `command_space`-INDEPENDENT
    observations (7D/side each, follower's own signals - see
    `control_loop.py::_tick()`'s `extra_obs` / `dataset_recorder.py::
    add_frame()`), so a recorded episode always carries every raw signal
    the follower server exposes via `/getstate`, regardless of which single
    representation (`observation.state`) happened to be the PRIMARY
    control-space that tick - pick whichever you actually want at training
    time instead of being stuck with only the one used for control:
      - `observation.joint_pos_raw`: raw 6 joint angles (rad) + gripper (m) -
        same as `observation.state` in `command_space: joint`, but ALSO
        recorded (as a bonus) in `command_space: ee`.
      - `observation.ee_pose_raw`: raw x,y,z,ax,ay,az (axis-angle, NOT
        quaternion) + gripper - the SDK's native EE format, ALSO recorded
        (as a bonus) in `command_space: joint`.
      - `observation.velocity`: joint velocities (rad/s for the 6 arm
        joints, m/s for the gripper carriage).
      - `observation.effort`: total per-joint motor effort (Nm for arm
        joints, N for the gripper carriage) - includes gravity/friction/any
        external load, nonzero even when holding still.
      - `observation.acceleration`: joint accelerations.

    `extra_features`: optional additional feature-spec entries (same shape
    this function's own dict values take - e.g. `{"next.reward": {"dtype":
    "float32", "shape": (1,), "names": ["reward"]}}`) merged into the
    returned dict as-is. Used by the infer app (`observation.intervened`,
    `observation.policy_action`, `next.reward`) and optionally by teleop
    (`next.reward`, only when its pedal-episode-control checkbox is on) -
    no existing caller passes this, so every dataset recorded before this
    parameter existed keeps its exact original schema.
    """
    action_n = len(config.sides) * config.action_dim_per_side
    state_n = len(config.sides) * config.state_dim_per_side
    aux_n = len(config.sides) * 7
    h, w = config.cameras.resolution

    features: dict = {
        "action": {"dtype": "float32", "shape": (action_n,), "names": config.action_names},
        "observation.state": {"dtype": "float32", "shape": (state_n,), "names": config.state_names},
        "observation.joint_pos_raw": {
            "dtype": "float32", "shape": (aux_n,), "names": _joint_space_aux_names(config.sides),
        },
        "observation.ee_pose_raw": {
            "dtype": "float32", "shape": (aux_n,), "names": _ee_pose_raw_names(config.sides),
        },
        "observation.velocity": {
            "dtype": "float32", "shape": (aux_n,), "names": _joint_space_aux_names(config.sides),
        },
        "observation.effort": {
            "dtype": "float32", "shape": (aux_n,), "names": _joint_space_aux_names(config.sides),
        },
        "observation.acceleration": {
            "dtype": "float32", "shape": (aux_n,), "names": _joint_space_aux_names(config.sides),
        },
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
    }
    for cam in config.cameras.devices[:4]:
        if not cam.serial:
            continue
        features[f"observation.images.{cam.name}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        }
    if extra_features:
        features.update(extra_features)
    return features
