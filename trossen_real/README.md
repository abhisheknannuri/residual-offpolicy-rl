# Trossen Teleop Client (`trossen_real/`)

**Two pieces, one physical follower robot (single-arm for now):**

- **`follower/follower_single_server.py`** — standalone Flask server, owns
  the follower robot connection. Endpoints and behavior closely mirror
  hil-serl's `serl_robot_infra/robot_servers/windowxai_follower_server.py`
  (same route names/payload shapes). Runs in its own terminal, on its own
  port. This is the **one** place robot-control logic lives — both the
  teleop app and a future RL gym-wrapper are just HTTP clients of it.
- **`teleop/app.py`** — the teleop data-collection web app (camera
  visualization + start/stop episode recording into a **LeRobot v2.1**
  dataset). Reads the leader directly via the SDK
  (`leader/trossen_leader_single.py`, no HTTP), talks to
  `follower_single_server.py` over HTTP (`teleop/follower_client.py`), owns
  cameras + recording + the browser UI.

For **RL training**, you don't run `teleop/app.py` at all — just
`follower/follower_single_server.py` plus your own gym-wrapper (built
separately) that talks to it the same way `app.py` does, via
`follower_client.py` or plain `requests` calls.

This doc explains the architecture, why it's shaped this way, exactly what
each file does, the HTTP API, the dataset schema produced, and how to run
and extend it. See the review-ready planning doc for the original
requirements/decisions trail: `plan-trossenTeleopClient.prompt.md` (chat
history) — this README documents the actual implementation as it stands.

---

## 1. High-level architecture

```
=====================  follower_single_server.py (standalone, own terminal)  =====================
                       ┌─────────────────────────────────────────┐
                       │   follower/follower_single_server.py       │
                       │  - owns the ONE follower robot connection  │
                       │  - hil-serl-shaped routes: /connect_to_robot,│
                       │    /getpos, /getq, /getstate, /pose,        │
                       │    /move_gripper, /jointreset, /clearerr,   │
                       │    /reset, /disconnect_from_robot, ...      │
                       │  - NO leader, NO UI, NO cameras, NO         │
                       │    dataset recording                        │
                       └───────────────┬───────────────────────────┘
                                       │ (in-process calls)
                                       ▼
                       follower/follower_single.py (TrossenFollowerSingle:
                         follower-only engine, safety clipping, two-stage
                         reset, disconnect-to-sleep-position, etc.)

=====================  app.py (teleop data collection, own terminal)  =====================
                       ┌─────────────────────────────────────────┐
                       │           teleop/app.py (Flask)            │
                       │  - serves web UI (static/)                │
                       │  - reads leader directly (SDK, no HTTP)   │
                       │  - talks to follower_single_server.py     │
                       │    over HTTP                               │
                       │  - owns cameras + LeRobot v2.1 recording   │
                       └───────────────┬───────────────────────────┘
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
        ▼                              ▼                                ▼
leader/trossen_leader_single.py  teleop/follower_client.py    cameras/camera_manager.py +
 (direct SDK, freedrive,          (thin HTTP client of         teleop/dataset_recorder.py
  no Flask - just a class)         follower_single_server.py)   (unchanged)
        ▲
        │ driven every tick by
  teleop/control_loop.py (20Hz): read leader delta (direct) -> read follower
  state (HTTP) -> add delta -> send absolute target (HTTP) -> record frame
```

This mirrors hil-serl's own separation: a robot-control HTTP server
(`windowxai_follower_server.py`) that any client can talk to, plus a
directly-read leader (`widowxai_leader.py`, no HTTP). The teleop app and an
RL gym-wrapper are just two different clients of the **same**
`follower_single_server.py` - "adapt the client to the use case", not the
server. An earlier draft merged everything into one Flask process with a
`mode` switch; that made every robot-control fix a two-file (or config-flag)
change to reason about. Splitting the follower server out into its own
process, reused unmodified by both use cases, removes that duplication.
The arm-driver-level SDK wrappers (`RealArm`/`MockArm`) shared by both the
follower engine and the leader class live in a single top-level
`arm_driver.py` so connect/freedrive/cleanup/mock-fallback behavior isn't
duplicated between them.

**Which do I run?**
- **Teleoperated data collection**: start `follower_single_server.py` first
  (own terminal), then `app.py` (own terminal), open the browser.
- **RL training/fine-tuning**: start `follower_single_server.py` only; your
  gym-wrapper (separate module) talks to it directly, no `app.py` involved.


---

## 2. Directory layout

```
trossen_real/
├── config.py                     # YAML -> dataclasses, builds LeRobot `features` dict
├── arm_driver.py                 # shared RealArm/MockArm/make_arm SDK wrappers (real or mock)
├── configs/
│   ├── trossen_station1_single.yaml   # example single-arm station config
│   ├── trossen_station2_dual.yaml     # dual-arm example (not yet supported by
│   │                                    follower_single_server.py/app.py - see §9)
│   └── trossen_station*_single.yaml   # additional single-arm stations
├── follower/
│   ├── follower_single.py        # TrossenFollowerSingle - follower-only control engine
│   └── follower_single_server.py # standalone follower-only Flask server (hil-serl-shaped)
├── leader/
│   └── trossen_leader_single.py  # TrossenSingleLeader - direct SDK leader class, no Flask
├── cameras/
│   └── camera_manager.py         # multi-camera background capture (real or mock)
├── teleop/
│   ├── app.py                    # teleop data-collection Flask app: UI + recording
│   ├── control_loop.py           # 20Hz teleop loop: leader (direct) -> follower (HTTP)
│   ├── dataset_recorder.py       # LeRobot v2.1 session/episode writer + axis-angle->quat state conversion (_to_sim_state)
│   └── follower_client.py        # thin HTTP client for follower_single_server.py
├── static/
│   ├── index.html                # single-page UI
│   ├── style.css
│   └── app.js                    # fetch()-based client, polls /api/health every 1s
├── scripts/
│   ├── test_dataset_recorder.py  # dev sanity check (no robot/camera needed)
│   └── replay_episode.py         # replay a recorded episode's actions against the REAL follower + verify tracking
└── datasets/                     # output: one folder per recording session (gitignored)
```

---

## 3. Station config schema (`configs/*.yaml`)

One YAML file = one physical station. Loaded by `config.py:load_station_config()`.

```yaml
station_name: trossen_station1_single
arm_mode: single              # only "single" is supported by follower_single_server.py/app.py for now

robots:
  follower: { single: {ip: "192.168.1.3"} }
  leader:   { single: {ip: "192.168.1.5"} }

# Where follower_single_server.py is running for this station. app.py (and
# any RL gym-wrapper) talk to it over HTTP - launch follower_single_server.py
# first.
follower_server_url: "http://127.0.0.1:5060"

cameras:
  resolution: [256, 256]      # final target size (H,W) - real cameras are captured natively
                               # (640x480) then optionally cropped (roi) and resized down to this
  fps: 30                     # background capture rate (independent of control rate)
  devices:                    # up to 4; UI renders them in order into a 2x2 grid
    - name: cam_high
      serial: ""              # RealSense serial; "" => synthetic mock feed for this slot
      # roi: [x_min, y_min, x_max, y_max]   # OPTIONAL - pixel box in the native
      # 640x480 capture frame (NOT `resolution` above) - crop to this box BEFORE
      # resizing, e.g. to properly frame a task-relevant region for a wrist
      # camera instead of squashing the whole native FOV down uniformly. See
      # `scripts/select_camera_roi.py` for a small calibration helper (grabs a
      # live frame, either an interactive click-and-drag `cv2.selectROI()`
      # window or a headless save-to-disk-and-inspect workflow, then prints a
      # ready-to-paste YAML snippet + preview images of the resulting crop).

### Camera auto-recovery / health monitoring (`camera_manager.py`)

A RealSense pipeline can get stuck mid-session (`RuntimeError: Frame didn't
arrive within 1000` - a known, usually-transient librealsense issue: USB
bandwidth/power glitches, a hub hiccup, etc.). `CameraManager` handles this
itself instead of requiring someone to notice and manually restart the
process:

- **Auto-reconnect**: after `RECONNECT_AFTER_FAILURES` (5) CONSECUTIVE
  failed reads on a real camera, its pipeline is closed and reopened from
  scratch (with a short `RECONNECT_BACKOFF_S` pause first) - keeps
  retrying indefinitely on a backoff, doesn't give up after some fixed
  number of attempts, so a camera that comes back later (cable reseated,
  USB hub power-cycled) recovers on its own without a server restart.
- **NEVER silently fabricates data**: unlike the initial `start()`-time
  connection attempt (which DOES fall back to a synthetic mock feed if a
  camera never connects at all, so the UI/dev workflow still works without
  hardware), a RUNTIME reconnect deliberately never falls back to a mock -
  `dataset_recorder.py`'s `_recorded_camera_names` is computed from the
  station config's `serial` field, not from whether a camera is currently
  a real vs. mock object, so silently swapping to a mock at runtime would
  mean synthetic frames get recorded into the dataset AS IF they were real
  camera data, with nothing anywhere flagging that it happened. Instead, a
  broken real camera just keeps serving its last genuinely-real frame
  (stale, but real) while `health()` correctly reports it unhealthy.
- **`/api/health`'s `cameras` field now reflects actual frame freshness**
  (within `STALE_AFTER_S`, 3.0s), not just "was constructed at `start()`
  time" (the previous, silently-always-green behavior) - a camera stuck
  reconnecting now correctly shows as unhealthy. `camera_details` (also in
  `/api/health`) gives richer per-camera diagnostics
  (`last_frame_age_s`/`consecutive_failures`/`reconnect_count`) for
  anything that wants to actually MONITOR/ALERT on this (e.g. an RL
  training loop polling the endpoint), not just show a status dot - the UI
  also surfaces this as a tooltip on each camera's status row.
- Log spam fix: a stuck camera used to log a full traceback every single
  failed read (once per second) forever - now logs the traceback once on
  the FIRST failure, then just a short one-line warning when a reconnect
  is actually attempted.
- Verified via mocked cameras that fail then recover (confirms
  auto-reconnect + `health()` correctly returns healthy again, and
  `is_mock()` stays `False` throughout) and cameras that never recover
  (confirms `health()` stays unhealthy and `is_mock()` stays `False`
  forever - never fabricates data - while reconnect attempts keep
  happening on a backoff, not giving up).

control:
  frequency_hz: 20            # control-loop / recording rate (matches repo convention)
  # "ee" (default): cartesian EE-pose control (x,y,z,ax,ay,az+gripper) -
  # our own scheme, built for sim/robomimic state-format compatibility.
  # "joint": raw joint-space control (6 arm joints+gripper), matches
  # Trossen's own official `lerobot_trossen` reference implementation
  # exactly - see "`control.command_space`" below for the full explanation.
  command_space: ee
  # "absolute" (default): follower target = the leader's raw CURRENT
  # pose/joints, forwarded directly - matches Trossen's own reference
  # (always mirrors absolute, never delta). ONLY makes sense if leader/
  # follower share a consistent frame (ideally identical mounting) so the
  # leader's raw coordinates land inside the follower's reachable
  # workspace - otherwise the follower just sits clipped at its
  # ee_bounds/joint_max_relative_target boundary.
  # "delta" (command_space: ee only): follower target = follower's CURRENT
  # pose + the leader's pose delta since the last tick. Can lag behind fast
  # leader motion (each new target is computed from wherever the follower
  # CURRENTLY is, not the previously intended target, so tracking error
  # doesn't get corrected, it compounds).
  # See §4's "`control.command_space`"/"`control.control_mode`" for exactly
  # how the recorded action is computed in each mode.
  control_mode: absolute
  # NOT 0.044 (hardware max): a fast (0.1s) gripper move was observed on
  # real hardware to overshoot to 0.044262, faulting the arm ("Joint limit
  # exceeded", idles + drops the connection). hil-serl's own move_gripper()
  # clips to exactly [0.009, 0.044] with no margin here and can hit the same
  # fault; small margin instead. Adjust if your gripper's overshoot differs.
  gripper_open: 0.04
  # NOT 0.0: the real controller's hard lower limit is -0.004, so commanding
  # exactly 0.0 leaves zero margin - confirmed on real hardware, a tiny
  # overshoot faults the arm ("Joint limit exceeded", idles + drops the
  # connection). 0.009 matches hil-serl's own safe internal gripper clip.
  gripper_closed: 0.009
  # goal_time_s (time given to the arm to reach each commanded pose/joints
  # every tick) is DERIVED as goal_time_multiplier / frequency_hz - matches
  # Trossen's own `lerobot_trossen` reference exactly
  # (`min_time_to_move_multiplier`, their own comment: "smaller multiplier
  # -> faster but jerkier; larger -> smoother but more lag", their default
  # 3.0/30Hz=0.1s). 2.0 here reproduces this repo's previous fixed
  # goal_time_s: 0.1 at frequency_hz: 20 exactly - not a NEW hardcoded
  # constant, a ratio that stays sensible if frequency_hz ever changes.
  goal_time_multiplier: 2.0
  # Slower goal_time for discrete UI/API gripper commands (the Open/Close
  # Gripper button, /api/gripper) - the derived per-tick goal_time above is
  # tuned for small continuous per-tick deltas and overshoots the limit on
  # a full-stroke move.
  gripper_button_goal_time_s: 1.5
  # Safety cap for control_mode: absolute + command_space: ee ONLY (ignored
  # otherwise) - see "`control.control_mode`" below.
  absolute_mode_max_speed_m_s: 0.3
  # Safety cap for command_space: joint ONLY - per-tick per-arm-joint clamp
  # (rad) mirroring Trossen's own `max_relative_target` safety net.
  joint_max_relative_target: 0.5

ee_bounds:                    # safety clip box per arm side, applied to every command
  single: { xyz_min: [...], xyz_max: [...] }

# Two-stage reset (see follower/follower_single.py, mirrors hil-serl's
# windowxai_follower_server.py::reset()):
#   1. Always: joint-space move to `staged_joint_positions` (6 joints + gripper),
#      gripper opened.
#   2. Only if `randomize_ee`: read back the EE pose stage 1 landed on, sample
#      a random offset (+/- xyz_offset, meters), move there via EE-pose
#      (cartesian) control - never joint control.
reset:
  single:
    staged_joint_positions: [0.0, 1.5707963267948966, 1.5707963267948966, -1.5707963267948966, 0.0, 0.0, 0.044]
    randomize_ee: false
    xyz_offset: [0.025, 0.025, 0.025]
    randomize_angles: false      # if true, also randomize axis-angle orientation
    angle_offset: [0.1, 0.1, 0.1]
    goal_time_s: 3.0             # slower than regular moves - a reset is a big jump

dataset:
  save_root: trossen_real/datasets
```

---

## 4. Action / state representation

Internally (SDK, control loop, `/pose`/`/getstate` HTTP), everything uses
the Trossen SDK's native pose format (`[x,y,z,ax,ay,az]`, meters +
axis-angle radians, confirmed directly from the installed `trossen_arm`
SDK's own docstrings) - matching hil-serl's own real-hardware WidowXAI gym
env exactly (`trossen_windowxai.py::_get_obs()` uses this raw pose as-is,
with no conversion). This section describes `command_space: ee` (the
default) - see "`control.command_space`" below for `command_space: joint`,
which uses a completely different (simpler, no-conversion) representation.

| | dim | meaning |
|---|---|---|
| **action** (internal / control loop) | 7 | `[dx, dy, dz, dax, day, daz, gripper]` — first 6 are **deltas** read directly off the leader arm since the last tick (`leader/trossen_leader_single.py::get_action()`); gripper is the leader's **absolute** position (not a delta) |
| **observation.state** (internal / control loop) | 7 | `[x, y, z, ax, ay, az, gripper]` — the follower's **absolute** pose + gripper position |

`control_loop.py` computes the new absolute follower target client-side
(`current_pose + delta`) and sends that to `follower_single_server.py`'s
`/pose` - the server itself does no delta math, exactly mirroring
hil-serl's gym-env pattern (`TrossenWindowxAIEnv.step()`).

**What's actually written to the recorded dataset is different for
`observation.state` (but NOT for `action`)** - see `dataset_recorder.py::_to_sim_state()`:

| | dim | meaning |
|---|---|---|
| **action** (recorded) | 7 | Same as internal, unchanged - `[dx, dy, dz, dax, day, daz, gripper]`. Already matches this repo's own sim/robomimic action convention (`get_action_names()` in `resfit/lerobot/dataset/convert_robomimic_to_lerobot.py`: Δpos(3) + Δrot axis-angle(3) + gripper) - axis-angle deltas are fine for small per-tick rotations, no conversion needed. |
| **observation.state** (recorded) | 9 | `[eef_pos(3), eef_quat(4), gripper_qpos(2)]` - converted from the internal 7D axis-angle format via `scipy.spatial.transform.Rotation.from_rotvec(...).as_quat()`. |

Why the state gets converted but the action doesn't: this repo's
sim-trained BC/RL pipeline (`RESIDUAL_LEARNING.md` §12,
`BC_POLICY_TRAINING.md`) was built from robomimic/robosuite datasets whose
`observation.state` is `robot0_eef_pos`(3) + `robot0_eef_quat`(4) +
`robot0_gripper_qpos`(2) = 9D with a **quaternion** orientation - a
policy/normalizer trained on that shape+representation can't be handed raw
7D axis-angle data (wrong dims, and even zero-padded to 9D the axis-angle
numbers would be nonsense in quaternion slots, corrupting normalization).
The action space, on the other hand, already uses axis-angle deltas on
both sides (sim and here), so no conversion is needed there.

- **Quaternion component order**: scipy's `Rotation.as_quat()` default is
  `[x,y,z,w]` (scalar-last) - confirmed to match `robot0_eef_quat` in this
  repo's existing sim datasets (`scratch/stage_reset_summary.md` §B: robosuite's
  own `transform_utils` outputs quaternions in xyzw too). This is
  deliberately NOT MuJoCo's internal `qpos` convention (`[w,x,y,z]`,
  scalar-first) - that's a different context (setting simulator joint
  state), unrelated to this observation feature; don't conflate the two.
- **Gripper duplication**: our hardware has a single continuous gripper
  position (one carriage joint on leader/follower), not two independently
  actuated parallel fingers like Panda's. `gripper_qpos_0`/`gripper_qpos_1`
  are both set to the same value as the simplest faithful mapping - revisit
  if this turns out to matter for policy transfer.
- **`StationConfig.action_dim_per_side` (7) vs `state_dim_per_side` (9) are
  deliberately different** - unlike most of this file's "action/state are
  symmetric 7D" framing elsewhere, the recorded dataset's two features are
  NOT the same dimensionality, exactly because of the conversion above.
- Verified via `trossen_real/scripts/test_dataset_recorder.py`: reloads
  the recorded dataset and checks `observation.state` is shape `(9,)`, the
  quaternion is unit-norm, and `gripper_qpos_0 == gripper_qpos_1`.

### `control.command_space`: EE (cartesian) vs raw joint control

Configurable per-station via `control.command_space: ee | joint`
(`control_loop.py::_tick()`):

- **`ee` (default)**: cartesian EE-pose control, everything described in
  §4 above (7D pose+gripper action/state internally, converted to a 9D
  sim/robomimic eef_pos+eef_quat+gripper_qpos `observation.state` when
  recorded). This is our own scheme, built for compatibility with this
  repo's existing sim-trained BC/RL pipeline.
- **`joint`**: raw joint-space control - matches Trossen's own OFFICIAL
  `lerobot_trossen` reference implementation
  (`widowxai_follower.py`/`widowxai_leader.py`) exactly: the leader's raw
  ABSOLUTE joint positions (6 arm joints (rad) + gripper carriage (m), the
  gripper is just another entry in this same 7D vector, not a
  separately-treated quantity - matches their `left_carriage_joint`
  convention) are read directly every tick and forwarded as the
  follower's goal, safety-clamped by `joint_max_relative_target` (mirrors
  their own `max_relative_target`/`ensure_safe_goal_position()` - a
  runaway-jump guard, not a routine limiter). NO delta math, NO
  axis-angle/quaternion conversion anywhere - both `action` and
  `observation.state` are recorded as the raw 7D joint vector, exactly
  Trossen's own `<joint>.pos` observation-feature convention. Requires
  `control_mode: absolute` (validated at config-load time) - Trossen's own
  reference never does delta joint control. The recorded `action` is the
  EXACT (post-safety-clamp) goal actually sent that tick, not the
  leader's raw un-clamped intent - matches their own `send_action()`
  return-value convention ("action actually sent is saved in the
  dataset").
- Verified via a mock-based in-process test exercising `follower_single.py::goal_joint()`,
  `leader/trossen_leader_single.py::get_joint_positions()`,
  `control_loop.py::_tick()`'s joint branch, and
  `dataset_recorder.py::add_frame()` writing the raw 7D joint vector
  without the EE/quaternion conversion - not yet exercised against real
  hardware.

### `control.control_mode`: how the leader drives the follower (and what gets recorded)

Configurable per-station via `control.control_mode: delta | absolute`
(`control_loop.py::_tick()`, `command_space: ee` only - `joint` always
uses absolute mirroring, see above):

- **`absolute` (default)**: follower target = the leader's raw current
  pose,
  clipped to the station's `ee_bounds` client-side (matching what the
  follower server will do anyway) BEFORE computing the recorded delta -
  this matters: recording the *unclipped* aspirational target would make
  the dataset's `action` describe a move the follower never actually made
  whenever the leader's pose exceeds `ee_bounds`. Recorded `action[:6]` is
  `clipped_target_pose - current_pose` - i.e. still a **delta**, matching
  the same action-space convention as `delta` mode (and the sim/robomimic
  action convention - see §4), NOT the leader's own tick-to-tick delta
  (that's a different physical quantity once the follower isn't just
  mirroring leader deltas 1:1). Gripper is unaffected either way - always
  the leader's raw absolute gripper reading, recorded unclipped. Matches
  Trossen's own `lerobot_trossen` reference, which ALWAYS mirrors the
  leader's absolute position/joints directly, never a delta.
- **`delta`**: follower target = follower's current pose + the leader's
  pose delta since the last tick. Recorded `action[:6]` is exactly that
  leader delta (since `target - current_pose` and `leader_action[:6]` are
  the same value by construction in this mode).
- **Why `absolute` is now the default**: `delta` mode can visibly lag during fast
  leader motion - each tick's target is computed from wherever the
  follower's tracking *actually* is (which may still be catching up from
  the previous tick), not from the previously *intended* target, so
  tracking error doesn't get corrected on the next tick, it compounds.
  `absolute` mode sidesteps this by never referencing the follower's own
  (possibly lagging) state for the target computation at all, and matches
  Trossen's own proven reference design - but it ONLY makes sense if the
  leader and follower are mounted with a consistent, known relationship
  (ideally an identical base frame/mirrored setup), otherwise the
  leader's raw coordinates won't map onto a useful part of the follower's
  workspace and it'll just sit clipped at its boundary. We already sync
  leader and follower to the same pose at every reset
  (`TrossenSingleLeader.sync_to_joints()`), which is exactly the
  precondition `absolute` mode needs.
- **Safety: `absolute_mode_max_speed_m_s` caps large jumps (absolute mode
  only).** `goal_time_s` (derived, see `control.command_space` above)
  assumes small continuous per-tick
  moves - true for BOTH modes during steady, continuous teleop (bounded by
  human hand speed either way). But `absolute` mode's target has NO
  reference to the follower's own position at all, so the very first tick
  after enabling teleop (or after any period the leader/follower weren't
  being tracked together) can be an arbitrarily large jump if the leader
  happens to be far from wherever the follower currently is - impossible
  in `delta` mode by construction (it only ever looks at the leader's
  tick-to-tick CHANGE, never its absolute position). Fixed: in `absolute`
  mode, the goal_time actually used for that tick is
  `max(goal_time_s, jump_distance_m / absolute_mode_max_speed_m_s)`
  (default cap `0.3` m/s) - small continuous deltas still get the fast
  `goal_time_s`, but a large one-off jump is automatically given
  proportionally more time instead of being commanded just as fast.
- Verified via a mock-based test: `delta` mode's recorded action still matches the raw leader
  delta exactly (unchanged behavior); `absolute` mode's recorded action
  correctly equals `target - current_pose` including when the leader's
  pose is clipped by `ee_bounds` (confirmed the recorded delta reflects
  the *clipped* target, not the raw out-of-bounds one); a large (0.6m)
  jump in absolute mode was confirmed to scale `goal_time` up
  proportionally (`2.0s` at the default speed cap) while a small (1cm)
  continuous delta stayed at the fast `0.1s`, and `delta` mode's goal_time
  is confirmed unaffected regardless of magnitude.

---

## 4a. Reset behavior (two-stage)

Mirrors hil-serl's `windowxai_follower_server.py::reset()` closely:

1. **Always**: joint-space move to `reset.single.staged_joint_positions`
   (6 joints + gripper, via `driver.set_all_positions()`), gripper commanded
   open. Verified live on real hardware with `[0.0, pi/2, pi/2, -pi/2, 0.0, 0.0, 0.044]`
   (the exact hil-serl default).
2. **Only if `randomize_ee: true`**: read back the EE pose stage 1 landed
   on, sample a random offset within `+/- xyz_offset` (and, if
   `randomize_angles: true`, `+/- angle_offset`), then move there via
   **EE-pose (cartesian) control** - never joint control - clipped against
   `ee_bounds` either way.

`/reset` also accepts an optional `{"pose_to_reach": [7]}` body (matching
hil-serl's `reset(pose_to_reach=...)` signature exactly): if given, stage 2
moves directly there instead of randomizing.

Both stages use `reset.single.goal_time_s` (default `3.0`, matching
hil-serl). Stage 1's `staged_joint_positions` already includes the gripper
target as its 7th element, so it's a single blocking move taking exactly
`goal_time_s` (verified live: 3.00s at the default) - an earlier version
made a redundant second gripper call here that doubled this to ~6s; that
duplicate call has been removed. Stage 2 (if triggered) is a second
blocking move of its own, also `goal_time_s`.

**Leader-sync-on-reset moved from the follower engine to `app.py`.** In
this split architecture the follower engine
(`follower/follower_single.py`) has no access to the leader at all - the
leader is a separate class (`leader/trossen_leader_single.py`) used by a
different orchestrator (`teleop/app.py`), potentially on a different
machine entirely. So `app.py`'s `/api/reset` route now does it itself:
stops the control loop (so its background thread can't send stale `/pose`
commands concurrently with the reset - these would otherwise queue up
behind the follower engine's command lock and fire off the instant the
reset finishes, looking like the follower "drifting" on its own right
after a reset), calls `follower.reset()`, reads back the follower's actual
post-reset JOINT positions (`follower.get_state()["q"]`), briefly takes the
leader out of freedrive to move to those same joint angles
(`TrossenSingleLeader.sync_to_joints()` - joint-space, not cartesian, since
both arms are the same model/mounting so matching joint angles gives a
matching EE pose deterministically regardless of how far the leader's
current configuration is from the target; a cartesian/IK-based move was
observed on real hardware to sometimes land near a degenerate
"home"-looking configuration instead when the jump was large), returns it
to freedrive, re-baselines the delta-action reference (`rebaseline()`),
then restarts the control loop. Without the leader sync, the leader stays
wherever freedrive last left it (could be far from the follower's new
pose) while only the follower jumps to the staged position - harmless for
the **delta** xyz/orientation action (only the increment matters), but the
gripper action is **absolute**, so the very next control-loop tick would
otherwise snap the follower's just-reset gripper to whatever the leader's
gripper happens to
read at that instant.

---

## 5. Dataset format (LeRobot v2.1)

We deliberately use the repo's already-vendored `deps/lerobot` (editable
install, package name `lerobot`) rather than pulling in real LeRobot v3 —
avoids having two incompatible `lerobot` packages in one venv, and this
repo's BC/RL training pipeline already expects v2.1. See
`resfit/lerobot/dataset/convert_robomimic_to_lerobot.py` for the sibling
usage this was modeled on.

- **One `LeRobotDataset` per "session"** (`dataset_recorder.start_session(task_name)`),
  folder named `{YYYYMMDD}_{HHMMSS}_{task_name}` under `dataset.save_root`.
- **One episode per Start/Stop Recording press** (`start_recording()` /
  `add_frame()` per tick / `stop_recording()` -> `LeRobotDataset.save_episode()`).
  Standard v2.1 layout results: one parquet + one mp4 per camera per episode,
  proper `meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl}`.
- Features written per frame: `action`, `observation.state`, `next.done`,
  5 always-present auxiliary observations (see below), and one
  `observation.images.<camera_name>` (dtype `"video"`, shape
  `(256,256,3)`, names `["height","width","channel"]`) **only for cameras
  with a real `serial` set**. A mock (`serial: ""`) slot renders a synthetic
  feed in the web UI so the 2x2 layout can be previewed/developed without
  hardware, but is never a real observation - `build_lerobot_features()`
  excludes it from the schema entirely, and `DatasetRecorder.add_frame()`
  drops it from every frame even though the control loop passes images for
  all 4 configured slots every tick. No feature, no video file, no
  meta/stats for a mock camera - checked directly against the reloaded
  dataset's `features` dict.
- **Auxiliary observations (always present, `command_space`-independent)**:
  `observation.state`/`action` only ever reflect ONE representation (the
  active `command_space`'s), but the follower server's `/getstate` always
  returns pose+q+gripper_pos+dq+efforts+accelerations together regardless
  of mode - so every recorded frame also always carries, at zero extra
  network cost (see `control_loop.py::_tick()`):
  - `observation.joint_pos_raw` (7D/side: 6 arm joints (rad) + gripper (m))
    - same as `observation.state` in `command_space: joint`, ALSO recorded
      as a bonus in `command_space: ee`.
  - `observation.ee_pose_raw` (7D/side: x,y,z,ax,ay,az axis-angle + gripper,
    the SDK's raw format - NOT the 9D sim-convention quaternion format used
    by `observation.state` in `command_space: ee`) - ALSO recorded as a
    bonus in `command_space: joint`.
  - `observation.velocity` (7D/side, rad/s for arm joints, m/s for gripper).
  - `observation.effort` (7D/side, Nm for arm joints, N for gripper -
    includes gravity/friction/any external load, nonzero even holding still).
  - `observation.acceleration` (7D/side).

  Pick whichever keys you actually want at training time instead of being
  stuck with only the one representation used for control that session.
  `DatasetRecorder.add_frame()`'s `extra_obs` param requires exactly these
  5 keys (`dataset_recorder.AUX_OBS_KEYS`) - raises `ValueError` if
  incomplete, rather than a more confusing `LeRobotDataset` "missing
  features" error.
- If `stop_recording()` is called with zero buffered frames, nothing is
  saved (avoids empty episodes from an accidental click).

Validated end-to-end with `trossen_real/scripts/test_dataset_recorder.py`
(records synthetic episodes with one real + several mock camera slots,
reloads via `LeRobotDataset(...)`, checks episode/frame counts, fps,
confirms the mock cameras are absent from both the schema and every frame,
confirms all 5 auxiliary observations are present with the right shape,
and confirms `add_frame()` rejects an incomplete `extra_obs`). Also
verified end-to-end through `ControlLoop._tick()` itself (mocked hardware,
both `command_space` modes) - ticks run and the recorded dataset reloads
with all 5 auxiliary features present.

---

## 5a. Replaying a recorded episode against the real robot (`scripts/replay_episode.py`)

Standalone verification tool: replays one recorded episode's `action`
sequence against the REAL follower and reports how closely the robot's
actual motion tracked what was recorded (position/orientation/gripper
error per step, plus mean/max at the end) - the way to answer "did the
robot actually do the motion the dataset says it did."

**No separate reset-pose storage needed.** `/api/start_recording` already
runs the follower's staged/randomized reset *before* `start_recording()`
begins (see §6b), so episode frame 0's `observation.state` already **is**
the post-reset pose for that specific episode (including wherever
`randomize_ee` landed that time) - the script just reads it back and moves
there directly, no extra bookkeeping required anywhere else in the stack.

Sequence: (1) load one episode via `LeRobotDataset(..., episodes=[N])`,
(2) convert frame 0's 9D sim-convention state back to the SDK's native
axis-angle pose (inverse of `dataset_recorder.py::_to_sim_state()`) and
move the follower there directly (slow, default 3.0s - matches the reset
convention for a big one-off jump, not the fast per-tick teleop time),
(3) replay each frame's 7D `action` as `target = follower_current_pose +
action[:6]`, gripper = `action[6]` (clipped) - this is **mode-agnostic**:
works identically whether the episode was originally recorded with
`control_mode: delta` or `absolute` (§4's `control_mode` note), since the
recorded `action[:6]` always means "the delta this tick's command
represented from that tick's observed state" either way, (4) after each
step, diffs the follower's *actual* resulting pose against the dataset's
recorded state for the next frame and reports the error.

**Supports both `command_space` modes.** `ee`: as described above
(delta-applied against the follower's current pose, 9D-state<->7D-pose
conversion via `_sim_state_to_pose7()`, error reported as pos(mm)/ori(deg)/
gripper(mm)). `joint`: the recorded `action` is already the exact absolute
joint goal that was sent that tick (see `control.command_space` above) - no
delta math, `target = action` directly, sent via `/move_to_joint_positions`;
error reported as joint(deg, L2 norm over the 6 arm joints)/gripper(mm).
`--action-source state-delta` only applies to `ee` mode (raises a clear
error if combined with `joint`). Verified end-to-end with a mocked follower
server for both modes (argument validation, start-pose move, per-step
target computation, error metrics) - not yet exercised against real
hardware in `joint` mode.

**`--config` is now OPTIONAL.** `dataset_recorder.py::start_session()`
copies the EXACT station config YAML used at recording time into the
session dir itself (`<dataset>/station_config.yaml`, see
`DatasetRecorder.SAVED_CONFIG_FILENAME`) - this script ALWAYS prefers that
saved copy when present (it's ground truth for how the episode was
actually recorded), regardless of whatever the original config file says
NOW (it may have been retuned since) or whatever `--config` name someone
happens to pass. `--config` is only actually a fallback for datasets
recorded before this feature existed (raises `FileNotFoundError` if
neither a saved copy nor `--config` is available). If BOTH a saved copy
AND `--config` are given, the saved copy still wins, but a mismatch on
anything that changes replay behavior (`command_space`, `control_mode`,
`frequency_hz`, `goal_time_multiplier`) is loudly warned about - this is
exactly the "I modified the config and forgot" failure mode this exists to
catch. Verified (mock): resolves from the saved copy with no `--config`
given, warns and still uses the saved copy on a real mismatch, and falls
back to (and warns about) `--config` when no saved copy exists.

Run (needs `follower_single_server.py` already running + connected for the
same station - see §8; must be the ONLY thing driving that follower, i.e.
not also connected via `app.py`'s teleop UI at the same time):

```bash
python -m trossen_real.scripts.replay_episode \
    --dataset trossen_real/datasets/<session_dir> \
    --episode 0 \
    --max-steps 20   # start small - see SAFETY note below
```

- `--dataset` is the session **folder** (contains `meta/`, `data/`,
  `videos/`), not a path to a specific parquet file.
- `--config` is normally NOT needed (see above) - only pass it for a
  dataset recorded before the saved-config feature existed.
- `--max-steps N` replays only the first N actions - use this for a first,
  careful look before replaying a whole episode unattended.

**SAFETY**: this drives the real robot through an entire recorded episode
automatically, at speed, with no human pressing anything per step. Stay at
the robot able to hit an E-stop / kill the process; start with
`--max-steps` on a short episode.

Verified (mock-only, pure Python - NOT yet run against real hardware): the
axis-angle↔quaternion pose round-trip used to read back the start pose,
and the position/orientation/gripper error computation against known
synthetic offsets.

**Fixed a real timing bug found on the first real-hardware run**: each
step used to make TWO `get_state()` HTTP calls - one to compute the
delta target, one (extra, unaccounted) purely for the tracking-error
comparison, made *after* the pacing sleep. That second call's latency
wasn't included in the loop's `elapsed`/`sleep` timing, so the ACTUAL
achieved step period silently drifted from the intended one. Fixed: one
`get_state()` call per step, reused for both the next step's delta
baseline AND this step's error comparison. Also now tracks and prints
the ACHIEVED control frequency per step and the average at the end.

**A follow-up attempt to fix the remaining steady ~15-28mm error by
switching to `blocking=True` made things WORSE, not better - reverted.**
The `trossen_arm` SDK's own `set_cartesian_positions()` docstring is
explicit: `blocking` - "Whether to block until the goal positions are
reached, default is true." The reasoning seemed sound (with
`blocking=False`, each `/pose` call returns the instant the controller
*accepts* the command, not when it's *reached* it, so with
`goal_time_s=0.1s` per move but only a ~0.05-0.055s tick period, every
step's target gets sent before the follower finishes the PREVIOUS
one) - but confirmed live on real hardware, forcing `blocking=True`
caused RUNAWAY divergence (74mm growing to 300mm within ~20 steps) and
a real joint-limit fault (`500` from `/pose`), not an improvement. Root
cause of *that*: the recorded `action` in delta mode is the **leader's**
raw tick-to-tick delta, not what the follower actually achieved - the
ORIGINAL recording's follower systematically under-travels each delta
in real time (that's the same overlapping-move "lag" that's normal
during live teleop too). Forcing every replayed step to FULLY complete
its recorded delta means the follower travels FURTHER per step than it
ever actually did during recording - and that overshoot compounds with
nothing to correct it, unlike the original recording/live deployment
scenario where the SAME damping applies to every tick consistently.
**Reverted to non-blocking by default** (`--blocking` flag still
available to reproduce/experiment with the divergence, off by default) -
this matches how the live teleop control loop actually runs AND how this
action sequence will actually be consumed by a deployed policy (both
non-blocking, ~20Hz, overlapping moves), so it's the representative way
to replay it. **The remaining steady, bounded tracking gap (tens of mm)
seen with non-blocking replay is very likely an expected, real
characteristic of this control scheme - not a bug** - the same
real-time tracking lag exists during live teleoperation too.

---

## 6a. HTTP API — `follower/follower_single_server.py` (robot control, port 5060 by default)

No `/api/` prefix - matches hil-serl's `windowxai_follower_server.py` route
names/payload shapes exactly:

| Route | Method | Purpose |
|---|---|---|
| `/connect_to_robot` | POST | Connect the follower |
| `/disconnect_from_robot` | POST | Move to staged->sleep position, then release the connection |
| `/configure` | POST | Re-run the safe staged-position reset |
| `/getpos` | POST | `{"pose": [x,y,z,ax,ay,az]}` |
| `/getq` | POST | `{"q": [6 joints + gripper]}` |
| `/get_gripper` | POST | `{"gripper": float}` |
| `/getstate` | POST | `{"pose":[6], "q":[7], "gripper_pos":float, "dq":[7], "efforts":[7]}` |
| `/pose`, `/move_to_ee_pose` | POST | body `{"arr":[7], "blocking"?, "min_time_to_move"?}` — absolute target `[x,y,z,ax,ay,az,gripper]` |
| `/move_gripper` | POST | body `{"gripper_pos":float, "blocking"?, "min_time_to_move"?}` — arbitrary continuous position |
| `/jointreset` | POST | Two-stage reset (see §4a) |
| `/clearerr` | POST | Same as `/jointreset` (matches hil-serl: a full reset, not a light error clear) |
| `/reset` | POST | body optional `{"pose_to_reach": [7]}` |
| `/health` (GET) | — | Our own addition (not in hil-serl), for convenience |

Connection-required routes return HTTP 409 with an `error` message if
called before `/connect_to_robot`.

## 6b. HTTP API — `teleop/app.py` (teleop UI, port 5050 by default, `/api/` prefix)

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI (`static/index.html`) |
| `/api/stations` | GET | List config files under `configs/` |
| `/api/connect` | POST `{config_name}` | Load config, connect the leader directly + tell `follower_single_server.py` to connect, start cameras + control loop (state-polling only - see §7's teleop-active note) |
| `/api/disconnect` | POST | Stop control loop/cameras, park the leader through the follower's staged->sleep joint waypoints then disconnect it + tell `follower_single_server.py` to disconnect (blocked if a session/recording is active) |
| `/api/health` | GET | Leader/follower/camera connection status (`cameras`, now reflects whether a camera has ACTUALLY produced a frame recently, not just whether it was constructed - see below), richer per-camera diagnostics (`camera_details`), session/recording/`teleop_active` state, tick counter, control-loop error (if it auto-stopped), ACTUAL achieved control frequency (`actual_frequency_hz`, rolling-window measured, vs. `target_frequency_hz` from config) — polled by the UI every ~1s |
| `/api/get_state` | GET | Latest follower state (7D) + gripper open/closed |
| `/api/reset` | POST | Manual reset: staged/randomize move on the follower + leader joint sync, same sequence `/api/start_recording` runs automatically per episode (see §7) |
| `/api/gripper` | POST `{open?}` | Toggle (or set) the gripper via `follower_single_server.py`'s `/move_gripper`; toggling reflects current live state. No-op while `teleop_active` (the leader's absolute gripper position would override it next tick) - UI disables the button in that window |
| `/api/start_session` | POST `{task_name}` | "Initiate Teleop": create the dataset folder for this session AND switch the control loop into actively driving the follower from the leader (`teleop_active=True`) |
| `/api/stop_session` | POST | "End Session": switch the control loop back to state-polling only (`teleop_active=False`), then end the session (must not be recording) |
| `/api/start_recording` | POST | Runs the same reset as `/api/reset` first (fresh known pose per episode, mirroring a gym env's `reset()`), then begins buffering the new episode |
| `/api/stop_recording` | POST | Save the episode (or discard if empty) |
| `/video_feed/<cam_name>` | GET | MJPEG stream (`multipart/x-mixed-replace`), ~15fps to the browser regardless of camera capture rate |

An RL gym-wrapper (its own module, not part of this repo) just talks
directly to `follower_single_server.py` - it never goes through `app.py`:

```python
import requests
BASE = "http://<station-pc-ip>:5060"
requests.post(f"{BASE}/connect_to_robot")
obs = requests.post(f"{BASE}/getstate").json()
requests.post(f"{BASE}/pose", json={"arr": [x, y, z, ax, ay, az, gripper]})
requests.post(f"{BASE}/reset")
```
(or reuse `trossen_real/teleop/follower_client.py`'s `FollowerClient` class directly.)

---

## 7. Key design decisions (and why)

- **The follower has exactly one owner: `follower/follower_single_server.py`.**
  Both the teleop app and a future RL gym-wrapper are HTTP clients of it,
  never owners. This avoids the "fix it in two places" problem an earlier
  merged-server draft had, and matches hil-serl's own
  `windowxai_follower_server.py` design directly.
- **The leader is read directly via the SDK, never through Flask.** Only
  `app.py`'s control loop needs the leader's pose, so there's no HTTP hop
  for it - `leader/trossen_leader_single.py`'s `TrossenSingleLeader` wraps
  the same `RealArm`/`MockArm` classes (`arm_driver.py`) used for the
  follower, so connect/freedrive/cleanup/mock-fallback behavior is shared,
  not duplicated.
- **`follower/follower_single.py`'s `TrossenFollowerSingle` is the shared
  engine, `follower/follower_single_server.py` is just a thin
  hil-serl-shaped route layer on top of it.** All the hardware-tested
  logic (safety-box clipping, two-stage reset, disconnect-to-sleep-position,
  per-call `goal_time`/`blocking` overrides) lives in one place and is
  exercised identically regardless of which route called it.
- **Client computes the delta, server only does absolute moves.** Matches
  hil-serl's gym-env pattern exactly: `control_loop.py` reads the
  follower's current pose over HTTP, adds the leader's delta, and sends
  the new absolute target back - `follower_single_server.py` never does
  delta math itself, keeping its API simple and RL-friendly (a policy
  naturally outputs absolute or delta actions the same way).
- **Mock hardware/camera fallback is automatic.** `arm_driver.py` and
  `camera_manager.py` try to import `trossen_arm` / `pyrealsense2`; if
  unavailable (or a camera slot's `serial` is `""`), they transparently
  substitute an in-process mock so the entire stack is testable without
  hardware. `/health` on both servers reports `using_mock_hardware`.
- **Cameras run faster than the control loop, decoupled.** Each camera has
  its own background thread capturing at its native rate (config `fps`,
  e.g. 30). The 20Hz control loop just grabs whichever frame is currently
  latest — no blocking, small (<50ms) timestamp skew accepted. The browser
  preview is further throttled to ~15fps independent of both.
- **A re-entrant lock (`TrossenFollowerSingle._cmd_lock`) guards every
  arm-commanding call** on the follower server side, so a UI-triggered
  Reset/gripper click can never race a concurrent `/pose` call.
- **Leader-sync-on-reset lives in `app.py`, not the follower engine** — in
  this split architecture the follower engine has no access to the leader
  at all (see §4a), so `_do_reset()` (shared by `/api/reset` and
  `/api/start_recording`) temporarily disables `teleop_active`, does the
  leader realignment itself (`TrossenSingleLeader.sync_to_joints()`,
  joint-space) using the follower's actual post-reset joint positions as
  the target, then restores whatever `teleop_active` was before the call.
- **The control loop only actively drives the follower while
  `teleop_active` is True** (toggled by `/api/start_session`"Initiate
  Teleop"/`/api/stop_session` "End Session") - state polling (the
  snapshot backing `/api/get_state`/`/api/health`, used for the gripper
  button's live display and camera preview) runs continuously from connect
  to disconnect regardless. This used to run unconditionally from
  `/api/connect` onward, which meant the leader's continuously-streamed
  **absolute** gripper position silently overrode the manual gripper
  button within one 50ms tick no matter when it was clicked - the button
  looked broken. Now the leader is only read (and the follower only
  commanded) inside the Initiate-Teleop/End-Session window; outside it,
  manual controls (gripper button, Reset) work normally, and the UI
  disables the gripper button while `teleop_active` since it would
  otherwise be a no-op.
- **Gripper button shows the ACTION a click will perform, not just the
  current state** — if the gripper currently reads open, the button reads
  "Close Gripper" in red; if closed, "Open Gripper" in green. Updates every
  health poll (~1s) so it tracks live state (e.g. after a reset opens the
  gripper, or an RL policy closes it). Disabled while `teleop_active` (see
  above).
- **Gripper safety margins on BOTH extremes, not just the closed side.**
  The real controller's hard limits are `[-0.004, 0.044]`; commanding
  exactly `0.0`/`0.044` leaves zero margin, and a tiny overshoot **actually
  faulted the arm on real hardware** at both ends during testing - closed
  side: `Joint limit exceeded` at `0.0`; open side: overshot to `0.044262`
  during a fast (`goal_time_s=0.1`) button-triggered full-stroke move,
  same fault. `gripper_closed: 0.009` matches hil-serl's own safe internal
  clip; `gripper_open: 0.04` is our own added margin (hil-serl itself clips
  to exactly `0.044` with no margin there and can hit the same fault) -
  adjust either if your gripper's actual overshoot behavior differs.
- **Discrete gripper-button commands use a slower, dedicated goal_time
  (`gripper_button_goal_time_s`, default `1.5`s), not the fast per-tick
  `goal_time_s` (`0.1`s).** The fast value is tuned for small continuous
  teleop deltas each tick; reusing it for a full open/close stroke
  triggered by a single button click is what caused the overshoot fault
  above. `/api/gripper` passes this explicitly and blocks until the move
  completes, rather than relying on the follower server's fast default.
- **Follower/leader disconnect is best-effort at every stage, never
  all-or-nothing.** `follower/follower_single.py::disconnect()` tries the
  mode switch, staged move, and sleep move as three INDEPENDENT
  try/excepts - if the staged move fails partway (transient comms error),
  the sleep/home move is still attempted rather than skipped, so the arm
  isn't left sitting at some random pose with torque cut. `app.py`'s
  `/api/disconnect` parks the leader through the SAME staged->sleep joint
  waypoints before releasing it
  (`TrossenSingleLeader.disconnect(park_waypoints=[staged, sleep])`) - this
  does leave freedrive and will fight the operator's hand if they're
    still holding the leader at disconnect time, which is an accepted
    tradeoff over leaving it wherever freedrive last let it rest. If the
  driver itself is unresponsive (all move attempts fail), the connection is
  still released rather than hanging - there's nothing more that can be
  done in that case.
- **The control loop stops itself after `MAX_CONSECUTIVE_FAILURES` (5)
  consecutive tick failures** (e.g. follower server unreachable) instead of
  silently retrying forever - `ControlLoop.get_error()` then reports why,
  surfaced through `/api/health`'s `control_loop_error` field and shown in
  red in the UI's tick-status line.

---

## 8. Running it

**Teleoperated data collection — two terminals:**

```bash
# Terminal 1: the robot-control server (start this first)
cd residual-offpolicy-rl
source .venv/bin/activate
python -m trossen_real.follower.follower_single_server --config trossen_station1_single --port 5060

# Terminal 2: the teleop UI
cd residual-offpolicy-rl
source .venv/bin/activate
python -m trossen_real.teleop.app --port 5050
# open http://localhost:5050
```

In the UI: pick a station config -> **Connect** (leader/follower connected,
but not yet actively linked - gripper button and **Reset** work freely
here for manual setup) -> enter a task name -> **Initiate Teleop** (now the
leader's deltas actively drive the follower every tick) -> **Start
Recording** (runs the same reset as the **Reset** button automatically,
then buffers frames) -> **Stop Recording** -> repeat Start/Stop Recording
per episode -> **End Session** when done (stops the leader from driving
the follower; gripper button/Reset work freely again) -> **Disconnect**.

**RL training/fine-tuning — just the robot-control server:**

```bash
cd residual-offpolicy-rl
source .venv/bin/activate
python -m trossen_real.follower.follower_single_server --config trossen_station1_single --port 5060
# add --auto-connect to skip the explicit POST /connect_to_robot call
```

Then, from your gym-wrapper (own module, not part of this repo yet):

```python
from trossen_real.teleop.follower_client import FollowerClient
client = FollowerClient("http://<station-pc-ip>:5060")
client.connect()
state = client.get_state()
client.move_to_ee_pose([x, y, z, ax, ay, az, gripper])
client.reset()
```

Dev sanity check (no robot/camera required):

```bash
python -m trossen_real.scripts.test_dataset_recorder
```

Replay a recorded episode against the real robot (see §5a for full details/safety notes) - needs `follower_single_server.py` already running + connected. `--config` is normally NOT needed (the exact config used to record is saved inside the dataset itself and preferred automatically):

```bash
python -m trossen_real.scripts.replay_episode \
    --dataset trossen_real/datasets/<session_dir> --episode 0 --max-steps 20
```

## 9. Known gaps / before using on real hardware

- **Dual-arm is not supported by `follower_single_server.py`/`app.py` yet**
  - only `arm_mode: single`. A `trossen_station2_dual.yaml` example remains
  for reference but neither server will load it (raises a clear error).
- Fill in real IPs and camera serial numbers in the example configs before
  connecting.
- No "discard episode" button yet (only Start/Stop Recording, save-or-skip
  based on frame count) — flagged as an easy follow-up if you want it.
- `TrossenSingleLeader` has been verified against real hardware only in
  isolation (connect/read); a full live teleop drag-test (physically moving
  the leader and confirming the follower tracks it) needs to be done by a
  human at the robot, since nothing here can move the leader itself.

---

## 10. Verified against real hardware

Both `follower_single_server.py`'s full route set (`connect_to_robot`,
`getstate` with the new `q`/`dq`/`efforts` fields, `jointreset`,
`move_gripper`, `pose`, `disconnect_from_robot`) and the full `app.py`
control-loop integration (leader read -> follower HTTP round-trip ->
recording) have been exercised end-to-end - the former directly against
the real robot, the latter against a real `follower_single_server.py`
process with a mocked leader (no physical leader arm was moved by these
tests). Also caught and fixed on real hardware:

- A `trossen_arm` driver/controller **firmware version mismatch**
  (`LogicError: major and minor versions ... must match`) - the venv had a
  newer driver (`1.11.0`) than the controller's firmware (`1.10.0`); fixed
  by pinning the venv to `trossen_arm==1.10.0` to match.
- The `gripper_closed` safety margin bug described in §7.

---

## 11. Shutdown & resource cleanup

Both `follower_single_server.py` and `app.py` install explicit handlers for
`SIGTERM`, `SIGINT`, and `SIGHUP` (in addition to `atexit`), because
`atexit` callbacks only run on a *normal* interpreter exit or a caught
`KeyboardInterrupt` — `SIGTERM` (what `kill <pid>` sends by default, and
what many "kill terminal" actions send) and `SIGHUP` do **not** trigger
`atexit` on their own. Without this, killing the terminal or `kill <pid>`
would silently skip cleanup.

On any of those signals, both processes run the same graceful shutdown:
`follower_single_server.py` moves the follower to the staged->sleep
position then releases the connection (`TrossenFollowerSingle.disconnect()`,
which calls `driver.cleanup()`, not just a state-flag flip); `app.py`
disconnects the leader directly and tells `follower_single_server.py` to
disconnect too. Safe to run twice (e.g. once via an explicit route, again
via the signal handler at exit) — every `disconnect()`/`close()` involved
is idempotent.

Verified live: sent a real `SIGTERM` to a running server process and
confirmed the shutdown log line + full process exit (checked with `pgrep`
that nothing was left running).

**`app.py`'s shutdown is bounded even if a connect attempt is stuck.**
`/api/connect` holds `state.lock` for its ENTIRE duration - and a
wrong/unreachable leader/follower IP can block inside the SDK's own
connect call for many seconds (see below). `_cleanup()` used to do a plain
`with state.lock:`, so a stuck connect attempt also blocked the shutdown
signal handler from ever acquiring that same lock - Ctrl+C/`kill` appeared
to do nothing at all (repeated presses just re-logged "received signal",
never actually exiting), confirmed on real hardware with a typo'd leader
IP. Fixed: `_cleanup()` now does `state.lock.acquire(timeout=3.0)` - if it
can't get the lock in time (nothing was connected yet anyway if a connect
is still in progress, so there's nothing unsafe about skipping the
graceful disconnect), the signal handler forces immediate termination via
`os._exit(1)` instead of waiting - necessary because Flask's per-request
threads aren't daemon threads, so a stuck one would otherwise keep the
whole process alive even after `sys.exit()`.

**Connect attempts also fail fast now instead of hanging for up to 40s.**
`RealArm.connect()` used to call `configure()` with its own 20s-default
timeout, and (from an earlier fix) unconditionally retried once on ANY
failure - doubling the wait to ~40s for a genuinely wrong/unreachable IP,
which isn't a transient issue retrying would help with. Now: `configure()`
is called with an explicit `timeout=5.0` (`RealArm.CONNECT_TIMEOUT_S`,
still generous - real connects complete in well under 1s), and the retry
only triggers if the exception message contains the specific known
transient `"Connection reset by peer"` signature; a genuine timeout raises
immediately instead. The web UI's Connect button also turns into "Cancel
connecting..." while a connect is in flight (client-side `AbortController`)
- note this only stops the BROWSER from waiting; the server-side attempt
already in flight can't be interrupted mid-call and keeps running in the
background (now bounded to a few seconds either way thanks to the shorter
timeout) rather than being truly killed.

**Hard limit that no code can work around:** `kill -9` (`SIGKILL`) cannot
be caught by any process, ever — the OS terminates it immediately, no
Python code runs. If that happens, background threads are daemon threads
so nothing hangs, the OS reclaims all sockets/file descriptors on its own,
and the arm simply holds its last commanded pose (a hardware/firmware
property, not something this codebase controls).

Never run `app.py` against a `follower_server.py` for a **different**
station's follower, and never run two `follower_server.py` processes
against the same physical arm at the same time.

