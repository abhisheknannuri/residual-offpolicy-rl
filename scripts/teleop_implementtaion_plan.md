# Plan: Trossen Teleop Client (Web UI + Data Collection)

## Decisions locked in with user
- Use resfit's existing venv + vendored `deps/lerobot` (v2.1 API: `LeRobotDataset.create/add_frame/save_episode`). NOT true v3. Follow v2.1 meta/data/video layout exactly (per-episode parquet + per-episode mp4, `meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl}`).
- Single Flask process per station (like hil-serl's `trossen_arm_mujoco/trossen_real/data_collector.py` + `robot_hardware.py` pattern) - NOT a 3-tier split. This one process: connects to leader(s)+follower(s) via trossen_arm SDK directly (no HTTP hop to itself), owns cameras, runs 20Hz control loop thread, records to LeRobot dataset, AND serves the web UI (HTML/CSS/JS + MJPEG video feeds) from the same app.
- HTTP API routes on this Flask app (goal_ee, reset, get_state, connect, gripper) double as the future RL gym-wrapper client interface (reusable remotely later, same as `windowxai_follower_server.py` endpoints) - internally the control loop calls the same underlying Python functions directly (no self HTTP calls).
- Leader arm(s): read directly via `trossen_arm` SDK (gravity-comp/freedrive mode), NOT its own Flask server. Config still stores leader IP(s).
- Max 2 arms per station (single or dual). Config picks `arm_mode: single|dual`.
- Cameras: 4 hardcoded UI slots (2x2 grid), 256x256, no resize. Background capture thread per camera at native rate (~30fps) for smooth MJPEG preview; 20Hz recording loop grabs LATEST available frame each tick (no blocking, skew <50ms accepted).
- Action/state representation (matches repo's ACTION_NORMALIZATION.md + RESIDUAL_LEARNING.md conventions and Trossen SDK's native format): EE pose = `[x,y,z,ax,ay,az]` (meters, axis-angle radians) + gripper (1D, continuous 0.0-0.044m, absolute not delta). Per-arm action = 7D delta pose (dx,dy,dz,dax,day,daz) + absolute gripper target = 7 values. Per-arm state = 7D absolute pose+gripper. Dual arm -> concat to 14D (left first, then right), matching hil-serl's `TrossenHardwareInterface`/`send_follower_command` convention.
- Reset button: ALWAYS resets FOLLOWER(s) to a randomized absolute EE pose (position sampled uniformly in configured xyz min/max box; if `randomize_angles: true` also sample axis-angle in configured range, else fixed default orientation) + opens gripper. Optional config flag `sync_leader_on_reset` (default **false** for safety): if true, leader(s) briefly switch to position mode, drive to the SAME sampled pose as follower, then return to gravity-comp/freedrive for continued teleop. Manual leader realignment by human is the default/fallback path.
- Gripper button: single toggle button per arm; label/icon reflects CURRENT gripper state (open/closed, polled from `/api/get_state`); click commands the opposite state. Used for manual testing / pausing (e.g., holding an object during RL runs).
- Dataset session scope: ONE `LeRobotDataset` created per connect-session, `task_name` set once via a text field before starting, folder name `{date}_{time}_{task_name}` under `teleop_client/datasets/`. Multiple Start/Stop Recording button presses append multiple episodes into that same dataset (standard v2.1 behavior: each episode its own parquet+mp4 files, `save_episode()` per stop).
- No new pyproject dependency needed for Flask (already a resfit dependency). Need to ADD: `pyrealsense2` (or vendor `rs_capture.py`-style wrapper), Trossen's `trossen_arm` SDK (external, likely already installed system-wide per hil-serl precedent - verify).

## Templates / reference code found during research
- `hil-serl/serl_robot_infra/trossen_env/teleoperator/widowxai_leader.py` - direct SDK leader read pattern, delta-pose computation, gravity-comp freedrive mode setup.
- `hil-serl/serl_robot_infra/robot_servers/windowxai_follower_server.py` - Flask endpoint naming/shape precedent (`/pose`, `/getstate`, `/reset`, `/move_gripper`, `/clearerr`).
- `hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/trossen_real/robot_hardware.py` (`TrossenHardwareInterface`) - dual-arm (leader_left/right, follower_left/right) management, `set_mode("teleop"/"policy")`, `get_leader_action()`/`send_follower_command()` 14D layout, background per-camera capture threads, `config.yaml` schema (robots/cameras/control/data/webui sections) - **closest existing precedent to what we're building**.
- `hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/trossen_real/data_collector.py` - single Flask app control-panel precedent (routes `/`, `/status`, `/start`, `/stop`, `/home`, `/delete`; background `main_loop()` thread) - **primary architecture template**.
- `hil-serl/serl_robot_infra/trossen_env/camera/rs_capture.py` + `multi_video_capture.py` - RealSense capture + threaded multi-camera latest-frame-wins queue pattern to reuse/adapt.
- `resfit/lerobot/dataset/convert_robomimic_to_lerobot.py` (lines ~520-606) - exact working `LeRobotDataset.create()/add_frame()/save_episode()` call pattern against the repo's vendored v2.1 `deps/lerobot` - use as the direct code template for `dataset_recorder.py`.
- `resfit/rl_finetuning/reward_models/reward_client.py` - existing Flask client style conventions (health/check pattern) to mirror for any future remote gym-wrapper client.
- `ACTION_NORMALIZATION.md`, `RESIDUAL_LEARNING.md` - confirm 7D single-arm action convention (3 delta pos + 3 delta axis-angle rot + 1 gripper), fps=20.

## Steps

### Phase 1: Config schema + repo scaffolding
1. Create `teleop_client/` folder at repo root with subfolders: `configs/`, `server/`, `static/`, `datasets/` (gitignored).
2. Define YAML station config schema (`configs/trossen_station1_single.yaml`, `configs/trossen_station2_dual.yaml` as examples):
   - `station_name`, `arm_mode: single|dual`
   - `robots.follower` / `robots.leader`: IP(s), flat for single, `{left:{ip}, right:{ip}}` for dual
   - `cameras`: ordered list (name + serial), first 4 used as UI slots; `resolution: [256,256]`, `fps`
   - `control`: `frequency_hz: 20`, gripper open/closed limits
   - `ee_bounds`: per-arm xyz min/max safety clip box (+ optional axis-angle bounds)
   - `reset`: xyz_min/max, `randomize_angles` (default false) + angle_min/max, `sync_leader_on_reset` (default false)
   - `dataset.save_root`

### Phase 2: Hardware interface (depends on Phase 1 schema)
3. `server/hardware_interface.py` - `TrossenHardwareInterface` class (adapt from `robot_hardware.py`): connects to 1-2 leader + 1-2 follower `TrossenArmDriver`s per config; `connect()/disconnect()`, `set_mode("teleop"/"policy")`, `get_leader_action()` (delta pose + abs gripper per arm), `get_follower_state()` (abs pose+gripper per arm), `send_follower_command(action, goal_time)`, `move_to_pose(arm, pose)` for reset, `set_gripper(arm, open|close)`, `health()` (per-arm connection check), safety-box clipping using `ee_bounds`.

### Phase 3: Camera manager (parallel with Phase 2)
4. `server/camera_manager.py` - adapt `rs_capture.py` + `multi_video_capture.py`: one `RSCapture`-like reader per configured camera (resolution 256x256, fps from config), one background thread per camera pushing into a single-slot "latest frame" holder (thread-safe), `get_latest_frame(name)` non-blocking getter, `get_all_latest()` for recording tick, `start()/stop()`.

### Phase 4: Dataset recorder (depends on Phase 1 schema; independent of 2/3)
5. `server/dataset_recorder.py` - wraps vendored `deps/lerobot`'s `LeRobotDataset`. Build `features` dict from `arm_mode` (7D single / 14D dual action+state, per-camera video features from config's camera list, `next.done`). `start_session(task_name)` -> `LeRobotDataset.create(repo_id=folder_name, fps=20, root=.../{date}_{time}_{task_name}, robot_type="trossen_"+arm_mode, features=features, use_videos=True)`. `start_episode()`/`add_frame(action, state, images, done)` (calls `dataset.add_frame(frame=..., task=task_name)`) /`stop_episode()` (calls `dataset.save_episode()`).

### Phase 5: Control loop + Flask app (depends on Phases 2-4)
6. `server/control_loop.py` - background thread, fixed 20Hz tick: if in teleop mode, `get_leader_action()` -> compute follower target (current + delta, clipped to `ee_bounds`) -> `send_follower_command()` -> `get_follower_state()` -> `camera_manager.get_all_latest()` -> if recording flag set, `dataset_recorder.add_frame(...)`; publish latest tick data (state/gripper/connection flags) into a shared thread-safe object for the Flask routes/UI to read.
7. `server/app.py` - Flask app, single process:
   - UI routes: `GET /` (index.html + station dropdown, populated from `configs/*.yaml`), `GET /static/*`.
   - API routes (RL-reusable): `GET /api/stations`, `POST /api/connect {config_name}` (loads yaml, instantiates hardware_interface+camera_manager, starts control_loop thread in "idle/teleop" mode), `GET /api/health` (per-arm + per-camera connection status, polled), `GET /api/get_state`, `POST /api/goal_ee {arm,pose}` (absolute target - for future remote gym wrapper reuse), `POST /api/reset` (randomized follower reset + optional leader sync), `POST /api/gripper {arm,state}` (toggle).
   - Recording routes: `POST /api/start_session {task_name}` (calls `dataset_recorder.start_session`), `POST /api/start_recording` (begin buffering episode + set recording flag), `POST /api/stop_recording` (`save_episode()`, clear flag).
   - Streaming: `GET /video_feed/<cam_name>` MJPEG (`multipart/x-mixed-replace`) reading from `camera_manager.get_latest_frame`.
   - Launch: `python server/app.py --port 5050` (no per-station CLI config arg needed - config chosen at runtime via dropdown/`/api/connect`); run with `threaded=True` (or waitress) to serve concurrent video streams + API polling.

### Phase 6: Web UI (depends on Phase 5 routes existing, can be stubbed/mocked in parallel)
8. `static/index.html` + `style.css` + `app.js`: station dropdown (from `/api/stations`) + Connect button (green/red status dot per arm/camera via `/api/health` polling every ~1s) -> Task name text field + "Initiate Teleop"/start-session button (creates dataset folder) -> 2x2 hardcoded video grid (`<img src="/video_feed/camN">` per slot, no CSS scaling - render at native 256x256) -> Start Recording / Stop Recording buttons (toggle recording flag, episode counter) -> Reset button -> per-arm Gripper toggle button(s) (label reflects live state).

### Phase 7: Wiring & end-to-end state machine (depends on all prior phases)
9. Wire full state machine in `app.js`/`app.py`: `disconnected -> connecting -> connected/idle -> session_active -> recording <-> idle` states; button enable/disable per state; periodic `/api/health` drives connection indicators independently for robots vs cameras.
10. Safety pass: verify `ee_bounds` clipping applied before every `send_follower_command`; verify reset path always opens gripper; verify SIGINT/app-shutdown gracefully disconnects arms (mirror `windowxai_follower_server.py`'s signal handler).

## Relevant files (all NEW)
- `teleop_client/configs/trossen_station1_single.yaml`, `teleop_client/configs/trossen_station2_dual.yaml` - station configs
- `teleop_client/server/hardware_interface.py` - leader/follower SDK control, adapt from `hil-serl/.../robot_hardware.py`
- `teleop_client/server/camera_manager.py` - adapt from `hil-serl/.../camera/rs_capture.py` + `multi_video_capture.py`
- `teleop_client/server/dataset_recorder.py` - LeRobot v2.1 writer, template from `resfit/lerobot/dataset/convert_robomimic_to_lerobot.py` (lines ~520-606)
- `teleop_client/server/control_loop.py` - 20Hz teleop/record loop
- `teleop_client/server/app.py` - single Flask app (UI + API + MJPEG)
- `teleop_client/static/index.html`, `style.css`, `app.js` - web UI

## Verification
1. Unit-level: instantiate `dataset_recorder` against a mock/dummy features dict, record a few synthetic frames, confirm `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/episodes_stats.jsonl`, `data/chunk-000/episode_000000.parquet`, `videos/chunk-000/<cam>/episode_000000.mp4` all produced correctly and loadable via `LeRobotDataset(repo_id=..., root=...)` (mirrors `verify_dataset_ready.py` pattern).
2. Hardware smoke test (single arm first): connect via `/api/connect`, confirm `/api/health` green, manually backdrive leader, confirm follower mirrors motion within safety bounds, confirm `/video_feed/<cam>` renders at exactly 256x256 in browser with no scaling.
3. Full teleop session: set task name, start session, start recording an episode, perform a short manipulation, stop recording, hit Reset, confirm follower moves to a new randomized pose within configured box + gripper opens, repeat for 2-3 episodes, then inspect resulting dataset folder against `dataset_inspector.py`/`verify_dataset_ready.py` tooling from `DatasetUtil/tools/`.
4. Dual-arm config smoke test once single-arm path verified.

## Further Considerations
1. **Discard/re-record episode**: user only asked for Start/Stop Recording, but real usage will produce bad takes. Recommend adding a lightweight "Discard episode" button (clears in-progress buffer instead of calling `save_episode()`) - cheap to add, high practical value. Confirm whether to include in v1 scope.
2. **Hardware-less dev/testing mode**: no simulator exists for this real-robot path. Recommend a `--mock` flag on `hardware_interface`/`camera_manager` (fake poses + synthetic color frames) purely so UI/control-loop/recording logic can be exercised without physical robots during development. Optional, skip if not wanted.
3. **`trossen_arm` SDK + `pyrealsense2` availability**: confirm these are already installed in the target machine's environment (repo currently has neither in `pyproject.toml`/`constraints-uv.txt`) - need to add as dependencies or confirm system-wide install per hil-serl precedent.
