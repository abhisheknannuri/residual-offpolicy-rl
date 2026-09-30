# Eye-in-hand calibration (RealSense on a Trossen wrist)

Finds **`T_gripper_cam`** — where the wrist camera sits in the gripper/EE frame.

No ROS 2, no interbotix. Your stack drives the arm with the `trossen_arm` SDK
behind `follower_single_server.py`, so this talks HTTP to that server for poses
and opens the RealSense directly by serial from the station config.

Everything runs in the repo venv. **Call the interpreter directly, never `uv`:**

```bash
.venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand --help
```

## Why the pasted snippet wouldn't have worked

| Snippet | Reality here |
|---|---|
| `InterbotixManipulatorXS` | you use `trossen_arm` via the follower server |
| `aruco.estimatePoseCharucoBoard` | **removed in OpenCV 5** (`.venv` has 5.0.0) |
| `cv2.calibrateHandEye` | **not in this OpenCV build** either |
| factory intrinsics only | fine, but the *board's* square size matters more |

So the board pose uses `matchImagePoints` + `solvePnP`, and the hand-eye solver
is implemented here in numpy — checked against `cv2.calibrateHandEye` (OpenCV
4.11) to **0.000 mm / 0.0000 deg**.

## Recommended workflow (record, then replay)

Pose recording needs **no camera**, so teleop can hold the RealSense while you
bookmark viewpoints. Then stop teleop and let the script drive the arm back.

```bash
# 0. print a board (once). PRINT AT 100% SCALE.
.venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand print-board --out board.png
#    then MEASURE a square with calipers and pass the measured mm as --square-len

# 1. follower server (own terminal)
.venv/bin/python -m trossen_real.follower.follower_single_server \
    --config trossen_station1_single --port 5095

# 2. teleop the arm; press ENTER at each good viewpoint (10-20 of them)
.venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand record-poses \
    --config trossen_station1_single

# 3. STOP teleop (it holds the camera), then replay + solve
.venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand run \
    --config trossen_station1_single --camera cam_right_wrist \
    --motion replay --poses trossen_real/calibration/results/poses_*.json

# 4. check it on the real robot
.venv/bin/python -m trossen_real.calibration.calibrate_eye_in_hand verify \
    --result trossen_real/calibration/results/<result>.json
```

Other motion modes: `--motion manual` (you move the arm from anywhere — teleop or
`curl`; SPACE captures) and `--motion guided` (a preset joint sweep around wherever
the arm starts). The arm is **stiff** — the server exposes no freedrive route, so
it can't be pushed by hand.

## The two things that decide your accuracy

**1. Rotate about several different axes.** Translating the arm tells you nothing
about the camera's rotation, and rotating about a single axis leaves one degree of
freedom unresolved — *while every residual still looks perfect*. That silent failure
is real and reproduced in `tests/test_handeye.py::test_single_rotation_axis_is_flagged_by_residuals`
(60 mm error at a 0.0000 deg residual). So the tools report a **rotation-axis
spread** during capture and in the final report: want **> 45 deg**.

**2. Measure the printed square.** A 2% print-scale error becomes a 2% distance
bias, and reprojection error stays perfect
(`tests/test_charuco.py::test_square_size_error_scales_translation`).

The honest accuracy number is **board-in-base spread**: the board doesn't move, so
every capture must place it at the same spot in the base frame. Want < ~3 mm.

## Files

| File | What |
|---|---|
| `calibrate_eye_in_hand.py` | the CLI: `print-board`, `record-poses`, `run`, `solve`, `verify` |
| `handeye.py` | AX=XB solvers (Park, Tsai) + nonlinear refinement + diagnostics |
| `charuco.py` | board spec, detection, pose, intrinsics (OpenCV 4/5 safe) |
| `results/` | saved calibrations and pose lists (JSON) |

Results keep the **raw captures**, so you can re-solve later without touching the
robot: `solve --captures <file>`.

## Output

```json
{
  "T_gripper_cam": [[...4x4...]],
  "translation_m": [...], "quaternion_xyzw": [...], "rpy_deg_xyz_extrinsic": [...],
  "T_base_target": [[...]],
  "diagnostics": {"target_spread_mm": ..., "rotation_axis_spread_deg": ..., ...},
  "captures": [...]
}
```

`T_gripper_cam` maps a point in **camera** coords to **gripper/EE** coords. The
arm's EE pose from `/getpos` is `(x, y, z, axis-angle)` in the **base** frame, so
a point seen by the camera lands in the base frame as:

```python
p_base = T_base_gripper @ T_gripper_cam @ p_cam
```

It does not depend on capture resolution, so calibrating at 1280x720 stays valid
for the 640x480-with-ROI frames the RL pipeline uses.

## Tests (no hardware needed)

```bash
PYTHONPATH=. .venv/bin/python trossen_real/calibration/tests/test_handeye.py
PYTHONPATH=. .venv/bin/python trossen_real/calibration/tests/test_charuco.py
PYTHONPATH=. .venv/bin/python trossen_real/calibration/tests/test_pipeline_end_to_end.py
```

The last one renders synthetic views of the board from a known camera-in-gripper
transform, runs them through the real detector, JSON and `solve`, and recovers it
to 0.5 mm / 0.04 deg.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Could not open RealSense ...` | teleop/infer app still holds the camera |
| board never detected | wrong `--squares-x/-y` or `--dictionary` for your printout |
| high board-in-base spread | board moved, wrong square size, blurry frames, or FK error |
| `park vs tsai` disagree a lot | not enough rotation variety — record more poses |
| arm faults during replay | `curl -X POST <server>/clearerr` then `/jointreset` |
