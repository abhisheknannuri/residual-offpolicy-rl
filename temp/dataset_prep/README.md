# Dataset preprocessing: order and tools

How a recorded Station 3 dataset becomes the ACT training dataset and the
residual-RL offline dataset. Every step writes a NEW directory and never
changes its input. Run the dry run of a step before its real pass.

```
record (teleop / infer app)      v2.1, absolute joint actions
  │
  ├─ 1. delta-joint conversion   v2.1   trossen_real/scripts/convert_to_delta_joint_dataset.py
  │      (+ merge of datasets, if any)
  ├─ 2. still-frame thinning     v2.1   temp/dataset_prep/thin_still_frames.py            -> <name>_thin
  ├─ 3. gripper binarization     v2.1   temp/gripper/binarize_gripper_dataset.py           -> <name>_binGrip
  │      (optional: when the robot runs the GripperController)
  └─ 4. v2.1 -> v3               v3     DatasetUtil tools/convert_v21_to_v3.py             -> ACT training
```

- **ACT training** uses the v3 result of step 4.
- **Residual RL on the whole task** uses a **v2.1** dataset, because the RL
  offline-buffer builder reads v2.1 (`offline_data.root` in
  `scripts/TrossenStation3Real/RL/conf/station3.yaml`). Which step's v2.1 it
  should be is still open; see "Residual RL" below.

Steps 2 and 3 are independent of each other and can be left out separately.
Step 3 can also run on a v3 dataset (it reads both); doing it on v2.1 keeps the
v2.1 and v3 versions identical.

## 1. Delta-joint conversion

`action[t] = target[t] - observation.state[t]` for joints 0-5, the gripper stays
absolute. It also trims leading/trailing dead time with its own heuristic on the
action (not used for anything else; mid-episode stillness is step 2).

## 2. Still-frame thinning (`temp/dataset_prep/`)

Drops the middle of long still stretches (holding, hesitating before the
release), so they are a smaller share of the training data.

- **Rule** (`still_rule.py`, one copy, used by the viewer and the script):
  - still = every joint's measured position changes less than its threshold
    between consecutive frames, `|q[t+1, j] - q[t, j]| < joint_thr[j]`
    (observation.state, rad/frame), and the measured gripper less than
    `grip_thr`;
  - only runs of at least `min_run` still frames count;
  - in a run, keep the first `keep_head` and the last `keep_tail` frames, drop
    the rest;
  - never drop within `protect` frames of a gripper label switch.

  Measured positions, not the action: a holding arm's delta action is mostly
  the gravity/tracking offset and looks like motion. Per joint, not the norm:
  a slow single-joint alignment must not hide behind five still joints.
- **Nothing is merged or summed.** Each kept frame keeps its own action (target
  minus its own state), which is still correct without its dropped neighbours.
  Summing deltas would count the holding offset many times, and a multi-step
  delta would be executed in one tick.
- **Choose the thresholds** with the viewer (Python server, the page holds no
  data), then dry run, then the real pass:
  ```bash
  .venv/bin/python temp/dataset_prep/still_explorer.py --dataset <v2.1> \
      --viz-url http://localhost:7860/local/<same episodes in the LeRobot visualizer>
  # open http://127.0.0.1:7871
  .venv/bin/python temp/dataset_prep/thin_still_frames.py --src <v2.1> --dry-run   # report only
  .venv/bin/python temp/dataset_prep/thin_still_frames.py --src <v2.1>             # writes <name>_thin
  ```
  Defaults (chosen with the viewer on ACT Iter1): `joint_thr` 0.002 rad/frame
  for all joints, `grip_thr` 0.5 mm/frame, `min_run` 20, `keep_head` 10,
  `keep_tail` 5, `protect` 10. All can be overridden on the command line.
- **The real pass re-encodes both camera videos** (AV1, only the kept frames;
  slow) and rebuilds timestamps, indices and stats through LeRobot's
  `add_frame` / `save_episode`. The dry-run report checks per episode that
  every gripper label switch survives and that the last frame is kept. It also
  lists the episodes that lose `next.reward` frames: the success hold at the end
  is itself a still stretch.
- `observation.intervened`, `next.done` and every other column are copied
  per kept frame.

## 3. Gripper binarization (`temp/gripper/`)

`action[:, 6]` becomes exactly `closed` (4 mm) or `open` (40 mm): above 16 mm
is open, below 10 mm closed, in between the previous label, and a label shorter
than 3 frames keeps the one before it (operator twitches). The thresholds sit in
the gap of the ACT Iter1 data: every width held for 15+ frames is <= 10 mm or
>= 20 mm, and the recovery episodes release at only 20-29 mm. The same
thresholds are used by `GripperController` on the robot, so they must match.

```bash
.venv/bin/python temp/gripper/binarize_gripper_dataset.py --src <dataset> --dry-run
.venv/bin/python temp/gripper/binarize_gripper_dataset.py --src <dataset>   # writes <name>_binGrip
```

The report replays the controller on the original widths, as if ACT copied the
data exactly: every label switch caught, how many ticks late, the misses.

**An ACT trained on binarized labels must run with the GripperController**:
its output jumps 4 <-> 40 mm, and sent raw that is a full stroke in one tick,
the kind of fast gripper move that faulted the arm before. In the infer app:
station config `control.gripper.mode: binary`, or "binary" in the Connect
panel's Gripper dropdown; Connect refuses `raw` for such a checkpoint (it reads
the checkpoint's action stats from the policy server's `/health`). The RL env
does not use it yet.

The controller itself: `trossen_real/inference/gripper_controller.py`
(`GripperController`, `binarize`, `gripper_from_config`); tests:
`trossen_real/inference/tests/test_gripper_mode.py` (config, checkpoint check,
infer loop with fakes), `temp/gripper/sim_gripper.py` (simulated gripper, fake
ACT and human), `temp/gripper/real_gripper_test.py` (real follower + leader +
pedal, fake ACT).

## 4. v2.1 -> v3

```bash
cd <DatasetUtil>
.venv/bin/python tools/convert_v21_to_v3.py --repo-id <name> --root <folder>
```
Renames `<folder>/<name>` to `<name>_old` (the v2.1 copy) and writes v3 at
`<name>`. Keep the `_old` copy: steps 2 and 3 read v2.1.

## Residual RL

The RL offline buffer is built from a v2.1 dataset. Two things to decide when
it is the whole task (not the stage-2 cut):

- **Gripper**: if the base ACT was trained on binarized labels, the offline
  actions should be binarized the same way, so the buffer's actions match what
  the base policy and controller produce online.
- **Thinning: probably not for RL.** RL learns from transitions (state, action,
  reward, next state). After thinning, the next kept frame can be up to ~1 s
  later than the action it follows, the success hold loses reward frames, and
  the discounting no longer matches the 20 Hz the online env steps at. Use the
  un-thinned v2.1 (after step 1, binarized if needed) unless there is a reason
  to do otherwise.
