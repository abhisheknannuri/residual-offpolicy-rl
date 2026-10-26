# Stage-2 residual RL, Station 3: dataset split and reset (2026-10-22)

Residual RL trained only from SARM stage 2 ("cube above the hole") to the end
of the task. This needs (1) an offline dataset that starts at stage 2 and (2)
a reset that puts the robot into stage 2 with the cube in the gripper.

## 1. The stage-2 RL dataset

**Output:** `trossen_real/datasets/iterativeTraining/TrossenStation3/RL_Stage2/PickCubeAndInsert_TS3_Stage2_RL_deltajoint_v21`
v2.1 (what the RL buffer population reads), 99 episodes, 34,200 frames,
**3 cameras** (`cam_right_wrist`, `cam_left_wrist`, `cam_specialist`, 256×256),
delta-joint actions (joints 0-5 delta, gripper absolute), same columns as the
infer data. 5,521 frames have `next.reward`/`next.done` = 1.

**Inputs** (the episodes recorded with all 3 cameras):

| Source | Episodes used |
|---|---|
| `InferData/20261014_113503_PickCubeAndInsert_iter1` (ACT rollouts, HG-DAgger) | 48 (failures 3, 13, 34 dropped) |
| `RecoveryData/20261014_155431_PickCubeAndInsert_RecoveryEpisodes` (teleop) | 51 |

**Cut:** each episode is kept from the first frame of the first run of 5
consecutive frames that SARM predicts as stage 2 with confidence > 0.85, to
the episode's end. Checked against the hand labels of the same episodes: the
cut is a median 2 frames after the hand-labelled stage-2 start, 98/99 within
10 frames (one is 17 frames late). For the recovery episodes this cut keeps
the deliberately misaligned insert as well as the recovery.

**Columns:** the recovery episodes have no `observation.intervened` /
`observation.policy_action`; they were added as `intervened = True` (fully
teleoperated) and `policy_action = 0` (no policy ran), with the exact parquet
types of the infer data.

### Commands

Repo venv = `.venv/bin/python` (never `uv run` in this repo). The SARM reward
server and client are opensarm's (see `SARM/docs/real_robot_stage_labelling.md`).

```bash
cd ~/personal_abhi/temp/residual-offpolicy-rl
IT=trossen_real/datasets/iterativeTraining/TrossenStation3
O=$IT/RL_Stage2
OS=~/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/SARM/sarm_modified/opensarm
DU=~/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil

# 1. SARM reward server (opensarm folder, own venv)
cd $OS && CUDA_VISIBLE_DEVICES=0 uv run --with flask python reward_server.py \
    --config config/sarm_Station3_PickCubeAndInsert.yaml --host 127.0.0.1 --port 5080 --device cuda:0 &

# 2. per-frame SARM predictions (~18 s per episode)
cd $OS
../../opensarm/.venv/bin/python tools/sarm_label_client.py --server_url http://127.0.0.1:5080 \
    --dataset ~/personal_abhi/temp/residual-offpolicy-rl/$IT/InferData/20261014_113503_PickCubeAndInsert_iter1 \
    --episodes $(python3 -c "print(','.join(str(i) for i in range(51) if i not in (3,13,34)))") \
    --no-write --predictions ~/personal_abhi/temp/residual-offpolicy-rl/$O/sarm_predictions_iter1.parquet
../../opensarm/.venv/bin/python tools/sarm_label_client.py --server_url http://127.0.0.1:5080 \
    --dataset ~/personal_abhi/temp/residual-offpolicy-rl/$IT/RecoveryData/20261014_155431_PickCubeAndInsert_RecoveryEpisodes \
    --no-write --predictions ~/personal_abhi/temp/residual-offpolicy-rl/$O/sarm_predictions_recovery.parquet
cd ~/personal_abhi/temp/residual-offpolicy-rl

# 3. cut points (stage 2, 5 frames in a row, conf > 0.85)
for s in iter1 recovery; do
  .venv/bin/python scripts/TrossenStation3Real/RL/compute_sarm_stage_cut.py \
      --predictions $O/sarm_predictions_$s.parquet --stage 2 --report $O/cut_$s.json > $O/keep_from_$s.txt
done
# -> iter1: 48 episodes, 12780/23707 frames kept; recovery: 51, 21420/30754

# 4. delta-joint conversion from the cut (all features and cameras kept)
for s in iter1 recovery; do
  K=$(cat $O/keep_from_$s.txt); EPS=$(echo "$K" | tr ',' '\n' | cut -d: -f1 | paste -sd,)
  [ $s = iter1 ] && SRC=$IT/InferData/20261014_113503_PickCubeAndInsert_iter1 \
                 || SRC=$IT/RecoveryData/20261014_155431_PickCubeAndInsert_RecoveryEpisodes
  .venv/bin/python -m trossen_real.scripts.convert_to_delta_joint_dataset \
      --input-root $SRC --output-root $O/${s}_stage2_deltajoint --episodes "$EPS" --keep-from "$K" \
      > $O/${s}_convert.log 2>&1
done

# 5. same columns as iter1 for the recovery part
.venv/bin/python -m trossen_real.scripts.add_constant_features \
    --dataset $O/recovery_stage2_deltajoint --like $O/iter1_stage2_deltajoint \
    --set observation.intervened=1 --set observation.policy_action=0

# 6. merge (v2.1)
.venv/bin/python $DU/tools/merge_v21_datasets.py \
    --dataset $O/iter1_stage2_deltajoint --dataset $O/recovery_stage2_deltajoint \
    --force-task "PickCubeAndInsertStation3" \
    --output $O/PickCubeAndInsert_TS3_Stage2_RL_deltajoint_v21
```

Folder contents: the predictions (`sarm_predictions_*.parquet/.json`), the
cut points (`keep_from_*.txt`, `cut_*.json`), the two intermediate parts, the
conversion logs and the merged dataset.

### Cameras: RL 3, ACT 2 (`policy_camera`)

`conf/station3.yaml`:

```yaml
rl_camera:        # what the residual actor/critic see and the buffers store
- observation.images.cam_left_wrist
- observation.images.cam_right_wrist
- observation.images.cam_specialist
policy_camera:    # what is sent to ACT for the base action; null = rl_camera
- observation.images.cam_left_wrist
- observation.images.cam_right_wrist
```

`policy_camera: null` (the default) is the behaviour before this option: ACT
gets `rl_camera`, and existing buffer caches keep their hash. When set, it is
used for the base-policy query online (`TrossenResidualEnv._query_base_action`)
and offline (`_populate_offline_buffer`), and it is part of both buffer cache
keys (the stored base actions depend on it). The trainer refuses to start if
ACT needs a camera that is not in this list. Resolution is separate already:
RL frames are resized to `offline_data.image_size` (84), ACT gets 256×256.
Helper: `resfit/rl_finetuning/config/residual_td3.py::camera_keys`; tests:
`trossen_real/inference/tests/test_camera_split.py`.

## 2. SARM when an episode starts at stage 2

Run on 20 episodes of this dataset (which all start at stage 2): frames 0-9
are stage 0 only because the reward server forces the first `warmup_frames`
(10) of an episode to stage 0; from frame 10 the model predicts stage 2 with
confidence 1.00, then 3 and 4. The short history at the start is not a
problem. The warmup pin only matters if the stage detector queries more
often than every 10 ticks (the infer app queries once per ACT chunk, 12-15).

## 3. Stage-2 reset

`trossen_real/rl/stage2_reset.py`: `Stage2ResetFollower(FollowerClient)`
with `reset()` =

1. blocking joint move to `STAGE2_JOINTS`, gripper open
2. (optional) leader synced, position mode
3. operator puts the cube between the fingers, types `0` + Enter
4. gripper closed on the cube (`GRIPPER_CUBE_CLOSE` = 0.004), 2 s wait
5. randomization, `mode`:
   - `joint`: uniform ± `JOINT_NOISE` on base/shoulder/elbow/wrist pitch,
     clipped to `JOINT_MIN`/`JOINT_MAX`; wrist yaw/roll untouched (they set
     the cube's orientation to the hole)
   - `ee`: read the EE pose, ± `EE_XYZ_OFFSET` / ± `EE_ANGLE_OFFSET`, EE move
     (xyz clipped to `ee_bounds` by the follower server), joint control after
   - `none`: stay at `STAGE2_JOINTS`

The gripper stays closed on the cube in every mode. The constants are
hard-coded at the top of the file, from frame 0 of the 99 episodes:

| Constant | Value | From |
|---|---|---|
| `STAGE2_JOINTS` | [-0.2657, 1.6978, 1.3430, -1.1645, -0.0032, -0.2020] + gripper 0.040 | per-joint median |
| `JOINT_MIN` / `JOINT_MAX` | 10th / 90th percentile | per joint |
| `JOINT_NOISE` | ±0.05, 0.03, 0.03, 0.03 rad (joints 0-3) | chosen small; tune |
| `EE_XYZ_OFFSET` | ±0.015, 0.015, 0.010 m | chosen inside the EE spread (std x 0.026, y 0.050, z 0.011) |
| `EE_ANGLE_OFFSET` | ±0.03 rad | chosen inside the angle spread (~0.03-0.04) |

**Test it first** (follower server running):

```bash
.venv/bin/python scripts/TrossenStation3Real/RL/test_stage2_reset.py --mode joint --dry-run --n 10   # no motion
.venv/bin/python scripts/TrossenStation3Real/RL/test_stage2_reset.py --mode joint --n 5
.venv/bin/python scripts/TrossenStation3Real/RL/test_stage2_reset.py --mode ee --n 5 --leader
```

Unit tests (fake follower): `trossen_real/inference/tests/test_stage2_reset.py`.

**Selecting it:** `reset.single.mode` in the station config
(`trossen_station3_single.yaml`): `normal` (default) / `stage2_joint` /
`stage2_ee` / `stage2_none`. Used by:

- **the infer app**: Start Inference and Reset Robot run the stage-2 reset;
  the UI shows "put the cube between the gripper fingers" and a **Cube placed
  - close gripper** button the reset waits for. Stop Inference / Disconnect
  cancel a waiting reset.
- **the RL trainers** (`train_residual_td3[_distributed].py`): same reset,
  confirmed with `0` + Enter in the actor's terminal.

Both build the follower with `stage2_reset.make_follower()`.
