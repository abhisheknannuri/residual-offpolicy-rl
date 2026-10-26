# PickCubeAndInsert Station3 — Trial1 (2026-09-28)

Dataset prep + ACT training for the 5 teleop sessions recorded on Station 3
(2026-09-24 / 25). Same pipeline as the Station 1 "Trial1" work recorded in
`trossen_real/training_prompts.md`.

## Result — 3 v3 datasets, 178 episodes / 55,177 frames each

| Kind | Path (under `trossen_real/datasets/`) |
|---|---|
| Absolute joint | `PickCubeAndInsert_Station3_Trial1` |
| Delta joint | `PickCubeAndInsert_Station3_Trial1_deltajoint` |
| Relative (chunk-anchor) | `PickCubeAndInsert_Station3_Trial1_relativeDeltaTraining` |

All three verified by a real `LeRobotDataset` load (v3.0, `action` and
`observation.state` both `(7,)`). `*_old` siblings are the v2.1 backups the v3
converter keeps; the 5 raw sessions were never modified.

## What went in

| Session | Eps | Action taken |
|---|---|---|
| `20260924_132455` | 3 | kept as-is |
| `20260924_155210` | 60 | **dropped ep 21** — the classifier catches it automatically (reward pedal never held) |
| `20260924_164756` | 60 | kept as-is |
| `20260925_100304` | 6 | **dropped ep 5** — needed a manual `--exclude`; see below |
| `20260925_100751` | 51 | kept as-is |
| **Total** | **180 → 178** | 56,186 → 55,177 frames |

**Episode 5 of `20260925_100304` is the interesting one.** The automatic
classifier *keeps* it: the reward pedal was genuinely held, so by every
recorded signal it is a success. But it is 563 frames with 248 done-frames,
versus ~300/45 for its siblings — the operator marked success on an attempt
that failed. `next.reward`/`next.done` only record operator intent at press
time, never physical outcome, so no automatic check can catch this. Same
failure mode as Trial1's episodes 75 and 84.

## Running it

```bash
./prepare_dataset.sh          # already run once; re-running needs the outputs deleted first
```

Then copy the three datasets to the server and:

```bash
./train_act_abs.sh
./train_act_deltajoint.sh
./train_act_relative.sh
```

Hyper-parameters are identical to Trial1 (75k steps, batch 64, chunk 20,
`n_action_steps` 20, `use_vae=false`, lr 1e-4). Override without editing:

```bash
STEPS=50000 DATA_DIR=/some/other/path ./train_act_abs.sh
```

## Environments — this is the part that bites

Three different venvs are involved, and picking the wrong one fails:

| Variable | Path | Used for |
|---|---|---|
| `REPO_PY` | `<repo>/.venv/bin/python` | `trossen_real.scripts.*` (filter, delta convert) |
| `DU_PY` | `DatasetUtil/.venv/bin/python` | merge, **v2.1 → v3** |
| `LR_EDIT` | `GeneralistRewardModels/lerobot/.venv/bin/lerobot-edit-dataset` | relative-action stats |

The repo venv has **lerobot 0.1.0**, which cannot write v3; DatasetUtil's has
**0.4.4**, which can. **Never run `uv` on this machine** — it breaks the repo
venv. Every command calls an interpreter directly. (`uv run` on the *server*
for `lerobot-train` is fine and is what Trial1 used; override `TRAIN_CMD` if
you'd rather not.)

## Trimming — nothing is trimmed, on purpose

Both trims in `convert_to_delta_joint_dataset.py` were disabled on 2026-09-01
and the dry run here confirms `0 start + 0 end` across all 178 episodes:

- **Start-trim** assumed every episode opens with a genuine still period to
  calibrate a "resting floor" against. Not true for policy rollouts that start
  mid-motion — on one verified episode it discarded the whole approach, up to
  the gripper already touching the cube.
- **End-trim** decided purely from arm speed, with no awareness of
  `next.done`/`next.reward`. A success hold (arm still, pedal held) looks
  exactly like dead time, so it cut real terminal frames — 26 of 65 episodes in
  one batch, including the true `next.done=True` frame.

If you want dead time at the start removed, it needs a new heuristic (the old
functions are kept in the file, just not called). Trial1 also shipped with 0
trim, so this is consistent with the datasets already trained on.

## Notes

- `20260925_100751` carries `meta/lerobot_rl_labels.json` (`{"episodes": {}}`,
  empty) and `episodes_stats.jsonl.rl_bak`, both root-owned, from an RL
  labelling tool run on 2026-09-25. The empty label file had no effect, and the
  merge reads `episodes.jsonl`/parquet, so the merged output is unaffected.
- All 5 sessions already shared the task string `PickCubeAndInsertStation3` and
  an identical 16-key feature schema, so no `--force-task` was needed.
- Relative stats came out `mean=0.0098, std=0.0834, q01=-0.2908, q99=0.2470`
  (`relative_dims=6/7`, gripper excluded) — closely matching Trial1's
  `mean=0.0107, std=0.0734`, which is a good cross-check that the data is
  comparable.
- Disk: the new artifacts total ~1.1 GB and the volume is at 99%.
  `PickCubeAndInsert_Station3_Trial1_old` (213 MB) and
  `..._deltajoint_old` (191 MB) are v2.1 backups, safe to delete once you're
  happy with the v3 outputs.

---

# Iterative training 1 (2026-10-16)

Second ACT round: Trial1 plus the data collected with the Trial1 ACT. Two
datasets, trained with the **same command and hyper-parameters** as Trial1;
only the dataset changes. Evaluate both on the robot and keep the better one.

| Dataset (v3, under `trossen_real/datasets/iterativeTraining/TrossenStation3/ACT_Iter1/`) | Episodes | Frames | Contents |
|---|---|---|---|
| `PickCubeAndInsert_TS3_Iter1_deltajoint` | 226 | 78,884 | Trial1 (178) + iter1 rollouts (48) |
| `PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint` | 273 | 92,381 | the above + 47 recovery segments |

Built by `prepare_dataset_iter1.sh` (a record of what was run, with the
reasons in comments); recovery cut points by `compute_recovery_keep_from.py`.

## What went in

| Source | Eps | Action taken |
|---|---|---|
| `PickCubeAndInsert_Station3_Trial1_deltajoint_old` (v2.1 of the Trial1 delta dataset) | 178 | as-is |
| `InferData/20261014_113503_PickCubeAndInsert_iter1` (ACT rollouts with HG-DAgger interventions) | 51 → 48 | dropped 3, 13, 34 (failures, no reward pedal); delta-converted |
| `RecoveryData/20261014_155431_PickCubeAndInsert_RecoveryEpisodes` (teleop) | 51 → 47 | **only the recovery part kept** (below); delta-converted; dataset B only |

Both new sources: `cam_specialist` (added for SARM) is dropped, and so are the
infer app's `observation.intervened` / `observation.policy_action`, so the
schema matches Trial1. ACT keeps the same 2 wrist-camera input as Trial1, so
the trained policy runs with the same inference config.

## Why the recovery episodes are cut

Every recovery episode is a teleop demo of approach → grasp → a
**deliberately misaligned insert and release** → recovery. Training on whole
episodes would teach ACT the deliberate failure (tilting and letting go of
the cube). So each episode is kept only from 0.5 s after its failed release:
the first frame ACT sees is the cube unseated on the plate with the gripper
open. The release frame comes from the SARM stage labels (first gripper
opening during "cube being inserted"); 4 episodes without one are dropped.
Checked visually at 8 cut points.

What dataset B can still get wrong: the recoveries use several strategies
(push with closed fingers, re-grasp and slide, lift and re-insert, put the
cube down on the table and re-grasp). This ACT has `use_vae=false`, so it
averages conflicting behaviours for the same image. If B is worse than A,
that is the first suspect; a later round should record one strategy only
(re-grasp, lift, re-insert).

## Train on the server

Copy both v3 datasets into the server's `DATA_DIR`
(`/home/qte9489/Research/DatasetUtils/CubePickAndInsert_TrossenReal/Station3_Trial1`
by default, or set `DATA_DIR`), then:

```bash
DATASET=PickCubeAndInsert_TS3_Iter1_deltajoint                            ./train_act_deltajoint.sh  # A
DATASET=PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint               ./train_act_deltajoint.sh  # B
DATASET=PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint USE_VAE=true  ./train_act_deltajoint.sh  # B + CVAE
```

Runs 1 vs 2 differ only in data, 2 vs 3 only in `use_vae`. The full record
(every command, the checks, how to do the next iteration) is in
`trossen_real/datasets/iterativeTraining/TrossenStation3/ACT_Iter1/README.md`.

Checkpoints go to `outputs/train/ACT_BC_<DATASET>`. Same 75k steps, batch 64,
chunk 20/20, lr 1e-4 as Trial1. With ~43-67% more frames, 75k steps is
proportionally fewer epochs than Trial1; compare the 50k-75k checkpoints.

## Gotcha found on the way

The visualizer's RL (reward) tab had left `episode_*.parquet.rl_bak` backups
inside `data/chunk-000/`. lerobot 0.1.0 reads every file in `data/`, so the
load failed with "timestamps unexpectedly violate the tolerance". They were
moved to `<dataset>/.rl_backups/`.
Fixed in the visualizer (`backend/rl/common.py::backup_file`) on 2026-10-16:
new backups go to `<dataset>/.rl_backups/` directly.
