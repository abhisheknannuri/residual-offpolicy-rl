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
