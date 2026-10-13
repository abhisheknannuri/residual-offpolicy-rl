# Station 3: why full-task residual RL hurt ACT, and the final-stage plan

Date: 2026-10-13. Every number below is measured from files in this repo
(`eval_runs/index.jsonl`, `logs/residual/`, `logs/infer_*`, `RealTrainLogs/`,
the run checkpoints and buffers). Anything not verified says so.

---

## 1. Summary

- **Residual RL made ACT worse.** Best ACT checkpoints reach "cube over the
  hole" (stage 2) in 60-80% of eval runs. The latest residual checkpoint
  (`run_2026-10-13_13-21-22 / policy_step_70016`) never grasps the cube in
  20 of 33 runs (stage 0).
- **Your hypothesis is right, and the data shows why.** In the `delta_joint`
  action space the residual is as large as ACT's own per-tick step (not a small
  correction), it has a steady per-joint bias, and delta control integrates
  that bias every tick. ACT replays each 14-step chunk open-loop, so nothing
  pulls the arm back until the next chunk, which then starts from a pose the
  demonstrations never visited.
- **The critic cannot steer the actor anyway.** Q barely depends on the action
  (the whole residual range moves Q by ~0.003 out of ~0.7), so the actor
  follows mostly noise plus whatever bias it picked up.
- **The new plan fits these findings.** Improving ACT with human-gated data
  attacks the pick/transport failures directly, and restricting RL to the
  final insertion stage gives short episodes, a small precise residual, and a
  much denser success signal.
- **Effort.** Data collection with interventions and a third camera already
  works (0 new code). ACT retraining is configuration. The real work is the
  final-stage RL: stage reset, stage-filtered offline buffer, separate RL /
  ACT camera lists, and (optionally) SARM on the real env. Roughly 500-700
  lines of code plus tests, described per item in section 5.

---

## 2. What the evaluations show

Eval stages (`trossen_real/configs/eval/pick_cube_and_insert_station3.yaml`):
0 approach, 1 grasp and lift, 2 transport over the hole, 3 insert (success).
`furthest_stage` counts per checkpoint:

| Checkpoint | runs | success | stage 0 / 1 / 2 / 3 |
|---|---|---|---|
| ACT 050000 (the base used for RL) | 20 | 4 | 0 / 0 / 16 / 4 |
| ACT 060000 (best ACT) | 22 | 6 | 1 / 1 / 12 / 6 |
| ACT re-eval today (`bc_reeval_13Oct2026_14_36`) | 8 | 1 | 1 / 0 / 5 / 1 |
| offline RL `offline_step_100000` | 21 | 2 | 10 / 0 / 8 / 2 |
| online RL 10-09 `policy_step_72505` | 20 | 2 | 9 / 2 / 7 / 2 |
| online RL 10-13 `policy_step_70016` | 33 | 3 | **20** / 0 / 2 / 3 |

ACT almost always grasps and transports (stage 2), and fails at insertion.
Every RL checkpoint moves mass from stage 2 down to stage 0, i.e. the
residual breaks the part ACT already does well. Offline RL alone already shows
it (10/21 at stage 0), so it is not only an online-training problem.

The training run behind `policy_step_70016` (`RealTrainLogs/station3_online_2026-10-13_run1.jsonl`):
80,401 steps, 267 episodes, 72 pedal successes (most of them human-assisted:
16% of ticks intervened), 141 timeouts. Success did not increase over the run;
the last quarter has the fewest interventions and mostly timeouts.

## 3. Why the residual drives the arm away (measured)

> The full derivation, with every unit conversion, the proof that delta
> control accumulates the residual, and per-chunk measurements on both eval
> sessions joined to the measured joint positions, is in
> [RESIDUAL_DRIFT_MATH.md](RESIDUAL_DRIFT_MATH.md). Where the numbers below
> differ, that document is the reference: it uses both eval sessions (6,828
> ticks) and per-tick ratios instead of one session's medians.

From the residual client's per-tick log during the `policy_step_70016` evals
(`logs/residual/residual_trossen_station3_single_20261013_135955.jsonl`,
4,791 ticks; a second session `..._143037` agrees):

**3.1 The residual is not small in delta space.**

| joint | 0 | 1 | 2 | 3 | 4 | 5 | grip |
|---|---|---|---|---|---|---|---|
| median \|residual\| / median \|ACT action\| | **1.07** | 0.45 | **1.13** | 0.57 | 0.77 | 0.39 | 0.23 |

`action_scale = 0.15` (normalized) on joints 0-2 is 0.013-0.025 rad per tick,
about the size of ACT's typical per-tick delta. In ResFiT's original setting
(absolute actions) the same number is a small offset around a target; with
`delta_joint` it is a second policy of equal strength.

**3.2 The residual has a steady bias, and delta control integrates it.**

| joint | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| mean residual (normalized) | -0.041 | +0.017 | -0.034 | +0.026 | -0.003 | -0.015 |
| drift per 14-tick chunk (rad) | -0.049 | +0.039 | **-0.079** | +0.045 | -0.003 | -0.046 |
| drift rate (deg/s at 20 Hz) | -4.0 | +3.2 | **-6.5** | +3.7 | -0.3 | -3.8 |

Each tick commands `q_measured + delta`. A bias in `delta` is never
"returned": the next tick starts from wherever the arm is. The follower only
realizes part of each command (section 3.4), so the physical drift is smaller
than these numbers, but it accumulates the whole episode.

**3.3 ACT cannot correct inside a chunk.** ACT runs with `chunk_size 20`,
`n_action_steps 14`: one forward pass, then 14 cached actions (0.7 s)
executed without looking at the robot. Its deltas are relative to the
measured state, so the residual's drift is carried along, not undone. At the
next forward pass ACT sees a pose outside its training distribution and
starts from there. This is your observation: drift early in the chunk, an
unrecoverable pose by the end of it.

Re-querying ACT every tick would close this loop, but it is not affordable as
is: a fresh ACT forward takes **~75 ms median** at eval (`dt_base_ms`), more
than a 50 ms tick. These refills are also the cause of the ~7% slow ticks seen
in training (7.4% of ticks > 20 ms base latency = 1 in 14).

**3.4 Related findings from the 2026-10-12 training runs.**
- The follower realizes only 4-39% of the summed commanded deltas over 1 s:
  joints 1-2 carry a constant ~0.01 rad "holding" offset against gravity
  (present in the teleop demos too).
- Critic analysis over the buffers (checkpoints 2503, 2505, 5004): `|dQ/da|`
  ~0.02, `Q(base+pi) - Q(base)` ~ +0.007 on Q ~ 0.7, the critic does not rank
  human corrections above the actor's action (27-31%), and the actor's
  residual is uncorrelated with human corrections (cosine ~0). The critic is
  learning V(s), not which actions help.
- Fixed bugs that affected all runs before 2026-10-12 evening: flat 0.05
  exploration noise (yaml ignored), stored gripper != executed gripper, and
  15.7 Hz training vs 20 Hz data/eval (see the commit log).

## 4. Lessons that should shape the new pipeline

1. **Keep the residual small relative to ACT's step, not to the action box.**
   In `delta_joint`, choose `action_scale` from ACT's per-tick deltas in the
   stage where RL runs (e.g. 10-30% of the median |ACT action| per joint),
   not a fixed 0.15.
2. **Penalize residual bias.** `agent.actor.action_l2_reg_weight` is 0 today.
   A small value keeps the residual centred at zero unless the critic has a
   real reason. Any constant offset becomes drift in this action space.
3. **Use the human corrections directly.** The buffer already stores
   `intervened`. A BC term on intervened rows towards
   `clip(human - base, ±action_scale)` teaches the reachable part of every
   correction without depending on the critic's action sensitivity.
4. **Short horizon.** Insertion-only episodes (tens of ticks, not 350-500)
   make each action matter more to the return, which is exactly what the
   critic is missing now (with γ = 0.9975 over ~400 ticks, one 50 ms action
   barely changes Q).

---

## 5. Proposed pipeline: what exists, what changes, how much

### 5.1 Collect intervention data with ACT running (HG-DAgger style)

**Already works, no code change.**
- `trossen_real/infer_app` records a LeRobot dataset while ACT runs, with
  pedal interventions (`infer_loop.py:288-345`). Per frame:
  - `action`: the EXECUTED absolute command (leader position while you
    intervene, ACT's action otherwise)
  - `observation.policy_action`: what ACT proposed
  - `observation.intervened`, `next.reward` (reward pedal)
  - `observation.state`, `joint_pos_raw`, `ee_pose_raw`, velocity, effort,
    acceleration
- **Cameras:** the recorder writes every camera that has a serial in the
  station config (`dataset_recorder.py:133-137`); ACT only receives the
  cameras its checkpoint reports via `/health` (`app.py:434`). `cam_high` got
  its serial in commit `a046bb6`, so it is already recorded and never sent to
  ACT. To record a different extra camera, change the serials in
  `trossen_station3_single.yaml`. Nothing else changes.
- Check once before a long session: record 1 episode and confirm
  `meta/info.json` lists three `observation.images.*` keys.

**Decisions:**
- Which frames ACT retrains on. Options:
  - whole episodes (autonomous + corrected parts), filtered to successes
  - human-gated segments only (intervened frames plus a short lead-in)
  
  HG-DAgger uses all human-gated segments; adding whole successful episodes
  also reinforces ACT's good autonomous behaviour.
- Note that the very first ticks after an intervention starts can carry a
  jump between leader and follower. Look at a few episodes before deciding
  whether to trim them.

### 5.2 Process the data and retrain ACT

**Existing tools:**
- `trossen_real/scripts/filter_episodes.py` (drop bad episodes), DatasetUtil
  merge, `convert_to_delta_joint_dataset.py` (already handles the infer
  app's extra columns, see its lines 97-132), `convert_v21_to_v3.py`. Recipe:
  `scripts/TrossenStation3Real/prepare_dataset.sh`.
- ACT training: `scripts/TrossenStation3Real/train_act_deltajoint.sh` (the
  external lerobot repo).

**What is new:**
- **An ACT-only copy without the third camera.** lerobot builds the policy's
  inputs from the dataset's features, so a dataset with `cam_high` would make
  ACT expect three cameras. Keep two datasets:
  - full (3 cams) for RL and SARM
  - ACT copy with `observation.images.cam_high`, `observation.intervened` and
    `observation.policy_action` removed

  `lerobot-edit-dataset` (lerobot 0.4.x venv used in `prepare_dataset.sh`)
  has a feature-removal operation; the exact flag is NOT verified here. Check
  `lerobot-edit-dataset --help`. If unsupported, a ~60-line script with the
  same v2.1 read/write pattern as `convert_to_delta_joint_dataset.py` does it.
- **Merging old and new data.** The original 178 episodes have only the two
  wrist cameras. That is fine for ACT (2 cams), but affects RL (5.4.3).

**Effort:** 0-60 lines plus a new section in `prepare_dataset.sh`. Then the
usual ACT evaluation with the existing eval app and poses file.

### 5.3 Subtask labeling and SARM

**Exists in this repo (sim only):**
- `resfit/rl_finetuning/reward_models/reward_client.py`: HTTP client for the
  SARM/TCC reward server (`/predict_stage`, gated stage + confidence).
- `resfit/rl_finetuning/reward_models/pbrs_wrapper.py` (`StageAwarePBRSWrapper`,
  279 lines): online PBRS or milestone rewards from stages.
- `scripts/label_offline_pbrs.py` (418 lines): labels an offline v2.1 dataset
  through the same server, writes a sidecar parquet (`index`, `stage`,
  `reward_pbrs`) consumed via `offline_data.reward_parquet`.

**External (not in this repo):** the SARM training code and reward server
(`opensarm/reward_server.py` per the labeler's docstring), and the subtask
annotation tool.

**What is new:**
- **Annotation.** Subtasks matching the eval stages (approach, grasp_lift,
  transport, insert) keep eval, SARM and RL consistent.
- **Labeling the real dataset.** `label_offline_pbrs.py` defaults to sim keys
  (e.g. `observation.images.agentview`). Point it at the real dataset and the
  camera SARM was trained on. Small changes (~20 lines) if the key handling is
  hard-coded.
- **SARM on the real env (only if you want dense reward online).** The
  wrapper is only applied inside the sim vec-env
  (`train_residual_td3_distributed.py:727-760`); the real branch returns
  `TrossenResidualEnv` before that point. The wrapper expects a single
  unbatched env; the real env returns `(1, dim)` tensors and float images.
  Needs an adapter or a small integration in `TrossenResidualEnv.step()`:
  ~80-150 lines plus a test.

**Simpler alternative for the final stage:** if RL only runs the insertion,
the reward pedal (sparse success, already working) may be enough, and SARM is
used only offline:
- to find where the insertion stage starts in each episode (5.4.1, 5.4.2)
- optionally as a success check

### 5.4 Final-stage residual RL

#### 5.4.1 Stage reset (the main new piece)

What you described: move the arm to a random pose at the start of the
insertion stage, open the gripper, you place the cube, the gripper closes,
the episode starts.

**Exists:**
- `FollowerClient.reset(pose_to_reach=...)` moves to the staged joint pose,
  then to an explicit EE pose, clipped to `ee_bounds`
  (`follower_single.py:244-300`)
- `ResetConfig.randomize_ee` + `xyz_offset` (random EE offsets around a pose)
- `move_gripper`
- the pedal and `EpisodeBoundaryMonitor`
- every recorded frame stores `observation.ee_pose_raw`, so real insertion
  start poses can be taken from your own demonstrations

**New code:**
1. `scripts/extract_stage_start_poses.py` (~80 lines). For each episode, take
   the first frame of the insertion stage, using the SARM sidecar parquet or
   the eval annotations if SARM is not ready. Save its `ee_pose_raw` (and
   joint q) to a JSON file.
2. A stage-reset mode in `TrossenResidualEnv.reset()` (~100-150 lines, config
   section `stage_reset:`):
   - sample a start pose from the JSON, add a random offset (reuse
     `xyz_offset`), move there via `reset(pose_to_reach=...)`
   - open the gripper
   - wait for a pedal press ("cube placed"), with a timeout and a log line
   - close the gripper (`gripper_closed`), settle, then `policy.reset()` and
     return the first observation
   - Today `reset()` ends with `sync_to_follower` for the leader; keep that
     after the gripper closes.
3. ACT is then queried mid-task, from a pose it saw in demos only partway
   through an episode. ACT is stateless apart from its action queue
   (`policy.reset()` clears it), so this works, but check how well ACT does
   from these resets before adding RL (section 6).

#### 5.4.2 Offline buffer: insertion frames only

**Exists:** the offline buffer builder walks the dataset episode by episode
(`train_residual_td3_distributed.py:1100-1260`) and supports sidecar rewards
(`offline_data.reward_parquet`).

**New:** a per-episode start-frame filter (e.g. `offline_data.stage_parquet`
+ `offline_data.min_stage`) so only insertion-stage frames become
transitions. ~40-60 lines. It must be added to the offline cache key
(`offline_cache_meta`, around line 1380) or old caches will be reused.

**Side benefit:** today the builder queries ACT for every one of the 55,177
frames; insertion-only cuts that to the insertion frames.

#### 5.4.3 RL cameras separate from ACT cameras

**Today they are one list.** `rl_camera` is used for the ACT query AND the RL
observation:
- `TrossenResidualEnv`: `self.image_keys` feeds both `build_observation()`
  (ACT) and `_build_actor_obs()` (RL), `real_residual_env.py:116, 506, 526`
- the offline builder: `_dataset_sample_to_json_obs(sample, image_keys)` and
  the RL obs, around line 1093

**New:** a second list, e.g. `base_policy_cameras` (default: equal to
`rl_camera`, so nothing changes unless set), or read it from the policy
server's `/health`. Pass it to the env and the offline builder for the ACT
query only. ~40-60 lines plus a test.

**Deployment already works.** `residual_client.py` builds the RL observation
from the raw frames of all cameras, and ACT gets its own keys.

**Consequences:**
- the actor and critic get one ViT per RL camera, so a new RL camera set
  means new networks (no warm start from current checkpoints)
- `image_keys` is part of both cache keys, so a new camera set means new caches
- the original 178 demos have no `cam_high`. With `cam_high` in the RL
  cameras, the offline buffer can only use the new episodes (the builder
  needs the key in every frame). Either:
  - keep RL on the two wrist cameras for old data
  - record enough new insertion data
  - use `cam_high` only once the new dataset is big enough

#### 5.4.4 Configuration for the insertion stage (no code)

- `real_max_steps`: insertion length from the data plus margin (look at the
  stage durations after labeling; likely ~60-150 ticks instead of 350)
- `agent.actor.action_scale`: from ACT's per-tick deltas in the insertion
  frames (section 4.1)
- `agent.actor.action_l2_reg_weight` > 0
- `algo.learning_starts` and `buffer_size`: smaller, episodes are short
- reward: sparse pedal (as now) or SARM milestone/PBRS

#### 5.4.5 BC on interventions (optional, recommended)

The online buffer already has `intervened` per row (and the offline rows are
all `False`). Add a term to the actor loss for intervened rows only:
`λ · ||π(s) − clip(a_human − base, ±s)||²`. That's ~40-60 lines in
`q_agent.update_actor` plus config and a test. `bc_loss_coef` exists today but
belongs to the RFT path (`update_actor_rft`); read it before reusing.

#### 5.4.6 Evaluating the insertion stage

The eval app (`infer_app`) resets to the staged pose only. A final-stage eval
needs the same stage reset (and the human placing the cube), so either:
- reuse `TrossenResidualEnv` with RL off for evaluation
- add the stage reset to the eval app (~50-80 lines)

Per-stage success is also needed for the thesis comparison: ACT vs ACT +
residual on identical stage resets.

---

## 6. Order of work

1. Record intervention episodes with the infer app (3 cams) - **no code**.
2. Build the ACT-only dataset copy, retrain ACT, evaluate with the existing
   eval app - **config + small script**.
3. Label subtasks, train SARM, label the dataset offline (or annotate the
   insertion start frames by hand if SARM is not ready).
4. `extract_stage_start_poses.py` + stage reset in `TrossenResidualEnv`.
   **Evaluate ACT alone from stage resets first.** If ACT already inserts
   reliably from there, RL has little to gain; if it fails consistently in a
   specific way, that is what RL targets.
5. Stage-filtered offline buffer, `base_policy_cameras`, insertion config,
   optional BC-on-interventions.
6. RL on the insertion stage, evaluating every few thousand steps with the
   checkpoint-over-buffer analysis (`Q(base+π) − Q(base)`, `|dQ/da|`,
   residual bias) alongside real success rate.

## 7. Effort table

| Item | Exists | New code (approx.) | Files |
|---|---|---|---|
| Intervention recording + extra camera | yes | 0 | - |
| ACT-only dataset copy | partly (`lerobot-edit-dataset`, unverified flag) | 0-60 | `prepare_dataset.sh` / new script |
| ACT retrain | yes | 0 | external lerobot |
| SARM training + server | external | - | external |
| Offline SARM labeling of the real dataset | yes (sim keys) | ~20 | `scripts/label_offline_pbrs.py` |
| SARM reward on the real env (optional) | sim wrapper | 80-150 | `real_residual_env.py` or adapter |
| Stage start poses | no | ~80 | new script |
| Stage reset | parts (pose_to_reach, gripper, pedal) | 100-150 | `real_residual_env.py`, `config.py`, station yaml |
| Insertion-only offline buffer | builder + sidecar mechanism | 40-60 | both trainers |
| Separate ACT / RL cameras | no | 40-60 | env, both trainers |
| BC on interventions (optional) | `intervened` stored | 40-60 | `q_agent.py`, config |
| Final-stage eval | eval app | 50-80 | `infer_app` or env |
| Tests for the above | - | ~150-250 | `trossen_real/inference/tests/`, `resfit/.../tests/` |

Total new code: roughly 500-700 lines plus tests. None of it changes the
math verified in the buffer inspections, and every new behaviour can default
to off so existing configs and caches keep working.

## 8. Open questions

1. Retraining ACT: whole successful episodes, human-gated segments only, or
   both?
2. RL cameras: wrist only, or wrist + `cam_high` (which limits offline data
   to new episodes)?
3. Final-stage reward: pedal-only sparse success, or SARM shaping online?
4. Stage-start distribution: sample only the recorded start poses, or add
   random offsets around them (how large)?
5. Should the residual run at all in grasp/transport, or should ACT run alone
   until the insertion stage starts (needs online stage detection, i.e.
   SARM online)? The current plan resets directly into the stage, which
   avoids this question during training.
