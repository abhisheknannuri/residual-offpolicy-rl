# Failed insertions: data, labels, SARM and ACT recovery

Date: 2026-10-14. Station 3, PickCubeAndInsert.

The question: the cube sometimes ends up resting crooked on the hole instead
of inserted. How to record such episodes, label them, train SARM so it can
tell success from failure, and teach ACT to recover.

Sources checked:
- `infer_dataset/20261014_113503_PickCubeAndInsert_iter1` (51 episodes, 24,610
  frames, 3 cameras including `cam_specialist`)
- SARM code in
  `RLRewardResearchWS/SARM/sarm_modified/opensarm` (file:line refs below)
- this repo's reward client / PBRS wrapper

---

## 1. What the new data already contains

- Episode 13 (`cam_specialist`, last 6 s): the cube is lowered into the
  square hole, the gripper opens and leaves, and the cube stays tipped on the
  hole edge with two faces visible. Not inserted.
- Per-episode labels already in the dataset:

  | | episodes |
  |---|---|
  | reward pedal pressed (success) | 48 / 51 |
  | no reward (failure) | 3: episodes 3, 13, 34 |
  | any intervention | 41 / 51 |
  | episode 13 | 0 interventions, 0 reward: a pure autonomous failure |

  So success vs failure per episode is already recorded, through
  `next.reward`. What is missing is *which frames* show the failure and
  enough failure episodes to learn from.

---

## 2. How SARM actually works (from the code)

**One label per frame: `reward = stage + τ`**
(`lerobot/common/datasets/rm_lerobot_dataset.py:132, 186-201`;
`workspace/sarm_ws.py:305`). The integer part is the stage, the fractional
part τ ∈ [0, 1) is progress inside that stage.

**Two networks:**
- `StageTransformer` (`models/stage_estimator.py`): a causal transformer
  over a window of frames (CLIP image embeddings per camera + task text +
  robot state), with a classification head trained with cross-entropy on
  `floor(reward)`.
- `SubtaskTransformer` (`models/subtask_estimator.py`): the same inputs
  plus the stage one-hot, with a sigmoid head trained with MSE on τ.

  τ is annotated linearly in time inside each stage, so the progress head
  learns "how long have we been in this stage". This is the part you called
  bad; it can be ignored for the RL reward, which only uses the stage.

**"sparse" and "dense" are not success and failure.** They are two annotation
granularities: a coarse stage list and a finer one, each with its own heads
and its own dataset (`config/sarm_PickPlaceCan.yaml`:
`repo_id_sparse`, `repo_id_dense`, `use_dual_datasets`). The sim run used
sparse only. Putting failures in "dense" would mix two unrelated things.

**Rewind augmentation already teaches "progress can go backwards".** With
`max_rewind_steps > 0`, 80% of training windows get reversed frames appended
with reversed targets (`rm_lerobot_dataset.py:144-201`). It is synthetic
(played-backwards video), not real failures.

**The reward server only lets the stage go up.**
`apply_hysteresis(..., monotonic=True)` never proposes a stage below the
committed one (`reward_server.py:264-330`). As configured, once the cube
reaches the insert stage, a later tip-over cannot lower the reward.

---

## 3. Recommendation: label the state of the world, not the robot's motion

The classifier is the useful part of SARM. Make each stage mean "what state
is the cube in", so a failed insertion is simply a frame in an earlier stage:

| stage | name (annotation text) | the frame shows |
|---|---|---|
| 0 | approaching the cube | cube on the table, not grasped |
| 1 | cube grasped and lifted | cube in the gripper, off the table |
| 2 | cube above the hole | cube in the gripper, over the plate, not touching it |
| 3 | cube being inserted | cube on the plate but not seated: held or released, upright or tipped, being pushed or re-grasped |
| 4 | cube inserted | cube seated flush in the hole (including the retreat home afterwards) |

The exact labelling rules, with worked episodes, are in section 3.1.

How each episode type is labelled:

- **Success:** 0 → 1 → 2 → 3 → 4. The retreat home is stage 4, because the
  cube stays inserted.
- **Failure like episode 13:** 0 → 1 → 2 → 3, and then *stays 3* through
  the release and the retreat, because the cube is not seated. It never
  reaches stage 4.
- **Recovery on the plate (pushed, or released and re-grasped where it
  stands):** 0 → 1 → 2 → 3 → 4, with a long stage 3. The cube never leaves
  the plate, so the stage never goes down (episodes 60 and 93).
- **Recovery by lifting again:** 0 → 1 → 2 → 3 → 1 (re-grasped and lifted
  off the plate) → 2 → 3 → 4. The stage goes down only when the cube really
  goes back.

Why this works without changing the model:
- Success at the end of an episode = "stage 4 predicted with high
  confidence". The same classifier gives both the sub-stage and task success.
- Failure episodes that also go home are essential. Without them the model
  can learn "arm at the home pose = success" from the robot state. With
  them, the home pose occurs in both stage 3 and stage 4, and only the image
  of the cube separates them. Episode 13 is exactly this kind of data.
- If the classifier still leans on the robot state, train with
  `model.no_state: true` (an existing config option) and compare.

τ within a segment: linear as now for normal segments. For the segment after
a failed release (stage 3, nothing happening) a constant τ (e.g. 0.5) avoids
teaching false progress. The RL reward does not use τ anyway.

### 3.1 Labelling rules

**Rule: look only at the cube, never at the gripper.** The stage is where
the cube is in that frame. What the gripper is doing (opening, moving away,
coming back, pushing) does not change the stage. Labelling the gripper's
motion instead of the cube is the most common mistake (it happened in the
middle of episodes 60 and 93).

| The cube is… | stage |
|---|---|
| on the table at its start spot | 0 approaching the cube |
| in the gripper, off the table, not yet over the plate | 1 cube grasped and lifted |
| in the gripper, over the plate, not touching it | 2 cube above the hole |
| on the plate but not seated: held or released, upright or tipped, being pushed or re-grasped | 3 cube being inserted |
| flush in the hole | 4 cube inserted |

**Boundaries (the frame where the new stage begins):**

| boundary | first frame of the new stage |
|---|---|
| 0 → 1 | the fingers close on the cube (gripper position stops at the cube's width, ≈ 0.011) |
| 1 → 2 | the cube is over the plate |
| 2 → 3 | the cube first touches the plate or the hole edge |
| 3 → 4 | the cube is flush: no edge sticks out above the plate surface. Not when the gripper opens; the cube can still drop or tip after that |

**Going back to an earlier stage (only when the cube really goes back):**

| what happens to the cube | stage |
|---|---|
| released on the plate, tipped or upright, next to or on the hole | stays 3 |
| pushed with closed fingers | stays 3 |
| re-grasped where it stands on the plate and slid or pressed in | stays 3 |
| re-grasped and lifted off the plate | 1, then 2 once over the plate, then 3 |
| falls off the plate onto the table | 0 |
| seated, then pulled or knocked out | back to 3 (or lower, by where the cube ends up) |

**Checks before saving an episode:**
- The labels start at frame 0 and end on the episode's last frame.
- The last stage is 4 if and only if the reward pedal was pressed
  (`next.reward`). The converter (`stages_to_sarm_reward.py`) refuses an
  episode that breaks this.
- 4 never comes before the cube is visibly flush in `cam_specialist`.
- A drop to stage 0 or 1 in the middle of an episode needs the cube to have
  left the plate. If it didn't, the segment is 3.

**Worked episodes (merged SARM dataset):**

| episode | what happens | correct labels |
|---|---|---|
| 0 | clean success | 0: 0–103, 1: 104–145, 2: 146–177, 3: 178–299, 4: 300–344 |
| 60 | cube held beside the hole, released upright on the plate (~340), re-grasped where it stands (~410), slid in, flush at ~607 | 0: 0–95, 1: 96–151, 2: 152–198, 3: 199–606, 4: 607–652 |
| 93 | released at ~308, tips onto the hole edge, pushed with closed fingers four times, flush at ~606 | 0: 0–137, 1: 138–191, 2: 192–236, 3: 237–605, 4: 606–675 |

In episode 60, the release and re-grasp (340–408) had been labelled 0. In
episode 93, the tipped cube (307–368) had been labelled 4. Both should be 3.

### 3.2 What the stages tell you, and what they don't

**Per frame:** the stage the cube is in. This is SARM's output.

**Per episode, success or failure:** the last stage. It ends at 4 means
success, it ends at 3 means failure (the cube is left unseated, as in
episode 13). For a success detector at run time: 4 predicted for k
consecutive frames after the gripper has opened.

**Recovery is not a stage.** A single frame cannot tell "first insertion
attempt" from "recovering from a failed one": a held cube at the hole looks
the same either way. That is intended, because it is the same world state
and should get the same reward. Recovery is visible from the stage *over
time*, together with the gripper signal:

| signal | means |
|---|---|
| the gripper opens while the stage is 3 | a failed release (the cube was let go unseated) |
| stage 3 lasts much longer than usual (clean episode 0: 6 s; episodes 60 and 93: 18–21 s) | a struggle at the hole: pushing, re-grasping |
| the stage drops (3 → 1 or 3 → 0) | the cube was lifted off again, or fell off the plate |
| any of the above, then the episode ends at 4 | a recovered episode |
| any of the above, then it ends at 3 | a failed episode |

So "is a recovery under way" = a failed release has happened and stage 4
has not been reached yet. It can be computed from the saved labels plus the
gripper position in the data, without extra labels. A per-episode tag
("clean", "recovered", "failed") can be derived the same way, for picking
episodes for ACT or for statistics.

The recovery *strategy* (pushing in episode 93, re-grasping in episode 60)
is not visible in the stages. Both are 3 throughout. Section 6 explains why
mixing strategies is a problem for ACT; that has to be tracked separately,
by gripper position (closed on nothing ≈ 0.004 = pushing, ≈ 0.011 = holding
the cube) or by hand.

**Alternative, only if stage 3 vs 4 stays unreliable:** add a separate binary
"cube inserted" head (BCE) on the stage model's fused features. Roughly
30-50 lines in `stage_estimator.py` + `sarm_ws.py` and one label column. Try
the world-state stages first; they need no code change.

---

## 4. Recording plan

1. **Camera for SARM: `cam_specialist`.** It is the only camera that sees the
   hole after the gripper leaves (episode 13). Wrist cameras move away with
   the arm. Use `cam_specialist` (+ wrists if wanted) as SARM's
   `camera_names`. The original 178 teleop demos do not have this camera, so
   SARM trains on the new recordings only.
2. **Keep recording with the infer app as now.** When ACT leaves the cube
   tipped:
   - in some episodes, **do not intervene**: let ACT finish and go home, then
     press the reset pedal (no reward). These are pure failure examples like
     episode 13.
   - in others, **intervene and recover** with the leader, then press the
     reward pedal once the cube is seated. These are recovery examples.
3. **Create failure states on purpose** if ACT fails too rarely. During an
   intervention, place the cube crooked on the hole, release, then either
   leave it (failure) or recover it. The same labelling rules apply.
4. **Pick one recovery strategy and always use it.** Section 6 explains why.
   Re-grasp, lift, re-align, insert is the reliable one. Pushing the cube on
   its inclined face sometimes works and sometimes not, and gives ACT two
   conflicting behaviours for the same image.
5. **Quantity.** Currently 3 failures in 51 episodes. That is a judgment
   call, not a measured number, but aim for a few dozen failure endings and a
   similar number of recoveries before training SARM's stage 3 vs 4
   decision. Use class weights in the cross-entropy if stage 4 frames
   dominate.

---

## 5. Labelling the frames

**What exists:** `tools/dataset_inspector.py` in opensarm shows frames and an
existing `reward` column, but it does not write labels. The infer datasets
have no `reward` column.

**Proposed (small, new):**
- A per-episode JSON with the frame where each segment starts, e.g.
  `{"13": [[0,0],[61,1],[150,2],[235,3]], ...}` (segment start frame, stage).
- A converter (~100-150 lines) that writes the SARM `reward` column
  (stage + τ, constant τ after a failed release) into a copy of the v2.1
  dataset, plus the task text and the state-normalization JSON SARM expects
  (`state_norm_path`).
- Finding boundaries: step through `cam_specialist` in the inspector. The
  `observation.intervened` flag and the gripper position (`action[6]`)
  narrow down grasp and release frames, so most boundaries need a quick check
  rather than a search.
- Episode-level check: the last stage of an episode must be 4 when
  `next.reward` was pressed and must not be 4 when it was not. The converter
  should refuse annotations that break this.

**SARM config for Station 3 (new yaml, no code):**
- `camera_names`: specialist (+ wrists)
- `state_dim: 7`
- `num_classes_sparse: 5` with the five texts above
- `use_dual_datasets: false`
- `frame_gap` chosen for 20 Hz and 250-800-frame episodes (the sim used 5
  with 8 steps, a 2 s window)

**Evaluate on held-out episodes:**
- per-frame confusion between stages 3 and 4
- the final-stage accuracy per episode against the pedal label (success
  episodes must end at 4, failures at 3)

---

## 6. ACT and recovery

**ACT can learn recovery without being stage-conditioned.** ACT is
conditioned on images and state. If the dataset contains frames where the
cube is tipped on the hole and the next actions are "re-grasp", ACT learns to
act from that image. This is HG-DAgger: your interventions from failure
states are the training signal.

Three things decide whether it works:

1. **One strategy.** This ACT is trained with `use_vae=false`
   (`train_act_deltajoint.sh`), so it has no latent to choose between modes.
   Two different recovery behaviours for the same image get averaged, and
   the average of "re-grasp" and "push the inclined face" does neither.
2. **Enough recovery examples, from varied failure states.** A handful of
   recoveries among ~230 normal episodes will be outweighed. Include them all,
   and if needed oversample recovery episodes when retraining.
3. **Recovery must start from what ACT itself produces.** Failure states
   created by ACT (as in episode 13) match what ACT will see at test time
   better than staged ones. Use both, but prefer ACT's own.

**What SARM adds:** SARM does not drive ACT. It gives:
- an automatic success detector (stage 4) to replace the reward pedal
- stage labels to find recovery segments
- a reward for the insertion-stage RL that drops when the cube tips (section
  7)

---

## 7. Using it in the insertion-stage RL

- **Reward:** stage potentials or milestones as now, but with
  `monotonic: false` for the last stages, so a tip-over after reaching
  stage 4 lowers the reward instead of keeping it. Keep hysteresis so one
  noisy frame does not flip it.
- **Success / termination:** stage 4 confirmed for k consecutive queries
  *after the gripper has opened*, to avoid scoring "cube held in the hole"
  as inserted. Keep the reward pedal as ground truth until SARM's 3-vs-4
  accuracy on held-out episodes is measured.
- **Offline buffer:** the stage labels give the insertion-stage start frame
  per episode (`FINAL_STAGE_RL_PLAN.md` 5.4.2), and recovery segments give
  real examples of states RL must handle.

---

## 8. Work and order

| step | code | notes |
|---|---|---|
| Record failures (pure) and recoveries (one strategy) | 0 | infer app as now |
| Annotation JSON + converter to SARM `reward` column | ~100-150 lines | new script, plus episode-level consistency check |
| Station 3 SARM config | yaml | 5 world-state stages, specialist camera |
| Train and evaluate SARM (3 vs 4 confusion, per-episode success) | 0 | existing `train.py` / `eval.py` |
| Optional "inserted" BCE head | ~30-50 lines | only if 3 vs 4 is unreliable |
| `monotonic: false` for the final stages in the reward server / client | config | already a parameter |
| Retrain ACT with recovery episodes (oversampled if needed) | 0 | `FINAL_STAGE_RL_PLAN.md` 5.2 |

## 9. Open decisions

1. Is the retreat home part of stage 4, or its own stage 5? It is labelled 4
   above because the world state does not change. A separate stage adds a
   class without adding information about the cube.
2. ~~Where exactly does stage 3 start?~~ Decided 2026-10-15: at the first
   contact of the cube with the plate or the hole edge (section 3.1).
3. The recovery strategy (re-grasp is recommended).
