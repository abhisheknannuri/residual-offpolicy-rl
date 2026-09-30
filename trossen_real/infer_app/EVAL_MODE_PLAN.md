# Eval mode for the infer app — design & implementation plan

Written for: whoever implements this (you or me, next session). Assumes the
reader knows `infer_app/app.py`, `infer_loop.py` and `PEDAL_BEHAVIOR.md`.

Adds a config-driven **Evaluation mode** to the existing infer app. Its job is to
replace the hand-filled markdown tables in
`trossen_real/EVAL/ACT/TrossenStation3/Notes.md` with data the app records
itself, and to regenerate those same tables automatically.

**Non-eval inference must stay byte-for-byte unchanged.** Every change below is
additive and inert when eval mode is off.

This supersedes `plan-configDrivenEvalModeInferUi.prompt.md`.

---

## 0. The actual protocol (from `EVAL/ACT/TrossenStation3/Notes.md`)

This grounds every decision below; read it first.

- **4 sub-stages**, in order:
  `0` Approach the Red Cube · `1` Grasp & Lift · `2` Transport to above the hole ·
  `3` Insert the cube.
- **20 eval poses** = *cube placements on the table* (X, Y cm + yaw ∈ {0, ±45°}),
  generated deterministically by `scripts/generate_eval_poses.py`
  (`seed=42`, workspace 22.5 × 14.5 cm). The **human places the cube** at pose *N*
  before each run. It is not a robot pose.
- **Max steps 500.** Policy served with an eval-time `--n-action-steps 12`, which
  differs from the trained `n_action_steps=20` — so it must be recorded per run.
- Per checkpoint you fill a **20-row pose sheet**: `S0 S1 S2 S3 | Steps |
  Pass/Fail | Failure mode`, then roll it up into a **checkpoint summary** table.
- **15 checkpoints** (5K…75K) × 20 poses = **300 runs.**

Two things follow directly from that last number:

1. **Annotation must cost ~1 keystroke.** Anything heavier will not survive 300
   runs, and half-filled sheets are worse than none.
2. **The app must generate the tables.** Hand-transcribing 300 rows into markdown
   is where the errors will come from.

### What your sheet actually encodes

Pose 01 has only `S2` ticked; pose 02 only `S3`. So the columns are **not** four
independent booleans — they record the **furthest sub-stage completed**. That
collapses the annotation to a single choice:

```
furthest_stage ∈ { none, 0, 1, 2, 3 }     # 3 == full success
```

Rendered back out as the four `- [ ]`/`- [X]` columns for ../../docs/archive/Notes.md. One keypress
(`0`–`4`, or `` ` `` for none) instead of four clicks, and it cannot produce an
impossible row like "S3 done but S1 not".

---

## 1. What already exists — do not rebuild it

| You want | Already there | Where |
|---|---|---|
| Per-step intervention flag | `TickLogger` writes one flushed JSONL line per tick with `step` + `intervened`, **on by default** | `infer_loop.py:347`, `policy_client.py:29` |
| Per-frame intervention in training data | `observation.intervened` column | `infer_loop.py:314` |
| Why the run ended | `stop_reason` ∈ `manual`, `max_steps`, `reset_pedal`, `reward_release`, `error` | `infer_loop.py:359-371` |
| Success signal | `stop_reason == "reward_release"` = operator pedal-marked success | `episode_boundary.py` |
| Timeout signal | `stop_reason == "max_steps"` = the Timeout column in your summary table | `infer_loop.py:362` |
| Steps | `snapshot.step` | `infer_loop.py:328` |
| Checkpoint + eval hyperparams | `policy_health`: `checkpoint`, `n_action_steps`, `chunk_size`, `device`, `likely_action_space` | `app.py:328` |
| Videos, tick log, dataset episode | one MP4/camera, one JSONL, one LeRobot episode per Start/Stop | `infer_loop.py:186-198, 376-381` |
| Sanity gates | `check_camera_health` blocks start; `check_action_space_mismatch` needs `force_action_space` | `app.py:371`, `policy_client.py:236` |

**So `Steps`, `Pass/Fail`, `Timeout`, and the checkpoint identity are already
captured today.** The only genuinely new operator input is *furthest stage* and
an optional failure note.

---

## 2. Corrections to the previous plan

1. **Four independent stage checkboxes → one "furthest stage" choice** (§0).
2. **"Add intervention capture to the loop" is redundant** — §1. Keep a live
   counter for the UI; derive authoritative indices from the tick log.
3. **Do NOT extend `InferSnapshot`.** It is rebuilt from scratch every tick
   (`infer_loop.py:332`) — that is why `log_file`/`video_files` are re-passed each
   iteration and why `stop_reason` is only set in the `finally` block. A new field
   that isn't re-passed silently resets forever. Eval state lives in `AppState`
   (§4), so the trap is structurally impossible.
4. **The polled UI will destroy operator input.** `poll()` rewrites `innerHTML`
   of every status row every 500 ms (`app.js:200-267`). A notes box inside that
   region gets wiped mid-typing (§7).
5. **No link back to training data** — record `dataset_dir` +
   `dataset_episode_index`.
6. **Live stage marking by keypress was wrong** (§3).

---

## 3. Intervention and stage attribution

Your objection was exact: the tick log knows *when* you intervened but not *which
stage* that was, because there is no stage detection. My earlier answer — press
`1..N` during the run — does not survive contact with the hardware: **while
intervening, your hands are on the leader arm.** It would be marked late, wrong,
or not at all.

Three points instead:

**a. For a fair checkpoint comparison, eval runs should be autonomous.** If you
take over, that run no longer measures the policy. So any run with intervention
is flagged `assisted: true` and **excluded from the headline success rate**,
which is reported alongside an assisted count. This protects the comparison you
actually want.

**b. When you do intervene, one dropdown at annotation time is enough.**
`intervened_during_stage ∈ {0,1,2,3,multiple}`, shown *only* when the run
recorded intervention. You already know the answer the moment the run ends; it
costs one click and needs no detection logic.

**c. The precise data is still on disk either way.** Step indices, segment count,
first intervention step and total intervened steps are computed from the tick log
and written into the summary, regardless of whether you attribute a stage. If you
later build stage detection, you can re-attribute retroactively without re-running
anything.

So: no live marking, no stage-detection requirement, and the headline metric
stays clean. Optional extra (§10, P4, only if it earns itself): a post-run
timeline that shades intervention segments so you can click stage boundaries
after the fact — precise, hands-free, but 300 runs' worth of friction, so it is
explicitly *not* in the main path.

---

## 4. Backend state

```python
# new: trossen_real/infer_app/eval_session.py
@dataclass
class InterventionSegment: start_step: int; end_step: int      # inclusive

class EvalRun:            # one Start->Stop cycle == one pose attempt
    run_id, started_at, ended_at
    checkpoint_label, pose_index, operator, tags, run_notes
    intervention_segments: list[InterventionSegment]
    step_count, stop_reason
    dataset_episode_index: int | None
    artifacts: {tick_log, videos[], dataset_dir}
    annotation: {furthest_stage: int|None, failure_note: str,
                 intervened_during_stage: int|str|None}
    status: "running" | "pending_review" | "saved" | "discarded" | "unreviewed"

class EvalSession:        # one connect->disconnect cycle
    config, output_root
    checkpoint_label                 # set once per session
    pose_cursor: int                 # 1..20, auto-advances on save
    current: EvalRun | None
    pending: EvalRun | None          # stopped, awaiting annotation
    completed: list[EvalRun]
```

`AppState` gains exactly one field: `eval: EvalSession | None`.

**Loop integration is one optional parameter and two call sites** — no
`InferSnapshot` change:

```python
# infer_loop.py
def start(self, ..., eval_hook: EvalHook | None = None)
...
    if eval_hook: eval_hook.on_tick(step=step, intervened=intervened_now)   # tick loop
...
    if eval_hook: eval_hook.on_end(stop_reason=stop_reason, steps=step)     # finally block
```

`on_tick` only edge-detects intervention segments and bumps a counter — no I/O,
nothing that can raise. Both calls are wrapped in `try/except` + log: **an eval
bookkeeping bug must never stop the arm.**

---

## 5. Pose session tracking

Across 300 runs the likeliest data-quality failure is losing track of which pose
you are on. So the session owns a cursor:

- `scripts/generate_eval_poses.py` gains an **`--out poses.json`** flag (new flag
  only; default behaviour, seed and printing unchanged) writing
  `{"seed": 42, "workspace_cm": [22.5, 14.5], "cube_cm": 3.0,
    "poses": [{"index": 1, "x_cm": 17.14, "y_cm": 5.99, "yaw_deg": 45.0}, ...]}`.
- The eval config points at that file; the UI shows **"Pose 7 / 20 — place cube at
  X=5.39 Y=7.43 cm, yaw −45°"** and the pose thumbnail if
  `20_eval_configuration.png` is available.
- The cursor auto-advances on Save, stays put on Discard, and is manually
  settable (re-running a pose).
- `/api/eval/session` reports which pose indices are already done for the current
  checkpoint, so an interrupted session resumes correctly.

---

## 6. Run lifecycle

```
idle ──start──> running ──stop──> pending_review ──save────> saved ──> pose_cursor++
                                        │         └─discard─> discarded (cursor stays)
                                        └── start blocked (409) while pending
```

- **Start refused with 409 while a review is pending.** Chosen over auto-saving
  (an unscored run pollutes the set) and over auto-dropping (loses a real
  rollout). The UI shows a banner with one-key Discard.
- **Disconnect / process exit with a pending review** → auto-finalize as
  `unreviewed` (files intact, flagged), mirroring how `_cleanup()` already
  finalizes dataset sessions (`app.py:477`).
- `dataset_episode_index` = the recorder's `num_episodes` read *before* the run
  (`app.py:346`), since eval runs map 1:1 onto Start/Stop cycles.
- `stop_reason` seeds the annotation: `reward_release` → `furthest_stage=3`
  pre-selected; `max_steps` → marks the Timeout column. Always operator-overridable.

---

## 7. Frontend

**The split that matters:**

| Region | Ownership | Update rule |
|---|---|---|
| status rows, live telemetry (step, Hz, intervened count, current pose) | polled | rewritten every 500 ms, as today |
| checkpoint label, pose cursor, stage choice, notes | **operator-owned** | written only on explicit user action, on run start (clear), on save (clear). **Never** touched by `poll()` |

Enforced structurally: the owned region gets its own container and its own render
function that `poll()` cannot reach.

**Per-run interaction budget — the design target:**

| Key | Action |
|---|---|
| `Space` | Start / Stop inference |
| `` ` `` `0` `1` `2` `3` | furthest stage = none / S0 / S1 / S2 / S3 |
| `Enter` | Save (advances pose cursor) |
| `Backspace` | Discard |
| `n` | focus the failure-note box (optional) |

So a normal run is **Space → 2 → Enter**. The failure note and the
`intervened_during_stage` dropdown are optional and only appear when relevant.

Lives in `static/eval.js` + an `#eval-panel` section, so lifting it into a
separate UI later is a file move. No framework, no build step.

---

## 8. Artifacts

```
eval_runs/
  index.jsonl                        # one line per finalized run - aggregate from this
  2026-09-28/
    ACT_BC_Station3_Trial1_delta_55k/
      pose_07/
        run_20260928_141233_7f3a/{summary.json, timeline.jsonl}
```

`index.jsonl` is the point: every table in §9 is a one-pass read of one file.

**`summary.json`:**

```json
{
  "schema_version": 1,
  "run_id": "20260928_141233_7f3a",
  "status": "saved",
  "started_at": "...", "ended_at": "...", "duration_s": 42.3,
  "station": "trossen_station3_single",
  "eval_config": {"name": "pick_cube_and_insert_station3", "hash": "ab12cd34"},
  "checkpoint_label": "ACT_BC_..._deltajoint/055000",
  "pose": {"index": 7, "x_cm": 5.39, "y_cm": 7.43, "yaw_deg": -45.0, "seed": 42},
  "policy": {"checkpoint": "...", "device": "cuda:0", "chunk_size": 20,
             "n_action_steps": 12, "action_space": "delta_joint",
             "likely_action_space": "delta_joint"},
  "run": {"steps": 314, "max_steps": 500, "stop_reason": "reward_release",
          "achieved_hz_mean": 19.8, "timed_out": false},
  "annotation": {"furthest_stage": 3, "failure_note": "",
                 "intervened_during_stage": null, "operator": "abhi"},
  "intervention": {"enabled": true, "assisted": false,
                   "total_intervened_steps": 0, "segment_count": 0,
                   "first_intervention_step": null, "segments": []},
  "artifacts": {"tick_log": "logs/infer_...jsonl",
                "videos": ["infer_videos/2026-09-28/..._cam_right_wrist.mp4"],
                "dataset_dir": null, "dataset_episode_index": null},
  "discard": null
}
```

Discarded runs are identical but `"status": "discarded"` plus
`"discard": {"reason", "at", "operator_note"}`. **Nothing is ever deleted** —
videos, tick log and dataset episode stay put. Aggregation filters on `status`.

**`timeline.jsonl`** — one line per *event*, not per tick (the tick log already
holds per-tick data): `run_start`, `intervention_start`, `intervention_end`,
`run_end` (+`stop_reason`), each with `t` and `step`.

---

## 9. Report generation — the payoff

`trossen_real/scripts/eval_report.py` (new, offline, reads only `index.jsonl`):

```sh
.venv/bin/python -m trossen_real.scripts.eval_report \
  --eval-root eval_runs --format markdown
```

Emits **exactly the two tables already in ../../docs/archive/Notes.md**, so output can be pasted
straight back in:

- **Checkpoint summary** — `Eval Done (20 Poses)`, per-sub-stage reached counts,
  `Full Success (x/20)`, `Median Steps`, `Timeout (x/20)`.
- **Pose-wise sheet** per checkpoint — `S0..S3` rendered as `- [X]` from
  `furthest_stage`, `Steps`, `Pass/Fail`, `Failure Mode`.

Plus `--format csv` for plotting success-vs-training-step across the 15
checkpoints, and an `--include-assisted` flag (default: **excluded** from the
headline rate, counted separately — §3a).

This is what makes the whole feature worth building: the tables stop being
hand-maintained.

---

## 10. Implementation phases

**P0 — pure, testable, no hardware** — ✅ **DONE** (30/30 tests pass)
1. ✅ `infer_app/eval_config.py` — schema, loader, validation, rubric hash.
2. ✅ `infer_app/eval_session.py` — `EvalRun`/`EvalSession`/`EvalHook`,
   edge-detected intervention segments, pose cursor, pending-review rules.
3. ✅ `infer_app/eval_writer.py` — summary/timeline/index writers, soft discard,
   slugified deterministic paths.
4. ✅ `scripts/eval_report.py` — the two markdown tables + CSV, from `index.jsonl`.
5. ✅ `infer_app/tests/test_eval_core.py` — 30 tests (§11).
6. ✅ `configs/eval/pick_cube_and_insert_station3.yaml` +
   `poses_station3_cube_20.json`, and `--out` on `generate_eval_poses.py`
   (additive flag; no-arg behaviour unchanged).

```sh
PYTHONPATH=. .venv/bin/python trossen_real/infer_app/tests/test_eval_core.py
PYTHONPATH=. .venv/bin/python -m trossen_real.scripts.eval_report --eval-root eval_runs
```

**P1 — backend wiring** — ✅ **DONE** (10/10 API tests)
6. ✅ `AppState.eval` + `eval_sync_lock`; `--eval-dir`; `eval_config_name` on
   `/api/connect` (validated BEFORE any hardware is touched).
7. ✅ `eval_hook` param + two call sites in `infer_loop.py`.
8. ✅ `eval` block on `/api/start_inference`; pending-review 409; `max_steps`
   defaults to the eval config's 500.
9. ✅ `/api/eval/{configs,config,pose_cursor,annotate,finalize}`; eval state
   merged into `/api/health` under `"eval"`.
10. ✅ `unreviewed` auto-finalize in `api_disconnect()` and `_cleanup()`.
11. ✅ `--out` on `generate_eval_poses.py`.

**P2 — frontend** — ✅ **DONE** (11/11 wiring tests)
12. ✅ `#eval-panel` + `static/eval.js`; polled (`#eval-live`) / operator-owned
    (`#eval-review`) split, enforced by a test that app.js never names an
    owned element.
13. ✅ Pose cursor + placement hint ("Pose #07 | X=5.39 Y=7.43 cm | Yaw −45°"),
    checkpoint label, "go to pose".
14. ✅ Keyboard annotation, Save/Discard, assisted-run stage dropdown.

**P3 — hardening** — ✅ **DONE**
15. ✅ Validation: stage bounds, finalize/annotate while running, unknown pose
    index, missing checkpoint label, bad finalize action — all covered by tests.
16. ✅ Hardware runbook (§13 below).

### Deviation from the plan, on purpose

§7 proposed `Space` to Start/Stop. **Not implemented:** no keyboard shortcut in
`eval.js` can start or stop the robot, and a test asserts `eval.js` cannot even
reach those endpoints. A stray keypress must never command motion. Start/Stop
stay as explicit buttons; the keys are annotation-only and therefore post-run.
The per-run budget is still ~2 actions: click Stop (or let it time out), then
`2` + `Enter`.

**P4 — only if it earns itself**
17. Post-run intervention timeline with click-to-assign stage boundaries (§3).

---

## 11. Verification

**Automated (no hardware) — where the real coverage is:**
- intervention segments: starts intervened, ends intervened, single-tick blip,
  alternating, never intervened
- writer: save path; **discard leaves every artifact in place**; `index.jsonl`
  append-only, one line per finalized run; re-finalize refused
- report: `furthest_stage` → `- [X]` column rendering; median steps with an even
  count; timeout counting; assisted runs excluded from the headline rate;
  partially-complete checkpoint (7/20 poses done)
- pose cursor: advance on save, hold on discard, resume after interruption
- config: duplicate stage ids, empty stages, pose file missing/short
- **the hook never raises into the loop** (inject a failing writer, assert the
  tick loop survives)

**Manual on hardware:**
1. **Regression first — eval OFF:** connect/start/stop/disconnect; dataset +
   video + tick log identical to today. This is the one that must not break.
2. One real pose run → `summary.json` matches what you saw; `Space → 2 → Enter`
   works; cursor advances 7 → 8.
3. Discard → files present, `status="discarded"`, cursor unchanged.
4. Start while pending → 409 + banner; discard, then start.
5. Intervene mid-run → `assisted: true`, stage dropdown appears, run drops out of
   the headline rate but is counted.
6. Kill the app with a review pending → `status="unreviewed"`, nothing lost.
7. Run `eval_report.py` → paste into ../../docs/archive/Notes.md, compare against a hand-filled row.

---

## 12. Decisions

**Settled (from ../../docs/archive/Notes.md + your reply):**
- Stages: the 4 sub-stages verbatim from ../../docs/archive/Notes.md.
- Pose = cube placement (X, Y, yaw), 20 deterministic poses from seed 42;
  human places the cube, app tracks the cursor.
- Annotation = **furthest sub-stage reached**, one keystroke; rendered back to
  the four `- [X]` columns.
- Intervention → `assisted` flag + optional stage dropdown; assisted runs
  excluded from the headline success rate. No live marking, no stage detection.
- Soft discard; start blocked while a review is pending; shutdown finalizes
  `unreviewed`.
- The app generates the ../../docs/archive/Notes.md tables.

**Deliberately deferred:** automatic stage detection, and the post-run timeline
(P4). Neither is needed for the 300 runs in front of you.

**Operational note:** disk was at 99% during the Station3 dataset prep. 300 runs
× 2 cameras of MP4 is real volume — consider `record_video=false` for eval runs
where the tick log and summary suffice, or prune after each checkpoint.

---

## 13. Hardware runbook (as built)

### Automated tests — run these first, they need no hardware

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
PYTHONPATH=. .venv/bin/python trossen_real/infer_app/tests/test_eval_core.py      # 30
PYTHONPATH=. .venv/bin/python trossen_real/infer_app/tests/test_eval_api.py       # 10
PYTHONPATH=. .venv/bin/python trossen_real/infer_app/tests/test_eval_frontend.py  # 11
```

### 1. Regression check — the one that must not break

Start the app **without** touching the eval dropdown (leave it "off"):

```sh
.venv/bin/python -m trossen_real.infer_app.app --port 5080
```

Connect → Start → Stop → Disconnect exactly as before. Expect: no eval panel,
`/api/health`'s `eval` field is `{"eval_mode": false}`, and **no `eval_runs/`
directory is created**. Dataset recording and video are unaffected.

### 2. Eval session

```sh
# policy server (other terminal, its own venv - unchanged)
CUDA_VISIBLE_DEVICES=1 uv run python custom_scripts/policy_server.py \
  --checkpoint .../ACT_BC_PickCubeAndInsert_Station3_Trial1_deltajoint/checkpoints/055000/pretrained_model/ \
  --n-action-steps 12 --device cuda:0

# follower server (other terminal, unchanged)
.venv/bin/python -m trossen_real.follower.follower_single_server \
  --config trossen_station3_single --port <port>

# the app
.venv/bin/python -m trossen_real.infer_app.app --port 5080 --eval-dir eval_runs
```

In the browser: pick **Eval mode = `pick_cube_and_insert_station3`**, type the
**Checkpoint label** (e.g. `ACT_..._deltajoint/055000`) and your name, Connect.

Per pose, ~2 actions:
1. The panel shows **Pose 1/20** and where to place the cube. Place it.
2. **Start Inference.** It stops itself at 500 steps, or on the reward pedal
   (success) / reset pedal (abort).
3. Press `` ` `` `0` `1` `2` `3` for the furthest sub-stage, then **Enter** to save.
   The cursor advances to pose 2. `Backspace` discards instead (cursor holds).

Notes:
- A run stopped by releasing the reward pedal pre-selects stage 3 (full success);
  override if that is wrong.
- If you intervened, the panel says `ASSISTED` and offers a "intervened during"
  dropdown. Assisted runs are excluded from the headline rate.
- Starting a new run before saving the previous one is refused with a clear 409.
- Killing the app mid-review writes the run as `unreviewed` — nothing is lost.

### 3. Report

```sh
PYTHONPATH=. .venv/bin/python -m trossen_real.scripts.eval_report --eval-root eval_runs
PYTHONPATH=. .venv/bin/python -m trossen_real.scripts.eval_report --eval-root eval_runs --format csv
```

Paste the markdown straight into `EVAL/ACT/TrossenStation3/Notes.md`.

### Rollback

Every file modified in P1/P2 was backed up next to itself before editing:

```sh
ls trossen_real/infer_app/*.20260928_141157.bak trossen_real/infer_app/static/*.20260928_141157.bak
```

Restoring those five files returns the app to its pre-eval state; the new
`eval_*.py`, `static/eval.js`, `configs/eval/` and `scripts/eval_report.py` are
additive and can simply be ignored or deleted.
