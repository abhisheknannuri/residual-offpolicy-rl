# Robot web UI architecture — inventory, prior art, and a proposal

Status: **proposal. Nothing implemented.**

This document is written to be reviewed by people (and models) who do **not**
have this repo. Everything factual is stated with the file, line count or
measurement it came from, so a reviewer can challenge the premises and not just
the conclusions.

**Reviewers: the questions I most want attacked are in §9.** The two decisions
I am least sure about are (a) merging teleop and inference into one app, and
(b) how far to go toward LeRobot's robot abstraction given the version pin in
§5.3.

---

## 1. Context

A research setup for **residual off-policy RL on a real robot arm**: a frozen
behaviour-cloning policy (ACT) is served over HTTP, and a TD3 residual policy
learns a correction on top of it. The arm is a Trossen bimanual-capable
follower used in single-arm mode, with two wrist cameras, a leader arm for
human intervention, and a foot pedal.

Three browser UIs exist, built at different times by different means. They
overlap, duplicate each other, and are each hardcoded to one robot. The goal is
a plan that:

- consolidates where consolidation is genuinely right, not just tidier;
- makes **adding a different robot** a configuration/plugin act, not a fork;
- leaves room for features not yet imagined;
- reuses open-source work instead of rebuilding it.

Scale note, so reviewers calibrate: this is a **single-researcher, single-robot
lab**, not a fleet product. Solutions that need a team to operate are the wrong
answer here.

---

## 2. Inventory — what exists today

### 2.1 Teleop app — `trossen_real/teleop/`

Flask, vanilla JS. Drive the arm from the leader arm and record demonstrations
into a LeRobot dataset.

| file | lines |
| --- | --- |
| `app.py` | 556 |
| `control_loop.py` | 411 |
| `dataset_recorder.py` | 329 |
| `follower_client.py` | 117 |
| frontend (`index.html` + `app.js` + `style.css`) | 550 |

Endpoints: `/api/stations`, `/api/connect`, `/api/disconnect`, `/api/health`,
`/api/get_state`, `/api/reset`, `/api/gripper`, `/api/start_session`,
`/api/stop_session`, `/api/start_recording`, `/api/stop_recording`,
`/video_feed/<cam>`.

Features: station selection, connect/disconnect, live MJPEG camera feeds, joint
state readout, staged reset, gripper control, session and episode lifecycle,
pedal-driven episode boundaries, LeRobot dataset recording.

### 2.2 Inference / eval app — `trossen_real/infer_app/`

Flask, vanilla JS. Run a BC policy against the arm and score structured
evaluation campaigns.

| file | lines |
| --- | --- |
| `app.py` | 904 |
| `infer_loop.py` | 415 |
| `eval_session.py` | 520 |
| `eval_config.py` | 235 |
| `eval_writer.py` | 159 |
| `video_recorder.py` | 179 |
| frontend (`index.html` + `app.js` + `eval.js` + `style.css`) | 931 |

Endpoints: the same six connection/health/reset/video endpoints as teleop, plus
`/api/start_inference`, `/api/stop_inference`, `/api/eval/configs`,
`/api/eval/config`, `/api/eval/pose_cursor`, `/api/eval/annotate`,
`/api/eval/finalize`.

Features: everything teleop has for connection and video, plus policy-server
connection with health/action-space cross-check, chunked action fetching,
configurable image encoding (raw / zlib / jpeg), per-tick diagnostic logging,
H.264 video recording, **human intervention mid-episode** via leader + pedal,
and an eval campaign system: YAML-declared stages and cube poses, per-pose
progress, stage annotation, `summary.json` / `timeline.jsonl` / `index.jsonl`
per run.

### 2.3 Dataset visualizer — `lerobot-dataset-visualizer` (separate repo)

A fork of HuggingFace's
[lerobot-dataset-visualizer](https://github.com/huggingface/lerobot-dataset-visualizer),
substantially extended. **The most mature of the three by a wide margin.**

Stack: Next.js 15 (App Router), React 19, TypeScript, Tailwind 4, Recharts,
Three.js + `@react-three/fiber` + `@react-three/drei`, `urdf-loader`,
`hyparquet` (browser-side Parquet), Bun; plus an optional **FastAPI** backend
for dataset writes. Docker + Compose. ESLint, Prettier, `tsc --noEmit`, Bun
tests.

~40 TS/TSX files. Features, from its own README:

- dataset / episode navigation across orgs, local folders and the HF Hub
- video playback synchronised with interactive time-series charts
- overview, statistics (episode-length histogram, fps, totals)
- **action insights**: autocorrelation, state-action alignment, speed
  distribution, cross-episode variance heatmap
- **filtering**: flag low-movement / jerky / outlier-length episodes, export a
  ready-to-run LeRobot CLI command
- **3D URDF viewer** with end-effector trail, for SO-100/101, OpenArm bimanual,
  Unitree G1
- **annotations**: hand-edit the v3.1 language schema (`language_persistent` +
  `language_events`) — subtask, plan, memory, interjection, VQA atoms with
  bbox / keypoint / count / attribute / spatial answers; bboxes and keypoints
  render as draggable overlays on the video
- **RL tab**: view and edit sparse success labels (`next.reward` / `next.done`)
  per frame, staged server-side in `meta/lerobot_rl_labels.json`, committed to
  Parquet for both dataset v2.1 (one file per episode) and v3.0 (shared files
  plus `meta/episodes/*.parquet` stats rows), with `.rl_bak` backups
- lazy-loaded panels, pagination, chunked reads

This app is **not** the problem. It is the model to learn from.

---

## 3. What is actually wrong — measured, not asserted

### 3.1 Two Flask apps are one app wearing two hats

`/api/stations` and `/api/health` are **byte-identical** between the two apps
(verified by hashing the function bodies). `/api/connect` differs only slightly.
Both re-implement MJPEG streaming, station loading, connection lifecycle and
staged reset.

More importantly, the control loops are structurally the same thing. Each tick:
read follower state → read cameras → **obtain an action** → send to follower →
optionally record. The only real difference is where the action comes from.

And the inference app already demonstrates this, at
`infer_app/infer_loop.py:282-290`:

```python
result = self.policy.predict_full(obs)
...
if intervened_now:
    action = self.intervention.get_leader_action()
else:
    action = policy_action
```

It already switches between a **policy** action source and a **leader-arm**
action source at runtime. Teleop is that same loop with the leader source
pinned on. The abstraction is half-built by accident.

### 3.2 Two hand-rolled frontends, no shared code, no build step

~1,480 lines of vanilla HTML/CSS/JS across two `static/` directories with
nothing shared. Adding a control means editing two files in two places, or
accepting drift. There is no component model, no type checking, no tests on the
frontend. Meanwhile the third app has all of those.

The checkbox sprawl that prompted this review is a symptom: with no schema and
no component library, every new capability becomes another hand-written
`<input type=checkbox>` plus hand-written JS plus a hand-parsed request field.

### 3.3 Robot identity is hardcoded in all three, differently

- **Teleop / infer**: Trossen-specific throughout — `FollowerClient`,
  `TrossenSingleLeader`, station YAMLs, `action_dim=7` in places, "single arm
  mode" assumptions.
- **Visualizer**: `getRobotConfig()` (`urdf-viewer.tsx:45`) is an if/else chain of
  **substring matches** on the dataset's `robot_type` string
  (`lower.includes("so100")`, `lower.includes("openarm")`, …) with a
  fall-through default to SO-101. Joint units are **guessed by magnitude**
  (`>360` → servo ticks, `>6.3` → degrees, else radians). There is a hardcoded
  `G1_SDK_TO_URDF` dictionary mapping ~30 column names to URDF joint names.

That last one is the clearest evidence of the problem: three robots in, and the
mechanism is already string sniffing plus a per-robot lookup table pasted into a
React component. A fourth robot makes it worse, not linearly.

### 3.4 Eight logging sinks, four roots, no shared run id

Covered in detail in
[`docs/real/RESIDUAL_RL_DEPLOY_AND_EVAL_PLAN.md`](../real/RESIDUAL_RL_DEPLOY_AND_EVAL_PLAN.md)
§2. Summary: Python logging, a stdout tee, per-tick JSONL, MJPEG-era video,
LeRobot datasets, eval JSON/JSONL, latency dumps and RL training logs all land
in different places keyed only by wall-clock timestamp. Any UI that wants to
show "this run" has to join on timestamps.

**The UI work depends on that being fixed first.** A browser cannot usefully
display a run that has no identity.

---

## 4. Prior art — what the ecosystem already provides

Surveyed because the brief was explicitly "don't implement everything from
scratch".

### 4.1 LeLab — the closest thing to what we want

[LeLab](https://github.com/DavidLMS/leLab) puts the LeRobot workflow
(calibrate, teleoperate, record, train, deploy) in a browser. Stack: **FastAPI
backend + React/TypeScript + Tailwind + Vite, with WebSocket for live joint
data.**

Relevance: this is independent confirmation of the stack I would otherwise be
proposing from first principles, and it covers calibration and teleop+record,
which is most of our teleop app. Its eval/dataset-browsing story appears thin,
which is exactly where our other two apps are strong.

**Verdict:** adopt the architecture, and read its code before writing ours.
Whether to fork it or borrow from it is open question Q4.

### 4.2 HuggingFace lerobot-dataset-visualizer — already our upstream

Our visualizer is a fork. Staying mergeable with upstream is worth real effort:
we get dataset v3.0 support, new robots and new panels for free. Every
divergence we add is a merge conflict later.

**Verdict:** keep the fork thin and upstream-shaped. Push generic improvements
(a real robot registry, for instance) upstream rather than carrying them.

### 4.3 Rerun — visualisation we should not build

[Rerun](https://rerun.io) is purpose-built for robotics: time-aligned images,
scalars, transforms and 3D with a scrubbing viewer. **`rerun-sdk 0.35.0` is
already installed in this repo's venv.** Measured here: **0.79 ms/tick** to log
2 images + 9 scalars; a 350-step episode produces a ~15 MB `.rrd`. It also
ships `serve_web_viewer()` and `serve_grpc()`, so it can serve its viewer into
a browser tab.

**Verdict:** this is the answer for live signal inspection and post-hoc
debugging. Do not build time-series plotting, video scrubbing, or 3D into Flask.

### 4.4 Foxglove / MCAP

Mature, strong multi-panel layouts, MCAP is a good container. But it is
ROS-shaped, and this stack has no ROS. It would buy interop we do not need at
the cost of a dependency plus a writer plus a reader.

**Verdict:** skip. Revisit only if ROS enters the picture.

### 4.5 BEAVR

[BEAVR](https://arxiv.org/html/2508.09606v1) — open-source bimanual
multi-embodiment VR teleoperation, explicitly designed for heterogeneous
platforms from 7-DoF arms to humanoids.

**Verdict:** not adopting (no VR hardware), but its **multi-embodiment
abstraction** is worth reading as a design reference for §5.

### 4.6 What the survey did *not* turn up

No open-source tool covers **structured, resumable, multi-checkpoint policy
evaluation campaigns with human stage annotation**. Our eval app's design —
YAML stages, fixed object poses, per-pose progress, assisted-vs-clean
accounting — appears to be genuinely ours. That is the piece worth keeping and
investing in; almost everything else has a better off-the-shelf answer.

---

## 5. The robot abstraction

This is the part the brief cared most about, and it has a good answer that
already exists.

### 5.1 LeRobot already defines the contract

Modern LeRobot (v0.6.x) provides a `Robot` abstract base class
([docs](https://huggingface.co/docs/lerobot/en/integrate_hardware)) whose
contract is:

| member | meaning |
| --- | --- |
| `observation_features` → dict | shape/type of everything `get_observation()` returns; **must work before connecting** |
| `action_features` → dict | shape/type `send_action()` expects |
| `is_connected` → bool | comms established |
| `connect(calibrate=True)` | open comms, calibrate if needed, configure |
| `disconnect()` | release ports, threads, cameras |
| `is_calibrated` → bool | calibration loaded |
| `calibrate()` | learn ranges of motion, normalise raw positions |
| `configure()` | idempotent hardware setup (control modes, gains) |
| `get_observation()` → dict | the core read; keys match `observation_features` |
| `send_action(action)` → dict | the core write; returns what was actually sent |

Crucially, **the base class assumes nothing about form factor** — the docs say
so explicitly, and it is used for 5-DoF arms through humanoids.

There is a matching `Teleoperator` base class with `get_action()` and
`send_feedback()`. A leader arm, a gamepad and a phone are all teleoperators.

### 5.2 And it already defines the plugin mechanism

Two layers:

1. **Registry.** `@RobotConfig.register_subclass("my_cool_robot")` on a
   dataclass inheriting `RobotConfig`, resolved at runtime via draccus
   choice-types from `--robot.type=my_cool_robot`.
2. **Out-of-tree discovery.** A pip package named `lerobot_robot_*`,
   `lerobot_camera_*` or `lerobot_teleoperator_*`, following a naming
   convention (`FooConfig` ↔ `Foo`) and exposing both in `__init__.py`, is
   **auto-discovered with no changes to LeRobot's source.** Community examples
   exist (`lerobot-robot-xarm`, `lerobot-teleoperator-teleop`).

So the right shape for "easy to add another robot" is: **Trossen becomes
`lerobot_robot_trossen`, and the web UI talks to the `Robot` interface, never to
Trossen classes.** Adding a robot is then publishing a package, and the UI
discovers it.

### 5.3 The constraint that makes this not free

**The vendored LeRobot in this repo is 0.1.0, with the old
`lerobot/common/...` layout and `CODEBASE_VERSION = "v2.1"`. Upstream is
0.6.1.** `lerobot.robots` does not exist in the vendored copy — I checked:
`ModuleNotFoundError`.

And the pin is load-bearing: the ACT checkpoints and the offline dataset are
dataset-v2.1 artifacts, read by this exact vendored code. Upgrading LeRobot to
get `Robot` would mean migrating datasets and re-validating the BC checkpoints,
which is a research-schedule risk, not a refactor.

**Therefore: mirror the interface, do not import it.** Define our own
`Robot`-shaped protocol with the same member names and semantics, implement
Trossen against it, and keep it close enough that a later swap to the real
`lerobot.robots.Robot` is a rename plus deletion of our protocol — not a
redesign. Where the name is already right (`connect`, `disconnect`,
`is_connected`, `get_observation`, `send_action`), use it verbatim.

This is the single most important judgement call in the document and I would
like it challenged.

### 5.4 The piece LeRobot does *not* give us

LeRobot's contract covers **hardware I/O**. A UI needs more:

- **Presentation metadata:** display names, joint ordering for charts, units
  and limits for axes, which cameras to show where, URDF URL and scale,
  column-name → URDF-joint mapping.
- **Capability flags:** does this robot have a leader arm? a pedal? a gripper?
  bimanual? Can it be reset programmatically?

Today those live as substring matches and lookup tables (§3.3). They should be
**one declarative descriptor per robot**, served to the browser, so the UI
renders from data instead of branching on strings:

```yaml
# robots/trossen_station3_single.yaml   (illustrative)
id: trossen_station3_single
display_name: "Trossen Station 3 (single arm)"
embodiment: single_arm
joints:
  - {name: joint_0, unit: rad, limits: [-3.14, 3.14], chart_group: arm}
  # ...
  - {name: gripper, unit: normalized, limits: [0, 1], chart_group: gripper}
cameras:
  - {key: observation.images.cam_left_wrist,  role: wrist_left,  display: "Left wrist"}
  - {key: observation.images.cam_right_wrist, role: wrist_right, display: "Right wrist"}
capabilities: [leader_teleop, pedal, gripper, programmatic_reset, intervention]
urdf: {url: "...", scale: 1.0, column_to_joint: {joint_0: joint_0, ...}}
```

The visualizer's unit *guessing* (`>360` → ticks) becomes a declared `unit`.
`G1_SDK_TO_URDF` becomes that robot's `column_to_joint`. `hasURDFSupport()`
becomes "does the descriptor have a `urdf` block".

**This descriptor is generic enough to be worth upstreaming** to
lerobot-dataset-visualizer, which would retire our fork's divergence on the
robot-registry question entirely.

---

## 6. Proposed architecture

### 6.1 Three surfaces, not three apps

| surface | what it is | build or adopt |
| --- | --- | --- |
| **A. Robot Console** | drive the robot: teleop, inference, eval campaigns | **build** (merge of the two Flask apps) |
| **B. Data Browser** | datasets, run records, episodes, annotations, labels, Q plots | **extend** the existing Next.js visualizer |
| **C. Signal Inspector** | live and post-hoc time-aligned signals, images, 3D | **adopt Rerun** |

The split is by *interaction model*, which is why it holds: A is a realtime
control panel with a human and a moving robot; B is an archive you browse; C is
a scrubbing timeline. Those want different UIs, and forcing them into one page
is how you get the current checkbox wall.

Surfaces A and B stay separate deployments. They share the robot descriptor,
the design tokens, and a generated TypeScript API client — not a monolith.

### 6.2 Surface A: one console, pluggable action source

The merge argument from §3.1, made concrete:

```text
                     ┌──────────── ActionSource (protocol) ────────────┐
                     │  get_action(obs) -> action                      │
                     │  reset()                                        │
                     └─────────────────────────────────────────────────┘
                        ▲             ▲              ▲            ▲
                 LeaderArm      PolicyClient   ResidualPolicy   Replay
                 (teleop)       (BC infer)     (BC + RL)        (debug)

  one control loop:
    obs = robot.get_observation()
    action = source.get_action(obs)          # or intervention override
    robot.send_action(action)
    recorder.write(obs, action, extras)      # optional
```

Mode becomes **configuration, not a separate application**. Human intervention
stops being an inference-only feature and becomes "a second source that can
preempt the first", which is what it already is in the code.

What this collapses: one connection lifecycle, one MJPEG/WebRTC path, one
station loader, one reset, one recorder, one frontend.

What it must not break: the eval campaign system, pedal semantics, and the
exact recording conventions (pre-action observation ordering) that existing
datasets depend on.

### 6.3 Stack for Surface A

**FastAPI + React/TypeScript + Tailwind + Vite.** Reasons, in order:

1. It is what the mature third app already is (minus FastAPI), so the team of
   one already maintains this stack.
2. It is what LeLab independently chose for the same problem (§4.1).
3. FastAPI gives OpenAPI for free → a **generated** TypeScript client, so the
   checkbox/field drift in §3.2 becomes a type error instead of a runtime bug.
4. WebSocket for state/telemetry instead of polling.
5. Pydantic models are the config schema (§6.5) and the API schema at once.

Video: keep MJPEG initially because it works and is trivial; WebRTC only if
latency or bandwidth proves it necessary. Not a day-one concern.

### 6.4 Surface B: extend the visualizer to read run records

The visualizer already reads LeRobot datasets, renders synchronised video +
charts, and has an RL tab. The planned **run record** (one `run_id`,
`ticks.parquet`, `rl_obs.zarr`, `meta.json` — see the deploy/eval plan §3)
is deliberately Parquet-shaped so that `hyparquet` can read it in the browser
with the machinery that already exists.

Additions:

- a **Runs** section alongside Datasets: list, filter by checkpoint/pose/outcome
- an **Eval campaign** view: the checkpoint × pose grid, success/assist/stage
  outcomes, drill-through to one episode
- **Q-value plots** from `q_table.parquet`, same figures as the sim runs
- reuse the existing playback bar, time context and chart pipeline unchanged

### 6.5 Config: schema + presets, which is the answer to the checkbox wall

One Pydantic model tree per surface, with a `validate()` that returns
structured errors **and** the dependency rules, e.g.:

```
policy.mode == "residual"   requires  policy.residual_checkpoint
policy.collect_q            requires  policy.mode == "residual"
policy.jpeg_quality         ignored unless image_encoding == "jpeg"   (warn)
policy.chunk_steps != 0     requires  server supports /predict_chunk
```

`GET /api/config/schema` serves fields, types, defaults and rules; the form is
**generated** from it. A new field needs no hand-written HTML.

Then **presets** as YAML, because ~90% of runs are one of a handful of setups:
`teleop_record`, `bc_baseline`, `residual_eval`, `residual_actor_only`,
`debug_mock`. The UI becomes *pick a preset → see the diff → override a field
or two → Start*, with the long tail behind an "Advanced" disclosure. The chosen
preset plus the override diff is what gets recorded in the run's `meta.json`,
which also buys reproducibility.

### 6.6 What ties the three surfaces together

Only three things, deliberately:

1. **The robot descriptor** (§5.4) — one file per robot, served by A, consumed
   by A, B and C.
2. **The run record** — written by A, read by B and C.
3. **A generated API client + shared design tokens** — so A and B look and feel
   like one system without being one codebase.

No shared database, no message bus, no monorepo requirement.

---

## 7. Migration path

Ordered so that every step is independently useful and nothing is a flag day.

| phase | work | why here |
| --- | --- | --- |
| **0** | **Run record + run_id** (deploy/eval plan §2–3) | every UI feature below needs runs to have identity |
| **1** | Robot descriptor schema; write one for the Trossen station; serve it from the existing Flask app | pure addition, no UI change, immediately unblocks B |
| **2** | Extract `ActionSource` protocol and the shared control loop **inside the current Flask apps**; keep both frontends | the risky refactor, done where it can be tested against hardware without a UI rewrite |
| **3** | Extract the duplicated endpoints into one FastAPI service behind the existing frontends | API consolidation, still no UI rewrite |
| **4** | New React console for Surface A against that API; retire the two `static/` dirs | now a rewrite of ~1,480 lines of vanilla JS, not of the robot logic |
| **5** | Visualizer: Runs section + eval campaign grid + Q plots | the payoff for phase 0 |
| **6** | Replace visualizer robot sniffing with the descriptor; **offer upstream** | removes the worst hardcoding; may remove our fork divergence |
| **7** | Rerun integration: live `connect_grpc()` during runs, `.rrd` generated from run records | last because it is additive and independent |

Phases 1–3 are reversible and testable. Phase 4 is the only big-bang, and by
then the robot logic is already behind an API.

### Do a second robot as the test

The abstraction is unproven until something else uses it. The cheapest proof is
a **mock/simulated robot** implementing the same protocol — this repo already
has `follower_single_server_mock.py`, so a `MockRobot` descriptor plus adapter
would let the whole console run with no hardware. That is worth having anyway
for UI development, and it is the forcing function that keeps Trossen
assumptions from leaking back in.

---

## 8. Explicit non-goals

- **No ROS, no Foxglove, no MCAP.** Not justified by anything in this stack.
- **No multi-user, no auth, no RBAC.** Single researcher, trusted LAN.
- **No fleet management, no cloud deployment.** One robot, one workstation.
- **Not merging the data browser into the console.** §6.1 argues they are
  genuinely different interaction models.
- **Not building time-series/video/3D viewers.** That is Rerun and the existing
  visualizer.
- **Not upgrading vendored LeRobot as part of this.** §5.3 — separate decision
  with its own risk.
- **No WebRTC until MJPEG is measured to be the bottleneck.**

---

## 9. Questions I want reviewers to attack

**Q1 — Is merging teleop and inference right?** §3.1 and §6.2 argue they are
one loop with different action sources, and the code already switches sources
at runtime. Counter-argument: teleop is a *data collection* tool with different
users, failure modes and safety posture than an *evaluation* tool, and merging
couples two things that change for different reasons. Which risk is worse?

**Q2 — Mirror LeRobot's `Robot` interface, or bite the version upgrade?**
§5.3. Mirroring risks a permanent near-miss that never converges. Upgrading
risks the BC checkpoints and dataset v2.1 artifacts. Is there a third option —
e.g. an adapter package that depends on modern `lerobot` only for the ABC,
while the training code keeps the vendored copy?

**Q3 — Is the robot descriptor the right abstraction boundary?** It separates
*hardware I/O* (LeRobot's `Robot`) from *presentation and capabilities* (ours).
Is that a real seam, or will the two inevitably leak into each other? Is there
existing prior art for a declarative robot-presentation descriptor that I
missed?

**Q4 — Fork LeLab, or build Surface A fresh?** It already does calibrate /
teleoperate / record in FastAPI + React. Forking could save months but imports
its abstractions, which may not fit a residual-RL eval workflow. Building fresh
keeps the eval campaign system (§4.6, the one genuinely novel piece) at the
centre. Has anyone read LeLab's code closely enough to judge?

**Q5 — Is three surfaces one too many?** Could Rerun's `serve_web_viewer()`
absorb Surface C *and* chunks of B, leaving just a control panel and Rerun?
What breaks — annotation editing, campaign grids, dataset writes?

**Q6 — Phase ordering.** Phase 0 (run record) gates everything and touches the
training loop. Is it right to block UI work on it, or should Surfaces A/B
proceed against the existing eight sinks and migrate later?

**Q7 — What features do specialist robotics UIs have that this plan misses?**
The §4 survey found no off-the-shelf answer for evaluation campaigns, which
makes me suspect I am missing a term of art or a tool. Calibration wizards,
safety interlocks, dataset quality gates, teleop latency monitors, policy A/B
comparison — what else should be on the roadmap?

---

## 10. Appendix — measured facts

| fact | source |
| --- | --- |
| `/api/stations`, `/api/health` byte-identical across the two Flask apps | md5 of function bodies |
| Teleop: 1,413 Python + 550 frontend lines | `wc -l` |
| Infer/eval: 2,412 Python + 931 frontend lines | `wc -l` |
| Visualizer: ~40 TS/TSX files, Next.js 15 / React 19 / Tailwind 4 | `package.json`, file listing |
| `infer_loop.py:282-290` already switches policy ↔ leader action source | source |
| Visualizer robot selection is substring matching with SO-101 fallback | `urdf-viewer.tsx:45` (`getRobotConfig`) |
| Visualizer guesses joint units by magnitude (>360 ticks, >6.3 degrees) | `urdf-viewer.tsx:63` (`detectAndConvert`) |
| `G1_SDK_TO_URDF` hardcodes ~30 column→joint mappings in a component | `urdf-viewer.tsx` |
| Vendored LeRobot is 0.1.0, `lerobot/common/...` layout, `CODEBASE_VERSION="v2.1"` | `lerobot.__version__`, `lerobot_dataset.py:76` |
| `lerobot.robots` absent from vendored copy | `ModuleNotFoundError` |
| Upstream LeRobot is 0.6.1 | PyPI |
| `rerun-sdk 0.35.0` already installed | `importlib.metadata` |
| Rerun logging: 0.79 ms/tick for 2 images + 9 scalars; ~15 MB per 350-step episode | measured in this venv |
| Eight logging sinks across four roots, no shared run id | deploy/eval plan §2 |

## Sources

- [LeRobot — Bring Your Own Hardware](https://huggingface.co/docs/lerobot/en/integrate_hardware)
- [huggingface/lerobot](https://github.com/huggingface/lerobot)
- [huggingface/lerobot-dataset-visualizer](https://github.com/huggingface/lerobot-dataset-visualizer)
- [DavidLMS/leLab](https://github.com/DavidLMS/leLab)
- [LeRobot v0.6.0 release notes](https://huggingface.co/blog/lerobot-release-v060)
- [BEAVR: Bimanual, multi-Embodiment, Accessible VR Teleoperation](https://arxiv.org/html/2508.09606v1)
- [Rerun](https://rerun.io)
