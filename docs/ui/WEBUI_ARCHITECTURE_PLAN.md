# Robot web UI architecture — inventory, prior art, and a proposal

Status: **proposal. Nothing implemented.**

This document is written to be reviewed by people (and models) who do **not**
have this repo. Everything factual is stated with the file, line count or
measurement it came from, so a reviewer can challenge the premises and not just
the conclusions.

### If you are reviewing this

**The questions I most want attacked are in §9** (Q1-Q10). The decisions I am
least confident about, in order:

1. **Merging teleop and inference into one app** (§3.1, §6.2) — the argument is
   that they are one control loop with different action sources, and that camera
   exclusivity means they can never run concurrently anyway. Q1.
2. **Mirroring LeRobot's `Robot` interface rather than importing it** (§5.3) —
   forced by a version pin that is load-bearing for existing checkpoints. Risks
   a permanent near-miss. Q2.
3. **Where the camera bus stops** (§6.6) — a real in-process bus now,
   cross-process deferred behind a stated decision rule. Over- or under-built?
   Q8.

Context that constrains any proposal: **one researcher, one robot, one
workstation.** Solutions needing a team to operate are the wrong answer here,
however good they are in general. Non-goals are listed explicitly in §8 —
please argue with them rather than assuming they were overlooked.

Facts are separated from proposals throughout: §2-§4 and §10 are measured or
cited; §5-§7 are proposals. If a premise in the first group is wrong, most of
the second group changes.

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

### 3.4 Encode and disk I/O run on the control thread

In `infer_app/infer_loop.py`, inside the per-tick loop:

```python
images = self.cameras.get_all_latest()
if video_recorder is not None:
    video_recorder.write_frames(images)      # PyAV encode, inline
...
if recorder is not None:
    recorder.add_frame(...)                  # LeRobot dataset write, inline
```

So video encoding and dataset writes happen **inside the 50 ms tick budget**. A
disk hiccup or a slow encode steals directly from control. This is a real,
already-present latency hazard, visible in `achieved_hz`, and it is the reason
the frame bus in §6.6 is a deliverable rather than a tidy-up.

It also means frames currently have **no identity**: `get_all_latest()` returns
bare `dict[str, np.ndarray]` with no capture time and no sequence number.
`_last_frame_time` exists but is used only by `health()`, never handed to a
consumer. Consequences:

| question | answerable today? |
| --- | --- |
| is this frame fresh, or has the camera stalled? | **no** — `_latest` keeps serving the last real frame indefinitely |
| did I drop frames since my last read? | no |
| are the two wrist cameras from the same moment, or 50 ms apart? | **no** |
| for a recorded episode, when was each frame actually captured? | no |

The second-to-last matters for the residual policy specifically: it consumes two
wrist cameras and implicitly assumes they are synchronised. If one hiccups, the
policy acts on a stale view and the failure gets attributed to the policy.

### 3.5 Eight logging sinks, four roots, no shared run id

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

### 4.3 Rerun — a viewer, and specifically NOT an archive

[Rerun](https://rerun.io) is purpose-built for robotics: time-aligned images,
scalars, transforms and 3D with a scrubbing viewer. `rerun-sdk 0.35.0` is
already installed here. Measured in this venv: **0.79 ms/tick** to log 2 images
+ 9 scalars; ~15 MB `.rrd` per 350-step episode.

**It cannot be the logging archive.** Tested: the installed version has no
read-back API at all —

```text
import rerun.dataframe   ->  ModuleNotFoundError
submodules present: archetypes, blueprint, catalog, components,
                    datatypes, experimental, utilities
```

So an `.rrd` is write-only from your own code's perspective: you can view it,
but you cannot pull 7,000 ticks of images back out for a batched Q recompute,
or `groupby` a sweep in pandas. (Latest is 0.38.1 and newer versions do add a
dataframe API — but that is three minor versions of a 0.x format away, and it
does not fix the deeper issue: `.rrd` is a visualisation format whose
readability is tied to one SDK's version, and this is thesis data that must open
in 2029.)

**Verdict: a sink, not the source of truth.** Rerun earns its place for two
things nothing else covers — **live** streaming while the robot moves, and a
scrubbing timeline you do not have to build. Everything retrospective should be
served from the archive instead.

Note it is also the first genuinely useful *separate-process subscriber* (§6.7):
`rr.connect_grpc()` publishes from the camera-owning process to a viewer process
that can crash and restart without the robot noticing.

### 4.4 Foxglove / MCAP

Mature, strong multi-panel layouts, MCAP is a good container. But it is
ROS-shaped, and this stack has no ROS. It would buy interop we do not need at
the cost of a dependency plus a writer plus a reader.

**Verdict:** skip. Revisit only if ROS enters the picture.

### 4.5 Intrinsic Core — right ideas, wrong weight class

Alphabet's Intrinsic open-sourced [Intrinsic Core](https://github.com/intrinsic-ai/sdk)
at ROSCon in September 2026 (Apache 2.0): a local runtime, SDK and real-time
control framework for industrial robotics.

Why it does not fit: the runtime is a **k3s containerized environment** managing
process lifecycle, event scheduling and application state sync. That is a
sensible answer for industrial fleets and pure overhead for one arm on one
workstation.

What is worth stealing is conceptual: their **"skills"** abstraction — package
each capability as a reusable module behind a declared interface. That is the
same instinct as `ActionSource` and the robot descriptor below, independently
arrived at by a much larger team, which is mild evidence the seam is real.

### 4.6 BEAVR

[BEAVR](https://arxiv.org/html/2508.09606v1) — open-source bimanual
multi-embodiment VR teleoperation, explicitly designed for heterogeneous
platforms from 7-DoF arms to humanoids.

**Verdict:** not adopting (no VR hardware), but its **multi-embodiment
abstraction** is worth reading as a design reference for §5.

### 4.7 What the survey did *not* turn up

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

## 5A. Three protocols, each with one reason to change

The single most important design decision in this document. Three abstractions
are circling each other and **must not merge**, because each changes for a
different reason:

| protocol | job | changes when |
| --- | --- | --- |
| **`Robot`** | device I/O — `connect`, `disconnect`, `calibrate`, health, `get_observation`, `send_action` | you buy different hardware |
| **`Env`** (gym) | RL semantics — `reset`, `step`, spaces, horizon, reward | the task or RL formulation changes |
| **`ActionSource`** | where an action comes from — leader, BC, ResFiT, replay | you add a policy or an input device |

### `Env` already exists; do not reuse it for the UI

`TrossenResidualEnv` is already gym-shaped — `gym.spaces.Dict`,
`gym.spaces.Box`, `reset()`, `step()`, `metadata` — which is exactly why one
trainer runs both MuJoCo and the real arm. That abstraction is **done**.

The trap is reusing it for the UI. A UI needs connect/disconnect, calibration,
per-camera frames and health, and has no notion of "step" or "episode return".
Force one class to serve both and you get an `Env` with `calibrate()` bolted on,
or a `Robot` pretending to have a reward. LeRobot keeps `Robot` separate from any
env for this exact reason. Keep them separate.

### `ActionSource` — the seam that collapses teleop/infer/eval

```python
class ActionSource(Protocol):
    def get_action(self, obs) -> Action: ...
    def reset(self) -> None: ...
```

`LeaderArm`, `SpaceMouse`, `BCPolicy`, `ResFiT`, `ReplayFromDataset` are all
implementations. A new method is one small class.

**It must return the FINAL action, not a residual.** If the framework does
`clamp(base + residual)` it has baked ResFiT's combination rule and
normalisation into the framework — and not all residual methods combine the same
way (different spaces, different blending, chunked residuals). Each source owns
its own composition and hands back something executable. The framework's job is
to *ask* and to *log*, never to do the method's maths. This also correctly puts
"query the base policy" inside `ResFiT`, which is the thing that knows it needs
one.

Teleoperators are a sub-case LeRobot already names: `Teleoperator` with
`get_action()` / `send_feedback()`. Leader arm, gamepad, SpaceMouse, keyboard,
phone — all the same shape.

### One more small protocol

```python
class InterventionTrigger(Protocol):
    def is_active(self) -> bool: ...
```

Pedal today; keyboard or a gamepad button later. Trivial, and it stops
"intervention" from meaning "pedal" throughout the codebase.

## 6. Proposed architecture

### 6.1 Three surfaces, not three apps

| surface | what it is | build or adopt |
| --- | --- | --- |
| **A. Robot Console** | drive the robot: teleop, inference, eval execution | **build** (merge of the two Flask apps) |
| **B. Data Browser** | datasets, run records, episodes, annotations, labels, eval campaigns, Q plots | **extend** the existing Next.js visualizer |
| **(C.) Live Inspector** | live signals while the robot moves | **adopt Rerun** — a tool, not a pillar (§4.3) |

Rerun is deliberately parenthesised. On review it is **not** a third surface: it
is one optional subscriber. Once the Data Browser reads run records, Rerun's
only irreplaceable job is *live* viewing during a run (§4.3). Skip it entirely
if you only ever look at runs afterwards.

The split follows one rule: **does it need the exclusive devices?**

- Cameras are RealSense, claimed per-process (`cfg.enable_device(serial)`), so
  only one process can hold them.
- The leader arm and pedal are likewise held in-process.
- The **follower arm is already behind an HTTP service**
  (`follower_single_server.py`) — both current apps are just clients, so the arm
  is not the constraint. The cameras are.

Everything that must touch cameras/leader/pedal lives in A. Everything that only
reads files lives in B. That is a physical boundary, not a taste one, which is
why it will hold as features are added.

Secondary rule for the same reason, applied to eval:

| | where | why |
| --- | --- | --- |
| eval **execution** — trial cursor, scoring while fresh | **A** | needs the loop and the devices |
| eval **analysis** — checkpoint x trial grid, success rates, Q plots, campaign history | **B** | reads results only |

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

### 6.6 The camera bus — one producer per camera, many consumers

**Build this.** Not a seam-naming exercise: §3.4 shows encode and disk writes
currently run on the control thread, inside the 50 ms tick budget.

#### Consumers are two different species

| consumer | wants | behind? | work per frame |
| --- | --- | --- | --- |
| control loop | **latest only** | dropping old is *correct* | fast |
| web UI | latest, ~10 fps | drop fine | jpeg encode |
| Rerun | latest, subsampled | drop fine | cheap |
| **dataset recorder** | **every frame, in order** | drop = **corrupt dataset** | disk write, slow |
| **video saver** | every frame, in order | drop = gap in the video | encode, slow |

One `queue.Queue` fan-out cannot serve both, and the failure mode is nasty: a
slow consumer either **blocks the producer** (camera thread stalls, control loop
starves) or **silently drops** (a dataset with holes you never learn about).

This split is not invented here — DDS and Zenoh formalise it as *History QoS*:
`KEEP_LAST(depth=1)` versus `KEEP_ALL`. Both modes are needed.

#### Shape

```text
cam0 thread ─┐
cam1 thread ─┼─▶ CameraBus
             │     │
             │     ├── latest slot per camera   (seq, overwritten, no queue)
             │     │        └─▶ Grouper ──▶ FrameSet{frames, skew_ms, age_ms}
             │     │                 ├─▶ control loop   (pull: latest())
             │     │                 ├─▶ web UI thread  (pull: latest(), 10 Hz)
             │     │                 └─▶ Rerun thread   (pull: latest(), subsampled)
             │     │
             │     └── per-subscriber bounded queue   (one queue EACH, not shared)
             │              ├─▶ recorder thread    lossless=True
             │              └─▶ video thread       lossless=True
```

#### Five rules that make it correct

1. **Per-subscriber queues, never shared.** A shared queue means one consumer
   *steals* items from another. Each stream subscriber gets its own
   `queue.Queue(maxsize=N)`.

2. **The producer never blocks.** A full queue must not stall a camera thread,
   because that starves the control loop. Publish and move on, always.

3. **Each subscriber declares its contract, and drops are loud.**

   ```python
   bus.subscribe(mode="latest")                     # drops are normal
   bus.subscribe(mode="stream", lossless=True)      # a drop is an ERROR
   ```

   For `lossless=True`, a full queue means the recorder cannot keep up. You
   cannot make the disk faster, but you **can** refuse to pretend the episode is
   clean: raise, mark the episode failed, stop recording. A corrupt dataset
   discovered six weeks into training costs far more than an aborted episode.

4. **Work happens on the subscriber's thread.** The bus hands off a reference;
   the recorder's own thread encodes and writes. This is what gets encode off
   the control path, which is the entire point.

5. **Publish references, not copies.** Frames are freshly allocated per read and
   never mutated in place — `read()` ends in `cv2.cvtColor`/`resize`, which
   allocate, and the producer does `self._latest[name] = frame`, a rebind. So
   every subscriber can share one reference with `writeable=False` set to make
   the invariant enforced rather than conventional. Anyone who genuinely needs
   to mutate copies for themselves. (Today `get_latest_frame` does
   `frame.copy()` per reader while holding the camera's lock — guarding a hazard
   that does not exist, and blocking the producer for the duration.)

#### Frames need identity

```python
@dataclass(frozen=True)
class Frame:
    data: np.ndarray      # writeable=False, not copied
    camera: str
    seq: int              # monotonic per camera; a repeat means a stall
    t_capture: float      # perf_counter at driver read

@dataclass(frozen=True)
class FrameSet:
    frames: dict[str, Frame]
    skew_ms: float        # max - min t_capture across cameras
    age_ms: float         # now - oldest t_capture
```

This is what fixes §3.4's unanswerable questions, and it lets consumers be
strict instead of hopeful:

- control loop: `if fs.age_ms > 100: abort` — refuse to drive the arm from a
  frozen view rather than silently acting on it
- recorder: log `seq` and `t_capture` per frame, so a dataset is auditable and
  `seq` gaps *prove* drops
- run record: `skew_ms` as a per-tick column, so "were the cameras synced during
  that run" is a query, not a hope
- health: a repeating `seq` is a precise stall signal, better than a
  time-since-last heuristic

#### Grouping: one stage, and it never waits

The **Grouper** is a latest-value subscriber to every camera and a producer of
`FrameSet`. It lives there, not in each consumer, because the control loop and
the recorder both need grouped, skew-checked sets.

It must **never wait** for a slow camera — waiting makes tick latency equal to
the worst camera's latency. Take latest-of-each, compute `skew_ms`, publish, and
let the skew number reveal whether synchronisation is actually a problem.

#### Threads are the right tool here

The usual objection is the GIL. It does not bite: the expensive operations
**release it** — PyAV/cv2 encode, disk I/O, and torch forward passes all do. So
thread-per-consumer gives real parallelism for this workload and
`multiprocessing` is unnecessary. The GIL would only matter if the heavy work
were pure-Python loops.

#### Observability, because an opaque bus is worse than none

Every subscriber exposes queue depth, drop count, last-processed `seq`, and
processing time p50/p95 — all into the run record. The questions that matter
become columns:

| question | answer |
| --- | --- |
| did the recorder keep up? | `drops == 0` |
| was a camera stalled? | repeated `seq` |
| were the cameras synced? | `skew_ms` |
| did encode steal control budget? | it cannot any more, by construction |

#### Size

~150-200 lines, **stdlib only** (`threading`, `queue`). Do not pull in a
framework: ZMQ, Zenoh and DDS all solve the *cross-process* version of this
problem, and this is one process.

#### Cross-process: still deferred

The same `FrameBus` protocol is what a cross-process implementation would
satisfy later, so building the in-process bus does not foreclose it. But it stays
deferred, and the decision rule is unchanged.

Build a cross-process frame service when **any one** of these trips:

1. A consumer must survive the control process restarting, or vice versa.
2. A consumer is in another language or on another machine.
3. Two things genuinely need cameras **concurrently** and cannot be one process.
4. The control loop misses its deadline because of a subscriber's work — which
   the in-process bus above already fixes.

None is true today. (1) arrives first, and Rerun already covers its most likely
instance by publishing from the camera-owning process to a viewer that can crash
and restart freely.

#### If it is ever built: transport options

**gRPC and zero-copy are mutually exclusive.** gRPC serialises protobuf to a
socket — a copy by definition. Zero-copy means the consumer reads the same
physical memory the producer wrote. Pick one.

For reference the load is small: 2 cameras x 256x256x3 at 30 Hz is ~12 MB/s.

| option | verdict |
| --- | --- |
| **ZMQ PUB/SUB** | `pyzmq 27.1.0` **already installed** (agentlace uses it); idiom already known. Sane default. Copies, but 12 MB/s is nothing. |
| **`multiprocessing.shared_memory` ring buffer** | Best technical fit for same-machine image fan-out: true zero-copy, stdlib, no new dependency. Needs a ring of N slots plus per-slot sequence numbers (the seqlock pattern) so a reader can detect that a slot was overwritten mid-read. At 20 Hz with 8 slots the writer revisits a slot every 400 ms while a 400 KB read takes ~40 us — a ~10,000x margin, so torn reads are a theoretical rather than practical concern. |
| **Zenoh 1.x** | The modern non-ROS robotics answer — shared-memory zero-copy, ~5 us latency, 67 Gbps peak. `eclipse-zenoh` 1.10.1 on PyPI. Verify the **Python** binding exposes SHM before committing; the zero-copy story lives in the Rust core. |
| **gRPC** | Fine for control-plane calls and for Rerun (which uses it); wrong tool for a zero-copy image bus. |
| **ROS 2 / DDS** | **No.** Beyond having no ROS, DDS zero-copy is implemented only for `rclcpp`, so a Python stack gets none of the benefit while taking the whole dependency. |

#### Transport for everything else: keep REST

Measured per tick today:

| call | cost | transport matters? |
| --- | --- | --- |
| `follower.get_state()` | 7 floats over HTTP, ~1 ms | no |
| `cameras.get_all_latest()` | in-process | no |
| RL forward (encoder + actor + 10 critics) | **1.38 ms** local GPU | no |
| policy server | **once per 14 ticks** (chunked) | no longer the hot path |

Images *were* the bottleneck at 55 ms/call, dominated by base64 + JSON parsing,
and that is already fixed with jpeg (15.2x smaller) and chunking (14x fewer
calls). Adopting gRPC now would optimise a solved problem.

The one transport change worth making is **WebSocket for state push to the
browser** instead of polling — a UI responsiveness win, not a control-loop one.

### 6.7 Deployment: containers for the file-readers, host process for the robot

| component | containerise? |
| --- | --- |
| Data Browser + its FastAPI backend | **yes** — reads files, no devices, already has a Dockerfile and compose |
| Policy server | **yes** — HTTP in, HTTP out |
| **Robot console** | **no** |

RealSense passthrough into a container means device nodes, udev rules, USB
permissions and kernel-version coupling. Days of work, zero gain on a
single-user workstation. Containers buy isolation, reproducibility and
multi-tenancy; the reproducibility here comes from the uv lockfile, and the other
two are not needed.

A CLI launcher is the cheap ergonomic win instead: `resfit ui robot`,
`resfit ui browse`, `resfit ui inspect <run>`. An afternoon, and it retires the
"which port, which directory, which env var" tax permanently.

### 6.8 Logging: one tap, two kinds of sink

Rerun cannot be the archive (§4.3). So the tap fans out:

```text
loop tick
   └─ tap.put(record)          non-blocking, background thread, bounded queue
        ├─ ticks.parquet       ARCHIVE  - source of truth, pandas-readable
        ├─ rl_obs.zarr         ARCHIVE  - exact model input, batch-readable
        └─ rr.log(...)         VIEWER   - optional, live, regenerable
```

The rule: **archive in a format nobody owns; view in whatever is nicest this
year.** The Rerun side is then genuinely disposable — if `.rrd` breaks,
regenerate it from the archive and lose nothing.

### 6.9 What ties the surfaces together

Only three things, deliberately:

1. **The robot descriptor** (§5.4) — one file per robot, served by A, consumed
   by A and B.
2. **The run record** — written by A, read by B (and by Rerun, regenerably).
3. **A generated API client + shared design tokens** — so A and B look and feel
   like one system without being one codebase.

No shared database, no message bus, no monorepo requirement. Four protocols
inside A — `Robot`, `Env`, `ActionSource`, `FrameBus` — each with one reason to
change (§5A, §6.6).

---

## 7. Migration path

Ordered so that every step is independently useful and nothing is a flag day.

| phase | work | why here |
| --- | --- | --- |
| **0** | **Run record + run_id** (deploy/eval plan §2–3) | every UI feature below needs runs to have identity |
| **1** | Robot descriptor schema; write one for the Trossen station; serve it from the existing Flask app | pure addition, no UI change, immediately unblocks B |
| **1b** | Protocols: `ActionSource`, `FrameBus`, `InterventionTrigger`; `Frame`/`FrameSet` with `seq` + `t_capture` | an afternoon; every later split becomes a transport swap, and frames finally have identity (§3.4) |
| **1c** | **Build the in-process camera bus** (§6.6): latest-value + lossless-stream modes, per-subscriber queues, grouper, subscriber threads, drop/depth/skew metrics | gets PyAV encode and dataset writes **off the control thread** — fixes a present latency hazard, not a hypothetical one. ~150-200 lines, stdlib only |
| **2** | Extract the shared control loop **inside the current Flask apps** behind `ActionSource`; keep both frontends | the risky refactor, done where it can be tested against hardware without a UI rewrite |
| **3** | Extract the duplicated endpoints into one FastAPI service behind the existing frontends | API consolidation, still no UI rewrite |
| **4** | New React console for Surface A against that API; retire the two `static/` dirs | now a rewrite of ~1,480 lines of vanilla JS, not of the robot logic |
| **5** | Visualizer: Runs section + eval campaign grid + Q plots | the payoff for phase 0 |
| **6** | Replace visualizer robot sniffing with the descriptor; **offer upstream** | removes the worst hardcoding; may remove our fork divergence |
| **7** | Rerun integration: live `connect_grpc()` during runs, `.rrd` generated from run records | last because it is additive and independent — and doubles as the first proof the publish pattern works across processes |
| **8** | CLI launcher (`resfit ui robot` / `browse` / `inspect`) | cheap, do it whenever it annoys you enough |
| **—** | Frame service with a real transport | **not scheduled.** Build when §6.6's decision rule trips, not before |

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
- **No *cross-process* camera service, no serialisation format, no middleware
  dependency.** §6.6 — one machine, 12 MB/s. The in-process bus *is* being built
  (phase 1c) because it fixes §3.4; what stays deferred is the transport. The
  failure mode to avoid is not under-engineering, it is debugging your own
  middleware instead of the insertion task.
- **No gRPC for the image path.** It is a copy by definition, so it cannot be
  the zero-copy answer, and the image hot path is already solved by jpeg +
  chunking.
- **Not containerising the robot console.** §6.7.
- **Rerun is not the archive.** §4.3 — it has no read-back API in the installed
  version.

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
keeps the eval campaign system (§4.7, the one genuinely novel piece) at the
centre. Has anyone read LeLab's code closely enough to judge?

**Q5 — RESOLVED, but challenge it.** Rerun is demoted from a surface to an
optional subscriber, because the installed SDK has no read-back API so it cannot
be the archive (§4.3). Counter: newer versions add a dataframe API — is betting
on that better than maintaining a parquet schema? I say no, on format-lifetime
grounds for thesis data. Disagree if you think the maintenance cost of our own
schema is higher than the version risk.

**Q6 — Phase ordering.** Phase 0 (run record) gates everything and touches the
training loop. Is it right to block UI work on it, or should Surfaces A/B
proceed against the existing eight sinks and migrate later?

**Q7 — What features do specialist robotics UIs have that this plan misses?**
The §4 survey found no off-the-shelf answer for evaluation campaigns, which
makes me suspect I am missing a term of art or a tool. Calibration wizards,
safety interlocks, dataset quality gates, teleop latency monitors, policy A/B
comparison — what else should be on the roadmap?

**Q8 — Is the in-process bus the right stopping point?** §6.6 builds a real
bus (two subscription modes, per-subscriber queues, lossless contracts) but keeps
it in one process. Two ways this could be wrong: (a) the deferral is too
conservative and we should go cross-process now while the design is fresh;
(b) the bus is already over-built for one process and a pair of
`threading.Thread`s plus two queues would do. Which?

Sub-question with teeth: for `lossless=True`, is **aborting the episode** on a
dropped frame the right call, or too brittle for a human-in-the-loop session
where restarting a trial is expensive? The alternative is recording the gap and
flagging the episode, which keeps the run but admits a dataset you must
remember to filter.

**Q9 — Fault isolation sooner?** Today a dying Flask takes the control loop with
it, because Flask hosts the loop thread. Should splitting *that* (loop in its
own process, UI as a client) come before any of the feature work — i.e. is it a
phase 1 concern rather than an eventual one?

**Q10 — Four protocols or too many?** `Robot`, `Env`, `ActionSource`,
`FrameBus`. Each has a defensible single reason to change (§5A, §6.6), but four
abstractions in a single-researcher codebase is also how projects become
unreadable. Which, if any, should collapse?

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
| `rerun-sdk 0.35.0` has **no** read-back API (`import rerun.dataframe` fails) | tested in this venv |
| Rerun latest is 0.38.1 — installed copy is 3 minor versions behind | PyPI |
| Follower arm is already behind an HTTP service; both apps are clients | `follower_single_server.py`, `FollowerClient` |
| Cameras are RealSense, claimed per-process (`cfg.enable_device(serial)`) | `camera_manager.py:84` |
| Four camera subscribers already exist, all in one process | `infer_loop.py` (policy, video_recorder, recorder, /video_feed) |
| PyAV encode + LeRobot dataset writes run **inline on the control thread** | `infer_loop.py:274-275`, `:304+` |
| Frames carry no `seq` or capture time; `_last_frame_time` is used only by `health()` | `camera_manager.py:175,273,313` |
| `get_latest_frame()` copies per reader **while holding the camera lock** | `camera_manager.py:299-305` |
| `read()` allocates a fresh array per frame (`cv2.cvtColor`/`resize`), so published frames are never mutated in place | `camera_manager.py:88-100` |
| The two modes needed (latest-value vs stream) are DDS/Zenoh History QoS: `KEEP_LAST(1)` vs `KEEP_ALL` | prior art |
| `CameraManager` is one class in one file, imported by both apps | no duplicated camera code |
| Camera load is ~12 MB/s (2 x 256x256x3 @ 30 Hz) | arithmetic |
| `pyzmq 27.1.0` already installed via agentlace | `importlib.metadata` |
| `eclipse-zenoh` 1.10.1 available on PyPI, not installed | PyPI |
| DDS zero-copy is `rclcpp`-only, so unavailable to a Python stack | [Agnocast paper](https://arxiv.org/pdf/2506.16882) |
| `TrossenResidualEnv` is already gym-shaped (`spaces.Dict`, `spaces.Box`, `reset`, `step`) | `real_residual_env.py:183-252` |
| Per-tick: follower ~1 ms, RL forward 1.38 ms, policy server once per 14 ticks | measured |

## Sources

- [LeRobot — Bring Your Own Hardware](https://huggingface.co/docs/lerobot/en/integrate_hardware)
- [huggingface/lerobot](https://github.com/huggingface/lerobot)
- [huggingface/lerobot-dataset-visualizer](https://github.com/huggingface/lerobot-dataset-visualizer)
- [DavidLMS/leLab](https://github.com/DavidLMS/leLab)
- [LeRobot v0.6.0 release notes](https://huggingface.co/blog/lerobot-release-v060)
- [BEAVR: Bimanual, multi-Embodiment, Accessible VR Teleoperation](https://arxiv.org/html/2508.09606v1)
- [Rerun](https://rerun.io)
- [Intrinsic Core / intrinsic-ai/sdk](https://github.com/intrinsic-ai/sdk)
- [Intrinsic open-sources core robotics capabilities, ROSCon 2026](https://www.unite.ai/intrinsic-open-sources-core-robotics-capabilities-at-roscon-2026/)
- [Zenoh / ZettaScale — shared memory and zero-copy](https://www.therobotreport.com/zettascale-designs-zenoh-to-transcend-dds-for-automotive-ros-communications/)
- [Zenoh performance thread, Open Robotics Discourse](https://discourse.openrobotics.org/t/zenoh-performance/30494)
- [ROS 2 Agnocast — DDS zero-copy is rclcpp-only](https://arxiv.org/pdf/2506.16882)