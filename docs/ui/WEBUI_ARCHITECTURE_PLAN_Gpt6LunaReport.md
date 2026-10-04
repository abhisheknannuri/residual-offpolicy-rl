# Review: Robot Web UI Architecture Plan

## Verdict

**Architecture: 8/10. Execution plan: 6.5/10. Overall: 7.5/10.**

The plan is unusually grounded in the actual workload: one researcher, one
workstation, real hardware, existing data tools, and a real-time-ish control
loop. Its separation of robot I/O, RL semantics, action selection, data
inspection, and live visualization is sound. It also resists adopting ROS,
containers, or a distributed message bus without a present need.

I would build a robot-operations UI in this direction, but **not merge
teleoperation and policy execution into one undifferentiated controller**. I
would share the device adapters, telemetry, recording plumbing, and UI
components, while retaining explicit operating modes and safety state machines.
The current plan describes good software boundaries; it does not yet define
enough safety behavior to be an implementation-ready robot-control design.

This review assesses the plan as written. I have not independently checked its
inventory measurements, external-project claims, or cited library behavior
against source code.

## What Works Well

- **Good problem framing.** It distinguishes the device-owning console from
  dataset and run analysis, and ties that split to exclusive camera/leader
  ownership rather than UI preference.
- **Useful boundaries.** `Robot`, `Env`, and `ActionSource` have distinct
  reasons to change. Keeping the Gym environment out of the operator UI is
  especially sensible.
- **Strong attention to data correctness.** Frame identity, staleness, camera
  skew, run IDs, and archival formats are operationally important, not polish.
- **Pragmatic stack.** FastAPI/OpenAPI plus a typed React client fits the
  existing browser tooling. Keeping the robot process on the host is sensible
  for RealSense access.
- **Incremental migration.** Preserving the existing frontends while extracting
  shared behavior reduces the chance of a risky rewrite becoming a hardware
  bring-up project.

## Main Risks To Resolve

### 1. Presets are not safety controls

The plan suggests presets can preserve the different safety postures of
teleoperation, inference, and evaluation. Presets are configuration; they are
not an enforcement boundary. The document needs explicit answers for:

- Which component owns command arbitration when policy, leader, and reset
  commands overlap?
- What happens on stale camera frames, lost leader/pedal, policy timeout,
  browser disconnect, API crash, and follower-server disconnect?
- What command is sent on each failure: stop, hold, controlled retreat, or no
  new command? Which behavior is enforced below the browser process?
- Which limits are validated server-side on every command, and which stop
  mechanism remains available if the UI or Python process hangs?

I would define a server-side operating state machine and command guard before
consolidating control paths. The physical emergency stop and any robot-side
watchdog must remain independent of the browser. Make the active command source,
robot state, and fault state continuously visible to the operator.

### 2. Share the loop carefully

Teleop and inference do share a useful cycle shape, but that does not prove they
have the same lifecycle, operator intent, or failure policy. A shared loop with
mode-specific validation can work; a generic loop whose differences live only
in presets risks hiding consequential behavior.

I would first extract small shared services and typed action-source adapters,
then keep explicit `TeleopSession`, `PolicySession`, and `EvaluationSession`
state transitions. Share the mechanics; make authority changes, intervention,
recording, reset, and shutdown explicit. Add tests for transitions and
conflicting commands before retiring either existing app.

### 3. The camera bus promises more than its sketch establishes

The plan correctly identifies different latest-value and ordered-stream
consumers. However, it does not fully specify how every captured frame reaches
the recorder if the producer-facing path is a latest slot, or how a slow
subscriber is isolated without losing the frame. A bounded queue can detect
overrun; it cannot guarantee lossless recording when disk or encoding remains
slower than capture.

Specify the actual fan-out point, queue capacity assumptions, shutdown/drain
behavior, and what happens when a recorder queue fills. In particular, a
recording failure should mark or stop the recording clearly, but should not
accidentally throw through the robot-control thread and interrupt command
handling. Queue sizing and the claimed control-loop benefit need a measured
prototype under sustained slow-disk and slow-encoder fault injection. Read-only
NumPy flags are a useful convention, not a complete ownership guarantee.

### 4. Run-record reliability needs a failure contract

The run record is correctly treated as important, but a bounded non-blocking
logging tap can itself drop archive records. Define which data is mandatory,
how saturation is reported, whether a run can continue while its archive is
invalid, and how partial runs are recovered after process termination. Give
each record a schema version and store checkpoint identity, resolved config,
robot/station identity, calibration identity, software revision, and timing
metadata. Treat finalization as a verifiable operation, not just file presence.

Run IDs can be introduced early without requiring the entire Parquet/Zarr
pipeline to block UI development. Start with a small, versioned run manifest
and expand it as consumers need more data.

### 5. Hardware identity and UI metadata are not quite one descriptor

The descriptor is a good direction, but the example combines robot-model facts
with station-specific facts. A robot type, physical station, camera serials,
calibration, and locally enforced joint limits have different lifetimes and
should not silently share one record. Separate a robot model/capability profile
from a station deployment configuration, and specify which values are trusted
for command validation versus display only.

Also, mirroring LeRobot's interface is a reasonable compatibility choice, but
it does not automatically provide LeRobot's plugin discovery or guarantee a
future drop-in replacement. Keep the local protocol deliberately small and
prove compatibility with contract tests or an adapter when the upgrade happens.

### 6. Local-only still needs a network posture

"No auth on a trusted LAN" is a meaningful scope choice, but a robot command
service exposed beyond loopback can be triggered by another local/LAN client or
browser-originated request. Bind to loopback by default. If LAN access is
needed, document and implement an explicit access-control and request-origin
policy rather than relying on network trust alone.

## Answers To The Plan's Questions

- **Q1, merge apps:** Share APIs and components; unify only after explicit mode
  state machines and safety tests exist. Avoid a single implicit control mode.
- **Q2, LeRobot:** Mirror a minimal protocol for now. Do not promise the future
  change is only a rename; preserve an adapter boundary and contract tests.
- **Q3, descriptor:** Yes, with robot profile separated from station
  configuration and safety-critical values enforced in the backend.
- **Q4, LeLab:** Inspect and prototype against it before choosing. Reuse
  concepts or components where they fit; do not fork until the eval workflow
  and hardware lifecycle have been compared directly.
- **Q5, Rerun:** Keep it an optional visualization sink. Keep a documented,
  versioned archive that remains usable without Rerun. Revisit read-back claims
  when changing the pinned SDK, not based on a moving latest-version claim.
- **Q6, phase order:** Establish run identity and a minimum manifest early, but
  do not block console UX work on a complete archive. Use mock runs until the
  record contract is stable.
- **Q7, missing features:** Add preflight checks, command-source/authority
  display, watchdog and stale-data behavior, fault acknowledgement, safe
  startup/shutdown, calibration status, and a visible recording-validity state.
- **Q8, camera bus:** Prototype bounded per-consumer delivery and instrument it
  before committing to a 150-200-line design. Queue overflow should make data
  quality unmistakable; whether it also stops the robot is a separate safety
  decision.
- **Q9, fault isolation:** Specify process failure behavior before hardware
  rollout. A separate control process may be warranted, but it adds supervision
  and IPC; first define which process owns the watchdog and how control stops
  safely if the UI/API dies.
- **Q10, protocols:** Four is not inherently too many. Add each only with a
  contract test and a concrete consumer; avoid building abstractions before
  the first second implementation needs them.

## What I Would Build

1. Write down command authority, operating states, failure responses, and the
   independent emergency-stop/watchdog path. Test the state transitions with a
   mock robot and injected disconnects.
2. Add a versioned run manifest and stable run ID. Keep current recording
   behavior until the new archive can prove completeness and recovery.
3. Extract narrow robot, action-source, and recorder interfaces behind the
   existing apps. Preserve separate teleop, policy, and evaluation sessions.
4. Prototype asynchronous recording with queue-depth/drop metrics and
   deliberately slowed consumers. Move work off the control path only after
   verifying no silent data loss and bounded control latency.
5. Build the React console against the stabilized API, first in mock mode, then
   against hardware with an operator-visible preflight and fault state.
6. Add the run browser and optional Rerun output after the run schema has a real
   producer and at least one consumer.

That keeps the plan's strongest ideas while making operational safety and data
validity explicit gates; clean architecture cannot substitute for them.
