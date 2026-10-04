# Gemini Architecture Review: WEBUI_ARCHITECTURE_PLAN

## Overall Rating: 9.5/10

This is an exceptionally well-thought-out, pragmatic, and grounded architecture plan. It correctly identifies the pain points of the current system (code duplication, string-sniffing for robot identity, I/O on the control thread) and proposes solutions that are right-sized for a single-researcher, single-robot setup. It avoids the trap of over-engineering (no ROS, no k8s, no cross-process pub/sub until proven necessary) while establishing clean abstractions that will scale cleanly to a few more robots.

## Would I build a web UI for robot operations this way?

**Yes, absolutely.** If I were designing a system for this exact constraint profile (one workstation, research-focused, fast iteration), I would land on almost exactly this stack:
- **FastAPI + React/TS/Tailwind** is the undisputed sweet spot for local robotics web UIs today. It provides type safety from backend to frontend without the weight of enterprise frameworks.
- **Separating Control (Surface A) from Analysis (Surface B)** is a critical insight. They have entirely different device requirements (exclusive access vs none) and performance profiles.
- **The Parquet/Zarr archive + Rerun viewer split** is state-of-the-art. Rerun is brilliant, but its storage format is transient and tied to SDK versions; treating it purely as a sink solves the data-longevity problem.
- **In-process threaded camera bus:** This is exactly how to solve the "fast control loop vs slow dataset writer" problem in Python without the overhead of IPC or complex middleware.

---

## Responses to §9 Questions

### Q1 — Is merging teleop and inference right?
**Yes.** The argument that they are fundamentally the same control loop (read state -> get action -> step -> log) is correct. Duplicating this loop is a massive source of drift and bugs. The concern about safety posture or different failure modes is valid but better solved via strict **Configuration Presets** (as proposed in §6.5) rather than separate codebases. A `teleop` preset and an `inference` preset can enforce different safety bounds or logging requirements while sharing the bulletproof core loop.

### Q2 — Mirror LeRobot's `Robot` interface, or bite the version upgrade?
**Mirror it (via a `Protocol`).** The constraint is that upgrading LeRobot breaks compatibility with existing v2.1 datasets and BC checkpoints. In a research environment, data integrity and reproducibility trump architectural purity. Using a Python `typing.Protocol` to define the `Robot` shape gives you structural subtyping—it behaves exactly like the LeRobot ABC without needing to import it. When you eventually upgrade LeRobot, you just delete your `Protocol` and import theirs.

### Q3 — Is the robot descriptor the right abstraction boundary?
**Yes.** Kinematics (URDF) and Hardware I/O (`Robot` class) do not and should not know about UI presentation, camera display names, or pedal availability. The descriptor pattern is standard and necessary. It is effectively a "Manifest" or "Device Profile", which is very common in hardware abstraction layers and device management.

### Q4 — Fork LeLab, or build Surface A fresh?
**Build fresh.** LeLab solves a slightly different problem and brings its own opinions. Given that you already have a mature Data Browser in Next.js and you need to support a highly custom eval campaign system (which LeLab lacks), shoehorning your eval logic into a forked LeLab codebase will likely take as much time as building Surface A from scratch, and you'll be fighting their abstractions the whole way.

### Q5 — Rerun as viewer, not archive?
**100% Correct.** Rerun is a visualization tool, not a data lake. Relying on `.rrd` for long-term storage of thesis data is dangerous. Parquet + Zarr is the correct choice for the source of truth.

### Q6 — Phase ordering (Run record first?)
Phase 0 (Run record + identity) is fundamental. However, if it blocks UI work on Surface A (the console), you could temporarily mock the run ID generation in Surface A just to get the frontend built, but you shouldn't use it for real runs until Phase 0 is done. For Surface B (Data Browser), Phase 0 is absolutely a hard blocker.

### Q7 — What features do specialist robotics UIs have that this plan misses?
The plan is impressively complete. Two things to consider:
1. **Calibration/Setup Wizard:** Teleop/Inference often requires a setup checklist (zeroing the arm, checking camera feeds). A dedicated UI flow for this before "Start Session" is useful.
2. **System Diagnostics:** Live latency tracking (control loop hz, camera frame age, inference time). This could live in Rerun, but a simple traffic-light indicator in Surface A's header is invaluable for a researcher to know if the system is healthy before a run.

### Q8 — Is the in-process bus the right stopping point?
**It is exactly right.** Going cross-process now violates your "single researcher, single workstation" constraint by introducing IPC complexity and serialization overhead for no immediate gain. 
*On the `lossless=True` sub-question:* **Abort the episode.** If you are collecting training data, silent drops corrupt the dataset. A failed episode costs 5 minutes; a corrupt dataset costs weeks of debugging bad policy behavior. Fail loud and early.

### Q9 — Fault isolation sooner?
**No.** Splitting the control loop into its own process before feature work is premature optimization. Python's `threading` with the GIL released during I/O/compute (cv2, torch, disk) is sufficient for a 50ms tick budget. Keep it simple until profiling proves you are missing deadlines because of the UI thread.

### Q10 — Four protocols or too many?
`Robot`, `Env`, `ActionSource`, `FrameBus` are four distinct concerns.
- `Robot`: Talks to hardware.
- `Env`: Defines the RL task (reward, reset state).
- `ActionSource`: Provides the action (Human, BC, RL).
- `FrameBus`: Moves images around.
None of these overlap. Collapsing them would create God Objects. Four protocols is perfectly manageable and enforces clean boundaries.
