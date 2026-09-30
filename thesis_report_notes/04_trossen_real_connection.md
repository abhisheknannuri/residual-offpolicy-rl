# ResFiT on Real Hardware: How the Trossen Pipeline Connects to the Sim Architecture

This is a standalone reference document. It does **not** re-describe the
simulation architecture (see the main sim-architecture spec) — it only covers
how the real-hardware (Trossen arm) residual-RL pipeline plugs into the same
`train_residual_td3.py` / `QAgent` / replay-buffer machinery, and exactly
where it differs.

Primary source consulted first: `../docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md`
(repo root, dated 2026-08-13). Every claim below was re-verified against the
current code as of this writing (2026-08-23); where current code has moved
past that document, this doc says so explicitly and cites the newer code.

---

## 1. Where the real-hardware code lives

- **`trossen_real/`** (repo root) is the real-hardware package. Confirmed
  present with, among others:
  - `trossen_real/rl/real_residual_env.py` — the RL-facing env wrapper (§3).
  - `trossen_real/inference/policy_client.py` — HTTP client for the base
    policy server (§2).
  - `trossen_real/inference/intervention.py` — `InterventionManager` (§4).
  - `trossen_real/human_intervention/pedal_listener.py` — `PedalListener`.
  - `trossen_real/leader/trossen_leader_single.py` — `TrossenSingleLeader`.
  - `trossen_real/teleop/follower_client.py` — `FollowerClient`.
  - `trossen_real/follower/follower_single_server.py` — the follower-arm
    HTTP server process.
  - `trossen_real/cameras/camera_manager.py` — `CameraManager`.
  - `trossen_real/config.py` — station config loader (`load_station_config`).
  - `trossen_real/configs/*.yaml` — per-station YAML configs (e.g.
    `trossen_station1_single.yaml`, `trossen_station2_single.yaml`).
  - `trossen_real/scripts/test_real_residual_env.py` — standalone functional
    test for `TrossenResidualEnv` (§7).
  - `trossen_real/REAL_TRAINING_ARCHITECTURE.md` — a second, more detailed
    (and newer, mtime 2026-08-15) internal architecture doc with a full
    step-by-step sequence diagram; used here as a cross-check, not as the
    primary source.

- **`lerobot_trossen/`** (repo root, untracked per `git status`) is a
  **separate nested git repository** (`lerobot_trossen/.git` exists as its
  own repo — `packages/lerobot_robot_trossen`,
  `packages/lerobot_teleoperator_trossen`). It is Trossen's own vendored
  LeRobot `Robot`/`Teleoperator` integration package, not something
  `trossen_real/` or `train_residual_td3.py` imports from directly — no
  `import lerobot_trossen` reference was found anywhere under
  `resfit/` or `trossen_real/`. Treat it as an adjacent/reference repo, not
  part of the active real-hardware training pipeline.

- **`server/hardware_interface.py`**, a standalone `PolicyClient`/
  `FollowerClient`/`CameraManager` class search, and `custom_scripts/` were
  **NOT FOUND** anywhere in the repo. The only hit for "hardware_interface"
  is a stray, garbled untracked file at the repo root literally named
  `erver.hardware_interface as hw` (and a sibling `ed:", e)`) — these are
  clearly leftover artifacts of an accidental shell redirect (e.g. `python -c
  "... import server.hardware_interface as hw" > 'erver.hardware_interface
  as hw'`), not real Python modules; they contain no importable code relevant
  to this pipeline. `FollowerClient` is defined once, at
  `trossen_real/teleop/follower_client.py`. `CameraManager` is defined once,
  at `trossen_real/cameras/camera_manager.py`. `PolicyClient` is defined
  once, at `trossen_real/inference/policy_client.py:75`.

---

## 2. Base BC policy serving: `policy_server.py` (HTTP), not an in-process checkpoint

Real hardware does **not** load the frozen ACT checkpoint in-process the way
sim does. Instead:

- `trossen_real/inference/policy_client.py:75-116` defines `PolicyClient`, a
  thin HTTP client with `.health()` (`GET /health`), `.reset()`
  (`POST /reset`), `.predict()`/`.predict_full()` (`POST /predict`).
- Every comment referencing the server side is explicit that it lives in
  **this same repo**, at `custom_scripts/policy_server.py`, run as a
  **separate process** (and, per `scripts/train_residual_rl_real.sh:29-32`,
  a **separate venv** from `trossen_real/`'s own): *"Start policy_server.py
  (SEPARATE venv — the training repo's own, NOT this repo's .venv — own
  terminal, stays running): `cd <training_repo> && python
  custom_scripts/policy_server.py --checkpoint <path> --port 5070`"*
  (`scripts/train_residual_rl_real.sh:29-32`; also
  `trossen_real/inference/policy_client.py:8-13`,
  `trossen_real/infer_app/app.py:13-15`).
- **However, `custom_scripts/policy_server.py` itself does not exist
  anywhere in this repository.** `find . -iname "policy_server*"` and a
  full-repo grep for `def _predict`/`@app.post("/predict")` both came up
  empty outside of the *client*-side files
  (`trossen_real/inference/policy_client.py`,
  `trossen_real/infer_app/app.py`, `trossen_real/scripts/live_infer_deploy.py`,
  `trossen_real/rl/real_residual_env.py`,
  `trossen_real/scripts/test_real_residual_env.py`,
  `resfit/rl_finetuning/scripts/train_residual_td3.py`,
  `resfit/rl_finetuning/config/residual_td3.py`) — none of which *define*
  the server. **NOT FOUND: the server implementation is referenced
  extensively (contract, port 5070, `/health`, `/reset`, `/predict`,
  `--checkpoint`, `--n-action-steps`) but not present in this checkout.**
  It may live in a separate, not-yet-merged location, or was deleted/never
  committed. This means the real-hardware base-policy-serving half of the
  pipeline cannot currently be run end-to-end from this repo alone.

- API contract inferred from the client side (`trossen_real/inference/policy_client.py`):
  - `GET /health` → `{"loaded": bool, "checkpoint": str, "likely_action_space": str|None, ...}`
    (`policy_client.py:196-217`, `train_residual_td3.py:445-448`).
  - `POST /reset` → flushes the ACT policy's internal action-chunk queue
    (`policy_client.py:82-84`).
  - `POST /predict` with a JSON obs built by `build_observation()`
    (`policy_client.py:141-166`: `observation.state` = raw 7D joint vector,
    `observation.joint_pos_raw`/`ee_pose_raw`/`velocity`/`effort`/
    `acceleration`, plus one base64-encoded image per camera key) →
    `{"action": [...], "action_normalized": [...]}` (real units + the raw
    normalized network output, diagnostic-only) (`policy_client.py:99-113`).
  - `reconstruct_absolute_action()` (`policy_client.py:169-227`) turns the
    server's raw prediction into an absolute joint target: pass-through for
    `action_space="absolute"`, or `predicted[:6] + current_q[:6]` for
    `"delta_joint"` (gripper dim always passed through, then clipped to
    `gripper_bounds`). The docstring notes this reconstruction is only exact
    for the first step of a chunk, so `delta_joint` deployments should run
    the server with `--n-action-steps 1`.
  - `check_action_space_mismatch()` (`policy_client.py:230-254`) is a
    best-effort safety heuristic comparing the caller's chosen
    `action_space` against the checkpoint's own `train_config.json`
    (via the `"delta"` substring in `dataset.repo_id`).

- Contrast with sim: sim loads the ACT checkpoint **in-process** as an
  `ACTPolicy` object and calls `base_policy.select_action(raw_obs)` directly
  inside `BasePolicyVecEnvWrapper.step()`
  (`resfit/rl_finetuning/wrappers/residual_env_wrapper.py:148-149`). Real
  hardware explicitly skips this: `train_residual_td3.py:279-311` gates the
  entire MuJoCo-GL-setup + local-checkpoint-loading block on `not
  cfg.real_hardware`, sets `base_policy = eval_base_policy = None`, and
  hardcodes `cfg.actor_name = "residual_act"` — the rationale given in-code
  is that "the real checkpoint's lerobot version may not even match this
  repo's vendored `resfit.lerobot` ACT implementation"
  (`REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md:66`, corroborated by the
  `train_residual_td3.py:297-311` comments).

---

## 3. `TrossenResidualEnv` vs. `BasePolicyVecEnvWrapper`: the shared contract

Sim's env wrapper: `resfit/rl_finetuning/wrappers/residual_env_wrapper.py`
(`BasePolicyVecEnvWrapper`). Real's env wrapper:
`trossen_real/rl/real_residual_env.py` (`TrossenResidualEnv`, 399 lines).
The real file's own docstring states the intent directly: *"Real-hardware,
single-station counterpart to `BasePolicyVecEnvWrapper` ...
Deliberately mirrors `BasePolicyVecEnvWrapper`'s exact contract (same
`info["scaled_action"]` semantics, same `observation.base_action` obs
augmentation, same `(1, dim)`-shaped batched tensors...)"*
(`trossen_real/rl/real_residual_env.py:5-15`).

Contract points verified on both sides:

| Contract element | Sim (`residual_env_wrapper.py`) | Real (`real_residual_env.py`) |
|---|---|---|
| `reset()` returns | `(augmented_obs, info)` | `(obs, {})` — same shape |
| `step(residual_naction)` returns | `(obs, reward, terminated, truncated, info)` | identical 5-tuple |
| Base-action query | `self.base_policy.select_action(raw_obs)` in-process (`:102`, `:149`) | `PolicyClient.predict()` over HTTP inside `_query_base_action()` (`:376-380`) |
| Obs augmentation | `_augment_obs()`: adds `observation.base_action`, standardizes `observation.state` (`:170-178`) | `_augment_obs()`: identical two operations (`:382-387`), docstring says *"Mirrors `BasePolicyVecEnvWrapper._augment_obs()` exactly"* |
| Replay-buffer action | `info["scaled_action"] = combined_naction` (`:145`) | `info["scaled_action"] = stored_naction` (`:285`) — same key, same semantics (combined action unless overridden, see §4) |
| Combining base + residual | `combined_naction = self._last_base_naction + residual_naction` — **no clamp** (`:136`) | `combined_naction = torch.clamp(self._last_base_naction + residual_naction, -1.0, 1.0)` — **clamped to [-1,1]** (`:223`) |
| Batch shape | `(num_envs, dim)` from a real `VectorizedEnvWrapper` | `(1, dim)` always — a single real station pretending to be a batch-of-1 vec-env (docstring `:10-15`) |
| `env.vec_env` attribute | Real `VectorizedEnvWrapper` instance | Self-referential alias (`self.vec_env = self`, `:153`) so `env.vec_env.metadata[...]` works unchanged |
| `metadata["horizon"]` | MuJoCo env horizon | `max_steps` (`:149`) |
| `close()` | delegates to `vec_env.close()` | best-effort park + disconnect leader/pedal/follower, idempotent (`:327-345`) |
| Auto-reset on `terminated`/`truncated` | Handled internally by the sim vec-env (gymnasium autoreset) | `TrossenResidualEnv.step()` **manually re-implements** the same convention: on a terminal step it stashes the true terminal obs in `info["final_obs"]`/`info["final_info"]` and then calls `self.reset()` itself before returning, so the *returned* `next_obs` is already the fresh post-reset observation (`:317-323`) |

**One genuine, verified divergence**: sim's `combined_naction` is **not**
clamped before being unscaled and stepped
(`residual_env_wrapper.py:136-139`), whereas real hardware **clamps
`combined_naction` to `[-1, 1]` before doing anything else with it**
(`real_residual_env.py:223`). This is a real behavioral difference between
the two paths, not just a documentation gap — worth flagging if comparing
residual-magnitude statistics between sim and real runs.

**`get_envs()` real-hardware branch** (`resfit/rl_finetuning/scripts/train_residual_td3.py:404-493`):
connects `FollowerClient` (`:441-442`), asserts the `PolicyClient` reports a
loaded checkpoint (`:444-448`), starts `CameraManager` with a 1s settle
sleep (`:450-452`), optionally connects `TrossenSingleLeader` + constructs
`PedalListener` when `enable_intervention` (hard failure, no fallback, on
leader-connect exception: `:456-465`), then builds `TrossenResidualEnv`
(`:470-493`) wired with `rl_image_size=cfg.offline_data.image_size` so the
online and offline buffers can't drift out of sync in image resolution.

**One correction relative to the plan doc's phrasing**: the plan doc
(`REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md:68`) says *"The `eval_env`
call always passes `enable_intervention=False`."* That is true only for the
**non**-real-hardware path (`train_residual_td3.py:646`,
`enable_intervention=False` hardcoded when building a second sim env). For
real hardware, `eval_env` is not a second `get_envs()` call at all — it's
literally the same object (`eval_env = env`, `train_residual_td3.py:630`,
with an accompanying comment explaining a second `FollowerClient.connect()`
would 409 since the follower server refuses concurrent connections). This
distinction is moot in practice because periodic eval rollouts are skipped
entirely for real hardware (§8 below), but the mechanism differs from what
the plan doc's wording implies.

---

## 4. Human intervention: what happens to the stored transition

Verified end-to-end against `trossen_real/rl/real_residual_env.py`,
`trossen_real/inference/intervention.py`, and
`resfit/rl_finetuning/off_policy/rl/q_agent.py`. The plan doc's claim holds:

**Every step, autonomous or intervened, the agent's residual actor and the
base policy are both evaluated** (`real_residual_env.py:221-223`
unconditionally computes `combined_naction`). What differs is only what
gets *executed* and *stored*:

- **Autonomous** (`intervened_now == False`, `real_residual_env.py:238-246`):
  `env_action = action_scaler.unscale(combined_naction)` →
  `reconstruct_absolute_action(...)` → `executed_action`; `stored_naction =
  combined_naction` (the RL agent's own proposed action, unchanged). If
  intervention is enabled but not currently active, the leader is mirrored
  to track the follower (`intervention.mirror_to_leader(...)`, `:246`) so a
  human resting a hand on it feels the autonomous motion.
- **Intervened** (`intervened_now == True`, `real_residual_env.py:226-237`):
  `executed_action = leader.get_joint_positions()` — the human's real,
  physically-executed absolute joint readback (gripper dim clipped to
  `(gripper_closed, gripper_open)`, a safety fix `real_residual_env.py:227-236`
  explicitly calls out as closing a gap versus an earlier unclipped version).
  `stored_naction = self._to_stored_naction(executed_action,
  follower_state_before)` (`:237`, `:389-399`) — converts the human's action
  into the **same per-frame delta-joint / `[-1,1]`-normalized convention**
  every other transition uses (`raw[:6] = executed[:6] -
  follower_state_before["q"][:6]`, then `action_scaler.scale(raw)` —
  clip is `[-1,1]` only, never `±action_scale`).
- **`info["scaled_action"] = stored_naction`** in both branches
  (`real_residual_env.py:285`) — the RL-agent's own `combined_naction` is
  additionally exposed as `info["actor_proposed_action"]`
  (`:287`) purely for diagnostics/logging, explicitly commented "NEVER
  trained on directly."
- `train_residual_td3.py` reads `combined_action = info["scaled_action"]`
  at both the warm-up (`:1318`) and main-loop (`:1830`) call sites into
  `_add_transitions_to_buffer(...)` — there is **no branch anywhere in
  `train_residual_td3.py` that distinguishes intervened vs. autonomous
  transitions**; both flow into the exact same online
  `TensorDictPrioritizedReplayBuffer`.
- **No BC/imitation loss path exists for the intervened data.**
  `_compute_actor_bc_loss()` in `resfit/rl_finetuning/off_policy/rl/q_agent.py:503-504`
  begins with `assert not self.residual_actor, "Not implemented"` — this
  code path is unreachable for any residual actor (real or sim), confirming
  the design intent that intervention data is consumed purely via ordinary
  TD/critic learning (`Q(s, a_human)`), with the actor only improving
  indirectly through its existing Q-maximization gradient-ascent step.
- Release-edge handling (`trossen_real/inference/intervention.py:63-90`):
  `InterventionManager.tick()` is called at the start of every `step()`
  (`real_residual_env.py:221`); on the intervention **release** edge only
  (`True -> False`) it calls `policy.reset()` to flush the base policy
  server's stale action-chunk queue before the next `predict()` call
  (`intervention.py:80-83`).
- A safety fix not mentioned in the Aug-13 plan doc but present in current
  code: during intervention, the leader's raw gripper readback is now
  explicitly clipped to the follower's configured gripper bounds before
  being sent to the follower (`real_residual_env.py:227-236`) — the inline
  comment states this "was a real gap" in an earlier version that sent the
  leader's raw joints through unclipped.

---

## 5. Buffer/action conventions specific to real hardware

- **`num_envs=1` / `eval_num_envs=1` strictly enforced.**
  `train_residual_td3.py:573` asserts `cfg.num_envs == 1` unconditionally
  (all configs, sim included, are single-env per the n-step implementation);
  `:575` adds a real-hardware-specific `assert cfg.eval_num_envs == 1`.
  `get_envs()`'s real branch additionally asserts `num_envs == 1` again at
  `:431`. `ResidualTD3TrossenRealConfig` sets both to `1` by default
  (`resfit/rl_finetuning/config/residual_td3.py:405-406`).
- **Action clipping convention**: confirmed identical to the plan doc's
  claim. The *only* clip applied anywhere to a stored action is `[-1, 1]`
  via `ActionScaler.scale()` — inside `_to_stored_naction()`
  (`real_residual_env.py:399`) for human interventions, and implicitly via
  the `torch.clamp(..., -1.0, 1.0)` on `combined_naction`
  (`real_residual_env.py:223`) for autonomous transitions. There is **no**
  separate `±action_scale` clip anywhere in the real-hardware path — a
  residual target larger than `action_scale` (e.g. from a large human
  correction) is expected/tolerated, saturating through the actor's own
  `Tanh`, not clipped as a bug-fix.
- **Warm-up safety assertion**: `train_residual_td3.py:576-579` requires
  `cfg.algo.use_base_policy_for_warmup == True` whenever `real_hardware`,
  specifically to forbid the "pure random minus base" warm-up mode that
  "can imply arbitrarily large corrections and is not safe on real
  hardware" (comment verbatim in code).
- **Explicit reset-after-`done`**: real hardware has no vec-env auto-reset,
  so unlike sim (where `obs = next_obs` unconditionally works because the
  vec-env resets internally), `TrossenResidualEnv.step()` performs the
  reset itself before returning when `terminated or truncated`
  (`real_residual_env.py:317-323`) — this makes `train_residual_td3.py`'s
  downstream consumption code path-identical to sim without a real-hardware
  branch at the call site.

---

## 6. Camera / resolution differences: sim vs. real

- **Sim**: `resfit/dexmg/environments/dexmg.py` renders MuJoCo/robosuite
  offscreen frames at `camera_size` (constructor default `84`,
  `dexmg.py:90`, propagated to `camera_heights`/`camera_widths` at
  `:196-197`). The single-arm residual configs (`ResidualTD3CanConfig`,
  `ResidualTD3CubeLiftConfig`, `ResidualTD3SquareConfig`) inherit the base
  `rl_camera` default of **2 cameras**: `observation.images.agentview` +
  `observation.images.robot0_eye_in_hand`
  (`resfit/rl_finetuning/config/rlpd.py:283-288`). Dual-arm configs
  (`ResidualTD3BoxCleanConfig`, `ResidualTD3CoffeeConfig`) use **3**
  cameras (`residual_td3.py:438-443`, `:473-477`). One resolution serves
  both the base-policy query and the RL buffer in sim — there is no
  separate "policy resolution" vs. "buffer resolution" split.
- **Real**: station configs (`trossen_real/configs/trossen_station1_single.yaml:19-38`,
  same in `trossen_station2_single.yaml`) capture at native camera
  resolution (RealSense, 640×480 native per the ROI comment) and resize
  down to `cameras.resolution: [256, 256]` at capture time
  (`trossen_station1_single.yaml:20`, `trossen_real/config.py:45`, `:307`).
  Up to **4** camera slots are configured per station
  (`cam_high`, `cam_right_wrist`, `cam_left_wrist`, `cam_front` —
  `trossen_station1_single.yaml:22-38`; two of the four have real serials
  wired, `cam_high`/`cam_front` have empty `serial: ""`, meaning they fall
  back to a synthetic mock feed unless filled in). `scripts/train_residual_rl_real.sh`'s
  `RL_CAMERA` default uses only **2** of those four:
  `observation.images.cam_left_wrist` + `observation.images.cam_right_wrist`
  (`train_residual_rl_real.sh`, §15 block).
  - **Two separate resolutions are used, deliberately, for two different
    consumers** — this is real hardware's key departure from sim's
    single-resolution scheme: `_query_base_action()` always sends the
    **full 256×256 native-capture** frame to `policy_server.py`
    (`real_residual_env.py:376-380`, unaffected by `rl_image_size`),
    while `_build_actor_obs()` downsizes only the RL-facing copy stored in
    the online replay buffer to `rl_image_size` (bilinear + antialias via
    `F.interpolate`, `real_residual_env.py:365-373`), wired from
    `cfg.offline_data.image_size` (default `84` — same numeric value as
    sim's default `camera_size`, `train_residual_td3.py:491`,
    `train_residual_rl_real.sh` §14: `IMAGE_SIZE=84`). This split exists
    specifically because the frozen base checkpoint was trained/deployed
    against full-resolution frames, while an 84×84 buffer keeps memory
    bounded and matches the offline dataset's own resize convention
    (`REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md:115-124`, corroborated by
    the code cited above).

---

## 7. Testing status: statically verified vs. actually run on hardware

- **As of the plan doc (2026-08-13)**: "core implementation complete and
  statically verified (`get_errors` clean) ... has NOT yet been run on real
  hardware" (`REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md:3`), with an
  explicit "Still NOT done" list (§7.2 of that doc): no mock-hardware smoke
  test, no real-hardware run of `test_real_residual_env.py`, no real-hardware
  RL training-loop run, no `.sh` launch script.
- **Current code has moved past several of those gaps**:
  - `scripts/train_residual_rl_real.sh` **now exists** (last modified
    2026-08-14, one day after the plan doc) — a full launch script wiring
    `real_hardware=true` and all real-hardware CLI overrides, contradicting
    the plan doc's "No `.sh` launch script" item. This is a code fact I
    verified directly (`ls`/`cat` on the file), not present at the time the
    plan doc was written.
  - `trossen_real/REAL_TRAINING_ARCHITECTURE.md` was added even later
    (mtime 2026-08-15) with a full sequence-diagram walkthrough of
    `TrossenResidualEnv.step()`. I treated it as a secondary source and
    cross-checked its per-step claims directly against
    `real_residual_env.py` rather than trusting it outright — one caveat:
    that file contains a stray, evidently-unanswered internal question
    (embedded verbatim at line 67 of that file, about action normalization
    remapping) that reads as leftover scratch/debugging text rather than a
    verified architectural claim; I did not rely on that passage for
    anything in this document.
  - **Evidence suggestive of an actual (or at minimum hardware-adjacent)
    run having taken place**: an untracked file,
    `RealTrainLogs/real_env_diagnostic_log.jsonl` (repo root, mtime
    2026-08-14 16:44, i.e. ~7 minutes after `real_residual_env.py`'s last
    edit at 16:37 that same day), contains **746 real per-step JSONL
    records** matching exactly the `TickLogger` schema written by
    `TrossenResidualEnv.step()` (`real_residual_env.py:290-304`) — fields
    `step`, `intervened`, `reward`, `terminated`, `truncated`,
    `executed_action`, `stored_naction`, `actor_proposed_action`,
    `residual_naction`, `follower_state_before/after`, `achieved_hz`. The
    logged `follower_state` joint vectors (`q`) are **non-zero and vary
    continuously across steps** (e.g. first record's `q` ≈
    `[-0.0029, 1.5616, 1.5593, -1.5681, 0.0006, -0.0013, 0.0401]`, drifting
    to very different values 746 steps later), which rules out the code's
    documented `MockArm` fallback — `trossen_real/arm_driver.py:86`'s
    docstring states the mock arm's "6 arm joints (always 0 in mock)."
    `achieved_hz` values (~12.5 in the sampled record vs. the configured
    20 Hz target) are also consistent with a real network/HTTP/robot-SDK
    round-trip under load, not an instantaneous mock. **I cannot fully
    confirm from the repo alone that this was a full RL-training run vs.
    some other invocation** (e.g. a manual/interactive session against
    `TrossenResidualEnv`), and no corresponding wandb run ID, run directory,
    or checkpoint output was located to corroborate it further — but the
    file's existence, schema match, non-mock telemetry, and timing (same
    day as the final `real_residual_env.py` edit) are meaningfully stronger
    evidence than the plan doc's "not yet run" status from the day before.
    Treat this as **"likely exercised against real (or at least
    non-mock) hardware at least once, exact scope unconfirmed,"** not as
    proof of a completed, successful training run.
  - No `MockArm`-based smoke-test script (the plan doc's item 7.2.1) was
    found under `trossen_real/scripts/` or `trossen_real/rl/` — **still
    NOT FOUND**, consistent with the plan doc.
- **Bottom line for this document**: treat the real-hardware residual-RL
  pipeline as *implemented and at least partially exercised near real (or
  non-mock) hardware as of 2026-08-14*, but **not** as a pipeline with a
  confirmed, completed, successful end-to-end training run on record in
  this repo. Do not overclaim a finished/validated training run.

---

## 8. `../docs/setup/UV_SERVER_SETUP.md`: relevance to the real-hardware stack

`../docs/setup/UV_SERVER_SETUP.md` is a hand-off guide for building this
repo's **`.venv`** via `uv` on a fresh server — it covers `torchrl` built
from source against torch 2.11+cu128 (§2), headless MuJoCo/robosuite
rendering via `MUJOCO_GL=egl`/`osmesa` (§3), and a troubleshooting table of
dependency-resolution conflicts (`dexmimicgen`, `gymnasium`, `mujoco`,
`datasets`, etc., §4). **A full-text search of that file for
"trossen"/"policy_server"/"real hardware"/"follower" returned zero
matches** — it contains no real-hardware-specific setup instructions. It is
purely about the sim/RL-training Python environment that this same repo's
`train_residual_td3.py` process runs under (Process 3 in the topology
below); it says nothing about the separate `policy_server.py` process or
its venv, which `scripts/train_residual_rl_real.sh`'s pre-flight checklist
explicitly calls out as needing its **own, different** venv
(`train_residual_rl_real.sh:29-32`).

---

## 9. Process topology summary (for orientation)

Three separate OS processes, confirmed via `scripts/train_residual_rl_real.sh`'s
pre-flight checklist and `trossen_real/REAL_TRAINING_ARCHITECTURE.md`'s
topology diagram (cross-checked against the client-side code that talks to
each):

1. **Follower server** — `trossen_real.follower.follower_single_server`,
   `http://127.0.0.1:5060` by default (`trossen_station1_single.yaml:17`).
   Owns the physical follower arm SDK connection.
2. **Base policy server** — `custom_scripts/policy_server.py`,
   `http://127.0.0.1:5070` by default (`train_residual_rl_real.sh` §3).
   Serves the frozen ACT/BC checkpoint. **Server implementation NOT FOUND
   in this repo** (§2 above) — only its HTTP contract, as consumed by
   `PolicyClient`, exists here.
3. **Training engine** — `resfit/rl_finetuning/scripts/train_residual_td3.py`
   with `real_hardware=true`. Instantiates `TrossenResidualEnv`, which talks
   to process 1 and 2 over HTTP, and connects directly (not over HTTP) to
   the leader-arm SDK, the foot-pedal `evdev` device, and the cameras.

---

## Summary: sim vs. real, at a glance

| Aspect | Sim | Real hardware |
|---|---|---|
| Env class | `BasePolicyVecEnvWrapper` (`resfit/rl_finetuning/wrappers/residual_env_wrapper.py`) | `TrossenResidualEnv` (`trossen_real/rl/real_residual_env.py`) |
| Base policy | In-process `ACTPolicy.select_action()` | HTTP `PolicyClient.predict()` → `policy_server.py` (not found in repo) |
| Vectorization | `num_envs` configurable (typically many) | Hard-asserted `num_envs=1`, `eval_num_envs=1` |
| `combined_naction` clamp | None | `torch.clamp(..., -1, 1)` |
| Auto-reset on done | Vec-env internal (gymnasium) | Manually reimplemented inside `step()` |
| Human intervention | N/A | `InterventionManager`/`TrossenSingleLeader`/`PedalListener`; overrides `stored_naction` only, same buffer, no BC path |
| Image pipeline | One resolution (`camera_size`, default 84) for both base-policy query and RL buffer | Two resolutions: full native-res (256×256) to the base policy, `rl_image_size` (default 84) to the RL buffer |
| Cameras (single-arm default) | 2 (`agentview`, `robot0_eye_in_hand`) | 2 of up to 4 configured (`cam_left_wrist`, `cam_right_wrist` by default) |
| Periodic eval during training | Runs normally | Fully skipped; `save_freq` checkpointing is the only safety net |
| Tested on hardware | N/A | Statically verified certain; a non-mock diagnostic log from 2026-08-14 suggests at least partial real/non-mock exercising, but no confirmed completed training run is on record |
