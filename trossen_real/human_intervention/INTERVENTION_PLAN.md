# Plan: Human-Intervention Takeover (infer_app) + Residual-RL Gym Wrapper

TL;DR: Wire the already-built `TrossenSingleLeader` (freedrive/position mode switching) and `PedalListener` into the inference app's control loop so a human can take over mid-rollout (drives follower via leader's absolute joint readback), while the ACT policy keeps being called every tick (just ignored while intervened) via a new shared `InterventionManager` helper. Phase 2 designs a single-env (no vec-env) real-hardware gym wrapper for residual RL that reuses the same helper and overrides the stored buffer action with the human's actual executed action during intervention.

Backups taken before implementation: `trossen_real/infer_app/infer_loop.py.<timestamp>.bak`, `trossen_real/infer_app/app.py.<timestamp>.bak` (dated-copy convention, matching existing `resfit/dexmg/environments/dexmg.py.20260702_190034.bak` precedent in this repo).

**Steps**

**Phase 1 - Intervention in infer_app (concrete, implement first)**

1. New file `trossen_real/inference/intervention.py` - `InterventionManager(leader, pedal, policy)`:
   - `tick() -> bool`: reads `pedal.is_intervention()`; switches leader mode via `leader.set_freedrive_mode()`/`set_position_mode()` (already edge-safe - no-op if already in that mode); calls `policy.reset()` EXACTLY on the True->False (release) edge, tracked via an internal `_prev_intervened` bool. Must be called BEFORE that tick's `policy.predict_full()` so the reset takes effect before the next prediction.
   - `get_leader_action() -> np.ndarray`: `leader.get_joint_positions()` (7D absolute: 6 joints + gripper, no delta math).
   - `mirror_to_leader(action, goal_time) -> None`: `leader.track_joints(action, goal_time=goal_time, blocking=False)` - only call when NOT intervened.
2. *depends on 1* - Modify `trossen_real/infer_app/infer_loop.py`:
   - `InferenceLoop.__init__` gains optional `leader: TrossenSingleLeader | None = None`, `intervention: InterventionManager | None = None` params (both None => today's exact behavior, zero regression).
   - `_run()` tick restructured: `intervention.tick()` first, then always call `policy.predict_full()`/`reconstruct_absolute_action()`, then branch: if intervened send `leader.get_joint_positions()` to the follower, else send the policy action and mirror it to the leader.
   - `InferSnapshot` gains `intervention_enabled: bool`, `intervening: bool` fields.
   - `TickLogger.log(...)` call gains `intervened=intervened_now` field.
3. *parallel with 2* - Modify `trossen_real/infer_app/app.py`:
   - `AppState` gains `leader`, `pedal`, `intervention_enabled`.
   - `POST /api/connect` accepts `enable_intervention: bool = False`. If true, leader connection is REQUIRED - failure fails the whole request (and disconnects the follower already connected).
   - `POST /api/disconnect` and `_cleanup()` park+disconnect the leader the same staged->sleep way as the follower, plus `pedal.stop()`.
   - `GET /api/health` gains `intervention_enabled`, `intervening`, `leader_connected`.
4. *parallel with 2/3, lower priority* - Static UI: "Enable intervention" checkbox + status indicator.

**Phase 2 - Real-hardware residual-RL gym wrapper (single-env, no vec wrapper; implement after Phase 1 is hardware-verified)**

5. New file `trossen_real/rl/real_residual_env.py` - `TrossenResidualEnv` (plain class, gym-like `reset()`/`step()`/`close()`, NOT vectorized - single real station only):
   - Reuses `InterventionManager` internally.
   - `step(residual_naction)`: combines with cached base action (`combined_naction = base_naction + residual_naction`), sends to follower unless intervened (then sends leader's readback instead). Computes `stored_action` for the replay buffer: normally the combined action, but OVERRIDDEN with the human's actually-executed delta (`leader_abs_target[:6] - follower_state_before_tick[:6]`, normalized) when intervened - the caller's buffer-add code must use `info["stored_action"]`, not the `residual_naction` argument, when populating the buffer (documented deviation from the standard gym contract).
   - No auto-reset (real hardware) - caller calls `reset()` explicitly after a done step.
   - `close()`: same leader+follower park/disconnect as Phase 1.

**Relevant files**
- `trossen_real/inference/intervention.py` - NEW
- `trossen_real/infer_app/infer_loop.py` - modified
- `trossen_real/infer_app/app.py` - modified
- `trossen_real/infer_app/static/` - modified (UI)
- `trossen_real/leader/trossen_leader_single.py` - reused as-is
- `trossen_real/human_intervention/pedal_listener.py` - reused as-is
- `trossen_real/arm_driver.py` - reused as-is (`MockArm` fallback)
- `trossen_real/cameras/camera_manager.py` - reused as-is (`_MockCamera` fallback)
- `resfit/rl_finetuning/wrappers/residual_env_wrapper.py` - reference template for Phase 2
- New `trossen_real/rl/real_residual_env.py` (Phase 2)

**Verification**
1. `enable_intervention=false` (or omitted): byte-identical behavior to before these changes.
2. `enable_intervention=true` with no leader reachable: `/api/connect` fails cleanly, follower not left half-connected.
3. Mock-hardware test (no real robot/camera/pedal needed): connect with intervention enabled, start inference, simulate a pedal press mid-run, confirm follower switches to leader readback and back cleanly (via `TickLogger`'s `intervened` field).
4. Confirm `policy.reset()` is called exactly once per release edge.
5. Ctrl-C mid-run with intervention enabled: both leader and follower park to staged+sleep pose.
6. Phase 2: deferred, no automated verification yet (no training script exists).

**Decisions**
- `n_action_steps=1` (current delta_joint checkpoints) already avoids queue staleness by construction; the release-edge `policy.reset()` is a universal safety net for any future non-delta/chunked (`n_action_steps>1`) checkpoint.
- Leader connection is REQUIRED (not best-effort) when intervention is explicitly enabled.
- `InterventionManager` is shared between Phase 1 and Phase 2 to avoid duplicating pedal/leader/reset-edge logic.
- Phase 2 is single-env (no `gymnasium.vector`/`SyncVectorEnv` wrapping) since real residual RL only ever has one physical station.
- Mocking needs zero new code - `MockArm`, `_MockCamera`, and `PedalListener`'s dummy mode already exist and activate automatically when real hardware isn't present.
