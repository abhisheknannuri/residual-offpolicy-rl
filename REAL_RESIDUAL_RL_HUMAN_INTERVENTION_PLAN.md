# Real-Hardware Online Residual RL Training with Human Intervention — Implementation Record

Status: **core implementation complete and statically verified (`get_errors` clean). A manual real-hardware smoke-test script exists (`trossen_real/scripts/test_real_residual_env.py`) but has NOT yet been run on real hardware.** See §7 for exactly what's left.

## 0. Design (final, as implemented)

- Builds on Phase 1 (`trossen_real/human_intervention/INTERVENTION_PLAN.md`, already implemented/tested on real hardware): `TrossenSingleLeader`, `PedalListener`, `InterventionManager` (pedal check + leader mode switch + release-edge `policy.reset()` + `sync_to_follower()`) — all reused unchanged.
- **Single replay buffer, no BC/imitation pathway, no new loss code.** When a human intervenes, the transition's stored `action` is simply overridden with the human's actual executed action — it goes into the exact same online replay buffer as every autonomous transition. The critic learns from it via ordinary TD-learning; the actor improves only indirectly via its existing Q-maximization gradient ascent. No separate intervention buffer, no `update_actor_offline()`/BC-loss reuse (that path is dead for residual actors anyway — `_compute_actor_bc_loss()` has `assert not self.residual_actor`).
- Real single station: `num_envs=1`, `eval_num_envs=1` enforced everywhere. No vectorization.
- Buffer convention preserved exactly: the `action` field is ALWAYS the combined/executed action (never a bare residual); the only clip anywhere is `[-1, 1]` (via `ActionScaler`), never `±action_scale` — target residuals larger than `action_scale` are expected/tolerated (actor just saturates through its own `Tanh`), not a bug.

## 1. New file: `trossen_real/rl/real_residual_env.py`

`TrossenResidualEnv` — real-hardware, single-station drop-in replacement for `BasePolicyVecEnvWrapper`, presenting the same `reset()`/`step()`/`info["scaled_action"]`/obs-augmentation contract (batch dim 1, matching gymnasium vectorized-env auto-stacking) so `_add_transitions_to_buffer()`, `agent.act()`/`agent.update()`, checkpointing, and wandb logging all consume it with zero changes.

**Constructor** (as implemented):
```python
TrossenResidualEnv(
    follower, cameras, policy,            # FollowerClient, CameraManager, PolicyClient (HTTP to policy_server.py)
    control, reset_cfg,                   # ControlConfig / ResetConfig from the station config
    action_scaler, state_standardizer,
    image_keys, camera_resolution,
    action_dim: int = 7,
    action_space: str = "delta_joint",    # "absolute" | "delta_joint"
    leader=None, pedal=None,              # both None => intervention disabled entirely
    max_steps: int | None = None,
    settle_time_s: float = 2.0,           # sleep after the reset move (+leader sync), before obs/policy.reset()
)
```
`state_dim = action_dim` (both 7 — `command_space: joint` means state IS the raw 7D joint vector). `self.intervention = InterventionManager(leader, pedal, policy)` is built only when **both** `leader` and `pedal` are non-`None`; otherwise `self.intervention = None` and every branch checks `if self.intervention is not None` — this makes disabled-intervention behave byte-identical to a plain autonomous env.

**`reset()`**: `follower.reset()` → if intervention: `sync_to_follower(follower_state["q"], goal_time=reset_cfg.goal_time_s)` → `policy.reset()` → build obs via `_build_actor_obs()` + `_query_base_action()` + `_augment_obs()` → cache `self._last_base_naction` → `self._step_count = 0`.

**`step(residual_naction)`**:
1. `intervention.tick()` if present (pedal check + leader mode switch + release-edge `policy.reset()`).
2. `combined_naction = clamp(self._last_base_naction + residual_naction, -1, 1)`.
3. Read `follower_state_before`.
4. **If intervened now**: `executed_action = leader.get_joint_positions()` (real 7D absolute readback), `stored_naction = self._to_stored_naction(executed_action, follower_state_before)`.
   **Else**: `env_action = action_scaler.unscale(combined_naction)`, `executed_action = reconstruct_absolute_action(...)`, `stored_naction = combined_naction`; if intervention present (even though not currently intervened), `intervention.mirror_to_leader(executed_action, ...)` so the leader tracks the autonomous arm for a smooth future takeover.
5. `follower.move_to_joint_positions(executed_action, ...)`, sleep to hold `control.frequency_hz`.
6. Build `next_obs` by re-querying the base policy on the new follower state/images.
7. `reward`/`terminated`/`truncated` from `pedal.get_reward()`/`pedal.consume_reset()`/step-count-vs-`max_steps` (or `0.0`/`False`/`False` defaults if no pedal).
8. Returns `info = {"scaled_action": stored_naction, "intervened": intervened_now, "actor_proposed_action": combined_naction}` — `scaled_action` is what `_add_transitions_to_buffer()` actually stores; `actor_proposed_action` is diagnostic-only, never trained on.

**`_to_stored_naction(executed_action, follower_state_before)`**: for `delta_joint`, `raw[:6] = executed_action[:6] - follower_state_before["q"][:6]` (same convention as `convert_to_delta_joint_dataset.py`), then `action_scaler.scale(raw)` — clips to `[-1,1]` only, matching every other stored action.

**`close()`**: best-effort try/except park+disconnect leader (`park_waypoints=[reset_cfg.staged_joint_positions, [0.0]*7]`) + `pedal.stop()` if intervention present, then `follower.disconnect()`. Idempotent (safe to call twice).

No auto-reset — the training script must call `env.reset()` explicitly after a `done` step (see §3).

## 2. New config fields — `resfit/rl_finetuning/config/residual_td3.py`

Added to `ResidualTD3DexmgConfig` (all default off/inert — existing sim configs are byte-identical):
```python
real_hardware: bool = False           # gates every real-hardware code path in train_residual_td3.py
station_config_name: str = ""         # trossen_real/config.py::load_station_config()
policy_server_url: str = "http://127.0.0.1:5070"
real_action_space: str = "delta_joint"
real_max_steps: int | None = None     # per-episode step cap (no MuJoCo horizon to read)
enable_intervention: bool = False     # opt-in human takeover; leader connect failure is a hard error, no silent fallback
```
Also added `ResidualTD3TrossenRealConfig(ResidualTD3DexmgConfig)` — the concrete real-hardware task config (`task="trossen_real"`, `real_hardware=True`, `num_envs=eval_num_envs=1`, real camera keys, placeholder `offline_data`/`base_policy`/`wandb` fields meant to be overridden via CLI for an actual run).

## 3. `resfit/rl_finetuning/scripts/train_residual_td3.py` changes (all applied, `get_errors` clean)

1. **MuJoCo GL setup + local base-policy loading skipped for `real_hardware`.** The frozen base ACT policy is served remotely by `policy_server.py` (separate process, over HTTP via `PolicyClient`) and is never loaded in-process for real hardware (the real checkpoint's lerobot version may not even match this repo's vendored `resfit.lerobot` ACT implementation). `base_policy`/`eval_base_policy` become `None`; `cfg.actor_name` is hardcoded to `"residual_act"`.
2. **`cfg.offline_data.use_base_policy_for_base_actions` forced to `False`** (GT-as-base offline population) when `real_hardware=True`, with a warning — avoids needing two different code paths (HTTP vs in-process) to query the same base policy.
3. **`get_envs()` real-hardware branch**: loads the station config, connects `FollowerClient` + `PolicyClient` (asserts a policy is actually loaded), starts `CameraManager` (1s settle sleep), optionally (`enable_intervention`) connects `TrossenSingleLeader` — **required, hard failure if it can't connect, no silent degraded mode** — and constructs `PedalListener()`, then builds `TrossenResidualEnv`. The `eval_env` call always passes `enable_intervention=False` regardless of the train env's setting — no manual driving of eval rollouts.
4. **`assert cfg.eval_num_envs == 1`** added alongside the pre-existing `assert cfg.num_envs == 1`, when `real_hardware`.
5. **`horizon` unified**: one local `horizon = env.metadata["horizon"] if cfg.real_hardware else env.vec_env.metadata["horizon"]`, reused for both `wandb.summary["environment/horizon"]` call sites (previously 3 separate re-fetches, all sim-only).
6. **`env.render_viewer()`** guarded with `hasattr()` — no MuJoCo viewer for real hardware.
7. **Warm-up safety assertion**: `real_hardware=True` requires `cfg.algo.use_base_policy_for_warmup=True` — refuses the "pure random minus base" warm-up mode (can imply arbitrarily large corrections, unsafe on real hardware).
8. **Explicit reset-after-done**: both the warm-up loop and the main training loop replace the unconditional `obs = next_obs` with `if cfg.real_hardware and done.any(): obs, _ = env.reset() else: obs = next_obs` — sim vec-envs auto-reset internally, a single real env does not.
9. **Both periodic-eval blocks skipped entirely for real hardware** (offline TD3-BC phase's `run_dexmg_evaluation` call and the main loop's) — real hardware relies purely on the pre-existing, unconditional `save_freq` checkpointing; no interleaved eval rollouts, no "best success rate" tracking against a live eval env.
10. **Shutdown handlers**: right after `env`/`eval_env` are constructed, `atexit.register(...)` plus `signal.signal(SIGINT, ...)`/`signal.signal(SIGTERM, ...)` are installed to best-effort `env.close()`/`eval_env.close()` on normal exit, uncaught exceptions, or Ctrl-C — mirrors `infer_app/app.py::_cleanup()`'s pattern exactly (idempotent, one failure doesn't block the other).
11. `_extract_final_info_for_env()`/`_add_transitions_to_buffer()`'s `"final_info"` handling needed **no code change** — `TrossenResidualEnv`'s `info` dict never contains that key, so it already falls through to the correct `else` branch.

Everything else — `QAgent`/`Actor`/all loss functions, `_add_transitions_to_buffer()`, `TensorDictPrioritizedReplayBuffer`, `MultiStepTransform`, checkpointing, wandb logging, `_populate_offline_buffer()` — is completely unchanged and works against a real delta-joint dataset the same as any sim dataset.

## 4. Managing the real robot: startup / safety / shutdown

**Startup** (inside `get_envs()`'s real-hardware branch, in order): load station config → connect follower (hard fail if unreachable) → connect policy client + assert a checkpoint is loaded (hard fail otherwise) → start cameras, settle 1s → if `enable_intervention`: connect leader (hard fail if it can't connect) + construct pedal listener → construct `TrossenResidualEnv`.

**Safety guards carried over from Phase 1** (all still apply, unchanged): release-edge `policy.reset()` on takeover-release (via `InterventionManager.tick()`) prevents stale-action-queue jerks; `sync_to_follower()` never dips the leader into freedrive during reset (motors actively hold); leader connection is required-not-optional whenever intervention is enabled; the new warm-up safety assertion (§3.7) blocks the dangerous random-warm-up mode on real hardware.

**Shutdown**: `atexit` + `SIGINT`/`SIGTERM` handlers call `env.close()`/`eval_env.close()` best-effort (§3.10) — parks both arms, stops the pedal listener, disconnects the follower, regardless of whether training ended normally, crashed, or was Ctrl-C'd.

**Manual E-stop / mid-training safety**: the foot pedal's intervention button is available at all times during training, not just during some special "assist" mode — a human can take over at any point via the exact same mechanism used during autonomous rollout. Standard practice: physically stand at the robot for the first several hundred real-hardware steps of any new run.

## 5. Reward / eval strategy (v1)

- Reward source: `pedal.get_reward()` (binary, hil-serl-style convention) — no trained reward-model integration in v1.
- No periodic eval rollouts on real hardware (§3.9) — rely purely on `save_freq` checkpointing; evaluate saved checkpoints manually/offline afterward.

## 6. Scenarios considered while designing `TrossenResidualEnv`/the training-loop changes

- **Intervention starts mid-episode, then is released mid-episode** — `intervention.tick()`'s release-edge `policy.reset()` (Phase 1 code, reused) clears ACT's stale action-chunk queue so the base policy doesn't resume from a queue built against a now-stale observation.
- **`enable_intervention=False` for the whole run** — `self.intervention is None` short-circuits every intervention-related branch; behavior must be byte-identical to a plain autonomous residual rollout (no leader/pedal objects even constructed).
- **`enable_intervention=True` for training but eval must never be manually driven** — `eval_env` is always constructed with `leader=None, pedal=None` regardless of `cfg.enable_intervention`.
- **Leader fails to connect when intervention was requested** — hard failure, aborts the whole `get_envs()` call (and thus startup) rather than silently continuing without intervention capability.
- **Episode terminates (via pedal-triggered reset) during warm-up or the main loop** — real hardware has no auto-reset, so `done.any()` must trigger an explicit `env.reset()` call; unconditionally rolling `obs = next_obs` (the sim-only behavior) would silently keep training against a stale post-terminal observation.
- **Residual target larger than `action_scale` from a human correction** — expected and tolerated, not clipped to `±action_scale` anywhere (only `[-1,1]` via `ActionScaler`); the actor's own `Tanh` saturates, matching the pre-existing sim convention (`debug_offline/pct_target_exceeds_scale` metric already exists for exactly this).
- **Training crash / Ctrl-C mid-run** — `atexit`/`SIGINT`/`SIGTERM` handlers ensure the arms get parked and disconnected instead of being left powered/holding a pose indefinitely.
- **Real base-policy checkpoint incompatible with resfit's vendored ACT implementation** — sidestepped entirely by never loading the base policy in-process for real hardware; it's only ever queried over HTTP via the already-battle-tested (Phase 1) `PolicyClient`/`policy_server.py`.

### Reward/reset pedal timing bugs found and fixed (this round)

The reward and reset pedal semantics were re-derived carefully from the user's exact description ("I only press the reward pedal once the cube is properly inserted, and I might hold it while the policy keeps running for a bit before I also press reset — every transition during that whole hold should get reward=1, not just the last one"). Three real bugs were found and fixed:

1. **Press-then-release entirely within one control tick could be silently missed.** The old code called `pedal.get_reward()` (a plain level read) only once, at the END of `step()`, after the arm had already moved and the tick's sleep had elapsed. If the reward button was pressed and released faster than one control tick (~50-100ms), the level read at the end of the tick would show `False`, and that transition would incorrectly get `reward=0` even though the button WAS pressed while that action was being executed. **Fix**: `PedalListener` now has a `_reward_latch` set `True` by ANY press event, and a new `consume_reward_latched()` method that returns `latch OR current_level`, then resets the latch to the current level — this guarantees a press at any point since the previous step is captured, while continuous holding across many steps still correctly returns `True` every single step (not just once). `TrossenResidualEnv.step()` now calls `consume_reward_latched()` instead of `get_reward()`.
2. **Missing post-reset settle wait before capturing the first observation.** `infer_app/infer_loop.py` (Phase 1, already proven on real hardware) always sleeps `settle_time_s` (default 2.0s) after the staged-position reset move (+ leader sync) completes, and BEFORE calling `policy.reset()`/taking the first observation — this lets the arm fully stop vibrating/settling. `TrossenResidualEnv.reset()` was missing this entirely (jumped straight from the reset move to capturing obs). **Fix**: added a `settle_time_s` constructor param (wired through `cfg.real_settle_time_s`, default 2.0), and `reset()` now sleeps for it in the same position as `infer_loop.py` (reset move → leader sync → **settle sleep** → `policy.reset()` → first obs).
3. **A reward pedal held continuously across a reset boundary could leak a false-positive reward into the new episode's first step(s).** If the operator's foot is still on the reward pedal at the exact instant reset is triggered (e.g. they pressed reset with the other foot without releasing reward first), the level-based reward signal would still read `True` on the very first step(s) of the NEW episode — even though nothing has been achieved yet in that fresh episode. This can't be fully solved in code (physical foot position is ground truth), but it's mitigated two ways: (a) `reset()` discards any stale latch from the previous episode's tail so it can't carry over on its own, and (b) `reset()` now logs a clear warning if the reward pedal is STILL held after the settle window elapses, using the existing settle time as a natural grace period for the operator to release it before the new episode's first real step.
4. **Silent pedal disconnection.** `PedalListener`'s background thread already retries reconnecting for up to 2 minutes on a device error, but during/after that window all button states are frozen/stale with no signal surfaced to the caller. Added `PedalListener.is_healthy()` (False when in dummy mode or the device handle is `None`), checked (and logged as an `ERROR`) both at the start of `reset()` and every `step()` when a pedal is configured — makes a silent disconnect loud instead of invisible.

### Offline buffer image-resolution / OOM fix (this round)

The offline buffer population path (`_populate_offline_buffer()`, used when `cfg.algo.offline_fraction > 0`) has a `use_base_policy_for_base_actions` mode: instead of using the dataset's GT action as the "base action" reference (residual target = GT - GT = 0), it queries a base policy for each frame and uses ITS prediction as the base action (residual target = GT - base_policy_action), matching the online training convention more closely. The user intends to use this mode for real hardware too. Two real problems were found and fixed:

1. **OOM risk from storing full-resolution images in the buffer.** Camera frames are 256×256 (station config default); with tens of thousands of transitions × multiple cameras × 2 copies (obs + next.obs), a buffer storing everything at native resolution can reach double-digit GB, exceeding typical `/dev/shm`/RAM limits. `offline_data.image_size` (an existing config field, e.g. `84`) already resizes dataset images down before storing — this already worked correctly for sim, where `use_base_policy_for_base_actions` queries a LOCAL in-process `ACTPolicy`.
2. **For real hardware, resizing at dataset-LOAD time throws away the resolution needed for a good policy-server query.** The base policy is served remotely (`policy_server.py`, over HTTP) and is normally sent FULL-resolution frames (matching what live rollout sends via `_query_base_action()`). If the dataset is resized to 84×84 BEFORE `_populate_offline_buffer()` ever sees a frame, the only image available to query the policy with is the low-res one — a resolution the checkpoint was never actually deployed against.

**Fix**: when `cfg.real_hardware and cfg.offline_data.use_base_policy_for_base_actions`, the dataset is now loaded at **native resolution** (no resize transform at load time). `_populate_offline_buffer()` builds a JSON obs directly from the native-res sample tensors (`_dataset_sample_to_json_obs()` — auxiliary `joint_pos_raw`/`ee_pose_raw`/`velocity`/`effort`/`acceleration` keys are zero-filled, since ACT only ever reads `observation.state`, confirmed by reading `modeling_act.py` — these are schema-listed but completely inert/unused), queries `env.policy` (the SAME already-connected `PolicyClient` live rollout uses, passed in as `real_policy_client`) over HTTP to get the base action, THEN resizes the same sample's images down to `offline_data.image_size` (`post_resize`, applied per-sample) right before storing into the buffer. Memory-wise this is safe: the DataLoader still streams ONE native-res sample at a time (never the whole dataset at once) — only the STORED buffer entries need to be small, which they still are.

**Companion fix — the ONLINE buffer had the same latent mismatch.** `TrossenResidualEnv._build_actor_obs()` was building RL-facing image tensors at the raw camera capture resolution (e.g. 256×256, from `camera_resolution`), never resized — this would have (a) mismatched the offline buffer's 84×84 once training actually started, and (b) caused the network's vision encoder to be built for 256×256 input (since `img_c, img_h, img_w = env.observation_space[image_keys[0]].shape[1:]` in `train_residual_td3.py` derives the encoder's expected size from the env's `observation_space`, not a hardcoded constant). **Fix**: added an `rl_image_size` constructor param to `TrossenResidualEnv` — when set, `_build_actor_obs()` downsizes (bilinear + antialias, via `torch.nn.functional.interpolate`) the RL-facing copy of each frame before storing, and `observation_space`'s image `Box` shapes reflect that size instead of the raw camera resolution. `_query_base_action()` is completely unaffected — it always uses the original full-resolution frame from the same `images` dict, independent of this resize. `get_envs()` wires `rl_image_size=cfg.offline_data.image_size` — i.e. **one existing config field now drives BOTH buffers' image resolution**, so they can't drift out of sync.



## 7. Testing

### 7.1 Standalone real-hardware functional test (no RL actor/training loop)

`trossen_real/scripts/test_real_residual_env.py` — drives `TrossenResidualEnv` directly with a **zero residual action** every step (so the executed action is always exactly the base ACT policy's own prediction — the same autonomous behavior already validated by `live_infer_deploy.py`), using the REAL calibrated `ActionScaler`/`StateStandardizer` (fit from the actual training dataset's stats, not a placeholder — an uncalibrated scaler could clamp a real-unit action to the wrong range and command a bad joint target). This isolates testing to exactly the NEW code (reward/reset/settle/intervention plumbing) without the added risk/variance of an untrained residual actor.

Run:
```bash
python -m trossen_real.scripts.test_real_residual_env \
    --config trossen_station2_single \
    --policy-server-url http://127.0.0.1:5070 \
    --dataset-repo-id <repo_id> --dataset-root <root> \
    --action-space delta_joint \
    --max-steps 20 --num-episodes 2 \
    [--enable-intervention]
```
Prints `step, reward, terminated, truncated, intervened` every control tick. Suggested manual sequence (full detail in the script's own docstring):
1. No pedal presses — confirm plain-autonomous behavior (reward always 0, only truncation ends episodes).
2. Hold the reward pedal across several steps, release — confirm `reward=1.0` every step while held.
3. Press+release the reward pedal as fast as possible within one tick — confirm that step still shows `reward=1.0` (validates the latch fix).
4. Press the reset pedal — confirm the episode ends immediately (`terminated=True`), then a fresh reset (staged position + settle) happens automatically before the next episode's steps print.
5. Hold the reward pedal THROUGH a reset-pedal press — confirm the "reward pedal still held at episode start" warning prints.
6. (with `--enable-intervention`) Hold the intervention pedal, backdrive the leader — confirm `intervened=True` and smooth follower tracking; release — confirm a smooth, jerk-free resume.
7. Unplug the pedal mid-run — confirm the "lost its device connection" error repeats; do not continue training in this state.

### 7.2 Still NOT done (explicitly remaining)

1. **No mock-hardware (`MockArm`) smoke test** — unlike Phase 1, which had thorough `MockArm`-based test scripts (force `arm_driver.HAS_TROSSEN_SDK = False`, fake pedal/`PolicyClient` stubs) run before ever touching real hardware. §7.1's script requires real hardware; a mock-hardware equivalent (no robot needed at all) has not been written.
2. **No real-hardware run of `test_real_residual_env.py` performed yet** — write it, but it hasn't been executed against the actual station.
3. **No real-hardware run of the FULL RL training loop** (not even a short 200-step smoke run with a human physically present) — should only be attempted after §7.1 passes cleanly.
4. **No `.sh` launch script** for real-hardware training (analogous to `train_residual_rl_can_sparse.sh`, wiring `REAL_HARDWARE=true` and the new config overrides as CLI args) — not yet written.
5. Real-hardware eval/checkpoint-quality assessment relies entirely on manual/offline evaluation of saved checkpoints (no automated periodic eval loop for real hardware exists, by design — see §5).

---