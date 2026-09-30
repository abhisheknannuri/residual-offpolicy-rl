# Reward-shaping / pedal-boundary investigation & changes — report

Date: 2026-08-28
Files touched: `trossen_real/rl/real_residual_env.py`, `resfit/rl_finetuning/scripts/train_residual_td3.py`, `scripts/TrossenStation1Real/train_residual_rl_real.sh`
Backups made before editing: `real_residual_env.py.20260828_190319.bak`, `train_residual_td3.py.20260828_193423.bak`, `train_residual_rl_real.sh.20260828_190319.bak`

## 0. Why this happened

Started from a question about the `REWARD_SHAPING` comment block in `train_residual_rl_real.sh` §16. Investigation turned up that the comment described intended behavior, not verified behavior, and a real bug (stale reset-pedal press) was found by extending the same reasoning that had already been fixed once in the teleop/infer apps this session. This then surfaced a genuine Bellman-target-inflation risk when the reward pedal is held continuously through non-rewarding motion (e.g. backing away after a successful insertion) — see §8. Everything below is either (a) something verified directly against code/data, not assumed, or (b) an explicit decision the user made when given the tradeoff. §8 in particular documents two dead ends this investigation went through (a rejected "clip" band-aid, and a naive widening that would have crashed training) before arriving at the correct fix — kept in for accountability, not just the final answer.

**Dataset actually used by this script**: `OFFLINE_DATASET_ROOT=trossen_real/datasets/PickAndInsertCube_Station1_merged_deltajoint_old` — **136 total episodes** (`meta/info.json`), of which `OFFLINE_EPISODES=120` are loaded. All empirical claims below were verified against this exact dataset by reading every episode's parquet file directly with pandas, not sampled/assumed. (A different, older 161-episode dataset — `..._delta_rewardLabled_v21_old` — was checked too, for comparison only; it is not what this script loads, and an earlier draft of the `.sh` comment had incorrectly cited its "161 episodes" figure for this one.)

## 1. What was verified about `REWARD_SHAPING` (`_populate_offline_buffer()`, `train_residual_td3.py:1014-1027`)

```python
if offline_reward_map is not None:          # sidecar file - not used by this .sh
    step_reward = ...
elif reward_shaping and "next.reward" in sample:
    step_reward = float(sample["next.reward"].item())   # true branch
else:
    step_reward = float(sample["next.done"].item())      # false branch
```

Checked every one of the 136 episodes' parquet files directly:

| Check | Result |
|---|---|
| `next.done` mask == `next.reward > 0.5` mask, per episode | **136/136 identical** |
| `next.done=True` region shape per episode | One contiguous run from a "success onset" index to the literal last frame — 136/136, no gaps, no exceptions |
| Frames marked `True` per episode | min 24, max 70, mean 41.2 (out of mean 291-frame episodes) |
| `next.reward` value range | strictly `{0.0, 1.0}` (checked) |
| Last frame of every episode marked `next.done=True` | 136/136 (100%) — correctly separates episodes for `MultiStepTransform`'s FIFO, which is never explicitly reset between episodes |

**Consequence**: `REWARD_SHAPING=true` and `REWARD_SHAPING=false` currently produce **numerically identical** per-frame offline rewards on this dataset, because the two source columns hold the same values everywhere. Setting the flag to `false` (done, per your instruction) does not by itself change training behavior on this data — it only would if `next.done` is ever re-labeled independently of `next.reward` (e.g. a genuine single-terminal-frame marker). Documented this explicitly in the `.sh` comment so it isn't silently assumed to have done something it didn't.

## 2. What was verified about online rollout (`TrossenResidualEnv`, `get_envs()`)

- `get_envs()`'s `if cfg.real_hardware:` branch (`train_residual_td3.py:424`) returns a `TrossenResidualEnv` directly at line 474 and never reaches `create_vectorized_env(reward_shaping=cfg.reward_shaping, ...)` at line 530 (that call only exists on the sim path).
- `TrossenResidualEnv.__init__`'s full parameter list has no `reward_shaping` argument at all.
- Online reward was (and still is) computed exactly one way: from the reward pedal, unconditionally.

**Consequence**: `REWARD_SHAPING` has **zero code path to online/live-robot rollout**, in either direction. It only ever affects `_populate_offline_buffer()`, and per §1, doesn't even do that on this dataset today.

- Checked `algo.terminated_only_bootstrap` (`rlpd.py:178`): defaults to `false`, never overridden by this `.sh`. With it `false`, `MultiStepTransform` uses `done` (`terminated | truncated`, combined) uniformly for the n-step bootstrap-cutoff mask — i.e. a real success/reset and a plain `REAL_MAX_STEPS` timeout are already treated identically today. This flag is only required `true` (hard-enforced, `train_residual_td3.py:588-592`) when the separate PBRS `reward_model` system is enabled, which this script does not use. **No change made here — already correct for a sparse-reward run.**

## 3. Bug found and fixed: stale reset-pedal press surviving into the next episode

`TrossenResidualEnv.reset()` discarded a stale **reward** latch (`self.pedal.consume_reward_latched()`) but never drained a stale **reset-pedal press**. `PedalListener.consume_reset()` is edge-triggered but sticky — a press sets an internal flag that stays `True` until something calls `consume_reset()`, however long that takes.

Failure mode: press the reset pedal a second time while the previous reset's `settle_time_s` (2.0s in this `.sh`) is still running. That second press is never consumed by `reset()`, sits latched, and fires on the very first `step()` call of the brand-new episode — truncating it to exactly 1 step. This is the same class of bug already found and fixed earlier this session in `trossen_real/human_intervention/episode_boundary.py::EpisodeBoundaryMonitor.start_episode()` for the teleop/infer apps.

**Fix**: `TrossenResidualEnv` now constructs an `EpisodeBoundaryMonitor` (the exact same shared class teleop/infer use — not a reimplementation) whenever `pedal is not None`, and `reset()` calls `episode_monitor.start_episode()`, which drains both the stale reset-press and the stale reward latch (logging a warning if either was actually pending).

## 4. New behavior added (explicit user request): reward-pedal release also ends the episode

Previously, only a reset-pedal press set `terminated=True` online; releasing the reward pedal had no effect on episode boundaries (it only ever affected the `reward` value). This mirrored a real, pre-existing gap between teleop/infer (where reward-pedal-release already ends a recording, added earlier this session) and the RL training env (where it didn't).

**Change**: `step()` now computes `terminated` from `episode_monitor.check_episode_end()`, which returns `True` on either:
- a reset-pedal press (unchanged behavior), or
- the reward pedal's held→released edge, **if** it was held at some point during the current episode (new).

`reward` now comes from `episode_monitor.get_frame_reward()` (same `consume_reward_latched()` under the hood — value unchanged).

### Why this doesn't corrupt the Bellman backup — verified, not asserted

Raised and worked through with the user: does ending an episode via reward-release risk landing `reward=0, done=True` on the same (terminal) transition if the operator releases reward and only later presses reset?

- If the episode ends via the **new reward-release trigger**: `PedalListener.consume_reward_latched()` has a one-tick decay (`self._reward_latch = self.reward_pressed` is set at the end of every call, so the tick where release is *detected* still reads back `True` one more time before the *next* tick reads `False`). Since `check_episode_end()`'s release-edge and `get_frame_reward()`'s latch read happen on the *same* tick, this means `reward=1.0` and `terminated=True` land on the same transition **automatically, with no operator timing required**.
- If the episode instead ends via a **separately-timed reset-pedal press** after the reward pedal was already released and the latch has fully decayed (2+ ticks later): that transition legitimately gets `reward=0, done=True`. This is not a bug or data corruption — it's a correct reflection of what actually happened at that instant. The success signal isn't lost; it was already correctly recorded as `reward=1` on the earlier held-reward transitions, and standard TD-learning with n-step bootstrapping propagates that value backward through training the normal way. Recommended operator habit (the user's own proposal, confirmed sound): press reset while still holding reward (or immediately after releasing it) if you want the literal terminal transition to also carry `reward=1`.
- **Initially considered too narrow, then corrected (see §8)**: an earlier draft of this investigation stopped at "just use the narrow trigger, the reward/done desync on a manually-timed reset press isn't a correctness problem." That's true as far as it goes, but it doesn't cover the user's ACTUAL operating pattern (see §8): holding the reward pedal continuously through non-rewarding motion (e.g. backing away after insertion), which is a real, separate problem from the one this section covers. §8 has the full correction.

## 5. Files changed, precisely

**`trossen_real/rl/real_residual_env.py`**:
- Import `EpisodeBoundaryMonitor` from `trossen_real.human_intervention.episode_boundary` (the same class introduced earlier this session for teleop/infer — not duplicated).
- `__init__`: construct `self.episode_monitor = EpisodeBoundaryMonitor(pedal)` whenever `pedal is not None` (independent of `self.intervention`, which still requires both `leader` and `pedal`).
- `reset()`: replaced the old `self.pedal.consume_reward_latched()` stale-latch drain with `self.episode_monitor.start_episode()`, which drains both stale reward latch and stale reset-press.
- `step()`: replaced direct `self.pedal.consume_reward_latched()` / `self.pedal.consume_reset()` calls with `self.episode_monitor.get_frame_reward()` / `self.episode_monitor.check_episode_end()`; the returned stop reason (`"reset_pedal"` / `"reward_release"` / `None`) is now logged into the existing per-step `tick_logger` JSONL for offline diagnosis.
- Module docstring updated to describe the new behavior.
- **`step()` also computes a separate `buffer_done` value** (see §8) — kept fully separate from `terminated`/`truncated`/the auto-reset control flow, which remain exactly as described above.
- Nothing else in the file was touched — obs-building, action execution, intervention handling, and the auto-reset control flow at the end of `step()` are unchanged.

**`scripts/TrossenStation1Real/train_residual_rl_real.sh`**:
- §16 REWARD comment rewritten to state verified facts (see §1/§2 above) instead of unverified claims, and to describe the new reward-release-ends-episode behavior.
- `REWARD_SHAPING` set to `"false"` (per your instruction) — see §1 for why this currently has no numerical effect on this dataset.
- §9 `OFFLINE_EPISODES` comment corrected: this dataset has 136 episodes, not 161 (161 was a different, older dataset variant).
- Pre-flight checklist blurb updated to mention the new reward-pedal-release-ends-episode behavior and the "press reset while still holding reward" operator habit.

## 6. What was explicitly NOT changed, and why

- `algo.terminated_only_bootstrap` — left at its default `false`. Already correct for this sparse-reward run (see §2); only relevant to the separate PBRS `reward_model` system.
- `_get_reward()` / `_populate_offline_buffer()` — untouched. The `next.done`/`next.reward` identity issue (§1) is a dataset-labeling characteristic, not a code bug; no code change can fix it without either re-labeling the dataset or accepting that `REWARD_SHAPING` is inert on this data.
- `get_envs()`'s pedal/leader construction gating (`pedal` is only constructed when `enable_intervention=true`) — left as-is. This `.sh` already sets `ENABLE_INTERVENTION="true"`, so the pedal is available; decoupling pedal construction from intervention (as was done for the infer app earlier this session) was out of scope of what was asked here and would be an unrelated change to hardware-connection-requirement logic.
- `MultiStepTransform`/`rb_transforms.py` itself — untouched. §8's fix works entirely by changing what value gets fed into the existing, unmodified transform, not by changing the transform.
- `agent.clip_q_target_to_reward_range` — considered (see §8) and explicitly **not** used as the fix. It would have papered over the target-inflation symptom without addressing which states get labeled rewarding in the first place, and — separately from that objection — it's a global clamp; the `buffer_done` approach fixes the actual mechanism (bootstrap cutoff) at the source.

## 8. The real issue: holding the reward pedal through non-rewarding motion, and the fix that actually addresses it

This section exists because the investigation got this wrong twice before landing on the right answer — kept in deliberately, not cleaned up, because the reasoning for why the first two attempts were wrong is exactly what makes the third one trustworthy.

### The scenario

The user's actual operating pattern: hold the reward pedal down from the moment of success (e.g. cube insertion) continuously through some amount of *additional, non-rewarding* motion afterward (e.g. backing the gripper away), only releasing it (or pressing reset) once that motion is done. Under §4's design, every one of those in-between steps also gets `reward=1` (online reward is a direct, literal reflection of "is the pedal down right now" — it has no notion of what the arm is physically doing).

### Dead end #1 — `agent.clip_q_target_to_reward_range` (rejected)

First proposed clamping `target_q` to `[0,1]` via this existing, already-built config flag (`q_agent.py:338-339`, comment: *"Sparse rewards are in {0,1}"*). **The user correctly rejected this as a "cheat code"**: it doesn't stop the critic from being trained to predict a high value for the backward-motion states in the first place — reward=1 is still being attached to them, so the critic still learns "this state is worth ~1," which is the actual problem (mislabeled states), not just the symptom (unbounded targets). It also doesn't actually apply to this run: `clip_q_target_to_reward_range` defaults to `False` and is never set in `train_residual_rl_real.sh`, so target_q is genuinely unbounded above for a `reward=1, done=False` transition — `target_q = 1 + gamma^n * Q(next)`, which grows without limit as the critic's own estimate of post-success states rises, a real overestimation-compounding risk, not a hypothetical one.

### Dead end #2 — naively widening `terminated` (would have crashed training)

The user's own proposed fix — mark `done=True` whenever `reward=1` — is the *mathematically* correct direction (see below), but the first attempt at implementing it (widen the `terminated` tensor `step()` already returns) would have broken the training loop. Traced precisely: `train_residual_td3.py`'s main loop never calls `env.reset()` itself — it trusts `real_residual_env.py`'s own internal auto-reset to have already substituted a fresh post-reset observation into `next_obs` whenever a real physical reset occurred, and does `obs = next_obs` unconditionally (`train_residual_td3.py:1356`, `:1856`). But its `if done.any():` block (`:1811`) unconditionally reads `info["final_info"]`, which `real_residual_env.py` only populates when a *physical* reset actually happens. Widening `terminated` to fire on every reward=1 tick — without a physical reset happening — would make `done.any()` fire with `info["final_info"]` never set, crashing with `KeyError: 'final_info'` the first time the pedal was held past one step.

Also found, separately: even if that crash weren't an issue, widening `terminated` specifically would have been *aiming at the wrong field*. `MultiStepTransform` is constructed with no custom `done_key` (`train_residual_td3.py:748-758`), so it defaults to reading `"done"` for both trajectory segmentation and (since `use_terminated_for_bootstrap=algo.terminated_only_bootstrap=False` for this run) the bootstrap-cutoff mask itself (`rb_transforms.py:379`, the `else` branch). The `"terminated"` field stored in the buffer is currently **inert** for this run's configuration — it's only consulted when `terminated_only_bootstrap=True`, which it isn't.

### The actual fix — decoupled `buffer_done`, verified end-to-end

Two signals, doing two separate jobs, computed both in `real_residual_env.py::step()`:

- **`terminated`/`truncated`** (returned to the caller, unchanged from §4): drives whether `step()` physically calls `self.reset()`, and populates `info["final_obs"]`/`info["final_info"]`. This is what the training loop's `done.any()`/episode-count/wandb logging reads — completely unaffected by this fix, zero regression risk there.
- **`info["buffer_done"]`** (new): `True` whenever `terminated_now` (the narrow physical-reset trigger) is true, OR `truncated`, OR `reward_value >= 1.0` this step — including mid-hold, with no physical reset happening. Computed once per step (`real_residual_env.py`, right after `truncated`).

`_add_transitions_to_buffer()` (`train_residual_td3.py`) was given a small, additive change: it now reads `info.get("buffer_done", None)` and, when present, uses it to override *only* the transition's stored `next.done`/`next.terminated` fields (the ones `MultiStepTransform` actually reads) — falling back to the previous narrow `done[i]`/`terminated[i]` when the key isn't present, which is always true for sim envs (zero behavior change there, confirmed by a regression test — see §7).

Net effect: every reward=1 transition becomes its own non-bootstrapped, `target_q=reward=1` transition in the online buffer — structurally identical to how the offline dataset already treats its own wide `next.done` region (§1) — regardless of whether the robot is still physically moving. The mislabeling itself (backward motion being marked reward=1 at all) is not something any done-flag trick can fix — that's a matter of when you release the pedal, not a math bug — but the *numerical* consequence of doing so (runaway target inflation) is now bounded exactly the way the user asked for, via the actual bootstrap-cutoff mechanism rather than a post-hoc clamp.

**Files changed for this part**: `real_residual_env.py` (`buffer_done` computed in `step()`, added to `info`, logged to `tick_logger`); `resfit/rl_finetuning/scripts/train_residual_td3.py`'s `_add_transitions_to_buffer()` (reads `info.get("buffer_done")`, overrides only the stored `next.done`/`next.terminated`).

## 9. Verification performed

- `py_compile` + actual `import` of every file touched this session (`trossen_real.rl.real_residual_env`, `resfit.rl_finetuning.scripts.train_residual_td3`, and the teleop/infer files from earlier) — all clean.
- `bash -n` syntax check on the edited `.sh` — clean.
- `EpisodeBoundaryMonitor` itself (the component §4's fix depends on) was unit-tested earlier this session with a fake pedal covering: stale reset-press draining, genuine post-drain press still firing, reward held→released firing `reward_release`, and the one-tick latch-decay behavior.
- **§8's `buffer_done` fix was verified end-to-end against the REAL, unmodified `MultiStepTransform` and the REAL (patched) `_add_transitions_to_buffer()`** — not mocked math. Two scripted tests: (1) regression check confirming behavior is byte-identical to before when `info["buffer_done"]` is absent (the sim-env case); (2) a synthetic 5-step sequence (neutral → reward=1/not-physically-terminal → reward=1/not-physically-terminal → released/neutral → real reset) confirming the two held-reward steps are stored with `next.done=True` and `next.reward=1.0` exactly (no inflation), while the physically-continuing and real-reset steps behave correctly and distinctly. Both passed.
- **Not verified**: an actual real-hardware run. `TrossenResidualEnv` has no mock-hardware test harness in this repo (confirmed — `../../docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md` §7.2 lists this as a pre-existing gap), so this change could not be exercised end-to-end against the physical station. Recommend a short supervised run (a handful of episodes, human at the robot) exercising: a plain reset-pedal-press episode end, a reward-hold-then-release episode end, a deliberate double-press of the reset pedal during the settle window (confirming the next episode is no longer truncated to 1 step), and specifically your actual pattern — hold reward through some backward motion, then release — checked afterward via `scripts/inspect_replay_buffer.py` (renamed/generalized, see scripts/TrossenStation1Real/INTERVENTION_CLIPPING_AND_SYNC.md) to confirm the stored `next.done`/`next.reward` values on those transitions look as described here.
