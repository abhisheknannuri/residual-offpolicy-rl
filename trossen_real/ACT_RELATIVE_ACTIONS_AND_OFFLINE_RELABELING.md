# ACT chunk-anchor-relative actions: impact on this repo (currently: none)

Status as of 2026-08-31: **investigated, documented, deliberately not
adopted yet.** The actively-deployed checkpoint
(`policy_bc_PickAndInsertCubeStation1_merged_deltaJoint`) still uses the
static per-frame-delta dataset (`convert_to_delta_joint_dataset.py`) and
nothing in this repo changed. This doc exists so a future you (or me) knows
what was found, what to do about it, and doesn't have to re-derive it.

**No files in this repo (`residual-offpolicy-rl`) were modified for this
investigation.** All code changes live in the OTHER repo
(`.../GeneralistRewardModels/lerobot`) - see its own
`custom_scripts/RELATIVE_ACTIONS_ACT.md` for the full technical writeup
(math, exact diffs, verification performed). Read that first; this doc is
the "what does it mean for `residual-offpolicy-rl`" companion.

## The root cause (why the robot sometimes jumps / gripper doesn't close)

Verified end-to-end via actual code (both repos), not assumed from comments:

- Your dataset's delta target is `delta[t] = action[t] - state[t]`, a
  DIFFERENT reference state per frame, baked in statically at conversion
  time (`convert_to_delta_joint_dataset.py`).
- `trossen_real/inference/policy_client.py::reconstruct_absolute_action()`
  reconstructs `absolute[t+k] = predicted[k] + state_measured_right_now`.
  This is only correct for `k=0` (the first step of a freshly-inferred
  chunk) - for `k>0` (when `--n-action-steps > 1` at `policy_server.py`) it
  uses the WRONG reference state, and the error compounds.
- OpenPI's fix - and now ACT's, in the other repo (see its doc) - anchors
  the WHOLE chunk's delta to ONE state (the observation-time state), so
  reconstruction is `absolute[t+k] = predicted[k] + state_at_chunk_inference_time`
  for every `k` - well-defined regardless of chunk position.

This is a real, verified architectural mismatch, independent of anything
about the RL pipeline. It's a candidate root cause for the "robot jumps /
gripper sometimes doesn't close" symptom you asked about, specifically when
`--n-action-steps > 1`.

## Current mitigation (no code changed here)

For the checkpoint you're actually running today, the safe, zero-code-change
option is **`--n-action-steps 1`** on `policy_server.py` (re-infer every
tick). You said you'll evaluate the relative-actions retraining path later -
until then, `n_action_steps=1` is the only setting for which the CURRENT
checkpoint's reconstruction math is exactly correct.

## When you DO retrain with `use_relative_actions=True` (later)

Per the other repo's doc, you'll need to run its `recompute_stats
--operation.relative_action true` command on your ORIGINAL ABSOLUTE dataset,
then train with `--policy.use_relative_actions=true`. When that new
checkpoint is deployed:

- `PolicyClient.predict_full()`'s `"action"` will already be a correctly
  reconstructed ABSOLUTE action (server does it now, correctly, across all
  chunk positions - see the other repo's `policy_server.py` fix).
- `reconstruct_absolute_action(action_space="delta_joint", ...)` in THIS
  repo's `policy_client.py` must NOT be called for such a checkpoint -
  calling it would double-apply the state addition (once on the server, once
  here). Every current caller (`infer_app/infer_loop.py`,
  `scripts/live_infer_deploy.py`, `rl/real_residual_env.py`,
  `train_residual_td3.py`) currently assumes `action_space="delta_joint"`
  means "I must reconstruct" - that assumption breaks for a
  server-side-relative checkpoint.
- Proposed fix (not built - do this when you actually adopt the feature):
  add a `server_reconstructs_absolute` (or similar) field to
  `policy_server.py`'s `/health` response, and branch on it in
  `check_action_space_mismatch()`/every caller that currently calls
  `reconstruct_absolute_action()`, e.g.:
  ```python
  health = policy.health()
  if health.get("server_reconstructs_absolute"):
      absolute_action = predicted  # already absolute, use as-is
  else:
      absolute_action = reconstruct_absolute_action(action_space, predicted, follower_state, ...)
  ```
  This is a small, contained change, isolated to the handful of call sites
  above - not done now since you're not using the feature yet.

## Separate, more urgent finding: offline `base_action` relabeling bug

Found while checking whether switching the base policy's action format would
make the RL residual pipeline "messy" (it doesn't - see below) - but this is
a **real, independent, currently-live bug** regardless of relative vs.
static-delta actions, and regardless of whether you ever adopt the feature
above.

`train_residual_td3.py`'s offline buffer population
([train_residual_td3.py:1006-1042](../resfit/rl_finetuning/scripts/train_residual_td3.py))
iterates the offline dataset frame-by-frame (`shuffle=False`, sequential,
across episode boundaries) and calls `real_policy_client.predict(json_obs)`
once per frame to compute `observation.base_action` when
`use_base_policy_for_base_actions=True` - **which is the default**
(`residual_td3.py:20`) and what every real-hardware `.sh` script actually
uses (`train_residual_rl_real.sh`, `train_residual_rl_offline_pretrain.sh`,
`train_residual_rl_real_continue_from_offlineRL.sh`, all pass it through
explicitly as `"true"`).

**The loop never calls `real_policy_client.reset()`** between frames or
episodes. `policy_server.py`'s persistent `state.policy` keeps its internal
`n_action_steps` queue across ALL of these calls, since nothing ever clears
it. With `n_action_steps > 1` at the server during an offline-relabeling
run, only 1-in-N of these `/predict` calls does a genuine fresh inference on
that frame's actual image/state - the other N-1 silently pop an
already-planned action left over from queuing a chunk for a completely
unrelated, earlier frame (possibly from a different episode). The resulting
`base_action` label for those frames is not "what the frozen BC policy would
predict for this frame" - it's "whatever was still queued from a stale
chunk," which the RL residual math (`combined_action = base_action +
residual_action`) then trains against as if it were meaningful.

This is orthogonal to everything else in this doc: it exists for the CURRENT
static-delta checkpoint exactly as much as it would for a future relative
one, because it's a `policy_server.py`-queue/reset problem, not an action-
math problem.

**Impact depends entirely on what `n_action_steps` was actually used during
real offline-buffer-population runs** (not standalone BC-inference testing -
a separate, unrelated session) - you deferred checking this
("let's ignore this for now"). When you're ready to check: look at how you
actually launched `policy_server.py` in the terminal during the sessions
that built the offline buffer(s) currently in `offline_buffer_cache/`.

**Proposed fix (not applied - you said not to touch `train_residual_td3.py`
for this right now):** add `real_policy_client.reset()` immediately before
each `predict()` call in the relabeling loop - forces a genuine fresh
single-step inference every time, unconditionally correct regardless of
whatever `n_action_steps` the server happens to be configured with. Small,
one-line-plus-comment change, costs one extra cheap HTTP round-trip per
offline dataset frame (relabeling is an offline, one-time, non-real-time
pass - this cost is negligible there). Ready to apply whenever you want it;
say the word and I'll do it (with a backup first, as always for this file).

## Why the RL residual pipeline itself is NOT made messy by relative actions

You asked about this directly - confirmed, not assumed:

- The residual math (`combined_action = base_action + residual_action`,
  `real_residual_env.py`/`train_residual_td3.py`) only cares that
  `base_action` is a valid absolute joint target for the CURRENT tick -
  it's agnostic to how that absolute value was derived internally.
- The offline relabeling loop above always uses each frame's OWN recorded
  state as the query context - i.e. chunk-position-0 semantics
  (`delta[t+0] = action[t] - state[t]`), which is IDENTICAL between the old
  static-per-frame scheme and the new chunk-anchor-relative scheme (they
  only differ for chunk positions `k>0`, which offline relabeling never
  uses). So switching the underlying BC checkpoint to a relative-trained one
  would NOT itself change what `base_action` means to the RL code - only the
  (separate, above) reset/queue bug would, and that bug predates and is
  independent of this whole investigation.

## Summary of what to do, in order, whenever you pick this back up

1. Decide whether to check historical `n_action_steps` usage for your
   existing offline buffers (the reset/queue bug above) - independent of
   everything else here.
2. If/when adopting relative actions: run the other repo's
   `recompute_stats --operation.relative_action true` on your absolute
   dataset, retrain ACT with `--policy.use_relative_actions=true`.
3. Add the proposed `/health` field + branch in this repo's callers (small,
   contained, listed above) before deploying such a checkpoint through this
   repo's apps.
4. Optionally: the chunk-fetch-once bandwidth optimization you raised - a
   separate, unrelated feature, not scoped here.
