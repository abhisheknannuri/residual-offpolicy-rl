# Foot pedal behavior — teleop app, infer app, and (for contrast) the RL training env

This document describes exactly what each of the 3 foot-pedal buttons does in
**each of the 3 places** they're used in this repo. It exists because the same
physical pedal means three genuinely different things depending on which app
reads it, and getting that confused is the easiest way to misinterpret a
recorded dataset or a training run.

Hardware mapping (`trossen_real/human_intervention/pedal_listener.py`,
`PedalListener`, unchanged by this feature):

| evdev code | Pedal   |
|---|---|
| `BTN_0` / `BTN_MISC` (256) | **Reward** |
| `BTN_1` (257) | **Intervention** |
| `BTN_2` (258) | **Reset** |

`PedalListener` itself only exposes raw, app-agnostic primitives:
- `get_reward()` — 1 if currently held, else 0 (plain level read).
- `consume_reward_latched()` — like `get_reward()`, but also catches a
  press-then-release that happens entirely within one control tick (a
  background thread updates the raw state faster than any control loop
  polls it). **Quirk to know about**: because it stores `self._reward_latch
  = self.reward_pressed` at the end of every call, the tick where the pedal
  is *released* still reads back `True` one more time before the *following*
  tick reads `False` — i.e. reward "decays" one tick late. This is
  pre-existing behavior (used by `real_residual_env.py` already) and is
  relied on, not incidental: it means the frame at the exact release
  instant still correctly carries `reward=1`.
- `is_intervention()` — level read of the intervention pedal.
- `consume_reset()` — edge-triggered: returns `True` exactly once for a
  press, however brief, then resets itself.

Two small NEW pieces of code sit on top of these primitives, both in
`trossen_real/human_intervention/episode_boundary.py`
(`EpisodeBoundaryMonitor`) and `trossen_real/inference/intervention.py`
(`InterventionManager`, pre-existing, infer-app-only). Everything below
describes their combined effect per app.

## Teleop app (`trossen_real/teleop/`)

No intervention concept exists here at all — the leader is *always* driving
the follower whenever teleop is active. The only new behavior is optional
(`enable_pedal_episode_control` checkbox, off by default — unchecked, this
app is byte-identical to before this feature existed):

| Pedal | Behavior |
|---|---|
| **Reset** | A single press ends the current recording immediately — exactly equivalent to clicking **Stop Recording** (episode is saved with whatever frames were buffered). Does NOT auto-start the next episode; click **Start Recording** yourself. |
| **Reward** | `next.reward` = 1.0 for every recorded frame while held (0.0 otherwise, with the one-tick decay quirk above). On the held→released edge — **only if it was held at some point during the current episode** — also ends the current recording (same as Reset: equivalent to Stop Recording, no auto-restart). |
| **Intervention** | Not read at all by teleop — has no effect. |

`next.reward` only exists as a column in the recorded dataset **when this
checkbox is enabled** — `build_lerobot_features()`'s schema is otherwise
completely unchanged from before this feature.

Reset/reward pedal presses are only consumed (and only have any effect)
while actively recording (`recorder.is_recording`) — pressing them between
episodes, before Start Recording, or after Stop Recording is a no-op.

## Infer app (`trossen_real/infer_app/`)

Two independent opt-in checkboxes, either or both may be on:
- `enable_intervention` — leader arm + intervention pedal, pre-existing (see
  `INTERVENTION_PLAN.md`), unaffected by this feature.
- `enable_dataset_recording` — LeRobot dataset recording + reset/reward
  pedal episode control, this feature. **Needs no leader at all.**

If only `enable_dataset_recording` is on (no leader), the intervention
pedal simply does nothing — there's no leader to hand control to.

| Pedal | Behavior |
|---|---|
| **Reset** | A single press ends the current inference run immediately after finishing the tick it landed on — exactly equivalent to clicking **Stop Inference** (episode is saved with whatever frames were recorded so far). Does NOT auto-start a new run; click **Start Inference** yourself. |
| **Reward** | `next.reward` = 1.0 for every recorded frame while held (same one-tick-decay quirk as teleop). On the held→released edge, if held at some point this episode, also ends the run (same as Reset). |
| **Intervention** (only if `enable_intervention` is also on) | Held down → the leader's own joint readback is sent to the follower instead of the policy's action (see `INTERVENTION_PLAN.md` for the full mechanism) — completely independent of reset/reward/dataset recording. The base policy is still queried every tick even while intervened; its proposed action is simply not applied to the follower, but IS still recorded (see below). |

When `enable_dataset_recording` is on, the recorded `infer_dataset` LeRobot
dataset carries two extra columns beyond the usual schema:
- `observation.intervened` (bool) — was this exact frame's action the
  leader's (intervened) or the policy's (autonomous)?
- `observation.policy_action` (7D, real joint units) — what the base
  policy *proposed* for this frame, regardless of whether it was actually
  used. Lets you compare "what the policy wanted to do" vs. "what actually
  happened" frame-by-frame, even on intervened frames.
- `next.reward` (float32) — the reward pedal signal, same convention as
  the hand-labeled dataset at `trossen_real/datasets/
  ..._rewardLabled_v21_old`.

One `infer_dataset/{timestamp}_{task_name}/` LeRobot dataset session is
created lazily on the **first** Start Inference press after Connect, and
accumulates one more episode per subsequent Start/Stop Inference cycle
(whether stopped manually, by `max_steps`, or by a pedal) until
**Disconnect** (or app shutdown, as a fallback), which finalizes it.

## RL training env (`trossen_real/rl/real_residual_env.py`) — for contrast only

**Not touched by this feature.** Included here only so the different
semantics don't get confused with the two apps above:
- **Reward**: `pedal.consume_reward_latched()` every step — does **not**
  by itself end an episode.
- **Reset**: `pedal.consume_reset()` sets `terminated=True` for that step
  — this is the **only** thing that ends an episode in the RL env. Holding
  the reward pedal straight through a reset can leak `reward=1` into the
  new episode's first step(s) if not released in time — `reset()` warns
  about this after the settle window (see
  `../../docs/real/REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN.md` §Reward/reset pedal timing
  bugs found and fixed).
- **Intervention**: identical mechanism to the infer app
  (`InterventionManager`, shared code), but the stored action for an
  intervened transition is the human's actual executed action, converted
  to the training convention (`_to_stored_naction()`) — there is no
  separate `observation.intervened`/`observation.policy_action` column
  concept in the online replay buffer; `info["intervened"]` and
  `info["actor_proposed_action"]` are diagnostic-only.

## Worked examples

**"I hold the reward pedal for 2 seconds then release it" (teleop or infer, recording active):**
Every frame while held gets `next.reward=1.0`. The frame at the exact tick
you released it *also* gets `next.reward=1.0` (one-tick decay). That same
tick, the episode-end check fires (`reward_release`) and the recording
stops right after that frame — equivalent to you clicking Stop
Recording/Stop Inference at that instant.

**"I tap the reset pedal quickly":**
`consume_reset()`'s edge-trigger fires regardless of how brief the press
was. The current tick finishes and records normally, then the run/recording
stops — equivalent to Stop Recording/Stop Inference.

**"I hold intervention through a reset-pedal press" (infer app only):**
Independent axes — the tick that catches the reset pedal press still
executes (either autonomously or under intervention, whichever was active
at that instant), gets recorded with the correct `observation.intervened`
value for that frame, and then the run stops.
