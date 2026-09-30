# ResFiT Architecture Spec for the TikZ Figure — Sim (CAN task)

Scope: **simulation only** (robosuite `PickPlaceCan`, the "Can" task, via `dexmg`).
Real-hardware (Trossen arm) is documented separately in
[`04_trossen_real_connection.md`](04_trossen_real_connection.md) — do not mix the two
when drawing the sim figure.

This document is a synthesis. Every numbered item below points into one of three
detail files, which carry the full `file:line` citation trail for every claim:

- [`01_base_bc_policy.md`](01_base_bc_policy.md) — the frozen ACT base policy
- [`02_residual_rl_core.md`](02_residual_rl_core.md) — obs plumbing, residual actor,
  action composition, critic ensemble, buffers, update rules, gradient flow, control loop
- [`03_reward_path_and_ablations.md`](03_reward_path_and_ablations.md) — PBRS/milestone
  reward-model path, and every `scripts/CanAblationStudies/*.sh` axis

All facts were re-verified against the **current on-disk code** as of 2026-08-23
(branch `dev/abhi-resfit-setup`), including several **uncommitted** modified files
(`resfit/dexmg/environments/dexmg.py`, `resfit/lerobot/policies/diffusion/modeling_diffusion.py`,
`resfit/lerobot/utils/load_policy.py`, `resfit/rl_finetuning/config/residual_td3.py`,
`resfit/rl_finetuning/config/rlpd.py`, `resfit/rl_finetuning/off_policy/rl/actor.py`,
`resfit/rl_finetuning/off_policy/rl/q_agent.py`, `resfit/rl_finetuning/scripts/train_residual_td3.py`,
`resfit/rl_finetuning/utils/evaluate_dexmg.py`, `resfit/rl_finetuning/utils/normalization.py`,
`resfit/rl_finetuning/utils/rb_transforms.py`, `scripts/train_bc_can.sh`,
`scripts/train_residual_rl_can.sh`, `scripts/train_residual_rl_can_sparse.sh`,
`scripts/train_residual_rl_lift.sh`) — the on-disk state is what actually executes and is
what's cited throughout, not the last-committed version.

---

## 1. Base BC policy — internal structure

**Class**: `ACTPolicy` (`resfit/lerobot/policies/act/modeling_act.py`) — **not**
`DiffusionPolicy**, even though a diffusion-policy module exists in the repo (unused for
this task; loaded only if a checkpoint's `config.json["type"]` says `"diffusion"`).
Traced end to end: shell script → `BasePolicyConfig.local_path` (takes priority over a
W&B run id if set) → `resolve_local_policy_dir` → `load_policy()` dispatches on
`config.json["type"]=="act"` → `ACTPolicy.from_pretrained(...)`. Full trace: §A–§C of
[`01_base_bc_policy.md`](01_base_bc_policy.md).

**Vision encoder**: `torchvision.models.resnet18`, ImageNet-pretrained
(`ResNet18_Weights.IMAGENET1K_V1`), `FrozenBatchNorm2d` (BN stats frozen, conv weights
not). **One shared backbone** processes both cameras (not per-camera weights). Input
`(B,3,84,84)` per camera → output `(B,512,3,3)` feature map → `Conv2d(512,512,k=1)` +
2D sinusoidal pos-embed → 9 tokens/camera × 2 cameras = **18 image tokens**.
- At BC training time (external LeRobot repo): **not frozen** — trained jointly at
  `lr=1e-5` (backbone) vs `lr=1e-4` (rest of network).
- During residual RL: **entire ACT policy frozen**, proven three ways: `.eval()` never
  undone, every call wrapped in `torch.no_grad()` on top of an already-`@torch.no_grad`
  `select_action`, and its parameters appear in none of the 3 RL optimizers. See §D.

**Text/task conditioning**: **NOT FOUND** anywhere in ACT. No CLIP text tower, no
language conditioning. (The only task-string in the whole pipeline —
`"pick up the can and place it in the bin"` — conditions the *external SARM/TCC reward
model*, unrelated to the BC policy; see item 6.)

**Image+state combination**: not concat/FiLM/cross-attention/dot-product in the usual
sense — it's a **DETR-style transformer**: 18 image tokens + 1 state token
(`Linear(9,512)`) + 1 zero-latent token (no VAE) = 20 tokens → 4-layer transformer
encoder (post-norm, 8 heads, FFN 512→3200→512, ReLU) → 1-layer decoder with 20 learned
query positions, cross-attending to the 20 encoder tokens → `Linear(512,7)` action head.

**Proprioceptive state**: raw concat of 3 robosuite keys (`robot0_eef_pos`(3) +
`robot0_eef_quat`(4) + `robot0_gripper_qpos`(2) = **9-D**), one `Linear(9,512)`
projection, no separate per-component encoder.

**Action chunking**: `chunk_size = n_action_steps = 20` (equal ⇒ no receding-horizon
re-query within a chunk, no temporal ensembling — `temporal_ensemble_coeff=null`). A
per-env `deque` action queue means the actual 20-token transformer forward pass runs
**once every 20 env steps**; the other 19 calls just pop a cached action, no compute.
At `control_freq=20`Hz this is a **1Hz effective replanning rate**. The base policy's
`select_action()` is still *called* every env step (cheap queue-pop when non-empty).

---

## 2. Observation plumbing

Single `robosuite.make()` render per env step produces both `observation.state` (9-D)
and `observation.images.{agentview, robot0_eye_in_hand}` (2 cameras, **84×84**, the
`camera_size=84` default). **There is only one resolution actually fed to any
network** — base policy, RL actor, RL critic, and (when enabled) the reward server all
consume the same 84×84 frames. A separate `sim.render()` call at 240×320 exists purely
for video-logging and is never fed to a network; "256×256" appears only as an unused
hypothetical in shell-script comments. Full trace: §2 of
[`02_residual_rl_core.md`](02_residual_rl_core.md).

Preprocessing: `uint8→float32`, `/255`, HWC→CHW in the env wrapper (no resize/crop — the
sim renders directly at 84×84); the RL encoder additionally does `obs - 0.5` centering
and, only during gradient updates (not action selection), a `RandomShiftsAug`
(pad-4-then-crop) augmentation.

Key routing:
- **Base policy** sees the raw (unaugmented, unstandardized) obs dict, its own camera
  keys.
- **RL actor/critic** see only `cfg.rl_camera` (same 2 cameras for Can), plus two
  augmented keys added by the wrapper: `observation.base_action` (the base policy's
  action, scaled to `[-1,1]`) and a **z-scored** `observation.state`.
- **Reward server** (only when explicitly enabled — off by default, see item 6) gets a
  single camera, `observation.images.agentview`.

---

## 3. Residual actor

**Class**: `Actor` (`resfit/rl_finetuning/off_policy/rl/actor.py`).

```
visual features [B, patches, dim] --flatten--> Linear(repr_dim, 128) → LayerNorm → ReLU
                                                          │
concat: [compressed_feat(128), state(9), base_action(7)] = 144-D  (Can task)
                                                          │
Linear(144,1024) → LayerNorm → ReLU → Linear(1024,1024) → LayerNorm → ReLU → Linear(1024,7) → Tanh
                                                          │
                                          mu ∈ [-1,1]  →  mu * action_scale
```

Input is **visual features (own encoder, not the base policy's) + proprioceptive state
+ the frozen base policy's own action** — no raw pixels reach the MLP directly, no
CLIP embedding. Output is 7-D (Can), range `[-action_scale, action_scale]` for the
distribution mean; the sampled action is separately hard-clamped to `[-1,1]`.

**Zero-init**: exact — `actor_last_layer_init_scale=0.0` triggers
`nn.init.normal_(weight, std=0.0)` / `nn.init.normal_(bias, std=0.0)` on the *last*
`Linear` of the policy MLP, i.e. weight and bias are set to **exactly 0**. `tanh(0)=0`
⇒ residual mean is exactly zero at step 0 (verified in code, not inferred).

---

## 4. Action composition

Literal line (`resfit/rl_finetuning/wrappers/residual_env_wrapper.py:136-139`):
```python
combined_naction = self._last_base_naction + residual_naction   # NOT clamped here
env_action = self.action_scaler.unscale(combined_naction)       # clamps to [-1,1] internally
```
So: **`a_base + a_res`, unclamped**, then the clamp happens *inside* `unscale()` for
whatever action actually drives the simulator. **Important code-level quirk for the
figure**: the value written into the replay buffer (`info["scaled_action"]`) is the
**unclamped** sum — three *other* call sites (target-Q computation, actor loss,
eval Q-logging) each independently recompute and explicitly clamp
`clamp(base+residual, -1, 1)`. This is numerically minor (action_scale is small,
0.1–0.2) but is a real inconsistency, not a documentation error — see
[`02_residual_rl_core.md`](02_residual_rl_core.md) §4 discrepancy #1.

**Alpha (`action_scale`) — scalar or per-dimension, both used in practice**:
| Script | value |
|---|---|
| Config default | `0.1` (scalar) |
| `train_residual_rl_can.sh` (dense) | `0.2` (scalar) |
| `train_residual_rl_can_sparse.sh` | `[0.15,0.15,0.15,0.05,0.05,0.05,0.1]` (**7-D per-dim**: xyz, rot, gripper) |
| `train_residual_rl_lift.sh` | `0.1` (scalar, matches default) |

The residual is queried **every RL env step** (not once per base-policy chunk) and
added to whatever base action the ACT queue currently pops — the residual actor has no
visibility into the base policy's internal 20-step chunk structure.

---

## 5. Critic ensemble

**Class**: `Critic` / `SpatialEmbQEnsemble` (`resfit/rl_finetuning/off_policy/rl/critic.py`).

- **Ensemble size**: `num_q=10`, independent per-head weights (batched via
  `vmap`/`functional_call`, not 10 sequential passes).
- **Per-head**: shared trunk (`Linear→LayerNorm→ReLU` proj to 1024, weighted-sum spatial
  pooling) → 2-hidden-layer `HeadMLP`: `Linear(→1024)→LayerNorm→ReLU ×2 → Linear(→1)`
  (LayerNorm after every hidden layer, **not** after the final output layer).
- **Input**: visual features from the **RL's own** encoder (shared with the actor, but
  gradients from the actor loss are `.detach()`-ed off it — only the critic loss trains
  it), state, and the **composed** (base+residual) action from the buffer — not the raw
  residual alone.
- **Target critics**: one `critic_target` (deep-copy of the full 10-head ensemble),
  Polyak `tau=0.005`, updated after *every* critic gradient step (not just delayed
  ones). `.train(False)`, all forward calls under `no_grad`.
- **TD target**: **min over a random 2-of-10 subset** (RED-Q style, `min_q_heads=2`,
  fresh random pair drawn every call) — not min-over-all-10, not a fixed pair.
- **Actor gradient**: **ensemble mean over all 10 heads** (`policy_gradient_type=
  "ensemble_mean"`), `-q.mean()`.

---

## 6. Reward path

**Default state for the two main per-task scripts (`train_residual_rl_can.sh`,
`train_residual_rl_can_sparse.sh`): the reward-model path is OFF**
(`REWARD_MODEL_ENABLED="false"`). They train on the plain robosuite env reward (dense or
sparse, toggled by `REWARD_SHAPING`). The PBRS/SARM path is **opt-in**, exercised only
by two dedicated scripts: `train_residual_rl_can_sarm_stage_pbrs.sh` and
`_milestone.sh`. **State this explicitly on the figure** so the reward-model box doesn't
read as the default path.

When enabled: `StageAwarePBRSWrapper` sends the single `agentview` 84×84 frame (batched
over up to `query_every_k=4` accumulated steps) to an **external** HTTP server's
`/predict_stage` endpoint, roughly every 4 env steps (≈5Hz at 20Hz control). The server
returns `gated_stage` (an int 0..3); everything else is optional/diagnostic.

**PBRS formula** (active values): `r = r_sparse + 0.995·φ(s') − φ(s)`,
`φ = [0.0, 0.05, 0.10, 0.15]`, `φ(terminal)=0`, `keep_sparse_term=True`.

**Milestone formula** (active values): ratchet payout
`Σ payouts[last_paid+1..stage]` with `payouts=[0.0,0.3,0.3,0.4]` (paid once per newly
reached stage, never re-paid) **plus** a `+1.0` success bonus gated on the *true*
simulator sparse-success flag at termination (max per-episode: `0.3+0.3+0.4+1.0=2.0`).

**Combination with env reward**: the shaped reward **replaces** the environment reward
entirely in the returned 5-tuple — not summed upstream anywhere.

**Storage vs. sample time**: computed and stored **at step/storage time**, verbatim, no
scaling; a shared `MultiStepTransform` (n=3) then **re-sums** the raw stored per-step
rewards into an n-step return **at sample time**, identically for PBRS and milestone (it
has no knowledge of reward mode).

**Streak-hysteresis gating — flag for the figure**: this is **not implemented anywhere
in this repo**. `grep -rn "streak"` across all `.py` files returns zero hits. The
detailed hysteresis pseudocode in `../docs/rewards/REWARD_MODEL_INTEGRATION.md` describes the
**external** SARM/TCC server's presumed internal behavior — this repo's client only
sends `hysteresis_k`/`conf_threshold`/`monotonic` as request parameters and receives an
already-gated `gated_stage` back. Do not draw a hysteresis box inside the ResFiT
codebase; if drawn at all, it belongs inside the external reward-server box.

---

## 7. Replay buffers and sampling

**Two separate buffers** (`online_rb`, `offline_rb`, both TorchRL
`TensorDictPrioritizedReplayBuffer`), not one buffer with a flag.

- `online_rb`: capacity 80k (Lift/Can-dense) or 100k (Can-sparse); starts **empty**,
  filled by an explicit random-exploration **warmup** loop (base_action + uniform noise,
  by default) until `learning_starts=10,000` transitions — the residual actor network is
  not used at all during this phase.
- `offline_rb`: sized dynamically to the whole offline demo dataset (subset); populated
  once from a LeRobot dataset, with the frozen base policy re-run on every dataset
  observation to fill `observation.base_action` (dataset ground-truth action stays as
  the stored `action`).

**Batch**: 256 total, split 128/128 (`offline_fraction=0.5`) and concatenated.

**n-step**: `n=3`, applied identically to **both** buffers via the same
`MultiStepTransform`.

**UTD**: `num_updates_per_iteration=4`, on **every single env step** — 4 critic updates,
of which exactly 1 (the last) also updates the actor. (Not "actor updated every 4th env
step" — it's every env step, at 1/4 the frequency of critic gradient calls.)

---

## 8. Update rules

**Critic loss** (MSE variant, the default in all 3 main scripts):
```python
q_all = critic(feat, state, action)                      # [10, B]
td_errors = (q_all - target_q).abs().mean(dim=0)          # mean |TD error| across heads first
critic_loss = (td_errors ** 2).mean()
```
**Bellman target**: `target_q = reward + γⁿ·𝟙[nonterminal] · min_{2-of-10}(critic_target(s', clip(base'+actor_target(s'))))`.

**Residual actor loss**:
```python
action_pred = actor(obs, stddev=0.0)     # NOTE: hardcoded 0.0, not the real stddev schedule — in-code comment flags this as "not fully verified"
combined = clamp(base_action + action_pred, -1, 1)
actor_loss = -critic.q_value_for_policy(feat, state, combined).mean() + action_l2_reg_weight * ||action_pred||²
```
`action_l2_reg_weight=0.0` in all 3 main scripts (present in code, numerically inactive).

**BC regularization**: `bc_loss_coef=0.0` in all 3 main scripts — the BC-regularized
actor-update path (`update_actor_rft`) is not reached in normal online training at all
(main loop always passes `bc_batch=None`). A **separate** TD3-BC-style BC loss
(`update_actor_offline`) exists but only runs during the distinct offline-RL
pretraining phase (item 9), with `offline_rl_bc_alpha=0.2` (a code comment notes: "paper
suggested value is 2.5 [TD3-BC paper, Fujimoto & Gu 2021] but the bc loss is based on
full action range not residual" — i.e. deliberately lowered for the residual setting).

**Delayed updates**: policy delay = 4 critic-updates-per-1-actor-update, every env step.

**Target policy smoothing**: `TruncatedNormal` noise, `sigma`=the `stddev` schedule
value (constant in all 3 scripts: 0.025 dense-Can/Lift-ish, 0.05 Lift, or a 7-D vector
for sparse-Can), `clip=stddev_clip=0.3`. Applied **only** to the target actor's action
inside the critic's target computation — never to the main actor at act-time (no noise)
or at actor-loss time (noise clip is passed but `stddev` is hardcoded to 0, so no noise
is actually sampled there either).

**Optimizers**: AdamW for all three trainable nets. `actor_lr=1e-6`, `critic_lr=1e-4`
(shared by the RL's own vision encoder too) — a **100× LR gap** between actor and
critic/encoder. Grad-clip norm 1.0 for both.

---

## 9. Offline pre-training stage

**Two mutually exclusive mechanisms** (a real code fact, not a documentation
simplification — see [`02_residual_rl_core.md`](02_residual_rl_core.md) §9
discrepancy #4):

1. **Plain critic warmup** (`critic_warmup_steps`, default 10,000): critic-only gradient
   updates before online interaction begins, on data already in the two buffers. Only
   runs if `train_offline_rl=False`.
2. **Offline TD3-BC pretraining** (`train_offline_rl=True`, `offline_rl_steps`,
   default 50,000; set to 25k/50k/75k/200k across ablation scripts): critic updated
   every step with the ordinary critic loss; actor updated every `policy_delay=2` steps
   with a **TD3-BC loss** — `-λ·Q(s,a) + MSE(action_pred, target_residual)`,
   `λ = alpha / |Q_target|.mean()`, `alpha=0.2`. Data drawn from **both** buffers
   (offline demos + whatever's in the online buffer post-warmup) via
   `EpochUnifiedDataset`, `fixed_ratio` (default 0.5) or `proportional` sampling.

**Checkpoint transfer**: after the offline-RL phase, the **whole `QAgent`** state dict
(actor + critic + target nets + encoder, no partial transfer) is optionally reloaded
from either the `latest` (default, in-memory) or `best` (highest offline-eval success
rate seen) checkpoint before online training starts — this choice (`best` vs `latest`)
is itself swept across the `scripts/CanAblationStudies/*_best.sh` variants.

**CanAblationStudies (`/scripts/CanAblationStudies/`, 25 scripts) — 6 verified axes**,
each with its own config field and consuming code line (full table in
[`03_reward_path_and_ablations.md`](03_reward_path_and_ablations.md) §9):
critic-warmup on/off, BC-checkpoint quality ("AvgBC" — a naming convention for a
different W&B artifact version, not a code-defined concept), offline-RL on/off + step
count (25k/50k/75k/200k, with 200k = **pure offline, zero online steps**), offline-RL
sampling method + ratio, offline→online checkpoint choice (best/latest), and residual
action-scale schedule (scalar vs. the 7-D per-dimension vector from item 4).

---

## 10. Gradient flow

| Module | Status | Proof |
|---|---|---|
| Base ACT policy | **FROZEN** | `.eval()` never undone; every call `no_grad()`-wrapped on top of an already-`@torch.no_grad` method; absent from all 3 optimizers |
| Residual actor | **TRAINED** | `actor_opt = AdamW(actor.parameters(), lr=1e-6)` |
| RL's own vision encoder (shared actor+critic) | **TRAINED**, by critic loss only | `encoder_opt` uses `critic_lr`; `obs["feat"].detach()` before the actor-loss call strips actor gradients from reaching it |
| Critic ensemble (10 heads) | **TRAINED** | `critic_opt = AdamW(critic.parameters(), lr=1e-4)` |
| Target critic | **FROZEN, Polyak-only** | `.train(False)`, no optimizer, `no_grad()` forward, `soft_update_params(tau=0.005)` |
| Target actor | **FROZEN, Polyak-only, but left in `.train(True)` mode** (asymmetric vs. target critic — a real code quirk, numerically inert only because dropout=0) | No optimizer; `assert actor_target.training` in code |

**Important scoping note for the figure**: the residual actor/critic do **not** share
or fine-tune the base ACT policy's ResNet18. They run a completely separate `VitEncoder`
over the same camera frames. Do not draw one shared vision-encoder box for both paths.

---

## 11. Control flow (online loop, pseudocode)

See [`02_residual_rl_core.md`](02_residual_rl_core.md) §11 for the fully
file:line-annotated version. Condensed:

```
while global_step <= total_timesteps:
    stddev = schedule(global_step)
    residual_action = agent.act(obs, stddev)                 # actor forward, no base-policy involvement here

    next_obs, reward, terminated, truncated, info = env.step(residual_action)
      # inside the env wrapper:
      #   combined = last_base_action + residual_action        (UNCLAMPED)
      #   env_action = unscale(combined)                        (clamped [-1,1] here)
      #   raw_obs, reward, ... = sim.step(env_action)
      #   next_base_action = base_policy.select_action(raw_obs) (no_grad, cached chunk pop or fresh 20-step inference)
      #   augmented_obs = {..., observation.base_action: next_base_action, observation.state: standardize(state)}
      #   info["scaled_action"] = combined                      (this is what gets stored)

    buffer.add(obs, action=info["scaled_action"], reward, next_obs, done)   # -> online_rb, MultiStepTransform intercepts
    obs = next_obs
    global_step += num_envs

    if global_step % update_every_n_steps == 0:
        for i in range(4):                                    # UTD=4
            batch = concat(online_rb.sample(128), offline_rb.sample(128))
            update_actor = (i == 3)                            # only the last of 4
            agent.update(batch, stddev, update_actor)
              # critic: target = r + γ^n·1[nonterm]·min_2of10(critic_target(s', clip(base'+actor_target(s')+smoothing_noise)))
              #         loss = MSE(mean_heads(|Q - target|))  → encoder_opt.step(); critic_opt.step()
              #         soft_update(critic, critic_target, 0.005)
              if update_actor:
                  feat = feat.detach()
                  action_pred = actor(obs, stddev=0.0)          # hardcoded, not the schedule value
                  combined = clamp(base_action + action_pred, -1, 1)
                  actor_loss = -critic.q_value_for_policy(...).mean(dim=heads)   # ensemble MEAN
                  actor_loss.backward(); actor_opt.step()
                  soft_update(actor, actor_target, 0.005)
```

Phases **before** this loop (in order): populate `offline_rb` from the demo dataset →
random-warmup fill `online_rb` to 10,000 transitions → **either** offline TD3-BC
pretraining **or** plain critic warmup (mutually exclusive, item 9) → the loop above.

---

## 12. Provenance

I only have the **arXiv abstract** for the ResFiT paper (2509.19301) — I did not fetch
or read the full PDF, so anything below marked "paper" is inferred from (a) that
abstract, (b) whether a mechanism lives in the **committed** baseline code (this repo's
`README.md` calls itself "the official release of code for" the paper) vs. in
**uncommitted diffs / untracked directories / dedicated ablation scripts**, and (c)
whether it matches a named prior algorithm this repo's own filenames reference
(`rlpd.py` → almost certainly RLPD, Ball et al. 2023; TD3-BC alpha comment explicitly
names "the paper" as TD3-BC, Fujimoto & Gu 2021). Treat "your own addition" verdicts
below as high-confidence given the git-status evidence; treat "paper" verdicts as
lower-confidence since I haven't read the full ResFiT text.

| Item | Verdict | Basis |
|---|---|---|
| Stage-aware PBRS shaping | **Your own addition** | Abstract states the method needs only "sparse binary reward signals" — dense PBRS/SARM shaping is additive on top. The wrapper (`pbrs_wrapper.py`), the SARM/TCC HTTP client, and the two dedicated `*_sarm_stage_*.sh` scripts are all off-by-default in the two main per-task scripts. |
| Streak-hysteresis gating | **Not in this repo at all** — not "yours" or "the paper's" as *code*; it's prose describing an external server's presumed behavior | `grep -rn "streak"` = 0 hits repo-wide. Described only in `../docs/rewards/REWARD_MODEL_INTEGRATION.md`'s prose about the external SARM/TCC service. |
| Per-dimension alpha (`action_scale`) | **Your own addition/experiment** | `Any`-typed field generic enough to take a scalar (used in 2 of 3 main scripts) or a list; the per-dim vector is only exercised in the sparse-Can script and the `*_listedResScale.sh` ablation round — reads as your own exploration, not a documented paper default. |
| n-step=3 and 50/50 offline/online sampling | **Likely inherited from RLPD** (Ball et al., "Efficient Online RL with Offline Data"), which this file is literally named after (`rlpd.py`) and is known for exactly this symmetric-sampling recipe — **not independently confirmed against the ResFiT paper's own text** | Filename + the well-known RLPD design; abstract-only access to ResFiT itself. |
| BC coefficient (`offline_rl_bc_alpha=0.2`) | **A deliberate deviation from a cited prior paper (TD3-BC), present in the base/committed code, not obviously your addition** | In-code comment: *"paper suggested value is 2.5 (but the bc loss is based on full action range not residual)"* — this describes adapting TD3-BC's alpha for the residual setting; whether this specific value was set by the ResFiT authors or tuned later is not determinable from the repo alone. |
| Checkpoint-selection rule (`best` vs `latest`) | **Your own addition** | Only exercised as a swept axis in `scripts/CanAblationStudies/*_best.sh` (6 of 25 scripts) — reads as your own ablation design, not a documented paper default (`latest` is the config default). |

---

## 13. Figure guidance

### Suggested 6–8 execution-order steps (figure caption badges)

1. **Env step** — robosuite renders one 84×84 frame per camera (2 cams) + reads 9-D
   proprioceptive state, one render call.
2. **Base policy query** — frozen ACT (ResNet18 → 4-layer transformer encoder →
   1-layer decoder) is queried once every 20 env steps and predicts a 20-step action
   chunk; every step in between just pops the next cached action (no compute).
3. **Residual query** — a separate, trainable actor (its own ViT encoder → 128-D
   compressed feature + state + base action → 2-layer MLP → Tanh) outputs a small
   correction every single env step, mean-initialized to exactly zero.
4. **Composition & execution** — `a = clip(a_base + α·a_res, -1, 1)` is sent to the
   simulator (`α` = a scalar or 7-D per-dimension bound, 0.1–0.2 typical).
5. *(optional, off by default)* **External reward scoring** — every ~4 steps, the
   agentview frame is sent to an external stage-classifier server; its returned stage
   index drives a potential-based (PBRS) or milestone-ratchet shaped reward that
   **replaces** the sim reward.
6. **Buffer write** — `(s, a_composed, r, s')` is written to one of two replay buffers
   (online, empty-start + warmup-filled; offline, preloaded from demo data), with an
   n-step (n=3) return resolved later at sample time.
7. **Gradient step** — every env step: sample 128 online + 128 offline transitions →
   4 critic updates (10-head ensemble, min-of-random-2 target, Polyak τ=0.005) → 1
   delayed actor update (ensemble-mean Q-gradient, LR 100× smaller than the critic's).
8. *(optional prior phase, not per-step)* **Offline warm start** — before online
   training, either a critic-only warmup or a full offline TD3-BC pretraining phase
   initializes the actor/critic (mutually exclusive with each other).

### Omit from the diagram (real, but adds clutter without aiding understanding)

- The unclamped-vs-clamped buffer-storage inconsistency (item 4) — a code detail, not a
  designed architectural feature; footnote it in text if you must, don't draw two paths.
- The hardcoded `stddev=0.0` in the actor loss, and the target-actor `.train(True)` vs
  target-critic `.train(False)` asymmetry — both are code quirks, not intended design.
- The `vmap`/`functional_call` batched-ensemble implementation trick — draw the 10
  critic heads conceptually, not their PyTorch execution mechanism.
- `RandomShiftsAug` data augmentation — a training-time regularizer, not a structural
  block.
- The second 240×320 `sim.render()` call — video-logging only, never touches a network.
- BC/RA-BC regularization terms — present in code but numerically inactive
  (`bc_loss_coef=0.0`, `action_l2_reg_weight=0.0`) in all 3 main training scripts.
- Priority-replay importance weights — inactive (`sampling_strategy="uniform"`).
- The entire real-hardware/Trossen path — belongs in a separate figure per
  [`04_trossen_real_connection.md`](04_trossen_real_connection.md).

### Likely contradictions with prior written material

I don't have your Chapter 4 text to check directly — the items below are contradictions
found between **the current code** and **this repo's own pre-existing analysis docs**
(`../docs/algorithms/RESIDUAL_LEARNING.md`, `../docs/policies/ACT_ARCHITECTURE.md`, `../docs/training/RESIDUAL_RL_TRAINING.md`,
`../docs/rewards/REWARD_MODEL_INTEGRATION.md`), which may be the same source your chapter draws from —
worth cross-checking your text against these specifically:

1. Composed action is stored **unclamped** in the buffer; `../docs/algorithms/RESIDUAL_LEARNING.md` and
   `../docs/algorithms/CRITIC_LOSSES_EXPLAINED.md` both imply/state it's clamped there.
2. `action_scale` (α) is documented elsewhere as always scalar (`0.2`); it is actually a
   **7-D per-dimension vector** in the sparse-Can config and in the `listedResScale`
   ablations.
3. "Actor updated every 4th step" (prior docs' phrasing) actually means every 4th
   *gradient call*, not every 4th *env step* — the actor updates once per env step, same
   as the critic, just less often per gradient-call batch.
4. Critic warmup and offline TD3-BC pretraining are described in places as compatible;
   they are **mutually exclusive** in code (offline-RL running disables warmup
   entirely, regardless of `critic_warmup_steps`'s value).
5. `../docs/policies/ACT_ARCHITECTURE.md`'s worked numeric example (token counts, state/action dims) is
   for **TwoArmCoffee/GR1** (3 cams, 36-D state, 24-D action), not Can (2 cams, 9-D
   state, 7-D action) — the architectural description transfers, the numbers don't.
6. The streak-hysteresis mechanism in `../docs/rewards/REWARD_MODEL_INTEGRATION.md` reads as
   code-grounded but describes an **external, unverifiable-from-here** server, not
   anything implemented in this repo.
