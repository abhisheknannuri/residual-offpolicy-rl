# ResFiT Residual RL Core — Code-Grounded Architecture Notes

Repo: `residual-offpolicy-rl` (ResFiT). All line numbers verified against the working tree
as of 2026-08-23 (uncommitted modifications per `git status`). Where a claim depends on a
file listed as modified/uncommitted, this is noted explicitly. All paths are relative to the
repo root `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl` unless given absolute.

---

## #2 Observation plumbing

### Env → wrapper stack (top to bottom of the call chain)

1. `robosuite.make()` — raw MuJoCo env, created in
   `resfit/dexmg/environments/dexmg.py:218` (`RobosuiteGymWrapper.__init__`).
2. `RobosuiteGymWrapper` (`resfit/dexmg/environments/dexmg.py:61-628`) — converts the
   robosuite obs dict into `{"observation.state": ..., "observation.images.<cam>": ...}`.
3. `VectorizedEnvWrapper` (`dexmg.py:814-894`) — numpy → torch, wraps
   `gym.vector.SyncVectorEnv`/`AsyncVectorEnv`.
4. `BasePolicyVecEnvWrapper` (`resfit/rl_finetuning/wrappers/residual_env_wrapper.py:23-211`)
   — runs the frozen base policy, composes actions, augments obs with
   `observation.base_action` and standardizes `observation.state`.

### Exact observation dict keys/shapes produced per env step

From `RobosuiteGymWrapper._process_obs()` (`dexmg.py:403-453`):

- `observation.state`: `float32`, shape `(state_dim,)`. Built by concatenating the keys
  returned by `_get_expected_low_dim_keys(env_name)` (`dexmg.py:515-551`). For single-arm
  Panda tasks (Lift/Can/Square/Threading):
  `["robot0_eef_pos"(3), "robot0_eef_quat"(4), "robot0_gripper_qpos"(2)]` → **9-D** total
  (`dexmg.py:518-522`). Two-arm Panda tasks add `robot1_eef_pos/quat/gripper_qpos` (6 more
  dims, `dexmg.py:524-529`). Humanoid tasks (Coffee/Pouring/CanSort) use a different 18-D set
  (`dexmg.py:531-538`).
- `observation.images.<camera_name>`: `float32` in `[0,1]`, shape `(3, H, W)`, `H=W=camera_size`
  (default 84). Built at `dexmg.py:434-441` via `img.astype(np.float32)/255.0` then
  `np.transpose(img,(2,0,1))`. The camera set per task comes from
  `_get_expected_image_keys(env_name)` (`dexmg.py:455-513`), e.g. Lift/Can/Square:
  `["agentview_image","robot0_eye_in_hand_image"]` → 2 cameras.

Both `observation.state` low-dim keys and `observation.images.*` are extracted from a
**single** `robosuite.env.step()`/`env.reset()` observation dict — there is only **one**
simulation/render call per env step for the policy-facing images
(`env_kwargs["camera_heights"]=self.camera_size`, `env_kwargs["camera_widths"]=self.camera_size`,
set at `dexmg.py:196-197`, default `camera_size=84`, `dexmg.py:90`).

### `BasePolicyVecEnvWrapper._augment_obs()` (residual_env_wrapper.py:170-178)

After each `env.step()`, the wrapper adds two things on top of the raw obs dict:

```python
augmented_obs["observation.base_action"] = base_naction          # scaled to [-1,1], (action_dim,)
augmented_obs["observation.state"] = state_standardizer.standardize(augmented_obs["observation.state"])
```
(`residual_env_wrapper.py:174-176`)

So the **final observation dict** seen by the QAgent (actor + critic) is:
`{"observation.images.<cam1>", "observation.images.<cam2>", ..., "observation.state" (z-scored),
"observation.base_action" (∈[-1,1], scaled)}`.

### Which keys go where

- **Base policy** (`ACTPolicy.select_action`, called inside `BasePolicyVecEnvWrapper.reset()`
  line 101-102 and `.step()` line 148-149, both wrapped in `torch.no_grad()`) is given the
  **raw** `raw_obs` dict (unstandardized state, un-augmented) straight from
  `VectorizedEnvWrapper`. Its own image keys are `list(base_policy.config.image_features.keys())`
  (`residual_env_wrapper.py:63`), independently determined by the BC policy's own training
  config — not necessarily identical to `cfg.rl_camera`, though in practice they match the
  task's full camera set.
- **RL actor & critic** (`QAgent`) only see the cameras listed in `cfg.rl_camera`
  (`resfit/rl_finetuning/config/rlpd.py:283-288`, per-task overrides in
  `residual_td3.py:438-444` etc.) — via `self.rl_cameras` in
  `resfit/rl_finetuning/off_policy/rl/q_agent.py:56`, consumed in `_encode()` (`q_agent.py:211-239`).
  For Lift/Can/Square this is the same 2-camera set the base policy uses; for box-cleanup/coffee
  it's an explicit 2-3 camera subset.
- **Reward server** (optional, `RewardModelConfig`, `resfit/rl_finetuning/config/residual_td3.py:78-155`,
  disabled by default `enabled: bool = False`): when enabled, only **one** camera is sent —
  `image_key: str = "observation.images.agentview"` (`residual_td3.py:113`), consumed by
  `StageAwarePBRSWrapper` (`resfit/dexmg/environments/dexmg.py:778-807`).

### Two image resolutions? — CORRECTED FROM ASSUMPTION

There are **not** two resolutions fed to the policy. There is exactly **one** camera
resolution used for the observation the policy/critic/base-policy see: `camera_size=84`
(`dexmg.py:90`, `create_vectorized_env(..., camera_size: int = 84, ...)` `dexmg.py:901`, never
overridden by `get_envs()` in `train_residual_td3.py:526-540` — no `camera_size=` kwarg is
passed there, so the 84 default is used).

A **second, separate** render call exists purely for **video capture** (not fed to any
network): `RobosuiteGymWrapper.render()` (`dexmg.py:553-576`) calls
`self.env.sim.render(camera_name=..., height=self.render_size[0], width=self.render_size[1])`
where `self.render_size` defaults to **`(240, 320)`** when not explicitly set
(`dexmg.py:121-126`), NOT 256×256. This is a distinct `sim.render()` call, separate from the
84×84 `use_camera_obs=True` render robosuite performs internally for the policy observation.
`render()` is only invoked by the evaluation/video-logging path (`evaluate_dexmg.py`), never by
the training-observation path. `256×256` appears **only** in shell-script comments as a
hypothetical example for `offline_data.image_size` (the *offline dataset* resize knob,
`resfit/rl_finetuning/config/residual_td3.py:24-27`) — e.g.
`scripts/train_residual_rl_can_sparse.sh:332,585,598` — it is never an actual configured value
in any of the three main training scripts (all leave `IMAGE_SIZE=84`).

### Where image preprocessing happens, exact values

1. `RobosuiteGymWrapper._process_obs()` (`dexmg.py:434-437`): `uint8→float32`, `/255.0`,
   `(H,W,C)→(C,H,W)` transpose. No crop/resize here (robosuite renders directly at 84×84).
2. `VitEncoder.forward()` (`resfit/rl_finetuning/off_policy/networks/encoder.py:29-37`):
   ```python
   if obs.max() > 5: obs = obs / 255.0     # handles uint8-stored replay-buffer images
   obs = obs - 0.5                          # center to [-0.5, 0.5]
   ```
   (encoder.py:30-32). No resize/crop at this stage either — patch conv strides consume the
   84×84 image directly.
3. `QAgent._encode()` (`resfit/rl_finetuning/off_policy/rl/q_agent.py:211-239`): converts
   uint8 replay-buffer images to float32 `/255.0` (line 224-226), then if `augment=True`
   applies `RandomShiftsAug(pad=4)` (`q_agent.py:144,231`) — random-shift data augmentation
   (pad by 4px, random crop back to 84×84), applied during **training** critic/actor updates
   (`update()` calls `_encode(obs, augment=True)`, `q_agent.py:788,791`) but **not** during
   action selection (`act()` calls `_encode(obs, augment=False)`, `q_agent.py:260`).
4. Offline dataset images can optionally be resized via `T.Resize((image_size,image_size))`
   (`train_residual_td3.py:345-350`) — controlled by `offline_data.image_size`, `None`/unset
   in all three shell scripts examined except a commented-out `IMAGE_SIZE=84` line (i.e. it
   is explicitly set to 84, a no-op resize since the dataset is already 84×84).

---

## #3 Residual actor

**Class**: `Actor(nn.Module)`, file `resfit/rl_finetuning/off_policy/rl/actor.py:69-189`.

### Architecture

```
obs["feat"]  (visual features [B, num_patch_total, patch_dim])
    │  .flatten(1,-1)                                        (actor.py:164, since cfg.spatial_emb=0 default)
    ▼
compress = Linear(repr_dim, feature_dim=128) → LayerNorm(128) → Dropout(0) → ReLU   (actor.py:93-98)
    │
    ▼  concat with prop
[compressed_feat(128), observation.state(state_dim), observation.base_action(action_dim)]   (actor.py:167-175)
    │
    ▼
policy = build_fc(policy_in_dim, hidden_dim=1024, action_dim, num_layer=2, layer_norm=1, dropout=0)
   =  Linear(policy_in_dim, 1024) → LayerNorm(1024) → Dropout(0) → ReLU
    → Linear(1024, 1024)          → LayerNorm(1024) → Dropout(0) → ReLU
    → Linear(1024, action_dim)    → Tanh                                    (actor.py:13-29)
    │
    ▼
mu ∈ [-1,1]^action_dim
scaled_mu = mu * action_scale_tensor                                        (actor.py:181)
    │
    ▼
TruncatedNormal(scaled_mu, std)   low=-1, high=1 (hard clamp)                (actor.py:187, common_utils/utils.py:154-182)
```

**IMPORTANT — the `SpatialEmb` branch is explicitly unsupported for the residual actor**:
`forward()` asserts `assert not self.residual_actor, "Not implemented"` if
`cfg.spatial_emb > 0` (`actor.py:160-162`). Since `ActorConfig.spatial_emb: int = 0` by
default (`resfit/rl_finetuning/config/rlpd.py:74`) and none of the three shell scripts
override it, the residual actor always uses the plain `Linear→LayerNorm→Dropout→ReLU`
compress branch (`actor.py:93-99`), never the critic's `SpatialEmb` attention-pooling module.

### Exact input & concatenated dim

Input is **visual features + state + base action**, concatenated
(`actor.py:167-175`, `torch.cat(all_input, dim=-1)`):

- Compressed visual feature: `feature_dim = 128` (`ActorConfig.feature_dim`, `rlpd.py:69`).
- Proprioceptive state: `prop_dim = observation.state` dim (task-dependent, e.g. 9 for Lift).
- Base action: `action_dim` — added because `residual_actor=True` triggers
  `self.prop_dim += action_dim` in `Actor.__init__` (`actor.py:77-79`) and
  `all_input.append(obs["observation.base_action"])` in `forward()` (`actor.py:171-173`).

So for Lift (state=9, action_dim=7): `policy_in_dim = 128 + (9+7) = 144`.
No raw pixels, no CLIP/vision-embedding besides the shared MinViT encoder features — actor
takes **only** the compressed visual feature, standardized state, and the frozen base
policy's normalized action. No image is passed to the actor's `build_fc` MLP directly.

### Output dim and range

`action_dim`-dimensional. `mu ∈ [-1,1]` after `Tanh` (actor.py:177), then
`scaled_mu = mu * action_scale_tensor ∈ [-action_scale, action_scale]` (actor.py:181).
`action_scale_tensor` is either a scalar or a per-dimension vector — see `to_native_list`
handling at `actor.py:118-124`. The **sampled** action from `TruncatedNormal` is hard-clamped
to `[-1+eps, 1-eps]` regardless of `action_scale` (`common_utils/utils.py:165-168`), i.e. the
`action_scale`-derived bound only constrains the **mean**, not the hard sample bound (though
with the tiny stddevs used in practice — 0.025–0.05 — samples rarely leave the
`±action_scale` neighborhood).

### Zero initialization — exact mechanism

`agent.actor.actor_last_layer_init_scale = 0.0` (config default in
`ResidualTD3DexmgConfig.agent`, `resfit/rl_finetuning/config/residual_td3.py:266-276`; also
explicitly set `ACTOR_LAST_LAYER_INIT=0.0` in all three shell scripts). In
`Actor._initialize_weights()` (`actor.py:126-157`):
```python
if cfg.actor_last_layer_init_scale is not None:      # 0.0
    final_layer = <last nn.Linear found by reverse-iterating self.policy.modules()>
    utils.initialize_layer_weights(final_layer, cfg.actor_last_layer_init_distribution, cfg.actor_last_layer_init_scale)
```
`actor_last_layer_init_distribution` default `"normal"` (`rlpd.py:83`). In
`initialize_layer_weights()` (`resfit/rl_finetuning/off_policy/common_utils/utils.py:79-114`),
the `"normal"` branch with `scale=0.0` does:
```python
nn.init.normal_(layer.weight, mean=0.0, std=scale)   # std=0.0 → exact 0
nn.init.normal_(layer.bias, mean=0.0, std=scale)      # std=0.0 → exact 0
```
(`utils.py:91-94`). This is a `N(0, 0²)` distribution, which collapses to a point mass at 0,
so both weight and bias of the **last** `Linear` layer of `self.policy` are set to exactly
0.0. Since `tanh(0)=0`, `mu=0` at init, so `scaled_mu=0` — the residual actor's mean output is
exactly zero at step 0, matching the intended "start = BC baseline" behavior. Verified
directly in code, not inferred.

---

## #4 Action composition

### Literal composition line

`resfit/rl_finetuning/wrappers/residual_env_wrapper.py:136-139` (`BasePolicyVecEnvWrapper.step()`):
```python
combined_naction = self._last_base_naction + residual_naction        # line 136 — NOT clamped here
env_action = self.action_scaler.unscale(combined_naction)             # line 139 — clamps internally
```
`ActionScaler.unscale()` (`resfit/rl_finetuning/utils/normalization.py:123-141`) does:
```python
scaled_clamped = torch.clamp(scaled_action, -1.0, 1.0)   # line 138
return action_min + (scaled_clamped + 1.0) * (action_max - action_min) / 2.0
```
So the action **actually executed in the environment** is clamped to `[-1,1]` (in normalized
space) before unscaling — but the clamp happens **inside `unscale()`**, not as an explicit
`torch.clamp(base+residual, -1, 1)` line in the wrapper itself.

### DISCREPANCY vs. prior docs — what's stored in the replay buffer is the UNCLAMPED sum

`info["scaled_action"] = combined_naction` (`residual_env_wrapper.py:145`) stores the **raw,
unclamped** `base_naction + residual_naction` sum — not the clamped value used to actually
step the environment. This is what flows into `_add_transitions_to_buffer(actions=combined_action, ...)`
(`train_residual_td3.py:1830-1844`, and identically in the warmup loop at
`train_residual_td3.py:1317-1332`), and ultimately becomes `batch["action"]` fed to
`update_critic()`. `CRITIC_LOSSES_EXPLAINED.md` and `RESIDUAL_RL_TRAINING.md` both state or
imply that the buffer stores `clamp(base+residual, -1, 1)`; **this is not what the current
code does** — see "Discrepancies" section below.

By contrast, wherever the code explicitly recomputes a combined action for
critic/actor-loss purposes, it **does** clamp explicitly:
- Target Q computation: `next_action = torch.clamp(next_obs["observation.base_action"] + next_residual_action, -1.0, 1.0)` (`resfit/rl_finetuning/off_policy/rl/q_agent.py:329`).
- Actor loss: `combined_action = torch.clamp(obs["observation.base_action"] + action_pred, -1.0, 1.0)` (`q_agent.py:443`).
- Eval-time Q-value logging: `q_actions = torch.clamp(obs["observation.base_action"] + actions, -1.0, 1.0)` (`resfit/rl_finetuning/utils/evaluate_dexmg.py:283`).

So the critic's current-action input (`batch["action"]`, from the buffer) is the
**unclamped** raw sum, while its target-action input and the actor's own training target are
**explicitly clamped**. When base+residual doesn't approach ±1 (the typical regime, since
`action_scale` is small: 0.1–0.2), this discrepancy is numerically negligible, but it is a
real inconsistency in the code as written.

### Scalar or per-dimension vector alpha (action_scale)?

Can be **either**. `ActorConfig.action_scale: Any = 1.0` (`rlpd.py:89`), handled via
`to_native_list()` in `actor.py:119-124` and `normalization.py:64-67`, which accepts a
scalar, list, or OmegaConf ListConfig and converts to a `torch.Tensor` (per-dim if a list).

**Actual numeric values used** (verified against the shell scripts, which are what actually
executes — CLI overrides win over `residual_td3.py` dataclass defaults):

| Config default (`residual_td3.py:266-276`) | `train_residual_rl_can.sh` (dense) | `train_residual_rl_can_sparse.sh` | `train_residual_rl_lift.sh` |
|---|---|---|---|
| `action_scale=0.1` | `ACTION_SCALE=0.2` (scalar, line 157) | `ACTION_SCALE="[0.15,0.15,0.15,0.05,0.05,0.05,0.1]"` (**7-D per-dim vector**, line 166) | `ACTION_SCALE=0.1` (scalar, line 148 — matches default) |

The sparse-Can script uses an explicit **per-dimension residual bound**: first 3 action dims
(likely position deltas) capped at ±0.15, next 3 (orientation) at ±0.05, gripper at ±0.1 —
and correspondingly `RANDOM_ACTION_NOISE_SCALE` and `STDDEV_MAX`/`STDDEV_MIN` are also 7-D
vectors matching that per-dim scale (`scripts/train_residual_rl_can_sparse.sh:166,265-267,312`).

### Clip range and where

Combined action is clamped to `[-1, 1]` (normalized action space) — see above. This happens
(a) implicitly for actual env execution, inside `ActionScaler.unscale()`
(`normalization.py:138`), and (b) explicitly wherever the critic/actor loss recomputes the
combined action (`q_agent.py:329,443`; `evaluate_dexmg.py:283`).

### Chunking / re-query points

The residual is queried **every single RL env step** (one `agent.act()` call per
`env.step()`, `train_residual_td3.py:1763,1769`) and applied to whatever base action the base
policy currently offers for that step. The base policy (`ACTPolicy`, ACT-style with action
chunking) internally maintains its own action queue (`select_action()` in
`resfit/lerobot/policies/act/modeling_act.py:155-221`) — it runs one forward pass and queues
`n_action_steps` actions, popping one per call. The residual actor has **no visibility** into
this internal chunk structure — it is queried fresh every env step and its output is added
to whichever single base action the ACT queue currently pops (`residual_env_wrapper.py:101-102,
148-149`), i.e. the residual is **not** applied once-per-chunk, it's applied at every step,
including to base actions that came from a cached (not freshly-inferred) chunk element.

---

## #5 Critic ensemble

**Class**: `Critic(nn.Module)` wrapping `SpatialEmbQEnsemble`, file
`resfit/rl_finetuning/off_policy/rl/critic.py:250-598`.

### Ensemble size

`num_q: int = 10` (`CriticConfig.num_q`, `resfit/rl_finetuning/config/rlpd.py:49`). All three
shell scripts set `NUM_Q_HEADS=10` explicitly (e.g.
`scripts/train_residual_rl_can.sh:431`), matching the default.

### Per-head architecture

Shared trunk (`SpatialEmbQEnsemble`, `critic.py:479-598`):
```
feat[B, num_patch_total, patch_dim], action[B, action_dim], prop[B, prop_dim]
   → concat(feat_transposed, repeated_action, repeated_prop) along feature dim
   → input_proj = Linear(proj_in_dim, emb_dim=1024) → LayerNorm(1024) → ReLU   (critic.py:511-516)
   → weighted-sum pool: z = (learned_weight * y).sum(dim=1)                    (critic.py:517-518,568)
   → z = concat(z, prop, action)                                                (critic.py:570-573)
```
(`emb_dim = CriticConfig.spatial_emb = 1024` default, `rlpd.py:46`; `fuse_patch=1` default,
`rlpd.py:42`, controls whether patches or the action/prop dims are the "pooled" axis —
`critic.py:499-504`.)

Per-head `HeadMLP` (`critic.py:444-476`, instantiated ×`num_q=10` and vmapped for
efficiency via `torch.func.stack_module_state`/`vmap` — `critic.py:520-529,591-597`):
```
Linear(emb_dim+action_dim+prop_dim, hidden_dim=1024) → LayerNorm(1024) → ReLU
Linear(1024, 1024)                                    → LayerNorm(1024) → ReLU
Linear(1024, output_dim)                                                        (critic.py:447-465)
```
`hidden_dim=1024` (`CriticConfig.hidden_dim`, `rlpd.py:43`), `num_layers=2` hidden layers
(`CriticConfig.num_layers`, `rlpd.py:55`). `output_dim=1` for `mse` loss (all 3 shell scripts
use `CRITIC_LOSS_TYPE="mse"`), or `n_bins=51` for `hl_gauss`/`c51` (`critic.py:256-259`).
LayerNorm is placed **after every hidden `Linear`, before `ReLU`**, and **not** after the
final output `Linear` (`critic.py:456-458,463`). `use_layer_norm=True` by default
(`CriticConfig.use_layer_norm`, `rlpd.py:57`), never overridden in the three shell scripts.

The 10 heads are **fully independent** MLPs (separate weights) sharing only the trunk —
implemented via `stack_module_state`+`vmap`+`functional_call` for batched efficiency
(`critic.py:524-529,591-597`), not 10 sequential forward passes.

### Critic forward-call input — exact source

`Critic.forward(feat, prop, act, ...)` (`critic.py:302-326`). Called from
`update_critic()` as `self.critic(obs["feat"], obs["observation.state"], action)`
(`q_agent.py:377,345,351`), where:
- `obs["feat"]` = shared MinViT encoder output (`self._encode(obs, augment=True)`,
  `q_agent.py:788`) — the **same encoder** the actor uses (single `self.encoders` ModuleList,
  `q_agent.py:62,111`), but the critic's own gradients flow into it (encoder optimizer uses
  `critic_lr`, `q_agent.py:111`).
- `action` = `batch["action"]` from the replay buffer — the **composed** (base+residual)
  action, per §4 above (note: unclamped as stored, see the discrepancy noted there), **not**
  the raw residual alone.

### Target critics

**One** `critic_target` per QAgent — `self.critic_target = copy.deepcopy(self.critic)`
(`q_agent.py:88`), containing all 10 target heads (deep-copied ensemble, not 10 separate
target networks constructed independently). Explicitly frozen from autograd's perspective via
`self.critic_target.train(False)` (`q_agent.py:150`) and all target-Q computation happens
inside `torch.no_grad()` (`q_agent.py:312`). Updated only via Polyak averaging, never by a
gradient step / optimizer (no `critic_target_opt` exists).

**Polyak tau**: `critic_target_tau: float = 0.01` (`QAgentConfig` default, `rlpd.py:101`),
overridden to **0.005** in `ResidualTD3DexmgConfig.agent` (`residual_td3.py:270`) and by all
three shell scripts (`CRITIC_TARGET_TAU=0.005`). Applied via
`utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)`
(`q_agent.py:808`, called after **every** `update_critic()` call — i.e. every gradient step,
not just delayed ones):
```python
target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)   # common_utils/utils.py:60
```

### TD target aggregation — min-over-subset (not min-over-all, not mean)

`Critic.q_value()` (`critic.py:328-340`), used for the Bellman target
(`q_agent.py:334`, `self.critic_target.q_value(...)`):
```python
num_heads = min(self.cfg.min_q_heads, q_out.shape[0])          # min_q_heads=2, num_heads=10 → 2
idx = torch.randperm(q_out.shape[0])[:num_heads]                 # random 2-of-10 subset, re-sampled every call
return torch.min(q_out.index_select(0, idx), dim=0).values       # min over that random subset of 2
```
`min_q_heads: int = 2` default (`CriticConfig.min_q_heads`, `rlpd.py:59`), and all three shell
scripts set `MIN_Q_HEADS=2` (matching default) — this is the RED-Q-style "min of a random
2-of-K subset", **not** min over all 10 heads, and **not** a fixed pair (a fresh random pair
is drawn every forward call, both for critic-loss target computation and for the
IBRL/`q_value()`-consumer code paths).

### Actor gradient aggregation — ensemble mean

`Critic.q_value_for_policy()` (`critic.py:342-363`), consumed by the actor loss
(`q = self.critic.q_value_for_policy(obs["feat"], obs["observation.state"], combined_action)`,
`q_agent.py:447`):
```python
if self.cfg.policy_gradient_type == "ensemble_mean":
    return q_out.mean(dim=0)              # critic.py:355 — mean over ALL 10 heads
```
`policy_gradient_type: str = "ensemble_mean"` default (`CriticConfig`, `rlpd.py:53`), and all
three shell scripts set `POLICY_GRADIENT_TYPE="ensemble_mean"` explicitly (matching default).
The code also supports `"min_random_pair"` (min over `min_q_heads` random heads, `critic.py:356-360`)
and `"q1"` (just head 0, standard TD3, `critic.py:361-362`), but neither is used by the
scripts examined. The `-q.mean()` actor loss is at `q_agent.py:448`.

---

## #7 Replay buffers

### Two separate buffers, not one buffer + flag

`online_rb` and `offline_rb`, both `TensorDictPrioritizedReplayBuffer` (TorchRL), created at
`train_residual_td3.py:744-758` (online) and `:848-862` (offline). Both attach an identical
`MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma, use_terminated_for_bootstrap=cfg.algo.terminated_only_bootstrap)`
(`train_residual_td3.py:750-754,854-858`; transform defined in
`resfit/rl_finetuning/utils/rb_transforms.py:13-393`) — n-step returns are applied to **both**
buffers identically.

### Capacities

- `online_rb`: `LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu")`
  (`train_residual_td3.py:745`). `RLPDAlgoConfig.buffer_size: int = 200_000` default
  (`rlpd.py:147`); shell scripts: Can dense/Lift `BUFFER_SIZE=80000`, Can sparse
  `BUFFER_SIZE=100000`.
- `offline_rb`: `max_offline_transitions = estimated_transitions` when
  `offline_fraction>0` (`train_residual_td3.py:840-846`), computed as
  `total_frames_used - num_episodes` (one boundary transition dropped per episode,
  `:821-832`) from the actual HuggingFace dataset metadata (`dataset.meta.episodes[...]["length"]`
  for the first `offline_data.num_episodes` episodes) — **not** a fixed config number; it is
  dynamically sized to fit the whole offline dataset (subset).

### What's preloaded into each

- `offline_rb`: populated once from the offline demonstration dataset by
  `_populate_offline_buffer()` (`train_residual_td3.py:896` onward) — converts every
  consecutive frame pair in the LeRobot dataset into `(s,a,r,s')` transitions. Two modes
  controlled by `offline_data.use_base_policy_for_base_actions` (default `True`,
  `resfit/rl_finetuning/config/residual_td3.py:20`): Mode 2 (default) runs the frozen base
  policy on every dataset observation to populate `observation.base_action`, while the stored
  `action` field remains the dataset's ground-truth action (see code comments at
  `q_agent.py:1555-1573`).
- `online_rb`: **not** preloaded from BC rollouts — it starts empty and is filled by an
  explicit **random-exploration warmup** loop (`train_residual_td3.py:1265-1355`) that runs
  until `len(online_rb) >= cfg.algo.learning_starts` (default 10,000, all 3 shell scripts:
  `LEARNING_STARTS=10000`). During warmup the *residual* actor network is **not** used at all
  — actions are `base_action + uniform_noise` when `use_base_policy_for_warmup=True` (default,
  `train_residual_td3.py:1279-1289`), or pure-uniform-random when `False`.

### Batch size and offline/online split

`algo.batch_size: int = 256` default (`rlpd.py:146`), all 3 scripts:
`BATCH_SIZE=256`. Split via
```python
online_batch_size = int(cfg.algo.batch_size * (1 - cfg.algo.offline_fraction))
offline_batch_size = int(cfg.algo.batch_size * cfg.algo.offline_fraction)
```
(`train_residual_td3.py:737-738`). `offline_fraction: float = 0.5` default (`rlpd.py:159`),
all 3 shell scripts: `OFFLINE_FRACTION=0.5` → 128 online + 128 offline per batch, concatenated
(`torch.cat([online_batch, offline_batch], dim=0)`, `train_residual_td3.py:1984`, and
identically in `_run_critic_warmup` at `:1470`).

### N-step returns

`n: int` = `algo.n_step`. `RLPDAlgoConfig.n_step: int = 3` default (`rlpd.py:165`). All 3 main
shell scripts explicitly set `N_STEP=3` (matching default). Applied identically to **both**
online and offline buffers via the shared `MultiStepTransform` construction (see above) —
same `n_steps` value for both, no separate config per buffer.

### UTD ratio / gradient steps per env step

`algo.num_updates_per_iteration: int = 4` default (`rlpd.py:154`), all 3 shell scripts:
`UTD=4`. `algo.update_every_n_steps: int = 1` (`rlpd.py:156`), all 3 scripts:
`UPDATE_EVERY_N_STEPS=1`. In the main loop (`train_residual_td3.py:1967-2050`), on **every**
env step (`global_step % update_every_n_steps == 0`), a `while i < 4` loop runs **4 critic
gradient updates**, of which **exactly 1** (the last, `i=3`) also updates the actor
(`actor_update_cadence = num_updates_per_iteration // actor_updates_per_iteration = 4//1 = 4`,
`update_actor = (i+1) % actor_update_cadence == 0`, `train_residual_td3.py:1969,1990`). So the
ratio is **4 critic updates : 1 actor update, every single env step** — not "actor updated
every 4th env step" (see Discrepancies section).

---

## #8 Update rules

### Critic loss (MSE, default in all 3 shell scripts — `CRITIC_LOSS_TYPE="mse"`)

```python
q_all = self.critic(obs["feat"], obs["observation.state"], action).squeeze(-1)   # [K=10, B]
td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)                 # [B], mean |TD error| across heads
critic_loss = (td_errors ** 2).mean()                                            # scalar
```
(`q_agent.py:377-388`). This is **not** a per-head MSE averaged across heads — it first
averages the per-head absolute TD error, *then* squares, *then* averages over the batch (heads
are coupled in the loss). With prioritized-replay importance weights (`sampling_strategy`
default `"uniform"` in all 3 scripts, so this branch is inactive):
`weighted_td_errors = td_errors**2 * importance_weights` (`q_agent.py:382-385`).

**Bellman target** (`q_agent.py:312-339`):
```python
next_residual_action = actor_target(next_obs, stddev, clip=stddev_clip)          # target-smoothing noise
next_action = clamp(next_obs["observation.base_action"] + next_residual_action, -1, 1)
target_q_min = critic_target.q_value(next_obs["feat"], next_obs["observation.state"], next_action)  # min-of-2-random-of-10
target_q = reward + discount * target_q_min                                       # discount = gamma^n * nonterminal
```
`discount` (passed in as `effective_discount = batch["gamma"] * batch["nonterminal"]`,
`q_agent.py:786`) already encodes `γⁿ · 𝟙[nonterminal]` from `MultiStepTransform`. Optional:
`clip_q_target_to_reward_range` (default `False`, `QAgentConfig`, `rlpd.py:126`) would clamp
`target_q` to `[0,1]` for sparse rewards — not set by any of the 3 shell scripts, so inactive.

### Residual actor loss

```python
action_pred = actor(obs, stddev=0.0, clip=stddev_clip)     # NOTE: stddev hardcoded to 0.0, see below
action_l2_penalty = cfg.actor.action_l2_reg_weight * mean(sum(action_pred**2, dim=-1))
combined_action = clamp(obs["observation.base_action"] + action_pred, -1, 1)
q = critic.q_value_for_policy(obs["feat"], obs["observation.state"], combined_action)   # ensemble mean, §5
actor_loss_base = -q.mean()
actor_loss_total = actor_loss_base + action_l2_penalty
```
(`q_agent.py:424-452`). **Active coefficient**: `action_l2_reg_weight = 0.0` in all 3 shell
scripts (`ACTION_L2_REG=0.0`, e.g. `scripts/train_residual_rl_lift.sh:640`) → the L2
regularization term is present in code but numerically **inactive** (contributes 0) for all
three main training configs.

Code comment flag (verbatim, `q_agent.py:429-432`):
```python
action_pred: torch.Tensor = self._act_default(
    obs=obs, eval_mode=False,
    # stddev=stddev,
    # NOTE: This fix has not been fully verified yet.
    stddev=0.0,
    clip=self.cfg.stddev_clip, use_target=False,
)
```
So despite the caller (`update_actor`) receiving a `stddev` schedule value, the actor-loss
action is computed with **`std=0` hardcoded**, i.e. the actor loss uses the (near-)
deterministic mean action, not a noisy sample — even though the surrounding sampling
distribution mechanics (`TruncatedNormal`) are otherwise in place.

### BC regularization — inactive in the main scripts

`bc_loss_coef: float = 0.0` (`QAgentConfig`, `rlpd.py:119`), and all three shell scripts set
`BC_LOSS_COEF=0.0`. The BC-regularized path (`update_actor_rft`, `q_agent.py:684-768`, mixing
`actor_loss_total + bc_loss_coef * ratio * bc_loss`) is only reachable if the caller passes a
non-`None` `bc_batch`; the standard `update()` call in the main loop passes `bc_batch=None`
(`train_residual_td3.py:2023`, `agent.update(batch, stddev, update_actor, bc_batch=None,
ref_agent=agent)`), routing to plain `update_actor()` (no BC term) instead. So BC
regularization of the residual actor is **not used** in normal online training — it is a
separate machinery (`update_actor_offline`, TD3-BC style, `q_agent.py:586-682`) used **only**
during the distinct offline-RL pretraining phase, see §9.

### Delayed policy update frequency

Every env step, critic updated 4×, actor updated 1× (last iteration only) — see §7. This is
the TD3 "delayed policy update" (Fujimoto et al.), here expressed as `num_updates_per_iteration
// actor_updates_per_iteration = 4`.

### Target policy smoothing — noise, sigma, clip, where applied

Implemented via `TruncatedNormal.sample(clip=...)` (`common_utils/utils.py:170-182`):
```python
eps = standard_normal(shape) * self.scale        # self.scale = stddev (the schedule value)
if clip is not None: eps = clamp(eps, -clip, clip)
x = self.loc + eps
x = clamp(x, low+eps_const, high-eps_const)        # hard clamp to [-1,1]
```
Applied **only** to the target-actor's action, via `_act_default(..., clip=self.cfg.stddev_clip,
use_target=True)` inside `update_critic()`'s target computation (`q_agent.py:318-324`),
**never** to the main actor's action selection at act-time or actor-loss time (those pass
`clip=None` at act-time, `q_agent.py:266`; and `clip=self.cfg.stddev_clip` at actor-loss time
but with `stddev=0` so no noise is actually sampled, `q_agent.py:427-434`).
- **sigma** (`stddev`): schedule value at the current `global_step`, from
  `algo.stddev_schedule = "linear(stddev_max, stddev_min, stddev_step)"`
  (`rlpd.py:222-229`). All 3 main shell scripts use constant schedules
  (`STDDEV_MAX==STDDEV_MIN`): Can dense/Lift = `0.025`/`0.05` resp.; Can sparse =
  a 7-D vector `[0.015,0.015,0.015,0.003,0.003,0.003,0.0125]`.
- **clip range**: `stddev_clip` = `agent.stddev_clip`, `QAgentConfig` default `0.3`
  (`rlpd.py:102`); all 3 shell scripts: `STDDEV_CLIP=0.3` (matching default; the sparse script
  uses `${STDDEV_CLIP:-0.3}`, unset → falls back to 0.3).
- Gated by `target_action_noise: bool = True` default (`QAgentConfig`, `rlpd.py:129`); all 3
  shell scripts: `TARGET_ACTION_NOISE="true"`. When `True`,
  `eval_mode = not target_action_noise = False` → the noisy `dist.sample(clip=stddev_clip)`
  branch executes (`q_agent.py:295-298,320`); when `False`, `eval_mode=True` uses the
  deterministic `dist.mean`.

### Optimizer + LR per network

`torch.optim.AdamW` for all three (`q_agent.py:111-113`):
- `encoder_opt`: `AdamW(self.encoders.parameters(), lr=cfg.critic_lr)` — encoder shares the
  **critic's** LR, no independent LR.
- `critic_opt`: `AdamW(self.critic.parameters(), lr=cfg.critic_lr)`.
- `actor_opt`: `AdamW(self.actor.parameters(), lr=cfg.actor_lr)`.

Config defaults `QAgentConfig`: `actor_lr=1e-4, critic_lr=1e-4` (`rlpd.py:99-100`), overridden
in `ResidualTD3DexmgConfig.agent`: `actor_lr=1e-6` (`residual_td3.py:268`), `critic_lr=1e-4`
(`:269`). All 3 shell scripts confirm: `ACTOR_LR=1e-6`, `CRITIC_LR=1e-4` — **100× LR gap**
between actor and critic/encoder. Gradient clipping:
`critic_grad_clip_norm=1.0`, `actor_grad_clip_norm=1.0` (both `QAgentConfig` defaults and all
3 shell scripts, `q_agent.py:412-413,577`).

---

## #9 Offline pre-training stage

**This is separate from, and mutually exclusive with, the "critic warmup" phase** — verified
in `train_residual_td3.py:1719`: `if cfg.algo.critic_warmup_steps > 0 and not _is_resuming and
not _did_offline_rl:` — the plain critic-only warmup is **skipped entirely** if
`train_offline_rl=True` and `offline_rl_steps>0`.

### Order of phases (from code, `train_residual_td3.py`)

1. Populate `offline_rb` from dataset (`:896` onward).
2. Random-exploration warmup fills `online_rb` to `learning_starts` transitions
   (`:1265-1355`) — **neither** the residual actor **nor** the offline-RL phase run yet.
3. **If `algo.train_offline_rl=True`** (optional, off by default —
   `ResidualTD3AlgoConfig.train_offline_rl: bool = False`, `residual_td3.py:193`): run
   `algo.offline_rl_steps` (default 50,000, `residual_td3.py:194`) TD3-BC-style gradient
   steps using `EpochUnifiedDataset` sampling from both buffers combined
   (`train_residual_td3.py:1514-1691`, class in
   `resfit/rl_finetuning/off_policy/rl/unified_buffer.py`).
4. **Else / after**: if `critic_warmup_steps>0` and offline-RL was NOT run, run that many
   critic-only updates (`update_actor=False`) before starting online interaction
   (`:1719-1731`).
5. Main online training loop.

### Which networks are pretrained offline, on what data, how many steps

During the offline-RL phase (`:1541-1665`):
- **Critic** (+ shared encoder) updated **every** step via the normal `agent.update(...,
  update_actor=False, ...)` call (`:1549`) — same critic loss as online (§8).
- **Actor** updated only every `policy_delay=2` steps (`:1539,1552`) via
  `agent.update_actor_offline(obs, target_action, cfg.algo.offline_rl_bc_alpha)`
  (`:1575`, method defined `q_agent.py:586-682`) — a **TD3-BC** loss:
  ```python
  lmbda = alpha / q_target.abs().mean().clamp(min=1e-5)              # q_agent.py:490
  actor_loss_q = -(lmbda * q).mean()                                  # q_agent.py:493
  bc_loss = F.mse_loss(action_pred, target_residual)                  # q_agent.py:496, target_residual = GT_action - base_action
  actor_loss_total = actor_loss_q + bc_loss + action_l2_penalty       # q_agent.py:499
  ```
  `alpha = cfg.algo.offline_rl_bc_alpha`, default `0.2` (`residual_td3.py:195`, code comment:
  "paper suggested value is 2.5 (but the bc loss is based on full action range not residual)").
- Data: `EpochUnifiedDataset(td_offline, td_online, sampling_method, offline_ratio)`
  (`train_residual_td3.py:1532-1536`) — draws from **both** the populated offline dataset
  buffer and whatever's in the online buffer after warmup (i.e., BC-demo transitions AND the
  small amount of random-exploration rollout data collected in step 2 above), NOT purely
  offline demo data. `offline_rl_sampling_method` default `"fixed_ratio"`,
  `offline_rl_offline_ratio` default `0.5` (`residual_td3.py:199-200`).
- **Steps**: `offline_rl_steps` — set to `25000` in `sparse_with25kOffRL.sh`, `200000` in
  `sparse_just200kOffRL.sh` (see ablation-script diff below), `75000` in the main
  `train_residual_rl_can_sparse.sh` (`OFFLINE_RL_STEPS=75000`, line 709).

### Which checkpoint transfers into online training, and why

After the offline-RL loop finishes, code explicitly reloads a checkpoint into the live `agent`
object before online training starts (`train_residual_td3.py:1671-1689`):
```python
load_choice = cfg.algo.offline_rl_load_ckpt   # default "latest"
if load_choice == "best":
    ckpt_to_load = model_save_dir / "offline_best" / "checkpoint.pt"
elif load_choice == "latest":
    pass  # agent in memory already holds the final (last-step) weights
else:
    ckpt_to_load = model_save_dir / f"offline_step_{load_choice}" / "checkpoint.pt"
if ckpt_to_load and ckpt_to_load.exists():
    agent.load_state_dict(torch.load(ckpt_to_load)["agent_state_dict"])
```
`offline_rl_load_ckpt: str = "latest"` default (`residual_td3.py:198`). **Both actor and
critic (and target networks, and encoder) transfer together** — `agent.load_state_dict` loads
the whole `QAgent` `nn.Module` state dict, there is no separate actor-only or critic-only
transfer path. The "why": `best` picks the checkpoint with the highest offline-eval success
rate seen during the offline phase (tracked via `best_eval_success_rate`,
`:1621-1641`), guarding against late-stage TD3-BC overfitting; `latest` simply continues with
whatever the offline phase ended on.

### CanAblationStudies — flags that actually differ between scripts

Verified via `diff` on the `^[A-Z_]*=` variable assignments (all scripts otherwise identical
templates):

- **`sparse_criticWarmup.sh` vs `sparse_noCriticWarmup.sh`**: only difference is
  `CRITIC_WARMUP=10000` vs `CRITIC_WARMUP=0` (and the auto-generated `WANDB_NAME`). Neither
  sets `TRAIN_OFFLINE_RL` (both leave the `ResidualTD3AlgoConfig` default `train_offline_rl=False`
  — offline-RL phase never runs in this pair; this ablation isolates the plain critic-only
  warmup phase, §9 step 4).
- **`sparse_with25kOffRL.sh` vs `sparse_just200kOffRL.sh`**: `sparse_just200kOffRL.sh` sets
  `TOTAL_TIMESTEPS=0` (i.e. **no online RL phase at all** — the "main training loop" runs
  zero iterations) and `OFFLINE_RL_STEPS=200000` (vs. `25000` and `TOTAL_TIMESTEPS=200000` in
  the `with25kOffRL` variant). This ablation isolates **pure offline TD3-BC pretraining with
  no online fine-tuning** against a hybrid 25k-offline + 200k-online run.
- The `ablation_checklist.md` (`scripts/CanAblationStudies/ablation_checklist.md`) documents
  a 4-round sweep: Round 0 = critic-warmup on/off × BC-quality (latest vs "AvgBC");
  Round 1 = offline-RL sampling method (`fixed_ratio` @ 0.25/0.5/0.75 vs `proportional`);
  Round 2 = checkpoint choice (`best` vs `latest`) × offline steps (25k/50k/75k) × BC quality;
  Round 3 = scalar vs per-dimension `action_scale` ("listedResScale"); Round 4 = 5-seed
  confirmation of the winning config. (Read directly, not run.)

---

## #10 Gradient flow — TRAINED vs FROZEN per module

| Module | Status | Evidence (file:line) |
|---|---|---|
| **Base policy** (`ACTPolicy`, frozen BC) | **FROZEN** | Set to `.eval()` at load (`train_residual_td3.py:329,332`); every call site wraps `select_action()` in `torch.no_grad()` (`residual_env_wrapper.py:101-102,148-149`); **not** registered in any optimizer's `param_groups` — no `base_policy.parameters()` reference anywhere in `q_agent.py`'s three `torch.optim.AdamW(...)` calls (`q_agent.py:111-113`). No explicit `requires_grad_(False)` call was found (`grep` of `resfit/lerobot/utils/load_policy.py` — freezing is achieved purely by omission from the optimizer graph + `no_grad()` context, not by disabling gradients on the parameters themselves.) |
| **Residual actor** (`self.actor`) | **TRAINED** | `self.actor_opt = AdamW(self.actor.parameters(), lr=cfg.actor_lr)` (`q_agent.py:113`); `self.train(True)` at init calls `self.actor.train(training)` (`q_agent.py:151,183`); updated in `update_actor()`/`update_actor_offline()`/`update_actor_rft()` (`q_agent.py:525-584,586-682,684-768`). |
| **Vision encoders** (`self.encoders`, shared by actor+critic) | **TRAINED** (by critic loss only) | `self.encoder_opt = AdamW(self.encoders.parameters(), lr=cfg.critic_lr)` (`q_agent.py:111`), stepped inside `update_critic()` (`q_agent.py:406,419`). Explicitly **detached** before the actor loss: `obs["feat"] = obs["feat"].detach()` (`q_agent.py:815`, in `update()` right before `update_actor`/`update_actor_rft` is called) — so gradient from the actor loss never reaches the encoder. Optional full freeze via `cfg.freeze_encoder` (`QAgentConfig.freeze_encoder: bool = False` default, `rlpd.py:124`) sets `param.requires_grad=False` for all encoder params (`q_agent.py:105-108`) — **not** used by any of the 3 main shell scripts (`FREEZE_ENCODER="false"` in all three). |
| **Critic ensemble** (`self.critic`, all 10 heads) | **TRAINED** | `self.critic_opt = AdamW(self.critic.parameters(), lr=cfg.critic_lr)` (`q_agent.py:112`), stepped in `update_critic()` (`q_agent.py:407,420`). |
| **Target critic** (`self.critic_target`) | **FROZEN (no gradients), updated only via Polyak** | `self.critic_target.train(False)` (`q_agent.py:150`); no optimizer references it; all its forward calls are inside `torch.no_grad()` (`q_agent.py:312`); updated via `utils.soft_update_params(self.critic, self.critic_target, tau)` (`q_agent.py:808`), a plain in-place `.data.copy_()` — not an autograd op (`common_utils/utils.py:58-60`). |
| **Target actor** (`self.actor_target`) | **FROZEN (no gradients), updated only via Polyak, but left in `.train(True)` mode** | No optimizer references it; `assert self.actor_target.training` (`q_agent.py:314`) confirms it is **not** put in eval mode (unlike `critic_target`) — `QAgent.train()` (`q_agent.py:180-188`) never touches `actor_target`, and `copy.deepcopy(self.actor)` (`q_agent.py:89`) inherits the default `training=True` state, which is never subsequently changed. This means the target actor's `Dropout` layers are active (though `dropout=0` in all 3 shell scripts' implicit `ActorConfig.dropout` default, so numerically a no-op). Updated via `utils.soft_update_params(self.actor, self.actor_target, tau)` (`q_agent.py:823`). |

---

## #11 Control-flow pseudocode — online training loop

Reimplementation-level pseudocode of the loop in `train_residual_td3.py`, annotated with
source lines. (Phases before the `while` loop — offline buffer population, warmup, optional
offline-RL phase, optional critic-warmup — are covered in §9; this focuses on the steady-state
online loop, `train_residual_td3.py:1755` onward.)

```python
# train_residual_td3.py:1755
while global_step <= cfg.algo.total_timesteps:

    # ---- (1) Residual-action query ----------------------------------------------------
    with torch.no_grad(), utils.eval_mode(agent):                        # :1761
        stddev = utils.schedule(cfg.algo.stddev_schedule, global_step)   # :1762  (schedule → constant 0.025/0.05/vector per script)
        action = agent.act(obs, eval_mode=False, stddev=stddev, cpu=False)  # :1763
        #   -> QAgent.act() (q_agent.py:251-276): obs["feat"] = self._encode(obs, augment=False)  (q_agent.py:260)
        #   -> self._act_default(eval_mode=False, clip=None, use_target=False) (q_agent.py:262-268)
        #      -> Actor.forward(obs, std=stddev) -> TruncatedNormal(scaled_mu, std) (actor.py:159-189)
        #      -> dist.sample(clip=None) (q_agent.py:298, common_utils/utils.py:170-182)  <- residual action, NOT combined

    if cfg.algo.progressive_clipping_steps > 0:                          # :1765-1767 (=0 in all 3 scripts -> no-op)
        action = action * min(1.0, global_step / progressive_clipping_steps)

    # ---- (2) Environment step: base-policy query + composition happens INSIDE env.step() -
    next_obs, reward, terminated, truncated, info = env.step(action)     # :1769
    #   env == BasePolicyVecEnvWrapper (residual_env_wrapper.py:23)
    #   -> combined_naction = self._last_base_naction + residual_naction        (residual_env_wrapper.py:136)  [UNCLAMPED, see §4]
    #   -> env_action = self.action_scaler.unscale(combined_naction)            (residual_env_wrapper.py:139)  [clamps to [-1,1] internally]
    #   -> raw_obs, reward, terminated, truncated, info = self.vec_env.step(env_action)  (residual_env_wrapper.py:142)
    #      (vec_env = VectorizedEnvWrapper -> gym AsyncVectorEnv/SyncVectorEnv -> RobosuiteGymWrapper.step(), dexmg.py:367-401)
    #   -> info["scaled_action"] = combined_naction                             (residual_env_wrapper.py:145)  [stored for buffer]
    #   -> base_action_next = self.base_policy.select_action(raw_obs)   [torch.no_grad(), frozen ACT]  (residual_env_wrapper.py:148-149)
    #   -> if terminated.any(): self.base_policy.reset(env_ids=...)             (residual_env_wrapper.py:154-156)
    #   -> augmented_obs = self._augment_obs(raw_obs, base_naction_next)        (residual_env_wrapper.py:159, :170-178)
    #      -> augmented_obs["observation.base_action"] = base_naction_next
    #      -> augmented_obs["observation.state"] = state_standardizer.standardize(state)
    done = terminated | truncated                                        # :1770

    env.render_viewer()   # no-op if headless=True                       # :1772-1774

    # ---- (episode-end bookkeeping, wandb episode_return log) ----------  :1801-1826

    # ---- (3) Replay-buffer write --------------------------------------------------------
    combined_action = info["scaled_action"]                              # :1830   [the UNCLAMPED base+residual sum]
    _add_transitions_to_buffer(obs, next_obs, actions=combined_action,   # :1831-1844
        reward=reward, done=done, info=info, online_rb=online_rb, terminated=terminated, ...)
        # -> builds a TensorDict{"obs":curr_obs, "next":{"obs":next_obs,"done","terminated","reward"},
        #                        "action":combined_action, "_priority":10.0}   (train_residual_td3.py:204-220)
        # -> online_rb.add(td)   (train_residual_td3.py:222)
        #    -> MultiStepTransform._inv_call() intercepts, buffers last n_step transitions,
        #       and only WRITES the oldest one once n_step frames have accumulated, with
        #       next.obs/reward/gamma/nonterminal all rewritten for the n-step horizon
        #       (rb_transforms.py:193-240, 243-393)
    obs = next_obs                                                       # :1846

    # ---- (periodic eval, checkpoint save) ----------------------------  :1852-1962 (every eval_interval_every_steps / save_freq)

    global_step += cfg.num_envs                                          # :1903

    # ---- (4) Gradient updates: 4 critic + 1 actor, every env step -----------------------
    if global_step % cfg.algo.update_every_n_steps == 0 or global_step == cfg.num_envs:  # :1967
        i = 0
        actor_update_cadence = cfg.algo.num_updates_per_iteration // cfg.algo.actor_updates_per_iteration  # = 4  :1969
        while i < cfg.algo.num_updates_per_iteration:                     # 4 iterations   :1971
            online_batch = online_rb.sample(online_batch_size)            # :1977   (128)
            offline_batch = offline_rb.sample(offline_batch_size)         # :1982   (128, since offline_fraction=0.5>0)
            batch = torch.cat([online_batch, offline_batch], dim=0)       # :1984   (256 total)

            update_actor = (i + 1) % actor_update_cadence == 0            # :1990   True only on i=3

            metrics = agent.update(batch, stddev, update_actor,           # :2023
                                    bc_batch=None, ref_agent=agent)
            #   QAgent.update() (q_agent.py:770-826):
            #     obs["feat"] = self._encode(obs, augment=True)                          (q_agent.py:788)  [RandomShiftsAug]
            #     next_obs["feat"] = self._encode(next_obs, augment=True)  [no_grad]     (q_agent.py:790-791)
            #     effective_discount = batch["gamma"] * batch["nonterminal"]             (q_agent.py:786)
            #     critic_metric = self.update_critic(obs, action, reward, effective_discount, next_obs, stddev, ...)  (q_agent.py:799-807)
            #       -> target: next_residual = actor_target(next_obs, stddev, clip=stddev_clip) [smoothing noise]  (q_agent.py:318-324)
            #       -> next_action = clamp(next_obs["base_action"] + next_residual, -1, 1)                          (q_agent.py:329)
            #       -> target_q = reward + effective_discount * critic_target.q_value(...)  [min-of-2-of-10]        (q_agent.py:334-336)
            #       -> critic_loss = MSE-of-mean-abs-TD-error (q_agent.py:377-388); backward(); clip_grad_norm_(1.0); encoder_opt.step(); critic_opt.step()  (q_agent.py:406-420)
            #     soft_update_params(critic, critic_target, tau=0.005)                    (q_agent.py:808)
            #     if update_actor:
            #       obs["feat"] = obs["feat"].detach()                                    (q_agent.py:815)  [no encoder grad from actor]
            #       actor_metric = self.update_actor(obs, stddev)                         (q_agent.py:818)
            #         -> action_pred = actor(obs, std=0.0) [hardcoded, not the stddev arg]  (q_agent.py:427-434)
            #         -> combined_action = clamp(obs["base_action"] + action_pred, -1, 1)  (q_agent.py:443)
            #         -> q = critic.q_value_for_policy(...) [ensemble MEAN over 10 heads]  (q_agent.py:447, critic.py:353-355)
            #         -> actor_loss = -q.mean() + action_l2_reg_weight * ||action_pred||^2 (q_agent.py:448-450, =0 in all 3 scripts)
            #         -> backward(); clip_grad_norm_(1.0); actor_opt.step()               (q_agent.py:573-582)
            #       soft_update_params(actor, actor_target, tau=0.005)                    (q_agent.py:823)
            i += 1
```

---

## Discrepancies with existing docs

1. **Combined action stored in the replay buffer is UNCLAMPED, contrary to `RESIDUAL_LEARNING.md`
   and `CRITIC_LOSSES_EXPLAINED.md`'s implicit assumption.** `RESIDUAL_LEARNING.md` (§3,
   "Clamping to [-1,1]") states the composed action `a_t = clip(a_base + a_res, -1, 1)` is
   what's used throughout, and `ORIGINAL_VS_IBRL_QAGENT.md` §2 similarly claims
   `info["scaled_action"] = combined_naction # ← clamp(base + residual, -1, 1)` (paraphrased,
   with a clamp implied). The **actual code** at `residual_env_wrapper.py:136-145` computes
   `combined_naction = base + residual` with **no clamp**, and it is this unclamped value that
   is stored via `info["scaled_action"]` and lands in `batch["action"]` for the critic-loss's
   current-action term. The clamp only happens inside `ActionScaler.unscale()`
   (`normalization.py:138`) for the value actually sent to the simulator, and explicitly at
   3 other call sites (target-Q computation, actor loss, eval Q-logging) — but **not** for the
   buffer-stored `action` field. This is numerically minor when `action_scale` is small
   (0.1–0.2, keeping sums well inside `[-1,1]` in the typical case) but is a genuine code-level
   inconsistency worth flagging precisely as such rather than assuming a clamp exists.

2. **Actor gets updated every env step, not "every 4th step."** `RESIDUAL_RL_TRAINING.md`
   §12 ("Step (4): Gradient Updates (UTD=4)") and §2's hyperparameter table both phrase this
   as "actor updated every 4th step," which reads as "once every 4 env steps." The actual
   behavior (`train_residual_td3.py:1967-1990`) is: **on every single env step**, 4 critic
   updates run, and exactly 1 of those 4 also updates the actor — so the actor is updated
   once per env step too, just at 1/4 the frequency *of gradient calls* relative to the
   critic, not at 1/4 the frequency *of env steps*.

3. **`RESIDUAL_LEARNING.md` §5 states `agent.actor.action_scale=0.2` is "the command-line
   arg"** used generically; in fact the config **default** is `0.1`
   (`ResidualTD3DexmgConfig.agent.actor.action_scale`, `residual_td3.py:272`), and the actual
   value differs per script: `0.2` (Can dense), a **7-D per-dimension vector**
   `[0.15,0.15,0.15,0.05,0.05,0.05,0.1]` (Can sparse), and `0.1` (Lift, matching the config
   default). None of the docs mention that `action_scale` can be (and, in the sparse-Can
   ablation config, actually is) a per-dimension vector rather than a scalar.

4. **`TD3_ALGORITHM.md`/`RESIDUAL_RL_TRAINING.md`'s "critic warmup" description doesn't
   mention it is mutually exclusive with the offline-RL (TD3-BC) phase.** Per
   `train_residual_td3.py:1719` (`if cfg.algo.critic_warmup_steps > 0 and not _is_resuming and
   not _did_offline_rl`), if `algo.train_offline_rl=True` (as in the sparse-Can main script and
   several `CanAblationStudies` scripts), the plain critic-only warmup phase is **skipped
   entirely**, regardless of `critic_warmup_steps`'s configured value. The two "warm the
   critic up before the actor listens to it" mechanisms are alternatives, not both applied.

5. **`ORIGINAL_VS_IBRL_QAGENT.md`'s comparison of `QAgent` vs `QAgent_ibrl` is accurate for
   the `_act_default`/critic-training claims checked here** (verified against
   `q_agent.py:278-300,824` and `critic.py`) — no discrepancy found in that doc for the
   portions cross-checked against current `q_agent.py`/`critic.py`, though it was not the
   primary target of this pass and `ibrl_q_agent.py` itself was not re-read in full.

6. **Actor-loss `stddev` is silently hardcoded to `0.0`** (`q_agent.py:429-432`, with an
   inline `# NOTE: This fix has not been fully verified yet.` comment) rather than using the
   schedule value passed into `update_actor(obs, stddev)`. None of the reviewed prior docs
   mention this; `RESIDUAL_RL_TRAINING.md`'s pseudocode for "Actor Update" implies the actor
   loss action uses the same stochastic sampling as elsewhere, omitting this hardcoded
   override.

7. **Target actor network is left in `.train(True)` mode** while the target critic is
   explicitly set to `.train(False)` (`q_agent.py:150` vs. no equivalent call for
   `actor_target`, confirmed by the `assert self.actor_target.training` at `q_agent.py:314`).
   None of the reviewed docs mention this asymmetry; it means the target actor's `Dropout`
   layers (inert only because `dropout=0` in all three shell scripts) are technically live
   during target-action sampling, unlike the target critic.

## Not investigated / out of scope for this pass

- `resfit/rl_finetuning/off_policy/rl/ibrl_q_agent.py` was read only via the pre-existing
  `ORIGINAL_VS_IBRL_QAGENT.md` doc, not independently re-verified line-by-line against current
  code in this pass (flagged in item 5 above).
- `resfit/rl_finetuning/off_policy/rl/unified_buffer.py` (`EpochUnifiedDataset`, used only in
  the offline-RL phase, §9) was referenced by grep/usage-site only, not read in full — its
  exact `"fixed_ratio"` vs `"proportional"` sampling implementation was not verified line by
  line.
- `resfit/lerobot/policies/diffusion/modeling_diffusion.py`, `resfit/lerobot/utils/load_policy.py`
  (beyond a targeted `grep` for `requires_grad`/`eval()`), and `temp.py` were not read — they
  are in the modified-files list from `git status` but outside the scope of questions #2–#11
  as posed.
