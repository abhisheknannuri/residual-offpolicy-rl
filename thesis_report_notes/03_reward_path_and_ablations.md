# #6 Reward Path & #9 Ablations — Verified Findings

Repo: `residual-offpolicy-rl` (ResFiT). All line numbers verified against the
working tree as of 2026-08-23 (branch `dev/abhi-resfit-setup`). Every claim
below was re-checked directly against current code — this is not a
reproduction of the prior docs, though it agrees with
`../docs/rewards/STAGE_AWARE_REWARD_PIPELINE.md` almost everywhere (see "Discrepancies"
section for the few places checked separately).

---

## #6 Reward path

### 6.1 Which frames go to the reward server, at what resolution, what cadence

- Only the **training env** gets the reward-model wrapper — never eval envs.
  This is set up in the closure `get_envs()`: the wrapper is only attached
  when `apply_reward_model=True`, which is passed `True` only for the
  training-env branch. `resfit/rl_finetuning/scripts/train_residual_td3.py:413,499`
- The frame source is `obs[image_key]`, default
  `image_key: str = "observation.images.agentview"`
  (`resfit/rl_finetuning/config/residual_td3.py:113`, and default of the
  `StageAwarePBRSWrapper.__init__` parameter,
  `resfit/rl_finetuning/reward_models/pbrs_wrapper.py:62`). Neither
  `train_residual_rl_can_sarm_stage_pbrs.sh` nor `_milestone.sh` overrides
  `reward_model.image_key`, so the **agentview** camera is what's sent (not
  the wrist camera `robot0_eye_in_hand`, which also exists in the obs dict per
  `resfit/dexmg/environments/dexmg.py` camera config).
- Resolution: **84×84**, hardcoded default in
  `resfit/dexmg/environments/dexmg.py:90` (`camera_size: int = 84`), fed into
  `robosuite.make(camera_heights=self.camera_size, camera_widths=self.camera_size, ...)`
  at `dexmg.py:196-197`. Neither `train_residual_rl_can_sarm_stage_pbrs.sh` nor
  `_milestone.sh` passes a `camera_size` / `CAMERA_SIZE` override (grepped,
  no match), so the effective resolution sent to the reward server is 84×84.
- Frame format sent over the wire: `_extract()` takes `obs[image_key]` as
  `[C,H,W]` float in `[0,1]`, transposes to `[H,W,C]`, scales by 255 and casts
  to `uint8` — `pbrs_wrapper.py:112-120`.
- **Cadence**: a server call fires when
  `step_ctr % query_every_k == 0 OR truncated OR (terminated AND reward_mode == "milestone")`
  — `pbrs_wrapper.py:203-208`. `query_every_k` is `4` in both
  `train_residual_rl_can_sarm_stage_pbrs.sh:676` and
  `..._milestone.sh:183` (config default is `5`,
  `resfit/rl_finetuning/config/residual_td3.py:106` — **the shell-script
  override of `4` is what actually runs**). At 20 Hz sim control, `k=4` ⇒
  reward-server query rate ≈ **5 Hz**. Between queries the wrapper reuses the
  last cached `gated_stage`; it does not call the server every env step.
- Every env step (regardless of whether a server call fires) 1 frame + 1
  state vector is appended to `self._pending_frames` /
  `self._pending_states` (`pbrs_wrapper.py:194-196`); on a query these
  accumulated (≤`k`) frames are sent as one batched request
  (`np.stack(...)`, `pbrs_wrapper.py:125-126`), then cleared.
- On `reset()`, exactly 1 frame is queued and force-queried
  (`reset=True`) even though only 1 pending frame exists — `pbrs_wrapper.py:182-185`.

### 6.2 What the server returns — exact response schema as parsed by the client

`resfit/rl_finetuning/reward_models/reward_client.py:93-154`
(`RewardModelClient.predict_stage`) posts to `{server_url}/predict_stage`
with:
- `files`: `frames` (`.npy`, uint8 `[N,H,W,3]`), `states` (`.npy`, float32
  `[N,state_dim]`) — `reward_client.py:122-125`.
- `data`: `session_id`, `task`, `num_anchors`, `reset` (`"1"`/`"0"`),
  `hysteresis_k`, `conf_threshold` (formatted `f"{v:.6f}"`), `monotonic`
  (`"1"`/`"0"`) — `reward_client.py:126-134`.

The client reads the JSON response and uses these fields
(`pbrs_wrapper.py:151-164`):
- `out["gated_stage"]` — **required**, `int(...)` — raises `KeyError` if
  absent, no `.get()` fallback.
- `out.get("raw_stages", None)` — optional, list; `raw_stages[-1]` used as
  debug-only "pre-hysteresis" stage.
- `out.get("stage_conf", self._stage_conf)` — optional float, keeps last
  value if absent.
- `out.get("server_latency_s", 0.0)` — optional float.
- Client then adds its own `client_latency_s` (measured wall-clock POST
  round-trip) before returning — `reward_client.py:143,153`.
- On HTTP status ≥ 400 the client raises `RuntimeError` using the response's
  `"error"` JSON field or raw text — `reward_client.py:145-150`.

The module docstring in `reward_client.py:13-31` additionally documents
`raw_confs` and `taus` as fields the server *may* return, but the client code
never reads either — confirmed by grep, no `raw_confs` / `taus` occurrence
outside that docstring.

`GET /health` — client only checks `str(info.get("status","")).lower() == "ok"`
(`reward_client.py:73-79`); other fields the docstring mentions
(`model_type`, `num_stages`) are not read anywhere in code.

`POST /reset` — client sends `{"session_id": ...}`, only checks
`raise_for_status()`; return body is ignored (`reward_client.py:81-88`).

**Server-side logic (how `gated_stage`/hysteresis are actually computed) is
not present in this repo** — no file matching `predict_stage`/`gated_stage`
implementation was found anywhere under `resfit/` or elsewhere in the
repository (searched: `grep -rln "predict_stage\|gated_stage\|def health"` —
only hits were the client (`reward_client.py`), the wrapper
(`pbrs_wrapper.py`), the offline labeler (`scripts/label_offline_pbrs.py`),
the rollout recorder, and unrelated real-hardware camera/policy files). The
server is an external process, out of this repo's scope.

### 6.3 EXACT shaped-reward formula, both modes, with active numeric values

Both formulas are computed in `StageAwarePBRSWrapper.step()`,
`resfit/rl_finetuning/reward_models/pbrs_wrapper.py:190-245`. There is a
single wrapper class, mode-switched by `self.reward_mode` (`"pbrs"` |
`"milestone"`) — no separate `milestone_wrapper.py` exists anywhere in the
repo (confirmed by `find` — only `pbrs_wrapper.py` under
`reward_models/`).

**PBRS** (`pbrs_wrapper.py:224-231`, literal):
```python
if bool(terminated):
    phi_next = 0.0
else:
    phi_next = self.potentials[self._current_stage]
r_shaped = (r_sparse if self.keep_sparse_term else 0.0) + self.gamma * phi_next - self._phi_current
self._phi_current = phi_next
r_out = float(r_shaped)
```
i.e. `r = (r_sparse if keep_sparse_term else 0) + γ·φ(s') − φ(s)`, with
`φ(terminal) = 0`.

- `keep_sparse_term`: config default `True`
  (`resfit/rl_finetuning/config/residual_td3.py:128`); both
  `train_residual_rl_can_sarm_stage_pbrs.sh:690` and `_milestone.sh` (unused
  there) set `REWARD_KEEP_SPARSE_TERM="true"` — **active value: `True`**.
- `γ` = `cfg.reward_model.pbrs_gamma` if set, else falls back to
  `cfg.algo.gamma` (`train_residual_td3.py:499-501`). Config default of
  `pbrs_gamma` is `None` (`residual_td3.py:125`); neither `_pbrs.sh` nor
  `_milestone.sh` sets `pbrs_gamma` (grepped, no match) — **active value:
  falls through to `algo.gamma`**. `algo.gamma` is set to `0.995` in both
  scripts (`train_residual_rl_can_sarm_stage_pbrs.sh` header defines
  `GAMMA=0.995`, forwarded as `algo.gamma="${GAMMA}"`) — **active `γ = 0.995`**.
- `potentials` (the `φ` lookup table): config default
  `(0.0, 0.05, 0.10, 0.15)` (`residual_td3.py:123`); both scripts set
  `REWARD_POTENTIALS="[0.0,0.05,0.1,0.15]"`
  (`train_residual_rl_can_sarm_stage_pbrs.sh:679`) — same values, so
  **active `φ = [0.0, 0.05, 0.10, 0.15]`** (4 stages, `num_stages=len(potentials)=4`,
  `pbrs_wrapper.py:78`).

**Milestone** (`pbrs_wrapper.py:210-221`, literal):
```python
r_out = 0.0
if self._current_stage > self._last_paid_stage:
    for _s in range(self._last_paid_stage + 1, self._current_stage + 1):
        r_out += self.milestone_payouts[_s]
    self._last_paid_stage = self._current_stage
if bool(terminated) and r_sparse > 0.5:
    r_out += self.milestone_success_bonus
r_out = float(r_out)
r_shaped = r_out
```
i.e. `r = Σ_{s=last_paid+1..stage} payouts[s]` (paid once per newly-reached
stage, ratchet — `self._last_paid_stage` never decreases), **plus**
`success_bonus` iff `terminated AND r_sparse > 0.5` (gated on the *true*
simulator sparse-success flag, not on model confidence).

- `milestone_payouts`: config default `(0.0, 0.1, 0.1, 0.1)`
  (`residual_td3.py:139`, i.e. `Any = ...` field
  `MilestonePayouts` around `residual_td3.py:138-139`); `_milestone.sh:198`
  overrides to `MILESTONE_PAYOUTS="[0.0,0.3,0.3,0.4]"` — **active value:
  `[0.0, 0.3, 0.3, 0.4]`**.
- `milestone_success_bonus`: config default `0.7`; `_milestone.sh:199`
  overrides to `MILESTONE_SUCCESS_BONUS="1.0"` — **active value: `1.0`**.
- `γ` (gamma) is **not used** in the milestone formula at all — confirmed, no
  `self.gamma` reference in the `if self.reward_mode == "milestone":` branch
  (`pbrs_wrapper.py:210-222`).
- Max theoretical per-episode milestone return with active values:
  `0.3 + 0.3 + 0.4 + 1.0 = 2.0` (matches a comment in
  `train_residual_rl_can_sarm_stage_milestone.sh:194`).

**pbrs vs milestone — which script is which entry point**: PBRS by
`scripts/train_residual_rl_can_sarm_stage_pbrs.sh` (reward_mode defaults to
`"pbrs"`, never overridden there); milestone by
`scripts/train_residual_rl_can_sarm_stage_milestone.sh`, which appends
`reward_model.reward_mode=milestone` plus the two milestone overrides above
to the Hydra CMD array (confirmed at
`train_residual_rl_can_sarm_stage_milestone.sh` — grepped
`REWARD_MODE="milestone"` at line 197 and the corresponding
`reward_model.reward_mode=...` append inside the `if REWARD_MODEL_ENABLED`
block later in the file).

**Important — the two per-task "default" scripts do NOT run the reward
model at all.** `scripts/train_residual_rl_can.sh:682` and
`scripts/train_residual_rl_can_sparse.sh:686` both set
`REWARD_MODEL_ENABLED="false"` (default, comment says "← set true to use the
reward model"). Both of these scripts are the ones showing as modified/
uncommitted in `git status`. Only the two dedicated
`*_sarm_stage_pbrs.sh` / `*_sarm_stage_milestone.sh` scripts set
`REWARD_MODEL_ENABLED="true"`. So: **as currently configured, `can.sh` and
`can_sparse.sh` train on the plain robosuite env reward** (dense if
`REWARD_SHAPING="true"` in `can.sh:558`, sparse if `REWARD_SHAPING="false"`
in `can_sparse.sh:572`) — the SARM/PBRS/milestone reward path is opt-in via a
separate pair of scripts, not the default training path.

### 6.4 How the shaped reward combines with the env's own reward — REPLACES, not added

`env.step()` inside the wrapper first calls the underlying env and captures
`r_sparse = float(reward)` — the environment's own (sparse or dense,
depending on `reward_shaping=` at env construction) reward
(`pbrs_wrapper.py:191-192`). This `r_sparse` value is then **only used as an
input term** to the PBRS/milestone formulas above; the wrapper's `step()`
returns:
```python
return obs, r_out, bool(terminated), bool(truncated), info
```
`pbrs_wrapper.py:245` — **`r_out` fully replaces the environment's `reward`**
in the 5-tuple passed up the wrapper stack; there is no further combination
(no `env_reward + shaped_reward` anywhere upstream). This is confirmed
downstream too: `VectorizedEnvWrapper.step()` in `dexmg.py` just converts
`rewards` (already `r_out` from the sub-env) to a `torch.float32` tensor with
no numeric transform — grepped for any additive combination of a reward-model
term with a separate env-reward term in `train_residual_td3.py` and found
none; `env.step(action)` at the main-loop call site
(`train_residual_td3.py`, main loop) receives `reward` that is already
`r_out`.

### 6.5 Storage time vs replay-sample time — computed at STEP time, stored verbatim, then n-step-summed at SAMPLE time

- **Step/storage time**: `_add_transitions_to_buffer()` in
  `train_residual_td3.py` stores `reward[i]` — which is `r_out` from the
  wrapper's `step()`, unmodified — directly into
  `TensorDict({"next": {"reward": reward[i], ...}})`, with no scaling or
  clipping applied at insertion. (Verified: no post-processing of `reward`
  between `env.step()` and the `TensorDict` construction in
  `_add_transitions_to_buffer`.)
- **Sample time**: a `MultiStepTransform` (n-step return) is attached to both
  the online and offline replay buffers
  (`train_residual_td3.py`, `MultiStepTransform(n_steps=cfg.algo.n_step,
  gamma=cfg.algo.gamma, use_terminated_for_bootstrap=cfg.algo.terminated_only_bootstrap)`).
  Its implementation, `resfit/rl_finetuning/utils/rb_transforms.py`
  (`MultiStepTransform._multi_step_func`), discount-sums the *raw per-step*
  `next.reward` values stored above over a sliding window of `n_step`
  consecutive transitions:
  `summed_reward = r_t + γ·r_{t+1} + ... `, capped/truncated early if `done`
  fires within the window, renaming the raw 1-step value to
  `next.original_reward` and overwriting `next.reward` with the n-step sum.
  This transform is **identical** for PBRS and milestone — it has no
  knowledge of `reward_mode`, it just sums whatever scalar was stored.
- `n_step`: both `_pbrs.sh`/`_milestone.sh` set `N_STEP=3`.
  `algo.terminated_only_bootstrap=true` is forced in the Hydra CMD when the
  reward model is enabled (`train_residual_td3.py:583-608` warns, does not
  hard-fail, if it's not set) — this matters for PBRS specifically because the
  telescoping identity `Σ γ^t(γφ(s_{t+1}) − φ(s_t))` only collapses correctly
  to `γ^nφ(s_{t+n}) − φ(s_t)` if `φ(absorbing)=0` is applied at the true
  terminal, not at truncation.
- **Offline buffer path**: no live HTTP calls happen while
  `train_residual_td3.py` is running for the *offline* buffer. A separate
  script, `scripts/label_offline_pbrs.py`, is run beforehand, streams demo
  frames through the same `/predict_stage` endpoint with the same cadence
  and formulas, and writes a sidecar parquet
  (`pbrs_sarm_progress.parquet` / `milestone_sarm_progress.parquet`, present
  at repo root) with a `reward_pbrs` or `reward_milestone` column. At RL
  startup, if `offline_data.reward_parquet` is set, `train_residual_td3.py`
  reads embedded `labeler_config` metadata from the parquet and hard-errors
  if it mismatches the live `cfg.reward_model` config (e.g. wrong
  `reward_mode`, `gamma`, `potentials`, `milestone_payouts`). During
  offline-buffer population the per-transition reward is resolved by
  priority: (1) sidecar parquet value if present, else (2) the dataset's own
  dense reward if `reward_shaping=True`, else (3) sparse `1.0`/`0.0` from the
  episode's `done` flag — then stored into `next.reward` and goes through the
  same `MultiStepTransform` as the online path when sampled.

Note: `can.sh` / `can_sparse.sh` currently set `OFFLINE_REWARD_PARQUET=""`
(empty, lines 698/702), so for those two default scripts, offline-buffer
reward falls back to case (2)/(3) above — no reward-model labeling is used
offline either, consistent with `REWARD_MODEL_ENABLED="false"` in those same
scripts.

### 6.6 Stage-aware component and streak/hysteresis gating

**What defines a "stage"**: `self._current_stage`, an integer in
`[0, num_stages-1]` (`num_stages = len(potentials)`, i.e. **4** with the
active `REWARD_POTENTIALS`/`MILESTONE_PAYOUTS` config). It is not a
locally-computed subtask index or phase counter — it is entirely **returned
by the external reward server** as `out["gated_stage"]`
(`pbrs_wrapper.py:151`), then client-side re-clamped:
```python
if self.monotonic:
    stage = max(self._current_stage, stage)
stage = min(max(stage, 0), self.num_stages - 1)
```
(`pbrs_wrapper.py:152-154`) — a defense-in-depth monotonic clamp on top of
the server already being told `monotonic=1`.

**Streak-hysteresis gating**: grepped `resfit/rl_finetuning/` (all `.py`
files) for `streak`, `hysteresis`, `gate`, `gating` (case-insensitive).
- `streak`: **NOT FOUND** anywhere in the codebase (`.py`, `.md`, `.sh`)
  except inside `../docs/rewards/REWARD_MODEL_INTEGRATION.md`'s own prose (see below).
- `hysteresis`/`gating`: present only as (a) a client-side parameter
  `hysteresis_k` that is **sent to the server per request** and never
  interpreted client-side beyond being forwarded
  (`pbrs_wrapper.py:65,84,135`; `reward_client.py:101,111,113,131`;
  `residual_td3.py:147` — comment "consecutive high-conf predictions to
  upgrade a stage"), and (b) an unrelated IBRL Q-value gating mechanism in
  `resfit/rl_finetuning/off_policy/rl/ibrl_q_agent.py:481,597` (action
  selection between base/residual actor via a target-critic comparison — not
  a reward-shaping mechanism, out of scope for this reward-path question).

**Conclusion**: the actual streak/hysteresis *gating logic* (a persistent
per-session counter that must see `hysteresis_k` consecutive high-confidence
votes before advancing a stage) is **not implemented anywhere in this repo**.
`REWARD_MODEL_INTEGRATION.md:319-372` contains a detailed Python-pseudocode
description of exactly this mechanism ("Hysteresis — it's a running streak,
NOT 'last 5 must match'", with a `count_toward_next` counter reset on any
low-confidence/lower-stage vote) — but this describes the **external SARM/TCC
reward server's** behavior, which lives outside this repository. No file in
this repo (`grep -rln "predict_stage\|gated_stage"`) implements it; the
client only ever sends `hysteresis_k`/`conf_threshold`/`monotonic` as request
parameters and receives back an already-gated `gated_stage` integer. This
should be treated as **documentation of external/inferred server behavior,
not verified in-repo code** — flagged explicitly since the doc's confident
pseudocode could otherwise read as code-grounded.

---

## #9 Ablations — `scripts/CanAblationStudies/*.sh`

25 scripts found (matches expected count), read in full as text; every flag
below was grepped into `resfit/rl_finetuning/config/residual_td3.py` and
`resfit/rl_finetuning/scripts/train_residual_td3.py` (plus
`unified_buffer.py`, `actor.py`, `normalization.py`) to confirm what it
actually changes. `scripts/CanAblationStudies/ablation_checklist.md`
describes the intended round structure (Round 0: critic warmup + BC
robustness; Round 1: offline-RL sampling method; Round 2: checkpoint-loading
× offline-RL-steps × BC robustness; Round 3: action-scale schedule; Round 4:
final 5-seed confirmation).

### Axis 1 — Critic warmup (on/off, and how much)

- **CLI/env var**: `CRITIC_WARMUP` (shell) → Hydra override
  `algo.critic_warmup_steps="${CRITIC_WARMUP}"`
  (`scripts/CanAblationStudies/sparse_criticWarmup.sh:760`).
- **Config field**: `critic_warmup_steps: int = 10_000`
  (`resfit/rl_finetuning/config/residual_td3.py:164`).
- **Code that reads it**: `train_residual_td3.py:1719-1721` —
  `if cfg.algo.critic_warmup_steps > 0 and not _is_resuming and not _did_offline_rl:` then
  runs `_run_critic_warmup(...)`, which loops
  `for i in range(cfg.algo.critic_warmup_steps): ...` doing critic-only
  updates (no actor update) — `train_residual_td3.py:1455-1459` (function
  body).
- **Values used across scripts**: `0` (`sparse_noCriticWarmup*.sh`, and every
  `sparse_with{25k,50k,75k}OffRL*.sh` / `sparse_just200kOffRL.sh` script) vs
  `10000` (`sparse_criticWarmup.sh`, `sparse_criticWarmup_withAvgBC.sh`,
  `sparse_criticWarmup_listedResScale.sh`). Note: warmup is skipped entirely
  whenever `_did_offline_rl` is true (offline-RL scripts implicitly disable
  it via the `and not _did_offline_rl` guard, independent of the
  `CRITIC_WARMUP` value) — confirmed all offline-RL ablation scripts also set
  `CRITIC_WARMUP=0` explicitly, consistent with warmup being irrelevant once
  offline TD3-BC pretraining runs.

### Axis 2 — "AvgBC" (weaker base-policy checkpoint) on/off

- **CLI/env vars**: `BASE_WT_TYPE` and `BASE_WT_VERSION` → Hydra overrides
  `base_policy.wt_type="${BASE_WT_TYPE}"`,
  `base_policy.wt_version="${BASE_WT_VERSION}"`
  (`sparse_criticWarmup.sh:736-737`). Same `BASE_WANDB_ID` run id is used in
  both variants (`resfit-robomimic-can-bc/pzqj1tmd`).
- **Code that reads it**: `train_residual_td3.py:317-320` —
  `download_policy_from_wandb(cfg.base_policy.wandb_id, step=cfg.base_policy.wt_type, artifact_version=cfg.base_policy.wt_version)`.
  Inside `resfit/lerobot/utils/load_policy.py:18-40`
  (`download_policy_from_wandb`): if `step == "latest"` it pulls W&B artifact
  `run_{id}_latest:{artifact_version}`; if `step == "best"` it pulls
  `run_{id}_best:{artifact_version}`.
- **Values used**: default/baseline = `BASE_WT_TYPE="latest"`,
  `BASE_WT_VERSION="v9"` (→ artifact `run_pzqj1tmd_latest:v9`); "withAvgBC"
  variants = `BASE_WT_TYPE="best"`, `BASE_WT_VERSION="v0"` (→ artifact
  `run_pzqj1tmd_best:v0`) — every `*_withAvgBC*.sh` script (6 of them) uses
  this pairing.
- **Caveat (NOT FOUND)**: nothing in the RL-side code names or defines what
  makes this specific checkpoint an "average" BC policy — "AvgBC" is purely a
  *naming convention* the ablation-study author applied to the
  `wt_type="best"`/`v0` W&B artifact from the BC training run; the actual
  distinction (e.g., whether "best" here means an EMA/weight-averaged
  checkpoint vs. simply the lowest-validation-loss iterate from BC training)
  lives in the BC training pipeline, not in this ablation/RL code, and was
  not traced further per the scope of this task.

### Axis 3 — Offline-RL pretraining amount (k-steps) and on/off

- **CLI/env vars**: `TRAIN_OFFLINE_RL` (bool) →
  `algo.train_offline_rl="${TRAIN_OFFLINE_RL}"`; `OFFLINE_RL_STEPS` (int) →
  `algo.offline_rl_steps="${OFFLINE_RL_STEPS}"`
  (`sparse_with25kOffRL.sh:767-768`).
- **Config fields**: `train_offline_rl: bool = False`
  (`residual_td3.py:193`), `offline_rl_steps: int = 50000`
  (`residual_td3.py:194`).
- **Code that reads it**: `train_residual_td3.py:1514` —
  `if getattr(cfg.algo, "train_offline_rl", False):` gates the entire
  offline TD3-BC phase (pools `td_offline`/`td_online` tensordicts into an
  `EpochUnifiedDataset`, `resfit/rl_finetuning/off_policy/rl/unified_buffer.py:71`,
  and runs `for i in range(1, cfg.algo.offline_rl_steps + 1):` TD3-BC updates,
  `train_residual_td3.py:1541`). `_did_offline_rl` flag at
  `train_residual_td3.py:1394,1718` (`train_offline_rl AND offline_rl_steps>0`)
  is what later disables critic warmup (Axis 1) and also determines whether
  the specified checkpoint-load step (Axis 5) runs.
- **Values used**: `TRAIN_OFFLINE_RL="false"` (no offline RL at all) in
  `sparse_noCriticWarmup*.sh`, `sparse_criticWarmup*.sh` (Round 0 scripts);
  `TRAIN_OFFLINE_RL="true"` with `OFFLINE_RL_STEPS ∈ {25000, 50000, 75000,
  200000}` in the `sparse_with{25k,50k,75k}OffRL*.sh` and
  `sparse_just200kOffRL.sh` scripts respectively. `sparse_just200kOffRL.sh`
  additionally sets `TOTAL_TIMESTEPS=0` (vs `200000` in the 25k/50k/75k
  scripts) — i.e. it runs **only** the 200k-step offline phase with **no**
  subsequent online RL loop at all (confirmed by diff:
  `sparse_with25kOffRL.sh` vs `sparse_just200kOffRL.sh` differs only in
  `TOTAL_TIMESTEPS`, `OFFLINE_RL_STEPS`, and `WANDB_NAME`).

### Axis 4 — Offline-RL sampling method + ratio

- **CLI/env vars**: `OFFLINE_RL_SAMPLING_METHOD` (`"fixed_ratio"` |
  `"proportional"`) → `algo.offline_rl_sampling_method="${...}"`;
  `OFFLINE_RL_OFFLINE_RATIO` (float) →
  `algo.offline_rl_offline_ratio="${...}"`
  (`sparse_with25kOffRL_ratio025.sh:772-773`).
- **Config fields**: `offline_rl_sampling_method: str = "fixed_ratio"`,
  `offline_rl_offline_ratio: float = 0.5` (`residual_td3.py:199-200`).
- **Code that reads it**: `train_residual_td3.py:1529-1534` constructs
  `EpochUnifiedDataset(td_offline, td_online, sampling_method=offline_sampling, offline_ratio=offline_ratio)`.
  Inside `unified_buffer.py:96-112` (constructor) and `:132-172` (`sample()`):
  `"fixed_ratio"` forces `off_batch = int(batch_size * offline_ratio)`
  offline samples per batch, remainder from online
  (`unified_buffer.py:139`); `"proportional"` pools both buffers and samples
  by their natural size ratio (`unified_buffer.py:172`+). Docstring at
  `unified_buffer.py:78-82` states this explicitly.
- **Values used**: baseline `fixed_ratio` @ `0.5`
  (`sparse_with25kOffRL.sh`); `proportional` (`sparse_with25kOffRL_proportional.sh`,
  which also sets `TRAIN_OFFLINE_RL="false"` and a `RESUME_CKPT` path — this
  variant resumes from an already-offline-RL-trained checkpoint rather than
  redoing the offline phase, so it's testing the *online*-phase sampling
  behavior via resume, not a fresh offline run); `fixed_ratio @ 0.25`
  (`sparse_with25kOffRL_ratio025.sh`); `fixed_ratio @ 0.75`
  (`sparse_with25kOffRL_ratio075.sh`).

### Axis 5 — Offline-RL checkpoint to load before online phase ("best" vs "latest")

- **CLI/env var**: `OFFLINE_RL_LOAD_CKPT` (`"latest"` | `"best"` | a step
  number) → `algo.offline_rl_load_ckpt="${OFFLINE_RL_LOAD_CKPT}"`
  (`sparse_with25kOffRL_best.sh:771`).
- **Config field**: `offline_rl_load_ckpt: str = "latest"`
  (`residual_td3.py:198`).
- **Code that reads it**: `train_residual_td3.py:1672-1690` — after the
  offline TD3-BC phase completes, `load_choice = getattr(cfg.algo,
  "offline_rl_load_ckpt", "best")`; if `"best"`, loads
  `model_save_dir / "offline_best" / "checkpoint.pt"` into the agent before
  starting online RL; if `"latest"`, keeps the in-memory (final-step)
  weights as-is; otherwise treats `load_choice` as a specific step number and
  loads `offline_step_{N}/checkpoint.pt`.
- **Values used**: `"latest"` (baseline `sparse_with{25k,50k,75k}OffRL.sh`,
  and the `_withAvgBC.sh` / `_listedResScale.sh` / `_proportional.sh`
  variants) vs `"best"` in every `*_best.sh` script (6 of them:
  `sparse_with{25k,50k,75k}OffRL_best.sh` and
  `sparse_with{25k,50k,75k}OffRL_withAvgBC_best.sh`).

### Axis 6 — Residual action-scale schedule (scalar vs per-dimension "listed")

- **CLI/env vars**: `ACTION_SCALE` → `agent.actor.action_scale="${ACTION_SCALE}"`
  (verified consumer path below); companion vars `STDDEV_MAX`, `STDDEV_MIN`
  (→ `algo.stddev_max`/`algo.stddev_min`) and `RANDOM_ACTION_NOISE_SCALE`
  (→ `algo.random_action_noise_scale`) are changed **together** with
  `ACTION_SCALE` in the `*_listedResScale.sh` scripts (confirmed via diff:
  all three change together in `sparse_with25kOffRL_listedResScale.sh` vs
  baseline).
- **Config fields**: `action_scale=0.1` (actor default,
  `residual_td3.py:272`, inside `ActorConfig`); `stddev_max: Any = 0.05`,
  `stddev_min: Any = 0.05` (`residual_td3.py:182-183`);
  `random_action_noise_scale: Any = 0.2` (`residual_td3.py:171`).
- **Code that reads `action_scale`**: `resfit/rl_finetuning/off_policy/rl/actor.py:119-124` —
  `act_scale = to_native_list(cfg.action_scale)`, stored as a registered
  buffer `action_scale_tensor`; used in `forward()` at
  `actor.py:179-181`: `scaled_mu = mu * self.action_scale_tensor` — i.e. it
  scales the actor network's raw output mean before it's used as the
  (truncated-normal) residual-action distribution mean. Also passed into
  `ActionScaler` (`resfit/rl_finetuning/utils/normalization.py:41,56,64-67,78`)
  which expands the unscale range by `(1 + action_scale)`.
- **`random_action_noise_scale`** governs the uniform-noise magnitude used
  during the pure-random warmup phase (`residual = uniform_noise *
  random_action_noise_scale`, per in-script comment and consumed at
  `train_residual_td3.py:1283,1295`, and threaded into env creation at
  `train_residual_td3.py:778`).
- **Values used**: scalar `0.2` (baseline, all non-`listedResScale` scripts)
  vs the 7-D per-joint list `[0.15,0.15,0.15,0.05,0.05,0.05,0.1]` (in
  `sparse_criticWarmup_listedResScale.sh`,
  `sparse_noCriticWarmup_listedResScale.sh`,
  `sparse_with{25k,50k,75k}OffRL_listedResScale.sh` — 5 scripts). The
  matching `STDDEV_MAX`/`STDDEV_MIN` in the `listedResScale` variants change
  from scalar `0.025` to the list `[0.015,0.015,0.015,0.003,0.003,0.003,0.0125]`.

### Summary table

| Axis | Flag(s) | Config field(s) | Code that consumes it | Values in ablation scripts |
|---|---|---|---|---|
| Critic warmup | `CRITIC_WARMUP` | `algo.critic_warmup_steps` (default 10000) | `train_residual_td3.py:1719-1721`, `:1455-1459` | 0, 10000 |
| BC checkpoint quality ("AvgBC") | `BASE_WT_TYPE`, `BASE_WT_VERSION` | `base_policy.wt_type`, `base_policy.wt_version` | `train_residual_td3.py:317-320`; `load_policy.py:18-40` | latest/v9 (baseline) vs best/v0 (AvgBC) |
| Offline-RL on/off + amount | `TRAIN_OFFLINE_RL`, `OFFLINE_RL_STEPS`, `TOTAL_TIMESTEPS` | `algo.train_offline_rl` (False), `algo.offline_rl_steps` (50000) | `train_residual_td3.py:1394,1514,1541,1718` | off; 25k/50k/75k/200k (200k = offline-only, `TOTAL_TIMESTEPS=0`) |
| Offline-RL sampling method+ratio | `OFFLINE_RL_SAMPLING_METHOD`, `OFFLINE_RL_OFFLINE_RATIO` | `algo.offline_rl_sampling_method` ("fixed_ratio"), `algo.offline_rl_offline_ratio` (0.5) | `train_residual_td3.py:1529-1534`; `unified_buffer.py:96-112,132-172` | fixed_ratio@{0.25,0.5,0.75}, proportional |
| Offline→online checkpoint choice | `OFFLINE_RL_LOAD_CKPT` | `algo.offline_rl_load_ckpt` ("latest") | `train_residual_td3.py:1672-1690` | "latest", "best" |
| Residual action-scale schedule | `ACTION_SCALE`, `STDDEV_MAX/MIN`, `RANDOM_ACTION_NOISE_SCALE` | `agent.actor.action_scale` (0.1 default), `algo.stddev_max/min` (0.05), `algo.random_action_noise_scale` (0.2) | `actor.py:119-124,179-181`; `normalization.py:41-78`; `train_residual_td3.py:1283,1295,778` | scalar 0.2 vs per-joint list `[0.15,0.15,0.15,0.05,0.05,0.05,0.1]` (and matching stddev lists) |

---

## Discrepancies with existing docs

1. **`../docs/rewards/REWARD_MODEL_INTEGRATION.md`'s hysteresis pseudocode is server-side,
   unverifiable in-repo.** `REWARD_MODEL_INTEGRATION.md:319-372` presents a
   confident, specific Python-pseudocode explanation of a "running streak"
   hysteresis counter (`count_toward_next`, resets on any low-confidence
   frame, needs `hysteresis_k` consecutive votes to advance one stage). This
   repo contains **no implementation** of that logic — `grep -rn "streak"`
   across all `.py` files returns zero hits, and no file implements
   `predict_stage`/`gated_stage` server-side. `../docs/rewards/STAGE_AWARE_REWARD_PIPELINE.md`
   is explicit that it does not trace server internals; readers of
   `../docs/rewards/REWARD_MODEL_INTEGRATION.md` alone could mistake its pseudocode for
   traced code. Treat that section of `../docs/rewards/REWARD_MODEL_INTEGRATION.md` as an
   explanation of an external service, not this repo's code.
2. **`../docs/rewards/REWARD_AND_SUCCESS.md` (Feb 2026) predates the reward-model feature
   entirely** — it contains zero mentions of `reward_model`, `pbrs`,
   `milestone`, `SARM`, or `TCC` (grepped, no hits). It is stale for any
   reward-model question and was not otherwise used as a source in this
   report.
3. **`../docs/rewards/STAGE_AWARE_REWARD_PIPELINE.md` — confirmed accurate**, including its
   line-number citations into `pbrs_wrapper.py`, `reward_client.py`,
   `residual_td3.py`, and `train_residual_td3.py`, and its correction that
   PBRS/milestone share one wrapper class. One addition beyond what it
   states: it does not call out that `train_residual_rl_can.sh` and
   `train_residual_rl_can_sparse.sh` (the two scripts flagged as
   uncommitted/modified for this task) both currently ship with
   `REWARD_MODEL_ENABLED="false"` — i.e., as configured today, neither of
   the "regular" per-task training scripts exercises the reward-model path
   at all; only the two dedicated `*_sarm_stage_*.sh` scripts do. This is
   worth stating explicitly for the thesis figure so it doesn't imply the
   reward-model path is the default/primary one.
4. No discrepancy found in the ablation-script/config mapping — the
   `ablation_checklist.md` round structure matches what the flags actually
   do in code (critic-warmup screening in Round 0, sampling-method screening
   in Round 1, checkpoint-loading × step-count × BC-robustness in Round 2,
   action-scale in Round 3).

---

## Files referenced (absolute paths)

- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/rl_finetuning/reward_models/pbrs_wrapper.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/rl_finetuning/reward_models/reward_client.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/rl_finetuning/config/residual_td3.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/rl_finetuning/scripts/train_residual_td3.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/rl_finetuning/off_policy/rl/unified_buffer.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/rl_finetuning/off_policy/rl/actor.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/rl_finetuning/utils/normalization.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/lerobot/utils/load_policy.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/resfit/dexmg/environments/dexmg.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/train_residual_rl_can_sarm_stage_pbrs.sh`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/train_residual_rl_can_sarm_stage_milestone.sh`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/train_residual_rl_can.sh`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/train_residual_rl_can_sparse.sh`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/label_offline_pbrs.py`
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/CanAblationStudies/*.sh` (25 scripts)
- `/home/qte9489/personal_abhi/temp/residual-offpolicy-rl/scripts/CanAblationStudies/ablation_checklist.md`
