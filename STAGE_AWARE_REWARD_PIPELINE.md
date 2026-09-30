# Stage-Aware External Reward Model Pipeline (SARM / TCC via HTTP)

Traced directly from code, with file:line citations for every claim. Covers the
two entry points:

- `scripts/train_residual_rl_can_sarm_stage_pbrs.sh`      (`reward_mode="pbrs"`, the default)
- `scripts/train_residual_rl_can_sarm_stage_milestone.sh` (`reward_mode="milestone"`)

**Important correction up front:** PBRS and milestone are **not two separate
wrapper classes**. Both live inside the single class `StageAwarePBRSWrapper` in
`resfit/rl_finetuning/reward_models/pbrs_wrapper.py`, selected at construction
time by a `reward_mode: "pbrs" | "milestone"` argument. There is no
`milestone_wrapper.py` anywhere in the repo.

Everything below is verified against the code in this repo as of this writing.
Server-side behavior (SARM / TCC internals) is **not** traced — only what the
client sends and what shape/fields it expects back, per the wire protocol the
client code implements.

---

## 0. Exact parameters used by each entry script

Both scripts share every hyperparameter except the reward block (the milestone
script's header comment says so explicitly: "Everything ... IDENTICAL — only
the reward differs").

| Param | `_pbrs.sh` | `_milestone.sh` |
|---|---|---|
| `REWARD_MODEL_ENABLED` | `true` | `true` |
| `REWARD_MODEL_BACKEND` | `sarm` | `sarm` |
| `REWARD_SERVER_URL` | `http://127.0.0.1:8003` | `http://127.0.0.1:8003` |
| `REWARD_TASK_PROMPT` | `"pick up the can and place it in the bin"` | same |
| `REWARD_QUERY_EVERY_K` | `4` | `4` |
| `REWARD_HYSTERESIS_K` | `4` | `4` |
| `REWARD_CONF_THRESHOLD` | `0.80` | `0.80` |
| `REWARD_POTENTIALS` | `[0.0, 0.05, 0.1, 0.15]` — 4 stage potentials | `[0.0, 0.05, 0.1, 0.15]` — **unused as potentials**, only its length (4) sets `num_stages` |
| `REWARD_KEEP_SPARSE_TERM` | `true` — used | `true` — **ignored** in milestone mode |
| `REWARD_IMAGE_VFLIP` | `false` | `false` |
| `TERMINATED_ONLY_BOOTSTRAP` | `true` | `true` |
| `REWARD_MODE` | not set → defaults to `"pbrs"` | `"milestone"` |
| `MILESTONE_PAYOUTS` | not set → default `(0.0, 0.1, 0.1, 0.1)` (unused, mode is pbrs) | `[0.0, 0.3, 0.3, 0.4]` |
| `MILESTONE_SUCCESS_BONUS` | not set → default `0.7` (unused) | `1.0` |
| `pbrs_gamma` | not set in either script → falls back to `algo.gamma = 0.995` | n/a in milestone mode |
| `GAMMA` (`algo.gamma`) | `0.995` | `0.995` |
| `V_MIN` / `V_MAX` (critic) | `0.0` / `80.0` | `0.0` / `80.0` |
| `CRITIC_LOSS_TYPE` | `mse` | `mse` |
| `OFFLINE_REWARD_PARQUET` | `./pbrs_sarm_progress.parquet` | `./milestone_sarm_progress.parquet` |
| `OFFLINE_REWARD_COLUMN` | `reward_pbrs` | `reward_milestone` |
| `WANDB_PROJECT` | `robomimic-can-residual-sarm-stage-pbrs` | `robomimic-can-residual-sarm-stage-milestone` |

Max theoretical per-episode milestone return: `0.3+0.3+0.4 (payouts) + 1.0 (success bonus) = 2.0`
(matches the comment at `_milestone.sh:194`). MSE critic loss doesn't clamp, so `V_MAX=80.0` covers it with margin.

Source: `scripts/train_residual_rl_can_sarm_stage_pbrs.sh:672-707`,
`scripts/train_residual_rl_can_sarm_stage_milestone.sh:179-211`.

---

## 1. Shell script → Hydra CLI overrides

Both scripts build a bash array `CMD` of `key=value` Hydra overrides. The
`reward_model.*` overrides are appended **conditionally**, only when
`REWARD_MODEL_ENABLED == "true"`:

```bash
# pseudocode, both scripts
if REWARD_MODEL_ENABLED == "true":
    CMD += [
        "reward_model.enabled=true",
        f"reward_model.backend={REWARD_MODEL_BACKEND}",
        f"reward_model.server_url={REWARD_SERVER_URL}",
        f"reward_model.task_prompt={REWARD_TASK_PROMPT}",
        f"reward_model.query_every_k={REWARD_QUERY_EVERY_K}",
        f"reward_model.hysteresis_k={REWARD_HYSTERESIS_K}",
        f"reward_model.conf_threshold={REWARD_CONF_THRESHOLD}",
        f"reward_model.potentials={REWARD_POTENTIALS}",
        f"reward_model.keep_sparse_term={REWARD_KEEP_SPARSE_TERM}",
        f"reward_model.image_vflip={REWARD_IMAGE_VFLIP}",
        "algo.terminated_only_bootstrap=true",
    ]
    # milestone script ONLY, additionally:
    CMD += [
        "reward_model.reward_mode=milestone",
        f"reward_model.milestone_payouts={MILESTONE_PAYOUTS}",
        f"reward_model.milestone_success_bonus={MILESTONE_SUCCESS_BONUS}",
    ]

run: python resfit/rl_finetuning/scripts/train_residual_td3.py --config-name=residual_td3_can_config <CMD overrides>
```

Source: `_pbrs.sh:830-844`, `_milestone.sh:333-350`.

---

## 2. Hydra structured config — `RewardModelConfig`

Every `reward_model.X=...` override above lands on this dataclass
(`resfit/rl_finetuning/config/residual_td3.py:78-154`):

```python
@dataclass
class RewardModelConfig:
    enabled: bool = False
    backend: str = "sarm"                    # "sarm" | "tcc" — informational only, server decides
    server_url: str = "http://127.0.0.1:8001"
    request_timeout_s: float = 20.0
    require_server: bool = True              # abort at startup if /health fails

    task_prompt: str = "pick up the can and place it in the bin"
    query_every_k: int = 5
    client_history_len: int = 40             # documented but NOT read anywhere in pbrs_wrapper.py — see §7
    image_key: str = "observation.images.agentview"
    image_vflip: bool = False

    # PBRS params
    num_stages: int = 4
    potentials: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15)
    pbrs_gamma: float | None = None          # None -> falls back to algo.gamma
    keep_sparse_term: bool = True

    # Reward mode
    reward_mode: str = "pbrs"                # "pbrs" | "milestone"
    milestone_payouts: tuple[float, ...] = (0.0, 0.1, 0.1, 0.1)
    milestone_success_bonus: float = 0.7

    # Server-side gating knobs (sent to server per request, not applied client-side)
    hysteresis_k: int = 5
    conf_threshold: float = 0.80
    monotonic: bool = True

    # Offline dataset labeling
    offline_key: str | None = None
```

Registered on the top-level config at `ResidualTD3DexmgConfig` /
`RLPDDexmgConfig` via `reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)`
(`residual_td3.py:291`).

---

## 3. `main()` in `train_residual_td3.py` — pre-flight checks (in order)

```python
# residual_td3.py main(), ~line 573 onward
assert cfg.num_envs == 1   # n-step implementation requires exactly 1 training env

if cfg.reward_model.enabled:
    if not cfg.algo.terminated_only_bootstrap:
        log_warning(...)   # does NOT abort, just warns

    client = RewardModelClient(server_url=cfg.reward_model.server_url,
                                task_prompt=cfg.reward_model.task_prompt,
                                request_timeout_s=cfg.reward_model.request_timeout_s)
    if client.check():                      # GET /health, catches all exceptions -> True/False
        log_info(f"Reward server OK: {client.health()}")
    elif cfg.reward_model.require_server:    # default True in both scripts
        raise RuntimeError("Reward server not reachable ...")   # TRAINING ABORTS HERE
    else:
        log_warning("continuing without server")
```

Source: `train_residual_td3.py:573-608`. This check runs **before** the offline
dataset is loaded and before any env is constructed — a dead reward server
fails fast, not mid-training (mid-training failures are a separate hard-fail
path, see §5.5).

---

## 4. `get_envs()` — assembling the `reward_model_cfg` dict

Inside `get_envs()` (a closure in `main()`), when building the *training* env
(`apply_reward_model=True`, only for the training env — never for eval envs):

```python
# train_residual_td3.py:497-523
reward_model_cfg = None
force_sync = False
if apply_reward_model and cfg.reward_model.enabled:
    pbrs_gamma = cfg.reward_model.pbrs_gamma
    if pbrs_gamma is None:
        pbrs_gamma = cfg.algo.gamma            # both scripts: neither sets pbrs_gamma -> uses algo.gamma=0.995

    reward_model_cfg = {
        "enabled": True,
        "server_url": cfg.reward_model.server_url,
        "task_prompt": cfg.reward_model.task_prompt,
        "request_timeout_s": cfg.reward_model.request_timeout_s,
        "session_prefix": f"{cfg.reward_model.backend}-{cfg.task}",   # e.g. "sarm-can"
        "potentials": list(cfg.reward_model.potentials),
        "pbrs_gamma": float(pbrs_gamma),
        "query_every_k": cfg.reward_model.query_every_k,
        "image_key": cfg.reward_model.image_key,
        "keep_sparse_term": cfg.reward_model.keep_sparse_term,
        "hysteresis_k": cfg.reward_model.hysteresis_k,
        "conf_threshold": cfg.reward_model.conf_threshold,
        "monotonic": cfg.reward_model.monotonic,
        "image_vflip": cfg.reward_model.get("image_vflip", False),
        "reward_mode": cfg.reward_model.get("reward_mode", "pbrs"),
        "milestone_payouts": list(cfg.reward_model.get("milestone_payouts", (0.0, 0.1, 0.1, 0.1))),
        "milestone_success_bonus": float(cfg.reward_model.get("milestone_success_bonus", 0.7)),
    }
    force_sync = True   # forces SyncVectorEnv so the blocking HTTP call runs in-process,
                         # not across an AsyncVectorEnv subprocess pipe
```

Source: `train_residual_td3.py:495-540`. `force_sync=True` is then threaded
into `create_vectorized_env(..., force_sync=force_sync)` — this is why the
reward model can *only* run against the single training env, never the
parallel eval envs.

---

## 5. Env construction — `make_dexmimicgen_env()` in `dexmg.py`

```python
# resfit/dexmg/environments/dexmg.py:756-811, inside the per-subenv factory _make()
env = RobosuiteGymWrapper(env_name=..., camera_size=..., headless=..., reward_shaping=..., horizon=...)
# (optional RewardManipulationWrapper — not used by either sarm-stage script)

if reward_model_cfg is not None and reward_model_cfg.get("enabled", False):
    from resfit.rl_finetuning.reward_models.reward_client import RewardModelClient
    from resfit.rl_finetuning.reward_models.pbrs_wrapper import StageAwarePBRSWrapper

    client = RewardModelClient(
        server_url=reward_model_cfg["server_url"],
        task_prompt=reward_model_cfg.get("task_prompt", ""),
        request_timeout_s=reward_model_cfg.get("request_timeout_s", 20.0),
    )
    session_id = f"{reward_model_cfg['session_prefix']}-env{env_id}"   # e.g. "sarm-can-env0"
    env = StageAwarePBRSWrapper(
        env,
        client=client,
        session_id=session_id,
        potentials=reward_model_cfg["potentials"],
        gamma=reward_model_cfg["pbrs_gamma"],
        query_every_k=reward_model_cfg.get("query_every_k", 5),
        image_key=reward_model_cfg.get("image_key", "observation.images.agentview"),
        keep_sparse_term=reward_model_cfg.get("keep_sparse_term", True),
        hysteresis_k=reward_model_cfg.get("hysteresis_k", 5),
        conf_threshold=reward_model_cfg.get("conf_threshold", 0.80),
        monotonic=reward_model_cfg.get("monotonic", True),
        image_vflip=reward_model_cfg.get("image_vflip", False),
        task_prompt=reward_model_cfg.get("task_prompt", ""),
        reward_mode=reward_model_cfg.get("reward_mode", "pbrs"),
        milestone_payouts=reward_model_cfg.get("milestone_payouts"),
        milestone_success_bonus=reward_model_cfg.get("milestone_success_bonus", 0.7),
    )
return env
```

Source: `dexmg.py:740-811`. This is the *only* place `StageAwarePBRSWrapper`
is instantiated — one instance per sub-env (here, exactly one sub-env, since
`cfg.num_envs == 1` is asserted).

### 5.1 `RewardModelClient` — HTTP client (client-side only; server internals not traced)

`resfit/rl_finetuning/reward_models/reward_client.py`, a thin `requests`-based
synchronous client. Three endpoints it calls:

```
GET  {server_url}/health
    -> client expects JSON with a "status" key; client.check() returns
       str(info.get("status","")).lower() == "ok"   (reward_client.py:67-79)

POST {server_url}/reset
    form: {"session_id": str}
    -> client only checks HTTP status (raise_for_status); return value ignored
       (reward_client.py:81-88)

POST {server_url}/predict_stage
    files:
        frames: "frames.npy"  -- np.save()'d ndarray, dtype uint8, shape [N, H, W, 3]
                                  (N = number of frames queued since last query;
                                   chronological, most-recent last)
        states: "states.npy"  -- np.save()'d ndarray, dtype float32, shape [N, state_dim]
    form data:
        session_id: str
        task: str                    (task_prompt)
        num_anchors: str(int)        (== N, frames.shape[0])
        reset: "1" | "0"
        hysteresis_k: str(int)
        conf_threshold: f"{float:.6f}"
        monotonic: "1" | "0"
    -> client reads JSON response and REQUIRES/USES these keys:
        "gated_stage"        int    (required — client does int(out["gated_stage"]), KeyError if absent)
        "stage_conf"         float  (optional, default kept from previous value via .get())
        "raw_stages"         list[int]  (optional, .get(..., None))
        "server_latency_s"   float  (optional, default 0.0 via .get())
       client ADDS a "client_latency_s" field (wall-clock round trip) before returning.
       On HTTP status >= 400: raises RuntimeError with the server's "error" field or raw text.
```

Source: `reward_client.py:14-31` (protocol docstring), `93-154` (`predict_stage` implementation).

**What I can infer about server behavior from the client code only** (not
verified against server source, per your instruction to skip server-side):
the server must maintain per-`session_id` state across calls (since hysteresis
requires memory of past predictions), must return `gated_stage` as an integer
in `[0, num_stages-1]`, and must accept `reset=1` to clear that per-session
state (used on env `reset()` — see §6). Everything else about *how* the server
computes `gated_stage`/`stage_conf` is outside this trace.

---

## 6. `StageAwarePBRSWrapper` — the actual reward transform

`resfit/rl_finetuning/reward_models/pbrs_wrapper.py`. State held per instance:

```python
self.potentials          # list[float], len == num_stages   (from REWARD_POTENTIALS)
self.num_stages           # len(potentials)
self.gamma                 # pbrs_gamma (algo.gamma fallback)
self.query_every_k         # int
self.keep_sparse_term      # bool — PBRS only
self.hysteresis_k, self.conf_threshold, self.monotonic   # sent to server each call
self.reward_mode           # "pbrs" | "milestone"
self.milestone_payouts     # list[float], len == num_stages
self.milestone_success_bonus  # float
self._last_paid_stage      # int, milestone-only ratchet high-water mark
self._pending_frames, self._pending_states   # list, cleared after each server query
self._step_ctr              # int, counts env steps since last reset
self._current_stage          # int, monotonic-clamped gated stage
self._raw_stage               # int, debug-only pre-hysteresis argmax stage
self._stage_conf               # float
self._phi_current               # float, PBRS-only running potential
```

### 6.1 `_extract(obs)` — frame/state conversion (both modes, identical)

```python
def _extract(obs: dict) -> (frame, state):
    img = obs[image_key]                        # [C, H, W] float32 in [0, 1]  (torch/np, from env obs)
    img = transpose(img, (1, 2, 0))              # -> [H, W, C]
    img = clip(img * 255.0, 0, 255).astype(uint8)  # -> [H, W, C] uint8, [0,255]
    if image_vflip: img = img[::-1].copy()       # both scripts: image_vflip=False, no-op here
    state = obs[state_key].astype(float32).reshape(-1)   # -> [state_dim] float32
    return img, state
```
Source: `pbrs_wrapper.py:112-120`.

### 6.2 `reset()`

```python
def reset(**kwargs):
    obs, info = env.reset(**kwargs)
    pending_frames.clear(); pending_states.clear()
    step_ctr = 0
    current_stage = 0; raw_stage = 0; stage_conf = 1.0
    last_server_latency = 0.0; last_client_latency = 0.0

    img, state = _extract(obs)
    pending_frames.append(img); pending_states.append(state)
    _query(reset=True)                 # forces a server call even with only 1 pending frame,
                                        # and tells server to clear this session_id's hysteresis state
    phi_current = potentials[current_stage]       # PBRS bookkeeping (harmless if mode=milestone)
    last_paid_stage = current_stage                # milestone bookkeeping (harmless if mode=pbrs)
    return obs, info
```
Source: `pbrs_wrapper.py:171-188`. Both `_phi_current` and `_last_paid_stage`
are unconditionally initialized here regardless of `reward_mode` — only one of
the two is actually consumed later, depending on mode.

### 6.3 `_query(reset: bool)` — shared by both modes

```python
def _query(reset):
    if pending_frames is empty: return
    frames = stack(pending_frames, axis=0)   # [N, H, W, 3] uint8
    states = stack(pending_states, axis=0)   # [N, state_dim] float32
    try:
        out = client.predict_stage(frames, states, session_id=..., num_anchors=len(frames),
                                    reset=reset, hysteresis_k=..., conf_threshold=..., monotonic=..., task=...)
    except Exception as e:
        # HARD FAIL — no silent fallback / no stale-reward continuation
        raise RuntimeError(f"Reward server call failed mid-training ({e}). Aborting RL: "
                            "restart the reward server and resume from checkpoint.") from e

    stage = int(out["gated_stage"])
    if monotonic: stage = max(current_stage, stage)      # client-side monotonic re-clamp
                                                            # (defense in depth; server is ALSO told monotonic=1)
    stage = clip(stage, 0, num_stages - 1)
    current_stage = stage
    raw_stage = clip(int(out.get("raw_stages", [current_stage])[-1]), 0, num_stages - 1)  # debug only
    stage_conf = float(out.get("stage_conf", stage_conf))
    last_server_latency = float(out.get("server_latency_s", 0.0))
    last_client_latency = float(out.get("client_latency_s", 0.0))
    pending_frames.clear(); pending_states.clear()
```
Source: `pbrs_wrapper.py:122-166`. Note: a reward-server outage or error
**aborts the entire training run** — there is no degraded/offline mode.

### 6.4 `step(action)` — where the two modes diverge

```python
def step(action):
    obs, reward, terminated, truncated, info = env.step(action)   # reward == r_sparse, the RAW sim reward (0.0 or 1.0)
    r_sparse = float(reward)

    img, state = _extract(obs)
    pending_frames.append(img); pending_states.append(state)
    step_ctr += 1

    # Query cadence — when does a server call actually fire this step:
    #   - every query_every_k steps (both scripts: k=4, so 20Hz sim -> ~5Hz query rate), OR
    #   - always on truncation (flush whatever's pending), OR
    #   - milestone mode ONLY: also on termination (so the final stage is scored
    #     before the success bonus is paid — matches the offline labeler's flush-on-terminal)
    if step_ctr % query_every_k == 0 or truncated or (terminated and reward_mode == "milestone"):
        _query(reset=False)

    # ---- MODE BRANCH ----
    if reward_mode == "milestone":
        r_out = 0.0
        if current_stage > last_paid_stage:
            for s in range(last_paid_stage + 1, current_stage + 1):   # sums SKIPPED stages too
                r_out += milestone_payouts[s]
            last_paid_stage = current_stage
        if terminated and r_sparse > 0.5:
            r_out += milestone_success_bonus       # gated on TRUE sim success flag, NOT model confidence
        r_shaped = r_out                              # (alias; no separate shaping value in this mode)

    else:  # "pbrs"
        phi_next = 0.0 if terminated else potentials[current_stage]
        r_shaped = (r_sparse if keep_sparse_term else 0.0) + gamma * phi_next - phi_current
        phi_current = phi_next
        r_out = float(r_shaped)

    # ---- debug info, both modes ----
    info["reward_model_stage"]              = int(current_stage)
    info["reward_model_raw_stage"]          = int(raw_stage)
    info["reward_model_stage_conf"]         = float(stage_conf)
    info["reward_model_server_latency_s"]   = float(last_server_latency)
    info["reward_model_client_latency_s"]   = float(last_client_latency)
    info["reward_model_r_sparse"]           = r_sparse
    info["reward_model_r_shaped"]           = float(r_shaped)
    info["reward_model_reward"]             = r_out
    info["reward_model_r_milestone"]        = r_out if reward_mode == "milestone" else 0.0
    info["reward_model_last_paid_stage"]    = int(last_paid_stage)

    return obs, r_out, terminated, truncated, info   # <-- r_out FULLY REPLACES the env's reward
```
Source: `pbrs_wrapper.py:190-245`.

**Key semantic differences, stated precisely:**

| | PBRS | Milestone |
|---|---|---|
| Formula | `r = (r_sparse or 0) + γ·φ(stage') − φ(stage)` | `r = Σ payouts[last_paid+1 .. stage] (once) + success_bonus·1[terminated ∧ r_sparse>0.5]` |
| Policy-invariance | Yes — telescopes under n-step, doesn't change optimal policy | No — explicitly a dense/discrete shaping reward, changes the optimal policy by design (it's an ablation) |
| Uses `gamma` | Yes | No |
| Uses `potentials` values | Yes, as the per-stage `φ` table | No — only `len(potentials)` is used, to set `num_stages` |
| Uses `keep_sparse_term` | Yes | No (ignored) |
| Extra terminal query | No (φ(terminal)=0 by convention, no need to re-score) | Yes — always queries on `terminated` so the final stage is captured before paying the bonus |
| Can regress/repay a stage | No — `φ` is re-evaluated fresh every step from `current_stage`, so revisiting a stage just re-emits its (fixed) potential difference | No — `_last_paid_stage` is a ratchet, each stage pays out exactly once for the whole episode |

---

## 7. Vectorization → torch conversion

The wrapped single env is placed inside a `gym.vector.SyncVectorEnv` (forced,
per §4) by `create_vectorized_env()` (`dexmg.py:897+`), then wrapped once more
by `VectorizedEnvWrapper`:

```python
# dexmg.py:831-848
def step(self, actions):
    if isinstance(actions, torch.Tensor): actions = actions.cpu().numpy()
    obs, rewards, terminated, truncated, info = self.vec_env.step(actions)  # numpy, batched over sub-envs
    obs = self._convert_obs_to_torch(obs, device)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)     # <- reward dtype fixed to float32 here
    terminated = torch.tensor(terminated, device=device, dtype=torch.bool)
    truncated  = torch.tensor(truncated,  device=device, dtype=torch.bool)
    return obs, rewards, terminated, truncated, info
```
`rewards` at this point is `r_out` from §6.4, batched into a `[num_envs]`
tensor (`num_envs == 1` here) — no numeric transform happens in this layer,
only dtype/device conversion.

`BasePolicyVecEnvWrapper` sits one layer above this (adds the base-policy +
residual action composition; not reward-relevant, out of scope here) — it's
what `env` refers to in the main loop below.

---

## 8. Main training loop — `env.step()` call site

```python
# train_residual_td3.py:1755-1770, inside `while global_step <= total_timesteps:`
stddev = schedule(cfg.algo.stddev_schedule, global_step)
action = agent.act(obs, eval_mode=False, stddev=stddev)
if progressive_clipping_steps > 0:
    action *= min(1.0, global_step / progressive_clipping_steps)   # both scripts: PROGRESSIVE_CLIPPING=0, no-op

next_obs, reward, terminated, truncated, info = env.step(action)   # reward == r_out from §6.4/§7, unmodified
done = terminated | truncated
```
Source: `train_residual_td3.py:1760-1770`.

Every 50 steps, if `cfg.reward_model.enabled`, the debug info keys from §6.4
are averaged and logged to W&B under `reward_model/*` (`train_residual_td3.py:1777-1793`).
If a training-rollout recorder is active (`SAVE_TRAINING_ROLLOUTS=true` /
`TRAINING_ROLLOUT_PROB=0.2` in both scripts, single-env only), it also consumes
these same `info["reward_model_*"]` keys to annotate recorded frames
(`train_rollout_recorder.py:123-127`) — this is a visualization/debug path,
not part of the reward computation itself.

---

## 9. Building the replay-buffer transition — `_add_transitions_to_buffer()`

```python
# train_residual_td3.py:162-222, called once per env.step() in the main loop
combined_action = info["scaled_action"]   # the ACTUAL executed action (base_policy + residual), NOT the raw residual

def _add_transitions_to_buffer(obs, next_obs, actions, reward, done, info, ..., terminated=None):
    if terminated is None: terminated = done
    for i in range(num_envs):     # num_envs == 1 here
        if done[i] and info["final_obs"][i] is not None:
            next_obs_i = info["final_obs"][i]        # terminal-observation handling (autoreset correctness)
        else:
            next_obs_i = {k: v[i] for k, v in next_obs.items()}
        curr_obs_i = {k: v[i] for k, v in obs.items()}
        # keep only image_keys + lowdim_keys, convert images float->uint8 for storage
        to_uint8(curr_obs_i, image_keys); to_uint8(next_obs_i, image_keys)

        td = TensorDict({
            "obs": TensorDict(curr_obs_i),
            "next": TensorDict({
                "obs": TensorDict(next_obs_i),
                "done": done[i],                 # bool  = terminated[i] | truncated[i]
                "terminated": terminated[i],      # bool  = TRUE absorbing terminal only
                "reward": reward[i],               # float32 scalar tensor — THIS IS r_out, VERBATIM, no further transform
            }),
            "action": actions[i],                   # = combined_action[i] = scaled_action, the executed action
            "_priority": 10.0,                        # fixed high initial priority (for prioritized sampling)
        }).unsqueeze(0)

        online_rb.add(td)
```
Source: `train_residual_td3.py:162-222`, call site `1828-1839+`.

**This is the direct answer to "what gets written into the buffer": exactly
`r_out` from §6.4 — the PBRS-shaped value or the milestone ratchet value,
whichever mode is active — stored as `next.reward`, with no scaling,
clipping, or further transform applied at insertion time.**

---

## 10. At *sampling* time — n-step return computation (both modes, identical mechanism)

Not part of insertion, but this is the "certain kind of manipulation" that
happens to the stored reward before it's used in a Bellman target. Attached as
a replay-buffer transform:

```python
# train_residual_td3.py:750-753 (online_rb) / :854-857 (offline_rb)
transform = MultiStepTransform(
    n_steps=cfg.algo.n_step,                                   # both scripts: N_STEP=3
    gamma=cfg.algo.gamma,                                       # 0.995
    use_terminated_for_bootstrap=cfg.algo.terminated_only_bootstrap,  # both scripts: True
)
```

`MultiStepTransform._multi_step_func` (`rb_transforms.py:243-360`), applied
over a sliding window of `n_steps` consecutive stored transitions:

```python
done = window["next"]["done"]                 # [.., n_steps] bool
if use_terminated_for_bootstrap:
    terminated = window["next"]["terminated"]  # separate signal, required present or KeyError

reward = window["next"]["reward"]              # [.., n_steps], the raw per-step r_out values stored in §9
summed_reward, time_to_obs = _get_reward(gamma, reward, done, n_steps)
    # = r_t + gamma*r_{t+1} + ... up to n_steps, OR fewer if `done` fires early within the window

nonterminal = (time_to_obs == n_steps) & (~future_done_at_horizon)
    # bootstrap mask for the Bellman target: target = summed_reward + gamma^n * nonterminal * Q(s_{t+n})
    # "future_done_at_horizon" is evaluated from `terminated` (not `done`) when
    # use_terminated_for_bootstrap=True — so a TRUNCATED episode still bootstraps,
    # only a TRUE terminal (terminated=True) blocks the Q-bootstrap term.

tensordict.rename_key_(("next","reward"), ("next","original_reward"))   # keeps the raw 1-step reward too
tensordict.set(("next","reward"), summed_reward)                          # overwrites "reward" with the n-step sum
```

This step is **identical** for PBRS and milestone — the transform has no
knowledge of `reward_mode`; it just discount-sums whatever scalar was stored
in `next.reward` per step. `terminated_only_bootstrap=True` is asserted (as a
warning, not a hard error) by both scripts because:
- **PBRS** needs it for the telescoping identity `Σ γ^t(γφ(s_{t+1})-φ(s_t))` to
  collapse correctly to `γ^n φ(s_{t+n}) - φ(s_t)`, which only holds if
  `φ(absorbing)=0` is applied exactly at the true terminal, not at truncation.
- **Milestone** needs it because a truncated (time-limit) episode must still
  bootstrap the Q-value (it's not a real end-of-task), while a true success
  terminal must not (the ratchet + bonus is the entire return from there).

---

## 11. Offline buffer path — no live HTTP calls at RL runtime

Separate from everything above. The offline replay buffer (pre-collected demo
data) does **not** call the reward server while `train_residual_td3.py` is
running. Instead:

1. A **separate, offline** script, `scripts/label_offline_pbrs.py`, is run
   *before* RL training. It streams the demo dataset's frames/states through
   the *same* `RewardModelClient` / `/predict_stage` endpoint, applies the
   *same* PBRS or milestone formula (mirroring §6.4 exactly — see
   `label_offline_pbrs.py:355-365` for the milestone ratchet, replicated from
   `pbrs_wrapper.py:210-222`), and writes a **sidecar parquet** with columns
   `index, episode_index, frame_index, stage, r_sparse, reward_pbrs` (PBRS
   mode) or `reward_milestone` (milestone mode) — source:
   `label_offline_pbrs.py:1-24, 76-113`.
2. It embeds a `labeler_config` JSON blob in the parquet's schema metadata,
   recording every param used (`gamma`, `potentials`, `reward_mode`,
   `milestone_payouts`, `milestone_success_bonus`, `num_stages`,
   `hysteresis_k`, `query_every_k`, `conf_threshold`, `keep_sparse_term`,
   `image_vflip`) — `label_offline_pbrs.py:207-209, 258`.
3. At RL startup, if `offline_data.reward_parquet` is set (both scripts set
   it, pointing at their respective `./pbrs_sarm_progress.parquet` /
   `./milestone_sarm_progress.parquet`), `train_residual_td3.py` reads that
   `labeler_config` metadata and **hard-errors** (`raise ValueError`, training
   aborts) if it's incompatible with the live `cfg.reward_model` config —
   e.g. a milestone parquet feeding a PBRS run, or mismatched
   `milestone_payouts`/`gamma`/`potentials`/`num_stages`. Server-gating knobs
   (`hysteresis_k`, `query_every_k`, `conf_threshold`, `image_vflip`) are only
   warned about, not hard-errored, since they only affect stage *assignment*,
   not reward-formula correctness. Source: `train_residual_td3.py:1080-1158`.
4. During offline-buffer population (`_populate_offline_buffer`,
   `train_residual_td3.py:896+`), the reward for each stored transition is
   resolved by priority: **(1)** the sidecar parquet value, keyed by the
   dataset's global frame `index`, treated exactly like a `next.reward` field;
   else **(2)** the dataset's own dense `next.reward` if `reward_shaping=True`;
   else **(3)** sparse `1.0`/`0.0` from the episode's `done` flag. Source:
   `train_residual_td3.py:1010-1023`. The resulting per-step reward is stored
   into `next.reward` in the offline `TensorDict` (`train_residual_td3.py:1035-1053`)
   the same way as the online path (§9), and goes through the *same*
   `MultiStepTransform` n-step computation (§10) when sampled.

---

## 12. Open items I did not verify (explicitly out of scope / need the server code)

- Anything about how the SARM/TCC server itself computes `gated_stage` /
  `stage_conf` / hysteresis internally — you said to skip this.
- `RewardModelConfig.client_history_len` (`residual_td3.py:112`, default 40)
  is documented in the dataclass but **is never read anywhere in
  `pbrs_wrapper.py`** — the wrapper's own `_pending_frames` buffer just grows
  by 1 per step and is fully drained on every `_query()` call (no fixed-size
  rolling window is enforced client-side). If this field is meant to bound
  memory or match the server's required window, it currently doesn't do
  either on the client side — flagging this rather than assuming intent.
