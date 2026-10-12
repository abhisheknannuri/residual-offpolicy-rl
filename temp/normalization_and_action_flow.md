# Station-3 residual RL: normalization and action flow (verified)

Sources for every number:
- Run `run_2026-10-09_11-00-05_..._seed42`, checkpoint `policy_step_42507`:
  `deployment.normalization` (the exact stats the scalers were built from) and
  `config` (the resolved config the run used).
- `scripts/TrossenStation3Real/RL/conf/station3.yaml`.
- Code at commit `44e8405`. File:line references given per step.

The one thing NOT verified is ACT's own normalization mode (section 2), because
that checkpoint is on the policy-server machine.

Units (`real_action_space: delta_joint`, 20 Hz):
- Joints 0-5: radians. A per-tick DELTA relative to the follower's measured
  joint position at that tick.
- Joint 6 (gripper): ABSOLUTE position. Never delta-converted. Physical range
  `gripper_closed = 0.004`, `gripper_open = 0.04`
  (`trossen_real/configs/trossen_station3_single.yaml:58-59`).

---

## 0. Big picture: three independent normalizations

| # | What | Type | Stats from | Used by |
|---|---|---|---|---|
| 1 | ACT actions/state | lerobot normalizer inside the ACT checkpoint | ACT's training dataset | ACT only, on the policy server |
| 2 | RL actions | **widened min-max** to [-1, 1] (`ActionScaler`) | dataset `action` min/max + `action_scale` + `min_action_range` | RL actor, critic, buffer |
| 3 | RL state | **z-score** (`StateStandardizer`) | dataset `observation.state` mean/std + `min_state_std` | RL actor, critic, buffer |

The policy server always returns the base action in REAL units. The RL side
then normalizes it again with #2. ACT's normalized numbers never reach RL.

---

## 1. Dataset

1. Teleop records ABSOLUTE joint targets `action_abs[t]` and joint state `q[t]`.
2. `convert_to_delta_joint_dataset.py` rewrites the action column:
   - `action[t][:6] = action_abs[t][:6] - q[t][:6]` (each row against its OWN state)
   - `action[t][6]  = action_abs[t][6]` (gripper stays absolute)
3. The dataset's `meta` stats are computed on this delta action column.

## 2. ACT (base policy): normalization #1

- lerobot trains ACT on the delta dataset. The normalizer and unnormalizer are
  saved inside the ACT checkpoint
  (`policy_preprocessor_step_4_normalizer_processor.safetensors`,
  `policy_postprocessor_step_0_unnormalizer_processor.safetensors`).
- **NOT VERIFIED:** which mode. lerobot's ACT default is `MEAN_STD` for STATE
  and ACTION. If so: ACT's network outputs `z = (a - mean) / std`, which is
  NOT bounded to [-1, 1], and the server unnormalizes with `a = z * std + mean`.
  Confirm in `config.json -> normalization_mapping` at
  `.../ACT_BC_PickCubeAndInsert_Station3_Trial1_deltajoint/checkpoints/050000/pretrained_model/`
  on the server machine.
- Whatever the mode, the server returns `"action"` in REAL units and RL uses
  that (`PolicyClient.predict()`, `trossen_real/inference/policy_client.py:116`).
  `"action_normalized"` is also returned but RL never reads it.

So: **`a_base_real` = real units (rad delta for 0-5, absolute gripper).**

---

## 3. RL ActionScaler: normalization #2, step by step

Code: `resfit/rl_finetuning/utils/normalization.py:76-91` (build),
`:136-153` (scale), `:155-172` (unscale).
Built with `action_scale = agent.actor.action_scale` and
`min_range_per_dim = offline_data.min_action_range`
(`train_residual_td3_distributed.py:561-566`).

### Inputs

From the dataset (`action` stats, delta units):

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 (grip) |
|---|---|---|---|---|---|---|---|
| `min` | -0.06867 | -0.17433 | -0.16899 | -0.12932 | -0.06065 | -0.23308 | 0.00400 |
| `max` | +0.07858 | +0.10910 | +0.12207 | +0.11292 | +0.06752 | +0.19493 | 0.04000 |

From `station3.yaml`:
- `agent.actor.action_scale = [0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.1]` (call it `s`)
- `offline_data.min_action_range = 0.1`

### Step 1: center

```
mid = (min + max) / 2
```
dim 0: (-0.06867 + 0.07858) / 2 = **0.00496**

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `mid` | +0.00496 | -0.03262 | -0.02346 | -0.00820 | +0.00343 | -0.01907 | +0.02200 |

### Step 2: half range

```
half = (max - min) / 2
```
dim 0: (0.07858 - (-0.06867)) / 2 = **0.07362**

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `half` | 0.07362 | 0.14172 | 0.14553 | 0.12112 | 0.06409 | 0.21401 | 0.01800 |

### Step 3: floor the half range

```
half = max(half, min_action_range / 2) = max(half, 0.1 / 2) = max(half, 0.05)
```
Only the gripper is below 0.05: 0.01800 becomes **0.05000**. Every other dim is unchanged.

Why it exists: a dim that barely varies in the data would get a tiny range, and
dividing by it would turn small real changes into huge normalized values. Side
effect here: the gripper box is now ~2.8x wider than its physical range.

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `half` (floored) | 0.07362 | 0.14172 | 0.14553 | 0.12112 | 0.06409 | 0.21401 | **0.05000** |

### Step 4: widen by action_scale

```
half_exp = half * (1 + s)
```
`s` is `action_scale` i.e `action_scale: [0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.1]`

dim 0: 0.07362 * 1.15 = **0.08467**. dim 3: 0.12112 * 1.05 = **0.12717**. dim 6: 0.05 * 1.10 = **0.05500**.

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `1 + s` | 1.15 | 1.15 | 1.15 | 1.05 | 1.05 | 1.05 | 1.10 |
| `half_exp` | 0.08467 | 0.16298 | 0.16736 | 0.12717 | 0.06729 | 0.22471 | 0.05500 |

`half_exp` is the most useful number in this file: **1.0 in normalized
space = `half_exp` in real units**, per dim.

### Step 5: the box limits

```
L_min = mid - half_exp
L_max = mid + half_exp
```
dim 0: 0.00496 - 0.08467 = **-0.07971**, 0.00496 + 0.08467 = **+0.08963**

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `L_min` | -0.07971 | -0.19559 | -0.19082 | -0.13538 | -0.06386 | -0.24378 | -0.03300 |
| `L_max` | +0.08963 | +0.13036 | +0.14390 | +0.11897 | +0.07073 | +0.20563 | +0.07700 |

### Step 6: the two functions

```
scale(a)   = 2 * (clamp(a, L_min, L_max) - L_min) / (L_max - L_min) - 1
           = (clamp(a, L_min, L_max) - mid) / half_exp                 -> in [-1, 1]

unscale(n) = L_min + (clamp(n, -1, 1) + 1) * (L_max - L_min) / 2
           = mid + clamp(n, -1, 1) * half_exp                          -> real units
```
Both clamp. Anything outside the box is cut to its edge.

### Step 7: where the dataset itself lands

```
scale(dataset max) = (max - mid) / half_exp = half / half_exp = 1 / (1 + s)   (dims 0-5)
```
dim 0: 0.07362 / 0.08467 = **0.86957**. dim 3: 0.12112 / 0.12717 = **0.95238**.
Gripper: physical 0.04 -> (0.04 - 0.022) / 0.055 = **0.32727**.

| dim | 0-2 | 3-5 | 6 |
|---|---|---|---|
| dataset range in normalized | ±0.86957 | ±0.95238 | ±0.32727 |
| headroom to ±1 = `1 - 1/(1+s)` | 0.13043 | 0.04762 | 0.67273 |
| residual cap (`s`) | 0.15 | 0.05 | 0.10 |

### Why widen, instead of plain min-max or z-score

What the code requires (verified, not opinion):
- The actor's residual is `tanh(...) * s`, so it lives in a hard box `[-s, s]`
  (`actor.py:28`, `:181`).
- Sampling noise goes through `TruncatedNormal(low=-1, high=1)`
  (`off_policy/common_utils/utils.py:155`).
- Combined action is `clamp(base_n + residual_n, -1, 1)` everywhere: env
  (`real_residual_env.py:266`), actor loss (`q_agent.py:443`), critic target
  (`q_agent.py:329`). `unscale()` clamps to ±1 again.

So the RL action space must be a bounded box with fixed edges. **z-score**
gives no fixed edges (values are unbounded), so "±1" would mean nothing
physically. **Min-max** gives a box, but with plain min-max the dataset
extremes sit exactly at ±1.

The widening is NOT what keeps the combined action inside [-1, 1]; the
`clamp(-1, 1)` already does that whatever the stats are. What the widening does:

1. **Leaves room for the residual at the edges.** With plain min-max, a base
   action at the dataset's extreme is already at ±1. Any residual pushing
   further would be clamped to 0 effect. With widening, the extreme sits at
   ±0.870 (dims 0-2), so there is 0.130 of room left.
2. **Defines the largest physical step the system can ever command.** Because
   `unscale` clamps to ±1, the follower's per-tick delta is limited to
   `[L_min, L_max]`. The intervention path uses the same box to clamp the
   human's delta (`real_residual_env.py:288-296`).

It is NOT a full guarantee of room: headroom (0.130 / 0.048) is slightly
smaller than the residual cap (0.15 / 0.05). At the very edge of the data, up to
13% of a full residual can still be clamped away.

### Residual and noise in real units

Real = normalized * `half_exp`.

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| residual cap normalized (`s`) | 0.15 | 0.15 | 0.15 | 0.05 | 0.05 | 0.05 | 0.10 |
| residual cap real (`s * half_exp`) | 0.01270 | 0.02445 | 0.02510 | 0.00636 | 0.00336 | 0.01124 | 0.00550 |
| noise std normalized (run 2026-10-09, bug) | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| noise std normalized (station3.yaml now) | 0.015 | 0.015 | 0.015 | 0.03 | 0.03 | 0.03 | 0.0125 |
| noise std real (station3.yaml now) | 0.00127 | 0.00244 | 0.00251 | 0.00382 | 0.00202 | 0.00674 | 0.00069 |
| noise std real (`0.05 * half_exp`) | 0.00423 | 0.00815 | 0.00837 | 0.00636 | 0.00336 | 0.01124 | 0.00275 |

Joint 4 example: the actor can move it at most 0.00336 rad (0.19°) per tick
beyond BC, and the exploration noise std is the same size as that cap.

---

## 4. RL StateStandardizer: normalization #3, step by step

Code: `normalization.py:225-290`. Built with
`min_std = offline_data.min_state_std = 0.1`.

This is a z-score, the same TYPE as lerobot's default MEAN_STD, but a separate
object with its own floor. Stats are the dataset's `observation.state`
(absolute joint positions, NOT delta). Nothing is widened or changed apart from the floor.

### Inputs

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `mean` | -0.17692 | +1.82664 | +1.36511 | -1.06004 | -0.00462 | -0.13981 | +0.02600 |
| `std` | 0.13621 | 0.25094 | 0.28547 | 0.25456 | 0.02730 | 0.24426 | 0.01316 |

### Step 1: floor std

```
std = max(std, min_std) = max(std, 0.1)
```
Dim 4: 0.02730 -> **0.1**. Dim 6: 0.01316 -> **0.1**. Others unchanged.

### Step 2: standardize

```
state_n = (q - mean) / std
```
Dim 4 and 6 are therefore squeezed about 3.7x and 7.6x compared with a true
z-score. A one-std move on the gripper only changes `state_n[6]` by ~0.13.

`state_n` is what is stored in the buffer as `observation.state` and fed to
the actor and the critic. ACT never sees it; ACT gets raw `q` through its own
normalizer.

---

## 5. Who sees what, in which space

| Quantity | Space | Shown to |
|---|---|---|
| `observation.state` | z-score (#3) | actor, critic |
| images | uint8 in buffer, /255 to [0,1] in `_encode` (`q_agent.py:226`) | ViT encoder -> actor, critic |
| `observation.base_action` = `base_n` | normalized #2, [-1, 1] | **actor only** |
| `residual_n` | normalized #2, [-s, s] (+ noise when sampling) | added to `base_n` |
| `combined_n` = `clamp(base_n + residual_n, -1, 1)` | normalized #2 | **critic** (as its action input) |
| buffer `action` | normalized #2 | critic |
| executed command | REAL units, absolute joint target | robot |

The critic's input is `[image features, state_n, action_n]`
(`critic.py:418-437`, `use_prop: 1`). It does NOT get `base_n` separately.
Nothing RL-side ever sees real-unit actions.

---

## 6. One full tick, with formulas

`real_residual_env.py:241-322`.

```
1. a_base_real = server.predict(raw obs)                      # real units
2. base_n      = scale(a_base_real)                           # #2 -> [-1, 1]; this is obs["observation.base_action"]
3. state_n     = (q - mean) / max(std, 0.1)                   # #3
4. mu          = tanh(MLP([vit(images), state_n, base_n]))   # each dim in [-1, 1]
   mean_n      = mu * s
                 # s = [0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.10]
                 # mean_n in [-0.15,+0.15] x3, [-0.05,+0.05] x3, [-0.10,+0.10]
                 # real:  ±0.01270, ±0.02445, ±0.02510, ±0.00636, ±0.00336, ±0.01124, ±0.00550
   residual_n  = clamp(mean_n + N(0, stddev), -1, 1)          # training rollout
                 # stddev = 0.05 every dim (this run); no stddev_clip here
                 # 99.73% of samples (±3σ = ±0.15) fall in:
                 #   [-0.30,+0.30] x3, [-0.20,+0.20] x3, [-0.25,+0.25]
                 # real:  ±0.02540, ±0.04889, ±0.05021, ±0.02543, ±0.01346, ±0.04494, ±0.01375
                 # eval: residual_n = mean_n (no noise)
5. combined_n  = clamp(base_n + residual_n, -1, 1)
6. env_action  = unscale(combined_n) = mid + combined_n * half_exp      # real delta
7. exec[:6]    = q_before[:6] + env_action[:6]                # absolute joint target
   exec[6]     = clip(env_action[6], 0.004, 0.04)             # absolute gripper
8. buffer "action" = combined_n                                # autonomous
```

Intervened tick replaces 4-8:
```
delta    = clip(leader[:6] - q_before[:6], L_min[:6], L_max[:6])
exec[:6] = q_before[:6] + delta
exec[6]  = clip(leader[6], 0.004, 0.04)
buffer "action" = scale([delta, exec[6]])
```
The residual the actor proposed on this tick is not stored in the buffer. Only
`info["actor_proposed_action"]` and `real_log_file` have it.

### Worked example: buffer row 30000 (autonomous)

Values read from the buffer, then converted with the tables above.

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `base_n` (stored) | -0.02101 | +0.26617 | +0.06117 | +0.05435 | -0.10760 | +0.09767 | -0.32765 |
| `action` = `combined_n` (stored) | -0.07656 | +0.28451 | -0.04949 | +0.09523 | -0.02631 | +0.17910 | -0.29091 |
| `residual_n` = action - base | -0.05554 | +0.01834 | -0.11065 | +0.04088 | **+0.08129** | **+0.08143** | +0.03674 |
| `base_real` = mid + base_n * half_exp | +0.00318 | +0.01076 | -0.01322 | -0.00129 | -0.00381 | +0.00287 | 0.00398 |
| `action_real` = mid + action * half_exp | -0.00152 | +0.01375 | -0.03174 | +0.00391 | +0.00166 | +0.02117 | 0.00600 |
| `residual_real` = residual_n * half_exp | -0.00470 | +0.00299 | -0.01852 | +0.00520 | +0.00547 | +0.01830 | +0.00202 |

Dim 0 by hand: `base_real = 0.00496 + (-0.02101)(0.08467) = 0.00318`.

Dims 4 and 5: residual 0.081 is above the 0.05 cap. That excess is the
N(0, 0.05) exploration noise (the cap only bounds `mu * s`).

State for the same row: stored `state_n[0] = -0.70219`, so
`q[0] = -0.70219 * 0.13621 + (-0.17692) = -0.27256` rad, and the commanded
absolute target is `q[0] + action_real[0] = -0.27256 - 0.00152 = -0.27409`.

---

## 7. Noise: every knob, where it is used, and the actual numbers

All noise values are in NORMALIZED action space (same units as `s`).
Real units = normalized * `half_exp` (section 3, step 4).

### 7.1 The one stddev that is actually used: `stddev_schedule`

```
stddev = utils.schedule(cfg.algo.stddev_schedule, step)
```

`stddev_schedule` is an OmegaConf interpolation, in both the dataclass
(`config/rlpd.py`) and station3.yaml:

```
stddev_schedule: "linear(${.stddev_max},${.stddev_min},${.stddev_step})"
```

It is resolved when read, i.e. after Hydra has merged the yaml and CLI
overrides, so it always reflects the final `stddev_max/min/step`.

```
stddev(t) = (1 - mix) * stddev_max + mix * stddev_min,   mix = clip(t / stddev_step, 0, 1)
```

- `stddev_max` and `stddev_min` may each be a scalar or a 7-element list.
  `utils.schedule` parses the arguments with `ast.literal_eval`, so lists inside
  `linear(...)` work.
- Current station3.yaml: max = min = `[0.015,0.015,0.015,0.03,0.03,0.03,0.0125]`,
  `stddev_step: 300000`. Constant; `stddev_step` has no effect while max == min.
- Decay example: `stddev_max: 0.05`, `stddev_min: 0.01`, `stddev_step: 50000`.
- To bypass the three keys, replace the interpolation with a literal
  (`"[0.01, ...]"` or `"linear(0.05,0.01,50000)"`).
- Not part of either buffer cache key; it is part of the actor/learner
  config fingerprint, so both machines must use the same yaml.
- Tests: `resfit/rl_finetuning/off_policy/common_utils/tests/test_stddev_schedule.py`.

History: before 2026-10-12, `stddev_schedule` was computed in
`RLPDAlgoConfig.__post_init__` from the dataclass DEFAULTS before the yaml was
merged, so `stddev_max/min/step` were silently ignored and every run used
`linear(0.05,0.05,300000)` = a flat **0.05 on all 7 dims**. That includes
`run_2026-10-09_11-00-05` and `run_2026-10-02_12-37-47` (confirmed in their
checkpoint configs), and every Station-1 / sim script that set
`algo.stddev_max/min` on the command line. Sections 7.2 and 6 use those
0.05 numbers because they describe that run's buffer.

This single value is used in THREE places:

| Where | Code | Noise | Clip on the noise |
|---|---|---|---|
| Rollout (exploration) | `train_residual_td3_distributed.py:2815-2816` -> `agent.act(stddev=...)` | `N(0, stddev)` | none (`clip=None`) |
| Critic target (target policy smoothing) | `train_residual_td3_distributed.py:2106, 2167` -> `update_critic` (`q_agent.py:318-323`) | `N(0, stddev)` on the target actor's residual | ±`stddev_clip` |
| Critic warm-up / offline critic updates | `train_residual_td3_distributed.py:1684, 1793` | same as critic target | ±`stddev_clip` |

The actor loss uses `stddev=0` (`q_agent.py:432`), so the actor is trained on
its deterministic output `mu * s`.

### 7.2 Rollout range of the residual

```
residual_n = clamp(mu * s + N(0, stddev), -1, 1)       # TruncatedNormal.sample(clip=None)
```

- `mu * s` is hard-bounded to `[-s, +s]` (tanh).
- The Gaussian is unbounded. ±3σ covers 99.73% of samples per dim, so 0.27% of
  samples per dim fall outside. Over 7 dims and tens of thousands of steps that
  is thousands of samples, so values beyond `s + 3σ` do appear in the buffer
  (measured max |residual| on dim 2: 0.34 = 0.15 + 3.8σ).
- The final `clamp(-1, 1)` makes `residual_n` ∈ [-1, 1] guaranteed. With
  `s ≤ 0.15` and σ = 0.05, reaching ±1 needs a 17σ sample, so in practice
  this clamp never fires. After that, `combined_n = clamp(base_n + residual_n, -1, 1)`
  guarantees the combined action is in [-1, 1].

With σ = 0.05 (this run), 99.73% of residuals lie in `[-(s + 0.15), +(s + 0.15)]`:

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `s` (actor cap) | 0.15 | 0.15 | 0.15 | 0.05 | 0.05 | 0.05 | 0.10 |
| 3σ | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| `s + 3σ` (normalized) | 0.30 | 0.30 | 0.30 | 0.20 | 0.20 | 0.20 | 0.25 |
| `s + 3σ` (real) | 0.02540 | 0.04889 | 0.05021 | 0.02543 | 0.01346 | 0.04494 | 0.01375 |
| 3σ / `s` | 1.0 | 1.0 | 1.0 | **3.0** | **3.0** | **3.0** | 1.5 |

The last row is the problem: on the wrist joints the noise reaches 3x anything
the actor itself can output, so most of what the critic sees on those dims is noise.

With the per-joint values `[0.015,0.015,0.015,0.03,0.03,0.03,0.0125]`:

| dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 3σ | 0.045 | 0.045 | 0.045 | 0.090 | 0.090 | 0.090 | 0.0375 |
| `s + 3σ` (normalized) | 0.195 | 0.195 | 0.195 | 0.140 | 0.140 | 0.140 | 0.1375 |
| `s + 3σ` (real) | 0.01651 | 0.03178 | 0.03264 | 0.01780 | 0.00942 | 0.03146 | 0.00756 |
| 3σ / `s` | 0.30 | 0.30 | 0.30 | **1.80** | **1.80** | **1.80** | 0.375 |

Joints 0-2 and the gripper become sensible. Wrist joints 3-5 are still
noise-dominated (σ 0.03 against a cap of 0.05). σ ≈ 0.01 on dims 3-5 would
give 3σ / s = 0.6, in line with the others.

### 7.3 `stddev_clip` (agent.stddev_clip = 0.3)

What it does (`TruncatedNormal.sample`, `utils.py:170-182`):

```
eps = N(0, stddev)
eps = clamp(eps, -stddev_clip, +stddev_clip)     # only when clip is passed
x   = clamp(mean + eps, -1, 1)
```

It caps the NOISE sample, not the action. It is passed only in the critic
target (and the actor loss, where stddev = 0 makes it irrelevant). At rollout
it is not passed, so exploration noise is never clipped by it.

This is TD3's "target policy smoothing": the target `Q(s', a')` is averaged
over a small neighborhood of the target actor's action, so the critic cannot
exploit a narrow spike in Q. The clip keeps that neighborhood bounded.

How it relates to stddev: it is meant to be a small multiple of the target
noise σ. In the TD3 paper (actions in [-1, 1]) σ = 0.2 and clip = 0.5, i.e.
clip ≈ 2.5σ.
- Now: σ = 0.05, clip = 0.3 = **6σ**. A 6σ sample essentially never happens,
  so the clip is inactive and the target noise is a plain N(0, 0.05).
- It is a single scalar (`stddev_clip: float`, `config/rlpd.py:102`), not per joint.
- A reasonable choice is 2-3x the LARGEST σ in `stddev_schedule`. For
  `[0.015,...,0.03,...,0.0125]` that is 0.06-0.09. It should also stay at or
  below the residual cap `s` on the dims you care about, otherwise the critic
  target averages over actions the actor can never output.
- Because the same `stddev_schedule` drives exploration AND target smoothing,
  changing it changes both.

### 7.4 Other noise / std settings in station3.yaml

| Key | Value | Used? | What it is |
|---|---|---|---|
| `algo.stddev_schedule` | `linear(${.stddev_max},${.stddev_min},${.stddev_step})` | **yes** | sections 7.1-7.3 |
| `algo.stddev_max/min/step` | `[0.015..]`, `[0.015..]`, 300000 | **yes** (since 2026-10-12 fix) | feed `stddev_schedule`, see 7.1 |
| `agent.stddev_clip` | 0.3 | yes, critic target only | 7.3 |
| `agent.target_action_noise` | true | yes | false = target actor uses `mu * s` with no noise |
| `algo.random_action_noise_scale` | `[0.15,0.15,0.15,0.05,0.05,0.05,0.1]` | warm-up only | uniform, not Gaussian: `residual_n = U(-1, 1) * scale`, so residual ∈ [-scale, +scale] exactly. Mean \|residual\| = scale/2 |
| `agent.actor.max_action_norm` | -1 | no (disabled) | would cap the L2 norm of the sampled action |
| `agent.critic.loss.sigma` | -1 | no | only for `hl_gauss` loss; this run uses `mse` |
| `offline_data.min_state_std` | 0.1 | yes | state normalization floor (section 4), not noise |
| `offline_data.min_action_range` | 0.1 | yes | action scaler floor (section 3), not noise |

---

## 8. Gripper: stored vs executed (fixed 2026-10-12)

Physical gripper [0.004, 0.04] = normalized [-0.32727, +0.32727] (section 3, step 7).
Joints 0-5 have only ONE clip (±1, applied identically to stored and executed;
the follower's `goal_joint` only clips the gripper), so they never mismatch.
The gripper has a SECOND, narrower clip (physical range) that used to apply
only on the way to the robot.

The bug (runs up to 2026-10-12): on autonomous ticks the buffer stored
`combined_n = clamp(base_n + residual_n, -1, 1)`, taken BEFORE the physical
gripper clip. In run_2026-10-09_11-00-05, **39.0% of autonomous transitions**
had |action[6]| > 0.32727 (max 0.595), executed as 0.004 / 0.04 but stored as
something else. Half of those came from ACT itself overshooting (base gripper
range 0.0006-0.0415 m), half from the residual pushing an edge value further.
Intervened transitions were never affected.

The fix, two parts:
1. Env stores what was executed: autonomous ticks use
   `stored_n = _to_stored_naction(executed_action, follower_state_before)`,
   the same conversion as intervened ticks.
2. One executable box everywhere. `ActionScaler.normalized_bounds(gripper_bounds)`
   returns ±1 on joints and `scale(gripper_closed/open)` on dim 6 (computed
   from the scaler stats and station config, never hardcoded). It is used by:
   - the env's `combined_n` (rollout)
   - `QAgent._combine()` in the critic target, the actor loss and the offline
     actor loss, set via `agent.set_action_bounds(...)` in both trainers when
     `real_hardware`. Sim runs keep plain ±1.

Updated tick formulas (section 6, steps 5 and 8):
```
combined_n = clip(base_n + residual_n, low, high)    # low/high = [-1 x6, -0.32727], [+1 x6, +0.32727]
buffer "action" = scale([exec[:6] - q_before[:6], exec[6]])   # = combined_n, gripper now always in range
```

Tests: `trossen_real/inference/tests/test_executed_action_storage.py` (fails on
the old env code for every out-of-range gripper case).

Existing buffers are unchanged: the online buffer of run_2026-10-09 still holds
the out-of-range values. The offline buffer's `action` was never affected
(dataset gripper is exactly 0.004-0.04).

---

## 9. What the buffer holds (per row)

| key | value |
|---|---|
| `obs/observation.state` | `state_n` at t |
| `obs/observation.base_action` | `base_n` at t (the one actually added) |
| `obs/observation.images.*` | 84x84 uint8 |
| `action` | section 6 step 8 (or the intervened formula) |
| `next/obs/*` | same at t+k, k = `steps_to_next_obs` |
| `next/reward` | n-step sum `r_t + γ r_{t+1} + γ² r_{t+2}` |
| `next/original_reward` | 1-step reward |
| `next/done`, `next/terminated` | `buffer_done` (reset pedal, truncation, or reward ≥ 1) |
| `gamma` | γ^k |
| `nonterminal` | false if the window hit a done |
| `intervened` | flag only, never read in training |

n = 3, γ = 0.9975. Within an episode k = 3. The last two rows before a done
have k = 2 and k = 1 (e.g. idx 1347/1348/1349 -> 3/2/1).

## 10. Losses

`q_agent.py:302-336, 425-450, 781-803`.

- Critic: `y = next.reward + gamma * nonterminal * min_2heads Q_target(s_{t+k}, clamp(base_n' + pi_target(s') + clip(N(0,0.05), ±0.3), -1, 1))`;
  regress `Q(s_t, action)` towards `y`.
- Actor: `loss = -Q(s, clamp(base_n + mu * s, -1, 1))` (no noise).
  `action_l2_reg_weight = 0`, `bc_loss_coef = 0`. No term uses `intervened`.
