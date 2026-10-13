# Residual drift in delta-joint control: the math, with measured numbers

Station 3, PickCubeAndInsert. Date 2026-10-13.

This document proves, with equations and measured data, how the residual RL
actor moves the arm when ACT runs in the `delta_joint` action space, and how
large that effect is. Every number comes from one of the sources in section 0.
Nothing is assumed. The one place where a model is fitted to data
(section 7) states the fit and its R².

---

## 0. Sources

| Id | Source | What it gives |
|---|---|---|
| D | `trossen_real/datasets/PickCubeAndInsert_Station3_Trial1_deltajoint_old`, all 178 episodes, 55,177 frames, read from the parquet files | dataset action statistics |
| C | checkpoint `runs_13Oct2026/run_2026-10-13_13-21-22.../models/policy_step_70016/checkpoint.pt` | `config` (action_scale, noise, ...), `deployment.normalization` (RL scaler stats) |
| E-ACT | `logs/infer_*.jsonl` of every ACT-only eval run (matched to `eval_runs/index.jsonl`): 179 runs, 9 ACT checkpoints, 80,474 ticks | ACT's normalized and real outputs |
| E-RL | `logs/residual/residual_*_20261013_135955.jsonl` and `..._143037.jsonl` (residual client, per tick: base action, residual, combined, action), joined to the infer logs of the same 32 eval runs of `policy_step_70016` | residual per tick + measured joint positions |
| Code | file:line references below | how each quantity is computed |

The join of E-RL to the infer logs is done tick by tick in order, keeping only
pairs whose real action matches exactly: 6,828 of 6,832 ticks matched,
maximum action difference 0.0.

Units: joints 0-5 in radians (per-tick delta), joint 6 = gripper position in
metres (absolute, never delta). Control rate 20 Hz. ACT: `chunk_size 20`,
`n_action_steps 14`.

---

## 1. Short answers to the questions

**Q1. "ACT can give any values between −1 and +1?"**
No. ACT has its own normalization, a mean/std z-score, which is not bounded.
It was measured at eval between −7.9 and +6.0 (E-ACT). The policy server
converts ACT's output back to real units before returning it. The RL code
never sees ACT's normalized numbers. Proof in section 2.

**Q2. "ACT gives data in real units, right?"**
Yes. `PolicyClient.predict()` returns the server's `"action"` field, which is
in real units: rad per tick for joints 0-5, metres for the gripper
(`trossen_real/inference/policy_client.py:116`).

**Q3. "Is action_scale applied to the normalized ones?"**
Yes, but not ACT's normalized ones. The RL code re-normalizes ACT's real-unit
action with its own min-max scaler (`ActionScaler`, section 3) into its own
[−1, 1] space. The residual `mu · action_scale` lives in that space.

**Q4. "action_scale extends the dataset stats to support the residual?"**
Yes. The RL scaler's half range is the dataset half range (floored at 0.05)
times `(1 + action_scale)`, per joint
(`resfit/rl_finetuning/utils/normalization.py:81-85`). Numbers in section 3.

**Q5. "Is the scaling applied to both the observation's base action and the
transition's action in the buffer?"**
Yes, with the same scaler:
- Online: `obs.observation.base_action = scale(ACT real)`
  (`trossen_real/rl/real_residual_env.py:530, 535`); `action =
  scale(executed − q_before)` for both autonomous and intervened ticks
  (`real_residual_env.py:334, 345, 547-549`).
- Offline: `observation.base_action = scale(ACT real)` and `action =
  scale(dataset GT action)` (`train_residual_td3_distributed.py:1166, 1181,
  1201, 1275`).

**Q6. "Are we doing something wrong?"**
The conversions are correct, proven exact on 6,828 real ticks (section 4).
What is wrong is the size of the residual. In this action space
`action_scale` is a fraction of the dataset's extreme delta range, which is
10-40 times the typical per-tick step. So `action_scale = 0.15` lets the
residual reach 1-3.3 times ACT's typical step on every tick (section 5).
Combined with delta control, which does not undo past deviations
(section 6), and ACT's 14-step open-loop chunks, the residual moves the arm
as much as ACT does (sections 7-8).

**Q7. "With s = 0.1, the range becomes [−1.1, 1.1]? BC 0.05 + RL 0.1 = 0.15?"**
- The range does not become [−1.1, 1.1]. The box is widened in real units by
  `(1 + s)`, and the normalized range stays [−1, 1]. What changes is that the
  dataset's extremes now sit at ±1/(1+s) instead of ±1.
- The sum 0.05 + 0.1 = 0.15 is correct in normalized units, but normalized 0
  is not "no motion". Real zero sits at `−mid/half_exp`, e.g. +0.200 for
  joint 1. So "0.05 normalized" is not a small real motion.

  With real numbers (section 5.3): ACT's typical step plus the full residual
  gives 2.0-4.3 times ACT's step, per joint, on every tick.

---

## 2. ACT's normalization (measured, not assumed)

For every ACT-only eval tick, the infer log stores both `policy_action_normalized`
(ACT network output) and `policy_action_real` (what the server returns).
Fitting `real = α + β · normalized` per joint, separately for each of the 9
ACT checkpoints (E-ACT):

- fit error ≤ 1.5 × 10⁻⁸ for every checkpoint
- α and β are identical for all checkpoints and equal the dataset's action
  mean and standard deviation (D) to every printed digit:

| joint | 0 | 1 | 2 | 3 | 4 | 5 | 6 (grip) |
|---|---|---|---|---|---|---|---|
| α = dataset mean μ | −0.000558 | +0.001915 | −0.010083 | +0.002089 | −0.000408 | +0.000194 | +0.022737 |
| β = dataset std σ | 0.013650 | 0.036591 | 0.035802 | 0.026866 | 0.007004 | 0.026473 | 0.016733 |

So ACT's normalization is

    z = (a_real − μ) / σ         (network space, unbounded)
    a_real = μ + σ · z           (what the server returns)

Measured range of z at eval: min [−4.11 −4.04 −3.41 −4.22 −4.27 −7.91 −1.25],
max [3.23 2.36 3.32 3.16 3.71 6.01 1.17]. Not bounded to [−1, 1].

---

## 3. The RL action scaler, step by step

Inputs:
- dataset action min/max per joint (D)
- `agent.actor.action_scale` = s = [0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.10] (C)
- `offline_data.min_action_range` = 0.1 (C)

Formulas (`normalization.py:76-91, 136-172`):

    mid      = (max + min) / 2
    half     = max( (max − min) / 2 , min_action_range / 2 )
    half_exp = half · (1 + s)
    L_min    = mid − half_exp ,  L_max = mid + half_exp
    scale(a)   = (clamp(a, L_min, L_max) − mid) / half_exp       ∈ [−1, 1]
    unscale(n) = mid + clamp(n, −1, 1) · half_exp

Values (C, recomputed from D):

| joint | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| dataset min | −0.06867 | −0.17433 | −0.16899 | −0.12932 | −0.06065 | −0.23308 | 0.00400 |
| dataset max | +0.07858 | +0.10910 | +0.12207 | +0.11292 | +0.06752 | +0.19493 | 0.04000 |
| mid | +0.00496 | −0.03262 | −0.02346 | −0.00820 | +0.00343 | −0.01907 | +0.02200 |
| half (before floor) | 0.07362 | 0.14172 | 0.14553 | 0.12112 | 0.06409 | 0.21401 | 0.01800 → 0.05 |
| 1 + s | 1.15 | 1.15 | 1.15 | 1.05 | 1.05 | 1.05 | 1.10 |
| **half_exp** (real units per 1.0 normalized) | 0.08467 | 0.16298 | 0.16736 | 0.12717 | 0.06729 | 0.22471 | 0.05500 |
| L_min | −0.07971 | −0.19559 | −0.19082 | −0.13538 | −0.06386 | −0.24378 | −0.03300 |
| L_max | +0.08963 | +0.13036 | +0.14390 | +0.11897 | +0.07073 | +0.20563 | +0.07700 |
| real 0 in normalized units (−mid/half_exp) | −0.0586 | **+0.2001** | **+0.1402** | +0.0645 | −0.0510 | +0.0849 | −0.4000 |

**ACT's real outputs inside this box (E-ACT, 80,474 ticks):**
- fraction outside [L_min, L_max]: 0 on every joint, so `scale()` never clips
  ACT
- in RL-normalized units ACT spans min [−0.80 −0.78 −0.68 −0.78 −0.52 −0.90
  −0.36], max [0.53 0.75 0.79 0.83 0.37 0.84 0.35]
- median |n| = [0.076 0.278 0.079 0.066 0.050 0.081 0.327]

---

## 4. The residual and the combined action (exact)

Actor (`actor.py:28, 181`): `mu = tanh(MLP(...)) ∈ [−1, 1]`, residual
`r = mu · s` (eval: no noise), so `|r_j| ≤ s_j`.

Deployment (`residual_client.py:360`) and training rollout both form

    n_b = scale(a_ACT)                     ACT action in RL-normalized units
    n_c = clamp(n_b + r, −1, 1)            combined (training also clips the gripper to its physical range)
    a_cmd = unscale(n_c)                   real-unit delta sent to the robot

When the clamp is inactive:

    a_cmd − a_ACT = (mid + (n_b + r)·half_exp) − (mid + n_b·half_exp) = r · half_exp      (1)

So **the residual's real-unit contribution per tick is exactly
ρ = r · half_exp**, independent of `mid`.

**Measured on E-RL (6,828 ticks):**
- `max |scale(a_ACT) − n_b| = 0.0`
- `max |unscale(n_c) − a_cmd| = 0.0`
- the clamp to ±1 was inactive on 100% of ticks
- `max |(a_cmd − a_ACT) − r·half_exp| = 4.6 × 10⁻⁸` (float32 rounding)

Equation (1) holds on real data.

Per-tick cap, `s · half_exp` (real units):

| joint | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| residual cap (rad / m per tick) | 0.01270 | 0.02445 | 0.02510 | 0.00636 | 0.00336 | 0.01124 | 0.00550 |
| per second at 20 Hz (rad/s) | 0.254 | 0.489 | 0.502 | 0.127 | 0.067 | 0.225 | – |
| per second (deg/s) | 14.6 | 28.0 | 28.8 | 7.3 | 3.9 | 12.9 | – |

---

## 5. Why s = 0.15 is large in this action space

### 5.1 Dataset step sizes vs the scaler range (D)

| joint | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| \|a\| q50 (typical per-tick step) | 0.00381 | 0.01297 | 0.01373 | 0.00649 | 0.00229 | 0.00534 | 0.03146 |
| \|a\| q90 | 0.02365 | 0.06332 | 0.06752 | 0.05035 | 0.01030 | 0.03510 | 0.03995 |
| \|a\| q99 | 0.04883 | 0.12636 | 0.10910 | 0.08888 | 0.02747 | 0.12131 | 0.03996 |
| half range / typical step | 19.3 | 10.9 | 10.6 | 18.7 | 28.0 | 40.1 | – |
| **residual cap / typical step** (s·half_exp / q50) | **3.33** | **1.88** | **1.83** | **0.98** | **1.47** | **2.10** | 0.17 |

Min-max scaling uses the extremes, and in a delta action space the extremes
are rare large steps, 10-40 times the typical step. A residual defined as a
fraction of that range is therefore 1-3.3 times a typical ACT step.

The gripper is the exception: it is an absolute position, so its residual
(at most 5.5 mm) does not accumulate (section 6) and is small compared with
its range.

### 5.2 Measured during the step-70016 evals (E-RL)

| joint | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| ACT \|a_ACT\| q50 | 0.00273 | 0.01118 | 0.01202 | 0.00465 | 0.00130 | 0.00353 | 0.02622 |
| residual \|ρ\| q50 | 0.00622 | 0.01681 | 0.01484 | 0.00580 | 0.00232 | 0.00712 | 0.00397 |
| residual \|ρ\| q90 | 0.01132 | 0.02370 | 0.02444 | 0.00632 | 0.00317 | 0.01085 | 0.00546 |
| **\|ρ\| / \|a_ACT\| (q50)** | **2.28** | **1.50** | **1.23** | **1.25** | **1.78** | **2.02** | 0.15 |
| mean residual, normalized | −0.0343 | +0.0123 | −0.0356 | +0.0228 | −0.0079 | −0.0100 | −0.0117 |
| mean residual, real per tick | −0.00290 | +0.00201 | −0.00595 | +0.00290 | −0.00053 | −0.00225 | −0.00064 |
| fraction of ticks at the cap (\|r\| ≥ 0.95 s) | 0.04 | 0.15 | 0.15 | 0.41 | 0.08 | 0.14 | 0.26 |

On every arm joint the deterministic residual is larger than ACT's own step
(median). It also has a non-zero mean on every joint.

### 5.3 Your example, with real numbers

Take ACT's typical step (dataset q50) and add the full residual cap in the
same direction. Normalize with section 3, add, convert back:

| joint | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| ACT step a (rad) | +0.00381 | +0.01297 | +0.01373 | +0.00649 | +0.00229 | +0.00534 |
| n_b = (a − mid)/half_exp | −0.0136 | +0.2797 | +0.2222 | +0.1155 | −0.0170 | +0.1087 |
| + s | +0.15 | +0.15 | +0.15 | +0.05 | +0.05 | +0.05 |
| n_c | +0.1364 | +0.4297 | +0.3722 | +0.1655 | +0.0330 | +0.1587 |
| a_cmd = mid + n_c·half_exp (rad) | +0.01651 | +0.03742 | +0.03883 | +0.01285 | +0.00565 | +0.01658 |
| **a_cmd / a** | **4.33** | **2.88** | **2.83** | **1.98** | **2.47** | **3.10** |

The same holds with both signs negative (identical ratios).

---

## 6. Why the residual's effect accumulates (proof)

Per arm joint, tick t, measured position q_t. ACT computes a chunk at tick
t₀ from the observation at t₀; its 14 deltas d_{t₀}, …, d_{t₀+13} are fixed
from then on (open loop). The residual adds ρ_t (equation 1). The command sent
at tick t (`policy_client.py:385`):

    x_t = q_t + d_t + ρ_t                                                      (2)

**Delta control (what runs now).** The follower moves toward x_t. Whatever
part of ρ_t it realizes ends up in q_{t+1}. The next command is
x_{t+1} = q_{t+1} + d_{t+1} + ρ_{t+1}, and d_{t+1} was fixed at t₀, so it
contains no term that depends on q_{t+1} or on ρ_t. Nothing removes the
deviation inside the chunk.

With any follower response that is linear in the commands,
q_{t+1} − q_t = Σ_k c_k (d_{t−k} + ρ_{t−k}). The displacement over a chunk
then splits exactly into an ACT part and a residual part:

    Δq_chunk = β_ACT · Σ d  +  β_res · Σ ρ                                     (3)

with no restoring term. Deviations add up: after N ticks without ACT
feedback the commanded residual offset is Σρ, bounded by N · s · half_exp,
not by s · half_exp.

**Absolute control (for comparison).** If actions were absolute targets,
x_t = A_t + ρ_t with A_t ACT's target. ρ_t does not appear in x_{t+1}, so
the deviation from ACT's plan at any time is at most |ρ_t| ≤ s · half_exp,
however long it runs. This is the setting where ResFiT's residual is a
"small correction".

**Across chunks.** ACT re-plans every 14 ticks from the drifted state. It
removes the drift only if it actively steers back. Section 7.3 measures
whether it does.

Worst-case commanded residual offset, N · s · half_exp (rad):

| joint | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| one tick | 0.0127 | 0.0245 | 0.0251 | 0.0064 | 0.0034 | 0.0112 |
| one chunk (14 ticks) | 0.1778 | 0.3423 | 0.3515 | 0.0890 | 0.0471 | 0.1573 |
| one chunk (deg) | 10.2 | 19.6 | 20.1 | 5.1 | 2.7 | 9.0 |

---

## 7. Measured drift (E-RL, 32 eval runs, `policy_step_70016`)

Chunk boundaries come from the residual log: a fresh ACT forward pass shows
as `dt_base_ms > 20` (505 refills). 473 chunks are exactly 14 ticks inside
one run and are used below.

### 7.1 Commanded residual per chunk

| joint | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| \|Σρ\| q50 (rad) | 0.0731 | 0.1937 | 0.1828 | 0.0776 | 0.0304 | 0.0870 |
| \|Σρ\| q90 | 0.1535 | 0.3237 | 0.3362 | 0.0884 | 0.0436 | 0.1506 |
| \|Σρ\| max | 0.1744 | 0.3416 | 0.3507 | 0.0889 | 0.0466 | 0.1572 |
| bound 14·s·half_exp | 0.1778 | 0.3423 | 0.3515 | 0.0890 | 0.0471 | 0.1573 |
| ACT \|Σd\| q50 | 0.0378 | 0.1555 | 0.1699 | 0.0717 | 0.0175 | 0.0557 |
| ACT \|Σd\| q90 | 0.1208 | 0.2859 | 0.4954 | 0.4106 | 0.0521 | 0.2483 |
| chunks where the residual keeps one sign for all 14 ticks | 57% | 56% | 62% | 82% | 71% | 63% |

The residual reaches its theoretical per-chunk maximum (q90 ≈ max ≈ bound).
In most chunks it pushes in one direction for the whole chunk, so its sum
does not cancel. Its median per-chunk sum is larger than ACT's on joints 0,
1, 2, 3, 4 and 5.

### 7.2 What the joints actually did

For each chunk, Δq = q(first tick of the next chunk) − q(first tick of this
chunk), from the measured follower state. Least-squares fit of equation (3)
per joint, no intercept:

| joint | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| β_ACT | 0.262 | 0.228 | 0.286 | 0.359 | 0.105 | 0.348 |
| β_res | 0.221 | 0.108 | 0.195 | 0.386 | 0.047 | 0.287 |
| R² | 0.86 | 0.73 | 0.89 | 0.98 | **0.29** | 0.97 |
| realized residual part \|β_res Σρ\| q50 (rad) | 0.0162 | 0.0210 | 0.0356 | 0.0299 | 0.0014 | 0.0250 |
| realized ACT part \|β_ACT Σd\| q50 (rad) | 0.0099 | 0.0354 | 0.0487 | 0.0257 | 0.0018 | 0.0194 |
| **share of the motion from the residual (q50)** | **59%** | **37%** | **39%** | **45%** | 40% | **51%** |

The fit explains 73-98% of the measured motion on joints 0, 1, 2, 3 and 5.
Joint 4 is not explained (R² 0.29), so its row is not evidence either way.

The follower realizes only part of each command (β < 1, see
`FINAL_STAGE_RL_PLAN.md` 3.4: gravity holding offset). β_res / β_ACT =
[0.84, 0.47, 0.68, 1.08, 0.45, 0.82] for joints 0-5: a commanded radian of
residual moves the real arm 45-108% as much as a commanded radian of ACT.
This is why the realized shares below are somewhat lower than the commanded
sums in 7.1 would suggest, but still large.

### 7.3 Does ACT's next chunk correct the residual's drift?

441 pairs of consecutive chunks in the same run. Regression of ACT's
commanded sum in chunk k+1 on the residual's commanded sum in chunk k
(negative = ACT steers back):

| joint | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| slope | −0.140 | +0.113 | +0.535 | +1.533 | −0.061 | −0.171 |
| correlation | −0.17 | +0.10 | **+0.38** | **+0.37** | −0.05 | −0.08 |

There is no consistent correction:
- joints 0 and 5: weak negative correlation
- joints 2 and 3: ACT's next chunk moves further in the same direction as the
  residual's drift

These are correlations; they do not say why ACT does this. They show that
ACT does not undo the drift.

### 7.4 Accumulated over an eval run

Realized residual displacement per run = β_res · Σρ over the whole run
(32 runs, median 181.5 ticks; uses the fit from 7.2):

| joint | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| q50 (rad) | 0.118 | 0.114 | 0.294 | 0.218 | (0.006) | 0.181 |
| q50 (deg) | 6.7 | 6.5 | **16.8** | **12.5** | (0.4) | 10.4 |
| q90 (deg) | 21.3 | 21.0 | **32.1** | **25.5** | (0.9) | **30.9** |

(joint 4 in brackets: R² 0.29, not reliable.)

---

## 8. Conclusion

1. The conversions between real units, ACT's z-score and the RL min-max
   space are correct: exact on 6,828 ticks (section 4).
2. ACT returns real-unit deltas. The RL re-normalizes them; `action_scale` is
   defined in the RL space as a fraction of the widened extreme range
   (sections 2-3).
3. Because delta extremes are 10-40 times the typical step, `action_scale =
   0.15` lets the residual add up to 1-3.3 typical ACT steps per tick, and at
   eval the deterministic residual was 1.2-2.3 times ACT's own step (median)
   on every arm joint (section 5).
4. In delta control a residual deviation is never removed inside a chunk
   (equation 2-3). In absolute control it would be bounded by one tick's
   residual (section 6).
5. Measured: per chunk the residual reaches its theoretical maximum, keeps
   one sign in 56-82% of chunks, produces 37-59% of the real joint motion on
   joints 0-3 and 5, and ACT's next chunk does not correct it. Over one run
   the residual alone moves joints 2, 3 and 5 by a median of 10-17°
   (q90 25-32°) (section 7).

Together these explain the eval result: ACT reaches the hole (stage 2) in
16/20 runs; with this residual the arm does not even grasp in 20/33 runs.

---

## 9. What follows for the next RL runs

1. **Set action_scale from ACT's step size, not the range.** For a per-tick
   cap of k times the typical step, the cap is `s·half_exp = s·(1+s)·half`,
   so solve `s² + s − k·q50/half = 0` per joint:
   `s = (−1 + sqrt(1 + 4·k·q50/half)) / 2`. With the current dataset stats
   (full task, joints 0-5):

   | k (cap / typical step) | s | cap (rad per tick) |
   |---|---|---|
   | 0.25 | [0.0128, 0.0224, 0.0231, 0.0132, 0.0089, 0.0062] | [0.00095, 0.00324, 0.00343, 0.00162, 0.00057, 0.00134] |
   | 0.5 | [0.0252, 0.0438, 0.0451, 0.0261, 0.0176, 0.0123] | [0.00190, 0.00649, 0.00687, 0.00324, 0.00115, 0.00267] |

   (checked: `s·(1+s)·half / q50` = k exactly). For the insertion-stage RL,
   recompute from the insertion frames, whose step sizes differ. Even
   at k = 0.5, 14 ticks of a one-signed residual still command 7 typical
   steps per chunk, so this has to be combined with item 2.
2. **Penalize the residual's mean.** `agent.actor.action_l2_reg_weight` is 0
   (C). Any constant residual integrates into drift (section 6).
3. **Fix the cache keys before changing action_scale.** `action_scale` changes
   the scaler box that every stored action is normalized with, but it is in
   neither `offline_cache_meta` nor `online_cache_meta`
   (`train_residual_td3_distributed.py`). Changing it with existing caches
   silently reuses buffers normalized with the old box. Add it to both keys
   first.
4. **Measure, don't guess.** The analysis in sections 7.1-7.4 can run on any
   eval: the residual log and infer log are enough. It should be repeated
   after every change above.
