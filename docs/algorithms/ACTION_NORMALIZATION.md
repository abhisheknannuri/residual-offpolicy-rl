# Action & State Normalization — Deep Dive with Worked Examples

This document explains the two normalization systems used in residual RL training: **ActionScaler** (min-max scaling to [-1, 1]) and **StateStandardizer** (z-score normalization). Includes complete formulas, code mapping, and a worked example with 3 action dimensions.

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)

---

## Table of Contents

1. [Why Normalize?](#1-why-normalize)
2. [ActionScaler — Min-Max Scaling](#2-actionscaler--min-max-scaling)
3. [StateStandardizer — Z-Score](#3-statestandardizer--z-score)
4. [Worked Example: 3 Action Dimensions](#4-worked-example-3-action-dimensions)
5. [Safety Guards Against Blow-Up](#5-safety-guards-against-blow-up)
6. [Where Normalization Happens in the Pipeline](#6-where-normalization-happens-in-the-pipeline)

---

## 1. Why Normalize?

### The Problem

Raw action dimensions have vastly different scales:

| Dimension | Range | Description |
|---|---|---|
| `joint_1_velocity` | [-2.1, 2.1] rad/s | Large base joint |
| `joint_5_velocity` | [-0.03, 0.03] rad/s | Small wrist joint |
| `gripper` | [0, 1] | Binary open/close |

If the actor's output is in the same space for all dimensions, the network must learn that a "1" means very different things for different joints. The gradient for `joint_5` is 70× smaller than for `joint_1`, making learning extremely uneven.

### The Solution

Map everything to [-1, 1]:
- `joint_1_velocity = 2.1` → normalized = 1.0
- `joint_5_velocity = 0.03` → normalized = 1.0
- `gripper = 1` → normalized = 1.0

Now "1.0" means "maximum positive value" for every dimension. The network can treat all outputs equally.

---

## 2. ActionScaler — Min-Max Scaling

**File**: `resfit/rl_finetuning/utils/normalization.py`, class `ActionScaler`

### Initialization

Given dataset statistics $a_{\min}^{(d)}, a_{\max}^{(d)}$ for each action dimension $d$:

**Step 1: Compute center and half-range**

$$a_{\text{mid}}^{(d)} = \frac{a_{\min}^{(d)} + a_{\max}^{(d)}}{2}$$

$$a_{\text{half}}^{(d)} = \frac{a_{\max}^{(d)} - a_{\min}^{(d)}}{2}$$

**Step 2: Apply minimum range safety guard**

$$a_{\text{half}}^{(d)} = \max\left(a_{\text{half}}^{(d)}, \frac{\text{min\_range}}{2}\right) = \max\left(a_{\text{half}}^{(d)}, 0.05\right)$$

where `min_range_per_dim = 0.1` (so `min_range / 2 = 0.05`).

**Step 3: Expand by action_scale factor**

$$a_{\text{expanded}}^{(d)} = a_{\text{half}}^{(d)} \times (1 + \text{action\_scale}) = a_{\text{half}}^{(d)} \times 1.2$$

> **Note**: The `action_scale` (default 0.2, set via `agent.actor.action_scale`) does NOT overwrite or replace the dataset statistics. The dataset min/max define the center and shape of the normalization range. The `action_scale` only **expands** that range by 20% to create headroom for the residual correction. This same value is also used inside the Actor network to bound the residual output (`Tanh × action_scale`). They are separate mechanisms that share the same config value — one bounds the residual, the other expands the normalization range to accommodate it.

**Step 4: Compute final limits**

$$a_{\text{final\_min}}^{(d)} = a_{\text{mid}}^{(d)} - a_{\text{expanded}}^{(d)}$$

$$a_{\text{final\_max}}^{(d)} = a_{\text{mid}}^{(d)} + a_{\text{expanded}}^{(d)}$$

### Scaling (raw → normalized)

$$a_{\text{norm}}^{(d)} = 2 \cdot \frac{\text{clip}(a^{(d)}, a_{\text{final\_min}}^{(d)}, a_{\text{final\_max}}^{(d)}) - a_{\text{final\_min}}^{(d)}}{a_{\text{final\_max}}^{(d)} - a_{\text{final\_min}}^{(d)}} - 1$$

Result: $a_{\text{norm}}^{(d)} \in [-1, 1]$

### Unscaling (normalized → raw)

$$a^{(d)} = a_{\text{final\_min}}^{(d)} + \frac{\text{clip}(a_{\text{norm}}^{(d)}, -1, 1) + 1}{2} \times (a_{\text{final\_max}}^{(d)} - a_{\text{final\_min}}^{(d)})$$

---

## 3. StateStandardizer — Z-Score

**File**: `resfit/rl_finetuning/utils/normalization.py`, class `StateStandardizer`

### Initialization

Given dataset statistics $\mu_s^{(d)}, \sigma_s^{(d)}$ for each state dimension $d$:

$$\sigma_{\text{safe}}^{(d)} = \max(\sigma_s^{(d)}, \text{min\_std}) = \max(\sigma_s^{(d)}, 0.1)$$

### Standardization

$$s_{\text{std}}^{(d)} = \frac{s^{(d)} - \mu_s^{(d)}}{\sigma_{\text{safe}}^{(d)}}$$

Result: $s_{\text{std}}$ has approximately mean=0 and std=1, making it neural-network-friendly.

### Where It's Applied

`_augment_obs()` in `BasePolicyVecEnvWrapper` applies standardization to `observation.state` at every step. The RL agent only ever sees standardized states.

---

## 4. Worked Example: 3 Action Dimensions

Let's trace through the full normalization pipeline with concrete numbers.

### Dataset Statistics

| Dim | Name | $a_{\min}$ | $a_{\max}$ | Range |
|---|---|---|---|---|
| 0 | x_velocity | -0.120 | 0.080 | 0.200 |
| 1 | y_velocity | -0.005 | 0.005 | 0.010 |
| 2 | gripper | 0.000 | 1.000 | 1.000 |

### Step 1: Center and Half-Range

| Dim | $a_{\text{mid}}$ | $a_{\text{half}}$ |
|---|---|---|
| 0 | (-0.120 + 0.080) / 2 = **-0.020** | (0.080 - (-0.120)) / 2 = **0.100** |
| 1 | (-0.005 + 0.005) / 2 = **0.000** | (0.005 - (-0.005)) / 2 = **0.005** |
| 2 | (0.000 + 1.000) / 2 = **0.500** | (1.000 - 0.000) / 2 = **0.500** |

### Step 2: Safety Guard (`min_range = 0.1`, `min_half = 0.05`)

| Dim | $a_{\text{half}}$ before | After max(·, 0.05) | Changed? |
|---|---|---|---|
| 0 | 0.100 | **0.100** | No |
| 1 | 0.005 | **0.050** | **Yes!** (0.005 < 0.05) |
| 2 | 0.500 | **0.500** | No |

**Dim 1 was rescued!** Without the safety guard, a range of 0.01 means that a tiny raw change of 0.001 would correspond to normalized change of 0.2 — extremely sensitive. The safety floor of 0.05 prevents this.

### Step 3: Expand by action_scale = 0.2

| Dim | $a_{\text{expanded}} = a_{\text{half}} \times 1.2$ |
|---|---|
| 0 | 0.100 × 1.2 = **0.120** |
| 1 | 0.050 × 1.2 = **0.060** |
| 2 | 0.500 × 1.2 = **0.600** |

### Step 4: Final Limits

| Dim | $a_{\text{final\_min}}$ | $a_{\text{final\_max}}$ | Final Range |
|---|---|---|---|
| 0 | -0.020 - 0.120 = **-0.140** | -0.020 + 0.120 = **0.100** | 0.240 |
| 1 | 0.000 - 0.060 = **-0.060** | 0.000 + 0.060 = **0.060** | 0.120 |
| 2 | 0.500 - 0.600 = **-0.100** | 0.500 + 0.600 = **1.100** | 1.200 |

### Forward Pass: scale(raw_action)

BC policy outputs raw action: $a = [0.050, 0.003, 0.800]$

**Dim 0**: $2 \times \frac{0.050 - (-0.140)}{0.100 - (-0.140)} - 1 = 2 \times \frac{0.190}{0.240} - 1 = 2 \times 0.792 - 1 = \mathbf{0.583}$

**Dim 1**: $2 \times \frac{0.003 - (-0.060)}{0.060 - (-0.060)} - 1 = 2 \times \frac{0.063}{0.120} - 1 = 2 \times 0.525 - 1 = \mathbf{0.050}$

**Dim 2**: $2 \times \frac{0.800 - (-0.100)}{1.100 - (-0.100)} - 1 = 2 \times \frac{0.900}{1.200} - 1 = 2 \times 0.750 - 1 = \mathbf{0.500}$

Normalized: $a_{\text{norm}} = [0.583, 0.050, 0.500]$

### Residual Applied

Actor outputs residual: $a_{\text{res}} = [-0.05, 0.10, -0.15]$

Combined: $a_{\text{combined}} = [0.583 + (-0.05), 0.050 + 0.10, 0.500 + (-0.15)]$
$= [0.533, 0.150, 0.350]$

Clamp to [-1, 1]: $[0.533, 0.150, 0.350]$ (all already in range)

### Inverse Pass: unscale(combined)

**Dim 0**: $-0.140 + \frac{0.533 + 1}{2} \times 0.240 = -0.140 + 0.767 \times 0.240 = -0.140 + 0.184 = \mathbf{0.044}$

**Dim 1**: $-0.060 + \frac{0.150 + 1}{2} \times 0.120 = -0.060 + 0.575 \times 0.120 = -0.060 + 0.069 = \mathbf{0.009}$

**Dim 2**: $-0.100 + \frac{0.350 + 1}{2} \times 1.200 = -0.100 + 0.675 \times 1.200 = -0.100 + 0.810 = \mathbf{0.710}$

### Summary

| | Raw (BC) | Normalized | + Residual | Unscaled |
|---|---|---|---|---|
| Dim 0 (x_vel) | 0.050 | 0.583 | 0.533 | 0.044 |
| Dim 1 (y_vel) | 0.003 | 0.050 | 0.150 | 0.009 |
| Dim 2 (gripper) | 0.800 | 0.500 | 0.350 | 0.710 |

The residual made small adjustments: dim 0 decreased slightly, dim 1 increased, gripper closed a bit more.

---

## 5. Safety Guards Against Blow-Up

### ActionScaler: `min_range_per_dim = 0.1`

**Problem**: If a dimension has a near-zero range (e.g., `min=-0.001, max=0.001`), the denominator in the scaling formula is $0.002$. A tiny raw change of $0.0005$ maps to a normalized change of $0.5$ — the network would need extreme precision.

**Fix**: Force `half_range ≥ 0.05`, ensuring no dimension has a range smaller than $0.1$.

### StateStandardizer: `min_std = 0.1`

**Problem**: If a state dimension has near-zero variance (e.g., a constant like the number of robots), dividing by $\sigma \approx 0$ produces infinities.

**Fix**: Force `std ≥ 0.1`: $\sigma_{\text{safe}} = \max(\sigma, 0.1)$

### ActionScaler: Double-safety in `scale()`

```python
# Runtime safeguard on the range
self._range = torch.maximum(self._range, torch.tensor(1e-8))
```

Even after the min_range guard, a second check at $10^{-8}$ prevents any possible division by zero.

### StateStandardizer: Double-safety in `standardize()`

```python
std_safe = torch.maximum(std, torch.tensor(1e-8))
```

Same double-check pattern.

---

## 6. Where Normalization Happens in the Pipeline

### Action Flow (Normalization Points)

```
BC Policy
    │
    ▼ raw action (e.g., [0.05, 0.003, 0.8])
    │
action_scaler.scale()
    │
    ▼ normalized action (e.g., [0.583, 0.050, 0.500])
    │
stored as obs["observation.base_action"]
    │
    ▼
Residual Actor produces residual (e.g., [-0.05, 0.10, -0.15])
    │
combined = base + residual (e.g., [0.533, 0.150, 0.350])
    │
action_scaler.unscale()
    │
    ▼ raw combined action (e.g., [0.044, 0.009, 0.710])
    │
env.step(raw_action)
```

### State Flow (Normalization Points)

```
Environment
    │
    ▼ raw state (e.g., [1.57, -0.02, 0.95, ...])
    │
state_standardizer.standardize()   ← happens inside _augment_obs()
    │
    ▼ standardized state (e.g., [-0.3, 0.1, 1.2, ...])
    │
stored as obs["observation.state"]
    │
    ▼
Used by Actor and Critic networks
```

### Summary Table

| Data | Normalization | When | Where |
|---|---|---|---|
| Actions (BC → RL agent) | `ActionScaler.scale()` | Every env step | `BasePolicyVecEnvWrapper.step()` |
| Actions (RL agent → env) | `ActionScaler.unscale()` | Every env step | `BasePolicyVecEnvWrapper.step()` |
| States | `StateStandardizer.standardize()` | Every env step | `BasePolicyVecEnvWrapper._augment_obs()` |
| Images | uint8 / 255.0 | During encoding | `QAgent._encode()` |

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)
