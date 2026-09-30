# N-Step Returns — Deep Dive with Numerical Walkthrough

This document explains **multi-step (n-step) TD returns** as used in the replay buffer's `MultiStepTransform`, with complete math, a worked example, and bias-variance analysis.

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)

---

## Table of Contents

1. [What is an N-Step Return?](#1-what-is-an-n-step-return)
2. [Math: From 1-Step to N-Step](#2-math-from-1-step-to-n-step)
3. [The Bias-Variance Tradeoff](#3-the-bias-variance-tradeoff)
4. [How MultiStepTransform Works](#4-how-multisteptransform-works)
5. [Numerical Walkthrough: n=5](#5-numerical-walkthrough-n5)
6. [Edge Cases: Episode Boundaries](#6-edge-cases-episode-boundaries)
7. [How N-Step Interacts with the Training Loop](#7-how-n-step-interacts-with-the-training-loop)

---

## 1. What is an N-Step Return?

In standard 1-step TD learning, the target for $Q(s_t, a_t)$ is:

$$y_t^{(1)} = r_t + \gamma \cdot Q(s_{t+1}, a_{t+1})$$

This bootstraps after **one** step — it uses only one real reward and then substitutes the critic's estimate for everything beyond that.

An **n-step return** looks further into the future:

$$y_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \cdot Q(s_{t+n}, a_{t+n})$$

This uses **n real rewards** before bootstrapping. In this codebase, $n = 5$.

---

## 2. Math: From 1-Step to N-Step

### 1-Step TD Target

$$y_t^{(1)} = r_t + \gamma \cdot Q(s_{t+1}, a_{t+1})$$

- Uses 1 real reward
- Bootstraps from step $t+1$
- High bias (critic estimate may be wrong), low variance (only 1 random variable)

### 2-Step TD Target

$$y_t^{(2)} = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot Q(s_{t+2}, a_{t+2})$$

### 5-Step TD Target (used in this codebase)

$$y_t^{(5)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \gamma^4 r_{t+4} + \gamma^5 Q(s_{t+5}, a_{t+5})$$

With $\gamma = 0.995$:

$$y_t^{(5)} = r_t + 0.995 r_{t+1} + 0.990 r_{t+2} + 0.985 r_{t+3} + 0.980 r_{t+4} + 0.975 \cdot Q(s_{t+5}, a_{t+5})$$

### ∞-Step TD Target (Monte Carlo)

$$y_t^{(\infty)} = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

No bootstrapping at all — uses only real rewards. Zero bias, but very high variance.

---

## 3. The Bias-Variance Tradeoff

### Why Not n=1?

With 1-step, the target relies entirely on the critic's estimate of $Q(s_{t+1}, a_{t+1})$. Early in training, this estimate is essentially random. A bad Q-estimate produces a bad target, which produces a worse Q-estimate — a spiral.

### Why Not n=∞ (Monte Carlo)?

Monte Carlo uses no bootstrapping, so it's unbiased. But in robotics:
- Episodes are long (300+ steps for manipulation tasks)
- Rewards are sparse (only at the end)
- The return has high variance because it depends on all future actions

### Why n=5 is a Good Middle Ground

| Property | n=1 | n=5 | n=∞ |
|---|---|---|---|
| Real rewards used | 1 | 5 | all |
| Bootstrap reliance | high | moderate | none |
| Bias | high | moderate | zero |
| Variance | low | moderate | high |
| Signal propagation | slow | fast | fastest |

**Signal propagation** is the key insight for sparse-reward tasks. If the reward only comes at the last step of a 300-step episode:

- **n=1**: The reward signal takes ~300 gradient updates to propagate backward through Q-values (each update pushes Q one step back)
- **n=5**: The reward signal propagates 5 steps at a time → ~60 updates to reach the beginning
- **n=∞**: Immediate propagation, but too noisy

With $\gamma = 0.995$ and $n = 5$, the bootstrap discount is $\gamma^5 = 0.975$, so we still heavily weight future returns while using 5 real steps.

---

## 4. How MultiStepTransform Works

**File**: `resfit/rl_finetuning/utils/rb_transforms.py`

The `MultiStepTransform` is attached to the replay buffer as a **transform**. It modifies transitions as they are **added** to the buffer (not when sampled). This means every transition in the buffer already has correct n-step returns.

### What It Does to Each Transition

Given a stream of transitions being added to the buffer:

```
Before n-step transform:
  (s₀, a₀, r₀, s₁) → (s₁, a₁, r₁, s₂) → (s₂, a₂, r₂, s₃) → ...

After n-step transform (n=5):
  (s₀, a₀, R₀⁵, s₅, γ⁵, nonterminal₀)
  where R₀⁵ = r₀ + γr₁ + γ²r₂ + γ³r₃ + γ⁴r₄
```

Each transformed transition replaces:
- `next.obs` → observation **5 steps** in the future (not 1)
- `next.reward` → **cumulative discounted reward** over 5 steps
- `gamma` → $\gamma^5$ (or $\gamma^k$ if episode ended in fewer than 5 steps)
- `nonterminal` → whether the bootstrap should be used (False if episode ended within 5 steps)

### Internal Buffer Mechanism

The transform maintains an internal buffer of the last `n_steps` frames. When a new frame arrives:

1. Concatenate with the internal buffer
2. If we have at least `n_steps` frames, compute the transform:
   - Sum discounted rewards with a convolution filter $[1, \gamma, \gamma^2, ..., \gamma^{n-1}]$
   - Gather the observation from $n$ steps ahead
   - Handle episode boundaries by resetting the sum at `done` flags
3. Output the transformed transition
4. Keep the last `n_steps` frames in the buffer for the next call

### The Reward Convolution

The reward summation is implemented as a 1D convolution for efficiency:

```python
filt = [γ^0, γ^1, γ^2, ..., γ^(n-1)] = [1, 0.995, 0.990, 0.985, 0.980]
summed_reward = conv1d(reward_sequence, filt)
```

This computes $R_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k}$ for every timestep $t$ simultaneously.

---

## 5. Numerical Walkthrough: n=5

### Scenario: Successful Episode

Consider a robotic manipulation episode that succeeds at step 12 (horizon = 15):

```
Step:   0    1    2    3    4    5    6    7    8    9   10   11   12
Reward: 0    0    0    0    0    0    0    0    0    0    0    0    1
Done:   F    F    F    F    F    F    F    F    F    F    F    F    T
```

$\gamma = 0.995$, $n = 5$

### Transition at t=0

**Original**: $(s_0, a_0, r_0 = 0, s_1)$

**After n-step transform**:
- Reward: $R_0 = 0 + 0.995 \times 0 + 0.990 \times 0 + 0.985 \times 0 + 0.980 \times 0 = 0$
- Next obs: $s_5$ (5 steps ahead)
- Gamma: $\gamma^5 = 0.995^5 = 0.9752$
- Nonterminal: True (episode didn't end in steps 0–4)

**Training target**: $y = 0 + 0.9752 \times Q(s_5, a_5)$

### Transition at t=8

**After n-step transform**:
- Reward: $R_8 = 0 + 0.995 \times 0 + 0.990 \times 0 + 0.985 \times 0 + 0.980 \times 1 = 0.980$
- Next obs: $s_{12}$ (the terminal state, but note: done at step 12)
- Gamma: $\gamma^5 = 0.9752$
- Nonterminal: **False** (the episode ended at step 12, which is within the 5-step lookahead)

**Training target**: $y = 0.980 + 0.9752 \times 0 = 0.980$ (no bootstrap — terminal state)

This is the key benefit: step 8 **immediately sees** the reward from step 12, even though it's 4 steps away. Without n-step, step 8 would need the Q-value chain $Q(s_9) \leftarrow Q(s_{10}) \leftarrow Q(s_{11}) \leftarrow Q(s_{12})$ to be accurate.

### Transition at t=10

**After n-step transform**:
- Reward: $R_{10} = 0 + 0.995 \times 0 + 0.990 \times 1 = 0.990$ (only 3 real steps, episode ends at 12)
- Next obs: $s_{12}$ (terminal)
- Gamma: $\gamma^3 = 0.985$ (only 3 steps, not 5)
- Nonterminal: **False**

**Training target**: $y = 0.990$

### Transition at t=11

**After n-step transform**:
- Reward: $R_{11} = 0 + 0.995 \times 1 = 0.995$ (only 2 real steps)
- Next obs: $s_{12}$ (terminal)
- Gamma: $\gamma^2 = 0.990$
- Nonterminal: **False**

**Training target**: $y = 0.995$

### Transition at t=12 (terminal)

**After n-step transform**:
- Reward: $R_{12} = 1$ (only 1 step — this IS the terminal step)
- Gamma: $\gamma^1 = 0.995$
- Nonterminal: **False**

**Training target**: $y = 1.0$

### Summary of All Transitions' Targets

| Step | $R_t^{(n)}$ | Gamma | Nonterminal | Target $y$ |
|---|---|---|---|---|
| 0 | 0 | 0.9752 | True | $0.9752 \cdot Q(s_5, a_5)$ |
| 7 | 0 | 0.9752 | True | $0.9752 \cdot Q(s_{12}, a_{12})$ — but nonterminal! |
| 8 | 0.980 | 0.9752 | False | 0.980 |
| 9 | 0.985 | 0.9752 | False | 0.985 |
| 10 | 0.990 | 0.985 | False | 0.990 |
| 11 | 0.995 | 0.990 | False | 0.995 |
| 12 | 1.000 | 0.995 | False | 1.000 |

Notice: Steps 8–12 all see the reward **directly** — no bootstrapping needed. Only steps 0–7 rely on Q-value estimates, and even they bootstrap from a state much closer to the reward.

---

## 6. Edge Cases: Episode Boundaries

### What Happens When Done=True Within N Steps

If the episode ends at step $t + k$ where $k < n$:

1. The reward sum only goes up to step $t + k$: $R_t = \sum_{j=0}^{k} \gamma^j r_{t+j}$
2. The gamma becomes $\gamma^{k+1}$ (not $\gamma^n$)
3. The `nonterminal` flag is set to **False** — no bootstrapping

This is handled correctly by the `_multi_step_func` in `rb_transforms.py`, which tracks `done` states using cumulative sums to identify episode boundaries.

### Consecutive Episodes

The transform handles multiple episodes in the same data stream. When a `done` flag is encountered, the reward accumulation resets — rewards from a new episode are never mixed with the previous one.

```
Episode 1:       [s₀, s₁, ..., s_T (done)]
Episode 2:       [s₀', s₁', ..., s_T' (done)]

Transform correctly separates:
- Transition at s_{T-2} only sees rewards from episode 1
- Transition at s₀' only sees rewards from episode 2
```

---

## 7. How N-Step Interacts with the Training Loop

### In the Critic Update

```python
# In q_agent.py update():
discount = batch["gamma"]           # γ^n (from MultiStepTransform, already computed)
nonterminal = batch["nonterminal"]   # False if episode ended within n steps

effective_discount = discount * nonterminal  # Zero out bootstrap for terminal

# Critic target:
target_Q = critic_target.q_value(next_obs, ..., next_action)  # Q(s_{t+n}, a_{t+n})
y = reward + effective_discount * target_Q
# reward is already the n-step cumulative sum R_t^(n) from the transform
```

The training loop doesn't need to know about n-step — it's all handled by the buffer transform. The batch arrives pre-processed with:
- `reward` = $R_t^{(n)}$ (cumulative discounted reward)
- `next.obs` = $s_{t+n}$ (observation n steps in the future)
- `gamma` = $\gamma^n$ or $\gamma^k$ if episode ended early
- `nonterminal` = whether to bootstrap

### Why `num_envs=1` is Required

The assertion `assert cfg.num_envs == 1` exists because the `MultiStepTransform` maintains an internal buffer that assumes transitions arrive sequentially from a single environment. With multiple environments, transitions from different envs would be interleaved, corrupting the n-step computation.

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)
