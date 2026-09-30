# TD3 Algorithm — Deep Dive with Math and Examples

This document explains the **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** algorithm as implemented in this codebase, with full mathematical derivations, pseudocode, and a numerical example.

**Reference**: [Fujimoto et al., 2018 — "Addressing Function Approximation Error in Actor-Critic Methods"](https://arxiv.org/abs/1802.09477)

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)

---

## Table of Contents

1. [Background: DDPG and Its Problems](#1-background-ddpg-and-its-problems)
2. [TD3's Three Fixes](#2-td3s-three-fixes)
3. [The Full TD3 Algorithm (Pseudocode)](#3-the-full-td3-algorithm-pseudocode)
4. [Extension: RED-Q Style Ensemble](#4-extension-red-q-style-ensemble)
5. [Numerical Example: One Complete Update Step](#5-numerical-example-one-complete-update-step)
6. [How This Codebase Implements TD3](#6-how-this-codebase-implements-td3)

---

## 1. Background: DDPG and Its Problems

### DDPG (Deep Deterministic Policy Gradient)

DDPG is an actor-critic algorithm for continuous action spaces:

- **Actor** $\mu_\phi(s)$: deterministic policy — maps state to action
- **Critic** $Q_\theta(s, a)$: estimates expected return for taking action $a$ in state $s$

**Critic update** — minimize TD error:

$$\mathcal{L}_{\text{critic}} = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}} \left[ \left( Q_\theta(s, a) - y \right)^2 \right]$$

where the TD target is:

$$y = r + \gamma \cdot Q_{\theta'}(s', \mu_{\phi'}(s'))$$

$\theta', \phi'$ are **target networks** — slow-moving copies updated via Polyak averaging:

$$\theta' \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta'$$

**Actor update** — follow the gradient of Q:

$$\nabla_\phi J = \mathbb{E}_{s \sim \mathcal{B}} \left[ \nabla_a Q_\theta(s, a) \big|_{a = \mu_\phi(s)} \cdot \nabla_\phi \mu_\phi(s) \right]$$

### The Problem: Overestimation Bias

DDPG's critic systematically **overestimates** Q-values. Here's why:

1. The TD target uses $\max_a Q(s', a)$ implicitly (the actor learns to pick the action that maximizes Q)
2. Q-values have estimation errors: $Q(s,a) = Q^*(s,a) + \epsilon(s,a)$
3. Taking the max over noisy estimates is biased upward: $\mathbb{E}[\max(X_1, X_2)] \geq \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$
4. This overestimation compounds over training, creating a feedback loop

The actor chases overestimated Q-values → the critic's targets include the actor's bad actions → both degrade.

---

## 2. TD3's Three Fixes

### Fix 1: Twin (Clipped Double) Critics

Instead of one Q-network, train **two independent critics** $Q_{\theta_1}$ and $Q_{\theta_2}$. Use the **minimum** for the TD target:

$$y = r + \gamma \cdot \min(Q_{\theta_1'}(s', a'), Q_{\theta_2'}(s', a'))$$

**Intuition**: If either critic thinks the action is bad, trust the pessimistic one. This counteracts overestimation because the minimum of two noisy estimates has less positive bias than either alone.

**Mathematically**: For two independent estimates with errors $\epsilon_1, \epsilon_2$:

$$\mathbb{E}[\min(Q^* + \epsilon_1, Q^* + \epsilon_2)] \leq Q^*$$

This introduces a slight *underestimation* bias, but underestimation is far less harmful than overestimation (the actor becomes conservative rather than overconfident).

### Fix 2: Delayed Actor Updates

Update the critic **more frequently** than the actor. In standard TD3: critic updates every step, actor updates every 2 steps. In this codebase: critic updates 4 times per env step, actor updates once.

**Why?** The actor's gradient depends on the critic being accurate. If both update at the same rate, the actor "chases" a constantly changing, noisy critic. By making the critic update more often, its estimates stabilize before the actor adjusts.

**In this codebase** (UTD=4, actor_updates_per_iteration=1):

```
Update cycle per env step:
  i=0: critic update only
  i=1: critic update only
  i=2: critic update only
  i=3: critic update + actor update  ← actor sees a critic that's had 4 corrections
```

### Fix 3: Target Policy Smoothing

Add **clipped noise** to the target action:

$$a' = \mu_{\phi'}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

Then clamp to valid range: $a' = \text{clip}(a', -1, 1)$

**Why?** Without noise, the critic can exploit narrow "spikes" in Q-value space — specific actions where Q happens to be high due to estimation error. Adding noise smooths these spikes:

$$Q_{\text{target}}(s', a') \approx \frac{1}{|\mathcal{N}|} \sum_{a' \in \mathcal{N}} Q(s', a')$$

This is a form of **regularization** — the critic can't rely on fragile, point-estimate Q-values.

**In this codebase**: Target noise uses `TruncatedNormal` with `stddev_clip = 0.3`:

```python
next_residual = actor_target(next_obs, stddev=stddev, clip=0.3)
```

---

## 3. The Full TD3 Algorithm (Pseudocode)

```
Initialize:
  Critic Q_θ₁, Q_θ₂ with random weights
  Actor μ_φ with random weights
  Target networks: θ₁' ← θ₁, θ₂' ← θ₂, φ' ← φ
  Replay buffer B ← ∅

for t = 1 to T:
  # --- Collect experience ---
  a = μ_φ(s) + ε,  ε ~ N(0, σ²)           # Exploration noise
  s', r, done = env.step(a)
  B.add(s, a, r, s', done)
  
  # --- Sample batch ---
  {(sᵢ, aᵢ, rᵢ, sᵢ', dᵢ)} ~ B
  
  # --- Compute target ---
  ã' = μ_φ'(s') + clip(ε, -c, c),  ε ~ N(0, σ_target²)    # Target smoothing
  ã' = clip(ã', -1, 1)
  y = r + γ(1-d) · min(Q_θ₁'(s', ã'), Q_θ₂'(s', ã'))      # Clipped double Q
  
  # --- Update critics ---
  L_critic = (1/B) Σ [(Q_θ₁(s, a) - y)² + (Q_θ₂(s, a) - y)²]
  θ₁, θ₂ ← θ₁, θ₂ - α_c · ∇L_critic
  
  # --- Delayed actor update (every d steps) ---
  if t mod d == 0:
    L_actor = -(1/B) Σ Q_θ₁(s, μ_φ(s))        # Maximize Q
    φ ← φ - α_a · ∇L_actor
    
    # Soft update targets
    θ₁' ← τθ₁ + (1-τ)θ₁'
    θ₂' ← τθ₂ + (1-τ)θ₂'
    φ'  ← τφ  + (1-τ)φ'
```

---

## 4. Extension: RED-Q Style Ensemble

This codebase uses a **10-head Q-ensemble** instead of TD3's standard 2 critics. This is inspired by RED-Q ([Chen et al., 2021](https://arxiv.org/abs/2101.05982)).

### Architecture

```
10 independent Q-heads (shared spatial trunk, independent MLP heads)

           Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇  Q₈  Q₉  Q₁₀
            │   │   │   │   │   │   │   │   │   │
            └───┴───┴───┴───┴───┴───┴───┴───┴───┘
                        │
                  Shared SpatialEmb Trunk
                  (visual features → compressed)
```

### Target Q: Min over 2 Random Heads

For computing TD targets:

$$Q_{\text{target}}(s', a') = \min_{i \in \text{random\_subset}(2, \{1,...,10\})} Q_{\theta_i'}(s', a')$$

At each update step, **randomly select 2** of the 10 heads and take their minimum. This provides:
- **Diversity**: Different random subsets prevent the same heads from always being "the pessimist"
- **Less extreme pessimism**: Min over 2 instead of min over 10 — reduces underestimation bias

### Actor Q: Mean over All Heads

For computing the actor's policy gradient:

$$Q_{\pi} = \frac{1}{10} \sum_{i=1}^{10} Q_{\theta_i}(s, \mu_\phi(s))$$

Using the **mean** gives a more stable gradient signal than using any single head or subset.

### Why 10 Heads?

| Approach | Target Q | Actor Q | Stability |
|---|---|---|---|
| TD3 (2 critics) | min(Q₁, Q₂) | Q₁ only | Good, but variance |
| RED-Q (10 heads) | min(random 2 of 10) | mean(all 10) | Better — ensemble smooths |

The ensemble provides:
1. **Better uncertainty estimation** — disagreement between heads indicates uncertain states
2. **Smoother loss landscape** — averaging 10 heads reduces gradient noise for the actor
3. **Computational efficiency** — PyTorch `vmap` runs all 10 heads in a single batched forward pass

---

## 5. Numerical Example: One Complete Update Step

Let's trace through a single critic + actor update with concrete numbers.

### Setup

- Action dimension: 14 (7 per robot arm for a dual-arm task)
- Discount: $\gamma = 0.995$
- N-step: 5 (handled by replay buffer transform, so the batch already has n-step returns)
- Batch size: 256

### Critic Update Example

**Step 1: Sample transition from buffer**

```
s = {images: [...], state: [0.2, -0.1, ...], base_action: [0.3, -0.5, ...]}
a = [0.35, -0.48, ...]     (combined base + residual, stored in buffer)
r = 0.0                     (no success yet)
s' = {images: [...], state: [0.21, -0.09, ...], base_action: [0.31, -0.52, ...]}
nonterminal = True
gamma_n = 0.995^5 = 0.9752  (5-step discount from MultiStepTransform)
```

**Step 2: Compute target action (with smoothing noise)**

```
# Target actor predicts next residual
next_residual_raw = actor_target(s')  → [-0.02, 0.01, ...]  (small corrections)

# Add clipped noise for smoothing
noise = clip(N(0, 0.025²), -0.3, 0.3) → [0.008, -0.012, ...]
next_residual = next_residual_raw + noise → [-0.012, -0.002, ...]

# Combine with base
next_action = clip(s'["base_action"] + next_residual, -1, 1)
           = clip([0.31 + (-0.012), -0.52 + (-0.002), ...], -1, 1)
           = [0.298, -0.522, ...]
```

**Step 3: Compute target Q**

```
# Run all 10 heads of target critic
Q_target_all = [0.42, 0.39, 0.45, 0.41, 0.38, 0.43, 0.40, 0.44, 0.37, 0.41]

# Randomly pick 2 heads (say indices 4 and 8)
Q_min = min(Q_target_all[4], Q_target_all[8]) = min(0.38, 0.37) = 0.37

# TD target
y = r + gamma_n * nonterminal * Q_min
  = 0.0 + 0.9752 * 1.0 * 0.37
  = 0.3608
```

**Step 4: Compute critic loss**

```
# Current critic predicts Q for (s, a)
Q_current_all = [0.35, 0.33, 0.40, 0.36, 0.34, 0.38, 0.35, 0.39, 0.32, 0.36]

# MSE across all heads (averaged)
loss_per_head = [(0.35-0.3608)², (0.33-0.3608)², ..., (0.36-0.3608)²]
             = [0.000119, 0.000945, 0.001536, 0.0000069, 0.000433, ...]
critic_loss = mean(loss_per_head) ≈ 0.000482
```

**Step 5: Backward + step**

```
critic_loss.backward()
clip_grad_norm_(encoder, 1.0)
clip_grad_norm_(critic, 1.0)
encoder_opt.step()    # lr = 1e-4
critic_opt.step()     # lr = 1e-4
```

**Step 6: Soft update target**

```
# For each parameter θ' in target critic:
θ' ← 0.005 * θ + 0.995 * θ'
# This is extremely slow — the target barely changes each step
```

### Actor Update Example (every 4th step)

**Step 1: Predict residual action**

```
# Actor predicts residual for same observation s (with detached features!)
residual = actor(s_detached) → [-0.015, 0.008, ...]  (in [-0.2, 0.2])

# Combine with base
combined = clip(s["base_action"] + residual, -1, 1)
         = clip([0.3 + (-0.015), -0.5 + 0.008, ...], -1, 1)
         = [0.285, -0.492, ...]
```

**Step 2: Evaluate with critic (mean of all 10 heads)**

```
Q_all = critic(s["feat"], s_state, combined)
      = [0.36, 0.34, 0.41, 0.37, 0.35, 0.39, 0.36, 0.40, 0.33, 0.37]

Q_mean = mean(Q_all) = 0.368
```

**Step 3: Compute actor loss**

```
actor_loss = -Q_mean = -0.368   (we MAXIMIZE Q by MINIMIZING -Q)
```

**Step 4: Backward + step**

```
actor_loss.backward()               # Gradient flows: Q → combined → residual → actor params
                                     # But NOT through encoder (features were detached)
clip_grad_norm_(actor, 1.0)
actor_opt.step()                     # lr = 1e-6 (very small — conservative updates)
soft_update_params(actor, actor_target, tau=0.005)
```

---

## 6. How This Codebase Implements TD3

### Key Differences from Standard TD3

| Feature | Standard TD3 | This Codebase |
|---|---|---|
| Number of critics | 2 | 10 (RED-Q style ensemble) |
| Target Q | min(Q₁, Q₂) | min(random 2 of 10) |
| Actor Q | Q₁ only | mean(all 10) |
| Actor update delay | every 2 critic updates | every 4 critic updates |
| Exploration | Gaussian noise on action | TruncatedNormal distribution |
| Action space | direct action | **residual** (base_action + residual) |
| Target tau | 0.005 | 0.005 (same) |
| Data | online only | 50% online + 50% offline demonstrations |

### Code Map

| Concept | File | Method |
|---|---|---|
| Critic update | `q_agent.py` | `update_critic()` |
| Actor update | `q_agent.py` | `update_actor()`, `_compute_actor_loss()` |
| Target action (with noise) | `q_agent.py` | `_act_default(use_target=True)` |
| Min over random 2 heads | `critic.py` | `q_value()` |
| Mean over all heads | `critic.py` | `q_value_for_policy()` |
| Soft update | `utils.py` | `soft_update_params()` |
| Exploration noise schedule | `utils.py` | `schedule()` |
| TruncatedNormal sampling | `utils.py` | `TruncatedNormal.sample()` |

Back to: [../training/RESIDUAL_RL_TRAINING.md](../training/RESIDUAL_RL_TRAINING.md)
