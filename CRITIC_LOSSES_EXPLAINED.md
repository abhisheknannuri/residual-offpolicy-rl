# Critic Losses — Bellman Backup & All 3 Loss Types Explained

This document explains the **exact** Bellman backup equation used in this codebase and how each of the 3 critic loss types (`mse`, `hl_gauss`, `c51`) implements it, with the actual code copied inline.

---

## Table of Contents

1. [The Bellman Backup (Shared By All 3 Losses)](#1-the-bellman-backup-shared-by-all-3-losses)
2. [Where `(1 - done)` Went — The N-Step Transform](#2-where-1---done-went--the-n-step-transform)
3. [Loss Type 1: MSE (Default)](#3-loss-type-1-mse-default)
4. [Loss Type 2: HL-Gauss (Distributional)](#4-loss-type-2-hl-gauss-distributional)
5. [Loss Type 3: C51 (Categorical Distributional)](#5-loss-type-3-c51-categorical-distributional)
6. [Summary Table](#6-summary-table)

---

## 1. The Bellman Backup (Shared By All 3 Losses)

### The Textbook Equation (1-Step)

$$
y = r + \gamma \cdot (1 - \text{done}) \cdot Q_{\text{target}}(s', a')
$$

### The N-Step Generalization (What This Repo Uses)

$$
y = \underbrace{\sum_{k=0}^{n-1} \gamma^k r_{t+k}}_{R^{(n)}} + \underbrace{\gamma^n \cdot \mathbf{1}_{\text{nonterminal}}}_{{\text{effective\_discount}}} \cdot Q_{\text{target}}(s_{t+n}, a_{t+n})
$$

The key difference: instead of looking 1 step ahead, this repo looks **n steps ahead** (default n=3 or n=5 depending on config). The `(1 - done)` term becomes `nonterminal`, and `γ` becomes `γⁿ`.

### Where The Target Is Built In Code

**File: `resfit/rl_finetuning/off_policy/rl/q_agent.py`, lines 630–641 (batch unpacking)**

```python
obs: dict[str, torch.Tensor] = batch["obs"]
action: torch.Tensor = batch["action"]
reward: torch.Tensor = batch[("next", "reward")]        # ← This is R^(n), the n-step summed reward
discount: torch.Tensor = batch["gamma"]                  # ← This is γ^n (from MultiStepTransform)
next_nonterminal: torch.Tensor = batch["nonterminal"]    # ← This is the (1-done) equivalent
next_obs: dict[str, torch.Tensor] = batch[("next", "obs")]  # ← This is s_{t+n}, NOT s_{t+1}

# To not bootstrap on terminal states we zero out the discount factor for terminal next states
effective_discount = discount * next_nonterminal
```

**This is the `(1 - done)` you were looking for.** It is `next_nonterminal`. The line `effective_discount = discount * next_nonterminal` does:

$$
\text{effective\_discount} = \gamma^n \cdot \mathbf{1}_{\text{nonterminal}}
$$

When `nonterminal = False` (i.e., the episode ended within the n-step lookahead), `effective_discount = 0`, so the Q-target disappears and the target becomes just the summed reward.

**File: `resfit/rl_finetuning/off_policy/rl/q_agent.py`, lines 313–339 (Bellman target)**

```python
with torch.no_grad():
    assert self.actor_target.training

    # Predict next residual action and form the combined next action
    next_residual_action = self._act_default(
        obs=next_obs,
        eval_mode=not self.cfg.target_action_noise,
        stddev=stddev,
        clip=self.cfg.stddev_clip,
        use_target=True,
    )

    if self.residual_actor:
        next_action = torch.clamp(next_obs["observation.base_action"] + next_residual_action, -1.0, 1.0)
    else:
        next_action = next_residual_action

    # Compute target Q using min over a random subset of 2 heads
    target_all = self.critic_target.q_value(next_obs["feat"], next_obs["observation.state"], next_action)
    target_q_min = target_all.squeeze(-1)  # [B]
    target_q = (reward + (discount * target_q_min)).detach()
```

**The last line is the Bellman backup:**

$$
\text{target\_q} = R^{(n)} + \text{effective\_discount} \cdot \min_{j \in \mathcal{H}} Q^{(j)}_{\text{target}}(s_{t+n}, a_{t+n})
$$

Where:
- `reward` = $R^{(n)}$ = n-step discounted reward sum (already computed by replay transform)
- `discount` = `effective_discount` = $\gamma^n \cdot \mathbf{1}_{\text{nonterminal}}$ (already zero when terminal)
- `target_q_min` = $\min_{j} Q^{(j)}_{\text{target}}$ (min over random subset of critic heads)

**This target is the same for all 3 loss types.** The difference is only in how the critic is trained to match this target.

---

## 2. Where `(1 - done)` Went — The N-Step Transform

You asked: "where is the `(1 - done)` part?"

It is computed by the `MultiStepTransform` attached to the replay buffer when transitions are inserted, NOT at training time.

**File: `resfit/rl_finetuning/utils/rb_transforms.py`, lines 277–283**

```python
tensordict.set("gamma", gamma ** (time_to_obs + 1))

# NOTE: This is changed from how the torchrl implementation does it.
future_done = done.gather(-1, idx_to_gather).squeeze(-1)
nonterminal = (time_to_obs == n_steps) & (~future_done)

tensordict.set("nonterminal", nonterminal)
```

**What this does:**

1. `gamma ** (time_to_obs + 1)` → The actual discount factor. If the episode ended early (in fewer than n steps), this is $\gamma^k$ where $k < n$.

2. `nonterminal` → `True` only if:
   - We got the full n-step lookahead (`time_to_obs == n_steps`), AND
   - The gathered future state is NOT terminal (`~future_done`)

So by the time the batch reaches `update_critic()`, the `(1-done)` masking is already baked into `batch["nonterminal"]`.

### Equivalence To Textbook

| Textbook (1-step) | This repo (n-step) | Where in code |
|---|---|---|
| $r$ | $R^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k}$ | `batch[("next", "reward")]` |
| $\gamma$ | $\gamma^n$ (or less if episode ended early) | `batch["gamma"]` |
| $(1 - \text{done})$ | `nonterminal` | `batch["nonterminal"]` |
| $s'$ | $s_{t+n}$ (the state n steps ahead) | `batch[("next", "obs")]` |

---

## 3. Loss Type 1: MSE (Default)

**Config:** `agent.critic.loss.type = "mse"`

### Math

Given the Bellman target $y$ computed above, the MSE critic loss is:

$$
\delta_b = \frac{1}{K} \sum_{k=1}^{K} \left| Q_k(s, a) - y \right|
$$

$$
L_{\text{critic}} = \frac{1}{B} \sum_{b=1}^{B} \delta_b^2
$$

Where:
- $K$ = number of critic heads (default 10)
- $B$ = batch size
- $Q_k(s, a)$ = prediction from the $k$-th critic head
- $y$ = the Bellman target (`target_q`)

**Note:** This is not a standard per-head MSE. It first averages the absolute TD error across all heads, then squares it, then averages over the batch. This means heads are coupled in their loss contribution.

### Code

**File: `resfit/rl_finetuning/off_policy/rl/q_agent.py`, lines 380–392**

```python
else:
    q_all = self.critic(obs["feat"], obs["observation.state"], action).squeeze(-1)  # [K,B]
    # Compute TD errors for prioritized experience replay (before taking mean)
    td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [B] - mean across heads

    # Apply importance sampling weights if provided (for prioritized experience replay)
    if importance_weights is not None:
        # Weight the squared TD errors by importance sampling weights
        weighted_td_errors = td_errors**2 * importance_weights
        critic_loss = weighted_td_errors.mean()
    else:
        # Mean squared error across heads and batch (uniform sampling)
        critic_loss = (td_errors**2).mean()
```

### Step By Step

1. `q_all` → shape `[K, B]` — all 10 critic heads predict Q for each batch sample
2. `target_q.unsqueeze(0)` → shape `[1, B]` — the shared Bellman target broadcast to all heads
3. `torch.abs(q_all - target_q.unsqueeze(0))` → per-head absolute TD errors `[K, B]`
4. `.mean(dim=0)` → average absolute error across heads → `[B]`
5. `td_errors**2` → square it → `[B]`
6. `.mean()` → average over batch → scalar loss

### Critic Network Output (MSE mode)

**File: `resfit/rl_finetuning/off_policy/rl/critic.py`, Critic.forward()**

```python
# MSE case: outputs are already scalars per head → [num_q, B, 1]
return logits_per_head
```

For MSE, each critic head outputs a single scalar Q-value (output_dim=1).

---

## 4. Loss Type 2: HL-Gauss (Distributional)

**Config:** `agent.critic.loss.type = "hl_gauss"`

### Math

Instead of predicting a single scalar Q-value, each critic head outputs a **histogram** (logits over bins) that represents a distribution over possible Q-values.

Given the same Bellman target $y$, the HL-Gauss loss converts $y$ into a soft Gaussian target distribution over bins, then minimizes cross-entropy:

$$
p_i^{\text{target}} = \frac{\Phi\left(\frac{b_{i+1} - y}{\sigma\sqrt{2}}\right) - \Phi\left(\frac{b_i - y}{\sigma\sqrt{2}}\right)}{Z}
$$

$$
L_{\text{HL-Gauss}} = -\sum_{i=1}^{N_{\text{bins}}} p_i^{\text{target}} \cdot \log \text{softmax}(\text{logits})_i
$$

Where:
- $\Phi$ = standard normal CDF
- $b_i$ = bin edges (linearly spaced between `v_min` and `v_max`)
- $\sigma$ = smoothing width (default: `0.75 * (v_max - v_min) / n_bins`)
- $Z$ = normalization constant
- $N_{\text{bins}}$ = 51 (default)

The Q-value is recovered as:

$$
Q = \sum_{i=1}^{N_{\text{bins}}} \text{softmax}(\text{logits})_i \cdot c_i
$$

Where $c_i$ are the bin centers.

### Code — Target To Probabilities

**File: `resfit/rl_finetuning/off_policy/rl/critic.py`, HLGaussLoss._target_to_probs()**

```python
def _target_to_probs(self, target):
    target = target.unsqueeze(-1)  # [batch] -> [batch, 1]

    # Compute normalized distances to bin edges
    norm_factor = math.sqrt(2) * self.sigma
    upper_edges = (self.bin_edges[1:].unsqueeze(0) - target) / norm_factor
    lower_edges = (self.bin_edges[:-1].unsqueeze(0) - target) / norm_factor

    # Compute log probabilities for each bin
    bin_log_probs = self._normal_cdf_log_difference(upper_edges, lower_edges)

    # Compute normalization factor (log of total probability mass)
    log_z = self._normal_cdf_log_difference(
        (self.bin_edges[-1] - target) / norm_factor, (self.bin_edges[0] - target) / norm_factor
    )

    # Convert to probabilities
    probs = torch.exp(bin_log_probs - log_z.unsqueeze(-1))

    # Safety check: if anything goes wrong, fall back to uniform
    valid_mask = torch.isfinite(probs).all(dim=-1, keepdim=True)
    uniform_probs = torch.ones_like(probs) / probs.shape[-1]
    return torch.where(valid_mask, probs, uniform_probs)
```

### Code — Loss (Cross-Entropy Against Soft Target)

**File: `resfit/rl_finetuning/off_policy/rl/critic.py`, HLGaussLoss.forward()**

```python
def forward(self, logits, target):
    tgt_probs = self._target_to_probs(target.detach())
    # Manual cross-entropy for soft labels: -sum(p * log_softmax(logits))
    log_probs = functional.log_softmax(logits, dim=-1)
    return -(tgt_probs * log_probs).sum(dim=-1).mean()
```

### Code — Usage In update_critic

**File: `resfit/rl_finetuning/off_policy/rl/q_agent.py`, lines 343–347**

```python
if self.critic.loss_cfg.type == "hl_gauss":
    # Compute logits for current Q heads and average HL-Gauss loss across heads
    q_per_head, logits_per_head = self.critic(obs["feat"], obs["observation.state"], action, return_logits=True)
    K = logits_per_head.shape[0]
    losses = [self.critic.hl_loss(logits_per_head[i], target_q) for i in range(K)]
    critic_loss = torch.stack(losses).mean()
```

### Step By Step

1. Forward pass returns both Q-values (for logging) and raw logits (for loss)
2. `logits_per_head` → shape `[K, B, 51]` — each head outputs 51 bin logits
3. For each head $k$: compute `hl_loss(logits_per_head[k], target_q)`
   - Convert scalar `target_q` into a Gaussian distribution over 51 bins
   - Cross-entropy between predicted bin probabilities and target distribution
4. Average the loss across all K heads

### How Q-Value Is Recovered

**File: `resfit/rl_finetuning/off_policy/rl/critic.py`, Critic.forward()**

```python
if self.loss_cfg.type == "hl_gauss":
    # expectation over bin centers → scalar Q, done per head
    q_per_head = self._logits_to_q(
        torch.softmax(logits_per_head, dim=-1),
        self.hl_loss.bin_centers,
    )  # [num_q, B, 1]
```

This is just: $Q = \sum_i \text{softmax}(\text{logits})_i \cdot c_i$

---

## 5. Loss Type 3: C51 (Categorical Distributional)

**Config:** `agent.critic.loss.type = "c51"`

### Math

C51 also predicts a distribution over Q-values, but uses a **categorical distribution** over fixed "atoms" (support points). Unlike HL-Gauss which converts the scalar target into a Gaussian, C51 **projects the entire next-state distribution** through the Bellman operator.

**Support:** $N$ atoms linearly spaced:
$$
z_i = v_{\min} + i \cdot \Delta z, \quad \Delta z = \frac{v_{\max} - v_{\min}}{N - 1}
$$

**Bellman projection for each atom:**
$$
\hat{z}_i = r + \gamma (1 - \text{done}) \cdot z_i
$$

Then clamp: $\hat{z}_i = \text{clamp}(\hat{z}_i, v_{\min}, v_{\max})$

**Distribute probability mass** of atom $i$ to its two nearest neighbors on the support:
$$
b = \frac{\hat{z}_i - v_{\min}}{\Delta z}, \quad l = \lfloor b \rfloor, \quad u = \lceil b \rceil
$$

$$
p_l^{\text{target}} \mathrel{+}= p_i^{\text{next}} \cdot (u - b)
$$
$$
p_u^{\text{target}} \mathrel{+}= p_i^{\text{next}} \cdot (b - l)
$$

**Loss:** Cross-entropy between predicted and projected target distribution:
$$
L_{\text{C51}} = -\sum_{i=0}^{N-1} p_i^{\text{target}} \cdot \log \text{softmax}(\text{logits})_i
$$

### Code — Support Setup

**File: `resfit/rl_finetuning/off_policy/rl/critic.py`, C51Loss.__init__()**

```python
class C51Loss(nn.Module):
    def __init__(self, v_min: float, v_max: float, num_atoms: int):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

        # Create support for value distribution
        support = torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

        # Delta z for projection
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
```

### Code — Bellman Projection

**File: `resfit/rl_finetuning/off_policy/rl/critic.py`, C51Loss.project_distribution()**

```python
def project_distribution(
    self, next_distribution: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Project Bellman update onto categorical support."""
    batch_size = rewards.size(0)

    # Compute target values for each atom: r + gamma * (1 - done) * support
    target_support = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * self.support.unsqueeze(0)

    # Clamp target support to valid range
    target_support = torch.clamp(target_support, self.v_min, self.v_max)

    # Compute indices and interpolation weights for projection
    b = (target_support - self.v_min) / self.delta_z
    lower = b.floor().long()  # Lower bound indices
    upper = b.ceil().long()  # Upper bound indices

    # Handle edge cases
    lower[(upper > 0) * (lower == upper)] -= 1
    upper[(lower < (self.num_atoms - 1)) * (lower == upper)] += 1

    # Project probabilities onto support
    target_distribution = torch.zeros_like(next_distribution)
    offset = (
        torch.linspace(
            0, (batch_size - 1) * self.num_atoms, batch_size, dtype=torch.long, device=target_distribution.device
        )
        .unsqueeze(1)
        .expand(batch_size, self.num_atoms)
    )

    # Lower bound projection
    target_distribution.view(-1).index_add_(
        0, (lower + offset).view(-1), (next_distribution * (upper.float() - b)).view(-1)
    )
    # Upper bound projection
    target_distribution.view(-1).index_add_(
        0, (upper + offset).view(-1), (next_distribution * (b - lower.float())).view(-1)
    )

    return target_distribution
```

**This is where you saw the explicit `(1 - done)` — because C51 needs to shift individual atoms, not just a scalar target.**

### Code — Cross-Entropy Loss

**File: `resfit/rl_finetuning/off_policy/rl/critic.py`, C51Loss.forward()**

```python
def forward(self, current_logits: torch.Tensor, target_distribution: torch.Tensor) -> torch.Tensor:
    """Compute C51 loss using cross-entropy between current and target distributions."""
    # Convert logits to log probabilities
    current_log_probs = functional.log_softmax(current_logits, dim=-1)

    # Compute cross-entropy loss
    return -(target_distribution * current_log_probs).sum(dim=-1).mean()
```

### Code — Usage In update_critic

**File: `resfit/rl_finetuning/off_policy/rl/q_agent.py`, lines 349–376**

```python
elif self.critic.loss_cfg.type == "c51":
    # Compute logits for current Q heads and C51 distributional loss
    q_per_head, logits_per_head = self.critic(obs["feat"], obs["observation.state"], action, return_logits=True)

    # Get next state distribution for C51 target computation
    with torch.no_grad():
        _, next_logits = self.critic_target(
            next_obs["feat"], next_obs["observation.state"], next_action, return_logits=True
        )
        # Take min over random subset of heads for next distribution
        num_heads = min(self.critic.cfg.min_q_heads, next_logits.shape[0])
        idx = torch.randperm(next_logits.shape[0], device=next_logits.device)[:num_heads]
        next_logits_min = torch.min(next_logits.index_select(0, idx), dim=0).values
        next_distribution = torch.softmax(next_logits_min, dim=-1)

        # Project the target distribution
        # The discount factor passed to this function is already discount = gamma * (1 - done)
        # For C51, we need to extract the done mask and gamma separately
        dones = (discount == 0.0).float()
        gamma = 0.99  # Assume standard gamma value
        target_distribution = self.critic.c51_loss.project_distribution(next_distribution, reward, dones, gamma)

    # Compute C51 loss for each head
    K = logits_per_head.shape[0]
    losses = [self.critic.c51_loss(logits_per_head[i], target_distribution) for i in range(K)]
    critic_loss = torch.stack(losses).mean()
```

### Step By Step

1. Get the next-state distribution from the target critic (softmax of logits)
2. Take min over random head subset for pessimism
3. Project through Bellman: shift each atom by $r + \gamma(1-\text{done}) z_i$, redistribute mass
4. For each current head: cross-entropy against the projected target distribution
5. Average losses across heads

### How Q-Value Is Recovered

**File: `resfit/rl_finetuning/off_policy/rl/critic.py`, C51Loss.logits_to_q_value()**

```python
def logits_to_q_value(self, logits: torch.Tensor) -> torch.Tensor:
    """Convert categorical logits to Q-value using expectation over support."""
    probs = functional.softmax(logits, dim=-1)
    return (probs * self.support).sum(dim=-1, keepdim=True)
```

$Q = \sum_i p_i \cdot z_i$ (expected value of the distribution)

---

## 6. Summary Table

| | MSE | HL-Gauss | C51 |
|---|---|---|---|
| **Critic output** | scalar Q (`[K, B, 1]`) | bin logits (`[K, B, 51]`) | atom logits (`[K, B, 51]`) |
| **Bellman target** | scalar $y$ | scalar $y$ (same) | projected distribution over atoms |
| **Where (1-done) lives** | `effective_discount` in q_agent | `effective_discount` in q_agent | explicit in `project_distribution()` |
| **Loss function** | $(|\bar{\delta}|)^2$ mean TD error squared | Cross-entropy vs Gaussian soft-label | Cross-entropy vs projected distribution |
| **Needs v_min/v_max?** | No | Yes | Yes |
| **File (loss class)** | inline in q_agent.py | critic.py `HLGaussLoss` | critic.py `C51Loss` |
| **File (target + loss call)** | q_agent.py L380-392 | q_agent.py L343-347 | q_agent.py L349-376 |

### The Complete Data Flow (All Types)

```
┌───────────────────────────────────────────────────────────────────────┐
│ REPLAY BUFFER (MultiStepTransform)                                     │
│                                                                         │
│  Raw transitions:  (s_t, a_t, r_t, done_t, s_{t+1})                   │
│                          ↓                                              │
│  N-step transform produces:                                             │
│    batch["(next, reward)"]  = Σ γ^k r_{t+k}     (summed reward)        │
│    batch["gamma"]           = γ^n                 (discount power)      │
│    batch["nonterminal"]     = can_bootstrap?      (the 1-done flag)     │
│    batch["(next, obs)"]     = s_{t+n}             (future state)        │
└───────────────────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────────────────┐
│ q_agent.py — update() function                                         │
│                                                                         │
│  effective_discount = batch["gamma"] * batch["nonterminal"]            │
│                          ↓                                              │
│  This IS the (1-done) masking. When nonterminal=False,                 │
│  effective_discount=0, so Q_target disappears from the backup.         │
└───────────────────────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────────────────────┐
│ q_agent.py — update_critic() function                                  │
│                                                                         │
│  target_q = reward + (discount * target_q_min)                         │
│                                                                         │
│  ┌─────────┐     ┌──────────────┐     ┌───────────────────────┐       │
│  │   MSE   │     │   HL-Gauss   │     │         C51           │       │
│  ├─────────┤     ├──────────────┤     ├───────────────────────┤       │
│  │ δ = |Q-y|    │ p = Gauss(y) │     │ project next dist     │       │
│  │ L = δ².mean │ L = -p·log(q)│     │ through r+γ(1-d)·z    │       │
│  │           │     │              │     │ L = -p_tgt·log(q)    │       │
│  └─────────┘     └──────────────┘     └───────────────────────┘       │
└───────────────────────────────────────────────────────────────────────┘
                          ↓
                    critic_loss.backward()
                    optimizer.step()
```

---

## Key Takeaway

**All 3 loss types use the same Bellman target computation.** The `(1 - done)` masking is always handled by `effective_discount = batch["gamma"] * batch["nonterminal"]`, computed from the N-step replay transform.

The only place you see a literal `(1 - done)` in the loss code is inside C51's `project_distribution()`, because C51 needs to shift individual probability atoms rather than a single scalar target. But even there, the `dones` flag is derived from `effective_discount == 0` (i.e., it is still the same terminal information, just re-extracted for the distribution projection API).
