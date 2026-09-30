# Original QAgent vs IBRL QAgent — Detailed Comparison & Notes

> **Author notes**: This document is for evaluation and note-taking purposes.
> The IBRL-style `QAgent_ibrl` code was written by a collaborator; I am testing,
> evaluating, and documenting differences.  **No code changes** are proposed here.

---

## Table of Contents

1. [High-Level Summary](#1-high-level-summary)
2. [What Action Does the Critic See?](#2-what-action-does-the-critic-see)
3. [Action Flow: Environment Execution](#3-action-flow-environment-execution)
4. [The `eps_greedy` Parameter — Full Explanation](#4-the-eps_greedy-parameter--full-explanation)
5. [Critical Issue: Is Online RL Wasted?](#5-critical-issue-is-online-rl-wasted)
6. [Missing WandB Logging: BC-vs-RL Selection Frequency](#6-missing-wandb-logging-bc-vs-rl-selection-frequency)
7. [Side-by-Side Code Comparison](#7-side-by-side-code-comparison)
8. [Summary of Differences](#8-summary-of-differences)

---

## 1. High-Level Summary

| Aspect | Original `QAgent` | IBRL `QAgent_ibrl` |
|--------|-------------------|---------------------|
| **Action selection** | Always returns `residual` | Returns `residual` or `zeros` based on Q-gating |
| **What critic evaluates** | `Q(s, base + residual)` — the **combined** action | `Q(s, base + residual)` vs `Q(s, base)` — compares both |
| **Actor training** | Maximize `Q(s, base + residual)` | **Same** — maximize `Q(s, base + residual)` |
| **Critic training** | Trained on `(s, combined_action)` from replay buffer | **Same** — trained on `(s, combined_action)` from replay buffer |
| **Conservatism** | None — always applies residual | Conservative — only applies residual when critic thinks it helps |

---

## 2. What Action Does the Critic See?

### In Both Implementations: The Critic is Trained on **Combined Actions**

This is a crucial point. The **replay buffer stores the combined (base + residual) action**, NOT just the residual.

**Evidence from the training script** (`train_residual_td3.py`):

```python
# Line ~771, ~1024 — Both warmup and main training loop:
combined_action = info["scaled_action"]
_add_transitions_to_buffer(
    ...,
    actions=combined_action,  # ← This is base + residual (clamped to [-1, 1])
    ...
)
```

And in the environment wrapper (`residual_env_wrapper.py`):

```python
# BasePolicyVecEnvWrapper.step():
combined_naction = self._last_base_naction + residual_naction
env_action = self.action_scaler.unscale(combined_naction)
raw_obs, reward, terminated, truncated, info = self.vec_env.step(env_action)
info["scaled_action"] = combined_naction  # ← stored for replay buffer
```

So the replay buffer `action` field contains: `clamp(base_naction + residual_naction, -1, 1)`.

### Critic `update_critic()` — Both are Consistent

In **both** `QAgent` and `QAgent_ibrl`, the critic's `update_critic()` method receives `action` from the replay buffer batch:

```python
# q_agent.py and ibrl_q_agent.py — identical logic:
action: torch.Tensor = batch["action"]  # This IS the combined action
...
q_all = self.critic(obs["feat"], obs["observation.state"], action)
```

And for the **target Q** (Bellman backup), both compute the next action as:

```python
next_action = torch.clamp(
    next_obs["observation.base_action"] + next_residual_action, -1.0, 1.0
)
target_q = self.critic_target.q_value(
    next_obs["feat"], next_obs["observation.state"], next_action
)
```

**Conclusion**: The critic in both versions is trained on `Q(s, base + residual)` —
the full combined action. This is correct and consistent.

---

## 3. Action Flow: Environment Execution

### Original QAgent

```
actor.act(obs) → residual                      # actor outputs residual
                                                # (scaled by action_scale, e.g. 0.1-0.2)
env_wrapper.step(residual)                      # wrapper does:
    combined = base_action + residual           #   add base
    env_action = unscale(combined)              #   convert to env space
    env.step(env_action)                        #   execute
```

The residual is **always** applied. The actor output goes straight to the env wrapper.

### IBRL QAgent

```
actor.act(obs) → calls _act_ibrl():
    residual = actor_network(obs)               # get RL residual
    candidate_0 = clamp(base + residual, -1, 1) # RL-augmented action
    candidate_1 = base                          # pure BC action
    Q_0 = critic_target(obs, candidate_0)       # Q for RL
    Q_1 = critic_target(obs, candidate_1)       # Q for BC
    if Q_0 > Q_1:
        return residual                         # "use RL"
    else:
        return zeros                            # "ignore RL, just use BC"

env_wrapper.step(selected_residual)             # wrapper does:
    combined = base_action + selected_residual  #   if zeros → combined = base only
    ...                                         #   if residual → combined = base + RL
```

The key insight: returning `zeros` from `_act_ibrl()` means the env wrapper applies
`base + 0 = base`, effectively executing pure BC.

### Important Consistency Note

In `_act_ibrl()`, the critic evaluates candidates as **full combined actions**:

```python
full_rl_action = torch.clamp(base_action + residual, -1.0, 1.0)  # candidate 0
bc_action = base_action                                            # candidate 1
```

This is **consistent** with how the critic was trained (on combined actions), so the
Q-comparison is valid.

### Critic Ensemble Aggregation in Q-Gating

The two candidates are **not** evaluated separately. They are batched together into a
single tensor of shape `[B*2, action_dim]` and passed through the critic in one forward
call:

```python
flat_actions = rl_bc_actions.flatten(0, 1)          # [B*2, action_dim]
qa = self.critic_target.q_value(flat_qfeats, flat_props, flat_actions)  # [B*2, 1]
```

Inside `Critic.q_value()` (see `critic.py`):

1. **All K ensemble heads** score every row: `q_out = self.forward(...)` → `[num_q, B*2, 1]`
2. A **random subset** of `min_q_heads` heads is selected via `torch.randperm(num_q)[:min_q_heads]`
3. The **min** over that subset gives one scalar per candidate: `[B*2, 1]`

Back in `_act_ibrl()`:

4. Reshape to `qa.view(B, 2)` — columns are `[Q(base+residual), Q(base)]`
5. `argmax(1)` picks the candidate with the higher pessimistic Q-value

**Key detail:** Because both candidates are batched into one `[B*2]` tensor through a
single `q_value()` call, the **same random head subset** (same `randperm`) judges both
candidates. This ensures a fair apples-to-apples comparison — neither candidate gets a
luckier or more pessimistic draw of critic heads.

---

## 4. The `eps_greedy` Parameter — Full Explanation

### What It Is

`eps_greedy` is the probability of using the **greedy** (Q-gated) action selection
in `_act_ibrl()`. It controls exploration diversity during action selection.

### How It Works

```python
def _act_ibrl(self, ..., eps_greedy: float, ...):
    # ... compute greedy selection (Q-gated) ...
    
    # Epsilon-greedy exploration (ONLY during training, NOT eval):
    if not eval_mode and eps_greedy < 1.0:
        eps = torch.rand((bsize, 1), device=qa.device)
        use_greedy = (eps < eps_greedy).float()
        # With prob (1 - eps_greedy): pick a RANDOM candidate (RL or BC)
        # With prob eps_greedy: use the greedy Q-gated selection
        rand_action_idx = torch.randint(0, num_action, (bsize,), device=qa.device)
        rand_is_residual = (rand_action_idx == 0).unsqueeze(1)
        rand_selected_residual = torch.where(rand_is_residual, residual, zero_residual)
        selected_residual = rand_selected_residual * (1 - use_greedy) + selected_residual * use_greedy
```

### Value Ranges and Effects

| `eps_greedy` value | Behaviour |
|--------------------|-----------|
| **1.0** (current default) | **Always greedy** — Q-gated switching decides every time. No random exploration in the RL-vs-BC choice. |
| **0.9** | 90% greedy (Q picks), 10% random pick between RL and BC |
| **0.5** | 50/50 between Q-gated and random selection |
| **0.0** | **Always random** — ignores Q-values entirely, 50/50 coin flip between RL and BC each step |

### Where It's Called

```python
# In act() — evaluation time:
action = self._act_ibrl(
    ...,
    eps_greedy=1.0,  # ← HARDCODED to 1.0 (always greedy)
    ...
)

# In update_critic() — Bellman target computation:
next_residual_action = self._act_ibrl(
    ...,
    eps_greedy=1.0,  # ← HARDCODED to 1.0 (always greedy)
    ...
)
```

**Both call sites hardcode `eps_greedy=1.0`.**

This means:
- The epsilon-greedy branch (`if not eval_mode and eps_greedy < 1.0`) is **never entered**.
- The Q-gated selection is **always deterministic** (pick whichever candidate has higher Q).

The `eps_greedy` parameter exists as a hook for future experimentation but is
currently inactive. If you wanted to add exploration diversity to the RL-vs-BC
choice, you would lower it below 1.0 during training data collection.

---

## 5. Critical Issue: Is Online RL Wasted?

### The Concern

> "eps_greedy is set to 1.0, so we always give residual as 0's, so we are
> doing online RL for nothing?"

### The Answer: **No, that's not what happens.**

`eps_greedy=1.0` means **always use greedy Q-gated selection** (not "always return zeros"). 

The Q-gated selection picks:
- **residual** (RL action) when `Q(s, base + residual) > Q(s, base)`
- **zeros** (pure BC) when `Q(s, base) >= Q(s, base + residual)`

So the output depends on which candidate has higher Q-value. The actor is being
trained to maximize `Q(s, base + residual)`, so as training progresses:

1. Early training: critic is uncertain → Q-gating may frequently pick BC (zeros)
2. Mid training: actor learns good residuals → Q-gating starts preferring RL
3. Late training: actor produces helpful residuals → Q-gating mostly picks RL

**The RL training is not wasted** — it's being used, but only when the critic
believes it's beneficial. This is the whole point of IBRL: conservative application
of learned residuals.

### However, There IS a Subtle Issue...

During **online data collection** in the training loop (not visible in this file),
the agent's `act()` method is called. Since `act()` internally calls `_act_ibrl()`
with `eps_greedy=1.0` and `eval_mode=False`:

1. The **actor still uses exploration noise** (`stddev > 0` during training) when
   computing the residual
2. But the **RL-vs-BC choice is always greedy** (no exploration in which candidate
   to pick)

This means: if the critic is poorly calibrated early on and always prefers BC,
the agent will **never explore with RL actions**, creating a chicken-and-egg problem.
The critic can't learn that RL actions are good if it never sees them executed.

**Wait — actually look more carefully:** In this IBRL implementation, `act()` is
only called during **evaluation** (the `assert not self.training` check). During
**training data collection**, the original training script uses its own exploration
logic (random noise during warmup, then `agent.act()` for online collection).

So the real question is: during online collection after warmup, does the agent
use `_act_ibrl()` or `_act_default()`? Since `QAgent_ibrl.act()` calls `_act_ibrl()`,
yes — online collected actions are Q-gated. If the critic always prefers BC early
on, exploration of RL residuals would be limited. This is a known limitation of
IBRL-style approaches.

---

## 6. Missing WandB Logging: BC-vs-RL Selection Frequency

### Current State

The `QAgent_ibrl` code does **NOT** log how often the critic picks BC vs RL.
There is no WandB metric tracking the Q-gating decision frequency.

### What Should Be Logged

To monitor IBRL behaviour, these metrics would be valuable:

1. **`ibrl/rl_selected_pct`** — % of batch where critic picked RL residual (idx=0)
2. **`ibrl/bc_selected_pct`** — % of batch where critic picked BC (idx=1)  
3. **`ibrl/q_rl_mean`** — Mean Q-value of the RL candidate
4. **`ibrl/q_bc_mean`** — Mean Q-value of the BC candidate
5. **`ibrl/q_advantage_mean`** — Mean of `Q(RL) - Q(BC)` (how much critic prefers RL)

These would give a plot showing: early on, BC is preferred → over time, RL gets
preferred more and more (if training is working).

### Where to Add (Notes for Future)

In `_act_ibrl()`, after computing `greedy_action_idx`:

```python
# NOTES: This is where logging should be added (not changing code now)
# rl_selected = (greedy_action_idx == 0).float()
# metrics could include:
#   "ibrl/rl_selected_pct": rl_selected.mean().item() * 100
#   "ibrl/q_rl_mean": qa.view(bsize, num_action)[:, 0].mean().item()
#   "ibrl/q_bc_mean": qa.view(bsize, num_action)[:, 1].mean().item()
#   "ibrl/q_advantage_mean": (qa.view(bsize, num_action)[:, 0] - qa.view(bsize, num_action)[:, 1]).mean().item()
```

The challenge is that `_act_ibrl()` currently returns only the action tensor,
not metrics. To add logging, the method signature would need to return a
metrics dict as well, or the logging would need to happen at the call site.

In the `act()` method specifically (which is called during eval):
```python
# Could be done like:
# self._last_ibrl_stats = {
#     "rl_selected_pct": (greedy_action_idx == 0).float().mean().item() * 100,
#     ...
# }
```

Then the evaluation function in `evaluate_dexmg.py` could read and aggregate
`agent._last_ibrl_stats` per step.

For **training** (in `update_critic`), the target action computation also uses
`_act_ibrl()`, so similar stats could be logged there.

---

## 7. Side-by-Side Code Comparison

### `act()` Method

**Original QAgent:**
```python
def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
    obs["feat"] = self._encode(obs, augment=False)
    action = self._act_default(obs=obs, eval_mode=eval_mode, stddev=stddev,
                                clip=None, use_target=False)
    return action  # ← always returns the raw residual
```

**IBRL QAgent:**
```python
def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
    obs["feat"] = self._encode(obs, augment=False)
    action = self._act_ibrl(obs=obs, eval_mode=eval_mode, stddev=stddev,
                             clip=None, eps_greedy=1.0, use_target=False)
    return action  # ← returns residual OR zeros (Q-gated)
```

### `update_critic()` — Target Action Computation

**Original QAgent:**
```python
next_residual_action = self._act_default(
    obs=next_obs, eval_mode=not self.cfg.target_action_noise,
    stddev=stddev, clip=self.cfg.stddev_clip, use_target=True,
)
next_action = torch.clamp(
    next_obs["observation.base_action"] + next_residual_action, -1.0, 1.0
)
```
→ Target always uses `base + residual` for Bellman backup.

**IBRL QAgent:**
```python
next_residual_action = self._act_ibrl(
    obs=next_obs, eval_mode=not self.cfg.target_action_noise,
    stddev=stddev, clip=self.cfg.stddev_clip,
    eps_greedy=1.0, use_target=True,
)
next_action = torch.clamp(
    next_obs["observation.base_action"] + next_residual_action, -1.0, 1.0
)
```
→ Target uses Q-gated selection: `base + residual` if RL is better, or `base + 0 = base` if BC is better.

**Key difference**: The Bellman target in IBRL is computed under the **IBRL policy**
(Q-gated), making the critic learn Q-values consistent with the actual IBRL
execution policy. This is correct — the critic should learn Q-values for the policy
that will actually be deployed.

### `_compute_actor_loss()` — Identical in Both

```python
# Both versions — the actor is ALWAYS trained to maximize Q(s, base + residual):
action_pred = self._act_default(obs=obs, ...)  # ← uses _act_default, NOT _act_ibrl!
combined_action = torch.clamp(obs["observation.base_action"] + action_pred, -1.0, 1.0)
q = self.critic.q_value_for_policy(obs["feat"], obs["observation.state"], combined_action)
actor_loss = -q.mean()
```

**Important**: The actor loss uses `_act_default()` in BOTH implementations.
The Q-gating only affects action **selection**, not actor **training**.
The actor always learns to produce good residuals regardless.

---

## 8. Summary of Differences

### What's the Same

- Critic is trained on **combined actions** (base + residual) from replay buffer
- Actor is trained to maximize `Q(s, base + residual)` — pure policy gradient
- Encoder, network architectures, optimizers — all identical
- Data collection and replay buffer storage — identical

### What's Different

| Component | Original | IBRL |
|-----------|----------|------|
| `act()` calls | `_act_default()` | `_act_ibrl()` (Q-gated) |
| Action output | Always `residual` | `residual` if Q says RL is better, else `zeros` |
| Bellman target | `base + target_residual` always | `base + target_residual` only if Q prefers it |
| Extra computation | None | 2x critic forward pass at action selection (for Q comparison) |
| Exploration of RL-vs-BC choice | N/A (always RL) | Controlled by `eps_greedy` (currently always greedy) |

### Key Insight

The original QAgent **trusts the RL residual unconditionally**. If the actor
produces a bad residual, it gets applied anyway.

The IBRL QAgent **uses the critic as a safety net**. If the critic thinks the
residual makes things worse, it's ignored. This is conservative — it can't do
worse than BC *if the critic is well-calibrated*. But if the critic is poorly
calibrated (especially early in training), it may reject good residuals or
accept bad ones.

### Potential Concern: Exploration Bottleneck

With `eps_greedy=1.0` (always greedy Q-gating), if the critic is pessimistic
about RL residuals early on, it will always select BC. This means:
- The replay buffer gets filled with BC-only trajectories
- The actor gradient signal comes only from the critic (which hasn't seen RL
  actions executed), creating a potential bootstrapping problem

In the original IBRL paper, this is mitigated by:
1. Pre-filling the buffer with BC demonstrations (which we do via offline data)
2. The actor still trains via policy gradient regardless of Q-gating
3. Over time, the critic learns that some residuals help and starts selecting them

But in practice, monitoring the RL selection frequency (see section 6) is
essential to verify that this bootstrapping actually happens.
