# Value Function, Reward Types & Warmup Phases

## 1. What Is Q(s, a)?

The Q-function estimates the **total discounted future return** starting from state `s`, taking action `a`, then following the policy:

$$Q(s_t, a_t) = r_t + \gamma \cdot r_{t+1} + \gamma^2 \cdot r_{t+2} + \cdots$$

In the Bellman equation form:

$$Q(s_t, a_t) = r_t + \gamma \cdot Q(s_{t+1}, a_{t+1})$$

So $Q$ includes **both** the immediate reward $r_t$ **and** all discounted future rewards. It is *not* just the future part — the current reward is included.

The `value/q_trajectories` plot in WandB shows $Q(s_t, a_t)$ at every timestep $t$ during an eval rollout. It is the critic's **prediction of total remaining return** from that point onward.

---

## 2. Sparse vs Dense: Reward Structure

### Robosuite Lift Reward Components (with `reward_scale=1.0`)

| Component    | Value (raw) | Value (scaled ÷2.25) | When active |
|-------------|-------------|----------------------|-------------|
| Reaching     | `1 - tanh(10·dist)` ∈ [0, 1] | [0, 0.444] | Always (until success) |
| Grasping     | 0 or 0.25    | 0 or 0.111          | When cube grasped |
| **Success**  | **2.25**     | **1.0**              | Cube lifted >4cm above table |

Key: success check runs **first** (`if success → 2.25, elif shaping → reaching + grasping`). Dense sub-rewards max out at `1.25 / 2.25 = 0.556` per step. On success the reward jumps to exactly `1.0`.

| Reward Type | Per-step reward     | On success | On failure |
|-------------|--------------------:|------------|------------|
| **Sparse**  | 0                   | 1.0        | 0 every step |
| **Dense**   | 0.0 – 0.556         | 1.0        | 0.0 – 0.556 every step |

---

## 3. Q-Values: What the Charts Should Look Like

### Configuration

| Parameter | Value |
|-----------|-------|
| Discount $\gamma$ | 0.995 |
| Horizon | 100 steps |
| N-step returns | variable (default 3) |

### 3.1 Sparse Reward Q-Values

With sparse reward, $Q(s_t, a_t) = \gamma^{T-t}$ where $T$ is the success timestep. All intermediate rewards are 0.

**Q(t=0) for different success timings:**

| Scenario | Success at step | Q(t=0) |
|----------|----------------:|-------:|
| Immediate success | 0 | 1.000 |
| Fast success | 10 | 0.951 |
| Good success | 20 | 0.905 |
| Medium success | 50 | 0.778 |
| Slow success | 80 | 0.670 |
| End of horizon | 99 | 0.609 |
| **Failed** | never | **0.000** |

**Q along a trajectory** (success at step 50):

| Step $t$ | Q($s_t, a_t$) | What it means |
|---------:|--------------:|---------------|
| 0 | 0.778 | "I expect to succeed in ~50 steps" |
| 10 | 0.818 | "Success is 40 steps away" |
| 20 | 0.860 | "30 more steps to go" |
| 30 | 0.905 | "Getting closer" |
| 40 | 0.951 | "Almost there" |
| 49 | 0.995 | "Next step is success" |

**What the `q_trajectories` chart looks like (sparse):**

```
Q(s,a)
1.0 ┤                                          ╭──── success!
    │                                     ╭────╯
    │                                ╭────╯
0.8 ┤                           ╭────╯
    │                      ╭────╯
    │                 ╭────╯
0.6 ┤            ╭────╯
    │       ╭────╯
    │  ╭────╯
    ├──╯
0.0 ┤
    └──────────────────────────────────────────── Step
    0    10    20    30    40    50
```

- Starts near 0 at step 0 (before critic learns), then grows toward ~0.6–1.0
- Monotonically increasing along a successful trajectory (getting closer to reward)
- Failed trajectories stay near 0 throughout
- **Range: [0, 1]**

### 3.2 Dense Reward Q-Values

With dense reward, Q accumulates **both** per-step shaping rewards and the success bonus.

**Q(t=0) for different trajectories:**

| Scenario | Success at | Q(t=0) | Breakdown |
|----------|--------:|---------:|-----------|
| Fast success | step 20 | **8.0** | ~20 steps of shaping + discounted 1.0 |
| Medium success | step 50 | **17.2** | ~50 steps of shaping + discounted 1.0 |
| Slow success | step 90 | **27.0** | ~90 steps of shaping + discounted 1.0 |
| **Failed** (no lift) | never | **34.1** | 100 steps of max shaping, no success bonus |

**Counterintuitive result:** Failed trajectories can have **higher** Q(t=0) than successful ones! This is because:
- A failed trajectory collects shaping reward for all 100 steps (never terminates early)
- A fast success collects fewer per-step rewards before the episode ends
- The shaping sub-rewards (reaching, grasping) are denser than the success bonus

**Q along a good trajectory** (success at step 30):

| Step $t$ | r(t) | Q($s_t, a_t$) | Meaning |
|---------:|-----:|--------------:|---------|
| 0 | 0.000 | 11.29 | Total future: ~30 steps of shaping + success |
| 10 | 0.296 | 10.51 | Reaching the cube |
| 15 | 0.556 | 8.98 | Reached + grasped |
| 20 | 0.556 | 6.38 | Holding, approaching lift |
| 29 | 0.556 | 1.55 | About to succeed |

**Q along a failed trajectory** (no success, 100 steps):

| Step $t$ | r(t) | Q($s_t, a_t$) | Meaning |
|---------:|-----:|--------------:|---------|
| 0 | 0.000 | 34.13 | Full horizon of shaping ahead |
| 25 | 0.370 | 34.05 | Reaching slowly |
| 50 | 0.556 | 24.63 | Half the shaping left |
| 75 | 0.556 | 13.09 | Quarter left |
| 99 | 0.556 | 0.56 | Only this step's reward |

**What the `q_trajectories` chart looks like (dense):**

```
Q(s,a)
35 ┤ ╭──── failed (high Q from dense shaping!)
   │ │ ╲
30 ┤ │   ╲
   │ │    ╲
25 ┤ │     ╲     ╭── slow success
   │ │      ╲   ╱  ╲
20 ┤ │       ╲ ╱    ╲
   │ │        ╳      ╲
15 ┤ │       ╱╲╲      ╲
   │ ╭──── ╱   ╲╲     ╲
10 ┤ │fast╱     ╲╲     ╲
   │ │  ╱        ╲╲     ╲
 5 ┤ │╱           ╲╲     ╲
   │╱              ╲╲     ╲
 0 ┤                ╲╲     └── terminates
   └────────────────────────────────── Step
   0    20    40    60    80   100
```

- Starts **high** (10–35) and **decreases** toward 0 along the trajectory
- Successful episodes: Q drops to ~1 right before success, then episode ends
- Failed episodes: Q decreases smoothly from ~34 toward 0 at step 99
- **Range: [0, ~35]** — much larger than sparse!
- Q chart going **down** is NORMAL for dense rewards (fewer future rewards remaining)

### 3.3 Side-by-Side Comparison

| Aspect | Sparse | Dense |
|--------|--------|-------|
| **Q(t=0) range** | [0, 1] | [0, ~35] |
| **Q shape along trajectory** | ↗ increases toward 1 | ↘ decreases from high value |
| **Q at success step** | ~1.0 | ~1.0 (final step only) |
| **Failed episode Q(t=0)** | 0 | ~34 (highest!) |
| **V_MAX for distributional critics** | 1.0 | 70.0 |
| **Direction of Q curve** | Rising = "approaching reward" | Falling = "spending down future reward budget" |

---

## 4. Why Dense Q Decreases Along a Trajectory

In sparse reward, rewards are zero until success, so Q only grows as you approach the reward. Think of Q as "how close am I to the reward?" — it only goes up.

In dense reward, you receive reward at every step. Q represents the **remaining total** of all future rewards. As you collect rewards step by step, the remaining total shrinks. Think of Q as "how much reward is left in my budget?" — it goes down as you spend it.

The Bellman equation makes this clear:

$$Q(s_t, a_t) = \underbrace{r_t}_{\text{spent now}} + \gamma \cdot \underbrace{Q(s_{t+1}, a_{t+1})}_{\text{remaining budget}}$$

Each step, $r_t$ is "withdrawn" from the Q-budget, so the remaining $Q(s_{t+1})$ is smaller.

---

## 5. Training Warmup Phases

The RL training has **three distinct phases** before main training begins:

### Phase 1: Random Action Warmup (`learning_starts`, default: 10,000 steps)

| | Details |
|--|---------|
| **What happens** | Collect transitions into the online replay buffer using random actions |
| **Actions** | `base_policy(obs) + uniform_noise(±0.2)` (with `use_base_policy_for_warmup=True`) |
| **Critic updates** | **None** — zero gradient steps |
| **Actor updates** | **None** — zero gradient steps |
| **Purpose** | Fill the online buffer with diverse initial data so learning can start |

### Phase 2: Critic Warmup (`critic_warmup_steps`, default: 10,000 updates)

| | Details |
|--|---------|
| **What happens** | Train the critic (Q-function) from replay data, no new env interaction |
| **Data source** | Mixed batches: 50% offline buffer + 50% online buffer |
| **Critic updates** | **10,000** gradient steps (critic + encoder) |
| **Actor updates** | **None** — `update_actor=False` is passed explicitly |
| **Purpose** | Let Q-function converge before actor relies on Q-gradients for improvement |

### Phase 3: First Eval (`eval_first=True`, at global_step=0)

| | Details |
|--|---------|
| **When** | After both warmup phases, at the start of the main training loop |
| **Actor state** | **Untrained** — last layer is zero-initialized (`actor_last_layer_init_scale=0.0`) |
| **Actor output** | Exactly `0.0` for all dimensions (zero residual) |
| **Env action** | `base_policy(obs) + 0.0 = base_policy(obs)` — pure BC baseline |
| **Purpose** | Measures what the BC policy alone can do, before any RL improvement |

> The first eval is a **pure BC baseline measurement**. The actor outputs exactly zero,
> so the env executes only the frozen base policy's actions. This gives you the starting
> success rate that RL will improve upon.

### Phase 4: Main Training Loop

| | Details |
|--|---------|
| **Critic updates** | 4 per env step (UTD=4) |
| **Actor updates** | 1 per env step (every 4th critic update) |
| **Data source** | Mixed: offline + online buffers |
| **Actions** | `base_policy(obs) + actor(obs)` with exploration noise |

The actor starts receiving gradient updates from the **very first gradient step** of the main loop. There is no gradual switch — once warmup is done, both critic and actor train together.

### Timeline Visualization

> **Key: `global_step` starts at 0 AFTER all warmup finishes.** The 10K env steps and 10K critic
> updates happen before the main loop and are **invisible in WandB** — nothing is logged to WandB
> during warmup. All WandB charts begin at `global_step=0`.

```
                    INVISIBLE TO WANDB
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  Phase 1: Fill Online Buffer                │
    │  ─────────────────────────                  │
    │  • 10K env steps with BC + noise            │
    │  • Actions = base_policy(obs) + noise(±0.2) │
    │  • NO critic updates (zero gradient steps)  │
    │  • NO actor updates (zero gradient steps)   │
    │  • NO WandB logging                         │
    │  • Result: online buffer has 10K transitions │
    │                                             │
    │  Phase 2: Critic Warmup                     │
    │  ──────────────────────                     │
    │  • NO new env interaction (no new data)     │
    │  • 10K critic gradient updates              │
    │  • Batches: 50% online buf + 50% offline buf│
    │  • Actor gets ZERO updates                  │
    │  • NO WandB logging*                        │
    │  • Only prints to terminal every 100 steps  │
    │  • Result: critic already trained on 10K    │
    │    batches, actor still outputs all zeros   │
    │                                             │
    └─────────────────────────────────────────────┘
                         │
                         ▼
              ╔══════════════════════╗
              ║  global_step = 0     ║  ← THIS IS WHERE WANDB STARTS
              ╚══════════════════════╝
                         │
                         ▼
    ┌─────────────────────────────────────────────┐
    │  First Eval (at global_step=0)              │
    │  ────────────────────────────               │
    │  • Actor outputs exactly 0 (zero-init)      │
    │  • Env runs: base_policy(obs) + 0 = pure BC │
    │  • NO exploration noise during eval         │
    │  • Logs to WandB at step=0:                 │
    │    - eval/success_rate (= BC baseline ~74%) │
    │    - value/q_trajectories                   │
    │      (critic already knows stuff from 10K   │
    │       warmup, so Q-values are NOT random)   │
    └─────────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────┐
    │  Main Training Loop (global_step=1,2,3,...) │
    │  ──────────────────────────────────────     │
    │  Each step:                                 │
    │  1. Collect: env step with actor+noise      │
    │  2. Train: UTD=4 critic updates             │
    │            + 1 actor update (every 4th)     │
    │  3. Log to WandB (train/critic_qt, etc.)    │
    │  4. Eval every N steps → WandB eval metrics │
    │                                             │
    │  • Actor receives gradients from step 1     │
    │  • Critic continues training (had 10K head  │
    │    start but that's not in the step count)  │
    └─────────────────────────────────────────────┘
```

### What This Means for Your WandB Charts

1. **`global_step=0` in WandB is actually step 20K of wall-clock time** — 10K env steps + 10K critic updates happened before this point, but the code sets `global_step = 0` and that's what WandB uses as the x-axis.

2. **The first `eval/success_rate` at step 0** = pure BC baseline (~74%). The actor hasn't learned anything yet, so residual = 0, and the env executes only the frozen base policy.

3. **The first `value/q_trajectories` at step 0** is already meaningful — the critic has been training for 10K steps during warmup, so it has a rough estimate of Q-values. These Q-predictions come from the warmup-trained critic evaluating the pure-BC rollout. They won't be random.

4. **`train/critic_qt` starts at step 1** (first main loop iteration). The critic's 10K warmup updates are NOT reflected in this chart — you only see the post-warmup values. So `critic_qt` at step 1 already reflects a partially-trained critic, not a randomly-initialized one.

5. **Both actor and critic train from step 1** — the actor starts from zero and the critic continues from its 10K warmup state. There's no gradual transition.

### What `train/critic_qt` Shows During Warmup

During **Phase 2** (critic warmup), the critic is training on buffered data. You'll see `train/critic_qt` (the mean target Q-value across training batches) evolve:

- **Sparse**: Starts near 0, slowly increases as the critic learns to predict success probability. Most transitions have reward=0, so Q stays small unless the critic sees success transitions from the offline buffer.
- **Dense**: Starts higher (5–20) because most transitions have non-zero reward. The critic quickly learns to sum up the shaping rewards across steps.

---

## 6. N-Step Returns: How They Work and Why They Matter Differently

### 6.1 What the Code Does

The `MultiStepTransform` in `rb_transforms.py` is applied **at sampling time** from the replay buffer. When a transition at step $t$ is sampled, it transforms it:

**1-step (n=1)** — standard TD:

$$y_t = r_t + \gamma \cdot Q(s_{t+1}, a_{t+1})$$

**n-step** — multi-step TD:

$$y_t = \underbrace{r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1} r_{t+n-1}}_{R_n\text{ (n-step reward sum)}} + \gamma^n \cdot Q(s_{t+n}, a_{t+n})$$

The transform modifies three fields in the sampled batch:

| Field | Meaning |
|-------|---------|
| `next.reward` | Replaced with $R_n$ — the discounted sum of the next $n$ rewards |
| `next.obs` | Replaced with $s_{t+n}$ — the observation $n$ steps ahead |
| `gamma` | Set to $\gamma^{n'}$ where $n'$ is the actual number of steps taken |
| `nonterminal` | True only if we took the full $n$ steps **and** $s_{t+n}$ is not terminal |

At episode boundaries: if done=True at step $t+k$ where $k < n$, the sum **truncates** at step $k$, and `nonterminal=False` (no bootstrapping — the target is pure Monte Carlo for those last steps).

In `q_agent.py`, the Bellman target becomes:

```python
effective_discount = batch["gamma"] * batch["nonterminal"]   # γ^n if nonterminal, else 0
target_q = reward + effective_discount * Q_target(s_{t+n})
```

### 6.2 Core Insight: Direct Signal vs Bootstrapping

Every transition sampled from the replay buffer falls into one of two categories:

| Category | What happens | How fast the critic learns |
|----------|-------------|--------------------------|
| **Direct signal** (★) | $R_n > 0$ — the n-step reward sum is nonzero | Fast: critic gets explicit supervision |
| **Bootstrap only** (·) | $R_n = 0$ — must rely entirely on $Q(s_{t+n})$ | Slow: depends on Q being accurate already |

With bootstrapping-only transitions, learning is a chain reaction: the critic at $s_{t+n}$ must already be accurate for $s_t$ to learn. Information propagates **one step per training iteration**. With direct signal, the critic learns immediately from the reward.

**N-step increases the number of direct-signal transitions** — but only when rewards are sparse.

### 6.3 Sparse vs Dense: The Key Difference

**Sparse reward**: Only the single success step has $r \neq 0$. With n=1, only that one transition carries signal. With n=3, the 3 transitions preceding success also get nonzero $R_n$.

**Dense reward**: Every step has $r > 0$. Even with n=1, **all** transitions carry direct signal. Increasing n just makes $R_n$ slightly larger, but doesn't unlock new information.

| | Sparse (n=1) | Sparse (n=3) | Dense (n=1) | Dense (n=3) |
|-|:---:|:---:|:---:|:---:|
| **Transitions with direct signal** | 1/10 | 3/10 | **10/10** | **10/10** |
| **Benefit of increasing n** | **3× more signal** | — | negligible | — |

### 6.4 Trade-offs of Higher n

| | Pro | Con |
|-|-----|-----|
| **Higher n** | More reward signal per transition (big win for sparse) | Higher variance: the n-step sum accumulates noise from $n$ actions. Also uses a less-accurate bootstrap (Q at $s_{t+n}$ is harder to predict than $s_{t+1}$) |
| **Lower n** | Low variance, more reliance on Q accuracy | Slower credit assignment for sparse rewards |

| n | Sparse | Dense |
|--:|--------|-------|
| 1 | Slow — reward signal propagates 1 step per training iteration | Fine — every transition already has signal |
| 3 | Good — 3× faster propagation, moderate variance | Marginal — slightly reduces bootstrap bias |
| 5 | Better — 5× faster propagation, more variance | Risk — higher variance may hurt with no benefit to signal coverage |

---

## 7. Worked Example: 10-Step Episode, Step by Step

A concrete episode with **10 steps** (t=0 to t=9), **success at step 9** (done=True), $\gamma = 0.995$.

### 7.1 Reward Sequences

| Step $t$ | $r_t$ (sparse) | $r_t$ (dense) | What's happening |
|---------:|:--------------:|:-------------:|------------------|
| 0 | 0 | 0.10 | Arm starts moving toward cube |
| 1 | 0 | 0.15 | Approaching |
| 2 | 0 | 0.20 | Closer |
| 3 | 0 | 0.25 | Near cube |
| 4 | 0 | 0.35 | Contact |
| 5 | 0 | 0.40 | Grasping |
| 6 | 0 | 0.45 | Grasped, lifting |
| 7 | 0 | 0.50 | Lifting |
| 8 | 0 | 0.55 | Almost there |
| 9 | **1.0** | **1.0** | **Success! Cube lifted.** |

### 7.2 Sparse Reward — What the Critic Sees

Each row = one transition sampled from replay buffer. $R_n$ = n-step reward sum. ★ = has direct reward signal (the critic can learn from this without needing Q to be accurate). · = must bootstrap (learning depends on Q being correct at the bootstrap state).

**n=1** — only 1 transition (step 9) carries signal, 9 transitions are blind:

| $t$ | $R_1$ | Bootstrap? | TD Target $y_t$ | Signal? |
|----:|------:|:----------:|-----------------|:-------:|
| 0 | 0 | $\gamma Q(s_1)$ | $0 + 0.995 \cdot Q(s_1)$ | · |
| 1 | 0 | $\gamma Q(s_2)$ | $0 + 0.995 \cdot Q(s_2)$ | · |
| 2 | 0 | $\gamma Q(s_3)$ | $0 + 0.995 \cdot Q(s_3)$ | · |
| 3 | 0 | $\gamma Q(s_4)$ | $0 + 0.995 \cdot Q(s_4)$ | · |
| 4 | 0 | $\gamma Q(s_5)$ | $0 + 0.995 \cdot Q(s_5)$ | · |
| 5 | 0 | $\gamma Q(s_6)$ | $0 + 0.995 \cdot Q(s_6)$ | · |
| 6 | 0 | $\gamma Q(s_7)$ | $0 + 0.995 \cdot Q(s_7)$ | · |
| 7 | 0 | $\gamma Q(s_8)$ | $0 + 0.995 \cdot Q(s_8)$ | · |
| 8 | 0 | $\gamma Q(s_9)$ | $0 + 0.995 \cdot Q(s_9)$ | · |
| 9 | **1.0** | terminal | **1.0** | ★ |

> **1/10** transitions have signal. The reward at step 9 must chain-propagate backwards through Q: $s_9 \to s_8 \to s_7 \to \cdots \to s_0$. Learning from $s_0$ requires Q at $s_1$ to be accurate, which requires Q at $s_2$, etc. — a 9-step chain.

**n=2** — steps 8–9 now carry signal:

| $t$ | $R_2$ | Bootstrap? | TD Target $y_t$ | Signal? |
|----:|------:|:----------:|-----------------|:-------:|
| 0 | 0 | $\gamma^2 Q(s_2)$ | $0 + 0.990 \cdot Q(s_2)$ | · |
| 1 | 0 | $\gamma^2 Q(s_3)$ | $0 + 0.990 \cdot Q(s_3)$ | · |
| 2 | 0 | $\gamma^2 Q(s_4)$ | $0 + 0.990 \cdot Q(s_4)$ | · |
| 3 | 0 | $\gamma^2 Q(s_5)$ | $0 + 0.990 \cdot Q(s_5)$ | · |
| 4 | 0 | $\gamma^2 Q(s_6)$ | $0 + 0.990 \cdot Q(s_6)$ | · |
| 5 | 0 | $\gamma^2 Q(s_7)$ | $0 + 0.990 \cdot Q(s_7)$ | · |
| 6 | 0 | $\gamma^2 Q(s_8)$ | $0 + 0.990 \cdot Q(s_8)$ | · |
| 7 | 0 | $\gamma^2 Q(s_9)$ | $0 + 0.990 \cdot Q(s_9)$ | · |
| 8 | **0.995** | terminal | **0.995** | ★ |
| 9 | **1.0** | terminal | **1.0** | ★ |

> **2/10** transitions have signal. $R_2$ at step 8 = $0 + \gamma \cdot 1.0 = 0.995$. The bootstrap chain is now only 8 steps, not 9.

**n=3** — steps 7–9 now carry signal:

| $t$ | $R_3$ | Bootstrap? | TD Target $y_t$ | Signal? |
|----:|------:|:----------:|-----------------|:-------:|
| 0 | 0 | $\gamma^3 Q(s_3)$ | $0 + 0.985 \cdot Q(s_3)$ | · |
| 1 | 0 | $\gamma^3 Q(s_4)$ | $0 + 0.985 \cdot Q(s_4)$ | · |
| 2 | 0 | $\gamma^3 Q(s_5)$ | $0 + 0.985 \cdot Q(s_5)$ | · |
| 3 | 0 | $\gamma^3 Q(s_6)$ | $0 + 0.985 \cdot Q(s_6)$ | · |
| 4 | 0 | $\gamma^3 Q(s_7)$ | $0 + 0.985 \cdot Q(s_7)$ | · |
| 5 | 0 | $\gamma^3 Q(s_8)$ | $0 + 0.985 \cdot Q(s_8)$ | · |
| 6 | 0 | $\gamma^3 Q(s_9)$ | $0 + 0.985 \cdot Q(s_9)$ | · |
| 7 | **0.990** | terminal | **0.990** | ★ |
| 8 | **0.995** | terminal | **0.995** | ★ |
| 9 | **1.0** | terminal | **1.0** | ★ |

> **3/10** transitions have signal. The bootstrap chain shrinks from 9 (n=1) to 7 (n=3).

### 7.3 Dense Reward — What the Critic Sees

**n=1** — already all 10 transitions carry signal:

| $t$ | $r_t$ | $R_1$ | Bootstrap? | TD Target $y_t$ | Signal? |
|----:|------:|------:|:----------:|-----------------|:-------:|
| 0 | 0.10 | 0.10 | $0.995 Q(s_1)$ | $0.10 + 0.995 Q(s_1)$ | ★ |
| 1 | 0.15 | 0.15 | $0.995 Q(s_2)$ | $0.15 + 0.995 Q(s_2)$ | ★ |
| 2 | 0.20 | 0.20 | $0.995 Q(s_3)$ | $0.20 + 0.995 Q(s_3)$ | ★ |
| 3 | 0.25 | 0.25 | $0.995 Q(s_4)$ | $0.25 + 0.995 Q(s_4)$ | ★ |
| 4 | 0.35 | 0.35 | $0.995 Q(s_5)$ | $0.35 + 0.995 Q(s_5)$ | ★ |
| 5 | 0.40 | 0.40 | $0.995 Q(s_6)$ | $0.40 + 0.995 Q(s_6)$ | ★ |
| 6 | 0.45 | 0.45 | $0.995 Q(s_7)$ | $0.45 + 0.995 Q(s_7)$ | ★ |
| 7 | 0.50 | 0.50 | $0.995 Q(s_8)$ | $0.50 + 0.995 Q(s_8)$ | ★ |
| 8 | 0.55 | 0.55 | $0.995 Q(s_9)$ | $0.55 + 0.995 Q(s_9)$ | ★ |
| 9 | 1.00 | 1.00 | terminal | **1.00** | ★ |

> **10/10** — every transition already has direct signal.

**n=3** — $R_3$ is larger but signal count unchanged:

| $t$ | $R_3$ | Bootstrap? | TD Target $y_t$ | Signal? |
|----:|------:|:----------:|-----------------|:-------:|
| 0 | 0.447 | $0.985 Q(s_3)$ | $0.447 + 0.985 Q(s_3)$ | ★ |
| 1 | 0.597 | $0.985 Q(s_4)$ | $0.597 + 0.985 Q(s_4)$ | ★ |
| 2 | 0.795 | $0.985 Q(s_5)$ | $0.795 + 0.985 Q(s_5)$ | ★ |
| 3 | 0.994 | $0.985 Q(s_6)$ | $0.994 + 0.985 Q(s_6)$ | ★ |
| 4 | 1.194 | $0.985 Q(s_7)$ | $1.194 + 0.985 Q(s_7)$ | ★ |
| 5 | 1.343 | $0.985 Q(s_8)$ | $1.343 + 0.985 Q(s_8)$ | ★ |
| 6 | 1.492 | $0.985 Q(s_9)$ | $1.492 + 0.985 Q(s_9)$ | ★ |
| 7 | **2.037** | terminal | **2.037** | ★ |
| 8 | **1.545** | terminal | **1.545** | ★ |
| 9 | **1.00** | terminal | **1.00** | ★ |

> **10/10** — same signal count. $R_3$ is ~3× larger than $R_1$ (sums 3 rewards), and the last 3 steps no longer bootstrap (pure MC targets), but no new transitions become informative.

### 7.4 Summary: N-Step Impact at a Glance

| | n=1 | n=2 | n=3 | n=5 |
|-|:---:|:---:|:---:|:---:|
| **Sparse: transitions with signal** | 1/10 (10%) | 2/10 (20%) | 3/10 (30%) | 5/10 (50%) |
| **Sparse: bootstrap chain length** | 9 steps | 8 steps | 7 steps | 5 steps |
| **Dense: transitions with signal** | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| **Dense: bootstrap chain length** | 9 steps | 8 steps | 7 steps | 5 steps |

For sparse rewards, going from n=1 to n=5 is a **5× improvement** in direct supervision. For dense rewards, it's **unchanged** — the benefit is only the minor reduction in bootstrap chain length (less compounding Q-error).

### 7.5 Practical Recommendation

| Reward type | Best n | Reasoning |
|-------------|:------:|-----------|
| **Sparse** | 3–5 | Big win: more signal per transition, faster credit assignment. n=5 is the codebase default for original tasks. |
| **Dense** | 1–3 | Small win: reduces bootstrap bias slightly. But higher n adds variance from summing noisy rewards. n=1 is perfectly fine. |

---

## 8. Quick Reference: What to Expect in WandB

| WandB Metric | Sparse (healthy) | Dense (healthy) |
|---|---|---|
| `train/critic_qt` | 0 → grows to 0.6–1.0 | 5–25 → stabilizes around 15–25 |
| `value/q_trajectories` | Rising curves from ~0 to ~1 | Falling curves from ~20–35 to ~0 |
| `eval/success_rate` | 0.7 → 0.9+ over 20K steps | Should also improve, but from higher Q baseline |
| `train/critic_loss` | Decreasing | Decreasing |
| First eval success | = BC baseline (~74%) | = BC baseline (~74%) |
