# Notes

## Seeding & Randomization in residual-offpolicy-rl

### TL;DR

| What | Seeded? | How | Actually deterministic? |
|---|---|---|---|
| PyTorch (CPU + CUDA) | RL: YES, BC: only if `--seed` passed | `torch.manual_seed()` + `torch.cuda.manual_seed_all()` | YES (weight init, dropout, etc.) |
| NumPy global (`np.random`) | RL: YES, BC: only if `--seed` passed | `np.random.seed()` | YES — and **this is what robosuite's object placement uses** |
| Python `random` module | RL: YES, BC: only if `--seed` passed | `random.seed()` | YES |
| cuDNN deterministic | RL: configurable (`torch_deterministic`), BC: NO | `torch.backends.cudnn.deterministic` | Default `False` = non-deterministic cuDNN kernels |
| Robosuite env `self.rng` | **NO** | `robosuite.make()` called WITHOUT `seed=` parameter | `self.rng = np.random.default_rng(None)` = random every time |
| Robosuite object placement (cube pos) | **Partially** via `np.random.seed()` | `UniformRandomSampler` uses global `np.random.uniform()` | YES for RL (since `np.random.seed()` is called), NO for BC (unless `--seed` given) |
| Robosuite robot joint noise | **Partially** via `np.random.seed()` | `robot.reset()` uses global `np.random.randn()` | Same as above — but magnitude=0.0 by default (no noise applied) |
| `DexMimicGenEnv.seed()` | **NO-OP** | Method exists but does nothing: `return [seed]` | env.seed() calls in RL training silently skipped (wrapper has no `.seed()`) |
| `DexMimicGenEnv.reset(seed=)` | **IGNORED** | Accepts `seed` kwarg but doesn't use it | Always ignored |
| DataLoader shuffle (BC) | Indirectly via `torch.manual_seed()` | PyTorch's default `RandomSampler` uses global torch RNG | Deterministic only if `--seed` was passed |
| Replay buffer sampling (RL) | Indirectly via `torch.manual_seed()` | TorchRL's `TensorDictPrioritizedReplayBuffer` uses torch RNG | Deterministic if seed was set |
| Eval envs (RL) | **Same as training envs** — no separate seeding | `env.seed(cfg.seed+1)` attempted but silently skipped (no `.seed()` method) | NOT separately seeded from training env |
| Eval envs (BC) | **Not seeded at all** | `create_vectorized_env()` has no seed parameter | Fully random |
| AsyncVectorEnv workers | **NOT seeded** | `context="spawn"` creates fresh processes with no seed propagation | Each worker gets random state from OS |
| `PYTHONHASHSEED` | **NOT set** anywhere | — | Python dict ordering not deterministic |

**Bottom line: seeding is INCOMPLETE. The `--seed` / `seed=` config controls PyTorch weight init and global numpy/random, but robosuite's internal `self.rng` (used for some things) is NEVER seeded. Object placement happens to use global `np.random` so it IS controlled by `np.random.seed()`, and robot joint noise is disabled by default (magnitude=0.0). But eval env seeding is effectively a no-op.**


### Detailed Breakdown

---

#### 1. BC Training (`resfit/lerobot/scripts/train_bc_dexmg.py`)

**Seed source:** CLI argument `--seed`, default=`None` (optional)

**Seeding code** ([train_bc_dexmg.py](resfit/lerobot/scripts/train_bc_dexmg.py#L432-L434)):
```python
if cfg.seed is not None:
    set_seed(cfg.seed)
    logger.info(colored(f"Random seed set to {cfg.seed}", "yellow"))
```

**`set_seed()` from LeRobot** ([deps/lerobot/lerobot/common/utils/random_utils.py](deps/lerobot/lerobot/common/utils/random_utils.py#L168-L173)):
```python
def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**What's seeded in BC training:**
- `random.seed()` — Python stdlib ✓
- `np.random.seed()` — NumPy global ✓
- `torch.manual_seed()` — PyTorch CPU ✓
- `torch.cuda.manual_seed_all()` — PyTorch GPU ✓

**What's NOT seeded in BC training:**
- `torch.backends.cudnn.deterministic` — never set (default `False`)
- `PYTHONHASHSEED` — never set
- Eval environments — `create_vectorized_env()` has no seed parameter
- DataLoader workers — no `worker_init_fn` or `generator` passed

**BC DataLoader** ([train_bc_dexmg.py](resfit/lerobot/scripts/train_bc_dexmg.py#L567-L575)):
```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,       # ← uses torch's global RNG (seeded if --seed given)
    num_workers=cfg.num_workers,
    pin_memory=device.type != "cpu",
    drop_last=True,
    persistent_workers=cfg.num_workers > 0,
)
```

`shuffle=True` creates a `RandomSampler` that uses the global PyTorch RNG. If `set_seed()` was called, the shuffling order is deterministic. BUT: `num_workers > 0` spawns worker processes that load data — their internal RNG state is NOT explicitly seeded (no `worker_init_fn`). In practice, PyTorch's default `worker_init_fn` seeds each worker with `base_seed + worker_id`, so it's reasonably deterministic.

**BC evaluation envs:** Created via `create_vectorized_env()` with no seed — **fully random every eval**.

---

#### 2. RL Training (`resfit/rl_finetuning/scripts/train_residual_td3.py`)

**Seed source:** Hydra config `seed: int | None = None`, auto-generated if not provided

**Seeding code** ([train_residual_td3.py](resfit/rl_finetuning/scripts/train_residual_td3.py#L308-L325)):
```python
# Seeding (must be done before environment creation)
if cfg.seed is None:
    cfg.seed = random.randint(0, 2**32 - 1)   # ← auto-generate

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)

torch.backends.cudnn.deterministic = cfg.torch_deterministic  # default: False

print(f"Set random seed to {cfg.seed}")
```

**Env seeding attempt** ([train_residual_td3.py](resfit/rl_finetuning/scripts/train_residual_td3.py#L356-L360)):
```python
# Seed environments explicitly for reproducibility
if hasattr(env, "seed"):
    env.seed(cfg.seed)
if hasattr(eval_env, "seed"):
    eval_env.seed(cfg.seed + 1)
```

**Why this is a NO-OP:** `env` is a `BasePolicyVecEnvWrapper` ([residual_env_wrapper.py](resfit/rl_finetuning/wrappers/residual_env_wrapper.py#L23)) which does NOT have a `.seed()` method. `hasattr(env, "seed")` returns `False`. The `__getattr__` fallback would delegate to `.vec_env` but `hasattr` catches the `AttributeError` and treats it as `False`. **Both calls silently do nothing.**

Even if it reached `DexMimicGenEnv.seed()`, that's also a no-op:
```python
def seed(self, seed=None):
    # For robosuite environments, we don't have direct seeding control
    # This is a no-op for compatibility
    return [seed]
```

**What actually controls randomness in robosuite envs:** The global `np.random` state (seeded by `np.random.seed(cfg.seed)` above). See Section 4 below.

---

#### 3. Environment Creation — No Seed Propagation

**`create_vectorized_env()`** ([dexmg.py](resfit/dexmg/environments/dexmg.py#L667)):
```python
def create_vectorized_env(
    env_name, num_envs, device, camera_size, render_size, debug, video_key, headless,
) -> VectorizedEnvWrapper:
    # ← NO seed parameter
    
    env_fns = []
    for env_id in range(num_envs):
        env_fns.append(make_dexmimicgen_env(env_name, camera_size, ...))
    
    if debug:
        vec_env = gym.vector.SyncVectorEnv(env_fns, ...)  # same process
    else:
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns,
            context="spawn",  # ← fresh processes, NO seed propagation
            ...
        )
```

**`robosuite.make()` call** ([dexmg.py](resfit/dexmg/environments/dexmg.py#L214)):
```python
self.env = robosuite.make(**env_kwargs)
# env_kwargs does NOT include 'seed' or 'initialization_noise'
```

**Consequences:**
- `robosuite.make()` gets `seed=None` → `self.rng = np.random.default_rng(None)` → random every time
- `initialization_noise` not passed → defaults to `{"magnitude": 0.0, "type": "gaussian"}` → **no joint noise at all**
- `AsyncVectorEnv` with `context="spawn"` creates new processes; `np.random.seed()` from the parent does NOT propagate to child processes

---

#### 4. What Robosuite Actually Randomizes and How

Robosuite has TWO random number sources:

**Source A: `self.rng` (instance-level `np.random.Generator`)** — [base.py](deps/robosuite/robosuite/environments/base.py#L142):
```python
self.seed = seed
self.rng = np.random.default_rng(seed)  # seed=None → unseeded
```
This is passed to some internal components but in this version of robosuite, the key randomization functions actually use the GLOBAL numpy RNG instead.

**Source B: Global `np.random`** — What `UniformRandomSampler` and `robot.reset()` actually use:

Object placement ([placement_samplers.py](deps/robosuite/robosuite/utils/placement_samplers.py#L167-L198)):
```python
# UniformRandomSampler.sample() uses global np.random:
return np.random.uniform(high=maximum, low=minimum)   # line 167
return np.random.uniform(high=maximum, low=minimum)   # line 183
rot_angle = np.random.uniform(high=2 * np.pi, low=0)  # line 196
```

Robot joint noise ([robot.py](deps/robosuite/robosuite/robots/robot.py#L248-L250)):
```python
if self.initialization_noise["type"] == "gaussian":
    noise = np.random.randn(len(self.init_qpos)) * self.initialization_noise["magnitude"]
elif self.initialization_noise["type"] == "uniform":
    noise = np.random.uniform(-1.0, 1.0, len(self.init_qpos)) * self.initialization_noise["magnitude"]
```

**KEY INSIGHT: Since both object placement and joint noise use GLOBAL `np.random`, and RL training calls `np.random.seed(cfg.seed)`, these ARE seeded for RL training (in the main process). However:**
- **BC training:** only seeded if `--seed` explicitly passed
- **AsyncVectorEnv workers:** spawned processes get their OWN `np.random` state (NOT the parent's seeded state), so eval environments running in sub-processes are NOT deterministically seeded
- **SyncVectorEnv (debug mode):** runs in the main process, so shares the parent's `np.random` state — IS seeded
- **Joint noise magnitude=0.0 by default**, so the randomness there doesn't matter in practice

---

#### 5. The Multi-Eval-Env Problem (Your Question)

**Setup (RL):** `eval_num_envs: int = 4` and `eval_num_episodes: int = 50`

**What happens:**
1. 4 eval environments are created via `AsyncVectorEnv(context="spawn")`
2. Each spawned worker process gets its own unseeded `np.random` state
3. During eval, all 4 envs `.reset()` at the start
4. As episodes finish, individual envs auto-reset (via `AutoresetMode.SAME_STEP`)
5. The eval loop counts finished episodes until 50 are done

**Are all 4 envs seeing the same scenarios?** NO — because:
- Each AsyncVectorEnv worker is a separate process with its OWN random state
- The 4 workers are NOT seeded to the same value
- They get different random numpy states from the OS at spawn time
- So cube positions, object placements, etc. differ across the 4 workers

**Your worry about "5 envs with same seed = only 10 distinct settings in 50 episodes":** This does NOT happen in the current code because the envs are NOT seeded identically (or at all, in async mode). The 4 workers each produce different random scenarios.

**However:** Even in `SyncVectorEnv` (debug mode), the envs share the SAME `np.random` global state (since they run in the main process). This means env 0's reset consumes some `np.random` draws, then env 1's reset consumes the next draws, etc. They won't see identical scenarios because the RNG advances sequentially across the envs. But the sequence IS deterministic if `np.random.seed()` was called.

**BC eval** (`eval_num_envs: int = 5` by default): Same story — `AsyncVectorEnv` workers are not seeded.

---

#### 6. Replay Buffer Sampling

**Online buffer** ([train_residual_td3.py](resfit/rl_finetuning/scripts/train_residual_td3.py#L417-L428)):
```python
online_rb = TensorDictPrioritizedReplayBuffer(
    storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
    alpha=alpha, beta=beta, eps=1e-6,
    priority_key="_priority",
    transform=MultiStepTransform(n_steps=cfg.algo.n_step, gamma=cfg.algo.gamma),
    pin_memory=True,
    prefetch=cfg.algo.prefetch_batches,
    batch_size=online_batch_size,
)
```

**No explicit seed passed.** TorchRL's replay buffer sampling uses PyTorch's global RNG internally, which IS seeded by `torch.manual_seed(cfg.seed)` in RL training. So batch sampling order is deterministic given the same seed and buffer contents.

---

#### 7. Dead Code: `set_all_seeds()` with Offsets

[helper.py](resfit/rl_finetuning/off_policy/common_utils/helper.py#L212-L217):
```python
def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed + 1)      # offset +1
    torch.manual_seed(rand_seed + 2)   # offset +2
    torch.cuda.manual_seed_all(rand_seed + 3)  # offset +3
```

**This is NEVER CALLED.** Defined but has zero callers in the codebase. The actual training scripts use their own inline seeding (same seed for all RNGs, no offset).

---

#### 8. Seed in Run Names

RL training embeds the seed in the W&B run name ([train_residual_td3.py](resfit/rl_finetuning/scripts/train_residual_td3.py#L833)):
```
resfit__<timestamp>__<Task>_n<nstep>_utd<utd>_buf<buf>_off<off>ep_lr<lr>__seed<seed>
```

Example from workspace:
```
run_2026-02-20_09-40-38_resfit__...__TwoArmCoffee_n5_utd4_buf200000_off250ep_lr1e-06__seed1200418472
```

---

### Summary Diagram

```
cfg.seed = 42 (or auto-generated random int)
      │
      ├──→ random.seed(42)           ← Python stdlib
      ├──→ np.random.seed(42)        ← global numpy RNG
      │         │
      │         ├──→ SyncVectorEnv (debug=True): envs share this RNG
      │         │    └── robosuite UniformRandomSampler uses np.random.uniform()
      │         │    └── robot.reset() uses np.random.randn() (but magnitude=0.0 → no noise)
      │         │
      │         └──→ AsyncVectorEnv (debug=False): workers DON'T inherit this RNG
      │              └── each worker gets random OS-provided numpy state
      │
      ├──→ torch.manual_seed(42)     ← PyTorch CPU
      │         └── DataLoader shuffle, replay buffer sampling, network init
      │
      ├──→ torch.cuda.manual_seed_all(42)  ← PyTorch CUDA
      │
      └──→ torch.backends.cudnn.deterministic = False (default)

    env.seed(42)        ← SILENTLY SKIPPED (wrapper has no .seed() method)
    eval_env.seed(43)   ← SILENTLY SKIPPED (same reason)
    
    robosuite.make(...)  ← NO seed= parameter passed
         └── self.rng = np.random.default_rng(None) = random
```

### Gaps / Things to Fix If You Want Full Determinism

1. **Pass `seed=` to `robosuite.make()`** — would seed `self.rng` in the base env (though currently most randomization uses global `np.random` anyway)
2. **Seed AsyncVectorEnv workers** — either via `worker_init_fn`-style pattern or by passing seeds through env factory functions
3. **Set `PYTHONHASHSEED`** — for fully deterministic Python dict iteration
4. **Set `torch.backends.cudnn.deterministic = True`** — or pass `torch_deterministic=True` in RL config
5. **Seed BC training by default** — currently BC only seeds if `--seed` is explicitly passed; consider auto-generating like RL does
6. **Fix the `env.seed()` / `eval_env.seed()` calls** — either add `.seed()` to the wrapper chain or remove the dead code
7. **Add `initialization_noise` to `robosuite.make()` call** — currently robot joints have zero noise, which might be intentional but should be documented
