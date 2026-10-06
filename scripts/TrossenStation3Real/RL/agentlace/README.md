# Station 3 - distributed RL (agentlace)

Four modes, each split into an **actor** (laptop, robot attached) and a
**learner** (GPU server).

- [Copy-paste: how to run each mode](#copy-paste-how-to-run-each-mode)
- [Before anything: the three prerequisites](#before-anything-the-three-prerequisites)
- [Which script goes on which machine](#which-script-goes-on-which-machine)
- [Why the two configs may differ, and what is checked](#why-the-two-configs-may-differ-and-what-is-checked)
- [The offline buffer cache](#the-offline-buffer-cache)
- [Config keys](#config-keys)
- [Troubleshooting](#troubleshooting)

---

## Copy-paste: how to run each mode

Every command is run **from the repo root**. `PYTHON_BIN` is not optional - see
[the note below](#python_bin-is-not-optional).

### 00. Both machines, every time

```sh
# laptop: tunnel to the server. Three forwards, not two.
ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 -R 5070:localhost:5070 user@server

# laptop: the base BC policy server
python custom_scripts/policy_server.py --port 5070 ...     # its own venv

# laptop: the follower server, for any mode with an actor
.venv/bin/python -m trossen_real.follower.follower_single_server \
    --config trossen_station3_single --port 5060
```

### 01. Populate buffers

```sh
# SERVER  (start this first)
PYTHON_BIN=.venv/bin/python \
  ./scripts/TrossenStation3Real/RL/agentlace/01_populate_buffers.sh

# LAPTOP
PYTHON_BIN=.venv/bin/python \
  ./scripts/TrossenStation3Real/RL/agentlace/01_populate_buffers_actor.sh
```

Add `offline_data.root=/abs/path/to/dataset` **on the server** if the dataset is
not where `station3.yaml` points.

### 02. Critic warm-up + offline RL — **server only, no actor**

```sh
PYTHON_BIN=.venv/bin/python \
  ./scripts/TrossenStation3Real/RL/agentlace/02_offline_rl.sh
```

It sets `algo.offline_pretrain_only: true`, which makes the trainer skip
`get_envs()` entirely. Running it as an actor raises *"offline_pretrain_only=True
touches no environment - run it with role=learner only"*
(`train_residual_td3_distributed.py:2390`).

Produces the checkpoints the next two modes resume from, under
`run_<timestamp>__<name>/models/`:

```text
models/critic_warmup_final/checkpoint.pt
models/offline_step_5000/checkpoint.pt   ... every offline_save_freq (5000)
```

### 03. Critic warm-up, then online RL

```sh
# SERVER
PYTHON_BIN=.venv/bin/python \
  ./scripts/TrossenStation3Real/RL/agentlace/03_online_rl_from_criticwarmup.sh

# LAPTOP
PYTHON_BIN=.venv/bin/python \
  ./scripts/TrossenStation3Real/RL/agentlace/03_online_rl_from_criticwarmup_actor.sh
```

Does its own 15 000-step critic warm-up first, then goes online. Takes no
checkpoint.

### 04. Online RL resuming from a checkpoint

**This is the one that takes `RESUME_CKPT`, and only on the learner.**

```sh
# SERVER  (start this first)
RESUME_CKPT=/abs/path/to/run_<ts>__<name>/models/offline_step_90000/checkpoint.pt \
PYTHON_BIN=.venv/bin/python \
  ./scripts/TrossenStation3Real/RL/agentlace/04_online_rl_from_checkpoint.sh

# LAPTOP  -- NO RESUME_CKPT. Deliberately.
PYTHON_BIN=.venv/bin/python \
  ./scripts/TrossenStation3Real/RL/agentlace/04_online_rl_from_checkpoint_actor.sh
```

Either kind of checkpoint is valid:

```text
.../models/offline_step_<N>/checkpoint.pt       continue from OFFLINE RL
.../models/critic_warmup_final/checkpoint.pt    continue from CRITIC WARM-UP
```

**Why the actor takes no `RESUME_CKPT`:** `resume_ckpt` is read only inside
`_learner_body` (`train_residual_td3_distributed.py:761`, the load at `:1583`). The actor never
loads a checkpoint - it receives weights over PUB/SUB. Setting it on the actor
would silently do nothing.

Mode 04 sets `critic_warmup_steps: 0`, `train_offline_rl: false`,
`offline_rl_steps: 0`, `total_timesteps: 200000` - the checkpoint already
carries a warm critic.

### Overrides

Anything after the script goes straight to hydra:

```sh
./04_online_rl_from_checkpoint.sh algo.total_timesteps=50000
./04_online_rl_from_checkpoint_actor.sh dist.ip=10.0.0.5
```

> **An override that changes training must be passed to BOTH machines.**
> `algo.*` is part of the fingerprint, so `algo.total_timesteps=50000` on the
> server alone makes the actor refuse to start. See
> [the exclude list](#why-the-two-configs-may-differ-and-what-is-checked) for
> the keys that may legitimately differ.

---

## Before anything: the three prerequisites

### `PYTHON_BIN` is not optional

The scripts default to bare `python`, which on this laptop is the **conda env**
(`~/miniforge3/envs/residual/bin/python`), not the repo venv. Set it on every
invocation, or activate the venv first:

```sh
PYTHON_BIN=.venv/bin/python ./scripts/.../01_populate_buffers.sh
```

### The tunnel needs three forwards, not two

The learner binds (`TrainerServer`); the actor dials out on both ports
(`TrainerClient`). The **third** forward is the reverse one, so the learner can
reach `policy_server.py`, which runs on the laptop:

```sh
ssh -N \
  -L 5588:localhost:5588 \
  -L 5589:localhost:5589 \
  -R 5070:localhost:5070 \
  user@server
```

With it, `policy_server_url: http://127.0.0.1:5070` works unchanged on both
machines and `dist.ip=localhost` works on the actor. Without a tunnel, pass the
server's address instead: `./..._actor.sh dist.ip=10.0.0.5`.

### Start the learner first

It has work to do before it wants the actor (mode 01: one policy query per
dataset frame). The actor prints `connecting to learner ... (waits until
reachable)` and blocks, so overlapping them is harmless.

---

## Which script goes on which machine

| mode | learner (GPU server) | actor (laptop + robot) |
| --- | --- | --- |
| 01 populate buffers | `01_populate_buffers.sh` | `01_populate_buffers_actor.sh` |
| 02 critic warm-up + offline RL | `02_offline_rl.sh` | *none* |
| 03 critic warm-up then online RL | `03_online_rl_from_criticwarmup.sh` | `03_online_rl_from_criticwarmup_actor.sh` |
| 04 online RL from a checkpoint | `04_online_rl_from_checkpoint.sh` | `04_online_rl_from_checkpoint_actor.sh` |

The unsuffixed scripts default to `role=learner`
(`conf/execution/distributed.yaml`), so they need no override. The `_actor.sh`
ones force `role=actor` and are otherwise identical - same config, same hashed
values. Overrides still pass through and a later one wins, so
`./01_populate_buffers_actor.sh role=single` works for the single-process
parity check.

### The split is not symmetric

| | learner | actor |
| --- | --- | --- |
| builds the OFFLINE buffer | yes, from the dataset | no |
| needs the dataset | yes | no |
| needs `policy_server.py` | yes, for the offline buffer | yes, for every rollout step |
| needs the robot / cameras / leader / pedal | **no** | **yes** |
| gradient updates, checkpoints, `resume_ckpt` | yes | no |
| `wandb` logging | yes | no |
| `enable_intervention` | ignored | honoured |

The learner builds the offline buffer itself and queries `policy_server.py` over
plain HTTP (`_LazyRemotePolicyEnv`, `:264`) - no env is constructed. `get_envs()`
is called only inside `_actor_body` (`:2407`, the call at `:2464`), which is why a GPU box with no
hardware attached can run the learner half of mode 01.

---

## Why the two configs may differ, and what is checked

The actor and learner **do** run with different configs - `role` differs by
definition, and in mode 04 only the learner has `resume_ckpt`. That is fine,
because there is an explicit allow-list.

At the handshake the learner sends its config fingerprint; the actor computes
its own and `_verify_fingerprint` (`:194`) **refuses to run on a mismatch**,
printing a per-key `actor=... learner=...` diff. The reason it is strict is in
the code comment:

> a silently mismatched `action_scale` / `n_step` / `image_size` /
> normalization knob raises no error anywhere - it just trains a critic in one
> action space and acts in another.

### Keys allowed to differ (`_FINGERPRINT_EXCLUDE`, `:160`)

| key | why |
| --- | --- |
| `role` | the whole point |
| `dist` | ports/codec are enforced by agentlace's own `TrainerConfig` hash |
| `seed` | may be randomly drawn per process |
| `actor_name`, `eval_num_envs` | mutated inside `main()` |
| `real_log_file` | per-machine path |
| **`resume_ckpt`** | **only the learner loads checkpoints** |
| `policy_server_url` | may legitimately differ (tunnels) |
| `wandb` | only the learner logs |
| `no_cleanup`, `headless`, `save_video` | per-machine display/cleanup choices |

**Everything else must match exactly** - including all of `algo.*`,
`offline_data.*`, `reward_shaping.*`, `base_policy.*` and the image/action
settings.

---

## The offline buffer cache

A **separate** hash from the fingerprint, and `resume_ckpt` is not in it - so
pointing mode 04 at a different checkpoint cannot invalidate your cache.

Built from an explicit metadata dict (`:1329`) and hashed to an 8-character
directory under `offline_buffer_cache/`:

```text
task, wandb_name, dataset_name, num_episodes,
use_base_policy_for_base_actions, min_action_range, min_state_std,
reward_shaping, reward_parquet, reward_column,
offline_root, remap_preset, image_keys,
n_step, gamma, schema, base_policy_wandb_id,
sampling_strategy, normalized_actions
```

Change any of those and the hash changes, the cache misses, and the buffer is
**rebuilt from the dataset** - which needs the dataset *and* a reachable
`policy_server.py` on the server. That is why the `-R 5070` forward should stay
up even for modes 03 and 04.

Modes 03 and 04 keep `offline_fraction: 0.5`, so they still load the offline
buffer; they just do not train on it offline.

---

## Config keys

These use `conf/dist_NN_*.yaml`, which compose the **distributed schema** plus
the identical `conf/station3.yaml` and `conf/modes/*.yaml` the plain scripts
use, plus `conf/execution/distributed.yaml` for role and transport. Nothing that
affects training or the buffer hashes is duplicated.

| key | default | |
| --- | --- | --- |
| `role` | `learner` | `single` \| `actor` \| `learner` |
| `dist.ip` | `localhost` | where the actor finds the learner |
| `dist.port` | `5588` | REQ/REP, actor -> learner |
| `dist.broadcast_port` | `5589` | PUB/SUB, learner -> actor (weights) |
| `dist.steps_per_update` | `50` | weight publish cadence, in gradient steps |
| `dist.target_utd` | `null` | learner free-runs; `4` matches the single-process UTD |

### What each mode sets

| | 01 | 02 | 03 | 04 |
| --- | --- | --- | --- | --- |
| `critic_warmup_steps` | 0 | 20000 | 15000 | **0** |
| `train_offline_rl` | false | true | false | false |
| `offline_rl_steps` | 0 | 75000 | 0 | 0 |
| `total_timesteps` | 0 | — | 75000 | 200000 |
| `offline_pretrain_only` | false | **true** | false | false |
| `resume_ckpt` | null | null | null | **required** |

`agentlace` is not pip-installed - it is imported from the clone at the repo
root, restored by `bash scripts/clone_deps.sh`. See `DEPENDENCIES.md`.

---

## Troubleshooting

**`Actor and learner configs differ - refusing to run`**
You passed a hydra override to one machine only. The error prints the exact
keys; pass the same override to both. If the differing key is in
[the exclude list](#why-the-two-configs-may-differ-and-what-is-checked), that is
a bug - report it.

> That error ends with *"Run both roles from the SAME script in
> `scripts/TrossenStation1Real/agentlace/`"*. **Ignore the path** - the trainer
> is shared by both stations and the message names Station 1. For Station 3 the
> scripts are in `scripts/TrossenStation3Real/RL/agentlace/` (note the extra
> `RL/` level). The advice itself is right: use the same mode's pair.

**`set RESUME_CKPT to a checkpoint.pt`**
Mode 04's learner script requires it. Use an absolute path to a
`.../models/<something>/checkpoint.pt`.

**The offline buffer starts rebuilding when you expected a cache hit**
A hashed value changed. The trainer `pprint`s the whole `offline_cache_meta`
dict just before hashing - compare it against the run that built the cache.

**`offline_pretrain_only=True requires the offline buffer to already exist`**
Mode 02 cannot populate the cache itself (it never builds an env). Run mode 01
first.

**`ModuleNotFoundError: trossen_real`**
Run the scripts as given - they `cd` to the repo root and use `-m`.
`trossen_real` is not part of the editable install (`pyproject.toml`'s
`packages.find` includes only `resfit*`), so invoking the file by path fails.

**The actor sits at `connecting to learner ...`**
Expected until the learner binds. If it never connects: check the tunnel, and
that `dist.ip` points at the learner.
