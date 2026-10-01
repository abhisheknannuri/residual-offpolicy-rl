# Station 3 - residual RL

Config lives in YAML. The `.sh` files are three-line wrappers that pick a config
and hand everything else to Hydra.

```
conf/station3.yaml        every configurable parameter, in one file, organised
                          like the dataclasses. HASHED values marked.
conf/modes/*.yaml         what each mode changes - values only, no schema
conf/execution/*.yaml     role + transport, distributed only
conf/NN_*.yaml            plain entry: schema + station3 + mode
conf/dist_NN_*.yaml       distributed entry: dist schema + the SAME station3 + the SAME mode
check_configs.sh          composes all 8 and asserts the HASHED values match
```

| script | hardware | what it does |
| --- | --- | --- |
| `01_populate_buffers.sh` | robot | offline buffer from the dataset, then the online warm-up, then stop |
| `02_offline_rl.sh` | none | critic warm-up + offline TD3-BC from the caches |
| `03_online_rl_from_criticwarmup.sh` | robot | critic warm-up in-process, then online RL |
| `04_online_rl_from_checkpoint.sh` | robot | resume from 02's checkpoint, then online RL |

`agentlace/` holds the same four for the actor/learner split.

## Where the values came from

`station3.yaml` lists every parameter the schema defines, so the whole surface is
visible. Values come from `scripts/TrossenStation1Real/*.sh` where those scripts
set one — all five agree on every value, so there is a single Station-1 answer —
and from the dataclass default otherwise. Most of the untouched ones (the ViT
shape, the reward-model block, critic widths) are defaults nobody has tuned, so
treat them as inherited rather than chosen.

`check_station1_parity.py` re-derives the Station-1 values straight from the
shell scripts and diffs them against this file, so a value cannot quietly drift
or be mistyped.

Station-3's own decisions are the dataset, station name, W&B project, base ACT
50k, `algo.learning_starts` 15000 and `real_max_steps` 350.

Inline comments are condensed from `train_residual_rl_real.sh`, which keeps the
long-form reasoning.

## The HASHED values are yours to keep stable

Both replay-buffer caches are addressed by a hash of their config. **Change one
marked `# HASHED` and every script gets a different, empty buffer** - you
re-collect at the robot and rebuild the offline buffer. There is no guard rail;
`station3.yaml` says which values those are and `docs/real/BUFFER_CACHES.md`
says which hash each one lands in.

`check_configs.sh` will tell you if the modes have drifted apart.

## Running

```sh
# laptop - follower
python -m trossen_real.follower.follower_single_server --config trossen_station3_single --port 5060

# GPU box - frozen base policy (ACT 50k)
uv run python custom_scripts/policy_server.py \
    --checkpoint .../ACT_BC_PickCubeAndInsert_Station3_Trial1_deltajoint/checkpoints/050000/pretrained_model/ \
    --n-action-steps 15
```

```sh
./01_populate_buffers.sh
./02_offline_rl.sh
./03_online_rl_from_criticwarmup.sh
RESUME_CKPT=/abs/.../models/offline_step_75000/checkpoint.pt ./04_online_rl_from_checkpoint.sh
```

Anything after the script goes straight to Hydra, so a one-off tweak needs no
edit:

```sh
./03_online_rl_from_criticwarmup.sh algo.total_timesteps=1000 wandb.mode=disabled
./02_offline_rl.sh --cfg job            # print the resolved config and exit
```

### Testing without a robot

Everything except the leader arm already stands in for itself: the cameras fall
back to a mock feed when no RealSense is found (`camera_manager.py:_make_camera`)
and the pedal falls back to dummy mode with no VEC device. Only the follower
needs a stand-in, and the leader has no fallback by design - `get_envs()` raises
rather than degrade silently.

So two things: run the mock follower, and turn intervention off.

```sh
python -m trossen_real.follower.follower_single_server_mock \
    --config trossen_station3_single --port 5060

./01_populate_buffers.sh enable_intervention=false
```

With a policy server up, that builds the offline buffer end to end at a desk.
Verified: 417 transitions from one episode, 32 `/predict_chunk` calls (≈13 frames
per round trip at the server's `n_action_steps=14`), 3 `/reset` calls.

Only the OFFLINE buffer is meaningful this way. The online warm-up will run, but
nothing moves, so the transitions it collects are not worth keeping.

### Distributed

```sh
./agentlace/03_online_rl_from_criticwarmup.sh role=learner
./agentlace/03_online_rl_from_criticwarmup.sh role=actor dist.ip=10.0.0.5
./agentlace/03_online_rl_from_criticwarmup.sh role=single    # parity check
```

They reuse `conf/station3.yaml` and `conf/modes/*.yaml` unchanged - only the
schema and the `execution` overlay differ - so the plain and distributed paths
cannot drift. Verified by `check_configs.sh`.

## How a config is assembled

`--config-dir conf` puts that directory on Hydra's search path;
`--config-name 01_populate` makes `conf/01_populate.yaml` the primary config.
Its `defaults:` list is the recipe, applied **top to bottom, later wins**:

```yaml
defaults:
  - residual_td3_trossen_real_config   # 1. the SCHEMA - a dataclass, not a file
  - station3                           # 2. conf/station3.yaml
  - modes/01_populate                  # 3. conf/modes/01_populate.yaml
  - _self_                             # 4. this file's own keys, last so they win
```

```
dataclass defaults (177 params)
  <- station3.yaml            the shared values
  <- modes/01_populate.yaml   what this mode changes
  <- command-line overrides   win over everything
```

Order is load-bearing: `station3.yaml` sets `algo.total_timesteps: 75000` and
`modes/01_populate.yaml` sets `0`; the composed config is `0`, because the mode
comes later.

Three things that are not obvious:

* **`residual_td3_trossen_real_config` is not a file.** It is the dataclass
  registered by `cs.store(name=...)` in `resfit/rl_finetuning/config/residual_td3.py`.
  That is where the 177 parameters and their types come from, and why a
  misspelled key is rejected at compose time rather than at runtime.
* **A subdirectory is a config GROUP.** `modes/01_populate` means "group `modes`,
  option `01_populate`". Nothing registers the folder - Hydra finds it because it
  is under the search path, and the directory name is the group name. Groups can
  be swapped from the command line, which is how `execution=single` works.
* **`# @package _global_` is required in every group file.** Without it a group's
  contents nest under the group name, and you get
  `Key 'modes' not in 'ResidualTD3TrossenRealConfig'`. It says "merge these keys
  at the root instead".

Only the `dist_*` configs list `execution: distributed`; the plain ones have no
`execution` entry, which is why `execution=` cannot be overridden on them.
`conf/execution/single.yaml` is not used by any config by default - it exists so
`execution=single` works as a swap on a distributed config.

## Overriding

Each `.sh` picks one config and passes everything after it straight to Hydra.

```sh
./01_populate_buffers.sh                       # conf/01_populate.yaml as-is
./01_populate_buffers.sh algo.learning_starts=5000 wandb.mode=disabled
./02_offline_rl.sh --cfg job --resolve         # print the resolved config, run nothing
```

Any key in `station3.yaml` can be overridden this way, including a HASHED one -
which will send you to a different buffer, so check `--cfg job` first if you are
not sure.

### Picking a different config entirely

The `.sh` files are a convenience. To run a config they do not name, call the
entrypoint yourself:

```sh
python resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-dir scripts/TrossenStation3Real/RL/conf \
    --config-name 03_online_from_criticwarmup
```

### Distributed: `role` and the `execution` group

The distributed configs include `execution: distributed`, which sets
`role: learner` plus the transport defaults. Two ways to change it:

```sh
./agentlace/03_online_rl_from_criticwarmup.sh role=actor dist.ip=10.0.0.5
./agentlace/03_online_rl_from_criticwarmup.sh execution=single    # swaps the whole group
```

`role=...` sets one key; `execution=single` swaps the whole overlay (role plus
its transport block). Both verified.

**`execution=` only works on the distributed configs.** On a plain one it fails
with `Could not override 'execution'. No match in the defaults list.` - correct,
because `role`/`dist` exist only in the distributed schema
(`ResidualTD3TrossenRealDistConfig`), and the plain entrypoint would reject them.

## Where the buffers live: single vs distributed are NOT interchangeable

In **single** mode one process owns both replay buffers, so the caches are
written on **that machine**.

In **distributed** mode they are the learner's. The actor never holds a replay
buffer at all - its `online_rb` is an `_NStepSink` that assembles n-step
transitions and pushes them over the wire
(`train_residual_td3_distributed.py:2428`), while every
`online_buffer_cache/<hash>/` read and write happens inside `_learner_body`.

So the offline and online caches land on **whichever machine runs the learner**.
Populating on the laptop with `./01_populate_buffers.sh` does not give the GPU
server a buffer, and vice versa - the hash is the same, the disk is not. Decide
where you want the data to live before spending robot time on it.

## Settings

| | | |
| --- | --- | --- |
| base policy | ACT **50k** | the only checkpoint that reached S2 on 20/20 poses |
| `algo.learning_starts` | **15000** | ~30 episodes, ~12 min of robot time |
| `real_max_steps` | **350** | matches Station-1's RL scripts |
| dataset | `..._deltajoint_old` | **must** be the `_old` v2.1 copy - the vendored lerobot is `CODEBASE_VERSION = "v2.1"` |
| `offline_data.num_episodes` | **178** | the whole dataset |

Everything else is Station-1's values.

## Notes

* **Interpreter is yours.** `PYTHON_BIN` defaults to `python`; set it or activate
  the venv. On this laptop bare `python` is still the conda env, until the
  environment rework lands.
* `offline_data.root` uses `${oc.env:RESFIT_ROOT,<abs path>}`, **not**
  `${hydra:runtime.cwd}`. The latter resolves to the launch directory - verified:
  running from `/tmp` produced `root: /tmp/trossen_real/...`, a different hash and
  a silently rebuilt buffer. Set `RESFIT_ROOT` on another machine.
* `01` stops via `total_timesteps: 0`, which still runs one online step. Ctrl+C
  after `Warm-up done. Online buffer size = N` is equally safe - the buffer is
  dumped immediately before that line prints.
* `no_cleanup: true` must stay. With it false the trainer `rmtree`s the run
  directory on success, deleting the checkpoints and `online_buffer_final`.
* After an online-RL session, carry the buffer forward with
  `scripts/promote_online_buffer.py` - see `docs/real/BUFFER_CACHES.md`.
