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
set one, and from the dataclass default otherwise — most of the untouched ones
(the ViT shape, the reward-model block, critic widths) are defaults nobody has
tuned, so treat them as inherited rather than chosen.

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

### Distributed

```sh
./agentlace/03_online_rl_from_criticwarmup.sh role=learner
./agentlace/03_online_rl_from_criticwarmup.sh role=actor dist.ip=10.0.0.5
./agentlace/03_online_rl_from_criticwarmup.sh role=single    # parity check
```

They reuse `conf/station3.yaml` and `conf/modes/*.yaml` unchanged - only the
schema and the `execution` overlay differ - so the plain and distributed paths
cannot drift. Verified by `check_configs.sh`.

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
