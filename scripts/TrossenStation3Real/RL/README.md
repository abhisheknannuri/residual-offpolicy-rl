# Station 3 - residual RL run scripts

Four modes, one shared config. The distributed (agentlace) variants in
`agentlace/` are the same four, reusing the same config file.

```
_common_rl.sh    every shared setting; hashed values are `readonly`
_run_rl.sh       assembles and runs the command (shared, incl. by agentlace/)
01_populate_buffers.sh               robot   offline buffer + online warm-up, then stop
02_offline_rl.sh                     no robot  critic warm-up + offline TD3-BC
03_online_rl_from_criticwarmup.sh    robot   critic warm-up then online RL
04_online_rl_from_checkpoint.sh      robot   resume from 02's checkpoint, then online RL
agentlace/                           the same four, actor/learner split
```

## Why the config is shared, and `readonly`

Both replay-buffer caches are addressed by a hash of their config
(`docs/real/BUFFER_CACHES.md`). Change one hashed value in one script and that
script quietly builds its **own** buffer instead of reusing the one you stood at
the robot to collect.

This is not hypothetical: Station-1's scripts use `algo.learning_starts=10000`
in four files and `15000` in the fifth, so they do **not** share an online
buffer.

So every hashed value lives in `_common_rl.sh` and is declared `readonly`. A mode
script that tries to change one gets

```
_common_rl.sh: line N: LEARNING_STARTS: readonly variable
```

instead of a silently different cache. Non-hashed values (critic warm-up length,
total timesteps, resume checkpoint) stay overridable per mode.

## Settings chosen for Station 3

| | | |
| --- | --- | --- |
| base policy | ACT **50k** | 4/20 full success, and the only checkpoint that reached S2 on 20/20 poses |
| `algo.learning_starts` | **15000** | ~30 episodes, ~12 min of robot time at 20 Hz |
| `real_max_steps` | **350** | matches Station-1's RL scripts |
| offline dataset | `PickCubeAndInsert_Station3_Trial1_deltajoint_old` | **must** be the `_old` (v2.1) copy - the vendored lerobot is `CODEBASE_VERSION = "v2.1"` and cannot read the v3.0 sibling |
| `offline_data.num_episodes` | **178** | the whole dataset (55,177 frames) |

Everything else is copied from Station 1, including
`action_scale = [0.15,0.15,0.15,0.05,0.05,0.05,0.1]`.

## Running

Start these first, in their own terminals:

```sh
# laptop - follower
python -m trossen_real.follower.follower_single_server \
    --config trossen_station3_single --port 5060

# GPU box - frozen base policy
uv run python custom_scripts/policy_server.py \
    --checkpoint .../ACT_BC_PickCubeAndInsert_Station3_Trial1_deltajoint/checkpoints/050000/pretrained_model/ \
    --n-action-steps 15
```

Then:

```sh
./01_populate_buffers.sh                                   # at the robot
./02_offline_rl.sh                                         # anywhere, no hardware
./03_online_rl_from_criticwarmup.sh                        # at the robot
RESUME_CKPT=/abs/.../offline_step_75000/checkpoint.pt \
    ./04_online_rl_from_checkpoint.sh                      # at the robot
```

`DRY_RUN=1 ./<script>` prints the exact command without running anything.

### Distributed

Same four scripts, with the role as the first argument:

```sh
# GPU server
./agentlace/03_online_rl_from_criticwarmup.sh learner

# laptop at the robot (or keep localhost and SSH-forward 5588/5589)
LEARNER_IP=10.0.0.5 ./agentlace/03_online_rl_from_criticwarmup.sh actor

# no networking, to check parity against the non-distributed script
./agentlace/03_online_rl_from_criticwarmup.sh single
```

They source `../_common_rl.sh` - the **same** file - so the hashed config cannot
drift between the plain and distributed paths. Verified: both produce byte-
identical hashed arguments.

## After an online-RL session

Everything collected during online RL is written to
`run_<...>/online_buffer_final/`, **not** back to the hashed cache, so the next
run would start from the warm-up buffer again. To carry it forward:

```sh
.venv/bin/python scripts/promote_online_buffer.py \
    --from run_<...> --to <online hash> --apply
```

Note the buffer is circular at `algo.buffer_size=70000`: past that, the oldest
transitions - which are your original warm-up data - are evicted. See
`docs/real/BUFFER_CACHES.md`.

## Known rough edges

* `01` needs the robot connected even for the **offline** buffer. Not a real
  requirement: `_populate_offline_buffer()` only queries `policy_server.py`, but
  it reaches the policy client via `env.policy`, and `get_envs()` connects the
  follower, cameras and leader first (`train_residual_td3.py:476-494`).
  Incidental coupling; left alone rather than changing the trainer.
* `01` stops via `total_timesteps=0`, which still runs **one** online step before
  exiting. Ctrl+C after `Warm-up done. Online buffer size = N` is equally safe -
  the buffer is dumped to disk immediately before that line prints.
* `no_cleanup` must stay `true`. With it false the trainer runs
  `shutil.rmtree(run_cache_dir)` on success, deleting the checkpoints and
  `online_buffer_final`.
