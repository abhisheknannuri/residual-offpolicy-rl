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

## Gripper filter (2026-11-05)

**Why.** In the run of 2026-11-05 the residual's gripper output flipped between
its limits from tick to tick (commanded 1 <-> 21 mm, up to 34 mm measured in one
tick). One full-stroke open overshot the 40 mm target to 44.015 mm, past the
controller's 44 mm limit, and the arm went idle. Clipping the target cannot
prevent that: every command is "reach this width in `goal_time_s` = 0.125 s"
(linear interpolation below 0.2 s), so a bigger jump is a faster move.

**What.** `trossen_real/inference/gripper_filter.py` `GripperFilter`, applied in
`TrossenResidualEnv.step()` to the final gripper width (ACT + residual, or the
leader while intervening), before it is sent, mirrored to the leader or stored.
The buffer therefore stores what was executed. Station config
`control.gripper_filter`:

| | | |
|---|---|---|
| `max_step` | 0.006 m | the command changes by at most this per tick, any source. Just above the fastest gripper motion in the demos (5.9 mm/tick) and the thin ACT's live runs (5.2) |
| `confirm_ticks` | 3 | the policy's request must lie on the same side of the command this many ticks in a row before it moves (as GripperController). 1 = speed limit only |

Effects: flipping of any size never moves the gripper and does not drift it; a
steady change starts 2 ticks (0.1 s) late, then follows at <= 6 mm/tick; a full
4 -> 40 mm stroke takes 6 ticks (0.3 s); after an intervention the policy must
agree for 3 ticks before the gripper moves away from where the human left it.
The filter starts each episode at the width the reset commanded.

The action scale is independent of it: the scale sets how far the residual can
move the gripper from ACT's width (0.3: +-19.5 mm, enough to release the cube
from ACT's 4 mm hold), the filter how fast and how steadily it gets there.

**Proof.**
`.venv/bin/python -m pytest trossen_real/inference/tests/test_gripper_filter.py -v`
(no robot): speed limit, clipping, flipping of 8.6-18 mm at every phase and
pattern, no ratchet after a motion, no drift under exploration noise, steady
requests and ramps, release while the residual flips, NaN, hand-back, 20 random
fuzz runs (<= 6 mm/tick and inside the range always), and the env on fake
hardware (reset start width, flipping residual, ACT release, hand-back + leader
mirror, stored == sent). Replayed on the requests of the 14:17 session of
2026-11-05: max jump 31.5 -> 6.0 mm, jumps over 6 mm 307 -> 0, reversals over
1 mm 329 -> 2, episodes opening past 30 mm 14 -> 12.

**Reading the log.** The env's `real_log_file` now has `gripper_requested` (the
width before the filter) next to `executed_action[6]` (what was sent).

**Opening cap (`open_gap`, added after a second fault at 16:22).** With the speed
limit in place the arm still faulted (44.009 mm): the command went 22 -> 28 -> 34
-> 40 mm in three ticks while the fingers lagged at ~28 mm, so the motor was told
to cover ~12 mm in one 0.125 s move and arrived too fast. The joints and the
gripper go out in ONE `set_all_positions` call with one goal_time; the lag is
the gripper motor's own response, not our code. Now, while opening, the command
may be at most `control.gripper_filter.open_gap` (9 mm) wider than the measured
fingers (`follower_state_before`). Closing is never capped: a held cube blocks
the fingers at its width and that gap is the grip. 9 mm: the demos reach the top
with the command up to 9.6 mm ahead and never faulted, but a 20 Hz log cannot
see the overshoot spike, so the value is NOT proven safe. Measure it on the
robot (follower_single_server.py stopped; only the gripper moves, 10-30 mm):

```bash
.venv/bin/python temp/gripper/gripper_overshoot_test.py --list     # methods, no robot
.venv/bin/python temp/gripper/gripper_overshoot_test.py            # all, 3 repeats, ~3 min
```

It compares one-tick jumps, 6 mm/tick, 6 mm/tick with open_gap 3/6/9 mm, and
the gripper sent alone (`set_gripper_position`, quintic goal_time 0.25/0.5 s),
sampling the gripper at the SDK's rate. Read the table's opening overshoot: a
method is safe at 40 mm if it stays well below 4 mm (the margin to 44 mm). Set
`open_gap` from it.

First run (`real_logs/overshoot_20261105_164535`, 10 <-> 30 mm, no cube): no
method overshot (peak 29.95 mm, low 10.05 mm), a one-tick 20 mm jump included,
so "fast arrival overshoots" is NOT confirmed. The script's default run now
covers what the faults had and that run did not: opening to 34 mm (`top34_*`),
and opening right after squeezing the cube at 1 mm (`squeeze_*`, you hold the
cube; 16:22 opened after ~1 s squeezing at 12.1 mm). `--all` adds the plain
cases.

Second run (`overshoot_20261105_165514`): no overshoot at 34 mm or after a
squeeze either. **Cause found in the fault logs:** both faults (14:30, 16:22)
came ~0.25 s after ACT's open to 40 mm with the fingers at 23-28 mm, and the
follower got NO command after it: SARM had seen the cube inserted and the env
was waiting for the confirmation pedal. Normally the next step replaces each
0.125 s move after 50 ms; with no next step the robot finishes that fast move
alone and overshoots. The gripper filter cannot act then (it runs per step).
**Fix:** `TrossenResidualEnv._hold_still()` sends one hold command before the
wait (joints at their measured positions, gripper at the narrower of the last
command and the measured width: an opening stops, a grip on the cube is kept).
Test: `test_env_sarm_success.py::test_confirmation_wait_starts_with_a_hold_command`.
Confirmed on the robot (run 2026-11-05 17:09, `once`: one 10 -> 30 mm command
in 0.125 s, nothing after it): the fingers stopped at 37.95 mm, 8 mm past the
target, and stayed there; the next command (1.5 s back to 30 mm) faulted the arm
at 44.075 mm. With commands every 50 ms (the first two runs) the same moves end
within 0.05 mm. `once` is removed from the script; `oncehold` (the fix) is
available with `--only`, not in the default run. Any other gap in the 20 Hz
command stream right after a large gripper move is the same risk. Logs and plots: `temp/gripper/real_logs/overshoot_<time>/`.
Tests: `test_gripper_filter.py` covers the cap (63 tests).
