# Station 3 - distributed (agentlace)

The same four modes as the parent folder, split into an actor (laptop, robot)
and a learner (GPU server).

## Which script goes on which machine

| mode | learner (GPU server) | actor (laptop + robot) |
| --- | --- | --- |
| 01 populate buffers | `01_populate_buffers.sh` | `01_populate_buffers_actor.sh` |
| 02 critic warm-up + offline RL | `02_offline_rl.sh` | *none - see below* |
| 03 critic warm-up then online RL | `03_online_rl_from_criticwarmup.sh` | `03_online_rl_from_criticwarmup_actor.sh` |
| 04 online RL from a checkpoint | `04_online_rl_from_checkpoint.sh` | `04_online_rl_from_checkpoint_actor.sh` |

The unsuffixed scripts default to `role=learner` (`conf/execution/distributed.yaml`),
so they need no override. The `_actor.sh` ones force `role=actor` and are
otherwise identical - same config, same hashed values. Overrides still pass
through, and a later one wins, so `./01_populate_buffers_actor.sh role=single`
works if you want the single-process parity check.

**Mode 02 has no actor.** It sets `algo.offline_pretrain_only: true`, which
makes the trainer skip `get_envs()` entirely; running it as an actor raises
`"offline_pretrain_only=True touches no environment - run it with role=learner
only"` (`train_residual_td3_distributed.py:2277`). Run it on the server alone.

## The split is not symmetric

| | learner | actor |
| --- | --- | --- |
| builds the OFFLINE buffer | yes, from the dataset | no |
| needs the dataset | yes | no |
| needs `policy_server.py` | yes, for the offline buffer | yes, for every rollout step |
| needs the robot / cameras / leader / pedal | **no** | **yes** |
| gradient updates, checkpoints, `resume_ckpt` | yes | no |
| `enable_intervention` | ignored | honoured |

The learner builds the offline buffer itself and queries `policy_server.py` over
plain HTTP (`_RemotePolicy`, `:267-287`) - no env is constructed. `get_envs()` is
called only inside `_actor_body` (`:2339`), which is why a GPU box with no
hardware attached can run the learner half of mode 01.

`resume_ckpt` is read only in `_learner_body` (`:1477`), so the mode 04 actor
script deliberately takes no `RESUME_CKPT` - the actor receives weights over
PUB/SUB.

## Networking

The learner binds (`TrainerServer`); the actor dials out on both ports
(`TrainerClient`). Three forwards are needed, not two - the learner also has to
reach `policy_server.py`, which usually runs on the laptop. From the **laptop**:

```sh
ssh -N \
  -L 5588:localhost:5588 \
  -L 5589:localhost:5589 \
  -R 5070:localhost:5070 \
  user@server
```

The `-R` is what lets the learner query the policy server. With it,
`policy_server_url: http://127.0.0.1:5070` works unchanged on both machines and
`dist.ip=localhost` works on the actor. Without a tunnel, pass the server's
address instead: `./..._actor.sh dist.ip=10.0.0.5`.

## Running mode 01 end to end

Tunnel up first, then `policy_server.py` on the laptop, then:

```sh
# server
PYTHON_BIN=.venv/bin/python ./scripts/TrossenStation3Real/RL/agentlace/01_populate_buffers.sh

# laptop
PYTHON_BIN=.venv/bin/python ./scripts/TrossenStation3Real/RL/agentlace/01_populate_buffers_actor.sh
```

Start the learner first - it has one policy query per dataset frame to get
through before it wants the actor. The actor prints
`connecting to learner ... (waits until reachable)` and blocks, so overlap is
harmless. On the server add `offline_data.root=/abs/path/to/dataset` if the
dataset is not where `station3.yaml` points.

## Config keys

These use `conf/dist_NN_*.yaml`, which compose the **distributed schema** plus
the identical `conf/station3.yaml` and `conf/modes/*.yaml` the plain scripts use,
plus `conf/execution/distributed.yaml` for role and transport. Nothing that
affects training or the buffer hashes is duplicated.

| key | default | |
| --- | --- | --- |
| `role` | `learner` | `single` \| `actor` \| `learner` |
| `dist.ip` | `localhost` | where the actor finds the learner |
| `dist.port` | `5588` | REQ/REP, actor -> learner |
| `dist.broadcast_port` | `5589` | PUB/SUB, learner -> actor (weights) |
| `dist.steps_per_update` | `50` | weight publish cadence, in gradient steps |
| `dist.target_utd` | `null` | learner free-runs; `4` matches the single-process UTD |

`agentlace` is not pip-installed - it is imported from the clone at the repo
root, restored by `bash scripts/clone_deps.sh`. See `DEPENDENCIES.md`.
