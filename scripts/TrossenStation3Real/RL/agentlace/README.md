# Station 3 - distributed (agentlace) variants

The same four modes as the parent folder, split into an actor (laptop, robot)
and a learner (GPU server) over agentlace's ZMQ transport.

These scripts **source `../_common_rl.sh`** - the identical file the
non-distributed scripts use - and add only role and transport settings via
`_common_dist.sh`. Nothing that affects training or the buffer hashes is
duplicated here, so the two paths cannot drift apart.

```sh
./03_online_rl_from_criticwarmup.sh learner             # GPU server
LEARNER_IP=10.0.0.5 ./03_online_rl_from_criticwarmup.sh actor   # laptop
./03_online_rl_from_criticwarmup.sh single              # parity check, no networking
```

Over an SSH tunnel, keep `LEARNER_IP=localhost` and forward both ports:

```sh
ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 user@server
```

| env var | default | |
| --- | --- | --- |
| `LEARNER_IP` | `localhost` | where the actor finds the learner |
| `LEARNER_PORT` | `5588` | REQ/REP, actor -> learner (transitions, stats) |
| `LEARNER_BROADCAST_PORT` | `5589` | PUB/SUB, learner -> actor (weights) |
| `STEPS_PER_UPDATE` | `50` | weight publish cadence, in gradient steps |
| `TARGET_UTD` | `null` | learner free-runs; set `4` to match the single-process UTD |

`agentlace` itself is not pip-installed - it is imported from the clone at the
repo root. `bash scripts/clone_deps.sh` restores it. See `DEPENDENCIES.md`.

Design and background: `scripts/TrossenStation1Real/RESFIT_DISTRIBUTED_MIGRATION.md`.
