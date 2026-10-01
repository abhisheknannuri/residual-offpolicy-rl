# Station 3 - distributed (agentlace)

The same four modes as the parent folder, split into an actor (laptop, robot)
and a learner (GPU server).

```sh
./03_online_rl_from_criticwarmup.sh role=learner
./03_online_rl_from_criticwarmup.sh role=actor dist.ip=10.0.0.5
./03_online_rl_from_criticwarmup.sh role=single     # no networking, parity check
```

Over an SSH tunnel keep `dist.ip=localhost` and forward both ports:

```sh
ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 user@server
```

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
