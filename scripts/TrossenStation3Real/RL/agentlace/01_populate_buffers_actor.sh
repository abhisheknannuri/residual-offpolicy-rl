#!/usr/bin/env bash
# =============================================================================
# [distributed | ACTOR] POPULATE BOTH BUFFERS - run this on the ROBOT machine
# =============================================================================
# The actor half of 01_populate_buffers.sh. Run that one on the GPU server (it
# defaults to role=learner) and this one on the laptop wired to the arm.
#
# The split is NOT symmetric:
#   learner  builds the OFFLINE buffer itself from the dataset, querying
#            policy_server.py over plain HTTP (_RemotePolicy,
#            train_residual_td3_distributed.py:267-287). No robot, no env.
#   actor    this script. The only side that calls get_envs() (:2339), so the
#            only side that touches follower/cameras/leader/pedal. It collects
#            the ONLINE warm-up and streams the transitions to the learner.
#
# Needs on THIS machine: the robot up, and policy_server.py reachable at
# policy_server_url (station3.yaml defaults to http://127.0.0.1:5070).
#
# Finding the learner - dist.ip is a plain hydra override:
#     ./01_populate_buffers_actor.sh dist.ip=10.0.0.5
# Over an SSH tunnel the learner looks local, so the default works. From the
# laptop, forwarding both RL ports plus policy_server BACK to the server:
#     ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 \
#            -R 5070:localhost:5070 user@server
#
# Config: conf/dist_01_populate.yaml -> conf/station3.yaml + conf/modes/01_populate.yaml
#         + conf/execution/distributed.yaml, with role=actor forced below.
# Anything after this script goes straight to hydra, and a later override wins,
# so even `role=` can still be changed from the command line if you need to.
#
# PYTHON_BIN picks the interpreter (default: `python`). Set it, or activate the
# venv - bare `python` is the conda env on this laptop until the environment
# rework lands.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# `-m` so sys.path[0] is the repo root (we just cd'd there), not the
# script's own directory. `trossen_real` is not part of the editable
# install - pyproject.toml's packages.find includes only "resfit*" - so
# running the file by path dies on `ModuleNotFoundError: trossen_real`.
exec "${PYTHON_BIN:-python}" -m resfit.rl_finetuning.scripts.train_residual_td3_distributed \
    --config-dir scripts/TrossenStation3Real/RL/conf \
    --config-name dist_01_populate \
    role=actor \
    "$@"
