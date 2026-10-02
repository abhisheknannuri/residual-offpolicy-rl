#!/usr/bin/env bash
# =============================================================================
# [distributed | ACTOR] CRITIC WARM-UP THEN ONLINE RL - run on the ROBOT machine
# =============================================================================
# The actor half of 03_online_rl_from_criticwarmup.sh. Run that one on the GPU
# server (it defaults to role=learner) and this one on the laptop.
#
#   learner  warms the critic from the two cached buffers, then runs the
#            gradient loop and publishes weights every dist.steps_per_update
#            steps. No robot, no env.
#   actor    this script. Rolls out on the real arm with the latest published
#            weights and streams transitions back. Honours enable_intervention,
#            so the leader arm and foot pedal work here, not on the learner.
#
# Needs on THIS machine: the robot up, and policy_server.py reachable at
# policy_server_url (station3.yaml defaults to http://127.0.0.1:5070).
#
# Finding the learner - dist.ip is a plain hydra override:
#     ./03_online_rl_from_criticwarmup_actor.sh dist.ip=10.0.0.5
# Over an SSH tunnel the learner looks local, so the default works. From the
# laptop, forwarding both RL ports plus policy_server BACK to the server:
#     ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 \
#            -R 5070:localhost:5070 user@server
#
# Config: conf/dist_03_online_from_criticwarmup.yaml -> conf/station3.yaml +
#         conf/modes/03_online_from_criticwarmup.yaml +
#         conf/execution/distributed.yaml, with role=actor forced below.
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
    --config-name dist_03_online_from_criticwarmup \
    role=actor \
    "$@"
