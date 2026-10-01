#!/usr/bin/env bash
# =============================================================================
# [distributed] POPULATE BOTH BUFFERS - robot required
# =============================================================================
# Builds the offline buffer from the dataset, collects the online warm-up, then
# stops. Nothing is trained.
#
# The robot must be up even for the OFFLINE buffer: _populate_offline_buffer()
# only queries policy_server.py, but it reaches the policy client via env.policy,
# and get_envs() connects follower/cameras/leader first (train_residual_td3_distributed.py
# :476-494). Incidental coupling, left alone rather than changing the trainer.
#
# Config: conf/dist_01_populate.yaml  ->  conf/station3.yaml + conf/modes/dist_01_populate.yaml
# Everything is in those files. Anything after this script is passed straight to
# hydra, so a one-off tweak needs no edit:
#     ./01_populate_buffers.sh algo.total_timesteps=1000
#
# Role: pass it like any other hydra override -
#     ./01_populate_buffers.sh role=learner
#     ./01_populate_buffers.sh role=actor dist.ip=10.0.0.5
#
# PYTHON_BIN picks the interpreter (default: `python`). Set it, or activate the
# venv - bare `python` is the conda env on this laptop until the environment
# rework lands.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# PYTHONPATH: `python path/to/script.py` puts the SCRIPT's directory on
# sys.path, not the repo root - and `trossen_real` is NOT part of the
# editable install (pyproject.toml packages.find includes only "resfit*"),
# so without this the run dies on `ModuleNotFoundError: trossen_real`.
exec env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python}" resfit/rl_finetuning/scripts/train_residual_td3_distributed.py \
    --config-dir scripts/TrossenStation3Real/RL/conf \
    --config-name dist_01_populate \
    "$@"
