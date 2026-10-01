#!/usr/bin/env bash
# =============================================================================
# POPULATE BOTH BUFFERS - robot required
# =============================================================================
# Builds the offline buffer from the dataset, collects the online warm-up, then
# stops. Nothing is trained.
#
# The robot must be up even for the OFFLINE buffer: _populate_offline_buffer()
# only queries policy_server.py, but it reaches the policy client via env.policy,
# and get_envs() connects follower/cameras/leader first (train_residual_td3.py
# :476-494). Incidental coupling, left alone rather than changing the trainer.
#
# Config: conf/01_populate.yaml  ->  conf/station3.yaml + conf/modes/01_populate.yaml
# Everything is in those files. Anything after this script is passed straight to
# hydra, so a one-off tweak needs no edit:
#     ./01_populate_buffers.sh algo.total_timesteps=1000
#
# PYTHON_BIN picks the interpreter (default: `python`). Set it, or activate the
# venv - bare `python` is the conda env on this laptop until the environment
# rework lands.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${PYTHON_BIN:-python}" resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-dir scripts/TrossenStation3Real/RL/conf \
    --config-name 01_populate \
    "$@"
