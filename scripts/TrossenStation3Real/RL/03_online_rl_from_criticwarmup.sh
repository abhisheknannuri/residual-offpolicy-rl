#!/usr/bin/env bash
# =============================================================================
# CRITIC WARM-UP THEN ONLINE RL - robot required
# =============================================================================
# Warms the critic from the cached buffers in this same process, then goes
# straight into online RL. No checkpoint is loaded.
#
# Config: conf/03_online_from_criticwarmup.yaml  ->  conf/station3.yaml + conf/modes/03_online_from_criticwarmup.yaml
# Everything is in those files. Anything after this script is passed straight to
# hydra, so a one-off tweak needs no edit:
#     ./03_online_rl_from_criticwarmup.sh algo.total_timesteps=1000
#
# PYTHON_BIN picks the interpreter (default: `python`). Set it, or activate the
# venv - bare `python` is the conda env on this laptop until the environment
# rework lands.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# `-m` so sys.path[0] is the repo root (we just cd'd there), not the
# script's own directory. `trossen_real` is not part of the editable
# install - pyproject.toml's packages.find includes only "resfit*" - so
# running the file by path dies on `ModuleNotFoundError: trossen_real`.
exec "${PYTHON_BIN:-python}" -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-dir scripts/TrossenStation3Real/RL/conf \
    --config-name 03_online_from_criticwarmup \
    "$@"
