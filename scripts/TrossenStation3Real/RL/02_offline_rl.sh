#!/usr/bin/env bash
# =============================================================================
# CRITIC WARM-UP + OFFLINE RL - no hardware
# =============================================================================
# Runs entirely off the two caches that 01 wrote. Connects nothing.
# Requires the online cache to already hold >= algo.learning_starts transitions.
#
# Config: conf/02_offline_rl.yaml  ->  conf/station3.yaml + conf/modes/02_offline_rl.yaml
# Everything is in those files. Anything after this script is passed straight to
# hydra, so a one-off tweak needs no edit:
#     ./02_offline_rl.sh algo.total_timesteps=1000
#
# PYTHON_BIN picks the interpreter (default: `python`). Set it, or activate the
# venv - bare `python` is the conda env on this laptop until the environment
# rework lands.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# PYTHONPATH: `python path/to/script.py` puts the SCRIPT's directory on
# sys.path, not the repo root - and `trossen_real` is NOT part of the
# editable install (pyproject.toml packages.find includes only "resfit*"),
# so without this the run dies on `ModuleNotFoundError: trossen_real`.
exec env PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python}" resfit/rl_finetuning/scripts/train_residual_td3.py \
    --config-dir scripts/TrossenStation3Real/RL/conf \
    --config-name 02_offline_rl \
    "$@"
