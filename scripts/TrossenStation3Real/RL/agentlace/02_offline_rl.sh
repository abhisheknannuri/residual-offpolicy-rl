#!/usr/bin/env bash
# =============================================================================
# [distributed] CRITIC WARM-UP + OFFLINE RL - no hardware
# =============================================================================
# Runs entirely off the two caches that 01 wrote. Connects nothing.
# Requires the online cache to already hold >= algo.learning_starts transitions.
#
# Config: conf/dist_02_offline_rl.yaml  ->  conf/station3.yaml + conf/modes/dist_02_offline_rl.yaml
# Everything is in those files. Anything after this script is passed straight to
# hydra, so a one-off tweak needs no edit:
#     ./02_offline_rl.sh algo.total_timesteps=1000
#
# Role: pass it like any other hydra override -
#     ./02_offline_rl.sh role=learner
#     ./02_offline_rl.sh role=actor dist.ip=10.0.0.5
#
# PYTHON_BIN picks the interpreter (default: `python`). Set it, or activate the
# venv - bare `python` is the conda env on this laptop until the environment
# rework lands.
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# Say which interpreter this is using. The laptop has two working envs (the
# uv .venv and a conda one) that differ on av/cv2 versions, and picking the
# wrong one raises no error - it just runs somewhere else. NOTE the variable
# is PYTHON_BIN; PYTHON=... is silently ignored.
echo "[$(basename "${BASH_SOURCE[0]}")] python: $("${PYTHON_BIN:-python}" -c 'import sys;print(sys.executable)')" >&2

exec "${PYTHON_BIN:-python}" resfit/rl_finetuning/scripts/train_residual_td3_distributed.py \
    --config-dir scripts/TrossenStation3Real/RL/conf \
    --config-name dist_02_offline_rl \
    "$@"
