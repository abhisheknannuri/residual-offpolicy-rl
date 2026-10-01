#!/usr/bin/env bash
# =============================================================================
# [distributed] CRITIC WARM-UP THEN ONLINE RL - robot required
# =============================================================================
# Warms the critic from the cached buffers in this same process, then goes
# straight into online RL. No checkpoint is loaded.
#
# Config: conf/dist_03_online_from_criticwarmup.yaml  ->  conf/station3.yaml + conf/modes/dist_03_online_from_criticwarmup.yaml
# Everything is in those files. Anything after this script is passed straight to
# hydra, so a one-off tweak needs no edit:
#     ./03_online_rl_from_criticwarmup.sh algo.total_timesteps=1000
#
# Role: pass it like any other hydra override -
#     ./03_online_rl_from_criticwarmup.sh role=learner
#     ./03_online_rl_from_criticwarmup.sh role=actor dist.ip=10.0.0.5
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
    --config-name dist_03_online_from_criticwarmup \
    "$@"
