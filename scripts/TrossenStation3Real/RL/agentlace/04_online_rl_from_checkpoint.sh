#!/usr/bin/env bash
# =============================================================================
# [distributed] ONLINE RL FROM A CHECKPOINT - robot required
# =============================================================================
# Resumes from a checkpoint 02 produced; critic_warmup_steps is 0 because the
# checkpoint already carries a warm critic.
#
#   RESUME_CKPT=/abs/.../models/offline_step_75000/checkpoint.pt ./04_...sh
#   RESUME_CKPT=/abs/.../models/critic_warmup_final/checkpoint.pt ./04_...sh
#
# Config: conf/dist_04_online_from_checkpoint.yaml  ->  conf/station3.yaml + conf/modes/dist_04_online_from_checkpoint.yaml
# Everything is in those files. Anything after this script is passed straight to
# hydra, so a one-off tweak needs no edit:
#     ./04_online_rl_from_checkpoint.sh algo.total_timesteps=1000
#
# Role: pass it like any other hydra override -
#     ./04_online_rl_from_checkpoint.sh role=learner
#     ./04_online_rl_from_checkpoint.sh role=actor dist.ip=10.0.0.5
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
    --config-name dist_04_online_from_checkpoint \
    resume_ckpt="${RESUME_CKPT:?set RESUME_CKPT to a checkpoint.pt - see run_<ts>_<name>/models/ from 02}" \
    "$@"
