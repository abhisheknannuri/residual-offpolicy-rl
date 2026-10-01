#!/usr/bin/env bash
# =============================================================================
# 04 - ONLINE RL RESUMING FROM A CHECKPOINT.  Robot required.
# =============================================================================
# Starts online RL from weights 02 produced, with critic_warmup_steps=0 because
# the critic is already warm - that is what the checkpoint carries.
#
# Point RESUME_CKPT at either:
#   .../models/offline_step_<N>/checkpoint.pt    continue from OFFLINE RL
#   .../models/critic_warmup_final/checkpoint.pt continue from the CRITIC WARM-UP
# Both are the same code path; only the file differs.
#
# Override without editing this file:
#   RESUME_CKPT=/abs/path/to/checkpoint.pt ./04_online_rl_from_checkpoint.sh
#
# Same buffer note as 03: promote the run's online buffer afterwards if you want
# the next session to start from it.
#
# DISTRIBUTED: pass the role as the first argument -
#     ./04_online_rl_from_checkpoint.sh learner      # GPU server
#     LEARNER_IP=<server> ./04_online_rl_from_checkpoint.sh actor    # laptop at the robot
#     ./04_online_rl_from_checkpoint.sh single       # no networking, parity check
# =============================================================================
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/../_common_rl.sh"   # the SAME shared config as the non-distributed scripts
source "${_HERE}/_common_dist.sh"    # role + transport only

MODE_NAME="[dist] 04 online RL from checkpoint"
MODE_DESC="resume from an offline-RL (or critic-warm-up) checkpoint, then online RL"

CRITIC_WARMUP=0
DO_OFFLINE_RL="false"
OFFLINE_RL_STEPS=0
TOTAL_TIMESTEPS=75000
RESUME_CKPT="${RESUME_CKPT:-}"

if [[ -z "${RESUME_CKPT}" ]]; then
    echo "ERROR: RESUME_CKPT is empty." >&2
    echo "       Set it at the top of this script, or pass it inline:" >&2
    echo "         RESUME_CKPT=/abs/path/to/checkpoint.pt $0" >&2
    echo "       Look under run_<timestamp>_<name>/models/ from script 02:" >&2
    echo "         offline_step_<N>/checkpoint.pt    or    critic_warmup_final/checkpoint.pt" >&2
    exit 1
fi
if [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "ERROR: RESUME_CKPT does not exist: ${RESUME_CKPT}" >&2
    exit 1
fi

source "${_HERE}/../_run_rl.sh"
