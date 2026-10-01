#!/usr/bin/env bash
# =============================================================================
# 02 - CRITIC WARM-UP + OFFLINE RL.  No robot, no policy server, no cameras.
# =============================================================================
# Runs entirely off the two caches 01 wrote. algo.offline_pretrain_only=true
# makes train_residual_td3.py skip get_envs() outright - nothing is connected -
# and exit after the offline phases, before the online loop.
#
# REQUIRES both caches to already exist, with the online one holding at least
# LEARNING_STARTS transitions. If the hash does not match you get a loud
# RuntimeError naming every value to check - that is this folder's shared
# _common_rl.sh doing its job, so do not override hashed values here.
#
# Produces checkpoints under run_<timestamp>_<name>/models/:
#   critic_warmup_final/checkpoint.pt   -> feed to 03
#   offline_step_<N>/checkpoint.pt      -> feed to 04
#
# DISTRIBUTED: pass the role as the first argument -
#     ./02_offline_rl.sh learner      # GPU server
#     LEARNER_IP=<server> ./02_offline_rl.sh actor    # laptop at the robot
#     ./02_offline_rl.sh single       # no networking, parity check
# =============================================================================
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/../_common_rl.sh"   # the SAME shared config as the non-distributed scripts
source "${_HERE}/_common_dist.sh"    # role + transport only

MODE_NAME="[dist] 02 offline RL (no robot)"
MODE_DESC="critic warm-up then offline TD3-BC, purely from the cached buffers"

OFFLINE_PRETRAIN_ONLY="true"
CRITIC_WARMUP=20000
DO_OFFLINE_RL="true"
OFFLINE_RL_STEPS=75000
RESUME_CKPT=""

# Nothing on the robot is touched, so do not ask for a leader connection.
ENABLE_INTERVENTION="false"

source "${_HERE}/../_run_rl.sh"
