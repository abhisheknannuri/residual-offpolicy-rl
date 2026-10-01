#!/usr/bin/env bash
# =============================================================================
# 03 - CRITIC WARM-UP, THEN ONLINE RL.  Robot required.
# =============================================================================
# The "ignore offline RL entirely" path: warm the critic from the two cached
# buffers in this same process, then go straight into online RL. No checkpoint
# from 02 is loaded.
#
# The online buffer is loaded from its hashed cache, so the warm-up collection
# from 01 is already in it and is NOT re-collected (the warm-up block is skipped
# whenever the cache already holds >= LEARNING_STARTS transitions).
#
# AFTERWARDS: everything this run collects lands in
# run_<...>/online_buffer_final/ and is NOT written back to the hashed cache.
# To carry it into the next session:
#   .venv/bin/python scripts/promote_online_buffer.py \
#       --from run_<...> --to <online hash> --apply
# See docs/real/BUFFER_CACHES.md.
# =============================================================================
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/_common_rl.sh"

MODE_NAME="03 online RL after critic warm-up"
MODE_DESC="critic warm-up from the cached buffers, then online RL at the robot"

CRITIC_WARMUP=15000
DO_OFFLINE_RL="false"
OFFLINE_RL_STEPS=0
TOTAL_TIMESTEPS=75000
RESUME_CKPT=""

source "${_HERE}/_run_rl.sh"
