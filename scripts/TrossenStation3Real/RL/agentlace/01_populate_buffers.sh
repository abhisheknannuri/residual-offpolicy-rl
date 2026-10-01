#!/usr/bin/env bash
# =============================================================================
# 01 - POPULATE BOTH BUFFERS.  Robot required.
# =============================================================================
# Builds the offline buffer from the dataset, then collects the online warm-up
# at the robot, then stops. Nothing is trained.
#
# BEFORE RUNNING, on this laptop:
#   python -m trossen_real.follower.follower_single_server \
#       --config trossen_station3_single --port 5060
# and on the GPU box (base policy - 50k, chosen from the eval campaign):
#   uv run python custom_scripts/policy_server.py \
#       --checkpoint .../ACT_BC_PickCubeAndInsert_Station3_Trial1_deltajoint/checkpoints/050000/pretrained_model/ \
#       --n-action-steps 15
#
# WHY THE ROBOT IS NEEDED FOR THE *OFFLINE* BUFFER TOO
# ----------------------------------------------------
# It is not needed in principle. `_populate_offline_buffer()` reads the dataset
# and queries policy_server.py for each base action - no robot involved. But it
# reaches the policy client via `env.policy`, and `get_envs()` builds `env` by
# connecting the follower, cameras and leader FIRST (train_residual_td3.py
# :476-494). So the coupling is incidental, not fundamental. We are not changing
# the code, so: connect the robot.
#
# WHAT GETS WRITTEN
#   offline_buffer_cache/<hash>/   once, reused by every later script
#   online_buffer_cache/<hash>/    every ONLINE_WARMUP_SAVE_FREQ transitions AND
#                                  at the end - so a dropped connection mid-
#                                  collection is resumable: just re-run this.
#
# HOW IT STOPS
#   critic_warmup_steps=0 and total_timesteps=0 mean the online loop runs a
#   single step and exits cleanly (the loop is `while global_step <= 0`). You can
#   also just Ctrl+C once you see "Warm-up done. Online buffer size = N" - the
#   buffer was dumped to disk immediately before that line is printed.
#
# DISTRIBUTED: pass the role as the first argument -
#     ./01_populate_buffers.sh learner      # GPU server
#     LEARNER_IP=<server> ./01_populate_buffers.sh actor    # laptop at the robot
#     ./01_populate_buffers.sh single       # no networking, parity check
# =============================================================================
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/../_common_rl.sh"   # the SAME shared config as the non-distributed scripts
source "${_HERE}/_common_dist.sh"    # role + transport only

MODE_NAME="[dist] 01 populate buffers"
MODE_DESC="offline buffer from dataset + online warm-up at the robot, then stop"

CRITIC_WARMUP=0
DO_OFFLINE_RL="false"
OFFLINE_RL_STEPS=0
TOTAL_TIMESTEPS=0
RESUME_CKPT=""

source "${_HERE}/../_run_rl.sh"
