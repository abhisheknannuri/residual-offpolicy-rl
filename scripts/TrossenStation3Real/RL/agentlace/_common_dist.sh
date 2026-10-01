#!/usr/bin/env bash
# =============================================================================
# _common_dist.sh - role + transport for the DISTRIBUTED Station-3 scripts.
#
# Sourced by every script in this folder, AFTER ../_common_rl.sh. Never run
# directly.
#
# Everything that affects training or the buffer hashes comes from
# ../_common_rl.sh - this file adds only who-talks-to-whom. That is deliberate:
# the actor and the learner must agree on every hyper-parameter, and a silent
# mismatch (different action_scale, n_step, normalization...) raises no error,
# it just has the actor acting in one action scale while the critic was trained
# in another.
# =============================================================================

# ── Role ─────────────────────────────────────────────────────────────────────
# First positional arg, else $ROLE, else "single".
#   single  - one process does everything, no networking. Use it to check parity
#             against the non-distributed scripts before trusting the split.
#   actor   - laptop: robot, cameras, pedal/leader, policy inference, n-step
#   learner - GPU server: buffers, critic/actor updates, checkpoints, W&B
if [[ -n "${1:-}" && "${1}" != -* ]]; then
    ROLE="${1}"
    shift
fi
ROLE="${ROLE:-single}"
case "${ROLE}" in
    single|actor|learner) ;;
    *) echo "ERROR: ROLE must be single | actor | learner (got '${ROLE}')" >&2; exit 1 ;;
esac

# ── Transport ────────────────────────────────────────────────────────────────
# On the actor, point LEARNER_IP at the GPU server:
#     LEARNER_IP=10.0.0.5 ./03_online_rl_from_criticwarmup.sh actor
# Over an SSH tunnel, keep localhost and forward both ports:
#     ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 user@server
LEARNER_IP="${LEARNER_IP:-localhost}"
LEARNER_PORT="${LEARNER_PORT:-5588}"                      # REQ/REP  actor -> learner
LEARNER_BROADCAST_PORT="${LEARNER_BROADCAST_PORT:-5589}"  # PUB/SUB  learner -> actor

# ── Sync cadence ─────────────────────────────────────────────────────────────
# Publish encoder+actor weights every N GRADIENT steps (not env steps).
# 50 is HIL-SERL's value (examples/experiments/config.py steps_per_update).
STEPS_PER_UPDATE="${STEPS_PER_UPDATE:-50}"

# Learner rate-limit as gradient-steps per env-step.
#   null -> learner free-runs at server throughput (what you want for speed)
#   4    -> pins the effective UTD to the single-process value, for an
#           apples-to-apples comparison against the non-distributed scripts
TARGET_UTD="${TARGET_UTD:-null}"

DIST=1
