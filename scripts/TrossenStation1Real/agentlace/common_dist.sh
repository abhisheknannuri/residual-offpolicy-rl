#!/usr/bin/env bash
# =============================================================================
# common_dist.sh - shared role / transport settings for the distributed
#                  (actor + learner) ResFiT scripts in this folder.
#
# Sourced by every train_*.sh here. Never run directly.
#
# Design rule: the actor and the learner run THE SAME script file with a
# different role argument, so they cannot drift apart on any hyper-parameter.
# A silent actor/learner config mismatch (different action_scale, n_step,
# normalization, ...) produces no error - just an actor acting in one action
# scale while the critic was trained in another. Do not "simplify" this by
# maintaining two separate scripts.
# =============================================================================

# ── Role ─────────────────────────────────────────────────────────────────────
# First positional arg, else $ROLE, else "single".
#   single  - today's behaviour: one process, env + training, no networking.
#             Use this to verify parity before touching the distributed path.
#   actor   - laptop: env, cameras, pedal/leader, policy inference, n-step, outbox
#   learner - GPU server: buffers, critic/actor updates, checkpoints, wandb
if [[ -n "${1:-}" && "${1}" != -* ]]; then
    ROLE="${1}"
    shift
fi
ROLE="${ROLE:-single}"

case "${ROLE}" in
    single|actor|learner) ;;
    *) echo "ERROR: ROLE must be one of: single | actor | learner  (got '${ROLE}')" >&2
       exit 1 ;;
esac

# ── Transport ────────────────────────────────────────────────────────────────
# On the actor, LEARNER_IP must point at the GPU server:
#     LEARNER_IP=10.0.0.5 ./train_residual_rl_real.sh actor
# Over an SSH tunnel, keep localhost and forward both ports:
#     ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 user@server
LEARNER_IP="${LEARNER_IP:-localhost}"
LEARNER_PORT="${LEARNER_PORT:-5588}"             # REQ/REP  actor -> learner (transitions, stats)
LEARNER_BROADCAST_PORT="${LEARNER_BROADCAST_PORT:-5589}"  # PUB/SUB learner -> actor (weights)

# ── Sync cadence ─────────────────────────────────────────────────────────────
# Publish encoder+actor weights every N *gradient* steps (not env steps).
# 50 is HIL-SERL's value (examples/experiments/config.py steps_per_update).
# Raise it if the weight blob turns out to be large on your ViT config.
STEPS_PER_UPDATE="${STEPS_PER_UPDATE:-50}"

# Optional learner rate-limit, expressed as gradient-steps per env-step.
# null  => learner free-runs at server throughput (what you want for speed).
# 4     => pins the effective UTD to today's single-process value, which is
#          what you need for an apples-to-apples comparison against your
#          existing sim/laptop runs. See RESFIT_DISTRIBUTED_MIGRATION.md §8.
TARGET_UTD="${TARGET_UTD:-null}"

# ── Interpreter ──────────────────────────────────────────────────────────────
# NOTE: do NOT use `uv run` here - it rebuilds/modifies the env. Point this at
# an interpreter directly. Default matches the originals' bare `python`.
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "────────────────────────────────────────────────────────────────"
echo " ROLE=${ROLE}   learner=${LEARNER_IP}:${LEARNER_PORT}/${LEARNER_BROADCAST_PORT}"
echo " steps_per_update=${STEPS_PER_UPDATE}   target_utd=${TARGET_UTD}"
echo " python=${PYTHON_BIN}"
echo "────────────────────────────────────────────────────────────────"
