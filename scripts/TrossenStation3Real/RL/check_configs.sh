#!/usr/bin/env bash
# =============================================================================
# check_configs.sh - compose every mode's config without running anything.
#
# Hydra validates override keys and types at compose time, so this catches a
# misspelled key or a wrong-typed value before you are standing at the robot.
# It found a real one: TARGET_ACTION_NOISE was set to 0.1 when
# agent.target_action_noise is a bool (rlpd.py:129).
#
# Also asserts that the HASHED parameters are byte-identical across every mode -
# if they are not, the modes silently use different replay-buffer caches.
#
#     ./check_configs.sh
# =============================================================================
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_HERE}/../../.." && pwd)"
PY="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

HASHED_RE='^(algo\.(n_step|gamma|batch_size|buffer_size|learning_starts|offline_fraction|sampling_strategy|random_action_noise_scale)|real_max_steps|offline_data\.(name|root|num_episodes|use_base_policy_for_base_actions|min_action_range|min_state_std)|rl_camera|reward_shaping)='

# 04_* refuses to run without an existing checkpoint file, which is the point -
# give it a real (empty) one so its config can still be composed here.
_FAKE_CKPT="$(mktemp)"
trap 'rm -f "${_FAKE_CKPT}"' EXIT

fail=0
ref=""
for f in "${_HERE}"/0*.sh "${_HERE}"/agentlace/0*.sh; do
    name="${f#${_HERE}/}"
    mapfile -t ARGS < <(RESUME_CKPT="${_FAKE_CKPT}" DRY_RUN=1 "$f" 2>/dev/null)
    if [[ ${#ARGS[@]} -lt 3 ]]; then
        printf '  %-46s SKIP (no command produced)\n' "$name"; continue
    fi
    entry="${ARGS[1]}"
    overrides=("${ARGS[@]:2}")

    if out=$("$PY" "$entry" "${overrides[@]}" --cfg job 2>&1); then
        printf '  %-46s composes OK\n' "$name"
    else
        printf '  %-46s FAILED\n' "$name"
        echo "$out" | grep -viE "robosuite|^\[" | grep -iE "error|key |validation" | head -3 | sed 's/^/        /'
        fail=1
    fi

    hashed=$(printf '%s\n' "${overrides[@]}" | grep -E "$HASHED_RE" | sort)
    if [[ -z "$ref" ]]; then ref="$hashed"; ref_name="$name"
    elif [[ "$hashed" != "$ref" ]]; then
        printf '        HASHED PARAMS DIFFER from %s:\n' "$ref_name"
        diff <(echo "$ref") <(echo "$hashed") | sed 's/^/          /'
        fail=1
    fi
done

echo
if [[ $fail -eq 0 ]]; then
    echo "  all modes compose, and their hashed parameters are identical"
else
    echo "  PROBLEMS FOUND (see above)"
fi
exit $fail
