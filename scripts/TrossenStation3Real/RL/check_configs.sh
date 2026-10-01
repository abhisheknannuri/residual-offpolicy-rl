#!/usr/bin/env bash
# =============================================================================
# check_configs.sh - compose every config without running anything.
#
# Hydra validates keys and types at compose time, so this catches a misspelled
# key or a wrong-typed value before you are standing at the robot. It has
# already caught a bool field set to 0.1, and a HASHED path that silently
# changed with the working directory.
#
# Also asserts the HASHED values are identical across every mode - if they are
# not, the modes use different replay-buffer caches without telling you.
#
#     ./check_configs.sh
# =============================================================================
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_HERE}/../../.." && pwd)"
PY="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CONF="${_HERE}/conf"
cd "${REPO_ROOT}"

PLAIN="resfit/rl_finetuning/scripts/train_residual_td3.py"
DIST="resfit/rl_finetuning/scripts/train_residual_td3_distributed.py"

# Keys that feed a replay-buffer cache hash (docs/real/BUFFER_CACHES.md).
HASHED_RE='^(task|reward_shaping|real_max_steps|  (n_step|gamma|batch_size|buffer_size|learning_starts|offline_fraction|sampling_strategy|random_action_noise_scale|name|root|num_episodes|use_base_policy_for_base_actions|min_action_range|min_state_std)):'

fail=0; ref=""; ref_name=""
for cfg in 01_populate 02_offline_rl 03_online_from_criticwarmup 04_online_from_checkpoint; do
  for variant in "" "dist_"; do
    entry="${PLAIN}"; [[ -n "$variant" ]] && entry="${DIST}"
    name="${variant}${cfg}"
    if out=$("$PY" "$entry" --config-dir "$CONF" --config-name "$name" \
             resume_ckpt=null --cfg job --resolve 2>&1); then
      printf '  %-34s composes OK\n' "$name"
    else
      printf '  %-34s FAILED\n' "$name"
      echo "$out" | grep -viE "robosuite|^\[" | grep -iE "error|key |validation" | head -3 | sed 's/^/        /'
      fail=1; continue
    fi
    hashed=$(echo "$out" | grep -E "$HASHED_RE" | sort)
    if [[ -z "$ref" ]]; then ref="$hashed"; ref_name="$name"
    elif [[ "$hashed" != "$ref" ]]; then
      printf '        HASHED VALUES DIFFER from %s:\n' "$ref_name"
      diff <(echo "$ref") <(echo "$hashed") | sed 's/^/          /'
      fail=1
    fi
  done
done

echo
if [[ $fail -eq 0 ]]; then
  echo "  all 8 configs compose, and their HASHED values are identical"
else
  echo "  PROBLEMS FOUND (see above)"
fi
exit $fail
