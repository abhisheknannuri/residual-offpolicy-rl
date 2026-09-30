#!/usr/bin/env bash
# ============================================================================
# clone_deps.sh - restore the vendored dependencies a fresh checkout is missing.
# ============================================================================
# deps/ and agentlace/ are .gitignore'd (each is a foreign repo with its own
# .git), so a clone of this repository cannot build or run until this has been
# executed. Run it BEFORE scripts/setup_uv_env.sh.
#
#   bash scripts/clone_deps.sh
#
# Idempotent: an existing clone is left alone unless it is on the wrong commit,
# in which case the script reports it and stops rather than resetting work you
# may have in there.
#
# See DEPENDENCIES.md for where each pin came from.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# path|url|commit
DEPS=(
  "deps/robosuite|https://github.com/ARISE-Initiative/robosuite|77a4751233c29456a5381209e30dd0dbf39a6557"
  "deps/mimicgen|https://github.com/NVlabs/mimicgen|72bd767c255545f462e7ccfb2731f2e5d4c1d9bb"
  "deps/dexmimicgen|https://github.com/NVlabs/dexmimicgen|e606f36a38b1d4ba8f56d06d6c0cd059b20ebbaf"
  "deps/lerobot|https://github.com/huggingface/lerobot|69901b9b6a2300914ca3de0ea14b6fa6e0203bd4"
  # NOT installed into the venv - imported from this path at runtime by
  # resfit/rl_finetuning/off_policy/distributed/transport.py. See DEPENDENCIES.md.
  "agentlace|https://github.com/youliangtan/agentlace|76984f9"
)

fail=0
for entry in "${DEPS[@]}"; do
    IFS='|' read -r path url commit <<< "$entry"

    if [ -d "$path/.git" ]; then
        have="$(git -C "$path" rev-parse HEAD)"
        if [ "${have:0:${#commit}}" = "$commit" ]; then
            echo "  ok      $path @ ${commit:0:8}"
        else
            echo "  WRONG   $path is at ${have:0:8}, expected ${commit:0:8}"
            echo "          not touching it - you may have local work. To fix:"
            echo "            git -C $path fetch && git -C $path checkout $commit"
            fail=1
        fi
        continue
    fi

    echo "  clone   $path"
    mkdir -p "$(dirname "$path")"
    git clone --quiet "$url" "$path"
    git -C "$path" checkout --quiet "$commit"
    echo "  ok      $path @ ${commit:0:8}"
done

if [ "$fail" -ne 0 ]; then
    echo ""
    echo "Some clones are on unexpected commits (see above). Fix them, then re-run."
    exit 1
fi

echo ""
echo "All dependencies present. Next: bash scripts/setup_uv_env.sh"
