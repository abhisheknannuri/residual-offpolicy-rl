#!/usr/bin/env bash
# Show exactly how each generated script differs from its original.
# Expect ONLY: banner, common_dist.sh source block, entrypoint, CONFIG_NAME,
# and the dist.* CMD arguments. Anything else means the generator drifted.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in train_residual_rl_*.sh; do
    echo "══════════════════════════════════════════════════════════════"
    echo " $f"
    echo "══════════════════════════════════════════════════════════════"
    diff "../$f" "$f"
    echo
done
