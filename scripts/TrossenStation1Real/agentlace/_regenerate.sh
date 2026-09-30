#!/usr/bin/env bash
# Regenerate every script in this folder from its parent-folder original.
# Run this after editing any ../train_residual_rl_*.sh so hyper-parameters
# never drift between the single-process and distributed variants.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
"${PYTHON_BIN:-python}" ./_regenerate.py
echo "✓ regenerated. Review with:  ./_verify_diff.sh"
