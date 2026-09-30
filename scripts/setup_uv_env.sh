#!/usr/bin/env bash
# ============================================================================
# setup_uv_env.sh — Reproducible uv environment for ResFiT (no conda needed).
# ============================================================================
# Mirrors the known-good conda `residual` env WITHOUT conda:
#   * torch 2.11.0 + torchvision/torchaudio/torchcodec 0.x  (CUDA 12.8 wheels)
#   * tensordict 0.9.1
#   * torchrl 0.9.2 built FROM SOURCE against torch 2.11 with gcc-11 (the
#     published torchrl wheels break the C++ ABI under torch 2.11 -> the
#     `_wrap_outputs` / SumSegmentTree undefined-symbol crash).
#   * robosuite / mimicgen / dexmimicgen / lerobot  -> editable from deps/
#   * resfit itself                                 -> editable (`-e .`)
#
# Usage:
#   bash scripts/setup_uv_env.sh            # create .venv and install everything
#   source .venv/bin/activate               # then use it (or call .venv/bin/python)
#
# Requirements on the machine:
#   * uv (>=0.4)                    curl -LsSf https://astral.sh/uv/install.sh | sh
#   * NVIDIA driver for CUDA 12.8   (torch 2.11 +cu128 wheels)
#   * a C++ toolchain for the torchrl SOURCE build. Prefer gcc-11..gcc-13 whose
#     libstdc++ CXXABI is <= the RUNTIME libstdc++ (a too-new gcc yields a .so
#     that needs a CXXABI the runtime lacks -> `GLIBCXX/CXXABI ... not found`).
#   * ffmpeg on PATH               (imageio uses it for eval/training videos;
#                                    the code auto-picks `which ffmpeg`).
#   * for headless rendering, export MUJOCO_GL=egl (or osmesa) when RUNNING.
#
# Do NOT use `uv run` / `uv sync` afterwards — they re-sync the project and would
# uninstall the manually-built torchrl. Use `.venv/bin/python` or activate.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"
PYVER="${PYVER:-3.10}"
CU_INDEX="${CU_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCHRL_REF="${TORCHRL_REF:-v0.9.2}"

CC_BIN="$(command -v gcc-11 || command -v gcc)"
CXX_BIN="$(command -v g++-11 || command -v g++)"
echo "Using CC=$CC_BIN CXX=$CXX_BIN for the torchrl source build"

echo "==> [1/5] Creating uv venv ($VENV, python $PYVER)"
uv venv --python "$PYVER" "$VENV"

UVPIP=(uv pip install --python "$VENV")

echo "==> [2/5] Installing torch stack (CUDA 12.8, pinned)"
"${UVPIP[@]}" \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 torchcodec==0.11.0 \
    --index-url "$CU_INDEX"

echo "==> [3/5] Installing tensordict + building torchrl $TORCHRL_REF from source"
"${UVPIP[@]}" tensordict==0.9.1
CC="$CC_BIN" CXX="$CXX_BIN" "${UVPIP[@]}" \
    --no-build-isolation --no-deps \
    "torchrl @ git+https://github.com/pytorch/rl.git@${TORCHRL_REF}"

echo "==> [4/5] Installing resfit + vendored editable deps (torch stack pinned)"
# The vendored editable deps hard-pin versions the resfit code can't use, so we
# feed uv BOTH files:
#   * -c constraints-uv.txt  -> keep the CUDA torch stack from being re-resolved.
#   * --overrides uv-overrides.txt -> REPLACE lerobot's hard gymnasium==0.29.1
#     (needs 1.x autoreset), datasets (3.x API), and mujoco (robosuite needs 3.3).
#     A plain constraint here is "unsatisfiable" against a hard `==` pin.
# dexmimicgen additionally hard-pins numpy==1.23.3 (a stale 2023 pin); the whole
# stack runs on numpy 2.2.6, so install it WITHOUT its deps (its real runtime
# deps come from the other editables + the torch stack).
"${UVPIP[@]}" -c constraints-uv.txt --overrides uv-overrides.txt \
    -e deps/robosuite \
    -e deps/mimicgen \
    -e deps/lerobot \
    -e .
"${UVPIP[@]}" -c constraints-uv.txt --overrides uv-overrides.txt --no-deps -e deps/dexmimicgen

echo "==> [5/5] Sanity check"
"$VENV/bin/python" - <<'PY'
import torch
# The torch stack must be the single matched CUDA build: torch 2.11.0(+cu128) with
# torchvision 0.26.0 and the source-built torchrl 0.9.2. If ANYTHING later bumps
# torch (a `uv run`/`uv sync`, or a `uv pip install` without -c constraints-uv.txt),
# torchvision's C++ ops stop binding -> "operator torchvision::nms does not exist".
assert torch.__version__.startswith("2.11.0"), (
    f"torch must be 2.11.0(+cu128) but is {torch.__version__}. Something upgraded it. "
    "Reinstall the matched stack:\n"
    "  uv pip install --python .venv --index-url https://download.pytorch.org/whl/cu128 "
    "torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 torchcodec==0.11.0"
)
import torchvision
import torchvision.ops  # binds the torchvision C++ ops; fails loudly on a torch/vision ABI mismatch
import torchrl
from torchrl.data import ReplayBuffer  # exercises the compiled torchrl C++ extension
from torchrl.data.replay_buffers.storages import LazyMemmapStorage  # noqa: F401
import tensordict, gymnasium, mujoco, transformers  # noqa: F401
import robosuite, lerobot  # noqa: F401
print(f"torch      {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"torchvision {torchvision.__version__}  (nms binds OK)")
print(f"torchrl    {torchrl.__version__}  C++ OK")
print(f"tensordict {tensordict.__version__}  gymnasium {gymnasium.__version__}  mujoco {mujoco.__version__}")
print("resfit uv env OK")
PY

echo ""
echo "✓ uv env ready at $VENV"
echo "  Activate:  source $VENV/bin/activate"
echo "  Or run:    $VENV/bin/python <script>   (avoid 'uv run' — it re-syncs the project)"
