# ResFiT — uv environment setup on a server (no conda)

This is a hand-off guide for setting up the **resfit RL training / PBRS labeling**
environment with `uv` on a fresh server. It captures every issue we already hit
and fixed on the laptop, so you (or an assistant on the server) can reproduce the
known-good environment quickly.

The whole thing is driven by three files at the repo root:
- `scripts/setup_uv_env.sh` — the installer.
- `constraints-uv.txt` — pins the CUDA torch stack (never re-resolved).
- `uv-overrides.txt` — **overrides** the vendored deps' bad hard pins.

---

## 0. TL;DR

```bash
# from the repo root
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not installed
sudo apt-get install -y ffmpeg                     # or ensure ffmpeg is on PATH
bash scripts/setup_uv_env.sh                       # builds .venv (a few minutes)
source .venv/bin/activate                          # then use .venv/bin/python
```

If it finishes with `resfit uv env OK`, you're done. If anything fails, jump to
**§4 Troubleshooting** — every failure we saw is listed there with the fix.

> Do **NOT** run `uv run` or `uv sync` in this repo afterwards. They re-sync the
> project to `pyproject.toml` and will uninstall the manually-built `torchrl`.
> Always use `.venv/bin/python` (or activate the venv).

---

## 1. Prerequisites on the machine

| Need | Why | Check |
|---|---|---|
| `uv` >= 0.4 | installer | `uv --version` |
| NVIDIA driver for **CUDA 12.8** | torch 2.11 `+cu128` wheels | `nvidia-smi` |
| Python **3.10** | matches the vendored deps | uv can fetch it |
| **gcc/g++** (ideally 11–13) | `torchrl` is built **from source** | `gcc --version` |
| **ffmpeg** on `PATH` | imageio writes eval/training videos | `which ffmpeg` |
| GL libs for headless MuJoCo | offscreen rendering | see §3 |
| **CUDA toolkit / NPP libs** | `torchcodec` links `libnppicc.so.12`; the driver alone is NOT enough | `ls /usr/local/cuda/targets/x86_64-linux/lib/libnppicc.so.12` |

> This list has only ever been checked on a laptop that already satisfied all of
> it. A bare server turned up a missing prerequisite not listed here - see
> `SETUP_FIXES.md` §3b, still unresolved. Treat the list as incomplete.

The trickiest one is the C++ toolchain — see §2.

---

## 2. Why `torchrl` is built from source (and the ABI caveat)

The published `torchrl==0.9.2` **PyPI wheel** is ABI-built against an older torch
and crashes on torch 2.11 with `undefined symbol: ... _wrap_outputs` /
`SumSegmentTree...`. The pytorch cu128 index only has the *newer* `torchrl 0.13.x`
(needs `tensordict 0.13` and has API changes vs this code, which targets 0.9.2).
So the script builds **torchrl 0.9.2 from git** against the installed torch 2.11:

```bash
CC=gcc-11 CXX=g++-11 uv pip install --no-build-isolation --no-deps \
    "torchrl @ git+https://github.com/pytorch/rl.git@v0.9.2"
```

**ABI caveat (the main server risk):** the compiled `.so` links against the
libstdc++ of whatever gcc built it. If you build with a gcc **newer** than the
libstdc++ present at *runtime*, `import torchrl` fails with
`GLIBCXX_x.x.x not found` / `CXXABI_x.x.x not found`. On the laptop, gcc **11.4**
produced a `.so` needing only `CXXABI_1.3.13`, which the runtime had → works.

If `import torchrl` fails on the server with a GLIBCXX/CXXABI error:
1. Build with an **older** gcc: `CC=gcc-11 CXX=g++-11 bash scripts/setup_uv_env.sh`
   (the script auto-prefers `gcc-11` if present; install it: `apt-get install g++-11`).
2. OR point the runtime at a newer libstdc++:
   `export LD_LIBRARY_PATH=/path/to/newer/libstdc++/lib:$LD_LIBRARY_PATH`
   (e.g. a conda env's `lib`, or `apt-get install libstdc++6` from a newer toolchain).
3. Verify: `.venv/bin/python -c "import torchrl; from torchrl.data import ReplayBuffer; print(torchrl.__version__, 'C++ OK')"`

The build takes ~1–2 min; that's expected.

---

## 3. Headless rendering (MuJoCo / robosuite)

The RL env renders offscreen. When **running** training/eval, export:

```bash
export MUJOCO_GL=egl        # preferred on servers with a GPU
# or, if EGL is unavailable:
export MUJOCO_GL=osmesa     # needs libosmesa6 (apt-get install libosmesa6)
```

If env creation fails on GL, install: `apt-get install -y libegl1 libgl1 libglfw3`.

`ffmpeg` must be on `PATH` — imageio uses it to write eval and training-rollout
videos. The code auto-detects it (`shutil.which("ffmpeg")`) and sets
`IMAGEIO_FFMPEG_EXE`, because imageio-ffmpeg's bundled binary can fail to launch
in fresh envs and then **hangs** on a network download. If you ever see a hang at
video-writing time, `export IMAGEIO_FFMPEG_EXE=$(which ffmpeg)` explicitly.

---

## 4. Troubleshooting (every failure we hit + fix)

All of these are already handled by the script/constraints/overrides — this table
is so you can recognize a variant quickly.

| Symptom | Cause | Fix (already applied) |
|---|---|---|
| `torchrl ... undefined symbol _wrap_outputs` | PyPI torchrl wheel vs torch 2.11 ABI | build from source (step 3) |
| `import torchrl` → `CXXABI_/GLIBCXX_ not found` | built with too-new gcc | build with gcc-11 or fix `LD_LIBRARY_PATH` (§2) |
| `dexmimicgen==0.1 depends on numpy==1.23.3 and numpy==2.2.6 ... unsatisfiable` | stale hard pin | `dexmimicgen` installed `--no-deps` |
| `lerobot==0.1.0 depends on gymnasium==0.29.1 and gymnasium==1.1.1 ... unsatisfiable` | constraint can't override a hard `==` | moved gymnasium to `uv-overrides.txt` (`--overrides`) |
| `module 'gymnasium.vector' has no attribute 'AutoresetMode'` | gymnasium 0.29.1 installed | override forces `gymnasium==1.1.1` |
| `mj_fullM(): incompatible function arguments` at env reset | mujoco 3.10 changed signature | override forces `mujoco==3.3.2` |
| `torch.stack(...): must be tuple of Tensors, not Column` in LeRobotDataset | datasets 5.x lazy Column | override forces `datasets==3.6.0` |
| `ModuleNotFoundError: transformers / tabulate / accelerate` | lerobot optional extras | added to `pyproject.toml` deps |
| hang when writing a video | imageio-ffmpeg downloads its binary | code uses system ffmpeg; `export IMAGEIO_FFMPEG_EXE=$(which ffmpeg)` |
| `operator torchvision::nms does not exist` | **torch got upgraded** (e.g. to 2.12.x) so torchvision/torchrl no longer match | reinstall the matched torch stack (see box below) |
| a `uv run`/`uv sync` removed torchrl OR bumped torch | project re-sync ignores the pinned stack | reinstall stack / `bash scripts/setup_uv_env.sh`; use `.venv/bin/python` instead |

> **Repair the torch stack** (fixes `torchvision::nms does not exist` and torchrl ABI):
> ```bash
> uv pip install --python .venv --index-url https://download.pytorch.org/whl/cu128 \
>   torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 torchcodec==0.11.0
> ```
> The whole stack must be the **same matched build**: torch **2.11.0+cu128** ↔
> torchvision **0.26.0+cu128** ↔ the source-built torchrl **0.9.2**. If `uv pip list`
> shows torch at anything other than `2.11.0+cu128` (e.g. `2.12.1`), that's the bug —
> something ran `uv run`/`uv sync` or a `uv pip install` without `-c constraints-uv.txt`
> and pulled a newer torch via lerobot's `torch>=2.2.1`. Never use `uv run`/`uv sync`.

If a **new** editable-dep hard-pin conflict appears (same shape as the gymnasium
one), add the wanted version to `uv-overrides.txt` — that's the general fix.
Note: keep the **torch stack** (`torch/torchvision/torchaudio/torchcodec`) in
`constraints-uv.txt`, NOT in overrides — an override would try to re-resolve them
from PyPI (CPU wheels) instead of keeping the installed `+cu128` builds.

---

## 5. Verify

```bash
.venv/bin/python - <<'PY'
import torch, torchrl, tensordict, gymnasium, mujoco, transformers, robosuite, lerobot
from torchrl.data import ReplayBuffer  # exercises the compiled C++ extension
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("torchrl", torchrl.__version__, "C++ OK")
print("gymnasium", gymnasium.__version__, "mujoco", mujoco.__version__)
PY
```
Expected: `torch 2.11.0+cu128 ... torchrl 0.9.2+... C++ OK ... gymnasium 1.1.1 mujoco 3.3.2`.

---

## 6. Running things under the uv env (no conda)

Adjust the **paths** to your server (base ACT checkpoint, offline dataset).

**Reward server (SARM)** — this is a *separate* component with its own deps/checkpoints
(see the SARM/opensarm setup). It must be running before labeling or RL training,
on the URL the training points at (e.g. `http://127.0.0.1:8003`). Health check:
`curl -s http://127.0.0.1:8003/health`.

**Offline PBRS labeling:**
```bash
.venv/bin/python scripts/label_offline_pbrs.py \
  --dataset_root /PATH/TO/SARM-robosuite-can-mh-stages_v21 \
  --server_url http://127.0.0.1:8003 \
  --task_prompt "pick up the can and place it in the bin" \
  --output ./pbrs_sarm_progress.parquet \
  --gamma 0.995 --hysteresis_k 4 --query_every_k 4
```
Defaults are already gamma 0.995 / hysteresis 4 / query 4. The parquet stores the
full labeling config; RL loading hard-errors if it disagrees with the RL config.

**RL training:** edit the paths at the top of
`scripts/train_residual_rl_can_sarm_stage_pbrs.sh` (`BASE_LOCAL_PATH`,
`OFFLINE_ROOT`, `REWARD_SERVER_URL`) then run it with the uv python on PATH:
```bash
source .venv/bin/activate
export MUJOCO_GL=egl
bash scripts/train_residual_rl_can_sarm_stage_pbrs.sh
```
(The script calls bare `python`; activating the venv makes that `.venv/bin/python`.)
Training-rollout debug videos + per-frame stage/PBRS JSON land in the run's
`outputs/<run>/training_rollouts/` and W&B `training/rollout_*`.

**Watch resources:** the reward server + RL both use the GPU; keep an eye on
`nvidia-smi`. Reduce `algo.batch_size` / `algo.buffer_size` / `offline_data.num_episodes`
if you hit OOM.
