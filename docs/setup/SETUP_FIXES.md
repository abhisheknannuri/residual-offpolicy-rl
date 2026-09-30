# Setup Fixes for ResFiT

Fixes applied while setting up the project on a machine with RTX 4090 Mobile (16 GB), Ubuntu 22.04, Python 3.10.

---

## 1. Missing `__init__.py` files

**Problem:** `ModuleNotFoundError: No module named 'resfit'`

**Why:** The `resfit/` directory and its subdirectories had no `__init__.py` files, so Python didn't recognize them as importable packages.

**Fix:** Created empty `__init__.py` in all package directories:

```bash
cd /path/to/residual-offpolicy-rl
for dir in \
  resfit \
  resfit/dexmg \
  resfit/dexmg/environments \
  resfit/lerobot \
  resfit/lerobot/configs \
  resfit/lerobot/policies \
  resfit/lerobot/policies/act \
  resfit/lerobot/policies/diffusion \
  resfit/lerobot/scripts \
  resfit/lerobot/utils \
  resfit/rl_finetuning \
  resfit/rl_finetuning/config \
  resfit/rl_finetuning/off_policy \
  resfit/rl_finetuning/off_policy/networks \
  resfit/rl_finetuning/scripts \
  resfit/rl_finetuning/utils \
  resfit/rl_finetuning/wrappers; do
  [[ -f "$dir/__init__.py" ]] || touch "$dir/__init__.py"
done
```

Also added the project root to `PYTHONPATH` (in `~/.bashrc`) so it works from any directory:

```bash
export PYTHONPATH="/path/to/residual-offpolicy-rl:$PYTHONPATH"
```

---

## 2. torchcodec incompatible with PyTorch version

**Problem:** `RuntimeError: Could not load libtorchcodec` — torchcodec 0.4.0 (installed by `setup_dexmg.sh`) is incompatible with PyTorch 2.10.0 (installed by `setup_lerobot.sh`).

**Why:** The setup scripts run sequentially. `setup_lerobot.sh` upgrades torch to latest (2.10.0), then `setup_dexmg.sh` pins `torchcodec==0.4.0` which only supports older PyTorch versions.

**Fix:** Reinstall torchcodec from the PyTorch CUDA wheel index (gets a compatible version):

```bash
pip uninstall -y torchcodec
pip install --no-cache-dir torchcodec --index-url https://download.pytorch.org/whl/cu128
```

This installed `torchcodec 0.10.0+cu128` which is compatible with PyTorch 2.10.0.

---

## 3. FFmpeg not installed in conda environment

**Problem:** torchcodec couldn't find FFmpeg shared libraries (tried versions 4-7, all failed).

**Why:** The `micromamba install` command in `setup_dexmg.sh` may have failed silently. The system FFmpeg (4.4) was too old and not linked into the conda env.

**Fix:** Install FFmpeg 7 via conda:

```bash
conda install -n residual -c conda-forge "ffmpeg>=6,<8" -y
```

---

## 4. CUDA Out of Memory during training

**Problem:** `torch.OutOfMemoryError` when running BC training with `--batch_size 256 --eval_num_envs 16` on a 16 GB GPU.

**Why:** 16 async evaluation environments (each with MuJoCo rendering) plus the model + batch_size=256 exceed 16 GB VRAM. The display server (Xorg + GNOME) also consumes ~1 GB.

**Fix:** Reduce batch size and eval environment count:

```bash
python resfit/lerobot/scripts/train_bc_dexmg.py \
    --dataset ankile/dexmg-two-arm-coffee \
    --policy act \
    --steps 200000 \
    --batch_size 128 \
    --wandb_project dexmg-bc \
    --eval_env TwoArmCoffee \
    --rollout_freq 5000 \
    --eval_video_key observation.images.frontview \
    --eval_render_size 224 \
    --eval_num_envs 4 \
    --eval_num_episodes 20 \
    --wandb_enable
```

If still tight, prepend: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 5. pip dependency warnings (safe to ignore)

**Problem:** pip shows `ERROR: pip's dependency resolver...` warnings about missing lerobot deps (cmake, flask, opencv-python-headless, pymunk, pyserial, pyzmq, rerun-sdk, zarr) and gymnasium version mismatch.

**Why:** lerobot was installed with `--no-deps` and only the deps needed for training were installed separately. The gymnasium conflict (0.29.1 vs 1.1.1) is intentional — dexmimicgen requires 1.1.1.

**Fix:** None needed. These are pip warnings, not actual errors. The missing packages are for physical robot control, visualization, and other features not used in training.

---

## 6. HuggingFace CLI login command

**Problem:** `hf: command not found`

**Why:** The correct CLI command is `huggingface-cli`, not `hf`.

**Fix:**

```bash
huggingface-cli login
```

---

## 7. torchrl C++ extension ABI mismatch (`SumSegmentTreeFp32` undefined symbol)

**Problem:** `NameError: name 'SumSegmentTreeFp32' is not defined` when creating a `TensorDictPrioritizedReplayBuffer`. The underlying error is:

```
ImportError: .../torchrl/_torchrl.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c106detail...
```

**Why:** The project's `setup_rlpd_robosuite.sh` script installs `torchrl==0.9.2` and `tensordict==0.9.1`, which were compiled against an older PyTorch C++ ABI. When paired with PyTorch 2.10.0 (installed by `setup_lerobot.sh`), the pre-built `.so` files reference symbols that no longer exist in the newer `libtorch`. torchrl catches the `ImportError` silently and falls back to a pure-Python mode that lacks `SumSegmentTreeFp32`.

**Fix:** Upgrade torchrl and tensordict to versions compiled for PyTorch 2.10:

```bash
pip install torchrl==0.11.1 tensordict==0.11.0
```

Verify the fix:

```bash
python -c "from torchrl._torchrl import SumSegmentTreeFp32; print('OK')"
```

---

## 8. MuJoCo on-screen viewer window not appearing with `headless=false`

**Problem:** Running RL training with `headless=false` sets `MUJOCO_GL=glfw` and `has_renderer=True` in robosuite, but no MuJoCo viewer window pops up on screen.

**Why:** Setting `has_renderer=True` creates robosuite's viewer object, but the viewer window only updates when `env.render()` is explicitly called. The Gymnasium gym wrapper's `step()` method never called the robosuite environment's `.render()` — so the viewer was created but never refreshed.

**Fix:** Added an `env.render()` call inside `RobosuiteGymWrapper.step()` when `headless=False`. This calls robosuite's `viewer.render()` each step, which swaps the framebuffer and displays the scene in the on-screen window.

---

## 9. Matplotlib/Tkinter crash during Residual RL training (`Tcl_AsyncDelete`)

```sh
Exception ignored in: <function Image.__del__ at 0x7731f80fb6d0>
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/tkinter/__init__.py", line 4056, in __del__
    self.tk.call('image', 'delete', self.name)
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7731e8c72830>
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/tkinter/__init__.py", line 388, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7731e8c72830>
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/tkinter/__init__.py", line 388, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7731e8c72830>
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/tkinter/__init__.py", line 388, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7731e8c72830>
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/tkinter/__init__.py", line 388, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
RuntimeError: main thread is not in main loop
Tcl_AsyncDelete: async handler deleted by the wrong thread
Aborted (core dumped)
(residual) qte9489@cw011081522:~/personal_abhi/temp/residual-offpolicy-rl$ Process Worker<AsyncVectorEnv>-0:
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 738, in _async_worker
    command, data = pipe.recv()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 383, in _recv
    raise EOFError
EOFError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 848, in _async_worker
    pipe.send((None, False))
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 206, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 411, in _send_bytes
    self._send(header + buf)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 368, in _send
    n = write(self._handle, buf)
BrokenPipeError: [Errno 32] Broken pipe
Process Worker<AsyncVectorEnv>-1:
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 738, in _async_worker
    command, data = pipe.recv()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 383, in _recv
    raise EOFError
EOFError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 848, in _async_worker
    pipe.send((None, False))
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 206, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 411, in _send_bytes
    self._send(header + buf)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 368, in _send
    n = write(self._handle, buf)
BrokenPipeError: [Errno 32] Broken pipe
Process Worker<AsyncVectorEnv>-0:
Process Worker<AsyncVectorEnv>-2:
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 738, in _async_worker
    command, data = pipe.recv()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 383, in _recv
    raise EOFError
EOFError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 848, in _async_worker
    pipe.send((None, False))
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 206, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 411, in _send_bytes
    self._send(header + buf)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 368, in _send
    n = write(self._handle, buf)
BrokenPipeError: [Errno 32] Broken pipe
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 738, in _async_worker
    command, data = pipe.recv()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 383, in _recv
    raise EOFError
EOFError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 848, in _async_worker
    pipe.send((None, False))
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 206, in send
    self._send_bytes(_ForkingPickler.dumps(obj))
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 411, in _send_bytes
    self._send(header + buf)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 368, in _send
    n = write(self._handle, buf)
BrokenPipeError: [Errno 32] Broken pipe
Process Worker<AsyncVectorEnv>-3:
Traceback (most recent call last):
  File "~/miniforge3/envs/residual/lib/python3.10/site-packages/gymnasium/vector/async_vector_env.py", line 738, in _async_worker
    command, data = pipe.recv()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
    buf = self._recv(4)
  File "~/miniforge3/envs/residual/lib/python3.10/multiprocessing/connection.py", line 383, in _recv
    raise EOFError
EOFError

During handling of the above exception, another exception occurred:
```

**Symptom:** Training crashes after some number of evaluations (nondeterministic — could be step 10K or step 90K) with:

```
RuntimeError: main thread is not in main loop
Tcl_AsyncDelete: async handler deleted by the wrong thread
Aborted (core dumped)
```

Followed by `BrokenPipeError` from `AsyncVectorEnv` worker pipes.

**Why:** The evaluation utility `resfit/rl_finetuning/utils/evaluate_dexmg.py` imports `matplotlib.pyplot` at the top of the file to generate Q-value diagnostic plots (saved as PNGs and uploaded to WandB). The plots themselves work fine — the problem is a **side effect of the import**.

When `import matplotlib.pyplot as plt` runs, matplotlib auto-selects a GUI backend. On Linux with a display server, the default is **TkAgg**, which imports `tkinter` and initializes the **Tcl event loop** as a side effect — even though no window is ever opened. This happens once at process startup, before any evaluation runs.

The Tcl event loop state then sits dormant in memory for the entire training run. The crash occurs when Python's garbage collector happens to collect Tcl-related objects from a **non-main thread** (e.g., during `AsyncVectorEnv` worker cleanup or a background GC cycle). Tcl asserts that cleanup must happen on the main thread and calls `abort()`. This kills the main process, which then breaks the pipes to all AsyncVectorEnv worker processes.

**Why it's nondeterministic:** The crash depends on when Python's GC runs, memory pressure, and thread scheduling. It could happen at the 1st evaluation or the 100th. In the observed case, 9 evaluations (steps 10K–90K) completed successfully before the GC timing aligned badly at ~93K steps.

**What the plots are:** Q-value trajectory plots showing predicted Q-values over episode steps, with successful episodes in green and failed ones in red. They are saved as PNG files to `outputs/<run_name>/` and uploaded to WandB under the `value/q_trajectories` panel. No interactive window is ever shown — the plots are always rendered to files.

**Fix:** Force the non-interactive `Agg` backend before importing pyplot in `evaluate_dexmg.py`:

```python
# resfit/rl_finetuning/utils/evaluate_dexmg.py
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — prevents Tkinter/Tcl thread conflicts
import matplotlib.pyplot as plt
```

The `Agg` backend renders to in-memory buffers and files without any GUI toolkit — Tkinter is never imported, and the Tcl event loop is never created. The output PNG files are **byte-for-byte identical** to what TkAgg would produce. The only thing lost is the ability to call `plt.show()` (which was never used).

**Import chain that caused the crash:**
```
train_residual_td3.py
  → from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
    → import matplotlib.pyplot as plt              ← triggers backend selection
      → matplotlib auto-selects TkAgg backend
        → import tkinter                           ← Tcl event loop initialized
          → Tcl state sits dormant in main process
            → AsyncVectorEnv spawns workers
              → GC collects Tcl objects from wrong thread (eventually)
                → Tcl_AsyncDelete asserts main thread → abort()
```

---
---



### v1.6 (legacy, incompatible)
- Old format: one big monolithic parquet file, stats in safetensors, no chunking
- **Cannot be loaded** by current code — raises an error telling you to convert

### v2.0 (major rewrite)
- Introduced the current structure: per-episode parquet files, chunked directories, videos organized by camera, `info.json` with full feature schema, `tasks.jsonl`, `episodes.jsonl`
- **Can still be loaded** by v2.1 code (with a warning)

### v2.1 (current — what your dataset uses)
- Only change from v2.0: **per-episode statistics** (`episodes_stats.jsonl`) replaced the old global `stats.json`. Aggregate stats are now computed on-the-fly from episode-level stats.

### Key differences table

| | v1.6 | v2.0 | v2.1 |
|---|---|---|---|
| Data format | Single parquet | Per-episode parquets in chunks | Same as v2.0 |
| Stats | `stats.safetensors` | `stats.json` (global) | `episodes_stats.jsonl` (per-episode) |
| Metadata dir | `meta_data/` | `meta/` | `meta/` |
| Tasks support | No | Yes | Yes |
| Backward compat | Must convert to v2 | Loads with warning | Current |

The version string lives in `meta/info.json` under `"codebase_version"`. When loading, LeRobot checks it — major mismatch = error, minor behind = warning, ahead of code = error.


# LeRobot Dataset Format (v2.1)

This section explains the LeRobot dataset format and specifically the `ankile/dexmg-two-arm-coffee` dataset used in this project.

## Where is the dataset stored?

```
~/.cache/huggingface/lerobot/ankile/dexmg-two-arm-coffee/
```

It gets downloaded automatically the first time you run `train_bc_dexmg.py`. Total size: ~655 MB.

## Top-level structure

```
dexmg-two-arm-coffee/
├── meta/                          # Metadata (describes everything)
│   ├── info.json                  # Master config: features, shapes, fps, paths
│   ├── episodes.jsonl             # One line per episode: index, task, length
│   ├── tasks.jsonl                # Maps task_index → task name
│   └── episodes_stats.jsonl       # Per-episode statistics (min/max/mean/std)
├── data/                          # Tabular data (states + actions)
│   ├── chunk-000/                 # Episodes 0–999
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ... (1000 files)
│   └── chunk-001/                 # Episodes 1000–1013
│       ├── episode_001000.parquet
│       └── ... (14 files)
├── videos/                        # Camera recordings (one mp4 per episode per camera)
│   ├── chunk-000/
│   │   ├── observation.images.agentview/
│   │   │   ├── episode_000000.mp4
│   │   │   └── ...
│   │   ├── observation.images.robot0_eye_in_left_hand/
│   │   │   └── ...
│   │   └── observation.images.robot0_eye_in_right_hand/
│   │       └── ...
│   └── chunk-001/
│       └── (same camera folders, episodes 1000–1013)
└── README.md
```

## Chunks explained

Datasets are split into **chunks** of 1000 episodes each (configurable via `chunks_size` in `info.json`). This avoids having thousands of files in a single folder.

- `chunk-000` → episodes 0 to 999 (1000 episodes)
- `chunk-001` → episodes 1000 to 1013 (14 episodes, the remainder)

The chunk for a given episode is: `episode_index // chunks_size`

## The parquet files (data/)

Each parquet file stores the **tabular data for one episode** — one row per timestep (frame). No images are stored here; images come from the video files.

### Columns in each parquet file

| Column | Type | Shape | What it is |
|---|---|---|---|
| `index` | int64 | scalar | Global index across all episodes (row 0 of ep0 = 0, row 0 of ep1 = 290, etc.) |
| `frame_index` | int64 | scalar | Frame index within this episode (0, 1, 2, ..., N-1) |
| `episode_index` | int64 | scalar | Which episode this row belongs to |
| `timestamp` | float32 | scalar | Time in seconds = `frame_index / fps` (fps=20, so 0.0, 0.05, 0.1, ...) |
| `task_index` | int64 | scalar | Index into `tasks.jsonl` (0 = "TwoArmCoffee" here) |
| `observation.state` | float32 | [36] | Robot state: left/right end-effector position (3), quaternion (4), gripper joint positions (11) × 2 arms |
| `action` | float32 | [24] | Action vector sent to the robot controller |
| `next.done` | bool | scalar | `True` only on the last frame of the episode |

### Example row (episode 0, frame 0)

```
index:             0
frame_index:       0
episode_index:     0
timestamp:         0.0
task_index:        0
next.done:         False
observation.state: [-0.303, -0.248, 1.097, -0.470, 0.528, ...]  (36 values)
action:            [-0.301, -0.255, 1.089, -1.257, 1.278, ...]  (24 values)
```

Episode 0 has 290 rows (frames), so the last row has `frame_index=289`, `next.done=True`, `timestamp=14.45`.

## The video files (videos/)

Each video file is **one camera view of one episode**, encoded as mp4 (AV1 codec). There are 3 cameras:

| Camera key | What it sees |
|---|---|
| `observation.images.agentview` | Third-person view of the scene |
| `observation.images.robot0_eye_in_left_hand` | Left hand wrist camera |
| `observation.images.robot0_eye_in_right_hand` | Right hand wrist camera |

Video specs: 84×84 pixels, 20 fps, AV1 codec, yuv420p pixel format.

During training, `torchcodec` decodes individual frames from these videos on-the-fly using the `timestamp` from the parquet data.

## The metadata files (meta/)

### info.json

The master config file. Key fields:

```json
{
  "codebase_version": "v2.1",
  "total_episodes": 1014,
  "total_frames": 326707,
  "total_videos": 3042,     // 1014 episodes × 3 cameras
  "fps": 20,
  "chunks_size": 1000,
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": { ... }       // Schema for every column (dtype, shape, names)
}
```

The `data_path` and `video_path` templates show how file paths are constructed from episode index and chunk number.

### episodes.jsonl

One JSON object per line, one per episode:

```json
{"episode_index": 0, "tasks": ["TwoArmCoffee"], "length": 290}
{"episode_index": 1, "tasks": ["TwoArmCoffee"], "length": 317}
```

`length` = number of frames (rows in the parquet file = frames in each video).

### tasks.jsonl

Maps task indices to task names:

```json
{"task_index": 0, "task": "TwoArmCoffee"}
```

This dataset has only one task.

## Alignment rules (what must match)

These are the consistency requirements — if you ever create your own dataset:

1. **Parquet rows = video frames.** Episode 0 has 290 parquet rows → each of its 3 videos must have exactly 290 frames (at 20 fps = 14.5 seconds).

2. **One parquet file per episode, one video file per episode per camera.** Episode 5 → `data/chunk-000/episode_000005.parquet` + `videos/chunk-000/observation.images.agentview/episode_000005.mp4` (and same for the other 2 cameras).

3. **Chunk boundaries must be consistent.** Episodes 0–999 in `chunk-000`, 1000+ in `chunk-001`, for both `data/` and `videos/`.

4. **`info.json` must accurately describe the features.** The `shape` in `info.json` must match the actual array sizes in the parquet (e.g., action shape [24] means every action column has exactly 24 floats).

5. **`total_frames` must equal the sum of all episode lengths.** Here: 326,707 total frames across 1,014 episodes.

6. **`total_videos` = `total_episodes` × number of camera keys.** 1,014 × 3 = 3,042.

7. **`fps` must match video fps.** Both are 20.

8. **Timestamps must be consistent.** `timestamp = frame_index / fps`. Frame 0 → 0.0s, frame 1 → 0.05s, etc.

## This specific dataset: ankile/dexmg-two-arm-coffee

| Property | Value |
|---|---|
| Task | TwoArmCoffee (bimanual robot making coffee) |
| Robot | GR1 humanoid (fixed lower body) with two dexterous hands |
| Episodes | 1,014 demonstrations |
| Total frames | 326,707 |
| FPS | 20 |
| Avg episode length | ~322 frames (~16 seconds) |
| State dim | 36 (left arm: pos[3] + quat[4] + gripper[11], right arm: same) |
| Action dim | 24 |
| Cameras | 3 (agentview, left hand, right hand) at 84×84 |
| Total size | ~655 MB |
