# Docker Setup Guide — ResFiT on Remote GPU Servers

Complete guide for setting up the ResFiT (Residual Fine-Tuning with Off-Policy RL) training environment using Docker on any GPU server. Covers first-time setup, daily usage, VS Code Dev Containers, and troubleshooting.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Server Prerequisites](#2-server-prerequisites)
3. [First-Time Setup](#3-first-time-setup)
4. [Building the Docker Image](#4-building-the-docker-image)
5. [Running Training](#5-running-training)
6. [Data Persistence & Volumes](#6-data-persistence--volumes)
7. [VS Code Dev Container (Remote Development)](#7-vs-code-dev-container-remote-development)
8. [Managing Containers on Shared Servers](#8-managing-containers-on-shared-servers)
9. [Updating Code](#9-updating-code)
10. [Environment Variables & Secrets](#10-environment-variables--secrets)
11. [Training Scripts Reference](#11-training-scripts-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick Reference / Cheat Sheet](#13-quick-reference--cheat-sheet)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  HOST SERVER (shared, untouched)                                │
│                                                                 │
│  ~/qte9489/residual-offpolicy-rl/                               │
│  ├── .env              ← secrets (WANDB_API_KEY, HF_TOKEN)      │
│  ├── data/             ← all training outputs (bind mount)      │
│  │   ├── run_2026-*/   ← RL training runs                      │
│  │   ├── bc_run_*/     ← BC training runs                      │
│  │   ├── offline_buffer_cache/                                  │
│  │   └── online_buffer_cache/                                   │
│  └── (git repo)                                                 │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  DOCKER CONTAINER (qte9489-resfit)                        │  │
│  │                                                           │  │
│  │  /app/              ← project code (baked into image)     │  │
│  │  /app/data/         ← mounted from host ./data/           │  │
│  │  /app/deps/         ← cloned repos (lerobot, robosuite)   │  │
│  │                                                           │  │
│  │  Python 3.10 + PyTorch 2.10 + CUDA 12.8                  │  │
│  │  MuJoCo 3.3.2 + robosuite + all dependencies             │  │
│  │  EGL rendering (headless, no display needed)              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Docker?

- **Isolation:** Nothing installed on the host. Other users are unaffected.
- **Reproducibility:** Same environment everywhere — laptop, server A, server B.
- **No conda behind VPN:** Corporate proxies/VPNs often break conda SSL. Docker uses pip only (standard HTTPS to pypi.org).
- **GPU passthrough:** NVIDIA Container Toolkit gives containers direct GPU access.
- **Persistence:** Training data lives on host filesystem via bind mounts. Container can be destroyed and recreated freely.

---

## 2. Server Prerequisites

Before you begin, the server needs exactly two things:

### 2.1 Docker Engine (20.10+)

```bash
# Check if installed:
docker --version

# If not installed (Ubuntu):
# Ask your admin, or if you have sudo:
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
# Log out and back in for group to take effect
```

### 2.2 NVIDIA Container Toolkit

```bash
# Check if installed:
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# If this shows your GPU info → you're good!
# If it fails with "could not select device driver":

# Install (Ubuntu, requires sudo):
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2.3 Verify

```bash
# This should show your GPU:
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 3. First-Time Setup

```bash
# 1. Clone the repo
cd ~
mkdir -p qte9489  # or wherever you keep projects
cd qte9489
git clone https://github.com/Abhi-0212000/residual-offpolicy-rl.git
cd residual-offpolicy-rl
git checkout dev/abhi-resfit-setup

# 2. Create secrets file
cp .env.example .env
nano .env
```

Fill in your `.env`:

```ini
WANDB_API_KEY=your_actual_wandb_api_key
HF_TOKEN=your_actual_huggingface_token
# Optional:
# CUDA_VISIBLE_DEVICES=0
# WANDB_ENTITY=your_username
```

**How to get the keys:**
- **WANDB_API_KEY:** Go to https://wandb.ai/authorize → copy the key
- **HF_TOKEN:** Go to https://huggingface.co/settings/tokens → create a token with "Read" access

```bash
# 3. Create data directory (for training outputs)
mkdir -p data
```

---

## 4. Building the Docker Image

```bash
# Build (first time takes ~20-30 min, subsequent builds use cache)
docker compose build 2>&1 | tee build.log
```

### What the build does (3 stages):

| Stage | What | Cached? |
|-------|------|---------|
| **base** | Ubuntu 22.04 + CUDA 12.8 + system packages + Python 3.10 | Yes (until base image changes) |
| **deps** | PyTorch, lerobot, robosuite, dexmimicgen, mimicgen, all pip packages | Yes (until dependency files change) |
| **final** | Copies project code into image | Rebuilds when code changes |

### Build verification

The build automatically runs these checks at the end:

```
PyTorch 2.10.0+cu128, CUDA 12.8    ✓
torchrl C++ OK                       ✓
MuJoCo 3.3.2                        ✓
robosuite OK                         ✓
wandb 0.25.0                         ✓
```

If you see all ✓, the image is ready.

### Rebuild after code changes

```bash
git pull
docker compose build   # only rebuilds the "final" stage (seconds, not minutes)
```

---

## 5. Running Training

### 5.1 Interactive Shell

```bash
# Enter the container (--rm = auto-delete when you exit)
docker compose run --rm train bash

# Inside the container, you have everything:
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import resfit; print('OK')"

# Run training scripts:
bash scripts/train_bc_act.sh
bash scripts/train_residual_rl.sh

# Exit when done:
exit
```

### 5.2 Run Training Directly (One-Liner)

```bash
# BC training:
docker compose run --rm train bash scripts/train_bc_act.sh

# Residual RL training:
docker compose run --rm train bash scripts/train_residual_rl.sh

# Any custom command:
docker compose run --rm train python resfit/lerobot/scripts/train_bc_dexmg.py --help
```

### 5.3 Detached Mode (Survives SSH Disconnect)

**This is what you want for long training runs:**

```bash
# Start BC training in background:
docker compose run -d --name qte9489-bc-train train bash scripts/train_bc_act.sh

# Start RL training in background:
docker compose run -d --name qte9489-rl-train train bash scripts/train_residual_rl.sh

# Check logs (live):
docker logs -f qte9489-rl-train

# Check logs (last 100 lines):
docker logs --tail 100 qte9489-rl-train

# Check if still running:
docker ps | grep qte9489

# Stop a run:
docker stop qte9489-rl-train

# Remove stopped container:
docker rm qte9489-rl-train
```

### 5.4 Using tmux/screen (Alternative to Detached Mode)

```bash
# Create a tmux session:
tmux new -s training

# Inside tmux, start the container interactively:
docker compose run --rm train bash
bash scripts/train_residual_rl.sh

# Detach from tmux: Ctrl+B, then D
# Reattach later:
tmux attach -t training
```

---

## 6. Data Persistence & Volumes

### What persists where

| Data | Container Path | Host Path | Type | Survives container removal? |
|------|---------------|-----------|------|-----------------------------|
| **Training runs** (`run_*`, `bc_run_*`) | `/app/data/` | `./data/` | Bind mount | **Yes** |
| **Buffer caches** | `/app/data/offline_buffer_cache/` | `./data/offline_buffer_cache/` | Bind mount | **Yes** |
| **HuggingFace datasets** (~655MB) | `/root/.cache/huggingface/` | Docker named volume `hf-cache` | Named volume | **Yes** (survives `docker compose down`) |
| **WandB cache** | `/root/.local/share/wandb/` | Docker named volume `wandb-cache` | Named volume | **Yes** |
| **Code** | `/app/` | Baked into image | Image layer | Immutable (rebuild to update) |
| **Deps** (lerobot, robosuite, etc.) | `/app/deps/` | Baked into image | Image layer | Immutable |

### How `CACHE_DIR` works

Both training scripts read the `CACHE_DIR` environment variable:

```python
_CACHE_ROOT = Path(os.environ.get("CACHE_DIR", ".")).expanduser().resolve()
```

Docker Compose sets `CACHE_DIR=/app/data`, which is mounted to `./data/` on the host. So:
- `run_*` directories → `./data/run_*/`
- `bc_run_*` directories → `./data/bc_run_*/`
- `offline_buffer_cache/` → `./data/offline_buffer_cache/`
- `online_buffer_cache/` → `./data/online_buffer_cache/`

### Checking your data from the host

```bash
# See all training runs:
ls -la data/

# Check disk usage:
du -sh data/*

# Runs are directly accessible without entering the container
```

### Nuclear cleanup (if you need to free disk)

```bash
# Remove named volumes (HF cache, wandb cache):
docker compose down -v

# Remove training data (CAREFUL — deletes all runs):
rm -rf data/*

# Remove the Docker image:
docker rmi qte9489-resfit:latest
```

---

## 7. VS Code Dev Container (Remote Development)

The repo includes a `.devcontainer/devcontainer.json` for full VS Code integration. This gives you an IDE experience inside the container with IntelliSense, debugging, and all extensions.

### 7.1 Prerequisites

On your **local machine** (laptop):
1. [VS Code](https://code.visualstudio.com/)
2. [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension
3. [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

### 7.2 Connect to Server via SSH

1. Open VS Code → `Ctrl+Shift+P` → **"Remote-SSH: Connect to Host..."**
2. Enter: `ssh qte9489@<server-ip>`
3. Once connected, open the folder: `~/qte9489/residual-offpolicy-rl/`

### 7.3 Reopen in Container

1. VS Code should auto-detect `.devcontainer/devcontainer.json` and show a popup:
   **"Folder contains a Dev Container configuration. Reopen folder in container?"**
2. Click **"Reopen in Container"**
3. Or manually: `Ctrl+Shift+P` → **"Dev Containers: Reopen in Container"**

### 7.4 What you get

- **Python IntelliSense** with Pylance (understands all imports including resfit, torch, robosuite)
- **Integrated terminal** inside the container (no need to `docker exec`)
- **Jupyter notebook** support for experimentation
- **GitLens** for git history
- **Docker extension** for monitoring containers
- Python interpreter auto-set to `/usr/bin/python3.10`

### 7.5 Dev Container Configuration

```jsonc
// .devcontainer/devcontainer.json
{
  "name": "ResFiT Dev",
  "dockerComposeFile": "../docker-compose.yml",   // uses the same compose file
  "service": "train",                              // uses the "train" service
  "workspaceFolder": "/app",                       // opens /app in VS Code
  "customizations": {
    "vscode": {
      "extensions": [                              // auto-installed in container
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "ms-azuretools.vscode-docker",
        "eamodio.gitlens"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python3.10"
      }
    }
  },
  "remoteUser": "root"                             // needed for GPU access
}
```

### 7.6 Running Training from Dev Container

Once inside the Dev Container, use the integrated terminal:

```bash
# The terminal is already inside the container!
bash scripts/train_bc_act.sh
```

Or use VS Code's **Run and Debug** (F5) with a `launch.json` if you want breakpoint debugging.

### 7.7 Troubleshooting Dev Container

| Problem | Solution |
|---------|----------|
| "Cannot connect to Docker" | Make sure Docker is running on the server and your user is in the `docker` group |
| Container won't start | Run `docker compose build` on the server first |
| GPU not available | Check `nvidia-smi` works on the host |
| Extensions not loading | Rebuild container: `Ctrl+Shift+P` → "Dev Containers: Rebuild Container" |

---

## 8. Managing Containers on Shared Servers

### 8.1 Naming Convention

All containers and images are prefixed with your username:
- Image: `qte9489-resfit:latest`
- Container: `qte9489-resfit-train` (auto-set by docker-compose)
- Detached runs: `qte9489-bc-train`, `qte9489-rl-train` (set via `--name`)

This ensures other users know whose containers/images these are.

### 8.2 Being a Good Citizen

```bash
# Check what's running (yours and others):
docker ps

# Check disk usage:
docker system df

# Clean up YOUR stopped containers:
docker container prune --filter "label=com.docker.compose.project=residual-offpolicy-rl"

# NEVER run these on shared servers:
# docker system prune -a     ← would delete OTHER users' images!
# docker volume prune         ← would delete OTHER users' data!
```

### 8.3 Resource Limits

The `docker-compose.yml` includes:
- `shm_size: '16gb'` — shared memory for PyTorch DataLoader
- `network_mode: host` — uses host network (works through VPN)
- GPU access via NVIDIA Container Toolkit

To limit to a specific GPU (if server has multiple):

```bash
# In .env:
CUDA_VISIBLE_DEVICES=0

# Or per-run:
docker compose run --rm -e CUDA_VISIBLE_DEVICES=0 train bash scripts/train_rl.sh
```

---

## 9. Updating Code

### 9.1 After pushing changes from your laptop

```bash
# On the server:
cd ~/qte9489/residual-offpolicy-rl
git pull

# If only Python code changed (fast, ~seconds):
docker compose build

# If dependencies changed (rare, ~minutes):
docker compose build --no-cache
```

### 9.2 After changing dependencies

If you add a new pip package:

1. Add it to the Dockerfile's `pip install` block
2. Push to GitHub
3. On server: `git pull && docker compose build`

### 9.3 Syncing training data to your laptop

```bash
# From your LAPTOP (not the server):
rsync -avz --progress \
    qte9489@<server-ip>:~/qte9489/residual-offpolicy-rl/data/ \
    ~/path/to/local/data/

# Or just specific runs:
rsync -avz --progress \
    qte9489@<server-ip>:~/qte9489/residual-offpolicy-rl/data/run_2026-02-24_*/ \
    ~/path/to/local/data/
```

---

## 10. Environment Variables & Secrets

### `.env` file (gitignored, never committed)

```ini
# Required:
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token

# Optional:
CUDA_VISIBLE_DEVICES=0           # Limit to specific GPU
WANDB_ENTITY=your_wandb_username # Default WandB entity
WANDB_PROJECT=resfit             # Default WandB project
```

### Environment set by Docker Compose

These are set automatically in every container:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONPATH` | `/app` | Makes `import resfit` work |
| `MUJOCO_GL` | `egl` | Headless MuJoCo rendering (no display) |
| `PYOPENGL_PLATFORM` | `egl` | OpenGL via EGL (no X11 needed) |
| `MPLBACKEND` | `Agg` | Non-interactive matplotlib (prevents Tkinter crash) |
| `CACHE_DIR` | `/app/data` | Where all training outputs go |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Prevents CUDA OOM on smaller GPUs |

---

## 11. Training Scripts Reference

All scripts are in `scripts/` with every parameter documented inline.

### `scripts/train_bc_act.sh` — BC Policy Training

Trains a base BC policy from demonstrations using the ACT (Action Chunking Transformer) architecture. **Must run before RL training.**

Key parameters to adjust:

```bash
STEPS=200000          # Training steps. 200K standard
BATCH_SIZE=128        # 128 for 16GB GPU, 256 for 48GB
EVAL_NUM_ENVS=4       # 4 for 16GB, 8-16 for 48GB
EVAL_NUM_EPISODES=100 # 100 for real runs, 20 for quick tests
```

### `scripts/train_residual_rl.sh` — Residual RL Training

Fine-tunes the frozen BC policy with TD3 residual RL. **Requires a completed BC run.**

Key parameters:

```bash
BASE_WANDB_ID="dexmg-bc/zp7niccu"  # ← CHANGE to your BC run's wandb_project/run_id
TOTAL_TIMESTEPS=500000               # 500K standard, 300K for quick experiments
ACTION_SCALE=0.2                     # ±20% max residual correction
UTD=4                                # Updates per env step
BUFFER_SIZE=80000                    # 80K for 16GB RAM, 200K for 48GB
```

### `scripts/resume_residual_rl.sh` — Resume Interrupted RL

Resumes from a checkpoint. **You must fill in 3 values:**

```bash
RESUME_CKPT="run_2026-.../models/latest"  # Path to checkpoint
WANDB_CONTINUE_ID="1wrldnus"               # 8-char WandB run ID
SEED=1987747100                             # Must match original run
```

---

## 12. Troubleshooting

### Build fails with SSL errors

**Cause:** Corporate VPN/proxy intercepting HTTPS.

```bash
# If pip fails (unlikely, pip usually works through VPN):
# Add to Dockerfile before pip install:
ENV PIP_TRUSTED_HOST="pypi.org pypi.python.org files.pythonhosted.org"

# If git clone fails:
git config --global http.sslVerify false  # temporary workaround
```

### `CUDA error: no kernel image is available for execution`

**Cause:** PyTorch was compiled for a different GPU architecture.

```bash
# Check GPU compute capability:
python -c "import torch; print(torch.cuda.get_device_capability())"
# RTX 4090 = (8, 9), RTX PRO 5000 = (10, 0), A100 = (8, 0)

# PyTorch 2.10 + cu128 supports all modern GPUs. If you see this error,
# the CUDA version mismatch is likely the issue.
```

### Container exits immediately

```bash
# Check the logs:
docker logs qte9489-rl-train

# Common cause: .env file missing or malformed
cat .env
```

### "Permission denied" on `./data/`

```bash
# Fix permissions on host:
chmod 777 data/

# Or run container as your user (not root):
docker compose run --rm --user "$(id -u):$(id -g)" train bash
```

### Out of disk space

```bash
# Check Docker disk usage:
docker system df

# Remove unused images (safe):
docker image prune

# Check data directory:
du -sh data/*
```

### WandB login issues

```bash
# Inside container, verify:
echo $WANDB_API_KEY  # should show your key

# Manual login (if .env not working):
wandb login
```

### `shm_size` error / DataLoader crashes

The `docker-compose.yml` sets `shm_size: '16gb'`. If you still see shared memory errors:

```bash
# Check inside container:
df -h /dev/shm

# If it shows less than 16GB, the host may have limits.
# Reduce num_workers in training scripts.
```

---

## 13. Quick Reference / Cheat Sheet

```bash
# ═══════════════════════════════════════════════
# FIRST-TIME SETUP (once per server)
# ═══════════════════════════════════════════════
git clone https://github.com/Abhi-0212000/residual-offpolicy-rl.git
cd residual-offpolicy-rl && git checkout dev/abhi-resfit-setup
cp .env.example .env && nano .env        # fill in keys
mkdir -p data
docker compose build 2>&1 | tee build.log

# ═══════════════════════════════════════════════
# DAILY USAGE
# ═══════════════════════════════════════════════
# Interactive shell:
docker compose run --rm train bash

# Run BC training (detached):
docker compose run -d --name qte9489-bc train bash scripts/train_bc_act.sh

# Run RL training (detached):
docker compose run -d --name qte9489-rl train bash scripts/train_residual_rl.sh

# Check logs:
docker logs -f qte9489-rl

# Check GPU usage from host:
nvidia-smi

# ═══════════════════════════════════════════════
# CODE UPDATES
# ═══════════════════════════════════════════════
git pull && docker compose build

# ═══════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════
docker stop qte9489-bc qte9489-rl    # stop running containers
docker rm qte9489-bc qte9489-rl      # remove stopped containers
# docker compose down -v              # remove volumes too (CAREFUL)

# ═══════════════════════════════════════════════
# DATA ACCESS (from host)
# ═══════════════════════════════════════════════
ls data/                               # list all runs
du -sh data/*                          # check sizes
```

---

## File Reference

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: CUDA 12.8 + Python 3.10 + all dependencies |
| `docker-compose.yml` | GPU access, volumes, environment, networking |
| `.env.example` | Template for secrets (copy to `.env`) |
| `.env` | Your secrets (gitignored, never committed) |
| `.dockerignore` | Keeps Docker build context small |
| `.devcontainer/devcontainer.json` | VS Code Dev Container configuration |
| `docker-run.sh` | Convenience wrapper for common Docker commands |
| `scripts/train_bc_act.sh` | BC policy training script with all params |
| `scripts/train_residual_rl.sh` | Residual RL training script with all params |
| `scripts/resume_residual_rl.sh` | Resume interrupted RL training |

---

## Version History

| Date | Change |
|------|--------|
| 2026-02-24 | Initial Docker setup. No conda (VPN incompatible). System Python 3.10 + pip. |
| 2026-02-24 | Added training scripts, persistent volumes, devcontainer config. |
