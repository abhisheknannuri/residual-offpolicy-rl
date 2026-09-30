# Workspace audit + cleanup plan

Updated 2026-09-30. **Status: §5.1-§5.4 are DONE and committed locally.
NOTHING HAS BEEN PUSHED.** Two commits sit on `dev/abhi-resfit-setup` ahead of
`origin`:

```
5c1cbd0  Add trossen_real: the real-robot stack                  (89 files, 1.2 MB)
a829378  Ignore vendored clones and recorded data; pin dependencies
```

Remaining: §5.5 docs move, §6-§7 the uv environment work.

Safety net in place: branch `backup/pre-cleanup-20260930`, tarball
`~/workspace_code_backup_20260930.tar.gz` (41 MB), and every deleted file copied
to `~/resfit_bak_archive/root_junk_20260930/`.

---

## 1. Git - settled, no action needed

```
origin    https://github.com/abhisheknannuri/residual-offpolicy-rl.git   <- yours, push here
upstream  https://github.com/amazon-far/residual-offpolicy-rl.git       <- Amazon's, never push
```

You forked `amazon-far`; that URL is correct and current. `origin` already carries
your new username and `git config user.email` is unchanged, so **pushes just work
and your 36 existing commits stay attributed to you**. No history rewrite - the
old `Abhi-0212000` author name in past commits is cosmetic and GitHub links by
email anyway.

Push target for the new work: `origin dev/abhi-resfit-setup` (already tracking).

---

## 2. Why the working tree is 66 GB

| size | path | in `.gitignore`? |
|---|---|---|
| 14 GB | `online_buffer_cache/` | yes |
| 11 GB | `.venv/` | yes |
| 11 GB | `offline_buffer_cache/` | yes |
| 7.0 GB | `trossen_real/datasets/` | yes |
| 6.1 + 1.9 + 1.5 GB | three `run_2026-08-*/` dirs | yes |
| 4.6 GB | `outputs/` | yes |
| 2.1 GB | `TD3_BC/` (**includes its own 2 GB `.venv`**) | **no** |
| 1.7 GB | `deps/` | yes |
| 1.4 GB | `wandb/` | yes |
| 1.0 GB | `infer_videos/` | **no** |
| 911 MB | `openpi/` (includes its own 385 MB `.git`) | **no** |
| 783 MB | `artifacts/` | yes |
| 711 MB | `infer_dataset/` | **no** |
| 559 MB | `.git/` | - |
| 417 MB | `logs/` | yes |
| 156 MB | `trossen_real/EVAL/` | **no** |
| 111 MB | `hil-serl/` | **no** |
| 96 MB | `RealTrainLogs/` | **no** |
| 24 MB | `infer_logs/` | **no** |

Short version: **63 of the 66 GB is caches, checkpoints, datasets and venvs** -
all regenerable, none of it belongs in git. Only ~1.1 GB is currently at risk of
being committed, and §5.2 removes that.

---

## 3. The five side-clones - **one of them is NOT just context**

You asked me to double-check they are unused. Result:

| clone | pinned commit | referenced by resfit? | verdict |
|---|---|---|---|
| `openpi` | - | **prose only** (2 comment lines in `convert_to_delta_joint_dataset.py`) | gitignore |
| `hil-serl` | - | **prose only** (comments in `trossen_real/config.py`) | gitignore |
| `lerobot_trossen` | - | **prose only** (comments + one yaml comment) | gitignore |
| `TD3_BC` | - | **nothing** (only refers to itself) | gitignore |
| **`agentlace`** | `76984f9` | **IMPORTED AT RUNTIME** | **must be recorded** |

> ### `agentlace` is a real dependency
> `resfit/rl_finetuning/off_policy/distributed/transport.py` does
> `from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer`,
> and `_ensure_agentlace_on_path()` deliberately adds `<repo>/agentlace` to
> `sys.path`. It is **not installed in either venv** - the distributed trainer
> only works because the clone happens to sit in the repo root.
>
> So: gitignore the *folder* (it has its own `.git`), but it must go into
> `DEPENDENCIES.md` and the clone script alongside `deps/*`, pinned at `76984f9`,
> or the distributed training path breaks on any fresh checkout.

Verified: no `import` of `openpi`, `hil_serl`, `serl_launcher`, `lerobot_trossen`
or `TD3_BC` anywhere outside those clones, and none of them is installed in either
environment. Safe to ignore.

---

## 4. `trossen_real/` is not in git at all

64 `.py` + 17 `.md`, about **4 MB of code**: infer app, eval mode, teleop,
calibration, follower/leader servers, dataset recorder, inference client. Zero
version history. The 7 GB attributed to this directory is `datasets/` (already
ignored) plus `EVAL/` (156 MB, going into `.gitignore` per your instruction).

This is the biggest risk in the workspace and the cheapest thing on this list to
fix. It gets committed and pushed to `origin dev/abhi-resfit-setup` in §5.4.

---

## 5. Cleanup

### 5.1 Junk at the repo root - **DONE**

Every file below was copied to `~/resfit_bak_archive/root_junk_20260930/` before
deletion. Nothing was lost.

| file | what it was |
|---|---|
| `ts` | captured `help()` output for `set_cartesian_positions` |
| `hutil`, `t_client()` | captured `less` help screens |
| `sen_arm, pydoc` | captured `pydoc` output for `TrossenArmDriver.configure` |
| `t.mock import MagicMock` | captured `less` help screen |
| `tatus --short teleop_client` | captured `less` help screen |
| `output.log` | a stray July run log |
| `.gitignore.20260901_113118.bak` | superseded |
| `camera_*.png` (5), `q_traj_50000.png` | stray debug images |
| `info.json`, `temp_notes.md` | stray scratch files |

> The last three in that list had **spaces in their real names** - `sen_arm, pydoc`,
> not `sen_arm,`. An earlier pass "deleted" the names without the spaces, which
> silently did nothing, and reading them appeared to show empty files because the
> files did not exist. Found by `git status --porcelain -z`, which prints the raw
> name. Worth remembering: git quotes odd filenames in normal output.

`scratch.py`, `inspect_lerobot.py` and `test_omegaconf.py` moved to `sandbox/`
(tracked) for review another day. Verified: nothing imports them.

### 5.2 `.gitignore` - **DONE** (commit `a829378`)

Measured effect: what `git add -A` would stage went from **3853 files / 1.10 GB**
to **252 files / 0.02 GB**. The rules applied:

```gitignore
# ── Side-clones kept as reading material for coding agents. Each has its own
#    .git, so committing them would create broken gitlinks. NOTE: agentlace is
#    NOT just reading material - it is imported by
#    resfit/rl_finetuning/off_policy/distributed/transport.py. It is ignored here
#    because it is a foreign clone, and pinned in DEPENDENCIES.md instead.
/openpi/
/hil-serl/
/lerobot_trossen/
/TD3_BC/
/agentlace/

# ── Recorded robot data and logs (large, machine-specific, regenerable)
/infer_videos/
/infer_dataset/
/infer_logs/
/RealTrainLogs/
/eval_runs/
/eval_runs_OLD_*/
trossen_real/EVAL/
/videos/
/videos_can/
/videos_can_action_scale02/

# ── Nested virtualenvs (TD3_BC ships a 2 GB one)
**/.venv/
**/venv/

# ── Timestamped backups: keep them on disk, keep them OUT of git
*.bak

# ── Agent / editor state
.claude/

# ── Scratch
/scratch/
/scratch.py
/temp.py
/temp_notes.md
```

**`.bak` files stay exactly where they are** - 78 of them, ignored by git, never
deleted. They are the only history for files that were never committed.

Check before committing:

```sh
git status --porcelain --untracked-files=all | grep -c '^??'   # want ~200, not 3853
```

> One trade-off to be aware of, since you asked for `trossen_real/EVAL/` to be
> ignored wholesale: that also excludes `docs/archive/Notes.md`, `placement_mat.pdf` and the
> pose configs - the *conclusions* of the eval campaign, not just its artifacts.
> If you want those in git, the alternative is
> `trossen_real/EVAL/*/*/runs/` + `trossen_real/EVAL/*/*/analysis/` (ignores the
> 156 MB, keeps ~200 KB of notes and figures). **Say which you prefer** - I have
> written the wholesale version above, as instructed.

### 5.3 Record the vendored dependencies - **DONE** (commit `a829378`)

`deps/` is gitignored and is not submodules, so **a fresh clone cannot build the
environment**. Create a tracked `DEPENDENCIES.md` + `scripts/clone_deps.sh`:

| dep | upstream | commit |
|---|---|---|
| robosuite | `ARISE-Initiative/robosuite` | `77a47512` |
| mimicgen | `NVlabs/mimicgen` | `72bd767` |
| dexmimicgen | `NVlabs/dexmimicgen` | `e606f36` |
| lerobot | `huggingface/lerobot` | `69901b9b` |
| **agentlace** | `youliangtan/agentlace` | `76984f9` |

### 5.4 Commit `trossen_real/` - **COMMITTED, NOT PUSHED** (commit `5c1cbd0`)

After 5.2, so datasets/EVAL are already ignored:

```sh
git add trossen_real/
git status --short | head -60        # expect ~80 files / ~4 MB - READ THIS LIST
git commit -m "Add trossen_real: real-robot stack (infer app, eval mode, teleop, calibration)"
git push origin dev/abhi-resfit-setup
```

Then the 19 modified `resfit/` files, in small thematic commits rather than one blob.

### 5.5 Documentation layout - **NOT STARTED**, do this LAST

31 `.md` files at the root, **22 of them cross-referenced by other files**, so
this is a move *plus* a link rewrite, not a plain `mv`. Proposed:

```
README.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md     stay at root (GitHub convention)
docs/overview/    docs/overview/ResFi-ResidualFine-tuningWithOff-PolicyRL.md
docs/algorithms/  TD3_ALGORITHM  RESIDUAL_LEARNING  CRITIC_LOSSES_EXPLAINED
                  NSTEP_RETURNS  VALUE_FUNCTION_AND_WARMUP  ORIGINAL_VS_IBRL_QAGENT
                  LAYERNORM_VS_GRADCLIP  ACTION_NORMALIZATION  REPLAY_BUFFERS
docs/policies/    ACT_ARCHITECTURE  ACT_TRANSFORMER  VISION_BACKBONE  BC_POLICY_TRAINING
docs/training/    RESIDUAL_RL_TRAINING  TD3_TRAINING_END_TO_END  CHECKPOINTING_AND_RESUME
                  EXPERIMENT_GUIDE  WANDB_BEST_PRACTICES
docs/rewards/     REWARD_AND_SUCCESS  REWARD_MODEL_INTEGRATION  STAGE_AWARE_REWARD_PIPELINE
docs/data/        DATASET_GUIDE
docs/setup/       UV_SERVER_SETUP  SETUP_FIXES  DOCKER_SETUP
docs/real/        REAL_RESIDUAL_RL_HUMAN_INTERVENTION_PLAN
docs/archive/     docs/archive/Notes.md  temp_notes.md
```

`trossen_real/*.md` stay beside their code - docs that live next to what they
describe get updated; docs in a distant folder rot.

This needs a script with a dry-run and a link-checker that fails loudly rather
than a blind `sed`. **Not written yet** - say the word.

---

## 6. The environment - the real problem

### 6.1 What is actually true today

* You use **uv**. Conda came from Amazon and is blocked at BMW, so it is not a
  long-term option. `.venv` is therefore the one that has to work.
* Both environments exist and **both currently work**, with an identical torch
  stack, but they differ: `av` **15.1.0 vs 17.1.0**, `cv2` **4.11.0 vs 5.0.0**.
* In a fresh login shell, `VIRTUAL_ENV` points at the conda env, so bare `python`
  is **conda's** - worth knowing even if you always activate `.venv` manually.
* **`ffmpeg` on `PATH` comes from conda** (`miniforge3/envs/residual/bin/ffmpeg`).
  The setup doc requires system ffmpeg or imageio hangs downloading its own. If
  conda goes away, this breaks. Record it as a system prerequisite.
* Toolchain is right for the torchrl build: `gcc` defaults to **11.4.0**, and the
  installed `_torchrl*.so` needs only `GLIBCXX_3.4.21`, well under the runtime's.

### 6.2 Why `uv` breaks it right now

The env was never built from `pyproject.toml`. `scripts/setup_uv_env.sh` does four
things `pyproject.toml` does not currently express:

1. torch from the **pytorch cu128 index**, not PyPI;
2. **torchrl 0.9.2 built from git source** with gcc-11 (the PyPI wheel has an ABI
   mismatch against torch 2.11);
3. four **editable deps** from `deps/`;
4. `--overrides uv-overrides.txt` to defeat hard pins (`gymnasium==0.29.1`, etc.),
   and `--no-deps` for dexmimicgen's stale `numpy==1.23.3`.

`uv sync` / `uv run` sync the venv to `pyproject.toml` + `uv.lock`, and none of the
four survive. `uv.lock` is from **10 July** and describes an environment that no
longer exists. So "never run uv" is currently a rule enforced by memory - which is
exactly what you want to stop relying on.

### 6.3 The fix: make `uv sync` the whole install (no workarounds)

**All four mechanisms exist in your installed uv 0.11.16. I tested them today**
in a scratch project - `uv lock` resolved cleanly and the lock file recorded both
the override and the editable path source:

| need | pyproject key | tested |
|---|---|---|
| torch from cu128 index | `[[tool.uv.index]] name=... url=... explicit=true` + `[tool.uv.sources] torch={index=...}` | accepted |
| torchrl from git | `[tool.uv.sources] torchrl={git=..., rev="v0.9.2"}` | accepted |
| build torchrl against installed torch | `[tool.uv] no-build-isolation-package=["torchrl"]` | accepted |
| beat the deps' hard pins | `[tool.uv] override-dependencies=["gymnasium==1.1.1", ...]` | **verified working** - lock recorded `overrides = [{ name = "gymnasium", specifier = "==1.1.1" }]` and resolved 1.1.1 over the dep's `==0.29.1` |
| editable vendored deps | `[tool.uv.sources] robosuite={path="deps/robosuite", editable=true}` | **verified** - lock recorded `source = { editable = "deps/robosuite" }` |
| dexmimicgen's stale `numpy==1.23.3` | same `override-dependencies` with `numpy==2.2.6` - **replaces `--no-deps`**, which is a blunt instrument | to prove |

Target end state - one command for the whole team:

```sh
git clone https://github.com/abhisheknannuri/residual-offpolicy-rl.git
cd residual-offpolicy-rl
bash scripts/clone_deps.sh      # deps/ + agentlace at pinned commits
uv sync                         # <- everything, from the committed uv.lock
```

**Two known gaps** that `pyproject.toml` cannot express, and how to handle them:

1. **`CC=gcc-11 CXX=g++-11` for the torchrl build.** Not expressible. Handle with
   a thin `scripts/setup.sh` that exports them and calls `uv sync`, or a `.env`.
   Default `gcc` here is already 11.4, so it works unset on *this* machine - not
   on a machine with gcc-14.
2. **Build ordering.** `no-build-isolation-package = ["torchrl"]` means torchrl
   compiles against packages already in the env, so **torch must be installed
   before torchrl builds**. Whether `uv sync` guarantees that in one pass is the
   single biggest unknown. If it does not, the fallback is a two-step
   `uv sync --no-install-package torchrl && uv sync` - still declarative, still
   one script, no memorised rules.

**This is exactly what the dummy environment is for.** Do not touch the working
`.venv` to find out.

### 6.4 Interim guard (optional, 2 lines, reversible)

Until 6.3 is proven, this makes the current footgun impossible:

```toml
[tool.uv]
managed = false
```

Tested against uv 0.11.16: `uv sync` is **refused** (`error: The project is marked
as unmanaged`), and `uv run python ...` uses `.venv/bin/python` **without syncing
or uninstalling anything**. It turns a rule-you-must-remember into an error
message. It is *removed* as part of 6.3 once `uv sync` is the real installer.

---

## 7. The dummy environment - where the fixing actually happens

Goal: reproduce the current environment in a throwaway folder, **fix it properly
there**, prove it, and only then apply to the real workspace.

```
/home/qte9489/personal_abhi/temp/resfit-env-test/
```

Disk: **130 GB free**, this needs ~15 GB.

### Step 1 - snapshot what "correct" means, from the working env

Do this **first**. It is the target the dummy env must hit.

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
.venv/bin/python -m pip freeze --all | sort > ~/env_target_uv_20260930.txt
wc -l ~/env_target_uv_20260930.txt        # ~395 packages
```

### Step 2 - clone the code, locally and offline

```sh
mkdir -p /home/qte9489/personal_abhi/temp/resfit-env-test
cd /home/qte9489/personal_abhi/temp/resfit-env-test
git clone --branch dev/abhi-resfit-setup \
    /home/qte9489/personal_abhi/temp/residual-offpolicy-rl repo
cd repo
```

Cloning from the **local path** tests what is actually committed, with no push and
no dependency on GitHub being reachable from BMW.

### Step 3 - recreate `deps/` and `agentlace` at the pinned commits

```sh
mkdir -p deps
git clone https://github.com/ARISE-Initiative/robosuite deps/robosuite  && git -C deps/robosuite    checkout 77a47512
git clone https://github.com/NVlabs/mimicgen            deps/mimicgen   && git -C deps/mimicgen     checkout 72bd767
git clone https://github.com/NVlabs/dexmimicgen         deps/dexmimicgen && git -C deps/dexmimicgen checkout e606f36
git clone https://github.com/huggingface/lerobot        deps/lerobot    && git -C deps/lerobot      checkout 69901b9b
git clone https://github.com/youliangtan/agentlace      agentlace       && git -C agentlace         checkout 76984f9
```

(This becomes `scripts/clone_deps.sh` in §5.3.)

### Step 4 - baseline: prove the OLD path still works here

Before changing anything, run the existing installer in the clean copy. If it
fails, that is itself the finding.

```sh
env -u VIRTUAL_ENV bash scripts/setup_uv_env.sh     # expect: "resfit uv env OK"
.venv/bin/python -m pip freeze --all | sort > /tmp/env_fresh_oldpath.txt
diff -u ~/env_target_uv_20260930.txt /tmp/env_fresh_oldpath.txt | head -60
```

`env -u VIRTUAL_ENV` matters - the conda `VIRTUAL_ENV` in your shell otherwise
confuses uv about which environment it is targeting.

**Expect drift.** The working `.venv` picked up `av 17.1.0` / `cv2 5.0.0` from
unconstrained transitive resolution months ago; a build today resolves whatever
PyPI serves now. The diff is the deliverable, not a failure - every line of it is
a version that should become an explicit pin.

### Step 5 - build the proper declarative setup, in the dummy only

1. Fold `constraints-uv.txt` + `uv-overrides.txt` + the `setup_uv_env.sh` logic
   into `pyproject.toml` using the keys in 6.3.
2. Add the drifted versions from step 4 as explicit pins, so the env is
   reproducible rather than accidental.
3. Delete the stale `uv.lock`, regenerate with `uv lock`.
4. `rm -rf .venv && uv sync` **from scratch**. Repeat until it is one command.
5. Resolve the torchrl build-ordering question (6.3, gap 2) empirically.

### Step 6 - acceptance test (all must pass in the dummy before touching the real workspace)

```sh
cd /home/qte9489/personal_abhi/temp/resfit-env-test/repo

# 1. versions match the working env
.venv/bin/python - <<'PY'
import importlib
want={"torch":"2.11.0+cu128","torchvision":"0.26.0+cu128","torchrl":"0.9.2",
      "tensordict":"0.9.1","numpy":"2.2.6","gymnasium":"1.1.1","mujoco":"3.3.2",
      "transformers":"4.48.0","datasets":"3.6.0"}
bad=[]
for m,v in want.items():
    got=getattr(importlib.import_module(m),"__version__","?")
    ok=got.startswith(v); print(f"{'OK ' if ok else 'BAD'} {m:<14}{got}")
    if not ok: bad.append(m)
raise SystemExit(1 if bad else 0)
PY

# 2. compiled extensions bind (this is what a bad torchrl build breaks)
.venv/bin/python -c "
import torchvision.ops
from torchrl.data import ReplayBuffer, LazyTensorStorage
import torch; print('C++ ext OK, cuda', torch.cuda.is_available())"

# 3. every test suite
.venv/bin/python -m pytest trossen_real/inference/tests/ trossen_real/infer_app/tests/ \
                          trossen_real/calibration/tests/ -q

# 4. the distributed path, which needs agentlace on sys.path
.venv/bin/python -c "
from resfit.rl_finetuning.off_policy.distributed import transport
transport._ensure_agentlace_on_path(); import agentlace; print('agentlace', agentlace.__file__)"

# 5. video encode/decode - the libraries that differ between your two envs
.venv/bin/python -c "import av, cv2; print('av', av.__version__, 'cv2', cv2.__version__)"

# 6. uv must no longer be able to wreck it
uv sync            # must be a clean no-op, not a reinstall
.venv/bin/python -c "import torchrl; print('torchrl survived uv sync:', torchrl.__version__)"
```

Test 6 is the one that closes the whole problem: **`uv sync` must be safe to run
twice.** When that passes, the "never use uv" rule can be deleted from the docs
instead of being remembered.

### Step 7 - only then, apply to the real workspace

With a proven `pyproject.toml` + `uv.lock`, the real workspace gets the new files
and a `uv sync`. Keep the old `.venv` renamed (`mv .venv .venv.old`) until a full
real-robot session has run on the new one, then delete it.

---

## 7b. Dummy-environment results (measured 2026-09-30)

Built at `/home/qte9489/personal_abhi/temp/resfit-env-test/repo` from a real
`git clone` of GitHub at `b183a15`, then `clone_deps.sh`, then
`setup_uv_env.sh`. All of the following is measured, not predicted.

### The good news: the installer reproduces the environment

| step | result |
| --- | --- |
| `git clone` from GitHub | 13 MB, 468 files, 1.2 s |
| `bash scripts/clone_deps.sh` | all 5 deps at their pinned commits, 66 s |
| `bash scripts/setup_uv_env.sh` | `resfit uv env OK`, torchrl compiled, `cuda=True` |
| 112 tests | pass, same as the working workspace |
| agentlace / distributed import | OK |

**Every pinned package matched the working environment exactly** - torch,
torchvision, torchaudio, torchcodec, tensordict, numpy, gymnasium, mujoco,
transformers, datasets, and also `av 17.1.0` and `opencv-python 5.0.0.93`, which
I had expected to drift. torchrl even resolved to the identical git commit
`0dc98d5`. The constraints and overrides do their job.

### Gap 1 - the real-robot dependencies are not installed at all

`pyrealsense2` (2.56.5.9235) and `trossen-arm` (1.10.0) are in the working
environment but **nothing installs them** - not `pyproject.toml`, not
`setup_uv_env.sh`. A teammate following the runbook gets an environment that
cannot talk to a camera or an arm.

It fails quietly, which is worse than failing loudly: `camera_manager.py` and
`arm_driver.py` both wrap the import in `try/except ImportError` and set
`HAS_REALSENSE` / `HAS_TROSSEN_SDK`, so every module still imports and all 112
tests still pass. The failure only surfaces when hardware is actually used.

**Fix:** add both to `pyproject.toml` under an optional `real` extra (they are
useless on a training server with no hardware), and have the runbook install it
for station machines.

### Gap 2 - 47 unpinned transitive packages drifted

None of them broke anything, but they are unpinned by accident rather than by
decision. The ones worth a look: `numba 0.66.0 -> 0.68.0`,
`llvmlite 0.48.0 -> 0.50.0`, `pyarrow 24.0.0 -> 25.0.1`,
`rerun-sdk 0.35.0 -> 0.38.1`, `mink 1.2.0 -> 1.3.0`,
`accelerate 1.14.0 -> 1.15.0`, and `packaging 26.2 -> 25.0` (a *downgrade*).

**Fix:** once the environment is trusted, freeze it into the lock and let `uv
sync` enforce it.

### Gap 3 - the working environment has CUDA cruft the fresh one does not

The working `.venv` carries **4 stray `nvidia-*-cu13` packages** alongside the 19
`cu12` ones; the fresh build has only the 19 `cu12`. Left over from a past torch
upgrade. Harmless, but it means the working env is *dirtier* than a fresh build -
a point in favour of switching to a rebuilt one.

### The footgun, proven

I ran `uv sync` in the disposable copy. One command:

| | before | after |
| --- | --- | --- |
| torch | 2.11.0+cu128 | **2.13.0+cu130** |
| torchvision | installed | **uninstalled** |
| torchrl | 0.9.2 (source build) | **uninstalled** |
| tensordict | installed | **uninstalled** |
| robosuite | installed | **uninstalled** |
| lerobot | installed | **uninstalled** |
| pytest | installed | **uninstalled** |

This is no longer an argument from documentation. It is what happens.

### The guard, proven

Adding to `pyproject.toml`:

```toml
[tool.uv]
managed = false
```

| command | result |
| --- | --- |
| `uv sync` | `error: The project is marked as unmanaged` |
| `uv run python ...` | uses `.venv/bin/python3`, syncs nothing |
| `bash scripts/setup_uv_env.sh` | still works - full rebuild from scratch passed |

Rebuilt from scratch with the guard in place and re-ran everything: all pinned
versions match, C++ extensions bind, 112 tests pass, agentlace imports, `uv sync`
is refused, and the environment is untouched by the attempt.

### What to do next

1. **Apply the guard to the real workspace now** - two lines, tested, reversible,
   and it removes the "never type uv" rule that is currently enforced by memory.
2. **Add `pyrealsense2` + `trossen-arm` as a `real` extra** - Gap 1 is the one
   that would actually bite a teammate.
3. Then the bigger job from section 6.3: fold `constraints-uv.txt`,
   `uv-overrides.txt` and the installer's logic into `pyproject.toml` so
   `uv sync` becomes the installer rather than something to defend against. The
   mechanisms are verified; the open question stays build ordering for the
   torchrl source build.

---

## 7c. The env is now fixed properly - `uv sync` IS the installer

The recommendation in 7b ("keep the .sh script, add a guard") was a half-measure
and was rejected, correctly. The right question was *why does the shell script
exist at all*. Answer: it does not have to. It was written pip-style, and every
one of its four jobs is expressible in `pyproject.toml`. Built and verified in
the test env on 2026-09-30.

### What replaced what

| the script did | `pyproject.toml` now says |
| --- | --- |
| `uv pip install --index-url .../cu128 torch torchvision torchaudio torchcodec` | `[[tool.uv.index]] name="pytorch-cu128" explicit=true` + `[tool.uv.sources]` for those four |
| `CC=gcc-11 uv pip install --no-build-isolation --no-deps "torchrl @ git+...@v0.9.2"` | `[tool.uv.sources] torchrl = { git=..., rev="v0.9.2" }` + `no-build-isolation-package = ["torchrl"]` |
| `uv pip install -e deps/robosuite -e deps/mimicgen -e deps/lerobot -e .` | `[tool.uv.sources]` `{ path="deps/...", editable=true }` |
| `-c constraints-uv.txt --overrides uv-overrides.txt` | `override-dependencies = [gymnasium, datasets, mujoco, numpy]` |
| `--no-deps` for dexmimicgen's stale `numpy==1.23.3` | the `numpy==2.2.6` override - narrower, keeps its real deps |
| (nothing - this was simply missing) | `[project.optional-dependencies] real = [pyrealsense2, trossen-arm]` |

### Measured results

| test | result |
| --- | --- |
| `uv lock` | 179 packages resolved in 2.2 s |
| `uv sync` from an empty `.venv` | **2 m 18 s**, torchrl compiled from git |
| all 13 pinned versions | match the working environment **exactly** |
| torchvision nms + torchrl C++ extension | bind, `cuda=True` |
| editable installs | resolve to `deps/*` |
| 112 tests | pass |
| agentlace / distributed import | OK |
| **`uv sync` a second time** | **`Checked 171 packages` in 0.034 s - clean no-op** |
| `uv sync --extra real` | installs pyrealsense2 2.56.5.9235 + trossen-arm 1.10.0; `HAS_REALSENSE=True`, `HAS_TROSSEN_SDK=True` |
| `uv run python ...` | uses `.venv/bin/python3` |
| wipe `.venv`, `uv sync` again | **173 packages, byte-for-byte identical to the first build** |

The build-ordering worry from 6.3 did not materialise: uv installs torch before
compiling torchrl against it, in a single `uv sync`.

### Consequences

* **`scripts/setup_uv_env.sh` can be deleted.** So can `constraints-uv.txt` and
  `uv-overrides.txt` - their content now lives in `pyproject.toml`.
* **`[tool.uv] managed = false` is NOT needed.** It was a guard against `uv sync`
  being wrong. `uv sync` is now the correct command, so there is nothing to guard
  against and nothing for anyone to remember.
* **The 51 unpinned transitive packages are now pinned by `uv.lock`.** They still
  differ from the current working `.venv` - but that env's versions were
  accidental, and the lock makes them deterministic from here on.
* Setup for the whole team becomes:

```sh
git clone https://github.com/abhisheknannuri/residual-offpolicy-rl.git
cd residual-offpolicy-rl
bash scripts/clone_deps.sh      # the 5 vendored repos at pinned commits
uv sync                         # everything else; add --extra real on a station
```

### The conda dependency is gone

Two prerequisites were listed above as "belongs in the README". Both were
examined; neither is a conda dependency, and one was a bug.

**gcc is not a conda thing.** `gcc`/`g++`/`cc` all resolve to `/usr/bin/...`
(dpkg package `gcc-11`, version 11.4.0). It is a system build tool, already
present, needed only to compile torchrl. The only real note is that a machine
whose default gcc is much newer produces a `.so` needing a CXXABI the runtime may
lack, in which case `CC=gcc-11 CXX=g++-11 uv sync`. That is a README line, not a
conda install.

**ffmpeg was a bug, not a prerequisite.** There are three sources on this
machine, none requiring conda:

| source | version | needs anything installed? |
| --- | --- | --- |
| `imageio-ffmpeg`'s bundled static binary | **7.0.2** | no - ships in the wheel, `uv sync` installs it, `uv.lock` pins it |
| PyAV's own linked FFmpeg libraries (557 codecs) | - | no - needs no external binary at all |
| `/usr/bin/ffmpeg` | 4.4.2 | system package, already present |

Conda's copy was simply **shadowing** `/usr/bin/ffmpeg` on PATH.

The reason it became load-bearing: `train_rollout_recorder.py` and
`evaluate_dexmg.py` both did

```python
if not os.environ.get("IMAGEIO_FFMPEG_EXE"):
    _sys_ffmpeg = shutil.which("ffmpeg")      # <- picks up conda's
```

justified by a comment saying imageio-ffmpeg's bundled binary "can fail to launch
in fresh envs and then block `get_ffmpeg_exe()` on a network download".
**Measured: `get_ffmpeg_exe()` returns a path inside the venv in 2.3 ms and
downloads nothing** - imageio-ffmpeg 0.6.0 vendors the binary. So the workaround
was guarding against something that does not happen here, and in doing so made a
conda install a prerequisite of a conda-free project. It also chose the system's
ffmpeg 4.4.2 over the bundled 7.0.2.

Fixed by `resfit/rl_finetuning/utils/ffmpeg_setup.py`, which inverts the order:
an explicit `IMAGEIO_FFMPEG_EXE` wins, then the bundled binary, then PATH. Both
call sites now use it.

Verified with conda **stripped entirely from PATH** (`env -i`, no `VIRTUAL_ENV`):
torch/torchrl/CUDA fine, PyAV fine, 112 tests pass, and both video paths write
real files - the infer-app recorder through PyAV and the training-rollout
recorder through imageio using the bundled ffmpeg 7.0.2.

### Still to decide before applying to the real workspace

The proposed files are saved as `pyproject.proposed.toml` and `uv.lock.proposed`
at the repo root. Applying them means replacing `pyproject.toml` / `uv.lock`,
deleting `scripts/setup_uv_env.sh`, `constraints-uv.txt` and
`uv-overrides.txt`, and rebuilding this workspace's `.venv` - so it is worth
doing deliberately, with `mv .venv .venv.old` kept until a real robot session has
run on the new one.

---

## 8. Suggested order

| # | step | risk | reversible |
|---|---|---|---|
| 1 | backup tar + `git branch backup/pre-cleanup-20260930` | none | - |
| 2 | §5.2 `.gitignore` | none - no files touched | yes |
| 3 | §5.3 `DEPENDENCIES.md` + `clone_deps.sh` | none | yes |
| 4 | §5.4 commit + push `trossen_real/` | none | `git reset` |
| 5 | §7 dummy env, steps 1-6 | none - separate folder | `rm -rf` |
| 6 | §7 step 7 apply to real workspace | medium | keep `.venv.old` |
| 7 | §5.5 docs move + link rewrite | medium - needs the script | `git reset --hard` |

Backup before anything:

```sh
cd /home/qte9489/personal_abhi/temp/residual-offpolicy-rl
tar --exclude='.venv' --exclude='deps' --exclude='*.pyc' --exclude='__pycache__' \
    --exclude='trossen_real/datasets' --exclude='infer_videos' --exclude='infer_dataset' \
    --exclude='online_buffer_cache' --exclude='offline_buffer_cache' --exclude='outputs' \
    --exclude='wandb' --exclude='TD3_BC' --exclude='openpi' --exclude='hil-serl' \
    -czf ~/workspace_code_backup_$(date +%Y%m%d).tar.gz .
git branch backup/pre-cleanup-$(date +%Y%m%d)
```

**Nothing in steps 1-5 pushes** except §5.4, which pushes only to your own fork.
