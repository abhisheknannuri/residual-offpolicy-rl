# Vendored dependencies

Five repositories this project needs but does not contain. They are all
`.gitignore`d (each has its own `.git`; committing them would create gitlinks
nobody can fetch), so **a fresh clone cannot build or run until they are
restored**. Restore them with:

```sh
bash scripts/clone_deps.sh
```

| what | where it goes | upstream | pinned commit |
| --- | --- | --- | --- |
| robosuite | `deps/robosuite` | https://github.com/ARISE-Initiative/robosuite | `77a4751233c29456a5381209e30dd0dbf39a6557` |
| mimicgen | `deps/mimicgen` | https://github.com/NVlabs/mimicgen | `72bd767c255545f462e7ccfb2731f2e5d4c1d9bb` |
| dexmimicgen | `deps/dexmimicgen` | https://github.com/NVlabs/dexmimicgen | `e606f36a38b1d4ba8f56d06d6c0cd059b20ebbaf` |
| lerobot | `deps/lerobot` | https://github.com/huggingface/lerobot | `69901b9b6a2300914ca3de0ea14b6fa6e0203bd4` |
| **agentlace** | `agentlace/` (repo root) | https://github.com/youliangtan/agentlace | `76984f9` |

The first four are installed editable by `scripts/setup_uv_env.sh`
(`uv pip install -e deps/...`). Three of their four SHAs are confirmed by the
original setup scripts (`resfit/lerobot/setup_lerobot.sh`,
`resfit/dexmg/setup_dexmg.sh`); see the caveat on mimicgen below.

## agentlace is different - it is NOT installed

`agentlace` is **imported at runtime and never pip-installed into the venv**.
`resfit/rl_finetuning/off_policy/distributed/transport.py` calls
`_ensure_agentlace_on_path()`, which adds `<repo>/agentlace` to `sys.path`. The
distributed actor/learner training path therefore works only because that clone
sits in the repo root. Without it:

```
ImportError: agentlace is not installed and no clone was found at <repo>/agentlace
```

It also lives one level deeper than it looks - the package is at
`agentlace/agentlace/`. With the repo root on `sys.path`, a bare
`import agentlace` "succeeds" by picking up the outer *folder* as an empty
namespace package and then every submodule import fails; `_ensure_agentlace_on_path()`
detects that (a real package has `__file__`, a namespace package does not) and
fixes it. Do not "simplify" that function away.

## Caveat: mimicgen was never pinned upstream

`resfit/dexmg/setup_dexmg.sh` clones mimicgen and does `git checkout main`, so the
commit you get depends on the day you ran it. `72bd767c...` is what this machine
happens to have, and it is what the working environment was built against - hence
pinning it here. If a fresh install behaves differently, mimicgen's drift is the
first thing to check.

## Not dependencies

These clones sit in the repo root purely as reading material for coding agents.
Verified 2026-09-30: nothing outside them imports them, and none of them is
installed in either environment. They are `.gitignore`d and need no restoring.

| clone | only referenced by |
| --- | --- |
| `openpi/` | two comment lines in `trossen_real/scripts/convert_to_delta_joint_dataset.py` |
| `hil-serl/` | comments in `trossen_real/config.py` |
| `lerobot_trossen/` | comments in `trossen_real/config.py` and one station yaml |
| `TD3_BC/` | nothing at all (also ships its own 2 GB `.venv`) |

## Related: the two dependency-installation paths

There are **two**, and they disagree. Know which one you are using.

| | `scripts/setup_uv_env.sh` | `resfit/rl_finetuning/setup_rlpd_robosuite.sh` |
| --- | --- | --- |
| package manager | `uv` into `.venv` | `pip` into a conda env |
| torchrl | **0.9.2, built from git source** | `torchrl==0.8.0` from PyPI |
| tensordict | 0.9.1 | 0.8.2 |
| torchcodec | 0.11.0 | 0.4.0 |
| ffmpeg | expects system ffmpeg on PATH | `micromamba install -n residual ffmpeg` |
| status | **current** | superseded, kept for provenance |

The conda path is Amazon's original. It is where the `ffmpeg` currently on this
machine's PATH came from (`miniforge3/envs/residual/bin/ffmpeg`) - worth knowing,
because the uv setup assumes a *system* ffmpeg and imageio hangs downloading its
own if there is none. See `WORKSPACE_CLEANUP_PLAN.md` §6.
