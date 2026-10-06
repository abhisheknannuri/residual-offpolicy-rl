# Distributed (actor/learner) ResFiT — test scaffolding

Design doc: [`../RESFIT_DISTRIBUTED_MIGRATION.md`](../RESFIT_DISTRIBUTED_MIGRATION.md).
HIL-SERL reverse-engineering: [`../../../hil-serl/ARCHITECTURE.md`](../../../hil-serl/ARCHITECTURE.md).

**Nothing in your original code was modified.** Every file here is new, and the
`train_residual_rl_*.sh` copies are *generated* from their parents in `../` so the
hyper-parameters cannot be mistyped.

## Files

| File | What it is |
|---|---|
| `train_residual_rl_real.sh` | generated copy of `../train_residual_rl_real.sh` |
| `train_residual_rl_real_withOffRL.sh` | generated copy |
| `train_residual_rl_real_continue_from_criticwarmup.sh` | generated copy |
| `train_residual_rl_real_continue_from_offlineRL.sh` | generated copy |
| `train_residual_rl_offline_pretrain.sh` | generated copy, **forced `ROLE=learner`** (touches no env) |
| `common_dist.sh` | role parsing, IP/ports, sync cadence, `PYTHON_BIN` |
| `_regenerate.py` / `_regenerate.sh` | rebuild the copies after editing an original |
| `_verify_diff.sh` | show exactly how each copy differs from its original |

## One script, both roles — on purpose

```bash
# GPU server
./train_residual_rl_real.sh learner

# laptop
LEARNER_IP=10.0.0.5 ./train_residual_rl_real.sh actor

# unchanged single-process behaviour (parity baseline)
./train_residual_rl_real.sh single
```

Both nodes run **the same file**, so they cannot drift apart on any
hyper-parameter. A silent actor/learner config mismatch (different `action_scale`,
`n_step`, normalization…) raises no error — it just makes the actor act in one
action scale while the critic was trained in another. Do not split these into
separate actor/learner scripts.

## Over SSH

Keep `LEARNER_IP=localhost` and forward both ports:

```bash
ssh -N -L 5588:localhost:5588 -L 5589:localhost:5589 user@gpu-server
```

Same trick applies to your ACT/DP `policy_server.py` if you ever move it — but
you're running it on the laptop, so it stays a plain localhost call.

## Interpreter

`PYTHON_BIN` defaults to bare `python` (matching the originals). **Never `uv run`** —
point it at an interpreter directly:

```bash
PYTHON_BIN=/path/to/.venv/bin/python ./train_residual_rl_real.sh learner
```

## Verifying the copies

```bash
./_verify_diff.sh
```

Expect exactly five kinds of change per file and nothing else:
banner, `common_dist.sh` source block, entrypoint → `train_residual_td3_distributed.py`,
`CONFIG_NAME` → `..._dist_config`, and the `dist.*` args in `CMD`.
Already confirmed clean — every hyper-parameter line is byte-identical.

## Status — implemented and tested

| Piece | Where |
|---|---|
| Entrypoint (**hand-maintained** — edit it directly) | `resfit/rl_finetuning/scripts/train_residual_td3_distributed.py` |
| n-step stream, outbox/inbox, weight sync, transport, comms | `resfit/rl_finetuning/off_policy/distributed/*.py` |
| Unit tests (13) | `.../distributed/tests/test_distributed_parity.py` |
| End-to-end localhost test (19 checks) | `.../distributed/tests/e2e_localhost.py` |

`role=single` calls your original `train_residual_td3.main()` directly — same code, not a copy.
`role=learner` / `role=actor` contain ~900 lines that were originally copied from that trainer.

> **The entrypoint is no longer generated.** It used to be built from a template
> by `build_entrypoint.py`, which spliced regions out of `train_residual_td3.py`
> by line number and anchor-checked them. That broke: `train_residual_td3.py`
> drifted, splice 305-326 stopped matching its anchor, `--check` failed, and the
> file was being hand-edited anyway while still carrying a "DO NOT EDIT" header.
> The template and builder have been removed. **Edit the entrypoint directly.**
>
> The `verbatim from train_residual_td3.py:A-B` comments are kept as
> provenance — they say where a region came from, which is useful when
> comparing behaviour against the single-process trainer. Nothing checks them
> and the line numbers drift, so read them as "this came from there", not as
> "these lines are identical today".

Run the tests (no uv):

```bash
PYTHONPATH=. .venv/bin/python resfit/rl_finetuning/off_policy/distributed/tests/test_distributed_parity.py
PYTHONPATH=. .venv/bin/python resfit/rl_finetuning/off_policy/distributed/tests/e2e_localhost.py
```

The e2e test runs the real generated entrypoint as two processes over ZMQ with only the
robot env and the dataset faked; outputs go to a `/tmp/resfit_dist_e2e_*` dir, never the repo.

## Install

Nothing to install. agentlace is imported straight from the clone at `<repo>/agentlace`,
and the wire codec defaults to stdlib `zlib` (`dist.codec`), so the missing `lz4` in
`.venv` doesn't matter. `pyzmq` is already present. Both nodes must use the same
`dist.codec` — agentlace refuses to connect otherwise.
