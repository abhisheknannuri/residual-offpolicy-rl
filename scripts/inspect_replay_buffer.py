#!/usr/bin/env python3
"""
inspect_replay_buffer.py

Inspect an ONLINE or OFFLINE replay buffer cache (memmap format, written by
torchrl's TensorDictPrioritizedReplayBuffer + optimized_replay_buffer_dumps()
in train_residual_td3.py) - prints the full key/dtype/shape schema, verifies
MultiStepTransform was applied at insertion time, and reports reward/done/
intervention/action-magnitude statistics.

Works IDENTICALLY for both trossen_real/.../offline_buffer_cache/<hash>/ and
.../online_buffer_cache/<hash>/ - the on-disk format is the same either way
(same TensorDict schema); only the SEMANTICS of what's inside differ (offline:
static dataset labels; online: live pedal-driven signals) - see
scripts/TrossenStation1Real/BUFFER_POPULATION_ARCHITECTURE.md for the full
architecture reference this tool was built to verify.

IMPORTANT gotcha, handled automatically here: the raw memmap files are
physically sized to the storage's ALLOCATED capacity (e.g. algo.buffer_size),
not the number of transitions actually written. This script always reads the
true count from writer/metadata.json's "cursor" field and slices every
tensor to [:cursor] before computing any statistic - reading the raw,
unsliced storage will include uninitialized garbage in the tail (confirmed:
offline_buffer_cache/b7bfa8d3 has a physical length of 35144 but a cursor of
35142 - the last 2 slots contain nonsense gamma/steps_to_next_obs values).

Usage:
    python scripts/inspect_replay_buffer.py <cache_dir>
    python scripts/inspect_replay_buffer.py <cache_dir> --full-schema
    python scripts/inspect_replay_buffer.py <cache_dir> --around 9500
    python scripts/inspect_replay_buffer.py <cache_dir> --action-scale 0.15,0.15,0.15,0.05,0.05,0.05,0.1

(Renamed/generalized from the original offline-only scripts/temp/inspect_offline_buffer.py.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tensordict import TensorDict


def load_valid(cache_dir: str) -> tuple[TensorDict, int, int]:
    """Load the storage TensorDict and slice it to the actually-written range.

    Returns (valid_td, cursor, raw_len). `raw_len` is the physical memmap
    length (the storage's allocated capacity); `cursor` is the number of
    transitions actually written (from writer/metadata.json). Always use
    `valid_td`, never index the raw td past `cursor` - see module docstring.
    """
    cache_path = Path(cache_dir)
    storage_dir = cache_path / "storage"
    if not storage_dir.exists():
        raise FileNotFoundError(f"No storage/ directory in {cache_path}")

    td = TensorDict.load_memmap(str(storage_dir))
    raw_len = td.batch_size[0]

    writer_meta_path = cache_path / "writer" / "metadata.json"
    if writer_meta_path.exists():
        cursor = json.load(open(writer_meta_path))["cursor"]
    else:
        cursor = raw_len

    if cursor < raw_len:
        print(
            f"NOTE: raw storage length={raw_len} but writer cursor={cursor} - "
            f"the last {raw_len - cursor} slot(s) are UNINITIALIZED (never written). "
            f"Slicing to [:{cursor}]."
        )
    return td[:cursor], cursor, raw_len


def print_metadata(cache_dir: str) -> None:
    cache_path = Path(cache_dir)
    for name in (
        "buffer_metadata.json",
        "user_metadata.json",
        "storage/storage_metadata.json",
        "sampler/sampler_metadata.json",
        "writer/metadata.json",
    ):
        p = cache_path / name
        if p.exists():
            print(f"\n--- {name} ---")
            print(json.dumps(json.load(open(p)), indent=2))


def print_schema(td: TensorDict, prefix: str = "") -> None:
    """Recursively print every leaf key's dtype/shape - the full transition schema."""
    for key in sorted(td.keys()):
        val = td[key]
        if hasattr(val, "keys"):
            print(f"  {prefix}{key}/")
            print_schema(val, prefix=prefix + "  ")
        elif isinstance(val, torch.Tensor):
            print(f"  {prefix}{key}: shape={tuple(val.shape)}, dtype={val.dtype}")
        else:
            print(f"  {prefix}{key}: {type(val)}")


def verify_nstep_transform(td: TensorDict) -> None:
    print(f"\n{'=' * 70}\nN-STEP TRANSFORM VERIFICATION (MultiStepTransform, rb_transforms.py)\n{'=' * 70}")
    has_all = True
    for key in ("gamma", "nonterminal", "steps_to_next_obs"):
        present = key in td.keys()
        has_all = has_all and present
        if present:
            u = td[key].unique()
            shown = u[:15].tolist()
            more = "" if len(u) <= 15 else f" (+{len(u) - 15} more)"
            print(f"  '{key}': shape={tuple(td[key].shape)}, dtype={td[key].dtype}, unique={shown}{more}")
        else:
            print(f"  '{key}': NOT FOUND")
    has_orig = "next" in td.keys() and "original_reward" in td["next"].keys()
    print(f"  'next.original_reward' present: {has_orig}")
    if has_all and has_orig:
        print("  -> MultiStepTransform WAS applied at insertion time (pre-computed n-step transitions,")
        print("     not computed lazily at sample time). 'next.reward' is the n-step SUM; ")
        print("     'next.original_reward' is the raw 1-step reward before that.")
    else:
        print("  -> one or more n-step keys missing - transform may not have run for this buffer.")


def reward_done_stats(td: TensorDict) -> None:
    print(f"\n{'=' * 70}\nREWARD / DONE / TERMINATED / INTERVENED STATISTICS\n{'=' * 70}")
    n = td.batch_size[0]
    reward = td["next", "reward"]
    orig = td["next", "original_reward"] if "original_reward" in td["next"].keys() else None
    done = td["next", "done"]
    terminated = td["next", "terminated"] if "terminated" in td["next"].keys() else None

    print(f"  total valid transitions: {n}")
    u = reward.unique()
    print(f"  next.reward (n-step summed): nonzero={int((reward.abs() > 1e-8).sum())}/{n}, "
          f"unique(<=20)={[round(v, 6) for v in u[:20].tolist()]}")
    if orig is not None:
        uo = orig.unique()
        print(f"  next.original_reward (raw 1-step): nonzero={int((orig.abs() > 1e-8).sum())}/{n}, "
              f"unique={[round(v, 6) for v in uo.tolist()]}")
    print(f"  next.done: True count = {int(done.sum())}/{n}")
    if terminated is not None:
        print(f"  next.terminated: True count = {int(terminated.sum())}/{n}")
        mismatch = int((done != terminated).sum())
        print(f"  done != terminated mismatches: {mismatch}/{n}"
              f"{' (expected 0 whenever a buffer_done override is in play)' if mismatch == 0 else '  <-- investigate'}")
    if "intervened" in td.keys():
        interv = td["intervened"]
        n_interv = int(interv.sum())
        print(f"  intervened: True count = {n_interv}/{n}")
        if n_interv > 0:
            idx = torch.where(interv)[0]
            print(f"    first 10 intervened indices: {idx[:10].tolist()}")


def action_scale_stats(td: TensorDict, action_scale: list[float]) -> None:
    """Compares the buffer's IMPLIED RESIDUAL (stored 'action' minus
    'obs.observation.base_action') against a given per-dimension
    action_scale - use this to check whether intervention-derived actions
    imply a residual larger than the residual actor's own output range
    (expected/tolerated by design - see INTERVENTION_CLIPPING_AND_SYNC.md,
    NOT a bug).

    IMPORTANT: compares the RESIDUAL, not the raw stored 'action' - the
    stored 'action' is the FULL combined action (base + residual), so
    comparing IT directly against action_scale would almost always "exceed"
    trivially (the base policy's own action is not itself bounded to
    action_scale) and tells you nothing about the residual specifically.
    """
    if "action" not in td.keys():
        print("\n(no top-level 'action' key found - skipping action-scale comparison)")
        return
    if "obs" not in td.keys() or "observation.base_action" not in td["obs"].keys():
        print("\n(no 'obs.observation.base_action' key found - cannot compute implied residual, skipping)")
        return
    action = td["action"]
    base_action = td["obs", "observation.base_action"]
    residual = action - base_action
    print(f"\n{'=' * 70}\nIMPLIED RESIDUAL (action - base_action) vs. action_scale={action_scale}\n{'=' * 70}")
    print(f"  action shape={tuple(action.shape)}, dtype={action.dtype}")
    print(f"  residual per-dim min: {[round(v, 4) for v in residual.min(0).values.tolist()]}")
    print(f"  residual per-dim max: {[round(v, 4) for v in residual.max(0).values.tolist()]}")
    scale_t = torch.tensor(action_scale, dtype=action.dtype)
    exceeds = (residual.abs() > scale_t).any(dim=-1)
    n_exceed = int(exceeds.sum())
    print(f"  transitions with |implied residual| exceeding action_scale in >=1 dim: {n_exceed}/{len(action)}")
    if "intervened" in td.keys():
        interv = td["intervened"]
        n_both = int((exceeds & interv).sum())
        n_auto = int((exceeds & ~interv).sum())
        n_interv_total = int(interv.sum())
        n_auto_total = len(action) - n_interv_total
        print(f"    of which intervened=True: {n_both}/{n_interv_total} "
              f"(expected/tolerated for intervention per design, see doc)")
        print(f"    of which intervened=False (autonomous): {n_auto}/{n_auto_total} "
              f"(should be near-zero - the actor's OWN mean output is architecturally bounded "
              f"to action_scale via Tanh; small positive counts here can come from the target-policy-"
              f"smoothing noise added during training, see INTERVENTION_CLIPPING_AND_SYNC.md §4 point 2)")


def show_window(td: TensorDict, center: int, before: int = 8, after: int = 3) -> None:
    n = td.batch_size[0]
    lo, hi = max(0, center - before), min(n, center + after)
    print(f"\n{'=' * 70}\nTRANSITIONS {lo}..{hi - 1} (centered on index {center})\n{'=' * 70}")
    for i in range(lo, hi):
        row = [f"idx={i:6d}", f"reward={td['next', 'reward'][i].item():.6f}"]
        if "original_reward" in td["next"].keys():
            row.append(f"orig_r={td['next', 'original_reward'][i].item():.4f}")
        row.append(f"done={bool(td['next', 'done'][i])}")
        if "terminated" in td["next"].keys():
            row.append(f"terminated={bool(td['next', 'terminated'][i])}")
        if "nonterminal" in td.keys():
            row.append(f"nonterminal={bool(td['nonterminal'][i])}")
        if "gamma" in td.keys():
            row.append(f"gamma={td['gamma'][i].item():.6f}")
        if "steps_to_next_obs" in td.keys():
            row.append(f"steps={int(td['steps_to_next_obs'][i])}")
        if "intervened" in td.keys():
            row.append(f"interv={bool(td['intervened'][i])}")
        print("  " + "  ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir", help="path to an offline_buffer_cache/<hash>/ or online_buffer_cache/<hash>/ directory")
    ap.add_argument("--full-schema", action="store_true", help="print every leaf key's dtype/shape")
    ap.add_argument("--around", type=int, default=None, help="show transitions centered on this index instead of the first nonzero reward")
    ap.add_argument("--action-scale", type=str, default=None,
                     help="comma-separated per-dim action_scale to compare stored actions against, "
                          "e.g. 0.15,0.15,0.15,0.05,0.05,0.05,0.1 (match agent.actor.action_scale from the .sh)")
    args = ap.parse_args()

    print_metadata(args.cache_dir)
    td, cursor, raw_len = load_valid(args.cache_dir)
    print(f"\nLoaded {cursor} valid transitions (raw storage length {raw_len}).")

    if args.full_schema:
        print(f"\n{'=' * 70}\nFULL KEY SCHEMA\n{'=' * 70}")
        print_schema(td)

    verify_nstep_transform(td)
    reward_done_stats(td)

    if args.action_scale:
        scale = [float(x) for x in args.action_scale.split(",")]
        action_scale_stats(td, scale)

    if args.around is not None:
        show_window(td, args.around)
    else:
        reward = td["next", "reward"]
        nz = torch.where(reward.abs() > 1e-8)[0]
        if len(nz) > 0:
            show_window(td, nz[0].item())
        else:
            print("\n(no nonzero next.reward found in this buffer - nothing to show around)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()
