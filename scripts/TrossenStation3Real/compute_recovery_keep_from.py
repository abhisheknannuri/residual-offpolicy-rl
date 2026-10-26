#!/usr/bin/env python3
"""Print the --keep-from argument for convert_to_delta_joint_dataset.py that
keeps only the RECOVERY part of deliberately-failed recovery episodes.

A recovery episode is a teleop demo of: approach, grasp, deliberately
misaligned insert, release (cube unseated on the plate), recovery. The cut is
MARGIN frames after the first "failed release": the first frame where the
gripper opens (observation.state[gripper] crosses OPEN_THRESHOLD upward)
while the episode is in the "cube being inserted" stage, before the final
"cube inserted" segment. Stages come from a stage-labelled copy of the
episodes (LeRobot SARM schema, v2.1 meta/episodes.jsonl), e.g. the SARM
dataset labelled in the visualizer.

Episodes without a failed release are left out (printed to stderr).

Usage (repo venv; prints "ep:frame,ep:frame,..." on stdout):
    .venv/bin/python scripts/TrossenStation3Real/compute_recovery_keep_from.py \\
        --labelled trossen_real/datasets/iterativeTraining/TrossenStation3/SARM/20261015_PickCubeAndInsert_Station3_iter1_plus_recovery_SARM_v21 \\
        --first 48 --count 51
--first/--count select the labelled episodes that are the recovery dataset's
episodes 0..count-1 (in the SARM dataset they follow the 48 iter1 episodes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_THRESHOLD = 0.02  # m; closed on the cube ~0.011, open ~0.03-0.04


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", required=True, type=Path, help="Stage-labelled v2.1 dataset.")
    ap.add_argument("--first", type=int, required=True, help="Labelled episode index of recovery episode 0.")
    ap.add_argument("--count", type=int, required=True, help="Number of recovery episodes.")
    ap.add_argument("--margin", type=int, default=10, help="Frames after the failed release (default 10 = 0.5 s).")
    ap.add_argument("--inserting", default="cube being inserted")
    ap.add_argument("--inserted", default="cube inserted")
    ap.add_argument("--gripper-index", type=int, default=6)
    args = ap.parse_args()

    eps = {}
    for line in (args.labelled / "meta" / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            eps[int(e["episode_index"])] = e

    out, skipped = [], []
    for k in range(args.count):
        i = args.first + k
        e = eps[i]
        names, starts = e["sparse_subtask_names"], e["sparse_subtask_start_frames"]
        if args.inserting not in names or args.inserted not in names:
            skipped.append(k)
            continue
        first_inserting = starts[names.index(args.inserting)]
        final_inserted = starts[len(names) - 1 - names[::-1].index(args.inserted)]
        df = pd.read_parquet(args.labelled / f"data/chunk-000/episode_{i:06d}.parquet", columns=["observation.state"])
        g = np.stack(df["observation.state"].to_numpy())[:, args.gripper_index]
        opens = [t for t in range(max(first_inserting, 1), final_inserted - 5)
                 if g[t] > OPEN_THRESHOLD and g[t - 1] <= OPEN_THRESHOLD]
        if not opens:
            skipped.append(k)
            continue
        out.append(f"{k}:{opens[0] + args.margin}")

    print(",".join(out))
    print(f"{len(out)} episodes kept, {len(skipped)} without a failed release: {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
