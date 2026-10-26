#!/usr/bin/env python3
"""Per-episode cut points where a SARM stage starts, as the --keep-from
argument of trossen_real.scripts.convert_to_delta_joint_dataset.

Input: the per-frame predictions parquet written by opensarm's
tools/sarm_label_client.py (--predictions), with columns episode_index,
frame_index, raw_stage, confidence.

Cut = the first frame of the first run of --run consecutive frames whose
raw_stage == --stage with confidence > --min-conf (a hysteresis, so a single
false prediction cannot set the cut). Everything from the cut to the episode's
end is kept. Episodes with no such run are left out and listed on stderr.

Usage (repo venv):
    .venv/bin/python scripts/TrossenStation3Real/RL/compute_sarm_stage_cut.py \\
        --predictions /path/iter1_sarm_predictions.parquet --stage 2 > keep_from.txt
Prints "ep:frame,ep:frame,..." on stdout; --report writes a JSON with the
details (cut frame, run length, confidence, frames kept per episode).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def first_run(stage: np.ndarray, conf: np.ndarray, target: int, min_conf: float, run: int) -> int | None:
    ok = (stage == target) & (conf > min_conf)
    n = 0
    for t, v in enumerate(ok):
        n = n + 1 if v else 0
        if n == run:
            return t - run + 1
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--stage", type=int, default=2, help="Stage index the kept part starts at (2 = cube above the hole).")
    ap.add_argument("--min-conf", type=float, default=0.85)
    ap.add_argument("--run", type=int, default=5, help="Consecutive frames required.")
    ap.add_argument("--episodes", default=None, help="Comma-separated episode indices (default: all in the file).")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.predictions)
    wanted = None if args.episodes is None else {int(e) for e in args.episodes.split(",")}
    out, rows, missing = [], [], []
    for ep, g in df.groupby("episode_index", sort=True):
        if wanted is not None and ep not in wanted:
            continue
        g = g.sort_values("frame_index")
        stage, conf = g["raw_stage"].to_numpy(), g["confidence"].to_numpy()
        cut = first_run(stage, conf, args.stage, args.min_conf, args.run)
        if cut is None:
            missing.append(int(ep))
            continue
        out.append(f"{int(ep)}:{cut}")
        rows.append({"episode": int(ep), "cut_frame": cut, "length": len(g), "kept_frames": len(g) - cut,
                     "conf_at_cut": float(conf[cut])})
    if wanted is not None:
        missing += sorted(wanted - set(df["episode_index"].unique()))
    print(",".join(out))
    kept = sum(r["kept_frames"] for r in rows)
    total = sum(r["length"] for r in rows)
    print(f"{len(rows)} episodes cut at stage {args.stage} ({args.run} frames > {args.min_conf}); "
          f"{kept}/{total} frames kept; no detection: {missing}", file=sys.stderr)
    if args.report:
        args.report.write_text(json.dumps({"settings": {k: str(v) for k, v in vars(args).items()},
                                           "episodes": rows, "no_detection": missing}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
