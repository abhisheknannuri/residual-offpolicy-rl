# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Compare control-loop rate across inference runs - the one number that says
whether a latency change worked.

Reads the per-tick JSONL logs the infer app writes
(`logs/infer_<station>_<timestamp>.jsonl`), so it needs no robot and no server.
Pass several and it prints one row each, oldest first, which is exactly how you
read an A/B/C test:

    .venv/bin/python -m trossen_real.scripts.infer_latency_report logs/infer_*.jsonl

`achieved_hz` is logged per tick by the loop itself. `p10` matters as much as
the median: with chunked fetching the loop runs fast and then pauses once per
chunk for the round trip, so a high median with a low p10 is the signature of
"chunking is working, but the round trip is still expensive".
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize(path: Path) -> dict | None:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    hz = [r["achieved_hz"] for r in rows if r.get("achieved_hz")]
    if not hz:
        return None
    # The first tick's rate is measured against loop start, not a previous tick,
    # so it is meaningless - drop it rather than letting it drag the minimum.
    hz = hz[1:] or hz
    return {
        "name": path.name,
        "ticks": len(rows),
        "median": st.median(hz),
        "mean": st.fmean(hz),
        "p10": _pct(hz, 0.10),
        "p90": _pct(hz, 0.90),
        "min": min(hz),
        "intervened": sum(1 for r in rows if r.get("intervened")),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="per-tick JSONL logs (logs/infer_*.jsonl)")
    ap.add_argument("--target-hz", type=float, default=20.0)
    args = ap.parse_args(argv)

    paths = sorted({Path(p) for p in args.logs}, key=lambda p: p.name)
    rows = [s for s in (summarize(p) for p in paths if p.exists()) if s]
    if not rows:
        print("no ticks with achieved_hz found in those logs")
        return 1

    print(f"{'log':<52}{'ticks':>6}{'median':>8}{'mean':>7}{'p10':>7}{'p90':>7}{'min':>7}"
          f"{'% of ' + str(int(args.target_hz)) + 'Hz':>10}")
    for s in rows:
        print(f"{s['name']:<52}{s['ticks']:>6}{s['median']:>8.1f}{s['mean']:>7.1f}"
              f"{s['p10']:>7.1f}{s['p90']:>7.1f}{s['min']:>7.1f}"
              f"{100 * s['median'] / args.target_hz:>9.0f}%")
    if len(rows) > 1:
        first, last = rows[0], rows[-1]
        print(f"\n{first['name']} -> {last['name']}: "
              f"median {first['median']:.1f} Hz -> {last['median']:.1f} Hz "
              f"({last['median'] / max(first['median'], 1e-9):.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
