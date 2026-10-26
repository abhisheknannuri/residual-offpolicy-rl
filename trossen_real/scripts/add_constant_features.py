# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Add constant-valued columns to a LeRobot v2.1 dataset, IN PLACE, so its
schema matches another dataset before merging.

`merge_v21_datasets.py` fills a column that one input lacks with zeros. That is
wrong when the right value is not zero, e.g. `observation.intervened` for a
fully teleoperated dataset (every frame was a human). This writes the column
with an explicit value instead. The feature spec (dtype, shape, names) is
copied from `--like`, the dataset it will be merged with.

Writes: the column in every episode parquet, `meta/info.json` features,
`meta/episodes_stats.jsonl` and `meta/stats.json` (if present).

Usage (repo venv):
    .venv/bin/python -m trossen_real.scripts.add_constant_features \\
        --dataset <recovery_delta> --like <iter1_delta> \\
        --set observation.intervened=1 --set observation.policy_action=0
A value is broadcast to the feature's shape; bools take 1/0/true/false.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _value(spec: dict, text: str) -> np.ndarray:
    dtype = spec["dtype"]
    if dtype == "bool":
        v = text.strip().lower() in ("1", "true", "yes")
        return np.full(tuple(spec["shape"]), v, dtype=bool)
    return np.full(tuple(spec["shape"]), float(text), dtype=np.dtype(dtype))


def _stats(v: np.ndarray, n: int) -> dict:
    x = v.astype(np.float64)
    return {"min": x.tolist(), "max": x.tolist(), "mean": x.tolist(), "std": np.zeros_like(x).tolist(), "count": [n]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="v2.1 dataset to modify in place.")
    ap.add_argument("--like", required=True, type=Path, help="v2.1 dataset whose feature specs to copy.")
    ap.add_argument("--set", action="append", required=True, metavar="FEATURE=VALUE")
    args = ap.parse_args()

    info_path = args.dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    like = json.loads((args.like / "meta" / "info.json").read_text())["features"]
    values = {}
    for item in args.set:
        name, text = item.split("=", 1)
        if name not in like:
            raise SystemExit(f"{name} is not a feature of --like {args.like}")
        if name in info["features"]:
            raise SystemExit(f"{name} already exists in {args.dataset}")
        values[name] = _value(like[name], text)

    # Parquet type of each new column, exactly as --like stores it (e.g. a 7-dim
    # feature is fixed_size_list<float>[7], not list<float>), so all files of
    # the merged dataset share one schema.
    like_info = json.loads((args.like / "meta" / "info.json").read_text())
    like_parquet = next((args.like / "data").rglob("*.parquet"))
    like_schema = pq.read_schema(like_parquet)
    del like_info

    chunk = int(info.get("chunks_size") or 1000)
    stats_path = args.dataset / "meta" / "episodes_stats.jsonl"
    stats = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    for e in stats:
        ep = int(e["episode_index"])
        p = args.dataset / info["data_path"].format(episode_chunk=ep // chunk, episode_index=ep)
        table = pq.read_table(p)  # existing columns keep their exact parquet types
        n = table.num_rows
        for name, v in values.items():
            # lerobot stores shape-(1,) features as a plain scalar column (see
            # get_hf_features_from_features), larger shapes as one list per row.
            col = [v.item()] * n if v.shape == (1,) else [v.tolist()] * n
            table = table.append_column(name, pa.array(col, type=like_schema.field(name).type))
            e["stats"][name] = _stats(v, n)
        pq.write_table(table, p)
    stats_path.write_text("".join(json.dumps(e) + "\n" for e in stats))

    gpath = args.dataset / "meta" / "stats.json"
    if gpath.exists():
        g = json.loads(gpath.read_text())
        total = sum(int(e["stats"][next(iter(values))]["count"][0]) for e in stats)
        for name, v in values.items():
            g[name] = _stats(v, total)
        gpath.write_text(json.dumps(g, indent=4))

    for name in values:
        info["features"][name] = like[name]
    info_path.write_text(json.dumps(info, indent=4))
    print(f"added {sorted(values)} to {len(stats)} episodes of {args.dataset}")


if __name__ == "__main__":
    main()
