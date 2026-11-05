#!/usr/bin/env python3
"""Local web app to choose the still-frame thresholds.

Python loads the LeRobot v2.1 dataset once (one parquet per episode) and does
all the work with still_rule.py; the page (still_explorer.html, no data in it)
only asks for what it shows:
  GET  /api/meta      joints, fps, default rule, episode list
  GET  /api/hist      per joint speed histograms: all frames vs the 3 s before
                      each gripper open (alignment + insertion, the precise part)
  POST /api/totals    rule -> frames still / dropped, per episode counts
  POST /api/episode   rule + episode -> positions, speeds, gripper, per-frame code

Speed of a joint = how much its measured position (observation.state, rad)
changes from one frame to the next: |q[t+1] - q[t]| in rad/frame (x20 = rad/s).
Gripper labels for the protected windows: gripper_controller.binarize on
action[:, 6].

Reads only; nothing is written.

    .venv/bin/python temp/dataset_prep/still_explorer.py \\
        --dataset trossen_real/datasets/iterativeTraining/TrossenStation3/ACT_Iter1/PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint_old \\
        --viz-url http://localhost:7860/local/PickCubeAndInsert_TS3_Iter1_withRecovery_deltajoint_binGrip
then open http://127.0.0.1:7871
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from flask import Flask, jsonify, request, send_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: trossen_real is not an installed package
from trossen_real.inference.gripper_controller import binarize  # noqa: E402
from still_rule import StillRule, classify, speeds  # noqa: E402

JOINTS = ["j0 base", "j1 shoulder", "j2 elbow", "j3 wrist pitch", "j4 wrist yaw", "j5 wrist roll", "gripper"]
PRE_OPEN_S = 3.0
BINS = np.logspace(-6, -0.5, 56)  # rad/frame (m/frame for the gripper)


class Dataset:
    def __init__(self, root: Path):
        info = json.loads((root / "meta" / "info.json").read_text())
        if not str(info.get("codebase_version", "")).startswith("v2"):
            raise SystemExit(f"expects a LeRobot v2.x dataset (per-episode files), got {info.get('codebase_version')}")
        chunk = int(info.get("chunks_size") or 1000)
        self.root, self.fps = root, int(info["fps"])
        pre = int(PRE_OPEN_S * self.fps)
        self.eps = []
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            ep = int(json.loads(line)["episode_index"])
            t = pq.read_table(root / info["data_path"].format(episode_chunk=ep // chunk, episode_index=ep),
                              columns=["action", "observation.state"])
            a = np.asarray(t.column("action").to_pylist(), dtype=np.float64)
            q = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float64)
            lab = np.asarray(binarize(a[:, 6]))
            sw = (np.flatnonzero(np.diff(lab) != 0) + 1).tolist()
            opens = [s for s in sw if lab[s] > lab[s - 1]]
            in_pre = np.zeros(len(q), bool)
            for s in opens:
                in_pre[max(0, s - pre): s] = True
            self.eps.append(dict(ep=ep, q=q, sp=speeds(q), lab=lab, sw=sw, in_pre=in_pre))
        sp = np.vstack([e["sp"] for e in self.eps])
        pre_mask = np.concatenate([e["in_pre"] for e in self.eps])
        x = np.clip(sp, BINS[0], BINS[-1])
        self.h_all = [np.histogram(x[:, j], BINS)[0].tolist() for j in range(7)]
        self.h_pre = [np.histogram(x[pre_mask, j], BINS)[0].tolist() for j in range(7)]


def rule_from(body) -> StillRule:
    r = StillRule()
    for f in fields(StillRule):
        if f.name in (body or {}):
            v = body[f.name]
            setattr(r, f.name, [float(x) for x in v] if f.name == "joint_thr" else
                    float(v) if f.name == "grip_thr" else max(0, int(v)))
    if len(r.joint_thr) != 6:
        raise ValueError("joint_thr needs 6 values")
    return r


def make_app(ds: Dataset, viz_url: str) -> Flask:
    app = Flask(__name__)

    @app.errorhandler(ValueError)
    def bad_rule(exc):
        return jsonify(error=str(exc)), 400

    @app.get("/")
    def index():
        return send_file(HERE / "still_explorer.html")

    @app.get("/api/meta")
    def meta():
        return jsonify(joints=JOINTS, fps=ds.fps, pre_open_s=PRE_OPEN_S, rule=asdict(StillRule()),
                       dataset=str(ds.root), viz=viz_url, episodes=[e["ep"] for e in ds.eps])

    @app.get("/api/hist")
    def hist():
        return jsonify(bins=BINS.tolist(), h_all=ds.h_all, h_pre=ds.h_pre)

    @app.post("/api/totals")
    def totals():
        rule = rule_from(request.get_json())
        rows, tot = [], np.zeros(5, int)
        pre_frames = drop_pre = 0
        for i, e in enumerate(ds.eps):
            c = classify(e["sp"], e["sw"], rule)
            n = np.bincount(c, minlength=5)
            dp = int(((c == 3) & e["in_pre"]).sum())
            tot += n
            pre_frames += int(e["in_pre"].sum())
            drop_pre += dp
            rows.append(dict(i=i, ep=e["ep"], T=len(c), d=int(n[3]), dp=dp))
        return jsonify(frames=int(tot.sum()), still=int(tot[1:].sum()), dropped=int(tot[3]), protected=int(tot[4]),
                       pre_frames=pre_frames, dropped_pre=drop_pre, episodes=rows)

    @app.post("/api/episode")
    def episode():
        body = request.get_json()
        e = ds.eps[int(body["i"])]
        c = classify(e["sp"], e["sw"], rule_from(body.get("rule")))
        return jsonify(ep=e["ep"], q=e["q"].T.tolist(), sp=e["sp"].T.tolist(), lab=e["lab"].tolist(),
                       code=c.tolist(), in_pre=e["in_pre"].astype(int).tolist())

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--viz-url", default="", help="LeRobot visualizer base URL of the same episodes (optional).")
    ap.add_argument("--port", type=int, default=7871)
    args = ap.parse_args()
    ds = Dataset(args.dataset.resolve())
    print(f"{len(ds.eps)} episodes loaded; open http://127.0.0.1:{args.port}")
    make_app(ds, args.viz_url).run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
