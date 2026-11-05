#!/usr/bin/env python3
"""Copy a LeRobot v2.1 or v3 dataset with the gripper ACTION as clean closed/open labels.

    action[:, 6]  ->  gripper_controller.binarize(widths, closed, open, close_below, open_above)

Everything else is copied unchanged (observation.state keeps the measured
gripper width, the joint deltas are untouched); the videos are hard-linked, not
copied. meta/stats.json and the per-episode stats (v3 meta/episodes/*.parquet,
v2.1 meta/episodes_stats.jsonl) are recomputed for the gripper dim of
`action`. The source is never written.

Also checks the labels backwards (report.json + plots, in --report, not in the
dataset):
  - per episode: label switches, frames changed, partial widths (inside the band),
    short label segments (< 3 frames, would be a flicker)
  - replay: GripperController (confirm_ticks, ramp) on the ORIGINAL widths, as if
    ACT had imitated the data exactly: every label switch it catches and how
    many ticks late, and the switches it misses

    .venv/bin/python temp/gripper/binarize_gripper_dataset.py --src <dataset> --dry-run
    .venv/bin/python temp/gripper/binarize_gripper_dataset.py --src <dataset>   # writes <dataset>_binGrip
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: trossen_real is not an installed package
from trossen_real.inference.gripper_controller import GripperController, binarize  # noqa: E402

G = 6  # gripper dim of `action`


def segments(labels):
    """[(start, length, value)] runs of equal labels."""
    out, s = [], 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[s]:
            out.append((s, k - s, labels[s]))
            s = k
    return out


def replay(widths, labels, args):
    """Controller on the original widths vs the labels: delay of each caught switch, misses."""
    ctrl = GripperController(args.closed, args.open, args.close_below, args.open_above, args.confirm_ticks, args.max_step,
                             limits=(args.closed, args.open))
    ctrl.reset(labels[0])
    ctrl.is_open = labels[0] == args.open
    states = []
    for w in widths:
        ctrl.step(float(w))
        states.append(ctrl.is_open)
    want = [lab == args.open for lab in labels]
    sw_label = [k for k in range(1, len(want)) if want[k] != want[k - 1]]
    sw_ctrl = [k for k in range(1, len(states)) if states[k] != states[k - 1]]
    delays, missed = [], 0
    for i, k in enumerate(sw_label):
        nxt = sw_label[i + 1] if i + 1 < len(sw_label) else len(want)
        hit = [c for c in sw_ctrl if k <= c < nxt and states[c] == want[k]]
        if hit:
            delays.append(hit[0] - k)
        else:
            missed += 1
    return delays, missed, len(sw_ctrl), states


def plot_episode(ep, w, lab, meas, states, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, np.asarray(w) * 1000, color="#999999", lw=1.2, label="recorded action (before)")
    ax.plot(t, np.asarray(lab) * 1000, color="#2266cc", lw=2, label="label (after)")
    ax.plot(t, np.asarray(meas) * 1000, color="#2a9d5c", lw=1, label="observation.state gripper")
    ax.step(t, np.where(states, args.open, args.closed) * 1000 + 0.6, where="post", color="#e07a1f", lw=1,
            label="controller replay on recorded (state)")
    for b in (args.close_below, args.open_above):
        ax.axhline(b * 1000, color="#bbbbbb", lw=0.6, ls=":")
    ax.set_title(f"episode {ep}", fontsize=10, loc="left")
    ax.set_xlabel("frame (20 Hz)")
    ax.set_ylabel("gripper (mm)")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def stats_of(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()), std=float(x.std()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", type=Path, default=None, help="Default: <src>_binGrip next to the source.")
    ap.add_argument("--closed", type=float, default=0.004)
    ap.add_argument("--open", type=float, default=0.040)
    ap.add_argument("--close-below", type=float, default=0.010)
    ap.add_argument("--open-above", type=float, default=0.016)
    ap.add_argument("--confirm-ticks", type=int, default=3, help="Replay check only.")
    ap.add_argument("--max-step", type=float, default=0.003, help="Replay check only.")
    ap.add_argument("--report", type=Path, default=None, help="Default: temp/gripper/dataset_reports/<dst name>.")
    ap.add_argument("--plots", type=int, default=12, help="Episode plots (the ones with the most switches + random).")
    ap.add_argument("--dry-run", action="store_true", help="Report only, write no dataset.")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = (args.dst or src.with_name(src.name + "_binGrip")).resolve()
    report_dir = (args.report or HERE / "dataset_reports" / dst.name).resolve()
    info = json.loads((src / "meta" / "info.json").read_text())
    version = str(info.get("codebase_version", ""))
    if not version.startswith(("v2.1", "v3")):
        raise SystemExit(f"expects a LeRobot v2.1 or v3 dataset, got {version}")
    if info["features"]["action"]["names"][G] != "gripper":
        raise SystemExit("action dim 6 is not 'gripper'")
    if not args.dry_run and dst.exists():
        raise SystemExit(f"{dst} exists; delete it or pass --dst")

    # ---- labels ---------------------------------------------------------------------
    data_files = sorted(glob.glob(str(src / "data" / "*" / "*.parquet")))
    tables = {f: pq.read_table(f) for f in data_files}
    per_ep, new_cols = {}, {}
    for f, tab in tables.items():
        act = np.asarray(tab.column("action").to_pylist(), dtype=np.float64)
        state = np.asarray(tab.column("observation.state").to_pylist(), dtype=np.float64)
        epi = np.asarray(tab.column("episode_index").to_pylist())
        new = act.copy()
        for ep in np.unique(epi):
            m = np.flatnonzero(epi == ep)  # rows of one episode are contiguous and in frame order
            w = act[m, G]
            lab = binarize(w, args.closed, args.open, args.close_below, args.open_above)
            new[m, G] = lab
            per_ep[int(ep)] = dict(w=w, lab=np.asarray(lab), meas=state[m, G])
        new_cols[f] = new

    # ---- report ---------------------------------------------------------------------
    report_dir.mkdir(parents=True, exist_ok=True)
    lo_w, hi_w = args.close_below, args.open_above
    rows, all_delays, total_missed = [], [], 0
    for ep, d in sorted(per_ep.items()):
        w, lab = d["w"], d["lab"]
        segs = segments(list(lab))
        delays, missed, n_ctrl, states = replay(w, lab, args)
        d["states"] = states
        all_delays += delays
        total_missed += missed
        rows.append(dict(
            episode=ep, frames=len(w), label_switches=len(segs) - 1,
            opens=sum(1 for k in range(1, len(lab)) if lab[k] > lab[k - 1]),
            changed_gt_1mm=int((np.abs(lab - w) > 0.001).sum()),
            partial_frames=int(((w > lo_w) & (w < hi_w)).sum()),
            short_segments=sum(1 for s, n, _ in segs[1:-1] if n < 3),
            replay_switches=n_ctrl, replay_missed=missed,
            replay_delay_max=max(delays) if delays else None))
    n = len(rows)
    summary = dict(
        src=str(src), dst=str(dst), closed=args.closed, open=args.open, close_below=args.close_below, open_above=args.open_above,
        band_mm=[round(lo_w * 1000, 1), round(hi_w * 1000, 1)],
        episodes=n, frames=int(sum(r["frames"] for r in rows)),
        frames_changed_gt_1mm=int(sum(r["changed_gt_1mm"] for r in rows)),
        partial_frames=int(sum(r["partial_frames"] for r in rows)),
        label_switches_per_episode=dict(zip(["min", "median", "max"],
                                            np.percentile([r["label_switches"] for r in rows], [0, 50, 100]).tolist())),
        episodes_without_open=[r["episode"] for r in rows if r["opens"] == 0],
        episodes_with_short_segments=[r["episode"] for r in rows if r["short_segments"]],
        replay=dict(label_switches=len(all_delays) + total_missed, caught=len(all_delays), missed=total_missed,
                    delay_ticks=dict(zip(["min", "median", "max"],
                                         np.percentile(all_delays, [0, 50, 100]).tolist())) if all_delays else None),
    )
    (report_dir / "report.json").write_text(json.dumps(dict(summary=summary, episodes=rows), indent=1))
    print(json.dumps(summary, indent=1))

    busiest = sorted(rows, key=lambda r: -r["label_switches"])[: args.plots // 2]
    rng = np.random.default_rng(0)
    rest = [r for r in rows if r not in busiest]
    pick = busiest + [rest[i] for i in rng.choice(len(rest), min(len(rest), args.plots - len(busiest)), replace=False)]
    for r in pick:
        d = per_ep[r["episode"]]
        plot_episode(r["episode"], d["w"], d["lab"], d["meas"], d["states"], args,
                     report_dir / f"episode_{r['episode']:04d}.png")
    print(f"report + {len(pick)} episode plots: {report_dir}")
    if args.dry_run:
        return

    # ---- write the copy ----------------------------------------------------------------
    def link_or_copy(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("videos"))
    if (src / "videos").exists():
        shutil.copytree(src / "videos", dst / "videos", copy_function=link_or_copy)

    for f, tab in tables.items():
        out = dst / Path(f).relative_to(src)
        i = tab.schema.get_field_index("action")
        col = pa.array(new_cols[f].astype(np.float32).tolist(), type=tab.schema.field("action").type)
        pq.write_table(tab.set_column(i, tab.schema.field("action"), col), out)

    all_g = np.concatenate([new_cols[f][:, G] for f in data_files])
    stats_path = dst / "meta" / "stats.json"
    if stats_path.exists():  # a v2.1 dataset written by LeRobotDataset.create has none (aggregated on load)
        stats = json.loads(stats_path.read_text())
        for k, v in stats_of(all_g).items():
            stats["action"][k][G] = v
        stats_path.write_text(json.dumps(stats, indent=4))

    ep_stats = dst / "meta" / "episodes_stats.jsonl"  # v2.1
    if version.startswith("v2"):
        lines = [json.loads(x) for x in ep_stats.read_text().splitlines() if x.strip()]
        for e in lines:
            for k, v in stats_of(per_ep[int(e["episode_index"])]["lab"]).items():
                e["stats"]["action"][k][G] = v
        ep_stats.write_text("".join(json.dumps(e) + "\n" for e in lines))
    for f in sorted(glob.glob(str(dst / "meta" / "episodes" / "*" / "*.parquet"))):  # v3
        tab = pq.read_table(f)
        eps = tab.column("episode_index").to_pylist()
        for k in ("min", "max", "mean", "std"):
            name = f"stats/action/{k}"
            vals = tab.column(name).to_pylist()
            for row, ep in enumerate(eps):
                vals[row][G] = stats_of(per_ep[int(ep)]["lab"])[k]
            i = tab.schema.get_field_index(name)
            tab = tab.set_column(i, tab.schema.field(name), pa.array(vals, type=tab.schema.field(name).type))
        pq.write_table(tab, f)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
