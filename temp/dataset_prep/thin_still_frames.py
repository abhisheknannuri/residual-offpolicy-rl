#!/usr/bin/env python3
"""Drop the middle of long still stretches from a LeRobot v2.1 dataset.

Which frames go is decided by still_rule.classify (code 3 = dropped), the same
rule still_explorer.py shows. Every kept frame is copied unchanged: its action
is target - its own measured state, so it stays correct without the dropped
neighbours (nothing is merged or summed). The videos are re-encoded with only
the kept frames (decoded once per camera per episode, then add_frame /
save_episode, as convert_to_delta_joint_dataset.py does); timestamps,
frame_index, index and all stats are rebuilt by LeRobot.

Gripper labels for the protected windows come from gripper_controller.binarize
on action[:, 6]. The gripper action itself is NOT changed here (run
binarize_gripper_dataset.py on the v3 result).

    # 1. dry run: report only (report.json in --report), writes no dataset
    .venv/bin/python temp/dataset_prep/thin_still_frames.py --src <v2.1 dataset> --dry-run
    # 2. the real pass (slow: AV1 re-encode), writes --dst (default <src without _old>_thin)
    .venv/bin/python temp/dataset_prep/thin_still_frames.py --src <v2.1 dataset>

Then: v3 conversion as usual, then binarize_gripper_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from trossen_real.inference.gripper_controller import binarize  # noqa: E402
from still_rule import StillRule, classify, speeds  # noqa: E402

logger = logging.getLogger("thin")
PRE_OPEN_S = 3.0


def label_switches(gripper_action):
    lab = np.asarray(binarize(gripper_action))
    return lab, (np.flatnonzero(np.diff(lab) != 0) + 1).tolist()


def analyze(ds, rule):
    """Per episode: kept frame indices + numbers for the report."""
    out = []
    pre = int(PRE_OPEN_S * ds.meta.fps)
    for ep in range(ds.meta.total_episodes):
        a0, a1 = int(ds.episode_data_index["from"][ep]), int(ds.episode_data_index["to"][ep])
        raw = ds.hf_dataset[a0:a1]
        a = np.asarray(raw["action"], dtype=np.float64)
        q = np.asarray(raw["observation.state"], dtype=np.float64)
        lab, sw = label_switches(a[:, 6])
        code = classify(speeds(q), sw, rule)
        keep = np.flatnonzero(code != 3)
        in_pre = np.zeros(len(q), bool)
        for s in sw:
            if lab[s] > lab[s - 1]:
                in_pre[max(0, s - pre): s] = True
        # the label switches must survive: same sequence of labels after thinning
        _, sw_after = label_switches(a[keep, 6])
        rew = np.asarray(raw["next.reward"], dtype=np.float64).ravel() if "next.reward" in raw else np.zeros(len(q))
        out.append(dict(
            episode=ep, frames=len(q), kept=len(keep), dropped=int((code == 3).sum()),
            dropped_pre_open=int(((code == 3) & in_pre).sum()), protected=int((code == 4).sum()),
            label_switches=len(sw), label_switches_after=len(sw_after),
            reward_frames=int((rew > 0).sum()), reward_frames_after=int((rew[keep] > 0).sum()),
            last_frame_kept=bool(len(keep) and keep[-1] == len(q) - 1),
            keep=keep.tolist()))
    return out


def write(ds, dst: Path, results, rule, src: Path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.video_utils import decode_video_frames
    from trossen_real.scripts.convert_to_delta_joint_dataset import (  # LeRobot I/O helpers only
        AUTO_MANAGED_KEYS, _cast_passthrough_value, _passthrough_keys, _to_hwc_uint8)

    features = {k: v for k, v in ds.meta.features.items() if k not in AUTO_MANAGED_KEYS}
    new = LeRobotDataset.create(repo_id=dst.name, fps=ds.meta.fps, root=str(dst), robot_type=ds.meta.robot_type,
                                features=features, use_videos=True, image_writer_threads=4)
    keys = _passthrough_keys(ds)
    cams = tuple(ds.meta.camera_keys)
    for r in results:
        ep = r["episode"]
        a0, a1 = int(ds.episode_data_index["from"][ep]), int(ds.episode_data_index["to"][ep])
        raw = ds.hf_dataset[a0:a1]
        arrays = {k: np.asarray(raw[k]) for k in ("action", *keys, "timestamp", "task_index")}
        keep = r["keep"]
        ts = arrays["timestamp"][keep].astype(np.float64).tolist()
        imgs = {c: decode_video_frames(ds.root / ds.meta.get_video_file_path(ep, c), ts, ds.tolerance_s,
                                       ds.video_backend).numpy() for c in cams}
        task = ds.meta.tasks[int(arrays["task_index"][0])]
        for j, i in enumerate(keep):
            frame = {"action": arrays["action"][i].astype(np.float32)}
            for k in keys:
                frame[k] = _cast_passthrough_value(ds, k, arrays[k][i])
            for c in cams:
                frame[c] = _to_hwc_uint8(imgs[c][j])
            new.add_frame(frame=frame, task=task)
        new.save_episode()
        logger.info("episode %d: %d -> %d frames", ep, r["frames"], r["kept"])
    (dst / "THIN_STILL_FRAMES.md").write_text(
        "# Still-frame thinning\n\nGenerated by `temp/dataset_prep/thin_still_frames.py` from\n"
        f"`{src}`.\n\n```json\n" + json.dumps(dict(rule=asdict(rule), episodes=[
            {k: v for k, v in r.items() if k != "keep"} for r in results]), indent=1) + "\n```\n")
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None, help="Default: temp/dataset_prep/reports/<dst name>.")
    ap.add_argument("--dry-run", action="store_true")
    d = StillRule()
    ap.add_argument("--joint-thr", default=",".join(str(x) for x in d.joint_thr), help="6 values, rad/frame")
    ap.add_argument("--grip-thr", type=float, default=d.grip_thr, help="m/frame")
    for k in ("min_run", "keep_head", "keep_tail", "protect"):
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=getattr(d, k))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rule = StillRule(joint_thr=[float(x) for x in args.joint_thr.split(",")], grip_thr=args.grip_thr,
                     min_run=args.min_run, keep_head=args.keep_head, keep_tail=args.keep_tail, protect=args.protect)
    if len(rule.joint_thr) != 6:
        raise SystemExit("--joint-thr needs 6 values")
    src = args.src.resolve()
    info = json.loads((src / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v2"):
        raise SystemExit(f"expects a LeRobot v2.x dataset, got {info.get('codebase_version')}")
    name = src.name[:-4] if src.name.endswith("_old") else src.name
    dst = (args.dst or src.with_name(name + "_thin")).resolve()
    report = (args.report or HERE / "reports" / dst.name).resolve()
    if not args.dry_run and dst.exists():
        raise SystemExit(f"{dst} exists; delete it or pass --dst")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="source", root=str(src))
    results = analyze(ds, rule)

    frames = sum(r["frames"] for r in results)
    summary = dict(
        src=str(src), dst=str(dst), rule=asdict(rule), episodes=len(results), frames=frames,
        kept=sum(r["kept"] for r in results), dropped=sum(r["dropped"] for r in results),
        dropped_share=round(sum(r["dropped"] for r in results) / frames, 4),
        dropped_in_3s_before_open=sum(r["dropped_pre_open"] for r in results),
        protected=sum(r["protected"] for r in results),
        episodes_with_changed_label_switches=[r["episode"] for r in results if r["label_switches"] != r["label_switches_after"]],
        episodes_losing_reward_frames=[r["episode"] for r in results if r["reward_frames"] != r["reward_frames_after"]],
        episodes_losing_last_frame=[r["episode"] for r in results if not r["last_frame_kept"]],
        dropped_per_episode=dict(zip(["min", "median", "max"],
                                     np.percentile([r["dropped"] for r in results], [0, 50, 100]).tolist())),
    )
    report.mkdir(parents=True, exist_ok=True)
    (report / "report.json").write_text(json.dumps(dict(summary=summary, episodes=[
        {k: v for k, v in r.items() if k != "keep"} for r in results]), indent=1))
    print(json.dumps(summary, indent=1))
    print(f"report: {report / 'report.json'}")
    if args.dry_run:
        return
    new = write(ds, dst, results, rule, src)
    print(f"wrote {dst}: {new.meta.total_episodes} episodes, {new.meta.total_frames} frames")


if __name__ == "__main__":
    main()
