# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Gather every artifact of an eval campaign into one self-contained tree.

A finished run leaves its pieces in three unrelated places - the run dir
(`eval_runs/...`), the per-tick log (`logs/infer_*.jsonl`) and the camera
recordings (`infer_videos/<date>/...`). That is fine while running but useless
as a record: the logs and videos are named by wall-clock time, sit next to
hundreds of unrelated files, and get rotated away. This COPIES (never moves,
never deletes) all of them into

    <dest>/runs/ckpt_<step>/pose_<NN>/run_<id>/
        summary.json  timeline.jsonl  tick_log.jsonl  videos/cam_*.mp4

so a checkpoint's evidence is one directory, and writes `MANIFEST.csv` mapping
every copied file back to its original path - the copy is the archive, the
manifest is the provenance.

Discarded runs are copied too, into `discarded_<id>/`, because "we threw this
one away" is itself data; nothing downstream counts them.

    .venv/bin/python -m trossen_real.scripts.collect_eval_artifacts \
        --eval-root eval_runs --dest trossen_real/EVAL/ACT/TrossenStation3

Idempotent: a file already present at the destination with the same size is
left alone, so re-running after more evals only copies what is new.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

CAM_SUFFIXES = ("cam_high", "cam_front", "cam_left_wrist", "cam_right_wrist")

# Codecs a Chromium-based player (VS Code, browsers) can already decode. A
# recording in one of these is copied as-is; only the legacy mpeg4 ones need
# converting.
WEB_CODECS = {"h264", "av1", "vp9", "vp8"}


def _ckpt_dirname(label: str) -> str:
    """'ACT_.../checkpoints/010000' -> 'ckpt_010000'."""
    return "ckpt_" + label.rsplit("/", 1)[-1]


def _video_name(src: Path) -> str:
    """'091700_trossen_station3_single_cam_high.mp4' -> 'cam_high.mp4'.

    Falls back to the original name if the camera tag is not recognised, so an
    unexpected camera is archived rather than silently dropped.
    """
    stem = src.stem
    for cam in CAM_SUFFIXES:
        if stem.endswith(cam):
            return cam + src.suffix
    return src.name


def _codec(path: Path) -> str:
    """Codec name of a video file, or "" if it holds no decodable stream."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    return r.stdout.strip()


def _transcode_h264(src: Path, dst: Path, manifest: list, repo: Path,
                    dest_root: Path) -> str:
    """Re-encode an MPEG-4 Part 2 recording to H.264 while archiving it.

Recordings made before 2026-09-30 came from `cv2.VideoWriter` with fourcc
    `mp4v` = MPEG-4 Part 2, which no Chromium-based player (VS Code, browsers)
    decodes - they were unwatchable exactly where you want to watch them. Those
    get converted here, with the system ffmpeg, on the archived COPY; the
    originals in `infer_videos/` are left alone.

    `video_recorder.py` now writes AV1 through PyAV, matching the dataset, so
    newer recordings hit the WEB_CODECS shortcut above and are copied untouched.
    Skips work already done: the destination is only written once.
    """
    if not src.exists():
        manifest.append((str(dst.relative_to(dest_root)), str(src), -1, "MISSING"))
        return "missing"
    if _codec(src) in WEB_CODECS:
        # Recorded after video_recorder.py moved to PyAV: already AV1, same as
        # the dataset. Re-encoding would only lose quality and add size.
        return _copy(src, dst, manifest, repo, dest_root)
    if dst.exists() and dst.stat().st_size > 0:
        # Probe rather than assume: an earlier pass may have archived an
        # undecodable stub here, and mislabelling that as h264 in the manifest
        # would be a lie that nothing downstream could catch.
        codec = _codec(dst)
        manifest.append((str(dst.relative_to(dest_root)), _origin(src, repo),
                         dst.stat().st_size,
                         f"ok({codec})" if codec else "ok(empty-no-transcode)"))
        return "skipped"
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(dst)],
            capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            # A run aborted before its first frame leaves a ~258-byte MP4 header
            # with nothing in it, which ffmpeg rightly refuses. Archive the bytes
            # verbatim rather than dropping them: "this run recorded nothing" is
            # part of the record, and the manifest says so.
            dst.unlink(missing_ok=True)
            shutil.copy2(src, dst)
            manifest.append((str(dst.relative_to(dest_root)), _origin(src, repo),
                             dst.stat().st_size, "ok(empty-no-transcode)"))
            return "copied"
    manifest.append((str(dst.relative_to(dest_root)), _origin(src, repo),
                     dst.stat().st_size, "ok(h264)"))
    return "copied"


def _origin(src: Path, repo: Path) -> str:
    try:
        return str(src.relative_to(repo))
    except ValueError:
        return str(src)


def _copy(src: Path, dst: Path, manifest: list, repo: Path, dest_root: Path) -> str:
    """Copy src->dst unless already identical in size. Returns 'copied'|'skipped'|'missing'."""
    if not src.exists():
        manifest.append((str(dst.relative_to(dest_root)), str(src), -1, "MISSING"))
        return "missing"
    size = src.stat().st_size
    if dst.exists() and dst.stat().st_size == size:
        state = "skipped"
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        state = "copied"
    manifest.append((str(dst.relative_to(dest_root)), _origin(src, repo), size, "ok"))
    return state


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", default="eval_runs")
    ap.add_argument("--dest", required=True, help="campaign folder, e.g. trossen_real/EVAL/ACT/TrossenStation3")
    ap.add_argument("--repo-root", default=".", help="root the original paths in summary.json are relative to")
    ap.add_argument("--config-hash", default=None,
                    help="only collect runs with this eval_config hash (default: all)")
    ap.add_argument("--no-videos", action="store_true", help="skip the .mp4 files (metadata only)")
    ap.add_argument("--video-codec", choices=("copy", "h264"), default="h264",
                    help="h264 (default) re-encodes the archived copy so VS Code and "
                         "browsers can play it; copy keeps the original mpeg4 bytes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    eval_root = Path(args.eval_root)
    if not eval_root.is_absolute():
        eval_root = repo / eval_root
    dest_root = Path(args.dest)
    if not dest_root.is_absolute():
        dest_root = repo / dest_root
    runs_root = dest_root / "runs"

    summaries = sorted(eval_root.glob("*/*/*/*/summary.json"))
    if not summaries:
        print(f"no runs found under {eval_root}", file=sys.stderr)
        return 1

    manifest: list = []
    tally = {"copied": 0, "skipped": 0, "missing": 0}
    n_runs = 0
    for sfile in summaries:
        s = json.loads(sfile.read_text())
        if args.config_hash and s.get("eval_config", {}).get("hash") != args.config_hash:
            continue
        n_runs += 1
        saved = s.get("status") == "saved"
        prefix = "run_" if saved else "discarded_"
        run_dir = (runs_root / _ckpt_dirname(s["checkpoint_label"])
                   / f"pose_{s['pose']['index']:02d}" / f"{prefix}{s['run_id']}")
        if args.dry_run:
            print(run_dir.relative_to(dest_root))
            continue

        for name, src in (("summary.json", sfile),
                          ("timeline.jsonl", sfile.parent / "timeline.jsonl")):
            tally[_copy(src, run_dir / name, manifest, repo, dest_root)] += 1

        art = s.get("artifacts") or {}
        if art.get("tick_log"):
            tally[_copy(repo / art["tick_log"], run_dir / "tick_log.jsonl",
                        manifest, repo, dest_root)] += 1
        if not args.no_videos:
            put = _copy if args.video_codec == "copy" else _transcode_h264
            for v in art.get("videos") or []:
                src = repo / v
                tally[put(src, run_dir / "videos" / _video_name(src),
                          manifest, repo, dest_root)] += 1

    if args.dry_run:
        print(f"\n{n_runs} runs would be collected")
        return 0

    # Filtered index: the campaign's own ledger, independent of the live one.
    idx_src = eval_root / "index.jsonl"
    if idx_src.exists():
        lines = [l for l in idx_src.read_text().splitlines() if l.strip()]
        if args.config_hash:
            lines = [l for l in lines if json.loads(l).get("eval_config_hash") == args.config_hash]
        (runs_root / "index.jsonl").write_text("\n".join(lines) + "\n")

    with (runs_root / "MANIFEST.csv").open("w", newline="") as fh:
        # \n, not csv's default \r\n: this file gets grepped and awk'd, and a
        # trailing \r makes the last column never compare equal.
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["copied_path", "original_path", "bytes", "state"])
        w.writerows(sorted(manifest))

    total = sum(b for _, _, b, st in manifest if st.startswith("ok"))
    print(f"{n_runs} runs -> {runs_root}")
    print(f"files copied={tally['copied']} skipped(identical)={tally['skipped']} "
          f"MISSING={tally['missing']}")
    print(f"archive size {total / 1e6:.0f} MB, manifest rows {len(manifest)}")
    return 1 if tally["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
