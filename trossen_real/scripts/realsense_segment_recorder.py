# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Long-running, memory-safe RealSense -> rolling-.mp4-chunks recorder.

Design (and why, vs. a naive single-loop version):

- pyrealsense2 + `subprocess` piping into the system `ffmpeg` binary only -
  no cv2/numpy required to run this (get_data() -> bytes() is enough).
- ffmpeg's own `-f segment` muxer does the chunking (one long-running ffmpeg
  process, not one process per chunk) - it cuts a new file every
  `--chunk-duration-sec` and forces a keyframe at each cut, so every
  finished chunk is a complete, independently-playable .mp4 even if this
  script is later killed mid-chunk.
- Capture and ffmpeg-feeding run in TWO threads connected by a `queue.Queue`
  bounded to `QUEUE_MAXSIZE` frames, DROP-OLDEST on backpressure. A naive
  single-loop version (capture, then `stdin.write()`, repeat) blocks frame
  capture on the write() call - if ffmpeg ever stalls even briefly (e.g.
  exactly at a segment cut, doing an internal file close/open), capture
  blocks too. That's fine for a fire-and-forget diagnostic recorder running
  in its own process, but this is meant to run alongside a real-time robot
  control loop - the fix is a small BOUNDED queue (not an unbounded list -
  still satisfies "no unmanaged accumulation" for hours-long runs) so a
  slow/stalled writer drops old frames instead of ever blocking the
  capture thread.
- Tries `h264_nvenc` first (hardware encode - near-zero CPU, matters for
  "run concurrently with a real-time policy loop"), and falls back to
  `libx264` automatically if nvenc isn't available on this machine (probed
  once at startup with a throwaway 1-frame encode, not assumed) - the
  target server's GPU is unknown, so this must not be hardcoded either way.
- Handles SIGINT AND SIGTERM (not just KeyboardInterrupt) so a normal
  `kill`/service-stop also closes stdin and lets ffmpeg finalize the
  in-progress chunk cleanly instead of leaving a truncated file. Even a
  `kill -9` on THIS process still lets ffmpeg finish its current segment
  properly in practice (confirmed by testing) - killing this process closes
  its end of the stdin pipe, which delivers a clean EOF to ffmpeg.
- Each frame gets a metadata record (system local time, UTC/epoch time -
  the one universal, cross-machine-comparable clock - and the RealSense
  hardware timestamp) written to one JSON file per chunk
  (`chunk_000_metadata.json` alongside `chunk_000.mp4`, etc.), so a frame in
  any chunk can be tied back to wall-clock time later (e.g. to sync against
  a robot's own onboard log).
- `--fps` is validated against the camera's ACTUALLY supported profiles at
  startup (queried live, not assumed) - RealSense cameras only support a
  fixed set of fps values per resolution, not arbitrary ones.

Usage:
    .venv/bin/python -m trossen_real.scripts.realsense_segment_recorder
    # or with a short chunk duration to quickly see segmenting work:
    .venv/bin/python -m trossen_real.scripts.realsense_segment_recorder --chunk-duration-sec 10 --max-runtime-sec 35
    # lower fps (must be one this camera actually supports at this resolution):
    .venv/bin/python -m trossen_real.scripts.realsense_segment_recorder --fps 15
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import queue
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import pyrealsense2 as rs

logger = logging.getLogger("realsense_segment_recorder")

# Hardcoded for local testing on this workstation - change to the target
# camera's serial when running elsewhere (see docstring above / hand-off
# notes for the server-side swap).
CAMERA_SERIAL = "230322273189"

WIDTH_DEFAULT, HEIGHT_DEFAULT, FPS_DEFAULT = 640, 480, 30
CHUNK_DURATION_SEC_DEFAULT = 300  # 5 minutes
OUTPUT_DIR_DEFAULT = "eval_recordings"
QUEUE_MAXSIZE = 2  # bounded - drop-oldest on backpressure, never blocks capture


def _ffmpeg_bin() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH - install it (system package, not pip).")
    return ffmpeg


def _supported_fps_values(serial: str, width: int, height: int) -> list[int]:
    """Query the REAL device for which fps values it actually supports at
    this resolution (rgb8 color stream) - confirmed by testing: this
    particular D405 only supports {5, 15, 30, 60, 90} at 640x480, NOT an
    arbitrary value like 20. Returns [] if the device/serial isn't found."""
    ctx = rs.context()
    for d in ctx.query_devices():
        if d.get_info(rs.camera_info.serial_number) != serial:
            continue
        fps_set = set()
        for s in d.query_sensors():
            for p in s.get_stream_profiles():
                if p.stream_type() == rs.stream.color and p.format() == rs.format.rgb8:
                    vp = p.as_video_stream_profile()
                    if (vp.width(), vp.height()) == (width, height):
                        fps_set.add(vp.fps())
        return sorted(fps_set)
    return []


def _validate_fps(serial: str, width: int, height: int, fps: int) -> None:
    supported = _supported_fps_values(serial, width, height)
    if not supported:
        logger.warning("Could not query supported fps values for serial=%s at %dx%d (device not found/still "
                        "initializing?) - proceeding without validation.", serial, width, height)
        return
    if fps not in supported:
        raise ValueError(f"fps={fps} is not supported by camera {serial} at {width}x{height}. "
                          f"This camera supports: {supported} at this resolution. "
                          f"(confirmed by querying the real device, not assumed)")


def _probe_nvenc(ffmpeg: str, width: int, height: int) -> bool:
    """Try a throwaway 1-frame encode with h264_nvenc; return True if it
    actually works on this machine (not just listed in `ffmpeg -encoders` -
    the encoder can be listed but fail to open if the GPU/driver doesn't
    support it, e.g. no NVENC-capable GPU, or driver too old)."""
    frame = b"\x00" * (width * height * 3)
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", "1", "-i", "-",
        "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, input=frame, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
        return proc.returncode == 0
    except Exception:
        return False


def _build_ffmpeg_cmd(ffmpeg: str, output_dir: Path, width: int, height: int, fps: int,
                       chunk_duration_sec: int, use_nvenc: bool) -> list[str]:
    codec_args = (
        # NOTE: do NOT add `-tune ll` (nvenc's low-latency tune) - confirmed
        # by testing it silently breaks `-force_key_frames` (segmenting
        # never cuts, one giant file instead of chunks). Not needed anyway -
        # this writes to disk, not a live low-latency stream.
        # `-forced-idr 1` is REQUIRED for accurate cut points - without it,
        # nvenc snaps "forced" keyframes to its own internal GOP boundaries
        # instead of the exact requested time, confirmed by testing: a 35s
        # run with --chunk-duration-sec 10 produced chunks of
        # 16.7s/8.3s/8.3s/1.7s instead of 10s/10s/10s/5s. With
        # `-forced-idr 1`, cuts land exactly on schedule (matches libx264,
        # which honors -force_key_frames correctly by default).
        ["-c:v", "h264_nvenc", "-preset", "p1", "-forced-idr", "1"]
        if use_nvenc
        else ["-c:v", "libx264", "-preset", "ultrafast"]
    )
    return [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        *codec_args,
        "-pix_fmt", "yuv420p",
        # Explicit forced-keyframe schedule - REQUIRED for the segment muxer
        # to actually cut on time when re-encoding (confirmed by testing:
        # without this, a 34s run with --chunk-duration-sec 10 produced ONE
        # 34s chunk, not four ~10s ones - the segmenter waits for a keyframe
        # that never arrives on schedule, since nvenc in particular doesn't
        # reliably honor the segment muxer's internal force-keyframe hook on
        # its own GOP schedule). This works regardless of encoder/GOP config.
        "-force_key_frames", f"expr:gte(t,n_forced*{chunk_duration_sec})",
        "-f", "segment", "-segment_time", str(chunk_duration_sec), "-reset_timestamps", "1",
        str(output_dir / "chunk_%03d.mp4"),
    ]


def _now_meta() -> dict:
    """One frame's timestamp record, taken at capture time. `unix_epoch_utc`
    is THE universal, cross-machine-comparable value (POSIX epoch seconds -
    identical meaning on every correctly clock-synced machine on Earth,
    independent of timezone) - use this to sync against a robot's onboard
    log later. `utc_iso`/`local_iso` are human-readable views of that same
    instant, for reading directly without doing the epoch math yourself."""
    t = time.time()
    utc_dt = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc)
    local_dt = datetime.datetime.fromtimestamp(t)
    return {
        "unix_epoch_utc": t,
        "utc_iso": utc_dt.isoformat(),
        "local_iso": local_dt.isoformat(),
    }


def _capture_loop(pipeline: rs.pipeline, frame_q: "queue.Queue[tuple[dict, bytes]]",
                   stop_event: threading.Event, stats: dict) -> None:
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            continue  # timeout waiting for a frame - just retry, don't crash the thread
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        meta = _now_meta()
        # RealSense's own device-clock timestamp (ms) - NOT synced to any
        # global clock, don't use it across machines. Useful only for
        # measuring inter-frame capture jitter within this one session, with
        # better precision than repeated time.time() calls from Python
        # (which carry OS scheduling jitter on top of the real capture time).
        meta["realsense_hw_timestamp_ms"] = color_frame.get_timestamp()
        data = bytes(color_frame.get_data())
        stats["captured"] += 1
        item = (meta, data)
        try:
            frame_q.put_nowait(item)
        except queue.Full:
            try:
                frame_q.get_nowait()  # drop oldest
                stats["dropped"] += 1
            except queue.Empty:
                pass
            try:
                frame_q.put_nowait(item)
            except queue.Full:
                pass


class _ChunkMetadataWriter:
    """Mirrors ffmpeg's own chunk boundaries (frames_per_chunk =
    fps * chunk_duration_sec) so `chunk_NNN_metadata.json` lines up 1:1 with
    `chunk_NNN.mp4` - verified against real output (see the session's
    HOW_TO_POSTPROCESS.txt for how to spot-check this yourself). Writes a
    chunk's json as soon as that chunk fills up, so - like the mp4 chunks
    themselves - every ALREADY-COMPLETE chunk's json is safe on disk even if
    this script dies before finishing the next one; `flush_partial()` (called
    on clean shutdown) writes out the final, possibly-partial chunk too."""

    def __init__(self, session_dir: Path, frames_per_chunk: int) -> None:
        self.session_dir = session_dir
        self.frames_per_chunk = frames_per_chunk
        self.chunk_index = 0
        self.session_frame_index = 0
        self._buffer: list[dict] = []

    def add_frame(self, meta: dict) -> None:
        record = {
            "frame_index_in_chunk": len(self._buffer),
            "frame_index_session": self.session_frame_index,
            **meta,
        }
        self._buffer.append(record)
        self.session_frame_index += 1
        if len(self._buffer) >= self.frames_per_chunk:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        path = self.session_dir / f"chunk_{self.chunk_index:03d}_metadata.json"
        path.write_text(json.dumps({"chunk_index": self.chunk_index, "frames": self._buffer}, indent=2))
        logger.info("Wrote %s (%d frames)", path.name, len(self._buffer))
        self.chunk_index += 1
        self._buffer = []

    def flush_partial(self) -> None:
        self._flush()


def _writer_loop(proc: subprocess.Popen, frame_q: "queue.Queue[tuple[dict, bytes]]",
                  stop_event: threading.Event, meta_writer: _ChunkMetadataWriter) -> None:
    while not stop_event.is_set() or not frame_q.empty():
        try:
            meta, data = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            proc.stdin.write(data)
        except (BrokenPipeError, OSError):
            logger.exception("ffmpeg stdin closed unexpectedly - stopping writer loop.")
            return
        meta_writer.add_frame(meta)


def _write_how_to_postprocess(session_dir: Path, fps: int) -> None:
    doc = f"""How to post-process this session's recording
==============================================

Chunks in this folder: chunk_000.mp4, chunk_001.mp4, ... - each one is an
independently valid, playable .mp4 (recorded at {fps} fps). They are NOT
merged automatically. Per-frame timestamp metadata is in the matching
chunk_NNN_metadata.json (see session_info.json for the full run config).

1) Merge ALL chunks into one continuous video (lossless, no re-encoding):

    cd {session_dir}
    for f in chunk_*.mp4; do echo "file '$f'"; done | sort > _concat_list.txt
    ffmpeg -f concat -safe 0 -i _concat_list.txt -c copy merged.mp4

2) Play back the merged video at Nx speed (e.g. 2x here - change the "2.0"
   for a different multiplier; this re-encodes, since changing playback
   speed isn't a lossless stream-copy operation):

    ffmpeg -i merged.mp4 -filter:v "setpts=PTS/2.0" -an merged_2x.mp4

   For 4x: setpts=PTS/4.0, for half-speed slow-motion: setpts=PTS*2.0, etc.

3) Look up a frame's real-world time: open chunk_NNN_metadata.json, find the
   entry with the matching frame_index_in_chunk. `unix_epoch_utc` is the
   universal, cross-machine value (POSIX epoch seconds, same meaning on any
   correctly clock-synced machine anywhere) - use it to line this recording
   up against a robot's own onboard log timestamps. `utc_iso`/`local_iso`
   are the same instant, just formatted for a human to read directly.
"""
    (session_dir / "HOW_TO_POSTPROCESS.txt").write_text(doc)


def run(output_root: Path, width: int, height: int, fps: int, chunk_duration_sec: int,
        max_runtime_sec: float | None) -> None:
    _validate_fps(CAMERA_SERIAL, width, height, fps)

    # Each run gets its own timestamped subfolder. Chunk files are named by
    # SEQUENCE NUMBER within a session (chunk_000.mp4, chunk_001.mp4, ...),
    # not by timestamp - always starting back at 000. Without a per-session
    # subfolder, running this script twice into the same --output-dir would
    # silently overwrite the first run's chunk_000.mp4 (ffmpeg's -y flag) -
    # confirmed this is a real risk, not hypothetical. The session folder
    # name itself carries the timestamp instead.
    session_dir = output_root / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    output_dir = session_dir
    logger.info("Recording session dir: %s (chunk_000.mp4, chunk_001.mp4, ... within it)", output_dir)
    ffmpeg = _ffmpeg_bin()

    use_nvenc = _probe_nvenc(ffmpeg, width, height)
    logger.info("Encoder: %s", "h264_nvenc (hardware)" if use_nvenc else "libx264 (software fallback)")

    (session_dir / "session_info.json").write_text(json.dumps({
        "camera_serial": CAMERA_SERIAL, "width": width, "height": height, "fps": fps,
        "chunk_duration_sec": chunk_duration_sec, "encoder": "h264_nvenc" if use_nvenc else "libx264",
        "session_start": _now_meta(),
    }, indent=2))
    _write_how_to_postprocess(session_dir, fps)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(CAMERA_SERIAL)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    pipeline.start(cfg)
    logger.info("RealSense pipeline started (serial=%s, %dx%d@%d)", CAMERA_SERIAL, width, height, fps)

    cmd = _build_ffmpeg_cmd(ffmpeg, output_dir, width, height, fps, chunk_duration_sec, use_nvenc)
    logger.info("ffmpeg cmd: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=width * height * 3)

    frame_q: "queue.Queue[tuple[dict, bytes]]" = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()
    stats = {"captured": 0, "dropped": 0}
    meta_writer = _ChunkMetadataWriter(session_dir, frames_per_chunk=fps * chunk_duration_sec)

    cap_thread = threading.Thread(target=_capture_loop, args=(pipeline, frame_q, stop_event, stats), daemon=True)
    writer_thread = threading.Thread(target=_writer_loop, args=(proc, frame_q, stop_event, meta_writer), daemon=True)
    cap_thread.start()
    writer_thread.start()

    def _shutdown(signum=None, frame=None) -> None:
        logger.info("Shutting down (signal=%s) - captured=%d dropped=%d", signum, stats["captured"], stats["dropped"])
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    start_t = time.monotonic()
    try:
        while not stop_event.is_set():
            time.sleep(1.0)
            if max_runtime_sec is not None and (time.monotonic() - start_t) >= max_runtime_sec:
                logger.info("max-runtime-sec reached, stopping.")
                stop_event.set()
                break
            if int(time.monotonic() - start_t) % 30 == 0:
                logger.info("captured=%d dropped=%d queue_size=%d", stats["captured"], stats["dropped"], frame_q.qsize())
    finally:
        stop_event.set()
        writer_thread.join(timeout=5)
        meta_writer.flush_partial()
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass
        proc.wait(timeout=15)
        pipeline.stop()
        logger.info("Done. captured=%d dropped=%d", stats["captured"], stats["dropped"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT,
                         help="Parent dir - each run creates its own timestamped subfolder in here.")
    parser.add_argument("--width", type=int, default=WIDTH_DEFAULT)
    parser.add_argument("--height", type=int, default=HEIGHT_DEFAULT)
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT,
                         help="Must be a value this camera actually supports at --width x --height "
                              "(validated against the real device at startup - e.g. this D405 at "
                              "640x480 only supports 5/15/30/60/90, NOT arbitrary values like 20).")
    parser.add_argument("--chunk-duration-sec", type=int, default=CHUNK_DURATION_SEC_DEFAULT)
    parser.add_argument("--max-runtime-sec", type=float, default=None,
                         help="For testing only - stop automatically after this many seconds.")
    args = parser.parse_args()
    run(Path(args.output_dir), args.width, args.height, args.fps, args.chunk_duration_sec, args.max_runtime_sec)


if __name__ == "__main__":
    main()
