# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Video recorder for inference app (`trossen_real.infer_app`).

Records active camera streams during inference sessions. Files are saved into
date-based subfolders:

    infer_videos/YYYY-MM-DD/HHMMSS_<station>_<cam_name>.mp4

**Same encoder as the dataset.** This uses PyAV (`av`) with the encoding
parameters taken from LeRobot's own `encode_video_frames()` - the function that
writes the dataset videos - so an inference recording and a dataset episode come
out byte-compatible in codec terms, and there is one video pipeline in this repo
rather than two.

It used to use `cv2.VideoWriter` with fourcc `mp4v`, which produces **MPEG-4
Part 2**. That is a codec no Chromium-based player decodes, so the recordings
would not play in VS Code or a browser while the dataset videos (AV1) did. The
cause was never a setting: OpenCV and PyAV each bundle their own FFmpeg build,
and OpenCV's has no software H.264/AV1 encoder here at all (its only H.264
encoder is `h264_v4l2m2m`, which needs a V4L2 hardware device this machine does
not have, so asking cv2 for `avc1` silently records nothing). PyAV's build has
libx264 and libsvtav1. Hence the port rather than a one-line fourcc change.

Encoding cost at 256x256 was measured before switching: ~0.7 ms/frame, so ~2.8 ms
for 4 cameras against a 50 ms budget at 20 Hz. Encoding happens inline in the
control loop, as it did before.

SVT-AV1 prints a config banner to stderr when a stream opens (4 lines x 4
cameras, once per run). That comes from the C library and cannot be silenced
through Python logging; the dataset path prints the same banner when it saves an
episode.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Encoding settings pulled from the dataset writer itself, so the two paths
# cannot drift apart when LeRobot changes its defaults. Falls back to the values
# LeRobot ships today if the signature ever stops carrying them.
_LEROBOT_FALLBACK = {"vcodec": "libsvtav1", "pix_fmt": "yuv420p", "g": 2, "crf": 30}


def _dataset_encoding() -> dict:
    try:
        from lerobot.common.datasets.video_utils import encode_video_frames

        params = inspect.signature(encode_video_frames).parameters
        out = {}
        for key, default in _LEROBOT_FALLBACK.items():
            p = params.get(key)
            out[key] = default if p is None or p.default is inspect.Parameter.empty else p.default
        return out
    except Exception:
        logger.warning("Could not read LeRobot's encode_video_frames() defaults; "
                       "using the built-in copy %s", _LEROBOT_FALLBACK)
        return dict(_LEROBOT_FALLBACK)


class InferVideoRecorder:
    """Records live camera RGB frames to .mp4 files per camera during an inference run."""

    def __init__(
        self,
        base_dir: Path | str,
        station_name: str,
        camera_names: list[str],
        fps: int = 20,
        resolution: tuple[int, int] = (256, 256),
    ) -> None:
        self.base_dir = Path(base_dir)
        self.station_name = station_name
        self.camera_names = camera_names
        self.fps = fps
        self.target_h, self.target_w = resolution

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")

        # Date subfolder: infer_videos/YYYY-MM-DD/
        self.out_dir = self.base_dir / date_str
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._containers: dict[str, av.container.OutputContainer] = {}
        self._streams: dict[str, av.video.stream.VideoStream] = {}
        self._warned: set[str] = set()
        self.saved_paths: dict[str, str] = {}

        enc = _dataset_encoding()
        self.codec = enc["vcodec"]
        self.pix_fmt = enc["pix_fmt"]
        options = {}
        if enc.get("g") is not None:
            options["g"] = str(enc["g"])
        if enc.get("crf") is not None:
            options["crf"] = str(enc["crf"])

        for cam_name in camera_names:
            file_name = f"{time_str}_{station_name}_{cam_name}.mp4"
            file_path = self.out_dir / file_name
            try:
                container = av.open(str(file_path), mode="w")
                stream = container.add_stream(self.codec, self.fps, options=options)
                stream.pix_fmt = self.pix_fmt
                stream.width = self.target_w
                stream.height = self.target_h
            except Exception:
                logger.exception("Failed to open %s encoder for camera '%s' at %s; "
                                 "this camera will not be recorded.",
                                 self.codec, cam_name, file_path)
                continue
            self._containers[cam_name] = container
            self._streams[cam_name] = stream
            self.saved_paths[cam_name] = str(file_path)
            logger.info("Recording video for camera '%s' (%s) -> %s",
                        cam_name, self.codec, file_path)

    @property
    def writers(self) -> dict:
        """Back-compat alias: callers only ever used this for truthiness/keys."""
        return self._streams

    def write_frames(self, images: dict[str, np.ndarray]) -> None:
        """Write latest RGB frames to corresponding camera video streams."""
        for cam_name, stream in self._streams.items():
            img = images.get(cam_name)
            if img is None:
                continue
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if img.shape[:2] != (self.target_h, self.target_w):
                img = cv2.resize(img, (self.target_w, self.target_h),
                                 interpolation=cv2.INTER_AREA)
            try:
                # PyAV takes RGB directly - no BGR round-trip, unlike cv2.
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
                for packet in stream.encode(frame):
                    self._containers[cam_name].mux(packet)
            except Exception:
                # A dropped frame must never take down the control loop, and a
                # per-frame traceback at 20 Hz would bury the console - warn once
                # per camera and keep going.
                if cam_name not in self._warned:
                    self._warned.add(cam_name)
                    logger.exception("Error encoding a frame for camera '%s'; "
                                     "further errors for this camera are suppressed.", cam_name)

    def close(self) -> None:
        """Flush encoders and close all containers."""
        for cam_name, stream in list(self._streams.items()):
            container = self._containers.get(cam_name)
            try:
                # Without this flush the tail of the run is lost: the encoder
                # holds frames back for lookahead.
                for packet in stream.encode():
                    container.mux(packet)
            except Exception:
                logger.exception("Error flushing encoder for camera '%s'", cam_name)
            try:
                container.close()
            except Exception:
                logger.exception("Error closing video container for camera '%s'", cam_name)
        self._streams.clear()
        self._containers.clear()
        if self.saved_paths:
            logger.info("Closed video writers. Saved videos: %s", list(self.saved_paths.values()))
