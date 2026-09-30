# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Multi-camera capture: one background thread per camera, latest-frame-wins.

Adapted from hil-serl's `serl_robot_infra/trossen_env/camera/rs_capture.py`
+ `multi_video_capture.py`. Cameras run at their own native rate in the
background purely for a smooth live preview; the 20Hz control/record loop
just grabs whatever is currently in `latest_frame` (no blocking, small
timestamp skew accepted).

If `pyrealsense2` is not installed, or a camera slot has no `serial` set in
the station config, that slot falls back to a synthetic mock feed (moving
color bar + camera name + timestamp) so the UI/recording path can be
developed and tested without physical cameras.

Real cameras are captured at a native RealSense resolution (not the
dataset's target resolution, which most sensors don't support natively -
see `_RealSenseCamera`), optionally cropped to a per-camera region of
interest (`CameraDevice.roi`), and resized down to `cameras.resolution` in
software before being handed to the rest of the app.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from trossen_real.config import CameraDevice, StationConfig

logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs

    HAS_REALSENSE = True
except ImportError:
    rs = None
    HAS_REALSENSE = False


class _RealSenseCamera:
    """Blocking single-camera reader (threading is added by `CameraManager`).

    RealSense color sensors only expose a fixed set of native stream
    profiles (e.g. 640x480, 848x480, 1280x720 - confirmed via
    `sensor.get_stream_profiles()`); most cameras (e.g. the D405) have NO
    native 256x256 profile, so requesting the dataset's target resolution
    directly raises `Couldn't resolve requests`. We instead always request
    a widely-supported native resolution (`_CAPTURE_RESOLUTION`) and resize
    each frame down to the configured target resolution in software.

    If `roi` is given (`[x_min, y_min, x_max, y_max]` pixel coords in this
    NATIVE capture resolution - see `CameraDevice.roi`), the frame is
    CROPPED to that box first, THEN resized to the target resolution -
    lets a wrist camera's relevant region fill the frame instead of the
    whole (often mostly-irrelevant) native field of view getting squashed
    down uniformly.
    """

    _CAPTURE_RESOLUTION = (480, 640)  # (h, w) - supported by all current RealSense color sensors

    def __init__(
        self, serial: str, resolution: tuple[int, int], fps: int, roi: list[int] | None = None
    ) -> None:
        self._target_h, self._target_w = resolution
        cap_h, cap_w = self._CAPTURE_RESOLUTION
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            if not (0 <= x_min < x_max <= cap_w and 0 <= y_min < y_max <= cap_h):
                raise ValueError(
                    f"roi={roi} is out of bounds for the native capture resolution "
                    f"{cap_w}x{cap_h} (w x h) - expected 0 <= x_min < x_max <= {cap_w} and "
                    f"0 <= y_min < y_max <= {cap_h}."
                )
        self._roi = roi
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, cap_w, cap_h, rs.format.bgr8, fps)
        self._pipeline.start(cfg)

    def read(self) -> tuple[bool, np.ndarray | None]:
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        color = frames.get_color_frame()
        if not color:
            return False, None
        img = np.asanyarray(color.get_data())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self._roi is not None:
            x_min, y_min, x_max, y_max = self._roi
            img = img[y_min:y_max, x_min:x_max]
        if img.shape[:2] != (self._target_h, self._target_w):
            img = cv2.resize(img, (self._target_w, self._target_h), interpolation=cv2.INTER_AREA)
        return True, img

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            logger.exception("Error while stopping RealSense pipeline; ignoring.")


class _MockCamera:
    """Synthetic frame source: moving bar + name + timestamp, at the configured resolution."""

    def __init__(self, name: str, resolution: tuple[int, int]) -> None:
        self.name = name
        self.h, self.w = resolution
        self._t0 = time.time()

    def read(self) -> tuple[bool, np.ndarray]:
        t = time.time() - self._t0
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Slowly shifting background color so the feed visibly looks "live".
        hue = int((t * 20) % 180)
        bg = cv2.cvtColor(np.full((1, 1, 3), (hue, 180, 60), dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
        img[:, :] = bg
        bar_x = int((np.sin(t) * 0.5 + 0.5) * (self.w - 10))
        cv2.rectangle(img, (bar_x, 0), (bar_x + 10, self.h), (255, 255, 255), -1)
        cv2.putText(img, self.name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{t:6.1f}s (mock)", (4, self.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        return True, img

    def close(self) -> None:
        pass


class CameraManager:
    """Owns up to 4 cameras (real or mock), each on its own background thread.

    Auto-recovers from a stuck/dead RealSense pipeline (the "Frame didn't
    arrive within 1000" `RuntimeError` - a known, usually-transient
    librealsense issue: USB bandwidth/power glitches, a hub hiccup, etc.)
    instead of spinning forever on the same broken pipeline object: after
    `RECONNECT_AFTER_FAILURES` consecutive failed reads, the camera is
    closed and re-opened from scratch. `health()` reflects whether a camera
    has ACTUALLY produced a frame recently (not just whether it was
    successfully constructed at `start()` time) - a silently-dead camera
    (pipeline stuck, reconnect attempts failing) now correctly shows as
    unhealthy instead of staying "green" forever, so `/api/health` (and
    anything polling it, e.g. an RL training loop) can actually detect and
    alert on this instead of only noticing via a wall of stack traces in
    the server log.
    """

    # After this many CONSECUTIVE failed reads (each with its own internal
    # ~1s RealSense timeout, so this is roughly this-many-seconds of no
    # frames), close and recreate the camera's pipeline from scratch rather
    # than continuing to hammer the same stuck one.
    RECONNECT_AFTER_FAILURES = 5
    # Brief pause before attempting a reconnect - avoids hammering the USB
    # bus/controller immediately after a failure that may itself have been
    # caused by a transient bus issue.
    RECONNECT_BACKOFF_S = 1.0
    # `health()` reports a camera as unhealthy if its last successful frame
    # is older than this (or no frame has arrived within this long after
    # `start()` - a startup grace period, not an instant "unhealthy").
    STALE_AFTER_S = 3.0

    def __init__(self, config: StationConfig) -> None:
        self.config = config
        self._cams: dict[str, _RealSenseCamera | _MockCamera] = {}
        self._latest: dict[str, np.ndarray] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._threads: list[threading.Thread] = []
        self._running = False
        self._is_mock: dict[str, bool] = {}
        self._start_time: dict[str, float] = {}
        self._last_frame_time: dict[str, float] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._reconnect_count: dict[str, int] = {}
        self._status_lock = threading.Lock()

    @property
    def camera_names(self) -> list[str]:
        return [d.name for d in self.config.cameras.devices[:4]]

    def start(self) -> None:
        self._running = True
        resolution = self.config.cameras.resolution
        fps = self.config.cameras.fps
        for dev in self.config.cameras.devices[:4]:
            cam, is_mock = self._make_camera(dev, resolution)
            self._cams[dev.name] = cam
            self._is_mock[dev.name] = is_mock
            self._locks[dev.name] = threading.Lock()
            self._latest[dev.name] = np.zeros((*resolution, 3), dtype=np.uint8)
            self._start_time[dev.name] = time.time()
            self._consecutive_failures[dev.name] = 0
            self._reconnect_count[dev.name] = 0
            thread = threading.Thread(target=self._stream_loop, args=(dev.name, fps), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _make_camera(self, dev: CameraDevice, resolution: tuple[int, int]) -> tuple[_RealSenseCamera | _MockCamera, bool]:
        if dev.serial and HAS_REALSENSE:
            try:
                return _RealSenseCamera(dev.serial, resolution, self.config.cameras.fps, roi=dev.roi), False
            except Exception:
                logger.exception("Camera '%s' (serial=%s) failed to connect, falling back to mock.", dev.name, dev.serial)
        elif dev.serial and not HAS_REALSENSE:
            logger.warning("pyrealsense2 not installed - camera '%s' will use a mock feed.", dev.name)
        return _MockCamera(dev.name, resolution), True

    def _reconnect(self, name: str, dev: CameraDevice, resolution: tuple[int, int]) -> None:
        """Close and recreate a stuck real camera's pipeline from scratch.

        Deliberately NEVER falls back to a `_MockCamera` here (unlike the
        INITIAL `_make_camera()` call at `start()` time, which does) - this
        camera's config has a real `serial` set, so `dataset_recorder.py`'s
        `_recorded_camera_names` (computed from config, NOT from
        `is_mock()`) will keep recording frames under this camera's name
        regardless of what's actually producing them. Silently swapping to
        a synthetic mock feed here would mean fabricated frames get
        recorded into the dataset AS IF they were real camera data, with
        nothing anywhere flagging that it happened - a much worse failure
        mode than a camera that's simply (visibly, via `health()`) stale.
        So: on a failed reconnect, keep the OLD (broken) camera object in
        place - `_latest[name]` just keeps serving the last genuinely-real
        frame (stale, but real) until a LATER reconnect attempt succeeds -
        and let `health()`/`get_camera_status()` correctly report this
        camera as unhealthy in the meantime so it's actually visible
        (instead of silently-wrong data with a green status dot).
        """
        logger.warning(
            "Camera '%s' had %d consecutive failed reads - attempting reconnect (attempt #%d)...",
            name, self._consecutive_failures[name], self._reconnect_count[name] + 1,
        )
        old_cam = self._cams[name]
        try:
            old_cam.close()
        except Exception:
            logger.exception("Error closing camera '%s' before reconnect; continuing anyway.", name)
        time.sleep(self.RECONNECT_BACKOFF_S)
        try:
            new_cam = _RealSenseCamera(dev.serial, resolution, self.config.cameras.fps, roi=dev.roi)
        except Exception:
            logger.exception(
                "Camera '%s' reconnect FAILED (serial=%s) - will keep retrying every ~%.0fs. "
                "Check the physical connection/USB port/hub power. Still reporting the last "
                "genuinely-real frame (now stale) rather than fabricating synthetic data.",
                name, dev.serial, self.RECONNECT_AFTER_FAILURES / max(self.config.cameras.fps, 1) + self.RECONNECT_BACKOFF_S,
            )
            with self._status_lock:
                self._consecutive_failures[name] = 0  # reset the counter so we wait a full cycle before retrying again
                self._reconnect_count[name] += 1
            return
        self._cams[name] = new_cam
        with self._status_lock:
            self._consecutive_failures[name] = 0
            self._reconnect_count[name] += 1
        logger.info("Camera '%s' reconnected successfully.", name)

    def _stream_loop(self, name: str, fps: int) -> None:
        period = 1.0 / max(fps, 1)
        dev = next(d for d in self.config.cameras.devices[:4] if d.name == name)
        resolution = self.config.cameras.resolution
        while self._running:
            t_start = time.time()
            cam = self._cams[name]
            try:
                ok, frame = cam.read()
                if ok and frame is not None:
                    with self._locks[name]:
                        self._latest[name] = frame
                    with self._status_lock:
                        self._last_frame_time[name] = time.time()
                        self._consecutive_failures[name] = 0
                else:
                    with self._status_lock:
                        self._consecutive_failures[name] += 1
            except Exception:
                with self._status_lock:
                    self._consecutive_failures[name] += 1
                # Only log the full traceback on the FIRST failure and then
                # again right before a reconnect attempt - logging it every
                # single tick (as this used to) just floods the server log
                # with an identical stack trace once per second without
                # adding any new information.
                if self._consecutive_failures[name] == 1:
                    logger.exception("Camera '%s' read failed - will retry, and reconnect after %d consecutive failures.",
                                      name, self.RECONNECT_AFTER_FAILURES)

            if (
                not self._is_mock.get(name, True)
                and self._consecutive_failures[name] >= self.RECONNECT_AFTER_FAILURES
            ):
                self._reconnect(name, dev, resolution)

            elapsed = time.time() - t_start
            time.sleep(max(0.0, period - elapsed))

    def get_latest_frame(self, name: str) -> np.ndarray | None:
        lock = self._locks.get(name)
        if lock is None:
            return None
        with lock:
            frame = self._latest.get(name)
            return None if frame is None else frame.copy()

    def get_all_latest(self) -> dict[str, np.ndarray]:
        return {name: self.get_latest_frame(name) for name in self.camera_names}

    def is_mock(self, name: str) -> bool:
        return self._is_mock.get(name, True)

    def health(self) -> dict[str, bool]:
        """Per camera: True if it's ACTUALLY produced a frame recently
        (within `STALE_AFTER_S`), not just whether it was constructed.

        Mock cameras always report True (they can't "die" the way a real
        RealSense pipeline can - see `_MockCamera.read()`, which always
        succeeds). A real camera stuck reconnecting (see `_reconnect()`,
        which deliberately keeps retrying the REAL camera forever rather
        than ever silently falling back to a mock feed at runtime) will
        correctly show False here until it's actually producing frames
        again, unlike the previous behavior (`name in self._cams`, always
        True once `start()` ran regardless of whether frames were ever
        arriving).
        """
        now = time.time()
        result: dict[str, bool] = {}
        for name in self.camera_names:
            if self._is_mock.get(name, True):
                result[name] = True
                continue
            last = self._last_frame_time.get(name)
            if last is None:
                # No frame yet - only unhealthy once past the startup grace period.
                result[name] = (now - self._start_time.get(name, now)) < self.STALE_AFTER_S
            else:
                result[name] = (now - last) < self.STALE_AFTER_S
        return result

    def get_camera_status(self) -> dict[str, dict]:
        """Richer per-camera diagnostics than `health()`'s plain bool - for
        anything that wants to actually monitor/alert on camera health
        (e.g. an RL training loop polling `/api/health`), not just show a
        status dot in the UI."""
        now = time.time()
        healthy = self.health()
        status: dict[str, dict] = {}
        for name in self.camera_names:
            last = self._last_frame_time.get(name)
            status[name] = {
                "healthy": healthy[name],
                "is_mock": self._is_mock.get(name, True),
                "last_frame_age_s": (now - last) if last is not None else None,
                "consecutive_failures": self._consecutive_failures.get(name, 0),
                "reconnect_count": self._reconnect_count.get(name, 0),
            }
        return status

    def stop(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=2.0)
        for cam in self._cams.values():
            cam.close()
        self._cams.clear()
        self._threads.clear()
