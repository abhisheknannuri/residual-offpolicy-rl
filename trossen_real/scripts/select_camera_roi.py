# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Helper for picking a per-camera ROI (`CameraDevice.roi` in a station
config), for cropping a wrist/other camera to a task-relevant region before
it gets resized down to `cameras.resolution` (see `camera_manager.py`).

There's no dedicated "LeRobot dataset ROI calibration" tool to reach for
here (checked - no such utility ships with this repo's vendored
`deps/lerobot`, and none is documented upstream that we could verify) -
this script is a small, self-contained substitute built directly on the
`opencv-python` dependency this repo already has (`cv2.selectROI()`, an
interactive click-and-drag box selector that ships with OpenCV itself, no
new dependency needed).

IMPORTANT: calibrate against a LIVE camera frame at its NATIVE capture
resolution (640x480 - see `_RealSenseCamera._CAPTURE_RESOLUTION`), NOT a
frame pulled from an already-recorded dataset - recorded frames have
already been cropped/resized down to `cameras.resolution` (e.g. 256x256),
so pixel coordinates measured on them don't correspond to the native-frame
ROI box this config field expects.

Two ways to use it:

  1. Interactive (needs a display - X11 forwarding over SSH, or run
     directly on the robot workstation): connects to the named camera,
     grabs one live frame, opens an OpenCV window where you click-and-drag
     a box around the region of interest, press ENTER/SPACE to confirm (or
     `c` to cancel) - see https://docs.opencv.org/4.x/da/dc9/tutorial_py_trackbar.html
     for the underlying widget. Prints the resulting
     `roi: [x_min, y_min, x_max, y_max]` as a ready-to-paste YAML snippet.

         python -m trossen_real.scripts.select_camera_roi \\
             --config trossen_station2_single --camera cam_left_wrist --interactive

  2. Headless (no display needed - e.g. developing over a plain SSH shell):
     grabs one live frame and saves it to disk
     (`<out-dir>/<camera>_raw_640x480.png`) so you can open it in any image
     viewer/VS Code and read off pixel coordinates yourself (hovering in
     VS Code's image preview shows the cursor position) - this is how the
     `cam_left_wrist` ROI `[252, 207, 582, 376]` in
     `trossen_station2_single.yaml` was actually determined. Once you have
     candidate coordinates, pass them back with `--roi` to save an
     annotated preview (raw frame with the box drawn on it, plus what the
     final cropped+resized frame will actually look like) BEFORE
     committing them to the station config - iterate until the preview
     looks right, then paste the printed YAML snippet in.

         python -m trossen_real.scripts.select_camera_roi \\
             --config trossen_station2_single --camera cam_left_wrist
         # inspect the saved raw frame, pick candidate x_min,y_min,x_max,y_max, then:
         python -m trossen_real.scripts.select_camera_roi \\
             --config trossen_station2_single --camera cam_left_wrist \\
             --roi 252 207 582 376

Run (from the repo root, inside the resfit venv, needs `pyrealsense2` and
the camera physically connected - this is NOT the mock feed path):

    python -m trossen_real.scripts.select_camera_roi --config <name> --camera <name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from trossen_real.cameras.camera_manager import HAS_REALSENSE, _RealSenseCamera
from trossen_real.config import load_station_config


def _grab_one_frame(serial: str, fps: int) -> np.ndarray:
    """One live frame at the native 640x480 capture resolution, no crop/resize."""
    cam = _RealSenseCamera(serial, resolution=_RealSenseCamera._CAPTURE_RESOLUTION, fps=fps, roi=None)
    try:
        for _ in range(5):  # a couple of warm-up reads - first frame(s) can be dark/stale
            ok, frame = cam.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read a frame from camera serial={serial}.")
        return frame
    finally:
        cam.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Station config name (must list --camera under cameras.devices)")
    parser.add_argument("--camera", required=True, help="Camera device name (e.g. cam_left_wrist) - must have a real serial set")
    parser.add_argument(
        "--roi", type=int, nargs=4, default=None, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help="Preview this exact ROI (pixel coords in the 640x480 native frame) without opening an interactive window",
    )
    parser.add_argument("--interactive", action="store_true", help="Open an OpenCV window to click-and-drag select the ROI (needs a display)")
    parser.add_argument("--out-dir", default="/tmp/roi_calibration", help="Where to save preview images (default /tmp/roi_calibration)")
    args = parser.parse_args()

    if not HAS_REALSENSE:
        raise RuntimeError("pyrealsense2 is not installed in this environment - can't grab a live camera frame.")

    config = load_station_config(args.config)
    dev = next((d for d in config.cameras.devices if d.name == args.camera), None)
    if dev is None:
        raise ValueError(f"No camera named '{args.camera}' in {args.config} - available: {[d.name for d in config.cameras.devices]}")
    if not dev.serial:
        raise ValueError(f"Camera '{args.camera}' has no serial set in {args.config} (mock slot) - nothing to calibrate against.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to camera '{args.camera}' (serial={dev.serial})...")
    frame = _grab_one_frame(dev.serial, config.cameras.fps)
    cap_h, cap_w = frame.shape[:2]
    print(f"Grabbed one live frame at native capture resolution {cap_w}x{cap_h} (w x h).")

    raw_path = out_dir / f"{args.camera}_raw_{cap_w}x{cap_h}.png"
    cv2.imwrite(str(raw_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Saved raw frame to {raw_path} - open it in any image viewer/VS Code to read off pixel coordinates if needed.")

    roi = args.roi
    if args.interactive:
        print("Opening interactive ROI selector - click-and-drag a box, then press ENTER/SPACE to confirm (or 'c' to cancel).")
        x, y, w, h = cv2.selectROI(f"Select ROI - {args.camera}", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), showCrosshair=True)
        cv2.destroyAllWindows()
        if w == 0 or h == 0:
            print("Selection cancelled (zero-size box) - nothing to report.")
            return
        roi = [x, y, x + w, y + h]

    if roi is None:
        print(
            "\nNo --roi/--interactive given - just saved the raw frame for manual inspection. "
            "Once you have candidate coordinates, re-run with --roi X_MIN Y_MIN X_MAX Y_MAX to preview them."
        )
        return

    x_min, y_min, x_max, y_max = roi
    if not (0 <= x_min < x_max <= cap_w and 0 <= y_min < y_max <= cap_h):
        raise ValueError(
            f"roi={roi} is out of bounds for the native capture resolution {cap_w}x{cap_h} (w x h)."
        )

    annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    annotated_path = out_dir / f"{args.camera}_raw_with_roi_box.png"
    cv2.imwrite(str(annotated_path), annotated)

    cropped = frame[y_min:y_max, x_min:x_max]
    target_h, target_w = config.cameras.resolution
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    preview_path = out_dir / f"{args.camera}_cropped_{target_w}x{target_h}_preview.png"
    cv2.imwrite(str(preview_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

    print(f"\nSaved preview images:")
    print(f"  {annotated_path}  (raw frame with the ROI box drawn on it)")
    print(f"  {preview_path}  (what the actual recorded frame will look like: crop -> resize to {target_w}x{target_h})")
    print(f"\nIf this looks right, paste this into {args.config}.yaml under cameras.devices -> {args.camera}:\n")
    print(f"    - name: {args.camera}")
    print(f"      serial: \"{dev.serial}\"")
    print(f"      roi: [{x_min}, {y_min}, {x_max}, {y_max}]")


if __name__ == "__main__":
    main()
