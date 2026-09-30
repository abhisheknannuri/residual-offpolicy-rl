# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Point imageio at an ffmpeg binary, without depending on the machine having one.

`import imageio` resolves its encoder once, so `configure_ffmpeg()` must be
called BEFORE that import - hence a tiny module of its own rather than a
function buried in a recorder.

Order of preference, and why:

1. `IMAGEIO_FFMPEG_EXE` if the user set it - an explicit choice always wins.
2. **The binary bundled inside `imageio-ffmpeg`.** It ships in the wheel, so it
   is installed by `uv sync`, pinned by `uv.lock`, and present on any machine
   that installed this project. On this repo's env that is a static ffmpeg 7.0.2.
3. A system `ffmpeg` on PATH, as a last resort.

An earlier version had 2 and 3 the other way round, on the grounds that
`get_ffmpeg_exe()` "can fail to launch in fresh environments and then blocks on
a network download". Measured 2026-09-30: it returns a path inside the venv in
2.3 ms and downloads nothing, because imageio-ffmpeg 0.6.0 vendors the binary.
Preferring PATH instead meant picking up whatever ffmpeg happened to be there -
on this laptop that was a **conda** env's copy, which quietly made a conda
install a prerequisite of a project that is otherwise conda-free. It also picked
the system's ffmpeg 4.4.2 over the bundled 7.0.2.

If the bundled binary really is unusable on some machine, set
`IMAGEIO_FFMPEG_EXE` explicitly there; that is rule 1 and it still wins.
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)


def configure_ffmpeg() -> str | None:
    """Set `IMAGEIO_FFMPEG_EXE` if it is not already set. Returns the path used."""
    existing = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if existing:
        return existing

    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled):
            os.environ["IMAGEIO_FFMPEG_EXE"] = bundled
            return bundled
    except Exception:
        logger.debug("imageio-ffmpeg's bundled binary is unavailable; falling back to PATH",
                     exc_info=True)

    system = shutil.which("ffmpeg")
    if system:
        os.environ["IMAGEIO_FFMPEG_EXE"] = system
        return system

    logger.warning(
        "No ffmpeg found - neither imageio-ffmpeg's bundled binary nor one on PATH. "
        "Video writing will fail. Install imageio-ffmpeg, or set IMAGEIO_FFMPEG_EXE."
    )
    return None
