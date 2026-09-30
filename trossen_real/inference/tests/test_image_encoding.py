# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""`encode_image()` must shrink the payload without changing what the policy sees.

The failure this guards against is not "compression is broken" - that would be
obvious. It is a silent RGB/BGR swap on one of the compressed paths (cv2 works
in BGR, the wire format is RGB), which would look exactly like "the checkpoint
got worse" and nothing else.

The server's own `_decode_image()` is loaded from `policy_server.py` where it
lives, so this tests the real contract rather than a re-implementation of it. If
that file is not on this machine the tests skip instead of quietly passing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from trossen_real.inference.policy_client import IMAGE_ENCODINGS, encode_image

POLICY_SERVER = Path(
    "/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/"
    "RLRewardResearchWS/GeneralistRewardModels/lerobot/custom_scripts/policy_server.py"
)


@pytest.fixture(scope="module")
def decode_image():
    if not POLICY_SERVER.exists():
        pytest.skip(f"policy_server.py not found at {POLICY_SERVER}")
    spec = importlib.util.spec_from_file_location("_psrv_under_test", POLICY_SERVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_psrv_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod._decode_image


@pytest.fixture
def frame():
    """A frame with structure, not noise: flat regions, an edge, and a saturated
    red patch standing in for the cube - the thing a channel swap would move."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 256, dtype=np.uint8)[None, :]
    img[:128] = img[:128] // 2
    img[40:80, 100:150] = (220, 30, 40)     # RGB red
    return img


@pytest.mark.parametrize("encoding", IMAGE_ENCODINGS)
def test_roundtrip_shape_and_range(decode_image, frame, encoding):
    tensor = decode_image(encode_image(frame, encoding))
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype.is_floating_point
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


@pytest.mark.parametrize("encoding", ["raw", "zlib"])
def test_lossless_encodings_are_bit_exact(decode_image, frame, encoding):
    back = (decode_image(encode_image(frame, encoding)).permute(1, 2, 0).numpy() * 255).round()
    assert np.array_equal(back.astype(np.uint8), frame)


def test_no_channel_swap_on_any_encoding(decode_image, frame):
    """Every encoding must agree with `raw` channel for channel."""
    ref = decode_image(encode_image(frame, "raw"))
    for encoding in IMAGE_ENCODINGS:
        got = decode_image(encode_image(frame, encoding))
        per_channel = [float((got[c] - ref[c]).abs().mean()) for c in range(3)]
        assert max(per_channel) < 0.02, f"{encoding}: channel mismatch {per_channel}"
    # And the red patch must still be red after a JPEG round trip - a swap would
    # keep the per-channel means small overall but move this block to blue.
    jpeg = decode_image(encode_image(frame, "jpeg")).permute(1, 2, 0).numpy()
    patch = jpeg[50:70, 110:140].reshape(-1, 3).mean(axis=0)
    assert patch[0] > patch[2] + 0.3, f"red patch is not red after jpeg: {patch}"


def test_jpeg_is_much_smaller_and_close_enough(decode_image, frame):
    raw = encode_image(frame, "raw")["data_b64"]
    jpeg_payload = encode_image(frame, "jpeg", jpeg_quality=95)
    assert len(jpeg_payload["data_b64"]) * 5 < len(raw)      # at least 5x smaller
    back = decode_image(jpeg_payload).permute(1, 2, 0).numpy() * 255
    mean_err = np.abs(back - frame.astype(float)).mean()
    # The LeRobot dataset stores images as AV1 crf=30, whose own mean abs error
    # on real frames measured ~1.39/255. Staying under that keeps inference no
    # further from the training distribution than training already was.
    assert mean_err < 1.39, f"jpeg q95 mean abs error {mean_err:.2f} exceeds the dataset's own AV1 error"


def test_encoding_key_is_absent_for_raw(frame):
    """A server that predates compression support must still understand us."""
    assert "encoding" not in encode_image(frame, "raw")
    assert encode_image(frame, "zlib")["encoding"] == "zlib"
    assert encode_image(frame, "jpeg")["encoding"] == "jpeg"


def test_unknown_encoding_is_rejected(frame):
    with pytest.raises(ValueError, match="encoding must be one of"):
        encode_image(frame, "webp")


def test_server_rejects_unknown_encoding(decode_image, frame):
    payload = encode_image(frame, "raw")
    payload["encoding"] = "webp"
    with pytest.raises(ValueError, match="unknown image encoding"):
        decode_image(payload)


def test_server_rejects_shape_mismatch(decode_image, frame):
    payload = encode_image(frame, "jpeg")
    payload["shape"] = [128, 128, 3]
    with pytest.raises(ValueError, match="declares"):
        decode_image(payload)


def test_build_observation_end_to_end_with_jpeg(decode_image, frame):
    """The exact shape of the configuration the runbook tells you to use:
    `INFER_IMAGE_ENCODING=jpeg` all the way from camera frames to the tensor the
    policy is fed."""
    from trossen_real.inference.policy_client import build_observation

    follower_state = {
        "q": [0.1] * 7, "pose": [0.0] * 6, "gripper_pos": 0.04,
        "dq": [0.0] * 7, "efforts": [0.0] * 7, "accelerations": [0.0] * 7,
    }
    images = {"cam_right_wrist": frame, "cam_left_wrist": frame}
    keys = ["observation.images.cam_right_wrist", "observation.images.cam_left_wrist"]

    obs = build_observation(follower_state, images, keys, image_encoding="jpeg", jpeg_quality=95)
    assert obs["observation.state"] == [0.1] * 7
    raw_obs = build_observation(follower_state, images, keys)          # default stays raw
    assert "encoding" not in raw_obs[keys[0]]

    for key in keys:
        assert obs[key]["encoding"] == "jpeg"
        assert len(obs[key]["data_b64"]) * 5 < len(raw_obs[key]["data_b64"])
        tensor = decode_image(obs[key])
        assert tensor.shape == (3, 256, 256)
        assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
