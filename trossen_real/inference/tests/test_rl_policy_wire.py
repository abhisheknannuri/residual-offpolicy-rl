# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""The RL paths must honour `policy_image_encoding` on the wire.

Threading a config value through four files is exactly the kind of change that
looks done and is not: miss one call site and that path silently keeps sending
512 KB of raw pixels while the config says `jpeg`. These tests assert on the
request body that would actually be sent.

Two call sites are covered:

* online  - `TrossenResidualEnv._query_base_action()` via `build_observation()`
* offline - `_populate_offline_buffer()` via `_dataset_sample_to_json_obs()`,
  which is a closure inside `train_residual_td3.py` and is tested through the
  same `encode_image()` contract it calls.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from trossen_real.inference.policy_client import (
    IMAGE_ENCODINGS,
    build_observation,
    encode_image,
)

IMAGE_KEYS = ["observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]


@pytest.fixture
def follower_state():
    return {
        "q": [0.1] * 7, "pose": [0.0] * 6, "gripper_pos": 0.04,
        "dq": [0.0] * 7, "efforts": [0.0] * 7, "accelerations": [0.0] * 7,
    }


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    base = np.zeros((256, 256, 3), np.uint8)
    base[:, :, 1] = np.linspace(0, 255, 256, dtype=np.uint8)[None, :]
    noisy = np.clip(base.astype(int) + rng.integers(-8, 8, base.shape), 0, 255).astype(np.uint8)
    return {"cam_left_wrist": noisy, "cam_right_wrist": noisy}


def _body_bytes(obs: dict) -> int:
    return sum(len(obs[k]["data_b64"]) for k in IMAGE_KEYS)


@pytest.mark.parametrize("encoding", IMAGE_ENCODINGS)
def test_build_observation_marks_the_encoding(follower_state, images, encoding):
    obs = build_observation(follower_state, images, IMAGE_KEYS, encoding)
    for key in IMAGE_KEYS:
        if encoding == "raw":
            # Absent, not "raw": a server predating compression must still
            # understand the body.
            assert "encoding" not in obs[key]
        else:
            assert obs[key]["encoding"] == encoding


def test_compression_actually_shrinks_the_request(follower_state, images):
    raw = _body_bytes(build_observation(follower_state, images, IMAGE_KEYS, "raw"))
    zlib_ = _body_bytes(build_observation(follower_state, images, IMAGE_KEYS, "zlib"))
    jpeg = _body_bytes(build_observation(follower_state, images, IMAGE_KEYS, "jpeg"))
    assert zlib_ < raw, "zlib did not shrink the body"
    assert jpeg * 5 < raw, f"jpeg should be >5x smaller, got {raw / jpeg:.1f}x"


def test_jpeg_quality_is_honoured(follower_state, images):
    hi = _body_bytes(build_observation(follower_state, images, IMAGE_KEYS, "jpeg", 95))
    lo = _body_bytes(build_observation(follower_state, images, IMAGE_KEYS, "jpeg", 60))
    assert lo < hi, "lowering jpeg_quality did not shrink the body"


def test_raw_is_byte_exact(follower_state, images):
    """The default path must still be the original bytes, unmodified."""
    obs = build_observation(follower_state, images, IMAGE_KEYS, "raw")
    got = np.frombuffer(base64.b64decode(obs[IMAGE_KEYS[0]]["data_b64"]), np.uint8)
    assert np.array_equal(got.reshape(256, 256, 3), images["cam_left_wrist"])


def test_the_online_env_passes_its_encoding_through(monkeypatch, follower_state, images):
    """`TrossenResidualEnv._query_base_action` must use the env's configured
    encoding, not the `build_observation` default. Constructing the real env
    needs hardware, so this checks the call the method makes."""
    import trossen_real.rl.real_residual_env as env_mod

    seen = {}

    def spy(fs, imgs, keys, encoding="raw", quality=95):
        seen["encoding"] = encoding
        seen["quality"] = quality
        return build_observation(fs, imgs, keys, encoding, quality)

    monkeypatch.setattr(env_mod, "build_observation", spy)

    class _Env:
        image_keys = IMAGE_KEYS
        policy_image_encoding = "jpeg"
        policy_jpeg_quality = 80
        device = "cpu"

        class policy:
            @staticmethod
            def predict(obs):
                return np.zeros(7, dtype=np.float32)

        class action_scaler:
            @staticmethod
            def scale(x):
                return x

    env_mod.TrossenResidualEnv._query_base_action(_Env(), follower_state, images)
    assert seen == {"encoding": "jpeg", "quality": 80}


def test_offline_helper_contract(images):
    """`_dataset_sample_to_json_obs` calls `encode_image(hwc, encoding, quality)`.
    Pin that signature so a reordered argument is caught here rather than by a
    55,177-frame population run sending raw bytes."""
    payload = encode_image(images["cam_left_wrist"], "jpeg", 80)
    assert payload["encoding"] == "jpeg"
    smaller = encode_image(images["cam_left_wrist"], "jpeg", 40)
    assert len(smaller["data_b64"]) < len(payload["data_b64"])
