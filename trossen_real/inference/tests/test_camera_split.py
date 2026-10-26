"""RL cameras vs base-policy (ACT) cameras: `rl_camera` / `policy_camera`."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from resfit.rl_finetuning.config.residual_td3 import camera_keys

WRISTS = ["observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
ALL3 = [*WRISTS, "observation.images.cam_specialist"]


def test_camera_keys_default_is_rl_camera():
    rl, pol = camera_keys(OmegaConf.create({"rl_camera": WRISTS, "policy_camera": None}))
    assert rl == WRISTS and pol == WRISTS


def test_camera_keys_absent_field_is_rl_camera():
    # sim configs have no policy_camera at all
    rl, pol = camera_keys(OmegaConf.create({"rl_camera": "observation.images.agentview"}))
    assert rl == pol == ["observation.images.agentview"]


def test_camera_keys_split():
    rl, pol = camera_keys(OmegaConf.create({"rl_camera": ALL3, "policy_camera": WRISTS}))
    assert rl == ALL3 and pol == WRISTS


def _query(env_attrs, monkeypatch):
    import trossen_real.rl.real_residual_env as env_mod

    seen = {}

    def spy(fs, imgs, keys, encoding="raw", quality=95):
        seen["keys"] = list(keys)
        return {}

    monkeypatch.setattr(env_mod, "build_observation", spy)

    class _Env:
        policy_image_encoding = "raw"
        policy_jpeg_quality = 95
        device = "cpu"

        class policy:
            @staticmethod
            def predict(obs):
                return np.zeros(7, dtype=np.float32)

        class action_scaler:
            @staticmethod
            def scale(x):
                return x

    env = _Env()
    for k, v in env_attrs.items():
        setattr(env, k, v)
    env_mod.TrossenResidualEnv._query_base_action(env, {"q": [0.0] * 7}, {})
    return seen["keys"]


def test_env_sends_only_policy_cameras_to_act(monkeypatch):
    assert _query({"image_keys": ALL3, "policy_image_keys": WRISTS}, monkeypatch) == WRISTS


def test_env_without_policy_cameras_sends_rl_cameras(monkeypatch):
    assert _query({"image_keys": WRISTS}, monkeypatch) == WRISTS
