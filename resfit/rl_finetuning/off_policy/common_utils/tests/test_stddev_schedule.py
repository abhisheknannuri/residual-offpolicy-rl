"""Exploration-noise schedule: config resolution + utils.schedule parsing.

Regression for a bug where stddev_max/min/step from yaml/CLI were silently
ignored: stddev_schedule was computed in __post_init__ from the dataclass
defaults BEFORE Hydra merged the yaml, so every run used linear(0.05,0.05,300000).

Run:  <python> -m pytest resfit/rl_finetuning/off_policy/common_utils/tests/ -v
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from resfit.rl_finetuning.config.residual_td3 import ResidualTD3AlgoConfig
from resfit.rl_finetuning.off_policy.common_utils import utils

PER_JOINT = [0.015, 0.015, 0.015, 0.03, 0.03, 0.03, 0.0125]


@dataclass
class _Root:
    algo: ResidualTD3AlgoConfig = field(default_factory=ResidualTD3AlgoConfig)


def _merged(algo_overrides: dict):
    return OmegaConf.merge(OmegaConf.structured(_Root), {"algo": algo_overrides})


# --- config resolution --------------------------------------------------------
def test_defaults_unchanged():
    assert _merged({}).algo.stddev_schedule == "linear(0.05,0.05,300000)"


def test_yaml_values_reach_schedule():
    cfg = _merged({"stddev_max": PER_JOINT, "stddev_min": PER_JOINT, "stddev_step": 15000})
    np.testing.assert_allclose(utils.schedule(cfg.algo.stddev_schedule, 0), PER_JOINT, rtol=1e-6)
    np.testing.assert_allclose(utils.schedule(cfg.algo.stddev_schedule, 10**6), PER_JOINT, rtol=1e-6)


def test_scalar_decay_from_yaml():
    cfg = _merged({"stddev_max": 0.05, "stddev_min": 0.01, "stddev_step": 1000})
    s = cfg.algo.stddev_schedule
    assert utils.schedule(s, 0) == pytest.approx(0.05)
    assert utils.schedule(s, 500) == pytest.approx(0.03)
    assert utils.schedule(s, 5000) == pytest.approx(0.01)


def test_explicit_schedule_still_wins():
    cfg = _merged({"stddev_max": 0.5, "stddev_schedule": "[0.01, 0.02]"})
    np.testing.assert_allclose(utils.schedule(cfg.algo.stddev_schedule, 0), [0.01, 0.02])


def test_resolved_container_has_no_interpolation():
    # checkpoint.py / wandb save the config with resolve=True.
    cfg = _merged({"stddev_max": PER_JOINT})
    saved = OmegaConf.to_container(cfg, resolve=True)["algo"]["stddev_schedule"]
    assert "${" not in saved and saved.startswith("linear([0.015")


# --- utils.schedule parsing ---------------------------------------------------
@pytest.mark.parametrize(
    ("schdl", "step", "expected"),
    [
        ("0.1", 7, 0.1),
        ("linear(0.05,0.05,300000)", 123, 0.05),
        ("linear(1.0, 0.0, 100)", 25, 0.75),
        ("linear(1.0, 0.0, 100)", 1000, 0.0),
        ("step_linear(1.0, 0.5, 10, 0.0, 10)", 5, 0.75),
        ("step_linear(1.0, 0.5, 10, 0.0, 10)", 15, 0.25),
    ],
)
def test_scalar_forms(schdl, step, expected):
    out = utils.schedule(schdl, step)
    assert isinstance(out, float)
    assert out == pytest.approx(expected)


def test_list_forms():
    np.testing.assert_allclose(utils.schedule(str(PER_JOINT), 0), PER_JOINT, rtol=1e-6)
    out = utils.schedule("linear([1.0, 0.2], [0.0, 0.2], 10)", 5)
    np.testing.assert_allclose(out, [0.5, 0.2], rtol=1e-6)
    out = utils.schedule("step_linear([1.0, 1.0], [0.5, 0.5], 10, [0.0, 0.1], 10)", 15)
    np.testing.assert_allclose(out, [0.25, 0.3], rtol=1e-6)


def test_unknown_form_raises():
    with pytest.raises(NotImplementedError):
        utils.schedule("cosine(1.0, 0.0, 10)", 0)


# --- per-joint std reaches the sampler ----------------------------------------
def test_per_joint_std_in_truncated_normal():
    torch.manual_seed(0)
    std = torch.tensor(utils.schedule(str(PER_JOINT), 0))
    loc = torch.zeros(20000, 7)
    x = utils.TruncatedNormal(loc, std).sample(clip=None)
    np.testing.assert_allclose(x.std(0).numpy(), PER_JOINT, rtol=0.05)
    x = utils.TruncatedNormal(loc, std).sample(clip=0.02)
    assert x.abs().max().item() <= 0.02 + 1e-6
