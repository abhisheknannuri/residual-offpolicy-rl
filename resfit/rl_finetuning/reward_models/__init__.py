# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""External stage-aware reward-model integration for residual RL.

Provides a synchronous HTTP client (:mod:`reward_client`) that talks to a reward
server (SARM / TCC) and a gym wrapper (:mod:`pbrs_wrapper`) that converts the
server's gated stage predictions into a Potential-Based Reward Shaping (PBRS)
signal for the training environment.
"""

from resfit.rl_finetuning.reward_models.reward_client import RewardModelClient
from resfit.rl_finetuning.reward_models.pbrs_wrapper import StageAwarePBRSWrapper

__all__ = ["RewardModelClient", "StageAwarePBRSWrapper"]
