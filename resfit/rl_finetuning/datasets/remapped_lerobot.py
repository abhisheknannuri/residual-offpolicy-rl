# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""On-the-fly feature-key remapping for LeRobot datasets.

Some datasets store the same demonstrations under non-standard feature names
(e.g. the SARM robosuite Can v2.1 dataset uses ``agentview-images-rgb`` /
``state`` / ``actions`` instead of the resfit-standard
``observation.images.agentview`` / ``observation.state`` / ``action``).

Rather than rewrite the dataset files or change resfit throughout, this thin
subclass renames the keys *at load time* — in ``__getitem__`` (per sample) and in
``meta.stats`` (so normalization finds the standard names) — and derives
``next.done`` (True on each episode's last frame) when the dataset lacks it.
"""

from __future__ import annotations

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Preset maps: {source dataset key -> resfit-standard key}. Select via
# ``offline_data.remap_preset``.
REMAP_PRESETS: dict[str, dict[str, str]] = {
    # SARM robosuite PickPlaceCan, LeRobot v2.1 (non-standard feature names).
    "sarm_v21": {
        "agentview-images-rgb": "observation.images.agentview",
        "robot0-eye-in-hand-images-rgb": "observation.images.robot0_eye_in_hand",
        "state": "observation.state",
        "actions": "action",
    },
}


class RemappedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that renames feature keys to resfit-standard names on the fly.

    Args (in addition to LeRobotDataset's):
        key_map: {source_key -> target_key}. Applied to each sampled item and to
            ``meta.stats``. Non-present keys are ignored.
        add_next_done: when True and ``next.done`` is absent, derive it as
            ``frame_index == episode_length - 1`` (the demo's terminal frame).
    """

    def __init__(self, *args, key_map: dict[str, str] | None = None, add_next_done: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_map = dict(key_map or {})
        self.add_next_done = add_next_done

        # Remap normalization statistics so downstream code finds e.g. "action"
        # and "observation.state" even though the dataset stores "actions"/"state".
        stats = getattr(self.meta, "stats", None)
        if stats:
            for src, dst in self.key_map.items():
                if src in stats and dst not in stats:
                    stats[dst] = stats[src]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        for src, dst in self.key_map.items():
            if src in item and dst not in item:
                item[dst] = item.pop(src)
        if self.add_next_done and "next.done" not in item:
            ep = int(item["episode_index"].item())
            length = int(self.meta.episodes[ep]["length"])
            fi = int(item["frame_index"].item())
            item["next.done"] = torch.tensor(fi == length - 1, dtype=torch.bool)
        return item
