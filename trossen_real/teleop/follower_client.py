# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Thin synchronous HTTP client for `follower_single_server.py`.

Used by the teleop data-collection app (`app.py`) today, and reusable as-is
by a future RL gym-wrapper - both are just HTTP clients of the same
follower server, per the "one server, adapt the client" design.
"""

from __future__ import annotations

import numpy as np
import requests

from trossen_real.call_stats import track_call


class FollowerClient:
    def __init__(self, base_url: str, request_timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        # Short tag for the live call_stats view - e.g. "follower:5095" -
        # distinguishes multiple follower servers (dual-arm stations) at a
        # glance without needing the full base_url.
        self._stats_client = f"follower:{self.base_url.rsplit(':', 1)[-1]}"

    def _post(self, path: str, json: dict | None = None):
        with track_call(self._stats_client, path):
            r = requests.post(f"{self.base_url}{path}", json=json, timeout=self.request_timeout_s)
            r.raise_for_status()
        return r

    def connect(self) -> None:
        self._post("/connect_to_robot")

    def disconnect(self, skip_staged_position: bool = False) -> None:
        """`skip_staged_position=True`: skip the staged->sleep positioning
        move and disconnect immediately from wherever the arm currently is -
        see `follower_single.py::TrossenFollowerSingle.disconnect()`'s
        docstring for what this opts out of. Default False is identical to
        calling this before the parameter existed."""
        self._post("/disconnect_from_robot", json={"skip_staged_position": skip_staged_position})

    def health(self) -> dict:
        with track_call(self._stats_client, "/health"):
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
        return r.json()

    def check(self) -> bool:
        try:
            return bool(self.health().get("connected", False))
        except Exception:
            return False

    def get_state(self) -> dict:
        """{"pose":[6], "q":[7], "gripper_pos":float, "dq":[7], "efforts":[7], "accelerations":[7]}

        ALWAYS returns both the EE pose AND joint representation together
        (and velocity/effort/acceleration), regardless of the station's
        `command_space` - the follower server doesn't branch on that, so
        this is a good source for auxiliary observations (see
        `control_loop.py::_tick()`/`dataset_recorder.py`'s `extra_obs`).
        """
        return self._post("/getstate").json()

    def get_pose(self) -> np.ndarray:
        return np.asarray(self._post("/getpos").json()["pose"], dtype=np.float32)

    def get_gripper(self) -> float:
        return float(self._post("/get_gripper").json()["gripper"])

    def move_to_ee_pose(
        self,
        pose7: np.ndarray,
        blocking: bool = False,
        min_time_to_move: float | None = None,
    ) -> None:
        """pose7: [x,y,z,ax,ay,az,gripper], absolute target."""
        body = {"arr": np.asarray(pose7, dtype=np.float32).tolist(), "blocking": blocking}
        if min_time_to_move is not None:
            body["min_time_to_move"] = min_time_to_move
        self._post("/pose", json=body)

    def move_to_joint_positions(
        self,
        joint7: np.ndarray,
        blocking: bool = False,
        min_time_to_move: float | None = None,
    ) -> None:
        """joint7: [joint_0..joint_5, gripper], absolute target (`command_space: joint`)."""
        body = {"arr": np.asarray(joint7, dtype=np.float32).tolist(), "blocking": blocking}
        if min_time_to_move is not None:
            body["min_time_to_move"] = min_time_to_move
        self._post("/move_to_joint_positions", json=body)

    def move_gripper(
        self,
        gripper_pos: float,
        blocking: bool = False,
        min_time_to_move: float | None = None,
    ) -> None:
        body = {"gripper_pos": float(gripper_pos), "blocking": blocking}
        if min_time_to_move is not None:
            body["min_time_to_move"] = min_time_to_move
        self._post("/move_gripper", json=body)

    def reset(self, pose_to_reach: np.ndarray | None = None) -> None:
        body = {}
        if pose_to_reach is not None:
            body["pose_to_reach"] = np.asarray(pose_to_reach, dtype=np.float32).tolist()
        self._post("/reset", json=body or None)

    def clear_error(self) -> None:
        self._post("/clearerr")
