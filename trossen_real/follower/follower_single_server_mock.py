# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock follower server - identical HTTP API, no robot.

Run it exactly like the real one:

    python -m trossen_real.follower.follower_single_server_mock \\
        --config trossen_station3_single --port 5060

Why this exists: building the OFFLINE replay buffer does not logically need a
robot - `_populate_offline_buffer()` reads the dataset and queries
`policy_server.py` for base actions. But it reaches the policy client through
`env.policy`, and `get_envs()` connects the follower, cameras and leader before
building the env (`train_residual_td3.py:476-494`). So the whole pipeline refuses
to start without hardware. This stands in for the arm so the offline path -
single AND distributed - can be exercised at a desk.

It does NOT simulate anything. Joint positions move instantly to whatever is
commanded, velocities/efforts/accelerations are zero, and the EE pose is a fixed
free-floating default. That is enough for the offline buffer, which never reads
the follower state for its transitions, and enough to let the online loop START -
but an online rollout against this is meaningless, because no action has any
effect on anything.

HOW THE API IS KEPT IDENTICAL
-----------------------------
This module imports the real server's Flask `app` and every route from
`follower_single_server`, and swaps ONLY `state.hardware`. There is no second
copy of the route table to drift out of sync - a route added to the real server
is automatically present here.

`trossen_real/arm_driver.py` already has a `MockArm`, but it is only selected
when the trossen_arm SDK is missing (`HAS_TROSSEN_SDK`). The SDK is installed on
this laptop, so the real path is taken and it tries to reach the arm's IP. This
forces the mock regardless.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from trossen_real.config import StationConfig, load_station_config
from trossen_real.follower import follower_single_server as real
from trossen_real.log_setup import setup_logging

logger = logging.getLogger(__name__)

# Matches arm_driver.MockArm's free-floating default: [x, y, z, ax, ay, az].
_MOCK_POSE = np.array([0.3, 0.0, 0.3, 0.0, 1.57, 0.0], dtype=np.float32)
_MOCK_GRIPPER_OPEN = 0.044


class MockFollowerSingle:
    """Satisfies every `state.hardware.*` call the real routes make.

    The route table calls exactly these ten; verified by grepping
    `state.hardware.` in follower_single_server.py. If a route is added that
    needs an eleventh, this class raises AttributeError loudly rather than
    returning something plausible.
    """

    def __init__(self, config: StationConfig) -> None:
        self.config = config
        self.sides = ["single"]
        # Plain attribute, not a property: ServerState.connected (server:108) reads
        # `self.hardware.connected` directly - missed by grepping `state.hardware.`,
        # found by running it.
        self.connected = False
        # 6 arm joints + gripper. Commands land here verbatim.
        self._q = np.zeros(7, dtype=np.float32)
        self._q[6] = _MOCK_GRIPPER_OPEN
        self._pose = _MOCK_POSE.copy()

    # -- lifecycle --------------------------------------------------------
    def connect(self) -> None:
        self.connected = True
        logger.warning("[MOCK] follower 'connected' - no robot is involved.")

    def disconnect(self, skip_staged_position: bool = False) -> None:
        # Plain attribute, not a property: ServerState.connected (server:108) reads
        # `self.hardware.connected` directly - missed by grepping `state.hardware.`,
        # found by running it.
        self.connected = False

    def health(self) -> dict[str, bool]:
        return {side: self.connected for side in self.sides}

    def using_mock_hardware(self) -> bool:
        return True

    # -- state ------------------------------------------------------------
    def get_follower_state(self) -> dict[str, np.ndarray]:
        """Per side: 7D = pose (6) + gripper (1), matching the real signature."""
        return {
            side: np.concatenate([self._pose, [self._q[6]]]).astype(np.float32)
            for side in self.sides
        }

    def get_follower_joint_state(self, side: str) -> dict[str, np.ndarray]:
        """q/dq/efforts/accelerations, 7D each. Only q is meaningful here."""
        z = np.zeros(7, dtype=np.float32)
        return {
            "q": self._q.copy(),
            "dq": z.copy(),
            "efforts": z.copy(),
            "accelerations": z.copy(),
        }

    # -- motion -----------------------------------------------------------
    def goal_ee(self, side, arr, goal_time=None, blocking=False) -> None:
        arr = np.asarray(arr, dtype=np.float32)
        self._pose = arr[:6].copy()
        self._q[6] = float(arr[6])

    def goal_joint(self, side, joint7, goal_time=None, blocking=False) -> None:
        self._q = np.asarray(joint7, dtype=np.float32).copy()

    def set_gripper_position(self, side, position, goal_time=None, blocking=False) -> None:
        self._q[6] = float(position)

    def reset(self, rng=None, pose_to_reach=None) -> None:
        staged = getattr(self.config.reset.get("single", None), "staged_joint_positions", None)
        if staged is not None:
            self._q = np.asarray(staged, dtype=np.float32).copy()
        else:
            self._q = np.zeros(7, dtype=np.float32)
            self._q[6] = _MOCK_GRIPPER_OPEN
        if pose_to_reach and "single" in pose_to_reach:
            self._pose = np.asarray(pose_to_reach["single"], dtype=np.float32)[:6].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Station config name under trossen_real/configs/ (arm_mode: single)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5060)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--auto-connect", action="store_true",
                        help="Connect immediately instead of waiting for POST /connect_to_robot")
    args = parser.parse_args()

    setup_logging(f"follower_server_MOCK_{args.config}_port{args.port}")

    config = load_station_config(args.config)
    if config.arm_mode != "single":
        raise ValueError(
            f"follower_single_server_mock.py only supports arm_mode: single "
            f"(got '{config.arm_mode}' from '{args.config}')."
        )
    real.state.config = config

    if args.auto_connect:
        real.state.hardware = MockFollowerSingle(config)
        real.state.hardware.connect()

    logger.warning(
        "=" * 70 + "\n"
        "  MOCK FOLLOWER SERVER - no robot. Joint commands are stored and echoed\n"
        "  back; nothing moves. Fine for building the OFFLINE buffer; an online\n"
        "  rollout against this is meaningless.\n" + "=" * 70
    )
    real.app.run(host=args.host, port=args.port, debug=args.debug,
                 threaded=True, use_reloader=False)


# `/connect_to_robot` on the real server constructs TrossenFollowerSingle. Point
# it at the mock instead - this is the only behavioural difference between the
# two servers, and doing it by substitution (rather than copying the routes)
# is what keeps the APIs identical.
real.TrossenFollowerSingle = MockFollowerSingle  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
