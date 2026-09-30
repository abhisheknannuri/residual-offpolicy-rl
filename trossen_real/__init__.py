# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Trossen teleop client: single-station Flask app for teleoperated data collection.

Run the follower server first (owns the robot connection), in its own terminal::

    python -m trossen_real.follower.follower_single_server --config trossen_station2_single --port 5060

Then the teleop orchestrator app, in another terminal::

    python -m trossen_real.teleop.app --port 5050

Then open http://localhost:5050 in a browser, pick a station config from
``trossen_real/configs/``, and click Connect.
"""
