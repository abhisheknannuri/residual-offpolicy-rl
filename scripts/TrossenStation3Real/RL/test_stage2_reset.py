#!/usr/bin/env python3
"""Try the stage-2 reset (trossen_real/rl/stage2_reset.py) on the robot.

Needs follower_single_server.py running for the station. Same pieces the RL
trainer uses: FollowerClient (here its Stage2ResetFollower subclass) and, with
--leader, TrossenSingleLeader synced in position mode.

    # 1. no motion: print the targets that would be sampled
    .venv/bin/python scripts/TrossenStation3Real/RL/test_stage2_reset.py --mode joint --dry-run --n 10
    # 2. real resets, one after another (Enter between them)
    .venv/bin/python scripts/TrossenStation3Real/RL/test_stage2_reset.py --mode joint --n 5
    .venv/bin/python scripts/TrossenStation3Real/RL/test_stage2_reset.py --mode ee --n 5 --leader

After every reset it prints the target and where the arm actually ended up.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# trossen_real is not an installed package (WORKSPACE_CLEANUP_PLAN.md): run from
# anywhere by putting the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from trossen_real.config import load_station_config  # noqa: E402
from trossen_real.rl import stage2_reset as s2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="trossen_station3_single")
    ap.add_argument("--mode", choices=["joint", "ee", "none"], default="joint")
    ap.add_argument("--n", type=int, default=3, help="Number of resets.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--leader", action="store_true",
                    help="Connect the leader and sync it after each reset, as the infer app / RL env do "
                         "(gripper at the commanded close on the cube).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Sample and print targets only. 'ee' mode reads the current EE pose (no motion).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_station_config(args.station)
    print(f"STAGE2_JOINTS      {np.round(s2.STAGE2_JOINTS, 4).tolist()}")
    print(f"GRIPPER_CUBE_CLOSE {s2.GRIPPER_CUBE_CLOSE}")
    if args.mode == "joint":
        print(f"noise on joints {s2.JOINT_NOISE_INDICES}: +/- {s2.JOINT_NOISE} rad")
        print(f"JOINT_MIN {s2.JOINT_MIN}\nJOINT_MAX {s2.JOINT_MAX}")
    elif args.mode == "ee":
        print(f"EE offset +/- {s2.EE_XYZ_OFFSET} m, +/- {s2.EE_ANGLE_OFFSET} rad; "
              f"ee_bounds {cfg.ee_bounds['single']}")

    leader = None
    if args.leader and not args.dry_run:
        from trossen_real.leader.trossen_leader_single import TrossenSingleLeader

        leader = TrossenSingleLeader(cfg.leader_ips["single"],
                                     gripper_bounds=(cfg.control.gripper_closed, cfg.control.gripper_open))
        leader.connect()

    follower = s2.Stage2ResetFollower(cfg.follower_server_url, mode=args.mode, seed=args.seed)
    try:
        if args.dry_run:
            for i in range(args.n):
                t = follower.sample_target()
                print(f"[{i}] {t}")
            return
        follower.connect()
        for i in range(args.n):
            if i:
                input("Enter for the next reset (take the cube out first if you like)... ")
            follower.reset()
            st = follower.get_state()
            if leader is not None:
                # same as InterventionManager.sync_to_follower (position mode, rebaseline)
                target = s2.leader_sync_joints(follower, st["q"])
                leader.sync_to_joints(target, goal_time=s2.MOVE_TIME_S, end_in_freedrive=False)
                leader.rebaseline()
                print(f"[{i}] leader synced to {np.round(target, 4).tolist()} "
                      f"(reads {np.round(leader.get_joint_positions(), 4).tolist()})")
            t = follower.last_target
            print(f"[{i}] target: {t}")
            print(f"[{i}] reached q:    {np.round(st['q'], 4).tolist()}")
            print(f"[{i}] reached pose: {np.round(st['pose'], 4).tolist()}")
            if t and t["space"] == "joint":
                print(f"[{i}] |q - target| max {np.abs(np.asarray(st['q'][:6]) - np.asarray(t['joints'][:6])).max():.4f} rad")
    finally:
        if leader is not None:
            leader.disconnect()


if __name__ == "__main__":
    main()
