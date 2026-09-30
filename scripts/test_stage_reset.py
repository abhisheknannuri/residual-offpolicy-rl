#!/usr/bin/env python3
"""
Standalone test of the "final placement stage" reset for NutAssemblySquare.

Reset strategy (the proper one): restore the FULL flattened MuJoCo sim state
(arm + gripper + both nuts, all mutually consistent) recorded from a real demo,
instead of rebuilding the pose piece by piece. We then VALIDATE the reset
(nut grasped + lifted + near peg, and stays grasped for a few hold steps). If a
state does not validate, we retry with another one until we get a good start.
This guarantees every episode begins in a clean, physically-valid grasp.

Usage:
    python scripts/test_stage_reset.py --trials 1000 --hold-steps 30
"""

import argparse
import os

import numpy as np
import robosuite

NPZ_PATH = "/home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil/NutSquare/stage_start_states.npz"

# Square peg XY (fixed in the pegs arena).
PEG_XY = np.array([0.23, 0.1])

LIFT_Z = 0.90          # nut must be at least this high (table ~0.82)
PEG_NEAR_MAX = 0.15    # nut within this planar distance of the peg
PEG_NEAR_MIN = 0.03    # but not already sitting on the peg
NUT_FELL_Z = 0.90      # below this => nut has dropped
SETTLE_STEPS = 10      # hold steps used inside the validated reset
CLOSE_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # +1 = CLOSE for Panda


def is_grasped(env):
    return env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=[g for g in env.nuts[0].contact_geoms],
    )


def nut_xyz(env):
    return env.sim.data.get_joint_qpos("SquareNut_joint0")[:3].copy()


def qualifies(env):
    p = nut_xyz(env)
    dist = np.linalg.norm(p[:2] - PEG_XY)
    return (p[2] > LIFT_Z) and (PEG_NEAR_MIN < dist < PEG_NEAR_MAX) and is_grasped(env)


def stage_reset(env, states, max_retries=50):
    """Restore a validated stage-start state. Returns (success, idx, attempts)."""
    n = states.shape[0]
    for attempt in range(1, max_retries + 1):
        env.reset()  # clears the internal step counter and re-randomizes
        idx = np.random.randint(0, n)
        env.sim.set_state_from_flattened(states[idx])
        env.sim.data.qvel[:] = 0.0
        env.sim.forward()
        env.robots[0].composite_controller.reset()

        if not qualifies(env):
            continue

        # Hold gripper closed for a few steps; make sure the grasp survives.
        good = True
        for _ in range(SETTLE_STEPS):
            env.step(CLOSE_ACTION)
            if nut_xyz(env)[2] < NUT_FELL_Z or not is_grasped(env):
                good = False
                break
        if good:
            return True, idx, attempt

    return False, -1, max_retries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--hold-steps", type=int, default=30,
                        help="Extra hold steps after a validated reset to stress-test stability.")
    parser.add_argument("--max-retries", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(NPZ_PATH):
        print(f"Error: npz not found at {NPZ_PATH}")
        return

    data = np.load(NPZ_PATH, allow_pickle=True)
    if "states" not in data:
        print("Error: npz has no 'states' array. Re-run extract_stage_start_states.py.")
        return
    states = data["states"]
    print(f"Loaded {states.shape[0]} stage-start states, dim {states.shape[1]}")

    print("Creating robosuite environment...")
    env = robosuite.make(
        "NutAssemblySquare",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    reset_failures = 0     # could not find a valid start within max_retries
    hold_failures = 0      # nut fell during the extra hold-steps stress test
    total_attempts = 0

    for trial in range(1, args.trials + 1):
        ok, idx, attempts = stage_reset(env, states, max_retries=args.max_retries)
        total_attempts += attempts
        if not ok:
            reset_failures += 1
            continue

        # Extra stress test: keep holding and confirm the nut stays put.
        fell = False
        for _ in range(args.hold_steps):
            env.step(CLOSE_ACTION)
            if nut_xyz(env)[2] < NUT_FELL_Z or not is_grasped(env):
                fell = True
                break
        if fell:
            hold_failures += 1

        if trial % 100 == 0:
            print(f"Trial {trial}/{args.trials} | reset_fail={reset_failures} "
                  f"hold_fail={hold_failures} avg_attempts={total_attempts/trial:.2f}")

    print("\n" + "=" * 50)
    print(f"Trials:                 {args.trials}")
    print(f"Reset failures:         {reset_failures} ({100*reset_failures/args.trials:.2f}%)")
    print(f"Hold-stability failures:{hold_failures} ({100*hold_failures/args.trials:.2f}%)")
    print(f"Avg reset attempts:     {total_attempts/args.trials:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
