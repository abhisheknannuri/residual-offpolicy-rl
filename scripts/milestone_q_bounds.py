#!/usr/bin/env python3
"""
Theoretical Q-value bounds for the stage-aware MILESTONE reward.

Question: "Can the TRUE value function ever reach 6-8 for the milestone reward,
or is that pure critic overestimation?"

This script answers it with pure math (no neural net). It enumerates every
plausible milestone reward sequence for an episode of up to `--horizon` steps —
all stage-entry timings, all success timings, and all failure cases — computes
the EXACT discounted return G[t] from every timestep (the value the critic
*should* converge to), and reports the global maximum.

It also verifies that the n-step (n=3) Bellman target equals the Monte-Carlo
return at the fixed point, i.e. **n-step does NOT raise the ceiling** — a common
worry. If the true max is ~2.0 but your Q-plots show 6-8, the gap is 100%
critic overestimation (function approximation + off-policy bootstrapping on
out-of-distribution / truncated states).

Run:
    .venv/bin/python scripts/milestone_q_bounds.py
    .venv/bin/python scripts/milestone_q_bounds.py --payouts 0.0 0.1 0.1 0.1 --success_bonus 0.7
    .venv/bin/python scripts/milestone_q_bounds.py --success_bonus 2.0   # what-if
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Exhaustive true-value (Q) bounds for the milestone reward")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n_step", type=int, default=3)
    # Milestone design (index = stage; index 0 = start, never paid).
    p.add_argument("--payouts", type=float, nargs="+", default=[0.0, 0.3, 0.3, 0.4])
    p.add_argument("--success_bonus", type=float, default=1.0)
    # Timing grid: how finely to sample the step at which each stage / success occurs.
    p.add_argument("--grid", type=int, default=40, help="number of candidate step positions to sample in [1, horizon-1]")
    return p.parse_args()


def build_reward_sequence(stage_steps, success_step, horizon, payouts, bonus):
    """Per-step milestone reward for one episode.

    Args:
        stage_steps: list of (stage, step) for the stages actually entered, monotonic
            in both (a ratchet). Stages not entered are simply absent.
        success_step: step index of the TRUE simulator success (terminal), or None
            for a failure (episode runs to `horizon` and truncates).
        Convention (matches the wrapper/labeler): the reward of the transition that
        LANDS on step t is credited at step t. Skipped stages are summed. The
        success bonus is paid on the terminal success step.
    Returns:
        r: float64 reward array of length T (= success_step+1, or horizon on failure).
        succeeded: bool.
    """
    T = (success_step + 1) if success_step is not None else horizon
    r = np.zeros(T, dtype=np.float64)
    last_paid = 0
    for stage, step in sorted(stage_steps, key=lambda x: x[1]):
        if 0 <= step < T and stage > last_paid:
            r[step] += float(sum(payouts[k] for k in range(last_paid + 1, stage + 1)))
            last_paid = stage
    if success_step is not None and 0 <= success_step < T:
        r[success_step] += float(bonus)
    return r, (success_step is not None)


def mc_return(r, gamma):
    """G[t] = r[t] + gamma*r[t+1] + ...  == Q(s_t, a_t) at the true Bellman fixed point."""
    G = np.zeros_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        G[t] = acc
    return G


def nstep_target_oracle(r, G, gamma, n):
    """n-step target with the ORACLE tail value V(s_{t+n}) = G[t+n].

    This is the target the critic is regressed toward when its bootstrap is exact.
    We verify it equals G (so n-step returns cannot raise the value ceiling; they
    only change bootstrapping variance). The bootstrap is masked at the terminal
    (terminated-only bootstrap): if t+n reaches/passes the end, no tail is added.
    """
    T = len(r)
    tgt = np.zeros(T, dtype=np.float64)
    for t in range(T):
        acc = 0.0
        horizon_reached = False
        for i in range(n):
            if t + i < T:
                acc += (gamma ** i) * r[t + i]
            else:
                horizon_reached = True
                break
        if (t + n) < T and not horizon_reached:
            acc += (gamma ** n) * G[t + n]  # bootstrap the true tail
        tgt[t] = acc
    return tgt


def main():
    a = parse_args()
    gamma, H, N = a.gamma, a.horizon, a.n_step
    payouts = [float(x) for x in a.payouts]
    bonus = float(a.success_bonus)
    num_stages = len(payouts)
    max_possible_total = sum(payouts) + bonus  # ratchet cap, undiscounted

    # Candidate step positions (a grid; the maximum is front-loaded so a coarse
    # grid that includes small steps captures it, but we also inject 1,2,3,... to
    # be safe for the "rewards as early as possible" worst case).
    grid = sorted(set(np.linspace(1, H - 1, a.grid).astype(int)) | set(range(1, min(2 * num_stages + 2, H))))

    stages = list(range(1, num_stages))  # 1..num_stages-1  (e.g. 1,2,3)

    global_max = -np.inf
    global_max_desc = None
    fail_max = -np.inf          # max true Q over FAILURE episodes (no bonus)
    succ_max = -np.inf
    max_nstep_err = 0.0
    n_cases = 0

    # Enumerate: which subset of stages is reached (a prefix, since monotonic),
    # the step at which each reached stage is entered (strictly increasing on the
    # grid), and whether/when success happens (>= last stage step) or failure.
    for reached in range(0, num_stages):  # 0..num_stages-1 (max stage index reached)
        reached_stages = stages[:reached]  # e.g. reached=3 -> [1,2,3]
        # choose strictly-increasing entry steps for the reached stages
        for entry_steps in itertools.combinations(grid, len(reached_stages)):
            stage_steps = list(zip(reached_stages, entry_steps))
            last_step = entry_steps[-1] if entry_steps else 0
            # success can only happen if the LAST stage was reached (placing) and
            # at a step >= last stage entry; else failure.
            success_options = [None]  # failure always possible
            if reached == num_stages - 1:  # reached the final "placing" stage
                success_options += [s for s in grid if s >= last_step]

            for ss in success_options:
                r, succeeded = build_reward_sequence(stage_steps, ss, H, payouts, bonus)
                if r.size == 0:
                    continue
                G = mc_return(r, gamma)
                tgt = nstep_target_oracle(r, G, gamma, N)
                max_nstep_err = max(max_nstep_err, float(np.max(np.abs(tgt - G))))
                m = float(np.max(G))
                n_cases += 1
                if m > global_max:
                    global_max, global_max_desc = m, (stage_steps, ss, succeeded)
                if succeeded:
                    succ_max = max(succ_max, m)
                else:
                    fail_max = max(fail_max, m)

    # Analytic upper bound: all remaining rewards paid on consecutive steps
    # starting at the state itself (most front-loaded reachable case).
    # stage1@t, stage2@t+1, stage3@t+2, success@t+3  -> G[t]
    analytic = sum(payouts[s] * (gamma ** (s - 1)) for s in stages) + bonus * (gamma ** (num_stages - 1))

    print("=" * 74)
    print(f"MILESTONE true-value (Q) bounds   gamma={gamma}  horizon={H}  n_step={N}")
    print(f"payouts (per stage) = {payouts}   success_bonus = {bonus}")
    print(f"ratchet cap (undiscounted total reward per episode) = {max_possible_total:.4f}")
    print("-" * 74)
    print(f"enumerated reward sequences ........... {n_cases:,}")
    print(f"n-step(n={N}) target vs MC return: max|err| = {max_nstep_err:.3e}  "
          f"(==0 => n-step does NOT change the fixed point / ceiling)")
    print("-" * 74)
    print(f"MAX true Q over ALL sequences ......... {global_max:.4f}")
    print(f"MAX true Q over SUCCESS episodes ...... {succ_max:.4f}")
    print(f"MAX true Q over FAILURE episodes ...... {fail_max:.4f}   (no success bonus)")
    print(f"analytic most-front-loaded bound ...... {analytic:.4f}")
    print(f"loose upper bound (undiscounted total)  {max_possible_total:.4f}")
    ss_desc, succ = (global_max_desc[1], global_max_desc[2]) if global_max_desc else (None, None)
    print(f"argmax case: stage_steps={global_max_desc[0]}  success_step={ss_desc}  succeeded={succ}")
    print("=" * 74)
    print("CONCLUSION:")
    print(f"  The true value function can NEVER exceed ~{global_max:.2f} for this design.")
    print(f"  A learned critic showing 6-8 is overestimating by ~{8.0 / max(global_max,1e-9):.1f}x")
    print(f"  the true maximum -> function-approximation / off-policy overestimation,")
    print(f"  concentrated on out-of-distribution & truncated (failure) states.")
    print("=" * 74)


if __name__ == "__main__":
    main()
