#!/usr/bin/env python3
"""GripperController on the REAL follower + leader + foot pedal, with a fake ACT.

Moves the real robot. Start follower_single_server.py first.

  1. connect the follower, move it to the staged position (Enter to confirm)
  2. connect the leader, sync it to the follower (position mode, 3 s)
  3. sessions, one after the other, each started with Enter (s = skip, q = quit):
       - the simulator's fake ACT patterns (sim_gripper.py: clean, flicker,
         untrained random, chunked gibberish, NaN/inf, hesitant, ...). For the
         ones the simulator ran with a human, the hint says when and what to do
         with the pedal; do it by hand, roughly.
       - manual sessions: ACT closed / open / alternating / gibberish for longer,
         intervene whenever you like, do the opposite, release, repeat.
     Reset pedal = end the current session early.

Per tick, like the infer app (infer_loop.py): intervention pedal held -> leader
in freedrive, follower gets the leader's 6 joints, gripper = controller(ACT,
leader width). Not held -> leader in position mode mirroring the follower, the
arm holds where it is, gripper = controller(ACT). The fake ACT only drives the
gripper; the arm only moves when you move the leader.

The gripper limits are the station's control.gripper_closed/gripper_open, so a
fully opened leader opens the follower fully (the controller's `open` width is
only the policy's target).

Logs + plots per session: temp/gripper/real_logs/<time>/.

    .venv/bin/python temp/gripper/real_gripper_test.py
    .venv/bin/python temp/gripper/real_gripper_test.py --list
    .venv/bin/python temp/gripper/real_gripper_test.py --only clean_release,manual_alternate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # repo root: trossen_real is not an installed package

import sim_gripper as sim  # noqa: E402
from trossen_real.inference.gripper_controller import GripperController  # noqa: E402

C, O = sim.C, sim.O


# ---- sessions: (name, what, ACT widths per tick, hint) ------------------------------
HINTS = {  # where the simulator's intervention windows don't translate to a hand
    "pedal_chatter": "3-6 s: tap the intervention pedal quickly, many times, while opening the leader gripper",
    "unsynced_leader": "the real leader is always synced to the follower, so this case cannot happen here: "
                       "just open the leader 3-5 s, release",
}


def sessions(hz: int) -> list[tuple[str, str, np.ndarray, str]]:
    out = []
    for sc in sim.scenarios():
        hint = "; ".join(
            f"{iv.start / hz:.1f}-{iv.end / hz:.1f} s: pedal down, move the leader gripper to "
            f"{iv.target * 1000:.0f} mm, release" for iv in sc.interventions)
        hint = HINTS.get(sc.name, hint) or "no intervention needed (try one if you like)"
        out.append((sc.name, sc.what, sc.act, hint))
    rng = np.random.default_rng(1)
    s = lambda sec: int(sec * hz)  # noqa: E731
    out += [
        ("manual_act_closed", "ACT says closed for 30 s", np.full(s(30), C),
         "intervene: open the leader (a bit, fully), release; repeat"),
        ("manual_act_open", "ACT says open for 30 s", np.full(s(30), O),
         "intervene: close the leader, release; repeat"),
        ("manual_alternate", "ACT 4 s open / 4 s closed for 60 s", np.tile(np.r_[np.full(s(4), O), np.full(s(4), C)], 8)[:s(60)],
         "intervene at any time, do the opposite of ACT, release at any time"),
        ("manual_gibberish", "untrained ACT: random width chunks of 0.5 s for 60 s",
         np.repeat(rng.uniform(C - 0.005, O + 0.005, s(60) // s(0.5) + 1), s(0.5))[:s(60)],
         "intervene at any time, release at any time"),
    ]
    return out


# ---- one session on the robot ------------------------------------------------------------
def run_session(name, act, ctrl, follower, leader, pedal, q_arm, hz, goal_time, log_path):
    rows, dt = [], 1.0 / hz
    pedal.consume_reset()  # ignore a press from before the session
    was_iv = False
    with open(log_path, "w") as f:
        for k, a in enumerate(act):
            t0 = time.time()
            if pedal.consume_reset():
                print("    reset pedal: session ended early")
                break
            iv = pedal.is_intervention()
            if iv:
                leader.set_freedrive_mode()
                lq = np.asarray(leader.get_joint_positions(), dtype=float)
                q_arm, human = lq[:6].tolist(), float(lq[6])
            else:
                leader.set_position_mode()
                human = None
            if iv != was_iv:
                print(f"    {k / hz:5.1f} s  intervention {'ON' if iv else 'OFF'}")
            was_open = ctrl.is_open
            cmd = ctrl.step(float(a), human)
            if not iv and ctrl.is_open != was_open:
                print(f"    {k / hz:5.1f} s  gripper -> {'OPEN' if ctrl.is_open else 'CLOSED'}")
            action = list(q_arm) + [cmd]
            follower.move_to_joint_positions(action, blocking=False, min_time_to_move=goal_time)
            if not iv:
                leader.track_joints(action, goal_time=goal_time, blocking=False)
            st = follower.get_state()
            row = dict(t=k / hz, act=float(a) if np.isfinite(a) else None, intervened=iv, human=human,
                       cmd=cmd, is_open=ctrl.is_open, measured=float(st["gripper_pos"]))
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            was_iv = iv
            time.sleep(max(0.0, dt - (time.time() - t0)))
    if leader is not None:
        leader.set_position_mode()
    return rows, q_arm


def plot(name, what, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    t = np.array([r["t"] for r in rows])
    g = lambda k: np.array([np.nan if r[k] is None else r[k] for r in rows], float)  # noqa: E731
    iv = np.array([r["intervened"] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k in np.flatnonzero(iv):
        ax.axvspan(t[k], t[k] + (t[1] - t[0] if len(t) > 1 else 0.05), color="#f2c14e", alpha=0.25, lw=0)
    ax.plot(t, np.clip(g("act"), C - 0.012, O + 0.012) * 1000, ".", ms=3, color="#999999", label="fake ACT width (vote)")
    ax.plot(t, g("human") * 1000, color="#e07a1f", lw=1.5, label="leader width (human)")
    ax.plot(t, g("cmd") * 1000, color="#2266cc", lw=2, label="controller command")
    ax.plot(t, g("measured") * 1000, color="#2a9d5c", lw=1, label="follower measured")
    ax.set_title(f"REAL {name}: {what}", fontsize=10, loc="left")
    ax.set_xlabel("s; yellow = intervention")
    ax.set_ylabel("gripper width (mm)")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="trossen_station3_single")
    ap.add_argument("--list", action="store_true", help="Print the sessions and exit (no robot).")
    ap.add_argument("--only", default="", help="Comma-separated session names.")
    ap.add_argument("--reset-mode", default="normal",
                    help="'normal' = staged position (default); 'config' = the station's reset.single.mode.")
    ap.add_argument("--max-step", type=float, default=0.003, help="m/tick ramp of the controller.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from trossen_real.config import load_station_config

    cfg = load_station_config(args.station)
    hz, goal_time = cfg.control.frequency_hz, cfg.control.goal_time_s
    todo = sessions(hz)
    if args.only:
        names = set(args.only.split(","))
        unknown = names - {n for n, *_ in todo}
        if unknown:
            raise SystemExit(f"unknown sessions: {sorted(unknown)}")
        todo = [s for s in todo if s[0] in names]
    if args.list:
        for name, what, act, hint in todo:
            print(f"{name:30s} {len(act) / hz:5.1f} s  {what}\n{'':30s} hint: {hint}")
        return
    if cfg.arm_mode != "single" or cfg.control.command_space != "joint":
        raise SystemExit("needs arm_mode single and command_space joint (as the infer app)")

    from trossen_real.human_intervention.pedal_listener import PedalListener
    from trossen_real.leader.trossen_leader_single import TrossenSingleLeader
    from trossen_real.rl.stage2_reset import leader_sync_joints, make_follower

    pedal = PedalListener()
    if not pedal.is_healthy():
        raise SystemExit("no foot pedal found: the intervention sessions need it")

    reset_mode = cfg.reset["single"].mode if args.reset_mode == "config" else args.reset_mode
    follower = make_follower(cfg.follower_server_url, reset_mode)
    follower.connect()
    lo, hi = cfg.control.gripper_closed, cfg.control.gripper_open
    print(f"follower {cfg.follower_server_url}, reset mode {reset_mode}, gripper limits [{lo}, {hi}] m, {hz} Hz")
    input("Enter: move the follower to the reset position... ")
    follower.reset()

    leader = TrossenSingleLeader(cfg.leader_ips["single"], start_in_freedrive=False, gripper_bounds=(lo, hi))
    leader.connect()
    try:
        target = leader_sync_joints(follower, follower.get_state()["q"])
        print(f"syncing the leader to {np.round(target, 4).tolist()} (3 s)")
        leader.sync_to_joints(target, goal_time=3.0, end_in_freedrive=False)
        leader.rebaseline()

        ctrl = GripperController(C, O, confirm_ticks=3, max_step=args.max_step, limits=(lo, hi))
        ctrl.reset(target[6])
        q_arm = list(target[:6])
        out = HERE / "real_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)

        for name, what, act, hint in todo:
            ans = input(f"\n[{name}] {len(act) / hz:.0f} s - {what}\n    hint: {hint}\n"
                        f"    Enter = start, s = skip, q = quit: ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            ctrl.reset(ctrl.command)  # continue from the current width, clear the vote count
            rows, q_arm = run_session(name, act, ctrl, follower, leader, pedal, q_arm, hz, goal_time,
                                      out / f"{name}.jsonl")
            plot(name, what, rows, out / f"{name}.png")
            cmd = np.array([r["cmd"] for r in rows])
            print(f"    done: {len(rows)} ticks, max command step {np.abs(np.diff(cmd)).max() * 1000 if len(cmd) > 1 else 0:.1f} mm,"
                  f" {sum(r['intervened'] for r in rows)} intervened ticks -> {out / (name + '.png')}")
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C)")
    finally:
        try:
            leader.set_position_mode()
        finally:
            leader.disconnect()
        print("leader disconnected; the follower stays where it is (still connected to its server)")


if __name__ == "__main__":
    main()
