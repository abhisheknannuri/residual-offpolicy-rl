#!/usr/bin/env python3
"""How far does the REAL follower gripper overshoot, per way of commanding it?

Moves the real follower's GRIPPER only; the six arm joints are re-sent at the
positions they have at the start, so the arm holds still. Talks to the arm
directly through the trossen_arm SDK (like follower_single.py), so STOP
follower_single_server.py first: the controller takes one connection.

Why: the RL env faulted the arm twice on 2026-11-05 ("Joint 6 position limit
exceeded ... 0.044009" at 16:22, "0.044015" at 14:31) while every command was
<= 40 mm. A 20 Hz log cannot see a spike, so this samples the gripper as fast as
the SDK returns it (~1 kHz).

First run (real_logs/overshoot_20261105_164535, 10 <-> 30 mm, no cube): no
method overshot at all (peak 29.95 mm, low 10.05 mm), one-tick 20 mm jumps
included. Second run (overshoot_20261105_165514): opening to 34 mm and opening right
after squeezing the cube did not overshoot either. Both faults had one thing
those runs did not: the open to 40 mm was the LAST command (the env waited for
the SARM pedal), so the robot finished that 0.125 s move alone. Confirmed by
the third run (see REMOVED below); `oncehold` tests the env's fix.

Safety: commands stay within [--bottom, --top] (10-30 mm), the "top34" cases go
to 34 mm (refused above 34), and the squeeze commands the station's
gripper_closed (1 mm) on the cube, as the RL does. The fault limit is 44 mm.
Ctrl+C at any time: the gripper goes back to the rest width slowly and the
connection is released.

Methods (a name is <case>_<how>):
  how
    jump          one set_all_positions to the end width, goal_time 0.125 s
                  (RL 14:31: 18.7 -> 40 mm in one tick)
    step6         6 mm per tick at 20 Hz, set_all_positions 0.125 s
                  (GripperFilter without the opening cap; RL 16:22)
    step6_gap<g>  as step6, while opening at most g mm wider than the measured
                  fingers (GripperFilter open_gap), --gaps
    gripper_q<t>  one set_gripper_position(end, goal_time=t), arm untouched;
                  t > 0.2 s is quintic (ends with zero speed), --quintic
    oncehold      ONE set_all_positions to the end width (0.125 s), then 50 ms
                  later (one tick) a hold at the measured width: the env's fix
                  (TrossenResidualEnv._hold_still). Not in the default run.
  REMOVED: `once` (the same command with nothing after it, as the env did while
  waiting for the SARM pedal). Run 2026-11-05 17:09, 10 -> 30 mm: the fingers
  stopped at 37.95 mm (8 mm past the target) and stayed there; the next command
  (1.5 s back to 30 mm) faulted the arm at 44.075 mm. Do not send a lone fast
  gripper command.
  case
    plain         bottom -> top (open) and top -> bottom (close), no cube
    top34         bottom -> 34 mm (open) and back (close), no cube
    squeeze       YOU hold the cube between the fingers; the gripper closes on
                  it at 1 mm for --squeeze-s s (20 Hz commands, like the RL),
                  then opens to top; the cube drops: hold it by its end, keep
                  your fingers out of the jaws. Opening only.
Each is repeated --repeats times; the gripper goes to the start width slowly
(1.5 s) before each.

Output (temp/gripper/real_logs/overshoot_<time>/): one JSON per trial (time,
measured width, command), summary.json, one plot per method, and a table:
overshoot = how far the fingers went past the end width (opening: above it,
closing: below it), settle = time until they stay within 0.5 mm of it, peak
speed = largest change over 10 ms windows.

    .venv/bin/python temp/gripper/gripper_overshoot_test.py --list               # no robot
    .venv/bin/python temp/gripper/gripper_overshoot_test.py                      # every method except oncehold, squeeze
    .venv/bin/python temp/gripper/gripper_overshoot_test.py --only plain_oncehold --top 20
    .venv/bin/python temp/gripper/gripper_overshoot_test.py --only top34_jump,squeeze_jump,squeeze_step6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root: trossen_real is not an installed package

MM = 0.001
TOP_MAX_MM = 34.0
BOTTOM_MIN_MM = 4.0
CASES = ("plain", "top34", "squeeze")


def methods(gaps, quintic):
    hows = [("jump", "one command to the end width, set_all_positions 0.125 s"),
            ("step6", "6 mm per tick, set_all_positions 0.125 s")]
    hows += [(f"step6_gap{g:g}", f"6 mm per tick, opening at most {g:g} mm ahead of the fingers") for g in gaps]
    hows += [(f"gripper_q{t:g}", f"one set_gripper_position, goal_time {t:g} s") for t in quintic]
    hows += [("oncehold", "ONE set_all_positions 0.125 s, then a hold at the measured width 50 ms later")]
    case_what = {"plain": "no cube", "top34": "open to 34 mm, no cube", "squeeze": "after squeezing the cube at 1 mm"}
    return [(f"{c}_{h}", f"{what}; {case_what[c]}") for c in CASES for h, what in hows]


def split(name):
    case, how = name.split("_", 1)
    return case, how


def run_move(arm, arm_q, how, start, end, hz, goal_time, window):
    """One move start -> end (m) from where the gripper is now. Returns samples
    [(t, measured, command)]."""
    samples, period = [], 1.0 / hz
    opening = end > start
    command, next_tick, t0 = start, 0.0, time.perf_counter()
    gap = float(how[len("step6_gap"):]) * MM if how.startswith("step6_gap") else None
    if how.startswith("gripper_q"):
        command = end
        arm.set_gripper_position(end, goal_time=float(how[len("gripper_q"):]), blocking=False)
    if how == "oncehold":
        command = end
        arm.set_joint_positions(list(arm_q) + [end], goal_time=goal_time, blocking=False)
    held = how != "oncehold"
    while (t := time.perf_counter() - t0) < window:
        measured = arm.get_gripper_position()
        if not held and t >= period:
            held = True
            command = min(command, measured) if opening else max(command, measured)
            arm.set_joint_positions(list(arm_q) + [command], goal_time=goal_time, blocking=False)
        if not how.startswith(("gripper_q", "oncehold")) and t >= next_tick:
            next_tick += period
            if how == "jump":
                command = end
            else:
                step = min(0.006, abs(end - command))
                new = command + (step if opening else -step)
                if gap is not None and opening:
                    new = min(new, max(command, measured + gap))
                command = new
            arm.set_joint_positions(list(arm_q) + [command], goal_time=goal_time, blocking=False)
        samples.append((t, measured, command))
    return samples


def squeeze(arm, arm_q, closed, seconds, hz, goal_time):
    """Command `closed` at hz for `seconds`, like the RL holding the cube.
    Returns samples [(t, measured, command)], t negative (before the opening)."""
    samples, period, t0 = [], 1.0 / hz, time.perf_counter()
    next_tick = 0.0
    while (t := time.perf_counter() - t0) < seconds:
        if t >= next_tick:
            next_tick += period
            arm.set_joint_positions(list(arm_q) + [closed], goal_time=goal_time, blocking=False)
        samples.append((t - seconds, arm.get_gripper_position(), closed))
    return samples


def summarize(samples, start, end):
    s = [x for x in samples if x[0] >= 0]
    t = np.array([x[0] for x in s])
    m = np.array([x[1] for x in s])
    opening = end > start
    over = (m.max() - end) if opening else (end - m.min())
    outside = np.abs(m - end) > 0.0005
    settle = float(t[np.where(outside)[0][-1]]) if outside.any() else 0.0
    grid = np.arange(0.0, t[-1], 0.010)  # the SDK's value updates in steps: speed over 10 ms windows
    speed = np.abs(np.diff(np.interp(grid, t, m))).max() / 0.010 if len(grid) > 1 else 0.0
    return dict(overshoot_mm=round(float(over) / MM, 3), peak_mm=round(float(m.max() if opening else m.min()) / MM, 3),
                settle_s=round(settle, 3), peak_speed_mm_s=round(float(speed) / MM, 1),
                samples=len(m), sample_hz=round(len(m) / max(float(t[-1]), 1e-6), 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="trossen_station3_single")
    ap.add_argument("--bottom", type=float, default=10.0, help="mm, closing end / opening start")
    ap.add_argument("--top", type=float, default=30.0, help=f"mm, opening end / closing start (<= {TOP_MAX_MM:g})")
    ap.add_argument("--gaps", default="3,6,9", help="mm, open_gap values for step6_gap<g>")
    ap.add_argument("--quintic", default="0.25,0.5", help="s, goal_time values for gripper_q<t>")
    ap.add_argument("--squeeze-s", type=float, default=3.0, help="s the cube is squeezed before opening")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--window", type=float, default=2.0, help="s recorded per move")
    ap.add_argument("--rest", type=float, default=20.0, help="mm, width the gripper is left at")
    ap.add_argument("--only", default="", help="comma-separated method names")
    ap.add_argument("--list", action="store_true", help="list the methods and exit (no robot)")
    args = ap.parse_args()

    gaps = [float(x) for x in args.gaps.split(",") if x]
    quintic = [float(x) for x in args.quintic.split(",") if x]
    todo = methods(gaps, quintic)
    if args.only:
        names = set(args.only.split(","))
        unknown = names - {n for n, _ in todo}
        if unknown:
            sys.exit(f"unknown methods: {sorted(unknown)}; --list shows them")
        todo = [m for m in todo if m[0] in names]
    else:
        todo = [m for m in todo if split(m[0])[0] != "squeeze" and split(m[0])[1] != "oncehold"]
    if args.list:
        for n, what in todo:
            print(f"  {n:22s} {what}")
        return
    if not (BOTTOM_MIN_MM <= args.bottom < args.top <= TOP_MAX_MM):
        sys.exit(f"need {BOTTOM_MIN_MM:g} <= --bottom < --top <= {TOP_MAX_MM:g} mm")
    if not BOTTOM_MIN_MM <= args.rest <= TOP_MAX_MM:
        sys.exit(f"--rest must be within {BOTTOM_MIN_MM:g}-{TOP_MAX_MM:g} mm")

    from trossen_real.arm_driver import RealArm
    from trossen_real.config import load_station_config

    cfg = load_station_config(args.config)
    hz, goal_time, closed = cfg.control.frequency_hz, cfg.control.goal_time_s, cfg.control.gripper_closed
    ip = cfg.follower_ips["single"]
    bottom, top, rest = args.bottom * MM, args.top * MM, args.rest * MM
    out_dir = HERE / "real_logs" / f"overshoot_{datetime.now():%Y%m%d_%H%M%S}"

    n_sq = sum(split(n)[0] == "squeeze" for n, _ in todo)
    print(f"Follower {ip}. The arm holds still; only the gripper moves ({args.bottom:g}-{args.top:g} mm, "
          f"top34 cases to 34 mm, squeeze cases close to {closed / MM:g} mm on the cube).")
    print(f"{len(todo)} methods x {args.repeats} repeats, {args.window:g} s per move, {hz} Hz ticks, "
          f"goal_time {goal_time:g} s." + (f" {n_sq * args.repeats} squeezes: you hold the cube." if n_sq else ""))
    print("follower_single_server.py must NOT be running. Keep hands out of the jaws.")
    input("Enter to connect and start (Ctrl+C to abort) ... ")

    arm = RealArm(ip, "follower")
    arm.connect()
    results = []
    try:
        arm_q = [float(x) for x in arm.get_joint_positions()[:6]]
        print(f"arm joints held at {np.round(arm_q, 3).tolist()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, what in todo:
            case, how = split(name)
            print(f"\n{name}: {what}")
            hi = TOP_MAX_MM * MM if case == "top34" else top
            moves = [("open", bottom, hi)] if case == "squeeze" else [("open", bottom, hi), ("close", hi, bottom)]
            for rep in range(args.repeats):
                for direction, start, end in moves:
                    pre = []
                    if case == "squeeze":
                        arm.set_gripper_position(top, goal_time=1.5, blocking=True)
                        input(f"  #{rep}: put the cube between the fingers and hold it by its end, Enter ... ")
                        pre = squeeze(arm, arm_q, closed, args.squeeze_s, hz, goal_time)
                        start = pre[-1][1]
                        print(f"  squeezed: command {closed / MM:g} mm, fingers at {start / MM:.2f} mm; opening")
                    else:
                        arm.set_gripper_position(start, goal_time=1.5, blocking=True)
                        time.sleep(0.5)
                    samples = pre + run_move(arm, arm_q, how, start, end, hz, goal_time, args.window)
                    s = summarize(samples, start, end)
                    s.update(method=name, direction=direction, repeat=rep)
                    results.append(s)
                    (out_dir / f"{name}_{direction}_{rep}.json").write_text(json.dumps(
                        dict(summary=s, start=start, end=end, samples=samples)))
                    print(f"  {direction:5s} #{rep}: overshoot {s['overshoot_mm']:6.2f} mm  "
                          f"(peak {s['peak_mm']:6.2f}), settle {s['settle_s']:.2f} s, "
                          f"peak speed {s['peak_speed_mm_s']:.0f} mm/s, {s['sample_hz']:.0f} Hz sampling")
    except KeyboardInterrupt:
        print("\nAborted.")
    finally:
        try:
            arm.set_gripper_position(rest, goal_time=1.5, blocking=True)
        finally:
            arm.disconnect()

    if not results:
        return
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"\n{'method':22s} {'dir':5s} {'overshoot mm (max)':>19s} {'peak mm':>8s} {'settle s':>9s} {'speed mm/s':>11s}")
    for name, _ in todo:
        for direction in ("open", "close"):
            rs = [r for r in results if r["method"] == name and r["direction"] == direction]
            if rs:
                worst = max(rs, key=lambda r: r["overshoot_mm"])
                print(f"{name:22s} {direction:5s} {worst['overshoot_mm']:19.2f} {worst['peak_mm']:8.2f} "
                      f"{max(r['settle_s'] for r in rs):9.2f} {max(r['peak_speed_mm_s'] for r in rs):11.0f}")
    print("\nAn opening is safe at 40 mm if its overshoot stays well below 4 mm (the margin to 44 mm).")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for name, _ in todo:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, direction in zip(axes, ("open", "close")):
                for f in sorted(out_dir.glob(f"{name}_{direction}_*.json")):
                    d = json.loads(f.read_text())
                    smp = np.array(d["samples"])
                    ax.plot(smp[:, 0], smp[:, 1] / MM, label=f"measured #{f.stem[-1]}")
                    ax.plot(smp[:, 0], smp[:, 2] / MM, "--", alpha=0.6, label=f"command #{f.stem[-1]}")
                    ax.axhline(d["end"] / MM, color="k", lw=0.5)
                ax.set(title=f"{name} {direction}", xlabel="s (0 = start of the move)", ylabel="mm")
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"{name}.png", dpi=100)
            plt.close(fig)
    except ImportError:
        pass
    print(f"Logs and plots: {out_dir}")


if __name__ == "__main__":
    main()
