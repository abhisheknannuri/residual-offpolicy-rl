#!/usr/bin/env python3
"""Simulate GripperController against bad ACT outputs and human interventions.

No robot, no network: a fake gripper (1-tick latency, first-order lag, stalls on
the cube), fake policies (clean, flickering, untrained random, chunked gibberish,
NaN/inf, hesitant ...) and fake interventions (partial open, full open, close
on the cube, pedal chatter, unsynced leader, grab mid-ramp). Every scenario runs
twice: NAIVE = what the real loop does today (absolute width passthrough) and
CTRL = GripperController. Checks per scenario, plots to --out.

    .venv/bin/python temp/gripper/sim_gripper.py               # all scenarios
    .venv/bin/python temp/gripper/sim_gripper.py -k human      # names containing "human"
Exit code 1 if a check fails.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: trossen_real is not an installed package
from trossen_real.inference.gripper_controller import GripperController  # noqa: E402

CFG = GripperController()  # the defaults under test
C, O = CFG.closed, CFG.open
CUBE_WIDTH = 0.011  # measured width while the fingers stall on the cube (dataset)
FAST_STROKE = 0.010  # m in one tick: the kind of fast move that overshot and faulted the arm (config.py)


# ---- fake gripper ---------------------------------------------------------------
class FakeGripper:
    """Follows the command one tick late, ~2 ticks to settle (goal_time 0.1 s at 20 Hz),
    and cannot close past the cube while it holds one."""

    def __init__(self, width, cube):
        self.width, self.cube, self._pending = width, cube, width

    def step(self, command):
        target, self._pending = self._pending, command
        self.width += 0.6 * (target - self.width)
        if self.cube:
            self.width = max(self.width, CUBE_WIDTH)
        return self.width


# ---- fake humans ----------------------------------------------------------------
@dataclass
class Intervention:
    start: int
    end: int  # exclusive: the pedal is released at this tick
    target: float  # width the human moves the leader gripper to
    speed: float = 0.002  # m/tick of the human's hand
    synced: bool = True  # leader synced to the follower at the start (as the real InterventionManager does)
    start_width: float | None = None  # leader width when not synced


@dataclass
class Scenario:
    name: str
    what: str
    act: np.ndarray  # policy width per tick (ACT + residual), may hold NaN / inf / out of range
    interventions: list = field(default_factory=list)
    cube: bool = True
    checks: list = field(default_factory=list)  # (description, fn(ctrl_result, naive_result) -> bool)
    limits: tuple | None = None  # hardware clip range for the controller (None = [closed, open])


def run(sc: Scenario, use_ctrl: bool) -> dict:
    T = len(sc.act)
    ctrl = GripperController(CFG.closed, CFG.open, CFG.close_below, CFG.open_above, CFG.confirm_ticks, CFG.max_step, sc.limits)
    grip = FakeGripper(CUBE_WIDTH if sc.cube else C, sc.cube)
    cmd = C
    out = {k: np.full(T, np.nan) for k in ("cmd", "width", "human", "state")}
    out["mode"], out["switch_ticks"] = [], []
    human = None
    for t in range(T):
        iv = next((i for i in sc.interventions if i.start <= t < i.end), None)
        if iv is None:
            human = None
        else:
            if human is None or t == iv.start:
                human = cmd if iv.synced else iv.start_width
            human += max(-iv.speed, min(iv.speed, iv.target - human))
        if use_ctrl:
            was = ctrl.is_open
            cmd = ctrl.step(float(sc.act[t]), human)
            st, mode = int(ctrl.is_open), "human" if human is not None else "policy"
            if human is None and ctrl.is_open != was:
                out["switch_ticks"].append(t)
        else:  # today: the width goes straight to the follower, clipped to the limits
            a = sc.act[t]
            src = human if human is not None else a
            if src is not None and math.isfinite(src):
                cmd = min(max(src, C), O)
            st, mode = int(cmd > (C + O) / 2), "human" if human is not None else "policy"
        out["cmd"][t], out["width"][t], out["state"][t] = cmd, grip.step(cmd), st
        out["human"][t] = np.nan if human is None else human
        out["mode"].append(mode)
    d = np.diff(out["cmd"])
    out["max_drop"] = float(max(0.0, -d.min())) if len(d) else 0.0
    out["max_step"] = float(np.abs(d).max()) if len(d) else 0.0
    out["fast_strokes"] = int((np.abs(d) > FAST_STROKE).sum())
    out["flips"] = int((np.diff(out["state"]) != 0).sum())
    return out


# ---- fake policies --------------------------------------------------------------
rng = np.random.default_rng(0)


def closed_then_open(T, t_open):
    a = np.full(T, C)
    a[t_open:] = O
    return a


def scenarios() -> list[Scenario]:
    S = []
    lim = CFG.max_step + 1e-9
    hlim = lim  # one max_step for the policy and the human

    def common(extra_lim=lim):
        return [
            ("command always finite and inside [closed, open]",
             lambda r, n: bool(np.all(np.isfinite(r["cmd"])) and r["cmd"].min() >= C - 1e-12 and r["cmd"].max() <= O + 1e-12)),
            (f"no tick moves the command more than {extra_lim*1000:.1f} mm", lambda r, n: r["max_step"] <= extra_lim),
            ("no fast stroke", lambda r, n: r["fast_strokes"] == 0),
        ]

    T = 250
    S.append(Scenario("clean_release", "ACT closed, then a clean open at tick 120",
                      closed_then_open(T, 120), checks=common() + [
        ("exactly one switch", lambda r, n: r["switch_ticks"] == [120 + CFG.confirm_ticks - 1]),
        ("ends fully open", lambda r, n: abs(r["cmd"][-1] - O) < 1e-9),
        ("naive snaps open in one tick (for contrast)", lambda r, n: n["fast_strokes"] >= 1)]))

    a = np.full(T, C)
    for t in range(10, 150, 9):  # 1-2 tick spikes to open, as a noisy regressor near the release
        a[t:t + (1 if t % 2 else 2)] = O
    a[150:] = O
    S.append(Scenario("flicker_then_release", "closed with 1-2 tick open spikes every 9 ticks, real open at 150",
                      a, checks=common() + [
        ("spikes ignored, one switch at the real release", lambda r, n: r["switch_ticks"] == [150 + CFG.confirm_ticks - 1]),
        ("naive flips many times", lambda r, n: n["flips"] > 10)]))

    S.append(Scenario("hover_in_band", "ACT hovers inside the band (10.5-15.5 mm)",
                      rng.uniform(CFG.close_below + 0.0005, CFG.open_above - 0.0005, T), checks=common() + [
        ("no switch inside the hysteresis band", lambda r, n: r["flips"] == 0)]))

    S.append(Scenario("untrained_random", "untrained ACT: uniform random width every tick",
                      rng.uniform(C - 0.01, O + 0.01, T), checks=common() + [
        (f"switches at least confirm_ticks ({CFG.confirm_ticks}) apart",
         lambda r, n: np.all(np.diff(r["switch_ticks"]) >= CFG.confirm_ticks)),
        ("naive makes fast strokes", lambda r, n: n["fast_strokes"] > 10)]))

    chunks = np.repeat(rng.uniform(C - 0.005, O + 0.005, T // 10 + 1), 10)[:T]
    S.append(Scenario("untrained_chunks", "gibberish chunks: a random width held for 10 ticks (one ACT chunk)",
                      chunks, checks=common() + [
        (f"switches at least confirm_ticks apart", lambda r, n: np.all(np.diff(r["switch_ticks"]) >= CFG.confirm_ticks))]))

    a = closed_then_open(T, 150)
    a[[20, 21, 40, 60, 61, 62, 80, 100]] = [np.nan, np.nan, np.inf, -np.inf, 1.0, 5.0, -1.0, np.nan]
    S.append(Scenario("invalid_values", "NaN / inf / out of range spikes, then an open at 150", a, checks=common() + [
        ("no switch from garbage; one at the release",
         lambda r, n: r["switch_ticks"] == [150 + CFG.confirm_ticks - 1])]))

    a = np.full(T, C)
    a[100:160] = np.linspace(C, O, 60)
    a[160:] = O
    S.append(Scenario("slow_regressed_release", "ACT opens smoothly over 3 s (regression to the mean)", a,
                      checks=common() + [
        ("one switch, confirm_ticks after u passes open_above",
         lambda r, n, a=a: r["switch_ticks"] == [int(np.argmax(a > CFG.open_above)) + CFG.confirm_ticks - 1]),
        ("ends fully open", lambda r, n: abs(r["cmd"][-1] - O) < 1e-9)]))

    a = np.full(T, C)
    a[100:105], a[115:] = O, O
    S.append(Scenario("hesitant_release", "opens 5 ticks, closes 10 (a real 0.5 s close), opens for good", a,
                      checks=common() + [
        ("open, close, open: 3 switches, ends open", lambda r, n: r["flips"] == 3 and r["state"][-1] == 1)]))

    a = np.full(T, C)
    a[100:105], a[109:] = O, O
    S.append(Scenario("hesitant_short_close", "opens 5 ticks, closes only 4, opens for good", a, checks=common() + [
        ("follows ACT (3 switches) but ramped, ends open", lambda r, n: r["flips"] == 3 and r["state"][-1] == 1),
        ("naive: fast strokes", lambda r, n: n["fast_strokes"] == 3)]))

    # ---- human interventions --------------------------------------------------------
    def handback_checks(rel, human_end_width=None):
        return [
            ("on the release it continues from the human's width",
             lambda r, n: abs(r["cmd"][rel] - r["cmd"][rel - 1]) <= lim),
            ("then ramps, never snaps", lambda r, n: r["max_drop"] <= lim),
        ]

    S.append(Scenario("human_partial_open_act_close", "ACT says closed; human opens to 15 mm (60-100) and lets go",
                      np.full(T, C), [Intervention(60, 100, 0.015)], checks=common(hlim) + handback_checks(100) + [
        ("ends closed", lambda r, n: abs(r["cmd"][-1] - C) < 1e-9),
        ("naive snaps closed on release", lambda r, n: n["cmd"][99] - n["cmd"][100] > 0.009)]))

    S.append(Scenario("human_full_open_act_close", "ACT says closed; human opens fully and lets go",
                      np.full(T, C), [Intervention(60, 100, O, speed=0.004)], checks=common(hlim) + handback_checks(100) + [
        ("ends closed (ACT's call), via a ramp", lambda r, n: abs(r["cmd"][-1] - C) < 1e-9),
        ("naive: fast stroke on release", lambda r, n: n["fast_strokes"] >= 1)]))

    a = closed_then_open(T, 40)
    S.append(Scenario("human_open_act_agrees", "ACT open; human holds open 30-33 mm (80-120), lets go",
                      a, [Intervention(80, 120, 0.031)], checks=common(hlim) + [
        ("no dip after the release", lambda r, n: np.all(np.diff(r["cmd"][119:]) >= -1e-12)),
        ("ends fully open", lambda r, n: abs(r["cmd"][-1] - O) < 1e-9)]))

    S.append(Scenario("human_closes_act_open", "ACT open; human closes on the cube (60-100), lets go",
                      np.full(T, O), [Intervention(60, 100, C, speed=0.004)], checks=common(hlim) + [
        ("starts opening only after confirm_ticks of ACT's open vote",
         lambda r, n: r["switch_ticks"][-1:] == [100 + CFG.confirm_ticks - 1]),
        ("ends open (ACT's call), via a ramp", lambda r, n: abs(r["cmd"][-1] - O) < 1e-9 and r["max_step"] <= hlim)]))

    chatter = [Intervention(s, s + 3, 0.030, speed=0.004) for s in range(60, 120, 6)]
    S.append(Scenario("pedal_chatter", "intervention pedal tapped 3 on / 3 off, human opening; ACT closed",
                      np.full(T, C), chatter, checks=common(hlim) + [
        ("no snaps", lambda r, n: r["max_drop"] <= hlim),
        ("ends closed", lambda r, n: abs(r["cmd"][-1] - C) < 1e-9)]))

    S.append(Scenario("unsynced_leader", "leader gripper physically open (35 mm) when the pedal goes down",
                      np.full(T, C), [Intervention(60, 100, O, synced=False, start_width=O)], checks=common(hlim) + [
        ("the jump to the leader is rate limited", lambda r, n: r["max_step"] <= hlim),
        ("naive jumps", lambda r, n: n["fast_strokes"] >= 1)]))

    S.append(Scenario("human_opens_leader_fully", "station limits [1, 40] mm; ACT closed; human opens the leader to 40 mm",
                      np.full(T, C), [Intervention(60, 120, 0.040, speed=0.004)], limits=(0.001, 0.040), checks=[
        ("follower reaches the leader's 40 mm (not capped at open=35)",
         lambda r, n: abs(np.nanmax(r["cmd"][60:120]) - 0.040) < 1e-9),
        ("lags a fast human by a few mm, then catches up exactly (not stuck)",
         lambda r, n: np.allclose(r["cmd"][80:120], r["human"][80:120])),
        ("ramps closed after the release", lambda r, n: r["max_drop"] <= lim and abs(r["cmd"][-1] - C) < 1e-9)]))

    S.append(Scenario("grab_mid_ramp", "ACT opens at 60, human grabs at 64 mid-ramp and closes, lets go at 90",
                      closed_then_open(T, 60), [Intervention(64, 90, C, speed=0.004)], checks=common(hlim) + [
        ("human takes over from where the ramp was", lambda r, n: r["max_step"] <= hlim),
        ("ends open (ACT still says open)", lambda r, n: abs(r["cmd"][-1] - O) < 1e-9)]))

    # ---- residual authority: ACT closed, residual pushes open from tick 100 --------------
    half = (O - C) / 2
    for scale in (0.05, 0.5, 1.0):
        a = np.full(T, C)
        a[100:] += scale * half * 1.0  # residual at its +1 limit, real width = scale * half range
        opens = bool(a[-1] > CFG.open_above)
        S.append(Scenario(f"residual_scale_{scale}", f"ACT closed, residual +1 x action_scale {scale} from tick 100",
                          a, checks=common() + [
            (("opens" if opens else "cannot open") + f" (width {a[-1]*1000:.1f} mm)",
             lambda r, n, opens=opens: (r["flips"] == 1) == opens)]))
    return S


# ---- plots ------------------------------------------------------------------------
def plot(sc, r, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(sc.act))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for iv in sc.interventions:
        ax.axvspan(iv.start, iv.end, color="#f2c14e", alpha=0.25, lw=0)
    a = np.clip(np.where(np.isfinite(sc.act), sc.act, np.nan), C - 0.012, O + 0.012)
    ax.plot(t, a * 1000, ".", ms=3, color="#999999", label="policy width (vote)")
    ax.plot(t, r["human"] * 1000, color="#e07a1f", lw=1.5, label="human (leader)")
    ax.plot(t, n["cmd"] * 1000, "--", color="#c0392b", lw=1.2, label="NAIVE command (today)")
    ax.plot(t, r["cmd"] * 1000, color="#2266cc", lw=2, label="CTRL command")
    ax.plot(t, r["width"] * 1000, color="#2a9d5c", lw=1, label="CTRL measured width (sim)")
    for b in (CFG.close_below, CFG.open_above):
        ax.axhline(b * 1000, color="#bbbbbb", lw=0.6, ls=":")
    for s in r["switch_ticks"]:
        ax.axvline(s, color="#2266cc", lw=0.6, alpha=0.5)
    ax.set_title(f"{sc.name}: {sc.what}", fontsize=10, loc="left")
    ax.set_xlabel("tick (20 Hz); yellow = intervention, blue lines = CTRL state switch")
    ax.set_ylabel("gripper width (mm)")
    ax.set_ylim((C - 0.013) * 1000, (O + 0.013) * 1000)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", default="", help="Only scenarios whose name contains this.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "plots"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"controller: closed {C} open {O} close_below {CFG.close_below} open_above {CFG.open_above} confirm_ticks {CFG.confirm_ticks} max_step {CFG.max_step}\n")
    failed = 0
    for sc in scenarios():
        if args.k not in sc.name:
            continue
        r, n = run(sc, True), run(sc, False)
        print(f"{sc.name:30s} CTRL max step {r['max_step']*1000:5.1f} mm, fast {r['fast_strokes']:2d}, flips {r['flips']:2d}"
              f"  | NAIVE max step {n['max_step']*1000:5.1f} mm, fast {n['fast_strokes']:2d}, flips {n['flips']:2d}")
        for desc, fn in sc.checks:
            ok = bool(fn(r, n))
            failed += not ok
            print(f"    {'ok  ' if ok else 'FAIL'} {desc}")
        if not args.no_plots:
            plot(sc, r, n, out / f"{sc.name}.png")
    print(f"\n{'all checks passed' if not failed else f'{failed} check(s) FAILED'}; plots in {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
