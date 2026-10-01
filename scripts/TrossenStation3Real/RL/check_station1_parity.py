# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Check conf/station3.yaml against the Station-1 shell scripts it was derived from.

station3.yaml was written by hand from scripts/TrossenStation1Real/*.sh, and
hand-copying 70-odd values is exactly the kind of job that goes wrong quietly:
16 of them were wrong on the first pass, including algo.stddev_max, which is a
per-joint list in Station-1 and had been written as a single float.

This parses the Station-1 scripts - the shell variable assignments and the
`CMD=(...)` array that maps them onto hydra keys - and diffs the result against
station3.yaml. Values that are meant to differ for Station 3 (dataset, station
name, W&B project, per-run paths) are listed in STATION3_SPECIFIC and skipped.

    .venv/bin/python scripts/TrossenStation3Real/RL/check_station1_parity.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STATION1 = REPO / "scripts/TrossenStation1Real"
STATION3_YAML = HERE / "conf/station3.yaml"

# Deliberately different for Station 3, or per-run.
STATION3_SPECIFIC = {
    "offline_data.name", "offline_data.root", "offline_data.num_episodes",
    "station_config_name", "wandb.project", "wandb.name", "wandb.group",
    "wandb.entity", "wandb.mode", "real_log_file", "resume_ckpt",
}

# Set per mode, so station3.yaml only carries a base value and conf/modes/*.yaml
# overrides it. Not meaningful to compare against any single Station-1 script.
MODE_SPECIFIC = {
    "algo.critic_warmup_steps", "algo.offline_pretrain_only",
    "algo.train_offline_rl", "algo.offline_rl_steps",
    "algo.offline_save_freq", "algo.critic_warmup_save_freq",
    "algo.total_timesteps",
}

# train_residual_rl_real.sh is the canonical full-online run; the others are
# variations on it.
CANONICAL = "train_residual_rl_real.sh"

# These land in a replay-buffer cache key, so a disagreement between Station-1
# scripts means they do not share a buffer - worth reporting loudly.
HASHED = {
    "task", "reward_shaping", "rl_camera", "real_max_steps",
    "algo.n_step", "algo.gamma", "algo.batch_size", "algo.offline_fraction",
    "algo.buffer_size", "algo.learning_starts", "algo.sampling_strategy",
    "algo.random_action_noise_scale", "offline_data.name", "offline_data.root",
    "offline_data.num_episodes", "offline_data.use_base_policy_for_base_actions",
    "offline_data.min_action_range", "offline_data.min_state_std",
}


def station1_values(script: Path) -> dict[str, str]:
    """{hydra key: value} for everything the script passes to the trainer."""
    text = script.read_text()
    cmd = re.search(r"CMD=\((.*?)\n\)", text, re.S)
    if not cmd:
        raise SystemExit(f"no CMD=( ... ) array found in {script}")

    var_to_key: dict[str, str] = {}
    for line in cmd.group(1).split("\n"):
        m = re.match(r'\s*"?([a-z_][\w.]*)="?\$\{([A-Z_][A-Z0-9_]*)\}"?', line.strip())
        if m:
            var_to_key[m.group(2)] = m.group(1)
    for m in re.finditer(r'CMD\+=\(([a-z_][\w.]*)="\$\{([A-Z_]+)\}"\)', text):
        var_to_key[m.group(2)] = m.group(1)

    values: dict[str, str] = {}
    for m in re.finditer(r'^([A-Z_][A-Z0-9_]*)=("?)([^\n#]*?)\2\s*(?:#.*)?$', text, re.M):
        values[m.group(1)] = m.group(3).strip()

    return {var_to_key[v]: val for v, val in values.items() if v in var_to_key}


def flatten(node, prefix="") -> dict[str, object]:
    out = {}
    for k, v in (node or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def norm(x) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, list):
        return "[" + ",".join(str(i) for i in x) + "]"
    return str(x).strip("'\"")


def equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-12
    except ValueError:
        return a.replace(" ", "") == b.replace(" ", "")


def main() -> int:
    scripts = sorted(STATION1.glob("train_residual_rl_*.sh"))
    scripts = [s for s in scripts if not s.name.endswith(".bak")]
    if not scripts:
        raise SystemExit(f"no Station-1 scripts found under {STATION1}")

    per_script = {s.name: station1_values(s) for s in scripts}
    if CANONICAL not in per_script:
        raise SystemExit(f"{CANONICAL} not found under {STATION1}")
    base = per_script[CANONICAL]

    # A HASHED value that differs between Station-1 scripts means those scripts
    # do NOT share a replay-buffer cache. Report it - it is a real bug there, and
    # the reason station3.yaml keeps these in one place.
    clashes = []
    for name, vals in per_script.items():
        if name == CANONICAL:
            continue
        for k in HASHED:
            a, b = base.get(k, "<absent>"), vals.get(k, "<absent>")
            if b != "<absent>" and a != "<absent>" and not equal(norm(a), norm(b)):
                clashes.append((k, CANONICAL, a, name, b))
    if clashes:
        print("  Station-1 scripts disagree on HASHED values - they do not share a buffer:")
        for k, n1, a, n2, b in clashes:
            print(f"    {k:<28} {n1} = {a!r}   vs   {n2} = {b!r}")
        print()

    mine = flatten(yaml.safe_load(STATION3_YAML.read_text()))
    bad = []
    for key, s1 in sorted(base.items()):
        if key in STATION3_SPECIFIC or key in MODE_SPECIFIC or str(s1).startswith("${"):
            continue
        got = mine.get(key, "<absent>")
        if not equal(norm(got), norm(s1)):
            bad.append((key, norm(s1), norm(got)))

    if bad:
        print(f"\n  {len(bad)} value(s) in station3.yaml disagree with Station 1:\n")
        print(f"  {'key':<44}{'Station-1':<46}station3.yaml")
        for k, a, b in bad:
            print(f"  {k:<44}{a:<46}{b}")
        return 1

    shared = len(set(base) - STATION3_SPECIFIC - MODE_SPECIFIC)
    print(f"  station3.yaml matches {CANONICAL} on all {shared} shared values "
          f"({len(STATION3_SPECIFIC & set(base))} Station-3-specific, "
          f"{len(MODE_SPECIFIC & set(base))} set per mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
