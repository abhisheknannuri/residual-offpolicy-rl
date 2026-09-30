"""Build ``resfit/rl_finetuning/scripts/train_residual_td3_distributed.py``.

The distributed entrypoint reuses large, already-debugged regions of
``train_residual_td3.py``'s ``main()`` (offline-buffer population, reward-parquet
validation, offline RL, critic warmup, the update loop, logging, checkpointing).
Those regions live inside one 2200-line function and cannot be imported, and the
original file must not be edited. Hand-retyping ~900 lines is how silent bugs
get in, so instead the template marks regions with::

    #@@SPLICE <first> <last> [shift=+N|-N]@@

and this script substitutes the original's exact lines. Every region's first and
last line are checked against ``ANCHORS`` below, so if ``train_residual_td3.py``
is edited and its line numbers move, the build FAILS instead of splicing the
wrong code. ``shift`` re-indents a region; it refuses regions containing
triple-quoted strings (re-indenting would change string contents).

Usage (no uv - call an interpreter directly):
    <python> resfit/rl_finetuning/off_policy/distributed/tools/build_entrypoint.py [--check]

``--check`` exits non-zero if the generated file on disk is out of date.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
ORIGINAL = REPO / "resfit/rl_finetuning/scripts/train_residual_td3.py"
TEMPLATE = Path(__file__).with_name("train_residual_td3_distributed.template.py")
OUTPUT = REPO / "resfit/rl_finetuning/scripts/train_residual_td3_distributed.py"

# (first, last) -> (expected first line, expected last line), compared after .strip()
ANCHORS: dict[tuple[int, int], tuple[str, str]] = {
    (305, 326): ('device_str = "cuda" if torch.cuda.is_available() else "cpu"', "torch.backends.cudnn.allow_tf32 = True"),
    (337, 375): ("base_policy: ACTPolicy | None = None", 'raise ValueError(f"Unknown base policy type: {type(base_cfg)}")'),
    (377, 437): ("# Load dataset and get normalization functions early", ")"),
    (439, 583): ("def get_envs(", ")"),
    (588, 603): ("if cfg.seed is None:", 'print(f"Set random seed to {cfg.seed}")'),
    (608, 643): ('assert cfg.num_envs == 1, "Only support 1 environment for now because of how n_step is implemented"', ")"),
    (700, 732): ("# Seed environments explicitly for reproducibility", "signal.signal(signal.SIGTERM, _signal_handler)"),
    (791, 797): ("# Set up actor learning rate warmup", ")"),
    (806, 814): ('alpha = cfg.algo.priority_alpha if cfg.algo.sampling_strategy == "prioritized_replay" else 0.0', ""),
    (831, 934): ("", ")"),
    (936, 1347): ("# Normalization functions already defined above - use them", 'print("Skipping offline buffer population for online-only training")'),
    (1363, 1383): ("if cfg.algo.offline_pretrain_only:", 'print(f"Warm-up: filling online buffer with {cfg.algo.learning_starts - len(online_rb)} random steps…")'),
    (1405, 1408): ("_save_freq = cfg.algo.online_warmup_save_freq", ")"),
    (1416, 1469): ("if cfg.algo.use_base_policy_for_warmup:", ")"),
    (1488, 1496): ("if next_save_threshold is not None and len(online_rb) >= next_save_threshold:", "next_save_threshold += _save_freq"),
    (1498, 1498): ("obs = next_obs  # roll state", "obs = next_obs  # roll state"),
    (1499, 1507): ("online_cache_dir.mkdir(parents=True, exist_ok=True)", "loaded_online_from_cache = True  # treat as cached going forward"),
    (1509, 1572): ("_hp_parts: list[str] = [", ""),
    (1575, 1605): ("global_step = 0", "training_timer = TrainingTimer()"),
    (1607, 1705): ("def _run_critic_warmup(", ""),
    (1709, 1947): ('if getattr(cfg.algo, "train_offline_rl", False):', "return"),
    (1949, 1967): ("# Reward-model debug: randomly record annotated TRAINING rollouts. Only for", ")"),
    (1970, 2060): ("iter_start = time.time()", "obs = next_obs  # roll"),
    (2127, 2187): ("# 1) Save a timestamped checkpoint for history", "optimized_replay_buffer_dumps(online_rb, online_buffer_dir)"),
    (2193, 2276): ("i = 0", "i += 1"),
    (2284, 2406): ("sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0", "print(print_msg)"),
    (2408, 2457): ('print(f"Training finished in {time.time() - train_start_time:.2f} seconds.")', 'print("Run directory cleaned up successfully.")'),
}

MARKER = re.compile(r"^\s*#@@SPLICE (\d+) (\d+)(?: shift=([+-]\d+))?@@\s*$")


def _reindent(lines: list[str], shift: int, where: str) -> list[str]:
    if shift == 0:
        return lines
    if any('"""' in ln or "'''" in ln for ln in lines):
        raise SystemExit(f"{where}: refusing to re-indent a region containing triple-quoted strings")
    out = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
        elif shift > 0:
            out.append(" " * shift + ln)
        else:
            if not ln.startswith(" " * -shift):
                raise SystemExit(f"{where}: cannot dedent line by {-shift}: {ln!r}")
            out.append(ln[-shift:])
    return out


def build() -> str:
    orig = ORIGINAL.read_text().splitlines()
    out: list[str] = []
    used: set[tuple[int, int]] = set()
    for lineno, line in enumerate(TEMPLATE.read_text().splitlines(), 1):
        m = MARKER.match(line)
        if not m:
            out.append(line)
            continue
        a, b = int(m.group(1)), int(m.group(2))
        shift = int(m.group(3) or 0)
        where = f"template:{lineno} SPLICE {a}-{b}"
        if (a, b) not in ANCHORS:
            raise SystemExit(f"{where}: no anchor registered for this range")
        exp_first, exp_last = ANCHORS[(a, b)]
        region = orig[a - 1 : b]
        if region[0].strip() != exp_first or region[-1].strip() != exp_last:
            raise SystemExit(
                f"{where}: train_residual_td3.py changed under this splice.\n"
                f"  expected first: {exp_first!r}\n  actual   first: {region[0].strip()!r}\n"
                f"  expected last : {exp_last!r}\n  actual   last : {region[-1].strip()!r}\n"
                "Re-derive the line range, update ANCHORS and the template, rebuild."
            )
        used.add((a, b))
        tag = f"train_residual_td3.py:{a}-{b}" + (f" (re-indented {shift:+d})" if shift else "")
        indent = " " * (len(region[0]) - len(region[0].lstrip()) + shift) if region[0].strip() else ""
        out.append(f"{indent}# ↓↓↓ verbatim from {tag}")
        out.extend(_reindent(region, shift, where))
        out.append(f"{indent}# ↑↑↑ end verbatim {tag}")
    unused = set(ANCHORS) - used
    if unused:
        raise SystemExit(f"anchors registered but not used by the template: {sorted(unused)}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the generated file is stale")
    args = ap.parse_args()
    text = build()
    compile(text, str(OUTPUT), "exec")  # syntax check before touching disk
    if args.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != text:
            print(f"STALE: {OUTPUT} does not match template + original. Rebuild it.")
            return 1
        print(f"OK: {OUTPUT.name} is up to date with its template and train_residual_td3.py")
        return 0
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
