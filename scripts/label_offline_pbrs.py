#!/usr/bin/env python3
"""
Label an offline LeRobot (v2.1) dataset with stage-aware PBRS rewards.

This is the offline counterpart of the online StageAwarePBRSWrapper: it streams
each episode's frames + states through the SAME reward server (SARM / TCC) with
the SAME server-side hysteresis, then applies the identical Potential-Based
Reward Shaping (PBRS) transform

    r_pbrs = (r_sparse if keep_sparse_term else 0) + gamma * phi(stage') - phi(stage)

so the offline replay-buffer rewards used by RLPD are consistent with what the
policy sees online.

The result is written to a SIDECAR parquet (a NEW key; the dataset's own rewards
are never overwritten):

    columns: index, episode_index, frame_index, stage, r_sparse, reward_pbrs

The reward server is model-agnostic: this labeler always sends frames + states +
task; the SARM server uses all of them, the TCC server uses only the frames. So
point ``--server_url`` at whichever server is running — ``/health`` reports the
model type. IMPORTANT: label the SAME LeRobot dataset that resfit loads as
``offline_data`` so the global frame ``index`` keys align, then feed the sidecar
to training via ``offline_data.reward_parquet`` / ``reward_column``.

Prerequisites:
  * The reward server must already be running (start opensarm/reward_server.py or
    tcc_torch/reward_server.py first).
  * Run inside the resfit conda env:  conda run -n residual python scripts/label_offline_pbrs.py ...

Example:
    conda run -n residual python scripts/label_offline_pbrs.py \
        --dataset_root /home/qte9489/personal_abhi/Thesis-Docs/Reward_Func/reward_func_ws/RLRewardResearchWS/DatasetUtil/PickPlaceCan/SARM-robosuite-can-mh-stages_v21 \
        --server_url http://127.0.0.1:8001 \
        --task_prompt "pick up the can and place it in the bin" \
        --output ./pbrs_sarm_progress.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
import json

# Make the resfit package importable when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "deps" / "lerobot"))

from resfit.rl_finetuning.reward_models.reward_client import RewardModelClient  # noqa: E402


def _to_uint8_hwc(img, vflip: bool) -> np.ndarray:
    """Convert a LeRobot image ([C,H,W] float[0,1] or [H,W,C] uint8) to HWC uint8."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if vflip:
        arr = arr[::-1].copy()
    return arr


def parse_args():
    p = argparse.ArgumentParser(description="Label offline dataset with PBRS stage rewards via reward server")
    p.add_argument("--dataset_root", type=str, default=None, help="Local LeRobot v2.1 dataset root")
    p.add_argument("--repo_id", type=str, default="offline", help="LeRobot repo_id (used with --dataset_root)")
    p.add_argument("--server_url", type=str, default="http://127.0.0.1:8001")
    p.add_argument("--task_prompt", type=str, default="pick up the can and place it in the bin")
    p.add_argument("--image_key", type=str, default="observation.images.agentview")
    p.add_argument("--state_key", type=str, default="observation.state")
    p.add_argument("--output", type=str, required=True, help="Output sidecar parquet path")
    # gamma MUST match the RL critic discount (algo.gamma) so the offline PBRS
    # rewards telescope identically to the online shaping. query_every_k /
    # hysteresis_k default to the same values the online StageAwarePBRSWrapper
    # uses so offline and online stage assignment stay consistent.
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--potentials", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15])
    p.add_argument("--query_every_k", type=int, default=4)
    p.add_argument("--hysteresis_k", type=int, default=4)
    p.add_argument("--conf_threshold", type=float, default=0.80)
    p.add_argument("--keep_sparse_term", action="store_true", default=True)
    p.add_argument("--no_keep_sparse_term", dest="keep_sparse_term", action="store_false")
    p.add_argument("--image_vflip", action="store_true", default=False)
    # Reward mode. "pbrs" writes column reward_pbrs (potential-based). "milestone"
    # writes column reward_milestone (ratchet one-time stage payouts + terminal
    # success bonus); gamma/potentials are then irrelevant.
    p.add_argument("--reward_mode", type=str, default="pbrs", choices=["pbrs", "milestone"])
    p.add_argument("--milestone_payouts", type=float, nargs="+", default=[0.0, 0.1, 0.1, 0.1])
    p.add_argument("--milestone_success_bonus", type=float, default=0.7)
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="Subset of episodes (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    potentials = [float(x) for x in args.potentials]
    num_stages = len(potentials)
    # Reward column + milestone payouts (used only in milestone mode). A distinct
    # column name keeps PBRS and milestone parquets from being confused, and the
    # RL config-guard also cross-checks reward_mode.
    reward_col = "reward_pbrs" if args.reward_mode == "pbrs" else "reward_milestone"
    milestone_payouts = [float(x) for x in args.milestone_payouts]

    client = RewardModelClient(args.server_url, task_prompt=args.task_prompt)
    if not client.check():
        raise RuntimeError(
            f"Reward server not reachable at {args.server_url}. Start the SARM/TCC server first."
        )
    print(f"Reward server OK: {client.health()}")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if args.dataset_root:
        ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    else:
        ds = LeRobotDataset(args.repo_id)

    num_episodes = ds.num_episodes
    ep_from = ds.episode_data_index["from"]
    ep_to = ds.episode_data_index["to"]
    episodes = args.episodes if args.episodes else list(range(num_episodes))
    print(f"Dataset: {num_episodes} episodes, labeling {len(episodes)}")

    # Resolve image/state keys against the actual dataset schema. Different
    # datasets use different names (e.g. LeRobot-standard
    # 'observation.images.agentview'/'observation.state' vs the SARM v2.1
    # 'agentview-images-rgb'/'state'), so fall back to auto-detection.
    sample0 = ds[int(ep_from[episodes[0]].item())]
    available = list(sample0.keys())

    image_key = args.image_key
    if image_key not in sample0:
        img_cands = [k for k, v in sample0.items() if getattr(v, "ndim", 0) >= 3]
        pref = [k for k in img_cands if "agentview" in k.lower()]
        if pref:
            image_key = pref[0]
        elif img_cands:
            image_key = img_cands[0]
        else:
            raise KeyError(f"No image key found. Available keys: {available}")
        print(f"[auto] image_key '{args.image_key}' not present; using '{image_key}'")

    state_key = args.state_key
    if state_key not in sample0:
        for c in ("observation.state", "state"):
            if c in sample0:
                state_key = c
                break
        else:
            raise KeyError(f"No state key found. Available keys: {available}")
        print(f"[auto] state_key '{args.state_key}' not present; using '{state_key}'")

    print(f"Using image_key='{image_key}', state_key='{state_key}'")

    # Optional ground-truth stage source (for reporting accuracy). The SARM
    # datasets store a continuous progress value in 'reward' (= stage + tau);
    # GT stage = floor(reward), clamped to [0, num_stages-1].
    gt_key = None
    for c in ("reward", "next.reward"):
        if c in sample0:
            gt_key = c
            break
    acc_total = 0
    acc_correct = 0

    rows = {
        "index": [],
        "episode_index": [],
        "frame_index": [],
        "stage": [],
        "r_sparse": [],
        reward_col: [],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.parent / (out_path.name + ".tmp")

    # ------------------------------------------------------------------
    # Config fingerprint + checkpoint/resume.
    # The parquet stores every labeling parameter in its schema metadata. On
    # restart we only resume if ALL of them match, so a resumed file can never
    # mix rewards computed under different hysteresis / potentials / server /
    # dataset settings. Otherwise we refuse (use a new --output or delete).
    # ------------------------------------------------------------------
    health = client.health()
    config = {
        "gamma": float(args.gamma),
        "potentials": [float(x) for x in potentials],
        "query_every_k": int(args.query_every_k),
        "hysteresis_k": int(args.hysteresis_k),
        "conf_threshold": float(args.conf_threshold),
        "keep_sparse_term": bool(args.keep_sparse_term),
        "image_vflip": bool(args.image_vflip),
        "num_stages": int(num_stages),
        "reward_mode": str(args.reward_mode),
        "milestone_payouts": milestone_payouts,
        "milestone_success_bonus": float(args.milestone_success_bonus),
        "image_key": image_key,
        "state_key": state_key,
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "repo_id": args.repo_id,
        "server_model_type": health.get("model_type"),
        "server_scheme": health.get("scheme"),
    }
    config_json = json.dumps(config, sort_keys=True)

    completed: set[int] = set()
    if out_path.exists():
        existing = pq.read_table(out_path)
        meta = existing.schema.metadata or {}
        stored_json = (meta.get(b"labeler_config", b"") or b"").decode() or "{}"
        if json.loads(stored_json) != json.loads(config_json):
            raise SystemExit(
                f"Refusing to resume: labeling config in {out_path} differs from the current run.\n"
                f"  stored : {stored_json}\n"
                f"  current: {config_json}\n"
                f"Use a different --output, or delete the existing file to relabel from scratch."
            )
        d = existing.to_pydict()
        for k in rows:
            rows[k] = list(d[k])
        completed = {int(e) for e in d["episode_index"]}
        acc_total = int((meta.get(b"acc_total", b"0") or b"0").decode())
        acc_correct = int((meta.get(b"acc_correct", b"0") or b"0").decode())
        print(
            f"Resuming from {out_path}: {len(completed)} episodes already done, "
            f"{len(rows['index'])} frames, running stage_acc={acc_correct}/{acc_total}."
        )

    def save_checkpoint():
        """Atomically write the current rows + config + accuracy to the parquet."""
        table = pa.table(
            {
                "index": np.array(rows["index"], dtype=np.int64),
                "episode_index": np.array(rows["episode_index"], dtype=np.int64),
                "frame_index": np.array(rows["frame_index"], dtype=np.int64),
                "stage": np.array(rows["stage"], dtype=np.int32),
                "r_sparse": np.array(rows["r_sparse"], dtype=np.float32),
                reward_col: np.array(rows[reward_col], dtype=np.float32),
            }
        )
        if table.num_rows:
            table = table.sort_by("index")
        table = table.replace_schema_metadata(
            {
                b"labeler_config": config_json.encode(),
                b"acc_total": str(acc_total).encode(),
                b"acc_correct": str(acc_correct).encode(),
            }
        )
        pq.write_table(table, tmp_path)
        tmp_path.replace(out_path)  # atomic rename on the same filesystem

    pending = [e for e in episodes if e not in completed]
    if not pending:
        print("All requested episodes already labeled. Nothing to do.")
    # Reuse a SINGLE server session for all episodes: each episode issues
    # reset=True on its first frame, which clears the server-side rolling buffer
    # and hysteresis. Using one session_id (instead of one per episode) prevents
    # the server's session dict from growing unboundedly (which OOM-killed a run).
    session_id = "offline-label"
    ep_bar = tqdm(pending, desc="episodes", unit="ep")
    for ep in ep_bar:
        start = int(ep_from[ep].item())
        end = int(ep_to[ep].item())  # exclusive
        n = end - start

        # PBRS / milestone episode state
        current_stage = 0
        last_paid_stage = 0
        pending_frames: list[np.ndarray] = []
        pending_states: list[np.ndarray] = []
        step_ctr = 0

        def _flush(reset: bool):
            nonlocal current_stage
            if not pending_frames:
                return
            frames = np.stack(pending_frames, axis=0)
            states = np.stack(pending_states, axis=0).astype(np.float32)
            out = client.predict_stage(
                frames,
                states,
                session_id=session_id,
                num_anchors=frames.shape[0],
                reset=reset,
                hysteresis_k=args.hysteresis_k,
                conf_threshold=args.conf_threshold,
                monotonic=True,
                task=args.task_prompt,
            )
            stage = int(out["gated_stage"])
            current_stage = max(current_stage, stage)
            current_stage = min(max(current_stage, 0), num_stages - 1)
            pending_frames.clear()
            pending_states.clear()

        # Accumulate this episode into LOCAL buffers; only commit + checkpoint
        # once the whole episode succeeds, so a crash mid-episode discards only
        # this episode and never corrupts the on-disk file.
        ep_rows = {k: [] for k in rows}
        ep_acc_total = 0
        ep_acc_correct = 0

        # next.reward (source-frame) convention: reward at frame i is the PBRS
        # reward of transition i -> i+1.
        prev_gidx = prev_frame_index = prev_stage = None
        phi_prev = None

        try:
            for local_i in tqdm(range(n), desc=f"ep {ep}", unit="f", leave=False):
                gidx = start + local_i
                sample = ds[gidx]
                img = _to_uint8_hwc(sample[image_key], args.image_vflip)
                state = np.asarray(sample[state_key], dtype=np.float32).reshape(-1)
                terminated = local_i == (n - 1)  # demos end in success (true terminal)

                pending_frames.append(img)
                pending_states.append(state)

                # Query cadence: reset on frame 0, then every k steps (and on the
                # final frame). After this, current_stage is the gated stage.
                if local_i == 0:
                    _flush(reset=True)
                else:
                    step_ctr += 1
                    if (step_ctr % args.query_every_k == 0) or terminated:
                        _flush(reset=False)

                if gt_key is not None:
                    gt_val = float(np.asarray(sample[gt_key]).reshape(-1)[0])
                    gt_stage = min(num_stages - 1, max(0, int(np.floor(gt_val))))
                    ep_acc_total += 1
                    ep_acc_correct += int(current_stage == gt_stage)

                if local_i == 0:
                    phi_prev = potentials[current_stage]
                    last_paid_stage = current_stage
                    prev_gidx, prev_frame_index, prev_stage = gidx, local_i, current_stage
                    continue

                r_sparse = 1.0 if terminated else 0.0
                if args.reward_mode == "milestone":
                    # Ratchet milestone: pay once per newly-entered stage (skipped
                    # stages summed) + dominant terminal success bonus gated on the
                    # true success flag. Same math as the online wrapper.
                    prev_reward = 0.0
                    if current_stage > last_paid_stage:
                        for _s in range(last_paid_stage + 1, current_stage + 1):
                            prev_reward += milestone_payouts[_s]
                        last_paid_stage = current_stage
                    if terminated and r_sparse > 0.5:
                        prev_reward += float(args.milestone_success_bonus)
                else:
                    phi_curr = 0.0 if terminated else potentials[current_stage]
                    prev_reward = (r_sparse if args.keep_sparse_term else 0.0) + args.gamma * phi_curr - phi_prev
                    phi_prev = phi_curr

                ep_rows["index"].append(prev_gidx)
                ep_rows["episode_index"].append(ep)
                ep_rows["frame_index"].append(prev_frame_index)
                ep_rows["stage"].append(prev_stage)
                ep_rows["r_sparse"].append(float(r_sparse))
                ep_rows[reward_col].append(float(prev_reward))

                prev_gidx, prev_frame_index, prev_stage = gidx, local_i, current_stage
        except KeyboardInterrupt:
            print(
                f"\nInterrupted during episode {ep}. {len(completed)} completed episodes "
                f"are safely checkpointed in {out_path}. Re-run the same command to resume."
            )
            return
        except Exception as e:  # noqa: BLE001
            print(
                f"\nError during episode {ep}: {type(e).__name__}: {e}\n"
                f"The {len(completed)} previously completed episodes remain safe in {out_path}. "
                f"Re-run to resume from episode {ep}."
            )
            raise

        # Episode finished -> commit local buffers and atomically checkpoint.
        for k in rows:
            rows[k].extend(ep_rows[k])
        acc_total += ep_acc_total
        acc_correct += ep_acc_correct
        completed.add(int(ep))
        save_checkpoint()

        _post = {"final_stage": current_stage}
        if gt_key is not None and acc_total > 0:
            _post["stage_acc"] = f"{100.0 * acc_correct / acc_total:.1f}%"
        ep_bar.set_postfix(_post)

    save_checkpoint()  # idempotent final write
    print(f"Wrote {len(rows['index'])} labeled frames to {out_path}")
    if gt_key is not None and acc_total > 0:
        print(
            f"Gated-vs-GT stage accuracy: {100.0 * acc_correct / acc_total:.1f}% "
            f"({acc_correct}/{acc_total} frames, GT from '{gt_key}')"
        )
    else:
        print("Gated-vs-GT stage accuracy: n/a (no GT reward field found)")


if __name__ == "__main__":
    main()
