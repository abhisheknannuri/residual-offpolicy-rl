# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Offline dataset-preprocessing pass: converts a `command_space: joint`
LeRobot v2.1 dataset's absolute joint actions into PER-FRAME DELTA actions
(joints 0-5 only - gripper is left absolute/untouched), and trims each
episode's leading AND trailing "dead time" (frames recorded before the
operator actually started moving the leader arm, and/or frames recorded
after the task was already done while the operator took a moment to click
"Stop Recording").

See `trossen_real/datasets/PickAndInsertCube.md` ("Delta-joint action
preprocessing" section) for the full design rationale, the re-verification
of `~/work/aiire_control/docs/pi_policy_data_pipeline.md` this is based on,
and exactly why the design below differs from that doc's own pi0/openpi
mechanism (chunk-start-referenced delta) in a way that's deliberately
BETTER SUITED to this dataset's specific action semantics - short version:

  action[t] (this dataset) = "the absolute joint target commanded THIS
  tick, computed client-side as current_joints[t] + safety-clamped delta"
  (see `control_loop.py::_tick()`'s `command_space == "joint"` branch).

  So `action[t] - state[t]` (state = observation.state = current_joints[t],
  measured at the START of the SAME tick, BEFORE the action was computed)
  recovers EXACTLY the real, bounded, physically-small commanded delta for
  that tick - not an approximation, and not dependent on any chunk-start
  reference. Gripper (last dim) is excluded from this subtraction and
  copied through unchanged, matching the "exclude_joints=['gripper']"
  convention used by both `openpi`'s `DeltaActions` and this repo's own
  `lerobot_v3` fork's `RelativeActionsProcessorStep` (see the doc).

Produces a NEW, separate v2.1 dataset directory (never modifies the
original in place) with:
  - `action` = [delta_joint_0, ..., delta_joint_5, gripper_absolute]
  - `observation.state` and all 5 auxiliary observations: UNCHANGED
    (still absolute) - only `action` changes representation.
  - Each episode's leading AND trailing near-zero-motion frames trimmed
    (see `compute_trim_cutoff()`/`compute_end_trim_cutoff()` - the latter
    is the exact same detection logic run on the reversed per-frame speed
    signal, symmetric by construction). Middle-of-episode frames are NEVER
    trimmed, no matter how still the arm gets mid-task (e.g. holding a
    grasp) - only a single leading run and a single trailing run are ever
    considered for removal.
  - `meta/info.json`/`stats.json`/`episodes.jsonl`/`episodes_stats.jsonl`/
    `tasks.jsonl` all regenerated from scratch by the standard
    `LeRobotDataset.create()`/`add_frame()`/`save_episode()` API (same API
    `dataset_recorder.py` itself uses to record data) - not hand-patched,
    so they're guaranteed self-consistent.
  - videos re-encoded (same codec/pix_fmt as the original, via the same
    `encode_video_frames()` default parameters) containing only the
    surviving (post-trim) frames.

Usage:
    # 1. ALWAYS dry-run first - reports per-episode trim cutoffs with ZERO
    #    writes, so you can sanity-check thresholds against real numbers
    #    before spending time on the (slow, video-encoding) real pass.
    python -m trossen_real.scripts.convert_to_delta_joint_dataset \\
        --input-root trossen_real/datasets/20260805_..._PickAndInsertCube_old \\
        --dry-run

    # 2. Try a small subset first.
    python -m trossen_real.scripts.convert_to_delta_joint_dataset \\
        --input-root trossen_real/datasets/20260805_..._PickAndInsertCube_old \\
        --output-root /tmp/delta_joint_test \\
        --episodes 0,1,2

    # 3. Full conversion.
    python -m trossen_real.scripts.convert_to_delta_joint_dataset \\
        --input-root trossen_real/datasets/20260805_..._PickAndInsertCube_old \\
        --output-root trossen_real/datasets/20260805_..._PickAndInsertCube_deltajoint
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import decode_video_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUTO_MANAGED_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _passthrough_keys(ds: LeRobotDataset) -> tuple[str, ...]:
    """Every column copied through UNCHANGED for each surviving frame -
    everything except "action" (gets the delta transform), the 5
    auto-managed bookkeeping columns (`add_frame()`/`save_episode()`
    regenerate these automatically), and camera/video keys (handled
    separately - batch-decoded from video, see `convert_episode()`).

    Derived from the SOURCE dataset's OWN schema (`ds.meta.features`), NOT a
    fixed hardcoded list - so this adapts automatically to whatever columns
    a given dataset actually has. This used to be a hardcoded tuple
    (`PASSTHROUGH_OBS_KEYS`) covering exactly teleop's schema at the time it
    was written. Running this against an infer-app-collected dataset (which
    has 3 EXTRA columns - `observation.intervened`, `observation.
    policy_action`, `next.reward`, see `infer_loop.py`) then failed
    `validate_frame()` with "Missing features" - the OUTPUT dataset's
    schema (`new_features` in `main()`, below) was already generic (copied
    wholesale from `ds.meta.features`), but this per-frame COPY list wasn't
    - it silently never populated those 3 columns even though the schema it
    was writing into demanded them. Confirmed and fixed by making this
    generic too, the same way `new_features` already was.
    """
    camera_keys = set(ds.meta.camera_keys)
    return tuple(k for k in ds.meta.features if k != "action" and k not in AUTO_MANAGED_KEYS and k not in camera_keys)


def _cast_passthrough_value(ds: LeRobotDataset, key: str, value) -> np.ndarray:
    """Cast + reshape one passthrough column's per-frame value to whatever
    dtype AND shape its OWN schema entry declares - bool columns
    (`next.done`, `observation.intervened`) stay bool; everything else
    (float32 columns) gets cast to float32 - then reshaped to the declared
    shape regardless of how HF happened to hand it back this time.

    Both dtype AND shape are read from `ds.meta.features[key]` rather than
    hardcoded/assumed, because they turn out to NOT be uniform across
    passthrough columns: a shape-`(7,)` column like `observation.state`
    round-trips through `np.asarray(raw[key])[i]` already correctly shaped,
    but a shape-`(1,)` SCALAR column can come back from the underlying HF
    dataset as a bare 0-d value (shape `()`) rather than a length-1 array -
    confirmed happens for `next.reward` (`next.done`/`observation.
    intervened` didn't hit this before only because their bool branch
    already wrapped explicitly, which was correct for them but coincidental
    - not a general float-column solution). `.reshape(shape)` handles both
    the correctly-shaped and bare-scalar cases uniformly, driven by the
    schema instead of by guessing which shape a given column happens to
    arrive in.
    """
    dtype = ds.meta.features[key]["dtype"]
    shape = tuple(ds.meta.features[key]["shape"])
    np_dtype = bool if dtype == "bool" else np.float32
    return np.asarray(value, dtype=np_dtype).reshape(shape)

# Companion file written into the OUTPUT dataset root recording exactly how
# it was produced - mirrors `dataset_recorder.SAVED_CONFIG_FILENAME`'s
# "the dataset should be self-describing" philosophy.
PROVENANCE_FILENAME = "DELTA_JOINT_TRANSFORM.md"


@dataclass
class TrimConfig:
    """See `compute_trim_cutoff()`. Defaults were empirically calibrated
    against the real `PickAndInsertCube` dataset (109 episodes) - see
    `PickAndInsertCube.md` for the calibration numbers. Re-check with
    `--dry-run` on any new dataset before trusting these defaults blindly;
    every real episode inspected showed a consistent ~0.008-0.012 rad
    "resting floor" (encoder/tracking noise, present even while
    perfectly still) followed by a clear, sustained rise once the operator
    actually started moving - these defaults are tuned to that shape.
    """

    baseline_window: int = 10        # frames used to estimate the per-episode "resting floor"
    threshold_multiplier: float = 2.5  # cutoff threshold = max(floor * this, min_abs_speed)
    min_abs_speed: float = 0.015     # rad - absolute floor under the multiplier-based threshold
    sustain_frames: int = 4          # consecutive frames that must all exceed threshold
    max_trim_frames: int = 150       # safety cap (~7.5s @ 20Hz) - never trim more than this
    min_remaining_frames: int = 50   # if trimming would leave fewer frames, skip trimming entirely


def compute_trim_cutoff(speed: np.ndarray, cfg: TrimConfig) -> tuple[int, str]:
    """`speed[t]` = per-frame motion magnitude (max|delta_joint| over the 6
    arm joints, `action[:6]-state[:6]`) for one episode, in original
    (pre-trim) frame order.

    Returns `(cutoff, note)` - `cutoff` frames `[0, cutoff)` should be
    dropped; `note` is empty unless something noteworthy happened (capped
    by `max_trim_frames`, no rise detected, or trimming skipped because too
    little would remain).

    Algorithm: estimate a per-episode "resting floor" from the SMALLEST
    speed seen in the first `baseline_window` frames (min, not mean/median
    - robust even if the operator started moving almost immediately, since
    real motion frames would only ever push the observed min UP relative
    to true floor, never down). Threshold = floor * multiplier, floored at
    `min_abs_speed` so a pathologically-quiet baseline can't produce a
    near-zero threshold that triggers on pure noise. Cutoff = first index
    where `sustain_frames` CONSECUTIVE frames all exceed that threshold
    (requiring sustained rise, not a single blip, avoids false-triggering
    on one noisy sample).
    """
    n = len(speed)
    window = min(cfg.baseline_window, n)
    floor = float(np.min(speed[:window])) if window > 0 else 0.0
    threshold = max(floor * cfg.threshold_multiplier, cfg.min_abs_speed)

    cutoff = 0
    found = False
    for i in range(max(n - cfg.sustain_frames, 0)):
        if speed[i:i + cfg.sustain_frames].min() > threshold:
            cutoff = i
            found = True
            break

    note = ""
    if not found:
        note = "no sustained motion-onset found - not trimming"
        cutoff = 0
    elif cutoff > cfg.max_trim_frames:
        note = f"cutoff {cutoff} exceeded max_trim_frames, capped to {cfg.max_trim_frames}"
        cutoff = cfg.max_trim_frames

    if n - cutoff < cfg.min_remaining_frames:
        note = (f"trim would leave only {n - cutoff} frames (< min_remaining_frames="
                f"{cfg.min_remaining_frames}) - not trimming")
        cutoff = 0

    return cutoff, note


def compute_end_trim_cutoff(speed: np.ndarray, cfg: TrimConfig) -> tuple[int, str]:
    """Symmetric counterpart to `compute_trim_cutoff()`, for trailing dead
    time at the END of an episode (e.g. operator finished the task, then
    took a moment to reach for/click "Stop Recording" while the arm just
    held its final pose).

    Reuses the EXACT same detection logic by reversing the input: the
    "resting floor" is estimated from the LAST `baseline_window` frames
    (assumed to be the trailing static hold, if one exists) instead of the
    first, and the scan for "sustained motion" runs backward from the end.
    Returns the number of trailing frames `[n-end_cutoff, n)` that should be
    dropped (0 if no trailing dead time is detected, e.g. the operator
    clicked stop immediately after finishing).
    """
    end_cutoff, note = compute_trim_cutoff(speed[::-1], cfg)
    return end_cutoff, note


def _gripper_mask(action_names: list[str]) -> np.ndarray:
    """True where a dim is a gripper channel (kept absolute), False where
    it's an arm joint (converted to delta). Name-based (not positional),
    matching `lerobot_v3`'s own `RelativeActionsProcessorStep._build_mask`
    convention - works for both single-arm (`[...,"gripper"]`) and
    dual-arm (`[...,"left_gripper",...,"right_gripper"]`) naming."""
    return np.array([name.lower() == "gripper" or name.lower().endswith("_gripper") for name in action_names])


def _to_hwc_uint8(frame_chw_float: np.ndarray) -> np.ndarray:
    """(3,H,W) float32 [0,1] (as returned by `decode_video_frames`) -> (H,W,3)
    uint8 [0,255] (the format `add_frame()`/`write_image()` expects - the
    same convention the ORIGINAL raw camera frames were captured/recorded
    in, see `video_utils.py`'s own HWC-uint8-to-CHW-float encode direction
    this is the exact inverse of)."""
    hwc = np.transpose(frame_chw_float, (1, 2, 0))
    return np.clip(hwc * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _load_action_names_and_validate_joint_mode(ds: LeRobotDataset) -> list[str]:
    action_names = ds.meta.features["action"]["names"]
    joint_like = [n for n in action_names if n.lower().startswith("joint_") or n.lower().endswith("_joint")]
    ee_like = [n for n in action_names if n.lower() in ("dx", "dy", "dz") or n.lower().endswith(("_dx", "_dy", "_dz"))]
    if ee_like or not joint_like:
        raise ValueError(
            f"This script only supports command_space: joint datasets (action names look like "
            f"['joint_0',...,'gripper']). Got action names={action_names!r}, which looks like "
            f"command_space: ee instead - refusing to run (delta-vs-state math here is specific "
            f"to how `command_space: joint` actions are computed, see the module docstring)."
        )
    return action_names


def analyze_episode(ds: LeRobotDataset, ep: int, gripper_mask: np.ndarray, cfg: TrimConfig) -> dict:
    ep_start = int(ds.episode_data_index["from"][ep])
    ep_end = int(ds.episode_data_index["to"][ep])
    actions = np.asarray(ds.hf_dataset[ep_start:ep_end]["action"], dtype=np.float32)
    states = np.asarray(ds.hf_dataset[ep_start:ep_end]["observation.state"], dtype=np.float32)
    joint_idx = ~gripper_mask
    speed = np.max(np.abs(actions[:, joint_idx] - states[:, joint_idx]), axis=1)
    length = ep_end - ep_start

    # START-TRIM DISABLED (2026-09-01): compute_trim_cutoff()'s "resting floor"
    # estimate (min of the first baseline_window frames) assumes every episode
    # opens with a genuine still period to calibrate against - true for teleop
    # data (operator pauses before grabbing the leader), but NOT for infer-app/
    # policy-rollout episodes, which can start already mid-motion (the policy
    # issuing real commands from tick 0, no equivalent "wait for the operator"
    # pause). Confirmed on episode 60 of the 2026-09-01 PickAndInsertCube_TS1_
    # Iter1 batch, both quantitatively (speed[0:15] shows smooth, continuous,
    # already-in-progress motion, never near-zero - the "floor" of 0.056 is
    # just the slowest point of an ALREADY-MOVING trajectory, not real
    # stillness, so the derived threshold ends up higher than most of the
    # genuine reach motion) AND visually (extracted real frames from the
    # video: frame 0 already mid-reach toward the cube, well before the
    # computed cutoff at frame 102, where the gripper is already AT the cube
    # about to grip - the entire real approach got discarded as "dead time").
    # Left disabled here (function kept below, not deleted) pending a proper,
    # separately-derived heuristic from inspecting the full human-collected
    # (teleop) datasets - NOT reworked ad hoc, since there's no reliable
    # per-frame signal (no pedal, no next.done) to distinguish genuine dead
    # time from slow real motion for this data lineage.
    # cutoff, note = compute_trim_cutoff(speed, cfg)
    cutoff, note = 0, ""
    # END-TRIM DISABLED (2026-09-01): compute_end_trim_cutoff() decides what to cut
    # PURELY from arm motion speed, with zero awareness of next.done/next.reward -
    # a successful episode's tail (arm holds still while the reward pedal is held to
    # mark success) looks EXACTLY like the "dead time" this was meant to remove, so
    # it reliably trimmed away real, semantically-important frames - up to and
    # including the true terminal next.done=True frame itself (confirmed: hit
    # 26/65 episodes in the 2026-09-01 PickAndInsertCube_TS1_Iter1 batch). Left
    # disabled here (function kept below, not deleted, in case a done/reward-aware
    # version of end-trimming is worth reviving later) rather than reworked, since
    # the real fix belongs upstream anyway: next.done should be WIDE (True for
    # every frame where next.reward==1, not just a single terminal tick) at
    # COLLECTION time in infer_loop.py/control_loop.py - see the verified OLD
    # dataset convention (PickAndInsertCube_Station1_merged_deltajoint_old): its
    # done==True frames are EXACTLY its reward==1 frames, every episode checked.
    # end_cutoff, end_note = compute_end_trim_cutoff(speed, cfg)
    end_cutoff, end_note = 0, ""

    # Combined safety check: if BOTH trims together would eat too much of the
    # episode, prioritize the START trim (dead-time-before-recording-started
    # is the more clearly-understood/common case) and drop the END trim
    # rather than risk an over-aggressive combined cut.
    if length - cutoff - end_cutoff < cfg.min_remaining_frames:
        end_note = (f"start+end trim would leave only {length - cutoff - end_cutoff} frames "
                    f"(< min_remaining_frames={cfg.min_remaining_frames}) - dropping the END trim, keeping START")
        end_cutoff = 0

    note_parts = [f"start: {note}" if note else "", f"end: {end_note}" if end_note else ""]
    combined_note = " | ".join(p for p in note_parts if p)

    return {
        "episode_index": ep,
        "ep_start": ep_start,
        "ep_end": ep_end,
        "length": length,
        "cutoff": cutoff,
        "end_cutoff": end_cutoff,
        "remaining": length - cutoff - end_cutoff,
        "note": combined_note,
    }


def convert_episode(
    ds: LeRobotDataset,
    new_ds: LeRobotDataset,
    ep: int,
    gripper_mask: np.ndarray,
    cfg: TrimConfig,
    passthrough_keys: tuple[str, ...],
) -> dict:
    """Read one episode from `ds`, apply trim + delta-action conversion,
    write the result into `new_ds` (must already exist via `.create()`)."""
    info = analyze_episode(ds, ep, gripper_mask, cfg)
    ep_start, ep_end, cutoff, end_cutoff = info["ep_start"], info["ep_end"], info["cutoff"], info["end_cutoff"]
    keep_end = (ep_end - ep_start) - end_cutoff  # local index (exclusive) of the last kept frame + 1

    columns = ["action", *passthrough_keys, "timestamp", "task_index"]
    raw = ds.hf_dataset[ep_start:ep_end]
    arrays = {key: np.asarray(raw[key]) for key in columns}

    actions = arrays["action"].astype(np.float32)
    states = arrays["observation.state"].astype(np.float32)
    joint_idx = ~gripper_mask
    delta_actions = actions.copy()
    delta_actions[:, joint_idx] = actions[:, joint_idx] - states[:, joint_idx]
    # Gripper dim(s) left exactly as-is (still absolute) - `joint_idx` mask excludes them.

    task_index = int(arrays["task_index"][0])
    task_str = ds.meta.tasks[task_index]

    kept_timestamps = arrays["timestamp"][cutoff:keep_end].astype(np.float64).tolist()
    n_kept = len(kept_timestamps)

    # Batch-decode only the SURVIVING frames' images, once per camera per
    # episode (far fewer, much faster video-reader calls than one `ds[idx]`
    # per frame - see module docstring / PickAndInsertCube.md).
    decoded_images: dict[str, np.ndarray] = {}
    for cam_key in ds.meta.camera_keys:
        video_path = ds.root / ds.meta.get_video_file_path(ep, cam_key)
        frames = decode_video_frames(video_path, kept_timestamps, ds.tolerance_s, ds.video_backend)
        decoded_images[cam_key] = frames.numpy()  # (T, 3, H, W) float32 [0,1]

    for j in range(n_kept):
        i = cutoff + j
        frame = {"action": delta_actions[i]}
        for key in passthrough_keys:
            frame[key] = _cast_passthrough_value(ds, key, arrays[key][i])
        for cam_key in ds.meta.camera_keys:
            frame[cam_key] = _to_hwc_uint8(decoded_images[cam_key][j])
        new_ds.add_frame(frame=frame, task=task_str)

    new_ds.save_episode()
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", required=True, help="Path to the source v2.1 dataset (command_space: joint).")
    parser.add_argument("--output-root", default=None, help="Path for the new dataset (required unless --dry-run).")
    parser.add_argument("--dry-run", action="store_true", help="Only report per-episode trim cutoffs; write nothing.")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode indices to process (default: all).")
    parser.add_argument("--baseline-window", type=int, default=TrimConfig.baseline_window)
    parser.add_argument("--threshold-multiplier", type=float, default=TrimConfig.threshold_multiplier)
    parser.add_argument("--min-abs-speed", type=float, default=TrimConfig.min_abs_speed)
    parser.add_argument("--sustain-frames", type=int, default=TrimConfig.sustain_frames)
    parser.add_argument("--max-trim-frames", type=int, default=TrimConfig.max_trim_frames)
    parser.add_argument("--min-remaining-frames", type=int, default=TrimConfig.min_remaining_frames)
    args = parser.parse_args()

    cfg = TrimConfig(
        baseline_window=args.baseline_window,
        threshold_multiplier=args.threshold_multiplier,
        min_abs_speed=args.min_abs_speed,
        sustain_frames=args.sustain_frames,
        max_trim_frames=args.max_trim_frames,
        min_remaining_frames=args.min_remaining_frames,
    )

    input_root = Path(args.input_root)
    ds = LeRobotDataset(repo_id="source", root=str(input_root))
    action_names = _load_action_names_and_validate_joint_mode(ds)
    gripper_mask = _gripper_mask(action_names)
    logger.info("action_names=%s -> gripper dims (kept absolute)=%s", action_names,
                [n for n, g in zip(action_names, gripper_mask) if g])

    all_episodes = list(range(ds.meta.total_episodes))
    episodes = (
        [int(e) for e in args.episodes.split(",")] if args.episodes else all_episodes
    )

    results = [analyze_episode(ds, ep, gripper_mask, cfg) for ep in episodes]
    total_len = sum(r["length"] for r in results)
    total_start_cutoff = sum(r["cutoff"] for r in results)
    total_end_cutoff = sum(r["end_cutoff"] for r in results)
    total_cutoff = total_start_cutoff + total_end_cutoff
    for r in results:
        flag = f"  <-- {r['note']}" if r["note"] else ""
        logger.info(
            "episode %3d: length=%4d  start_trim=%4d  end_trim=%4d  remaining=%4d%s",
            r["episode_index"], r["length"], r["cutoff"], r["end_cutoff"], r["remaining"], flag,
        )
    logger.info(
        "TOTAL over %d episodes: %d frames -> %d frames (trimmed %d start + %d end = %d, %.1f%%)",
        len(results), total_len, total_len - total_cutoff, total_start_cutoff, total_end_cutoff, total_cutoff,
        100.0 * total_cutoff / total_len if total_len else 0.0,
    )

    if args.dry_run:
        logger.info("--dry-run: nothing written.")
        return

    if not args.output_root:
        raise ValueError("--output-root is required unless --dry-run is given.")
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"--output-root {output_root} already exists - refusing to overwrite.")

    new_features = {
        k: v for k, v in ds.meta.features.items() if k not in AUTO_MANAGED_KEYS
    }
    new_ds = LeRobotDataset.create(
        repo_id=output_root.name,
        fps=ds.meta.fps,
        root=str(output_root),
        robot_type=f"{ds.meta.robot_type}_delta_joint",
        features=new_features,
        use_videos=True,
        image_writer_threads=4,
    )

    passthrough_keys = _passthrough_keys(ds)
    logger.info("passthrough columns (copied through unchanged): %s", passthrough_keys)

    written = []
    for ep in episodes:
        info = convert_episode(ds, new_ds, ep, gripper_mask, cfg, passthrough_keys)
        written.append(info)
        logger.info("wrote episode %d -> %d frames (trimmed %d start + %d end)",
                    ep, info["remaining"], info["cutoff"], info["end_cutoff"])

    # Provenance: saved station config (if present in the source dataset)
    # plus a note describing exactly how this dataset was derived.
    src_config = input_root / "station_config.yaml"
    if src_config.exists():
        shutil.copy2(src_config, output_root / "station_config.yaml")

    provenance = {
        "source_dataset": str(input_root.resolve()),
        "trim_config": vars(cfg),
        "episodes_processed": episodes,
        "per_episode_trim": [{"episode_index": r["episode_index"], "start_cutoff": r["cutoff"],
                               "end_cutoff": r["end_cutoff"], "original_length": r["length"],
                               "new_length": r["remaining"]}
                              for r in written],
        "gripper_dims_kept_absolute": [n for n, g in zip(action_names, gripper_mask) if g],
        "delta_dims": [n for n, g in zip(action_names, gripper_mask) if not g],
    }
    (output_root / PROVENANCE_FILENAME).write_text(
        "# Delta-joint action transform provenance\n\n"
        "Generated by `trossen_real/scripts/convert_to_delta_joint_dataset.py`. "
        "See `trossen_real/datasets/PickAndInsertCube.md` for the full design doc.\n\n"
        "```json\n" + json.dumps(provenance, indent=2, default=str) + "\n```\n"
    )
    logger.info("Done. New dataset at %s (%d episodes, %d frames total).",
                output_root, new_ds.meta.total_episodes, new_ds.meta.total_frames)


if __name__ == "__main__":
    main()
