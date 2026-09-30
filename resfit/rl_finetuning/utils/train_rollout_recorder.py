# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Randomly record annotated TRAINING rollouts for reward-model debugging.

The stage-aware PBRS reward is only as good as the reward model's stage
predictions. A common failure mode is a *false* stage advance (e.g. the arm
nudges the placement bin without ever grasping the can, yet the model reports a
late stage), which injects a spurious positive shaping reward. To catch this we
periodically capture whole TRAINING episodes (not eval): every step's rendered
frame is overlaid with the raw (pre-hysteresis) predicted stage, the gated stage
actually used, the confidence, and the per-step PBRS reward. A per-frame JSON is
written alongside the mp4, both saved under ``<output_dir>/<run>/training_rollouts/``
and uploaded to W&B under ``training/`` keyed by global step (so successive
recordings do not overwrite each other and are navigable by the step slider).

Only meaningful for single-env training (one episode timeline at a time).
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

# imageio-ffmpeg ships a bundled ffmpeg that can fail to launch in fresh (e.g. uv)
# environments; get_ffmpeg_exe() then blocks on a network download. Prefer a
# working system ffmpeg if the user hasn't explicitly pinned one. Must run before
# `import imageio`.
if not os.environ.get("IMAGEIO_FFMPEG_EXE"):
    _sys_ffmpeg = shutil.which("ffmpeg")
    if _sys_ffmpeg:
        os.environ["IMAGEIO_FFMPEG_EXE"] = _sys_ffmpeg

import imageio
import numpy as np
from PIL import Image, ImageDraw

import wandb


def _scalar(info: dict, key: str, default: float = 0.0) -> float:
    """Read a (possibly vectorized) scalar from a Gymnasium info dict."""
    v = info.get(key, None)
    if v is None:
        return default
    arr = np.asarray(v).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


class TrainingRolloutRecorder:
    """Sample-and-record annotated training episodes for reward-model debugging."""

    def __init__(
        self,
        *,
        output_dir,
        run_name: str | None,
        fps: int = 20,
        sample_prob: float = 0.2,
        seed: int = 0,
        num_stages: int = 4,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and float(sample_prob) > 0.0
        self.sample_prob = float(sample_prob)
        self.fps = int(fps)
        self.num_stages = int(num_stages)
        self._rng = random.Random(int(seed))

        base = Path(str(output_dir or "outputs")) / ((run_name or "run").split("__")[0])
        self.dir = base / "training_rollouts"
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

        self._ep_counter = 0
        self._frames: list[np.ndarray] = []
        self._rows: list[dict] = []
        self._recording = self._decide()

    def _decide(self) -> bool:
        return self.enabled and (self._rng.random() < self.sample_prob)

    # ------------------------------------------------------------------
    def _annotate(
        self,
        frame: np.ndarray,
        *,
        step: int,
        raw_stage: int,
        gated_stage: int,
        conf: float,
        r_shaped: float,
        r_sparse: float,
    ) -> np.ndarray:
        img = Image.fromarray(np.ascontiguousarray(frame))
        draw = ImageDraw.Draw(img)
        # Flag a suspicious gap where the raw model wants a higher stage than the
        # gated one currently allows (hysteresis is actively suppressing a jump).
        raw_gt_gated = raw_stage >= 0 and raw_stage > gated_stage
        y = 6
        for text, color in (
            (f"step {step}", (255, 255, 255)),
            (f"raw {raw_stage} / gated {gated_stage}", (255, 80, 80) if raw_gt_gated else (0, 255, 255)),
            (f"conf {conf:.2f}", (255, 255, 0)),
            (f"r_shaped {r_shaped:+.3f}", (0, 255, 0)),
            (f"r_sparse {r_sparse:+.1f}", (255, 128, 0)),
        ):
            draw.text((6, y), text, fill=color)
            y += 13
        return np.asarray(img)

    # ------------------------------------------------------------------
    def record_step(self, *, env, info: dict, global_step: int, done: bool) -> None:
        """Capture the current step's frame + reward-model stats (if recording)."""
        if not self._recording:
            return
        raw_stage = int(_scalar(info, "reward_model_raw_stage", -1))
        gated_stage = int(_scalar(info, "reward_model_stage", 0))
        conf = _scalar(info, "reward_model_stage_conf", 0.0)
        r_shaped = _scalar(info, "reward_model_r_shaped", 0.0)
        r_sparse = _scalar(info, "reward_model_r_sparse", 0.0)
        step_in_ep = len(self._rows) + 1

        # On the done step the env has already auto-reset, so render() would show
        # the *next* episode's first frame — skip the frame but keep the stats row.
        if not done:
            try:
                frame = np.asarray(env.render())
            except Exception:
                frame = None
            if frame is not None:
                if frame.ndim == 4:
                    frame = frame[0]
                self._frames.append(
                    self._annotate(
                        frame,
                        step=step_in_ep,
                        raw_stage=raw_stage,
                        gated_stage=gated_stage,
                        conf=conf,
                        r_shaped=r_shaped,
                        r_sparse=r_sparse,
                    )
                )

        self._rows.append(
            {
                "step_in_episode": step_in_ep,
                "global_step": int(global_step),
                "raw_stage": raw_stage,
                "gated_stage": gated_stage,
                "stage_conf": round(conf, 4),
                "r_shaped": round(r_shaped, 5),
                "r_sparse": round(r_sparse, 3),
            }
        )

    # ------------------------------------------------------------------
    def end_episode(self, *, global_step: int, success: bool | None = None) -> None:
        """Flush the current episode (if it was being recorded) and re-sample."""
        if self._recording and self._rows:
            self._flush(global_step=global_step, success=success)
        self._frames = []
        self._rows = []
        self._ep_counter += 1
        self._recording = self._decide()

    def _flush(self, *, global_step: int, success: bool | None) -> None:
        tag = f"train_ep{self._ep_counter:04d}_step{int(global_step)}"
        json_path = self.dir / f"{tag}.json"
        vid_path: Path | None = self.dir / f"{tag}.mp4"

        if self._frames:
            try:
                writer = imageio.get_writer(vid_path, fps=self.fps, macro_block_size=1)
                for fr in self._frames:
                    writer.append_data(fr)
                writer.close()
            except Exception:
                vid_path = None
        else:
            vid_path = None

        if success is None:
            success = any(r["r_sparse"] > 0.5 for r in self._rows)
        max_raw = max((r["raw_stage"] for r in self._rows), default=-1)
        max_gated = max((r["gated_stage"] for r in self._rows), default=0)
        final_gated = int(self._rows[-1]["gated_stage"]) if self._rows else 0
        sum_r_shaped = float(sum(r["r_shaped"] for r in self._rows))
        # False-progress heuristic: the gated stage advanced past stage 0 but the
        # episode never actually succeeded — worth eyeballing the saved video.
        suspicious_false_progress = bool(max_gated >= 1 and not success)

        summary = {
            "tag": tag,
            "global_step": int(global_step),
            "episode_len": len(self._rows),
            "success": bool(success),
            "max_raw_stage": int(max_raw),
            "max_gated_stage": int(max_gated),
            "final_gated_stage": final_gated,
            "sum_r_shaped": sum_r_shaped,
            "suspicious_false_progress": suspicious_false_progress,
            "frames": self._rows,
        }
        try:
            json_path.write_text(json.dumps(summary, indent=2))
        except Exception:
            pass

        if wandb.run is not None:
            log = {
                "training/rollout_max_gated_stage": max_gated,
                "training/rollout_max_raw_stage": max_raw,
                "training/rollout_final_gated_stage": final_gated,
                "training/rollout_success": int(bool(success)),
                "training/rollout_sum_r_shaped": sum_r_shaped,
                "training/rollout_suspicious_false_progress": int(suspicious_false_progress),
            }
            if vid_path is not None:
                log["training/rollout_video"] = wandb.Video(str(vid_path), format="mp4", caption=tag)
            try:
                wandb.log(log, step=int(global_step))
            except Exception:
                pass
