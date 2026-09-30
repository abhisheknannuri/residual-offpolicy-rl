# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Writes eval artifacts to disk. See `EVAL_MODE_PLAN.md` §8.

Layout::

    <output_root>/
      index.jsonl                         # one line per finalized run
      2026-09-28/
        <checkpoint_slug>/
          pose_07/
            run_20260928_141233_7f3a/
              summary.json
              timeline.jsonl

`index.jsonl` exists so that aggregation (`scripts/eval_report.py`) is a one-pass
read of a single file instead of a directory walk over hundreds of runs.

**Soft discard is an invariant of this module:** a discarded run is written
exactly like a saved one, with `status="discarded"` and a populated `discard`
block. Nothing here ever deletes or truncates a video, tick log or dataset
episode - those belong to the infer app and are merely referenced by path.
Aggregation filters on `status`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from trossen_real.infer_app.eval_config import EvalConfig
from trossen_real.infer_app.eval_session import EvalRun

logger = logging.getLogger(__name__)

INDEX_NAME = "index.jsonl"
SUMMARY_NAME = "summary.json"
TIMELINE_NAME = "timeline.jsonl"

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(value: str, *, fallback: str = "unknown", max_len: int = 80) -> str:
    """Filesystem-safe path segment. Checkpoint labels are often full paths
    (`.../checkpoints/055000/pretrained_model/`), so separators collapse to '_'
    and the result is truncated - the untruncated label is still recorded
    verbatim inside summary.json, so nothing is lost."""
    slug = _SLUG_RE.sub("_", (value or "").strip()).strip("._-")
    slug = re.sub(r"_{2,}", "_", slug)
    if not slug:
        return fallback
    return slug[-max_len:].lstrip("._-") or fallback


class EvalWriter:
    def __init__(self, output_root: str | Path) -> None:
        self.root = Path(output_root)

    # ---- paths -------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def run_dir(self, run: EvalRun) -> Path:
        date = time.strftime("%Y-%m-%d", time.localtime(run.started_at))
        return (
            self.root / date / slugify(run.checkpoint_label, fallback="unlabelled")
            / f"pose_{run.pose_index:02d}" / f"run_{run.run_id}"
        )

    # ---- writing -----------------------------------------------------------
    def write_run(self, run: EvalRun, config: EvalConfig, *, station: str = "",
                  policy: dict | None = None) -> dict:
        """Write summary.json + timeline.jsonl and append to index.jsonl.

        Returns the summary dict actually written. Safe to call for both saved
        and discarded runs - the only difference is `status`/`discard`.
        """
        pose_obj = None
        try:
            p = config.pose(run.pose_index)
            pose_obj = {"index": p.index, "x_cm": p.x_cm, "y_cm": p.y_cm,
                        "yaw_deg": p.yaw_deg, "seed": config.poses_seed}
        except Exception:  # pose vanished from the config - record the index alone
            logger.warning("pose %s not in eval config %s", run.pose_index, config.eval_name)

        summary = run.summary_dict(config, station=station, policy=policy, pose=pose_obj)

        d = self.run_dir(run)
        d.mkdir(parents=True, exist_ok=True)
        summary["artifacts"] = dict(summary.get("artifacts") or {})
        summary["artifacts"]["eval_dir"] = str(d)

        (d / SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
        with open(d / TIMELINE_NAME, "w") as f:
            for ev in run.timeline_events():
                f.write(json.dumps(ev) + "\n")
        self._append_index(summary)
        logger.info("eval run %s (%s) written to %s", run.run_id, run.status, d)
        return summary

    def _append_index(self, summary: dict) -> None:
        """One flat line per finalized run. Append-only and flushed, so killing
        the app never corrupts previously-indexed runs."""
        a = summary.get("annotation", {})
        i = summary.get("intervention", {})
        r = summary.get("run", {})
        row = {
            "run_id": summary["run_id"],
            "status": summary["status"],
            "started_at": summary["started_at"],
            "checkpoint_label": summary["checkpoint_label"],
            "eval_config": summary.get("eval_config", {}).get("name"),
            "eval_config_hash": summary.get("eval_config", {}).get("hash"),
            "pose_index": (summary.get("pose") or {}).get("index"),
            "steps": r.get("steps"),
            "max_steps": r.get("max_steps"),
            "stop_reason": r.get("stop_reason"),
            "timed_out": r.get("timed_out"),
            "furthest_stage": a.get("furthest_stage"),
            "success": a.get("success"),
            "failure_note": a.get("failure_note", ""),
            "assisted": i.get("assisted"),
            "total_intervened_steps": i.get("total_intervened_steps"),
            "intervened_during_stage": a.get("intervened_during_stage"),
            "operator": a.get("operator", ""),
            "eval_dir": summary.get("artifacts", {}).get("eval_dir"),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()

    # ---- reading -----------------------------------------------------------
    def read_index(self) -> list[dict]:
        return read_index(self.index_path)


def read_index(path: str | Path) -> list[dict]:
    """Read index.jsonl, skipping malformed lines rather than failing the whole
    report - a truncated last line (killed mid-write) must not hide 299 good runs."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("%s:%d is not valid JSON - skipping", p, n)
    return rows
