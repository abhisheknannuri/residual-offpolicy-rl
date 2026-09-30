# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Eval-mode runtime state: one `EvalSession` per connect->disconnect cycle,
one `EvalRun` per Start->Stop cycle (== one attempt at one cube pose).

See `EVAL_MODE_PLAN.md` for the design. Two things about this module matter more
than the rest:

1. **This state lives on `AppState`, never on `InferSnapshot`.** `InferSnapshot`
   is reconstructed from scratch on every tick (`infer_loop.py`), so a field that
   isn't re-passed each iteration silently resets to its default forever. Keeping
   eval state here makes that failure mode structurally impossible.

2. **`EvalHook` must never raise into the control loop.** It is called from
   inside the inference tick; an eval bookkeeping bug must not be able to stop a
   moving arm. Every callback is wrapped - failures are logged once and the hook
   disables itself, exactly like `TickLogger` does.

Step numbering matches `TickLogger`'s: `on_tick` is called with the post-increment
step, so step 1 is the first tick. Intervention segment indices therefore line up
with the per-run tick-log JSONL without any off-by-one adjustment.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from trossen_real.infer_app.eval_config import EvalConfig

logger = logging.getLogger(__name__)

STATUS_RUNNING = "running"
STATUS_PENDING = "pending_review"
STATUS_SAVED = "saved"
STATUS_DISCARDED = "discarded"
STATUS_UNREVIEWED = "unreviewed"  # app shut down with a review still pending

# Value used for `intervened_during_stage` when the operator took over across
# more than one sub-stage.
STAGE_MULTIPLE = "multiple"


@dataclass
class InterventionSegment:
    """A contiguous run of ticks where the operator was driving. Inclusive."""

    start_step: int
    end_step: int

    @property
    def n_steps(self) -> int:
        return self.end_step - self.start_step + 1

    def to_list(self) -> list[int]:
        return [self.start_step, self.end_step]


class _SegmentTracker:
    """Edge-detects intervention segments from a per-tick boolean.

    Handles the awkward cases explicitly: a run that begins already intervened,
    one that ends still intervened (closed by `finalize`), and single-tick blips.
    """

    def __init__(self) -> None:
        self.segments: list[InterventionSegment] = []
        self._open_start: int | None = None
        self._last_step: int = 0
        self.total_steps: int = 0

    def on_tick(self, step: int, intervened: bool) -> None:
        self._last_step = step
        if intervened:
            self.total_steps += 1
            if self._open_start is None:
                self._open_start = step
        elif self._open_start is not None:
            self.segments.append(InterventionSegment(self._open_start, step - 1))
            self._open_start = None

    def finalize(self) -> None:
        if self._open_start is not None:
            self.segments.append(InterventionSegment(self._open_start, self._last_step))
            self._open_start = None

    @property
    def first_step(self) -> int | None:
        return self.segments[0].start_step if self.segments else None


@dataclass
class EvalAnnotation:
    """What the operator supplies after the run stops.

    `furthest_stage` is the single piece of judgement the protocol needs: the
    highest sub-stage index completed, or `None` for 'reached none'. The four
    `- [X]` columns in Notes.md are rendered FROM this - see `eval_report.py`.
    """

    furthest_stage: int | None = None
    failure_note: str = ""
    intervened_during_stage: int | str | None = None
    operator: str = ""
    annotated: bool = False  # False until the operator actually touched it

    def to_dict(self) -> dict:
        return {
            "furthest_stage": self.furthest_stage,
            "failure_note": self.failure_note,
            "intervened_during_stage": self.intervened_during_stage,
            "operator": self.operator,
            "annotated": self.annotated,
        }


@dataclass
class EvalRun:
    run_id: str
    checkpoint_label: str
    pose_index: int
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    status: str = STATUS_RUNNING
    step_count: int = 0
    max_steps: int | None = None
    stop_reason: str | None = None
    tags: list[str] = field(default_factory=list)
    run_notes: str = ""
    intervention_enabled: bool = False
    intervention_segments: list[InterventionSegment] = field(default_factory=list)
    total_intervened_steps: int = 0
    achieved_hz_samples: list[float] = field(default_factory=list)
    dataset_dir: str | None = None
    dataset_episode_index: int | None = None
    tick_log: str | None = None
    videos: list[str] = field(default_factory=list)
    annotation: EvalAnnotation = field(default_factory=EvalAnnotation)
    discard_reason: str = ""

    # ---- derived -----------------------------------------------------------
    @property
    def assisted(self) -> bool:
        """True if the operator drove at any point - such a run does not measure
        the policy, so it is excluded from the headline success rate."""
        return self.total_intervened_steps > 0

    @property
    def timed_out(self) -> bool:
        return self.stop_reason == "max_steps"

    @property
    def duration_s(self) -> float | None:
        return None if self.ended_at is None else self.ended_at - self.started_at

    def is_success(self, config: EvalConfig) -> bool:
        return self.annotation.furthest_stage == config.last_stage_index

    def summary_dict(self, config: EvalConfig, *, station: str = "", policy: dict | None = None,
                     pose: dict | None = None) -> dict:
        hz = self.achieved_hz_samples
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "station": station,
            "eval_config": {"name": config.eval_name, "hash": config.config_hash,
                            "task_label": config.task_label},
            "checkpoint_label": self.checkpoint_label,
            "pose": pose if pose is not None else {"index": self.pose_index},
            "policy": policy or {},
            "run": {
                "steps": self.step_count,
                "max_steps": self.max_steps,
                "stop_reason": self.stop_reason,
                "timed_out": self.timed_out,
                "achieved_hz_mean": (sum(hz) / len(hz)) if hz else None,
            },
            "annotation": {**self.annotation.to_dict(),
                           "success": self.is_success(config),
                           "tags": list(self.tags),
                           "run_notes": self.run_notes},
            "intervention": {
                "enabled": self.intervention_enabled,
                "assisted": self.assisted,
                "total_intervened_steps": self.total_intervened_steps,
                "segment_count": len(self.intervention_segments),
                "first_intervention_step": (
                    self.intervention_segments[0].start_step if self.intervention_segments else None
                ),
                "segments": [s.to_list() for s in self.intervention_segments],
            },
            "artifacts": {
                "tick_log": self.tick_log,
                "videos": list(self.videos),
                "dataset_dir": self.dataset_dir,
                "dataset_episode_index": self.dataset_episode_index,
            },
            "discard": None if self.status != STATUS_DISCARDED else {
                "reason": self.discard_reason,
                "at": self.ended_at,
            },
        }

    def timeline_events(self) -> list[dict]:
        events = [{"t": self.started_at, "step": 0, "event": "run_start",
                   "pose_index": self.pose_index, "checkpoint_label": self.checkpoint_label}]
        for seg in self.intervention_segments:
            events.append({"step": seg.start_step, "event": "intervention_start"})
            events.append({"step": seg.end_step, "event": "intervention_end", "n_steps": seg.n_steps})
        events.append({"t": self.ended_at, "step": self.step_count, "event": "run_end",
                       "stop_reason": self.stop_reason})
        return events


class EvalHook:
    """Bridges the inference loop to an `EvalRun`. Never raises; see module docstring."""

    def __init__(self, run: EvalRun) -> None:
        self.run = run
        self._tracker = _SegmentTracker()
        self._disabled = False

    def _guard(self, what: str, fn, *args, **kwargs) -> None:
        if self._disabled:
            return
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.exception("eval hook %s failed - disabling eval bookkeeping for this run "
                             "(inference itself is unaffected)", what)
            self._disabled = True

    # ---- called from the inference tick ------------------------------------
    def on_tick(self, step: int, intervened: bool, achieved_hz: float | None = None) -> None:
        def _do():
            self._tracker.on_tick(step, intervened)
            self.run.step_count = step
            self.run.total_intervened_steps = self._tracker.total_steps
            if achieved_hz is not None:
                self.run.achieved_hz_samples.append(achieved_hz)

        self._guard("on_tick", _do)

    def on_end(self, stop_reason: str | None, steps: int) -> None:
        def _do():
            self._tracker.finalize()
            self.run.intervention_segments = list(self._tracker.segments)
            self.run.total_intervened_steps = self._tracker.total_steps
            self.run.step_count = steps
            self.run.stop_reason = stop_reason
            self.run.ended_at = time.time()
            self.run.status = STATUS_PENDING

        self._guard("on_end", _do)
        # Even if bookkeeping broke, the run must not be left marked "running" -
        # otherwise the UI would block the next Start forever.
        if self.run.status == STATUS_RUNNING:
            self.run.status = STATUS_PENDING
            self.run.ended_at = self.run.ended_at or time.time()

    # ---- live view for the UI ----------------------------------------------
    def live_stats(self) -> dict:
        return {
            "step": self.run.step_count,
            "intervened_steps": self._tracker.total_steps,
            "segment_count": len(self._tracker.segments) + (0 if self._tracker._open_start is None else 1),
            "intervening_now": self._tracker._open_start is not None,
        }


class EvalSessionError(RuntimeError):
    """Operator/flow errors (wrong order, bad pose index) - surfaced as 4xx."""


class EvalSession:
    """One connect->disconnect cycle's worth of eval state."""

    def __init__(self, config: EvalConfig, output_root: str | None = None,
                 checkpoint_label: str = "", operator: str = "") -> None:
        self.config = config
        self.output_root = output_root or config.output_root
        self.checkpoint_label = checkpoint_label
        self.operator = operator
        self.current: EvalRun | None = None
        self.pending: EvalRun | None = None
        self.completed: list[EvalRun] = []
        self.pose_cursor: int = config.pose_indices()[0] if config.poses else 1
        self.hook: EvalHook | None = None
        # Poses already saved for this checkpoint in EARLIER sessions, read back
        # from index.jsonl at connect. A 20-pose x 15-checkpoint campaign spans
        # many app restarts; without this the cursor would reset to pose 1 every
        # time and silently invite duplicate work.
        self._history_rows: list[dict] = []
        self.history_saved_poses: set[int] = set()
        self.history_hash_mismatch: bool = False

    # ---- resume across restarts --------------------------------------------
    def load_history(self, rows: list[dict]) -> None:
        """Seed already-finished poses from a previous session's index.jsonl.

        Only `status == "saved"` rows for the SAME eval config and the SAME
        checkpoint label count - a discarded or unreviewed run means that pose
        still needs doing, which is exactly why discard does not advance the
        cursor in the first place.
        """
        self._history_rows = list(rows or [])
        self._recompute_history()

    def _recompute_history(self) -> None:
        label = (self.checkpoint_label or "").strip()
        done: set[int] = set()
        mismatch = False
        if label:
            for r in self._history_rows:
                if r.get("status") != STATUS_SAVED:
                    continue
                if (r.get("checkpoint_label") or "").strip() != label:
                    continue
                if r.get("eval_config") and r["eval_config"] != self.config.eval_name:
                    continue
                idx = r.get("pose_index")
                if not isinstance(idx, int):
                    continue
                # Counted as done, but flagged: a different rubric hash means the
                # earlier run was scored against different stages/poses.
                if r.get("eval_config_hash") and r["eval_config_hash"] != self.config.config_hash:
                    mismatch = True
                done.add(idx)
        self.history_saved_poses = done
        self.history_hash_mismatch = mismatch
        remaining = self.remaining_pose_indices()
        if remaining:
            self.pose_cursor = remaining[0]

    # ---- pose bookkeeping --------------------------------------------------
    def saved_pose_indices(self) -> list[int]:
        in_session = {r.pose_index for r in self.completed if r.status == STATUS_SAVED}
        return sorted(in_session | self.history_saved_poses)

    def remaining_pose_indices(self) -> list[int]:
        done = set(self.saved_pose_indices())
        return [i for i in self.config.pose_indices() if i not in done]

    def set_pose_cursor(self, index: int) -> None:
        self.config.pose(index)  # raises if unknown
        self.pose_cursor = index

    def _advance_cursor(self) -> None:
        remaining = [i for i in self.remaining_pose_indices() if i > self.pose_cursor]
        if remaining:
            self.pose_cursor = remaining[0]
        else:
            rest = self.remaining_pose_indices()
            if rest:
                self.pose_cursor = rest[0]

    # ---- run lifecycle -----------------------------------------------------
    def begin_run(self, *, pose_index: int | None = None, checkpoint_label: str | None = None,
                  max_steps: int | None = None, tags: list[str] | None = None,
                  run_notes: str = "", intervention_enabled: bool = False) -> EvalRun:
        if self.pending is not None:
            raise EvalSessionError(
                f"run {self.pending.run_id} (pose {self.pending.pose_index}) is still awaiting "
                "Save or Discard - finalize it before starting the next run"
            )
        if self.current is not None:
            raise EvalSessionError("a run is already in progress")
        label = checkpoint_label or self.checkpoint_label
        if not label.strip():
            raise EvalSessionError("checkpoint_label is required in eval mode")
        if label != self.checkpoint_label:
            # Switching checkpoints mid-session: re-derive which poses are done
            # for the NEW label (and move the cursor), or the operator would
            # inherit the previous checkpoint's progress.
            self.checkpoint_label = label
            self._recompute_history()
        idx = self.pose_cursor if pose_index is None else int(pose_index)
        self.config.pose(idx)  # raises on unknown pose index
        self.pose_cursor = idx

        run = EvalRun(
            run_id=f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}",
            checkpoint_label=label,
            pose_index=idx,
            max_steps=max_steps if max_steps is not None else self.config.max_steps_default,
            tags=list(tags or []),
            run_notes=run_notes,
            intervention_enabled=intervention_enabled,
        )
        run.annotation.operator = self.operator
        self.current = run
        self.hook = EvalHook(run)
        return run

    def end_run(self) -> EvalRun | None:
        """Move the active run to pending review. Idempotent."""
        if self.current is None:
            return self.pending
        run = self.current
        if run.status == STATUS_RUNNING:  # loop never called on_end (e.g. start failed)
            run.status = STATUS_PENDING
            run.ended_at = run.ended_at or time.time()
        self.pending = run
        self.current = None
        return run

    def annotate(self, *, furthest_stage: int | None = ..., failure_note: str | None = None,
                 intervened_during_stage: int | str | None = ..., operator: str | None = None) -> EvalRun:
        run = self._require_pending()
        if furthest_stage is not ...:
            self.config.validate_stage(furthest_stage)
            run.annotation.furthest_stage = furthest_stage
            run.annotation.annotated = True
        if failure_note is not None:
            run.annotation.failure_note = failure_note
            run.annotation.annotated = True
        if intervened_during_stage is not ...:
            if isinstance(intervened_during_stage, int) and not isinstance(intervened_during_stage, bool):
                self.config.validate_stage(intervened_during_stage)
            elif intervened_during_stage not in (None, STAGE_MULTIPLE):
                raise EvalSessionError(
                    f"intervened_during_stage must be a stage index, {STAGE_MULTIPLE!r}, or null"
                )
            run.annotation.intervened_during_stage = intervened_during_stage
            run.annotation.annotated = True
        if operator is not None:
            run.annotation.operator = operator
            self.operator = operator
        return run

    def finalize(self, action: str, *, reason: str = "") -> EvalRun:
        run = self._require_pending()
        if action == "save":
            run.status = STATUS_SAVED
        elif action == "discard":
            run.status = STATUS_DISCARDED
            run.discard_reason = reason
        else:
            raise EvalSessionError(f"action must be 'save' or 'discard', got {action!r}")
        self.pending = None
        self.hook = None
        self.completed.append(run)
        if run.status == STATUS_SAVED:
            self._advance_cursor()
        return run

    def abort_current(self) -> None:
        """Drop the active run without recording it - for when the inference loop
        refused to start, so the run never actually happened. Not the same as
        `abandon_pending()`, which preserves a run that DID happen."""
        self.current = None
        self.hook = None

    def abandon_pending(self, reason: str = "app shut down with a review pending") -> EvalRun | None:
        """Shutdown path: keep the data, flag it, never silently drop it."""
        run = self.pending or self.current
        if run is None:
            return None
        run.status = STATUS_UNREVIEWED
        run.discard_reason = reason
        run.ended_at = run.ended_at or time.time()
        self.pending = None
        self.current = None
        self.hook = None
        self.completed.append(run)
        return run

    def _require_pending(self) -> EvalRun:
        if self.current is not None:
            raise EvalSessionError("cannot annotate or finalize while a run is in progress - stop it first")
        if self.pending is None:
            raise EvalSessionError("no run is awaiting review")
        return self.pending

    # ---- UI state ----------------------------------------------------------
    def to_dict(self) -> dict:
        pose = self.config.pose(self.pose_cursor) if self.config.poses else None
        pending = self.pending
        return {
            "eval_mode": True,
            "eval_config": self.config.eval_name,
            "eval_config_hash": self.config.config_hash,
            "checkpoint_label": self.checkpoint_label,
            "operator": self.operator,
            "pose_cursor": self.pose_cursor,
            "pose_hint": pose.describe() if pose else None,
            "pose": {"index": pose.index, "x_cm": pose.x_cm, "y_cm": pose.y_cm,
                     "yaw_deg": pose.yaw_deg} if pose else None,
            "n_poses": self.config.n_poses,
            "saved_pose_indices": self.saved_pose_indices(),
            "remaining_pose_indices": self.remaining_pose_indices(),
            "history_saved_poses": sorted(self.history_saved_poses),
            "history_hash_mismatch": self.history_hash_mismatch,
            "runs_saved": sum(1 for r in self.completed if r.status == STATUS_SAVED),
            "runs_discarded": sum(1 for r in self.completed if r.status == STATUS_DISCARDED),
            "run_active": self.current is not None,
            "eval_run_id": (self.current or pending).run_id if (self.current or pending) else None,
            "live": self.hook.live_stats() if self.hook is not None else None,
            "pending_review": None if pending is None else {
                "run_id": pending.run_id,
                "pose_index": pending.pose_index,
                "steps": pending.step_count,
                "stop_reason": pending.stop_reason,
                "assisted": pending.assisted,
                "total_intervened_steps": pending.total_intervened_steps,
                "segment_count": len(pending.intervention_segments),
                "annotation": pending.annotation.to_dict(),
                "suggested_furthest_stage": (
                    self.config.last_stage_index if pending.stop_reason == "reward_release" else None
                ),
            },
        }
