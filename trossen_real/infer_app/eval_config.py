# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Eval-mode configuration: sub-stages, the deterministic pose list, and the
scoring rubric. See `EVAL_MODE_PLAN.md` (this directory) for the design and
`EVAL/ACT/TrossenStation3/Notes.md` for the protocol this encodes.

Deliberately separate from the station config (`trossen_real/config.py`): a
station config describes HARDWARE (arm IPs, cameras, control rates), while this
describes a TASK's evaluation rubric. The same station evaluates different tasks,
and the same rubric can be run on different stations.

A "pose" here is the **cube's placement on the table** (x, y, yaw), which the
human sets by hand before each run - NOT a robot pose. The 20 poses come from
`scripts/generate_eval_poses.py` (fixed seed), so every checkpoint is scored
against the identical set of starting configurations.

`config_hash` covers only the parts that define the RUBRIC (stages + poses +
max_steps), not bookkeeping like `output_root`. Two runs with the same hash were
scored the same way; a differing hash is a warning that results are not directly
comparable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"


class EvalConfigError(ValueError):
    """Raised for any malformed eval config - always names the offending field."""


@dataclass(frozen=True)
class EvalStage:
    id: str
    name: str


@dataclass(frozen=True)
class EvalPose:
    """One cube placement. Units are centimetres/degrees, matching
    `generate_eval_poses.py` and the printed sheet in docs/archive/Notes.md."""

    index: int  # 1-based, as printed ("Pose #07")
    x_cm: float
    y_cm: float
    yaw_deg: float

    def describe(self) -> str:
        return f"Pose #{self.index:02d} | X={self.x_cm:.2f} Y={self.y_cm:.2f} cm | Yaw {self.yaw_deg:+.1f}deg"


@dataclass
class EvalConfig:
    eval_name: str
    task_label: str
    stages: list[EvalStage]
    poses: list[EvalPose]
    max_steps_default: int
    output_root: str = "eval_runs"
    poses_seed: int | None = None
    operators: list[str] = field(default_factory=list)
    source_path: Path | None = None

    # ---- derived -----------------------------------------------------------
    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def n_poses(self) -> int:
        return len(self.poses)

    @property
    def last_stage_index(self) -> int:
        """Completing this stage == full task success."""
        return len(self.stages) - 1

    @property
    def config_hash(self) -> str:
        payload = {
            "eval_name": self.eval_name,
            "stages": [[s.id, s.name] for s in self.stages],
            "poses": [[p.index, round(p.x_cm, 4), round(p.y_cm, 4), round(p.yaw_deg, 4)] for p in self.poses],
            "max_steps_default": self.max_steps_default,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]  # noqa: S324

    def pose(self, index: int) -> EvalPose:
        for p in self.poses:
            if p.index == index:
                return p
        raise EvalConfigError(
            f"pose index {index} is not in this eval config (have {self.pose_indices()})"
        )

    def pose_indices(self) -> list[int]:
        return [p.index for p in self.poses]

    def validate_stage(self, furthest_stage: int | None) -> None:
        """`None` means 'reached no stage at all' - a legitimate outcome."""
        if furthest_stage is None:
            return
        if not isinstance(furthest_stage, int) or isinstance(furthest_stage, bool):
            raise EvalConfigError(f"furthest_stage must be an int or None, got {furthest_stage!r}")
        if not 0 <= furthest_stage < self.n_stages:
            raise EvalConfigError(
                f"furthest_stage {furthest_stage} out of range - this config has {self.n_stages} "
                f"stages (valid: 0..{self.last_stage_index}, or None for 'none reached')"
            )

    def to_dict(self) -> dict:
        return {
            "eval_name": self.eval_name,
            "task_label": self.task_label,
            "hash": self.config_hash,
            "stages": [{"index": i, "id": s.id, "name": s.name} for i, s in enumerate(self.stages)],
            "poses": [
                {"index": p.index, "x_cm": p.x_cm, "y_cm": p.y_cm, "yaw_deg": p.yaw_deg} for p in self.poses
            ],
            "poses_seed": self.poses_seed,
            "max_steps_default": self.max_steps_default,
            "operators": list(self.operators),
            "n_stages": self.n_stages,
            "n_poses": self.n_poses,
        }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def list_available_eval_configs() -> list[str]:
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def _load_poses_file(path: Path) -> tuple[list[EvalPose], int | None]:
    """Read the JSON written by `generate_eval_poses.py --out`."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalConfigError(f"could not read poses_file {path}: {exc}") from exc
    entries = raw.get("poses") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise EvalConfigError(f"poses_file {path} has no 'poses' list")
    seed = raw.get("seed") if isinstance(raw, dict) else None
    return _parse_poses(entries, str(path)), seed


def _parse_poses(entries: list, where: str) -> list[EvalPose]:
    poses: list[EvalPose] = []
    for i, e in enumerate(entries):
        try:
            poses.append(
                EvalPose(
                    index=int(e.get("index", i + 1)),
                    x_cm=float(e["x_cm"]),
                    y_cm=float(e["y_cm"]),
                    yaw_deg=float(e["yaw_deg"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvalConfigError(f"{where}: pose entry {i} is malformed ({exc}) - need x_cm, y_cm, yaw_deg") from exc
    seen = [p.index for p in poses]
    dupes = {i for i in seen if seen.count(i) > 1}
    if dupes:
        raise EvalConfigError(f"{where}: duplicate pose indices {sorted(dupes)}")
    return poses


def load_eval_config(name_or_path: str) -> EvalConfig:
    """Load by name (from `trossen_real/configs/eval/`) or explicit path."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise EvalConfigError(
            f"eval config not found: {name_or_path} (looked in {CONFIG_DIR}). "
            f"Available: {list_available_eval_configs()}"
        )
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise EvalConfigError(f"{path} is not valid YAML: {exc}") from exc

    for key in ("eval_name", "stages"):
        if not raw.get(key):
            raise EvalConfigError(f"{path}: '{key}' is required and must be non-empty")

    stages: list[EvalStage] = []
    for i, s in enumerate(raw["stages"]):
        if not isinstance(s, dict) or "id" not in s:
            raise EvalConfigError(f"{path}: stages[{i}] must be a mapping with an 'id'")
        stages.append(EvalStage(id=str(s["id"]), name=str(s.get("name", s["id"]))))
    ids = [s.id for s in stages]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise EvalConfigError(f"{path}: duplicate stage ids {sorted(dupes)}")

    poses_seed = raw.get("poses_seed")
    if raw.get("poses_file"):
        pose_path = Path(raw["poses_file"])
        if not pose_path.is_absolute():
            pose_path = (path.parent / pose_path).resolve()
        poses, file_seed = _load_poses_file(pose_path)
        poses_seed = poses_seed if poses_seed is not None else file_seed
    elif raw.get("poses"):
        poses = _parse_poses(raw["poses"], str(path))
    else:
        raise EvalConfigError(f"{path}: needs either 'poses' (inline) or 'poses_file'")

    max_steps = int(raw.get("max_steps_default", 500))
    if max_steps <= 0:
        raise EvalConfigError(f"{path}: max_steps_default must be > 0, got {max_steps}")

    return EvalConfig(
        eval_name=str(raw["eval_name"]),
        task_label=str(raw.get("task_label", raw["eval_name"])),
        stages=stages,
        poses=poses,
        max_steps_default=max_steps,
        output_root=str(raw.get("output_root", "eval_runs")),
        poses_seed=poses_seed,
        operators=[str(o) for o in (raw.get("operators") or [])],
        source_path=path,
    )
