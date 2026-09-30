"""P0 tests for eval mode: config, session/hook, writer, report. No hardware.

Run (repo .venv only - never uv):
    PYTHONPATH=. .venv/bin/python trossen_real/infer_app/tests/test_eval_core.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from trossen_real.infer_app.eval_config import (
    EvalConfig,
    EvalConfigError,
    EvalPose,
    EvalStage,
    load_eval_config,
)
from trossen_real.infer_app.eval_session import (
    STAGE_MULTIPLE,
    STATUS_DISCARDED,
    STATUS_PENDING,
    STATUS_SAVED,
    STATUS_UNREVIEWED,
    EvalHook,
    EvalSession,
    EvalSessionError,
    _SegmentTracker,
)
from trossen_real.infer_app.eval_writer import EvalWriter, read_index, slugify
from trossen_real.scripts.eval_report import render_checkpoint_summary, render_pose_sheet

REPO = Path(__file__).resolve().parents[3]
REAL_CONFIG = REPO / "trossen_real/configs/eval/pick_cube_and_insert_station3.yaml"


def tiny_config(n_stages=4, n_poses=3) -> EvalConfig:
    return EvalConfig(
        eval_name="t", task_label="T",
        stages=[EvalStage(id=f"s{i}", name=f"S{i}") for i in range(n_stages)],
        poses=[EvalPose(index=i, x_cm=float(i), y_cm=0.0, yaw_deg=0.0) for i in range(1, n_poses + 1)],
        max_steps_default=500,
    )


def run_session(tmp: Path, n_poses=3):
    cfg = tiny_config(n_poses=n_poses)
    return cfg, EvalSession(cfg, output_root=str(tmp), checkpoint_label="ckpt_55k"), EvalWriter(tmp)


# ---------------------------------------------------------------------------
# intervention segment edge cases  (the math everything else depends on)
# ---------------------------------------------------------------------------
def _segments(pattern: list[bool]) -> list[list[int]]:
    t = _SegmentTracker()
    for step, iv in enumerate(pattern, 1):
        t.on_tick(step, iv)
    t.finalize()
    return [s.to_list() for s in t.segments]


def test_segments_never_intervened():
    assert _segments([False] * 5) == []


def test_segments_starts_intervened():
    assert _segments([True, True, False, False]) == [[1, 2]]


def test_segments_ends_still_intervened():
    """Run stopped mid-intervention - the open segment must be closed, not lost."""
    assert _segments([False, True, True]) == [[2, 3]]


def test_segments_single_tick_blip():
    assert _segments([False, True, False]) == [[2, 2]]


def test_segments_alternating():
    assert _segments([True, False, True, False, True]) == [[1, 1], [3, 3], [5, 5]]


def test_segments_whole_run():
    assert _segments([True] * 4) == [[1, 4]]


def test_segment_totals_and_first_step():
    t = _SegmentTracker()
    for step, iv in enumerate([False, False, True, True, False, True], 1):
        t.on_tick(step, iv)
    t.finalize()
    assert t.total_steps == 3
    assert t.first_step == 3
    assert [s.to_list() for s in t.segments] == [[3, 4], [6, 6]]


# ---------------------------------------------------------------------------
# hook safety: must never raise into the control loop
# ---------------------------------------------------------------------------
def test_hook_never_raises_into_the_loop():
    cfg, session, _ = run_session(Path("/tmp"))
    run = session.begin_run(pose_index=1)
    hook = EvalHook(run)

    class Boom(list):
        def append(self, *_a, **_k):
            raise RuntimeError("disk on fire")

    run.achieved_hz_samples = Boom()
    for step in range(1, 6):  # must not raise despite the sabotaged field
        hook.on_tick(step=step, intervened=(step == 2), achieved_hz=20.0)
    hook.on_end(stop_reason="manual", steps=5)
    # and the run must still leave "running" so the UI can't deadlock
    assert run.status == STATUS_PENDING
    print("  hook survived a failing callback and still ended the run")


# ---------------------------------------------------------------------------
# session lifecycle
# ---------------------------------------------------------------------------
def test_pending_review_blocks_next_start():
    cfg, session, _ = run_session(Path("/tmp"))
    session.begin_run(pose_index=1)
    session.hook.on_end("manual", 10)
    session.end_run()
    try:
        session.begin_run(pose_index=2)
    except EvalSessionError as e:
        assert "Save or Discard" in str(e)
        print(f"  blocked as expected: {str(e)[:60]}...")
        return
    raise AssertionError("starting a run with a pending review must be refused")


def test_save_advances_cursor_discard_does_not():
    cfg, session, _ = run_session(Path("/tmp"), n_poses=3)
    assert session.pose_cursor == 1
    session.begin_run()
    session.hook.on_end("reward_release", 100)
    session.end_run()
    session.annotate(furthest_stage=3)
    session.finalize("save")
    assert session.pose_cursor == 2, session.pose_cursor

    session.begin_run()
    session.hook.on_end("manual", 50)
    session.end_run()
    session.finalize("discard", reason="operator error")
    assert session.pose_cursor == 2, "discard must not advance the cursor"
    assert session.remaining_pose_indices() == [2, 3]


def test_cannot_annotate_or_finalize_while_running():
    cfg, session, _ = run_session(Path("/tmp"))
    session.begin_run(pose_index=1)
    for fn in (lambda: session.annotate(furthest_stage=1), lambda: session.finalize("save")):
        try:
            fn()
        except EvalSessionError:
            continue
        raise AssertionError("must refuse while a run is in progress")


def test_stage_bounds_validated():
    cfg, session, _ = run_session(Path("/tmp"))
    session.begin_run(pose_index=1)
    session.hook.on_end("manual", 5)
    session.end_run()
    for bad in (4, -1, 99):
        try:
            session.annotate(furthest_stage=bad)
        except EvalConfigError:
            continue
        raise AssertionError(f"furthest_stage={bad} must be rejected for a 4-stage config")
    session.annotate(furthest_stage=None)  # 'reached none' is legitimate
    session.annotate(furthest_stage=0)
    try:
        session.annotate(intervened_during_stage="halfway")
    except EvalSessionError:
        pass
    else:
        raise AssertionError("bad intervened_during_stage must be rejected")
    session.annotate(intervened_during_stage=STAGE_MULTIPLE)


def test_unknown_pose_index_rejected():
    cfg, session, _ = run_session(Path("/tmp"), n_poses=3)
    try:
        session.begin_run(pose_index=99)
    except EvalConfigError as e:
        assert "99" in str(e)
        return
    raise AssertionError("unknown pose index must be rejected")


def test_checkpoint_label_required():
    cfg = tiny_config()
    session = EvalSession(cfg, output_root="/tmp", checkpoint_label="")
    try:
        session.begin_run(pose_index=1)
    except EvalSessionError as e:
        assert "checkpoint_label" in str(e)
        return
    raise AssertionError("eval mode must require a checkpoint label")


def test_shutdown_keeps_pending_run_as_unreviewed():
    cfg, session, _ = run_session(Path("/tmp"))
    session.begin_run(pose_index=1)
    session.hook.on_end("manual", 42)
    session.end_run()
    run = session.abandon_pending()
    assert run.status == STATUS_UNREVIEWED and run.step_count == 42
    assert session.pending is None


def test_assisted_flag_follows_intervention():
    cfg, session, _ = run_session(Path("/tmp"))
    run = session.begin_run(pose_index=1, intervention_enabled=True)
    for step in range(1, 11):
        session.hook.on_tick(step=step, intervened=4 <= step <= 6)
    session.hook.on_end("manual", 10)
    assert run.assisted and run.total_intervened_steps == 3
    assert [s.to_list() for s in run.intervention_segments] == [[4, 6]]


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def _finish(session, writer, cfg, pose, stage, *, action="save", steps=200,
            stop_reason="reward_release", intervened=()):
    session.set_pose_cursor(pose)
    session.begin_run(pose_index=pose)
    for step in range(1, steps + 1):
        session.hook.on_tick(step=step, intervened=step in intervened, achieved_hz=20.0)
    session.hook.on_end(stop_reason, steps)
    session.end_run()
    if action == "save":
        session.annotate(furthest_stage=stage)
    run = session.finalize(action, reason="test discard" if action == "discard" else "")
    return writer.write_run(run, cfg, station="station3", policy={"n_action_steps": 12})


def test_writer_save_and_index():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, session, writer = run_session(tmp)
        s = _finish(session, writer, cfg, pose=1, stage=3)
        run_dir = Path(s["artifacts"]["eval_dir"])
        assert (run_dir / "summary.json").exists() and (run_dir / "timeline.jsonl").exists()
        assert s["status"] == STATUS_SAVED and s["annotation"]["success"] is True
        rows = read_index(writer.index_path)
        assert len(rows) == 1 and rows[0]["furthest_stage"] == 3 and rows[0]["success"] is True
        print(f"  wrote {run_dir.relative_to(tmp)}")


def test_discard_writes_everything_and_deletes_nothing():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, session, writer = run_session(tmp)
        s = _finish(session, writer, cfg, pose=1, stage=None, action="discard", stop_reason="manual")
        run_dir = Path(s["artifacts"]["eval_dir"])
        assert s["status"] == STATUS_DISCARDED
        assert s["discard"]["reason"] == "test discard"
        assert (run_dir / "summary.json").exists(), "a discarded run must still be written"
        assert (run_dir / "timeline.jsonl").exists()
        rows = read_index(writer.index_path)
        assert len(rows) == 1 and rows[0]["status"] == STATUS_DISCARDED
        print("  discarded run fully persisted; nothing deleted")


def test_index_survives_a_truncated_line():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, session, writer = run_session(tmp)
        _finish(session, writer, cfg, pose=1, stage=3)
        with open(writer.index_path, "a") as f:
            f.write('{"run_id": "half-written')  # killed mid-flush
        rows = read_index(writer.index_path)
        assert len(rows) == 1, "one bad line must not hide the good runs"


def test_timeline_events():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, session, writer = run_session(tmp)
        s = _finish(session, writer, cfg, pose=1, stage=2, steps=20, intervened=range(5, 9))
        events = [json.loads(x) for x in
                  (Path(s["artifacts"]["eval_dir"]) / "timeline.jsonl").read_text().splitlines()]
        kinds = [e["event"] for e in events]
        assert kinds == ["run_start", "intervention_start", "intervention_end", "run_end"]
        assert events[1]["step"] == 5 and events[2]["step"] == 8


def test_slugify_handles_checkpoint_paths():
    s = slugify("/home/a/outputs/train/ACT_BC_x_deltajoint/checkpoints/055000/pretrained_model/")
    assert "/" not in s and s and not s.startswith("_")
    assert slugify("") == "unknown"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _rows(*specs):
    """specs: (checkpoint, pose, furthest_stage, steps, timed_out, assisted, status)"""
    out = []
    for i, (ck, pose, fs, steps, to, asst, status) in enumerate(specs):
        out.append({
            "run_id": f"r{i}", "status": status, "started_at": 1000 + i,
            "checkpoint_label": ck, "pose_index": pose, "furthest_stage": fs,
            "success": fs == 3, "steps": steps, "timed_out": to, "assisted": asst,
            "total_intervened_steps": 5 if asst else 0, "failure_note": "",
            "intervened_during_stage": 1 if asst else None,
        })
    return out


def test_pose_sheet_ticks_only_the_furthest_stage():
    """Matches the hand-filled sheet: pose 01 shows S2 only, not S0+S1+S2."""
    rows = _rows(("c", 1, 2, 500, True, False, "saved"), ("c", 2, 3, 314, False, False, "saved"))
    sheet = render_pose_sheet(rows, "c", n_stages=4, n_poses=3, include_assisted=False)
    lines = {ln.split("|")[1].strip(): ln for ln in sheet.splitlines() if ln.startswith("| 0")}
    assert lines["01"] == "| 01 | - [ ] | - [ ] | - [X] | - [ ] | 500 | Fail |  |", lines["01"]
    assert lines["02"] == "| 02 | - [ ] | - [ ] | - [ ] | - [X] | 314 | Pass |  |", lines["02"]
    assert "| 03 | - [ ] | - [ ] | - [ ] | - [ ] |  |  |  |" in sheet, "unrun poses stay blank"
    print("  pose sheet reproduces the Notes.md row format exactly")


def test_summary_counts_are_cumulative_and_medians_right():
    rows = _rows(
        ("c", 1, 3, 100, False, False, "saved"),
        ("c", 2, 3, 200, False, False, "saved"),
        ("c", 3, 1, 500, True, False, "saved"),
        ("c", 4, None, 500, True, False, "saved"),
    )
    out = render_checkpoint_summary(rows, n_stages=4, n_poses=4, include_assisted=False)
    row = [ln for ln in out.splitlines() if ln.startswith("| c |")][0]
    cells = [c.strip() for c in row.split("|")[1:-1]]
    # ckpt, done, S0..S3 cumulative, success, median, timeout, notes
    assert cells[2:6] == ["3", "3", "2", "2"], cells
    assert cells[6] == "2/4" and cells[8] == "2/4", cells
    assert cells[7] == "350", cells  # median of 100,200,500,500
    print(f"  cumulative stage counts {cells[2:6]}, success {cells[6]}, median {cells[7]}")


def test_assisted_runs_excluded_by_default():
    rows = _rows(("c", 1, 3, 100, False, False, "saved"), ("c", 2, 3, 120, False, True, "saved"))
    excl = render_checkpoint_summary(rows, 4, 2, include_assisted=False)
    incl = render_checkpoint_summary(rows, 4, 2, include_assisted=True)
    assert "1/2" in excl and "assisted (excluded)" in excl
    assert "2/2" in incl
    sheet = render_pose_sheet(rows, "c", 4, 2, include_assisted=True)
    assert "ASSISTED (5 steps, stage 1)" in sheet


def test_discarded_and_unreviewed_never_counted():
    rows = _rows(("c", 1, 3, 100, False, False, "saved"),
                 ("c", 2, 3, 100, False, False, "discarded"),
                 ("c", 3, 3, 100, False, False, "unreviewed"))
    out = render_checkpoint_summary(rows, 4, 3, include_assisted=False)
    assert "1/3" in out and "2 pose(s) pending" in out


def test_rerun_of_a_pose_supersedes_the_earlier_attempt():
    rows = _rows(("c", 1, 1, 500, True, False, "saved"), ("c", 1, 3, 250, False, False, "saved"))
    sheet = render_pose_sheet(rows, "c", 4, 1, include_assisted=False)
    assert "250 | Pass" in sheet and "500" not in sheet


# ---------------------------------------------------------------------------
# resume across restarts (index.jsonl -> pose cursor)
# ---------------------------------------------------------------------------
def _hist(ckpt, poses, status="saved", cfg="t", hsh=None):
    return [{"status": status, "checkpoint_label": ckpt, "pose_index": p,
             "eval_config": cfg, "eval_config_hash": hsh} for p in poses]


def test_resume_skips_poses_already_saved_for_this_checkpoint():
    cfg = tiny_config(n_poses=5)
    s = EvalSession(cfg, output_root="/tmp", checkpoint_label="ckpt_55k")
    s.load_history(_hist("ckpt_55k", [1, 2, 3]))
    assert s.saved_pose_indices() == [1, 2, 3]
    assert s.pose_cursor == 4, "cursor must resume at the first unfinished pose"
    assert s.remaining_pose_indices() == [4, 5]
    print("  resumed at pose 4 of 5 from a previous session")


def test_resume_ignores_other_checkpoints():
    cfg = tiny_config(n_poses=5)
    s = EvalSession(cfg, output_root="/tmp", checkpoint_label="ckpt_70k")
    s.load_history(_hist("ckpt_55k", [1, 2, 3]))
    assert s.saved_pose_indices() == [] and s.pose_cursor == 1


def test_resume_ignores_discarded_and_unreviewed():
    cfg = tiny_config(n_poses=5)
    s = EvalSession(cfg, output_root="/tmp", checkpoint_label="ck")
    s.load_history(_hist("ck", [1], "saved") + _hist("ck", [2], "discarded")
                   + _hist("ck", [3], "unreviewed"))
    assert s.saved_pose_indices() == [1]
    assert s.pose_cursor == 2, "a discarded pose still needs doing"


def test_resume_ignores_a_different_eval_config():
    cfg = tiny_config(n_poses=5)
    s = EvalSession(cfg, output_root="/tmp", checkpoint_label="ck")
    s.load_history(_hist("ck", [1, 2], cfg="some_other_task"))
    assert s.saved_pose_indices() == []


def test_resume_flags_a_changed_rubric_hash():
    cfg = tiny_config(n_poses=5)
    s = EvalSession(cfg, output_root="/tmp", checkpoint_label="ck")
    s.load_history(_hist("ck", [1], hsh="deadbeef"))
    assert s.history_saved_poses == {1} and s.history_hash_mismatch is True
    print("  rubric-hash change surfaced rather than silently mixed")


def test_switching_checkpoint_recomputes_progress():
    cfg = tiny_config(n_poses=5)
    s = EvalSession(cfg, output_root="/tmp", checkpoint_label="ckpt_55k")
    s.load_history(_hist("ckpt_55k", [1, 2]) + _hist("ckpt_70k", [1, 2, 3, 4]))
    assert s.pose_cursor == 3
    s.begin_run(checkpoint_label="ckpt_70k")          # operator switches checkpoint
    assert s.saved_pose_indices() == [1, 2, 3, 4]
    assert s.current.pose_index == 5, "must start at the new checkpoint's first unfinished pose"
    print("  switching checkpoint re-derived progress (pose 5, not 3)")


def test_in_session_and_history_progress_combine():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, session, writer = run_session(tmp, n_poses=5)
        session.load_history(_hist("ckpt_55k", [1, 2]))
        assert session.pose_cursor == 3
        _finish(session, writer, cfg, pose=3, stage=3)
        assert session.saved_pose_indices() == [1, 2, 3]
        assert session.pose_cursor == 4


def test_resume_round_trips_through_a_real_index_file():
    """End to end: write runs, then start a fresh session from that index."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, session, writer = run_session(tmp, n_poses=5)
        _finish(session, writer, cfg, pose=1, stage=3)
        _finish(session, writer, cfg, pose=2, stage=1)
        _finish(session, writer, cfg, pose=3, stage=None, action="discard")
        fresh = EvalSession(cfg, output_root=str(tmp), checkpoint_label="ckpt_55k")
        fresh.load_history(writer.read_index())
        assert fresh.saved_pose_indices() == [1, 2], "the discarded pose 3 must still be pending"
        assert fresh.pose_cursor == 3
        print("  fresh session resumed from index.jsonl at pose 3")


def test_rerunning_a_pose_keeps_both_records():
    """Nothing is overwritten - both attempts stay on disk and in the index."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, session, writer = run_session(tmp, n_poses=3)
        a = _finish(session, writer, cfg, pose=1, stage=1, steps=500)
        session.set_pose_cursor(1)
        b = _finish(session, writer, cfg, pose=1, stage=3, steps=250)
        assert a["artifacts"]["eval_dir"] != b["artifacts"]["eval_dir"], "run dirs must be distinct"
        assert Path(a["artifacts"]["eval_dir"], "summary.json").exists()
        assert Path(b["artifacts"]["eval_dir"], "summary.json").exists()
        rows = read_index(writer.index_path)
        assert len(rows) == 2 and {r["furthest_stage"] for r in rows} == {1, 3}
        # the report keeps only the latest for the table, but both are on disk
        sheet = render_pose_sheet(rows, "ckpt_55k", 4, 3, include_assisted=False)
        assert "250 | Pass" in sheet and "| 500 |" not in sheet
        print("  both attempts kept on disk; report shows the latest")


# ---------------------------------------------------------------------------
# the real config
# ---------------------------------------------------------------------------
def test_real_station3_config_loads():
    cfg = load_eval_config(str(REAL_CONFIG))
    assert cfg.n_stages == 4 and cfg.n_poses == 20
    assert cfg.max_steps_default == 500 and cfg.poses_seed == 42
    assert cfg.stages[0].name == "Approach the Red Cube"
    assert cfg.stages[3].name == "Insert the Red Cube"
    assert cfg.pose_indices() == list(range(1, 21))
    print(f"  {cfg.eval_name}: {cfg.n_stages} stages, {cfg.n_poses} poses, hash={cfg.config_hash}")


def test_station3_poses_are_reproducible_from_the_script():
    """The poses must be a pure function of the recorded generator parameters -
    nothing hand-picked, nothing hand-edited. Regenerate in-process and compare."""
    import importlib.util
    import json as _json

    spec = importlib.util.spec_from_file_location(
        "gen_eval_poses", REPO / "trossen_real/scripts/generate_eval_poses.py")
    gen = importlib.util.module_from_spec(spec)
    sys.modules["gen_eval_poses"] = gen
    spec.loader.exec_module(gen)

    cfg = load_eval_config(str(REAL_CONFIG))
    raw = _json.loads((REPO / "trossen_real/configs/eval"
                       / "poses_station3_cube_20_stratified.json").read_text())
    g = raw["generator"]
    print(f"  regenerate_with: {g['regenerate_with']}")

    poses = gen.GENERATORS[g["method"]](n_poses=g["n_poses"], seed=g["seed"])
    if g["ordered_for_printing"]:
        poses = gen.order_poses_for_printing(poses, per_page=g["mat_per_page"])

    assert len(poses) == cfg.n_poses
    for i, (x, y, yaw) in enumerate(poses, 1):
        p = cfg.pose(i)
        assert abs(p.x_cm - x) < 1e-3 and abs(p.y_cm - y) < 1e-3 and p.yaw_deg == yaw, \
            f"pose {i} does not match a fresh regeneration - was the file hand-edited?"
    print(f"  all {len(poses)} poses reproduced exactly from (method, seed, n, per_page)")


def test_station3_pose_pages_are_in_order_and_do_not_overlap():
    """Mat page k must hold poses 5k+1..5k+5 (so you can print and work in order)
    with no two cubes on a page closer than the cube width."""
    import itertools
    import math as _m

    cfg = load_eval_config(str(REAL_CONFIG))
    cube_cm = 3.0
    for page in range(4):
        grp = [cfg.pose(i) for i in range(page * 5 + 1, page * 5 + 6)]
        assert [p.index for p in grp] == list(range(page * 5 + 1, page * 5 + 6))
        gaps = [_m.dist((a.x_cm, a.y_cm), (b.x_cm, b.y_cm))
                for a, b in itertools.combinations(grp, 2)]
        assert min(gaps) >= cube_cm, f"page {page + 1} has overlapping outlines ({min(gaps):.2f} cm)"
    print("  pages 1-4 = poses 1-5, 6-10, 11-15, 16-20; no overlaps")


def test_bad_configs_rejected():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "b.yaml"
        for text, why in (
            ("eval_name: x\nstages: []\n", "empty stages"),
            ("eval_name: x\nstages:\n  - id: a\n  - id: a\nposes: [{index: 1, x_cm: 0, y_cm: 0, yaw_deg: 0}]\n",
             "duplicate stage ids"),
            ("eval_name: x\nstages:\n  - id: a\n", "no poses"),
            ("eval_name: x\nstages:\n  - id: a\nposes: [{index: 1, x_cm: 0, y_cm: 0, yaw_deg: 0}]\n"
             "max_steps_default: 0\n", "max_steps 0"),
        ):
            bad.write_text(text)
            try:
                load_eval_config(str(bad))
            except EvalConfigError:
                continue
            raise AssertionError(f"should have rejected: {why}")


def test_config_hash_changes_with_rubric_not_bookkeeping():
    a = tiny_config()
    b = tiny_config()
    assert a.config_hash == b.config_hash
    b.output_root = "somewhere_else"
    assert a.config_hash == b.config_hash, "output_root is not part of the rubric"
    b.stages = b.stages[:3]
    assert a.config_hash != b.config_hash, "changing stages must change the hash"


# ---------------------------------------------------------------------------
# end-to-end: 3 poses -> tables
# ---------------------------------------------------------------------------
def test_end_to_end_session_to_markdown():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cfg, session, writer = run_session(tmp, n_poses=3)
        _finish(session, writer, cfg, pose=1, stage=3, steps=314, stop_reason="reward_release")
        _finish(session, writer, cfg, pose=2, stage=2, steps=500, stop_reason="max_steps")
        _finish(session, writer, cfg, pose=3, stage=1, steps=220, stop_reason="manual",
                intervened=range(10, 30))
        rows = read_index(writer.index_path)
        assert len(rows) == 3
        summary = render_checkpoint_summary(rows, 4, 3, include_assisted=False)
        sheet = render_pose_sheet(rows, "ckpt_55k", 4, 3, include_assisted=False)
        assert "| ckpt_55k |" in summary and "1 assisted (excluded)" in summary
        assert "| 01 | - [ ] | - [ ] | - [ ] | - [X] | 314 | Pass |" in sheet
        assert "| 02 | - [ ] | - [ ] | - [X] | - [ ] | 500 | Fail |" in sheet
        print("\n" + summary + "\n\n" + sheet)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            print(f"- {fn.__name__}")
            fn()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
