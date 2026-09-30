"""P2 tests: static wiring between index.html, app.js and eval.js.

No browser needed. Catches the failure this kind of vanilla-JS UI actually hits:
a `getElementById` that returns null because an id was renamed or never added,
which throws at load and silently breaks the whole panel.

Run (repo .venv only - never uv):
    PYTHONPATH=. .venv/bin/python trossen_real/infer_app/tests/test_eval_frontend.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import trossen_real.infer_app.app as appmod

STATIC = Path(__file__).resolve().parents[1] / "static"
HTML = (STATIC / "index.html").read_text()
APP_JS = (STATIC / "app.js").read_text()
EVAL_JS = (STATIC / "eval.js").read_text()

HTML_IDS = set(re.findall(r'id="([^"]+)"', HTML))


def _referenced_ids(js: str) -> set[str]:
    return set(re.findall(r'getElementById\(\s*"([^"]+)"\s*\)', js))


def test_every_id_eval_js_uses_exists_in_html():
    missing = sorted(_referenced_ids(EVAL_JS) - HTML_IDS)
    assert not missing, f"eval.js reaches for ids that index.html does not define: {missing}"
    print(f"  eval.js resolves {len(_referenced_ids(EVAL_JS))} ids, all present")


def test_every_id_app_js_uses_still_exists():
    """Regression: the eval edits must not have removed anything app.js needs."""
    missing = sorted(_referenced_ids(APP_JS) - HTML_IDS)
    assert not missing, f"app.js reaches for ids index.html no longer defines: {missing}"


def test_original_controls_all_survived():
    for el in ("station-select", "policy-url", "action-space", "force-action-space",
               "enable-intervention", "enable-dataset-recording", "connect-btn", "disconnect-btn",
               "settle-time", "max-steps", "skip-reset", "record-video", "task-name",
               "start-infer-btn", "stop-infer-btn", "video-grid", "follower-status",
               "policy-status", "camera-status", "infer-status", "intervention-status",
               "dataset-status", "error-status", "connection-status"):
        assert el in HTML_IDS, f"pre-existing control '{el}' disappeared from index.html"
    print(f"  all {len(HTML_IDS)} ids present, including every original control")


def test_eval_js_is_loaded_after_app_js():
    """eval.js registers the infer-health listener; app.js dispatches it. Order
    does not strictly matter for the listener, but app.js must define the
    evalConnectExtras hook call before eval.js overrides it."""
    i_app = HTML.index("/static/app.js")
    i_eval = HTML.index("/static/eval.js")
    assert i_app < i_eval, "eval.js must be loaded after app.js"


def test_app_js_dispatches_health_and_uses_the_hooks():
    assert 'CustomEvent("infer-health"' in APP_JS, "app.js must hand the poll payload to eval.js"
    assert "window.evalConnectExtras" in APP_JS
    assert "window.evalStartExtras" in APP_JS
    assert 'window.addEventListener("infer-health"' in EVAL_JS


def test_app_js_never_touches_the_eval_panel():
    """The whole point of the split: polling must not be able to clobber the
    operator-owned review form."""
    for owned in ("eval-panel", "eval-review", "eval-note", "eval-stage-buttons",
                  "eval-checkpoint", "eval-operator", "eval-assist-stage"):
        assert owned not in APP_JS, f"app.js must not touch '{owned}' - polling would wipe operator input"
    print("  app.js references no operator-owned eval element")


def test_review_form_is_only_rerendered_on_run_change():
    """Guard the specific line that prevents mid-typing clobber."""
    assert "renderedRunId" in EVAL_JS
    assert re.search(r"if\s*\(\s*pend\.run_id\s*===\s*renderedRunId\s*\)\s*return", EVAL_JS), \
        "renderReview must bail out when the pending run has not changed"


def test_no_keyboard_shortcut_can_command_the_robot():
    """Annotation keys only. A stray keypress must never start or stop motion."""
    for dangerous in ("/api/start_inference", "/api/stop_inference"):
        assert dangerous not in EVAL_JS, f"eval.js must not be able to call {dangerous}"
    print("  eval.js cannot start/stop inference; annotation keys only")


def test_keyboard_handler_yields_to_text_fields():
    assert 't.tagName === "INPUT"' in EVAL_JS and "TEXTAREA" in EVAL_JS, \
        "the keydown handler must not steal keys while the operator types a note"


def test_reset_button_present_and_guarded():
    assert "reset-robot-btn" in HTML_IDS
    assert "resetRobotBtn" in APP_JS and '"/api/reset"' in APP_JS
    # must be disabled while a run is in progress
    assert "els.resetRobotBtn.disabled = h.inferring" in APP_JS, \
        "Reset must be disabled while inference runs"
    print("  Reset Robot button present, wired, and disabled during a run")


def test_pose_control_is_not_described_as_motion():
    """It only labels which hand-placed cube configuration is on the table.
    Wording that implies the robot moves would be dangerously misleading."""
    row = HTML[HTML.index('id="eval-pose-row"'):HTML.index('id="eval-review"')]
    assert "Go to pose" not in HTML, "'Go to pose' reads as a motion command"
    assert "placed by hand" in row or "does not move" in row, \
        "the pose control must say the robot does not move"
    assert "label only" in row.lower()
    print("  pose control labelled as a label, not a move")


def test_start_sends_the_pose_number_shown_in_the_box():
    """Regression (2026-09-29): the run used the server cursor while the box
    showed something else, so a typed-but-not-Set value was recorded under the
    wrong pose and then visibly snapped back on the next poll."""
    assert "extra.pose_index" in EVAL_JS, "Start must send the pose number on screen"
    assert "ev.poseIndex.value" in EVAL_JS.split("evalStartExtras")[1][:600]


def test_pose_box_commits_on_change_not_only_on_button():
    assert 'ev.poseIndex.addEventListener("change", commitPoseCursor)' in EVAL_JS, \
        "blur/Enter must commit the pose number too"


def test_poll_cannot_overwrite_an_uncommitted_pose_edit():
    seg = EVAL_JS[EVAL_JS.index("function renderLive"):EVAL_JS.index("function renderReview")]
    assert "uncommitted" in seg, "an uncommitted edit must be flagged"
    assert "shown === e.pose_cursor" in seg, \
        "poll may only rewrite the box when it matches the server cursor"


def test_static_files_are_served():
    appmod.app.config["TESTING"] = True
    c = appmod.app.test_client()
    for path, needle in (("/", "eval-panel"), ("/static/eval.js", "infer-health"),
                         ("/static/app.js", "infer-health"), ("/static/style.css", "stage-btn")):
        r = c.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}"
        assert needle in r.get_data(as_text=True), f"{path} missing {needle!r}"
    print("  index.html, app.js, eval.js and style.css all served")


def test_eval_panel_starts_hidden():
    panel = re.search(r'<section id="eval-panel"[^>]*class="([^"]*)"', HTML)
    assert panel and "hidden" in panel.group(1), "eval panel must be hidden until eval mode is on"
    assert ".hidden" in (STATIC / "style.css").read_text(), "style.css must define .hidden"


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
