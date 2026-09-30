// Evaluation mode UI. See trossen_real/infer_app/EVAL_MODE_PLAN.md.
//
// Owns #eval-panel entirely. app.js never touches it - it only dispatches an
// "infer-health" event with each poll payload. That split is the whole point:
// app.js rewrites innerHTML twice a second, which would otherwise wipe whatever
// the operator is typing into the review form.
//
// Within this file the same rule applies again:
//   #eval-live   - rewritten on every poll (derived, safe to clobber)
//   #eval-review - re-rendered ONLY when the pending run id changes
//
// Deliberate omission: no keyboard shortcut starts or stops the robot. Keys are
// annotation-only (post-run), so a stray keypress can never command motion.

const ev = {
  panel: document.getElementById("eval-panel"),
  configSelect: document.getElementById("eval-config-select"),
  checkpoint: document.getElementById("eval-checkpoint"),
  operator: document.getElementById("eval-operator"),
  configName: document.getElementById("eval-config-name"),
  live: document.getElementById("eval-live"),
  poseIndex: document.getElementById("eval-pose-index"),
  poseSet: document.getElementById("eval-pose-set"),
  poseHint: document.getElementById("eval-pose-hint"),
  review: document.getElementById("eval-review"),
  reviewHead: document.getElementById("eval-review-head"),
  stageButtons: document.getElementById("eval-stage-buttons"),
  assistRow: document.getElementById("eval-assist-row"),
  assistStage: document.getElementById("eval-assist-stage"),
  note: document.getElementById("eval-note"),
  saveBtn: document.getElementById("eval-save-btn"),
  discardBtn: document.getElementById("eval-discard-btn"),
};

let evalConfig = null;      // stages/poses, from /api/connect or /api/eval/config
let renderedRunId = null;   // which pending run the review form currently shows
let chosenStage = undefined; // undefined = untouched, null = "none reached"
let lastSession = null;

async function evalApi(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${path} failed (${res.status})`);
  return data;
}

// ---- connect / start hooks consumed by app.js ------------------------------
window.evalConnectExtras = () => {
  const name = ev.configSelect.value.trim();
  if (!name) return {};
  return {
    eval_config_name: name,
    checkpoint_label: ev.checkpoint.value.trim(),
    operator: ev.operator.value.trim(),
  };
};

window.evalStartExtras = () => {
  if (!evalConfig) return null;
  const extra = {};
  const ck = ev.checkpoint.value.trim();
  if (ck) extra.checkpoint_label = ck;
  // Send the pose number the operator can SEE. Previously this relied solely on
  // the server-side cursor, so a typed-but-not-"Set" value was silently ignored
  // AND overwritten by the next poll - the run then got filed under the wrong
  // pose with nothing on screen to show it. What you see is what gets recorded.
  const idx = parseInt(ev.poseIndex.value, 10);
  if (Number.isInteger(idx)) extra.pose_index = idx;
  return extra;
};

// ---- setup -----------------------------------------------------------------
async function loadEvalConfigs() {
  try {
    const data = await evalApi("/api/eval/configs");
    for (const name of data.configs || []) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      ev.configSelect.appendChild(opt);
    }
  } catch (err) {
    console.error("could not list eval configs", err);
  }
}

function stageLabel(i) {
  const s = evalConfig && evalConfig.stages[i];
  return s ? `S${i} ${s.name}` : `S${i}`;
}

function buildStageButtons() {
  ev.stageButtons.innerHTML = "";
  const mk = (value, text, title) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "stage-btn";
    b.textContent = text;
    b.title = title;
    b.dataset.value = value === null ? "none" : String(value);
    b.addEventListener("click", () => setStage(value));
    ev.stageButtons.appendChild(b);
  };
  mk(null, "` none", "Reached no sub-stage");
  for (let i = 0; i < (evalConfig ? evalConfig.n_stages : 0); i++) {
    mk(i, `${i}`, stageLabel(i));
  }
  ev.assistStage.innerHTML = "";
  for (const [v, t] of [["", "(not set)"], ...Array.from(
    { length: evalConfig ? evalConfig.n_stages : 0 }, (_, i) => [String(i), stageLabel(i)]),
    ["multiple", "multiple stages"]]) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    ev.assistStage.appendChild(o);
  }
}

function setStage(value) {
  chosenStage = value;
  for (const b of ev.stageButtons.querySelectorAll(".stage-btn")) {
    const bv = b.dataset.value === "none" ? null : parseInt(b.dataset.value, 10);
    b.classList.toggle("selected", bv === value || (value === null && b.dataset.value === "none"));
  }
}

// ---- polled (safe to clobber) ----------------------------------------------
function renderLive(h, e) {
  const live = e.live || {};
  const done = (e.saved_pose_indices || []).length;
  const fromEarlier = (e.history_saved_poses || []).length;
  const bits = [
    `<b>${e.checkpoint_label || "(no checkpoint label)"}</b>`,
    `pose <b>${e.pose_cursor}</b>/${e.n_poses}`,
    `done ${done}/${e.n_poses}` + (fromEarlier ? ` (${fromEarlier} from earlier sessions)` : ""),
    e.runs_discarded ? `discarded ${e.runs_discarded}` : null,
    e.history_hash_mismatch
      ? `<span class="warn-text">earlier runs used a DIFFERENT rubric hash</span>` : null,
  ].filter(Boolean);
  if (done >= e.n_poses) bits.push(`<b>all poses done</b>`);
  if (h.inferring) {
    bits.push(`<span class="dot ok"></span> RUNNING step ${h.step}/${h.max_steps || "-"}`);
    if (live.intervened_steps) {
      bits.push(`<span class="warn-text">intervened ${live.intervened_steps} steps` +
        `${live.intervening_now ? " (NOW)" : ""}</span>`);
    }
  } else if (e.pending_review) {
    bits.push(`<span class="dot warn"></span> awaiting review`);
  } else {
    bits.push(`<span class="dot idle"></span> idle`);
  }
  ev.live.innerHTML = bits.join(" &nbsp;|&nbsp; ");
  ev.poseHint.textContent = e.pose_hint || "";
  // Only refresh the box when the operator is not editing it AND has no
  // uncommitted value in it - polling must never silently rewrite an edit.
  const shown = parseInt(ev.poseIndex.value, 10);
  const editing = document.activeElement === ev.poseIndex;
  if (!editing && (!Number.isInteger(shown) || shown === e.pose_cursor)) {
    ev.poseIndex.value = e.pose_cursor;
  }
  ev.poseIndex.classList.toggle("uncommitted",
    Number.isInteger(shown) && shown !== e.pose_cursor);
}

// ---- operator-owned (re-rendered only on run change) ------------------------
function renderReview(e) {
  const pend = e.pending_review;
  if (!pend) {
    ev.review.classList.add("hidden");
    renderedRunId = null;
    return;
  }
  if (pend.run_id === renderedRunId) return;   // do NOT touch the form mid-edit

  renderedRunId = pend.run_id;
  chosenStage = undefined;
  ev.note.value = pend.annotation.failure_note || "";
  ev.review.classList.remove("hidden");

  const assisted = pend.assisted;
  ev.assistRow.classList.toggle("hidden", !assisted);
  ev.assistStage.value = "";

  ev.reviewHead.innerHTML =
    `Pose <b>${pend.pose_index}</b> &nbsp; ${pend.steps} steps &nbsp; stopped via ` +
    `<b>${pend.stop_reason || "?"}</b>` +
    (assisted ? ` &nbsp; <span class="warn-text">ASSISTED - ${pend.total_intervened_steps} steps ` +
      `over ${pend.segment_count} segment(s); excluded from the headline rate</span>` : "");

  // reward-release means the operator already pedal-marked success
  const suggested = pend.suggested_furthest_stage;
  setStage(suggested !== null && suggested !== undefined ? suggested : undefined);
  if (suggested !== null && suggested !== undefined) chosenStage = suggested;
}

async function finalize(action) {
  if (!renderedRunId) return;
  if (action === "save" && chosenStage === undefined) {
    alert("Pick the furthest sub-stage first (` for none, 0-3).");
    return;
  }
  try {
    if (action === "save") {
      const body = { furthest_stage: chosenStage, failure_note: ev.note.value.trim() };
      if (!ev.assistRow.classList.contains("hidden") && ev.assistStage.value !== "") {
        body.intervened_during_stage = ev.assistStage.value === "multiple"
          ? "multiple" : parseInt(ev.assistStage.value, 10);
      }
      await evalApi("/api/eval/annotate", body);
      await evalApi("/api/eval/finalize", { action: "save" });
    } else {
      const reason = prompt("Discard reason (optional):", "") || "";
      await evalApi("/api/eval/finalize", { action: "discard", reason });
    }
    renderedRunId = null;
    chosenStage = undefined;
    ev.note.value = "";
    ev.review.classList.add("hidden");
  } catch (err) {
    alert(`${action} failed: ${err.message}`);
  }
}

// ---- events ----------------------------------------------------------------
window.addEventListener("infer-health", (e) => {
  const h = e.detail || {};
  const s = h.eval || { eval_mode: false };
  lastSession = s;
  if (!s.eval_mode) {
    ev.panel.classList.add("hidden");
    return;
  }
  if (!evalConfig) {
    evalApi("/api/eval/config").then((d) => {
      evalConfig = d.config;
      ev.configName.textContent = `${evalConfig.eval_name} (${evalConfig.n_stages} stages, ` +
        `${evalConfig.n_poses} poses, hash ${evalConfig.hash})`;
      buildStageButtons();
    }).catch((err) => console.error(err));
  }
  ev.panel.classList.remove("hidden");
  renderLive(h, s);
  renderReview(s);
});

async function commitPoseCursor() {
  const idx = parseInt(ev.poseIndex.value, 10);
  if (!Number.isInteger(idx)) return;
  try {
    await evalApi("/api/eval/pose_cursor", { pose_index: idx });
  } catch (err) {
    alert(err.message);
  }
}

ev.poseSet.addEventListener("click", commitPoseCursor);
// 'change' fires on blur and on Enter, so tabbing away or hitting Enter commits
// too - the Set button is now a convenience, not a requirement.
ev.poseIndex.addEventListener("change", commitPoseCursor);

ev.saveBtn.addEventListener("click", () => finalize("save"));
ev.discardBtn.addEventListener("click", () => finalize("discard"));

document.addEventListener("keydown", (e) => {
  if (ev.review.classList.contains("hidden")) return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) {
    if (e.key === "Enter" && t === ev.note) { e.preventDefault(); finalize("save"); }
    return;   // never steal keys from a field the operator is typing in
  }
  if (e.key === "`") { e.preventDefault(); setStage(null); }
  else if (/^[0-9]$/.test(e.key)) {
    const i = parseInt(e.key, 10);
    if (evalConfig && i < evalConfig.n_stages) { e.preventDefault(); setStage(i); }
  } else if (e.key === "Enter") { e.preventDefault(); finalize("save"); }
  else if (e.key === "Backspace") { e.preventDefault(); finalize("discard"); }
});

loadEvalConfigs();
