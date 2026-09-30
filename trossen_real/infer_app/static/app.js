const els = {
  stationSelect: document.getElementById("station-select"),
  policyUrl: document.getElementById("policy-url"),
  actionSpace: document.getElementById("action-space"),
  forceActionSpace: document.getElementById("force-action-space"),
  enableIntervention: document.getElementById("enable-intervention"),
  enableDatasetRecording: document.getElementById("enable-dataset-recording"),
  connectBtn: document.getElementById("connect-btn"),
  disconnectBtn: document.getElementById("disconnect-btn"),
  connectionStatus: document.getElementById("connection-status"),
  settleTime: document.getElementById("settle-time"),
  maxSteps: document.getElementById("max-steps"),
  skipReset: document.getElementById("skip-reset"),
  recordVideo: document.getElementById("record-video"),
  taskName: document.getElementById("task-name"),
  startInferBtn: document.getElementById("start-infer-btn"),
  stopInferBtn: document.getElementById("stop-infer-btn"),
  resetRobotBtn: document.getElementById("reset-robot-btn"),
  videoGrid: document.getElementById("video-grid"),
  followerStatus: document.getElementById("follower-status"),
  policyStatus: document.getElementById("policy-status"),
  cameraStatus: document.getElementById("camera-status"),
  inferStatus: document.getElementById("infer-status"),
  interventionStatus: document.getElementById("intervention-status"),
  datasetStatus: document.getElementById("dataset-status"),
  errorStatus: document.getElementById("error-status"),
};

let cameraNames = [];
let pollTimer = null;

async function api(path, options = {}) {
  const res = await fetch(path, {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Request to ${path} failed (${res.status})`);
  }
  return data;
}

function setConnectionPill(state) {
  els.connectionStatus.className = `status-pill ${state}`;
  els.connectionStatus.textContent = state;
}

async function loadStations() {
  const data = await api("/api/stations");
  els.stationSelect.innerHTML = "";
  for (const name of data.stations) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    els.stationSelect.appendChild(opt);
  }
}

function buildVideoGrid() {
  els.videoGrid.innerHTML = "";
  for (const name of cameraNames) {
    const cell = document.createElement("div");
    cell.className = "video-cell";
    cell.id = `video-cell-${name}`;
    const label = document.createElement("div");
    label.className = "video-label";
    label.textContent = name;
    const badge = document.createElement("div");
    badge.className = "cam-badge hidden";
    badge.id = `cam-badge-${name}`;
    const img = document.createElement("img");
    img.src = `/video_feed/${name}`;
    cell.appendChild(label);
    cell.appendChild(badge);
    cell.appendChild(img);
    els.videoGrid.appendChild(cell);
  }
}

function setConnectedUi(connected) {
  els.connectBtn.disabled = connected;
  els.disconnectBtn.disabled = !connected;
  els.startInferBtn.disabled = !connected;
  els.resetRobotBtn.disabled = !connected;
  els.stationSelect.disabled = connected;
  els.policyUrl.disabled = connected;
  els.actionSpace.disabled = connected;
  els.forceActionSpace.disabled = connected;
  els.enableIntervention.disabled = connected;
  els.enableDatasetRecording.disabled = connected;
  if (!connected) {
    els.stopInferBtn.disabled = true;
    els.resetRobotBtn.disabled = true;
    els.videoGrid.innerHTML = "";
    cameraNames = [];
    els.taskName.disabled = false;
  }
}

async function connect() {
  setConnectionPill("connecting");
  try {
    const data = await api("/api/connect", {
      method: "POST",
      body: {
        config_name: els.stationSelect.value,
        policy_server_url: els.policyUrl.value.trim(),
        action_space: els.actionSpace.value,
        force_action_space: els.forceActionSpace.checked,
        enable_intervention: els.enableIntervention.checked,
        enable_dataset_recording: els.enableDatasetRecording.checked,
        // eval.js contributes eval_config_name / checkpoint_label / operator when
        // eval mode is on; absent entirely otherwise.
        ...(window.evalConnectExtras ? window.evalConnectExtras() : {}),
      },
    });
    cameraNames = data.cameras || [];
    buildVideoGrid();
    setConnectionPill("connected");
    setConnectedUi(true);
    startPolling();
  } catch (err) {
    setConnectionPill("disconnected");
    alert(`Connect failed: ${err.message}`);
  }
}

async function disconnect() {
  stopPolling();
  try {
    await api("/api/disconnect", { method: "POST" });
  } catch (err) {
    console.error(err);
  }
  setConnectionPill("disconnected");
  setConnectedUi(false);
  els.followerStatus.textContent = "";
  els.policyStatus.textContent = "";
  els.cameraStatus.textContent = "";
  els.inferStatus.textContent = "";
  els.interventionStatus.textContent = "";
  els.datasetStatus.textContent = "";
  els.errorStatus.textContent = "";
}

async function startInference() {
  const body = {
    settle_time_s: parseFloat(els.settleTime.value) || 2.0,
    skip_reset: els.skipReset.checked,
    record_video: els.recordVideo ? els.recordVideo.checked : true,
    task_name: els.taskName.value.trim(),
  };
  const maxStepsVal = els.maxSteps.value.trim();
  if (maxStepsVal !== "") {
    body.max_steps = parseInt(maxStepsVal, 10);
  }
  if (window.evalStartExtras) {
    const extra = window.evalStartExtras();
    if (extra) body.eval = extra;
  }
  try {
    await api("/api/start_inference", { method: "POST", body });
  } catch (err) {
    alert(`Start inference failed: ${err.message}`);
  }
}

async function resetRobot() {
  // Blocking on the server (~reset goal_time). Disable the button meanwhile so a
  // second click can't queue another reset on top of the one in flight.
  els.resetRobotBtn.disabled = true;
  const label = els.resetRobotBtn.textContent;
  els.resetRobotBtn.textContent = "Resetting...";
  try {
    const r = await api("/api/reset", { method: "POST", body: {} });
    console.log("reset ok", r);
  } catch (err) {
    alert(`Reset failed: ${err.message}`);
  } finally {
    els.resetRobotBtn.textContent = label;
    els.resetRobotBtn.disabled = false;
  }
}

async function stopInference() {
  try {
    await api("/api/stop_inference", { method: "POST" });
  } catch (err) {
    console.error(err);
  }
}

function renderCameraBadges(cameraDetails) {
  for (const name of cameraNames) {
    const badge = document.getElementById(`cam-badge-${name}`);
    const cell = document.getElementById(`video-cell-${name}`);
    if (!badge || !cell) continue;
    const details = cameraDetails ? cameraDetails[name] : null;
    if (!details) {
      badge.className = "cam-badge hidden";
      cell.className = "video-cell";
      continue;
    }
    if (details.is_mock) {
      badge.textContent = "MOCK";
      badge.className = "cam-badge mock";
      cell.className = "video-cell unhealthy";
    } else if (!details.healthy) {
      badge.textContent = "STALE";
      badge.className = "cam-badge unhealthy";
      cell.className = "video-cell unhealthy";
    } else {
      badge.className = "cam-badge hidden";
      cell.className = "video-cell";
    }
  }
}

async function poll() {
  try {
    const h = await api("/api/health");
    if (!h.connected) {
      return;
    }
    els.followerStatus.innerHTML =
      `<span class="dot ${h.follower ? "ok" : "bad"}"></span> Follower: ${h.follower ? "ok" : "unreachable"}`;

    const ph = h.policy_health || {};
    els.policyStatus.innerHTML =
      `<span class="dot ok"></span> Policy: ${ph.checkpoint || "?"} ` +
      `(device=${ph.device}, chunk_size=${ph.chunk_size}, n_action_steps=${ph.n_action_steps}, ` +
      `action_space=${h.action_space}, likely_action_space=${ph.likely_action_space ?? "unknown"})`;

    const camLines = Object.entries(h.camera_details || {}).map(([name, d]) => {
      const cls = d.is_mock ? "bad" : d.healthy ? "ok" : "bad";
      const age = d.last_frame_age_s !== null && d.last_frame_age_s !== undefined ? d.last_frame_age_s.toFixed(2) : "?";
      return `<div class="status-row"><span class="dot ${cls}"></span>${name}: ` +
        `${d.is_mock ? "MOCK" : d.healthy ? "healthy" : "UNHEALTHY"} ` +
        `(last frame ${age}s ago, reconnects=${d.reconnect_count})</div>`;
    });
    els.cameraStatus.innerHTML = camLines.join("");
    renderCameraBadges(h.camera_details);

    if (h.camera_problems && h.camera_problems.length) {
      els.errorStatus.textContent = "Camera problems (inference would be refused): " + h.camera_problems.join("; ");
    } else if (h.error) {
      els.errorStatus.textContent = "Inference error: " + h.error;
    } else {
      els.errorStatus.textContent = "";
    }

    let inferLine;
    if (h.inferring && h.settling) {
      inferLine = "settling after reset...";
    } else if (h.inferring) {
      const stepStr = h.max_steps ? `${h.step}/${h.max_steps}` : `${h.step}`;
      const hz = h.achieved_hz ? h.achieved_hz.toFixed(1) : "?";
      inferLine = `RUNNING - step ${stepStr}, ${hz} Hz (target ${h.target_hz} Hz)` +
        (h.last_action ? ` - last action: [${h.last_action.map((v) => v.toFixed(3)).join(", ")}]` : "");
    } else {
      inferLine = "stopped" + (h.step ? ` (ran ${h.step} steps)` : "");
    }
    if (h.log_file) {
      inferLine += ` - log: ${h.log_file}`;
    }
    if (!h.inferring && h.stop_reason) {
      inferLine += ` - stopped via: ${h.stop_reason}`;
    }
    els.inferStatus.innerHTML = `<span class="dot ${h.inferring ? "ok" : "idle"}"></span> ${inferLine}`;

    els.startInferBtn.disabled = h.inferring;
    els.stopInferBtn.disabled = !h.inferring;
    // Never allow a reset to fight a running rollout for the arm. Skipped while
    // a reset is in flight (the handler owns the button then).
    if (els.resetRobotBtn.textContent !== "Resetting...") {
      els.resetRobotBtn.disabled = h.inferring;
    }

    if (h.dataset_recording_enabled) {
      els.taskName.disabled = h.dataset_session_active;
      const datasetLine = h.dataset_session_active
        ? `dataset session active - ${h.num_episodes} episode(s) saved${h.dataset_dir ? ` (${h.dataset_dir})` : ""}`
        : "dataset recording enabled - not started yet (type a task name and click Start Inference)";
      els.datasetStatus.innerHTML = `<span class="dot ${h.dataset_recording ? "ok" : "idle"}"></span> Dataset: ${datasetLine}`;
    } else {
      els.datasetStatus.innerHTML = `<span class="dot idle"></span> Dataset recording: disabled for this session`;
    }

    if (h.intervention_enabled) {
      const dotCls = h.intervening ? "warn" : "ok";
      const label = h.intervening
        ? "INTERVENING - follower is being driven by the leader, policy output is ignored"
        : `autonomous (leader ${h.leader_connected ? "connected" : "DISCONNECTED"}, mirroring policy)`;
      els.interventionStatus.innerHTML = `<span class="dot ${dotCls}"></span> Intervention: ${label}`;
    } else {
      els.interventionStatus.innerHTML = `<span class="dot idle"></span> Intervention: disabled for this session`;
    }

    // Hand the payload to eval.js. It owns its own DOM; nothing above this line
    // touches the eval panel, so operator input is never clobbered by polling.
    window.dispatchEvent(new CustomEvent("infer-health", { detail: h }));
  } catch (err) {
    console.error(err);
  }
}

function startPolling() {
  poll();
  pollTimer = setInterval(poll, 500);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

els.connectBtn.addEventListener("click", connect);
els.disconnectBtn.addEventListener("click", disconnect);
els.startInferBtn.addEventListener("click", startInference);
els.stopInferBtn.addEventListener("click", stopInference);
els.resetRobotBtn.addEventListener("click", resetRobot);

setConnectedUi(false);
loadStations();
