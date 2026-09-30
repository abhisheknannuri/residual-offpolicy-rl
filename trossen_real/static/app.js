const els = {
  stationSelect: document.getElementById("station-select"),
  enablePedalEpisodeControl: document.getElementById("enable-pedal-episode-control"),
  connectBtn: document.getElementById("connect-btn"),
  disconnectBtn: document.getElementById("disconnect-btn"),
  connectionStatus: document.getElementById("connection-status"),
  mockBadge: document.getElementById("mock-badge"),
  taskName: document.getElementById("task-name"),
  startSessionBtn: document.getElementById("start-session-btn"),
  stopSessionBtn: document.getElementById("stop-session-btn"),
  sessionDir: document.getElementById("session-dir"),
  startRecordingBtn: document.getElementById("start-recording-btn"),
  stopRecordingBtn: document.getElementById("stop-recording-btn"),
  episodeCounter: document.getElementById("episode-counter"),
  resetBtn: document.getElementById("reset-btn"),
  gripperButtons: document.getElementById("gripper-buttons"),
  armStatus: document.getElementById("arm-status"),
  cameraStatus: document.getElementById("camera-status"),
  tickStatus: document.getElementById("tick-status"),
  pedalStatus: document.getElementById("pedal-status"),
};

let cameraNames = [];
let pollTimer = null;
let connectAbortController = null;

async function api(path, options = {}) {
  const res = await fetch(path, {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: options.body ? JSON.stringify(options.body) : undefined,
    signal: options.signal,
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
  for (let i = 0; i < 4; i++) {
    const img = document.getElementById(`cam-${i}`);
    const label = img.previousElementSibling;
    if (i < cameraNames.length) {
      img.src = `/video_feed/${cameraNames[i]}`;
      label.textContent = cameraNames[i];
    } else {
      img.removeAttribute("src");
      label.textContent = "(unused)";
    }
  }
}

function buildGripperButton() {
  els.gripperButtons.innerHTML = "";
  const btn = document.createElement("button");
  btn.textContent = "Gripper: ...";
  btn.disabled = true;
  btn.addEventListener("click", toggleGripper);
  els.gripperButtons.appendChild(btn);
}

async function toggleGripper() {
  try {
    await api("/api/gripper", { method: "POST", body: {} });
  } catch (err) {
    alert(err.message);
  }
}

async function connect() {
  const config_name = els.stationSelect.value;
  if (!config_name) return;

  // A wrong/unreachable IP can make /api/connect take several seconds
  // (bounded server-side to ~5-10s, see arm_driver.py's CONNECT_TIMEOUT_S)
  // before failing - let the operator cancel waiting on it instead of
  // being stuck with an unresponsive-looking button. Note this only
  // aborts the BROWSER's wait; the connect attempt already in flight
  // server-side can't be interrupted mid-call and will still run to
  // completion (succeeding or failing on its own) in the background - the
  // short server-side timeout keeps that bounded to a few seconds.
  connectAbortController = new AbortController();
  setConnectionPill("connecting");
  els.stationSelect.disabled = true;
  els.connectBtn.textContent = "Cancel connecting...";

  try {
    const data = await api("/api/connect", {
      method: "POST",
      body: { config_name, enable_pedal_episode_control: els.enablePedalEpisodeControl.checked },
      signal: connectAbortController.signal,
    });
    cameraNames = data.cameras;
    buildVideoGrid();
    buildGripperButton();
    els.enablePedalEpisodeControl.disabled = true;
    els.connectBtn.disabled = true;
    els.disconnectBtn.disabled = false;
    els.startSessionBtn.disabled = false;
    els.resetBtn.disabled = false;
    for (const btn of els.gripperButtons.children) btn.disabled = false;
    startPolling();
  } catch (err) {
    setConnectionPill("disconnected");
    if (err.name !== "AbortError") {
      alert(err.message);
    }
  } finally {
    connectAbortController = null;
    els.stationSelect.disabled = false;
    els.connectBtn.textContent = "Connect";
  }
}

function onConnectBtnClick() {
  if (connectAbortController) {
    connectAbortController.abort();
    return;
  }
  connect();
}

async function disconnect() {
  try {
    await api("/api/disconnect", { method: "POST" });
  } catch (err) {
    alert(err.message);
    return;
  }
  stopPolling();
  setConnectionPill("disconnected");
  els.enablePedalEpisodeControl.disabled = false;
  els.connectBtn.disabled = false;
  els.disconnectBtn.disabled = true;
  els.startSessionBtn.disabled = true;
  els.stopSessionBtn.disabled = true;
  els.startRecordingBtn.disabled = true;
  els.stopRecordingBtn.disabled = true;
  els.resetBtn.disabled = true;
  els.gripperButtons.innerHTML = "";
  els.mockBadge.classList.add("hidden");
  els.pedalStatus.textContent = "";
  buildVideoGrid();
}

async function startSession() {
  const task_name = els.taskName.value.trim();
  if (!task_name) {
    alert("Enter a task name first.");
    return;
  }
  try {
    const data = await api("/api/start_session", { method: "POST", body: { task_name } });
    els.sessionDir.textContent = `session: ${data.session_dir}`;
    els.startSessionBtn.disabled = true;
    els.stopSessionBtn.disabled = false;
    els.startRecordingBtn.disabled = false;
  } catch (err) {
    alert(err.message);
  }
}

async function stopSession() {
  try {
    await api("/api/stop_session", { method: "POST" });
    els.sessionDir.textContent = "";
    els.startSessionBtn.disabled = false;
    els.stopSessionBtn.disabled = true;
    els.startRecordingBtn.disabled = true;
    els.stopRecordingBtn.disabled = true;
  } catch (err) {
    alert(err.message);
  }
}

async function startRecording() {
  try {
    await api("/api/start_recording", { method: "POST" });
    els.startRecordingBtn.disabled = true;
    els.stopRecordingBtn.disabled = false;
  } catch (err) {
    alert(err.message);
  }
}

async function stopRecording() {
  try {
    const data = await api("/api/stop_recording", { method: "POST" });
    els.episodeCounter.textContent = `episodes: ${data.num_episodes}`;
    els.startRecordingBtn.disabled = false;
    els.stopRecordingBtn.disabled = true;
  } catch (err) {
    alert(err.message);
  }
}

async function doReset() {
  els.resetBtn.disabled = true;
  try {
    await api("/api/reset", { method: "POST" });
  } catch (err) {
    alert(err.message);
  } finally {
    els.resetBtn.disabled = false;
  }
}

function renderStatusDot(ok) {
  return `<span class="dot ${ok ? "ok" : "bad"}"></span>`;
}

async function pollHealth() {
  try {
    const data = await api("/api/health");
    if (!data.connected) {
      setConnectionPill("disconnected");
      return;
    }
    setConnectionPill("connected");

    els.armStatus.innerHTML =
      `<div class="status-row">${renderStatusDot(data.leader)} leader</div>` +
      `<div class="status-row">${renderStatusDot(data.follower)} follower</div>`;

    els.cameraStatus.innerHTML = Object.entries(data.cameras)
      .map(([name, ok]) => {
        const d = (data.camera_details || {})[name];
        let title = "";
        if (d && !d.is_mock) {
          const age = d.last_frame_age_s !== null ? `${d.last_frame_age_s.toFixed(1)}s ago` : "never";
          title = ` title="last frame: ${age}, consecutive failures: ${d.consecutive_failures}, reconnects: ${d.reconnect_count}"`;
        }
        return `<div class="status-row"${title}>${renderStatusDot(ok)} ${name}</div>`;
      })
      .join("");

    els.tickStatus.textContent = `tick #${data.tick_count}, last update ${
      data.last_tick_age_s !== null ? data.last_tick_age_s.toFixed(2) : "?"
    }s ago, freq ${
      data.actual_frequency_hz !== null ? data.actual_frequency_hz.toFixed(1) : "?"
    }/${data.target_frequency_hz}Hz`;
    if (data.control_loop_error) {
      els.tickStatus.textContent += ` - CONTROL LOOP STOPPED: ${data.control_loop_error}`;
      els.tickStatus.style.color = "#f87171";
    } else {
      els.tickStatus.style.color = "";
    }

    els.episodeCounter.textContent = `episodes: ${data.num_episodes}`;
    els.startSessionBtn.disabled = data.session_active;
    els.stopSessionBtn.disabled = !data.session_active || data.recording;
    els.startRecordingBtn.disabled = !data.session_active || data.recording;
    els.stopRecordingBtn.disabled = !data.recording;

    if (data.pedal_episode_control_enabled) {
      els.pedalStatus.textContent = data.last_auto_stop_reason
        ? `Foot pedal episode control: enabled - last episode auto-ended via: ${data.last_auto_stop_reason}`
        : "Foot pedal episode control: enabled (reset pedal or reward-pedal release ends the current recording)";
    } else {
      els.pedalStatus.textContent = "Foot pedal episode control: disabled for this session";
    }

    const stateData = await api("/api/get_state").catch(() => null);
    if (stateData && els.gripperButtons.firstElementChild) {
      const btn = els.gripperButtons.firstElementChild;
      const isOpen = stateData.gripper_open;
      // Button shows the ACTION a click will perform (opposite of current
      // state): currently open -> next click closes it (red); currently
      // closed -> next click opens it (green).
      // Disabled while teleop is actively driving the follower - the
      // leader's continuously-streamed absolute gripper position would
      // just override a manual click on the very next tick.
      btn.textContent = data.teleop_active
        ? "Gripper (controlled by leader during teleop)"
        : isOpen ? "Close Gripper" : "Open Gripper";
      btn.disabled = data.teleop_active;
      btn.classList.toggle("action-close", !data.teleop_active && isOpen);
      btn.classList.toggle("action-open", !data.teleop_active && !isOpen);
    }
  } catch (err) {
    setConnectionPill("disconnected");
  }
}

function startPolling() {
  stopPolling();
  pollHealth();
  pollTimer = setInterval(pollHealth, 1000);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

els.connectBtn.addEventListener("click", onConnectBtnClick);
els.disconnectBtn.addEventListener("click", disconnect);
els.startSessionBtn.addEventListener("click", startSession);
els.stopSessionBtn.addEventListener("click", stopSession);
els.startRecordingBtn.addEventListener("click", startRecording);
els.stopRecordingBtn.addEventListener("click", stopRecording);
els.resetBtn.addEventListener("click", doReset);

loadStations();
