const state = {
  pollTimer: null,
  renderHandle: null,
  sessionStartMs: null,
  sessionTimer: null,
  currentFaceLandmarks: null,
  targetFaceLandmarks: null,
  currentPoseLandmarks: null,
  targetPoseLandmarks: null,
  faceDetected: false,
  poseDetected: false,
  running: false,
  fpsHistory: [],
  lastRenderTs: 0,
  cameraProbeResults: [],
};

const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
const LEFT_EYE = [362,385,387,263,373,380];
const RIGHT_EYE = [33,160,158,133,153,144];
const LEFT_IRIS = [474,475,476,477];
const RIGHT_IRIS = [469,470,471,472];
const MOUTH_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185];
const LEFT_BROW = [70,63,105,66,107,55,65,52,53,46];
const RIGHT_BROW = [336,296,334,293,300,285,295,282,283,276];

const trackerCanvas = document.getElementById("tracker-stage");
const trackerCtx = trackerCanvas.getContext("2d");
const videoFeed = document.getElementById("video-feed");

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

function setPill(id, label, tone) {
  const node = document.getElementById(id);
  if (!node) return;
  node.textContent = label;
  node.className = `pill ${tone}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function formatPercent(value) {
  return `${(clamp01(value) * 100).toFixed(1)}%`;
}

function resizeTracker() {
  const rect = trackerCanvas.getBoundingClientRect();
  trackerCanvas.width = Math.max(1, Math.floor(rect.width));
  trackerCanvas.height = Math.max(1, Math.floor(rect.height));
}

function cloneLandmarks(landmarks) {
  if (!Array.isArray(landmarks)) return null;
  return landmarks.map((point) => ({
    x: Number(point.x || 0),
    y: Number(point.y || 0),
    z: Number(point.z || 0),
    visibility: point.visibility == null ? 1 : Number(point.visibility),
  }));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpLandmarkSet(current, target, amount) {
  if (!Array.isArray(target) || target.length === 0) return null;
  if (!Array.isArray(current) || current.length !== target.length) return cloneLandmarks(target);
  return target.map((targetPoint, index) => {
    const currentPoint = current[index] || targetPoint;
    return {
      x: lerp(currentPoint.x, targetPoint.x, amount),
      y: lerp(currentPoint.y, targetPoint.y, amount),
      z: lerp(currentPoint.z, targetPoint.z, amount),
      visibility: lerp(currentPoint.visibility ?? 1, targetPoint.visibility ?? 1, amount),
    };
  });
}

function toCanvasPoints(landmarks, indices, width, height) {
  return indices
    .map((index) => landmarks[index])
    .filter(Boolean)
    .map((point) => [point.x * width, point.y * height]);
}

function drawPolygon(ctx, points, style = {}) {
  if (!points || points.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) ctx.lineTo(points[i][0], points[i][1]);
  if (style.closed !== false) ctx.closePath();
  if (style.fill) {
    ctx.fillStyle = style.fill;
    ctx.fill();
  }
  if (style.stroke) {
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = style.lineWidth || 1;
    ctx.stroke();
  }
}

function drawFaceProjection(ctx, landmarks, width, height) {
  drawPolygon(ctx, toCanvasPoints(landmarks, FACE_OVAL, width, height), {
    fill: "rgba(34,211,238,0.08)",
    stroke: "rgba(34,211,238,0.72)",
    lineWidth: 2,
  });
  drawPolygon(ctx, toCanvasPoints(landmarks, LEFT_BROW, width, height), { stroke: "rgba(129,140,248,0.95)", lineWidth: 2, closed: false });
  drawPolygon(ctx, toCanvasPoints(landmarks, RIGHT_BROW, width, height), { stroke: "rgba(129,140,248,0.95)", lineWidth: 2, closed: false });
  drawPolygon(ctx, toCanvasPoints(landmarks, LEFT_EYE, width, height), {
    fill: "rgba(52,211,153,0.12)",
    stroke: "rgba(45,212,191,0.96)",
    lineWidth: 2,
  });
  drawPolygon(ctx, toCanvasPoints(landmarks, RIGHT_EYE, width, height), {
    fill: "rgba(52,211,153,0.12)",
    stroke: "rgba(45,212,191,0.96)",
    lineWidth: 2,
  });
  drawPolygon(ctx, toCanvasPoints(landmarks, MOUTH_OUTER, width, height), {
    fill: "rgba(251,113,133,0.14)",
    stroke: "rgba(251,113,133,0.96)",
    lineWidth: 2,
  });

  ctx.fillStyle = "rgba(251,191,36,0.95)";
  [...LEFT_IRIS, ...RIGHT_IRIS].forEach((index) => {
    const point = landmarks[index];
    if (!point) return;
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawPoseProjection(ctx, landmarks, connections, width, height) {
  ctx.strokeStyle = "rgba(45,212,191,0.96)";
  ctx.lineWidth = 2.1;
  connections.forEach(([start, end]) => {
    const p1 = landmarks[start];
    const p2 = landmarks[end];
    if (!p1 || !p2) return;
    if ((p1.visibility ?? 1) < 0.35 || (p2.visibility ?? 1) < 0.35) return;
    ctx.beginPath();
    ctx.moveTo(p1.x * width, p1.y * height);
    ctx.lineTo(p2.x * width, p2.y * height);
    ctx.stroke();
  });
  ctx.fillStyle = "rgba(45,212,191,0.96)";
  landmarks.forEach((point) => {
    if (!point || (point.visibility ?? 1) < 0.35) return;
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, 3.2, 0, Math.PI * 2);
    ctx.fill();
  });
}

function updateRenderFps(now) {
  if (state.lastRenderTs > 0) {
    const fps = 1000 / Math.max(1, now - state.lastRenderTs);
    state.fpsHistory.push(fps);
    if (state.fpsHistory.length > 24) state.fpsHistory.shift();
    const avg = state.fpsHistory.reduce((sum, value) => sum + value, 0) / state.fpsHistory.length;
    setText("tracker-fps-chip", `${Math.round(avg)} fps`);
  }
  state.lastRenderTs = now;
}

function renderTracker(now) {
  updateRenderFps(now);
  resizeTracker();
  const width = trackerCanvas.width;
  const height = trackerCanvas.height;
  trackerCtx.clearRect(0, 0, width, height);

  trackerCtx.fillStyle = "#06111c";
  trackerCtx.fillRect(0, 0, width, height);
  trackerCtx.strokeStyle = "rgba(34,211,238,0.08)";
  trackerCtx.lineWidth = 1;
  for (let x = 0; x < width; x += 24) {
    trackerCtx.beginPath();
    trackerCtx.moveTo(x, 0);
    trackerCtx.lineTo(x, height);
    trackerCtx.stroke();
  }
  for (let y = 0; y < height; y += 24) {
    trackerCtx.beginPath();
    trackerCtx.moveTo(0, y);
    trackerCtx.lineTo(width, y);
    trackerCtx.stroke();
  }

  state.currentFaceLandmarks = state.faceDetected
    ? lerpLandmarkSet(state.currentFaceLandmarks, state.targetFaceLandmarks, 0.24)
    : null;
  state.currentPoseLandmarks = state.poseDetected
    ? lerpLandmarkSet(state.currentPoseLandmarks, state.targetPoseLandmarks, 0.22)
    : null;

  if (state.currentFaceLandmarks) drawFaceProjection(trackerCtx, state.currentFaceLandmarks, width, height);
  if (state.currentPoseLandmarks) drawPoseProjection(trackerCtx, state.currentPoseLandmarks, window.__lastPoseConnections || [], width, height);

  const empty = document.getElementById("tracker-stage-empty");
  if (empty) empty.classList.toggle("is-visible", !state.currentFaceLandmarks && !state.currentPoseLandmarks);

  state.renderHandle = requestAnimationFrame(renderTracker);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${url} -> ${response.status}`);
  return response.json();
}

function setHudDot(id, active) {
  const node = document.getElementById(id);
  if (!node) return;
  node.className = `hud-dot ${active ? "hud-dot-green" : "hud-dot-off"}`;
}

function updateTopRightSignals(signals) {
  setText("hud-smile", formatPercent(signals.smile));
  setText("hud-eyes", formatPercent(signals.eye_openness));
  setText("hud-gaze", formatPercent(signals.gaze));
}

function syncCameraControls(payload) {
  const cameraConfig = payload.camera_config || {};
  const sourceInput = document.getElementById("camera-source-input");
  const backendSelect = document.getElementById("camera-backend-select");
  if (sourceInput && document.activeElement !== sourceInput) {
    sourceInput.value = String(cameraConfig.source ?? 0);
  }
  if (backendSelect && document.activeElement !== backendSelect) {
    backendSelect.value = String(cameraConfig.backend || "auto");
  }
  setPill("camera-config-pill", payload.camera_running ? "Live" : "Idle", payload.camera_running ? "pill-ready" : "pill-neutral");
}

function updateCameraDiagnostics(payload) {
  const diag = payload.diagnostics || {};
  const frame = diag.frame || {};
  const parts = [
    `Camera: ${payload.camera_backend || "unbound"}`,
    `Blank frames: ${diag.blank_frame_count || 0}`,
  ];
  if (frame.available) {
    parts.push(`Frame μ ${Number(frame.mean || 0).toFixed(1)}`);
    parts.push(`σ ${Number(frame.std || 0).toFixed(1)}`);
    parts.push(frame.blank ? "Frame looks blank" : "Frame looks valid");
  }
  if (diag.last_camera_error) {
    parts.push(`Last error: ${diag.last_camera_error}`);
  }
  setText("camera-diagnostics", parts.join(" • "));
}

function renderProbeResults(results) {
  const container = document.getElementById("camera-probe-results");
  if (!container) return;
  if (!results || !results.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = results.map((item) => {
    const good = item.opened && item.diagnostics && !item.diagnostics.blank;
    const title = `source ${item.source} • ${item.backend}`;
    const meta = item.opened
      ? `opened=${item.opened} • backend=${item.backend_name || item.backend} • mean=${Number(item.diagnostics?.mean || 0).toFixed(1)} • std=${Number(item.diagnostics?.std || 0).toFixed(1)}`
      : `error=${escapeHtml(item.error || "open failed")}`;
    return `<div class="probe-item ${good ? "good" : "bad"}"><div class="probe-title">${escapeHtml(title)}</div><div class="probe-meta">${escapeHtml(meta)}</div></div>`;
  }).join("");
}

function updateState(payload) {
  const quality = payload.quality || {};
  const overlay = quality.overlay || {};
  const signals = quality.live_signals || {};
  const inference = payload.inference || {};
  const bridge = payload.bridge || {};
  const prototype = payload.prototype || {};
  const sync = payload.sync || {};

  state.faceDetected = Boolean(overlay.face_detected);
  state.poseDetected = Boolean(overlay.pose_detected);
  state.targetFaceLandmarks = state.faceDetected ? cloneLandmarks(overlay.face_landmarks || []) : null;
  state.targetPoseLandmarks = state.poseDetected ? cloneLandmarks(overlay.pose_landmarks || []) : null;
  window.__lastPoseConnections = overlay.pose_connections || [];

  setPill("camera-pill", payload.camera_running ? "Streaming" : "Idle", payload.camera_running ? "pill-ready" : "pill-neutral");
  setPill("readiness-pill", prototype.readiness || "idle", prototype.prediction_available ? "pill-ready" : "pill-warn");
  setPill("bridge-pill", bridge.available ? "Projected" : "Waiting", bridge.available ? "pill-ready" : "pill-neutral");
  setPill("expression-pill", quality.face_detected ? "Active" : "Waiting", quality.face_detected ? "pill-info" : "pill-neutral");
  setPill("risk-pill", inference.available ? "Model live" : "Tracking", inference.available ? "pill-ready" : "pill-info");

  setText("info-readiness", (prototype.readiness || "idle").replaceAll("_", " "));
  setText("readiness-message", prototype.message || "Waiting for tracker.");
  setText("face-detected", quality.face_detected ? "Locked" : "Searching");
  setText("loop-cadence", `${((sync.recommended_cadence_ms || 1000) / 1000).toFixed(1)}s`);
  setText("buffered-frames", String(payload.buffered_frames || 0));
  setText("last-api-state", payload.camera_running ? `#${sync.frame_seq || 0}` : "Idle");
  setText("video-message", prototype.message || "Tracking is booting.");
  setText("hud-face-text", quality.face_detected ? "Face locked" : "Face");
  setText("hud-pose-text", overlay.pose_detected ? "Pose locked" : "Pose");
  setText("hud-fps-chip", `${Math.round(sync.capture_fps || 0)} fps`);
  setHudDot("hud-face-dot", quality.face_detected);
  setHudDot("hud-pose-dot", overlay.pose_detected);
  setHudDot("tracker-face-dot", quality.face_detected);
  setHudDot("tracker-pose-dot", overlay.pose_detected);

  updateTopRightSignals(signals);
  syncCameraControls(payload);
  updateCameraDiagnostics(payload);

  const contextProgress = Math.min(1, (payload.buffered_frames || 0) / 30);
  setText("context-progress", formatPercent(contextProgress));
  const fill = document.getElementById("context-progress-fill");
  if (fill) fill.style.width = `${contextProgress * 100}%`;

  setText("bridge-state", bridge.available ? "Bridge ready" : (bridge.reason || "warming"));
  setText("bridge-visual-mean", Number(bridge.summary?.visual_mean_abs || 0).toFixed(4));
  setText("bridge-visual-std", Number(bridge.summary?.visual_std || 0).toFixed(4));
  setText("bridge-acoustic-mean", Number(bridge.summary?.acoustic_mean_abs || 0).toFixed(4));

  setText("mod-smile", formatPercent(signals.smile));
  setText("mod-brow", formatPercent(signals.brow));
  setText("mod-eyes", formatPercent(signals.eye_openness));
  setText("mod-mouth", formatPercent(signals.mouth_open));
  setText("mod-movement", formatPercent(Math.max(signals.face_movement || 0, signals.pose_movement || 0)));
  setText("mod-visual", formatPercent(signals.visual));
  setText("mod-body", formatPercent(signals.body));
  setText("mod-hands", formatPercent(signals.hands));
  setText("mod-gaze", formatPercent(signals.gaze));
  setText("mod-affect", formatPercent(signals.affect));

  setText("calibration-note", inference.available
    ? `Model probability ${(clamp01(inference.probability) * 100).toFixed(1)}% · ${String(inference.risk_band || "active").toUpperCase()}`
    : "Model output will appear after bridge context is ready."
  );
}

async function pollLiveState() {
  try {
    const payload = await fetchJson("/api/live-state", { cache: "no-store" });
    updateState(payload);
  } catch (error) {
    setText("video-message", `Live state fetch failed: ${error.message}`);
  }
}

async function startCamera() {
  try {
    const payload = await fetchJson("/api/camera/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (payload.status !== "ok") {
      throw new Error(payload.message || "camera_start_failed");
    }
    if (!videoFeed.src || !videoFeed.src.includes("/video_feed")) {
      videoFeed.src = `/video_feed?ts=${Date.now()}`;
    } else {
      videoFeed.src = `/video_feed?ts=${Date.now()}`;
    }
    state.running = true;
    if (!state.pollTimer) state.pollTimer = setInterval(pollLiveState, 140);
    await pollLiveState();
  } catch (error) {
    setText("video-message", `Camera start failed: ${error.message}`);
  }
}

async function applyCameraConfig() {
  const sourceInput = document.getElementById("camera-source-input");
  const backendSelect = document.getElementById("camera-backend-select");
  try {
    const payload = await fetchJson("/api/camera/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: sourceInput ? sourceInput.value : "0",
        backend: backendSelect ? backendSelect.value : "auto",
      }),
    });
    if (payload.status !== "ok" || payload.result?.ok === false) {
      throw new Error(payload.result?.message || "camera_config_failed");
    }
    updateState(payload.state || {});
    if (state.running) {
      videoFeed.src = `/video_feed?ts=${Date.now()}`;
    }
  } catch (error) {
    setText("camera-diagnostics", `Camera config failed: ${error.message}`);
  }
}

async function probeCameras() {
  try {
    const payload = await fetchJson("/api/camera/probe?max_sources=4", { cache: "no-store" });
    state.cameraProbeResults = payload.results || [];
    renderProbeResults(state.cameraProbeResults);
  } catch (error) {
    setText("camera-diagnostics", `Camera probe failed: ${error.message}`);
  }
}

async function stopCamera() {
  try {
    const payload = await fetchJson("/api/camera/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (payload.status !== "ok") {
      throw new Error(payload.message || "camera_stop_failed");
    }
    state.running = false;
    await pollLiveState();
  } catch (error) {
    setText("video-message", `Camera stop failed: ${error.message}`);
  }
}

async function resetSession() {
  try {
    await fetchJson("/api/reset-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    await pollLiveState();
  } catch (error) {
    setText("video-message", `Reset failed: ${error.message}`);
  }
}

function startSessionTimer() {
  state.sessionStartMs = Date.now();
  if (state.sessionTimer) clearInterval(state.sessionTimer);
  state.sessionTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.sessionStartMs) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const secs = String(elapsed % 60).padStart(2, "0");
    setText("session-timer", `${mins}:${secs}`);
  }, 1000);
}

function bindEvents() {
  document.getElementById("start-camera").addEventListener("click", startCamera);
  document.getElementById("pause-camera").addEventListener("click", stopCamera);
  document.getElementById("reset-session").addEventListener("click", resetSession);
  document.getElementById("apply-camera-config").addEventListener("click", applyCameraConfig);
  document.getElementById("probe-cameras").addEventListener("click", probeCameras);
  window.addEventListener("resize", resizeTracker);
}

function boot() {
  bindEvents();
  resizeTracker();
  startSessionTimer();
  pollLiveState();
  state.renderHandle = requestAnimationFrame(renderTracker);
  startCamera();
}

boot();
