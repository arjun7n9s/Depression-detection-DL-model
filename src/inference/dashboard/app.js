const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
const LEFT_EYE = [362,385,387,263,373,380];
const RIGHT_EYE = [33,160,158,133,153,144];
const LEFT_IRIS = [474,475,476,477];
const RIGHT_IRIS = [469,470,471,472];
const MOUTH_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185];
const LEFT_BROW = [70,63,105,66,107,55,65,52,53,46];
const RIGHT_BROW = [336,296,334,293,300,285,295,282,283,276];
const POSE_CONNECTIONS = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
const POSE_KEYPOINTS = [11,12,13,14,15,16,23,24,25,26,27,28];

const state = {
  stream: null,
  loopTimer: null,
  inFlight: false,
  running: false,
  captureWidth: 640,
  captureHeight: 480,
  cadenceMs: 140,
  recommendedCadenceMs: 1000,
  sessionStartMs: null,
  sessionTimerInterval: null,
  framesSent: 0,
  mpReady: false,
  faceLandmarker: null,
  poseLandmarker: null,
  animFrameId: null,
  overlayDetectIntervalMs: 110,
  lastOverlayDetectMs: 0,
  overlayMessage: "Overlay booting",
  faceLandmarks: null,
  faceBlendshapes: null,
  poseLandmarks: null,
  displayFaceLandmarks: null,
  displayPoseLandmarks: null,
  facePresent: false,
  posePresent: false,
  lastServerFaceDetected: false,
  lastServerBbox: null,
  prevFaceCentroid: null,
  prevPosePoints: null,
  faceMovement: 0,
  poseMovement: 0,
  handSignal: 0,
  visualSignal: 0,
  gazeSignal: 0,
  affectSignal: 0,
  bodySignal: 0,
  smileScore: 0,
  browScore: 0,
  eyeOpenness: 0,
  mouthOpen: 0,
  probHistory: [],
  maxHistory: 60,
  lastRenderTs: 0,
  renderFpsHistory: [],
};

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const trackerCanvas = document.getElementById("tracker-stage");
const trackerCtx = trackerCanvas ? trackerCanvas.getContext("2d") : null;
const sparkCanvas = document.getElementById("sparkline-canvas");
const sparkCtx = sparkCanvas ? sparkCanvas.getContext("2d") : null;

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

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function formatPercent(value) {
  return `${(clamp01(value) * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function humanizeBand(value) {
  if (!value) return "No context";
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function setMeter(labelId, fillId, value) {
  const pct = clamp01(Number(value) || 0);
  setText(labelId, formatPercent(pct));
  const fill = document.getElementById(fillId);
  if (fill) fill.style.width = `${pct * 100}%`;
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = Math.max(1, Math.floor(rect.width));
  overlay.height = Math.max(1, Math.floor(rect.height));
  if (trackerCanvas) {
    const trackerRect = trackerCanvas.getBoundingClientRect();
    trackerCanvas.width = Math.max(1, Math.floor(trackerRect.width));
    trackerCanvas.height = Math.max(1, Math.floor(trackerRect.height));
  }
}

function startSessionTimer() {
  state.sessionStartMs = Date.now();
  if (state.sessionTimerInterval) clearInterval(state.sessionTimerInterval);
  state.sessionTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.sessionStartMs) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const secs = String(elapsed % 60).padStart(2, "0");
    setText("session-timer", `${mins}:${secs}`);
  }, 1000);
}

function stopSessionTimer() {
  if (!state.sessionTimerInterval) return;
  clearInterval(state.sessionTimerInterval);
  state.sessionTimerInterval = null;
}

function updateRenderFps(now) {
  if (state.lastRenderTs > 0) {
    const fps = 1000 / Math.max(1, now - state.lastRenderTs);
    state.renderFpsHistory.push(fps);
    if (state.renderFpsHistory.length > 18) state.renderFpsHistory.shift();
    const avg = state.renderFpsHistory.reduce((sum, value) => sum + value, 0) / state.renderFpsHistory.length;
    setText("hud-fps-chip", `${Math.round(avg)} fps`);
    setText("tracker-fps-chip", `${Math.round(avg)} fps`);
  }
  state.lastRenderTs = now;
}

function drawSparkline() {
  if (!sparkCanvas || !sparkCtx) return;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, sparkCanvas.clientWidth * dpr);
  const height = Math.max(1, sparkCanvas.clientHeight * dpr);
  sparkCanvas.width = width;
  sparkCanvas.height = height;
  sparkCtx.clearRect(0, 0, width, height);
  if (state.probHistory.length < 2) return;

  const padX = 2;
  const padY = 4;
  const plotWidth = width - padX * 2;
  const plotHeight = height - padY * 2;
  const step = plotWidth / Math.max(1, state.maxHistory - 1);
  const startX = padX + (state.maxHistory - state.probHistory.length) * step;

  sparkCtx.strokeStyle = "rgba(148,163,184,0.16)";
  sparkCtx.lineWidth = 1;
  sparkCtx.setLineDash([4, 4]);
  sparkCtx.beginPath();
  sparkCtx.moveTo(padX, padY + plotHeight * 0.5);
  sparkCtx.lineTo(padX + plotWidth, padY + plotHeight * 0.5);
  sparkCtx.stroke();
  sparkCtx.setLineDash([]);

  const gradient = sparkCtx.createLinearGradient(0, padY, 0, padY + plotHeight);
  gradient.addColorStop(0, "rgba(34,211,238,0.22)");
  gradient.addColorStop(1, "rgba(34,211,238,0.02)");
  sparkCtx.beginPath();
  sparkCtx.moveTo(startX, padY + plotHeight);
  state.probHistory.forEach((value, index) => {
    sparkCtx.lineTo(startX + index * step, padY + plotHeight * (1 - clamp01(value)));
  });
  sparkCtx.lineTo(startX + (state.probHistory.length - 1) * step, padY + plotHeight);
  sparkCtx.closePath();
  sparkCtx.fillStyle = gradient;
  sparkCtx.fill();

  sparkCtx.beginPath();
  state.probHistory.forEach((value, index) => {
    const x = startX + index * step;
    const y = padY + plotHeight * (1 - clamp01(value));
    if (index === 0) sparkCtx.moveTo(x, y);
    else sparkCtx.lineTo(x, y);
  });
  sparkCtx.strokeStyle = "#22d3ee";
  sparkCtx.lineWidth = 2;
  sparkCtx.stroke();
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${url} -> ${response.status}`);
  return response.json();
}

async function initMediaPipe() {
  if (typeof FilesetResolver === "undefined" || typeof FaceLandmarker === "undefined" || typeof PoseLandmarker === "undefined") {
    state.overlayMessage = "Browser overlay runtime unavailable";
    return;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const makeFace = (delegate) => FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate,
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrices: false,
    });
    const makePose = (delegate) => PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    try { state.faceLandmarker = await makeFace("GPU"); } catch { state.faceLandmarker = await makeFace("CPU"); }
    try { state.poseLandmarker = await makePose("GPU"); } catch { state.poseLandmarker = await makePose("CPU"); }
    state.mpReady = Boolean(state.faceLandmarker && state.poseLandmarker);
    state.overlayMessage = state.mpReady ? "Overlay ready" : "Overlay unavailable";
  } catch (error) {
    console.warn("[MindSense] MediaPipe init failed:", error);
    state.overlayMessage = "Overlay init failed";
  }
}

function toCanvasPoints(landmarks, indices) {
  const points = [];
  for (const index of indices) {
    const point = landmarks[index];
    if (point) points.push([point.x * overlay.width, point.y * overlay.height]);
  }
  return points;
}

function drawPolygon(points, style = {}) {
  if (!points || points.length < 2) return;
  overlayCtx.beginPath();
  overlayCtx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) overlayCtx.lineTo(points[i][0], points[i][1]);
  if (style.closed !== false) overlayCtx.closePath();
  if (style.fill) {
    overlayCtx.fillStyle = style.fill;
    overlayCtx.fill();
  }
  if (style.stroke) {
    overlayCtx.strokeStyle = style.stroke;
    overlayCtx.lineWidth = style.lineWidth || 1;
    overlayCtx.stroke();
  }
}

function drawLandmarkPoints(landmarks, indices, fill, radius = 2.4) {
  overlayCtx.fillStyle = fill;
  indices.forEach((index) => {
    const point = landmarks[index];
    if (!point) return;
    overlayCtx.beginPath();
    overlayCtx.arc(point.x * overlay.width, point.y * overlay.height, radius, 0, Math.PI * 2);
    overlayCtx.fill();
  });
}

function cloneLandmarks(landmarks) {
  if (!Array.isArray(landmarks)) return null;
  return landmarks.map((point) => ({
    x: Number(point.x || 0),
    y: Number(point.y || 0),
    z: Number(point.z || 0),
    visibility: point.visibility ?? 1,
  }));
}

function lerpLandmarkSet(current, target, amount = 0.28) {
  if (!Array.isArray(target) || target.length === 0) return null;
  if (!Array.isArray(current) || current.length !== target.length) return cloneLandmarks(target);
  return target.map((targetPoint, index) => {
    const currentPoint = current[index] || targetPoint;
    return {
      x: lerp(currentPoint.x ?? targetPoint.x ?? 0, targetPoint.x ?? 0, amount),
      y: lerp(currentPoint.y ?? targetPoint.y ?? 0, targetPoint.y ?? 0, amount),
      z: lerp(currentPoint.z ?? targetPoint.z ?? 0, targetPoint.z ?? 0, amount),
      visibility: lerp(currentPoint.visibility ?? 1, targetPoint.visibility ?? 1, amount),
    };
  });
}

function syncRenderableLandmarks() {
  if (state.facePresent && Array.isArray(state.faceLandmarks) && state.faceLandmarks.length) {
    state.displayFaceLandmarks = lerpLandmarkSet(state.displayFaceLandmarks, state.faceLandmarks, 0.32);
  } else {
    state.displayFaceLandmarks = null;
  }

  if (state.posePresent && Array.isArray(state.poseLandmarks) && state.poseLandmarks.length) {
    state.displayPoseLandmarks = lerpLandmarkSet(state.displayPoseLandmarks, state.poseLandmarks, 0.30);
  } else {
    state.displayPoseLandmarks = null;
  }
}

function getRenderableFaceLandmarks() {
  return state.displayFaceLandmarks || state.faceLandmarks;
}

function getRenderablePoseLandmarks() {
  return state.displayPoseLandmarks || state.poseLandmarks;
}

function drawFaceOverlay() {
  const landmarks = getRenderableFaceLandmarks();
  if (!landmarks) return;
  drawPolygon(toCanvasPoints(landmarks, FACE_OVAL), {
    fill: "rgba(34,211,238,0.08)",
    stroke: "rgba(34,211,238,0.60)",
    lineWidth: 1.8,
  });
  drawPolygon(toCanvasPoints(landmarks, LEFT_BROW), {
    stroke: "rgba(129,140,248,0.88)",
    lineWidth: 1.8,
    closed: false,
  });
  drawPolygon(toCanvasPoints(landmarks, RIGHT_BROW), {
    stroke: "rgba(129,140,248,0.88)",
    lineWidth: 1.8,
    closed: false,
  });
  drawPolygon(toCanvasPoints(landmarks, LEFT_EYE), {
    fill: "rgba(52,211,153,0.12)",
    stroke: "rgba(52,211,153,0.94)",
    lineWidth: 1.7,
  });
  drawPolygon(toCanvasPoints(landmarks, RIGHT_EYE), {
    fill: "rgba(52,211,153,0.12)",
    stroke: "rgba(52,211,153,0.94)",
    lineWidth: 1.7,
  });
  drawPolygon(toCanvasPoints(landmarks, MOUTH_OUTER), {
    fill: "rgba(251,113,133,0.15)",
    stroke: "rgba(251,113,133,0.96)",
    lineWidth: 1.8,
  });
  drawLandmarkPoints(landmarks, LEFT_IRIS, "rgba(129,140,248,1.0)", 2.2);
  drawLandmarkPoints(landmarks, RIGHT_IRIS, "rgba(129,140,248,1.0)", 2.2);
}

function drawPoseOverlay() {
  const landmarks = getRenderablePoseLandmarks();
  if (!landmarks) return;
  overlayCtx.strokeStyle = "rgba(251,191,36,0.90)";
  overlayCtx.lineWidth = 2.5;
  for (const [startIndex, endIndex] of POSE_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    if (!start || !end) continue;
    if ((start.visibility ?? 1) < 0.35 || (end.visibility ?? 1) < 0.35) continue;
    overlayCtx.beginPath();
    overlayCtx.moveTo(start.x * overlay.width, start.y * overlay.height);
    overlayCtx.lineTo(end.x * overlay.width, end.y * overlay.height);
    overlayCtx.stroke();
  }
  overlayCtx.fillStyle = "rgba(251,191,36,0.98)";
  for (const index of POSE_KEYPOINTS) {
    const point = landmarks[index];
    if (!point) continue;
    if ((point.visibility ?? 1) < 0.35) continue;
    overlayCtx.beginPath();
    overlayCtx.arc(point.x * overlay.width, point.y * overlay.height, 4, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

function drawFallbackBbox() {
  if (!state.lastServerBbox) return;
  const [x0, y0, x1, y1] = state.lastServerBbox;
  const scaleX = overlay.width / state.captureWidth;
  const scaleY = overlay.height / state.captureHeight;
  const left = x0 * scaleX;
  const top = y0 * scaleY;
  const width = (x1 - x0) * scaleX;
  const height = (y1 - y0) * scaleY;
  overlayCtx.beginPath();
  overlayCtx.roundRect(left, top, width, height, 12);
  overlayCtx.fillStyle = state.lastServerFaceDetected ? "rgba(34,211,238,0.10)" : "rgba(251,113,133,0.05)";
  overlayCtx.strokeStyle = state.lastServerFaceDetected ? "rgba(34,211,238,0.72)" : "rgba(251,113,133,0.48)";
  overlayCtx.lineWidth = 2;
  overlayCtx.fill();
  overlayCtx.stroke();
}

function drawOverlayDiagnostics() {
  const cardWidth = 160;
  const cardHeight = 80;
  const cardX = overlay.width - cardWidth - 12;
  const cardY = 12;

  overlayCtx.fillStyle = "rgba(10,14,23,0.84)";
  overlayCtx.beginPath();
  overlayCtx.roundRect(cardX, cardY, cardWidth, cardHeight, 10);
  overlayCtx.fill();
  overlayCtx.font = "600 10px 'Segoe UI', sans-serif";
  overlayCtx.fillStyle = "rgba(255,255,255,0.62)";
  overlayCtx.fillText("LIVE TRACKING", cardX + 12, cardY + 20);

  const rows = [
    { label: "Smile", value: state.smileScore, color: "#34d399" },
    { label: "Eyes", value: state.eyeOpenness, color: "#22d3ee" },
    { label: "Gaze", value: state.gazeSignal, color: "#818cf8" },
  ];
  rows.forEach((row, index) => {
    const y = cardY + 34 + index * 15;
    overlayCtx.fillStyle = "rgba(255,255,255,0.18)";
    overlayCtx.fillRect(cardX + 42, y - 5, 88, 4);
    overlayCtx.fillStyle = row.color;
    overlayCtx.fillRect(cardX + 42, y - 5, 88 * clamp01(row.value), 4);
    overlayCtx.fillStyle = "rgba(255,255,255,0.70)";
    overlayCtx.fillText(row.label, cardX + 12, y);
  });
}

function toTrackerPoints(landmarks, indices) {
  const points = [];
  if (!trackerCanvas) return points;
  for (const index of indices) {
    const point = landmarks[index];
    if (point) points.push([point.x * trackerCanvas.width, point.y * trackerCanvas.height]);
  }
  return points;
}

function drawTrackerPolygon(points, style = {}) {
  if (!trackerCtx || !points || points.length < 2) return;
  trackerCtx.beginPath();
  trackerCtx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i += 1) trackerCtx.lineTo(points[i][0], points[i][1]);
  if (style.closed !== false) trackerCtx.closePath();
  if (style.fill) {
    trackerCtx.fillStyle = style.fill;
    trackerCtx.fill();
  }
  if (style.stroke) {
    trackerCtx.strokeStyle = style.stroke;
    trackerCtx.lineWidth = style.lineWidth || 1;
    trackerCtx.stroke();
  }
}

function drawTrackerPoints(landmarks, indices, fill, radius = 2.2) {
  if (!trackerCtx || !trackerCanvas) return;
  trackerCtx.fillStyle = fill;
  indices.forEach((index) => {
    const point = landmarks[index];
    if (!point) return;
    trackerCtx.beginPath();
    trackerCtx.arc(point.x * trackerCanvas.width, point.y * trackerCanvas.height, radius, 0, Math.PI * 2);
    trackerCtx.fill();
  });
}

function drawTrackerBackdrop() {
  if (!trackerCtx || !trackerCanvas) return;
  trackerCtx.clearRect(0, 0, trackerCanvas.width, trackerCanvas.height);
  const gradient = trackerCtx.createRadialGradient(
    trackerCanvas.width * 0.5,
    trackerCanvas.height * 0.42,
    trackerCanvas.height * 0.06,
    trackerCanvas.width * 0.5,
    trackerCanvas.height * 0.52,
    trackerCanvas.height * 0.86
  );
  gradient.addColorStop(0, "rgba(24, 38, 66, 0.82)");
  gradient.addColorStop(0.45, "rgba(12, 19, 34, 0.92)");
  gradient.addColorStop(1, "rgba(6, 10, 18, 1)");
  trackerCtx.fillStyle = gradient;
  trackerCtx.fillRect(0, 0, trackerCanvas.width, trackerCanvas.height);

  trackerCtx.save();
  trackerCtx.globalAlpha = 0.16;
  trackerCtx.strokeStyle = "rgba(148, 163, 184, 0.28)";
  trackerCtx.lineWidth = 1;
  for (let x = 20; x < trackerCanvas.width; x += 32) {
    trackerCtx.beginPath();
    trackerCtx.moveTo(x, 0);
    trackerCtx.lineTo(x, trackerCanvas.height);
    trackerCtx.stroke();
  }
  for (let y = 20; y < trackerCanvas.height; y += 32) {
    trackerCtx.beginPath();
    trackerCtx.moveTo(0, y);
    trackerCtx.lineTo(trackerCanvas.width, y);
    trackerCtx.stroke();
  }
  trackerCtx.restore();
}

function drawTrackerFaceOverlay() {
  const landmarks = getRenderableFaceLandmarks();
  if (!trackerCtx || !trackerCanvas || !landmarks) return;
  drawTrackerPolygon(toTrackerPoints(landmarks, FACE_OVAL), {
    fill: "rgba(34, 211, 238, 0.10)",
    stroke: "rgba(34, 211, 238, 0.92)",
    lineWidth: 2.1,
  });
  drawTrackerPolygon(toTrackerPoints(landmarks, LEFT_BROW), {
    stroke: "rgba(129, 140, 248, 0.96)",
    lineWidth: 2.0,
    closed: false,
  });
  drawTrackerPolygon(toTrackerPoints(landmarks, RIGHT_BROW), {
    stroke: "rgba(129, 140, 248, 0.96)",
    lineWidth: 2.0,
    closed: false,
  });
  drawTrackerPolygon(toTrackerPoints(landmarks, LEFT_EYE), {
    fill: "rgba(52, 211, 153, 0.16)",
    stroke: "rgba(52, 211, 153, 0.98)",
    lineWidth: 1.9,
  });
  drawTrackerPolygon(toTrackerPoints(landmarks, RIGHT_EYE), {
    fill: "rgba(52, 211, 153, 0.16)",
    stroke: "rgba(52, 211, 153, 0.98)",
    lineWidth: 1.9,
  });
  drawTrackerPolygon(toTrackerPoints(landmarks, MOUTH_OUTER), {
    fill: "rgba(251, 113, 133, 0.18)",
    stroke: "rgba(251, 113, 133, 0.98)",
    lineWidth: 2.0,
  });
  drawTrackerPoints(landmarks, LEFT_IRIS, "rgba(250, 204, 21, 1)", 2.6);
  drawTrackerPoints(landmarks, RIGHT_IRIS, "rgba(250, 204, 21, 1)", 2.6);
}

function drawTrackerPoseOverlay() {
  const landmarks = getRenderablePoseLandmarks();
  if (!trackerCtx || !trackerCanvas || !landmarks) return;
  trackerCtx.strokeStyle = "rgba(250, 204, 21, 0.98)";
  trackerCtx.lineWidth = 3;
  for (const [startIndex, endIndex] of POSE_CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    if (!start || !end) continue;
    if ((start.visibility ?? 1) < 0.35 || (end.visibility ?? 1) < 0.35) continue;
    trackerCtx.beginPath();
    trackerCtx.moveTo(start.x * trackerCanvas.width, start.y * trackerCanvas.height);
    trackerCtx.lineTo(end.x * trackerCanvas.width, end.y * trackerCanvas.height);
    trackerCtx.stroke();
  }
  trackerCtx.fillStyle = "rgba(250, 204, 21, 0.98)";
  for (const index of POSE_KEYPOINTS) {
    const point = landmarks[index];
    if (!point || (point.visibility ?? 1) < 0.35) continue;
    trackerCtx.beginPath();
    trackerCtx.arc(point.x * trackerCanvas.width, point.y * trackerCanvas.height, 4.2, 0, Math.PI * 2);
    trackerCtx.fill();
  }
}

function drawTrackerLabels() {
  if (!trackerCtx || !trackerCanvas) return;
  trackerCtx.save();
  trackerCtx.fillStyle = "rgba(226, 232, 240, 0.72)";
  trackerCtx.font = "700 11px 'Segoe UI', sans-serif";
  trackerCtx.fillText("LIVE SUBJECT PROJECTION", 18, trackerCanvas.height - 22);
  trackerCtx.restore();
}

function drawTrackerStage() {
  if (!trackerCtx || !trackerCanvas) return;
  drawTrackerBackdrop();
  if (state.posePresent) drawTrackerPoseOverlay();
  if (state.facePresent) drawTrackerFaceOverlay();
  drawTrackerLabels();
}

function getBlendshapeScore(categories, name) {
  const entry = categories.find((item) => item.categoryName === name);
  return entry ? entry.score : 0;
}

function extractBlendshapeSignals() {
  if (!state.faceBlendshapes || state.faceBlendshapes.length === 0) {
    state.smileScore = lerp(state.smileScore, 0, 0.35);
    state.browScore = lerp(state.browScore, 0, 0.35);
    state.eyeOpenness = lerp(state.eyeOpenness, 0, 0.35);
    state.mouthOpen = lerp(state.mouthOpen, 0, 0.35);
    return;
  }
  const categories = state.faceBlendshapes[0].categories || [];
  const smile = (getBlendshapeScore(categories, "mouthSmileLeft") + getBlendshapeScore(categories, "mouthSmileRight")) * 0.5;
  const brow = Math.max(
    getBlendshapeScore(categories, "browInnerUp"),
    getBlendshapeScore(categories, "browOuterUpLeft"),
    getBlendshapeScore(categories, "browOuterUpRight"),
    getBlendshapeScore(categories, "browDownLeft"),
    getBlendshapeScore(categories, "browDownRight")
  );
  const eyeClosed = (getBlendshapeScore(categories, "eyeBlinkLeft") + getBlendshapeScore(categories, "eyeBlinkRight")) * 0.5;
  const mouth = Math.max(getBlendshapeScore(categories, "jawOpen"), getBlendshapeScore(categories, "mouthOpen"));
  state.smileScore = lerp(state.smileScore, smile, 0.35);
  state.browScore = lerp(state.browScore, brow, 0.30);
  state.eyeOpenness = lerp(state.eyeOpenness, 1 - eyeClosed, 0.30);
  state.mouthOpen = lerp(state.mouthOpen, mouth, 0.30);
}

function computeFaceMovement() {
  if (!state.faceLandmarks || !state.faceLandmarks[1]) {
    state.prevFaceCentroid = null;
    return 0;
  }
  const nose = state.faceLandmarks[1];
  const centroid = { x: nose.x, y: nose.y };
  let movement = 0;
  if (state.prevFaceCentroid) {
    const dx = centroid.x - state.prevFaceCentroid.x;
    const dy = centroid.y - state.prevFaceCentroid.y;
    movement = Math.sqrt(dx * dx + dy * dy);
  }
  state.prevFaceCentroid = centroid;
  return movement;
}

function computePoseMovement() {
  if (!state.poseLandmarks || state.poseLandmarks.length === 0) {
    state.prevPosePoints = null;
    return 0;
  }
  let movement = 0;
  if (state.prevPosePoints && state.prevPosePoints.length === state.poseLandmarks.length) {
    let total = 0;
    let used = 0;
    for (const index of [11, 12, 15, 16]) {
      const curr = state.poseLandmarks[index];
      const prev = state.prevPosePoints[index];
      if (!curr || !prev) continue;
      total += Math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2);
      used += 1;
    }
    movement = used > 0 ? total / used : 0;
  }
  state.prevPosePoints = state.poseLandmarks.map((point) => ({ x: point.x, y: point.y, z: point.z, visibility: point.visibility ?? 1 }));
  return movement;
}

function updateTrackingSignals() {
  extractBlendshapeSignals();
  const faceMovement = computeFaceMovement();
  const poseMovement = computePoseMovement();

  state.faceMovement = lerp(state.faceMovement, faceMovement, 0.25);
  state.poseMovement = lerp(state.poseMovement, poseMovement, 0.25);
  state.handSignal = state.posePresent ? clamp01(state.poseMovement * 22) : lerp(state.handSignal, 0, 0.20);

  if (state.facePresent) {
    const stability = clamp01(1 - state.faceMovement * 18);
    state.visualSignal = lerp(state.visualSignal, 0.55 + 0.45 * stability, 0.18);
    state.gazeSignal = lerp(state.gazeSignal, clamp01(state.eyeOpenness * 0.75 + stability * 0.25), 0.18);
    state.affectSignal = lerp(state.affectSignal, clamp01(Math.max(state.smileScore, state.browScore * 0.9, state.mouthOpen * 0.8)), 0.20);
  } else {
    state.visualSignal = lerp(state.visualSignal, 0, 0.22);
    state.gazeSignal = lerp(state.gazeSignal, 0, 0.22);
    state.affectSignal = lerp(state.affectSignal, 0, 0.22);
  }

  if (state.posePresent) state.bodySignal = lerp(state.bodySignal, clamp01(0.35 + state.poseMovement * 20), 0.18);
  else state.bodySignal = lerp(state.bodySignal, 0, 0.20);

  setText("mod-smile", state.facePresent ? `${Math.round(state.smileScore * 100)}%` : "—");
  setText("mod-brow", state.facePresent ? `${Math.round(state.browScore * 100)}%` : "—");
  setText("mod-eyes", state.facePresent ? `${Math.round(state.eyeOpenness * 100)}%` : "—");
  setText("mod-mouth", state.facePresent ? `${Math.round(state.mouthOpen * 100)}%` : "—");
  setText("mod-movement", state.facePresent || state.posePresent ? `${Math.round(clamp01((state.faceMovement + state.poseMovement) * 24) * 100)}%` : "—");
  setText("mod-visual", formatPercent(state.visualSignal));
  setText("mod-body", formatPercent(state.bodySignal));
  setText("mod-hands", formatPercent(state.handSignal));
  setText("mod-gaze", formatPercent(state.gazeSignal));
  setText("mod-affect", formatPercent(state.affectSignal));

  if (state.facePresent) {
    if (state.smileScore >= 0.28) setPill("expression-pill", "Smiling", "pill-ready");
    else if (Math.max(state.browScore, state.mouthOpen) >= 0.24) setPill("expression-pill", "Expressive", "pill-info");
    else setPill("expression-pill", "Neutral", "pill-warn");
  } else {
    setPill("expression-pill", state.posePresent ? "Pose only" : "No face", state.posePresent ? "pill-warn" : "pill-neutral");
  }

  document.getElementById("hud-face-dot").className = `hud-dot ${state.facePresent ? "hud-dot-green" : "hud-dot-red"}`;
  document.getElementById("hud-face-text").textContent = state.facePresent ? "Face tracked" : "Face lost";
  document.getElementById("hud-pose-dot").className = `hud-dot ${state.posePresent ? "hud-dot-green" : "hud-dot-amber"}`;
  document.getElementById("hud-pose-text").textContent = state.posePresent ? "Pose tracked" : "Pose weak";
}

function detectMediaPipe(now) {
  if (!state.mpReady || video.readyState < 2) return;
  if (now - state.lastOverlayDetectMs < state.overlayDetectIntervalMs) return;
  state.lastOverlayDetectMs = now;

  try {
    const faceResult = state.faceLandmarker.detectForVideo(video, now);
    if (faceResult?.faceLandmarks?.length) {
      state.faceLandmarks = faceResult.faceLandmarks[0];
      state.faceBlendshapes = faceResult.faceBlendshapes || null;
      state.facePresent = true;
    } else {
      state.faceLandmarks = null;
      state.faceBlendshapes = null;
      state.facePresent = false;
    }
  } catch (error) {
    console.warn("[MindSense] Face detect failed:", error);
    state.faceLandmarks = null;
    state.faceBlendshapes = null;
    state.facePresent = false;
  }

  try {
    const poseResult = state.poseLandmarker.detectForVideo(video, now + 1);
    if (poseResult?.landmarks?.length) {
      state.poseLandmarks = poseResult.landmarks[0];
      state.posePresent = true;
    } else {
      state.poseLandmarks = null;
      state.posePresent = false;
    }
  } catch (error) {
    console.warn("[MindSense] Pose detect failed:", error);
    state.poseLandmarks = null;
    state.posePresent = false;
  }

  updateTrackingSignals();
}

function drawLiveOverlays(now) {
  resizeOverlay();
  syncRenderableLandmarks();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  updateRenderFps(now);
  drawTrackerStage();
  if (video.readyState < 2) {
    if (state.running) state.animFrameId = requestAnimationFrame(drawLiveOverlays);
    return;
  }

  overlayCtx.save();
  overlayCtx.translate(overlay.width, 0);
  overlayCtx.scale(-1, 1);
  if (state.facePresent) drawFaceOverlay();
  else drawFallbackBbox();
  if (state.posePresent) drawPoseOverlay();
  overlayCtx.restore();
  drawOverlayDiagnostics();
  drawTrackerStage();

  if (state.running) state.animFrameId = requestAnimationFrame(drawLiveOverlays);
}

function updateServerRisk(probability, band, available) {
  if (!available) {
    setPill("risk-pill", "Building context", "pill-warn");
    setText("tracker-band", "No context");
    return;
  }

  const tone = band === "elevated" ? "pill-danger" : band === "moderate" ? "pill-warn" : "pill-ready";
  setPill("risk-pill", "Model live", tone);
  setText("tracker-band", humanizeBand(band));
}

function updateHistory(probability, available) {
  if (!available) return;
  state.probHistory.push(clamp01(probability));
  if (state.probHistory.length > state.maxHistory) state.probHistory.shift();
  drawSparkline();
}

function renderPayload(payload) {
  const bridge = payload.bridge || {};
  const inference = payload.inference || {};
  const prototype = payload.prototype || {};
  const quality = payload.quality || {};
  const overlayData = quality.overlay || {};
  const liveSignals = quality.live_signals || {};
  const sync = payload.sync || {};

  state.lastServerFaceDetected = Boolean(quality.face_detected);
  state.lastServerBbox = Array.isArray(quality.bbox) ? quality.bbox : null;
  state.faceLandmarks = Array.isArray(overlayData.face_landmarks) && overlayData.face_landmarks.length ? overlayData.face_landmarks : null;
  state.poseLandmarks = Array.isArray(overlayData.pose_landmarks) && overlayData.pose_landmarks.length ? overlayData.pose_landmarks : null;
  state.facePresent = Boolean(overlayData.face_detected);
  state.posePresent = Boolean(overlayData.pose_detected);
  state.smileScore = Number(liveSignals.smile || 0);
  state.browScore = Number(liveSignals.brow || 0);
  state.eyeOpenness = Number(liveSignals.eye_openness || 0);
  state.mouthOpen = Number(liveSignals.mouth_open || 0);
  state.faceMovement = Number(liveSignals.face_movement || 0);
  state.poseMovement = Number(liveSignals.pose_movement || 0);
  state.visualSignal = Number(liveSignals.visual || 0);
  state.gazeSignal = Number(liveSignals.gaze || 0);
  state.affectSignal = Number(liveSignals.affect || 0);
  state.bodySignal = Number(liveSignals.body || 0);
  state.handSignal = Number(liveSignals.hands || 0);

  if (sync.recommended_cadence_ms) {
    state.recommendedCadenceMs = Number(sync.recommended_cadence_ms);
    setText("loop-cadence", `${(state.cadenceMs / 1000).toFixed(1)}s / ${(state.recommendedCadenceMs / 1000).toFixed(1)}s`);
    setText("tracker-cadence", `${(state.cadenceMs / 1000).toFixed(1)}s / ${(state.recommendedCadenceMs / 1000).toFixed(1)}s`);
  }

  const bufferedFrames = Number(payload.buffered_frames || 0);
  const requiredFrames = Number(bridge.required_frames || 30);
  const progress = requiredFrames > 0 ? Math.min(1, bufferedFrames / requiredFrames) : 0;
  setText("buffered-frames", String(bufferedFrames));
  setText("context-progress", formatPercent(progress));
  document.getElementById("context-progress-fill").style.width = `${progress * 100}%`;

  const readiness = prototype.readiness || "tracking_weak";
  const readinessTone = readiness === "live_inference_ready" ? "pill-ready" : readiness === "tracking_weak" ? "pill-danger" : "pill-warn";
  setPill("readiness-pill", humanizeBand(readiness), readinessTone);
  setText("info-readiness", humanizeBand(readiness));
  setText("readiness-message", prototype.message || state.overlayMessage);

  if (bridge.available) {
    setPill("bridge-pill", "Active", "pill-ready");
    setText("bridge-state", `Projected ${bridge.current_frames}/${bridge.required_frames}`);
    setText("bridge-visual-mean", formatNumber(bridge.summary?.visual_mean_abs));
    setText("bridge-visual-std", formatNumber(bridge.summary?.visual_std));
    setText("bridge-acoustic-mean", formatNumber(bridge.summary?.acoustic_mean_abs));
  } else {
    setPill("bridge-pill", bufferedFrames > 0 ? "Building" : "Waiting", "pill-warn");
    setText("bridge-state", bridge.reason ? humanizeBand(bridge.reason) : `${bufferedFrames}/${requiredFrames}`);
    setText("bridge-visual-mean", "0.0000");
    setText("bridge-visual-std", "0.0000");
    setText("bridge-acoustic-mean", "0.0000");
  }

  updateServerRisk(Number(inference.probability || 0), inference.risk_band || "low", Boolean(inference.available));
  updateHistory(Number(inference.probability || 0), Boolean(inference.available));

  const branchProbabilities = inference.branch_probabilities || {};
  setMeter("branch-acoustic", "branch-acoustic-fill", branchProbabilities.acoustic || 0);
  setMeter("branch-visual", "branch-visual-fill", branchProbabilities.visual || 0);
  setMeter("branch-fused", "branch-fused-fill", branchProbabilities.fused || 0);

  const gateWeights = inference.gate_weights || {};
  setMeter("gate-acoustic", "gate-acoustic-fill", gateWeights.acoustic || 0);
  setMeter("gate-visual", "gate-visual-fill", gateWeights.visual || 0);
  setMeter("gate-fused", "gate-fused-fill", gateWeights.fused || 0);

  const processingMs = sync.processing_ms != null ? `${sync.processing_ms.toFixed(1)}ms` : "—";
  const gapMs = sync.inter_request_ms != null ? `${Math.round(sync.inter_request_ms)}ms` : "—";
  const seqText = sync.frame_seq != null ? `#${sync.frame_seq}` : "—";
  setText("last-api-state", `${seqText} · ${processingMs}`);

  const calibration = inference.calibration || {};
  const qualityTrust = calibration.quality_trust != null ? formatPercent(calibration.quality_trust) : "—";
  const featureActivity = payload.feature_activity != null ? formatNumber(payload.feature_activity, 3) : "—";
  setText("tracker-trust", qualityTrust);
  setText("calibration-note", `Live tracking updates locally. Server seq ${seqText} · proc ${processingMs} · gap ${gapMs} · activity ${featureActivity} · trust ${qualityTrust}`);

  const faceState = state.facePresent ? "Tracked" : (quality.face_detected ? "Detected" : "No");
  setText("face-detected", faceState);

  setText("mod-smile", state.facePresent ? `${Math.round(state.smileScore * 100)}%` : "—");
  setText("mod-brow", state.facePresent ? `${Math.round(state.browScore * 100)}%` : "—");
  setText("mod-eyes", state.facePresent ? `${Math.round(state.eyeOpenness * 100)}%` : "—");
  setText("mod-mouth", state.facePresent ? `${Math.round(state.mouthOpen * 100)}%` : "—");
  setText("mod-movement", state.facePresent || state.posePresent ? `${Math.round(clamp01((state.faceMovement + state.poseMovement) * 24) * 100)}%` : "—");
  setText("mod-visual", formatPercent(state.visualSignal));
  setText("mod-body", formatPercent(state.bodySignal));
  setText("mod-hands", formatPercent(state.handSignal));
  setText("mod-gaze", formatPercent(state.gazeSignal));
  setText("mod-affect", formatPercent(state.affectSignal));

  if (state.facePresent) {
    if (state.smileScore >= 0.28) setPill("expression-pill", "Smiling", "pill-ready");
    else if (Math.max(state.browScore, state.mouthOpen) >= 0.24) setPill("expression-pill", "Expressive", "pill-info");
    else setPill("expression-pill", "Neutral", "pill-warn");
  } else {
    setPill("expression-pill", state.posePresent ? "Pose only" : "No face", state.posePresent ? "pill-warn" : "pill-neutral");
  }

  document.getElementById("hud-face-dot").className = `hud-dot ${state.facePresent ? "hud-dot-green" : "hud-dot-red"}`;
  document.getElementById("hud-face-text").textContent = state.facePresent ? "Face tracked" : "Face lost";
  document.getElementById("hud-pose-dot").className = `hud-dot ${state.posePresent ? "hud-dot-green" : "hud-dot-amber"}`;
  document.getElementById("hud-pose-text").textContent = state.posePresent ? "Pose tracked" : "Pose weak";
  document.getElementById("tracker-face-dot").className = `hud-dot ${state.facePresent ? "hud-dot-green" : "hud-dot-red"}`;
  document.getElementById("tracker-pose-dot").className = `hud-dot ${state.posePresent ? "hud-dot-green" : "hud-dot-amber"}`;
  const trackerEmpty = document.getElementById("tracker-stage-empty");
  if (trackerEmpty) trackerEmpty.classList.toggle("is-visible", !state.facePresent && !state.posePresent);

  if (!state.facePresent && !state.posePresent) {
    setText("video-message", "Face and pose tracking are currently absent. Step back into frame for overlays and live expression metrics.");
  } else if (inference.available) {
    setText("video-message", `Overlay tracking is live. Server probability is ${(Number(inference.probability || 0) * 100).toFixed(1)}% from the buffered ${requiredFrames}-step context.`);
  } else if (bridge.available) {
    setText("video-message", "Overlay tracking is live. Bridge is ready; the model is waiting for enough aligned context for inference.");
  } else {
    setText("video-message", "Overlay tracking is live. The server is collecting one aligned sample per second to match the training-time video cadence.");
  }
}

async function bootstrap() {
  try {
    const [health, bridgeStatus] = await Promise.all([
      fetchJson("/health"),
      fetchJson("/api/bridge-status"),
    ]);
    if (health.recommended_cadence_ms) {
      state.recommendedCadenceMs = Number(health.recommended_cadence_ms);
      setText("loop-cadence", `${(state.cadenceMs / 1000).toFixed(1)}s / ${(state.recommendedCadenceMs / 1000).toFixed(1)}s`);
    }
    const summary = bridgeStatus.summary || {};
    setPill("readiness-pill", "Ready", "pill-ready");
    setText("info-readiness", "Ready to stream");
    setText("readiness-message", `Bridge prep is available for ${summary.complete || 0} subjects. Overlay tracking should respond immediately; model inference follows the aligned live cadence.`);
  } catch (error) {
    console.error(error);
    setPill("readiness-pill", "Bootstrap issue", "pill-danger");
    setText("info-readiness", "Bootstrap failed");
    setText("readiness-message", "One or more server endpoints failed during dashboard initialization.");
  }
}

async function startCamera() {
  if (state.stream) return;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    resizeOverlay();
    state.running = true;
    state.framesSent = 0;
    setPill("camera-pill", "Streaming", "pill-ready");
    setText("video-message", "Camera is live. Face mask and pose skeleton overlays should respond immediately.");
    startSessionTimer();
    state.animFrameId = requestAnimationFrame(drawLiveOverlays);
    scheduleLoop();
  } catch (error) {
    console.error(error);
    setPill("camera-pill", "Camera blocked", "pill-danger");
    setText("video-message", "Camera access failed. Allow webcam permission and try again.");
  }
}

function stopLoop() {
  state.running = false;
  if (state.loopTimer) clearTimeout(state.loopTimer);
  if (state.animFrameId) cancelAnimationFrame(state.animFrameId);
  state.loopTimer = null;
  state.animFrameId = null;
  stopSessionTimer();
  setPill("camera-pill", "Paused", "pill-warn");
}

async function resetSession() {
  stopLoop();
  try {
    await fetchJson("/api/reset-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
  } catch (error) {
    console.error(error);
  }

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  state.faceLandmarks = null;
  state.faceBlendshapes = null;
  state.poseLandmarks = null;
  state.displayFaceLandmarks = null;
  state.displayPoseLandmarks = null;
  state.facePresent = false;
  state.posePresent = false;
  state.prevFaceCentroid = null;
  state.prevPosePoints = null;
  state.faceMovement = 0;
  state.poseMovement = 0;
  state.handSignal = 0;
  state.visualSignal = 0;
  state.gazeSignal = 0;
  state.affectSignal = 0;
  state.bodySignal = 0;
  state.smileScore = 0;
  state.browScore = 0;
  state.eyeOpenness = 0;
  state.mouthOpen = 0;
  state.lastServerFaceDetected = false;
  state.lastServerBbox = null;
  state.probHistory = [];
  state.renderFpsHistory = [];
  state.lastRenderTs = 0;
  drawSparkline();

  setText("session-timer", "00:00");
  setText("buffered-frames", "0");
  setText("context-progress", "0.0%");
  document.getElementById("context-progress-fill").style.width = "0%";
  setPill("risk-pill", "Waiting", "pill-neutral");
  setText("tracker-band", "No context");
  setText("tracker-trust", "—");
  setText("tracker-cadence", `${(state.cadenceMs / 1000).toFixed(1)}s / ${(state.recommendedCadenceMs / 1000).toFixed(1)}s`);
  setPill("bridge-pill", "Reset", "pill-neutral");
  setText("bridge-state", "Reset");
  setText("last-api-state", "Reset");
  setText("face-detected", "—");
  setText("video-message", "Session reset. Start streaming again to rebuild overlays and context.");
  setText("calibration-note", "Client tracking signals and server sync diagnostics will appear once the session restarts.");
  setPill("expression-pill", "Waiting", "pill-neutral");
  document.getElementById("tracker-face-dot").className = "hud-dot hud-dot-off";
  document.getElementById("tracker-pose-dot").className = "hud-dot hud-dot-off";
  const trackerEmpty = document.getElementById("tracker-stage-empty");
  if (trackerEmpty) trackerEmpty.classList.add("is-visible");

  ["branch-acoustic", "branch-visual", "branch-fused", "gate-acoustic", "gate-visual", "gate-fused"].forEach((id) => setText(id, "0.0%"));
  ["branch-acoustic-fill", "branch-visual-fill", "branch-fused-fill", "gate-acoustic-fill", "gate-visual-fill", "gate-fused-fill"].forEach((id) => {
    const node = document.getElementById(id);
    if (node) node.style.width = "0%";
  });
  ["mod-smile", "mod-brow", "mod-eyes", "mod-mouth", "mod-movement", "mod-visual", "mod-body", "mod-hands", "mod-gaze", "mod-affect"].forEach((id) => setText(id, "—"));

  if (state.stream) {
    state.running = true;
    setPill("camera-pill", "Streaming", "pill-ready");
    startSessionTimer();
    state.animFrameId = requestAnimationFrame(drawLiveOverlays);
    scheduleLoop();
  } else {
    setPill("camera-pill", "Idle", "pill-neutral");
  }
}

function scheduleLoop() {
  if (!state.running) return;
  if (state.loopTimer) clearTimeout(state.loopTimer);
  state.loopTimer = setTimeout(sendFrame, state.cadenceMs);
}

async function sendFrame() {
  if (!state.running || state.inFlight || video.readyState < 2) {
    scheduleLoop();
    return;
  }

  state.inFlight = true;
  setText("last-api-state", "Sending");
  try {
    const canvas = document.createElement("canvas");
    canvas.width = state.captureWidth;
    canvas.height = state.captureHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageBase64 = canvas.toDataURL("image/jpeg", 0.82).split(",")[1];

    const payload = await fetchJson("/api/extract-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: imageBase64 }),
    });

    state.framesSent += 1;
    renderPayload(payload);
  } catch (error) {
    console.error(error);
    setText("last-api-state", "Error");
  } finally {
    state.inFlight = false;
    scheduleLoop();
  }
}

function attachEvents() {
  document.getElementById("start-camera").addEventListener("click", startCamera);
  document.getElementById("pause-camera").addEventListener("click", stopLoop);
  document.getElementById("reset-session").addEventListener("click", resetSession);
  window.addEventListener("resize", resizeOverlay);
}

attachEvents();
bootstrap();
