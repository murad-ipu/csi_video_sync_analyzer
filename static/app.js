// CSI + video player. All times are seconds since the start of the session.
// The sync offset moves CSI later when positive: csi_t + offset lines up with frame_t.
"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video");
const COLORS = { surface: [26, 26, 25], line: "#383835", muted: "#a8a7a1", ink: "#f2f1ed", series: "#3987e5", cursor: "#f2f1ed" };
const MARGIN = { left: 42, right: 8, top: 4, bottom: 18 };

const state = {
  frame: 0, csi: 0,
  pinnedCsi: null,      // set by Shift+arrow so CSI stays put while the video seeks
  pendingFrame: null,   // target of a seek in progress, so quick arrow presses add up
  offsetS: 0, windowS: 10, normalized: true, renderQueued: false,
  mode: "time",         // "time", "frequency" (frequency.js) or "details" (details.js)
  detailsPanel: false,  // the CSI details panel under the video
};
const MODE_PANELS = {
  time: ["panelWaterfall", "panelFrame", "panelMotion"],
  frequency: ["freqControls", "panelSpectrogram", "panelSpectrum", "panelBands"],
  details: ["panelDetails"],
};
let meta, frameT, framePts, csiT, csiRssi, ampRaw, ampNorm, chartT, chartMotion, chartRssi, chartX;
let bins, K, motionPlot, rssiPlot;
const renderTimes = [];
window.playerDebug = { state, renderTimes };

// Binary search: index of the value nearest to v in a sorted typed array.
function nearest(values, v) {
  let low = 0, high = values.length - 1;
  while (high - low > 1) {
    const mid = (low + high) >> 1;
    if (values[mid] <= v) low = mid; else high = mid;
  }
  return Math.abs(values[low] - v) <= Math.abs(values[high] - v) ? low : high;
}

const ARRAY_TYPES = { f8: Float64Array, f4: Float32Array, i1: Int8Array, i8: BigInt64Array };

async function fetchArray(name) {
  const response = await fetch(`api/array/${name}`);
  if (!response.ok) throw new Error(`Could not load ${name}`);
  const buffer = await response.arrayBuffer();
  return new ARRAY_TYPES[meta.arrays[name]](buffer);
}

// One-hue ramp from near the surface to near white: brighter means stronger.
const COLOR_LUT = (() => {
  const stops = [[0.0, 18, 35, 61], [0.35, 24, 79, 149], [0.65, 42, 120, 214], [0.85, 127, 178, 238], [1.0, 227, 239, 252]];
  const lut = new Uint8Array(256 * 3);
  for (let n = 0; n < 256; n++) {
    const x = n / 255;
    let s = 0;
    while (s < stops.length - 2 && x > stops[s + 1][0]) s++;
    const a = stops[s], b = stops[s + 1], f = (x - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) lut[n * 3 + c] = a[c + 1] + (b[c + 1] - a[c + 1]) * f;
  }
  return lut;
})();

// Match a canvas to its CSS size at device resolution, drawing in CSS pixels.
function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const deviceWidth = Math.round(width * ratio), deviceHeight = Math.round(height * ratio);
  if (canvas.width !== deviceWidth || canvas.height !== deviceHeight) {
    canvas.width = deviceWidth;
    canvas.height = deviceHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "11px system-ui, 'Segoe UI', sans-serif";
  ctx.textBaseline = "middle";
  return { ctx, width, height };
}

function currentAmp() { return state.normalized ? ampNorm : ampRaw; }
function currentRange() { return state.normalized ? meta.amp_norm_range : meta.amp_raw_range; }

// The time at the center of every chart.
function cursorTime() {
  return state.pinnedCsi !== null ? csiT[state.csi] + state.offsetS : frameT[state.frame];
}

// --- Waterfall --------------------------------------------------------------------
const heatCanvas = document.createElement("canvas");
const heatCtx = heatCanvas.getContext("2d");
let heatImage = null;

function drawWaterfall() {
  const { ctx, width, height } = prepareCanvas($("waterfall"));
  const plotW = Math.max(1, Math.floor(width - MARGIN.left - MARGIN.right));
  const plotH = Math.max(1, height - MARGIN.top - MARGIN.bottom);
  if (!heatImage || heatImage.width !== plotW) {
    heatCanvas.width = plotW;
    heatCanvas.height = K;
    heatImage = heatCtx.createImageData(plotW, K);
  }
  const pixels = heatImage.data, amp = currentAmp(), [low, high] = currentRange();
  const scale = 255 / Math.max(1e-6, high - low);
  const center = cursorTime(), span = state.windowS * 2, start = center - state.windowS;
  // A column with no CSI frame close to it stays blank, so packet gaps are visible.
  const maxGap = Math.max(3 * meta.csi_step_s, span / plotW);

  for (let x = 0; x < plotW; x++) {
    const query = start + ((x + 0.5) / plotW) * span - state.offsetS;
    const index = nearest(csiT, query);
    const valid = Math.abs(csiT[index] - query) <= maxGap;
    for (let k = 0; k < K; k++) {
      const p = ((K - 1 - k) * plotW + x) * 4;  // lowest subcarrier at the bottom
      if (valid) {
        const level = Math.max(0, Math.min(255, Math.round((amp[index * K + k] - low) * scale)));
        pixels[p] = COLOR_LUT[level * 3]; pixels[p + 1] = COLOR_LUT[level * 3 + 1]; pixels[p + 2] = COLOR_LUT[level * 3 + 2];
      } else {
        pixels[p] = COLORS.surface[0]; pixels[p + 1] = COLORS.surface[1]; pixels[p + 2] = COLORS.surface[2];
      }
      pixels[p + 3] = 255;
    }
  }
  heatCtx.putImageData(heatImage, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(heatCanvas, MARGIN.left, MARGIN.top, plotW, plotH);

  // Axes: subcarrier bins on the left, session seconds below.
  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "right";
  ctx.fillText(String(bins[K - 1]), MARGIN.left - 5, MARGIN.top + 6);
  ctx.fillText(String(bins[K >> 1]), MARGIN.left - 5, MARGIN.top + plotH / 2);
  ctx.fillText(String(bins[0]), MARGIN.left - 5, MARGIN.top + plotH - 6);
  ctx.textAlign = "center";
  const step = [0.5, 1, 2, 5, 10, 20, 30, 60].find((s) => (s / span) * plotW >= 70) || 120;
  for (let t = Math.ceil(start / step) * step; t <= start + span; t += step) {
    ctx.fillText(`${t.toFixed(step < 1 ? 1 : 0)} s`, MARGIN.left + ((t - start) / span) * plotW, height - 8);
  }
  ctx.fillStyle = COLORS.cursor;
  ctx.fillRect(MARGIN.left + plotW / 2 - 0.75, MARGIN.top, 1.5, plotH);
}

$("waterfall").addEventListener("mousemove", (event) => {
  const box = event.target.getBoundingClientRect();
  const plotW = box.width - MARGIN.left - MARGIN.right, plotH = box.height - MARGIN.top - MARGIN.bottom;
  const fx = (event.clientX - box.left - MARGIN.left) / plotW, fy = (event.clientY - box.top - MARGIN.top) / plotH;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) { $("hoverWaterfall").textContent = ""; return; }
  const t = cursorTime() - state.windowS + fx * state.windowS * 2;
  const k = Math.min(K - 1, Math.floor((1 - fy) * K));
  const value = currentAmp()[nearest(csiT, t - state.offsetS) * K + k];
  $("hoverWaterfall").textContent = `${t.toFixed(2)} s · bin ${bins[k]} · ${value.toFixed(state.normalized ? 2 : 1)}`;
});
$("waterfall").addEventListener("mouseleave", () => { $("hoverWaterfall").textContent = ""; });

// --- Current frame ---------------------------------------------------------------
function traceFrame(ctx, index, toX, toY, amp) {
  ctx.beginPath();
  for (let k = 0; k < K; k++) {
    const x = toX(bins[k]), y = toY(amp[index * K + k]);
    // A jump between bins is a removed null subcarrier: leave a visible break.
    if (k === 0 || bins[k] !== bins[k - 1] + 1) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawFrame() {
  const { ctx, width, height } = prepareCanvas($("frame"));
  const plotW = width - MARGIN.left - MARGIN.right, plotH = height - MARGIN.top - MARGIN.bottom;
  const amp = currentAmp();
  const yMax = currentRange()[1] * 1.15;
  const yMin = state.normalized ? Math.min(0, currentRange()[0]) : 0;
  const firstBin = bins[0], lastBin = bins[K - 1];
  const toX = (bin) => MARGIN.left + ((bin - firstBin) / Math.max(1, lastBin - firstBin)) * plotW;
  const toY = (value) => MARGIN.top + plotH - ((Math.min(yMax, value) - yMin) / (yMax - yMin)) * plotH;

  ctx.strokeStyle = COLORS.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(MARGIN.left, MARGIN.top + plotH + 0.5);
  ctx.lineTo(MARGIN.left + plotW, MARGIN.top + plotH + 0.5);
  ctx.stroke();

  // History: each earlier frame in the last 0.5 s, drawn faintly.
  ctx.strokeStyle = "rgba(57, 135, 229, 0.14)";
  const t = csiT[state.csi];
  for (let i = state.csi - 1; i >= 0 && t - csiT[i] <= 0.5; i--) traceFrame(ctx, i, toX, toY, amp);

  ctx.strokeStyle = COLORS.series;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  traceFrame(ctx, state.csi, toX, toY, amp);

  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "right";
  ctx.fillText(yMax.toFixed(state.normalized ? 1 : 0), MARGIN.left - 5, MARGIN.top + 6);
  ctx.fillText(yMin.toFixed(state.normalized ? 1 : 0), MARGIN.left - 5, MARGIN.top + plotH - 6);
  ctx.textAlign = "center";
  const middleBin = Math.round((firstBin + lastBin) / 2);
  for (const bin of [firstBin, middleBin, lastBin]) ctx.fillText(`bin ${bin}`, Math.max(MARGIN.left + 16, Math.min(width - 24, toX(bin))), height - 8);
}

$("frame").addEventListener("mousemove", (event) => {
  const box = event.target.getBoundingClientRect();
  const fx = (event.clientX - box.left - MARGIN.left) / (box.width - MARGIN.left - MARGIN.right);
  if (fx < 0 || fx > 1) { $("hoverFrame").textContent = ""; return; }
  const k = nearest(Float64Array.from(bins), bins[0] + fx * (bins[K - 1] - bins[0]));
  $("hoverFrame").textContent = `bin ${bins[k]} · ${currentAmp()[state.csi * K + k].toFixed(state.normalized ? 2 : 1)}`;
});
$("frame").addEventListener("mouseleave", () => { $("hoverFrame").textContent = ""; });

// --- Motion and RSSI: two charts sharing one time axis ---------------------------
function drawCursorLine(plot) {
  const x = Math.round(plot.valToPos(cursorTime(), "x", true));
  const ctx = plot.ctx;
  ctx.save();
  ctx.fillStyle = COLORS.cursor;
  ctx.fillRect(x - 1, plot.bbox.top, 2, plot.bbox.height);
  ctx.restore();
}

function makeChart(element, values, range, hoverId, digits) {
  const axis = { stroke: COLORS.muted, grid: { stroke: COLORS.line, width: 1 }, ticks: { show: false }, font: "11px system-ui" };
  return new uPlot({
    width: element.clientWidth, height: element.clientHeight,
    padding: [4, 8, 0, 0],
    legend: { show: false },
    cursor: { y: false, points: { show: false }, drag: { x: false, y: false, setScale: false } },
    scales: { x: { time: false, auto: false }, y: { auto: false, range } },
    axes: [
      { ...axis, size: 18, values: (u, ticks) => ticks.map((t) => `${t} s`) },
      { ...axis, size: 42, grid: { show: false }, values: (u, ticks) => ticks.map((t) => Number(t.toPrecision(3))) },
    ],
    series: [{}, { stroke: COLORS.series, width: 1.5, points: { show: false } }],
    hooks: {
      draw: [drawCursorLine],
      setCursor: [(plot) => {
        const i = plot.cursor.idx;
        $(hoverId).textContent = i == null || values[i] == null ? "" : `${chartX[i].toFixed(2)} s · ${values[i].toFixed(digits)}`;
      }],
    },
  }, [chartX, values], element);
}

function updateChartWindows() {
  if (state.mode === "frequency") { updateBandWindow(); return; }
  if (state.mode !== "time") return;
  const center = cursorTime();
  for (const plot of [motionPlot, rssiPlot]) plot.setScale("x", { min: center - state.windowS, max: center + state.windowS });
}

function applyOffsetToCharts() {
  chartX = Float64Array.from(chartT, (t) => t + state.offsetS);
  motionPlot.setData([chartX, chartMotion], false);
  rssiPlot.setData([chartX, chartRssi], false);
  applyFrequencyOffset();
}

// A hidden chart measures zero, so only the charts on screen can be resized.
function resizeVisibleCharts() {
  if (state.mode === "frequency") { resizeBandPlot(); return; }
  if (state.mode !== "time") return;
  for (const [plot, id] of [[motionPlot, "chartMotion"], [rssiPlot, "chartRssi"]]) {
    const element = $(id);
    plot.setSize({ width: element.clientWidth, height: element.clientHeight });
  }
}

// --- Rendering --------------------------------------------------------------------
function updateInfo() {
  const frameTime = frameT[state.frame], csiTime = csiT[state.csi];
  $("infoVideo").textContent = `frame ${state.frame + 1} / ${meta.n_frames} · ${frameTime.toFixed(3)} s`;
  $("infoCsi").textContent = `frame ${state.csi + 1} / ${meta.n_csi} · ${csiTime.toFixed(3)} s`;
  $("infoDelta").textContent = `${((csiTime + state.offsetS - frameTime) * 1000).toFixed(1)} ms`;
  $("infoRssi").textContent = `${csiRssi[state.csi].toFixed(0)} dBm`;
  if (document.activeElement !== $("timeline")) $("timeline").value = state.frame;
}

function render() {
  const started = performance.now();
  state.renderQueued = false;
  if (state.mode === "frequency") {
    drawFrequencyPanels();
  } else if (state.mode === "time") {
    drawWaterfall();
    drawFrame();
  }
  updateChartWindows();
  updateDetails();
  updateInfo();
  renderTimes.push(performance.now() - started);
  if (renderTimes.length > 600) renderTimes.shift();
}

function queueRender() {
  if (state.renderQueued) return;
  state.renderQueued = true;
  requestAnimationFrame(render);
}

// --- Video to CSI mapping ----------------------------------------------------------
function syncCsiToFrame() {
  state.csi = state.pinnedCsi !== null ? state.pinnedCsi : nearest(csiT, frameT[state.frame] - state.offsetS);
}

function showMediaTime(mediaTime) {
  state.frame = nearest(framePts, mediaTime);
  syncCsiToFrame();
}

// Called by the browser for every video frame it presents, with its exact time.
function onVideoFrame(now, metadata) {
  showMediaTime(metadata.mediaTime);
  // An earlier seek can present a frame while a newer seek is still running.
  if (state.pendingFrame === state.frame) state.pendingFrame = null;
  render();  // drawn now so the charts appear together with this video frame
  video.requestVideoFrameCallback(onVideoFrame);
}

function seekToFrame(index) {
  const frame = Math.max(0, Math.min(meta.n_frames - 1, index));
  state.pendingFrame = frame;
  // A small nudge past the frame's start guards against landing on the frame before it.
  video.currentTime = framePts[frame] + 0.002;
}

function stepVideo(delta) {
  video.pause();
  state.pinnedCsi = null;
  seekToFrame((state.pendingFrame !== null ? state.pendingFrame : state.frame) + delta);
}

function stepCsi(delta) {
  video.pause();
  const next = Math.max(0, Math.min(meta.n_csi - 1, state.csi + delta));
  state.pinnedCsi = next;
  state.csi = next;
  const frame = nearest(frameT, csiT[next] + state.offsetS);
  if (frame !== state.frame) seekToFrame(frame);
  queueRender();
}

function togglePlay() {
  if (video.paused) {
    state.pinnedCsi = null;
    video.play();
  } else {
    video.pause();
  }
}

// --- View mode and saved settings ---------------------------------------------------
function setMode(mode, save) {
  state.mode = mode;
  for (const [name, ids] of Object.entries(MODE_PANELS)) {
    for (const id of ids) $(id).hidden = name !== mode;
    $(`mode_${name}`).setAttribute("aria-pressed", String(name === mode));
  }
  $("normalizedLabel").hidden = mode !== "time";  // that checkbox only affects the time panels
  $("right").dataset.mode = mode;
  resizeVisibleCharts();
  if (mode === "frequency") showFrequency();
  queueRender();
  if (save) savePlayerSettings();
}

function setDetailsPanel(show, save) {
  state.detailsPanel = show;
  $("detailsToggle").checked = show;
  $("detailsPanel").hidden = !show;
  queueRender();
  if (save) savePlayerSettings();
}

let settingsTimer = null;
function savePlayerSettings() {
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(() => {
    fetch("api/player_settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: state.mode, details_panel: state.detailsPanel, frequency: fq.settings }),
    }).catch(() => {});  // settings are a convenience: playback must never depend on them
  }, 300);
}

// --- Controls ----------------------------------------------------------------------
let saveTimer = null;
function setOffset(ms, save) {
  state.offsetS = ms / 1000;
  $("offset").value = ms;
  $("offsetValue").textContent = `${ms >= 0 ? "+" : ""}${ms.toFixed(0)} ms`;
  applyOffsetToCharts();
  syncCsiToFrame();
  queueRender();
  if (!save) return;
  $("offsetSaved").textContent = "";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try {
      const response = await fetch("api/sync_offset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sync_offset_ms: ms }),
      });
      $("offsetSaved").textContent = response.ok ? "saved" : "not saved";
    } catch {
      $("offsetSaved").textContent = "not saved";
    }
  }, 300);
}

function bindControls() {
  $("timeline").max = meta.n_frames - 1;
  $("play").onclick = togglePlay;
  video.onclick = togglePlay;
  video.onplay = () => { $("play").textContent = "Pause"; };
  video.onpause = () => { $("play").textContent = "Play"; };
  video.onended = () => { $("play").textContent = "Play"; };
  video.onerror = () => { $("infoStatus").textContent = "The browser could not play video.mp4"; };
  video.addEventListener("seeked", () => {
    if (video.seeking) return;  // a newer seek is still running
    // A seek landing on the frame already shown produces no new video frame callback.
    state.pendingFrame = null;
    showMediaTime(video.currentTime);
    queueRender();
  });

  $("timeline").addEventListener("input", (event) => {
    state.pinnedCsi = null;
    seekToFrame(Number(event.target.value));
  });
  $("timeline").addEventListener("change", (event) => event.target.blur());
  $("speed").onchange = (event) => { video.playbackRate = Number(event.target.value); event.target.blur(); };
  $("window").onchange = (event) => { state.windowS = Number(event.target.value); event.target.blur(); queueRender(); };
  $("normalized").onchange = (event) => { state.normalized = event.target.checked; event.target.blur(); queueRender(); };
  $("offset").addEventListener("input", (event) => setOffset(Number(event.target.value), true));
  $("offset").addEventListener("change", (event) => event.target.blur());
  for (const name of Object.keys(MODE_PANELS)) {
    $(`mode_${name}`).onclick = (event) => { event.target.blur(); setMode(name, true); };
  }
  $("detailsToggle").onchange = (event) => { event.target.blur(); setDetailsPanel(event.target.checked, true); };

  document.addEventListener("keydown", (event) => {
    const handlers = {
      " ": togglePlay,
      ArrowLeft: () => (event.shiftKey ? stepCsi(-1) : stepVideo(-1)),
      ArrowRight: () => (event.shiftKey ? stepCsi(1) : stepVideo(1)),
    };
    const handler = handlers[event.key];
    if (!handler) return;
    event.preventDefault();  // keeps sliders and the page from reacting too
    handler();
  }, true);

  // Space on a focused button would otherwise press it as well as play.
  document.addEventListener("keyup", (event) => { if (event.key === " ") event.preventDefault(); }, true);

  new ResizeObserver(() => {
    resizeVisibleCharts();
    queueRender();
  }).observe($("right"));
}

async function start() {
  $("infoStatus").textContent = "Loading CSI...";
  meta = await (await fetch("api/meta")).json();
  [frameT, framePts, csiT, csiRssi, ampRaw, ampNorm, chartT, chartMotion, chartRssi] = await Promise.all(
    ["frame_t", "frame_pts", "csi_t", "csi_rssi", "amp_raw", "amp_norm", "chart_t", "chart_motion", "chart_rssi"].map(fetchArray),
  );
  bins = meta.kept_bins;
  K = bins.length;
  state.offsetS = meta.sync_offset_ms / 1000;
  chartX = Float64Array.from(chartT, (t) => t + state.offsetS);

  motionPlot = makeChart($("chartMotion"), chartMotion, [0, meta.motion_max], "hoverMotion", 4);
  rssiPlot = makeChart($("chartRssi"), chartRssi, meta.rssi_range, "hoverRssi", 0);
  bindControls();
  setOffset(meta.sync_offset_ms, false);
  initFrequency();
  initDetails();
  const saved = meta.player_settings || {};
  setDetailsPanel(saved.details_panel === true, false);
  setMode(saved.mode in MODE_PANELS ? saved.mode : "time", false);

  const fit = meta.summary.fit_ok ? `clock fit residual ${meta.summary.fit_residual_rms_ms.toFixed(2)} ms` : "clock fit unreliable";
  $("infoStatus").textContent = fit;
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(onVideoFrame);
  } else {
    $("infoStatus").textContent = "This browser lacks requestVideoFrameCallback; use Chrome or Edge for exact frames";
    video.addEventListener("timeupdate", () => { showMediaTime(video.currentTime); queueRender(); });
  }
  queueRender();
}

start().catch((error) => { $("infoStatus").textContent = `Error: ${error.message}`; });
