// ESPectre detector panel (Time mode). Shows the offline replay of ESPectre's own
// Lightweight and High-Accuracy detectors, computed once by analyze.py: the score
// and threshold of each evaluation (both are probabilities from 0 to 1), and a strip
// per detector with its IDLE / MOTION state after the board's on/off hit counts.
// Times are CSI time; the sync offset is added when drawing, like the motion chart.
// Loaded before app.js and uses its state and helpers, only from inside functions.
"use strict";

const ESP_DETECTORS = [
  { name: "lightweight", label: "Lightweight", short: "LW", color: "#e08a3c" },
  { name: "high_accuracy", label: "High-Accuracy", short: "HA", color: "#b48ae8" },
];
const ESP_STATE_TEXT = ["not ready", "idle", "MOTION"];
const ESP_IDLE_COLOR = "#4a4a46";
const ESP_Y_RANGE = [-0.24, 1.02];  // the band below 0 holds the state strips

const esp = { available: false, t: null, x: null, list: [], plot: null };

async function loadEspectre() {
  const detectors = meta.espectre && meta.espectre.detectors;
  if (!detectors) return;
  esp.t = await fetchArray("esp_t");
  for (const spec of ESP_DETECTORS) {
    if (!(spec.name in detectors)) continue;
    const [score, threshold, stateCode] = await Promise.all(
      ["score", "threshold", "state"].map((part) => fetchArray(`esp_${spec.name}_${part}`)),
    );
    esp.list.push({ ...spec, score, threshold, stateCode, info: detectors[spec.name] });
  }
  esp.available = esp.t.length > 0 && esp.list.length > 0;
}

// Index of the last evaluation at or before time t (CSI time), or -1.
function espIndexAt(t) {
  if (!esp.available || t < esp.t[0]) return -1;
  let low = 0, high = esp.t.length - 1;
  while (low < high) {
    const mid = (low + high + 1) >> 1;
    if (esp.t[mid] <= t) low = mid; else high = mid - 1;
  }
  return low;
}

function espectreInfoText() {
  if (!esp.available) return "--";
  const i = espIndexAt(csiT[state.csi]);
  if (i < 0) return "before the first evaluation";
  return esp.list.map((d) => `${d.short} ${ESP_STATE_TEXT[d.stateCode[i]]} ${d.score[i].toFixed(2)}`).join(" · ");
}

// State strips under the score lines: detector color for MOTION, grey for idle, blank when not ready.
function drawEspectreStrips(plot) {
  const ctx = plot.ctx, box = plot.bbox;
  const top = plot.valToPos(-0.03, "y", true), bottom = box.top + box.height;
  const gap = Math.max(2, Math.round((bottom - top) * 0.12));
  const rowH = (bottom - top - gap * (esp.list.length - 1)) / esp.list.length;
  const min = plot.scales.x.min - state.offsetS, max = plot.scales.x.max - state.offsetS;
  const first = Math.max(0, espIndexAt(min)), last = espIndexAt(max);
  ctx.save();
  esp.list.forEach((d, row) => {
    const y = top + row * (rowH + gap);
    for (let i = first; i <= last; i++) {
      const code = d.stateCode[i];
      if (code === 0) continue;
      const x1 = plot.valToPos(esp.x[i], "x", true);
      const x2 = plot.valToPos(i + 1 < esp.x.length ? esp.x[i + 1] : esp.x[i] + 0.25, "x", true);
      ctx.fillStyle = code === 2 ? d.color : ESP_IDLE_COLOR;
      ctx.fillRect(x1, y, Math.max(1, x2 - x1), rowH);
    }
  });
  ctx.restore();
}

function espectreHover(plot) {
  const i = plot.cursor.idx;
  if (i == null || !esp.available) { $("hoverEspectre").textContent = ""; return; }
  const parts = esp.list.map((d) => `${d.short} ${d.score[i].toFixed(2)} / ${d.threshold[i].toFixed(2)} ${ESP_STATE_TEXT[d.stateCode[i]]}`);
  $("hoverEspectre").textContent = `${esp.x[i].toFixed(2)} s · ${parts.join(" · ")}`;
}

function espectreData() {
  const data = [esp.x];
  for (const d of esp.list) data.push(d.score, d.threshold);
  return data;
}

function buildEspectreLegend() {
  const legend = $("espLegend");
  legend.replaceChildren();
  for (const d of esp.list) {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = d.color;
    const threshold = d.info.start_threshold.toFixed(2);
    item.title = `${d.info.n_evaluations} evaluations; hits ${d.info.motion_on_hits} on / ${d.info.motion_off_hits} off`;
    item.append(swatch, `${d.label}: threshold ${threshold} (${d.info.threshold_source}), dashed line`);
    legend.append(item);
  }
  const board = esp.list[0].info.board_detector;
  if (board) {
    const note = document.createElement("span");
    note.textContent = `board ran ${board}`;
    legend.append(note);
  }
}

function initEspectre() {
  if (!esp.available) {
    $("espectreLabel").hidden = true;
    return;
  }
  esp.x = Float64Array.from(esp.t, (t) => t + state.offsetS);
  buildEspectreLegend();
  const element = $("chartEspectre");
  const axis = { stroke: COLORS.muted, grid: { stroke: COLORS.line, width: 1 }, ticks: { show: false }, font: "11px system-ui" };
  const series = [{}];
  for (const d of esp.list) {
    series.push({ stroke: d.color, width: 1.5, points: { show: false } });
    series.push({ stroke: d.color, width: 1, dash: [4, 4], points: { show: false } });
  }
  esp.plot = new uPlot({
    width: element.clientWidth, height: element.clientHeight,
    padding: [4, 8, 0, 0],
    legend: { show: false },
    cursor: { y: false, points: { show: false }, drag: { x: false, y: false, setScale: false } },
    scales: { x: { time: false, auto: false }, y: { auto: false, range: ESP_Y_RANGE } },
    axes: [
      { ...axis, size: 18, values: (u, ticks) => ticks.map((t) => `${t} s`) },
      { ...axis, size: 42, grid: { show: false }, splits: () => [0, 0.5, 1], values: (u, ticks) => ticks.map(String) },
    ],
    series,
    hooks: { draw: [drawEspectreStrips, drawCursorLine], setCursor: [espectreHover] },
  }, espectreData(), element);
}

function espectreVisible() {
  return esp.plot !== null && !$("panelEspectre").hidden;
}

function applyEspectreOffset() {
  if (!esp.plot) return;
  esp.x = Float64Array.from(esp.t, (t) => t + state.offsetS);
  esp.plot.setData(espectreData(), false);
}

function updateEspectreWindow(center) {
  if (espectreVisible()) esp.plot.setScale("x", { min: center - state.windowS, max: center + state.windowS });
}

function resizeEspectrePlot() {
  if (!espectreVisible()) return;
  const element = $("chartEspectre");
  esp.plot.setSize({ width: element.clientWidth, height: element.clientHeight });
}
