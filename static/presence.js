// Presence view: three slow features that may tell a still person from an empty room,
// computed once by analyze.py (frequency.compute_presence). They change slowly, so the
// charts show the whole recording with the video cursor, and a click jumps the video there.
// The empty-room baseline is chosen here and saved by the server, which also compares
// every point with it. Times are CSI time; the sync offset is added when drawing.
// Loaded before app.js and uses its state and helpers, only from inside functions.
"use strict";

const PRES_SERIES = "#3987e5";
const PRES_REFERENCE = "#a8a7a1";
const PRES_BASELINE_FILL = "rgba(25, 158, 112, 0.18)";
const PRES_BASELINE_TEXT = "#5fc49b";

const pres = {
  loaded: false, loading: false, error: "",
  t: null, breathing: null, breathingHz: null, level: null,
  baseline: null, distance: null, floor: null,
  plots: {},
};

// uPlot breaks a line at null, which is how a point without a full window is shown.
function nullable(values) {
  return Array.from(values, (v) => (v == null || Number.isNaN(v) ? null : v));
}

function presRange(values, fallback) {
  let low = Infinity, high = -Infinity;
  for (const v of values) if (v != null && Number.isFinite(v)) { low = Math.min(low, v); high = Math.max(high, v); }
  return Number.isFinite(low) ? [low, high] : fallback;
}

// --- Loading ----------------------------------------------------------------------
async function loadPresence() {
  if (pres.loaded || pres.loading) return;
  if (!meta.presence) {
    pres.error = "This session has no presence results. Restart the player with --recompute.";
    setPresenceStatus();
    return;
  }
  pres.loading = true;
  setPresenceStatus();
  try {
    [pres.t, pres.breathing, pres.breathingHz, pres.level] = await Promise.all(
      ["pres_t", "pres_breathing", "pres_breathing_hz", "pres_level"].map(fetchArray),
    );
    applyBaseline(await (await fetch("api/presence/baseline")).json());
    pres.loaded = true;
  } catch (error) {
    pres.error = `Could not load the presence results: ${error.message}`;
  } finally {
    pres.loading = false;
    setPresenceStatus();
    buildPresencePlots();
    queueRender();
  }
}

function applyBaseline(result) {
  pres.baseline = result.baseline;
  pres.distance = result.distance;
  pres.floor = result.level_floor;
}

async function sendBaseline(body) {
  $("presStatus").textContent = "Saving...";
  try {
    const response = await fetch("api/presence/baseline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "The server rejected this stretch");
    applyBaseline(result);
    setPresenceStatus();
    buildPresencePlots();
    queueRender();
  } catch (error) {
    $("presStatus").textContent = `Not saved: ${error.message}`;
  }
}

function setPresenceStatus() {
  const status = $("presStatus");
  if (pres.error) { status.textContent = pres.error; return; }
  if (pres.loading) { status.textContent = "Loading..."; return; }
  if (!pres.baseline) {
    status.textContent = "No empty-room baseline yet: move the video to a moment when the room is empty, then click the button.";
    return;
  }
  const [start, end] = pres.baseline.map((t) => (t + state.offsetS).toFixed(1));
  status.textContent = `Empty baseline: ${start} to ${end} s (green band). Click a chart to jump the video there.`;
}

// --- Charts -------------------------------------------------------------------------
function drawBaselineBand(plot) {
  if (!pres.baseline) return;
  const ctx = plot.ctx, box = plot.bbox, ratio = window.devicePixelRatio || 1;
  const x1 = plot.valToPos(pres.baseline[0] + state.offsetS, "x", true);
  const x2 = plot.valToPos(pres.baseline[1] + state.offsetS, "x", true);
  ctx.save();
  ctx.fillStyle = PRES_BASELINE_FILL;
  ctx.fillRect(x1, box.top, Math.max(1, x2 - x1), box.height);
  ctx.fillStyle = PRES_BASELINE_TEXT;
  ctx.font = `${11 * ratio}px system-ui, 'Segoe UI', sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("empty baseline", (x1 + x2) / 2, box.top + 4 * ratio);
  ctx.restore();
}

function drawPlotMessage(text) {
  return (plot) => {
    const ctx = plot.ctx, box = plot.bbox, ratio = window.devicePixelRatio || 1;
    ctx.save();
    ctx.fillStyle = COLORS.muted;
    ctx.font = `${12 * ratio}px system-ui, 'Segoe UI', sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, box.left + box.width / 2, box.top + box.height / 2);
    ctx.restore();
  };
}

function presenceX() {
  return Float64Array.from(pres.t, (t) => t + state.offsetS);
}

function makePresencePlot(id, { series, data, y, logValues, hover, extraDraw }) {
  const element = $(id);
  const axis = { stroke: COLORS.muted, grid: { stroke: COLORS.line, width: 1 }, ticks: { show: false }, font: "11px system-ui" };
  // A log chart plots log10 of its values on a plain axis and labels the ticks with the real values.
  const yValues = logValues
    ? (u, ticks) => ticks.map((t) => (10 ** t).toExponential(1))
    : (u, ticks) => ticks.map((t) => Number(t.toPrecision(3)));
  const plot = new uPlot({
    width: element.clientWidth, height: element.clientHeight,
    padding: [4, 8, 0, 0],
    legend: { show: false },
    cursor: { y: false, points: { show: false }, drag: { x: false, y: false, setScale: false } },
    scales: { x: { time: false, auto: false }, y: { auto: false, range: y } },
    axes: [
      { ...axis, size: 18, values: (u, ticks) => ticks.map((t) => `${t} s`) },
      { ...axis, size: 46, grid: { show: false }, values: yValues },
    ],
    series: [{}, ...series],
    hooks: {
      draw: [drawBaselineBand, ...(extraDraw ? [extraDraw] : []), drawCursorLine],
      setCursor: [(u) => {
        const i = u.cursor.idx;
        $(hover.id).textContent = i == null ? "" : `${u.data[0][i].toFixed(1)} s · ${hover.text(i)}`;
      }],
    },
  }, data, element);
  // Click a chart to jump the video to that moment.
  element.onclick = () => {
    if (plot.cursor.left == null || plot.cursor.left < 0) return;
    state.pinnedCsi = null;
    seekToFrame(nearest(frameT, plot.posToVal(plot.cursor.left, "x")));
  };
  plot.setScale("x", presenceXRange());
  return plot;
}

function presenceXRange() {
  return { min: pres.t[0] + state.offsetS, max: pres.t[pres.t.length - 1] + state.offsetS };
}

function buildPresencePlots() {
  for (const plot of Object.values(pres.plots)) plot.destroy();
  pres.plots = {};
  if (!pres.loaded || state.mode !== "presence") return;  // a hidden chart would measure zero
  const x = presenceX(), info = meta.presence;
  const fmtShare = (v) => (v == null ? "--" : v.toFixed(3));

  const breathing = nullable(pres.breathing);
  const flat = new Array(x.length).fill(info.flat_share);
  pres.plots.breathing = makePresencePlot("chartBreathing", {
    series: [
      { stroke: PRES_SERIES, width: 1.6, points: { show: false } },
      { stroke: PRES_REFERENCE, width: 1, dash: [4, 4], points: { show: false } },
    ],
    data: [x, breathing, flat],
    y: [0, Math.max(0.3, presRange(breathing, [0, 0.3])[1] * 1.1)],
    hover: {
      id: "hoverBreathing",
      text: (i) => {
        const hz = pres.breathingHz[i];
        const rate = Number.isNaN(hz) ? "" : ` · peak ${(hz * 60).toFixed(1)} per min (${hz.toFixed(2)} Hz)`;
        return `share ${fmtShare(breathing[i])}${rate}`;
      },
    },
  });

  const distance = pres.distance ? nullable(pres.distance) : new Array(x.length).fill(null);
  pres.plots.shape = makePresencePlot("chartShape", {
    series: [{ stroke: PRES_SERIES, width: 1.6, points: { show: false } }],
    data: [x, distance],
    y: [0, Math.max(1, presRange(distance, [0, 1])[1] * 1.1)],
    hover: { id: "hoverShape", text: (i) => (distance[i] == null ? "no value" : `${distance[i].toFixed(2)} % from the baseline`) },
    extraDraw: pres.baseline ? null : drawPlotMessage("Choose an empty-room baseline to see this chart"),
  });

  const level = nullable(pres.level).map((v) => (v != null && v > 0 ? v : null));  // a log needs > 0
  const logLevel = level.map((v) => (v == null ? null : Math.log10(v)));
  const [low, high] = presRange(logLevel, [-5, -1]);
  const floorLog = pres.floor != null && pres.floor > 0 ? Math.log10(pres.floor) : null;
  pres.plots.level = makePresencePlot("chartLevel", {
    series: [
      { stroke: PRES_SERIES, width: 1.6, points: { show: false } },
      { stroke: PRES_REFERENCE, width: 1, dash: [4, 4], points: { show: false } },
    ],
    data: [x, logLevel, new Array(x.length).fill(floorLog)],
    y: [low - 0.2, high + 0.2],
    logValues: true,
    hover: {
      id: "hoverLevel",
      text: (i) => {
        if (level[i] == null) return "no value";
        const ratio = pres.floor ? ` · ${(level[i] / pres.floor).toFixed(2)}× the baseline floor` : "";
        return `${level[i].toExponential(2)}${ratio}`;
      },
    },
  });
  $("breathingNote").textContent = `dashed: no peak at all (${info.flat_share.toFixed(3)})`;
  $("levelNote").textContent = pres.floor ? `dashed: the baseline's ${info.floor_percentile}th percentile` : "";
}

// --- Hooks called by app.js -----------------------------------------------------------
function initPresence() {
  $("presBaselineSet").onclick = (event) => {
    event.target.blur();
    if (!pres.loaded) return;
    const center = cursorTime() - state.offsetS, half = meta.presence.baseline_window_s / 2;
    sendBaseline({ start_s: center - half, end_s: center + half });
  };
  $("presBaselineClear").onclick = (event) => { event.target.blur(); sendBaseline({}); };
  $("presBaselineSet").title = meta.presence
    ? `Uses the ${meta.presence.baseline_window_s} s centered on the current video frame`
    : "";
}

function showPresence() {
  if (!pres.loaded) { loadPresence(); return; }
  setPresenceStatus();
  buildPresencePlots();
}

function presencePlotsVisible() {
  return state.mode === "presence" && Object.keys(pres.plots).length > 0;
}

// Redraws the cursor line; the data and axes stay as they are.
function updatePresence() {
  if (!presencePlotsVisible()) return;
  for (const plot of Object.values(pres.plots)) plot.redraw(false, false);
}

function applyPresenceOffset() {
  if (!presencePlotsVisible()) return;
  setPresenceStatus();
  buildPresencePlots();
}

function resizePresencePlots() {
  if (!presencePlotsVisible()) return;
  for (const [name, id] of [["breathing", "chartBreathing"], ["shape", "chartShape"], ["level", "chartLevel"]]) {
    const element = $(id);
    pres.plots[name].setSize({ width: element.clientWidth, height: element.clientHeight });
  }
}
