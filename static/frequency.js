// Frequency mode: spectrogram, current spectrum, and band power.
// Loaded before app.js and uses its state and helpers, only from inside functions.
// Column times are CSI time: adding state.offsetS places them on the video's clock.
"use strict";

const BAND_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const MISSING_GREY = [72, 72, 69];
const TIME_PANEL_IDS = ["panelWaterfall", "panelFrame", "panelMotion"];
const FREQUENCY_PANEL_IDS = ["freqControls", "panelSpectrogram", "panelSpectrum", "panelBands"];
const MAX_HELD_BYTES = 256 * 1024 * 1024;  // results the page keeps for instant switching
const MAX_RANGE_SAMPLES = 200000;
const BUSY_DELAY_MS = 120;

const fq = {
  settings: { window_s: 2, hop_s: 0.1, method: "mean", freq_max_hz: 50, color_auto: true, color_min: -60, color_max: 0 },
  data: null, held: new Map(), heldBytes: 0, requestId: 0,
  colorLimits: [-60, 0], spectrumRange: [-60, 0],
  bandPlot: null, bandX: null, bandValues: [], saveTimer: null,
  heatCanvas: document.createElement("canvas"), heatImage: null,
};

// --- Settings and mode -----------------------------------------------------------
function restoreSelect(id, value) {
  const select = $(id);
  if ([...select.options].some((option) => option.value === String(value))) select.value = String(value);
}

function readControls() {
  Object.assign(fq.settings, {
    window_s: Number($("freqWindow").value),
    hop_s: Number($("freqHop").value),
    method: $("freqMethod").value,
    freq_max_hz: Number($("freqRange").value),
    color_auto: $("colorAuto").checked,
    color_min: Number($("colorMin").value),
    color_max: Number($("colorMax").value),
  });
  $("colorMin").disabled = $("colorMax").disabled = fq.settings.color_auto;
}

function saveSettings() {
  clearTimeout(fq.saveTimer);
  fq.saveTimer = setTimeout(() => {
    fetch("api/player_settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: state.mode, frequency: fq.settings }),
    }).catch(() => {});  // settings are a convenience: playback must never depend on them
  }, 300);
}

function setMode(mode, save) {
  state.mode = mode;
  const frequency = mode === "frequency";
  for (const id of TIME_PANEL_IDS) $(id).hidden = frequency;
  for (const id of FREQUENCY_PANEL_IDS) $(id).hidden = !frequency;
  $("normalizedLabel").hidden = frequency;  // that checkbox only affects the time panels
  $("right").classList.toggle("frequency", frequency);
  $("modeTime").setAttribute("aria-pressed", String(!frequency));
  $("modeFrequency").setAttribute("aria-pressed", String(frequency));
  resizeVisibleCharts();
  if (frequency && !fq.data) requestFrequency();
  queueRender();
  if (save) saveSettings();
}

function initFrequency() {
  const saved = meta.player_settings || {};
  const stored = saved.frequency || {};
  restoreSelect("freqWindow", stored.window_s);
  restoreSelect("freqHop", stored.hop_s);
  restoreSelect("freqMethod", stored.method);
  restoreSelect("freqRange", stored.freq_max_hz);
  if (typeof stored.color_auto === "boolean") $("colorAuto").checked = stored.color_auto;
  if (Number.isFinite(stored.color_min)) $("colorMin").value = stored.color_min;
  if (Number.isFinite(stored.color_max)) $("colorMax").value = stored.color_max;
  readControls();

  for (const id of ["freqWindow", "freqHop", "freqMethod"]) {
    $(id).onchange = (event) => { event.target.blur(); readControls(); requestFrequency(); saveSettings(); };
  }
  for (const id of ["freqRange", "colorMin", "colorMax"]) {
    $(id).onchange = (event) => { event.target.blur(); readControls(); updateDisplayRanges(); queueRender(); saveSettings(); };
  }
  $("colorAuto").onchange = (event) => {
    event.target.blur();
    if (!event.target.checked) {  // start manual editing from the limits now on screen
      $("colorMin").value = Math.round(fq.colorLimits[0]);
      $("colorMax").value = Math.round(fq.colorLimits[1]);
    }
    readControls();
    updateDisplayRanges();
    queueRender();
    saveSettings();
  };
  $("modeTime").onclick = (event) => { event.target.blur(); setMode("time", true); };
  $("modeFrequency").onclick = (event) => { event.target.blur(); setMode("frequency", true); };
  bindFrequencyHover();
  setMode(saved.mode === "frequency" ? "frequency" : "time", false);
}

// --- Getting results from the backend --------------------------------------------
function setFrequencyStatus(text, busy) {
  $("freqStatus").textContent = text;
  $("freqStatus").classList.toggle("busy", busy);
}

async function fetchFrequencyArray(key, name, ArrayType) {
  const response = await fetch(`api/frequency/${key}/${name}`);
  if (!response.ok) throw new Error(`Could not load ${name}`);
  return new ArrayType(await response.arrayBuffer());
}

function holdResult(heldKey, result) {
  const bytes = result.power.byteLength + result.band_power.byteLength + result.t.byteLength + result.freqs.byteLength;
  result.bytes = bytes;
  fq.held.set(heldKey, result);
  fq.heldBytes += bytes;
  for (const [oldKey, old] of fq.held) {
    if (fq.heldBytes <= MAX_HELD_BYTES || oldKey === heldKey) break;
    fq.held.delete(oldKey);  // a Map iterates oldest first
    fq.heldBytes -= old.bytes;
  }
}

async function requestFrequency() {
  const id = ++fq.requestId;
  const { window_s, hop_s, method } = fq.settings;
  const heldKey = `${window_s}/${hop_s}/${method}`;
  const held = fq.held.get(heldKey);
  if (held) {
    fq.held.delete(heldKey);  // moved to the newest position
    fq.held.set(heldKey, held);
    setFrequencyStatus("", false);
    installResult(held);
    return;
  }
  const busyTimer = setTimeout(() => setFrequencyStatus("Computing...", true), BUSY_DELAY_MS);
  try {
    const response = await fetch("api/frequency", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ window_s, hop_s, method }),
    });
    const info = await response.json();
    if (!response.ok) throw new Error(typeof info.detail === "string" ? info.detail : "The server rejected these settings");
    const [t, freqs, power, band_power] = await Promise.all([
      fetchFrequencyArray(info.key, "t", Float64Array),
      fetchFrequencyArray(info.key, "freqs", Float64Array),
      fetchFrequencyArray(info.key, "power", Float32Array),
      fetchFrequencyArray(info.key, "band_power", Float32Array),
    ]);
    const result = { ...info, t, freqs, power, band_power };
    holdResult(heldKey, result);
    if (id !== fq.requestId) return;  // the settings changed while this one was running
    setFrequencyStatus("", false);
    installResult(result);
  } catch (error) {
    if (id === fq.requestId) setFrequencyStatus(`Error: ${error.message}`, false);
  } finally {
    clearTimeout(busyTimer);
  }
}

function installResult(result) {
  fq.data = result;
  const skipped = result.n_skipped ? ` · ${result.n_skipped} of ${result.n_columns} windows skipped` : "";
  $("freqInfo").textContent = `Resolution ${formatNumber(result.df_hz)} Hz · Sample rate ${formatNumber(result.fs_hz)} Hz${skipped}`;
  updateDisplayRanges();
  buildBandPlot(result);
  queueRender();
}

// --- Display ranges ----------------------------------------------------------------
function formatNumber(value) { return String(Number(value.toPrecision(3))); }
function displayMaxHz(ds) { return Math.min(fq.settings.freq_max_hz, ds.fs_hz / 2); }
function binAt(ds, hz) { return Math.max(0, Math.min(ds.n_freqs - 1, Math.round(hz / ds.df_hz))); }

function updateDisplayRanges() {
  const ds = fq.data;
  if (!ds || !ds.n_columns) return;
  const lastBin = binAt(ds, displayMaxHz(ds));
  const firstBin = Math.min(1, lastBin);  // 0 Hz only holds what the mean removal left over
  const bins = lastBin - firstBin + 1;
  // Whole columns are sampled, so a regular step cannot favor one frequency.
  const columnStep = Math.max(1, Math.floor((ds.n_columns * bins) / MAX_RANGE_SAMPLES));
  const sample = new Float32Array(Math.ceil(ds.n_columns / columnStep) * bins);
  let count = 0, highest = -Infinity;
  for (let c = 0; c < ds.n_columns; c += columnStep) {
    for (let b = firstBin; b <= lastBin; b++) {
      const value = ds.power[c * ds.n_freqs + b];
      if (Number.isNaN(value)) continue;
      sample[count++] = value;
      if (value > highest) highest = value;
    }
  }
  const sorted = sample.subarray(0, count).sort();
  const quantile = (q) => (count ? sorted[Math.min(count - 1, Math.floor(q * count))] : 0);
  fq.spectrumRange = count ? [quantile(0.005) - 3, highest + 3] : [-60, 0];

  let limits = [quantile(0.02), quantile(0.995)];
  if (!fq.settings.color_auto) {
    const low = fq.settings.color_min;
    limits = [low, fq.settings.color_max > low ? fq.settings.color_max : low + 1];
  }
  fq.colorLimits = limits;
  $("colorReadout").textContent = `color ${limits[0].toFixed(0)} to ${limits[1].toFixed(0)} dB`;
}

// The STFT column for the current video frame (CSI time = cursor time - offset).
function cursorColumn(ds) { return nearest(ds.t, cursorTime() - state.offsetS); }

function frequencyTicks(maxHz) {
  const step = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20].find((s) => maxHz / s <= 5) || 50;
  const ticks = [];
  for (let n = 0; n * step <= maxHz + 1e-9; n++) ticks.push(n * step);
  return ticks;
}

function drawCanvasMessage(ctx, width, height, text) {
  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
}

function emptyMessage() {
  if (!fq.data) return "Computing...";
  return `This recording is shorter than the ${formatNumber(fq.data.window_s)} s window`;
}

// --- Spectrogram -------------------------------------------------------------------
function drawSpectrogram() {
  const { ctx, width, height } = prepareCanvas($("spectrogram"));
  const ds = fq.data;
  if (!ds || !ds.n_columns) { drawCanvasMessage(ctx, width, height, emptyMessage()); return; }
  const plotW = Math.max(1, Math.floor(width - MARGIN.left - MARGIN.right));
  const plotH = Math.max(1, Math.floor(height - MARGIN.top - MARGIN.bottom));
  if (!fq.heatImage || fq.heatImage.width !== plotW || fq.heatImage.height !== plotH) {
    fq.heatCanvas.width = plotW;
    fq.heatCanvas.height = plotH;
    fq.heatImage = fq.heatCanvas.getContext("2d").createImageData(plotW, plotH);
  }
  const pixels = fq.heatImage.data, power = ds.power, nFreqs = ds.n_freqs;
  const [low, high] = fq.colorLimits, scale = 255 / Math.max(1e-6, high - low);
  const maxHz = displayMaxHz(ds), center = cursorTime(), span = state.windowS * 2, start = center - state.windowS;
  const maxDistance = Math.max(ds.hop_s * 0.51, span / plotW);

  // Each pixel row covers a range of bins and shows the strongest, so thin lines survive.
  const rowFirst = new Int32Array(plotH), rowLast = new Int32Array(plotH);
  for (let y = 0; y < plotH; y++) {
    const top = binAt(ds, maxHz * (1 - y / plotH)), bottom = binAt(ds, maxHz * (1 - (y + 1) / plotH));
    rowFirst[y] = bottom;
    rowLast[y] = Math.max(bottom, y === 0 ? top : top - 1);
  }

  for (let x = 0; x < plotW; x++) {
    const query = start + ((x + 0.5) / plotW) * span - state.offsetS;
    const column = nearest(ds.t, query);
    const inside = Math.abs(ds.t[column] - query) <= maxDistance;
    const missing = inside && Number.isNaN(power[column * nFreqs]);
    const offset = column * nFreqs;
    for (let y = 0; y < plotH; y++) {
      const p = (y * plotW + x) * 4;
      if (inside && !missing) {
        let value = power[offset + rowFirst[y]];
        for (let b = rowFirst[y] + 1; b <= rowLast[y]; b++) if (power[offset + b] > value) value = power[offset + b];
        const level = Math.max(0, Math.min(255, Math.round((value - low) * scale)));
        pixels[p] = COLOR_LUT[level * 3]; pixels[p + 1] = COLOR_LUT[level * 3 + 1]; pixels[p + 2] = COLOR_LUT[level * 3 + 2];
      } else {
        const fill = missing ? MISSING_GREY : COLORS.surface;
        pixels[p] = fill[0]; pixels[p + 1] = fill[1]; pixels[p + 2] = fill[2];
      }
      pixels[p + 3] = 255;
    }
  }
  fq.heatCanvas.getContext("2d").putImageData(fq.heatImage, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(fq.heatCanvas, MARGIN.left, MARGIN.top, plotW, plotH);

  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "right";
  for (const hz of frequencyTicks(maxHz)) {
    const y = MARGIN.top + plotH - (hz / maxHz) * plotH;
    ctx.fillText(`${formatNumber(hz)} Hz`, MARGIN.left - 5, Math.max(MARGIN.top + 6, Math.min(MARGIN.top + plotH - 6, y)));
  }
  ctx.textAlign = "center";
  const step = [0.5, 1, 2, 5, 10, 20, 30, 60].find((s) => (s / span) * plotW >= 70) || 120;
  for (let t = Math.ceil(start / step) * step; t <= start + span; t += step) {
    ctx.fillText(`${t.toFixed(step < 1 ? 1 : 0)} s`, MARGIN.left + ((t - start) / span) * plotW, height - 8);
  }
  ctx.fillStyle = COLORS.cursor;
  ctx.fillRect(MARGIN.left + plotW / 2 - 0.75, MARGIN.top, 1.5, plotH);
}

// --- Current spectrum ----------------------------------------------------------------
function drawSpectrum() {
  const { ctx, width, height } = prepareCanvas($("spectrum"));
  const ds = fq.data;
  if (!ds || !ds.n_columns) {
    $("spectrumWhen").textContent = "";
    drawCanvasMessage(ctx, width, height, emptyMessage());
    return;
  }
  const plotW = width - MARGIN.left - MARGIN.right, plotH = height - MARGIN.top - MARGIN.bottom;
  const maxHz = displayMaxHz(ds), [yMin, yMax] = fq.spectrumRange;
  const toX = (hz) => MARGIN.left + (hz / maxHz) * plotW;
  const toY = (db) => MARGIN.top + plotH - ((db - yMin) / (yMax - yMin)) * plotH;

  ctx.textAlign = "center";
  ds.bands.forEach((band, index) => {
    const left = Math.min(band.low_hz, maxHz), right = Math.min(band.high_hz, maxHz);
    if (right <= left) return;
    const x = toX(left), bandWidth = toX(right) - x;
    ctx.fillStyle = `${BAND_COLORS[index]}38`;
    ctx.fillRect(x, MARGIN.top, Math.max(1, bandWidth - 1), plotH);  // a 1 px gap separates neighbors
    if (ctx.measureText(band.name).width + 8 <= bandWidth) {
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(band.name, x + bandWidth / 2, MARGIN.top + 8);
    }
  });

  ctx.strokeStyle = COLORS.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(MARGIN.left, MARGIN.top + plotH + 0.5);
  ctx.lineTo(MARGIN.left + plotW, MARGIN.top + plotH + 0.5);
  ctx.stroke();

  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "right";
  ctx.fillText(`${yMax.toFixed(0)} dB`, MARGIN.left - 5, MARGIN.top + 6);
  ctx.fillText(`${yMin.toFixed(0)} dB`, MARGIN.left - 5, MARGIN.top + plotH - 6);
  ctx.textAlign = "center";
  for (const hz of frequencyTicks(maxHz)) {
    ctx.fillText(`${formatNumber(hz)} Hz`, Math.max(MARGIN.left + 16, Math.min(width - 24, toX(hz))), height - 8);
  }

  const column = cursorColumn(ds);
  $("spectrumWhen").textContent = `window centered at ${(ds.t[column] + state.offsetS).toFixed(2)} s`;
  const offset = column * ds.n_freqs;
  if (Number.isNaN(ds.power[offset])) {
    drawCanvasMessage(ctx, width, height, "No spectrum here: too many samples are missing in this window");
    return;
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH);
  ctx.clip();
  ctx.strokeStyle = COLORS.ink;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  const lastBin = binAt(ds, maxHz);
  for (let b = 0; b <= lastBin; b++) {
    const x = toX(ds.freqs[b]), y = toY(ds.power[offset + b]);
    if (b === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

// --- Band power ------------------------------------------------------------------------
function buildBandPlot(ds) {
  if (fq.bandPlot) { fq.bandPlot.destroy(); fq.bandPlot = null; }
  const legend = $("bandLegend");
  legend.replaceChildren(...ds.bands.map((band, index) => {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = BAND_COLORS[index];
    const note = band.n_bins ? "" : " (needs a longer window)";
    item.append(swatch, `${band.name} ${formatNumber(band.low_hz)}-${formatNumber(band.high_hz)} Hz${note}`);
    return item;
  }));
  if (!ds.n_columns) return;

  const nBands = ds.bands.length;
  let lowest = Infinity, highest = -Infinity;
  fq.bandValues = ds.bands.map((_, band) => Array.from({ length: ds.n_columns }, (__, column) => {
    const value = ds.band_power[column * nBands + band];
    if (Number.isNaN(value)) return null;  // uPlot breaks the line at null, which shows the gap
    if (value < lowest) lowest = value;
    if (value > highest) highest = value;
    return value;
  }));
  const range = Number.isFinite(lowest) ? [lowest - 3, highest + 3] : [-60, 0];
  fq.bandX = Float64Array.from(ds.t, (t) => t + state.offsetS);

  const element = $("chartBands");
  const axis = { stroke: COLORS.muted, grid: { stroke: COLORS.line, width: 1 }, ticks: { show: false }, font: "11px system-ui" };
  fq.bandPlot = new uPlot({
    width: element.clientWidth, height: element.clientHeight,
    padding: [4, 8, 0, 0],
    legend: { show: false },
    cursor: { y: false, points: { show: false }, drag: { x: false, y: false, setScale: false } },
    scales: { x: { time: false, auto: false }, y: { auto: false, range } },
    axes: [
      { ...axis, size: 18, values: (u, ticks) => ticks.map((t) => `${t} s`) },
      { ...axis, size: 42, grid: { show: false }, values: (u, ticks) => ticks.map((t) => Number(t.toPrecision(3))) },
    ],
    series: [{}, ...ds.bands.map((band, index) => ({ label: band.name, stroke: BAND_COLORS[index], width: 2, points: { show: false } }))],
    hooks: {
      draw: [drawCursorLine],
      setCursor: [(plot) => {
        const i = plot.cursor.idx;
        if (i == null) { $("hoverBands").textContent = ""; return; }
        const levels = ds.bands.map((band, b) => {
          const value = fq.bandValues[b][i];
          return `${band.name} ${value == null ? "--" : value.toFixed(1)}`;
        });
        $("hoverBands").textContent = `${fq.bandX[i].toFixed(2)} s · ${levels.join(" · ")} dB`;
      }],
    },
  }, [fq.bandX, ...fq.bandValues], element);
}

function resizeBandPlot() {
  if (!fq.bandPlot) return;
  const element = $("chartBands");
  fq.bandPlot.setSize({ width: element.clientWidth, height: element.clientHeight });
}

function applyFrequencyOffset() {
  if (!fq.bandPlot) return;
  fq.bandX = Float64Array.from(fq.data.t, (t) => t + state.offsetS);
  fq.bandPlot.setData([fq.bandX, ...fq.bandValues], false);
}

function updateBandWindow() {
  if (!fq.bandPlot) return;
  const center = cursorTime();
  fq.bandPlot.setScale("x", { min: center - state.windowS, max: center + state.windowS });
}

function drawFrequencyPanels() {
  drawSpectrogram();
  drawSpectrum();
}

// --- Hover read-outs -------------------------------------------------------------------
function plotFractions(event) {
  const box = event.target.getBoundingClientRect();
  return {
    fx: (event.clientX - box.left - MARGIN.left) / (box.width - MARGIN.left - MARGIN.right),
    fy: (event.clientY - box.top - MARGIN.top) / (box.height - MARGIN.top - MARGIN.bottom),
  };
}

function bindFrequencyHover() {
  $("spectrogram").addEventListener("mousemove", (event) => {
    const { fx, fy } = plotFractions(event), ds = fq.data;
    if (!ds || !ds.n_columns || fx < 0 || fx > 1 || fy < 0 || fy > 1) { $("hoverSpectrogram").textContent = ""; return; }
    const t = cursorTime() - state.windowS + fx * state.windowS * 2;
    const hz = (1 - fy) * displayMaxHz(ds);
    const value = ds.power[nearest(ds.t, t - state.offsetS) * ds.n_freqs + binAt(ds, hz)];
    $("hoverSpectrogram").textContent = `${t.toFixed(2)} s · ${hz.toFixed(2)} Hz · ${Number.isNaN(value) ? "no data" : `${value.toFixed(1)} dB`}`;
  });
  $("spectrogram").addEventListener("mouseleave", () => { $("hoverSpectrogram").textContent = ""; });

  $("spectrum").addEventListener("mousemove", (event) => {
    const { fx } = plotFractions(event), ds = fq.data;
    if (!ds || !ds.n_columns || fx < 0 || fx > 1) { $("hoverSpectrum").textContent = ""; return; }
    const bin = binAt(ds, fx * displayMaxHz(ds));
    const value = ds.power[cursorColumn(ds) * ds.n_freqs + bin];
    $("hoverSpectrum").textContent = `${ds.freqs[bin].toFixed(2)} Hz · ${Number.isNaN(value) ? "no data" : `${value.toFixed(1)} dB`}`;
  });
  $("spectrum").addEventListener("mouseleave", () => { $("hoverSpectrum").textContent = ""; });
}
