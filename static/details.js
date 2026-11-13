// CSI details: the contents of csi.npz for the CSI frame matched to the video.
// Shown in two places: the panel under the video (checkbox) and the Details view.
// Loaded before app.js and uses its state and helpers, only from inside functions.
"use strict";

const det = {
  loaded: false, loading: false, error: "",
  iq: null, amp: null, phase: null, fields: {},  // amp and phase are ESPectre's values, from Python
  views: [],          // { container, full, cells, rows, fieldCells, lastKey }
  keptIndex: null,    // subcarrier bin -> column in ampNorm, or -1
};

// Label, then a function returning the text for frame i.
const DETAIL_ROWS = [
  ["CSI frame", (i) => `${i + 1} / ${meta.n_csi}`],
  ["Session time", (i) => `${csiT[i].toFixed(3)} s`],
  ["On the video clock", (i) => `${(csiT[i] + state.offsetS).toFixed(3)} s (offset ${(state.offsetS * 1000).toFixed(0)} ms)`],
  ["Since previous frame", (i) => (i > 0 ? `${((csiT[i] - csiT[i - 1]) * 1000).toFixed(2)} ms` : "--")],
  ["csi_t_ns", (i) => rawField("csi_t_ns", i)],
  ["pc_receive_t_ns", (i) => rawField("pc_receive_t_ns", i)],
  ["Network delay", (i) => {
    const sent = det.fields.csi_t_ns, received = det.fields.pc_receive_t_ns;
    return sent && received ? `${(Number(received[i] - sent[i]) / 1e6).toFixed(2)} ms` : "--";
  }],
  ["perf_receive_t_ns", (i) => rawField("perf_receive_t_ns", i)],
  ["device_ticks_us", (i) => rawField("device_ticks_us", i)],
  ["wifi_rx_ts_us", (i) => rawField("wifi_rx_ts_us", i)],
  ["seq_num", (i) => rawField("seq_num", i)],
  ["stream_sequence", (i) => rawField("stream_sequence", i)],
  ["rssi_dbm", (i) => `${csiRssi[i].toFixed(0)} dBm`],
  ["noise_floor_dbm", (i) => {
    const value = rawField("noise_floor_dbm", i);
    return value === "-128" ? "not reported" : `${value} dBm`;
  }],
  ["channel", (i) => rawField("channel", i)],
  ["Mean amplitude", (i) => {
    const mean = det.fields.mean_amplitude;
    return mean ? `${mean[i].toFixed(2)} (${K} signal bins)` : "--";
  }],
];

function rawField(name, i) {
  const values = det.fields[name];
  return values ? String(values[i]) : "--";
}

// --- Loading ------------------------------------------------------------------------
async function ensureDetailsLoaded() {
  if (det.loaded || det.loading) return;
  if (!meta.detail_arrays.includes("csi_iq")) {
    det.error = "This session's csi.npz has no csi_data";
    return;
  }
  det.loading = true;
  try {
    const arrays = await Promise.all(meta.detail_arrays.map(fetchArray));
    const special = { csi_iq: "iq", csi_amp: "amp", csi_phase: "phase" };
    meta.detail_arrays.forEach((name, n) => {
      if (name in special) det[special[name]] = arrays[n]; else det.fields[name] = arrays[n];
    });
    det.keptIndex = new Int32Array(meta.n_subcarriers).fill(-1);
    bins.forEach((bin, k) => { if (bin < meta.n_subcarriers) det.keptIndex[bin] = k; });
    det.loaded = true;
  } catch (error) {
    det.error = `Could not load the CSI details: ${error.message}`;
  } finally {
    det.loading = false;
    for (const view of det.views) view.lastKey = null;
    queueRender();
  }
}

// --- Building a view --------------------------------------------------------------
// The DOM is built once; each frame only replaces the text of existing cells.
function buildDetailsView(container, full) {
  const view = { container, full, lastKey: null, fieldCells: [], cells: [], rows: [] };
  const fieldList = document.createElement("div");
  fieldList.className = "fields";
  for (const [label] of DETAIL_ROWS) {
    const item = document.createElement("div");
    const name = document.createElement("span");
    name.className = "label";
    name.textContent = label;
    const value = document.createElement("span");
    item.append(name, value);
    fieldList.append(item);
    view.fieldCells.push(value);
  }

  const columns = full
    ? ["Bin", "Q", "I", "Amplitude", "Normalized", "Phase (rad)"]
    : ["Bin", "Q", "I", "Amplitude"];
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const name of columns) {
    const cell = document.createElement("th");
    cell.textContent = name;
    head.append(cell);
  }
  const body = table.createTBody();
  for (let bin = 0; bin < meta.n_subcarriers; bin++) {
    const row = body.insertRow();
    const cells = columns.map(() => row.insertCell());
    cells[0].textContent = String(bin);
    view.rows.push(row);
    view.cells.push(cells);
  }

  const note = document.createElement("p");
  note.className = "note";
  note.textContent = "csi_data stores each subcarrier as two int8 values, Q first, then I. "
    + "Amplitude and phase are ESPectre's CSIPacket values for I + jQ: |I + jQ| and its angle. "
    + "Grey rows are guard and null subcarriers, which carry no signal.";
  const status = document.createElement("p");
  status.className = "note";
  view.status = status;

  const scroller = document.createElement("div");
  scroller.className = "tableScroll";
  scroller.append(table);
  container.classList.add(full ? "full" : "compact");
  container.replaceChildren(status, fieldList, scroller, note);
  det.views.push(view);
  return view;
}

function fillDetailsView(view, i) {
  view.status.hidden = true;
  DETAIL_ROWS.forEach(([, text], n) => { view.fieldCells[n].textContent = text(i); });
  const iq = det.iq, n = meta.n_subcarriers, width = n * 2;
  for (let bin = 0; bin < n; bin++) {
    const q = iq[i * width + bin * 2], iValue = iq[i * width + bin * 2 + 1];
    const cells = view.cells[bin], k = det.keptIndex[bin];
    view.rows[bin].classList.toggle("null", k < 0);
    cells[1].textContent = String(q);
    cells[2].textContent = String(iValue);
    cells[3].textContent = det.amp[i * n + bin].toFixed(2);
    if (view.full) {
      cells[4].textContent = k >= 0 ? ampNorm[i * K + k].toFixed(3) : "--";
      // A zero I/Q pair has no meaningful angle.
      cells[5].textContent = q === 0 && iValue === 0 ? "--" : det.phase[i * n + bin].toFixed(3);
    }
  }
}

// --- Updating ------------------------------------------------------------------------
function initDetails() {
  buildDetailsView($("detailsDock"), false);
  buildDetailsView($("detailsMain"), true);
}

function updateDetails() {
  const visible = det.views.filter((view) => !view.container.closest("[hidden]"));
  if (!visible.length) return;
  if (!det.loaded) {
    ensureDetailsLoaded();
    for (const view of visible) {
      view.status.hidden = false;
      view.status.textContent = det.error || "Loading the CSI frame contents...";
    }
    return;
  }
  // Redraw only when the frame or the offset changed; playback re-renders 30 times a second.
  const key = `${state.csi}/${state.offsetS}`;
  for (const view of visible) {
    if (view.lastKey === key) continue;
    fillDetailsView(view, state.csi);
    view.lastKey = key;
  }
}
