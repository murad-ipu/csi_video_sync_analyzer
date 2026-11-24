"""Replay a recorded session in the browser, video and CSI together.

    python player.py sessions/<folder>                  # opens http://localhost:8000
    python player.py sessions/<folder> --estimate-sync  # prints a suggested offset
    python player.py sessions/<folder> --recompute     # redo the saved analysis first
    python player.py sessions/<folder> --device <name> # one board of a multi-board recording

The player shows one board at a time, against the recording's shared video.
It only loads results: analyze.py computes them once into
<device folder>/derived/, and runs here only if they are missing or out of date.
Everything is sent to the browser as binary arrays, so the page never waits
for the server while you scrub or step.
"""

import argparse
import json
import re
import socket
import threading
import webbrowser
from pathlib import Path
from typing import Literal

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import analyze
import espectre_bridge
import espectre_path
import frequency
import timesync
# Re-exported for the tests; the player itself loads these results from analyze.py.
from frequency import MOTION_WINDOW_S, motion_signal, normalize_per_frame, select_subcarriers  # noqa: F401
from record import read_json, write_json_atomic

STATIC_DIR = Path(__file__).resolve().parent / "static"
MOTION_CHART_HZ = 25.0  # the charts need far fewer points than the 100 Hz source
SYNC_WINDOW_S = 0.2     # shorter window for the automatic offset estimate
# csi.npz fields shown by the CSI details view. Integers that can exceed 2**53
# go as int64 so the page shows them exactly.
DETAIL_FIELDS = {
    "csi_t_ns": "i8", "pc_receive_t_ns": "i8", "perf_receive_t_ns": "i8", "device_ticks_us": "i8",
    "wifi_rx_ts_us": "i8", "seq_num": "i8", "stream_sequence": "i8", "noise_floor_dbm": "f4", "channel": "f4",
}


# --- Signal processing (plain functions, unit tested) --------------------------------

def video_motion_energy(path: Path) -> np.ndarray:
    """Mean absolute difference between consecutive small grayscale frames."""
    energy, previous = [], None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            gray = frame.reformat(width=160, height=90, format="gray").to_ndarray().astype(np.int16)
            if previous is not None:
                energy.append(float(np.abs(gray - previous).mean()))
            previous = gray
    return np.array(energy)


# --- Session loading -----------------------------------------------------------------

def _espectre_payload(detectors: dict, t0_ns: int, arrays: dict) -> dict | None:
    """Add the ESPectre replay arrays for the page and return their summary.

    Both detectors evaluate on the same 250 ms grid, so they share one time array.
    """
    if not detectors:
        return None
    names = [name for name in espectre_bridge.DETECTORS if name in detectors]
    first = detectors[names[0]]
    arrays["esp_t"] = ("f8", (first["t_ns"] - t0_ns) / 1e9)
    summary = {"detectors": {}}
    for name in names:
        result = detectors[name]
        if not np.array_equal(result["t_ns"], first["t_ns"]):
            raise SystemExit(f"ESPectre {name} evaluated on a different grid; rerun with --recompute.")
        arrays[f"esp_{name}_score"] = ("f4", result["score"])
        arrays[f"esp_{name}_threshold"] = ("f4", result["threshold"])
        arrays[f"esp_{name}_state"] = ("u1", result["state"])
        summary["detectors"][name] = result["info"]
    return summary


def resolve_device(folder: Path, device: str | None = None) -> Path:
    """The device folder to show: the only board of the recording, or the one named."""
    folder = Path(folder)
    devices = analyze.device_folders(folder)
    names = [path.name for path in devices]
    if device:
        if folder / device in devices:
            return folder / device
        raise SystemExit(f"{folder} has no device {device!r}. Devices: {', '.join(names) or 'none'}")
    if len(devices) == 1:
        return devices[0]
    if not devices:
        return folder  # load_session reports what is missing
    raise SystemExit(f"{folder} holds {len(devices)} devices: {', '.join(names)}. "
                     f"Pick one with --device <name>, or pass its folder.")


def load_session(folder: Path, repo=None, recompute: bool = False) -> dict:
    """Load one board's session and its saved results (analyze.py), computing them only if needed.

    folder is a device folder; video.mp4 and frames.csv come from analyze.video_folder().
    """
    folder = Path(folder)
    video_dir = analyze.video_folder(folder)
    missing = [name for name in ("session.json", "frames.csv", "csi.npz", "csi_meta.json", "video.mp4")
               if not analyze.input_path(folder, name).exists()]
    if missing:
        hint = ""
        unfinished = [video_dir / "video.part.mp4", folder / "csi_stream.bin"]
        unfinished += [path / "csi_stream.bin" for path in analyze.device_folders(video_dir)]
        if any(path.exists() for path in unfinished):
            hint = f"\nThis session did not finish. Run: python record.py --recover {video_dir}"
        raise SystemExit(f"{folder} is missing: {', '.join(missing)}{hint}")

    espectre_path.setup(repo)
    from tools.lib.csi_io import load_npz_arrays  # applies the clone's subcarrier layout check

    session = read_json(folder / "session.json")
    frames = analyze.read_frames(folder)
    csi = load_npz_arrays(folder / "csi.npz")
    if frames.shape[0] == 0 or csi["csi_t_ns"].size == 0:
        raise SystemExit(f"{folder} has no video frames or no CSI frames to show.")
    try:
        manifest, computed = analyze.ensure_analysis(folder, repo, recompute)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if computed:
        analyze.print_summary(manifest)
    results = analyze.load_results(folder, csi)

    frame_t_ns, pts_us = frames[:, 1], frames[:, 3]
    csi_t_ns = np.asarray(csi["csi_t_ns"])
    t0_ns = int(results["t0_ns"])
    frame_t_s = (frame_t_ns - t0_ns) / 1e9
    csi_t_s = (csi_t_ns - t0_ns) / 1e9

    kept = results["kept_bins"]
    raw = results["amplitude"][:, kept]
    normalized = results["amp_norm"]
    motion = results["motion"]
    rssi = np.asarray(csi["rssi_dbm"]).astype(np.float32)

    csi_step_s = float(np.median(np.diff(csi_t_s))) if csi_t_s.size > 1 else 0.01
    stride = max(1, int(round(1.0 / (MOTION_CHART_HZ * max(csi_step_s, 1e-4)))))
    arrays = {
        "frame_t": ("f8", frame_t_s),
        "frame_pts": ("f8", pts_us / 1e6),
        "csi_t": ("f8", csi_t_s),
        "csi_rssi": ("f4", rssi),
        "amp_raw": ("f4", raw.ravel()),
        "amp_norm": ("f4", normalized.ravel()),
        "chart_t": ("f8", csi_t_s[::stride]),
        "chart_motion": ("f4", motion[::stride]),
        "chart_rssi": ("f4", rssi[::stride]),
    }
    espectre = _espectre_payload(results["detectors"], t0_ns, arrays)
    presence = results["presence"]
    if presence is not None:
        # Times are CSI time like every cached result; the page adds the sync offset.
        arrays["pres_t"] = ("f8", presence["t"])
        arrays["pres_breathing"] = ("f4", presence["breathing_share"])
        arrays["pres_breathing_hz"] = ("f4", presence["breathing_hz"])
        arrays["pres_level"] = ("f4", presence["level"])
    # Contents for the CSI details view. The page only downloads them when it is opened.
    # Amplitude and phase come from ESPectre (via analyze.py); the page never recomputes them.
    arrays["csi_iq"] = ("i1", np.asarray(csi["csi_data"]))
    arrays["csi_amp"] = ("f4", results["amplitude"])
    arrays["csi_phase"] = ("f4", results["phase"])
    arrays["mean_amplitude"] = ("f4", results["mean_amplitude"])
    detail_arrays = ["csi_iq", "csi_amp", "csi_phase", "mean_amplitude"]
    for name, dtype in DETAIL_FIELDS.items():
        if name in csi:
            arrays[name] = (dtype, np.asarray(csi[name]))
            detail_arrays.append(name)
    payloads = {name: np.asarray(values, dtype=dtype).tobytes() for name, (dtype, values) in arrays.items()}
    meta = {
        "n_frames": int(frame_t_s.size),
        "n_csi": int(csi_t_s.size),
        "kept_bins": kept.tolist(),
        "csi_step_s": csi_step_s,
        "amp_raw_range": [0.0, float(np.percentile(raw, 99.5))],
        "amp_norm_range": [float(np.percentile(normalized, 0.5)), float(np.percentile(normalized, 99.5))],
        "motion_max": float(np.percentile(motion, 99.9)) * 1.1 or 1.0,
        "rssi_range": [float(rssi.min()) - 2.0, float(rssi.max()) + 2.0],
        "motion_window_s": MOTION_WINDOW_S,
        "detail_arrays": detail_arrays,
        "n_subcarriers": int(np.asarray(csi["csi_data"]).shape[1] // 2),
        "espectre": espectre,
        "presence": presence["info"] if presence is not None else None,
        "sync_offset_ms": float(session.get("sync_offset_ms", 0)),
        "player_settings": session.get("player_settings", {}),
        # The board's name when the recording has its own folder for each board.
        "device": folder.name if video_dir != folder else None,
        "summary": {
            "status": session.get("status"),
            "fit_residual_rms_ms": session.get("time_fit", {}).get("residual_rms_ms"),
            "fit_ok": session.get("time_fit", {}).get("ok"),
        },
        "arrays": {name: dtype for name, (dtype, _) in arrays.items()},
    }
    return {
        "folder": folder, "video_path": video_dir / "video.mp4", "meta": meta, "payloads": payloads, "frame_t_s": frame_t_s,
        "csi_t_s": csi_t_s, "normalized": normalized, "presence": presence,
    }


def estimate_sync(data: dict) -> dict:
    """Compare video motion energy with a short-window CSI motion signal."""
    energy = video_motion_energy(data["video_path"])
    frame_t = data["frame_t_s"]
    # A difference between two frames describes the moment halfway between them.
    midpoints = (frame_t[1:] + frame_t[:-1])[: energy.size] / 2
    csi_motion = motion_signal(data["csi_t_s"], data["normalized"], SYNC_WINDOW_S)
    return timesync.estimate_offset_ms(midpoints, energy, data["csi_t_s"], csi_motion)


# --- Web app --------------------------------------------------------------------------

class SyncOffset(BaseModel):
    sync_offset_ms: float = Field(ge=-2000, le=2000, allow_inf_nan=False)


class FrequencyRequest(BaseModel):
    window_s: float
    hop_s: float = Field(ge=0.05, le=0.5, allow_inf_nan=False)
    method: Literal["mean", "pca1", "pca2", "motion"]

    @field_validator("window_s")
    @classmethod
    def window_is_offered(cls, value: float) -> float:
        if value not in frequency.WINDOW_CHOICES_S:
            raise ValueError(f"window_s must be one of {frequency.WINDOW_CHOICES_S}")
        return value


class FrequencySettings(FrequencyRequest):
    freq_max_hz: float = Field(gt=0, le=1000, allow_inf_nan=False)
    color_auto: bool
    color_min: float = Field(ge=-400, le=400, allow_inf_nan=False)
    color_max: float = Field(ge=-400, le=400, allow_inf_nan=False)


class PresenceBaseline(BaseModel):
    """An empty-room stretch in CSI time (no sync offset). Both None clears it."""
    start_s: float | None = Field(default=None, ge=-1e6, le=1e6, allow_inf_nan=False)
    end_s: float | None = Field(default=None, ge=-1e6, le=1e6, allow_inf_nan=False)


class PlayerSettings(BaseModel):
    mode: Literal["time", "frequency", "details", "presence"]
    details_panel: bool = False  # the CSI details panel under the video
    espectre_panel: bool = True  # the ESPectre detector panel in Time mode
    frequency: FrequencySettings


def create_app(data: dict) -> FastAPI:
    app = FastAPI(title="CSI video player", docs_url=None, redoc_url=None, openapi_url=None)
    folder, meta, payloads = data["folder"], data["meta"], data["payloads"]

    @app.middleware("http")
    async def check_page_files_every_time(request, call_next):
        # Without this, a browser may keep using an old copy of the page after an update.
        # "no-cache" still lets it reuse its copy, but only after the server confirms it is
        # current (a cheap 304 reply). The video is left alone: it never changes.
        response = await call_next(request)
        if request.url.path != "/video.mp4":
            response.headers["Cache-Control"] = "no-cache"
        return response
    write_lock = threading.Lock()
    compute_lock = threading.Lock()  # a second request for the same result then finds it cached
    amplitude = np.frombuffer(payloads["amp_raw"], dtype="f4").reshape(meta["n_csi"], len(meta["kept_bins"]))

    @app.get("/api/meta")
    def get_meta():
        return meta

    @app.get("/api/array/{name}")
    def get_array(name: str):
        if name not in payloads:
            raise HTTPException(404, "unknown array")
        return Response(payloads[name], media_type="application/octet-stream")

    @app.get("/video.mp4")
    def get_video():
        # FileResponse answers HTTP Range requests, which browsers need for seeking.
        return FileResponse(data["video_path"], media_type="video/mp4")

    @app.post("/api/sync_offset")
    def set_sync_offset(body: SyncOffset):
        with write_lock:
            session = read_json(folder / "session.json")
            session["sync_offset_ms"] = body.sync_offset_ms
            write_json_atomic(folder / "session.json", session)
            meta["sync_offset_ms"] = body.sync_offset_ms
        return {"sync_offset_ms": body.sync_offset_ms}

    @app.post("/api/player_settings")
    def set_player_settings(body: PlayerSettings):
        with write_lock:
            session = read_json(folder / "session.json")
            previous = session.get("player_settings", {}).get("frequency", {})
            # Bands are edited by hand in session.json, so the page never overwrites them.
            bands = previous.get("bands", frequency.bands_to_json(frequency.DEFAULT_BANDS))
            settings = {
                "mode": body.mode,
                "details_panel": body.details_panel,
                "espectre_panel": body.espectre_panel,
                "frequency": {**body.frequency.model_dump(), "bands": bands},
            }
            # The presence baseline has its own endpoint, so keep it as it is.
            if "presence" in session.get("player_settings", {}):
                settings["presence"] = session["player_settings"]["presence"]
            session["player_settings"] = settings
            write_json_atomic(folder / "session.json", session)
            meta["player_settings"] = settings
        return settings

    def presence_response(baseline) -> dict:
        """The saved baseline and every point's comparison with it."""
        presence = data["presence"]
        if presence is None:
            raise HTTPException(404, "This session has no presence results; rerun with --recompute.")
        if not baseline:
            return {"baseline": None, "distance": None, "level_floor": None}
        start_s, end_s = baseline
        try:
            result = frequency.presence_baseline(presence["t"], presence["shape"], presence["level"], start_s, end_s)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        # JSON has no NaN: points without a full window become null.
        distance = [None if not np.isfinite(v) else round(float(v), 4) for v in result["distance"]]
        return {"baseline": [start_s, end_s], "distance": distance, "level_floor": result["level_floor"]}

    @app.get("/api/presence/baseline")
    def get_presence_baseline():
        with write_lock:
            saved = read_json(folder / "session.json").get("player_settings", {}).get("presence", {})
        return presence_response(saved.get("baseline"))

    @app.post("/api/presence/baseline")
    def set_presence_baseline(body: PresenceBaseline):
        if (body.start_s is None) != (body.end_s is None):
            raise HTTPException(422, "Give both start_s and end_s, or neither to clear the baseline.")
        baseline = None if body.start_s is None else [body.start_s, body.end_s]
        if baseline and baseline[1] <= baseline[0]:
            raise HTTPException(422, "end_s must be after start_s.")
        response = presence_response(baseline)  # checked before anything is saved
        with write_lock:
            session = read_json(folder / "session.json")
            settings = session.setdefault("player_settings", {})
            settings["presence"] = {"baseline": baseline}
            write_json_atomic(folder / "session.json", session)
            meta["player_settings"] = settings
        return response

    @app.post("/api/frequency")
    def compute_frequency(body: FrequencyRequest):
        with write_lock:
            stored = read_json(folder / "session.json").get("player_settings", {}).get("frequency", {})
        try:
            bands = frequency.parse_bands(stored.get("bands"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        params = frequency.FrequencyParams(window_s=body.window_s, hop_s=body.hop_s, method=body.method, bands=bands)
        with compute_lock:
            key, info, was_cached = frequency.load_or_compute(folder, data["csi_t_s"], amplitude, params)
        return {**info, "key": key, "cached": was_cached}

    @app.get("/api/frequency/{key}/{name}")
    def get_frequency_array(key: str, name: str):
        if name not in frequency.CACHE_ARRAYS or not re.fullmatch(r"[0-9a-f]{20}", key):
            raise HTTPException(404, "unknown array")
        if not frequency.cache_path(folder, key).is_file():
            raise HTTPException(404, "unknown result")
        return Response(frequency.read_cached_array(folder, key, name), media_type="application/octet-stream")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def print_estimate(result: dict) -> None:
    print(f"Suggested sync offset: {result['offset_ms']:+.0f} ms   (correlation {result['correlation']:.2f})")
    if result["confident"]:
        print("Set the sync offset slider in the player to this value, then check the hand waves by eye.")
    else:
        print(f"Low confidence: {result['reason']}. Do not trust this number; repeat the sync test or align by eye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a CSI + video session in the browser.")
    parser.add_argument("session", help="Session folder made by record.py, or one device folder inside it")
    parser.add_argument("--device", help="Which board to show, for a recording with several")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument("--estimate-sync", action="store_true", help="Print a suggested sync offset and exit")
    parser.add_argument("--repo", help="Path to the ESPectre clone (default ../espectre)")
    parser.add_argument("--recompute", action="store_true", help="Redo the saved analysis even if it is up to date")
    args = parser.parse_args()

    print("Loading session...")
    folder = resolve_device(Path(args.session), args.device)
    if folder != Path(args.session):
        print(f"Device: {folder.name}")
    data = load_session(folder, args.repo, args.recompute)

    if args.estimate_sync:
        print("Decoding video to measure motion energy...")
        print_estimate(estimate_sync(data))
        return

    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"Port {args.port} is already in use. Close the other player or pass --port 8001.")

    url = f"http://localhost:{args.port}"
    print(f"Player ready at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(create_app(data), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
