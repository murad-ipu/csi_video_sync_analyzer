"""Replay a recorded session in the browser, video and CSI together.

    python player.py sessions/<folder>                  # opens http://localhost:8000
    python player.py sessions/<folder> --estimate-sync  # prints a suggested offset

All CSI signals are computed once at startup and sent to the browser as binary
arrays, so the page never waits for the server while you scrub or step.
"""

import argparse
import json
import socket
import threading
import webbrowser
from pathlib import Path

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import espectre_path
import timesync
from record import read_json, write_json_atomic

STATIC_DIR = Path(__file__).resolve().parent / "static"
MOTION_WINDOW_S = 0.5
MOTION_CHART_HZ = 25.0  # the charts need far fewer points than the 100 Hz source
SYNC_WINDOW_S = 0.2     # shorter window for the automatic offset estimate
MIN_NONZERO_FRACTION = 0.5


# --- Signal processing (plain functions, unit tested) --------------------------------

def select_subcarriers(amplitude: np.ndarray) -> np.ndarray:
    """Indices of subcarriers carrying signal; guard and null bins read zero."""
    return np.flatnonzero((amplitude > 0).mean(axis=0) >= MIN_NONZERO_FRACTION)


def normalize_per_frame(amplitude: np.ndarray) -> np.ndarray:
    """Divide each frame by its mean, which removes the per-packet AGC scale."""
    mean = amplitude.mean(axis=1, keepdims=True)
    return np.divide(amplitude, mean, out=np.zeros_like(amplitude), where=mean > 0)


def motion_signal(t_s: np.ndarray, normalized: np.ndarray, window_s: float = MOTION_WINDOW_S) -> np.ndarray:
    """Variance of normalized amplitude in a sliding window, averaged over subcarriers.

    The window is centered on each frame: a trailing window would make every
    motion peak appear late and look like a sync error.
    """
    if t_s.size == 0:
        return np.zeros(0, dtype=np.float32)
    data = normalized.astype(np.float64)
    zero = np.zeros((1, data.shape[1]))
    sum_1 = np.concatenate((zero, np.cumsum(data, axis=0)))
    sum_2 = np.concatenate((zero, np.cumsum(data**2, axis=0)))
    low = np.searchsorted(t_s, t_s - window_s / 2, side="left")
    high = np.searchsorted(t_s, t_s + window_s / 2, side="right")
    count = (high - low)[:, None].astype(np.float64)
    mean = (sum_1[high] - sum_1[low]) / count
    variance = (sum_2[high] - sum_2[low]) / count - mean**2
    return np.maximum(variance, 0.0).mean(axis=1).astype(np.float32)


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

def load_session(folder: Path, repo=None) -> dict:
    """Read the session files and precompute everything the page displays."""
    missing = [name for name in ("session.json", "frames.csv", "csi.npz", "video.mp4") if not (folder / name).exists()]
    if missing:
        hint = ""
        if (folder / "csi_stream.bin").exists() or (folder / "video.part.mp4").exists():
            hint = f"\nThis session did not finish. Run: python record.py --recover {folder}"
        raise SystemExit(f"{folder} is missing: {', '.join(missing)}{hint}")

    espectre_path.setup(repo)
    from tools.lib.csi_io import load_npz_arrays  # applies the clone's subcarrier layout check

    session = read_json(folder / "session.json")
    frames = np.loadtxt(folder / "frames.csv", delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    csi = load_npz_arrays(folder / "csi.npz")
    if frames.shape[0] == 0 or csi["csi_t_ns"].size == 0:
        raise SystemExit(f"{folder} has no video frames or no CSI frames to show.")

    frame_t_ns, pts_us = frames[:, 1], frames[:, 3]
    csi_t_ns = np.asarray(csi["csi_t_ns"])
    t0_ns = int(min(frame_t_ns[0], csi_t_ns[0]))
    frame_t_s = (frame_t_ns - t0_ns) / 1e9
    csi_t_s = (csi_t_ns - t0_ns) / 1e9

    kept = select_subcarriers(np.asarray(csi["amplitude"]))
    if kept.size == 0:
        raise SystemExit(f"{folder}: every subcarrier is empty.")
    raw = np.asarray(csi["amplitude"])[:, kept].astype(np.float32)
    normalized = normalize_per_frame(raw).astype(np.float32)
    motion = motion_signal(csi_t_s, normalized)
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
        "sync_offset_ms": float(session.get("sync_offset_ms", 0)),
        "summary": {
            "status": session.get("status"),
            "fit_residual_rms_ms": session.get("time_fit", {}).get("residual_rms_ms"),
            "fit_ok": session.get("time_fit", {}).get("ok"),
        },
        "arrays": {name: dtype for name, (dtype, _) in arrays.items()},
    }
    return {
        "folder": folder, "meta": meta, "payloads": payloads, "frame_t_s": frame_t_s,
        "csi_t_s": csi_t_s, "normalized": normalized,
    }


def estimate_sync(data: dict) -> dict:
    """Compare video motion energy with a short-window CSI motion signal."""
    energy = video_motion_energy(data["folder"] / "video.mp4")
    frame_t = data["frame_t_s"]
    # A difference between two frames describes the moment halfway between them.
    midpoints = (frame_t[1:] + frame_t[:-1])[: energy.size] / 2
    csi_motion = motion_signal(data["csi_t_s"], data["normalized"], SYNC_WINDOW_S)
    return timesync.estimate_offset_ms(midpoints, energy, data["csi_t_s"], csi_motion)


# --- Web app --------------------------------------------------------------------------

class SyncOffset(BaseModel):
    sync_offset_ms: float = Field(ge=-2000, le=2000, allow_inf_nan=False)


def create_app(data: dict) -> FastAPI:
    app = FastAPI(title="CSI video player", docs_url=None, redoc_url=None, openapi_url=None)
    folder, meta, payloads = data["folder"], data["meta"], data["payloads"]
    write_lock = threading.Lock()

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
        return FileResponse(folder / "video.mp4", media_type="video/mp4")

    @app.post("/api/sync_offset")
    def set_sync_offset(body: SyncOffset):
        with write_lock:
            session = read_json(folder / "session.json")
            session["sync_offset_ms"] = body.sync_offset_ms
            write_json_atomic(folder / "session.json", session)
            meta["sync_offset_ms"] = body.sync_offset_ms
        return {"sync_offset_ms": body.sync_offset_ms}

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
    parser.add_argument("session", help="Session folder made by record.py")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument("--estimate-sync", action="store_true", help="Print a suggested sync offset and exit")
    parser.add_argument("--repo", help="Path to the ESPectre clone (default ../espectre)")
    args = parser.parse_args()

    print("Loading session and computing CSI signals...")
    data = load_session(Path(args.session), args.repo)

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
