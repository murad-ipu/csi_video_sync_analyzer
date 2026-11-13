"""Compute every derived value of a session once and save it in <session>/derived/.

    python analyze.py sessions/<folder>              # skips work that is already up to date
    python analyze.py sessions/<folder> --recompute  # always recompute

record.py runs this at the end of every recording, and player.py runs it when
the saved results are missing or out of date. The player itself only loads.

Files written:
  derived/csi_values.npz             amplitude and phase from ESPectre's CSIPacket, only
                                     for sessions whose csi.npz predates storing phase
  derived/player_signals.npz         subcarriers kept, normalized amplitude, motion, mean amplitude
  derived/espectre_lightweight.npz   ESPectre Lightweight replay, one row per evaluation
  derived/espectre_high_accuracy.npz ESPectre High-Accuracy replay, one row per evaluation
  derived/manifest.json              what the results were computed from
  frequency_cache/<key>.npz          the Frequency view for the page's first settings

The results are recomputed when the manifest no longer matches: an input file
changed (csi.npz, frames.csv, csi_meta.json), the parameters changed, the
analysis code changed, or the ESPectre clone moved to another commit or has
local edits.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import espectre_bridge
import espectre_path
import frequency

DERIVED_DIR = "derived"
MANIFEST = "manifest.json"
INPUT_FILES = ("csi.npz", "frames.csv", "csi_meta.json")
# Hashing these means any change to the analysis code invalidates old results.
CODE_FILES = ("analyze.py", "espectre_bridge.py", "frequency.py")
ROOT = Path(__file__).resolve().parent
DETECTOR_FILES = {name: f"espectre_{name}.npz" for name in espectre_bridge.DETECTORS}


# --- Small helpers ----------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _write_npz(path: Path, **arrays) -> None:
    temp = path.with_suffix(".tmp")
    with open(temp, "wb") as file:
        np.savez(file, **arrays)
    os.replace(temp, path)


def _write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_frames(folder: Path) -> np.ndarray:
    return np.loadtxt(folder / "frames.csv", delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)


def session_t0_ns(frame_t_ns: np.ndarray, csi_t_ns: np.ndarray) -> int:
    """Session time zero: the first video frame or CSI frame, whichever came first."""
    return int(min(frame_t_ns[0], csi_t_ns[0]))


def _sensing(folder: Path) -> dict | None:
    meta = json.loads((folder / "csi_meta.json").read_text(encoding="utf-8"))
    return meta.get("sensing")


def _frequency_params(folder: Path) -> frequency.FrequencyParams:
    """The settings the page asks for first: the saved ones, else the defaults."""
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    saved = session.get("player_settings", {}).get("frequency", {})
    bands = frequency.parse_bands(saved.get("bands"))
    window_s = saved.get("window_s") if saved.get("window_s") in frequency.WINDOW_CHOICES_S else 2.0
    hop_s = saved.get("hop_s") if isinstance(saved.get("hop_s"), (int, float)) and 0.05 <= saved["hop_s"] <= 0.5 else 0.1
    method = saved.get("method") if saved.get("method") in frequency.METHODS else "mean"
    return frequency.FrequencyParams(window_s=float(window_s), hop_s=float(hop_s), method=method, bands=bands)


# --- Manifest ---------------------------------------------------------------------------

def fingerprint(folder: Path, repo=None) -> dict:
    """Everything the results depend on. Results are reused only if this is unchanged."""
    espectre_root = espectre_path.setup(repo)
    diff = _git(espectre_root, "diff", "HEAD")
    commit = _git(espectre_root, "rev-parse", "HEAD")
    code = hashlib.sha256()
    for name in CODE_FILES:
        code.update((ROOT / name).read_bytes())
    return {
        "inputs": {name: _sha256(folder / name) for name in INPUT_FILES},
        "code_sha256": code.hexdigest(),
        "espectre": {
            "commit": commit.strip() if commit else "unknown",
            # Local edits in the clone change results without changing the commit.
            "local_changes_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        },
        "parameters": {
            "player": {
                "min_nonzero_fraction": frequency.MIN_NONZERO_FRACTION,
                "motion_window_s": frequency.MOTION_WINDOW_S,
            },
            "espectre": espectre_bridge.espectre_parameters(),
        },
    }


def read_manifest(folder: Path) -> dict | None:
    try:
        return json.loads((folder / DERIVED_DIR / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_up_to_date(folder: Path, repo=None) -> bool:
    manifest = read_manifest(folder)
    if manifest is None or manifest.get("fingerprint") != fingerprint(folder, repo):
        return False
    return all((folder / DERIVED_DIR / name).is_file() for name in manifest.get("files", []))


# --- Computing --------------------------------------------------------------------------

def _csi_values(folder: Path, csi) -> tuple[np.ndarray, np.ndarray, str]:
    """Amplitude and phase. New recordings store them in csi.npz; older ones get them computed here."""
    if "phase" in csi and "amplitude" in csi:
        return np.asarray(csi["amplitude"], np.float32), np.asarray(csi["phase"], np.float32), "csi.npz"
    amplitude, phase = espectre_bridge.amplitude_phase(np.asarray(csi["csi_data"]))
    _write_npz(folder / DERIVED_DIR / "csi_values.npz", amplitude=amplitude, phase=phase)
    return amplitude, phase, "derived/csi_values.npz"


def run_analysis(folder: Path, repo=None, log=print) -> dict:
    """Compute and save every derived result for one session. Returns the manifest."""
    folder = Path(folder)
    espectre_path.setup(repo)
    from tools.lib.csi_io import load_npz_arrays  # applies the clone's subcarrier layout check

    derived = folder / DERIVED_DIR
    derived.mkdir(exist_ok=True)
    # Without a manifest the results count as missing, so a crash part-way is redone next time.
    (derived / MANIFEST).unlink(missing_ok=True)
    started = time.perf_counter()
    timings = {}

    csi = load_npz_arrays(folder / "csi.npz")
    frames = read_frames(folder)
    csi_t_ns = np.asarray(csi["csi_t_ns"], np.int64)
    if frames.shape[0] == 0 or csi_t_ns.size == 0:
        raise ValueError(f"{folder} has no video frames or no CSI frames to analyze.")
    t0_ns = session_t0_ns(frames[:, 1], csi_t_ns)
    csi_t_s = (csi_t_ns - t0_ns) / 1e9
    files = []

    # 1. Amplitude and phase (ESPectre CSIPacket).
    log("  amplitude and phase (ESPectre CSIPacket)...")
    step = time.perf_counter()
    amplitude, _, values_source = _csi_values(folder, csi)
    if values_source.startswith("derived/"):
        files.append("csi_values.npz")
    timings["csi_values_s"] = time.perf_counter() - step

    # 2. The player's own signals.
    log("  player signals (normalized amplitude, motion)...")
    step = time.perf_counter()
    kept = frequency.select_subcarriers(amplitude)
    if kept.size == 0:
        raise ValueError(f"{folder}: every subcarrier is empty.")
    raw = amplitude[:, kept]
    normalized = frequency.normalize_per_frame(raw).astype(np.float32)
    motion = frequency.motion_signal(csi_t_s, normalized)
    _write_npz(
        derived / "player_signals.npz",
        t0_ns=np.int64(t0_ns), kept_bins=kept.astype(np.int64), amp_norm=normalized, motion=motion,
        mean_amplitude=raw.mean(axis=1).astype(np.float32),
    )
    files.append("player_signals.npz")
    timings["player_signals_s"] = time.perf_counter() - step

    # 3. ESPectre detectors, replayed packet by packet.
    log("  ESPectre Lightweight and High-Accuracy replay...")
    step = time.perf_counter()
    results = espectre_bridge.replay_detectors(folder / "csi.npz", _sensing(folder))
    detectors = {}
    for name, result in results.items():
        info = result["info"]
        _write_npz(
            derived / DETECTOR_FILES[name],
            row=result["row"], t_ns=csi_t_ns[result["row"]], score=result["score"], threshold=result["threshold"],
            ready=result["ready"], raw_motion=result["raw_motion"], state=result["state"], info=json.dumps(info),
        )
        files.append(DETECTOR_FILES[name])
        detectors[name] = info
    timings["detectors_s"] = time.perf_counter() - step

    # 4. Frequency view for the page's first settings, in the existing cache.
    step = time.perf_counter()
    frequency_entry = None
    try:
        params = _frequency_params(folder)
        log(f"  frequency view ({params.method}, {params.window_s:g} s window)...")
        key, _, _ = frequency.load_or_compute(folder, csi_t_s, raw, params)
        frequency_entry = {"key": key, "window_s": params.window_s, "hop_s": params.hop_s, "method": params.method}
    except ValueError as exc:
        log(f"  frequency view skipped: {exc}")
    timings["frequency_s"] = time.perf_counter() - step
    timings["total_s"] = time.perf_counter() - started

    app_commit = _git(ROOT, "rev-parse", "HEAD")
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "app_commit": app_commit.strip() if app_commit else "unknown",
        "fingerprint": fingerprint(folder, repo),
        "files": files,
        "csi_values_source": values_source,
        "detectors": detectors,
        "frequency_default": frequency_entry,
        "timings": {name: round(value, 3) for name, value in timings.items()},
    }
    _write_json(derived / MANIFEST, manifest)
    return manifest


def ensure_analysis(folder: Path, repo=None, recompute: bool = False, log=print) -> tuple[dict, bool]:
    """Return (manifest, computed): reuse the saved results, or compute them if needed."""
    if not recompute and is_up_to_date(folder, repo):
        return read_manifest(folder), False
    log(f"Analyzing {folder} ...")
    return run_analysis(folder, repo, log), True


# --- Loading ------------------------------------------------------------------------------

def load_results(folder: Path, csi) -> dict:
    """Load the saved results. Call ensure_analysis first."""
    derived = Path(folder) / DERIVED_DIR
    manifest = read_manifest(folder)
    if manifest["csi_values_source"] == "csi.npz":
        amplitude, phase = np.asarray(csi["amplitude"], np.float32), np.asarray(csi["phase"], np.float32)
    else:
        with np.load(derived / "csi_values.npz") as stored:
            amplitude, phase = stored["amplitude"], stored["phase"]
    with np.load(derived / "player_signals.npz") as stored:
        signals = {name: stored[name] for name in stored.files}
    detectors = {}
    for name, file_name in DETECTOR_FILES.items():
        if file_name in manifest["files"]:
            with np.load(derived / file_name) as stored:
                detectors[name] = {key: stored[key] for key in stored.files if key != "info"}
                detectors[name]["info"] = json.loads(str(stored["info"]))
    return {"manifest": manifest, "amplitude": amplitude, "phase": phase, **signals, "detectors": detectors}


def print_summary(manifest: dict) -> None:
    timings = manifest["timings"]
    print(f"Analysis done in {timings['total_s']:.1f} s (ESPectre detectors {timings['detectors_s']:.1f} s).")
    for name, info in manifest["detectors"].items():
        ready = max(1, info["n_ready"])
        print(f"  {name:14s} threshold {info['start_threshold']:.3f} ({info['threshold_source']}), "
              f"{info['n_evaluations']} evaluations, MOTION {100 * info['n_motion'] / ready:.0f}% of ready ones")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and save the derived results of a session.")
    parser.add_argument("session", help="Session folder made by record.py")
    parser.add_argument("--recompute", action="store_true", help="Recompute even if the saved results are up to date")
    parser.add_argument("--repo", help="Path to the ESPectre clone (default ../espectre)")
    args = parser.parse_args()
    folder = Path(args.session)
    missing = [name for name in INPUT_FILES if not (folder / name).exists()]
    if missing:
        raise SystemExit(f"{folder} is missing: {', '.join(missing)}. Finish it with: python record.py --recover {folder}")
    manifest, computed = ensure_analysis(folder, args.repo, args.recompute)
    if computed:
        print_summary(manifest)
    else:
        print(f"{folder}: results are up to date (use --recompute to redo them).")


if __name__ == "__main__":
    main()
