"""Create a synthetic session so the player can be tried without hardware.

    python tests/make_demo_session.py <folder> --seconds 600 --offset-ms 80

The video shows a white square that moves during three bursts. CSI shows the
same bursts, earlier by --offset-ms, which is what camera delay looks like.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import espectre_bridge  # noqa: E402
import espectre_path  # noqa: E402
import record  # noqa: E402
import timesync  # noqa: E402

espectre_path.setup()

PC_START_NS = 1_780_000_000_000_000_000
NUM_SC = 64
NULL_BINS = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]


def burst_level(t_s, centers):
    """1.0 during a movement burst, 0.0 outside one."""
    return np.max([np.exp(-0.5 * ((t_s - c) / 0.35) ** 2) for c in centers], axis=0)


def make_session(folder: Path, seconds: float, offset_ms: float, width=1280, height=720) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(11)
    # Repeating bursts keep long sessions interesting.
    centers = np.arange(5.0, seconds - 2.0, 7.3) + rng.uniform(-1.5, 1.5, np.arange(5.0, seconds - 2.0, 7.3).size)

    # Video: 30 fps with small timing jitter (true variable frame rate).
    n_frames = int(seconds * 30)
    frame_t_ns = PC_START_NS + np.cumsum(rng.integers(32_000_000, 35_000_000, n_frames))
    relative_s = (frame_t_ns - PC_START_NS) / 1e9
    level = burst_level(relative_s, centers)
    # The square swings left and right around the center, always staying in view.
    speed = width / 70.0
    position = np.cumsum(level * speed * np.sign(np.sin(relative_s * 11)))
    half = height // 8
    encoder = record.VideoEncoder(folder / "video.part.mp4", width, height, 30)
    with open(folder / "frames.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "pc_t_ns", "perf_t_ns", "pts_us"])
        for n in range(n_frames):
            image = np.full((height, width, 3), 40, np.uint8)
            x = int(np.clip(width / 2 + position[n], half, width - half))
            image[height // 2 - half: height // 2 + half, x - half: x + half] = 235
            pts_us = int((frame_t_ns[n] - frame_t_ns[0]) // 1000)
            encoder.encode(image, pts_us)
            writer.writerow([n, int(frame_t_ns[n]), n, pts_us])
    encoder.close()
    record.remux_faststart(folder / "video.part.mp4", folder / "video.mp4")
    (folder / "video.part.mp4").unlink()

    # CSI: 100 per second, with a known clock relation and bursty arrival.
    n_csi = int(seconds * 100)
    ticks = 3_000_000 + np.arange(n_csi, dtype=np.int64) * 10_000 + rng.integers(-200, 200, n_csi)
    true_ns = PC_START_NS + (ticks - 3_000_000) * 1000
    received_ns = true_ns + np.rint((2.0 + rng.exponential(3.0, n_csi) + (7 - np.arange(n_csi) % 8) * 10.0) * 1e6).astype(np.int64)
    fit = timesync.fit_device_to_pc(ticks, received_ns)
    csi_t_s = (true_ns - PC_START_NS) / 1e9
    csi_level = burst_level(csi_t_s + offset_ms / 1000.0, centers)

    shape = 18.0 + 8.0 * np.sin(np.linspace(0, 9, NUM_SC))
    drift = 1.0 + 0.1 * rng.standard_normal((n_csi, NUM_SC))[:, :] * (0.08 + csi_level)[:, None]
    ripple = 1.0 + 0.25 * csi_level[:, None] * np.sin(np.linspace(0, 20, NUM_SC) + csi_t_s[:, None] * 25)
    agc = rng.choice([1.0, 1.0, 1.0, 0.9, 1.1], n_csi)[:, None]
    amplitude = shape * drift * ripple * agc
    amplitude[:, NULL_BINS] = 0
    iq = np.zeros((n_csi, NUM_SC * 2), np.int8)
    iq[:, 1::2] = np.clip(np.rint(amplitude), -127, 127)
    # Same as a real recording: amplitude and phase from ESPectre, computed from the stored I/Q.
    stored_amplitude, phase = espectre_bridge.amplitude_phase(iq)

    np.savez(
        folder / "csi.npz",
        pc_receive_t_ns=received_ns, perf_receive_t_ns=received_ns, csi_t_ns=fit.to_pc_ns(ticks),
        device_ticks_us=ticks, wifi_rx_ts_us=ticks & 0xFFFFFFFF, seq_num=np.arange(n_csi), stream_sequence=np.arange(n_csi),
        rssi_dbm=(-50 + rng.integers(-2, 3, n_csi)).astype(np.int8), noise_floor_dbm=np.full(n_csi, -96, np.int8),
        channel=np.full(n_csi, 6, np.uint8), csi_data=iq, amplitude=stored_amplitude, phase=phase,
        csi_target_pps=np.int64(100),
    )
    sensing = {"enabled": True, "detector": "high_accuracy", "threshold": 0.5,
               "motion_on_hits": 4, "motion_off_hits": 3, "csi_target_pps": 100}
    (folder / "csi_meta.json").write_text(json.dumps(
        {"subcarrier_count": NUM_SC, "record_version": 8, "synthetic": True, "sensing": sensing}))
    (folder / "session.json").write_text(json.dumps({"status": "synthetic", "time_fit": fit.to_dict(), "sync_offset_ms": 0}, indent=2))
    print(f"Wrote {folder}: {n_frames} frames, {n_csi} CSI frames, true offset {offset_ms:+.0f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--offset-ms", type=float, default=80.0)
    args = parser.parse_args()
    make_session(args.folder, args.seconds, args.offset_ms)
