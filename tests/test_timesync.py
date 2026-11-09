"""Unit tests for the time mapping, using synthetic data only."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import timesync  # noqa: E402

PC_START_NS = 1_780_000_000_000_000_000  # a realistic time.time_ns() value


def synthetic_stream(duration_s=300, pps=100, drift_ppm=35.0, seed=1):
    """Device ticks and bursty arrival times with a known true clock relation."""
    rng = np.random.default_rng(seed)
    n = duration_s * pps
    device_us = 5_000_000 + np.arange(n, dtype=np.int64) * (1_000_000 // pps)
    device_us = device_us + rng.integers(-300, 300, n)
    true_ns = PC_START_NS + np.rint((device_us - 5_000_000) * 1000.0 * (1 + drift_ppm / 1e6)).astype(np.int64)
    # One-sided delays: 1-3 ms base, and every 8 frames form a released burst.
    delay_ms = 1.0 + rng.exponential(2.0, n)
    delay_ms += (7 - np.arange(n) % 8) * 10.0
    # Occasional long stall (Wi-Fi retry or Python pause).
    stalls = rng.choice(n, n // 200, replace=False)
    delay_ms[stalls] += rng.uniform(50, 400, stalls.size)
    return device_us, true_ns, true_ns + np.rint(delay_ms * 1e6).astype(np.int64)


def test_fit_recovers_drift_and_true_time_through_bursts():
    device_us, true_ns, received_ns = synthetic_stream()
    fit = timesync.fit_device_to_pc(device_us, received_ns)

    assert fit.ok
    assert fit.drift_ppm == pytest.approx(35.0, abs=3.0)
    error_ms = (fit.to_pc_ns(device_us) - true_ns) / 1e6
    # Only the minimum path delay (about 1 ms here) stays uncorrected.
    assert np.abs(error_ms).max() < 3.0
    assert fit.residual_p95_ms < 2.0


def test_fit_beats_raw_receive_times():
    device_us, true_ns, received_ns = synthetic_stream(seed=2)
    fit = timesync.fit_device_to_pc(device_us, received_ns)
    corrected = np.abs(fit.to_pc_ns(device_us) - true_ns).mean()
    raw = np.abs(received_ns - true_ns).mean()
    assert corrected < raw / 10


@pytest.mark.parametrize("drift_ppm", [-120.0, 0.0, 80.0])
def test_fit_handles_negative_zero_and_positive_drift(drift_ppm):
    device_us, _, received_ns = synthetic_stream(drift_ppm=drift_ppm, seed=3)
    assert timesync.fit_device_to_pc(device_us, received_ns).drift_ppm == pytest.approx(drift_ppm, abs=3.0)


def test_fit_survives_a_window_where_every_packet_is_late():
    device_us, true_ns, received_ns = synthetic_stream(seed=4)
    late = (device_us > 100_000_000) & (device_us < 104_000_000)
    received_ns = received_ns.copy()
    received_ns[late] += 500_000_000
    fit = timesync.fit_device_to_pc(device_us, received_ns)
    assert np.abs((fit.to_pc_ns(device_us) - true_ns) / 1e6).max() < 3.0


def test_short_recording_falls_back_to_offset_only():
    device_us, _, received_ns = synthetic_stream(duration_s=1)
    fit = timesync.fit_device_to_pc(device_us, received_ns)
    assert fit.method == "offset_only"
    assert fit.slope_ns_per_us == 1000.0
    assert not fit.ok


def test_empty_input_is_reported_not_raised():
    fit = timesync.fit_device_to_pc([], [])
    assert fit.n_frames == 0 and not fit.ok


def test_frame_to_csi_picks_nearest_and_breaks_ties_left():
    csi = np.array([0, 10, 20, 30], dtype=np.int64) * 1_000_000
    frames = np.array([-5, 4, 5, 6, 26, 99], dtype=np.int64) * 1_000_000
    assert timesync.frame_to_csi_index(frames, csi).tolist() == [0, 0, 0, 1, 3, 3]


def test_positive_offset_moves_csi_later():
    csi = np.array([0, 100, 200], dtype=np.int64) * 1_000_000
    frame = np.array([200], dtype=np.int64) * 1_000_000
    # CSI index 1 (100 ms) + 100 ms offset lands on the frame at 200 ms.
    assert timesync.frame_to_csi_index(frame, csi, sync_offset_ms=100.0).tolist() == [1]
    assert timesync.frame_to_csi_index(frame, csi, sync_offset_ms=-100.0).tolist() == [2]


def test_frame_mapping_keeps_full_nanosecond_precision():
    # These two differ by 1 ns: a float64 round trip could not separate them.
    csi = np.array([PC_START_NS, PC_START_NS + 2], dtype=np.int64)
    frame = np.array([PC_START_NS + 1 + 1], dtype=np.int64)
    assert timesync.frame_to_csi_index(frame, csi).tolist() == [1]


def test_media_time_maps_to_exact_variable_rate_frame():
    pts_us = np.array([0, 33_000, 66_000, 166_000, 199_000])
    assert timesync.media_time_to_frame(pts_us, [0.0, 0.066, 0.166, 0.199]).tolist() == [0, 2, 3, 4]


def test_single_csi_frame_is_always_nearest():
    assert timesync.nearest_index(np.array([7]), np.array([1, 100])).tolist() == [0, 0]


def test_count_frame_gaps_uses_the_streams_own_rate():
    t_33ms = np.arange(0, 300) * 33_000_000
    assert timesync.count_frame_gaps(t_33ms) == 0
    t_with_gap = np.delete(t_33ms, [50, 51, 200])
    assert timesync.count_frame_gaps(t_with_gap) == 3
    assert timesync.count_frame_gaps(np.arange(0, 300) * 66_000_000) == 0


def motion_pair(true_offset_ms):
    """Three wave bursts seen by the camera, and by CSI earlier by the offset."""
    video_t = np.arange(0, 30, 1 / 30)
    csi_t = np.arange(0, 30, 0.01)

    def bursts(t):
        return sum(np.exp(-0.5 * ((t - c) / 0.15) ** 2) for c in (7.0, 14.5, 23.0))

    rng = np.random.default_rng(5)
    video = bursts(video_t) + rng.normal(0, 0.02, video_t.size)
    csi = bursts(csi_t + true_offset_ms / 1000.0) + rng.normal(0, 0.02, csi_t.size)
    return video_t, video, csi_t, csi


@pytest.mark.parametrize("true_offset_ms", [-200.0, 0.0, 120.0, 350.0])
def test_estimate_offset_recovers_known_shift(true_offset_ms):
    result = timesync.estimate_offset_ms(*motion_pair(true_offset_ms))
    assert result["confident"]
    assert result["offset_ms"] == pytest.approx(true_offset_ms, abs=20.0)


def test_estimate_offset_flags_unrelated_signals():
    rng = np.random.default_rng(6)
    t = np.arange(0, 30, 0.02)
    result = timesync.estimate_offset_ms(t, rng.normal(size=t.size), t, rng.normal(size=t.size), smooth_s=0.02)
    assert not result["confident"]
