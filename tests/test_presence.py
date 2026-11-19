"""Tests for the presence evidence: breathing share, channel shape, motion level, baseline."""

import numpy as np
import pytest

import frequency
from frequency import PresenceParams

FS = 100.0
K = 20
PARAMS = PresenceParams()


def timeline(seconds):
    return np.arange(int(seconds * FS)) / FS


def still_room(t, seed=3):
    """Normalized amplitude of a quiet channel: a fixed shape plus a little noise."""
    rng = np.random.default_rng(seed)
    shape = np.linspace(0.8, 1.2, K)
    return (shape * (1 + 0.003 * rng.standard_normal((t.size, K)))).astype(np.float32)


def breathe(normalized, t, hz, depth=0.01, columns=range(3, 12)):
    normalized[:, list(columns)] *= 1 + depth * np.sin(2 * np.pi * hz * t)[:, None]
    return normalized


def middle(result, t):
    """The evaluation point nearest the middle of the recording."""
    return int(np.argmin(np.abs(result["t"] - t[-1] / 2)))


# --- Breathing share --------------------------------------------------------------------

def test_a_breathing_rhythm_gives_a_large_share_at_its_rate():
    t = timeline(120.0)
    normalized = breathe(still_room(t), t, 0.25)
    result = frequency.breathing_evidence(t, normalized, frequency.presence_times(t, 1.0), PARAMS)
    i = middle({"t": frequency.presence_times(t, 1.0)}, t)
    assert result["share"][i] > 5 * result["flat_share"]
    assert result["peak_hz"][i] == pytest.approx(0.25, abs=result["df_hz"])


def test_noise_alone_stays_near_the_flat_share():
    t = timeline(120.0)
    centers = frequency.presence_times(t, 1.0)
    result = frequency.breathing_evidence(t, still_room(t), centers, PARAMS)
    assert np.nanmedian(result["share"]) < 2.5 * result["flat_share"]


def test_slow_drift_is_removed_before_the_spectrum():
    t = timeline(120.0)
    drifting = still_room(t) * (1 + 0.05 * (t / t[-1]))[:, None].astype(np.float32)
    drifting[:, :10] *= (1 - 0.04 * (t / t[-1]))[:, None]
    centers = frequency.presence_times(t, 1.0)
    result = frequency.breathing_evidence(t, drifting, centers, PARAMS)
    assert np.nanmedian(result["share"]) < 2.5 * result["flat_share"]


def test_the_breathing_window_must_fit_inside_the_recording():
    t = timeline(60.0)
    centers = frequency.presence_times(t, 1.0)
    result = frequency.breathing_evidence(t, breathe(still_room(t), t, 0.25), centers, PARAMS)
    half = PARAMS.breathing_window_s / 2
    assert np.isnan(result["share"][centers < half - 0.01]).all()
    assert np.isnan(result["share"][centers > t[-1] - half + 0.01]).all()
    assert np.isfinite(result["share"][(centers >= half + 1) & (centers <= t[-1] - half - 1)]).all()


def test_a_long_gap_leaves_no_breathing_value_for_windows_covering_it():
    t = timeline(120.0)
    normalized = breathe(still_room(t), t, 0.25)
    keep = (t < 50.0) | (t >= 60.0)  # a 10 s gap is a third of each window covering it
    centers = frequency.presence_times(t[keep], 1.0)
    result = frequency.breathing_evidence(t[keep], normalized[keep], centers, PARAMS)
    assert np.isnan(result["share"][(centers > 52) & (centers < 58)]).all()
    assert np.isfinite(result["share"][(centers > 20) & (centers < 30)]).all()


# --- Channel shape and baseline -----------------------------------------------------------

def test_shape_distance_is_small_on_the_baseline_and_grows_after_a_change():
    t = timeline(120.0)
    normalized = still_room(t)
    normalized[t >= 60.0, :5] *= 1.2  # something now reflects differently on a few subcarriers
    result = frequency.compute_presence(t, normalized, np.full(t.size, 1e-3, np.float32))
    baseline = frequency.presence_baseline(result["t"], result["shape"], result["level"], 20.0, 40.0)
    distance, points = baseline["distance"], result["t"]
    assert np.nanmax(distance[(points >= 20) & (points <= 40)]) < 0.5
    assert np.nanmin(distance[(points >= 70) & (points <= 110)]) > 5.0
    assert np.isnan(distance[points < 4.9]).all()  # no full 10 s window yet


def test_baseline_without_any_full_window_is_rejected():
    t = timeline(30.0)
    result = frequency.compute_presence(t, still_room(t), np.full(t.size, 1e-3, np.float32))
    with pytest.raises(ValueError, match="no complete"):
        frequency.presence_baseline(result["t"], result["shape"], result["level"], 0.0, 2.0)


def test_level_is_the_windowed_median_and_the_floor_its_baseline_percentile():
    t = timeline(60.0)
    motion = np.random.default_rng(4).exponential(1e-3, t.size).astype(np.float32)
    result = frequency.compute_presence(t, still_room(t), motion)
    i = int(np.argmin(np.abs(result["t"] - 30.0)))
    window = (t >= 25.0) & (t <= 35.0)
    assert result["level"][i] == pytest.approx(np.median(motion[window]), rel=1e-6)
    baseline = frequency.presence_baseline(result["t"], result["shape"], result["level"], 10.0, 40.0)
    rows = (result["t"] >= 10.0) & (result["t"] <= 40.0)
    assert baseline["level_floor"] == pytest.approx(np.percentile(result["level"][rows], 95), rel=1e-6)


def test_all_features_share_one_time_grid_in_csi_time():
    t = timeline(45.0) + 7.3
    result = frequency.compute_presence(t, still_room(t), np.full(t.size, 1e-3, np.float32))
    assert result["t"][0] == pytest.approx(8.0)
    assert np.diff(result["t"]) == pytest.approx(1.0)
    for name in ("breathing_share", "breathing_hz", "level"):
        assert result[name].shape == result["t"].shape
    assert result["shape"].shape == (result["t"].size, K)
