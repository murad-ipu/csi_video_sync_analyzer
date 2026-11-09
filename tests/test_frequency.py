"""Tests for the frequency view: resampling, gaps, STFT timing, peaks, cache, API."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import frequency
import make_demo_session
import player
from frequency import FrequencyParams, compute_spectra, resample_uniform

FS = 100.0
K = 20
SECONDS = 120.0


def timeline(seconds=SECONDS):
    return np.arange(int(seconds * FS)) / FS


def noisy_base(t):
    """Flat subcarriers with a little noise, plus two null bins that must be dropped."""
    rng = np.random.default_rng(5)
    amplitude = 20.0 * (1 + 0.002 * rng.standard_normal((t.size, K))) * np.linspace(0.8, 1.2, K)
    amplitude[:, [0, 11]] = 0
    return amplitude


def modulate(amplitude, t, hz, columns, depth=0.15):
    amplitude[:, columns] *= 1 + depth * np.sin(2 * np.pi * hz * t)[:, None]


def peak_hz(result, low=0.1, high=5.0):
    """Frequency of the strongest bin, from the median spectrum over time."""
    spectrum = np.nanmedian(result["power"], axis=0)
    freqs = result["freqs"]
    inside = np.flatnonzero((freqs >= low) & (freqs <= high))
    return freqs[inside[np.argmax(spectrum[inside])]]


def top_two_hz(result, df):
    """The two strongest separated peaks, lowest first."""
    spectrum = np.nanmedian(result["power"], axis=0).copy()
    freqs = result["freqs"]
    spectrum[freqs < 0.1] = -np.inf
    spectrum[freqs > 5.0] = -np.inf
    first = int(np.argmax(spectrum))
    spectrum[np.abs(freqs - freqs[first]) <= 4 * df] = -np.inf
    return sorted((float(freqs[first]), float(freqs[int(np.argmax(spectrum))])))


# --- Peaks ----------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["mean", "pca1"])
@pytest.mark.parametrize("hz", [1.0, 0.25])
def test_a_sine_on_some_subcarriers_peaks_at_its_frequency(method, hz):
    t = timeline()
    amplitude = noisy_base(t)
    modulate(amplitude, t, hz, [2, 3, 4, 5, 6, 7])
    params = FrequencyParams(window_s=20.0, hop_s=0.5, method=method)
    result = compute_spectra(t, amplitude, params)
    assert peak_hz(result) == pytest.approx(hz, abs=result["info"]["df_hz"])


@pytest.mark.parametrize("method", ["mean", "pca1"])
def test_both_sines_appear_together_on_the_same_subcarriers(method):
    t = timeline()
    amplitude = noisy_base(t)
    modulate(amplitude, t, 1.0, [2, 3, 4, 5, 6, 7])
    modulate(amplitude, t, 0.25, [2, 3, 4, 5, 6, 7])
    params = FrequencyParams(window_s=20.0, hop_s=0.5, method=method)
    result = compute_spectra(t, amplitude, params)
    df = result["info"]["df_hz"]
    low, high = top_two_hz(result, df)
    assert low == pytest.approx(0.25, abs=df)
    assert high == pytest.approx(1.0, abs=df)


def test_pca_separates_sines_on_different_subcarriers():
    t = timeline()
    amplitude = noisy_base(t)
    modulate(amplitude, t, 1.0, [2, 3, 4, 5, 6, 7], depth=0.2)
    modulate(amplitude, t, 0.25, [12, 13, 14, 15, 16, 17], depth=0.08)
    first = compute_spectra(t, amplitude, FrequencyParams(window_s=20.0, hop_s=0.5, method="pca1"))
    second = compute_spectra(t, amplitude, FrequencyParams(window_s=20.0, hop_s=0.5, method="pca2"))
    df = first["info"]["df_hz"]
    assert peak_hz(first) == pytest.approx(1.0, abs=df)
    assert peak_hz(second) == pytest.approx(0.25, abs=df)


def test_band_power_is_highest_in_the_band_holding_the_sine():
    t = timeline()
    amplitude = noisy_base(t)
    modulate(amplitude, t, 5.0, [2, 3, 4, 5, 6, 7])
    result = compute_spectra(t, amplitude, FrequencyParams(window_s=2.0, hop_s=0.5))
    names = [band["name"] for band in result["info"]["bands"]]
    assert names[int(np.argmax(np.nanmedian(result["band_power"], axis=0)))] == "movement"


def test_a_band_with_no_bins_is_reported_and_stays_nan():
    t = timeline(10.0)
    result = compute_spectra(t, noisy_base(t), FrequencyParams(window_s=0.5, hop_s=0.5))
    breathing = result["info"]["bands"][0]
    assert breathing["name"] == "breathing" and breathing["n_bins"] == 0
    assert np.isnan(result["band_power"][:, 0]).all()


# --- Resampling and gaps --------------------------------------------------------------

def test_resampling_interpolates_linearly_onto_an_exact_grid():
    t = np.array([0.0, 0.013, 0.031, 0.05])
    values = (t * 1000 + 5)[:, None].astype(np.float32)
    grid, resampled, missing = resample_uniform(t, values, FS, 0.05)
    assert grid == pytest.approx([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    assert resampled[:, 0] == pytest.approx(grid * 1000 + 5, abs=1e-3)
    assert not missing.any()


def test_a_short_gap_is_interpolated_and_a_long_gap_is_not():
    t = np.concatenate((np.arange(0, 1, 0.01), np.arange(1.04, 2, 0.01), np.arange(2.5, 3, 0.01)))
    values = np.full((t.size, 3), 7.0, np.float32)
    grid, resampled, missing = resample_uniform(t, values, FS, 0.05)
    short = (grid > 1.0) & (grid < 1.03)
    long = (grid > 2.0) & (grid < 2.49)
    assert short.any() and not missing[short].any()
    assert (resampled[short] == pytest.approx(7.0))
    assert long.any() and missing[long].all()
    assert (resampled[long] == 0).all()


def test_long_gaps_become_missing_windows_and_nothing_is_invented():
    t = timeline(30.0)
    amplitude = noisy_base(t)
    keep = (t < 10.0) | (t >= 14.0)
    params = FrequencyParams(window_s=2.0, hop_s=0.1)
    result = compute_spectra(t[keep], amplitude[keep], params)
    columns = result["t"]
    n_window = 200
    # A window may hold up to 10% missing samples (0.2 s) and still be used.
    limit = 0.1 * n_window / FS
    overlap = np.minimum(columns + 1.0, 14.0) - np.maximum(columns - 1.0, 10.0)
    tolerance = 2.0 / FS
    assert np.isnan(result["power"][overlap > limit + tolerance]).all()
    assert np.isnan(result["band_power"][overlap > limit + tolerance]).all()
    assert np.isfinite(result["power"][overlap < limit - tolerance]).all()
    assert result["info"]["n_skipped"] > 0


def test_missing_samples_in_a_usable_window_do_not_change_the_noise_level():
    t = timeline(60.0)
    amplitude = noisy_base(t)
    params = FrequencyParams(window_s=2.0, hop_s=0.5)
    full = compute_spectra(t, amplitude, params)
    # 0.1 s holes: over the 50 ms limit, but under 10% of each 2 s window.
    keep = np.ones(t.size, bool)
    for start in np.arange(1.0, 59.0, 7.0):
        keep[(t >= start) & (t < start + 0.1)] = False
    holed = compute_spectra(t[keep], amplitude[keep], params)
    assert holed["info"]["n_skipped"] == 0
    assert np.nanmedian(holed["power"]) == pytest.approx(np.nanmedian(full["power"]), abs=1.0)


def test_an_all_zero_frame_counts_as_lost():
    t = timeline(10.0)
    amplitude = noisy_base(t)
    amplitude[300:320] = 0
    result = compute_spectra(t, amplitude, FrequencyParams(window_s=1.0, hop_s=0.1))
    grid, missing = result["grid_t"], result["missing"]
    assert missing[(grid > 3.02) & (grid < 3.17)].all()


# --- Timing ---------------------------------------------------------------------------

@pytest.mark.parametrize("window_s,hop_s", [(2.0, 0.1), (0.5, 0.05), (5.0, 0.5)])
def test_column_time_is_the_window_center(window_s, hop_s):
    t = timeline(30.0) + 3.21
    result = compute_spectra(t, noisy_base(t), FrequencyParams(window_s=window_s, hop_s=hop_s))
    n, hop = int(round(window_s * FS)), int(round(hop_s * FS))
    grid = result["grid_t"]
    for column in (0, 1, 7):
        samples = grid[column * hop: column * hop + n]
        assert result["t"][column] == pytest.approx((samples[0] + samples[-1]) / 2, abs=1e-9)


def test_a_burst_peaks_in_the_column_centered_on_it():
    t = timeline(30.0)
    amplitude = noisy_base(t)
    burst = (t > 14.9) & (t < 15.1)
    amplitude[burst, 2:8] *= 1 + 0.4 * np.sin(2 * np.pi * 20 * t[burst])[:, None]
    result = compute_spectra(t, amplitude, FrequencyParams(window_s=2.0, hop_s=0.1))
    assert result["t"][np.nanargmax(result["band_power"][:, 3])] == pytest.approx(15.0, abs=0.1)


def test_a_recording_shorter_than_the_window_has_no_columns():
    t = timeline(3.0)
    result = compute_spectra(t, noisy_base(t), FrequencyParams(window_s=10.0, hop_s=0.1))
    assert result["info"]["n_columns"] == 0 and result["power"].shape[0] == 0


# --- Bands ----------------------------------------------------------------------------

def test_parse_bands_accepts_session_json_shape():
    assert frequency.parse_bands([{"name": "a", "low_hz": 1, "high_hz": 2}]) == (("a", 1.0, 2.0),)
    assert frequency.parse_bands(None) == frequency.DEFAULT_BANDS


@pytest.mark.parametrize("bad", [[], "x", [{"name": "a"}], [{"name": "a", "low_hz": 3, "high_hz": 2}]])
def test_parse_bands_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        frequency.parse_bands(bad)


# --- Web API and cache ----------------------------------------------------------------

@pytest.fixture(scope="module")
def session(tmp_path_factory):
    folder = tmp_path_factory.mktemp("freq") / "s1"
    make_demo_session.make_session(folder, 20.0, 0.0, width=320, height=240)
    return folder, player.load_session(folder)


@pytest.fixture(scope="module")
def client(session):
    return TestClient(player.create_app(session[1]))


REQUEST = {"window_s": 2.0, "hop_s": 0.1, "method": "mean"}
SETTINGS = {**REQUEST, "freq_max_hz": 20, "color_auto": False, "color_min": -50, "color_max": -5}


def test_api_returns_binary_arrays_of_the_stated_size(client):
    info = client.post("/api/frequency", json=REQUEST).json()
    key = info["key"]
    power = np.frombuffer(client.get(f"/api/frequency/{key}/power").content, "<f4")
    assert power.size == info["n_columns"] * info["n_freqs"] > 0
    assert np.frombuffer(client.get(f"/api/frequency/{key}/t").content, "<f8").size == info["n_columns"]
    assert np.frombuffer(client.get(f"/api/frequency/{key}/freqs").content, "<f8")[-1] == pytest.approx(50.0)
    bands = np.frombuffer(client.get(f"/api/frequency/{key}/band_power").content, "<f4")
    assert bands.size == info["n_columns"] * len(info["bands"])
    assert info["df_hz"] == pytest.approx(0.5)


def test_results_are_cached_per_parameter_set(client):
    settings = {"window_s": 1.0, "hop_s": 0.2, "method": "pca1"}
    first = client.post("/api/frequency", json=settings).json()
    other = client.post("/api/frequency", json={**settings, "method": "pca2"}).json()
    again = client.post("/api/frequency", json=settings).json()
    assert not first["cached"] and not other["cached"] and again["cached"]
    assert first["key"] == again["key"] != other["key"]


def test_a_damaged_cache_file_is_recomputed(client, session):
    folder, _ = session
    settings = {"window_s": 5.0, "hop_s": 0.5, "method": "mean"}
    key = client.post("/api/frequency", json=settings).json()["key"]
    frequency.cache_path(folder, key).write_bytes(b"not a zip file")
    assert client.post("/api/frequency", json=settings).json()["cached"] is False


def test_csi_times_in_the_response_exclude_the_sync_offset(client):
    """The page shifts columns itself, so the offset must not change the result."""
    before = client.post("/api/frequency", json=REQUEST).json()["key"]
    client.post("/api/sync_offset", json={"sync_offset_ms": 250})
    after = client.post("/api/frequency", json=REQUEST).json()
    client.post("/api/sync_offset", json={"sync_offset_ms": 0})
    assert before == after["key"] and after["cached"]


@pytest.mark.parametrize("bad", [
    {"window_s": 3.0, "hop_s": 0.1, "method": "mean"},
    {"window_s": 2.0, "hop_s": 0.01, "method": "mean"},
    {"window_s": 2.0, "hop_s": 0.1, "method": "ica"},
])
def test_bad_parameters_are_rejected(client, bad):
    assert client.post("/api/frequency", json=bad).status_code == 422


@pytest.mark.parametrize("path", ["0123456789abcdef0123/power", "0123456789abcdef0123/secret", "not-a-key/power"])
def test_unknown_results_are_404(client, path):
    assert client.get(f"/api/frequency/{path}").status_code == 404


def test_settings_are_saved_separately_and_other_fields_are_kept(client, session):
    folder, _ = session
    response = client.post("/api/player_settings", json={"mode": "frequency", "frequency": SETTINGS})
    assert response.status_code == 200
    saved = json.loads((folder / "session.json").read_text())
    assert saved["status"] == "synthetic" and saved["sync_offset_ms"] == 0
    stored = saved["player_settings"]
    assert stored["mode"] == "frequency" and stored["frequency"]["color_max"] == -5
    assert stored["frequency"]["bands"][0]["name"] == "breathing"
    assert client.get("/api/meta").json()["player_settings"]["mode"] == "frequency"


def test_hand_edited_bands_survive_saves_and_are_used(client, session):
    folder, _ = session
    path = folder / "session.json"
    client.post("/api/player_settings", json={"mode": "time", "frequency": SETTINGS})
    saved = json.loads(path.read_text())
    original = saved["player_settings"]["frequency"]["bands"]
    saved["player_settings"]["frequency"]["bands"] = [{"name": "custom", "low_hz": 1, "high_hz": 3}]
    path.write_text(json.dumps(saved))

    client.post("/api/player_settings", json={"mode": "frequency", "frequency": SETTINGS})
    assert json.loads(path.read_text())["player_settings"]["frequency"]["bands"][0]["name"] == "custom"
    assert client.post("/api/frequency", json=REQUEST).json()["bands"][0]["name"] == "custom"

    saved = json.loads(path.read_text())
    saved["player_settings"]["frequency"]["bands"] = original
    path.write_text(json.dumps(saved))


def test_invalid_hand_edited_bands_give_a_readable_error(client, session):
    folder, _ = session
    path = folder / "session.json"
    original = path.read_text()
    saved = json.loads(original)
    saved.setdefault("player_settings", {}).setdefault("frequency", {})["bands"] = [{"name": "x", "low_hz": 5, "high_hz": 1}]
    path.write_text(json.dumps(saved))
    response = client.post("/api/frequency", json=REQUEST)
    path.write_text(original)
    assert response.status_code == 400 and "low_hz" in response.json()["detail"]


def test_recorded_files_are_not_touched_by_the_frequency_view(client, session):
    folder, _ = session
    names = ("csi.npz", "frames.csv", "video.mp4", "csi_meta.json")
    before = {name: (folder / name).read_bytes() for name in names}
    client.post("/api/frequency", json={"window_s": 10.0, "hop_s": 0.5, "method": "pca2"})
    assert before == {name: (folder / name).read_bytes() for name in names}
