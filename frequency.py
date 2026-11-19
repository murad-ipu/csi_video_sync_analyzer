"""Frequency-domain view of CSI amplitude, used only by the player.

The steps, in order: resample onto a uniform grid, normalize each frame, then
for every analysis window remove the mean, apply a Hann window, take the FFT,
combine the subcarriers, and convert to dB.

The "motion" method analyzes the player's motion signal instead: per-frame
normalization and the 0.5 s sliding variance happen on the CSI frames first,
exactly as for the Time-mode chart, and that single signal is resampled.

All column times are in CSI time, without the sync offset. The page adds the
offset when drawing, as it does for the motion chart, so moving the offset
slider never needs a recompute.
"""

import hashlib
import json
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CACHE_DIR_NAME = "frequency_cache"
CACHE_VERSION = 1           # raise this when the processing changes
CACHE_ARRAYS = {"t": "f8", "freqs": "f8", "power": "f4", "band_power": "f4"}

SAMPLE_RATE_HZ = 100.0
MAX_GAP_S = 0.05
MAX_MISSING_FRACTION = 0.10
WINDOW_CHOICES_S = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
METHODS = ("mean", "pca1", "pca2", "motion")
PCA_COMPONENT = {"pca1": 1, "pca2": 2}
MOTION_WINDOW_S = 0.5       # the motion chart's window; the motion method uses the same
DEFAULT_BANDS = (
    ("breathing", 0.1, 0.5),
    ("slow movement", 0.5, 2.0),
    ("movement", 2.0, 10.0),
    ("fast/noise", 10.0, 50.0),
)
MAX_BANDS = 8               # the page has eight distinct line colors

MIN_NONZERO_FRACTION = 0.5
POWER_FLOOR = 1e-15         # keeps log10 finite for a perfectly flat signal
CHUNK_BYTES = 16 * 1024 * 1024   # windows handled by one worker at a time
WORKERS = max(1, (os.cpu_count() or 2) - 1)  # one core stays free for video playback
FLOAT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FrequencyParams:
    window_s: float = 2.0
    hop_s: float = 0.1
    method: str = "mean"
    fs_hz: float = SAMPLE_RATE_HZ
    max_gap_s: float = MAX_GAP_S
    max_missing: float = MAX_MISSING_FRACTION
    bands: tuple = DEFAULT_BANDS
    motion_window_s: float = MOTION_WINDOW_S


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


def parse_bands(raw) -> tuple:
    """Turn the bands from session.json into tuples, or raise ValueError."""
    if raw is None:
        return DEFAULT_BANDS
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_BANDS:
        raise ValueError(f"player_settings.frequency.bands must be a list of 1 to {MAX_BANDS} bands")
    bands = []
    for item in raw:
        try:
            band = (str(item["name"]), float(item["low_hz"]), float(item["high_hz"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Bad band {item!r}: needs name, low_hz and high_hz") from exc
        if not 0 <= band[1] < band[2]:
            raise ValueError(f"Bad band {item!r}: needs 0 <= low_hz < high_hz")
        bands.append(band)
    return tuple(bands)


def bands_to_json(bands: tuple) -> list:
    return [{"name": name, "low_hz": low, "high_hz": high} for name, low, high in bands]


def resample_uniform(t_s: np.ndarray, amplitude: np.ndarray, fs_hz: float, max_gap_s: float):
    """Linear interpolation onto an exact grid. Returns (grid, resampled, missing).

    A grid point between two frames further apart than max_gap_s is marked
    missing and holds zeros, so no invented values can reach the spectra.
    """
    if t_s.size < 2:
        return np.zeros(0), np.zeros((0, amplitude.shape[1]), np.float32), np.zeros(0, bool)
    count = int(np.floor((t_s[-1] - t_s[0]) * fs_hz + FLOAT_TOLERANCE)) + 1
    grid = t_s[0] + np.arange(count) / fs_hz
    right = np.searchsorted(t_s, grid, side="right").clip(1, t_s.size - 1)
    left = right - 1
    gap = t_s[right] - t_s[left]
    missing = gap > max_gap_s + FLOAT_TOLERANCE
    fraction = np.divide(grid - t_s[left], gap, out=np.zeros_like(grid), where=gap > 0).clip(0, 1)
    data = amplitude.astype(np.float32)
    resampled = data[left] + (data[right] - data[left]) * fraction[:, None].astype(np.float32)
    resampled[missing] = 0
    return grid, resampled, missing


def _remove_window_mean(windows: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Detrend each subcarrier using valid samples only; missing samples become 0."""
    weights = valid[:, None, :].astype(np.float32)
    mean = (windows * weights).sum(axis=-1, keepdims=True) / weights.sum(axis=-1, keepdims=True)
    return (windows - mean) * weights


def _spectral_density(series: np.ndarray, hann: np.ndarray, valid: np.ndarray, fs_hz: float) -> np.ndarray:
    """One-sided power density of (windows, channels, samples), shape (windows, channels, bins).

    Dividing by the Hann energy of the valid samples corrects for missing ones,
    and keeps the noise floor at the same level for every window length.
    """
    spectrum = np.fft.rfft(series * hann, axis=-1)
    energy = ((valid * hann) ** 2).sum(axis=-1)
    density = (spectrum.real**2 + spectrum.imag**2) / (fs_hz * energy[:, None, None])
    last = density.shape[-1] - 1 if hann.size % 2 == 0 else density.shape[-1]
    density[..., 1:last] *= 2  # DC and Nyquist have no mirrored negative frequency
    return density


def _project_on_component(windows: np.ndarray, component: int) -> np.ndarray:
    """Project (windows, K, samples) on one principal component, per window."""
    covariance = windows @ windows.transpose(0, 2, 1)
    _, vectors = np.linalg.eigh(covariance)  # eigenvalues come back in ascending order
    direction = vectors[:, :, -min(component, windows.shape[1])]
    return direction[:, None, :] @ windows


def _to_db(power: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(power, POWER_FLOOR))).astype(np.float32)


def _band_masks(freqs: np.ndarray, bands: tuple) -> list:
    # A bin on a shared edge belongs to the lower band, and 0 Hz belongs to none.
    return [(freqs > low + FLOAT_TOLERANCE) & (freqs <= high + FLOAT_TOLERANCE) for _, low, high in bands]


def prepare_grid(t_s: np.ndarray, amplitude: np.ndarray, params: FrequencyParams):
    """The uniform-grid input to the STFT: (grid, channels, missing).

    channels is (samples, 52-ish subcarriers) of normalized amplitude, or for
    the motion method (samples, 1) holding the motion signal.
    """
    kept = select_subcarriers(amplitude)
    measured = amplitude[:, kept]
    has_signal = measured.sum(axis=1) > 0  # an all-zero frame is a lost frame, so it becomes a gap
    t_used, a_used = t_s[has_signal], measured[has_signal]
    if params.method == "motion":
        # Same order as the Time-mode chart: normalize, then the sliding variance, on the real frames.
        motion = motion_signal(t_used, normalize_per_frame(a_used.astype(np.float32)), params.motion_window_s)
        # One channel: dividing it by its own mean would turn every sample into 1.
        return resample_uniform(t_used, motion[:, None], params.fs_hz, params.max_gap_s)
    grid, resampled, missing = resample_uniform(t_used, a_used, params.fs_hz, params.max_gap_s)
    return grid, normalize_per_frame(resampled), missing


def compute_spectra(t_s: np.ndarray, amplitude: np.ndarray, params: FrequencyParams) -> dict:
    """Run the whole analysis. power is (columns, bins) in dB; a skipped window is NaN."""
    grid, normalized, missing = prepare_grid(t_s, amplitude, params)

    n = max(2, int(round(params.window_s * params.fs_hz)))
    hop = max(1, int(round(params.hop_s * params.fs_hz)))
    freqs = np.fft.rfftfreq(n, 1.0 / params.fs_hz)
    df = params.fs_hz / n
    starts = np.arange(0, max(0, grid.size - n + 1), hop)
    columns = starts.size
    # np.hanning is symmetric, so the middle of a window is sample (n - 1) / 2.
    centers = grid[0] + (starts + (n - 1) / 2) / params.fs_hz if columns else np.zeros(0)

    masks = _band_masks(freqs, params.bands)
    power = np.full((columns, freqs.size), np.nan, np.float32)
    band_power = np.full((columns, len(params.bands)), np.nan, np.float32)

    usable = np.zeros(0, int)
    if columns:
        missing_sum = np.concatenate(([0], np.cumsum(missing)))
        missing_count = missing_sum[starts + n] - missing_sum[starts]
        usable = np.flatnonzero(missing_count <= params.max_missing * n + FLOAT_TOLERANCE)
    if usable.size:
        K = normalized.shape[1]
        component = PCA_COMPONENT.get(params.method, 0)  # 0: average the channels' spectra
        hann = np.hanning(n).astype(np.float32)
        window_view = np.lib.stride_tricks.sliding_window_view(normalized, n, axis=0)
        valid_view = np.lib.stride_tricks.sliding_window_view(~missing, n)

        def analyze(indices: np.ndarray) -> None:
            # Each chunk writes only its own rows, so the threads need no lock.
            valid = valid_view[starts[indices]]
            windows = _remove_window_mean(window_view[starts[indices]], valid)
            series = _project_on_component(windows, component) if component else windows
            density = _spectral_density(series, hann, valid, params.fs_hz).mean(axis=1)
            power[indices] = _to_db(density)
            for b, mask in enumerate(masks):
                if mask.any():
                    band_power[indices, b] = _to_db(density[:, mask].sum(axis=1) * df)

        chunk = max(1, int(CHUNK_BYTES / (n * K * 4)))
        chunks = [usable[begin:begin + chunk] for begin in range(0, usable.size, chunk)]
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(analyze, chunks))  # list() re-raises any error from a worker

    return {
        "t": centers,
        "freqs": freqs,
        "power": power,
        "band_power": band_power,
        "grid_t": grid,
        "missing": missing,
        "info": {
            "fs_hz": params.fs_hz,
            "window_s": n / params.fs_hz,
            "hop_s": hop / params.fs_hz,
            "df_hz": df,
            "method": params.method,
            "n_columns": columns,
            "n_freqs": int(freqs.size),
            "n_skipped": int(columns - usable.size),
            "bands": [
                {"name": name, "low_hz": low, "high_hz": high, "n_bins": int(mask.sum())}
                for (name, low, high), mask in zip(params.bands, masks)
            ],
        },
    }


# --- Disk cache -----------------------------------------------------------------------

def cache_key(source: Path, params: FrequencyParams) -> str:
    """A name that changes with the parameters and with the CSI file they came from."""
    stat = source.stat()
    description = {"version": CACHE_VERSION, "params": asdict(params), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:20]


def cache_path(folder: Path, key: str) -> Path:
    return folder / CACHE_DIR_NAME / f"{key}.npz"


def _read_cached_info(path: Path) -> dict | None:
    """The stored summary, or None if the file is absent or unreadable."""
    try:
        with np.load(path) as stored:
            return json.loads(str(stored["info"]))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def load_or_compute(folder: Path, t_s: np.ndarray, amplitude: np.ndarray, params: FrequencyParams) -> tuple:
    """Return (key, info, was_cached), computing and storing the result if needed."""
    key = cache_key(folder / "csi.npz", params)
    path = cache_path(folder, key)
    info = _read_cached_info(path)
    if info is not None:
        return key, info, True
    result = compute_spectra(t_s, amplitude, params)
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix(".tmp")
    with open(temp, "wb") as file:
        np.savez(file, info=json.dumps(result["info"]), **{name: result[name] for name in CACHE_ARRAYS})
    os.replace(temp, path)
    return key, result["info"], False


def read_cached_array(folder: Path, key: str, name: str) -> bytes:
    with np.load(cache_path(folder, key)) as stored:
        return stored[name].astype(CACHE_ARRAYS[name]).tobytes()


# --- Presence evidence ------------------------------------------------------------------
# Three slow features meant to tell a still person from an empty room. None of them is a
# validated detector: they are evidence to compare against the video. ESPectre has no
# stationary-presence detector yet (planned for its v3.3.0), so this is the app's own math.
# Every value is computed at one point per step_s from a window centered on that point.

@dataclass(frozen=True)
class PresenceParams:
    step_s: float = 1.0
    breathing_window_s: float = 30.0          # about 6 breaths at 12 per minute
    breathing_fs_hz: float = 10.0             # breathing needs nothing above 2 Hz
    breathing_band_hz: tuple = (0.15, 0.5)    # 9 to 30 breaths per minute
    reference_band_hz: tuple = (0.05, 2.0)    # the share is measured against this band
    peak_bins: int = 1                        # bins either side of the peak that count as the peak
    shape_window_s: float = 10.0
    level_window_s: float = 10.0
    baseline_window_s: float = 30.0           # the empty stretch taken around the cursor
    floor_percentile: float = 95.0
    max_missing: float = MAX_MISSING_FRACTION


def presence_times(t_s: np.ndarray, step_s: float) -> np.ndarray:
    """Evaluation times: whole steps from the first to the last CSI frame."""
    if t_s.size == 0:
        return np.zeros(0)
    first = np.ceil(t_s[0] / step_s) * step_s
    return np.arange(first, t_s[-1] + FLOAT_TOLERANCE, step_s)


def _inside(centers: np.ndarray, t_s: np.ndarray, window_s: float) -> np.ndarray:
    """Points whose whole window lies within the recording."""
    return ((centers - window_s / 2 >= t_s[0] - FLOAT_TOLERANCE)
            & (centers + window_s / 2 <= t_s[-1] + FLOAT_TOLERANCE))


def channel_shape(t_s: np.ndarray, normalized: np.ndarray, centers: np.ndarray, window_s: float) -> np.ndarray:
    """Average normalized amplitude of each subcarrier over the window around each point."""
    data = normalized.astype(np.float64)
    sums = np.concatenate((np.zeros((1, data.shape[1])), np.cumsum(data, axis=0)))
    low = np.searchsorted(t_s, centers - window_s / 2, side="left")
    high = np.searchsorted(t_s, centers + window_s / 2, side="right")
    count = (high - low)[:, None].astype(np.float64)
    shape = np.full((centers.size, data.shape[1]), np.nan)
    np.divide(sums[high] - sums[low], count, out=shape, where=count > 0)
    shape[~_inside(centers, t_s, window_s)] = np.nan
    return shape.astype(np.float32)


def motion_level(t_s: np.ndarray, motion: np.ndarray, centers: np.ndarray, window_s: float) -> np.ndarray:
    """Median of the motion signal over the window around each point.

    The median ignores a few short bursts, so it follows the steady background level:
    an empty room's noise, or the small fidgets of a person keeping still.
    """
    low = np.searchsorted(t_s, centers - window_s / 2, side="left")
    high = np.searchsorted(t_s, centers + window_s / 2, side="right")
    level = np.array([np.median(motion[a:b]) if b > a else np.nan for a, b in zip(low, high)], dtype=np.float32)
    level[~_inside(centers, t_s, window_s)] = np.nan
    return level


def breathing_evidence(t_s: np.ndarray, normalized: np.ndarray, centers: np.ndarray, params: PresenceParams) -> dict:
    """How much of the slow fluctuation sits in one narrow peak in the breathing band.

    Per window: resample to breathing_fs_hz, remove each subcarrier's straight-line
    trend (slow drift), take its Hann-windowed power spectrum, and scale it so that its
    power in the reference band sums to 1. Averaging the scaled spectra over the
    subcarriers gives the share of slow fluctuation at each frequency. The result is
    the largest share held by one peak (the peak bin and its neighbors) inside the
    breathing band. It is a ratio, so it does not depend on signal strength. A flat
    spectrum gives about flat_share; regular breathing gives clearly more.
    """
    fs = params.breathing_fs_hz
    n_points = centers.size
    share = np.full(n_points, np.nan, np.float32)
    peak_hz = np.full(n_points, np.nan, np.float32)
    n = int(round(params.breathing_window_s * fs))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    low_ref, high_ref = params.reference_band_hz
    reference = (freqs >= low_ref - FLOAT_TOLERANCE) & (freqs <= high_ref + FLOAT_TOLERANCE)
    low_band, high_band = params.breathing_band_hz
    band = np.flatnonzero((freqs >= low_band - FLOAT_TOLERANCE) & (freqs <= high_band + FLOAT_TOLERANCE))
    flat_share = (2 * params.peak_bins + 1) / max(1, int(reference.sum()))
    result = {"share": share, "peak_hz": peak_hz, "flat_share": float(flat_share), "df_hz": fs / n}

    # Resample at the full rate, then average blocks down to fs: the block mean keeps
    # faster fluctuations from folding into the breathing band.
    grid, channels, missing = resample_uniform(t_s, normalized, SAMPLE_RATE_HZ, MAX_GAP_S)
    factor = int(round(SAMPLE_RATE_HZ / fs))
    blocks = grid.size // factor
    if blocks < n or n_points == 0:
        return result
    data = channels[: blocks * factor].reshape(blocks, factor, -1).mean(axis=1)
    lost = missing[: blocks * factor].reshape(blocks, factor).any(axis=1)
    block_t = grid[: blocks * factor].reshape(blocks, factor).mean(axis=1)

    starts = np.searchsorted(block_t, centers - params.breathing_window_s / 2 - FLOAT_TOLERANCE)
    usable = np.flatnonzero((starts + n <= blocks) & _inside(centers, t_s, params.breathing_window_s))
    if usable.size:
        lost_sum = np.concatenate(([0], np.cumsum(lost)))
        lost_count = lost_sum[starts[usable] + n] - lost_sum[starts[usable]]
        usable = usable[lost_count <= params.max_missing * n]
    if usable.size == 0:
        return result

    windows = np.lib.stride_tricks.sliding_window_view(data, n, axis=0)[starts[usable]].astype(np.float64)
    valid = ~np.lib.stride_tricks.sliding_window_view(lost, n)[starts[usable]]
    weights = valid.astype(np.float64)[:, None, :]
    # Weighted straight-line fit per subcarrier, on valid samples only.
    u = np.arange(n) - (n - 1) / 2
    total = weights.sum(axis=-1, keepdims=True)
    mean = (windows * weights).sum(axis=-1, keepdims=True) / total
    u_mean = (u * weights).sum(axis=-1, keepdims=True) / total
    du = u - u_mean
    slope = ((weights * du * (windows - mean)).sum(axis=-1, keepdims=True)
             / (weights * du**2).sum(axis=-1, keepdims=True))
    residual = (windows - mean - slope * du) * weights

    power = np.abs(np.fft.rfft(residual * np.hanning(n), axis=-1)) ** 2
    reference_power = power[..., reference].sum(axis=-1, keepdims=True)
    scaled = np.divide(power, reference_power, out=np.zeros_like(power), where=reference_power > 0)
    spectrum = scaled.mean(axis=1)  # over subcarriers
    peak_sums = sum(spectrum[:, band + offset] for offset in range(-params.peak_bins, params.peak_bins + 1))
    best = np.argmax(peak_sums, axis=1)
    share[usable] = peak_sums[np.arange(usable.size), best]
    peak_hz[usable] = freqs[band[best]]
    return result


def compute_presence(t_s: np.ndarray, normalized: np.ndarray, motion: np.ndarray,
                     params: PresenceParams = PresenceParams()) -> dict:
    """All three presence features on one time grid, in CSI time (no sync offset)."""
    centers = presence_times(t_s, params.step_s)
    breathing = breathing_evidence(t_s, normalized, centers, params)
    return {
        "t": centers,
        "breathing_share": breathing["share"],
        "breathing_hz": breathing["peak_hz"],
        "shape": channel_shape(t_s, normalized, centers, params.shape_window_s),
        "level": motion_level(t_s, motion, centers, params.level_window_s),
        "info": {**asdict(params), "flat_share": breathing["flat_share"], "breathing_df_hz": breathing["df_hz"]},
    }


def presence_baseline(t: np.ndarray, shape: np.ndarray, level: np.ndarray, start_s: float, end_s: float,
                      floor_percentile: float = PresenceParams.floor_percentile) -> dict:
    """Compare every point with an empty-room stretch [start_s, end_s] in CSI time.

    distance: root-mean-square difference between each point's channel shape and the
    stretch's average shape, in percent of the normalized amplitude (which averages 1).
    level_floor: the stretch's motion level at floor_percentile.
    """
    rows = (t >= start_s - FLOAT_TOLERANCE) & (t <= end_s + FLOAT_TOLERANCE)
    rows &= np.isfinite(shape).all(axis=1)
    if not rows.any():
        raise ValueError("That stretch has no complete 10 s windows. Pick a moment further from the start or end.")
    reference = shape[rows].astype(np.float64).mean(axis=0)
    distance = 100.0 * np.sqrt(np.mean((shape.astype(np.float64) - reference) ** 2, axis=1))
    levels = level[rows & np.isfinite(level)]
    return {
        "distance": distance.astype(np.float32),
        "level_floor": float(np.percentile(levels, floor_percentile)) if levels.size else None,
        "n_points": int(rows.sum()),
    }
