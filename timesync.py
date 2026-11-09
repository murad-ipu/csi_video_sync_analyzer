"""Time alignment between the camera, the CSI stream, and the PC clock.

No hardware or file access here, so everything can be unit tested with
synthetic data.

Sign convention used everywhere: a positive sync offset moves CSI LATER
relative to the video, so a CSI frame lines up with the video frame whose time
equals csi_t + offset.
"""

from dataclasses import asdict, dataclass

import numpy as np

MIN_FRAMES_PER_WINDOW = 5
MAX_FIT_WINDOWS = 1500  # keeps the O(n^2) Theil-Sen slope fast
MAX_PLAUSIBLE_DRIFT_PPM = 500.0


@dataclass
class TimeFit:
    """Linear map from device microseconds to PC nanoseconds."""

    device_ref_us: int
    pc_ref_ns: int
    slope_ns_per_us: float
    offset_ns: float
    drift_ppm: float
    residual_rms_ms: float
    residual_p95_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    n_frames: int
    n_windows: int
    method: str
    ok: bool

    def to_pc_ns(self, device_us) -> np.ndarray:
        """Convert device ticks to corrected PC time (csi_t_ns)."""
        relative_us = np.asarray(device_us, dtype=np.int64) - self.device_ref_us
        relative_ns = self.offset_ns + self.slope_ns_per_us * relative_us.astype(np.float64)
        return self.pc_ref_ns + np.rint(relative_ns).astype(np.int64)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_device_to_pc(device_us, pc_receive_ns, window_s: float = 1.0) -> TimeFit:
    """Fit pc_time = offset + slope * device_time from bursty arrivals.

    Each record arrives at its true time plus a variable delay that is never
    negative (Wi-Fi retries, TCP batching, Python scheduling). The earliest
    arrival in each short window is therefore the closest one to the true
    clock relation, so the line is fitted through those. Theil-Sen (the median
    of all pairwise slopes) ignores windows with no fast packet.
    """
    device_us = np.asarray(device_us, dtype=np.int64)
    pc_receive_ns = np.asarray(pc_receive_ns, dtype=np.int64)
    if device_us.size != pc_receive_ns.size:
        raise ValueError("device_us and pc_receive_ns must have the same length")
    if device_us.size == 0:
        return TimeFit(0, 0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "empty", False)

    # Work relative to the first sample: floats stay accurate that way.
    device_ref_us, pc_ref_ns = int(device_us[0]), int(pc_receive_ns[0])
    x_us = (device_us - device_ref_us).astype(np.float64)
    y_ns = (pc_receive_ns - pc_ref_ns).astype(np.float64)

    # Long recordings use wider windows so the pairwise fit stays small.
    span_s = float(x_us.max() - x_us.min()) / 1e6
    window_us = max(window_s, span_s / MAX_FIT_WINDOWS) * 1e6
    bins = np.floor(x_us / window_us).astype(np.int64)
    delay_ns = y_ns - 1000.0 * x_us

    # Sorting by (bin, delay) puts each window's earliest arrival first.
    order = np.lexsort((delay_ns, bins))
    unique_bins, first, counts = np.unique(bins[order], return_index=True, return_counts=True)
    keep = counts >= MIN_FRAMES_PER_WINDOW
    chosen = order[first[keep]]
    x_win, y_win = x_us[chosen], y_ns[chosen]

    if x_win.size >= 3:
        i, j = np.triu_indices(x_win.size, k=1)
        dx = x_win[j] - x_win[i]
        valid = dx > 0
        slope = float(np.median((y_win[j] - y_win[i])[valid] / dx[valid]))
        method = "theil_sen_earliest_arrivals"
    else:
        # Too short to measure drift: only estimate the offset.
        slope = 1000.0
        method = "offset_only"
        x_win, y_win = x_us[[int(np.argmin(delay_ns))]], y_ns[[int(np.argmin(delay_ns))]]

    offset = float(np.median(y_win - slope * x_win))
    residual_ms = (y_win - (offset + slope * x_win)) / 1e6
    latency_ms = (y_ns - (offset + slope * x_us)) / 1e6
    drift_ppm = (slope / 1000.0 - 1.0) * 1e6

    return TimeFit(
        device_ref_us=device_ref_us,
        pc_ref_ns=pc_ref_ns,
        slope_ns_per_us=slope,
        offset_ns=offset,
        drift_ppm=drift_ppm,
        residual_rms_ms=float(np.sqrt(np.mean(residual_ms**2))),
        residual_p95_ms=float(np.percentile(np.abs(residual_ms), 95)),
        latency_median_ms=float(np.median(latency_ms)),
        latency_p95_ms=float(np.percentile(latency_ms, 95)),
        n_frames=int(device_us.size),
        n_windows=int(x_win.size),
        method=method,
        ok=method != "offset_only" and abs(drift_ppm) <= MAX_PLAUSIBLE_DRIFT_PPM,
    )


def nearest_index(sorted_values, queries) -> np.ndarray:
    """Index of the nearest value in a sorted array, for each query."""
    values = np.asarray(sorted_values)
    queries = np.asarray(queries)
    if values.size == 0:
        raise ValueError("sorted_values is empty")
    if values.size == 1:
        return np.zeros(queries.shape, dtype=np.int64)
    right = np.clip(np.searchsorted(values, queries), 1, values.size - 1)
    left = right - 1
    choose_left = (queries - values[left]) <= (values[right] - queries)
    return np.where(choose_left, left, right)


def frame_to_csi_index(frame_t_ns, csi_t_ns, sync_offset_ms: float = 0.0) -> np.ndarray:
    """Nearest CSI index for each video frame time, honoring the sync offset."""
    offset_ns = int(round(sync_offset_ms * 1e6))
    return nearest_index(np.asarray(csi_t_ns, np.int64), np.asarray(frame_t_ns, np.int64) - offset_ns)


def media_time_to_frame(pts_us, media_time_s) -> np.ndarray:
    """Video frame index for a browser mediaTime, using the real frame PTS."""
    return nearest_index(np.asarray(pts_us, np.float64), np.asarray(media_time_s, np.float64) * 1e6)


def count_frame_gaps(t_ns) -> int:
    """Estimate frames missing from a stream, using its own median interval.

    Using the measured interval means a webcam that settles at 15 fps in low
    light is not reported as dropping every second frame.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    if t_ns.size < 3:
        return 0
    intervals = np.diff(t_ns).astype(np.float64)
    median = float(np.median(intervals))
    if median <= 0:
        return 0
    late = intervals[intervals > 1.5 * median]
    return int(np.sum(np.rint(late / median) - 1))


def estimate_offset_ms(video_t_s, video_motion, csi_t_s, csi_motion,
                       max_lag_ms: float = 1000.0, step_ms: float = 10.0,
                       smooth_s: float = 0.2) -> dict:
    """Find the offset that best aligns CSI motion with video motion energy.

    Both signals are resampled onto one grid and smoothed the same way, then
    Pearson correlation is computed for every candidate shift of the CSI.
    """
    video_t_s, csi_t_s = np.asarray(video_t_s, float), np.asarray(csi_t_s, float)
    start, end = max(video_t_s[0], csi_t_s[0]), min(video_t_s[-1], csi_t_s[-1])
    step_s = step_ms / 1000.0
    if end - start < 2.0:
        return {"offset_ms": 0.0, "correlation": 0.0, "confident": False, "reason": "overlap under 2 s"}

    grid = np.arange(start, end, step_s)
    video = np.interp(grid, video_t_s, np.asarray(video_motion, float))
    csi = np.interp(grid, csi_t_s, np.asarray(csi_motion, float))
    kernel = np.ones(max(1, int(round(smooth_s / step_s))))
    kernel /= kernel.size
    video, csi = np.convolve(video, kernel, "same"), np.convolve(csi, kernel, "same")

    max_shift = min(int(round(max_lag_ms / step_ms)), grid.size // 3)
    shifts = np.arange(-max_shift, max_shift + 1)
    scores = np.zeros(shifts.size)
    for n, shift in enumerate(shifts):
        # shift > 0 moves CSI later: csi[i] is compared with video[i + shift].
        a = video[max(shift, 0): grid.size + min(shift, 0)]
        b = csi[max(-shift, 0): grid.size - max(shift, 0)]
        if a.std() > 0 and b.std() > 0:
            scores[n] = float(np.corrcoef(a, b)[0, 1])

    best = int(np.argmax(scores))
    at_edge = best in (0, shifts.size - 1)
    correlation = float(scores[best])
    confident = correlation >= 0.3 and not at_edge
    reason = "ok" if confident else ("best shift is at the search limit" if at_edge else "weak correlation")
    return {
        "offset_ms": float(shifts[best] * step_ms),
        "correlation": correlation,
        "confident": confident,
        "reason": reason,
    }
