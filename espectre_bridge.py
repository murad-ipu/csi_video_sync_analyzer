"""Every calculation this app takes from ESPectre, in one place.

- Per-packet amplitude and phase: tools/lib/csi_io.py, CSIPacket.amplitudes / .phases.
- Motion detector replay: the Lightweight and High-Accuracy detectors
  (tools/lib/lightweight_detector.py, tools/lib/high_accuracy_detector.py),
  driven by ESPectre's own replay parts: the production temporal sampler
  (tools/lib/temporal_replay.py), the 250 ms evaluation cadence and scored-tick
  rule (tools/lib/performance_report.py), startup calibration
  (tools/lib/dataset_metadata.py), and the motion on/off hit filter
  (micro_espectre/runtime_motion_policy.py).

Nothing here reimplements a detector. The replay loop is the same sequence of
calls as performance_report.replay_idle_stream, which only returns booleans; this
one also keeps the score, threshold and readiness of every evaluation. A test
checks that both give the same states.

Call espectre_path.setup() before using anything in this module.
"""

from pathlib import Path

import numpy as np

DETECTORS = ("lightweight", "high_accuracy")

# Effective state per evaluation, as stored and as sent to the page.
STATE_NOT_READY, STATE_IDLE, STATE_MOTION = 0, 1, 2


# --- Per-packet values ---------------------------------------------------------------

def amplitude_phase(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude and phase of every subcarrier, computed by ESPectre's CSIPacket.

    iq is (packets, 2 * subcarriers) int8 in the stored layout (Q first, then I),
    already passed through normalize_stored_csi_bin_layout.
    """
    from tools.lib.csi_io import CSIPacket

    iq = np.asarray(iq, dtype=np.int8)
    count, subcarriers = iq.shape[0], iq.shape[1] // 2
    amplitude = np.empty((count, subcarriers), np.float32)
    phase = np.empty((count, subcarriers), np.float32)
    for index in range(count):
        packet = CSIPacket(timestamp=0.0, seq_num=0, num_subcarriers=subcarriers, iq_raw=iq[index])
        amplitude[index] = packet.amplitudes
        phase[index] = packet.phases
    return amplitude, phase


# --- Detector replay -------------------------------------------------------------------

def board_settings(sensing: dict | None) -> dict:
    """The detection settings the board reported in /sensing before recording."""
    import config

    sensing = sensing or {}
    detector = sensing.get("detector")
    threshold = sensing.get("threshold")
    return {
        "detector": detector if detector in DETECTORS else None,
        "threshold": float(threshold) if isinstance(threshold, (int, float)) else None,
        "motion_on_hits": int(sensing.get("motion_on_hits") or config.MOTION_ON_HITS),
        "motion_off_hits": int(sensing.get("motion_off_hits") or config.MOTION_OFF_HITS),
        "csi_target_pps": int(sensing["csi_target_pps"]) if sensing.get("csi_target_pps") else None,
    }


def espectre_parameters() -> dict:
    """ESPectre configuration values the replay depends on (for the manifest)."""
    import config
    from tools.lib.high_accuracy_detector import HIGH_ACCURACY_DEFAULT_THRESHOLD
    from tools.lib.runtime_policy import production_filter_kwargs

    return {
        "evaluation_interval_ms": int(config.EVALUATION_INTERVAL_MS),
        "segmentation_window_size_ms": int(config.SEGMENTATION_WINDOW_SIZE_MS),
        "calibration_duration_ms": int(config.CALIBRATION_DURATION_MS),
        "default_subcarriers": [int(n) for n in config.DEFAULT_SUBCARRIERS],
        "high_accuracy_default_threshold": float(HIGH_ACCURACY_DEFAULT_THRESHOLD),
        "filters": production_filter_kwargs(),
    }


def load_packets(npz_path: Path, csi_target_pps: int | None = None) -> tuple[list, np.ndarray]:
    """ESPectre's sensing packet view of csi.npz, and the csi.npz row of each packet.

    The view keeps only HT20 sensing packets, so the row index is needed to put
    each evaluation on the right CSI time. Recordings made before csi.npz stored
    csi_target_pps get the board's value from /sensing, so the replay uses the
    board's rate instead of a measured one.
    """
    from tools.lib.csi_io import ht20_packet_mask, load_npz_packet_view

    view = load_npz_packet_view(npz_path)
    with np.load(npz_path) as stored:
        rows, row_len = stored["csi_data"].shape
        mask = ht20_packet_mask(
            stored["phy_mode"] if "phy_mode" in stored else None,
            stored["ltf_type"] if "ltf_type" in stored else None,
            stored["channel_width"] if "channel_width" in stored else None,
            row_len, row_len // 2, rows,
        )
    row_index = np.arange(rows) if mask is None else np.flatnonzero(mask)
    if row_index.size != len(view):
        raise ValueError(f"ESPectre kept {len(view)} packets but the HT20 mask selects {row_index.size}")
    packets = list(view)
    if csi_target_pps and packets and packets[0].get("csi_target_pps") is None:
        packets = [{**packet, "csi_target_pps": int(csi_target_pps)} for packet in packets]
    return packets, row_index


def _replay(detector, packets: list, interval_us: int, warmup: int, on_hits: int, off_hits: int) -> dict:
    """Run one detector over the packets, keeping every evaluation.

    The calls and their order match performance_report.replay_idle_stream.
    """
    import config
    from detector_interface import MotionState
    from tools.lib import performance_report as report
    from tools.lib.runtime_policy import RuntimeMotionPolicy
    from tools.lib.temporal_replay import apply_temporal_admission, iter_temporal_admissions, target_pps_for_packets

    target_pps = target_pps_for_packets(packets, interval_us)
    _, cadence = report.timing_cadence_for_window(warmup, interval_us)
    hits = RuntimeMotionPolicy(evaluation_interval_ms=config.EVALUATION_INTERVAL_MS,
                               motion_on_hits=on_hits, motion_off_hits=off_hits)
    columns = {name: [] for name in ("packet", "score", "threshold", "ready", "raw_motion", "state")}
    packets_since_reset = 0
    for admission in iter_temporal_admissions(
        packets, target_pps=target_pps, window_size_ms=config.SEGMENTATION_WINDOW_SIZE_MS,
        fallback_interval_us=interval_us,
    ):
        packet_index, packet = admission.packet_index, admission.packet
        if admission.reset_required:
            cadence.reset()
            packets_since_reset = 0
        apply_temporal_admission(detector, admission)
        cadence.note_packet(elapsed_us=admission.coverage_us)
        should_evaluate = cadence.should_evaluate()
        if should_evaluate:
            cadence.after_evaluation()
        detector.process_packet(
            report._packet_csi_data(packet),
            config.DEFAULT_SUBCARRIERS,
            rssi_dbm=packet.get("rssi_dbm"),
            timestamp_us=report._packet_timestamp_us(packet, packet_index, interval_us),
        )
        packets_since_reset = admission.slot_index + 1
        if not should_evaluate:
            continue
        detector.update_state()
        ready = report._is_scored_replay_evaluation(detector, packets_since_reset, warmup)
        raw_state = detector.get_state()
        # A not-ready evaluation holds the debounced state, as on the board.
        effective, _ = hits.apply_state(raw_state, ready=ready)
        columns["packet"].append(packet_index)
        columns["score"].append(float(detector.get_motion_metric()))
        columns["threshold"].append(float(detector.get_threshold()))
        columns["ready"].append(ready)
        columns["raw_motion"].append(raw_state == MotionState.MOTION)
        columns["state"].append(
            STATE_NOT_READY if not ready else STATE_MOTION if effective == MotionState.MOTION else STATE_IDLE
        )
    return {
        "packet": np.array(columns["packet"], np.int64),
        "score": np.array(columns["score"], np.float32),
        "threshold": np.array(columns["threshold"], np.float32),
        "ready": np.array(columns["ready"], bool),
        "raw_motion": np.array(columns["raw_motion"], bool),
        "state": np.array(columns["state"], np.uint8),
    }


def build_detector(name: str, packets: list, board: dict):
    """Build one detector the way ESPectre's replay tools do.

    Returns (detector, warmup, info). The board's threshold is used for the
    detector the board was running; the other one starts as on a fresh boot.
    """
    import config
    from tools.lib.dataset_metadata import (
        build_calibrated_lightweight_detector, build_lightweight_detector, derive_detector_timing,
        measure_packet_interval_us,
    )
    from tools.lib.high_accuracy_detector import HIGH_ACCURACY_DEFAULT_THRESHOLD, HighAccuracyDetector
    from tools.lib.runtime_policy import nominal_packet_interval_us, production_filter_kwargs
    from tools.lib.temporal_csi_sampler import minimum_valid_slots, temporal_window_slots
    from tools.lib.temporal_replay import target_pps_for_packets

    target_pps = target_pps_for_packets(packets, measure_packet_interval_us(packets))
    from_board = board["detector"] == name and board["threshold"] is not None

    if name == "high_accuracy":
        warmup = temporal_window_slots(target_pps, config.SEGMENTATION_WINDOW_SIZE_MS)
        threshold = board["threshold"] if from_board else HIGH_ACCURACY_DEFAULT_THRESHOLD
        detector = HighAccuracyDetector(threshold=threshold, window_size=warmup, **production_filter_kwargs())
        detector.set_minimum_valid_samples(minimum_valid_slots(warmup))
        source = "board (/sensing)" if from_board else "ESPectre trained default"
        return detector, warmup, {"threshold_source": source, "start_threshold": float(threshold)}

    # Lightweight: ESPectre's startup calibration on the first quiet 10-30 s.
    calibrated = build_calibrated_lightweight_detector(packets, selected_subcarriers=config.DEFAULT_SUBCARRIERS)
    if calibrated is not None:
        detector, threshold = calibrated
        source = "calibrated on this recording"
    else:
        # Calibration found no quiet window: the firmware keeps the base threshold.
        timing = derive_detector_timing(nominal_packet_interval_us(target_pps), config.SEGMENTATION_WINDOW_SIZE_MS)
        timing["window_packets"] = temporal_window_slots(target_pps, config.SEGMENTATION_WINDOW_SIZE_MS)
        detector = build_lightweight_detector(timing=timing)
        detector.on_startup_calibration_abandoned()
        threshold = detector.get_threshold()
        source = "base threshold (startup calibration failed)"
    info = {"calibrated_threshold": float(threshold) if calibrated is not None else None}
    if from_board:
        detector.set_threshold(board["threshold"])
        threshold, source = board["threshold"], "board (/sensing)"
    return detector, detector.get_window_size(), {**info, "threshold_source": source, "start_threshold": float(threshold)}


def replay_detectors(npz_path: Path, sensing: dict | None) -> dict:
    """Replay both ESPectre detectors over a recording.

    Returns {name: {"row": csi.npz row per evaluation, "score", "threshold",
    "ready", "raw_motion", "state", "info"}}.
    """
    from tools.lib.dataset_metadata import measure_packet_interval_us

    board = board_settings(sensing)
    packets, row_index = load_packets(npz_path, board["csi_target_pps"])
    results = {}
    if not packets:
        return results
    interval_us = measure_packet_interval_us(packets)
    for name in DETECTORS:
        detector, warmup, info = build_detector(name, packets, board)
        columns = _replay(detector, packets, interval_us, warmup, board["motion_on_hits"], board["motion_off_hits"])
        columns["row"] = row_index[columns.pop("packet")]
        info.update({
            "detector": name,
            "board_detector": board["detector"],
            "motion_on_hits": board["motion_on_hits"],
            "motion_off_hits": board["motion_off_hits"],
            "warmup_packets": int(warmup),
            "n_evaluations": int(columns["state"].size),
            "n_ready": int(columns["ready"].sum()),
            "n_motion": int((columns["state"] == STATE_MOTION).sum()),
        })
        results[name] = {**columns, "info": info}
    return results
