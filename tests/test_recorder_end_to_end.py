"""Record from the fake board and synthetic camera, then check every output file."""

import csv
import json
import shutil
import threading
from pathlib import Path

import av
import numpy as np
import pytest

import record
from fake_board import FakeBoard, SyntheticCamera

DURATION_S = 6.0


def recorder_args(board, out):
    return record.parse_args([
        "--target", f"127.0.0.1:{board.port}", "--duration", str(DURATION_S), "--out", str(out),
        "--autostart", "--no-window", "--width", "320", "--height", "240",
    ])


def camera_factory(index, width, height, fps):
    return SyntheticCamera(width, height, fps)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("sessions")
    with FakeBoard() as board:
        folder, session = record.run_recorder(recorder_args(board, out), camera_factory)
        yield board, Path(folder), session


def read_frames(folder):
    with open(folder / "frames.csv", newline="") as file:
        return list(csv.DictReader(file))


def test_all_session_files_exist_and_intermediates_are_gone(recording):
    _, folder, session = recording
    for name in ("video.mp4", "frames.csv", "csi.npz", "csi_meta.json", "diagnostics.csv", "session.json"):
        assert (folder / name).stat().st_size > 0, name
    assert not (folder / "csi_stream.bin").exists()
    assert not (folder / "video.part.mp4").exists()
    assert session["status"] == "complete"
    assert session["stop_reason"] == "duration"
    assert session["sync_offset_ms"] == 0


def test_board_is_released_and_only_get_requests_were_sent(recording):
    board, _, _ = recording
    assert board.csi_sessions == 1
    deadline = threading.Event()
    for _ in range(50):
        if not board.csi_open:
            break
        deadline.wait(0.1)
    assert not board.csi_open, "the /csi connection should be closed"
    # Without the D key nothing but GET is ever sent.
    assert {method for method, _ in board.requests} == {"GET"}


def test_video_is_h264_and_pts_match_frames_csv(recording):
    _, folder, _ = recording
    rows = read_frames(folder)
    with av.open(str(folder / "video.mp4")) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.format.name == "yuv420p"
        pts_us = [float(frame.pts * stream.time_base * 1_000_000) for frame in container.decode(stream)]
    assert len(pts_us) == len(rows)
    assert pts_us == pytest.approx([int(row["pts_us"]) for row in rows], abs=1.0)


def test_video_timing_and_count_look_right(recording):
    _, folder, session = recording
    rows = read_frames(folder)
    assert len(rows) == pytest.approx(DURATION_S * 30, rel=0.15)
    assert session["video"]["frames"] == len(rows)
    assert [int(row["frame_index"]) for row in rows] == list(range(len(rows)))
    assert session["video"]["average_fps"] == pytest.approx(30, rel=0.1)


def test_csi_arrays_are_complete_and_consistent(recording):
    _, folder, session = recording
    data = np.load(folder / "csi.npz")
    n = data["csi_t_ns"].size
    assert n == pytest.approx(DURATION_S * 100, rel=0.15)
    assert session["csi"]["frames"] == n
    assert session["csi"]["sequence_gaps"] == 0
    assert data["csi_data"].shape == (n, 128)
    assert data["amplitude"].shape == (n, 64)
    for key in ("pc_receive_t_ns", "device_ticks_us", "seq_num", "stream_sequence", "rssi_dbm"):
        assert data[key].size == n
    assert np.all(np.diff(data["csi_t_ns"]) > 0)
    q, i = data["csi_data"][0, 2 * 20], data["csi_data"][0, 2 * 20 + 1]
    assert data["amplitude"][0, 20] == pytest.approx(np.hypot(float(i), float(q)))


def test_time_fit_removes_burst_delay(recording):
    _, folder, session = recording
    fit = session["time_fit"]
    assert fit["ok"]
    assert abs(fit["drift_ppm"]) < 500
    data = np.load(folder / "csi.npz")
    # Records wait in a 50 ms burst, so raw arrival is spread over about 40 ms;
    # relative to the fitted time nothing should arrive much before it.
    delay_ms = (data["pc_receive_t_ns"] - data["csi_t_ns"]) / 1e6
    assert np.percentile(delay_ms, 1) > -10
    assert np.median(delay_ms) > 5


def test_meta_and_diagnostics_contents(recording):
    _, folder, _ = recording
    meta = json.loads((folder / "csi_meta.json").read_text())
    assert meta["subcarrier_count"] == 64 and meta["record_version"] == 8
    for key in ("device", "capabilities", "sensing", "wifi"):
        assert key in meta
    assert meta["sensing"]["mode"] == "sensing", "read before /csi was opened"
    with open(folder / "diagnostics.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) >= DURATION_S - 2
    assert rows[-1]["csi_accepted_pps"] == "100.0"
    assert rows[-1]["raw_csi_active"] == "True"
    assert rows[-1]["csi_stale_total"] == "", "missing fields stay empty"


def test_csi_npz_holds_espectre_values_and_the_board_rate(recording):
    _, folder, _ = recording
    data = np.load(folder / "csi.npz")
    assert data["phase"].shape == data["amplitude"].shape
    assert int(data["csi_target_pps"]) == 100
    q, i = float(data["csi_data"][0, 40]), float(data["csi_data"][0, 41])
    assert data["phase"][0, 20] == pytest.approx(np.arctan2(q, i), abs=1e-6)


def test_the_session_is_analyzed_when_recording_ends(recording):
    _, folder, _ = recording
    manifest = json.loads((folder / "derived" / "manifest.json").read_text())
    assert {"player_signals.npz", "espectre_lightweight.npz", "espectre_high_accuracy.npz"} <= set(manifest["files"])
    assert manifest["detectors"]["high_accuracy"]["board_detector"] == "high_accuracy"
    meta = json.loads((folder / "csi_meta.json").read_text())
    assert meta["sensing"]["detector"] == "high_accuracy"


def wait_for(condition, seconds=5.0):
    event = threading.Event()
    for _ in range(int(seconds / 0.05)):
        if condition():
            return True
        event.wait(0.05)
    return False


def test_detector_switch_sends_one_patch_and_shows_the_new_detector(tmp_path):
    with FakeBoard() as board:
        recorder = record.Recorder(recorder_args(board, tmp_path), camera_factory)
        recorder.start_preview()
        try:
            assert "high_accuracy" in recorder.detector_line()
            assert recorder.switch_detector()
            assert wait_for(lambda: recorder.switch_thread is None)
            assert board.detection["detector"] == "lightweight"
            assert recorder.board_info["sensing"]["detector"] == "lightweight"
            assert "lightweight" in recorder.detector_line()
            patches = [request for request in board.requests if request[0] != "GET"]
            assert patches == [("PATCH", "/espectre/v1/sensing")]
        finally:
            recorder.close()


def test_detector_switch_is_refused_while_recording(tmp_path):
    with FakeBoard() as board:
        recorder = record.Recorder(recorder_args(board, tmp_path), camera_factory)
        recorder.start_preview()
        try:
            recorder.recording.set()
            assert not recorder.switch_detector()
            recorder.recording.clear()
        finally:
            recorder.close()
        assert all(method == "GET" for method, _ in board.requests)


def test_second_session_gets_a_clear_error_when_board_is_busy(tmp_path):
    with FakeBoard() as board:
        board.csi_open = True
        with pytest.raises(SystemExit, match="409"):
            record.run_recorder(recorder_args(board, tmp_path), camera_factory)
        assert list(tmp_path.iterdir()) == [], "no session folder for a refused start"


def test_video_reaches_disk_while_recording(tmp_path):
    """A crash leaves the file as it is now, so finished fragments must be on disk."""
    part = tmp_path / "video.part.mp4"
    encoder = record.VideoEncoder(part, 320, 240, 30)
    frame = np.zeros((240, 320, 3), np.uint8)  # small frames are the hardest case
    for n in range(100):
        encoder.encode(frame, n * 33_333)

    # Copy the file without closing the encoder, like a killed process would leave it.
    crash_copy = tmp_path / "crash.mp4"
    shutil.copy(part, crash_copy)
    encoder.close()

    packets = record.remux_faststart(crash_copy, tmp_path / "video.mp4")
    assert packets >= 60, "at most about one 30-frame fragment may be lost"


def test_recover_without_readable_video_does_not_claim_frames(recording, tmp_path):
    _, source, _ = recording
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    for name in ("csi_meta.json", "session.json", "diagnostics.csv", "frames.csv"):
        shutil.copy(source / name, crashed / name)
    (crashed / "video.part.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 16)

    result = record.finalize_session(crashed, status="recovered")

    assert result["status"] == "recovered_no_video"
    assert result["video"]["frames"] == 0
    assert (crashed / "video.part.mp4").exists(), "the original is kept"


def test_recover_finishes_a_crashed_session(recording, tmp_path):
    """Simulate a crash: raw files only, a torn CSI record, and a torn CSV line."""
    _, source, _ = recording
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    for name in ("csi_meta.json", "session.json", "diagnostics.csv"):
        shutil.copy(source / name, crashed / name)
    session = json.loads((crashed / "session.json").read_text())
    session["status"] = "recording"
    session.pop("end_time")
    session.pop("run")
    (crashed / "session.json").write_text(json.dumps(session))

    # Rebuild the append-only CSI file from the finished arrays.
    data = np.load(source / "csi.npz")
    n = data["csi_t_ns"].size
    dtype = record.csi_record_dtype(128)
    records = np.zeros(n, dtype=dtype)
    records["pc_receive_t_ns"] = data["pc_receive_t_ns"]
    records["device_ticks_us"] = data["device_ticks_us"]
    records["stream_sequence"] = data["stream_sequence"]
    records["iq"] = data["csi_data"]
    with open(crashed / "csi_stream.bin", "wb") as file:
        file.write(record.CSI_STREAM_MAGIC + np.array([128, 0], dtype="<u4").tobytes())
        file.write(records.tobytes())
        file.write(b"\x01" * 37)  # a partly written record

    # Encode a real fragmented video, then pretend more rows were logged than saved.
    rows = read_frames(source)[:40]
    encoder = record.VideoEncoder(crashed / "video.part.mp4", 320, 240, 30)
    frame = np.zeros((240, 320, 3), np.uint8)
    for row in rows[:30]:
        encoder.encode(frame, int(row["pts_us"]))
    encoder.close()
    with open(crashed / "frames.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "pc_t_ns", "perf_t_ns", "pts_us"])
        writer.writerows([[row[k] for k in ("frame_index", "pc_t_ns", "perf_t_ns", "pts_us")] for row in rows])
        file.write("41,17800")  # a partly written line

    result = record.finalize_session(crashed, status="recovered")

    assert result["status"] == "recovered"
    assert result["csi"]["frames"] == n
    assert result["video"]["frames"] == 30
    assert len(read_frames(crashed)) == 30
    with av.open(str(crashed / "video.mp4")) as container:
        assert sum(1 for _ in container.decode(video=0)) == 30
