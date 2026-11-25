"""Record from two fake boards and a synthetic camera at once, then check every output file."""

import csv
import json
import shutil
import threading
from pathlib import Path

import av
import numpy as np
import pytest

import analyze
import player
import record
from fake_board import FakeBoard, SyntheticCamera

DURATION_S = 6.0


def recorder_args(boards, out, names=None):
    """One --target per board; names[i] (or None) becomes NAME=IP."""
    boards = boards if isinstance(boards, (list, tuple)) else [boards]
    names = names or [None] * len(boards)
    targets = []
    for board, name in zip(boards, names):
        address = f"127.0.0.1:{board.port}"
        targets += ["--target", f"{name}={address}" if name else address]
    return record.parse_args(targets + [
        "--duration", str(DURATION_S), "--out", str(out),
        "--autostart", "--no-window", "--width", "320", "--height", "240",
    ])


def camera_factory(index, width, height, fps):
    return SyntheticCamera(width, height, fps)


@pytest.fixture(scope="module")
def two_boards(tmp_path_factory):
    """Board "a" is named on the command line; the second is named after its address."""
    out = tmp_path_factory.mktemp("sessions")
    with FakeBoard() as board_a, FakeBoard() as board_b:
        folder, session = record.run_recorder(recorder_args([board_a, board_b], out, ["a", None]), camera_factory)
        names = ["a", f"127.0.0.1_{board_b.port}"]
        yield [board_a, board_b], Path(folder), session, names


@pytest.fixture(scope="module")
def recording(two_boards):
    """(board, its device folder, its session.json) for board "a"."""
    boards, folder, _, names = two_boards
    device = folder / names[0]
    return boards[0], device, json.loads((device / "session.json").read_text())


def read_frames(folder):
    with open(analyze.input_path(folder, "frames.csv"), newline="") as file:
        return list(csv.DictReader(file))


def wait_for(condition, seconds=5.0):
    event = threading.Event()
    for _ in range(int(seconds / 0.05)):
        if condition():
            return True
        event.wait(0.05)
    return False


# --- Layout of a multi-board recording ------------------------------------------------

def test_layout_shared_video_and_one_folder_per_board(two_boards):
    _, folder, session, names = two_boards
    assert sorted(path.name for path in folder.iterdir()) == sorted(["video.mp4", "frames.csv", "session.json", *names])
    assert analyze.device_folders(folder) == sorted(folder / name for name in names)
    assert session["status"] == "complete" and session["stop_reason"] == "duration"
    assert sorted(session["devices"]) == sorted(names)
    for name in names:
        device = folder / name
        for file_name in ("csi.npz", "csi_meta.json", "diagnostics.csv", "session.json"):
            assert (device / file_name).stat().st_size > 0, f"{name}/{file_name}"
        assert not (device / "csi_stream.bin").exists()
        assert not (device / "video.mp4").exists(), "the video is stored once, in the recording folder"
        assert session["devices"][name]["csi"]["frames"] > 0
        assert session["devices"][name]["time_fit"]["ok"]


def test_each_board_has_its_own_session_json(two_boards):
    boards, folder, session, names = two_boards
    for board, name in zip(boards, names):
        device = json.loads((folder / name / "session.json").read_text())
        assert device["device"] == name
        assert device["target"] == f"127.0.0.1:{board.port}"
        assert device["status"] == "complete" and device["stop_reason"] == "duration"
        assert device["sync_offset_ms"] == 0
        assert device["start_t_ns"] == session["start_t_ns"] and device["end_t_ns"] == session["end_t_ns"]
        meta = json.loads((folder / name / "csi_meta.json").read_text())
        assert meta["target"] == f"127.0.0.1:{board.port}"


def test_boards_record_the_same_stretch_of_time(two_boards):
    _, folder, _, names = two_boards
    starts, ends = [], []
    for name in names:
        times = np.load(folder / name / "csi.npz")["pc_receive_t_ns"]
        starts.append(times[0])
        ends.append(times[-1])
    assert max(starts) - min(starts) < 0.5e9
    assert max(ends) - min(ends) < 0.5e9


def test_boards_are_released_and_only_get_requests_were_sent(two_boards):
    boards, _, _, _ = two_boards
    for board in boards:
        assert board.csi_sessions == 1
        assert wait_for(lambda: not board.csi_open), "the /csi connection should be closed"
        # Without the D key nothing but GET is ever sent.
        assert {method for method, _ in board.requests} == {"GET"}


def test_every_board_is_analyzed_when_recording_ends(two_boards):
    _, folder, _, names = two_boards
    assert not (folder / "derived").exists()
    for name in names:
        manifest = json.loads((folder / name / "derived" / "manifest.json").read_text())
        assert {"player_signals.npz", "espectre_lightweight.npz", "espectre_high_accuracy.npz"} <= set(manifest["files"])
        assert manifest["detectors"]["high_accuracy"]["board_detector"] == "high_accuracy"
        meta = json.loads((folder / name / "csi_meta.json").read_text())
        assert meta["sensing"]["detector"] == "high_accuracy"
        assert analyze.is_up_to_date(folder / name), "the frames.csv one level up is part of the fingerprint"


def test_player_opens_one_board_of_the_recording(two_boards):
    _, folder, _, names = two_boards
    with pytest.raises(SystemExit, match="--device"):
        player.resolve_device(folder)
    with pytest.raises(SystemExit, match="no device"):
        player.resolve_device(folder, "missing")
    device = player.resolve_device(folder, names[1])
    assert device == folder / names[1]
    assert player.resolve_device(device) == device
    data = player.load_session(device)
    assert data["meta"]["device"] == names[1]
    assert data["video_path"] == folder / "video.mp4"
    assert data["meta"]["n_frames"] == len(read_frames(folder))


def test_target_names_and_validation():
    assert record.parse_target("kitchen=192.168.1.10") == ("kitchen", "192.168.1.10")
    assert record.parse_target("192.168.1.10") == ("192.168.1.10", "192.168.1.10")
    assert record.parse_target("10.0.0.5:8080") == ("10.0.0.5_8080", "10.0.0.5:8080")
    assert record.parse_target("my room=10.0.0.5") == ("my_room", "10.0.0.5")
    args = record.parse_args(["--target", "a=10.0.0.1,b=10.0.0.2", "--target", "10.0.0.3"])
    assert args.target == ["a=10.0.0.1", "b=10.0.0.2", "10.0.0.3"]
    bad_arguments = (
        ["--target", "x=10.0.0.1", "--target", "x=10.0.0.2"],  # same folder name
        ["--target", "10.0.0.1", "--target", "a=10.0.0.1"],    # same board twice
        ["--target", "a="],                                     # no address
    )
    for bad in bad_arguments:
        with pytest.raises(SystemExit):
            record.parse_args(bad)


# --- One board's files (board "a") ------------------------------------------------------

def test_all_session_files_exist_and_intermediates_are_gone(recording):
    _, folder, session = recording
    for name in ("csi.npz", "csi_meta.json", "diagnostics.csv", "session.json"):
        assert (folder / name).stat().st_size > 0, name
    for name in ("video.mp4", "frames.csv"):
        assert analyze.input_path(folder, name).stat().st_size > 0, name
    assert not (folder / "csi_stream.bin").exists()
    assert not (folder.parent / "video.part.mp4").exists()
    assert session["status"] == "complete"
    assert session["stop_reason"] == "duration"
    assert session["sync_offset_ms"] == 0


def test_video_is_h264_and_pts_match_frames_csv(recording):
    _, folder, _ = recording
    rows = read_frames(folder)
    with av.open(str(analyze.input_path(folder, "video.mp4"))) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.format.name == "yuv420p"
        pts_us = [float(frame.pts * stream.time_base * 1_000_000) for frame in container.decode(stream)]
    assert len(pts_us) == len(rows)
    assert pts_us == pytest.approx([int(row["pts_us"]) for row in rows], abs=1.0)


def test_video_timing_and_count_look_right(two_boards):
    _, folder, session, _ = two_boards
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


# --- Detector switch and refused starts -------------------------------------------------

def test_detector_switch_sends_one_patch_and_shows_the_new_detector(tmp_path):
    with FakeBoard() as board:
        recorder = record.Recorder(recorder_args(board, tmp_path), camera_factory)
        recorder.start_preview()
        state = recorder.boards[0]
        try:
            assert "high_accuracy" in state.detector_line()
            assert recorder.switch_detector()
            assert wait_for(lambda: not recorder.switching())
            assert board.detection["detector"] == "lightweight"
            assert state.board_info["sensing"]["detector"] == "lightweight"
            assert "lightweight" in state.detector_line()
            patches = [request for request in board.requests if request[0] != "GET"]
            assert patches == [("PATCH", "/espectre/v1/sensing")]
        finally:
            recorder.close()


def test_detector_switch_brings_every_board_to_the_same_detector(tmp_path):
    with FakeBoard() as board_a, FakeBoard() as board_b:
        board_b.detection["detector"] = "lightweight"  # already on the target: no request
        recorder = record.Recorder(recorder_args([board_a, board_b], tmp_path, ["a", "b"]), camera_factory)
        recorder.start_preview()
        try:
            assert recorder.switch_detector()
            assert wait_for(lambda: not recorder.switching())
            assert board_a.detection["detector"] == board_b.detection["detector"] == "lightweight"
            assert [r for r in board_a.requests if r[0] != "GET"] == [("PATCH", "/espectre/v1/sensing")]
            assert [r for r in board_b.requests if r[0] != "GET"] == []
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


def test_session_folder_name_can_be_chosen(tmp_path):
    with FakeBoard() as board:
        args = recorder_args(board, tmp_path)
        args.name, args.duration = "exp1", 1.0
        folder, _ = record.run_recorder(args, camera_factory)
        assert Path(folder) == tmp_path / "exp1"
        assert (tmp_path / "exp1" / "video.mp4").exists()
        assert (tmp_path / "exp1" / f"127.0.0.1_{board.port}" / "csi.npz").exists()
    # An existing folder is never reused, and bad names are refused.
    for name in ("exp1", "..\\up", "a/b", "x:"):
        with pytest.raises(SystemExit):
            record.parse_args(["--target", "10.0.0.1", "--out", str(tmp_path), "--name", name])
    assert record.parse_args(["--target", "10.0.0.1", "--out", str(tmp_path), "--name", "exp 2"]).name == "exp 2"


def test_one_busy_board_releases_the_others_and_creates_nothing(tmp_path):
    with FakeBoard() as free, FakeBoard() as busy:
        busy.csi_open = True
        with pytest.raises(SystemExit, match="board b already has a /csi session"):
            record.run_recorder(recorder_args([free, busy], tmp_path, ["a", "b"]), camera_factory)
        assert list(tmp_path.iterdir()) == [], "no session folder for a refused start"
        assert free.csi_sessions == 1
        assert wait_for(lambda: not free.csi_open), "the board that did open /csi is released again"


# --- Crash safety -------------------------------------------------------------------------

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


def copy_crashed_board(source_device: Path, target_device: Path) -> int:
    """Copy a board's start-time files and rebuild its csi_stream.bin. Returns the record count."""
    target_device.mkdir(parents=True)
    for name in ("csi_meta.json", "diagnostics.csv"):
        shutil.copy(source_device / name, target_device / name)
    session = json.loads((source_device / "session.json").read_text())
    for key in ("end_time", "end_t_ns", "duration_s", "stop_reason", "run", "csi", "time_fit"):
        session.pop(key, None)
    session["status"] = "recording"
    (target_device / "session.json").write_text(json.dumps(session))

    # Rebuild the append-only CSI file from the finished arrays.
    data = np.load(source_device / "csi.npz")
    n = data["csi_t_ns"].size
    dtype = record.csi_record_dtype(128)
    records = np.zeros(n, dtype=dtype)
    records["pc_receive_t_ns"] = data["pc_receive_t_ns"]
    records["device_ticks_us"] = data["device_ticks_us"]
    records["stream_sequence"] = data["stream_sequence"]
    records["iq"] = data["csi_data"]
    with open(target_device / "csi_stream.bin", "wb") as file:
        file.write(record.CSI_STREAM_MAGIC + np.array([128, 0], dtype="<u4").tobytes())
        file.write(records.tobytes())
        file.write(b"\x01" * 37)  # a partly written record
    return n


def crashed_recording_session(source: Path, target: Path) -> None:
    """The recording's session.json as it is while recording."""
    session = json.loads((source / "session.json").read_text())
    for key in ("end_time", "end_t_ns", "duration_s", "stop_reason", "run"):
        session.pop(key, None)
    session["status"] = "recording"
    session["devices"] = {name: {"target": info["target"]} for name, info in session["devices"].items()}
    (target / "session.json").write_text(json.dumps(session))


def test_recover_without_readable_video_does_not_claim_frames(two_boards, tmp_path):
    _, source, _, names = two_boards
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    crashed_recording_session(source, crashed)
    shutil.copy(source / "frames.csv", crashed / "frames.csv")
    copy_crashed_board(source / names[0], crashed / names[0])
    (crashed / "video.part.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 16)

    result = record.finalize_session(crashed, status="recovered")

    assert result["status"] == "recovered_no_video"
    assert result["video"]["frames"] == 0
    assert (crashed / "video.part.mp4").exists(), "the original is kept"
    assert json.loads((crashed / names[0] / "session.json").read_text())["status"] == "recovered_no_video"


def test_recover_finishes_a_crashed_session(two_boards, tmp_path):
    """Simulate a crash: raw files only, torn CSI records, and a torn CSV line."""
    _, source, _, names = two_boards
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    crashed_recording_session(source, crashed)
    counts = {name: copy_crashed_board(source / name, crashed / name) for name in names}

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
    assert result["video"]["frames"] == 30
    assert len(read_frames(crashed)) == 30
    assert "end_t_ns" in result, "the end time comes from the data after a crash"
    for name in names:
        device = json.loads((crashed / name / "session.json").read_text())
        assert result["devices"][name]["csi"]["frames"] == counts[name]
        assert device["csi"]["frames"] == counts[name]
        assert device["status"] == "recovered"
        assert device["end_t_ns"] == result["end_t_ns"]
        assert (crashed / name / "csi.npz").exists()
        assert not (crashed / name / "csi_stream.bin").exists()
    with av.open(str(crashed / "video.mp4")) as container:
        assert sum(1 for _ in container.decode(video=0)) == 30


def test_recover_still_finishes_an_older_single_folder_session(two_boards, tmp_path):
    """Sessions recorded before multi-board support keep the CSI beside the video."""
    _, source, _, names = two_boards
    crashed = tmp_path / "old"
    n = copy_crashed_board(source / names[0], crashed)
    shutil.copy(source / "frames.csv", crashed / "frames.csv")
    shutil.copy(source / "video.mp4", crashed / "video.part.mp4")

    result = record.finalize_session(crashed, status="recovered")

    assert analyze.device_folders(crashed) == [crashed]
    assert result["csi"]["frames"] == n
    assert "devices" not in result
    assert result["sync_offset_ms"] == 0
    assert (crashed / "csi.npz").exists() and (crashed / "video.mp4").exists()
