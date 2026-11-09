"""Record WiFi CSI and webcam video on one PC clock.

    python record.py --target <IP> --camera 0 --duration 300 --out sessions/
    python record.py --recover sessions/<folder>      # finish a crashed session

Press S in the preview window to start and Q to stop. Only GET requests are
sent to the board, so its settings and traffic source never change.

Crash safety: while recording, CSI goes to an append-only csi_stream.bin and
video to a fragmented video.part.mp4. A clean stop (or --recover) converts them
to csi.npz and a browser-friendly video.mp4 without re-encoding.
"""

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

import espectre_path
import timesync

# All video timestamps are integer microseconds, so frames.csv stores them exactly.
VIDEO_TIME_BASE = Fraction(1, 1_000_000)
FRAGMENTED_MP4 = "frag_keyframe+empty_moov+default_base_moof"
FRAME_QUEUE_SIZE = 64  # about 2 s of video if the encoder falls behind
CSI_STREAM_MAGIC = b"CSIVSYNC"
CSI_STREAM_HEADER_BYTES = 16
FLUSH_INTERVAL_S = 0.5
WINDOW_NAME = "CSI + video recorder"
NOISE_FLOOR_UNKNOWN = -128

DIAGNOSTIC_FIELDS = [
    "timestamp_ms", "csi_callback_pps", "csi_accepted_pps", "csi_occupancy", "wifi_rssi_dbm",
    "csi_hw_error_total", "csi_pending_frame_drops_total", "csi_missing_slots_total",
    "csi_excess_total", "csi_stale_total", "csi_out_of_order_total",
]
RAW_CSI_FIELDS = ["active", "raw_drop_total", "send_backpressure_total", "fresh_record_total", "stream_sequence"]


# --- Small file helpers -------------------------------------------------------------

def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON through a temp file, so a crash never leaves half a file."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csi_record_dtype(iq_bytes: int) -> np.dtype:
    """One fixed-size record of csi_stream.bin."""
    return np.dtype([
        ("pc_receive_t_ns", "<i8"), ("perf_receive_t_ns", "<i8"),
        ("device_ticks_us", "<u8"), ("wifi_rx_ts_us", "<u8"),
        ("seq_num", "<u4"), ("stream_sequence", "<u8"),
        ("raw_drop_total", "<u8"), ("send_backpressure_total", "<u8"),
        ("rssi_dbm", "i1"), ("noise_floor_dbm", "i1"), ("channel", "u1"),
        ("iq", "i1", (iq_bytes,)),
    ])


class CsiStreamWriter:
    """Append CSI records and flush them regularly so a crash loses under 1 s."""

    def __init__(self, path: Path):
        self.path = path
        self.file = None
        self.dtype = None
        self.iq_bytes = 0
        self.last_flush = time.monotonic()

    def write(self, fields: tuple, iq: np.ndarray) -> bool:
        """Return False for a record whose size differs from the first record."""
        if self.file is None:
            self.iq_bytes = int(iq.size)
            self.dtype = csi_record_dtype(self.iq_bytes)
            self.file = open(self.path, "wb")
            header = CSI_STREAM_MAGIC + np.array([self.iq_bytes, 0], dtype="<u4").tobytes()
            self.file.write(header)
        if iq.size != self.iq_bytes:
            return False
        record = np.empty(1, dtype=self.dtype)
        record[0] = (*fields, iq)
        self.file.write(record.tobytes())
        if time.monotonic() - self.last_flush >= FLUSH_INTERVAL_S:
            self.file.flush()
            self.last_flush = time.monotonic()
        return True

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def read_csi_stream(path: Path) -> np.ndarray:
    """Read csi_stream.bin, ignoring a partly written record at the end."""
    with open(path, "rb") as file:
        header = file.read(CSI_STREAM_HEADER_BYTES)
        if len(header) < CSI_STREAM_HEADER_BYTES or not header.startswith(CSI_STREAM_MAGIC):
            raise ValueError(f"{path} is not a CSI stream file")
        iq_bytes = int(np.frombuffer(header, "<u4", 1, len(CSI_STREAM_MAGIC))[0])
        return np.fromfile(file, dtype=csi_record_dtype(iq_bytes))


class VideoEncoder:
    """H.264 encoder writing a fragmented MP4 that stays valid after a crash."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        # flush_packets: each finished fragment goes straight to the OS. Without
        # it FFmpeg holds fragments in memory and a crash loses them.
        self.container = av.open(str(path), "w", format="mp4",
                                 options={"movflags": FRAGMENTED_MP4, "flush_packets": "1"})
        rate = max(1, int(round(fps)))
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.time_base = VIDEO_TIME_BASE
        # zerolatency: no frame reordering, so frames reach the file promptly.
        # One keyframe per second keeps browser seeking and frame stepping quick.
        self.stream.options = {"preset": "veryfast", "tune": "zerolatency", "crf": "20", "g": str(rate), "bf": "0"}

    def encode(self, bgr: np.ndarray, pts_us: int) -> None:
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = pts_us
        frame.time_base = VIDEO_TIME_BASE
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()


def remux_faststart(source: Path, target: Path, max_packets: int | None = None) -> int:
    """Copy H.264 packets to a normal MP4 with its index at the front.

    Returns how many packets were copied. A crash can leave a damaged final
    fragment, so a read error simply ends the copy there. max_packets stops at
    the last frame that has a timestamp in frames.csv.
    """
    count = 0
    temp = target.with_name(target.stem + ".tmp.mp4")
    with av.open(str(source)) as input_file:
        with av.open(str(temp), "w", format="mp4", options={"movflags": "+faststart"}) as output_file:
            input_stream = input_file.streams.video[0]
            output_stream = output_file.add_stream_from_template(input_stream)
            try:
                for packet in input_file.demux(input_stream):
                    if packet.dts is None:  # the demuxer's end-of-stream marker
                        continue
                    if max_packets is not None and count >= max_packets:
                        break
                    packet.stream = output_stream
                    output_file.mux(packet)
                    count += 1
            except av.FFmpegError as exc:
                print(f"Warning: video ends early at packet {count}: {exc}")
    os.replace(temp, target)
    return count


def read_frames_csv(path: Path) -> list:
    """Read frames.csv rows, ignoring a partly written last line."""
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) == 4 and all(cell.lstrip("-").isdigit() for cell in row):
                rows.append([int(cell) for cell in row])
    return rows


# --- Finishing a session (clean stop and --recover share this) ------------------------

def finalize_session(folder: Path, status: str = "complete", repo=None) -> dict:
    """Convert the crash-safe files to their final form and write the summary."""
    espectre_path.setup(repo)
    from tools.lib.csi_io import normalize_stored_csi_bin_layout

    session = read_json(folder / "session.json")
    meta = read_json(folder / "csi_meta.json")
    stream_path, part_path = folder / "csi_stream.bin", folder / "video.part.mp4"

    # CSI: fit the device clock to the PC clock, then build csi.npz.
    fit = timesync.fit_device_to_pc([], [])
    records = None
    if stream_path.exists():
        records = read_csi_stream(stream_path)
        fit = timesync.fit_device_to_pc(records["device_ticks_us"].astype(np.int64), records["pc_receive_t_ns"])
        # Uses the clone's layout check so subcarrier order stays correct for any chip.
        iq = normalize_stored_csi_bin_layout(np.array(records["iq"]))
        # Amplitude is the length of each (Q, I) pair.
        pairs = iq.reshape(iq.shape[0], -1, 2).astype(np.float32)
        amplitude = np.hypot(pairs[..., 0], pairs[..., 1])
        temp = folder / "csi.tmp.npz"
        with open(temp, "wb") as file:
            np.savez(
                file,
                pc_receive_t_ns=records["pc_receive_t_ns"],
                perf_receive_t_ns=records["perf_receive_t_ns"],
                csi_t_ns=fit.to_pc_ns(records["device_ticks_us"].astype(np.int64)),
                device_ticks_us=records["device_ticks_us"],
                wifi_rx_ts_us=records["wifi_rx_ts_us"],
                seq_num=records["seq_num"],
                stream_sequence=records["stream_sequence"],
                rssi_dbm=records["rssi_dbm"],
                noise_floor_dbm=records["noise_floor_dbm"],
                channel=records["channel"],
                csi_data=iq,  # int8 pairs, Q first then I, as in the V8 payload
                amplitude=amplitude,
            )
        os.replace(temp, folder / "csi.npz")
        meta["subcarrier_count"] = int(amplitude.shape[1])
        meta["iq_order"] = "csi_data holds int8 pairs: Q first, then I"
        write_json_atomic(folder / "csi_meta.json", meta)

    # Video: make it browser-friendly and keep frames.csv in step with it.
    frames = read_frames_csv(folder / "frames.csv")
    video_ok = False
    if part_path.exists():
        try:
            packets = remux_faststart(part_path, folder / "video.mp4", max_packets=len(frames))
            video_ok = packets > 0
            if packets < len(frames):  # frames logged but lost in the crash
                frames = frames[:packets]
        except (av.FFmpegError, IndexError) as exc:
            print(f"Warning: could not read {part_path.name}: {exc}")
    temp = folder / "frames.tmp.csv"
    with open(temp, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "pc_t_ns", "perf_t_ns", "pts_us"])
        writer.writerows(frames)
    os.replace(temp, folder / "frames.csv")

    if not video_ok:
        status = f"{status}_no_video"
        print("Warning: no video could be saved for this session. CSI data is intact.")

    # Summary. Without a video, its frames.csv rows describe nothing that exists.
    frame_times = np.array([row[1] for row in frames] if video_ok else [], dtype=np.int64)
    run = session.get("run", {})
    video_seconds = float(frame_times[-1] - frame_times[0]) / 1e9 if frame_times.size > 1 else 0.0
    session["video"] = {
        **session.get("video", {}),
        "frames": int(frame_times.size),
        "average_fps": (frame_times.size - 1) / video_seconds if video_seconds else 0.0,
        "duration_s": video_seconds,
        "dropped_estimated_from_gaps": timesync.count_frame_gaps(frame_times),
        "dropped_encoder_queue": run.get("video_queue_drops", 0),
    }
    n_csi = 0 if records is None else int(records.size)
    csi_seconds = 0.0
    sequence_gaps = board_drops = backpressure = 0
    if n_csi:
        csi_seconds = float(records["pc_receive_t_ns"][-1] - records["pc_receive_t_ns"][0]) / 1e9
        sequence_gaps = int(np.maximum(np.diff(records["stream_sequence"].astype(np.int64)) - 1, 0).sum())
        board_drops = int(records["raw_drop_total"][-1])
        backpressure = int(records["send_backpressure_total"][-1])
    session["csi"] = {
        "frames": n_csi,
        "average_pps": (n_csi - 1) / csi_seconds if csi_seconds else 0.0,
        "duration_s": csi_seconds,
        "sequence_gaps": sequence_gaps,
        "board_raw_drop_total": run.get("board_raw_drop_total", board_drops),
        "board_send_backpressure_total": run.get("board_send_backpressure_total", backpressure),
        "records_with_other_size_skipped": run.get("csi_size_mismatch", 0),
    }
    session["time_fit"] = fit.to_dict()
    session["status"] = status
    session["sync_offset_ms"] = session.get("sync_offset_ms", 0)
    # After a crash these come from the data, because the stop time was never written.
    last_times = [int(frame_times[-1])] if frame_times.size else []
    if n_csi:
        last_times.append(int(records["pc_receive_t_ns"][-1]))
    if last_times and "end_t_ns" not in session:
        session["end_t_ns"] = max(last_times)
        session["end_time"] = datetime.fromtimestamp(max(last_times) / 1e9).isoformat(timespec="milliseconds")
        session["duration_s"] = (max(last_times) - session["start_t_ns"]) / 1e9
    write_json_atomic(folder / "session.json", session)

    # The intermediates are only removed once their final forms exist.
    if records is not None and (folder / "csi.npz").exists():
        stream_path.unlink()
    if video_ok and (folder / "video.mp4").exists():
        part_path.unlink()
    return session


def print_summary(folder: Path, session: dict) -> None:
    video, csi, fit = session["video"], session["csi"], session["time_fit"]
    print(f"\nSession saved to {folder}")
    print(f"  status            : {session['status']} ({session.get('stop_reason', 'recovered')})")
    print(f"  video             : {video['frames']} frames, {video['average_fps']:.2f} fps, "
          f"{video['duration_s']:.1f} s, about {video['dropped_estimated_from_gaps']} missing, "
          f"{video['dropped_encoder_queue']} dropped by encoder queue")
    print(f"  CSI               : {csi['frames']} frames, {csi['average_pps']:.1f} per s, "
          f"{csi['sequence_gaps']} sequence gaps, {csi['board_raw_drop_total']} dropped on board")
    print(f"  clock fit         : {fit['method']}, drift {fit['drift_ppm']:.1f} ppm, "
          f"residual RMS {fit['residual_rms_ms']:.2f} ms, p95 {fit['residual_p95_ms']:.2f} ms")
    print(f"  CSI network delay : median {fit['latency_median_ms']:.1f} ms, p95 {fit['latency_p95_ms']:.1f} ms")
    if not fit["ok"]:
        print("  WARNING: the clock fit is not reliable (recording too short, or implausible drift).")


# --- Recorder -------------------------------------------------------------------------

def open_webcam(index: int, width: int, height: int, fps: float):
    """Open a USB webcam with DirectShow, asking for MJPG so 720p30 fits on USB."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep the delay as small as the driver allows
    return camera


class Recorder:
    """Owns the camera, encoder, CSI, and diagnostics threads."""

    def __init__(self, args, camera_factory=open_webcam):
        espectre_path.setup(args.repo)
        from src.python.espectre_cli.device_transport import DirectClient
        from tools.lib.csi_io import DirectRawCSIReceiver

        self.args = args
        self.camera_factory = camera_factory
        self.receiver = DirectRawCSIReceiver(args.target, derive_complex=False)
        self.client = DirectClient(self.receiver.control_endpoint, origin=self.receiver.origin, timeout=3.0)

        self.quit = threading.Event()         # everything stops
        self.recording = threading.Event()    # data is being saved
        self.camera = None
        self.camera_settings = {}
        self.threads = []

        self.session_dir = None
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.csi_writer = None
        self.encoder_thread = None
        self.csi_thread = None
        self.start_perf = 0
        self.expected_pps = 100.0

        self.latest_frame = None
        self.camera_times = deque(maxlen=120)
        self.video_queue_drops = 0
        self.frames_written = 0
        self.camera_error = None
        self.encoder_error = None
        self.csi_error = None

        self.csi_count = 0
        self.csi_receive_times = deque(maxlen=2000)
        self.csi_device_ticks = deque(maxlen=2000)
        self.csi_sequence_gaps = 0
        self.csi_size_mismatch = 0
        self.last_stream_sequence = None
        self.latest_packet_drop_total = 0

        self.latest_diagnostics = {}
        self.diagnostics_status = "waiting"
        self.diagnostics_file = None
        self.diagnostics_writer = None
        self.diagnostics_lock = threading.Lock()

    # Startup ---------------------------------------------------------------------------

    def read_board_info(self) -> dict:
        """Read everything needed for csi_meta.json before /csi is opened."""
        try:
            info = {name: self.client.request("get", name) for name in ("device", "capabilities", "sensing", "wifi")}
        except Exception as exc:
            raise SystemExit(f"Cannot reach the board at {self.args.target}: {exc}")
        if not info["sensing"].get("enabled", True):
            raise SystemExit("Sensing is disabled on the board, so /csi would fail. Enable it first.")
        self.expected_pps = float(info["sensing"].get("csi_target_pps") or 100.0)
        return info

    def open_camera(self) -> None:
        args = self.args
        camera = self.camera_factory(args.camera, args.width, args.height, args.fps)
        if not camera.isOpened():
            raise SystemExit(f"Cannot open camera {args.camera}. Is another app using it?")
        self.camera = camera
        fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
        self.camera_settings = {
            "index": args.camera,
            "requested": {"width": args.width, "height": args.height, "fps": args.fps},
            "actual": {
                "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(camera.get(cv2.CAP_PROP_FPS)),
                "fourcc": "".join(chr((fourcc >> (8 * n)) & 0xFF) for n in range(4)).strip("\0"),
            },
        }

    def start_preview(self) -> None:
        self.board_info = self.read_board_info()
        self.open_camera()
        for target in (self._camera_loop, self._diagnostics_loop):
            thread = threading.Thread(target=target, name=target.__name__, daemon=True)
            thread.start()
            self.threads.append(thread)

    # Threads ---------------------------------------------------------------------------

    def _camera_loop(self) -> None:
        failures = 0
        while not self.quit.is_set():
            # grab() returns when a frame arrives, so the stamp is taken before
            # the slower decode in retrieve().
            if not self.camera.grab():
                failures += 1
                if failures > 200:
                    self.camera_error = "camera stopped delivering frames"
                    return
                time.sleep(0.01)
                continue
            pc_t_ns, perf_t_ns = time.time_ns(), time.perf_counter_ns()
            ok, frame = self.camera.retrieve()
            if not ok:
                failures += 1
                continue
            failures = 0
            self.latest_frame = frame
            self.camera_times.append(perf_t_ns)
            if self.recording.is_set():
                try:
                    self.frame_queue.put_nowait((frame, pc_t_ns, perf_t_ns))
                except queue.Full:
                    self.video_queue_drops += 1

    def _encoder_loop(self) -> None:
        """Encode in its own thread so x264 can never delay frame capture."""
        path = self.session_dir / "frames.csv"
        encoder = None
        first_perf = last_pts = 0
        last_flush = time.monotonic()
        try:
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["frame_index", "pc_t_ns", "perf_t_ns", "pts_us"])
                while True:
                    item = self.frame_queue.get()
                    if item is None:
                        break
                    frame, pc_t_ns, perf_t_ns = item
                    if encoder is None:
                        # H.264 with yuv420p needs even dimensions.
                        height, width = frame.shape[0] & ~1, frame.shape[1] & ~1
                        encoder = VideoEncoder(self.session_dir / "video.part.mp4", width, height, self.args.fps)
                        first_perf = perf_t_ns
                        pts_us = 0
                    else:
                        # Real capture times give a true variable-frame-rate video.
                        pts_us = max(last_pts + 1, (perf_t_ns - first_perf) // 1000)
                    encoder.encode(frame[:height, :width], pts_us)
                    writer.writerow([self.frames_written, pc_t_ns, perf_t_ns, pts_us])
                    self.frames_written += 1
                    last_pts = pts_us
                    if time.monotonic() - last_flush >= FLUSH_INTERVAL_S:
                        file.flush()
                        last_flush = time.monotonic()
        except Exception as exc:
            self.encoder_error = f"{type(exc).__name__}: {exc}"
            # Keep draining so stop_recording() can still finish.
            while self.frame_queue.get() is not None:
                pass
        finally:
            if encoder is not None:
                try:
                    encoder.close()
                except Exception as exc:
                    self.encoder_error = self.encoder_error or f"closing video: {exc}"

    def _on_csi_packet(self, packet) -> None:
        """Called once per CSI record on the receiver thread."""
        if not self.recording.is_set():
            return
        pc_t_ns, perf_t_ns = time.time_ns(), time.perf_counter_ns()
        sequence = packet.raw_stream_sequence or 0
        if self.last_stream_sequence is not None:
            self.csi_sequence_gaps += max(0, sequence - self.last_stream_sequence - 1)
        self.last_stream_sequence = sequence
        self.latest_packet_drop_total = packet.raw_drop_total or 0
        noise = packet.noise_floor_dbm if packet.noise_floor_dbm is not None else NOISE_FLOOR_UNKNOWN
        fields = (
            pc_t_ns, perf_t_ns, packet.device_ticks_us or 0, packet.wifi_rx_ts_us or 0, packet.seq_num,
            sequence, packet.raw_drop_total or 0, packet.raw_send_backpressure_total or 0,
            packet.rssi_dbm or 0, noise, packet.channel or 0,
        )
        if not self.csi_writer.write(fields, packet.iq_raw):
            self.csi_size_mismatch += 1
            return
        self.csi_count += 1
        self.csi_receive_times.append(perf_t_ns)
        self.csi_device_ticks.append(packet.device_ticks_us or 0)

    def _csi_loop(self) -> None:
        try:
            self.receiver.run()
        except Exception as exc:
            if self.recording.is_set():
                self.csi_error = f"{type(exc).__name__}: {exc}"

    def _diagnostic_field_names(self) -> list:
        """Ask only for fields this firmware lists: an unknown name fails the request."""
        try:
            catalog = self.client.request("get", "diagnostics", {"fields": []}, timeout=2.0)
            names = {item.get("name") for item in catalog.get("fields", []) if isinstance(item, dict)}
        except Exception:
            return ["*"]
        if not names:
            return ["*"]
        fields = [name for name in DIAGNOSTIC_FIELDS if name in names]
        if any(str(name).startswith("raw_csi.") for name in names):
            fields.append("raw_csi")
        return fields or ["*"]

    def _diagnostics_loop(self) -> None:
        fields = self._diagnostic_field_names()
        next_poll = time.monotonic()
        while not self.quit.wait(max(0.0, next_poll - time.monotonic())):
            next_poll += 1.0
            pc_t_ns = time.time_ns()
            try:
                values = self.client.request("get", "diagnostics", {"fields": fields}, timeout=2.0)
            except Exception as exc:
                self.diagnostics_status = f"error: {exc}"
                continue
            self.diagnostics_status = "ok"
            self.latest_diagnostics = values
            with self.diagnostics_lock:
                if self.diagnostics_writer is not None:
                    raw = values.get("raw_csi") if isinstance(values.get("raw_csi"), dict) else {}
                    # A field the firmware does not report stays empty, never zero.
                    self.diagnostics_writer.writerow(
                        [pc_t_ns] + [values.get(name, "") for name in DIAGNOSTIC_FIELDS]
                        + [raw.get(name, "") for name in RAW_CSI_FIELDS]
                    )
                    self.diagnostics_file.flush()

    # Recording -------------------------------------------------------------------------

    def begin_recording(self) -> None:
        # Open /csi first: if the board refuses, nothing has been created on disk.
        self.receiver.add_callback(self._on_csi_packet)
        try:
            self.receiver.start()
        except RuntimeError as exc:
            if "409" in str(exc):
                raise RuntimeError("the board already has a /csi session open (HTTP 409); close the other client") from exc
            raise

        args = self.args
        now = datetime.now()
        self.session_dir = Path(args.out) / now.strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(self.session_dir / "csi_meta.json", {"target": args.target, "record_version": 8, **self.board_info})
        write_json_atomic(self.session_dir / "session.json", {
            "status": "recording",
            "start_time": now.isoformat(timespec="milliseconds"),
            "start_t_ns": time.time_ns(),
            "requested_duration_s": args.duration,
            "video": {"camera": self.camera_settings},
            "sync_offset_ms": 0,
        })

        self.csi_writer = CsiStreamWriter(self.session_dir / "csi_stream.bin")
        with self.diagnostics_lock:
            self.diagnostics_file = open(self.session_dir / "diagnostics.csv", "w", newline="", encoding="utf-8")
            self.diagnostics_writer = csv.writer(self.diagnostics_file)
            self.diagnostics_writer.writerow(["pc_t_ns"] + DIAGNOSTIC_FIELDS + [f"raw_csi_{name}" for name in RAW_CSI_FIELDS])

        self.encoder_thread = threading.Thread(target=self._encoder_loop, name="encoder", daemon=True)
        self.encoder_thread.start()
        self.start_perf = time.perf_counter_ns()
        self.recording.set()
        self.csi_thread = threading.Thread(target=self._csi_loop, name="csi", daemon=True)
        self.csi_thread.start()

    def elapsed_s(self) -> float:
        return (time.perf_counter_ns() - self.start_perf) / 1e9 if self.recording.is_set() else 0.0

    def failure(self) -> str | None:
        return self.csi_error or self.encoder_error or self.camera_error

    def stop_recording(self, reason: str) -> dict:
        end_ns, end_iso = time.time_ns(), datetime.now().isoformat(timespec="milliseconds")
        self.recording.clear()

        # Closing the connection is what ends the /csi session on the board.
        self.receiver.stop()
        self.csi_thread.join(timeout=5.0)
        self.frame_queue.put(None)
        self.encoder_thread.join(timeout=30.0)
        self.csi_writer.close()
        with self.diagnostics_lock:
            self.diagnostics_writer = None
            self.diagnostics_file.close()

        session = read_json(self.session_dir / "session.json")
        session.update({
            "end_time": end_iso,
            "end_t_ns": end_ns,
            "duration_s": (end_ns - session["start_t_ns"]) / 1e9,
            "stop_reason": reason,
            "run": {
                "video_queue_drops": self.video_queue_drops,
                "csi_size_mismatch": self.csi_size_mismatch,
                "board_raw_drop_total": self.receiver.raw_drop_total,
                "board_send_backpressure_total": self.receiver.raw_send_backpressure_total,
            },
        })
        write_json_atomic(self.session_dir / "session.json", session)
        return finalize_session(self.session_dir, repo=self.args.repo)

    def close(self) -> None:
        self.quit.set()
        for thread in self.threads:
            thread.join(timeout=3.0)
        if self.camera is not None:
            self.camera.release()
        self.client.close()

    # Live numbers ----------------------------------------------------------------------

    def video_fps(self) -> float:
        times = list(self.camera_times)
        return (len(times) - 1) * 1e9 / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0

    def csi_rate(self) -> float:
        """CSI frames received during the last second."""
        now = time.perf_counter_ns()
        return float(sum(1 for t in list(self.csi_receive_times) if now - t <= 1_000_000_000))

    def host_occupancy(self) -> float:
        """Share of the expected CSI time slots filled during the last second.

        Packets of a /csi session skip the board's own sampler, so its
        csi_occupancy reads 0 while recording. This uses device timestamps.
        """
        ticks = np.array(list(self.csi_device_ticks), dtype=np.int64)
        if ticks.size == 0:
            return 0.0
        slot_us = max(1, int(1_000_000 / self.expected_pps))
        recent = ticks[ticks > ticks[-1] - 1_000_000]
        return min(1.0, np.unique(recent // slot_us).size / self.expected_pps)

    def status_lines(self) -> list:
        diagnostics = self.latest_diagnostics
        if self.recording.is_set():
            limit = f" / {self.args.duration:.0f}s" if self.args.duration else ""
            return [
                f"REC {self.elapsed_s():6.1f}s{limit}   Q = stop",
                f"CSI {self.csi_rate():5.0f}/s   occupancy {self.host_occupancy() * 100:3.0f}% (host)",
                f"Video {self.video_fps():5.1f} fps   frames {self.frames_written}",
                f"Dropped  video queue {self.video_queue_drops}   CSI seq {self.csi_sequence_gaps}   "
                f"board {self.latest_packet_drop_total}",
            ]
        occupancy = diagnostics.get("csi_occupancy")
        occupancy_text = f"{occupancy * 100:.0f}%" if isinstance(occupancy, (int, float)) else "--"
        return [
            "PREVIEW   S = start   Q = quit",
            f"Video {self.video_fps():5.1f} fps   {self.camera_settings['actual']['width']}x{self.camera_settings['actual']['height']}",
            f"Board  CSI {diagnostics.get('csi_accepted_pps', '--')}/s   occupancy {occupancy_text}   "
            f"RSSI {diagnostics.get('wifi_rssi_dbm', '--')}   [{self.diagnostics_status[:40]}]",
        ]


def draw_status(frame: np.ndarray, lines: list) -> np.ndarray:
    """Return a copy of the frame with a dark status box in the top-left corner."""
    view = frame.copy()
    height = 12 + 26 * len(lines)
    box = view[:height, :min(760, view.shape[1])]
    box[:] = (box * 0.35).astype(np.uint8)
    for n, line in enumerate(lines):
        cv2.putText(view, line, (10, 28 + 26 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def run_recorder(args, camera_factory=open_webcam):
    """Run one recording. Returns (session folder, summary), or (None, None) if quit early."""
    recorder = Recorder(args, camera_factory)
    recorder.start_preview()
    show_window = not args.no_window
    start_requested = args.autostart
    message = ""
    result = (None, None)
    try:
        if show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            key = -1
            frame = recorder.latest_frame
            if show_window:
                if frame is not None:
                    cv2.imshow(WINDOW_NAME, draw_status(frame, recorder.status_lines() + ([message] if message else [])))
                key = cv2.waitKey(15) & 0xFF
                # Only meaningful once something has been shown in the window.
                window_closed = frame is not None and cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            else:
                time.sleep(0.02)
                window_closed = False

            is_recording = recorder.recording.is_set()
            stop_requested = key in (ord("q"), ord("Q")) or window_closed
            if not is_recording:
                if stop_requested:
                    break
                if recorder.camera_error:
                    raise SystemExit(f"Camera error: {recorder.camera_error}")
                if key in (ord("s"), ord("S")) or start_requested:
                    start_requested = False
                    try:
                        recorder.begin_recording()
                        message = ""
                        print(f"Recording to {recorder.session_dir}")
                    except Exception as exc:
                        message = f"Could not start: {exc}"
                        print(message)
                        if args.autostart:
                            raise SystemExit(message)
                continue

            reason = None
            if stop_requested:
                reason = "user"
            elif args.duration and recorder.elapsed_s() >= args.duration:
                reason = "duration"
            elif recorder.failure():
                reason = f"error: {recorder.failure()}"
                print(f"Stopping early, {reason}")
            if reason:
                folder = recorder.session_dir
                print("Stopping and finishing files...")
                result = (folder, recorder.stop_recording(reason))
                break
    except KeyboardInterrupt:
        # Ctrl+C still closes /csi and finishes the files.
        if recorder.recording.is_set():
            print("Ctrl+C: stopping and finishing files...")
            result = (recorder.session_dir, recorder.stop_recording("ctrl+c"))
    finally:
        if recorder.recording.is_set():
            # An unexpected exception: release the board and leave files for --recover.
            recorder.recording.clear()
            recorder.receiver.stop()
        recorder.close()
        if show_window:
            cv2.destroyAllWindows()
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Record WiFi CSI and webcam video on one PC clock.")
    parser.add_argument("--target", help="Board IP address (port 62587 is assumed)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--duration", type=float, default=0, help="Stop after this many seconds (0 = until Q)")
    parser.add_argument("--out", default="sessions", help="Parent folder for sessions")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--autostart", action="store_true", help="Start at once instead of waiting for S")
    parser.add_argument("--no-window", action="store_true", help="No preview window (needs --autostart and --duration)")
    parser.add_argument("--repo", help="Path to the ESPectre clone (default ../espectre)")
    parser.add_argument("--recover", metavar="FOLDER", help="Finish a session that did not stop cleanly")
    args = parser.parse_args(argv)
    if not args.recover and not args.target:
        parser.error("--target is required")
    if args.no_window and not (args.autostart and args.duration):
        parser.error("--no-window needs --autostart and --duration, otherwise nothing could stop it")
    return args


def main() -> None:
    args = parse_args()
    if args.recover:
        folder = Path(args.recover)
        print_summary(folder, finalize_session(folder, status="recovered", repo=args.repo))
        return
    if sys.version_info < (3, 13) and sys.platform == "win32":
        print("Note: before Python 3.13, time.time_ns() on Windows only ticks about every 15 ms.")
    folder, session = run_recorder(args)
    if folder is not None:
        print_summary(folder, session)
    else:
        print("Nothing was recorded.")


if __name__ == "__main__":
    main()
