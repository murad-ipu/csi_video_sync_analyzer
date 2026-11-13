"""A fake ESPectre Native board plus a synthetic camera, so no hardware is needed.

The board serves the read-only API and a GET /csi stream of V8 frames at about
100 per second. Frames are released in bursts and its clock runs fast, like the
real link. A second /csi request gets HTTP 409. Frames are built with the
struct definitions from tools/lib/csi_io.py. PATCH /sensing accepts only a
detector switch, and is refused (409) while /csi is open, like the real board.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from tools.lib.csi_io import (
    CSI_HEADER_STRUCT,
    RAW_CSI_FLAG_FRESH,
    RAW_CSI_FLAG_WIFI_RX_TS_VALID,
    RAW_CSI_HTTP_FRAME_STRUCT,
    RAW_CSI_PROTOCOL_VERSION,
    RAW_CSI_RECORD_MAGIC,
    RAW_CSI_RECORD_VERSION_V8,
    RAW_CSI_RESPONSE_MAGIC,
)

SESSION_ID = bytes.fromhex("00112233445566778899aabbccddeeff")
CLOCK_DRIFT_PPM = 40.0
BURST_FRAMES = 5
FRAME_INTERVAL_S = 0.01
NUM_SC = 64
# Centered HT20 layout: outer guards and the DC bin carry nothing.
NULL_BINS = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]


def payload(index):
    """int8 I/Q pairs with a fixed shape across subcarriers plus some noise."""
    rng = np.random.default_rng(index)
    values = rng.integers(10, 40, size=2 * NUM_SC, dtype=np.int8)
    for null_bin in NULL_BINS:
        values[2 * null_bin: 2 * null_bin + 2] = 0
    return values.tobytes()


def build_frame(sequence, ticks_us, raw_drop_total=0):
    iq = payload(sequence)
    record = CSI_HEADER_STRUCT.pack(
        RAW_CSI_RECORD_MAGIC, RAW_CSI_RECORD_VERSION_V8, CSI_HEADER_STRUCT.size, 3,
        RAW_CSI_FLAG_WIFI_RX_TS_VALID | RAW_CSI_FLAG_FRESH,
        sequence, len(iq) // 2, len(iq), 0x10B41DE8EC00, ticks_us, ticks_us & 0xFFFFFFFF, 0,
        6, -50, -96, 0, sequence, sequence, 2, 2, 1,
    ) + iq
    prefix = RAW_CSI_HTTP_FRAME_STRUCT.pack(
        RAW_CSI_RESPONSE_MAGIC, RAW_CSI_PROTOCOL_VERSION, RAW_CSI_RECORD_VERSION_V8,
        RAW_CSI_HTTP_FRAME_STRUCT.size, SESSION_ID, sequence, len(record), 0, sequence, raw_drop_total, 0,
    )
    return prefix + record


class FakeBoard:
    def __init__(self):
        self.lock = threading.Lock()
        self.csi_open = False
        self.csi_sessions = 0
        self.requests = []       # (method, path) of every request
        self.true_times = []     # (device ticks, true PC ns) of every frame sent
        self.detection = {"detector": "high_accuracy", "threshold": 0.5, "motion_on_hits": 4, "motion_off_hits": 3}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def resource(self, name):
        if name == "capabilities":
            return {
                "protocol_version": "1.0", "resources": [], "operations": [], "events": [],
                "features": {"csi": True},
                "csi": {"transport": "http", "protocol_version": 1, "record_version": 8, "frame_prefix_bytes": 60},
            }
        if name == "device":
            return {"device_id": "000010b41de8ec00", "name": "fake", "frontend": "native", "firmware": "3.1.0", "chip": "esp32s3"}
        if name == "sensing":
            return {"enabled": True, "ready": not self.csi_open, "calibrating": False,
                    "mode": "csi_collection" if self.csi_open else "sensing",
                    "traffic_generator_mode": "wifi_raw", "csi_target_pps": 100, **self.detection}
        if name == "wifi":
            return {"connected": True, "channel": 6, "rssi_dbm": -50}
        return None

    def diagnostics(self, body):
        fields = body.get("fields", [])
        if not fields:
            names = ["timestamp_ms", "csi_accepted_pps", "csi_occupancy", "wifi_rssi_dbm", "raw_csi.active", "raw_csi.raw_drop_total"]
            return {"fields": [{"name": name, "type": "number", "unit": ""} for name in names]}
        values = {
            "timestamp_ms": int(time.monotonic() * 1000), "csi_accepted_pps": 100.0, "wifi_rssi_dbm": -50,
            "csi_occupancy": 0.0 if self.csi_open else 0.98,
            "raw_csi": {"active": self.csi_open, "raw_drop_total": 0, "send_backpressure_total": 0},
        }
        return {key: value for key, value in values.items() if key in fields}

    def stream(self, handler):
        start_perf, start_ticks = time.perf_counter(), 2_000_000
        sequence = 0
        try:
            while True:
                chunk = bytearray()
                for _ in range(BURST_FRAMES):
                    sequence += 1
                    elapsed_s = sequence * FRAME_INTERVAL_S
                    ticks = start_ticks + int(elapsed_s * 1e6 * (1 + CLOCK_DRIFT_PPM / 1e6))
                    chunk += build_frame(sequence, ticks)
                # Release the burst once its last frame has "happened".
                wait = start_perf + sequence * FRAME_INTERVAL_S - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                handler.wfile.write(bytes(chunk))
                handler.wfile.flush()
        except OSError:
            pass  # the client closed the connection: collection ends
        finally:
            with self.lock:
                self.csi_open = False

    def _handler(self):
        board = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _json(self, status, obj):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _reject(self):
                board.requests.append((self.command, self.path))
                self._json(405, {"code": "method_not_allowed"})

            do_POST = do_PUT = do_DELETE = _reject

            def do_PATCH(self):
                """Only the detector switch is supported, as the recorder sends nothing else."""
                board.requests.append((self.command, self.path))
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
                if self.path.split("?")[0] != "/espectre/v1/sensing":
                    return self._json(405, {"code": "method_not_allowed"})
                if board.csi_open:
                    return self._json(409, {"code": "busy_raw_collection"})
                if set(body) != {"detector"} or body["detector"] not in ("lightweight", "high_accuracy"):
                    return self._json(400, {"code": "invalid_params"})
                board.detection["detector"] = body["detector"]
                return self._json(200, board.resource("sensing"))

            def do_GET(self):
                board.requests.append((self.command, self.path))
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
                path = self.path.split("?")[0]
                if path == "/espectre/v1/csi":
                    with board.lock:
                        if board.csi_open:
                            return self._json(409, {"code": "busy_raw_collection"})
                        board.csi_open = True
                        board.csi_sessions += 1
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    return board.stream(self)
                name = path.rsplit("/", 1)[-1]
                obj = board.diagnostics(body) if name == "diagnostics" else board.resource(name)
                return self._json(200, obj) if obj is not None else self._json(404, {"code": "not_found"})

        return Handler


class SyntheticCamera:
    """Stands in for cv2.VideoCapture: 30 fps with the frame number drawn in."""

    def __init__(self, width=320, height=240, fps=30.0):
        self.width, self.height, self.interval = width, height, 1 / fps
        self.index = 0
        self.next_time = time.perf_counter()
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                cv2.CAP_PROP_FPS: 1 / self.interval}.get(prop, 0.0)

    def grab(self):
        self.next_time += self.interval
        delay = self.next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self.index += 1
        return True

    def retrieve(self):
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        frame[:, self.index % self.width] = (0, 255, 0)
        return True, frame

    def release(self):
        self.released = True
