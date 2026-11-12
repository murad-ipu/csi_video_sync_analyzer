# CLAUDE.md

This file gives Claude Code guidance for working in this repository. README.md has the user guide (install, recording, replay, sync test, session file formats). This file covers what you need to change the code safely.

## What this is

A Windows tool that records WiFi CSI from an ESPectre Native board (ESP32, HTTP API on port 62587) and webcam video on one PC clock, then replays both in the browser frame by frame.

- `record.py` handles capture and session finalization (CLI with an OpenCV preview window).
- `player.py` is the FastAPI and uvicorn server for replay, plus the `--estimate-sync` mode.
- `timesync.py` holds the pure-numpy alignment math. It has no I/O.
- `espectre_path.py` puts the sibling ESPectre clone on `sys.path`.
- `static/` is a plain JS page with no build step. It uses a vendored uPlot and draws on canvas.

## Commands

```powershell
.venv\Scripts\Activate.ps1                 # venv in .venv (Python 3.14 currently)
python -m pytest tests -q                  # full suite, about 11 s, no hardware needed
python -m pytest tests/test_timesync.py -q # one file
python -m pytest tests -q -k offset        # by name
python tests\make_demo_session.py sessions\demo --seconds 60 --offset-ms 80
python player.py sessions\demo             # http://localhost:8000
python record.py --target <IP> --camera 0 --duration 300 --out sessions/
python record.py --recover sessions\<folder>
```

No linter or formatter is configured. Lines run up to about 120 characters.

## Hard external dependency: the ESPectre clone

This repository copies no ESPectre code. The clone at `..\espectre` (or `--repo`, or `ESPECTRE_REPO`) must exist, and this applies to the tests too, because `tests/conftest.py` calls `espectre_path.setup()`. Imports from the clone are deferred into functions, after `espectre_path.setup(repo)`:
- `tools.lib.csi_io`: `DirectRawCSIReceiver` (the /csi V8 stream client), `normalize_stored_csi_bin_layout`, `load_npz_arrays` (subcarrier layout check), and the V8 struct constants used by `tests/fake_board.py`.
- `src.python.espectre_cli.device_transport.DirectClient`: GET requests to `/device`, `/capabilities`, `/sensing`, `/wifi`, and `/diagnostics`.

Do not reimplement the V8 parser or the layout logic here. Always go through the clone.

## Invariants to preserve

- **The recorder only sends GET requests to the board.** It never changes settings. `test_recorder_end_to_end.py` asserts this. Opening `/csi` starts a session on the board, which pauses motion events. Closing the connection (`receiver.stop()`) ends the session. A second `/csi` client gets HTTP 409.
- **Sync offset sign:** a positive `sync_offset_ms` moves CSI *later*. A CSI frame at `csi_t` lines up with the video frame at `csi_t + offset`. This holds in `timesync.py`, `player.py`, and `app.js` (`state.offsetS`). Camera delay usually needs a positive offset.
- **Clocks:** video frames are stamped with `time.time_ns()` and `perf_counter_ns()` right after `camera.grab()`, before `retrieve()`. CSI times in `csi.npz` (`csi_t_ns`) are device ticks mapped to PC ns by `timesync.fit_device_to_pc`. That function runs Theil-Sen through the earliest arrival in each window, because network delay is never negative. `pc_receive_t_ns` keeps the raw arrival time.
- **Video PTS** are integer microseconds (`VIDEO_TIME_BASE = 1/1e6`), taken from real capture times. That makes the video variable frame rate, and `frames.csv` `pts_us` matches the MP4 exactly. The player maps the browser `mediaTime` to a frame index through `frame_pts`, so never assume a constant fps.
- **Crash safety:** while recording, data goes to an append-only `csi_stream.bin` (a 16-byte header `CSIVSYNC` + iq_bytes, then fixed-size records from `csi_record_dtype`) and a fragmented `video.part.mp4`. `finalize_session()` is shared by a clean stop and `--recover`. It builds `csi.npz` and remuxes to a faststart `video.mp4` without re-encoding, trims `frames.csv` to the packets that survived, and deletes the intermediates only after the final files exist. All JSON writes use `write_json_atomic`.
- **IQ order:** `csi_data` holds int8 pairs with Q first, then I, as in the V8 payload. Amplitude is `hypot` of each pair.
- **Diagnostics:** the recorder asks only for fields that the firmware's `/diagnostics` catalog lists, because an unknown field fails the request. Missing values are written as empty, never as 0.
- The board's own `csi_occupancy` reads 0 during a `/csi` session. That is why `Recorder.host_occupancy()` computes it from device ticks.

## Threads in record.py

`Recorder` runs four threads:
- camera (`_camera_loop`)
- diagnostics (1 Hz poll; it also runs in preview)
- encoder (`_encoder_loop`, fed by a bounded `frame_queue`; it counts drops and never blocks capture)
- csi (`receiver.run()`, which calls `_on_csi_packet` on the receiver thread)

The `recording` and `quit` events gate everything. `stop_recording()` ends the run in this order: stop the receiver, put `None` on the queue as a sentinel, join the threads, then finalize. `run_recorder(args, camera_factory)` is the entry point the tests use, with `--autostart --no-window`.

## Player data flow

`load_session()` computes everything once: it selects subcarriers (drops null and guard bins), normalizes each frame by its mean (this removes AGC), and computes motion as the centered sliding-window variance. It then serializes each array to raw bytes. `meta.arrays` gives the dtype (`f8` or `f4`) for each name. The browser fetches `/api/meta` and `/api/array/{name}`, and does all lookups client-side with a binary search. The browser needs `requestVideoFrameCallback` (Chrome or Edge) for exact frames. `POST /api/sync_offset` (range ±2000 ms) persists the offset to `session.json`. If you add an array, add it in `load_session()` and to the fetch list in `app.js` `start()`.

## Tests

- `test_timesync.py`: unit tests on synthetic data.
- `test_player.py`: signal functions, the FastAPI `TestClient`, and the sync estimate on a demo session.
- `test_recorder_end_to_end.py`: a module-scoped fixture records 6 s from `FakeBoard` (a local HTTP server that emits V8 frames in bursts with a 40 ppm drift) and `SyntheticCamera`, then checks every output file.
- `make_demo_session.py`: both a CLI and a test helper.
