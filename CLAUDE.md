# CLAUDE.md

This file gives Claude Code guidance for working in this repository. README.md has the user guide (install, recording, analysis, replay, sync test, session files, and the table of what comes from ESPectre). This file covers what you need to change the code safely.

## What this is

A Windows tool that records WiFi CSI from one or more ESPectre Native boards (ESP32, HTTP API on port 62587) and webcam video on one PC clock, analyzes each session once, then replays both in the browser frame by frame.

- `record.py` handles capture, session finalization, and the preview window (OpenCV). It runs the analysis at the end.
- `analyze.py` computes every derived value once into `<session>/derived/`, with `manifest.json` deciding when to recompute.
- `espectre_bridge.py` holds every calculation taken from ESPectre: amplitude and phase (`CSIPacket`), and the offline replay of the Lightweight and High-Accuracy detectors.
- `player.py` is the FastAPI and uvicorn server for replay, plus the `--estimate-sync` mode. It only loads results.
- `frequency.py` holds this app's own signal math (subcarrier selection, per-frame normalization, the motion signal, STFT, presence evidence) and the frequency disk cache.
- `timesync.py` holds the pure-numpy alignment math. It has no I/O.
- `espectre_path.py` puts the sibling ESPectre clone on `sys.path` and runs ESPectre's own `tools.lib.bootstrap.setup_paths()`.
- `static/` is a plain JS page with no build step. It uses a vendored uPlot and draws on canvas. `app.js` is the core; `frequency.js`, `details.js`, `espectre.js` and `presence.js` are per-feature scripts loaded before it.

## Commands

```powershell
.venv\Scripts\Activate.ps1                  # venv in .venv (Python 3.14 currently)
python -m pytest tests -q                   # full suite, about 25 s, no hardware needed
python -m pytest tests/test_analyze.py -q   # one file
python -m pytest tests -q -k replay         # by name
python tests\make_demo_session.py sessions\demo --seconds 60 --offset-ms 80
python analyze.py sessions\<folder> [--recompute]        # every board of the recording
python player.py sessions\demo [--recompute] # http://localhost:8000
python player.py sessions\<folder> --device <name>      # one board of several
python record.py --target <IP> --camera 0 --duration 300 --out sessions/
python record.py --target a=<IP1> --target b=<IP2> ...   # several boards, one folder each
python record.py --recover sessions\<folder>
```

No linter or formatter is configured. Lines run up to about 120 characters. Check JS syntax with `node --check static/<file>.js`.

## Hard external dependency: the ESPectre clone

This repository copies no ESPectre code. The clone at `..\espectre` (or `--repo`, or `ESPECTRE_REPO`) must exist, and this applies to the tests too, because `tests/conftest.py` calls `espectre_path.setup()`. Imports from the clone are deferred into functions, after `espectre_path.setup(repo)`.

- **ESPectre already provides it:** call it from the clone, through `espectre_bridge.py`. Do not reimplement the V8 parser, the layout logic, amplitude or phase, or any detector, sampler, cadence, calibration or hit rule.
- **ESPectre has no equivalent:** the calculation is this app's own and belongs in `frequency.py` or `timesync.py`.

The README table "What comes from ESPectre" lists every function used. Keep it current.

- **Replay loop:** `espectre_bridge._replay` mirrors `tools/lib/performance_report.replay_idle_stream`, call for call, but also keeps score, threshold and readiness. `test_analyze.py` asserts both give identical states. If ESPectre changes that function, update `_replay` to match.
- **Private helpers:** the bridge uses a few private helpers from `performance_report` (`_packet_csi_data`, `_packet_timestamp_us`, `_is_scored_replay_evaluation`). An ESPectre update can break them; the tests will show it.

## Invariants to preserve

- **Session layout:**
  - A recording folder holds the shared `video.mp4`, `frames.csv` and a recording-wide `session.json` (with `devices`). Each board has its own subfolder (`parse_target` gives its name) with `csi.npz`, `csi_meta.json`, `diagnostics.csv`, `csi_stream.bin` while recording, its own `session.json` (clock fit, `sync_offset_ms`, `player_settings`), `derived/` and `frequency_cache/`. New recordings always use subfolders, even with one board.
  - Older single-board sessions keep everything in one folder and must keep working.
  - `analyze.device_folders()` finds the board folders (the folder itself when it has `csi_meta.json`), and `analyze.video_folder()` / `analyze.input_path()` find the shared video files from a device folder. Never open `frames.csv` or `video.mp4` as `folder / name` in analysis or player code.
  - Analysis, the player and every cached result work on one device folder. The frequency cache key has no t0, and t0 is per board (first frame or that board's first CSI record).

- **Board requests are GET only, except one.** The single exception is the detector switch: the **D** key in preview sends `PATCH /sensing {"detector": ...}` via `Recorder.switch_detector()`, which brings every board to the detector the first board is not running (one PATCH per board that is not already on it, through `Board.switch_detector()`).
  - The switch is refused while recording.
  - A `sensing_lock` stops the 1 Hz `/sensing` refresh from overwriting a switch result.
  - `test_recorder_end_to_end.py` asserts that a recording without D is GET only, and that D sends exactly one PATCH per board that needs it.
- **The `/csi` session:**
  - Opening `/csi` starts a session on the board, which pauses motion events.
  - Closing the connection (`receiver.stop()`) ends the session.
  - A second `/csi` client gets HTTP 409.
  - `Recorder.begin_recording()` opens `/csi` on every board before creating any file; if one fails, the boards already opened are stopped again. Each attempt uses a fresh receiver (`Board.open_csi()`).
  - `Board.open_csi()` re-reads `/sensing` so `csi_meta.json` holds the settings in force at S.
- **Sync offset sign:** a positive `sync_offset_ms` (stored per board) moves CSI *later*. A CSI frame at `csi_t` lines up with the video frame at `csi_t + offset`. This holds in `timesync.py`, `player.py`, and the JS (`state.offsetS`). Cached results (frequency, ESPectre) are stored in CSI time *without* the offset, and the page adds it when drawing.
- **Clocks:** video frames are stamped with `time.time_ns()` and `perf_counter_ns()` right after `camera.grab()`, before `retrieve()`. CSI times in `csi.npz` (`csi_t_ns`) are device ticks mapped to PC ns by `timesync.fit_device_to_pc`. That function runs Theil-Sen through the earliest arrival in each window, because network delay is never negative. `pc_receive_t_ns` keeps the raw arrival time.
- **Session time zero:** session seconds count from `analyze.session_t0_ns()`, whichever of the first frame or first CSI record came first. The player and the analysis must use the same t0, because the frequency cache key does not include it.
- **Video PTS** are integer microseconds (`VIDEO_TIME_BASE = 1/1e6`), taken from real capture times. That makes the video variable frame rate, and `frames.csv` `pts_us` matches the MP4 exactly. The player maps the browser `mediaTime` to a frame index through `frame_pts`, so never assume a constant fps.
- **Crash safety while recording:**
  - Each board's data goes to its own append-only `<device>/csi_stream.bin` (a 16-byte header `CSIVSYNC` + iq_bytes, then fixed-size records from `csi_record_dtype`) and a fragmented `video.part.mp4`.
  - `finalize_session()` is shared by a clean stop and `--recover`, and takes the recording folder. It builds each board's `csi.npz` (`_finalize_csi`), writes each board's `session.json` and the recording's `devices` summary, remuxes to a faststart `video.mp4` without re-encoding, and trims `frames.csv` to the packets that survived.
  - It deletes the intermediates only after the final files exist.
  - Analysis runs after finalization. A failure there only prints a warning and never costs the recording.
  - All JSON writes are atomic.
- **The contents of `csi.npz`:**
  - `csi_data` holds raw int8 pairs with Q first, then I, as in the V8 payload.
  - `amplitude` and `phase` are ESPectre's `CSIPacket` values for I + jQ.
  - `csi_target_pps` is the board's rate, so ESPectre's replay doesn't fall back to a measured rate.
  - Older recordings lack `phase` and `csi_target_pps`. `analyze.py` writes their values to `derived/csi_values.npz` and never modifies `csi.npz`, `frames.csv`, `video.mp4` or `csi_meta.json`.
- **Compute once:**
  - The player only loads `derived/` results. `analyze.ensure_analysis()` recomputes when the manifest fingerprint changes: input file hashes, a hash of `CODE_FILES`, the ESPectre commit and its uncommitted diff, or parameters.
  - If you add a module whose code affects results, add it to `analyze.CODE_FILES`.
  - The page never recomputes values Python already sends (amplitude, phase, mean amplitude, normalization).
- **ESPectre replay settings:**
  - The detector the board was running uses the board's threshold from `/sensing`.
  - The other one uses ESPectre's default: High-Accuracy uses its trained 0.5, and Lightweight uses `build_calibrated_lightweight_detector` on the recording.
  - The on/off hits always come from `/sensing`.
  - Both detectors share one 250 ms evaluation grid, and the player relies on that (it sends a single `esp_t`).
  - State codes are 0 = not ready, 1 = idle, 2 = motion, after hit filtering.
- **Diagnostics:** the recorder asks only for fields that the firmware's `/diagnostics` catalog lists, because an unknown field fails the request. Missing values are written as empty, never as 0.
- The board's own `csi_occupancy` reads 0 during a `/csi` session. That is why `Recorder.host_occupancy()` computes it from device ticks.

## Threads in record.py

`Recorder` owns the camera, the encoder and one `Board` per `--target`. Everything about one board (HTTP client, receiver, CSI writer, diagnostics, detector switch, live numbers) lives in `Board`. The threads are:
- camera (`Recorder._camera_loop`)
- encoder (`Recorder._encoder_loop`, fed by a bounded `frame_queue`; it counts drops and never blocks capture)
- per board, diagnostics (`Board.diagnostics_loop`, a 1 Hz poll that runs in preview too, and refreshes `/sensing` there)
- per board, csi (`receiver.run()`, which calls `Board._on_csi_packet` on the receiver thread)
- per board, a short-lived switch thread for D

Every board request goes through `Board.request()`, which holds that board's `client_lock`. The recorder's `recording` and `quit` events are shared by every board and gate everything.

`stop_recording()` ends the run in this order: stop every board's receiver, put `None` on the queue as a sentinel, join the threads, write each board's run counters (`Board.close_files`), finalize, then analyze every board. `run_recorder(args, camera_factory)` is the entry point the tests use, with `--autostart --no-window`.

## Player data flow

`resolve_device()` picks the board folder (the only one, or `--device`). `load_session()` takes that device folder, reads `csi.npz` (through ESPectre's `load_npz_arrays`) and `frames.csv`, calls `analyze.ensure_analysis()`, and loads the results with `analyze.load_results()`. It then serializes each array to raw bytes. `meta.arrays` gives the dtype of each name (`f8`, `f4`, `i1`, `i8` or `u1`), and `ARRAY_TYPES` in `app.js` maps it to a typed array.

- **Fetching and lookup:** the browser fetches `/api/meta` and `/api/array/{name}`, and does all lookups client-side with a binary search. The browser needs `requestVideoFrameCallback` (Chrome or Edge) for exact frames.
- **Lazy arrays:** the arrays in `meta.detail_arrays` are fetched only when a details view opens.
- **ESPectre arrays:** `esp_t`, then `esp_<detector>_score`, `_threshold` and `_state`. The summary is in `meta.espectre`.
- **Saved state:** `POST /api/sync_offset` (range ±2000 ms) and `POST /api/player_settings` (mode, details panel, ESPectre panel, frequency settings) persist to `session.json`.
- **Presence:** `derived/presence.npz` holds one point per second (`pres_t`, `pres_breathing`, `pres_breathing_hz`, `pres_level`; the per-subcarrier `shape` stays on the server). `GET`/`POST /api/presence/baseline` reads or saves the empty baseline in `player_settings.presence` (CSI time) and returns each point's shape distance and the level floor, computed by `frequency.presence_baseline`. `POST /api/player_settings` keeps that key unchanged. The manifest stores the presence parameters through a JSON round trip, because a tuple would never compare equal to the list read back.
- **Adding an array:** compute it in `analyze.py` if it is derived, then add it in `load_session()`, and fetch it in `app.js` `start()` or in the feature script.

## Tests

- `test_timesync.py`: unit tests on synthetic data.
- `test_frequency.py`: the STFT math, the frequency API, and its cache.
- `test_presence.py`: the presence features on synthetic data (breathing rate, drift removal, gaps, baseline distance, level).
- `test_analyze.py` covers:
  - ESPectre amplitude equals the old `hypot` formula, on the demo session and on every real recording in `sessions/` (those tests are skipped when none exist)
  - the replay equals `replay_idle_stream`
  - the manifest and cache behavior, and that the player does not recompute
- `test_player.py`: signal functions, the FastAPI `TestClient`, the details and ESPectre arrays, and the sync estimate on a demo session.
- `test_recorder_end_to_end.py`: a module-scoped fixture records 6 s from two `FakeBoard`s at once (a local HTTP server that emits V8 frames in bursts with a 40 ppm drift, and accepts only a `PATCH /sensing` detector switch) and `SyntheticCamera`. It then checks the layout, every output file, each board's analysis, the player on one board, the detector switch, a refused start on one board, and `--recover` for both layouts.
- `make_demo_session.py`: both a CLI and a test helper. It writes the same `csi.npz` format as a real recording.
