# CSI + video sync

Record WiFi CSI from an ESPectre Native board and webcam video on one PC clock, then replay them together in the browser, frame by frame.

It only sends GET requests to the board, so settings and the traffic source never change. The one exception is the **D** key in the preview window, which switches the board's detector before you start (see [Record](#record)). While recording, the board pauses its motion events, and it resumes sensing when the recording stops.

Each recording is then analyzed once: ESPectre's own Lightweight and High-Accuracy detectors are replayed over it offline, and the player shows their scores and states next to the video.

## Install (Windows, once)

Needs Python 3.12+ (3.13+ recommended, because older versions read the Windows clock only every 15 ms) and the ESPectre clone at `..\espectre`. Use `--repo <path>` or the `ESPECTRE_REPO` variable if it is elsewhere.

```powershell
cd D:\Murad\Job\ESPectre\csi_video_sync
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Record

```powershell
python record.py --target <BOARD_IP> --camera 0 --duration 300 --out sessions/
```

In the preview window, press **S** to start and **Q** to stop. `--duration` counts from S. Other options: `--width`, `--height`, `--fps`, `--autostart`.

The status box shows elapsed time, CSI frames per second, occupancy, video fps, and drops. During recording, occupancy is computed on the PC, because the board's own `csi_occupancy` reads 0 during a `/csi` session.

Before recording, the status box also shows the board's motion detector (`lightweight` or `high_accuracy`), its threshold, the motion on/off hit counts, and whether it is ready or calibrating. Press **D** to switch the detector. This sends one `PATCH /sensing` request; it is the only request the recorder sends that is not a GET. D works only before S: during a recording the board refuses setting changes, and it runs no detector anyway. After a switch to Lightweight the board recalibrates, so keep the room quiet for about 10 s. The settings in force when you press S are saved in `csi_meta.json`.

When the recording stops, the session is analyzed automatically (a few seconds for 5 minutes, see [Analyze](#analyze)).

If the PC or program crashes, finish the session with the command below. It also runs the analysis.

```powershell
python record.py --recover sessions\<folder>
```

## Analyze

```powershell
python analyze.py sessions\<folder>              # only computes what is missing or out of date
python analyze.py sessions\<folder> --recompute  # always recompute
```

This computes everything the player shows, once, and saves it in the session's `derived` folder:

- **Amplitude and phase** of every subcarrier, from ESPectre's `CSIPacket`. New recordings already store them in `csi.npz`. For older recordings they are saved in `derived\csi_values.npz`; `csi.npz` is never changed.
- **The player's signals**: which subcarriers carry signal, per-frame normalized amplitude, the 0.5 s motion signal, and each frame's mean amplitude.
- **ESPectre detector replay**: both detectors, packet by packet, one row per 250 ms evaluation (see [ESPectre detectors](#espectre-detectors)).
- **The Frequency view** for the page's first settings, in `frequency_cache`.
- **`manifest.json`**: hashes of `csi.npz`, `frames.csv` and `csi_meta.json`, a hash of the analysis code, the ESPectre commit (plus a hash of any uncommitted changes in the clone), the parameters, and timings.

You rarely need to run it yourself: `record.py` runs it after every recording, and `player.py` runs it when the results are missing or the manifest no longer matches. Otherwise the player only loads the saved files. A changed input file, analysis code, parameter, or ESPectre version triggers a recompute; `--recompute` forces one. The `derived` folder is safe to delete.

## Replay

```powershell
python player.py sessions\<folder>
```

This opens http://localhost:8000 (use `--port` to change it). Chrome or Edge is recommended. Add `--recompute` to redo the saved analysis first.

| Key | Action |
| --- | --- |
| Space | Play or pause |
| Left / Right | Step one video frame |
| Shift+Left / Shift+Right | Step one CSI frame |

Use the timeline to jump anywhere. Offset changes save to `session.json` automatically.

## ESPectre detectors

In Time mode, the bottom panel shows ESPectre's two motion detectors, replayed offline on the recording. The **ESPectre** checkbox in the footer shows or hides it, and the player remembers the choice per session.

- **Lines**: each detector's score (solid) and threshold (dashed). Both are probabilities from 0 to 1, so they share one axis. Lightweight is orange, High-Accuracy is purple.
- **Strips**: under the lines, one strip per detector, with Lightweight on top. It is colored while the state is MOTION, grey while idle, and blank while the detector is not ready (its first second, or after a long gap in the CSI). The state includes the board's on/off hit counts, as on the board.
- **Info bar**: the **ESPectre** field shows both states and scores at the current CSI frame, for example `LW MOTION 0.77 · HA idle 0.12`.
- **Hover**: shows the time, score, threshold, and state of each detector.

Like the motion chart, the panel follows the sync offset.

### How the replay matches the board

The replay calls ESPectre's own code for every step: the temporal sampler, the 250 ms evaluation grid, readiness, subcarrier selection, filters, startup calibration, and hit counting. A test checks that it gives the same states as ESPectre's replay tool (`performance_report.replay_idle_stream`).

The board pauses detection while `/csi` is open, so its own scores cannot be recorded at the same time. The replay is therefore "what ESPectre's reference code says for this recording", not a copy of the board's output. The settings come from the `/sensing` snapshot in `csi_meta.json`:

| Setting | Detector the board was running | The other detector |
| --- | --- | --- |
| Threshold | the board's threshold | High-Accuracy: its trained default (0.5). Lightweight: ESPectre's startup calibration on the first quiet 10-30 s of the recording |
| On/off hits | the board's values | the board's values |
| Packet rate | the board's `csi_target_pps` | the same |

Lightweight's threshold also moves during a session: ESPectre lowers it after a long quiet stretch. The dashed line shows the threshold at each evaluation. If a recording starts with movement, Lightweight's own calibration can be off. The legend says where each threshold came from. To get the board's Lightweight threshold, switch the board to Lightweight with **D** before recording.

## Frequency mode

The **Time / Frequency** buttons next to Play switch the three panels on the right. The video, timeline, frame stepping, speed, and sync offset work the same in both modes. The time panels are drawn as before. The player remembers the mode and settings for each session.

The first time you use a setting it takes a moment (about 2 s for a 10-minute recording with the defaults, and up to about 20 s for the slowest settings). Results are stored in the session's `frequency_cache` folder, so going back to a setting is instant. You can delete that folder at any time.

### Panels

- **Spectrogram**: time runs left to right, frequency runs bottom to top, and brighter means more power. The white line is the current video frame. The footer's Window selector sets how much time is shown. Grey columns are windows with too much missing data.
- **Current spectrum**: power against frequency for the single analysis window nearest the current video frame. The title gives that window's center time. The shaded areas are the bands.
- **Band power**: one line per frequency band, with the same time axis and white line as the spectrogram. A break in a line is a window with too much missing data.

A spectrum needs a full window of data around its center time, so there are none in the first and last half-window of a recording. With a 20 s window, spectra start 10 s in. The panels label these stretches. Use a shorter window to see closer to the start and end.

### Settings

| Setting | What it does |
| --- | --- |
| Window | How much data goes into each spectrum. Longer windows separate nearby frequencies better but blur quick changes. |
| Hop | Time between one spectrum and the next. Smaller values look smoother and take longer to compute. |
| Combine | **Mean** averages the power of all subcarriers and is the general view. **PCA 1** finds the strongest shared pattern across subcarriers within each window and shows that pattern's spectrum. **PCA 2** shows the next strongest pattern. **Motion variance** analyzes the Time-mode motion signal instead of the amplitudes (see below). |
| Range | The frequencies shown. It only changes the drawing. |
| Auto color, Min, Max | The dB values at the dark and bright ends of the spectrogram colors. Auto picks them from the data in the visible frequency range. |
| Resolution | The spacing between frequency bins, which is 1 / window length. |
| Sample rate | The rate of the uniform grid the CSI is resampled onto (100 Hz). The highest visible frequency is half of this. |

Use a window of at least 10 to 20 s to look at breathing. A 2 s window has bins 0.5 Hz apart, so it cannot separate 0.25 Hz from nearby frequencies. If a window is too short for a band to hold any bin, the legend says so.

Mean and PCA values are not on the same dB scale, so compare levels within one method.

### Motion variance

This option takes the same motion signal the Time-mode chart draws, using every CSI frame rather than the chart's thinned-out points. That signal is the variance of each subcarrier's normalized amplitude over 0.5 s, averaged over the subcarriers. It is then resampled and analyzed like the amplitudes, with the same window, hop, gap handling, and bands.

It shows the rhythm of how active the scene is, not the rhythm of the signal itself:

- The 0.5 s variance window strongly weakens everything above about 2 Hz, so use the 0-2 Hz or 0-5 Hz range.
- Variance squares the signal, so a slow repeating movement appears at **twice** its rate. Breathing at 0.25 Hz would appear near 0.5 Hz.
- Its dB values are in variance units and cannot be compared with Mean or PCA.

For breathing and walking, the Mean and PCA options are usually more informative.

## CSI details

The **Details** button next to Time and Frequency shows what `csi.npz` holds for the CSI frame currently matched to the video, the same frame the header calls "CSI frame". The **CSI details** checkbox in the footer shows the same in a smaller panel under the video, so you can keep the charts on screen. Both follow playback, frame stepping, Shift+arrow CSI stepping, the timeline, and the sync offset. The player remembers the view and the checkbox per session.

- **Fields**: the timestamps (`csi_t_ns` is the board's capture time on the PC clock; `pc_receive_t_ns` is when the PC received it), the network delay between the two, sequence numbers, the time since the previous frame, RSSI, noise floor, channel, and the frame's mean amplitude (the value each frame is divided by when normalized).
- **Table**: one row per subcarrier with Q and I from `csi_data` and the amplitude |I + jQ|. The Details view also shows the normalized amplitude and the phase of I + jQ in radians. Amplitude and phase are ESPectre's `CSIPacket` values from the analysis; the page does not recompute them. Grey rows are guard and null subcarriers.

The page downloads this data the first time you open either view: about 4 MB per minute of recording.

### Reading the numbers

- Power is in dB of power density, using amplitude divided by each frame's mean, so the values have no physical unit and are only useful relative to each other. The noise floor stays at the same level when you change the window length.
- The timestamp of a spectrum is the center of its window, so a short event appears centered on when it happened, smeared over one window length.
- CSI frames up to 50 ms apart are joined by straight-line interpolation. Across a longer gap, nothing is invented: those samples are marked missing. A window with more than 10% missing samples is skipped and shown grey. A window with less counts its missing samples as zero after the mean is removed, and its power is corrected for the lost samples.

### Changing the bands

The bands are saved under `player_settings.frequency.bands` in `session.json` the first time the player saves settings. Edit them there, using up to 8 bands, then reload the page:

```json
"bands": [
  { "name": "breathing", "low_hz": 0.1, "high_hz": 0.5 },
  { "name": "movement", "low_hz": 0.5, "high_hz": 10 }
]
```

A band includes its upper edge but not its lower edge, so a bin on a shared edge is counted once.

## Sync test

1. Record 30 s. Stand so the camera sees you and the board.
2. Wait about 5 s, then wave a hand quickly close to the board. Do this 3 times, about 5 s apart, and hold still in between.
3. Open the session in the player. Step to the start of each wave and compare it with the motion spikes.
4. If the spikes are consistently early or late, move the sync offset slider until they line up. A positive offset moves CSI later. Camera delay usually means a positive value.
5. For a suggested value, run the command below. Trust it only if it says the result is confident, and always confirm by eye.

```powershell
python player.py sessions\<folder> --estimate-sync
```

## Session files

| File | Contents |
| --- | --- |
| `video.mp4` | H.264 video with real, variable frame times |
| `frames.csv` | `frame_index`, `pc_t_ns`, `perf_t_ns`, `pts_us` |
| `csi.npz` | `pc_receive_t_ns`, corrected `csi_t_ns`, `device_ticks_us`, sequence fields, `rssi_dbm`, `csi_data` (raw int8 pairs, Q then I), `amplitude` and `phase` (from ESPectre's `CSIPacket`), and `csi_target_pps`. Recordings made before this version have no `phase` or `csi_target_pps` |
| `csi_meta.json` | Subcarrier count, record version, and `/device`, `/capabilities`, `/sensing` (including the detector settings), `/wifi` read when S is pressed |
| `diagnostics.csv` | `/diagnostics` once per second. A field the firmware does not report stays empty |
| `session.json` | Times, counts, rates, camera settings, clock fit, `sync_offset_ms`, and `player_settings` (the player's view, CSI details checkbox, and frequency settings) |
| `derived\` | Results of `analyze.py`, with `manifest.json`. Safe to delete |
| `frequency_cache\` | Results saved by the player's frequency mode. Safe to delete |

## What comes from ESPectre, and what is this app's own

Every calculation ESPectre already provides is imported from the clone, never copied. They are all called from `espectre_bridge.py`, except the CSI receiver and the HTTP client.

| Value | ESPectre code used |
| --- | --- |
| Receiving the `/csi` stream (V8 records) | `tools/lib/csi_io.py`: `DirectRawCSIReceiver` |
| GET requests and the detector switch | `src/python/espectre_cli/device_transport.py`: `DirectClient` |
| Subcarrier order of the stored I/Q | `tools/lib/csi_io.py`: `normalize_stored_csi_bin_layout`, `load_npz_arrays` |
| Amplitude and phase per subcarrier | `tools/lib/csi_io.py`: `CSIPacket.amplitudes`, `CSIPacket.phases` |
| Packets for the replay (HT20 sensing view) | `tools/lib/csi_io.py`: `load_npz_packet_view`, `ht20_packet_mask` |
| Lightweight detector and its startup calibration | `tools/lib/lightweight_detector.py`, `tools/lib/dataset_metadata.py`: `build_calibrated_lightweight_detector` |
| High-Accuracy detector | `tools/lib/high_accuracy_detector.py`: `HighAccuracyDetector`, with `tools/lib/runtime_policy.py`: `production_filter_kwargs` |
| Temporal sampler, missing-slot handling | `tools/lib/temporal_replay.py`: `iter_temporal_admissions`, `apply_temporal_admission` |
| 250 ms evaluation grid, scored-evaluation rule | `tools/lib/performance_report.py`: `timing_cadence_for_window`, `_is_scored_replay_evaluation` |
| On/off hit counting | `src/python/micro_espectre/runtime_motion_policy.py`: `RuntimeMotionPolicy` |
| Settings (window, interval, subcarriers) | ESPectre's `config` |

ESPectre has no equivalent for these, so they are this app's own:

| Value | Where |
| --- | --- |
| Device clock to PC clock fit, video frame gaps, sync offset estimate | `timesync.py` |
| Signal subcarriers, per-frame normalization, 0.5 s motion signal | `frequency.py` (computed once by `analyze.py`) |
| Spectrogram, band power, PCA | `frequency.py` |
| Video motion energy for `--estimate-sync` | `player.py` |
| Live CSI rate, occupancy and fps in the recorder's status box | `record.py` |

Each value is computed once, in Python. The page only draws what it receives and finds the CSI frame for each video frame, which has to happen in the browser for every frame that is shown.

## Project files

- `record.py` runs the camera, encoder, CSI, and diagnostics threads and shows the live window. It writes crash-safe files during recording, converts them to their final form at the end, and then runs the analysis. CSI capture is `DirectRawCSIReceiver` from the ESPectre clone.
- `analyze.py` computes and saves every derived result of a session, and decides from `manifest.json` when they must be recomputed.
- `espectre_bridge.py` holds every calculation taken from ESPectre: amplitude and phase, and the detector replay.
- `player.py` loads the saved results and serves them with the video through a small FastAPI app. It also saves the sync offset and contains the automatic offset estimate.
- `frequency.py` holds the frequency-mode processing and its disk cache, using only numpy. The motion signal and per-frame normalization live here too.
- `timesync.py` holds the device-to-PC clock fit, the frame-to-CSI lookup, and the offset search. It has no hardware or file access, so it is tested with synthetic data.
- `espectre_path.py` finds the ESPectre clone and sets up its import paths. No ESPectre code is copied here.
- `static/` is the web page. `app.js` draws the waterfall and current-frame plots on canvas and uses a local copy of uPlot for the motion and RSSI charts. `frequency.js` draws the frequency-mode panels the same way, `espectre.js` draws the ESPectre detector panel, and `details.js` fills the CSI details tables.
- `tests/` holds the unit tests and an end-to-end recording test using a fake board and synthetic camera. `make_demo_session.py` builds a synthetic session for trying the player without hardware. Tests that need a real recording use any session in `sessions\` and are skipped when there is none.

## Tests and demo

```powershell
python -m pytest tests -q
python -m pytest tests\test_frequency.py -q   # frequency mode only
python -m pytest tests\test_analyze.py -q     # ESPectre reuse, analysis, and its cache
python tests\make_demo_session.py sessions\demo --seconds 60 --offset-ms 80
python player.py sessions\demo
```

## How to read the results

- Timing: a camera frame is stamped when it reaches the PC, after the scene happened. That delay is the main reason for a sync offset.
- Raw amplitude: automatic gain control rescales each packet, so raw amplitude can show vertical stripes that are not movement. Normalized amplitude removes that, and the motion signal always uses it.
- The clock fit cannot remove the smallest network delay, which is likely a few milliseconds.
- Blank columns in the waterfall are time with no CSI frames.
