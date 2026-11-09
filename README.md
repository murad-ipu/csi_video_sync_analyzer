# CSI + video sync

Record WiFi CSI from an ESPectre Native board and webcam video on one PC clock, then replay them together in the browser, frame by frame.

It only sends GET requests to the board, so settings and the traffic source never change. While recording, the board pauses its motion events, and it resumes sensing when the recording stops.

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

If the PC or program crashes, finish the session with:

```powershell
python record.py --recover sessions\<folder>
```

## Replay

```powershell
python player.py sessions\<folder>
```

This opens http://localhost:8000 (use `--port` to change it). Chrome or Edge is recommended.

| Key | Action |
| --- | --- |
| Space | Play or pause |
| Left / Right | Step one video frame |
| Shift+Left / Shift+Right | Step one CSI frame |

Use the timeline to jump anywhere. Offset changes save to `session.json` automatically.

## Frequency mode

The **Time / Frequency** buttons next to Play switch the three panels on the right. The video, timeline, frame stepping, speed, and sync offset work the same in both modes. The time panels are drawn as before. The player remembers the mode and settings for each session.

The first time you use a setting it takes a moment (about 2 s for a 10-minute recording with the defaults, and up to about 20 s for the slowest settings). Results are stored in the session's `frequency_cache` folder, so going back to a setting is instant. You can delete that folder at any time.

### Panels

- **Spectrogram**: time runs left to right, frequency runs bottom to top, and brighter means more power. The white line is the current video frame. The footer's Window selector sets how much time is shown. Grey columns are windows with too much missing data.
- **Current spectrum**: power against frequency for the single analysis window nearest the current video frame. The title gives that window's center time. The shaded areas are the bands.
- **Band power**: one line per frequency band, with the same time axis and white line as the spectrogram. A break in a line is a window with too much missing data.

### Settings

| Setting | What it does |
| --- | --- |
| Window | How much data goes into each spectrum. Longer windows separate nearby frequencies better but blur quick changes. |
| Hop | Time between one spectrum and the next. Smaller values look smoother and take longer to compute. |
| Combine | **Mean** averages the power of all subcarriers and is the general view. **PCA 1** finds the strongest shared pattern across subcarriers within each window and shows that pattern's spectrum. **PCA 2** shows the next strongest pattern. |
| Range | The frequencies shown. It only changes the drawing. |
| Auto color, Min, Max | The dB values at the dark and bright ends of the spectrogram colors. Auto picks them from the data in the visible frequency range. |
| Resolution | The spacing between frequency bins, which is 1 / window length. |
| Sample rate | The rate of the uniform grid the CSI is resampled onto (100 Hz). The highest visible frequency is half of this. |

Use a window of at least 10 to 20 s to look at breathing. A 2 s window has bins 0.5 Hz apart, so it cannot separate 0.25 Hz from nearby frequencies. If a window is too short for a band to hold any bin, the legend says so.

Mean and PCA values are not on the same dB scale, so compare levels within one method.

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
| `csi.npz` | `pc_receive_t_ns`, corrected `csi_t_ns`, `device_ticks_us`, sequence fields, `rssi_dbm`, `csi_data` (int8 pairs, Q then I), and `amplitude` |
| `csi_meta.json` | Subcarrier count, record version, and `/device`, `/capabilities`, `/sensing`, `/wifi` read before recording |
| `diagnostics.csv` | `/diagnostics` once per second. A field the firmware does not report stays empty |
| `session.json` | Times, counts, rates, camera settings, clock fit, `sync_offset_ms`, and `player_settings` (the player's mode and frequency settings) |
| `frequency_cache\` | Results saved by the player's frequency mode. Safe to delete |

## Project files

- `record.py` runs the camera, encoder, CSI, and diagnostics threads and shows the live window. It writes crash-safe files during recording and converts them to their final form at the end. CSI capture is `DirectRawCSIReceiver` from the ESPectre clone.
- `player.py` computes the CSI signals once at startup and serves them with the video through a small FastAPI app. It also saves the sync offset and contains the automatic offset estimate.
- `frequency.py` holds the frequency-mode processing and its disk cache, using only numpy.
- `timesync.py` holds the device-to-PC clock fit, the frame-to-CSI lookup, and the offset search. It has no hardware or file access, so it is tested with synthetic data.
- `espectre_path.py` finds the ESPectre clone and adds it to `sys.path`. No ESPectre code is copied here.
- `static/` is the web page. `app.js` draws the waterfall and current-frame plots on canvas and uses a local copy of uPlot for the motion and RSSI charts. `frequency.js` draws the frequency-mode panels the same way.
- `tests/` holds the unit tests and an end-to-end recording test using a fake board and synthetic camera. `make_demo_session.py` builds a synthetic session for trying the player without hardware.

## Tests and demo

```powershell
python -m pytest tests -q
python -m pytest tests\test_frequency.py -q   # frequency mode only
python tests\make_demo_session.py sessions\demo --seconds 60 --offset-ms 80
python player.py sessions\demo
```

## How to read the results

- Timing: a camera frame is stamped when it reaches the PC, after the scene happened. That delay is the main reason for a sync offset.
- Raw amplitude: automatic gain control rescales each packet, so raw amplitude can show vertical stripes that are not movement. Normalized amplitude removes that, and the motion signal always uses it.
- The clock fit cannot remove the smallest network delay, which is likely a few milliseconds.
- Blank columns in the waterfall are time with no CSI frames.
