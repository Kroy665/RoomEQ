# Setup and usage

Everything here assumes macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/) and Homebrew.
To see the app without any audio hardware first, run `uv run roomeq demo --open`.

- [1. Install](#1-install)
- [2. Real-time EQ (BlackHole)](#2-real-time-eq-blackhole)
- [3. The iPhone as measurement microphone](#3-the-iphone-as-measurement-microphone)
- [4. Measuring and auto-tuning](#4-measuring-and-auto-tuning)
- [5. Command reference](#5-command-reference)
- [6. Configuration](#6-configuration)
- [7. Microphone calibration](#7-microphone-calibration)

## 1. Install

```bash
brew install uv
git clone <this repo> roomeq && cd roomeq
uv sync                    # Python 3.12 + dependencies into .venv
uv run roomeq --help
uv run pytest              # 70+ tests, about a minute, no hardware needed
```

## 2. Real-time EQ (BlackHole)

```
apps → macOS output "BlackHole 2ch" → RoomEQ (resampler → EQ → gain → limiter) → your speakers
```

1. `brew install blackhole-2ch`, then log out and back in so it appears.
2. Recommended: `brew install switchaudio-osx`. RoomEQ then switches the system output to BlackHole when
   it starts and back to your speakers when it quits.
3. Quit any other system-wide EQ (e.g. eqMac) and disable its launch at login.
4. While your speakers are still the selected output, set the Mac volume to about 75%. With BlackHole
   as the output the volume keys no longer reach the speakers; use the speaker's own knob or remote, or
   RoomEQ's `+`/`-`. Turn the speaker down before the first start.
5. `uv run roomeq devices` must list "BlackHole 2ch". Put your speaker device name in the config
   (`audio.output_device`, default "External Headphones" = the 3.5 mm jack).

```bash
uv run roomeq run                      # latest preset, dashboard at http://localhost:8080
uv run roomeq run "autotune 10-03 0936" --volume -6
uv run roomeq run --blocksize 128      # ~30 ms latency for video; 512 if you ever hear crackles
```

Terminal keys: `b` bypass (level-matched A/B), `space` PANIC (instant bypass, −20 dB; again to recover),
`+`/`-` volume, `r` reload preset, `Enter` continue a measurement step, `q` quit.

## 3. The iPhone as measurement microphone

The iPhone and the Mac must be on the same Wi-Fi.

1. Start `roomeq run` (or `roomeq autotune`) and scan the QR code shown in the terminal or on the dashboard.
2. **First time only**, the page walks you through trusting RoomEQ's certificate:
   download it and tap **Allow** → **Settings → General → VPN & Device Management → RoomEQ Local CA →
   Install** → **Settings → General → About → Certificate Trust Settings →** switch on **RoomEQ Local CA**.
   Reopen the QR link; from then on it goes straight to the secure page.
3. Tap **Start microphone**, allow access, and keep Safari in front with the screen on.
4. macOS may ask whether Python may accept incoming connections: click **Allow**.

No shared Wi-Fi (guest network, client isolation)? Use `--tunnel` (cloudflared or ngrok): a public HTTPS
URL, no certificate setup. The Mac also lists the iPhone as a Continuity microphone;
`roomeq measure --mic "iPhone Microphone"` uses it directly, but Apple may apply voice processing on that
path, so the web recorder is preferred.

## 4. Measuring and auto-tuning

Before measuring: set the speaker's bass/treble knobs where you keep them, choose a moderate volume
(the level check says if it's too quiet or loud), and keep the room quiet.

```bash
uv run roomeq autotune --positions 3 --iterations 2   # opens the dashboard and starts
uv run roomeq verify                                   # 4 sweeps, phone not moved: is everything right?
```

Place the phone at ear height on something (not in your hand), screen up, bottom edge towards the
speakers. Follow the prompts: tap **Ready** on the phone, press **Continue** on the dashboard or **Enter**
in the terminal. During auto-tune rounds each position gets two sweeps (EQ off, EQ on): don't move the
phone between them. Results are saved as new presets and every measurement in
`~/.roomeq/measurements/`.

If auto-tune reports that the bass stays far above the target, turn the subwoofer down on the speaker
and run it again.

## 5. Command reference

| Command | What it does |
|---|---|
| `roomeq demo [--open]` | Full app with a simulated room and virtual phone (no hardware) |
| `roomeq run [preset]` | Real-time EQ + dashboard at http://localhost:8080 |
| `roomeq autotune [preset]` | `run`, and start auto-tune immediately |
| `roomeq verify [preset]` | 4-sweep check of the EQ path and speaker linearity |
| `roomeq measure` | Stand-alone measurement without the engine (plays straight to the speakers) |
| `roomeq solve [file]` | Re-solve a saved measurement, e.g. after changing the target |
| `roomeq export [preset]` | Filters as a table, Equalizer-APO text and JSON (for other EQ apps) |
| `roomeq simulate` | Measure → solve → verify on a virtual room in the terminal |
| `roomeq devices` / `presets` / `config` / `cert` | Housekeeping |

## 6. Configuration

`uv run roomeq config` writes `~/.roomeq/config.toml` (`ROOMEQ_HOME` moves the whole folder). It holds:

- `[audio]`: input (BlackHole) and output devices, sample rate, block size, limiter ceiling, soft start.
- `[measurement]`: positions, sweeps per position, sweep length/level, calibration file.
- `[target]`: bass lift and its transition frequency, treble tilt.
- `[solver]`: filter count, boost/cut limits, Q range, correction bands.
- `[server]`: HTTPS and HTTP ports.

## 7. Microphone calibration

Phone microphones are accurate enough for bass and low mids; above a few kHz they are not, which is why
RoomEQ does not correct there. If you have a calibration file (`frequency_hz, db` per line, dB = how much
the mic over-reads; UMIK/REW files work), set `measurement.calibration_file`. To make one, measure the
same spot with a calibrated USB microphone and the phone and use
`roomeq.dsp.calibration.derive_calibration`. Without a file, the phone is treated as flat and every result
says so.
