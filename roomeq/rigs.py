"""Real measurement rigs: play through the Mac's output, record with the phone.

``PhoneRig`` works with any *source* that exposes a growing sample counter, e.g. the web
recorder (``PhoneLink``) or a local CoreAudio input (``LocalRecorder``, e.g. the iPhone via
Continuity). The EQ for "measure through EQ" is applied offline to the test signal, which is
equivalent for a linear EQ and keeps measurement independent of the real-time engine.
"""

from __future__ import annotations

import select
import sys
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from scipy import signal as sps

from .dsp.biquad import Filter, FloatArray, apply_filters
from .server.link import PhoneError

Log = Callable[[str], None]
Player = Callable[[FloatArray, int], None]

POSITIONS = [
    ("Main seat", "Put the phone where your head is when you sit, at ear height, screen up and bottom edge "
                  "towards the speakers. Rest it on something (a stack of books or a chair back) and don't hold it."),
    ("Left of the seat", "Move the phone about 40 cm to the left of position 1, at the same height."),
    ("Right of the seat", "Move the phone about 40 cm to the right of position 1, at the same height."),
    ("In front of the seat", "Move the phone about 30 cm forward from position 1, at the same height."),
    ("Above the seat", "Put the phone back at position 1, but about 20 cm higher."),
]


class Source(Protocol):
    @property
    def written(self) -> int: ...
    @property
    def sample_rate(self) -> float: ...
    def wait_for(self, index: int, timeout: float) -> None: ...
    def read(self, start: int, stop: int) -> FloatArray: ...


class LocalSource:
    """Adapter giving a ``LocalRecorder`` the ``Source`` interface."""

    def __init__(self, rec) -> None:
        self.rec = rec

    @property
    def written(self) -> int:
        return self.rec.written

    @property
    def sample_rate(self) -> float:
        return float(self.rec.fs)

    def wait_for(self, index: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self.rec.written < index:
            if time.monotonic() > deadline:
                raise PhoneError("local input stopped delivering audio")
            time.sleep(0.05)

    def read(self, start: int, stop: int) -> FloatArray:
        return self.rec.read(start, stop)


def _stream_token(src: Source) -> int:
    info = getattr(src, "info", None)
    return getattr(info, "stream_id", 0) if info is not None else 0


def terminal_or_phone_ready(link=None) -> Callable[[str], None]:
    """Wait for Enter in the terminal, or the phone's "Ready" button (whichever comes first)."""

    def wait(prompt: str) -> None:
        print(prompt, flush=True)
        if link is not None:
            link.pop_events("ready")
        tty = sys.stdin.isatty()
        while True:
            if link is not None and link.pop_events("ready"):
                return
            if tty:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r:
                    sys.stdin.readline()
                    return
            else:
                time.sleep(0.2)

    return wait


@dataclass
class PhoneRig:
    source: Source
    player: Player
    fs: int
    log: Log = print
    notify: Callable[[dict], None] = lambda m: None
    wait_ready: Callable[[str], None] = lambda prompt: None
    tail_s: float = 2.0
    name: str = "phone"
    skip_next_prepare: bool = False

    def prepare_position(self, position: int, total: int) -> None:
        if self.skip_next_prepare:
            self.skip_next_prepare = False
            return
        title, text = POSITIONS[position % len(POSITIONS)]
        self.notify({"type": "position", "index": position, "total": total, "title": title, "text": text})
        self.wait_ready(f"\n>>> Position {position + 1}/{total}: {title}. {text}\n"
                        "    Press Enter here, or tap Ready on the phone.")

    def record(self, playback: FloatArray, position: int = 0, eq: Sequence[Filter] = (),
               preamp_db: float = 0.0) -> tuple[FloatArray, float]:
        sig = apply_filters(playback, list(eq), self.fs, preamp_db) if eq else np.asarray(playback, float)
        for attempt in range(2):
            try:
                return self._capture(sig)
            except PhoneError as exc:
                if attempt:
                    raise
                self.log(f"    ! {exc} - retrying this sweep")
                time.sleep(1.0)
        raise AssertionError("unreachable")

    def _capture(self, sig: FloatArray) -> tuple[FloatArray, float]:
        src = self.source
        fs_in = src.sample_rate
        token = _stream_token(src)
        mark = src.written
        src.wait_for(mark + int(0.3 * fs_in), timeout=5.0)          # stream is flowing
        start = max(0, src.written - int(0.3 * fs_in))
        self.notify({"type": "status", "text": "Measuring… keep quiet and don't move the phone.", "busy": True})
        self.player(sig, self.fs)
        end = start + int((len(sig) / self.fs + self.tail_s) * fs_in)
        src.wait_for(end, timeout=10.0)
        if _stream_token(src) != token:
            raise PhoneError("the phone reconnected during the sweep")
        return src.read(start, end), fs_in


@dataclass
class LevelReport:
    background_dbfs: float
    signal_dbfs: float
    peak_dbfs: float
    ok: bool
    advice: str


def level_check(rig: PhoneRig, level_dbfs: float = -18.0, seconds: float = 2.5) -> LevelReport:
    """Play band-limited pink noise and judge the recording level and background noise."""
    fs = rig.fs
    rng = np.random.default_rng(1)
    n = int(seconds * fs)
    white = rng.standard_normal(n)
    pink = sps.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                       [1, -2.494956002, 2.017265875, -0.522189400], white)
    pink = sps.sosfilt(sps.butter(2, [40, 5000], "bandpass", fs=fs, output="sos"), pink)
    ramp = int(0.1 * fs)
    env = np.ones(n)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    pink = pink * env
    pink *= 10 ** (level_dbfs / 20) / np.sqrt(np.mean(pink ** 2))
    sig = np.concatenate([np.zeros(int(0.8 * fs)), pink, np.zeros(int(0.3 * fs))])
    rec, fs_in = rig._capture(sig)

    def rms_db(x: FloatArray) -> float:
        return 10 * np.log10(max(float(np.mean(x ** 2)), 1e-20))

    bg = rms_db(rec[: int(0.25 * fs_in)])
    win = int(0.5 * fs_in)
    frames = [rec[i:i + win] for i in range(0, len(rec) - win, win // 2)]
    loud = max(rms_db(f) for f in frames) if frames else bg
    peak = 20 * np.log10(max(float(np.max(np.abs(rec))), 1e-10))
    margin = loud - bg
    if peak > -3:
        return LevelReport(bg, loud, peak, False, "Too loud: the phone is close to clipping. Turn the volume down.")
    if loud < -50:
        return LevelReport(bg, loud, peak, False, "Very quiet: turn the volume up (moderate listening level).")
    # the sweep's processing gain adds roughly 20 dB on top of this noise-burst margin
    if margin < 15:
        return LevelReport(bg, loud, peak, False,
                           f"Only {margin:.0f} dB above the background noise. Turn up a little or make the room quieter.")
    return LevelReport(bg, loud, peak, True, f"Good: test noise is {margin:.0f} dB above the background.")
