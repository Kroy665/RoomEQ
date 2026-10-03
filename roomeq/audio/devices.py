"""Audio device discovery and safe playback/recording helpers (sounddevice / PortAudio)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ..dsp.biquad import FloatArray


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    inputs: int
    outputs: int
    default_samplerate: float
    is_default_in: bool
    is_default_out: bool


def list_devices() -> list[Device]:
    import sounddevice as sd

    try:
        d_in, d_out = sd.default.device
    except Exception:
        d_in = d_out = -1
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(Device(i, d["name"], d["max_input_channels"], d["max_output_channels"],
                          float(d["default_samplerate"]), i == d_in, i == d_out))
    return out


def format_devices(devs: list[Device]) -> str:
    lines = [f"{'#':>3}  {'Name':<40} {'In':>3} {'Out':>4} {'Rate':>7}"]
    for d in devs:
        mark = (" <- default output" if d.is_default_out else "") + (" <- default input" if d.is_default_in else "")
        lines.append(f"{d.index:>3}  {d.name:<40} {d.inputs:>3} {d.outputs:>4} {d.default_samplerate:>7.0f}{mark}")
    return "\n".join(lines)


def _norm(s: str) -> str:
    return s.lower().replace("’", "'").strip()


def find_device(name: str, kind: str) -> Device:
    """Resolve a (partial, case-insensitive) device name. ``kind`` is 'input' or 'output'."""
    devs = [d for d in list_devices() if (d.inputs if kind == "input" else d.outputs) > 0]
    if not name:
        default = [d for d in devs if (d.is_default_in if kind == "input" else d.is_default_out)]
        if default:
            return default[0]
        raise ValueError(f"no {kind} device configured and no system default")
    exact = [d for d in devs if _norm(d.name) == _norm(name)]
    partial = [d for d in devs if _norm(name) in _norm(d.name)]
    hits = exact or partial
    if not hits:
        names = ", ".join(f'"{d.name}"' for d in devs)
        raise ValueError(f'{kind} device "{name}" not found. Available: {names}')
    if len(hits) > 1 and not exact:
        names = ", ".join(f'"{d.name}"' for d in hits)
        raise ValueError(f'{kind} device "{name}" is ambiguous: {names}')
    return hits[0]


class Player:
    """Blocking playback of a mono signal on all (up to 2) channels of an output device."""

    def __init__(self, device: str, max_level: float = 0.9):
        self.device = find_device(device, "output")
        self.max_level = max_level

    def __call__(self, signal: FloatArray, fs: int) -> None:
        import sounddevice as sd

        x = np.asarray(signal, dtype=np.float32)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > self.max_level:                      # never send a clipping signal
            x = x * (self.max_level / peak)
        ch = max(1, min(2, self.device.outputs))
        sd.play(np.repeat(x[:, None], ch, axis=1), fs, device=self.device.index, blocking=True, latency="high")


class LocalRecorder:
    """Continuously records from a local input device (e.g. the iPhone via Continuity) into a ring buffer."""

    def __init__(self, device: str, seconds: float = 120.0):
        import sounddevice as sd

        self.device = find_device(device, "input")
        self.fs = int(self.device.default_samplerate)
        self._buf = np.zeros(int(seconds * self.fs), dtype=np.float32)
        self._written = 0
        self._lock = threading.Lock()
        self._stream = sd.InputStream(device=self.device.index, channels=1, samplerate=self.fs,
                                      dtype="float32", latency="high", callback=self._cb)

    def _cb(self, indata, frames, _time, _status) -> None:  # audio thread: no allocation beyond slicing
        n = len(self._buf)
        with self._lock:
            pos = self._written % n
            first = min(frames, n - pos)
            self._buf[pos:pos + first] = indata[:first, 0]
            if first < frames:
                self._buf[: frames - first] = indata[first:, 0]
            self._written += frames

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    @property
    def written(self) -> int:
        with self._lock:
            return self._written

    def read(self, start: int, stop: int) -> FloatArray:
        n = len(self._buf)
        with self._lock:
            start = max(start, self._written - n)
            idx = np.arange(start, stop) % n
            return self._buf[idx].astype(np.float64)
