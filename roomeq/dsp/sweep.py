"""Measurement test signal: exponential sine sweep framed by two sync markers.

Layout (all at the Mac's output sample rate)::

    | pre | marker | gap | log sweep | tail | marker | post |

The two identical markers let the analyser find the recording's start *and* measure the
phone-vs-Mac clock drift from their separation, independently of the room's response
(both markers pass through the same room, so the room's delay cancels out).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .biquad import FloatArray


@dataclass(frozen=True)
class SweepConfig:
    fs: int = 48000
    f_start: float = 20.0
    f_end: float = 20000.0
    duration: float = 8.0
    level_dbfs: float = -12.0
    pre_silence: float = 0.5
    gap: float = 0.5
    tail: float = 1.5
    post_silence: float = 0.6
    marker_duration: float = 0.15
    marker_f_start: float = 300.0
    marker_f_end: float = 8000.0


@dataclass(frozen=True)
class TestSignal:
    config: SweepConfig
    signal: FloatArray          # full playback signal (mono)
    sweep: FloatArray           # sweep alone, same scaling as in ``signal``
    marker: FloatArray          # one marker, same scaling as in ``signal``
    marker1_start: int
    sweep_start: int
    marker2_start: int

    @property
    def fs(self) -> int:
        return self.config.fs

    @property
    def marker_separation(self) -> int:
        return self.marker2_start - self.marker1_start

    @property
    def duration_s(self) -> float:
        return len(self.signal) / self.config.fs


def exp_sweep(f1: float, f2: float, duration: float, fs: int,
              fade_in: float = 0.05, fade_out: float = 0.01) -> FloatArray:
    """Farina exponential sweep with half-Hann fades, unit amplitude."""
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    rate = duration / np.log(f2 / f1)
    x = np.sin(2.0 * np.pi * f1 * rate * (np.exp(t / rate) - 1.0))
    ni, no = int(fade_in * fs), int(fade_out * fs)
    if ni:
        x[:ni] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(ni) / ni)
    if no:
        x[-no:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(no) / no)
    return x


def make_marker(cfg: SweepConfig) -> FloatArray:
    n = int(round(cfg.marker_duration * cfg.fs))
    x = exp_sweep(cfg.marker_f_start, cfg.marker_f_end, cfg.marker_duration, cfg.fs, 0.0, 0.0)
    return x * np.hanning(n)


def make_test_signal(cfg: SweepConfig | None = None) -> TestSignal:
    cfg = cfg or SweepConfig()
    fs = cfg.fs
    amp = 10.0 ** (cfg.level_dbfs / 20.0)
    marker = make_marker(cfg) * amp
    sweep = exp_sweep(cfg.f_start, min(cfg.f_end, 0.45 * fs), cfg.duration, fs) * amp

    def silence(s: float) -> FloatArray:
        return np.zeros(int(round(s * fs)))

    parts = [silence(cfg.pre_silence), marker, silence(cfg.gap), sweep,
             silence(cfg.tail), marker, silence(cfg.post_silence)]
    m1 = len(parts[0])
    s0 = m1 + len(marker) + len(parts[2])
    m2 = s0 + len(sweep) + len(parts[4])
    return TestSignal(cfg, np.concatenate(parts), sweep, marker, m1, s0, m2)
