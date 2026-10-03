"""Synthetic room + speaker + phone, standing in for real hardware.

The default model loosely resembles a small 2.1 system (F&D A521X class: 5.25" sub, small
satellites) in a living room: subwoofer roll-off near 40 Hz, a hump from the bass knob, a
crossover dip, strong room modes, one deep narrow null, early reflections and a reverb tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from ..dsp.analysis import resample_nominal, resample_ratio
from ..dsp.biquad import Filter, FloatArray, apply_filters, filters_to_sos


@dataclass(frozen=True)
class RoomModel:
    hp_hz: float = 40.0                     # subwoofer low-frequency roll-off (4th order)
    lp_hz: float = 18000.0
    speaker: tuple[Filter, ...] = (
        Filter.peak(85, 3.0, 1.2),          # bass knob hump
        Filter.peak(170, -3.0, 2.0),        # sub/satellite crossover hole
        Filter.peak(2800, 2.0, 1.0),
        Filter.high_shelf(9000, -2.0),
    )
    modes: tuple[Filter, ...] = (
        Filter.peak(46, 7.0, 6.0),
        Filter.peak(68, 8.0, 5.0),
        Filter.peak(118, 6.0, 4.0),
        Filter.peak(240, 4.0, 3.0),
    )
    null: Filter | None = Filter.peak(97, -18.0, 10.0)
    reflections: tuple[tuple[float, float], ...] = ((2.3e-3, -6.0), (5.1e-3, -8.0), (8.7e-3, -10.0), (13e-3, -12.0))
    rt60_s: float = 0.35
    reverb_db: float = -14.0
    ir_length_s: float = 0.8
    position_variation: float = 1.0         # 0 = every position identical


@dataclass(frozen=True)
class PhoneMicModel:
    hp_hz: float = 22.0
    filters: tuple[Filter, ...] = (Filter.peak(6000, 3.0, 1.0), Filter.high_shelf(12000, -4.0))

    def response_db(self, freqs: FloatArray, fs: int) -> FloatArray:
        from ..dsp.biquad import response_db
        b, a = sps.butter(2, self.hp_hz, "highpass", fs=fs)
        _, h = sps.freqz(b, a, worN=freqs, fs=fs)
        return 20 * np.log10(np.abs(h)) + response_db(list(self.filters), freqs, fs)


def _vary(filters: tuple[Filter, ...], rng: np.random.Generator, amount: float) -> list[Filter]:
    out = []
    for f in filters:
        out.append(Filter(f.type, f.freq * (1 + amount * rng.uniform(-0.04, 0.04)),
                          f.gain_db + amount * rng.uniform(-2.0, 2.0) * np.sign(f.gain_db), f.q))
    return out


def room_ir(room: RoomModel, fs: int, position: int = 0) -> FloatArray:
    rng = np.random.default_rng(1000 + position)
    amt = room.position_variation if position else 0.0
    n = int(room.ir_length_s * fs)
    imp = np.zeros(n)
    imp[0] = 1.0
    h = sps.sosfilt(sps.butter(4, room.hp_hz, "highpass", fs=fs, output="sos"), imp)
    h = sps.sosfilt(sps.butter(2, room.lp_hz, "lowpass", fs=fs, output="sos"), h)
    flt = list(room.speaker) + _vary(room.modes, rng, amt)
    if room.null is not None:
        flt += _vary((room.null,), rng, amt)
    h = sps.sosfilt(filters_to_sos(flt, fs), h)
    direct = h.copy()
    lp = sps.butter(2, 6000, "lowpass", fs=fs, output="sos")
    for delay, gain_db in room.reflections:
        d = int(delay * (1 + amt * rng.uniform(-0.2, 0.2)) * fs)
        if d < n:
            h[d:] += 10 ** (gain_db / 20) * sps.sosfilt(lp, direct[: n - d])
    t = np.arange(n) / fs
    tail = rng.standard_normal(n) * np.exp(-6.91 * t / room.rt60_s)
    tail = sps.sosfilt(sps.butter(2, [150, 8000], "bandpass", fs=fs, output="sos"), tail)
    tail[: int(0.005 * fs)] = 0
    tail *= 10 ** (room.reverb_db / 20) * np.sqrt(np.sum(direct ** 2) / max(np.sum(tail ** 2), 1e-30))
    return h + tail


@dataclass
class SimConfig:
    mac_fs: int = 48000
    phone_fs: int = 48000
    drift_ppm: float = 35.0                 # phone clock relative to Mac clock
    noise_dbfs: float = -72.0               # background noise at the phone (RMS)
    acoustic_gain_db: float = -6.0          # playback dBFS -> recorded dBFS
    record_lead_s: float = 0.4              # phone starts recording this long before playback
    seed: int = 0
    room: RoomModel = field(default_factory=RoomModel)
    mic: PhoneMicModel = field(default_factory=PhoneMicModel)


class SimulatedRig:
    """Implements the MeasurementRig protocol without any hardware."""

    name = "simulation"

    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._irs: dict[int, FloatArray] = {}

    def ir(self, position: int) -> FloatArray:
        if position not in self._irs:
            self._irs[position] = room_ir(self.cfg.room, self.cfg.mac_fs, position)
        return self._irs[position]

    def prepare_position(self, position: int, total: int) -> None:  # no user to prompt
        pass

    def record(self, playback: FloatArray, position: int = 0, eq: list[Filter] | tuple = (),
               preamp_db: float = 0.0) -> tuple[FloatArray, float]:
        c = self.cfg
        y = apply_filters(playback, list(eq), c.mac_fs, preamp_db) if eq else np.asarray(playback, float)
        y = sps.fftconvolve(y, self.ir(position))
        b, a = sps.butter(2, c.mic.hp_hz, "highpass", fs=c.mac_fs)
        y = sps.lfilter(b, a, y)
        y = apply_filters(y, list(c.mic.filters), c.mac_fs)
        y *= 10 ** (c.acoustic_gain_db / 20)
        y = np.concatenate([np.zeros(int(c.record_lead_s * c.mac_fs)), y, np.zeros(int(0.3 * c.mac_fs))])
        y = resample_nominal(y, c.mac_fs, c.phone_fs)
        if c.drift_ppm:
            y = resample_ratio(y, 1.0 / (1.0 + c.drift_ppm * 1e-6))
        noise = self._rng.standard_normal(len(y))
        noise = sps.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                            [1, -2.494956002, 2.017265875, -0.522189400], noise)  # pink-ish
        noise *= 10 ** (c.noise_dbfs / 20) / max(np.std(noise), 1e-12)
        return np.clip(y + noise, -1.0, 1.0).astype(np.float32), float(c.phone_fs)
