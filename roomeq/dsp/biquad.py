"""RBJ Audio-EQ-Cookbook biquads, frequency responses and auto preamp.

Pure NumPy, no I/O. Every function maps directly onto a Swift/vDSP equivalent:
coefficients are plain arrays in SOS layout ``[b0, b1, b2, 1, a1, a2]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

IDENTITY_SOS: FloatArray = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])


class FilterType(str, Enum):
    PEAK = "peak"
    LOW_SHELF = "lowshelf"
    HIGH_SHELF = "highshelf"


@dataclass(frozen=True)
class Filter:
    """One EQ band. ``q`` is the cookbook Q (for shelves: the shelf Q, 0.707 = Butterworth-like)."""

    type: FilterType
    freq: float
    gain_db: float
    q: float

    @staticmethod
    def peak(freq: float, gain_db: float, q: float) -> "Filter":
        return Filter(FilterType.PEAK, float(freq), float(gain_db), float(q))

    @staticmethod
    def low_shelf(freq: float, gain_db: float, q: float = 0.707) -> "Filter":
        return Filter(FilterType.LOW_SHELF, float(freq), float(gain_db), float(q))

    @staticmethod
    def high_shelf(freq: float, gain_db: float, q: float = 0.707) -> "Filter":
        return Filter(FilterType.HIGH_SHELF, float(freq), float(gain_db), float(q))

    def to_dict(self) -> dict[str, float | str]:
        return {"type": self.type.value, "freq": self.freq, "gain_db": self.gain_db, "q": self.q}

    @staticmethod
    def from_dict(d: dict) -> "Filter":
        return Filter(FilterType(d.get("type", "peak")), float(d["freq"]), float(d["gain_db"]), float(d["q"]))


def peaking_sos(freqs: FloatArray | float, gains_db: FloatArray | float, qs: FloatArray | float,
                fs: float) -> FloatArray:
    """Vectorised RBJ peaking EQ. Returns an (n, 6) SOS array."""
    f0 = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    g = np.atleast_1d(np.asarray(gains_db, dtype=np.float64))
    q = np.atleast_1d(np.asarray(qs, dtype=np.float64))
    a = 10.0 ** (g / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha / a
    sos = np.empty((f0.size, 6))
    sos[:, 0] = (1.0 + alpha * a) / a0
    sos[:, 1] = (-2.0 * cw) / a0
    sos[:, 2] = (1.0 - alpha * a) / a0
    sos[:, 3] = 1.0
    sos[:, 4] = (-2.0 * cw) / a0
    sos[:, 5] = (1.0 - alpha / a) / a0
    return sos


def _shelf_sos(f: Filter, fs: float) -> FloatArray:
    a = 10.0 ** (f.gain_db / 40.0)
    w0 = 2.0 * np.pi * f.freq / fs
    cw = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * f.q)
    sq = 2.0 * np.sqrt(a) * alpha
    if f.type is FilterType.LOW_SHELF:
        b0 = a * ((a + 1) - (a - 1) * cw + sq)
        b1 = 2 * a * ((a - 1) - (a + 1) * cw)
        b2 = a * ((a + 1) - (a - 1) * cw - sq)
        a0 = (a + 1) + (a - 1) * cw + sq
        a1 = -2 * ((a - 1) + (a + 1) * cw)
        a2 = (a + 1) + (a - 1) * cw - sq
    else:
        b0 = a * ((a + 1) + (a - 1) * cw + sq)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - sq)
        a0 = (a + 1) - (a - 1) * cw + sq
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - sq
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def filter_sos(f: Filter, fs: float) -> FloatArray:
    """(1, 6) SOS row for a single filter."""
    if not 0.0 < f.freq < fs / 2.0:
        raise ValueError(f"filter frequency {f.freq} Hz outside (0, {fs / 2}) for fs={fs}")
    if f.q <= 0:
        raise ValueError(f"Q must be positive, got {f.q}")
    if f.type is FilterType.PEAK:
        return peaking_sos(f.freq, f.gain_db, f.q, fs)
    return _shelf_sos(f, fs)


def filters_to_sos(filters: Iterable[Filter], fs: float) -> FloatArray:
    """Stack filters into an (n, 6) SOS array; an empty list gives one identity section."""
    rows = [filter_sos(f, fs) for f in filters]
    return np.vstack(rows) if rows else IDENTITY_SOS.copy()


def sos_response_db(sos: FloatArray, freqs: FloatArray, fs: float) -> FloatArray:
    """Magnitude response (dB) of a cascade at arbitrary frequencies."""
    sos = np.atleast_2d(sos)
    w = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64) / fs
    z1 = np.exp(-1j * w)[None, :]
    z2 = z1 * z1
    num = sos[:, 0:1] + sos[:, 1:2] * z1 + sos[:, 2:3] * z2
    den = sos[:, 3:4] + sos[:, 4:5] * z1 + sos[:, 5:6] * z2
    mag2 = (np.abs(num) ** 2) / np.maximum(np.abs(den) ** 2, 1e-300)
    return 10.0 * np.log10(np.maximum(mag2, 1e-30)).sum(axis=0)


def response_db(filters: Sequence[Filter], freqs: FloatArray, fs: float) -> FloatArray:
    return sos_response_db(filters_to_sos(filters, fs), freqs, fs)


def dense_grid(fs: float, n: int = 2048) -> FloatArray:
    """Log-spaced grid from 10 Hz to just below Nyquist, used to find response maxima."""
    return np.geomspace(10.0, 0.49 * fs, n)


def auto_preamp_db(filters: Sequence[Filter], fs: float, margin_db: float = 0.5) -> float:
    """Preamp that cancels the largest positive excursion of the combined response, plus a margin.

    Uses the *combined* response, so overlapping boosts are accounted for. Returns 0 when the
    EQ never boosts.
    """
    if not filters:
        return 0.0
    peak = float(np.max(response_db(filters, dense_grid(fs), fs)))
    return 0.0 if peak <= 0.0 else -(peak + margin_db)


def apply_filters(x: FloatArray, filters: Sequence[Filter], fs: float, preamp_db: float = 0.0) -> FloatArray:
    """Offline filtering (simulation and tests). The real-time engine uses its own stateful kernel."""
    from scipy.signal import sosfilt

    y = sosfilt(filters_to_sos(filters, fs), x, axis=0)
    return y * 10.0 ** (preamp_db / 20.0)


def q_from_bandwidth(bw_octaves: float) -> float:
    """Cookbook relation between bandwidth in octaves and Q (for low f0 relative to fs)."""
    r = 2.0 ** bw_octaves
    return float(np.sqrt(r) / (r - 1.0))


def bandwidth_from_q(q: float) -> float:
    """Inverse of :func:`q_from_bandwidth`."""
    return float(2.0 / np.log(2.0) * np.arcsinh(1.0 / (2.0 * q)))
