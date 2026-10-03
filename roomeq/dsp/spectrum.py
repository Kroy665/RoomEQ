"""Frequency grids, fractional-octave smoothing and response averaging."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .biquad import FloatArray

NDArrayBool = NDArray[np.bool_]


def log_grid(f_min: float = 20.0, f_max: float = 20000.0, points_per_octave: int = 48) -> FloatArray:
    n = int(np.ceil(np.log2(f_max / f_min) * points_per_octave)) + 1
    return np.geomspace(f_min, f_max, n)


def smooth_power(lin_freqs: FloatArray, power: FloatArray, grid: FloatArray, fraction: float = 6.0) -> FloatArray:
    """Average linear-bin *power* over a 1/``fraction`` octave band centred on each grid frequency.

    Uses a cumulative integral so band edges falling between bins are handled exactly, and bands
    narrower than one bin degrade gracefully to linear interpolation.
    """
    df = np.diff(lin_freqs)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * df)))
    half = 2.0 ** (1.0 / (2.0 * fraction))
    lo = np.clip(grid / half, lin_freqs[0], lin_freqs[-1])
    hi = np.clip(grid * half, lin_freqs[0], lin_freqs[-1])
    width = hi - lo
    band = np.interp(hi, lin_freqs, cum) - np.interp(lo, lin_freqs, cum)
    point = np.interp(grid, lin_freqs, power)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(width > 0, band / np.where(width > 0, width, 1.0), point)
    return np.maximum(avg, 1e-30)


def smooth_db(freqs: FloatArray, db: FloatArray, fraction: float = 6.0) -> FloatArray:
    """Fractional-octave power smoothing of a response already sampled on a log grid."""
    p = 10.0 ** (db / 10.0)
    logf = np.log2(freqs)
    half = 1.0 / (2.0 * fraction)
    out = np.empty_like(p)
    for i, lf in enumerate(logf):
        m = np.abs(logf - lf) <= half
        out[i] = p[m].mean()
    return 10.0 * np.log10(out)


def power_average_db(responses_db: Sequence[FloatArray]) -> FloatArray:
    """Energy average of several responses on the same grid (the usual multi-position average)."""
    p = np.mean([10.0 ** (np.asarray(r) / 10.0) for r in responses_db], axis=0)
    return 10.0 * np.log10(p)


def interp_log(freqs_src: FloatArray, values: FloatArray, freqs_dst: FloatArray) -> FloatArray:
    """Interpolate on log-frequency; holds the end values outside the source range."""
    return np.interp(np.log(freqs_dst), np.log(freqs_src), values)


def band_mask(freqs: FloatArray, f_lo: float, f_hi: float) -> NDArrayBool:
    return (freqs >= f_lo) & (freqs <= f_hi)

