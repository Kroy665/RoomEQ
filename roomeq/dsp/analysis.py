"""Turn a phone recording of the test signal into a smoothed magnitude response.

Pipeline: nominal resample -> locate sync markers -> estimate & remove clock drift ->
align -> regularised deconvolution -> window the impulse response -> 1/N-octave magnitude ->
quality metrics (level, clipping, per-band SNR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy import signal as sps
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

from .biquad import FloatArray
from .spectrum import band_mask, log_grid, smooth_power
from .sweep import TestSignal


class MeasurementError(RuntimeError):
    """The recording could not be analysed (markers missing, too short, ...)."""


@dataclass(frozen=True)
class AnalysisConfig:
    smoothing_fraction: float = 6.0        # 1/6 octave
    window_left_ms: float = 10.0
    window_right_ms: float = 500.0
    grid_f_min: float = 20.0
    grid_f_max: float = 20000.0
    grid_points_per_octave: int = 48
    ir_pre_s: float = 0.1                  # where the IR peak is placed after alignment
    compensate_drift: bool = True
    # quality thresholds
    clip_level: float = 0.99
    min_peak_dbfs: float = -45.0
    min_snr_bass_db: float = 20.0          # median SNR over 30-500 Hz
    min_snr_mid_db: float = 15.0           # median SNR over 500-4000 Hz
    max_drift_ppm: float = 300.0


@dataclass
class Quality:
    peak_dbfs: float
    clipped_samples: int
    drift_ppm: float
    snr_db: FloatArray                    # per grid frequency
    snr_bass_db: float
    snr_mid_db: float
    warnings: list[str] = field(default_factory=list)
    reliable: bool = True


@dataclass
class Measurement:
    freqs: FloatArray                      # log grid
    db: FloatArray                         # smoothed magnitude, dB (arbitrary reference)
    ir: FloatArray                         # windowed impulse response at ``fs``
    ir_peak: int                           # peak index inside ``ir``
    fs: int
    spectrum: FloatArray                   # complex spectrum of windowed IR (for coherence)
    spectrum_freqs: FloatArray
    quality: Quality


# --------------------------------------------------------------------------- helpers

def resample_nominal(x: FloatArray, fs_in: float, fs_out: float) -> FloatArray:
    """Polyphase resampling between nominal rates (e.g. 44100 -> 48000)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    r = Fraction(fs_out / fs_in).limit_denominator(1000)
    return sps.resample_poly(np.asarray(x, dtype=np.float64), r.numerator, r.denominator)


def resample_ratio(x: FloatArray, ratio: float) -> FloatArray:
    """Stretch ``x`` so that ``len(out) ~= len(x) / ratio`` (band-limited FFT resampling).

    Used for clock-drift correction where ``ratio`` is within a few hundred ppm of 1.
    """
    n = len(x)
    n_in = next_fast_len(n)
    padded = np.zeros(n_in)
    padded[:n] = x
    n_out = int(round(n_in / ratio))
    y = sps.resample(padded, n_out)
    return y[: int(round(n / ratio))]


def _parabolic(c: FloatArray, i: int) -> float:
    if 0 < i < len(c) - 1:
        a, b, d = c[i - 1], c[i], c[i + 1]
        den = a - 2 * b + d
        if den != 0:
            return i + 0.5 * (a - d) / den
    return float(i)


def find_template(x: FloatArray, template: FloatArray, start: int = 0, stop: int | None = None) -> tuple[float, float]:
    """Sub-sample start index of ``template`` in ``x[start:stop]`` and the normalised peak height."""
    stop = len(x) if stop is None else min(stop, len(x))
    seg = x[max(start, 0):stop]
    if len(seg) < len(template):
        raise MeasurementError("recording too short to contain the sync marker")
    c = np.abs(sps.correlate(seg, template, mode="valid", method="fft"))
    i = int(np.argmax(c))
    norm = np.sqrt(np.sum(template ** 2) * max(np.sum(seg[i:i + len(template)] ** 2), 1e-30))
    return max(start, 0) + _parabolic(c, i), float(c[i] / norm)


MAX_PLAUSIBLE_PPM = 1000.0     # real phone-vs-Mac clocks differ by tens of ppm


def _two_markers(x: FloatArray, ts: TestSignal) -> tuple[float, float]:
    """Find both markers: the pair of strong correlation peaks whose spacing matches the test signal.

    Several candidate peaks are tried, and the second marker is only searched within
    +/-MAX_PLAUSIBLE_PPM of the expected spacing, so a reflection, a knock or the sweep itself
    cannot be mistaken for a marker.
    """
    m = ts.marker
    c = np.abs(sps.correlate(x, m, mode="valid", method="fft"))
    if c.size == 0:
        raise MeasurementError("recording shorter than one marker")
    sep = ts.marker_separation
    tol = int(sep * MAX_PLAUSIBLE_PPM * 1e-6) + 8
    work = c.copy()
    candidates = []
    for _ in range(6):
        a = int(np.argmax(work))
        if work[a] <= 0:
            break
        candidates.append(a)
        work[max(a - len(m), 0): a + len(m)] = 0.0
    best: tuple[float, int, int] | None = None
    for a in candidates:
        for cand in (a + sep, a - sep):
            lo, hi = max(cand - tol, 0), min(cand + tol + 1, len(c))
            if hi <= lo:
                continue
            b = lo + int(np.argmax(c[lo:hi]))
            score = min(c[a], c[b])
            if best is None or score > best[0]:
                best = (float(score), min(a, b), max(a, b))
    # both markers must tower over the correlation's typical level: for noise the weaker of a "pair"
    # within the narrow drift window is ~3x the median, real markers are 10x to >10000x
    if best is None or best[0] < 0.25 * c.max() or best[0] < 6.0 * float(np.median(c)):
        raise MeasurementError(
            "could not find both sync markers in the recording - was the phone recording for the "
            "whole sweep, and is the volume high enough?")
    return _parabolic(c, best[1]), _parabolic(c, best[2])


def _precise_separation(x: FloatArray, m1: float, m2: float, marker_len: int, fs: int) -> float:
    """Separation of the two recorded markers by cross-correlating the recordings with each other.

    Both markers carry identical room colouring, so this is unbiased by the room. The lag is only
    searched within the plausible clock-drift range.
    """
    pad_l, pad_r = int(0.01 * fs), int(0.08 * fs)
    i1, i2 = int(round(m1)), int(round(m2))
    n = marker_len + pad_l + pad_r
    s1 = x[max(i1 - pad_l, 0): max(i1 - pad_l, 0) + n]
    s2 = x[max(i2 - pad_l, 0): max(i2 - pad_l, 0) + n]
    if len(s1) != n or len(s2) != n:
        return m2 - m1
    c = sps.correlate(s2, s1, mode="full", method="fft")
    lags = sps.correlation_lags(len(s2), len(s1), mode="full")
    max_lag = int((i2 - i1) * MAX_PLAUSIBLE_PPM * 1e-6) + 8
    ok = np.abs(lags) <= max_lag
    c = np.where(ok, c, -np.inf)
    centre = int(np.argmax(c))
    frac = _parabolic(np.where(ok, c, c[centre]), centre) - centre
    return float((i2 - i1) + lags[centre] + frac)


def estimate_drift(x: FloatArray, ts: TestSignal) -> tuple[float, float]:
    """Returns ``(ratio, marker1_position)`` where ratio = recorded / nominal marker separation."""
    m1, m2 = _two_markers(x, ts)
    sep = _precise_separation(x, m1, m2, len(ts.marker), ts.fs)
    return sep / ts.marker_separation, m1


def deconvolve(y: FloatArray, x: FloatArray, fs: int, f_lo: float, f_hi: float,
               n_fft: int | None = None) -> FloatArray:
    """Regularised inverse filtering: h = IFFT( Y X* / (|X|^2 + eps(f)) ).

    ``eps`` is tiny inside the excitation band and large outside it, with half-octave cosine
    transitions, so out-of-band noise is not amplified.
    """
    n = n_fft or next_fast_len(len(y) + len(x))
    X = rfft(x, n)
    Y = rfft(y, n)
    f = rfftfreq(n, 1.0 / fs)
    p = np.abs(X) ** 2
    pmax = float(p.max())
    lo_edge, hi_edge = f_lo / np.sqrt(2.0), min(f_hi * np.sqrt(2.0), fs / 2.0)
    lf = np.log2(np.maximum(f, 1e-9))
    t_lo = np.clip((lf - np.log2(lo_edge)) / (np.log2(f_lo) - np.log2(lo_edge)), 0, 1)
    t_hi = np.clip((np.log2(hi_edge) - lf) / max(np.log2(hi_edge) - np.log2(f_hi), 1e-9), 0, 1)
    # w = 1 outside the excitation band, 0 inside
    w = 1.0 - (0.5 - 0.5 * np.cos(np.pi * t_lo)) * (0.5 - 0.5 * np.cos(np.pi * t_hi))
    eps = pmax * (1e-7 + w)
    H = Y * np.conj(X) / (p + eps)
    return irfft(H, n)


def window_ir(h: FloatArray, peak: int, fs: int, left_ms: float, right_ms: float) -> tuple[FloatArray, int]:
    """Cut [peak-left, peak+right] with a half-Hann fade-in and a Tukey-style fade-out (last 50%)."""
    nl, nr = int(left_ms * 1e-3 * fs), int(right_ms * 1e-3 * fs)
    start = max(peak - nl, 0)
    seg = h[start: peak + nr].copy()
    nl_eff = peak - start
    if nl_eff > 0:
        seg[:nl_eff] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(nl_eff) / nl_eff)
    nfade = (len(seg) - nl_eff) // 2
    if nfade > 0:
        seg[-nfade:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(nfade) / nfade)
    return seg, nl_eff


def magnitude(ir: FloatArray, fs: int, grid: FloatArray, fraction: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Smoothed dB on ``grid`` plus the raw complex spectrum and its frequency axis."""
    n = next_fast_len(max(len(ir), 2 * fs))
    H = rfft(ir, n)
    f = rfftfreq(n, 1.0 / fs)
    p = smooth_power(f, np.abs(H) ** 2, grid, fraction)
    return 10.0 * np.log10(p), H, f


# --------------------------------------------------------------------------- main entry

def analyze_recording(recording: FloatArray, fs_rec: float, ts: TestSignal,
                      cfg: AnalysisConfig | None = None) -> Measurement:
    cfg = cfg or AnalysisConfig()
    fs = ts.fs
    raw = np.asarray(recording, dtype=np.float64)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    peak = float(np.max(np.abs(raw))) if raw.size else 0.0
    clipped = int(np.count_nonzero(np.abs(raw) >= cfg.clip_level))

    x = resample_nominal(raw, fs_rec, fs)
    if len(x) < len(ts.signal) * 0.9:
        raise MeasurementError(
            f"recording is {len(x) / fs:.1f}s but the test signal is {ts.duration_s:.1f}s")

    ratio, m1 = estimate_drift(x, ts)
    drift_ppm = (ratio - 1.0) * 1e6
    if cfg.compensate_drift and abs(drift_ppm) > 0.05:
        x = resample_ratio(x, ratio)
        m1_est = m1 / ratio
        m1, _ = find_template(x, ts.marker, int(m1_est) - 200, int(m1_est) + len(ts.marker) + 200)

    pre = int(cfg.ir_pre_s * fs)
    offset = int(round(m1 - ts.marker1_start)) - pre
    L = len(ts.signal) + pre
    y = np.zeros(L)
    src_lo, dst_lo = max(offset, 0), max(-offset, 0)
    n_copy = min(L - dst_lo, len(x) - src_lo)
    if n_copy <= 0:
        raise MeasurementError("alignment failed - recording does not overlap the test signal")
    y[dst_lo:dst_lo + n_copy] = x[src_lo:src_lo + n_copy]

    f_hi = min(ts.config.f_end, 0.45 * fs)
    n_fft = next_fast_len(L + 3 * fs)
    h = deconvolve(y, ts.signal, fs, ts.config.f_start, f_hi, n_fft)

    search = h[: pre + int(0.2 * fs)]
    ir_peak = int(np.argmax(np.abs(search)))
    ir, peak_in_ir = window_ir(h, ir_peak, fs, cfg.window_left_ms, cfg.window_right_ms)

    grid = log_grid(cfg.grid_f_min, min(cfg.grid_f_max, f_hi), cfg.grid_points_per_octave)
    db, H, hf = magnitude(ir, fs, grid, cfg.smoothing_fraction)

    snr = _snr_per_band(h, ir_peak, ir, fs, grid, cfg, n_fft)
    q = _quality(peak, clipped, drift_ppm, snr, grid, cfg)
    return Measurement(grid, db, ir, peak_in_ir, fs, H, hf, q)


def _snr_per_band(h: FloatArray, peak: int, ir: FloatArray, fs: int, grid: FloatArray,
                  cfg: AnalysisConfig, n_fft: int) -> FloatArray:
    """Compare the windowed IR's spectrum with equal-length slices of the deconvolved noise floor.

    The noise slices come from after the room's decay and before the wrapped-around harmonic
    distortion products of the exponential sweep, so they contain only background noise.
    """
    nl = int(cfg.window_left_ms * 1e-3 * fs)
    nr = int(cfg.window_right_ms * 1e-3 * fs)
    win, _ = window_ir(np.ones(nl + nr), nl, fs, cfg.window_left_ms, cfg.window_right_ms)
    wlen = len(win)
    start = peak + int(1.0 * fs)
    stop = n_fft - int(3.0 * fs)
    slices = []
    pos = start
    while pos + wlen <= stop and len(slices) < 12:
        slices.append(h[pos:pos + wlen] * win)
        pos += wlen
    if not slices:
        return np.full(grid.shape, np.inf)
    n = next_fast_len(max(wlen, 2 * fs))
    f = rfftfreq(n, 1.0 / fs)
    noise_p = np.mean([np.abs(rfft(s, n)) ** 2 for s in slices], axis=0)
    sig_p = np.abs(rfft(ir, n)) ** 2
    s = smooth_power(f, sig_p, grid, cfg.smoothing_fraction)
    nn = smooth_power(f, noise_p, grid, cfg.smoothing_fraction)
    return 10.0 * np.log10(s / nn)


def _quality(peak: float, clipped: int, drift_ppm: float, snr: FloatArray, grid: FloatArray,
             cfg: AnalysisConfig) -> Quality:
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-12))
    bass = band_mask(grid, 30.0, 500.0)
    mid = band_mask(grid, 500.0, 4000.0)
    snr_bass = float(np.median(snr[bass])) if bass.any() else float("nan")
    snr_mid = float(np.median(snr[mid])) if mid.any() else float("nan")
    warnings: list[str] = []
    reliable = True
    if clipped:
        warnings.append(f"Recording clipped ({clipped} samples). Lower the volume and measure again.")
        reliable = False
    if peak_dbfs < cfg.min_peak_dbfs:
        warnings.append(f"Recording is very quiet (peak {peak_dbfs:.0f} dBFS). Turn the volume up.")
    if snr_bass < cfg.min_snr_bass_db:
        warnings.append(f"Low signal-to-noise in the bass ({snr_bass:.0f} dB). "
                        "Raise the volume or reduce background noise.")
        reliable = reliable and snr_bass >= cfg.min_snr_bass_db - 10
    if snr_mid < cfg.min_snr_mid_db:
        warnings.append(f"Low signal-to-noise in the mids ({snr_mid:.0f} dB).")
    if abs(drift_ppm) > cfg.max_drift_ppm:
        warnings.append(f"Unusually large clock drift ({drift_ppm:.0f} ppm); result may be wrong.")
        reliable = False
    return Quality(peak_dbfs, clipped, drift_ppm, snr, snr_bass, snr_mid, warnings, reliable)


# --------------------------------------------------------------------------- multi-sweep checks

@dataclass
class Repeatability:
    coherence: FloatArray        # per grid frequency, 1 = identical sweeps
    spread_db: FloatArray        # max - min of the repeated dB responses
    coherence_bass: float
    spread_bass_db: float
    warnings: list[str]
    reliable: bool


def repeatability(measurements: Sequence[Measurement], grid: FloatArray | None = None,
                  fraction: float = 6.0, min_coherence: float = 0.9,
                  max_spread_db: float = 2.0) -> Repeatability:
    """Coherence-style consistency of repeated sweeps taken at the *same* position.

    gamma^2(f) = |mean H_i|^2 / mean |H_i|^2, both terms fractional-octave smoothed.
    Background noise, movement or a changing room all reduce it.
    """
    if len(measurements) < 2:
        raise ValueError("need at least two sweeps")
    grid = measurements[0].freqs if grid is None else grid
    f = measurements[0].spectrum_freqs
    n = min(len(m.spectrum) for m in measurements)
    specs = np.array([m.spectrum[:n] for m in measurements])
    num = np.abs(specs.mean(axis=0)) ** 2
    den = (np.abs(specs) ** 2).mean(axis=0)
    coh = smooth_power(f[:n], num, grid, fraction) / smooth_power(f[:n], den, grid, fraction)
    coh = np.clip(coh, 0.0, 1.0)
    dbs = np.array([np.interp(grid, m.freqs, m.db) for m in measurements])
    spread = dbs.max(axis=0) - dbs.min(axis=0)
    bass = band_mask(grid, 30.0, 500.0)
    # Judge only where there is real signal: deep dips and the region below the speaker's roll-off
    # have little energy, so a tiny amount of noise swings their dB value without meaning anything.
    level = dbs.mean(axis=0)
    valid = bass & (level > np.median(level[bass]) - 10.0)
    valid = valid if valid.any() else bass
    cb = float(np.mean(coh[valid]))
    sb = float(np.percentile(spread[valid], 95))
    warnings: list[str] = []
    if cb < min_coherence:
        warnings.append(f"Repeated sweeps disagree (bass coherence {cb:.2f}). "
                        "Keep the phone still and the room quiet, then measure again.")
    if sb > max_spread_db:
        warnings.append(f"Repeated sweeps differ by up to {sb:.1f} dB in the bass.")
    return Repeatability(coh, spread, cb, sb, warnings, not warnings)
