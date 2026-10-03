"""Automatic parametric-EQ fitting.

Greedy placement plus joint refinement:

1. Align the target's level to the measurement (median over ``align_band``).
2. Find where to correct: from the speaker's low-frequency roll-off (auto-detected) up to
   ``gentle_band_max_hz``. Full-strength correction below ``full_band_max_hz``, only gentle,
   wide corrections above, nothing above ``gentle_band_max_hz``.
3. Mark deep *narrow* dips as room nulls. They are never boosted.
4. Repeatedly put a peaking filter on the largest weighted deviation, sized from the width of that
   deviation, then refine all filters together with bounded least squares. Cuts are preferred:
   deviations below the target count half as much as those above it.
5. Stop when the error stops improving, the remaining deviation is within tolerance, or the
   filter budget is spent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from .biquad import Filter, FilterType, FloatArray, auto_preamp_db, peaking_sos, q_from_bandwidth, response_db, sos_response_db
from .spectrum import band_mask
from .target import TargetCurve


@dataclass(frozen=True)
class SolverConfig:
    max_filters: int = 10
    max_boost_db: float = 3.0
    max_cut_db: float = 9.0
    q_min: float = 0.7
    q_max: float = 8.0
    boost_q_max: float = 2.5             # boosts must be broad
    f_min: float = 25.0                  # never correct below this
    full_band_max_hz: float = 500.0
    gentle_band_max_hz: float = 4000.0
    gentle_max_boost_db: float = 2.0
    gentle_max_cut_db: float = 4.0
    gentle_q_max: float = 2.0
    gentle_weight: float = 0.5
    boost_weight: float = 0.5            # dips count half as much as peaks
    null_depth_db: float = 6.0
    null_max_width_oct: float = 0.5
    detect_rolloff: bool = True
    boost_guard_oct: float = 0.5         # no boosts within this many octaves above the roll-off
    min_boost_hz: float = 35.0           # never boost below this, whatever roll-off detection says
    rolloff_threshold_db: float = 6.0
    align_band: tuple[float, float] = (200.0, 2000.0)
    tolerance_db: float = 0.75
    min_improvement_db: float = 0.02
    preamp_margin_db: float = 0.5
    max_total_cut_db: float = 12.0       # stacked cuts never go deeper than this
    spread_ref_db: float = 4.0           # position spread (std) at which a frequency counts half
    fs: float = 48000.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SolverConfig":
        d = dict(d)
        if "align_band" in d:
            d["align_band"] = tuple(d["align_band"])
        return SolverConfig(**d)


@dataclass
class SolveResult:
    filters: list[Filter]
    preamp_db: float
    freqs: FloatArray
    measured_db: FloatArray
    target_db: FloatArray          # level-aligned target
    eq_db: FloatArray              # response of filters (+ fixed filters), excluding preamp
    predicted_db: FloatArray       # measured + eq
    offset_db: float
    f_lo: float
    null_mask: FloatArray
    rms_before: float
    rms_after: float
    rms_bass_before: float
    rms_bass_after: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Band:
    lo: float
    hi: float
    max_boost: float
    max_cut: float
    q_max: float


class _Problem:
    def __init__(self, freqs: FloatArray, dev: FloatArray, cfg: SolverConfig, f_lo: float, null: FloatArray,
                 spread: FloatArray | None = None):
        self.f = freqs
        self.dev = dev
        self.cfg = cfg
        self.f_lo = f_lo
        self.null = null
        full = band_mask(freqs, f_lo, cfg.full_band_max_hz)
        gentle = (freqs > cfg.full_band_max_hz) & (freqs <= cfg.gentle_band_max_hz) & (freqs >= f_lo)
        self.w = np.where(full, 1.0, np.where(gentle, cfg.gentle_weight, 0.0))
        if spread is not None:
            # where listening positions disagree, the average is not a reliable thing to EQ towards
            self.w = self.w / (1.0 + (np.asarray(spread) / cfg.spread_ref_db) ** 2)
        self.active = self.w > 0
        self.above = freqs > cfg.gentle_band_max_hz
        self.below = freqs < f_lo
        self.boost_floor = max(f_lo * 2.0 ** cfg.boost_guard_oct if f_lo > cfg.f_min + 1 else f_lo,
                               cfg.min_boost_hz)
        self.no_boost = freqs < self.boost_floor

    def band_for(self, f0: float) -> _Band:
        c = self.cfg
        if f0 <= c.full_band_max_hz:
            return _Band(self.f_lo, c.full_band_max_hz, c.max_boost_db, c.max_cut_db, c.q_max)
        return _Band(c.full_band_max_hz * 0.8, c.gentle_band_max_hz,
                     min(c.gentle_max_boost_db, c.max_boost_db), min(c.gentle_max_cut_db, c.max_cut_db),
                     min(c.gentle_q_max, c.q_max))

    def eq_db(self, p: FloatArray) -> FloatArray:
        if p.size == 0:
            return np.zeros_like(self.f)
        p = p.reshape(-1, 3)
        sos = peaking_sos(2.0 ** p[:, 0], p[:, 1], np.exp(p[:, 2]), self.cfg.fs)
        return sos_response_db(sos, self.f, self.cfg.fs)

    def weighted_error(self, eq: FloatArray) -> FloatArray:
        err = self.dev + eq
        r = self.w * np.where(err > 0, err, self.cfg.boost_weight * err)
        r[self.null & (err < 0)] = 0.0
        return r

    def residual(self, p: FloatArray) -> FloatArray:
        eq = self.eq_db(p)
        c = self.cfg
        extra = np.concatenate([
            0.3 * eq[self.above],                                  # leave the top end alone
            1.0 * np.maximum(eq[self.no_boost], 0.0),              # don't boost near/below roll-off
            2.0 * np.maximum(eq[self.null], 0.0),                  # never push energy into a null
            3.0 * np.maximum(eq - c.max_boost_db, 0.0),            # stacked boosts still capped
            3.0 * np.maximum(-eq - c.max_total_cut_db, 0.0),       # ... and stacked cuts
        ])
        return np.concatenate([self.weighted_error(eq), extra])

    def score(self, p: FloatArray) -> float:
        return float(np.sqrt(np.mean(self.weighted_error(self.eq_db(p))[self.active] ** 2)))


def _contiguous(mask: FloatArray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _detect_rolloff(freqs: FloatArray, dev: FloatArray, cfg: SolverConfig) -> float:
    """Lowest frequency the speaker meaningfully reproduces (relative to the aligned target)."""
    if not cfg.detect_rolloff:
        return cfg.f_min
    region = (freqs >= cfg.f_min) & (freqs <= 200.0)
    idx = np.flatnonzero(region)
    for i in idx:
        if dev[i] > -cfg.rolloff_threshold_db:
            return float(max(freqs[i], cfg.f_min))
    return cfg.f_min


def _find_nulls(freqs: FloatArray, dev: FloatArray, active: FloatArray, cfg: SolverConfig) -> FloatArray:
    null = np.zeros_like(freqs, dtype=bool)
    deep = (dev < -cfg.null_depth_db) & active
    for a, b in _contiguous(deep):
        width = np.log2(freqs[b] / freqs[a]) if b > a else 0.0
        if width <= cfg.null_max_width_oct:
            # include the skirts down to a third of the threshold
            lo, hi = a, b
            while lo > 0 and dev[lo - 1] < -cfg.null_depth_db / 3:
                lo -= 1
            while hi < len(dev) - 1 and dev[hi + 1] < -cfg.null_depth_db / 3:
                hi += 1
            null[lo:hi + 1] = True
    return null


def correction_masks(freqs: FloatArray, measured_db: FloatArray, target: TargetCurve, cfg: SolverConfig
                     ) -> tuple[float, FloatArray, float]:
    """(f_lo, null_mask, offset_db) exactly as :func:`solve` derives them, for scoring other curves."""
    f = np.asarray(freqs, dtype=np.float64)
    tgt = target.evaluate(f)
    al = band_mask(f, *cfg.align_band)
    offset = float(np.median(measured_db[al] - tgt[al]))
    dev = measured_db - tgt - offset
    f_lo = _detect_rolloff(f, dev, cfg)
    return f_lo, _find_nulls(f, dev, band_mask(f, f_lo, cfg.gentle_band_max_hz), cfg), offset


def bass_excess_db(freqs: FloatArray, response_db: FloatArray, aligned_target_db: FloatArray, f_lo: float,
                   null_mask: FloatArray | None = None, lo: float = 40.0, hi: float = 120.0) -> float:
    """Median of (response - target) over the bass band: how much too loud the bass is overall."""
    m = band_mask(freqs, max(lo, f_lo), hi)
    if null_mask is not None:
        m &= ~null_mask
    return float(np.median(response_db[m] - aligned_target_db[m])) if m.any() else 0.0


def error_metrics(freqs: FloatArray, measured_db: FloatArray, target: TargetCurve, cfg: SolverConfig,
                  f_lo: float | None = None, null_mask: FloatArray | None = None,
                  offset_db: float | None = None) -> tuple[float, float]:
    """(rms over correction band, rms over bass band) of measured - aligned target, nulls excluded."""
    tgt = target.evaluate(freqs)
    if offset_db is None:
        al = band_mask(freqs, *cfg.align_band)
        offset_db = float(np.median(measured_db[al] - tgt[al]))
    err = measured_db - tgt - offset_db
    f_lo = cfg.f_min if f_lo is None else f_lo
    keep = ~null_mask if null_mask is not None else np.ones_like(freqs, dtype=bool)
    m_all = band_mask(freqs, f_lo, cfg.gentle_band_max_hz) & keep
    m_bass = band_mask(freqs, f_lo, cfg.full_band_max_hz) & keep
    rms = lambda m: float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")
    return rms(m_all), rms(m_bass)


def solve(freqs: FloatArray, measured_db: FloatArray, target: TargetCurve | None = None,
          cfg: SolverConfig | None = None, fixed_filters: Sequence[Filter] = (),
          initial: Sequence[Filter] = (), spread_db: FloatArray | None = None) -> SolveResult:
    """Fit peaking filters so that ``measured + EQ`` follows ``target``.

    ``measured_db`` must be the *unequalised* response (calibration already applied).
    ``fixed_filters`` (e.g. user tone shelves) are applied but not optimised.
    ``initial`` warm-starts the fit (used by the auto loop).
    ``spread_db`` (std across listening positions, per frequency) down-weights position-dependent
    features so the EQ is not tuned to peaks that exist only at some spots.
    """
    cfg = cfg or SolverConfig()
    target = target or TargetCurve()
    f = np.asarray(freqs, dtype=np.float64)
    meas = np.asarray(measured_db, dtype=np.float64)
    tgt = target.evaluate(f)
    fixed_db = response_db(list(fixed_filters), f, cfg.fs) if fixed_filters else np.zeros_like(f)

    al = band_mask(f, *cfg.align_band)
    offset = float(np.median(meas[al] + fixed_db[al] - tgt[al]))
    dev = meas + fixed_db - tgt - offset

    f_lo = _detect_rolloff(f, dev, cfg)
    active0 = band_mask(f, f_lo, cfg.gentle_band_max_hz)
    null = _find_nulls(f, dev, active0, cfg)
    prob = _Problem(f, dev, cfg, f_lo, null, spread_db)

    warnings: list[str] = []
    if f_lo > cfg.f_min + 1:
        warnings.append(f"The speakers roll off below about {f_lo:.0f} Hz; nothing below that is corrected.")
    for a, b in _contiguous(null):
        warnings.append(f"Deep narrow dip around {np.sqrt(f[a] * f[b]):.0f} Hz left alone: it is a room "
                        "null (cancellation). EQ cannot fill it; moving the seat or the subwoofer can.")

    params: list[list[float]] = []
    bounds: list[tuple[list[float], list[float]]] = []

    def add(f0: float, gain: float, q: float) -> None:
        band = prob.band_for(f0)
        span = 2.0 ** (1.0 / 3.0)
        flo, fhi = max(band.lo, f0 / span), min(band.hi, f0 * span)
        if fhi <= flo:
            flo, fhi = f0 / 1.01, f0 * 1.01
        if gain < 0:
            glo, ghi = -band.max_cut, 0.0
            qhi = band.q_max
        else:
            glo, ghi = 0.0, band.max_boost
            qhi = min(band.q_max, cfg.boost_q_max)
            flo = max(flo, prob.boost_floor)
            fhi = max(fhi, flo * 1.01)
        lo = [np.log2(flo), glo, np.log(cfg.q_min)]
        hi = [np.log2(fhi), max(ghi, glo + 1e-6), np.log(max(qhi, cfg.q_min * 1.0001))]
        x = [np.clip(np.log2(f0), lo[0], hi[0]), np.clip(gain, lo[1], hi[1]), np.clip(np.log(q), lo[2], hi[2])]
        params.append(x)
        bounds.append((lo, hi))

    def refine() -> None:
        if not params:
            return
        x0 = np.array(params).ravel()
        lo = np.concatenate([b[0] for b in bounds])
        hi = np.concatenate([b[1] for b in bounds])
        x0 = np.clip(x0, lo, hi)
        sol = least_squares(prob.residual, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=400)
        params[:] = sol.x.reshape(-1, 3).tolist()

    for flt in initial:
        if flt.type is FilterType.PEAK and abs(flt.gain_db) > 0.05:
            add(flt.freq, flt.gain_db, flt.q)
    refine()

    exhausted = np.zeros_like(f, dtype=bool)
    attempts = 0
    while len(params) < cfg.max_filters and attempts < 3 * cfg.max_filters:
        attempts += 1
        p_now = np.array(params).ravel()
        eq = prob.eq_db(p_now)
        err = dev + eq
        r = prob.weighted_error(eq)
        r[~prob.active | exhausted] = 0.0
        idx = int(np.argmax(np.abs(r)))
        if abs(r[idx]) < cfg.tolerance_db:
            break
        e = err[idx]
        sign = np.sign(e)
        if e < 0 and f[idx] < prob.boost_floor:
            exhausted[idx] = True
            continue
        near = [p for p in params if abs(p[0] - np.log2(f[idx])) <= 1.0 / 6.0 and np.sign(p[1]) == sign * -1]
        if near:
            # an existing filter already covers this spot; let refinement handle it
            exhausted |= np.abs(np.log2(f / f[idx])) <= 1.0 / 12.0
            continue
        lo_i = hi_i = idx
        while lo_i > 0 and np.sign(err[lo_i - 1]) == sign and abs(err[lo_i - 1]) >= abs(e) / 2:
            lo_i -= 1
        while hi_i < len(f) - 1 and np.sign(err[hi_i + 1]) == sign and abs(err[hi_i + 1]) >= abs(e) / 2:
            hi_i += 1
        bw = max(np.log2(f[hi_i] / f[lo_i]), 1.0 / 24.0)
        q0 = q_from_bandwidth(bw)
        band = prob.band_for(f[idx])
        if e > 0:
            gain0 = -min(e, band.max_cut)
            q0 = float(np.clip(q0, cfg.q_min, band.q_max))
        else:
            gain0 = min(-e, band.max_boost)
            q0 = float(np.clip(q0, cfg.q_min, min(band.q_max, cfg.boost_q_max)))

        before = prob.score(p_now)
        saved = ([list(p) for p in params], list(bounds))
        add(float(f[idx]), gain0, q0)
        refine()
        after = prob.score(np.array(params).ravel())
        if before - after < cfg.min_improvement_db:
            params[:], bounds[:] = saved
            exhausted |= np.abs(np.log2(f / f[idx])) <= 1.0 / 6.0

    filters = [Filter.peak(2.0 ** p[0], p[1], np.exp(p[2])) for p in params if abs(p[1]) >= 0.3]
    filters.sort(key=lambda x: x.freq)
    all_filters = list(fixed_filters) + filters
    eq_db = response_db(all_filters, f, cfg.fs) if all_filters else np.zeros_like(f)
    predicted = meas + eq_db

    rb, rbb = error_metrics(f, meas + fixed_db, target, cfg, f_lo, null, offset)
    ra, rba = error_metrics(f, predicted, target, cfg, f_lo, null, offset)
    excess = bass_excess_db(f, predicted, tgt + offset, f_lo, null)
    if excess > 4.0:
        warnings.append(f"Even with the EQ, the bass (40-120 Hz) stays about {excess:.0f} dB above the target: "
                        f"turn the subwoofer / bass level down by roughly {excess:.0f} dB on the speaker itself, "
                        "then measure again. That beats any EQ (less strain on the amp and driver).")
    preamp = auto_preamp_db(all_filters, cfg.fs, cfg.preamp_margin_db)
    return SolveResult(filters, preamp, f, meas, tgt + offset, eq_db, predicted, offset, f_lo, null,
                       rb, ra, rbb, rba, warnings)
