import numpy as np
import pytest

from roomeq.dsp.biquad import Filter, response_db
from roomeq.dsp.solver import SolverConfig, solve
from roomeq.dsp.spectrum import log_grid
from roomeq.dsp.target import FLAT, TargetCurve

FS = 48000
GRID = log_grid(20, 20000, 48)


def room(filters, noise=0.0, seed=0):
    db = response_db(filters, GRID, FS)
    return db + noise * np.random.default_rng(seed).standard_normal(len(GRID))


def check_limits(res, cfg=SolverConfig()):
    for f in res.filters:
        assert cfg.q_min - 1e-6 <= f.q <= cfg.q_max + 1e-6
        assert -cfg.max_cut_db - 1e-6 <= f.gain_db <= cfg.max_boost_db + 1e-6
        assert f.freq <= cfg.gentle_band_max_hz + 1e-6
        if f.freq > cfg.full_band_max_hz:
            assert f.q <= cfg.gentle_q_max + 1e-6
            assert -cfg.gentle_max_cut_db - 1e-6 <= f.gain_db <= cfg.gentle_max_boost_db + 1e-6
        if f.gain_db > 0:
            assert f.q <= cfg.boost_q_max + 1e-6


def test_recovers_known_peaks():
    true = [Filter.peak(52, 8, 5), Filter.peak(125, 6, 3), Filter.peak(310, 5, 2)]
    res = solve(GRID, room(true, noise=0.1), FLAT)
    check_limits(res)
    for t in true:
        match = [f for f in res.filters if abs(np.log2(f.freq / t.freq)) < 0.15 and f.gain_db < 0]
        assert match, f"no cut found near {t.freq} Hz: {res.filters}"
        best = min(match, key=lambda f: abs(f.gain_db + t.gain_db))
        assert best.gain_db == pytest.approx(-t.gain_db, abs=1.5)
    assert res.rms_after < 0.6
    assert res.rms_before > 2.0


def test_never_fills_deep_narrow_null():
    true = [Filter.peak(60, 6, 4), Filter.peak(95, -18, 10)]
    res = solve(GRID, room(true), FLAT)
    check_limits(res)
    near_null = [f for f in res.filters if 75 < f.freq < 120 and f.gain_db > 0.5]
    assert not near_null, res.filters
    assert any("null" in w for w in res.warnings)


def test_boost_and_cut_limits():
    true = [Filter.peak(80, 15, 4), Filter.peak(200, -6, 0.8), Filter.peak(1500, 6, 1)]
    res = solve(GRID, room(true), FLAT)
    check_limits(res)
    assert res.eq_db.max() <= 3.0 + 0.3


def test_no_correction_above_4k():
    true = [Filter.peak(6000, 6, 2), Filter.peak(12000, -8, 2)]
    res = solve(GRID, room(true), FLAT)
    assert all(f.freq <= 4000 for f in res.filters)
    hi = GRID > 6000
    assert np.max(np.abs(res.eq_db[hi])) < 1.5


def test_prefers_cuts():
    # symmetric +6 peak and -6 dip of equal width: the peak must get more correction
    true = [Filter.peak(70, 6, 2), Filter.peak(250, -6, 2)]
    res = solve(GRID, room(true), FLAT)
    at = lambda f0: res.eq_db[np.argmin(np.abs(GRID - f0))]
    assert -at(70) > at(250)


def test_rolloff_not_boosted():
    import scipy.signal as sps
    b, a = sps.butter(4, 45, "highpass", fs=FS)
    _, h = sps.freqz(b, a, worN=GRID, fs=FS)
    meas = 20 * np.log10(np.abs(h)) + room([Filter.peak(90, 5, 3)])
    res = solve(GRID, meas, TargetCurve())
    assert res.f_lo > 35
    assert all(f.gain_db <= 0 or f.freq > res.f_lo * 1.4 for f in res.filters)
    assert res.eq_db[GRID < res.f_lo].max() < 1.0


def test_target_alignment_and_preamp():
    res = solve(GRID, room([Filter.peak(100, 6, 3)]) + 17.0, TargetCurve())
    assert res.offset_db == pytest.approx(17.0, abs=1.0)
    if res.eq_db.max() > 0:
        assert res.preamp_db < -res.eq_db.max() + 0.1


def test_warm_start_keeps_good_solution():
    true = [Filter.peak(52, 8, 5), Filter.peak(125, 6, 3)]
    meas = room(true)
    first = solve(GRID, meas, FLAT)
    second = solve(GRID, meas, FLAT, initial=first.filters)
    assert second.rms_after <= first.rms_after + 0.05


def test_warns_when_bass_is_too_loud_for_eq():
    # a sub turned up ~18 dB: the EQ's cut limits can't bring it down, so say "turn the knob down"
    import scipy.signal as sps
    b, a = sps.butter(4, 40, "highpass", fs=FS)
    _, h = sps.freqz(b, a, worN=GRID, fs=FS)
    sub = 26.0 / (1 + (GRID / 130.0) ** 6)
    res = solve(GRID, 20 * np.log10(np.abs(h)) + sub, TargetCurve())
    assert any("turn the subwoofer / bass level down" in w for w in res.warnings), res.warnings
    assert all(f.gain_db <= 0 or f.freq >= 35 for f in res.filters)          # no deep-bass boosts
    ok = solve(GRID, room([Filter.peak(70, 6, 4)]), TargetCurve())
    assert not any("turn the subwoofer" in w for w in ok.warnings)
