import numpy as np
import pytest
from scipy.signal import sosfilt

from roomeq.dsp.biquad import (Filter, auto_preamp_db, bandwidth_from_q, filter_sos, filters_to_sos,
                               q_from_bandwidth, response_db, sos_response_db)

FS = 48000


def test_peaking_reference_coefficients():
    # Hand-computed from the RBJ cookbook for f0=1 kHz, +6 dB, Q=1, fs=48 kHz
    A = 10 ** (6 / 40)
    w0 = 2 * np.pi * 1000 / FS
    alpha = np.sin(w0) / 2
    a0 = 1 + alpha / A
    expected = np.array([(1 + alpha * A) / a0, -2 * np.cos(w0) / a0, (1 - alpha * A) / a0,
                         1.0, -2 * np.cos(w0) / a0, (1 - alpha / A) / a0])
    np.testing.assert_allclose(filter_sos(Filter.peak(1000, 6, 1), FS)[0], expected, rtol=1e-12)


@pytest.mark.parametrize("f0,gain,q", [(50, -9, 8), (100, 3, 0.7), (1000, 6, 2), (8000, -4, 1.5)])
def test_peaking_gain_at_centre_and_unity_far_away(f0, gain, q):
    r = response_db([Filter.peak(f0, gain, q)], np.array([f0, 1.0, FS * 0.4999]), FS)
    assert r[0] == pytest.approx(gain, abs=1e-9)
    assert r[1] == pytest.approx(0, abs=0.05)
    assert abs(r[2]) < 0.5


def test_peaking_bandwidth_matches_q():
    q = 2.0
    bw = bandwidth_from_q(q)
    assert q_from_bandwidth(bw) == pytest.approx(q)
    f0, gain = 100.0, 8.0
    edges = np.array([f0 * 2 ** (-bw / 2), f0 * 2 ** (bw / 2)])
    r = response_db([Filter.peak(f0, gain, q)], edges, FS)
    # cookbook defines BW between the half-gain (in dB) points
    np.testing.assert_allclose(r, gain / 2, atol=0.05)


@pytest.mark.parametrize("gain", [6.0, -6.0])
def test_shelves(gain):
    f = np.array([2.0, 1000.0, 23900.0])
    lo = response_db([Filter.low_shelf(1000, gain)], f, FS)
    hi = response_db([Filter.high_shelf(1000, gain)], f, FS)
    assert lo[0] == pytest.approx(gain, abs=0.01) and abs(lo[2]) < 0.01
    assert hi[2] == pytest.approx(gain, abs=0.01) and abs(hi[0]) < 0.01
    assert lo[1] == pytest.approx(gain / 2, abs=1e-6)
    assert hi[1] == pytest.approx(gain / 2, abs=1e-6)


def test_time_domain_sine_matches_response():
    flt = Filter.peak(200, -6, 4)
    t = np.arange(FS) / FS
    x = np.sin(2 * np.pi * 200 * t)
    y = sosfilt(filters_to_sos([flt], FS), x)
    amp = np.max(np.abs(y[FS // 2:]))
    assert 20 * np.log10(amp) == pytest.approx(-6, abs=0.05)


def test_zero_gain_is_identity_and_empty_cascade():
    f = np.geomspace(20, 20000, 50)
    np.testing.assert_allclose(response_db([Filter.peak(500, 0, 2)], f, FS), 0, atol=1e-9)
    np.testing.assert_allclose(sos_response_db(filters_to_sos([], FS), f, FS), 0, atol=1e-12)


def test_auto_preamp_uses_combined_response():
    flts = [Filter.peak(100, 3, 1), Filter.peak(130, 3, 1), Filter.peak(1000, -6, 2)]
    pre = auto_preamp_db(flts, FS, margin_db=0.5)
    peak = response_db(flts, np.geomspace(10, 23000, 4000), FS).max()
    assert peak > 3.0                         # overlapping boosts add up
    assert pre == pytest.approx(-(peak + 0.5), abs=0.02)
    assert auto_preamp_db([Filter.peak(100, -6, 2)], FS) == 0.0
    assert auto_preamp_db([], FS) == 0.0


def test_invalid_filters_rejected():
    with pytest.raises(ValueError):
        filter_sos(Filter.peak(30000, 3, 1), FS)
    with pytest.raises(ValueError):
        filter_sos(Filter.peak(100, 3, 0), FS)
