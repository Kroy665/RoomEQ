"""Deconvolution, alignment and drift compensation against synthetic rooms."""

import numpy as np
import pytest
from scipy import signal as sps

from roomeq.dsp.analysis import (AnalysisConfig, MeasurementError, analyze_recording, estimate_drift, magnitude,
                                 repeatability, resample_nominal, resample_ratio, window_ir)
from roomeq.dsp.biquad import Filter, filters_to_sos
from roomeq.dsp.spectrum import log_grid, power_average_db, smooth_power
from roomeq.dsp.sweep import SweepConfig, make_test_signal

FS = 48000
TS = make_test_signal(SweepConfig(fs=FS, duration=6.0))


def synthetic_ir(fs=FS):
    imp = np.zeros(int(0.5 * fs))
    imp[0] = 1.0
    flt = [Filter.peak(60, 8, 5), Filter.peak(150, -6, 2), Filter.peak(2000, 3, 1)]
    h = sps.sosfilt(filters_to_sos(flt, fs), imp)
    h = sps.sosfilt(sps.butter(2, 35, "highpass", fs=fs, output="sos"), h)
    h[int(0.004 * fs):] += 0.4 * h[: len(h) - int(0.004 * fs)]   # one reflection
    return h


def record(ir, drift_ppm=0.0, phone_fs=FS, lead_s=0.37, noise=1e-5, gain=0.5, seed=0):
    y = sps.fftconvolve(TS.signal, ir) * gain
    y = np.concatenate([np.zeros(int(lead_s * FS)), y, np.zeros(int(0.2 * FS))])
    y = resample_nominal(y, FS, phone_fs)
    if drift_ppm:
        y = resample_ratio(y, 1 / (1 + drift_ppm * 1e-6))
    return y + noise * np.random.default_rng(seed).standard_normal(len(y))


def truth_db(ir, grid, cfg=AnalysisConfig()):
    pad = np.concatenate([np.zeros(1000), ir])
    w, _ = window_ir(pad, 1000 + int(np.argmax(np.abs(ir))), FS, cfg.window_left_ms, cfg.window_right_ms)
    return magnitude(w, FS, grid, cfg.smoothing_fraction)[0]


def deviation(m, ir, lo, hi):
    t = truth_db(ir, m.freqs)
    band = (m.freqs > 200) & (m.freqs < 2000)
    d = m.db - t - np.median((m.db - t)[band])
    k = (m.freqs >= lo) & (m.freqs <= hi)
    return float(np.max(np.abs(d[k])))


def test_test_signal_layout():
    assert TS.marker_separation == TS.marker2_start - TS.marker1_start
    assert np.allclose(TS.signal[TS.marker1_start:TS.marker1_start + len(TS.marker)], TS.marker)
    assert np.allclose(TS.signal[TS.marker2_start:TS.marker2_start + len(TS.marker)], TS.marker)
    assert np.max(np.abs(TS.signal)) <= 10 ** (-12 / 20) + 1e-9


def test_recovers_response_without_drift():
    ir = synthetic_ir()
    m = analyze_recording(record(ir), FS, TS)
    assert deviation(m, ir, 30, 16000) < 0.3
    assert m.quality.reliable and not m.quality.warnings


@pytest.mark.parametrize("ppm", [-80.0, 35.0, 150.0])
def test_estimates_and_compensates_drift(ppm):
    ir = synthetic_ir()
    rec = record(ir, drift_ppm=ppm)
    ratio, _ = estimate_drift(rec, TS)
    assert (ratio - 1) * 1e6 == pytest.approx(ppm, abs=1.0)
    m = analyze_recording(rec, FS, TS)
    assert m.quality.drift_ppm == pytest.approx(ppm, abs=1.0)
    assert deviation(m, ir, 30, 16000) < 0.5


def test_drift_compensation_keeps_impulse_response_compact():
    # Uncorrected drift smears the IR; after 1/6-oct smoothing the magnitude barely moves, but the
    # smeared energy lands in the noise region and wrecks the high-frequency SNR estimate.
    ir = synthetic_ir()
    rec = record(ir, drift_ppm=150)
    good = analyze_recording(rec, FS, TS)
    bad = analyze_recording(rec, FS, TS, AnalysisConfig(compensate_drift=False))
    assert good.quality.snr_mid_db > bad.quality.snr_mid_db + 20


def test_sample_rate_mismatch_44k1_phone():
    ir = synthetic_ir()
    m = analyze_recording(record(ir, drift_ppm=40, phone_fs=44100), 44100, TS)
    assert m.quality.drift_ppm == pytest.approx(40, abs=1.5)
    assert deviation(m, ir, 30, 16000) < 0.5


def test_alignment_independent_of_recording_lead():
    ir = synthetic_ir()
    a = analyze_recording(record(ir, lead_s=0.1), FS, TS)
    b = analyze_recording(record(ir, lead_s=2.3), FS, TS)
    np.testing.assert_allclose(a.db, b.db, atol=0.05)


def test_inverted_polarity_still_works():
    ir = -synthetic_ir()
    m = analyze_recording(record(ir), FS, TS)
    assert deviation(m, ir, 30, 16000) < 0.3


def test_clipping_and_noise_are_flagged():
    ir = synthetic_ir()
    clipped = np.clip(record(ir, gain=6.0), -1, 1)
    assert not analyze_recording(clipped, FS, TS).quality.reliable
    noisy = analyze_recording(record(ir, noise=0.05, gain=0.03), FS, TS)
    assert noisy.quality.snr_bass_db < 20
    assert any("signal-to-noise" in w for w in noisy.quality.warnings)


def test_missing_markers_raise():
    with pytest.raises(MeasurementError):
        analyze_recording(np.random.default_rng(0).standard_normal(len(TS.signal)) * 1e-3, FS, TS)
    with pytest.raises(MeasurementError):
        analyze_recording(np.zeros(1000), FS, TS)


def test_repeatability():
    ir = synthetic_ir()
    same = [analyze_recording(record(ir, seed=s), FS, TS) for s in range(2)]
    rp = repeatability(same)
    assert rp.coherence_bass > 0.99 and rp.reliable
    ir2 = synthetic_ir()
    ir2[int(0.002 * FS):] += 0.9 * ir2[: len(ir2) - int(0.002 * FS)]     # something moved
    diff = [same[0], analyze_recording(record(ir2), FS, TS)]
    assert not repeatability(diff).reliable


def test_smoothing_flat_stays_flat_and_power_average():
    f = np.linspace(0, 24000, 24001)
    grid = log_grid()
    np.testing.assert_allclose(smooth_power(f, np.ones_like(f), grid), 1.0, rtol=1e-9)
    avg = power_average_db([np.zeros(3), np.full(3, -100.0)])
    np.testing.assert_allclose(avg, 10 * np.log10(0.5), atol=1e-6)


def test_repeatability_ignores_noise_inside_deep_dips():
    # Real-room case: sweeps agree everywhere except inside a deep notch, where little noise swings the dB.
    ir = sps.sosfilt(filters_to_sos([Filter.peak(130, -30, 6)], FS), synthetic_ir())
    reps = [analyze_recording(record(ir, noise=3e-4, seed=s), FS, TS) for s in range(2)]
    rp = repeatability(reps)
    k = (reps[0].freqs > 120) & (reps[0].freqs < 140)
    assert rp.spread_db[k].max() > rp.spread_bass_db       # the notch is noisier than the reported figure
    assert rp.reliable, rp.warnings


def test_marker_search_ignores_impostor_peaks():
    # a loud knock right where a wrong "second marker" would give ~1600 ppm (the bug seen on real hardware)
    ir = synthetic_ir()
    rec = record(ir, drift_ppm=15, gain=0.05)
    fake = int(0.37 * FS) + TS.marker2_start + int(TS.marker_separation * 1648e-6)
    rec[fake:fake + len(TS.marker)] += 0.2 * TS.marker[::-1] + 0.15 * TS.marker
    m = analyze_recording(rec, FS, TS)
    assert m.quality.drift_ppm == pytest.approx(15, abs=2)
    assert m.quality.reliable
