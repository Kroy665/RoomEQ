"""Real-time path: kernels vs. reference, glitch-free switching, limiter, resampler/drift, safety."""

import gc
import math
import tracemalloc

import numpy as np
import pytest
from scipy.signal import sosfilt

from roomeq.dsp.biquad import Filter, filters_to_sos
from roomeq.dsp.rt_kernels import MAX_SECTIONS, biquad_cascade, gain_limiter
from roomeq.engine.core import RS_UNDER, EngineCore, EngineSettings, UnsafePreset

FS = 48000
ROOM_EQ = [Filter.peak(70.6, -8.7, 4.62), Filter.peak(90, -9, 5.09), Filter.peak(208, 3, 2.5),
           Filter.peak(242, -9, 5.21), Filter.peak(489, 3, 1.16)]


def sine(f, seconds, fs=FS, amp=0.5, ch=2):
    t = np.arange(int(seconds * fs)) / fs
    return np.repeat((amp * np.sin(2 * np.pi * f * t))[:, None], ch, axis=1)


def run(core: EngineCore, x: np.ndarray, block: int = 256) -> np.ndarray:
    """Feed input and pull output in lock-step (same clock)."""
    out = np.zeros((len(x), x.shape[1]), dtype=np.float32)
    for i in range(0, len(x) - block + 1, block):
        core.process_input(x[i:i + block].astype(np.float32))
        core.process_output(out[i:i + block])
    return out


def test_biquad_kernel_matches_scipy_across_blocks():
    sos = filters_to_sos(ROOM_EQ, FS)
    x = np.random.default_rng(0).standard_normal((4096, 2))
    state = np.zeros((MAX_SECTIONS, 2, 2))
    out = np.zeros_like(x)
    padded = np.zeros((MAX_SECTIONS, 6)); padded[:len(sos)] = sos
    for i in range(0, 4096, 256):                       # state must carry across blocks
        biquad_cascade(x[i:i + 256], padded, len(sos), state, 0.5, out[i:i + 256])
    np.testing.assert_allclose(out, 0.5 * sosfilt(sos, x, axis=0), atol=1e-10)


def test_eq_change_is_glitch_free():
    core = EngineCore()
    core.set_volume(0.0)
    core.gst[0] = 1.0                                    # skip the soft start for this test
    x = sine(100, 3.0)
    out = np.zeros_like(x, dtype=np.float32)
    blk = 256
    for k, i in enumerate(range(0, len(x) - blk + 1, blk)):
        if k == 60:
            core.set_eq([Filter.peak(100, -9, 4)], name="a")
        if k == 160:
            core.set_eq([Filter.peak(100, 3, 1)], name="b")
        if k == 260:
            core.set_bypass(True)
        core.process_input(x[i:i + blk].astype(np.float32))
        core.process_output(out[i:i + blk])
    end = (len(x) // blk) * blk
    y = out[:end, 0].astype(float)
    # The input's steepest step is 0.5 * 2*pi*100/48000 = 0.00654. Every EQ here is <= 0 dB after
    # preamp, so any step above that (plus a hair) would be a click from the switching itself.
    assert np.max(np.abs(np.diff(y[2000:]))) < 0.5 * 2 * np.pi * 100 / FS * 1.05
    assert core.take_stats()["faults"] == 0


def test_eq_reaches_designed_gain_and_bypass_is_level_matched():
    core = EngineCore()
    core.gst[0] = 1.0
    pre = core.set_eq([Filter.peak(1000, 3, 1)], name="t")
    assert pre == pytest.approx(-3.0, abs=0.05)
    y = run(core, sine(1000, 1.0, amp=0.1))[-4800:, 0]
    assert 20 * np.log10(np.max(np.abs(y)) / 0.1) == pytest.approx(0.0, abs=0.1)   # +3 boost -3 preamp
    core.set_bypass(True)
    y = run(core, sine(1000, 1.0, amp=0.1))[-4800:, 0]
    assert 20 * np.log10(np.max(np.abs(y)) / 0.1) == pytest.approx(-3.0, abs=0.1)  # preamp kept in bypass


def test_limiter_never_exceeds_ceiling_and_is_transparent_below():
    ceiling = 10 ** (-1 / 20)
    L = 72
    for amp, expect_transparent in ((4.0, False), (0.5, True)):
        x = sine(60, 1.0, amp=amp) + sine(3000, 1.0, amp=amp / 4)
        orig = x.copy()
        gst = np.array([1.0, 1.0, 1e-4]); buf = np.zeros((L, 2)); lst = np.array([0.0, 1.0, 0.0])
        stats = np.array([0.0, 0.0, 1.0])
        params = np.array([ceiling, 1 - math.exp(-1 / (0.12 * FS))])
        for i in range(0, len(x), 256):
            gain_limiter(x[i:i + 256], gst, buf, lst, params, stats)
        assert np.max(np.abs(x)) <= ceiling + 1e-12
        if expect_transparent:
            np.testing.assert_allclose(x[L - 1:], orig[:len(x) - L + 1], atol=1e-12)   # pure delay
        else:
            # gain reduction did the work, not the safety clip: few samples sit exactly at the ceiling
            assert np.mean(np.isclose(np.abs(x), ceiling, atol=1e-9)) < 0.01


def test_soft_start_and_volume_ramp():
    core = EngineCore(EngineSettings(soft_start_s=1.0))
    y = run(core, sine(440, 1.5, amp=0.5))[:, 0]
    first = np.max(np.abs(y[:2400]))                 # first 50 ms
    later = np.max(np.abs(y[-4800:]))
    assert first < 0.05 * later
    core.set_volume(-12.0)
    y = run(core, sine(440, 1.0, amp=0.5))[-4800:, 0]
    assert 20 * np.log10(np.max(np.abs(y)) / 0.5) == pytest.approx(-12.0, abs=0.2)
    assert core.set_volume(+6.0) == 0.0              # trims can only attenuate


def test_panic_is_fast():
    core = EngineCore()
    core.gst[0] = 1.0
    core.set_eq([Filter.peak(100, -9, 4)], name="x")
    run(core, sine(1000, 0.5))
    core.panic()
    y = run(core, sine(1000, 0.1, amp=0.5))[:, 0]
    # within 10 ms the output is at the panic level (-20 dB)
    assert np.max(np.abs(y[480:])) < 0.5 * 10 ** (-20 / 20) * 1.05
    assert core.take_stats()["eq"] == "PANIC"
    core.clear_panic()
    assert core.take_stats()["eq"] == "on"


def test_unsafe_presets_rejected():
    core = EngineCore()
    with pytest.raises(UnsafePreset):
        core.set_eq([Filter.peak(100, 30, 1)])
    with pytest.raises(UnsafePreset):
        core.set_eq([Filter.peak(100, 3, 0.01)])
    with pytest.raises(UnsafePreset):
        core.set_eq([Filter.peak(100, 1, 1)] * 20)
    # preamp can be made more negative by the caller, never less than needed
    assert core.set_eq([Filter.peak(100, 6, 1)], preamp_db=0.0) <= -5.9


def simulate_clocks(core: EngineCore, fs_in_true: float, fs_out: float, seconds: float, freq: float = 440.0,
                    amp: float = 0.3, blk: int = 256) -> np.ndarray:
    """Producer (BlackHole) and consumer (output device) on independent clocks, as timed events.

    An input block is delivered when its last sample has been captured; an output block is
    requested when the device needs it. Each event carries its own timestamp.
    """
    next_in, next_out = blk / fs_in_true, 0.0
    inp = np.zeros((blk, 2), dtype=np.float32)
    out = np.zeros((int(seconds * fs_out) + blk, 2), dtype=np.float32)
    phase, o = 0.0, 0
    n = np.arange(blk)
    while next_out < seconds:
        if next_in <= next_out:
            inp[:, 0] = inp[:, 1] = amp * np.sin(phase + 2 * np.pi * freq * n / fs_in_true)
            phase += 2 * np.pi * freq * blk / fs_in_true
            core.process_input(inp, now=next_in)
            next_in += blk / fs_in_true
        else:
            core.process_output(out[o:o + blk], now=next_out)
            o += blk
            next_out += blk / fs_out
    return out[:o]


@pytest.mark.parametrize("fs_in,fs_out", [(48000, 48000), (44100, 48000), (48000, 44100)])
def test_resampler_quality(fs_in, fs_out):
    core = EngineCore(EngineSettings(fs_in=fs_in, fs_out=fs_out))
    core.gst[0] = 1.0
    y = simulate_clocks(core, fs_in, fs_out, 3.0, freq=1000.0, amp=0.25)[fs_out:, 0].astype(float)
    t = np.arange(len(y)) / fs_out
    A = np.column_stack([np.sin(2 * np.pi * 1000 * t), np.cos(2 * np.pi * 1000 * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    thdn_db = 20 * np.log10(np.std(resid) / np.std(A @ coef))
    assert np.hypot(coef[0], coef[1]) == pytest.approx(0.25, rel=0.01)
    assert thdn_db < -70, thdn_db
    assert core.take_stats()["underruns"] == 0


@pytest.mark.parametrize("ppm", [-150.0, 0.0, 90.0])
def test_clock_drift_tracked_without_glitches(ppm):
    """Two free-running clocks (BlackHole vs. headphone jack) for two simulated minutes."""
    core = EngineCore()
    simulate_clocks(core, FS * (1 + ppm * 1e-6), FS, 120.0)
    st = core.take_stats()
    assert st["underruns"] == 0 and st["overruns"] == 0
    assert st["drift_ppm"] == pytest.approx(ppm, abs=15.0)          # converged on the true clock ratio
    assert abs(st["buffer_fill"] - core.rs[4]) < 64


def test_output_path_does_not_grow_memory():
    core = EngineCore()
    core.set_eq(ROOM_EQ, name="room")
    inp = np.zeros((256, 2), dtype=np.float32)
    out = np.zeros((256, 2), dtype=np.float32)
    for _ in range(200):
        core.process_input(inp); core.process_output(out)
    gc.collect()
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for _ in range(5000):
        core.process_input(inp); core.process_output(out)
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    growth = sum(s.size_diff for s in snap1.compare_to(snap0, "filename"))
    assert growth < 16 * 1024, growth


def test_unstable_filter_state_is_contained():
    core = EngineCore()
    core.gst[0] = 1.0
    core.set_eq([Filter.peak(100, -6, 2)], name="x")
    run(core, sine(100, 0.5))
    core.state[core.ctl[0], 0, 0, 0] = np.nan              # simulate a numerical blow-up
    y = run(core, sine(100, 0.2))
    assert np.all(np.isfinite(y))
    assert core.take_stats()["faults"] >= 1


def test_buffer_target_adapts_to_bursty_devices():
    """Some drivers deliver several blocks at once; after a few resyncs the engine must settle."""
    core = EngineCore()
    blk, burst = 256, 4
    inp = np.zeros((blk, 2), dtype=np.float32)
    out = np.zeros((blk, 2), dtype=np.float32)
    t_out, k_in = 0.0, 0
    hist = []
    while t_out < 30.0:
        # input arrives as `burst` blocks back-to-back every burst*blk samples
        while (k_in // burst + 1) * burst * blk / FS <= t_out:
            core.process_input(inp, now=t_out)
            k_in += 1
        core.process_output(out, now=t_out)
        t_out += blk / FS
        hist.append(core.rs[RS_UNDER])
    assert hist[-1] <= 4                                   # a few resyncs while it learns...
    assert hist[-1] == hist[len(hist) // 3]                # ...then none for the last 20 s
    assert core.rs[4] > 784                                # it buffered more


@pytest.mark.parametrize("through", [True, False])
def test_test_signal_injection_replaces_music(through):
    core = EngineCore()
    core.gst[0] = 1.0
    core.set_eq([Filter.peak(1000, -6, 1)], name="x")       # preamp 0, so through-EQ is -6 dB at 1 kHz
    run(core, sine(440, 0.5, amp=0.5))                       # music playing, transition done
    tone = (0.25 * np.sin(2 * np.pi * 1000 * np.arange(FS) / FS)).astype(np.float32)
    core.start_injection(tone, through_eq=through)
    y = run(core, sine(440, 1.2, amp=0.5))[:, 0].astype(float)
    assert core.injection_done()
    core.stop_injection()
    seg = y[12000:36000]                                      # inside the tone (after limiter delay)
    expect = 0.25 * (10 ** (-6 / 20) if through else 1.0)
    assert np.max(np.abs(seg)) == pytest.approx(expect, rel=0.03)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / FS)
    assert spec[np.argmin(abs(f - 440))] < 1e-3 * spec.max()   # no music under the measurement
