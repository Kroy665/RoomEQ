"""Real-time EQ core, independent of any audio API.

    input callback  (BlackHole, fs_in)  -> ring buffer
    output callback (speakers,  fs_out) -> adaptive resampler -> 2-bank EQ -> gain/soft start -> limiter

Everything the callbacks touch is preallocated here; the callbacks only call numba kernels that
work in place. Control methods (set_eq, bypass, panic, volume) are called from other threads and
only write into the *inactive* EQ bank before flipping an integer flag, so no locks are needed
inside the audio path. (The GIL serialises Python-level access; the kernels hold it while they run.)
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..dsp.biquad import Filter, auto_preamp_db, filters_to_sos, response_db, dense_grid
from ..dsp.rt_kernels import (MAX_SECTIONS, SINC_TAPS, eq_process, gain_limiter, make_sinc_table, resample_read,
                              inject, process_block, ring_write, track_peak)

# ctl indices
ACTIVE, PHASE, COUNTER, WARMUP, FADE, FAULTS = range(6)
# resampler state indices
RS_POS, RS_STEP, RS_ADJ, RS_INTEG, RS_TARGET, RS_KP, RS_KI, RS_STARTED, RS_MAXADJ, RS_UNDER, RS_OVER, RS_FILL, RS_SINCE = range(13)

MAX_FILTER_GAIN_DB = 15.0
# test-signal injection modes
INJECT_OFF, INJECT_THROUGH_EQ, INJECT_BYPASS_EQ = 0, 1, 2


class UnsafePreset(ValueError):
    pass


@dataclass
class EngineSettings:
    fs_in: int = 48000
    fs_out: int = 48000
    channels: int = 2
    blocksize: int = 256
    buffer_blocks: float = 3.0             # resampler target fill, in input blocks (>= 2 + safety margin)
    ceiling_dbfs: float = -1.0
    soft_start_s: float = 1.5
    lookahead_ms: float = 1.5
    release_ms: float = 120.0
    warmup_ms: float = 150.0               # new EQ runs silently this long before the crossfade
    fade_ms: float = 40.0
    bypass_level_matched: bool = True      # bypass keeps the preamp so A/B is a fair comparison
    panic_volume_db: float = -20.0
    max_block: int = 8192
    drift_loop_s: float = 40.0             # clock-drift controller time constant
    inject_seconds: float = 30.0           # longest test signal that can be played through the engine


def validate(filters: Sequence[Filter], fs: float) -> None:
    if len(filters) > MAX_SECTIONS:
        raise UnsafePreset(f"too many filters ({len(filters)} > {MAX_SECTIONS})")
    for f in filters:
        if not (10.0 <= f.freq <= 0.45 * fs):
            raise UnsafePreset(f"filter frequency {f.freq:.1f} Hz out of range")
        if abs(f.gain_db) > MAX_FILTER_GAIN_DB:
            raise UnsafePreset(f"filter gain {f.gain_db:+.1f} dB exceeds ±{MAX_FILTER_GAIN_DB:.0f} dB")
        if not (0.1 <= f.q <= 20.0):
            raise UnsafePreset(f"filter Q {f.q:.2f} out of range")


class EngineCore:
    def __init__(self, s: EngineSettings | None = None):
        self.s = s = s or EngineSettings()
        ch, mb = s.channels, s.max_block
        # EQ banks
        self.sos = np.zeros((2, MAX_SECTIONS, 6))
        self.sos[:, :, 0] = 1.0
        self.sos[:, :, 3] = 1.0
        self.nsec = np.zeros(2, dtype=np.int64)
        self.gains = np.ones(2)
        self.state = np.zeros((2, MAX_SECTIONS, ch, 2))
        self.ctl = np.zeros(6, dtype=np.int64)
        self.tmp_a = np.zeros((mb, ch))
        self.tmp_b = np.zeros((mb, ch))
        self.work = np.zeros((mb, ch))
        self.eq_out = np.zeros((mb, ch))
        # gain / limiter
        self.gst = np.array([0.0, 1.0, 1.0 / max(s.soft_start_s * s.fs_out, 1.0)])
        L = max(2, int(s.lookahead_ms * 1e-3 * s.fs_out))
        self.lim_buf = np.zeros((L, ch))
        self.lst = np.array([0.0, 1.0, 0.0])
        self.lim_params = np.array([10 ** (s.ceiling_dbfs / 20), 1.0 - math.exp(-1.0 / (s.release_ms * 1e-3 * s.fs_out))])
        self.stats = np.array([0.0, 0.0, 1.0])
        # ring + resampler
        self.ring = np.zeros((max(16384, 8 * mb), ch), dtype=np.float32)
        self.widx = np.zeros(1, dtype=np.int64)
        self.table = make_sinc_table(cutoff=0.45 * min(1.0, s.fs_out / s.fs_in))
        self.rs = np.zeros(13)
        self.rs[RS_STEP] = s.fs_in / s.fs_out
        self.rs[RS_TARGET] = s.buffer_blocks * s.blocksize + SINC_TAPS // 2
        # Critically damped PI loop on the fill error (in samples): natural frequency w = 2*pi/T.
        # fill' = fs_in * (drift - adj)  ->  kp = 2w/fs_in, ki = w^2/fs_in per second of loop time.
        w = 2 * math.pi / s.drift_loop_s
        self.rs[RS_KP] = 2 * w / s.fs_in
        self.rs[RS_KI] = w * w / s.fs_in * (s.blocksize / s.fs_out)
        self.rs[RS_MAXADJ] = 2e-3
        self.rs[RS_FILL] = self.rs[RS_TARGET]
        # test-signal injection (measurements while the engine owns the output device)
        self.inj = np.zeros(int(s.inject_seconds * s.fs_out), dtype=np.float32)
        self.inj_st = np.zeros(3, dtype=np.int64)
        self.mute_input = False                    # silence music, e.g. for a whole measurement session
        # control-side state
        self._lock = threading.Lock()
        self._t_in = time.perf_counter()
        self._stats_time = 0.0
        self._last_output = 0.0
        self._startup_under = 0.0
        self._stats_cache: dict | None = None
        self.filters: list[Filter] = []
        self.preamp_db = 0.0
        self.preset_name = "flat"
        self.bypassed = False
        self.panicked = False
        self.volume_db = 0.0
        self.callbacks = 0
        self.callback_ns_max = 0
        self._warm_up_kernels()

    # ------------------------------------------------------------------ audio callbacks
    def process_input(self, indata: np.ndarray, now: float | None = None) -> None:
        ring_write(self.ring, self.widx, indata)
        self._t_in = time.perf_counter() if now is None else now

    def process_output(self, outdata: np.ndarray, now: float | None = None) -> None:
        t0 = time.perf_counter_ns()
        now = time.perf_counter() if now is None else now
        since = (now - self._t_in) * self.s.fs_in
        self.rs[RS_SINCE] = min(max(since, 0.0), 2.0 * self.s.blocksize)
        process_block(self.ring, self.widx, self.rs, self.table, self.work, self.stats, self.inj, self.inj_st,
                      self.mute_input, self.sos, self.nsec, self.gains, self.state, self.ctl, self.tmp_a,
                      self.tmp_b, self.eq_out, self.gst, self.lim_buf, self.lst, self.lim_params, outdata)
        self.callbacks += 1
        self._last_output = time.perf_counter()          # wall clock, even when ``now`` is simulated
        dt = time.perf_counter_ns() - t0
        if dt > self.callback_ns_max:
            self.callback_ns_max = dt

    def _warm_up_kernels(self) -> None:
        """Compile/run every kernel once so the first real callback doesn't pay for JIT compilation."""
        x = np.zeros((self.s.blocksize, self.s.channels), dtype=np.float32)
        out = np.zeros_like(x)
        self.process_input(x)
        self.process_output(out)
        self.widx[0] = 0
        self.rs[RS_POS] = 0.0
        self.rs[RS_STARTED] = 0.0
        self.rs[RS_UNDER] = 0.0
        self.gst[0] = 0.0
        self.callbacks = 0
        self.callback_ns_max = 0
        self._last_output = 0.0                     # no real audio has flowed yet

    # ------------------------------------------------------------------ control (other threads)
    def _load_bank(self, sos: np.ndarray, gain: float, warmup_ms: float, fade_ms: float, force: bool = False,
                   timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while self.ctl[PHASE] != 0:
            if time.perf_counter() - self._last_output > 0.2:
                # no audio is flowing (stream stopped or device gone): nothing can click, finish now
                self.ctl[ACTIVE] = 1 - self.ctl[ACTIVE]
                self.ctl[PHASE] = 0
                self.ctl[COUNTER] = 0
                break
            if force:
                self.ctl[PHASE] = 0                                # abort the running transition
                self.ctl[COUNTER] = 0
                break
            if time.monotonic() > deadline:
                raise TimeoutError("EQ transition did not finish (is the output stream running?)")
            time.sleep(0.005)
        oth = 1 - int(self.ctl[ACTIVE])
        k = sos.shape[0]
        self.sos[oth, :k] = sos
        self.nsec[oth] = k
        self.gains[oth] = gain
        self.state[oth] = 0.0
        self.ctl[WARMUP] = int(warmup_ms * 1e-3 * self.s.fs_out)
        self.ctl[FADE] = max(1, int(fade_ms * 1e-3 * self.s.fs_out))
        self.ctl[COUNTER] = 0
        self.ctl[PHASE] = 1                                        # the audio thread takes it from here

    def _apply(self, force: bool = False, fast: bool = False) -> None:
        warm, fade = (0.0, 5.0) if fast else (self.s.warmup_ms, self.s.fade_ms)
        pre = 10 ** (self.preamp_db / 20)
        if self.bypassed or self.panicked or not self.filters:
            g = pre if (self.s.bypass_level_matched and self.filters and not self.panicked) else 1.0
            self._load_bank(np.zeros((0, 6)), g, warm, fade, force)
        else:
            self._load_bank(filters_to_sos(self.filters, self.s.fs_out), pre, warm, fade, force)

    def set_eq(self, filters: Sequence[Filter], preamp_db: float | None = None, name: str = "custom") -> float:
        """Load a filter set (validated). The preamp is never allowed to leave headroom negative."""
        validate(filters, self.s.fs_out)
        needed = auto_preamp_db(list(filters), self.s.fs_out, margin_db=0.0)
        pre = needed if preamp_db is None else min(preamp_db, needed)
        with self._lock:
            self.filters = list(filters)
            self.preamp_db = pre
            self.preset_name = name
            self._apply()
        return pre

    def set_bypass(self, on: bool) -> None:
        with self._lock:
            self.bypassed = on
            self._apply()

    def panic(self) -> None:
        """Instant EQ bypass plus a big volume drop. Call again (or ``clear_panic``) to recover softly."""
        with self._lock:
            self.panicked = True
            pg = 10 ** (self.s.panic_volume_db / 20)
            self.gst[1] = pg
            self.gst[0] = min(self.gst[0], pg)
            self._apply(force=True, fast=True)

    def clear_panic(self) -> None:
        with self._lock:
            self.panicked = False
            self.gst[1] = 10 ** (self.volume_db / 20)               # ramps up at the soft-start rate
            self._apply()

    def set_volume(self, db: float) -> float:
        db = float(min(0.0, max(-60.0, db)))
        with self._lock:
            self.volume_db = db
            if not self.panicked:
                self.gst[1] = 10 ** (db / 20)
        return db

    def start_injection(self, signal: np.ndarray, through_eq: bool) -> None:
        """Queue a mono test signal (at fs_out). Music is replaced while it plays."""
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim != 1 or len(x) > len(self.inj):
            raise ValueError(f"test signal must be mono and at most {len(self.inj) / self.s.fs_out:.0f} s")
        self.inj_st[0] = INJECT_OFF
        self.inj[: len(x)] = x
        self.inj_st[1] = 0
        self.inj_st[2] = len(x)
        self.inj_st[0] = INJECT_THROUGH_EQ if through_eq else INJECT_BYPASS_EQ   # armed last

    def injection_done(self) -> bool:
        return self.inj_st[1] >= self.inj_st[2]

    def stop_injection(self) -> None:
        self.inj_st[0] = INJECT_OFF

    def soft_restart(self) -> None:
        self.gst[0] = 0.0

    # ------------------------------------------------------------------ reporting
    def latency_ms(self, stream_in_s: float = 0.0, stream_out_s: float = 0.0) -> float:
        buf = self.rs[RS_TARGET] / self.s.fs_in
        look = self.lim_buf.shape[0] / self.s.fs_out
        return 1e3 * (buf + look + stream_in_s + stream_out_s)

    def eq_curve_db(self, freqs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        f = dense_grid(self.s.fs_out, 512) if freqs is None else freqs
        if self.bypassed or self.panicked or not self.filters:
            return f, np.zeros_like(f)
        return f, response_db(self.filters, f, self.s.fs_out) + self.preamp_db

    def take_stats(self) -> dict:
        """Levels since the previous read. Reads closer than 250 ms share one snapshot, so the terminal
        and the dashboard can both poll without stealing each other's peaks."""
        now = time.monotonic()
        if now - self._stats_time < 0.25 and self._stats_cache is not None:
            return {**self._stats_cache, **self._live_stats()}
        self._stats_time = now
        peak_in, peak_out, min_lg = self.stats
        self.stats[:] = (0.0, 0.0, 1.0)
        db = lambda v: 20 * math.log10(max(v, 1e-9))
        out = {
            "preset": self.preset_name,
            "eq": "PANIC" if self.panicked else ("bypassed" if self.bypassed else ("on" if self.filters else "flat")),
            "volume_db": self.volume_db,
            "preamp_db": self.preamp_db,
            "peak_in_dbfs": db(peak_in),
            "peak_out_dbfs": db(peak_out),
            "limiter_gr_db": -db(min_lg),
            "drift_ppm": self.rs[RS_ADJ] * 1e6,
            "buffer_fill": float(self.rs[RS_FILL]),
            "underruns": int(self.rs[RS_UNDER] - self._startup_underruns()),
            "overruns": int(self.rs[RS_OVER]),
            "faults": int(self.ctl[FAULTS]),
            "callback_max_ms": self.callback_ns_max / 1e6,
            "gain": float(self.gst[0]),
        }
        self._stats_cache = out
        return out

    def _startup_underruns(self) -> float:
        """Resyncs during the first second are the buffer sizing itself while the output is still
        fading in (inaudible), so they are not reported as glitches."""
        if self.callbacks * self.s.blocksize < self.s.fs_out:
            self._startup_under = float(self.rs[RS_UNDER])
        return self._startup_under

    def _live_stats(self) -> dict:
        return {"preset": self.preset_name, "volume_db": self.volume_db, "preamp_db": self.preamp_db,
                "eq": "PANIC" if self.panicked else ("bypassed" if self.bypassed else ("on" if self.filters else "flat"))}
