"""Allocation-free real-time kernels (numba-compiled, plain loops, in-place on preallocated arrays).

These are deliberately written as straightforward per-sample loops over flat arrays so they port
1:1 to Swift/C. Nothing here allocates once compiled.

Layouts
-------
sos:    (2 banks, MAX_SECTIONS, 6)          b0 b1 b2 a0(=1) a1 a2, Direct Form II Transposed
state:  (2 banks, MAX_SECTIONS, CH, 2)
ring:   (RING, CH) float32 input ring buffer
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

MAX_SECTIONS = 16
SINC_TAPS = 32
SINC_PHASES = 512


@njit(cache=True, fastmath=False)
def biquad_cascade(x: np.ndarray, sos: np.ndarray, nsec: int, state: np.ndarray, gain: float,
                   out: np.ndarray) -> bool:
    """out[n, c] = gain * cascade(x[n, c]). Returns False (and zeroes state/out) on a non-finite value."""
    frames, ch = x.shape
    for c in range(ch):
        for n in range(frames):
            v = x[n, c]
            for k in range(nsec):
                b0 = sos[k, 0]; b1 = sos[k, 1]; b2 = sos[k, 2]; a1 = sos[k, 4]; a2 = sos[k, 5]
                y = b0 * v + state[k, c, 0]
                state[k, c, 0] = b1 * v - a1 * y + state[k, c, 1]
                state[k, c, 1] = b2 * v - a2 * y
                v = y
            out[n, c] = v * gain
    for c in range(ch):
        for k in range(nsec):
            if not (math.isfinite(state[k, c, 0]) and math.isfinite(state[k, c, 1])):
                state[:, :, :] = 0.0
                out[:, :] = 0.0
                return False
    return True


@njit(cache=True)
def eq_process(x: np.ndarray, sos: np.ndarray, nsec: np.ndarray, gains: np.ndarray, state: np.ndarray,
               ctl: np.ndarray, tmp_a: np.ndarray, tmp_b: np.ndarray, out: np.ndarray) -> int:
    """Two-bank EQ with warm-up and crossfade for glitch-free coefficient changes.

    ctl (int64): [active_bank, phase, counter, warmup_len, fade_len, faults]
      phase 0 = steady (only the active bank runs), 1 = warm-up (both run, output active),
      2 = crossfade (both run, mix active->other); at the end the other bank becomes active.
    Returns the phase after processing.
    """
    frames = x.shape[0]
    ch = x.shape[1]
    act = ctl[0]
    oth = 1 - act
    ok = biquad_cascade(x, sos[act], nsec[act], state[act], gains[act], tmp_a[:frames])
    if not ok:
        ctl[5] += 1
    if ctl[1] == 0:
        for n in range(frames):
            for c in range(ch):
                out[n, c] = tmp_a[n, c]
        return 0
    ok = biquad_cascade(x, sos[oth], nsec[oth], state[oth], gains[oth], tmp_b[:frames])
    if not ok:
        ctl[5] += 1
    for n in range(frames):
        if ctl[1] == 1:
            w = 0.0
            ctl[2] += 1
            if ctl[2] >= ctl[3]:
                ctl[1] = 2
                ctl[2] = 0
        elif ctl[1] == 2:
            t = ctl[2] / max(ctl[4], 1)
            w = 0.5 - 0.5 * math.cos(math.pi * t)          # raised-cosine (equal-gain, coherent signals)
            ctl[2] += 1
            if ctl[2] > ctl[4]:
                ctl[1] = 3
                w = 1.0
        else:
            w = 1.0
        for c in range(ch):
            out[n, c] = (1.0 - w) * tmp_a[n, c] + w * tmp_b[n, c]
    if ctl[1] == 3:                                          # switch over
        ctl[0] = oth
        ctl[1] = 0
        ctl[2] = 0
    return ctl[1]


@njit(cache=True)
def gain_limiter(x: np.ndarray, gst: np.ndarray, lim_buf: np.ndarray, lst: np.ndarray, params: np.ndarray,
                 stats: np.ndarray) -> None:
    """Master gain (smoothed + soft start), lookahead peak limiter and hard safety clip, in place.

    gst:    [current_gain, target_gain, ramp_per_sample]   (linear amplitude)
    lim_buf:(L, CH) delay line; lst: [write_pos, limiter_gain, attack_slope]
    params: [ceiling, release_coeff]
    stats:  [peak_in (see track_peak), peak_out, min_limiter_gain]  (updated, caller resets)

    Attack is a linear ramp that reaches the gain a new peak needs exactly when that peak leaves
    the delay line, so the output never exceeds the ceiling; the hard clip is only a safety net.
    """
    frames, ch = x.shape
    L = lim_buf.shape[0]
    ceiling = params[0]
    rel = params[1]
    g = gst[0]
    tgt = gst[1]
    step = gst[2]
    pos = int(lst[0])
    lg = lst[1]
    slope = lst[2]
    for n in range(frames):
        if g < tgt:
            g = min(g + step, tgt)
        elif g > tgt:
            g = max(g - 4.0 * step, tgt)                     # going down is faster than coming up
        for c in range(ch):
            lim_buf[pos, c] = x[n, c] * g
        peak = 0.0
        for i in range(L):
            for c in range(ch):
                a = abs(lim_buf[i, c])
                if a > peak:
                    peak = a
        target = 1.0 if peak <= ceiling else ceiling / peak
        if target < lg:
            need = (lg - target) / (L - 1) if L > 1 else lg - target
            if need > slope:
                slope = need
            lg = max(lg - slope, target)
        else:
            slope = 0.0
            lg = lg + (target - lg) * rel
        rd = (pos + 1) % L
        for c in range(ch):
            v = lim_buf[rd, c] * lg
            if v > ceiling:
                v = ceiling
            elif v < -ceiling:
                v = -ceiling
            x[n, c] = v
            a = abs(v)
            if a > stats[1]:
                stats[1] = a
        if lg < stats[2]:
            stats[2] = lg
        pos = rd
    gst[0] = g
    lst[0] = pos
    lst[1] = lg
    lst[2] = slope


@njit(cache=True)
def track_peak(x: np.ndarray, stats: np.ndarray, idx: int) -> None:
    """stats[idx] = max(stats[idx], max |x|)."""
    m = stats[idx]
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            a = abs(x[n, c])
            if a > m:
                m = a
    stats[idx] = m


@njit(cache=True)
def inject(buf: np.ndarray, st: np.ndarray, out: np.ndarray) -> None:
    """Write the next samples of a mono test signal into every channel of ``out``.

    st (int64): [mode, position, length]; past the end it writes silence.
    """
    pos = st[1]
    ln = st[2]
    for n in range(out.shape[0]):
        v = 0.0
        if pos < ln:
            v = buf[pos]
            pos += 1
        for c in range(out.shape[1]):
            out[n, c] = v
    st[1] = pos


@njit(cache=True)
def process_block(ring: np.ndarray, widx: np.ndarray, rs: np.ndarray, table: np.ndarray, work: np.ndarray,
                  stats: np.ndarray, inj: np.ndarray, inj_st: np.ndarray, mute: bool, sos: np.ndarray,
                  nsec: np.ndarray, gains: np.ndarray, state: np.ndarray, ctl: np.ndarray, tmp_a: np.ndarray,
                  tmp_b: np.ndarray, eq_out: np.ndarray, gst: np.ndarray, lim_buf: np.ndarray, lst: np.ndarray,
                  lim_params: np.ndarray, out: np.ndarray) -> None:
    """The whole output callback in one compiled call:
    resample -> input meter -> (mute / inject test signal) -> EQ -> gain + limiter -> device buffer.

    Being a single call matters in Python: no other thread can take the GIL half-way through a block.
    """
    n = out.shape[0]
    w = work[:n]
    e = eq_out[:n]
    resample_read(ring, widx, rs, table, w)
    track_peak(w, stats, 0)
    mode = inj_st[0]
    if mute or mode != 0:
        w[:, :] = 0.0
    if mode == 1:
        inject(inj, inj_st, w)
    eq_process(w, sos, nsec, gains, state, ctl, tmp_a, tmp_b, e)
    if mode == 2:
        inject(inj, inj_st, e)
    gain_limiter(e, gst, lim_buf, lst, lim_params, stats)
    for i in range(n):
        for c in range(out.shape[1]):
            out[i, c] = e[i, c]


def make_sinc_table(taps: int = SINC_TAPS, phases: int = SINC_PHASES, cutoff: float = 0.45,
                    beta: float = 8.0) -> np.ndarray:
    """Kaiser-windowed sinc, (phases + 1, taps). Row p is the filter for fractional delay p / phases."""
    half = taps // 2
    table = np.zeros((phases + 1, taps))
    for p in range(phases + 1):
        frac = p / phases
        t = np.arange(-half + 1, half + 1) - frac            # tap positions relative to the read point
        h = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(taps, beta)
        table[p] = h / h.sum()
    return table


@njit(cache=True)
def ring_write(ring: np.ndarray, widx: np.ndarray, x: np.ndarray) -> None:
    """Append x (frames, CH) to the ring; widx[0] is the absolute write index."""
    n = ring.shape[0]
    w = widx[0]
    xc = x.shape[1]
    for i in range(x.shape[0]):
        p = (w + i) % n
        for c in range(ring.shape[1]):
            ring[p, c] = x[i, c if c < xc else xc - 1]          # mono input feeds every channel
    widx[0] = w + x.shape[0]


@njit(cache=True)
def resample_read(ring: np.ndarray, widx: np.ndarray, rs: np.ndarray, table: np.ndarray, out: np.ndarray) -> int:
    """Pull out.shape[0] frames from the ring with a variable ratio, steering the fill level.

    rs (float64): [read_pos, nominal_step, adj, integ, target_fill, kp, ki, started, max_adj, underruns,
                   overruns, fill_avg, input_since]
    ``input_since`` = input samples produced since the last input callback (time-extrapolated by the
    caller), so the controller sees a smooth fill level instead of a 1-block sawtooth.
    kp is per sample of error; ki is per sample of error per output block.
    Returns 1 if this block was an underrun (silence written), else 0.
    """
    frames, ch = out.shape
    n = ring.shape[0]
    taps = table.shape[1]
    half = taps // 2
    phases = table.shape[0] - 1
    w = widx[0]
    pos = rs[0]
    target = rs[4]
    fill = w - pos
    if rs[7] == 0.0:                                          # wait for the buffer to prime
        if fill < target + half:
            out[:, :] = 0.0
            return 0
        rs[7] = 1.0
        pos = w + rs[12] - target                             # same fill measure the controller uses
        fill = w - pos
        rs[11] = target
    if fill < half + frames * rs[1] * 1.01 + 2:              # underrun: resync
        out[:, :] = 0.0
        rs[9] += 1.0
        # the devices deliver in bigger bursts than assumed: buffer one more block from now on
        target = min(target + frames * rs[1], n / 4.0)
        rs[4] = target
        rs[0] = w + rs[12] - target
        rs[11] = target
        return 1
    if fill > n - 2 * half - frames * 2:                     # overrun: jump forward
        rs[10] += 1.0
        pos = w + rs[12] - target
        fill = w - pos
        rs[11] = target
    # slow PI control of the ratio from the fill-level error (ppm-scale, inaudible)
    rs[11] += 0.02 * (fill + rs[12] - rs[11])
    err = rs[11] - target
    rs[3] += err * rs[6]
    if rs[3] > rs[8]:
        rs[3] = rs[8]
    elif rs[3] < -rs[8]:
        rs[3] = -rs[8]
    adj = err * rs[5] + rs[3]
    if adj > rs[8]:
        adj = rs[8]
    elif adj < -rs[8]:
        adj = -rs[8]
    rs[2] = adj
    step = rs[1] * (1.0 + adj)
    for i in range(frames):
        ip = math.floor(pos)
        frac = pos - ip
        fp = frac * phases
        p0 = int(fp)
        mu = fp - p0
        base = ip - half + 1
        for c in range(ch):
            acc0 = 0.0
            acc1 = 0.0
            for k in range(taps):
                v = ring[(base + k) % n, c]
                acc0 += table[p0, k] * v
                acc1 += table[p0 + 1, k] * v
            out[i, c] = acc0 + mu * (acc1 - acc0)
        pos += step
    rs[0] = pos
    return 0
