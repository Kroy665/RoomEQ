"""EQ verification from four sweeps at one fixed phone position.

    a: EQ bypassed            b: through the EQ
    c: EQ bypassed, 10 dB quieter            d: EQ bypassed again

Because the phone does not move, the room cancels out of every comparison:
    b - a      = the EQ as it actually reaches the speakers (should equal the designed EQ)
    c + 10 - a = how the speaker's own response changes with level (should be flat)
    d - a      = measurement noise / repeatability (the yardstick for the other two)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .analysis import Measurement
from .biquad import Filter, FloatArray, response_db

THIRD_OCTAVES = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
                 2500, 3150, 4000]


@dataclass
class VerifyReport:
    verdict: str
    lines: list[str]
    numbers: dict[str, float] = field(default_factory=dict)
    eq_designed: FloatArray | None = None
    eq_measured: FloatArray | None = None
    level_shape: FloatArray | None = None


def _stats(diff: FloatArray, mask: FloatArray) -> tuple[float, float, float]:
    """(median offset, rms of shape after removing the offset, max |shape|) over the mask."""
    off = float(np.median(diff[mask]))
    shape = diff[mask] - off
    return off, float(np.sqrt(np.mean(shape ** 2))), float(np.max(np.abs(shape)))


def verify_report(a: Measurement, b: Measurement, c: Measurement, d: Measurement, filters: Sequence[Filter],
                  preamp_db: float, fs: float, f_lo: float = 30.0, f_hi: float = 4000.0,
                  quieter_db: float = 10.0) -> VerifyReport:
    f = a.freqs
    # only judge where the speaker actually produces sound (skip the roll-off and deep nulls)
    lvl = a.db
    band = (f >= f_lo) & (f <= f_hi)
    mask = band & (lvl > np.median(lvl[band]) - 15.0)
    designed = response_db(list(filters), f, fs) + preamp_db if filters else np.zeros_like(f)
    measured_eq = b.db - a.db
    level_shape = c.db + quieter_db - a.db
    repeat = d.db - a.db

    eq_off, eq_rms, eq_max = _stats(measured_eq - designed, mask)
    lin_off, lin_rms, lin_max = _stats(level_shape, mask)
    rep_off, rep_rms, rep_max = _stats(repeat, mask)
    tol = max(1.0, 2.5 * rep_rms)

    lines = [
        "",
        "Verify results (same phone position for all sweeps, 30 Hz-4 kHz):",
        f"  repeatability        shape error {rep_rms:4.2f} dB rms, {rep_max:4.1f} dB max   (the noise floor)",
        f"  EQ as designed?      shape error {eq_rms:4.2f} dB rms, {eq_max:4.1f} dB max, level {eq_off:+.1f} dB",
        f"  same shape 10 dB quieter?  error {lin_rms:4.2f} dB rms, {lin_max:4.1f} dB max, level {lin_off:+.1f} dB",
        "",
        f"  {'Hz':>6} {'EQ designed':>12} {'EQ measured':>12} {'quieter - loud':>15}",
    ]
    for fc in THIRD_OCTAVES:
        i = int(np.argmin(np.abs(np.log(f / fc))))
        note = "" if mask[i] else "   (little signal here)"
        lines.append(f"  {fc:>6g} {designed[i]:>+12.1f} {measured_eq[i]:>+12.1f} "
                     f"{level_shape[i] - lin_off:>+15.1f}{note}")

    problems = []
    if rep_rms > 1.5:
        problems.append("the measurements themselves are not repeatable (noise, movement, or the phone was "
                        "moved). Keep the room quiet and the phone still, then verify again.")
    if lin_rms > tol:
        problems.append(f"the speaker changes its own frequency response with volume ({lin_rms:.1f} dB rms "
                        "between loud and 10 dB quieter). It probably has a dynamic-bass or bass-limiter "
                        "circuit, which undoes a fixed EQ. Measure and listen at the same volume, and if the "
                        "speaker has a bass-boost/EQ mode or a loud bass knob setting, turn it off/down.")
    if eq_rms > tol and lin_rms <= tol:
        problems.append(f"the EQ reaching the speakers differs from the design by {eq_rms:.1f} dB rms although "
                        "the speaker is level-linear. That points at the audio path (another EQ/effect running, "
                        "e.g. eqMac, or a RoomEQ bug). Please send this output.")
    elif eq_rms > tol:
        problems.append("the EQ's effect is distorted by the speaker's level-dependent behaviour (see above).")
    if not problems:
        verdict = (f"PASS: the EQ reaches the speakers as designed (within {eq_rms:.1f} dB rms) and the speaker "
                   "behaves the same at both levels. Differences between predicted and measured results in "
                   "auto-tune come from moving the phone between positions.")
    else:
        verdict = "PROBLEM: " + " Also: ".join(problems)
    lines += ["", verdict]
    return VerifyReport(verdict, lines,
                        {"repeat_rms": rep_rms, "eq_rms": eq_rms, "eq_offset": eq_off, "level_rms": lin_rms,
                         "level_offset": lin_off},
                        designed, measured_eq, level_shape)
