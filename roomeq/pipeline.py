"""Measurement sessions and the measure -> solve -> apply -> re-measure loop.

Hardware-agnostic: anything implementing :class:`MeasurementRig` works (the simulator now, the
phone + sound card later).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from .dsp.analysis import (AnalysisConfig, Measurement, MeasurementError, Repeatability, analyze_recording,
                           repeatability)
from .dsp.biquad import Filter, FloatArray, response_db
from .dsp.calibration import FLAT_WARNING, MicCalibration, flat
from .dsp.solver import SolveResult, SolverConfig, error_metrics, solve
from .dsp.spectrum import power_average_db
from .dsp.sweep import SweepConfig, TestSignal, make_test_signal
from .dsp.target import TargetCurve

Log = Callable[[str], None]


class MeasurementRig(Protocol):
    name: str

    def prepare_position(self, position: int, total: int) -> None:
        """Tell the user where to put the phone (blocks until ready)."""

    def record(self, playback: FloatArray, position: int = 0, eq: Sequence[Filter] = (),
               preamp_db: float = 0.0) -> tuple[FloatArray, float]:
        """Play ``playback`` (optionally through ``eq``) and return ``(recording, recording_fs)``."""


def _sweep(rig: "MeasurementRig", ts: TestSignal, position: int, eq: Sequence[Filter], preamp_db: float,
           analysis: AnalysisConfig, log: Log) -> Measurement:
    """One sweep; an unreliable or unreadable one is retaken once automatically."""
    last: Exception | None = None
    m: Measurement | None = None
    for attempt in range(2):
        try:
            rec, fs_rec = rig.record(ts.signal, position, eq, preamp_db)
            m = analyze_recording(rec, fs_rec, ts, analysis)
        except MeasurementError as exc:
            last = exc
            log(f"    ! {exc} - retaking this sweep")
            continue
        if m.quality.reliable or attempt == 1:
            return m
        log(f"    ! sweep unreliable ({'; '.join(m.quality.warnings) or 'quality check failed'}) - retaking it")
    if m is not None:
        return m
    raise last  # type: ignore[misc]


@dataclass
class MeasurementSet:
    freqs: FloatArray
    average_db: FloatArray                      # energy average over all sweeps, uncalibrated
    measurements: list[list[Measurement]]       # [position][repeat]
    repeatability: list[Repeatability]
    warnings: list[str]
    reliable: bool


def measure_set(rig: MeasurementRig, ts: TestSignal, positions: int = 3, repeats: int = 1,
                eq: Sequence[Filter] = (), preamp_db: float = 0.0,
                analysis: AnalysisConfig | None = None, log: Log = print) -> MeasurementSet:
    analysis = analysis or AnalysisConfig()
    per_pos: list[list[Measurement]] = []
    reps: list[Repeatability] = []
    warnings: list[str] = []
    reliable = True
    for p in range(positions):
        rig.prepare_position(p, positions)
        row: list[Measurement] = []
        for r in range(repeats):
            m = _sweep(rig, ts, p, eq, preamp_db, analysis, log)
            q = m.quality
            log(f"  position {p + 1}/{positions} sweep {r + 1}/{repeats}: peak {q.peak_dbfs:5.1f} dBFS, "
                f"SNR bass {q.snr_bass_db:4.0f} dB / mid {q.snr_mid_db:4.0f} dB, drift {q.drift_ppm:+6.1f} ppm"
                + ("" if q.reliable else "  [UNRELIABLE]"))
            for w in q.warnings:
                log(f"    ! {w}")
                warnings.append(f"position {p + 1}: {w}")
            reliable &= q.reliable
            row.append(m)
        if repeats > 1:
            rp = repeatability(row)
            reps.append(rp)
            log(f"    repeatability: bass coherence {rp.coherence_bass:.3f}, spread {rp.spread_bass_db:.2f} dB")
            for w in rp.warnings:
                log(f"    ! {w}")
                warnings.append(f"position {p + 1}: {w}")
            reliable &= rp.reliable
        per_pos.append(row)
    freqs = per_pos[0][0].freqs
    avg = power_average_db([m.db for row in per_pos for m in row])
    return MeasurementSet(freqs, avg, per_pos, reps, warnings, reliable)


def measure_pairs(rig: MeasurementRig, ts: TestSignal, positions: int, eq: Sequence[Filter], preamp_db: float,
                  analysis: AnalysisConfig | None = None, log: Log = print
                  ) -> tuple[MeasurementSet, MeasurementSet]:
    """At each position: one sweep without EQ, then one through the EQ, phone not moved in between.

    Comparing the two averages is fair even in rooms where the response changes a lot with position.
    """
    analysis = analysis or AnalysisConfig()
    off_rows: list[list[Measurement]] = []
    on_rows: list[list[Measurement]] = []
    warnings: list[str] = []
    reliable = True
    for p in range(positions):
        rig.prepare_position(p, positions)
        row = []
        for label, flt, pre in (("EQ off", (), 0.0), ("EQ on ", eq, preamp_db)):
            m = _sweep(rig, ts, p, flt, pre, analysis, log)
            q = m.quality
            log(f"  position {p + 1}/{positions} {label}: peak {q.peak_dbfs:5.1f} dBFS, SNR bass "
                f"{q.snr_bass_db:4.0f} dB / mid {q.snr_mid_db:4.0f} dB" + ("" if q.reliable else "  [UNRELIABLE]"))
            for w in q.warnings:
                log(f"    ! {w}")
                warnings.append(f"position {p + 1}: {w}")
            reliable &= q.reliable
            row.append(m)
        off_rows.append([row[0]])
        on_rows.append([row[1]])
    freqs = off_rows[0][0].freqs
    mk = lambda rows: MeasurementSet(freqs, power_average_db([r[0].db for r in rows]), rows, [], warnings, reliable)
    return mk(off_rows), mk(on_rows)


def position_spread(sets: Sequence[MeasurementSet]) -> FloatArray:
    """Per-frequency standard deviation (dB) across every position measured without EQ."""
    dbs = np.array([m.db for ms in sets for row in ms.measurements for m in row])
    if len(dbs) < 2:
        return np.zeros(dbs.shape[1])
    dbs = dbs - np.median(dbs, axis=1, keepdims=True)     # compare shapes, not levels
    return dbs.std(axis=0)


@dataclass
class Iteration:
    index: int
    filters: list[Filter]
    preamp_db: float
    measured_db: FloatArray          # calibrated, with ``filters`` applied (round 0: without EQ)
    rms_db: float
    rms_bass_db: float
    before_db: FloatArray | None = None   # same positions, EQ off (paired rounds only)
    rms_before_db: float | None = None

    @property
    def improvement_db(self) -> float:
        return 0.0 if self.rms_before_db is None else self.rms_before_db - self.rms_db


@dataclass
class AutotuneReport:
    freqs: FloatArray
    target_db: FloatArray            # level-aligned target
    initial: SolveResult             # solve on the first, unequalised measurement
    best: Iteration                  # index 0 = no EQ helped
    history: list[Iteration]
    warnings: list[str] = field(default_factory=list)

    @property
    def rms_before(self) -> float:
        """Without EQ, at the same positions as :attr:`rms_after`."""
        b = self.best
        return b.rms_before_db if b.rms_before_db is not None else self.history[0].rms_db

    @property
    def rms_after(self) -> float:
        return self.best.rms_db

    @property
    def before_db(self) -> FloatArray:
        return self.best.before_db if self.best.before_db is not None else self.history[0].measured_db


def autotune(rig: MeasurementRig, target: TargetCurve | None = None, solver: SolverConfig | None = None,
             calibration: MicCalibration | None = None, iterations: int = 3, positions: int = 3,
             repeats: int = 1, sweep: SweepConfig | None = None, analysis: AnalysisConfig | None = None,
             min_gain_db: float = 0.3, log: Log = print,
             on_measured: Callable[[str, MeasurementSet, Sequence[Filter], float], None] | None = None
             ) -> AutotuneReport:
    """Measure, solve, apply, re-measure.

    Round 0 measures without EQ and solves. Every later round measures each position twice, EQ off and
    EQ on, without moving the phone, so the improvement it reports is a fair comparison. The EQ-off
    sweeps of all rounds are pooled, which gives a better room average (and position spread) for the
    next solve. The round with the largest measured improvement wins; if none improves by at least
    ``min_gain_db``, the result is "no EQ".
    """
    target = target or TargetCurve()
    cal = calibration or flat()
    sweep = sweep or SweepConfig()
    ts = make_test_signal(sweep)
    solver = solver or SolverConfig(fs=float(sweep.fs))
    warnings: list[str] = []
    if cal.is_flat:
        warnings.append(FLAT_WARNING)
        log(f"! {FLAT_WARNING}")

    log(f"Round 0: measuring without EQ ({positions} positions x {repeats} sweeps) ...")
    ms = measure_set(rig, ts, positions, repeats, (), 0.0, analysis, log)
    if on_measured:
        on_measured("autotune round 0 (no EQ)", ms, (), 0.0)
    warnings += ms.warnings
    freqs = ms.freqs
    pool = [ms]
    base = cal.apply(freqs, ms.average_db)
    spread = position_spread(pool) if positions > 1 else None
    initial = solve(freqs, base, target, solver, spread_db=spread)
    warnings += initial.warnings
    f_lo, null = initial.f_lo, initial.null_mask

    def metrics(db: FloatArray) -> tuple[float, float]:
        return error_metrics(freqs, db, target, solver, f_lo, null)

    r0, rb0 = metrics(base)
    history = [Iteration(0, [], 0.0, base, r0, rb0)]
    best = history[0]
    log(f"Without EQ: RMS error {r0:.2f} dB (bass {rb0:.2f} dB)")

    current = initial
    for k in range(1, iterations + 1):
        if not current.filters:
            log("The solver found nothing worth correcting.")
            break
        log(f"Round {k}: {len(current.filters)} filters, preamp {current.preamp_db:+.1f} dB. Measuring each "
            "position with EQ off and on (don't move the phone in between) ...")
        off, on = measure_pairs(rig, ts, positions, current.filters, current.preamp_db, analysis, log)
        if on_measured:
            on_measured(f"autotune round {k} (EQ off)", off, (), 0.0)
            on_measured(f"autotune round {k} (EQ on)", on, current.filters, current.preamp_db)
        warnings += on.warnings
        before = cal.apply(freqs, off.average_db)
        after = cal.apply(freqs, on.average_db)
        rb, _ = metrics(before)
        r, rbass = metrics(after)
        it = Iteration(k, list(current.filters), current.preamp_db, after, r, rbass, before, rb)
        history.append(it)
        log(f"Round {k}: same positions, RMS error without EQ {rb:.2f} dB -> with EQ {r:.2f} dB "
            f"(improvement {it.improvement_db:+.2f} dB)")
        if it.improvement_db > max(best.improvement_db, 0.0) + (min_gain_db if best.index else 0.0):
            best = it
        elif best.index:
            log("No further improvement; keeping the best round.")
            break
        if k == iterations:
            break
        pool.append(off)
        pooled = cal.apply(freqs, power_average_db([p.average_db for p in pool]))
        nxt = solve(freqs, pooled, target, solver, initial=current.filters, spread_db=position_spread(pool))
        log(f"  re-solved on {sum(len(p.measurements) for p in pool)} positions: predicted "
            f"{nxt.rms_before:.2f} -> {nxt.rms_after:.2f} dB")
        current = nxt
    if best.index and best.improvement_db < min_gain_db:
        best = history[0]

    al_target = target.evaluate(freqs) + initial.offset_db
    return AutotuneReport(freqs, al_target, initial, best, history, list(dict.fromkeys(warnings)))
