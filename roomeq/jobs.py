"""Controller for a running engine: state for the dashboard, and background measure/autotune jobs.

While a job runs, the engine's output is muted for music and the test signals are injected into
the engine itself (through the EQ or bypassing it), so auto-tune verifies the *real* live EQ.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np

from .config import Config
from .dsp.biquad import Filter, FloatArray, response_db
from .dsp.calibration import FLAT_WARNING, MicCalibration, flat, load_calibration
from .dsp.solver import correction_masks, error_metrics, solve
from .dsp.spectrum import log_grid
from .dsp.sweep import make_test_signal
from .engine.core import EngineCore
from .pipeline import autotune, measure_set
from .presets import Preset, load_preset, save_preset
from .rigs import PhoneRig, level_check
from .server.link import PhoneError, PhoneLink

LIMITS = ("Phone mics are accurate for bass and low mids but rough above a few kHz, so RoomEQ corrects mainly "
          "below 500 Hz and nothing above 4 kHz. EQ cannot fix speaker placement or fill room nulls.")


class Cancelled(Exception):
    pass


class EnginePlayer:
    """``Player`` that plays test signals through the running engine instead of opening the device."""

    def __init__(self, core: EngineCore, out_latency_s: Callable[[], float], cancelled: Callable[[], bool]):
        self.core = core
        self.out_latency_s = out_latency_s
        self.cancelled = cancelled
        self.through_eq = False

    def __call__(self, signal: FloatArray, fs: int) -> None:
        if fs != self.core.s.fs_out:
            raise ValueError(f"test signal at {fs} Hz but the engine outputs {self.core.s.fs_out} Hz")
        self.core.start_injection(signal, self.through_eq)
        deadline = time.monotonic() + len(signal) / fs + 5.0
        try:
            while not self.core.injection_done():
                if self.cancelled():
                    raise Cancelled()
                if time.monotonic() > deadline:
                    raise PhoneError("the engine stopped playing (is the output device still connected?)")
                time.sleep(0.01)
            time.sleep(self.out_latency_s() + 0.05)               # let the last samples leave the device
        finally:
            self.core.stop_injection()


class EngineRig(PhoneRig):
    """Phone rig whose "with EQ" measurements apply the filters to the live engine."""

    def __init__(self, ctl: "Controller", job: "Job"):
        self.ctl = ctl
        self.engine_player = EnginePlayer(ctl.core, ctl.out_latency_s, lambda: job.cancel_flag.is_set())
        super().__init__(ctl.link, self.engine_player, ctl.core.s.fs_out, job.log, ctl.link.post, job.wait_ready)

    def record(self, playback, position=0, eq=(), preamp_db=0.0):
        core = self.ctl.core
        if eq:
            if list(eq) != core.filters or core.bypassed:
                core.set_bypass(False)
                core.set_eq(list(eq), preamp_db, name="auto-tune (testing)")
                time.sleep((core.s.warmup_ms + core.s.fade_ms) / 1000 + 0.1)
            self.engine_player.through_eq = True
        else:
            self.engine_player.through_eq = False
        return super().record(playback, position, (), 0.0)


@dataclass
class Job:
    kind: str
    state: str = "running"                  # running | waiting | done | failed | cancelled
    lines: list[str] = field(default_factory=list)
    prompt: str | None = None
    result: dict | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    _continue: threading.Event = field(default_factory=threading.Event)
    link: PhoneLink | None = None
    saved_eq: tuple | None = None

    def log(self, line: str) -> None:
        for ln in str(line).splitlines() or [""]:
            self.lines.append(ln)
        del self.lines[:-400]

    def wait_ready(self, prompt: str) -> None:
        self.log(prompt.strip())
        self.prompt = prompt.strip().lstrip(">").strip()
        self.state = "waiting"
        self._continue.clear()
        if self.link is not None:
            self.link.pop_events("ready")
        try:
            while True:
                if self.cancel_flag.is_set():
                    raise Cancelled()
                if self._continue.wait(0.1):
                    return
                if self.link is not None and self.link.pop_events("ready"):
                    return
        finally:
            self.prompt = None
            if self.state == "waiting":
                self.state = "running"

    def snapshot(self) -> dict:
        return {"kind": self.kind, "state": self.state, "lines": self.lines[-60:], "prompt": self.prompt,
                "result": self.result, "error": self.error}


class Controller:
    """Everything the dashboard can see and change while ``roomeq run`` is active."""

    def __init__(self, cfg: Config, core: EngineCore, link: PhoneLink, out_latency_s: Callable[[], float] = lambda: 0.02,
                 latency_ms: Callable[[], float] | None = None, devices: dict | None = None):
        self.cfg = cfg
        self.core = core
        self.link = link
        self.out_latency_s = out_latency_s
        self.latency_ms = latency_ms or (lambda: core.latency_ms())
        self.devices = devices or {}
        self.job: Job | None = None
        self._job_lock = threading.Lock()
        self.curves: dict[str, Any] = {}
        self.curves_version = 0
        self.preset_notes: list[str] = []
        self.rms: dict[str, float | None] = {"before": None, "after": None}
        self.demo = False
        self._masks_version = -1
        self._masks: tuple = (0.0, None, 0.0)

    # ------------------------------------------------------------------ presets & curves
    def load_preset(self, name: str | None) -> Preset:
        p = load_preset(name)
        self.core.set_eq(p.filters, p.preamp_db, p.name)
        self.set_curves_from_preset(p)
        return p

    def set_curves_from_preset(self, p: Preset) -> None:
        r = p.response or {}
        self.curves = {k: r.get(k) for k in ("freqs", "before", "after", "target")}
        self.curves["after_kind"] = r.get("after_kind", "predicted")
        self.curves["after_filters"] = [f.to_dict() for f in p.filters]
        self.preset_notes = list(p.notes)
        self.rms = {"before": p.rms_before, "after": p.rms_after}
        self.curves_version += 1

    def save_current(self, name: str) -> str:
        c = self.curves
        resp = {k: c.get(k) for k in ("freqs", "before", "after", "target")} if c.get("freqs") else None
        if resp is not None:
            resp["after_kind"] = c.get("after_kind", "predicted")
        p = Preset(name, list(self.core.filters), self.core.preamp_db, rms_before=self.rms.get("before"),
                   rms_after=self.rms.get("after"), notes=self.preset_notes, response=resp)
        path = save_preset(p)
        self.core.preset_name = name
        return str(path)

    def set_filters(self, filters: list[Filter]) -> float:
        pre = self.core.set_eq(filters, None, name=self.core.preset_name)
        self.curves_version += 1
        return pre

    def curves_payload(self) -> dict:
        """Curves for the dashboard. Every response is level-aligned to the target over the solver's
        alignment band (as the solver scores them), so "before", "after" and "predicted" overlay fairly
        even though measurements through the EQ include the preamp."""
        c = self.curves
        f = np.array(c["freqs"]) if c.get("freqs") else log_grid(20, 20000, 24)
        eq = response_db(self.core.filters, f, self.core.s.fs_out) if self.core.filters else np.zeros_like(f)
        out: dict[str, Any] = {"freqs": f.tolist(), "eq": (eq + self.core.preamp_db).tolist(),
                               "version": self.curves_version, "before": None, "after": None, "predicted": None,
                               "target": None, "rms_before": None, "rms_predicted": None, "rms_after": None}
        if not c.get("before"):
            return out
        solver = self.cfg.solver
        target = self.cfg.target
        tgt = target.evaluate(f)
        band = (f >= solver.align_band[0]) & (f <= solver.align_band[1])

        def aligned(curve: np.ndarray) -> np.ndarray:
            return curve - float(np.median(curve[band] - tgt[band]))

        before = aligned(np.array(c["before"]))
        # score like the solver: skip the region below the speakers' roll-off and the room nulls
        if self._masks_version != self.curves_version:
            self._masks = correction_masks(f, before, target, solver)
            self._masks_version = self.curves_version
        f_lo, nulls, _ = self._masks

        def rms(curve: np.ndarray) -> float:
            return error_metrics(f, curve, target, solver, f_lo=f_lo, null_mask=nulls, offset_db=0.0)[0]

        pred = aligned(before + eq)
        out.update(before=before.tolist(), predicted=pred.tolist(), target=tgt.tolist(),
                   rms_before=rms(before), rms_predicted=rms(pred))
        # a measured "after" only applies while the EQ is unchanged since it was measured
        if c.get("after") and c.get("after_kind") == "measured" and \
                c.get("after_filters") == [x.to_dict() for x in self.core.filters]:
            after = aligned(np.array(c["after"]))
            out.update(after=after.tolist(), rms_after=rms(after))
        return out

    def state(self) -> dict:
        st = self.core.take_stats()
        info = self.link.info
        return {
            "engine": st,
            "latency_ms": self.latency_ms(),
            "bypassed": self.core.bypassed,
            "panicked": self.core.panicked,
            "volume_db": self.core.volume_db,
            "preset": self.core.preset_name,
            "preamp_db": self.core.preamp_db,
            "filters": [f.to_dict() for f in self.core.filters],
            "notes": self.preset_notes,
            "rms": self.rms,
            "curves_version": self.curves_version,
            "devices": self.devices,
            "phone": {"connected": self.link.connected, "sample_rate": info.sample_rate if info else None,
                      "transport": info.transport if info else None},
            "job": self.job.snapshot() if self.job else None,
            "calibrated": bool(self.cfg.measurement.calibration_file),
            "demo": self.demo,
            "limits": LIMITS,
        }

    # ------------------------------------------------------------------ jobs
    def start_job(self, kind: str, positions: int, repeats: int, iterations: int = 3) -> Job:
        with self._job_lock:
            if self.job is not None and self.job.state in ("running", "waiting"):
                raise RuntimeError("a measurement is already running")
            job = Job(kind, link=self.link)
            self.job = job
        target = {"measure": self._run_measure, "autotune": self._run_autotune, "verify": self._run_verify}[kind]
        threading.Thread(target=self._wrap, args=(job, target, positions, repeats, iterations), daemon=True,
                         name=f"roomeq-{kind}").start()
        return job

    def continue_job(self) -> None:
        if self.job is not None:
            self.job._continue.set()

    def cancel_job(self) -> None:
        if self.job is not None:
            self.job.cancel_flag.set()

    def _calibration(self) -> MicCalibration:
        f = self.cfg.measurement.calibration_file
        return load_calibration(f) if f else flat()

    def _wrap(self, job: Job, fn: Callable, positions: int, repeats: int, iterations: int) -> None:
        core = self.core
        saved = (list(core.filters), core.preamp_db, core.preset_name, core.bypassed)
        job.saved_eq = saved
        final = "failed"
        try:
            core.mute_input = True
            self._wait_for_phone(job)
            fn(job, positions, repeats, iterations)
            final = "done"
            self.link.post({"type": "result", "done": True, "warnings": (job.result or {}).get("warnings", []),
                            "text": (job.result or {}).get("summary", "")})
        except Cancelled:
            final = "cancelled"
            job.log("Cancelled.")
            self.link.post({"type": "status", "text": "Measurement cancelled on the Mac."})
        except Exception as exc:                                # reported in the dashboard
            job.error = str(exc)
            job.log(f"Error: {exc}")
            job.log(traceback.format_exc(limit=3))
            self.link.post({"type": "status", "text": f"Measurement failed: {exc}"})
        finally:
            # music back first, whatever else happens
            core.stop_injection()
            core.mute_input = False
            if final != "done":                                  # put the previous EQ back
                filters, pre, name, byp = saved
                try:
                    core.set_eq(filters, pre if filters else 0.0, name)
                    core.set_bypass(byp)
                except Exception as exc:
                    job.log(f"! Could not restore the previous EQ: {exc}")
            job.state = final

    def _wait_for_phone(self, job: Job) -> None:
        if self.link.connected:
            return
        job.prompt = "Open the RoomEQ page on your phone and tap 'Start microphone'."
        job.state = "waiting"
        job.log(job.prompt)
        while not self.link.connected:
            if job.cancel_flag.is_set():
                raise Cancelled()
            time.sleep(0.2)
        job.prompt = None
        job.state = "running"
        job.log(f"Phone connected ({self.link.info.sample_rate:.0f} Hz).")

    def _level_check(self, job: Job, rig: EngineRig) -> None:
        rig.prepare_position(0, 1)
        for attempt in range(3):
            job.log("Level check: playing 2.5 s of noise ...")
            rep = level_check(rig)
            job.log(f"  background {rep.background_dbfs:.0f} dBFS, test noise {rep.signal_dbfs:.0f} dBFS, "
                    f"peak {rep.peak_dbfs:.0f} dBFS. {rep.advice}")
            self.link.post({"type": "status", "text": f"Level check: {rep.advice}"})
            if rep.ok or attempt == 2:
                return
            job.wait_ready("Adjust the speaker volume, then press Continue to check again.")

    def _measured_bass_excess(self, f: FloatArray, measured: FloatArray, f_lo: float, nulls) -> float:
        from .dsp.solver import bass_excess_db

        tgt = self.cfg.target.evaluate(f)
        band = (f >= self.cfg.solver.align_band[0]) & (f <= self.cfg.solver.align_band[1])
        aligned = measured - float(np.median(measured[band] - tgt[band]))
        return bass_excess_db(f, aligned, tgt, f_lo, nulls)

    def _save_set(self, label: str, ms, filters=(), preamp: float = 0.0) -> None:
        from .session import save_measurement

        cal = self._calibration()
        info = {"filters": [f.to_dict() for f in filters], "preamp_db": preamp,
                "volume_db": self.core.volume_db, "phone_rate": self.link.info.sample_rate if self.link.info else None}
        save_measurement(ms, cal.apply(ms.freqs, ms.average_db), cal, info, label)

    @staticmethod
    def _preset_name(kind: str) -> str:
        return f"{kind} {datetime.now():%m-%d %H%M}"

    def _run_measure(self, job: Job, positions: int, repeats: int, _iterations: int) -> None:
        rig = EngineRig(self, job)
        self._level_check(job, rig)
        rig.skip_next_prepare = True
        cal = self._calibration()
        warnings = [FLAT_WARNING] if cal.is_flat else []
        ts = make_test_signal(self._sweep())
        job.log(f"Measuring {positions} position(s) x {repeats} sweep(s) with the EQ bypassed ...")
        ms = measure_set(rig, ts, positions, repeats, (), 0.0, self.cfg.analysis(), job.log)
        self._save_set("measure (no EQ)", ms)
        warnings += ms.warnings
        meas = cal.apply(ms.freqs, ms.average_db)
        res = solve(ms.freqs, meas, self.cfg.target, self._solver())
        warnings += res.warnings
        name = self._preset_name("measure")
        self.core.set_bypass(False)
        self.core.set_eq(res.filters, res.preamp_db, name)
        self._finish(job, name, res.filters, ms.freqs, meas, res.predicted_db, "predicted",
                     res.target_db, res.rms_before, res.rms_after, warnings)

    def _run_autotune(self, job: Job, positions: int, repeats: int, iterations: int) -> None:
        rig = EngineRig(self, job)
        self._level_check(job, rig)
        rig.skip_next_prepare = True
        rep = autotune(rig, self.cfg.target, self._solver(), self._calibration(), iterations, positions, repeats,
                       self._sweep(), self.cfg.analysis(), log=job.log,
                       on_measured=lambda label, ms, flt, pre: self._save_set(label, ms, flt, pre))
        best = rep.best
        if best.index == 0:
            # Nothing measured better than no EQ: don't install (or save) a flat EQ, put back whatever
            # was running before, and say so plainly.
            filters, pre, name, byp = job.saved_eq
            if filters:
                self.core.set_eq(filters, pre, name)
            self.core.set_bypass(byp)
            tried = [h for h in rep.history[1:]]
            detail = "; ".join(f"round {h.index}: {h.rms_before_db:.1f} -> {h.rms_db:.1f} dB" for h in tried)
            msg = ("Auto-tune could not confirm an improvement at the same positions"
                   + (f" ({detail})" if detail else "") + ". Kept your previous EQ.")
            job.log(msg)
            job.result = {"summary": msg, "warnings": rep.warnings, "preset": None}
            return
        warnings = [w for w in rep.warnings if not w.startswith("Even with the EQ, the bass")]
        excess = self._measured_bass_excess(rep.freqs, best.measured_db, rep.initial.f_lo, rep.initial.null_mask)
        if excess > 4.0:
            warnings.insert(0, f"Measured with the EQ, the bass (40-120 Hz) is still about {excess:.0f} dB above "
                               f"the target: turn the subwoofer / bass knob down by roughly {excess:.0f} dB on the "
                               "speaker itself, then run auto-tune again.")
            job.log(f"! {warnings[0]}")
        rep.warnings[:] = warnings
        name = self._preset_name("autotune")
        self.core.set_bypass(False)
        self.core.set_eq(best.filters, best.preamp_db, name)
        self._finish(job, name, best.filters, rep.freqs, rep.before_db, best.measured_db, "measured",
                     rep.target_db, rep.rms_before, rep.rms_after, rep.warnings)

    def _run_verify(self, job: Job, _positions: int, _repeats: int, _iterations: int) -> None:
        """Single position, phone not moved: is the EQ applied as designed, and is the speaker linear?"""
        from dataclasses import replace

        from .dsp.analysis import analyze_recording
        from .dsp.verify import verify_report

        rig = EngineRig(self, job)
        self._level_check(job, rig)
        rig.skip_next_prepare = True
        sweep = self._sweep()
        ts = make_test_signal(sweep)
        ts_quiet = make_test_signal(replace(sweep, level_dbfs=sweep.level_dbfs - 10.0))
        an = self.cfg.analysis()

        def sweep_once(label: str, sig, eq=(), pre=0.0):
            job.log(f"  {label} ...")
            rec, fs = rig.record(sig.signal, 0, eq, pre)
            m = analyze_recording(rec, fs, sig, an)
            for w in m.quality.warnings:
                job.log(f"    ! {w}")
            return m

        job.log("Keep the phone exactly where it is for all four sweeps.")
        filters, pre = list(self.core.filters), self.core.preamp_db
        a = sweep_once("1/4 EQ bypassed", ts)
        if not filters:
            cal = self._calibration()
            res = solve(a.freqs, cal.apply(a.freqs, a.db), self.cfg.target, self._solver())
            filters, pre = res.filters, res.preamp_db
            job.log(f"  (no EQ was active: testing a fresh {len(filters)}-filter EQ solved from this sweep)")
        b = sweep_once("2/4 through the EQ", ts, filters, pre)
        c = sweep_once("3/4 EQ bypassed, 10 dB quieter", ts_quiet)
        d = sweep_once("4/4 EQ bypassed again (repeatability)", ts)
        rep = verify_report(a, b, c, d, filters, pre, float(self.core.s.fs_out))
        for line in rep.lines:
            job.log(line)
        self._save_verify(rep, a, b, c, d, filters, pre)
        job.result = {"summary": rep.verdict, "warnings": [], "preset": None}
        # leave the EQ as it was before Verify (the wrapper restores it unless we say otherwise)
        sf, sp, sn, sb = job.saved_eq
        if sf:
            self.core.set_eq(sf, sp, sn)
        else:
            self.core.set_eq([], 0.0, sn)
        self.core.set_bypass(sb)

    def _save_verify(self, rep, a, b, c, d, filters, pre) -> None:
        import json

        from .presets import home

        dd = home() / "measurements"
        dd.mkdir(parents=True, exist_ok=True)
        path = dd / f"{datetime.now():%Y%m%d-%H%M%S}-verify.json"
        r = lambda x: np.round(np.asarray(x), 3).tolist()
        path.write_text(json.dumps({
            "label": "verify", "created": datetime.now().isoformat(timespec="seconds"),
            "filters": [f.to_dict() for f in filters], "preamp_db": pre, "volume_db": self.core.volume_db,
            "freqs": r(a.freqs), "bypass_db": r(a.db), "through_eq_db": r(b.db), "quiet_db": r(c.db),
            "repeat_db": r(d.db), "verdict": rep.verdict, "numbers": rep.numbers,
        }), encoding="utf-8")

    def _finish(self, job: Job, name: str, filters: list[Filter], freqs: FloatArray,
                before: FloatArray, after: FloatArray, after_kind: str, target: FloatArray, rms_b: float,
                rms_a: float, warnings: list[str]) -> None:
        notes = list(dict.fromkeys(warnings))
        p = Preset(name, list(filters), self.core.preamp_db, rms_before=rms_b, rms_after=rms_a, notes=notes,
                   response={"freqs": np.asarray(freqs).tolist(), "before": np.asarray(before).tolist(),
                             "after": np.asarray(after).tolist(), "target": np.asarray(target).tolist(),
                             "after_kind": after_kind})
        path = save_preset(p)
        self.set_curves_from_preset(p)
        verb = "measured" if after_kind == "measured" else "predicted"
        summary = (f"{len(filters)} filters, preamp {self.core.preamp_db:+.1f} dB\n"
                   f"RMS error {rms_b:.1f} → {rms_a:.1f} dB ({verb})")
        job.result = {"summary": summary, "warnings": notes, "preset": name, "path": str(path),
                      "finished": datetime.now().isoformat(timespec="seconds")}
        job.log(summary)
        job.log(f"Applied and saved as preset '{name}'.")

    def _sweep(self):
        from dataclasses import replace
        return replace(self.cfg.sweep(), fs=self.core.s.fs_out)

    def _solver(self):
        from dataclasses import replace
        return replace(self.cfg.solver, fs=float(self.core.s.fs_out))
