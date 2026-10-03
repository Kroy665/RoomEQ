"""``roomeq run``: the real-time EQ with a keyboard-driven terminal UI."""

from __future__ import annotations

import select
import shutil
import sys
import time
from contextlib import contextmanager

from ..config import Config
from ..presets import Preset, load_preset
from .engine import AudioEngine, SystemOutput

KEYS = "b = bypass (A/B)   space = PANIC   +/- = volume   r = reload preset   q = quit"


@contextmanager
def _cbreak():
    if not sys.stdin.isatty():
        yield False
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _status_line(st: dict, latency_ms: float) -> str:
    import shutil

    eq = {"bypassed": "BYPASS", "PANIC": "PANIC!"}.get(st["eq"], "EQ on" if st["eq"] == "on" else "flat")
    glitches = st["underruns"] + st["overruns"]
    line = (f"{eq:<6} {st['preset'][:12]} | vol {st['volume_db']:+.0f} dB | in {st['peak_in_dbfs']:5.1f} "
            f"out {st['peak_out_dbfs']:5.1f} dBFS | limit {st['limiter_gr_db']:.1f} dB | {latency_ms:.0f} ms | "
            f"drift {st['drift_ppm']:+.0f} ppm | glitches {glitches}")
    width = max(20, shutil.get_terminal_size((100, 20)).columns - 1)
    return "\r" + line[:width].ljust(width)


def _load(name: str | None) -> Preset | None:
    try:
        return load_preset(name)
    except FileNotFoundError:
        return None


def run_engine(cfg: Config, preset_name: str | None, volume_db: float, switch_output: bool | None,
               blocksize: int | None = None, seconds: float | None = None, server: bool = True,
               tunnel: bool = False, job: dict | None = None, open_browser: bool = False) -> int:
    sys.setswitchinterval(0.001)                   # hand the GIL to the audio callbacks quickly
    engine = AudioEngine(cfg.audio, blocksize)
    core = engine.core
    preset = _load(preset_name)
    if preset is None and preset_name:
        print(f"error: preset '{preset_name}' not found", file=sys.stderr)
        return 1
    if preset is not None:
        pre = core.set_eq(preset.filters, preset.preamp_db, preset.name)
        print(f"Preset '{preset.name}': {len(preset.filters)} filters, preamp {pre:+.1f} dB")
    else:
        print("No preset saved yet - running flat (measure from the dashboard or with `roomeq autotune`).")
    core.set_volume(volume_db)

    sysout = SystemOutput()
    info = engine.start()
    print(f"Input  : {info.input_name} @ {info.fs_in} Hz")
    print(f"Output : {info.output_name} @ {info.fs_out} Hz"
          + ("  (resampling)" if info.fs_in != info.fs_out else ""))
    print(f"Block  : {info.blocksize} frames, added latency {info.latency_ms:.1f} ms "
          f"(buffering + 1.5 ms limiter look-ahead + device latency)")
    want_switch = sysout.available if switch_output is None else switch_output
    if want_switch and sysout.available:
        if sysout.switch_to(engine.dev_in.name, engine.dev_out.name):
            print(f"System output switched to {engine.dev_in.name} (restored to {sysout.previous} on exit)")
    elif want_switch:
        print("Tip: `brew install switchaudio-osx` lets RoomEQ switch the system output automatically.")
    if not (want_switch and sysout.available):
        print(f"Make sure System Settings > Sound > Output is set to {engine.dev_in.name}.")

    srv = ctl = None
    try:
        if server:
            from ..jobs import Controller
            from ..session import print_qr, start_server

            srv = start_server(cfg, tunnel, print, controller_factory=lambda link: Controller(
                cfg, core, link, out_latency_s=lambda: engine.output_latency_s(), latency_ms=engine.latency_ms,
                devices={"input": info.input_name, "output": info.output_name,
                         "fs_in": info.fs_in, "fs_out": info.fs_out}))
            ctl = srv.controller
            if preset is not None:
                ctl.set_curves_from_preset(preset)
            dash = f"http://localhost:{cfg.http_port}"
            print(f"\nDashboard: {dash}   (phone measurement page: {srv.setup_url})")
            if open_browser:
                import webbrowser
                webbrowser.open(dash)
            if job is not None:
                print("\nScan with your iPhone and tap 'Start microphone':")
                print_qr(srv.setup_url)
                ctl.start_job(**job)
    except Exception as exc:
        print(f"! Dashboard not available: {exc}")
        srv = ctl = None
    print(f"\n{KEYS}" + ("   Enter = continue a measurement step" if ctl else "") + "\n")

    t_end = None if seconds is None else time.monotonic() + seconds
    last = 0.0
    shown = 0
    try:
        with _cbreak() as interactive:
            while t_end is None or time.monotonic() < t_end:
                key = ""
                if interactive:
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r:
                        key = sys.stdin.read(1)
                else:
                    time.sleep(0.1)
                if key in ("q", "Q", "\x04"):
                    break
                if key in ("b", "B"):
                    core.set_bypass(not core.bypassed)
                elif key in (" ", "p", "P"):
                    core.clear_panic() if core.panicked else core.panic()
                elif key in ("+", "="):
                    core.set_volume(core.volume_db + 1.0)
                elif key in ("-", "_"):
                    core.set_volume(core.volume_db - 1.0)
                elif key in ("\n", "\r") and ctl is not None:
                    ctl.continue_job()
                elif key in ("r", "R"):
                    p = _load(core.preset_name if core.preset_name else None)
                    if p is not None:
                        if ctl is not None:
                            ctl.load_preset(p.name)
                        else:
                            core.set_eq(p.filters, p.preamp_db, p.name)
                # mirror measurement progress in the terminal, above the status line
                if ctl is not None and ctl.job is not None:
                    lines = ctl.job.lines
                    if shown > len(lines):
                        shown = 0
                    if len(lines) > shown:
                        width = shutil.get_terminal_size((100, 20)).columns - 1
                        sys.stdout.write("\r" + " " * width + "\r" + "\n".join(lines[shown:]) + "\n")
                        shown = len(lines)
                        last = 0.0
                now = time.monotonic()
                if now - last >= 0.5:
                    last = now
                    sys.stdout.write(_status_line(core.take_stats(), engine.latency_ms()))
                    sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if ctl is not None:
            ctl.cancel_job()
        core.set_volume(-60.0)                     # fade out instead of cutting off
        time.sleep(0.4)
        sysout.restore()
        if srv is not None:
            srv.stop()
        engine.stop()
        core._stats_time = 0.0                     # fresh numbers, not the 250 ms cache
        st = core.take_stats()
        print(f"\nStopped. Glitches: {st['underruns'] + st['overruns']} buffer, {engine.xruns} device; "
              f"worst callback {st['callback_max_ms']:.2f} ms of {1e3 * info.blocksize / info.fs_out:.1f} ms budget.")
    return 0
