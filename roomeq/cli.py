"""``roomeq`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import config_path, load_config, write_default
from .presets import Preset, format_apo, format_json, format_table, list_presets, load_preset, save_preset

LIMITS_NOTE = ("Limits: a phone mic is accurate for bass and low mids but rough above a few kHz, so RoomEQ "
               "corrects mainly below 500 Hz and not at all above 4 kHz. No EQ can fix speaker placement "
               "or fill room nulls.")


def cmd_simulate(a: argparse.Namespace) -> int:
    from .dsp.calibration import MicCalibration
    from .pipeline import autotune
    from .sim.room import SimConfig, SimulatedRig

    cfg = load_config()
    sim = SimConfig(mac_fs=a.mac_fs, phone_fs=a.phone_fs, drift_ppm=a.drift_ppm, noise_dbfs=a.noise_dbfs,
                    seed=a.seed)
    rig = SimulatedRig(sim)
    cal = None
    if a.calibrated:
        grid = np.geomspace(10, 0.49 * a.mac_fs, 400)
        cal = MicCalibration("simulated iPhone", grid, sim.mic.response_db(grid, a.mac_fs))
    sweep = replace(cfg.sweep(), fs=a.mac_fs)
    solver = replace(cfg.solver, fs=float(a.mac_fs))
    print(f"Simulation: Mac {a.mac_fs} Hz, phone {a.phone_fs} Hz, clock drift {a.drift_ppm:+.0f} ppm, "
          f"noise {a.noise_dbfs:.0f} dBFS\n")
    rep = autotune(rig, cfg.target, solver, cal, a.iterations, a.positions, a.repeats, sweep, log=print)
    preset = Preset("simulation", rep.best.filters, rep.best.preamp_db, rms_before=rep.rms_before,
                    rms_after=rep.rms_after, notes=rep.warnings,
                    response={"freqs": rep.freqs.tolist(), "before": rep.before_db.tolist(),
                              "after": rep.best.measured_db.tolist(), "target": rep.target_db.tolist()})
    print("\n" + format_table(preset))
    if rep.warnings:
        print("\nNotes:")
        for w in rep.warnings:
            print(f"  - {w}")
    print(f"\n{LIMITS_NOTE}")
    if a.save:
        print(f"\nSaved preset to {save_preset(preset)}")
    if a.json:
        print(format_json(preset))
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    p = load_preset(a.preset)
    if a.format == "json":
        print(format_json(p))
    elif a.format == "apo":
        print(format_apo(p))
    else:
        print(format_table(p))
        print("\nParametric text (Equalizer APO / AutoEQ format):\n" + format_apo(p))
        print("\nJSON:\n" + format_json(p))
    return 0


def cmd_presets(_: argparse.Namespace) -> int:
    names = list_presets()
    print("\n".join(names) if names else "no presets saved yet")
    return 0


def cmd_config(a: argparse.Namespace) -> int:
    path = write_default(overwrite=a.reset)
    print(f"Config file: {path}")
    cfg = load_config()
    print(json.dumps({"audio": vars(cfg.audio), "measurement": vars(cfg.measurement)}, indent=2))
    return 0


def cmd_devices(_: argparse.Namespace) -> int:
    from .audio.devices import format_devices, list_devices

    cfg = load_config()
    devs = list_devices()
    print(format_devices(devs))
    if not any("blackhole" in d.name.lower() for d in devs):
        print("\nBlackHole is not installed. For the real-time EQ run:  brew install blackhole-2ch  (then log out/in or reboot)")
    print(f"\nConfigured: input (BlackHole) = \"{cfg.audio.input_device}\", "
          f"output (speakers) = \"{cfg.audio.output_device}\"   [{config_path()}]")
    return 0


def cmd_measure(a: argparse.Namespace) -> int:
    from .session import run_measure

    cfg = load_config()
    m = cfg.measurement
    try:
        run_measure(cfg, positions=a.positions or m.positions, repeats=a.repeats or m.repeats, mic=a.mic,
                    use_tunnel=a.tunnel, through_preset=a.through_eq,
                    calibration=a.calibration or m.calibration_file or None, save_name=a.save,
                    do_level_check=not a.no_level_check)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    return 0


def cmd_solve(a: argparse.Namespace) -> int:
    from .dsp.calibration import load_calibration
    from .dsp.solver import solve
    from .presets import home

    cfg = load_config()
    mdir = home() / "measurements"
    path = Path(a.measurement) if a.measurement else max(mdir.glob("*.json"), default=None)
    if path is None or not path.exists():
        raise FileNotFoundError("no measurement found - run `roomeq measure` first")
    d = json.loads(path.read_text(encoding="utf-8"))
    f = np.array(d["freqs"])
    db = np.array(d["average_db"])
    cal_file = a.calibration or cfg.measurement.calibration_file
    if cal_file:
        db = load_calibration(cal_file).apply(f, db)
    res = solve(f, db, cfg.target, cfg.solver)
    notes = res.warnings if cal_file else ["Mic treated as flat (no calibration)."] + res.warnings
    preset = Preset(a.save, res.filters, res.preamp_db, rms_before=res.rms_before, rms_after=res.rms_after,
                    notes=notes, response={"freqs": f.tolist(), "before": db.tolist(),
                                           "after": res.predicted_db.tolist(), "target": res.target_db.tolist()})
    print(f"Measurement: {path.name}\n")
    print(format_table(preset) + "  (predicted)")
    for w in notes:
        print(f"  - {w}")
    print(f"\nSaved preset to {save_preset(preset)}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from .engine.run import run_engine

    cfg = load_config()
    if a.input:
        cfg.audio.input_device = a.input
    if a.output:
        cfg.audio.output_device = a.output
    if a.blocksize:
        cfg.audio.blocksize = a.blocksize
    switch = None if a.switch_output is None else a.switch_output
    job = None
    if getattr(a, "autotune", False):
        m = cfg.measurement
        job = {"kind": "autotune", "positions": a.positions or m.positions, "repeats": a.repeats or 1,
               "iterations": a.iterations or cfg.iterations}
    elif getattr(a, "verify", False):
        job = {"kind": "verify", "positions": 1, "repeats": 1}
    return run_engine(cfg, a.preset, a.volume, switch, seconds=a.seconds, server=not a.no_dashboard,
                      tunnel=a.tunnel, job=job, open_browser=a.open or job is not None)


def cmd_demo(a: argparse.Namespace) -> int:
    from .demo import run_demo

    return run_demo(load_config(), port=a.port, preset=a.preset, job_speed=a.speed, seconds=a.seconds,
                    open_browser=a.open)


def cmd_cert(a: argparse.Namespace) -> int:
    from .server.certs import ensure_certs, lan_ips, mdns_name

    if a.reset:
        import shutil
        from .presets import home
        shutil.rmtree(home() / "certs", ignore_errors=True)
    paths = ensure_certs()
    print(f"CA certificate (install on the phone): {paths.ca_der}")
    print(f"Server certificate: {paths.cert}  (valid for {', '.join(lan_ips() + [mdns_name()])})")
    if a.reset:
        print("New CA created: remove the old 'RoomEQ Local CA' profile from the phone and install this one.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="roomeq", description="Automatic room EQ using your phone as the mic.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="run the whole measure/solve/verify loop on a virtual room")
    s.add_argument("--iterations", type=int, default=3)
    s.add_argument("--positions", type=int, default=3)
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("--mac-fs", type=int, default=48000)
    s.add_argument("--phone-fs", type=int, default=48000)
    s.add_argument("--drift-ppm", type=float, default=35.0)
    s.add_argument("--noise-dbfs", type=float, default=-72.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--calibrated", action="store_true", help="use the simulated mic's calibration file")
    s.add_argument("--save", action="store_true", help="save the result as preset 'simulation'")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_simulate)

    e = sub.add_parser("export", help="print a preset as a table, parametric text and JSON")
    e.add_argument("preset", nargs="?", help="preset name or path (default: last saved)")
    e.add_argument("--format", choices=["all", "table", "json", "apo"], default="all")
    e.set_defaults(func=cmd_export)

    sub.add_parser("presets", help="list saved presets").set_defaults(func=cmd_presets)
    c = sub.add_parser("config", help=f"create/show the config file ({config_path()})")
    c.add_argument("--reset", action="store_true", help="overwrite with defaults")
    c.set_defaults(func=cmd_config)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)

    m = sub.add_parser("measure", help="measure the room with your phone and compute an EQ preset")
    m.add_argument("--positions", type=int, help="listening positions to average (default from config: 3)")
    m.add_argument("--repeats", type=int, help="sweeps per position (default from config: 2)")
    m.add_argument("--mic", metavar="DEVICE",
                   help='record from a local input instead of the web page, e.g. "iPhone Microphone" (Continuity)')
    m.add_argument("--tunnel", action="store_true", help="serve the phone page through cloudflared/ngrok")
    m.add_argument("--through-eq", nargs="?", const="", metavar="PRESET",
                   help="measure with a preset's EQ applied (default: last saved)")
    m.add_argument("--calibration", metavar="CSV", help="phone mic calibration file")
    m.add_argument("--save", default="room", metavar="NAME", help="preset name to save (default: room)")
    m.add_argument("--no-level-check", action="store_true")
    m.set_defaults(func=cmd_measure)

    so = sub.add_parser("solve", help="re-compute the EQ from a saved measurement (e.g. after changing the target)")
    so.add_argument("measurement", nargs="?", help="measurement JSON (default: latest)")
    so.add_argument("--calibration", metavar="CSV")
    so.add_argument("--save", default="room", metavar="NAME")
    so.set_defaults(func=cmd_solve)

    de = sub.add_parser("demo", help="the full app with a simulated room and phone (no audio hardware)")
    de.add_argument("--port", type=int, help="HTTP port (default from config: 8080)")
    de.add_argument("--preset", help="preset to load at start (from the demo's own preset folder)")
    de.add_argument("--speed", type=float, default=12.0, help="time speed-up while measuring (default 12x)")
    de.add_argument("--open", action="store_true", help="open the dashboard in the browser")
    de.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    de.set_defaults(func=cmd_demo)

    ce = sub.add_parser("cert", help="create/show the HTTPS certificate for the phone page")
    ce.add_argument("--reset", action="store_true", help="create a brand-new CA")
    ce.set_defaults(func=cmd_cert)

    r = sub.add_parser("run", help="start the real-time EQ (system audio via BlackHole -> EQ -> speakers)")
    r.add_argument("preset", nargs="?", help="preset name (default: last saved)")
    r.add_argument("--volume", type=float, default=0.0, help="master trim in dB (<= 0)")
    r.add_argument("--blocksize", type=int, help="frames per callback (default from config: 256)")
    r.add_argument("--input", help="override input device (default: BlackHole 2ch)")
    r.add_argument("--output", help="override output device (default from config)")
    sw = r.add_mutually_exclusive_group()
    sw.add_argument("--switch-output", dest="switch_output", action="store_true", default=None,
                    help="switch the macOS output to BlackHole while running (needs switchaudio-osx)")
    sw.add_argument("--no-switch-output", dest="switch_output", action="store_false")
    r.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    r.add_argument("--no-dashboard", action="store_true", help="engine only, no web server")
    r.add_argument("--tunnel", action="store_true", help="serve the phone page through cloudflared/ngrok")
    r.add_argument("--open", action="store_true", help="open the dashboard in the browser")
    r.set_defaults(func=cmd_run, autotune=False)

    at = sub.add_parser("autotune", help="run the EQ and auto-tune it: measure, apply, re-measure (with dashboard)")
    at.add_argument("preset", nargs="?", help="preset to start from (default: last saved)")
    at.add_argument("--positions", type=int, help="positions per round (default from config: 3)")
    at.add_argument("--repeats", type=int, help="sweeps per position (default 1)")
    at.add_argument("--iterations", type=int, help="measure/apply rounds (default from config: 3)")
    at.add_argument("--volume", type=float, default=0.0)
    at.add_argument("--blocksize", type=int)
    at.add_argument("--input")
    at.add_argument("--output")
    at.add_argument("--tunnel", action="store_true")
    at.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    at.set_defaults(func=cmd_run, autotune=True, switch_output=None, no_dashboard=False, open=True)

    ve = sub.add_parser("verify", help="check that the EQ reaches the speakers as designed and the speaker is "
                                       "level-linear (4 sweeps, phone not moved)")
    ve.add_argument("preset", nargs="?", help="preset to verify (default: last saved)")
    ve.add_argument("--volume", type=float, default=0.0)
    ve.add_argument("--blocksize", type=int)
    ve.add_argument("--input")
    ve.add_argument("--output")
    ve.add_argument("--tunnel", action="store_true")
    ve.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    ve.set_defaults(func=cmd_run, autotune=False, verify=True, switch_output=None, no_dashboard=False, open=True,
                    positions=None, repeats=None, iterations=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
