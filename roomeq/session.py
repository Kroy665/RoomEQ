"""Interactive measurement session used by ``roomeq measure`` (and later the dashboard)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Config
from .dsp.biquad import response_db
from .dsp.calibration import FLAT_WARNING, MicCalibration, flat, load_calibration
from .dsp.solver import solve
from .dsp.sweep import make_test_signal
from .pipeline import MeasurementSet, measure_set
from .presets import Preset, format_table, home, load_preset, save_preset
from .rigs import PhoneRig, level_check, terminal_or_phone_ready
from .server.link import PhoneLink

Log = Callable[[str], None]

LIMITS = ("Phone mics are accurate for bass and low mids but rough above a few kHz, so RoomEQ corrects mainly "
          "below 500 Hz and nothing above 4 kHz. EQ cannot fix speaker placement or fill room nulls.")


@dataclass
class Server:
    link: PhoneLink
    thread: object
    urls: object
    tunnel: object | None
    controller: object | None = None

    @property
    def setup_url(self) -> str:
        """URL to open on the phone: the tunnel, or the plain-HTTP page that redirects to HTTPS once trusted."""
        return self.urls.tunnel or self.urls.http[0]

    def stop(self) -> None:
        if self.tunnel is not None:
            self.tunnel.stop()
        self.thread.stop()


def start_server(cfg: Config, use_tunnel: bool = False, log: Log = print,
                 controller_factory: Callable | None = None) -> Server:
    from .server.app import ServerThread, ServerUrls, create_app
    from .server.certs import ensure_certs, lan_ips, mdns_name

    certs = ensure_certs()
    https_port, http_port = cfg.port, cfg.http_port
    ips = lan_ips()
    urls = ServerUrls(https=[f"https://{ip}:{https_port}" for ip in ips] + [f"https://{mdns_name()}:{https_port}"],
                      http=[f"http://{ip}:{http_port}" for ip in ips] + [f"http://localhost:{http_port}"])
    link = PhoneLink()
    controller = controller_factory(link) if controller_factory else None
    thread = ServerThread(create_app(link, urls, certs, controller), https_port, http_port, certs)
    thread.start()
    thread.wait_started()
    tunnel = None
    if use_tunnel:
        from .server.tunnel import start_tunnel
        log("Starting tunnel ...")
        tunnel = start_tunnel(http_port)
        urls.tunnel = tunnel.url
    return Server(link, thread, urls, tunnel, controller)


def print_qr(url: str) -> None:
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        pass


def save_measurement(ms: MeasurementSet, calibrated_db: np.ndarray, cal: MicCalibration, info: dict,
                     label: str = "") -> Path:
    d = home() / "measurements"
    d.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    slug = "".join(ch if ch.isalnum() else "-" for ch in label).strip("-")
    path = d / (f"{stamp}-{slug}.json" if slug else f"{stamp}.json")
    n = 2
    while path.exists():
        path = d / f"{stamp}-{slug}-{n}.json"
        n += 1
    data = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "info": info,
        "calibration": cal.name,
        "freqs": np.round(ms.freqs, 3).tolist(),
        "average_db": np.round(ms.average_db, 3).tolist(),
        "calibrated_db": np.round(calibrated_db, 3).tolist(),
        "sweeps": [
            {"position": p, "repeat": r, "db": np.round(m.db, 3).tolist(),
             "peak_dbfs": round(m.quality.peak_dbfs, 2), "snr_bass_db": round(m.quality.snr_bass_db, 1),
             "snr_mid_db": round(m.quality.snr_mid_db, 1), "drift_ppm": round(m.quality.drift_ppm, 2),
             "warnings": m.quality.warnings}
            for p, row in enumerate(ms.measurements) for r, m in enumerate(row)
        ],
        "warnings": ms.warnings,
        "reliable": ms.reliable,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def run_measure(cfg: Config, *, positions: int, repeats: int, mic: str | None, use_tunnel: bool,
                through_preset: str | None, calibration: str | None, save_name: str,
                do_level_check: bool = True, log: Log = print) -> Preset:
    from .audio.devices import LocalRecorder, Player

    player = Player(cfg.audio.output_device)
    log(f"Output device: {player.device.name}")
    fs = cfg.audio.samplerate
    server = None
    recorder = None
    try:
        if mic:
            from .rigs import LocalSource
            recorder = LocalRecorder(mic)
            recorder.start()
            log(f"Recording from local input: {recorder.device.name} ({recorder.fs} Hz)")
            source = LocalSource(recorder)
            rig = PhoneRig(source, player, fs, log, wait_ready=terminal_or_phone_ready(None))
            info: dict = {"mic": recorder.device.name, "sample_rate": recorder.fs}
        else:
            server = start_server(cfg, use_tunnel, log)
            link = server.link
            log("\nOn your iPhone, scan this code or open:\n")
            log(f"    {server.setup_url}\n")
            print_qr(server.setup_url)
            if not use_tunnel:
                log("First time only: that page walks you through trusting the RoomEQ certificate.\n"
                    f"Already trusted? You can also open {server.urls.https[0]} directly.")
            log("\nThen tap 'Start microphone'. Waiting for the phone ...")
            pinfo = link.wait_connected()
            log(f"Phone connected: {pinfo.sample_rate:.0f} Hz via {pinfo.transport}")
            processing = [k for k in ("echoCancellation", "noiseSuppression", "autoGainControl")
                          if pinfo.settings.get(k) is True]
            if processing:
                log(f"! The phone browser kept {', '.join(processing)} on; results may be less accurate.")
            rig = PhoneRig(link, player, fs, log, notify=link.post, wait_ready=terminal_or_phone_ready(link))
            info = {"mic": "phone (web)", "sample_rate": pinfo.sample_rate, "user_agent": pinfo.user_agent,
                    "settings": pinfo.settings, "transport": pinfo.transport}

        if do_level_check:
            rig.prepare_position(0, positions)
            while True:
                log("Level check: playing 2.5 s of noise ...")
                rep = level_check(rig)
                log(f"  background {rep.background_dbfs:.0f} dBFS, test noise {rep.signal_dbfs:.0f} dBFS, "
                    f"peak {rep.peak_dbfs:.0f} dBFS. {rep.advice}")
                rig.notify({"type": "status", "text": f"Level check: {rep.advice}"})
                if rep.ok:
                    break
                ans = input("Adjust the volume, then press Enter to re-check (or type 'go' to continue anyway): ")
                if ans.strip().lower() == "go":
                    break

        eq, preamp = [], 0.0
        if through_preset is not None:
            p = load_preset(through_preset or None)
            eq, preamp = list(p.filters), p.preamp_db
            log(f"Measuring through preset '{p.name}' ({len(eq)} filters, preamp {preamp:+.1f} dB)")

        cal = load_calibration(calibration) if calibration else flat()
        warnings: list[str] = []
        if cal.is_flat:
            warnings.append(FLAT_WARNING)
            log(f"! {FLAT_WARNING}")

        ts = make_test_signal(cfg.sweep())
        log(f"\nMeasuring {positions} position(s) x {repeats} sweep(s), {ts.duration_s:.0f} s each.")

        rig.skip_next_prepare = do_level_check             # position 1 was set up for the level check
        ms = measure_set(rig, ts, positions, repeats, eq, preamp, cfg.analysis(), log)
        warnings += ms.warnings
        meas = cal.apply(ms.freqs, ms.average_db)
        mpath = save_measurement(ms, meas, cal, info)
        log(f"\nSaved measurement to {mpath}")

        room = meas - response_db(eq, ms.freqs, cfg.solver.fs) if eq else meas
        res = solve(ms.freqs, room, cfg.target, cfg.solver, initial=eq)
        warnings += res.warnings
        if not ms.reliable:
            warnings.insert(0, "Some sweeps were unreliable (see above). Consider measuring again.")
        preset = Preset(save_name, res.filters, res.preamp_db, rms_before=res.rms_before, rms_after=res.rms_after,
                        notes=list(dict.fromkeys(warnings)),
                        response={"freqs": res.freqs.tolist(), "before": room.tolist(),
                                  "after": res.predicted_db.tolist(), "target": res.target_db.tolist()})
        path = save_preset(preset)
        log("\n" + format_table(preset) + "  (predicted)")
        if preset.notes:
            log("\nNotes:")
            for w in preset.notes:
                log(f"  - {w}")
        log(f"\n{LIMITS}\nSaved preset '{save_name}' to {path}")
        rig.notify({"type": "result", "done": True, "warnings": preset.notes,
                    "text": f"{len(res.filters)} filters, preamp {res.preamp_db:+.1f} dB\n"
                            f"RMS error {res.rms_before:.1f} → {res.rms_after:.1f} dB (predicted)"})
        return preset
    finally:
        if recorder is not None:
            recorder.stop()
        if server is not None:
            import time
            time.sleep(0.5)                                  # let the final message reach the phone
            server.stop()
