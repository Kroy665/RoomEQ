"""``roomeq demo``: the whole app without audio hardware.

The real engine core, solver, measurement pipeline, server and dashboard run unchanged. Only the
outside world is simulated: a music source feeding the engine, the room and speakers (the same model
as ``roomeq simulate``) and a phone that streams what it "hears" and taps Ready on its own. Time runs
faster while a measurement job is active so a full auto-tune takes seconds, not minutes.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
from scipy import signal as sps

from .config import Config
from .dsp.biquad import filters_to_sos
from .engine.core import EngineCore, EngineSettings
from .jobs import Controller
from .server.link import HEADER, PhoneLink
from .dsp.biquad import Filter
from .sim.room import RoomModel, _vary

# A typical small living room with a 2.1 system: a big bass build-up from room modes, a peak around
# 250 Hz and a sub/satellite hand-over dip. Each listening position gets its own variation.
DEMO_ROOM = RoomModel(
    modes=(Filter.peak(52, 7.0, 5.0), Filter.peak(70, 10.0, 5.0), Filter.peak(92, 8.0, 4.0),
           Filter.peak(245, 7.0, 3.0)),
    null=Filter.peak(150, -12.0, 6.0),
)

DEMO_STREAM_ID = 0xD3D0


class DemoWorld(threading.Thread):
    """Music in, room + phone out, paced in (accelerated) real time."""

    def __init__(self, core: EngineCore, link: PhoneLink, ctl: Controller, job_speed: float = 12.0,
                 ready_delay_s: float = 1.2, room: RoomModel | None = None, seed: int = 7):
        super().__init__(daemon=True, name="roomeq-demo-world")
        self.core, self.link, self.ctl = core, link, ctl
        self.job_speed = job_speed
        self.ready_delay_s = ready_delay_s
        self.stop_flag = threading.Event()
        fs = core.s.fs_out
        room = room or DEMO_ROOM
        hp = sps.butter(4, room.hp_hz, "highpass", fs=fs, output="sos")
        self.variants = []
        for pos in range(5):                                  # one room response per listening position
            rng = np.random.default_rng(100 + pos)
            modes = _vary(room.modes, rng, 1.0 if pos else 0.0)
            null = _vary((room.null,), rng, 1.0 if pos else 0.0) if room.null else []
            self.variants.append(np.vstack([filters_to_sos(list(room.speaker) + modes + list(null), fs), hp]))
        self.sos = self.variants[0]
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.rng = np.random.default_rng(seed)
        self.fs = fs
        self._msg_seen = 0

    def _music(self, t0: float, n: int) -> np.ndarray:
        """A slow chord progression with a bass line and soft noise: enough to make the meters move."""
        t = t0 + np.arange(n) / self.fs
        chords = [(110.0, 220.0, 277.2, 329.6), (98.0, 196.0, 246.9, 293.7), (87.3, 174.6, 220.0, 261.6),
                  (98.0, 196.0, 246.9, 311.1)]
        root = chords[int(t0 / 2.0) % len(chords)]
        env = 0.6 + 0.4 * np.abs(np.sin(np.pi * t * 2.0))
        x = sum(np.sin(2 * np.pi * f * t) * (0.5 if i == 0 else 0.22) for i, f in enumerate(root)) * env
        x += 0.03 * self.rng.standard_normal(n)
        return (0.18 * x).astype(np.float32)

    def run(self) -> None:
        blk, fs = 256, self.fs
        out = np.zeros((blk, 2), dtype=np.float32)
        music = np.zeros((blk, 2), dtype=np.float32)
        t, sent = 0.0, 0
        self.link.hello(fs, DEMO_STREAM_ID, "RoomEQ demo phone", {"echoCancellation": False,
                        "noiseSuppression": False, "autoGainControl": False}, "demo")
        waiting_since: float | None = None
        while not self.stop_flag.is_set():
            job = self.ctl.job
            busy = job is not None and job.state in ("running", "waiting")
            m = self._music(t, blk)
            music[:, 0] = m
            music[:, 1] = m
            t += blk / fs
            self.core.process_input(music, now=t)
            self.core.process_output(out, now=t)
            heard, self.zi = sps.sosfilt(self.sos, out[:, 0].astype(float) * 0.4, zi=self.zi)
            heard += 4e-5 * self.rng.standard_normal(blk)
            self.link.ingest(HEADER.pack(sent & 0xFFFFFFFF, DEMO_STREAM_ID) + heard.astype("<f4").tobytes())
            sent += blk
            # the virtual listener moves the phone where the instructions say ...
            for msg in self.link.messages_after(self._msg_seen):
                self._msg_seen = msg["id"]
                if msg.get("type") == "position":
                    self.sos = self.variants[int(msg["index"]) % len(self.variants)]
            # ... and taps "Ready" a moment after being asked
            if job is not None and job.state == "waiting" and self.link.connected:
                now = time.monotonic()
                waiting_since = waiting_since or now
                if now - waiting_since > self.ready_delay_s:
                    self.link.event({"type": "ready"})
                    waiting_since = None
            else:
                waiting_since = None
            time.sleep(blk / fs / (self.job_speed if busy else 1.0))


def run_demo(cfg: Config, port: int | None = None, preset: str | None = None, job_speed: float = 12.0,
             seconds: float | None = None, open_browser: bool = False) -> int:
    from .session import start_server

    if "ROOMEQ_HOME" not in os.environ:
        os.environ["ROOMEQ_HOME"] = str(Path.home() / ".roomeq" / "demo")   # never touch real presets
    if port:
        cfg.http_port = port
        cfg.port = port + 363
    cfg.measurement.sweep_seconds = 6.0
    core = EngineCore(EngineSettings())
    holder: dict = {}

    def factory(link: PhoneLink) -> Controller:
        ctl = Controller(cfg, core, link, devices={"input": "Demo music", "output": "Simulated room",
                                                   "fs_in": 48000, "fs_out": 48000})
        ctl.demo = True
        holder["ctl"] = ctl
        return ctl

    srv = start_server(cfg, controller_factory=factory)
    ctl = holder["ctl"]
    if preset:
        try:
            ctl.load_preset(preset)
        except FileNotFoundError:
            print(f"! preset '{preset}' not found in {os.environ['ROOMEQ_HOME']}")
    world = DemoWorld(core, srv.link, ctl, job_speed=job_speed)
    world.start()
    url = f"http://localhost:{cfg.http_port}"
    print(f"RoomEQ demo (simulated room and phone, no audio hardware): {url}")
    print(f"Presets for the demo live in {os.environ['ROOMEQ_HOME']}. Ctrl+C to stop.")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        t_end = None if seconds is None else time.monotonic() + seconds
        while t_end is None or time.monotonic() < t_end:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        world.stop_flag.set()
        world.join(timeout=2)
        srv.stop()
    return 0
