"""Phase (d): auto-tune through the live engine core, and the dashboard API."""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.signal import sosfilt, sosfilt_zi

from roomeq.config import Config
from roomeq.dsp.biquad import Filter, filters_to_sos
from roomeq.engine.core import EngineCore, EngineSettings
from roomeq.jobs import Controller
from roomeq.server.app import ServerUrls, create_app
from roomeq.server.link import HEADER, PhoneLink

FS = 48000
ROOM = [Filter.peak(65, 9, 5), Filter.peak(120, 6, 3), Filter.peak(300, -4, 2)]


class FakeRig(threading.Thread):
    """Audio device + room + phone: pumps the engine and streams what the 'phone' hears into the link."""

    def __init__(self, core: EngineCore, link: PhoneLink, speed: float = 10.0, dynamic_bass: bool = False):
        super().__init__(daemon=True)
        self.core, self.link, self.speed = core, link, speed
        self.stop = threading.Event()
        self.sos = filters_to_sos(ROOM, FS)
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.rng = np.random.default_rng(0)
        # "dynamic bass": an AGC on the band below 150 Hz, like many small 2.1 systems
        self.dynamic = dynamic_bass
        from scipy.signal import butter
        self.lp = butter(2, 150, "lowpass", fs=FS, output="sos")
        self.hp = butter(2, 150, "highpass", fs=FS, output="sos")
        self.zl = np.zeros((1, 2))
        self.zh = np.zeros((1, 2))
        self.env = 0.05

    def speaker(self, x: np.ndarray) -> np.ndarray:
        if not self.dynamic:
            return x
        lo, self.zl = sosfilt(self.lp, x, zi=self.zl)
        hi, self.zh = sosfilt(self.hp, x, zi=self.zh)
        self.env += 0.05 * (np.sqrt(np.mean(lo ** 2)) - self.env)
        gain = float(np.clip((0.05 / max(self.env, 1e-6)) ** 0.7, 0.5, 4.0))
        return lo * gain + hi

    def run(self) -> None:
        blk = 256
        music = np.zeros((blk, 2), dtype=np.float32)
        out = np.zeros((blk, 2), dtype=np.float32)
        t, sent = 0.0, 0
        self.link.hello(FS, 1)
        while not self.stop.is_set():
            t += blk / FS
            self.core.process_input(music, now=t)
            self.core.process_output(out, now=t)
            heard, self.zi = sosfilt(self.sos, self.speaker(out[:, 0].astype(float)) * 0.5, zi=self.zi)
            heard = heard + 3e-5 * self.rng.standard_normal(blk)
            self.link.ingest(HEADER.pack(sent, 1) + heard.astype("<f4").tobytes())
            sent += blk
            time.sleep(blk / FS / self.speed)


def make_controller(tmp_path, monkeypatch, sweep_s=3.0):
    monkeypatch.setenv("ROOMEQ_HOME", str(tmp_path))
    cfg = Config()
    cfg.measurement.sweep_seconds = sweep_s
    core = EngineCore(EngineSettings())
    core.gst[0] = 1.0
    link = PhoneLink()
    return cfg, core, link, Controller(cfg, core, link)


def drive(ctl: Controller, timeout: float = 120.0) -> None:
    """Press 'Continue' whenever the job waits, until it finishes."""
    deadline = time.monotonic() + timeout
    while ctl.job.state in ("running", "waiting"):
        assert time.monotonic() < deadline, ctl.job.lines[-20:]
        if ctl.job.state == "waiting":
            ctl.continue_job()
        time.sleep(0.05)


def test_autotune_through_live_engine(tmp_path, monkeypatch):
    cfg, core, link, ctl = make_controller(tmp_path, monkeypatch)
    rig = FakeRig(core, link)
    rig.start()
    try:
        ctl.start_job("autotune", positions=1, repeats=1, iterations=2)
        drive(ctl)
    finally:
        rig.stop.set()
        rig.join(timeout=5)
    job = ctl.job
    assert job.state == "done", "\n".join(job.lines[-30:])
    assert core.filters and core.preset_name.startswith("autotune")
    measured = sorted((tmp_path / "measurements").glob("*autotune-round*.json"))
    assert len(measured) == 5                                   # round 0 + (EQ off, EQ on) x 2 rounds
    assert not core.mute_input and core.inj_st[0] == 0           # music back, no injection left
    # the room's +9 dB mode at 65 Hz got a substantial cut
    assert any(abs(np.log2(f.freq / 65)) < 0.25 and f.gain_db < -4 for f in core.filters), core.filters
    c = ctl.curves_payload()
    assert c["after"] is not None, "auto-tune should report a *measured* after-curve"
    assert c["rms_after"] < 0.6 * c["rms_before"]
    assert any("same positions, RMS error without EQ" in ln for ln in job.lines)


def test_cancel_restores_previous_eq(tmp_path, monkeypatch):
    cfg, core, link, ctl = make_controller(tmp_path, monkeypatch)
    rig = FakeRig(core, link)
    rig.start()
    try:
        core.set_eq([Filter.peak(100, -3, 2)], name="mine")
        ctl.start_job("measure", positions=1, repeats=1)
        deadline = time.monotonic() + 10
        while ctl.job.state != "waiting":                   # first position prompt
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert core.mute_input                              # music muted during the session
        ctl.cancel_job()
        while ctl.job.state in ("running", "waiting"):
            time.sleep(0.02)
    finally:
        rig.stop.set()
        rig.join(timeout=5)
    assert ctl.job.state == "cancelled"
    assert core.filters == [Filter.peak(100, -3, 2)] and core.preset_name == "mine"
    assert not core.mute_input


def test_dashboard_api(tmp_path, monkeypatch):
    cfg, core, link, ctl = make_controller(tmp_path, monkeypatch)
    app = create_app(link, ServerUrls(https=["https://x:8443"], http=["http://x:8080"]), None, ctl)
    c = TestClient(app, base_url="http://localhost")
    assert "RoomEQ Dashboard" in c.get("/").text
    assert "Measure your room" in TestClient(app, base_url="http://192.168.1.3:8080").get("/").text

    s = c.get("/api/state").json()
    assert s["preset"] == "flat" and s["phone"]["connected"] is False and s["phone_url"] == "http://x:8080"

    r = c.post("/api/eq", json={"filters": [{"type": "peak", "freq": 80, "gain_db": -6, "q": 4},
                                            {"type": "peak", "freq": 300, "gain_db": 2, "q": 1}]})
    assert r.status_code == 200 and r.json()["preamp_db"] < -1.9
    assert c.post("/api/eq", json={"filters": [{"type": "peak", "freq": 80, "gain_db": 40, "q": 4}]}).status_code == 400
    assert len(core.filters) == 2                                  # unsafe edit was not applied

    assert c.post("/api/volume", json={"db": -12}).json()["volume_db"] == -12
    c.post("/api/bypass", json={"on": True})
    assert core.bypassed
    assert c.post("/api/panic", json={}).json()["panicked"] is True
    assert c.post("/api/panic", json={}).json()["panicked"] is False

    assert c.post("/api/presets/save", json={"name": "test one"}).status_code == 200
    assert {"id": "test_one", "name": "test one"} in c.get("/api/presets").json()["presets"]
    core.set_eq([], name="flat")
    assert c.post("/api/presets/load", json={"name": "test_one"}).status_code == 200
    assert len(core.filters) == 2

    curves = c.get("/api/curves").json()
    assert len(curves["eq"]) == len(curves["freqs"]) and curves["before"] is None
    assert c.get("/api/qr.svg").headers["content-type"].startswith("image/svg")
    assert c.post("/api/job/start", json={"kind": "nope"}).status_code == 400

    remote = TestClient(app, base_url="http://192.168.1.3:8080", client=("192.168.1.50", 5000))
    assert remote.post("/api/volume", json={"db": 0}).status_code == 403
    assert remote.get("/api/state").status_code == 200


def run_job(ctl, rig, kind, **kw):
    rig.start()
    try:
        ctl.start_job(kind, kw.get("positions", 1), kw.get("repeats", 1), kw.get("iterations", 2))
        drive(ctl)
    finally:
        rig.stop.set()
        rig.join(timeout=5)
    return ctl.job


@pytest.mark.parametrize("dynamic", [False, True])
def test_verify(tmp_path, monkeypatch, dynamic):
    cfg, core, link, ctl = make_controller(tmp_path, monkeypatch)
    mine = [Filter.peak(65, -8, 5), Filter.peak(120, -5, 3)]
    core.set_eq(mine, name="mine")
    job = run_job(ctl, FakeRig(core, link, dynamic_bass=dynamic), "verify")
    assert job.state == "done", "\n".join(job.lines[-30:])
    summary = job.result["summary"]
    if dynamic:
        assert summary.startswith("PROBLEM") and "changes its own frequency response with volume" in summary
    else:
        assert summary.startswith("PASS"), "\n".join(job.lines[-35:])
    assert core.filters == mine and core.preset_name == "mine"        # verify leaves the EQ alone
    assert list((tmp_path / "measurements").glob("*-verify.json"))


def test_failed_autotune_keeps_previous_eq(tmp_path, monkeypatch):
    import roomeq.pipeline as pl

    real = pl.solve

    def harmful(*a, **k):                                     # a solver that makes things worse
        r = real(*a, **k)
        r.filters = [Filter.peak(800, 12, 1.0)]
        r.preamp_db = -12.5
        return r

    monkeypatch.setattr(pl, "solve", harmful)
    cfg, core, link, ctl = make_controller(tmp_path, monkeypatch)
    mine = [Filter.peak(100, -3, 2)]
    core.set_eq(mine, name="mine")
    job = run_job(ctl, FakeRig(core, link), "autotune")
    assert job.state == "done"
    assert "could not confirm an improvement" in job.result["summary"]
    assert core.filters == mine and core.preset_name == "mine"
    assert not list((tmp_path / "presets").glob("autotune*.json"))     # no flat/bad preset saved
