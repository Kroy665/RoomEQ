"""``roomeq demo``: the simulated world drives a full auto-tune with no hardware and no human."""

import time

from roomeq.config import Config
from roomeq.demo import DemoWorld
from roomeq.engine.core import EngineCore, EngineSettings
from roomeq.jobs import Controller
from roomeq.server.link import PhoneLink


def test_demo_world_runs_autotune_unattended(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOMEQ_HOME", str(tmp_path))
    cfg = Config()
    cfg.measurement.sweep_seconds = 3.0
    core = EngineCore(EngineSettings())
    link = PhoneLink()
    ctl = Controller(cfg, core, link)
    ctl.demo = True
    world = DemoWorld(core, link, ctl, job_speed=40.0, ready_delay_s=0.05)
    world.start()
    try:
        deadline = time.monotonic() + 10
        while not link.connected:                         # the virtual phone connects by itself
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert ctl.state()["demo"] is True
        assert ctl.state()["phone"]["transport"] == "demo"
        ctl.start_job("autotune", positions=3, repeats=1, iterations=1)
        deadline = time.monotonic() + 180
        while ctl.job.state in ("running", "waiting"):    # nobody presses Continue: it taps Ready itself
            assert time.monotonic() < deadline, ctl.job.lines[-10:]
            time.sleep(0.1)
    finally:
        world.stop_flag.set()
        world.join(timeout=5)
    assert ctl.job.state == "done", "\n".join(ctl.job.lines[-20:])
    assert core.filters
    c = ctl.curves_payload()
    assert c["rms_after"] < c["rms_before"] - 1.0           # the demo room has a real problem to fix
