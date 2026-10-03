"""End-to-end: simulated room, phone, clock drift and noise -> autotune -> verified improvement."""

import numpy as np
import pytest

from roomeq.dsp.calibration import MicCalibration, derive_calibration, flat, parse_calibration
from roomeq.pipeline import autotune
from roomeq.presets import Preset, format_apo, format_table, load_preset, save_preset
from roomeq.sim.room import SimConfig, SimulatedRig


def quiet(_: str) -> None:
    pass


@pytest.mark.parametrize("phone_fs,drift", [(48000, 35.0), (44100, -60.0)])
def test_autotune_improves_simulated_room(phone_fs, drift):
    rig = SimulatedRig(SimConfig(phone_fs=phone_fs, drift_ppm=drift))
    rep = autotune(rig, iterations=3, positions=3, repeats=2, log=quiet)
    assert rep.rms_after < 0.7 * rep.rms_before
    assert rep.best.filters
    assert all(f.freq <= 4000 for f in rep.best.filters)
    assert any("null" in w for w in rep.warnings)
    assert any("flat" in w for w in rep.warnings)           # uncalibrated mic warning


def test_calibration_removes_mic_colouring():
    cfg = SimConfig()
    grid = np.geomspace(10, 23000, 400)
    cal = MicCalibration("sim", grid, cfg.mic.response_db(grid, 48000))
    rep_cal = autotune(SimulatedRig(cfg), calibration=cal, iterations=1, positions=1, log=quiet)
    rep_raw = autotune(SimulatedRig(cfg), calibration=flat(), iterations=1, positions=1, log=quiet)
    f = rep_cal.freqs
    k = (f > 4000) & (f < 8000)
    # the simulated mic has +3 dB at 6 kHz; calibration must remove it
    diff = rep_raw.history[0].measured_db[k] - rep_cal.history[0].measured_db[k]
    assert diff.max() > 2.0


def test_calibration_csv_parsing_and_derivation():
    text = '"Sens Factor =-1.2dB, SERNO: 700"\nfreq,db\n20, -1.5\n1000\t0.0\n10000; 2.5\n'
    cal = parse_calibration(text)
    assert cal.correction_db(np.array([1000.0]))[0] == pytest.approx(0.0)
    assert cal.correction_db(np.array([5.0]))[0] == pytest.approx(-1.5)        # held at the ends
    f = np.geomspace(20, 20000, 100)
    d = derive_calibration(f, np.full(100, 3.0) + np.log2(f / 1000), np.full(100, 3.0))
    assert d.correction_db(np.array([2000.0]))[0] == pytest.approx(1.0, abs=0.05)


def test_preset_roundtrip_and_export(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOMEQ_HOME", str(tmp_path))
    from roomeq.dsp.biquad import Filter
    p = Preset("living room", [Filter.peak(63, -6, 4.2), Filter.peak(125, 2, 1.5)], -2.5, rms_before=3.1, rms_after=1.2)
    save_preset(p)
    q = load_preset()
    assert q.name == "living room" and len(q.filters) == 2
    assert "Filter 1: ON PK Fc 63 Hz Gain -6.0 dB Q 4.20" in format_apo(q)
    assert "Preamp: -2.5 dB" in format_table(q)


class ReplacedEachRound(SimulatedRig):
    """The listener puts the phone at slightly different spots every time they are asked."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.visits = 0
        self.spot = 0

    def prepare_position(self, position, total):
        self.visits += 1
        self.spot = 100 + self.visits                       # a new, never-repeated spot

    def record(self, playback, position=0, eq=(), preamp_db=0.0):
        return super().record(playback, self.spot, eq, preamp_db)


def test_paired_rounds_are_fair_when_positions_vary_a_lot():
    from roomeq.sim.room import RoomModel

    rig = ReplacedEachRound(SimConfig(room=RoomModel(position_variation=3.0)))
    rep = autotune(rig, iterations=2, positions=3, log=quiet)
    # each round compares EQ off/on at identical spots, so a real improvement is measured as one
    for h in rep.history[1:]:
        assert h.rms_before_db is not None
    assert rep.best.index > 0
    assert rep.rms_after < rep.rms_before - 0.3
