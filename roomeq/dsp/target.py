"""Target curves (relative dB; the solver aligns the level automatically)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .biquad import FloatArray
from .spectrum import interp_log


@dataclass(frozen=True)
class TargetCurve:
    """Flat mids, smooth bass lift and gentle treble tilt.

    bass: ``bass_boost_db / (1 + (f / bass_transition_hz) ** bass_slope)``. Defaults give about
    +3.3 dB at 100 Hz, +3.9 dB at 60 Hz and +1 dB at 200 Hz.
    treble: ``treble_tilt_db_per_octave * log2(f / treble_start_hz)`` above ``treble_start_hz``.
    ``custom_points`` (freq, dB) pairs override the parametric curve when given.
    """

    bass_boost_db: float = 4.0
    bass_transition_hz: float = 150.0
    bass_slope: float = 4.0
    treble_tilt_db_per_octave: float = -1.0
    treble_start_hz: float = 2000.0
    custom_points: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def evaluate(self, freqs: FloatArray) -> FloatArray:
        f = np.asarray(freqs, dtype=np.float64)
        if self.custom_points:
            pts = sorted(self.custom_points)
            return interp_log(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), f)
        bass = self.bass_boost_db / (1.0 + (f / self.bass_transition_hz) ** self.bass_slope)
        treble = np.where(f > self.treble_start_hz,
                          self.treble_tilt_db_per_octave * np.log2(np.maximum(f, 1e-9) / self.treble_start_hz),
                          0.0)
        return bass + treble

    def to_dict(self) -> dict:
        d = asdict(self)
        d["custom_points"] = [list(p) for p in self.custom_points]
        return d

    @staticmethod
    def from_dict(d: dict) -> "TargetCurve":
        d = dict(d)
        d["custom_points"] = tuple(tuple(p) for p in d.get("custom_points", ()))
        return TargetCurve(**d)


FLAT = TargetCurve(bass_boost_db=0.0, treble_tilt_db_per_octave=0.0)
