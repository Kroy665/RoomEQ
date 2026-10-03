"""Measurement-microphone calibration (per phone model).

CSV convention (same as miniDSP/UMIK and REW): each row is ``frequency_hz, db`` where ``db``
is how much the microphone over-reads at that frequency. The corrected response is
``measured - db``. Header lines, quotes and lines such as ``"Sens Factor =..."`` are skipped;
separators may be commas, semicolons, tabs or spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .biquad import FloatArray
from .spectrum import interp_log

FLAT_WARNING = ("No microphone calibration loaded - treating the phone mic as flat. Bass and low-mid "
                "corrections are still useful, but expect errors of several dB above ~4 kHz and "
                "possibly below ~40 Hz.")

_NUM = re.compile(r"^[\s\"']*[-+]?\d")


@dataclass(frozen=True)
class MicCalibration:
    name: str
    freqs: FloatArray
    db: FloatArray

    @property
    def is_flat(self) -> bool:
        return self.freqs.size == 0

    def correction_db(self, freqs: FloatArray) -> FloatArray:
        if self.is_flat:
            return np.zeros_like(np.asarray(freqs, dtype=np.float64))
        return interp_log(self.freqs, self.db, np.asarray(freqs, dtype=np.float64))

    def apply(self, freqs: FloatArray, measured_db: FloatArray) -> FloatArray:
        return np.asarray(measured_db) - self.correction_db(freqs)


def flat() -> MicCalibration:
    return MicCalibration("flat (uncalibrated)", np.array([]), np.array([]))


def parse_calibration(text: str, name: str = "custom") -> MicCalibration:
    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        if not _NUM.match(line):
            continue
        parts = [p for p in re.split(r"[,;\t ]+", line.strip().replace('"', "").replace("'", "")) if p]
        try:
            f, d = float(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            continue
        if f > 0:
            rows.append((f, d))
    if len(rows) < 2:
        raise ValueError(f"calibration '{name}' has fewer than two valid (frequency, dB) rows")
    rows.sort()
    arr = np.array(rows)
    return MicCalibration(name, arr[:, 0], arr[:, 1])


def load_calibration(path: str | Path) -> MicCalibration:
    p = Path(path)
    return parse_calibration(p.read_text(encoding="utf-8", errors="replace"), p.stem)


def derive_calibration(freqs: FloatArray, phone_db: FloatArray, reference_db: FloatArray,
                       reference_cal: MicCalibration | None = None, normalise_hz: float = 1000.0) -> MicCalibration:
    """Build a phone calibration from simultaneous-position measurements with a reference mic.

    ``reference_cal`` is the reference mic's own calibration (e.g. the UMIK-1 file). The result
    is normalised to 0 dB at ``normalise_hz``.
    """
    ref = reference_db if reference_cal is None else reference_cal.apply(freqs, reference_db)
    diff = np.asarray(phone_db) - ref
    diff = diff - np.interp(np.log(normalise_hz), np.log(freqs), diff)
    return MicCalibration("derived", np.asarray(freqs, dtype=np.float64), diff)


def save_calibration(cal: MicCalibration, path: str | Path) -> None:
    lines = [f"# RoomEQ microphone calibration: {cal.name}", "# frequency_hz, db"]
    lines += [f"{f:.3f}, {d:.3f}" for f, d in zip(cal.freqs, cal.db)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
