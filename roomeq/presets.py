"""EQ presets (JSON) and export formats."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .dsp.biquad import Filter, FilterType


def home() -> Path:
    p = Path(os.environ.get("ROOMEQ_HOME", Path.home() / ".roomeq"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def presets_dir() -> Path:
    p = home() / "presets"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Preset:
    name: str
    filters: list[Filter]
    preamp_db: float
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    rms_before: float | None = None
    rms_after: float | None = None
    notes: list[str] = field(default_factory=list)
    response: dict | None = None      # {"freqs": [...], "before": [...], "after": [...], "target": [...]}

    def to_dict(self) -> dict:
        return {
            "name": self.name, "created": self.created, "preamp_db": round(self.preamp_db, 2),
            "filters": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.to_dict().items()}
                        for f in self.filters],
            "rms_before": self.rms_before, "rms_after": self.rms_after, "notes": self.notes,
            "response": self.response,
        }

    @staticmethod
    def from_dict(d: dict) -> "Preset":
        return Preset(d["name"], [Filter.from_dict(f) for f in d.get("filters", [])], float(d.get("preamp_db", 0.0)),
                      d.get("created", ""), d.get("rms_before"), d.get("rms_after"), d.get("notes", []),
                      d.get("response"))


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "preset"


def save_preset(p: Preset) -> Path:
    path = presets_dir() / f"{_slug(p.name)}.json"
    path.write_text(json.dumps(p.to_dict(), indent=2), encoding="utf-8")
    (presets_dir() / ".last").write_text(path.name, encoding="utf-8")
    return path


def load_preset(name_or_path: str | None = None) -> Preset:
    if name_or_path is None:
        last = presets_dir() / ".last"
        if not last.exists():
            raise FileNotFoundError("no presets saved yet - run `roomeq autotune` or `roomeq simulate --save`")
        path = presets_dir() / last.read_text(encoding="utf-8").strip()
    else:
        path = Path(name_or_path)
        if not path.exists():
            path = presets_dir() / f"{_slug(name_or_path)}.json"
    return Preset.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_presets() -> list[str]:
    return sorted(p.stem for p in presets_dir().glob("*.json"))


def list_preset_infos() -> list[dict[str, str]]:
    """``[{"id": file stem, "name": display name}]``, newest first."""
    out = []
    for p in sorted(presets_dir().glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            name = json.loads(p.read_text(encoding="utf-8")).get("name") or p.stem
        except (OSError, ValueError):
            continue
        out.append({"id": p.stem, "name": name})
    return out


_TYPE_LABEL = {FilterType.PEAK: "Peak", FilterType.LOW_SHELF: "LowShelf", FilterType.HIGH_SHELF: "HighShelf"}
_APO = {FilterType.PEAK: "PK", FilterType.LOW_SHELF: "LSC", FilterType.HIGH_SHELF: "HSC"}


def format_table(p: Preset) -> str:
    lines = [f"Preset: {p.name}", f"Preamp: {p.preamp_db:+.1f} dB", "",
             f"{'#':>2}  {'Type':<9} {'Freq (Hz)':>10} {'Gain (dB)':>10} {'Q':>6}",
             f"{'-' * 2}  {'-' * 9} {'-' * 10} {'-' * 10} {'-' * 6}"]
    for i, f in enumerate(p.filters, 1):
        lines.append(f"{i:>2}  {_TYPE_LABEL[f.type]:<9} {f.freq:>10.1f} {f.gain_db:>+10.1f} {f.q:>6.2f}")
    if p.rms_before is not None and p.rms_after is not None:
        lines += ["", f"RMS error vs target: {p.rms_before:.2f} dB -> {p.rms_after:.2f} dB"]
    return "\n".join(lines)


def format_apo(p: Preset) -> str:
    """Equalizer APO / AutoEQ "ParametricEQ.txt" style, accepted by many EQ apps."""
    lines = [f"Preamp: {p.preamp_db:.1f} dB"]
    for i, f in enumerate(p.filters, 1):
        lines.append(f"Filter {i}: ON {_APO[f.type]} Fc {f.freq:.0f} Hz Gain {f.gain_db:.1f} dB Q {f.q:.2f}")
    return "\n".join(lines)


def format_json(p: Preset) -> str:
    d = p.to_dict()
    d.pop("response", None)
    return json.dumps(d, indent=2)
