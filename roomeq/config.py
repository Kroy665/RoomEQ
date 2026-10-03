"""User configuration: ``$ROOMEQ_HOME/config.toml`` (default ``~/.roomeq/config.toml``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .dsp.analysis import AnalysisConfig
from .dsp.solver import SolverConfig
from .dsp.sweep import SweepConfig
from .dsp.target import TargetCurve
from .presets import home

DEFAULT_TOML = """\
# RoomEQ configuration. Delete a line to fall back to its default.

[audio]
# Run `roomeq devices` to see the exact names. Partial names are matched.
input_device = "BlackHole 2ch"
output_device = "External Headphones"   # where the speakers are connected
samplerate = 48000
blocksize = 256               # frames per callback; lower = less latency, higher = safer
limiter_ceiling_dbfs = -1.0
soft_start_s = 1.5

[measurement]
positions = 3                 # listening positions to average
repeats = 2                   # sweeps per position (2+ enables the repeatability check)
sweep_seconds = 8.0
sweep_level_dbfs = -12.0
through_eq = false            # measure with the current EQ applied
calibration_file = ""         # phone mic calibration CSV; empty = treat mic as flat

[target]
bass_boost_db = 4.0
bass_transition_hz = 150.0
treble_tilt_db_per_octave = -1.0
treble_start_hz = 2000.0

[solver]
max_filters = 10
max_boost_db = 3.0
max_cut_db = 9.0
q_min = 0.7
q_max = 8.0
f_min = 25.0
full_band_max_hz = 500.0
gentle_band_max_hz = 4000.0

[autotune]
iterations = 3

[server]
port = 8443                   # HTTPS (phone)
http_port = 8080              # plain HTTP (certificate download, tunnel, localhost)
"""


@dataclass
class AudioConfig:
    input_device: str = "BlackHole 2ch"
    output_device: str = "External Headphones"
    samplerate: int = 48000
    blocksize: int = 256
    limiter_ceiling_dbfs: float = -1.0
    soft_start_s: float = 1.5


@dataclass
class MeasureConfig:
    positions: int = 3
    repeats: int = 2
    sweep_seconds: float = 8.0
    sweep_level_dbfs: float = -12.0
    through_eq: bool = False
    calibration_file: str = ""


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    measurement: MeasureConfig = field(default_factory=MeasureConfig)
    target: TargetCurve = field(default_factory=TargetCurve)
    solver: SolverConfig = field(default_factory=SolverConfig)
    iterations: int = 3
    port: int = 8443
    http_port: int = 8080

    def sweep(self) -> SweepConfig:
        return SweepConfig(fs=self.audio.samplerate, duration=self.measurement.sweep_seconds,
                           level_dbfs=self.measurement.sweep_level_dbfs)

    def analysis(self) -> AnalysisConfig:
        return AnalysisConfig()


def _pick(cls: type, d: dict[str, Any], **extra: Any) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys in config: {', '.join(sorted(unknown))}")
    return cls(**{**d, **extra})


def config_path() -> Path:
    return home() / "config.toml"


def write_default(path: Path | None = None, overwrite: bool = False) -> Path:
    path = path or config_path()
    if path.exists() and not overwrite:
        return path
    path.write_text(DEFAULT_TOML, encoding="utf-8")
    return path


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    audio = _pick(AudioConfig, data.get("audio", {}))
    return Config(
        audio=audio,
        measurement=_pick(MeasureConfig, data.get("measurement", {})),
        target=_pick(TargetCurve, data.get("target", {})),
        solver=_pick(SolverConfig, data.get("solver", {}), fs=float(audio.samplerate)),
        iterations=int(data.get("autotune", {}).get("iterations", 3)),
        port=int(data.get("server", {}).get("port", 8443)),
        http_port=int(data.get("server", {}).get("http_port", 8080)),
    )
