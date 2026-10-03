"""sounddevice wrapper: BlackHole input stream + speaker output stream around an EngineCore."""

from __future__ import annotations

import gc
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from ..audio.devices import Device, find_device
from ..config import AudioConfig
from .core import EngineCore, EngineSettings


class EngineError(RuntimeError):
    pass


def _supported(device: Device, rate: int, kind: str) -> bool:
    import sounddevice as sd

    try:
        if kind == "input":
            sd.check_input_settings(device=device.index, samplerate=rate, channels=min(2, device.inputs))
        else:
            sd.check_output_settings(device=device.index, samplerate=rate, channels=min(2, device.outputs))
        return True
    except Exception:
        return False


@dataclass
class StreamInfo:
    input_name: str
    output_name: str
    fs_in: int
    fs_out: int
    blocksize: int
    latency_ms: float


class AudioEngine:
    def __init__(self, cfg: AudioConfig, blocksize: int | None = None):
        self.cfg = cfg
        self.dev_in = find_device(cfg.input_device, "input")
        self.dev_out = find_device(cfg.output_device, "output")
        if "blackhole" in self.dev_out.name.lower() or self.dev_out.name == self.dev_in.name:
            raise EngineError(f'output device "{self.dev_out.name}" would feed back into the input. '
                              "Set audio.output_device to your speakers (e.g. External Headphones).")
        fs_in = cfg.samplerate
        if not _supported(self.dev_in, fs_in, "input"):
            fs_in = int(self.dev_in.default_samplerate)
        fs_out = fs_in if _supported(self.dev_out, fs_in, "output") else int(self.dev_out.default_samplerate)
        self.blocksize = int(blocksize or cfg.blocksize)
        self.core = EngineCore(EngineSettings(fs_in=fs_in, fs_out=fs_out, blocksize=self.blocksize,
                                              ceiling_dbfs=cfg.limiter_ceiling_dbfs, soft_start_s=cfg.soft_start_s))
        self.xruns = 0
        self._in = None
        self._out = None

    # The callbacks run on CoreAudio threads. They only hand preallocated buffers to numba kernels.
    def _in_cb(self, indata, frames, time, status) -> None:
        if status:
            self.xruns += 1
        self.core.process_input(indata)

    def _out_cb(self, outdata, frames, time, status) -> None:
        if status:
            self.xruns += 1
        self.core.process_output(outdata)

    def start(self) -> StreamInfo:
        import sounddevice as sd

        s = self.core.s
        self._in = sd.InputStream(device=self.dev_in.index, channels=min(2, self.dev_in.inputs), samplerate=s.fs_in,
                                  blocksize=self.blocksize, dtype="float32", latency="low", callback=self._in_cb)
        self._out = sd.OutputStream(device=self.dev_out.index, channels=2 if self.dev_out.outputs >= 2 else 1,
                                    samplerate=s.fs_out, blocksize=self.blocksize, dtype="float32", latency="low",
                                    callback=self._out_cb)
        if self._out.channels != s.channels:
            raise EngineError("mono output devices are not supported")
        gc.collect()
        gc.disable()                      # no collector pauses while audio runs
        self._out.start()
        self._in.start()
        return StreamInfo(self.dev_in.name, self.dev_out.name, s.fs_in, s.fs_out, self.blocksize, self.latency_ms())

    def latency_ms(self) -> float:
        lin = self._in.latency if self._in is not None else 0.0
        lout = self._out.latency if self._out is not None else 0.0
        return self.core.latency_ms(lin, lout)

    def output_latency_s(self) -> float:
        return float(self._out.latency) if self._out is not None else 0.0

    def stop(self) -> None:
        for st in (self._out, self._in):                 # output first, so it never starves
            if st is not None:
                try:
                    st.stop()
                    st.close()
                except Exception:
                    pass
        self._in = self._out = None
        gc.enable()


class SystemOutput:
    """Switch the macOS default output with SwitchAudioSource (brew install switchaudio-osx), if present."""

    def __init__(self) -> None:
        self.tool = shutil.which("SwitchAudioSource")
        self.previous: str | None = None

    @property
    def available(self) -> bool:
        return self.tool is not None

    def current(self) -> str | None:
        if not self.tool:
            return None
        r = subprocess.run([self.tool, "-c", "-t", "output"], capture_output=True, text=True)
        return r.stdout.strip() or None

    def switch_to(self, name: str, fallback: str) -> bool:
        """Switch to ``name``; on restore go back to the previous output, or ``fallback`` if the
        previous output already was ``name`` (otherwise quitting would leave the Mac silent)."""
        if not self.tool:
            return False
        cur = self.current()
        self.previous = fallback if (cur is None or cur == name) else cur
        r = subprocess.run([self.tool, "-s", name, "-t", "output"], capture_output=True, text=True)
        return r.returncode == 0

    def restore(self) -> None:
        if self.tool and self.previous:
            subprocess.run([self.tool, "-s", self.previous, "-t", "output"], capture_output=True, text=True)
            self.previous = None
