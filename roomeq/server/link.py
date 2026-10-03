"""Thread-safe bridge between the phone's audio stream (server thread) and measurement code (main thread).

Audio frames carry the absolute index of their first sample, so gaps (dropped packets) are
detected and filled with silence, and duplicates are ignored. Control messages to the phone go
over the WebSocket when connected; HTTP-fallback clients poll for them.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from ..dsp.biquad import FloatArray

HEADER = struct.Struct("<II")       # first-sample index (uint32), stream id (uint32)


class PhoneError(RuntimeError):
    pass


@dataclass
class PhoneInfo:
    sample_rate: float = 0.0
    stream_id: int = 0
    user_agent: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    transport: str = ""


class PhoneLink:
    def __init__(self, seconds: float = 90.0, max_rate: int = 48000):
        self._cond = threading.Condition()
        self._buf = np.zeros(int(seconds * max_rate), dtype=np.float32)
        self._written = 0
        self.dropped = 0
        self.info: PhoneInfo | None = None
        self.last_seen = 0.0
        self._messages: list[dict] = []
        self._ws_send: Callable[[str], Awaitable[None]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._events: list[dict] = []

    # ---------------------------------------------------------------- phone side (server thread)
    def hello(self, sample_rate: float, stream_id: int, user_agent: str = "", settings: dict | None = None,
              transport: str = "ws") -> None:
        with self._cond:
            self.info = PhoneInfo(float(sample_rate), int(stream_id), user_agent, settings or {}, transport)
            self._written = 0
            self.dropped = 0
            self.last_seen = time.monotonic()
            self._cond.notify_all()

    def ingest(self, frame: bytes) -> None:
        if len(frame) < HEADER.size + 4:
            return
        start, sid = HEADER.unpack_from(frame)
        samples = np.frombuffer(frame, dtype="<f4", offset=HEADER.size)
        with self._cond:
            if self.info is None or sid != self.info.stream_id:
                return
            self.last_seen = time.monotonic()
            n = len(self._buf)
            if start > self._written:                       # gap: pad with silence
                gap = start - self._written
                self.dropped += gap
                self._write_locked(np.zeros(min(gap, n), dtype=np.float32), self._written + max(gap - n, 0))
                self._written = start
            skip = self._written - start                    # overlap: drop the duplicated part
            if skip >= len(samples):
                return
            self._write_locked(samples[skip:], self._written)
            self._written += len(samples) - skip
            self._cond.notify_all()

    def _write_locked(self, x: np.ndarray, at: int) -> None:
        n = len(self._buf)
        pos = at % n
        first = min(len(x), n - pos)
        self._buf[pos:pos + first] = x[:first]
        if first < len(x):
            self._buf[: len(x) - first] = x[first:]

    def event(self, ev: dict) -> None:
        with self._cond:
            self._events.append(ev)
            self.last_seen = time.monotonic()
            self._cond.notify_all()

    def attach_ws(self, loop: asyncio.AbstractEventLoop, send: Callable[[str], Awaitable[None]]) -> None:
        self._loop, self._ws_send = loop, send

    def detach_ws(self, send: Callable[[str], Awaitable[None]]) -> None:
        if self._ws_send is send:
            self._ws_send = None

    def messages_after(self, after: int) -> list[dict]:
        with self._cond:
            return [m for m in self._messages if m["id"] > after]

    # ---------------------------------------------------------------- measurement side (main thread)
    def post(self, msg: dict) -> None:
        with self._cond:
            msg = {**msg, "id": len(self._messages) + 1}
            self._messages.append(msg)
            send, loop = self._ws_send, self._loop
        if send is not None and loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(send(json.dumps(msg)), loop)
            except RuntimeError:
                pass

    @property
    def connected(self) -> bool:
        return self.info is not None and time.monotonic() - self.last_seen < 3.0

    @property
    def written(self) -> int:
        with self._cond:
            return self._written

    @property
    def sample_rate(self) -> float:
        if self.info is None:
            raise PhoneError("phone not connected")
        return self.info.sample_rate

    def wait_connected(self, timeout: float | None = None) -> PhoneInfo:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not (self.info is not None and self._written > 0):
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    raise PhoneError("timed out waiting for the phone to start streaming")
                self._cond.wait(0.5 if left is None else min(left, 0.5))
            return self.info

    def wait_for(self, index: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._written < index:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise PhoneError(
                        f"phone audio stalled ({(index - self._written) / max(self.sample_rate, 1):.1f}s missing). "
                        "Is the page still open and the screen on?")
                self._cond.wait(min(left, 0.25))

    def read(self, start: int, stop: int) -> FloatArray:
        with self._cond:
            n = len(self._buf)
            if stop > self._written:
                raise PhoneError("requested audio not received yet")
            start = max(start, self._written - n, 0)
            idx = np.arange(start, stop) % n
            return self._buf[idx].astype(np.float64)

    def pop_events(self, kind: str | None = None) -> list[dict]:
        with self._cond:
            keep, out = [], []
            for e in self._events:
                (out if kind is None or e.get("type") == kind else keep).append(e)
            self._events = keep
            return out
