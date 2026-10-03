"""Phone link: buffer logic, certificates, and a full measurement over a real HTTPS WebSocket."""

import datetime as dt
import json
import queue
import socket
import ssl
import struct
import threading
import time

import httpx
import numpy as np
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID

from roomeq.dsp.sweep import SweepConfig, make_test_signal
from roomeq.pipeline import measure_set
from roomeq.rigs import PhoneRig, level_check, terminal_or_phone_ready
from roomeq.server.app import ServerThread, ServerUrls, create_app
from roomeq.server.certs import ensure_certs
from roomeq.server.link import HEADER, PhoneLink
from roomeq.sim.room import SimConfig, SimulatedRig


def frame(start: int, sid: int, x: np.ndarray) -> bytes:
    return HEADER.pack(start, sid) + np.asarray(x, dtype="<f4").tobytes()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ----------------------------------------------------------------------------- unit

def test_link_handles_gaps_duplicates_and_stale_streams():
    link = PhoneLink(seconds=1, max_rate=1000)
    link.hello(1000, 7)
    link.ingest(frame(0, 7, np.ones(100)))
    link.ingest(frame(50, 7, np.full(100, 2.0)))           # overlaps by 50
    link.ingest(frame(200, 7, np.full(10, 3.0)))           # 50-sample gap
    link.ingest(frame(210, 99, np.full(10, 9.0)))          # stale stream id: ignored
    assert link.written == 210
    assert link.dropped == 50
    x = link.read(0, 210)
    assert np.all(x[:100] == 1) and np.all(x[100:150] == 2) and np.all(x[150:200] == 0) and np.all(x[200:] == 3)
    link.hello(1000, 8)                                    # reconnect resets the stream
    assert link.written == 0


def test_link_ring_wraps():
    link = PhoneLink(seconds=1, max_rate=100)              # 100-sample ring
    link.hello(100, 1)
    for i in range(5):
        link.ingest(frame(i * 60, 1, np.full(60, float(i))))
    np.testing.assert_array_equal(link.read(240, 300), np.full(60, 4.0))


def test_certificates_meet_ios_rules(tmp_path):
    paths = ensure_certs(tmp_path)
    ca = x509.load_pem_x509_certificate(paths.ca_pem.read_bytes())
    srv = x509.load_pem_x509_certificate(paths.cert.read_bytes())    # first cert in the chain file
    ca.public_key().verify(srv.signature, srv.tbs_certificate_bytes, ec.ECDSA(srv.signature_hash_algorithm))
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "127.0.0.1" in {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    eku = srv.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku
    assert srv.not_valid_after_utc - srv.not_valid_before_utc <= dt.timedelta(days=825)
    # stable across calls, re-issued when a new host is needed
    before = paths.cert.read_bytes()
    assert ensure_certs(tmp_path).cert.read_bytes() == before
    assert ensure_certs(tmp_path, extra_hosts=["new.local"]).cert.read_bytes() != before


# ----------------------------------------------------------------------------- integration

class FakePhone(threading.Thread):
    """Streams like phone.js: hello, then little-endian float32 frames, silence when idle, ~4x realtime."""

    def __init__(self, url: str, ctx: ssl.SSLContext, fs: int):
        super().__init__(daemon=True)
        self.url, self.ctx, self.fs = url, ctx, fs
        self.audio: "queue.Queue[np.ndarray]" = queue.Queue()
        self.messages: list[dict] = []
        self.stop_flag = threading.Event()
        self.error: Exception | None = None

    def run(self) -> None:
        from websockets.sync.client import connect

        try:
            with connect(self.url, ssl=self.ctx) as ws:
                sid = 1234
                ws.send(json.dumps({"type": "hello", "sampleRate": self.fs, "streamId": sid,
                                    "settings": {"echoCancellation": False}}))
                sent, chunk, pending = 0, 2048, np.zeros(0)
                while not self.stop_flag.is_set():
                    while True:
                        try:
                            pending = np.concatenate([pending, self.audio.get_nowait()])
                        except queue.Empty:
                            break
                    if len(pending):
                        x, pending = pending[:chunk], pending[chunk:]
                    else:
                        x = np.zeros(chunk)
                    ws.send(frame(sent, sid, x))
                    sent += len(x)
                    try:
                        msg = json.loads(ws.recv(timeout=0))
                        self.messages.append(msg)
                        if msg.get("type") == "position":
                            ws.send(json.dumps({"type": "ready"}))
                    except TimeoutError:
                        pass
                    time.sleep(chunk / self.fs / 4)
        except Exception as exc:  # surfaced in the test
            self.error = exc


@pytest.mark.parametrize("phone_fs", [48000, 44100])
def test_full_measurement_over_https_websocket(tmp_path, phone_fs):
    certs = ensure_certs(tmp_path)
    https_port, http_port = free_port(), free_port()
    link = PhoneLink()
    urls = ServerUrls(https=[f"https://127.0.0.1:{https_port}"], http=[f"http://127.0.0.1:{http_port}"])
    server = ServerThread(create_app(link, urls, certs), https_port, http_port, certs, host="127.0.0.1")
    server.start()
    server.wait_started()
    ctx = ssl.create_default_context(cafile=str(certs.ca_pem))   # proves the chain validates
    phone = FakePhone(f"wss://127.0.0.1:{https_port}/ws/phone", ctx, phone_fs)
    phone.start()
    try:
        sim = SimulatedRig(SimConfig(phone_fs=phone_fs, drift_ppm=42.0, record_lead_s=0.0))

        def player(sig, fs):     # "the room": convert playback into what the phone hears
            rec, _ = sim.record(sig)
            phone.audio.put(rec.astype(np.float64))

        info = link.wait_connected(timeout=10)
        assert info.sample_rate == phone_fs and info.transport == "ws"
        rig = PhoneRig(link, player, 48000, log=lambda s: None, notify=link.post,
                       wait_ready=terminal_or_phone_ready(link))
        lv = level_check(rig)
        assert lv.signal_dbfs > lv.background_dbfs + 20

        ts = make_test_signal(SweepConfig(duration=3.0))
        ms = measure_set(rig, ts, positions=2, repeats=1, log=lambda s: None)
        assert phone.error is None
        assert ms.reliable, ms.warnings
        for row in ms.measurements:
            assert row[0].quality.drift_ppm == pytest.approx(42.0, abs=2.0)
        assert [m["index"] for m in phone.messages if m["type"] == "position"] == [0, 1]

        # same result as analysing the simulator's output directly (no network)
        from roomeq.dsp.analysis import analyze_recording
        direct = analyze_recording(*sim.record(ts.signal), ts)
        k = (ms.freqs > 30) & (ms.freqs < 10000)
        a = ms.measurements[0][0].db
        d = (a - direct.db)[k]
        assert np.max(np.abs(d - np.median(d))) < 0.3
    finally:
        phone.stop_flag.set()
        phone.join(timeout=5)
        server.stop()


def test_http_fallback_endpoints(tmp_path):
    link = PhoneLink()
    app = create_app(link, ServerUrls(https=["https://x"], http=["http://x"]), ensure_certs(tmp_path))
    from fastapi.testclient import TestClient

    c = TestClient(app)
    assert c.post("/api/phone/control", json={"type": "hello", "sampleRate": 48000, "streamId": 5}).status_code == 200
    c.post("/api/phone/audio", content=frame(0, 5, np.ones(480)))
    assert link.written == 480 and link.info.transport == "http"
    link.post({"type": "status", "text": "hi"})
    msgs = c.get("/api/phone/messages?after=0").json()
    assert msgs[-1]["text"] == "hi"
    c.post("/api/phone/control", json={"type": "ready"})
    assert link.pop_events("ready")
    assert c.get("/roomeq-ca.cer").headers["content-type"] == "application/x-x509-ca-cert"
    assert c.get("/api/ping").headers["access-control-allow-origin"] == "*"
    assert "RoomEQ" in c.get("/").text
