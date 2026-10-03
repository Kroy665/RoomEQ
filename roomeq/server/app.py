"""FastAPI app: phone recorder page, audio ingest (WebSocket + HTTP fallback), CA download.

Runs HTTPS (for the phone: microphone needs a secure context) and plain HTTP (CA download,
localhost dashboard, and tunnels) side by side in a background thread.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .certs import CertPaths
from .link import PhoneLink

STATIC = Path(str(resources.files("roomeq.server") / "static"))


@dataclass
class ServerUrls:
    https: list[str] = field(default_factory=list)
    http: list[str] = field(default_factory=list)
    tunnel: str | None = None

    @property
    def phone(self) -> str:
        return self.tunnel or (self.https[0] if self.https else "")


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _is_local(req: Request) -> bool:
    return req.client is not None and req.client.host in LOCAL_HOSTS


def create_app(link: PhoneLink, urls: ServerUrls, certs: CertPaths | None, controller=None) -> FastAPI:
    app = FastAPI(title="RoomEQ", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.link = link
    app.state.urls = urls
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    nocache = {"Cache-Control": "no-store"}

    @app.get("/", response_class=HTMLResponse)
    def index(req: Request) -> Any:
        # The Mac (localhost) gets the dashboard; the phone (LAN address or tunnel) gets the mic page.
        if controller is not None and _is_local(req) and req.url.hostname in LOCAL_HOSTS:
            return FileResponse(STATIC / "dashboard.html", headers=nocache)
        return FileResponse(STATIC / "phone.html", headers=nocache)

    @app.get("/phone", response_class=HTMLResponse)
    def phone_page() -> Any:
        return FileResponse(STATIC / "phone.html", headers=nocache)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_page() -> Any:
        if controller is None:
            return HTMLResponse("<p>The dashboard needs the engine: run <code>roomeq run</code>.</p>", status_code=503)
        return FileResponse(STATIC / "dashboard.html", headers=nocache)

    if controller is not None:
        _dashboard_api(app, controller, urls)

    @app.get("/api/info")
    def info() -> Any:
        i = link.info
        return {"https": urls.https, "http": urls.http, "tunnel": urls.tunnel, "phone_url": urls.phone,
                "connected": link.connected, "sample_rate": i.sample_rate if i else None}

    @app.get("/api/ping")
    def ping() -> Any:
        # lets the plain-HTTP setup page detect whether this phone already trusts the RoomEQ CA
        return JSONResponse({"ok": True}, headers={"Access-Control-Allow-Origin": "*"})

    @app.get("/roomeq-ca.cer")
    def ca_der() -> Any:
        if certs is None:
            return Response(status_code=404)
        # this content type makes iOS Safari offer to install it as a profile
        return Response(certs.ca_der.read_bytes(), media_type="application/x-x509-ca-cert",
                        headers={"Content-Disposition": 'attachment; filename="RoomEQ-CA.cer"'})

    @app.websocket("/ws/phone")
    async def ws_phone(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()

        async def send(text: str) -> None:
            await ws.send_text(text)

        link.attach_ws(loop, send)
        try:
            for m in link.messages_after(0)[-1:]:          # replay the latest instruction
                await ws.send_text(json.dumps(m))
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    link.ingest(msg["bytes"])
                elif msg.get("text") is not None:
                    _control(link, json.loads(msg["text"]), "ws")
        except WebSocketDisconnect:
            pass
        finally:
            link.detach_ws(send)

    # HTTP fallback for browsers/networks where the WebSocket cannot be opened
    @app.post("/api/phone/control")
    async def control(req: Request) -> Any:
        _control(link, await req.json(), "http")
        return {"ok": True}

    @app.post("/api/phone/audio")
    async def audio(req: Request) -> Any:
        link.ingest(await req.body())
        return {"ok": True}

    @app.get("/api/phone/messages")
    def messages(after: int = 0) -> Any:
        return JSONResponse(link.messages_after(after))

    return app


def _dashboard_api(app: FastAPI, ctl, urls: ServerUrls) -> None:
    from fastapi import HTTPException

    from ..dsp.biquad import Filter
    from ..engine.core import UnsafePreset
    from ..presets import list_preset_infos

    def local(req: Request) -> None:
        if not _is_local(req):
            raise HTTPException(403, "controls are only available on the Mac running RoomEQ (localhost)")

    @app.get("/api/state")
    def state() -> Any:
        s = ctl.state()
        s["phone_url"] = urls.tunnel or (urls.http[0] if urls.http else "")
        return s

    @app.get("/api/curves")
    def curves() -> Any:
        return ctl.curves_payload()

    @app.post("/api/bypass")
    async def bypass(req: Request) -> Any:
        local(req)
        ctl.core.set_bypass(bool((await req.json()).get("on")))
        return {"ok": True}

    @app.post("/api/panic")
    async def panic(req: Request) -> Any:
        local(req)
        on = (await req.json()).get("on")
        if on is None:
            on = not ctl.core.panicked
        ctl.core.panic() if on else ctl.core.clear_panic()
        return {"ok": True, "panicked": ctl.core.panicked}

    @app.post("/api/volume")
    async def volume(req: Request) -> Any:
        local(req)
        return {"volume_db": ctl.core.set_volume(float((await req.json())["db"]))}

    @app.post("/api/eq")
    async def eq(req: Request) -> Any:
        local(req)
        body = await req.json()
        try:
            filters = [Filter.from_dict(f) for f in body.get("filters", [])]
            pre = ctl.set_filters(filters)
        except (UnsafePreset, KeyError, ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))
        return {"ok": True, "preamp_db": pre}

    @app.get("/api/presets")
    def presets() -> Any:
        return {"presets": list_preset_infos(), "current": ctl.core.preset_name}

    @app.post("/api/presets/load")
    async def preset_load(req: Request) -> Any:
        local(req)
        try:
            p = ctl.load_preset(str((await req.json())["name"]))
        except FileNotFoundError:
            raise HTTPException(404, "no such preset")
        except UnsafePreset as exc:
            raise HTTPException(400, str(exc))
        return {"ok": True, "name": p.name}

    @app.post("/api/presets/save")
    async def preset_save(req: Request) -> Any:
        local(req)
        name = str((await req.json()).get("name", "")).strip()
        if not name:
            raise HTTPException(400, "name required")
        return {"ok": True, "path": ctl.save_current(name)}

    @app.post("/api/job/start")
    async def job_start(req: Request) -> Any:
        local(req)
        b = await req.json()
        kind = b.get("kind")
        if kind not in ("measure", "autotune", "verify"):
            raise HTTPException(400, "kind must be measure, autotune or verify")
        try:
            ctl.start_job(kind, int(b.get("positions", 3)), int(b.get("repeats", 1)), int(b.get("iterations", 3)))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"ok": True}

    @app.post("/api/job/continue")
    def job_continue(req: Request) -> Any:
        local(req)
        ctl.continue_job()
        return {"ok": True}

    @app.post("/api/job/cancel")
    def job_cancel(req: Request) -> Any:
        local(req)
        ctl.cancel_job()
        return {"ok": True}

    @app.get("/api/qr.svg")
    def qr() -> Any:
        import io

        import qrcode
        import qrcode.image.svg

        url = urls.tunnel or (urls.http[0] if urls.http else "")
        img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, border=1)
        buf = io.BytesIO()
        img.save(buf)
        return Response(buf.getvalue(), media_type="image/svg+xml", headers=nocache_hdr())


def nocache_hdr() -> dict:
    return {"Cache-Control": "no-store"}


def _control(link: PhoneLink, m: dict, transport: str) -> None:
    if m.get("type") == "hello":
        link.hello(float(m["sampleRate"]), int(m["streamId"]), str(m.get("userAgent", "")),
                   m.get("settings") or {}, transport)
    else:
        link.event(m)


class ServerThread(threading.Thread):
    """uvicorn HTTPS + HTTP servers in one background event loop."""

    def __init__(self, app: FastAPI, https_port: int | None, http_port: int, certs: CertPaths | None,
                 host: str = "0.0.0.0"):
        super().__init__(daemon=True, name="roomeq-server")
        import uvicorn

        self._servers: list[uvicorn.Server] = []
        common = dict(host=host, log_level="warning", lifespan="off", ws_max_size=4 * 1024 * 1024)
        if https_port and certs:
            self._servers.append(uvicorn.Server(uvicorn.Config(
                app, port=https_port, ssl_certfile=str(certs.cert), ssl_keyfile=str(certs.key), **common)))
        self._servers.append(uvicorn.Server(uvicorn.Config(app, port=http_port, **common)))
        for s in self._servers:
            s.install_signal_handlers = lambda: None       # main thread keeps Ctrl+C
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:                      # surfaced by wait_started()
            self.error = exc

    async def _serve(self) -> None:
        await asyncio.gather(*(s.serve() for s in self._servers))

    def wait_started(self, timeout: float = 10.0) -> None:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error or not self.is_alive():
                raise RuntimeError(f"server failed to start: {self.error}")
            if all(s.started for s in self._servers):
                return
            time.sleep(0.05)
        raise RuntimeError("server did not start in time (port in use?)")

    def stop(self) -> None:
        for s in self._servers:
            s.should_exit = True
        self.join(timeout=5)
