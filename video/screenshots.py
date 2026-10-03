"""README screenshots.

Dashboard shots come from the running demo server (`roomeq demo --port 3200`, demo-home with the real
"Living room (measured)" preset). Phone shots come from a separate in-process server with Chromium's
fake microphone, so the virtual demo phone is not disturbed.

    uv run --with playwright python video/screenshots.py http://localhost:3200 http://192.168.1.2:3200
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "media"
OUT.mkdir(parents=True, exist_ok=True)
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3200"
LAN = sys.argv[2] if len(sys.argv) > 2 else BASE


def api(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read() or b"{}")


def wait_job(timeout: float = 180) -> None:
    t = time.time() + timeout
    while time.time() < t:
        st = api("/api/state")["job"]["state"]
        if st not in ("running", "waiting"):
            return
        time.sleep(1)
    raise TimeoutError("job did not finish")


def dashboard_shots(p) -> None:
    b = p.chromium.launch()
    # real room, dark: hero + chart
    api("/api/presets/load", {"name": "Living_room_measured"})
    ctx = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="dark")
    page = ctx.new_page()
    page.goto(BASE + "/?theme=dark")
    page.wait_for_timeout(2500)
    page.screenshot(path=str(OUT / "dashboard-dark.png"))
    ctx.close()

    ctx = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="light")
    page = ctx.new_page()
    page.goto(BASE + "/?theme=light")
    page.wait_for_timeout(2500)
    page.add_style_tag(content=".topbar{position:static}")   # keep the sticky header off close-ups
    card = page.locator(".chart-card")
    box = page.locator("#chart-resp").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.27, box["y"] + box["height"] * 0.45)
    page.wait_for_timeout(400)
    card.screenshot(path=str(OUT / "real-room-chart.png"))

    # filters, presets, measure card (light)
    page.locator("#filters-title").scroll_into_view_if_needed()
    page.evaluate("window.scrollTo(0, document.querySelector('#filters-title').getBoundingClientRect().top + window.scrollY - 90)")
    page.wait_for_timeout(600)
    page.screenshot(path=str(OUT / "dashboard-light.png"))

    # auto-tune running, then verify result (demo world, simulated room)
    api("/api/job/start", {"kind": "autotune", "positions": 3, "repeats": 1, "iterations": 2})
    page.wait_for_timeout(15000)
    page.evaluate("window.scrollTo(0, document.querySelector('#measure-title').getBoundingClientRect().top + window.scrollY - 90)")
    page.wait_for_timeout(1200)
    page.locator("#measure-title").locator("xpath=..").screenshot(path=str(OUT / "autotune-running.png"))
    wait_job()
    api("/api/job/start", {"kind": "verify", "positions": 1, "repeats": 1})
    wait_job()
    page.wait_for_timeout(1500)
    log = page.locator("#job-log")
    log.evaluate("e => e.style.maxHeight = '520px'")
    log.evaluate("e => e.scrollTop = e.scrollHeight")
    page.wait_for_timeout(300)
    page.locator("#measure-title").locator("xpath=..").screenshot(path=str(OUT / "verify-pass.png"))
    api("/api/presets/load", {"name": "Living_room_measured"})
    b.close()


def phone_shots(p) -> None:
    """The phone page in its three states, against a private server (fake mic)."""
    import os
    from dataclasses import replace

    os.environ["ROOMEQ_HOME"] = str(Path(__file__).resolve().parent / "demo-home")
    from roomeq.config import Config
    from roomeq.session import start_server

    srv = start_server(replace(Config(), port=19443, http_port=19080))
    link = srv.link
    lan = srv.urls.http[0]
    wav = Path(__file__).resolve().parent / "build" / "quiet-mic.wav"     # a room-level signal, not a full-scale tone
    b = p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                f"--use-file-for-fake-audio-capture={wav}"])
    ctx = b.new_context(viewport={"width": 393, "height": 760}, device_scale_factor=3)
    ctx.grant_permissions(["microphone"])
    page = ctx.new_page()
    page.goto(lan + "/")                       # LAN address: not a secure context -> first-visit setup
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "phone-setup.png"))
    page.goto("http://localhost:19080/phone")  # localhost is a secure context: the real mic path
    page.click("#start")
    link.wait_connected(timeout=10)
    link.post({"type": "position", "index": 1, "total": 3, "title": "Left of the seat",
               "text": "Move the phone about 40 cm to the left of position 1, at the same height."})
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "phone-measuring.png"))
    link.post({"type": "result", "done": True,
               "text": "7 filters, preamp -3.7 dB\nRMS error 4.6 → 2.6 dB (measured)",
               "warnings": ["Deep narrow dip around 366 Hz left alone: it is a room null (cancellation). "
                            "EQ cannot fill it; moving the seat or the subwoofer can."]})
    page.wait_for_timeout(1200)
    page.screenshot(path=str(OUT / "phone-result.png"))
    b.close()
    srv.stop()


with sync_playwright() as p:
    dashboard_shots(p)
    phone_shots(p)
print("saved to", OUT)
for f in sorted(OUT.glob("*.png")):
    print(" ", f.name)
