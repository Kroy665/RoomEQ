"""Optional public HTTPS tunnel (cloudflared or ngrok), so the phone needs no certificate setup."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import urllib.request


class Tunnel:
    def __init__(self, proc: subprocess.Popen, url: str, tool: str):
        self.proc, self.url, self.tool = proc, url, tool

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def available() -> str | None:
    for tool in ("cloudflared", "ngrok"):
        if shutil.which(tool):
            return tool
    return None


def start_tunnel(port: int, tool: str | None = None, timeout: float = 25.0) -> Tunnel:
    tool = tool or available()
    if tool is None:
        raise RuntimeError("neither cloudflared nor ngrok is installed (brew install cloudflared)")
    if tool == "cloudflared":
        proc = subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        found: list[str] = []

        def reader() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m and not found:
                    found.append(m.group(0))

        threading.Thread(target=reader, daemon=True).start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not found and proc.poll() is None:
            time.sleep(0.2)
        if not found:
            proc.terminate()
            raise RuntimeError("cloudflared did not report a URL")
        return Tunnel(proc, found[0], tool)

    proc = subprocess.Popen(["ngrok", "http", str(port), "--log", "stdout"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1) as r:
                for t in json.load(r).get("tunnels", []):
                    if t.get("public_url", "").startswith("https://"):
                        return Tunnel(proc, t["public_url"], tool)
        except OSError:
            pass
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("ngrok did not start (run `ngrok config add-authtoken <token>` once)")
