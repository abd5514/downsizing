"""Record the monitoring dashboard by driving headless Chrome over CDP.

Chrome's one-shot `--screenshot` flag writes the file but then refuses to exit,
so launching it once per frame is not workable. Instead one browser is started
with remote debugging enabled and Page.captureScreenshot is called repeatedly
over the DevTools protocol, which also keeps the page's own polling alive
between frames - the dashboard updates on screen exactly as a viewer would see
it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE = "/tmp/chrome_drowsim_cdp"


def start_chrome(url: str, port: int, width: int, height: int) -> subprocess.Popen:
    shutil.rmtree(PROFILE, ignore_errors=True)
    return subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--no-first-run", "--no-default-browser-check",
         f"--remote-debugging-port={port}", f"--user-data-dir={PROFILE}",
         f"--window-size={width},{height}", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def devtools_url(port: int, timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1) as r:
                for tab in json.load(r):
                    if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl"):
                        return tab["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError("Chrome did not expose a debuggable page")


async def capture(ws_url: str, out_dir: Path, fps: float, seconds: float) -> int:
    interval = 1.0 / fps
    n = int(seconds * fps)
    saved = 0
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        for i in range(n):
            started = time.monotonic()
            await ws.send(json.dumps({"id": i + 1, "method": "Page.captureScreenshot",
                                      "params": {"format": "png"}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == i + 1:
                    break
            data = msg.get("result", {}).get("data")
            if data:
                (out_dir / f"dash_{i:05d}.png").write_bytes(base64.b64decode(data))
                saved += 1
            delay = interval - (time.monotonic() - started)
            if delay > 0:
                await asyncio.sleep(delay)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011/")
    ap.add_argument("--out", default="data/dashboard_frames")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--port", type=int, default=9222)
    args = ap.parse_args()

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    proc = start_chrome(args.url, args.port, args.width, args.height)
    try:
        ws_url = devtools_url(args.port)
        print(f"[capture] connected; recording {args.seconds:.0f}s at {args.fps} fps")
        saved = asyncio.run(capture(ws_url, out, args.fps, args.seconds))
        print(f"[capture] {saved} frames -> {out}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(PROFILE, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
