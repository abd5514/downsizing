"""Render three videos per driver state from the site's own demo replay.

For each of ALERT, YAWNING and MICROSLEEP this writes:
  driver_<state>.mp4        the demo clip with ground truth and the detector's scores
  dashboard_<state>.mp4     the monitoring dashboard while that stretch replays
  side_by_side_<state>.mp4  both together, in step

The dashboard is recorded from a local server running the same code and the same
demo file the deployed site replays, so it shows exactly what a visitor sees.
Frame f of the clip is paired with the dashboard at loop start + f's packet time.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_dashboard import devtools_url, start_chrome  # noqa: E402

import websockets  # noqa: E402

CLIP = Path("data/demo_clip.mp4")
LABELS = Path("data/demo_clip_labels.json")
DEMO = Path("server/demo_telemetry.jsonl")
FPS = 10
DASH_FPS = 5
DASH_W, DASH_H = 1280, 720

GREEN, AMBER, RED = (109, 207, 62), (68, 181, 245), (68, 68, 239)
BG, FG, MUTED = (21, 17, 15), (237, 234, 232), (178, 163, 154)
CLASSES = [("Alert", 0, GREEN), ("Yawning", 2, AMBER), ("Microsleep", 1, RED)]

SEGMENTS = {
    "yawning": (5, 68, "YAWNING -> WARNING",
                "The model flags the yawn a few seconds in; the badge turns amber."),
    "alert": (82, 152, "ALERT -> NORMAL",
              "Eyes open and attentive: the alert score stays high, the badge stays green."),
    "microsleep": (135, 235, "MICROSLEEP -> CRITICAL",
                   "Eyes closed for 0.8 s: the rule raises CRITICAL at once."),
}


def frame_times(packets: list[dict]) -> np.ndarray:
    """Loop-relative time at which each clip frame reaches the server."""
    t0 = packets[0]["timestamp"]
    times = []
    for p in packets:
        times.extend([p["timestamp"] - t0] * len(p["features"]))
    return np.asarray(times)


def frame_scores(packets: list[dict]) -> np.ndarray:
    return np.asarray([row for p in packets for row in p["features"]], dtype=np.float32)


def wait_health(url: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url + "/health", timeout=1).read()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


async def record(ws_url: str, seconds: float) -> list[tuple[float, bytes]]:
    shots = []
    interval = 1.0 / DASH_FPS
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Emulation.setDeviceMetricsOverride",
                                  "params": {"width": DASH_W, "height": DASH_H,
                                             "deviceScaleFactor": 1, "mobile": False}}))
        while json.loads(await ws.recv()).get("id") != 1:
            pass
        msg_id = 1
        end = time.time() + seconds
        while time.time() < end:
            started = time.monotonic()
            msg_id += 1
            taken = time.time()
            await ws.send(json.dumps({"id": msg_id, "method": "Page.captureScreenshot",
                                      "params": {"format": "png"}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == msg_id:
                    break
            data = msg.get("result", {}).get("data")
            if data:
                shots.append((taken, base64.b64decode(data)))
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))
    return shots


def loop_starts(base: str) -> list[float]:
    with urllib.request.urlopen(base + "/events?limit=500", timeout=5) as r:
        ev = json.load(r)["events"]
    starts = [e["timestamp"] for e in ev
              if e["client_id"] == "DEMO_REPLAY" and e["state"] == "NORMAL"
              and str(e["source"]).lower() == "rule"]
    return sorted(starts)


def writer(path: Path, w: int, h: int) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )


def caption(img: np.ndarray, title: str, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.full((70, w, 3), BG, np.uint8)
    cv2.putText(bar, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, FG, 2, cv2.LINE_AA)
    cv2.putText(bar, text, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MUTED, 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def annotate_driver(frame: np.ndarray, truth: str, scores: np.ndarray) -> np.ndarray:
    img = frame.copy()
    panel = img[0:118, 0:300]
    img[0:118, 0:300] = (panel * 0.35 + np.array(BG) * 0.65).astype(np.uint8)
    cv2.putText(img, f"Ground truth: {truth}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, FG, 1, cv2.LINE_AA)
    for i, (name, idx, colour) in enumerate(CLASSES):
        y = 48 + i * 24
        value = float(np.clip(scores[idx], 0, 1))
        cv2.putText(img, name, (12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)
        cv2.rectangle(img, (100, y - 5), (240, y + 5), (52, 42, 37), -1)
        cv2.rectangle(img, (100, y - 5), (100 + int(140 * value), y + 5), colour, -1)
        cv2.putText(img, f"{value * 100:3.0f}%", (248, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, FG, 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="deliverables/feature_videos")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--cdp-port", type=int, default=9223)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    packets = [json.loads(l) for l in DEMO.read_text().splitlines() if l.strip()]
    times, scores = frame_times(packets), frame_scores(packets)
    truth = json.loads(LABELS.read_text())["labels"]
    cap = cv2.VideoCapture(str(CLIP))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    assert len(frames) == len(scores) == len(truth), "clip, telemetry and labels disagree"

    base = f"http://127.0.0.1:{args.port}"
    env = {k: v for k, v in os.environ.items() if k != "DROWSIM_PASSWORD"}
    env["PYTHONPATH"] = "."
    server = subprocess.Popen(
        [sys.executable, "server/server.py", "--host", "127.0.0.1", "--port", str(args.port),
         "--bilstm", "models/bilstm_attention.pt", "--demo", str(DEMO)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    chrome = None
    try:
        wait_health(base)
        chrome = start_chrome(base + "/dashboard", args.cdp_port, DASH_W, DASH_H)
        ws_url = devtools_url(args.cdp_port)
        loop_len = times[-1] + 3.0
        max_end = max(e for _, e, *_ in SEGMENTS.values()) / FPS
        print("recording the dashboard across one full demo loop ...")
        shots = asyncio.run(record(ws_url, loop_len + max_end + 8))
        starts = [s for s in loop_starts(base) if s > shots[0][0] + 1]
    finally:
        if chrome:
            chrome.terminate()
        server.terminate()
    if not starts:
        print("no complete demo loop was captured")
        return 1
    t0 = starts[0]
    shot_times = np.asarray([t for t, _ in shots])
    decoded: dict[int, np.ndarray] = {}

    def dash_at(t: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(shot_times, t), 0, len(shots) - 1))
        if i not in decoded:
            img = cv2.imdecode(np.frombuffer(shots[i][1], np.uint8), cv2.IMREAD_COLOR)
            if img.shape[:2] != (DASH_H, DASH_W):
                img = cv2.resize(img, (DASH_W, DASH_H), interpolation=cv2.INTER_AREA)
            decoded[i] = img
        return decoded[i]

    for state, (start, end, title, text) in SEGMENTS.items():
        drv = writer(out / f"driver_{state}.mp4", 640, 480 + 70)
        dsh = writer(out / f"dashboard_{state}.mp4", DASH_W, DASH_H + 70)
        sbs = writer(out / f"side_by_side_{state}.mp4", 960 + DASH_W, DASH_H + 70)
        for f in range(start, end):
            d = annotate_driver(frames[f], truth[f].upper(), scores[f])
            board = dash_at(t0 + times[f])
            drv.stdin.write(caption(d, title, text).tobytes())
            dsh.stdin.write(caption(board, title, text).tobytes())
            big = cv2.resize(d, (960, DASH_H), interpolation=cv2.INTER_LINEAR)
            sbs.stdin.write(caption(np.hstack([big, board]), title, text).tobytes())
        for p in (drv, dsh, sbs):
            p.stdin.close()
            p.wait()
        print(f"{state}: {(end - start) / FPS:.1f} s written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
