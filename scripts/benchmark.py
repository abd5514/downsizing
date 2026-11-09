"""Measure the numbers the report's performance table needs, on this machine.

Every figure printed here is measured, not quoted. Run it after training and
paste the output into the thesis - and if a value fails its requirement, say so
in the report rather than adjusting the requirement.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import requests

from drowsim import config as cfg


def rss_mb() -> float:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / (1024 * 1024) if os.uname().sysname == "Darwin" else peak / 1024


def bench_yolo(weights: str, imgsz: int, n: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(5):                      # warm-up; the first call is an outlier
        model.predict(frame, imgsz=imgsz, verbose=False)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(times)
    return {"mean": a.mean(), "std": a.std(), "p95": np.percentile(a, 95), "max": a.max()}


def bench_server(base: str, n: int) -> dict:
    session_id = uuid.uuid4().hex[:12]
    row = [0.9, 0.05, 0.05, 0.85, 0.12, 0.5, 0.45]
    times, failures = [], 0
    for seq in range(n):
        payload = {
            "client_id": "BENCH", "session_id": session_id, "seq": seq,
            "timestamp": seq / cfg.TELEMETRY_HZ, "fps": cfg.TARGET_FPS,
            "features": [row] * 3, "client_state": "NORMAL",
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(base.rstrip("/") + cfg.INFER_PATH, json=payload, timeout=5)
            if r.status_code != 200:
                failures += 1
            times.append((time.perf_counter() - t0) * 1000.0)
        except requests.RequestException:
            failures += 1
    a = np.array(times) if times else np.array([0.0])
    return {"mean": a.mean(), "p95": np.percentile(a, 95), "max": a.max(),
            "loss": failures / max(n, 1)}


def row(metric: str, value: str, requirement: str, ok: bool) -> str:
    return f"| {metric:<38} | {value:<22} | {requirement:<14} | {'PASS' if ok else 'FAIL'} |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=cfg.YOLO_IMG_SIZE)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--n-yolo", type=int, default=100)
    ap.add_argument("--n-net", type=int, default=500)
    args = ap.parse_args()

    print(f"Benchmarking with weights={args.weights} imgsz={args.imgsz}\n")
    y = bench_yolo(args.weights, args.imgsz, args.n_yolo)
    try:
        s = bench_server(args.server, args.n_net)
        server_up = True
    except Exception:
        s, server_up = {"mean": 0, "p95": 0, "max": 0, "loss": 1.0}, False

    fps = 1000.0 / y["mean"]
    end_to_end = y["mean"] + s["mean"]
    mem = rss_mb()

    print(f"| {'Metric':<38} | {'Measurement':<22} | {'Requirement':<14} | Status |")
    print(f"|{'-' * 40}|{'-' * 24}|{'-' * 16}|--------|")
    print(row("Mean YOLO inference latency",
              f"{y['mean']:.2f} +/- {y['std']:.2f} ms", "< 100 ms", y["mean"] < 100))
    print(row("Max YOLO inference latency", f"{y['max']:.2f} ms", "< 500 ms", y["max"] < 500))
    print(row("Achievable capture rate", f"{fps:.1f} FPS", ">= 10 FPS", fps >= 10))
    if server_up:
        print(row("Telemetry round-trip (local)", f"{s['mean']:.2f} ms", "< 500 ms",
                  s["mean"] < 500))
        print(row("End-to-end alert latency", f"{end_to_end:.2f} ms", "< 500 ms",
                  end_to_end < 500))
        print(row(f"Packet loss over {args.n_net} requests", f"{s['loss']:.1%}", "< 5%",
                  s["loss"] < 0.05))
    else:
        print(f"| {'Server benchmarks':<38} | {'server unreachable':<22} | "
              f"{'-':<14} | SKIP   |")
    print(row("Peak client RSS", f"{mem:.0f} MB", "< 500 MB", mem < 500))

    print("\nNote: the alert itself is raised locally by the rule engine, so the "
          "driver-facing latency is the YOLO figure alone; the round-trip only "
          "delays the dashboard and the model's contextual verdict.")

    # Written out so the slides and the report quote the measured figures rather
    # than numbers typed in by hand, which is how they go stale.
    out = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "yolo_mean_ms": round(y["mean"], 2),
        "yolo_std_ms": round(y["std"], 2),
        "yolo_max_ms": round(y["max"], 2),
        "achievable_fps": round(fps, 1),
        "roundtrip_mean_ms": round(s["mean"], 2) if server_up else None,
        "end_to_end_ms": round(end_to_end, 2) if server_up else None,
        "packet_loss": round(s["loss"], 4) if server_up else None,
        "peak_rss_mb": round(mem),
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/benchmark.json").write_text(json.dumps(out, indent=2))
    print("saved -> runs/benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
