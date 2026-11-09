"""Train and score each YOLO generation on identical data and settings.

The DROWSIM specification asks for a v8/v9/v10/v11 comparison. A table of
numbers copied from the Ultralytics release notes would say nothing about this
task, so each model is trained here on the same composites, with the same
hyperparameters and seed, and scored the same way: predicted class against
ground truth on the raw, unmodified FL3D test frames.

Already-trained models are skipped, so the sweep can be interrupted and resumed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODELS = [
    ("yolov8n.pt", "drowsim_yolov8n", "Anchor-free, decoupled head"),
    ("yolov9t.pt", "drowsim_yolov9t", "PGI + GELAN backbone"),
    ("yolov10n.pt", "drowsim_yolov10n", "NMS-free dual assignment"),
    ("yolo11n.pt", "drowsim_yolo11n", "C3k2 + C2PSA attention"),
]


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out", default="runs/model_comparison.json")
    args = ap.parse_args()

    py = sys.executable
    rows = []

    for weights, name, innovation in MODELS:
        target = Path("models") / f"{name}.pt"
        report_path = Path("runs") / f"{name}_test_report.json"

        if not target.exists():
            started = time.monotonic()
            rc = run([py, "scripts/train_yolo.py", "--model", weights,
                      "--name", name, "--epochs", str(args.epochs),
                      "--batch", str(args.batch), "--imgsz", str(args.imgsz)])
            train_minutes = (time.monotonic() - started) / 60.0
            if rc != 0 or not target.exists():
                print(f"[compare] {name}: training failed (rc={rc}), skipping")
                rows.append({"model": weights, "name": name,
                             "innovation": innovation, "error": f"train rc={rc}"})
                continue
        else:
            print(f"[compare] {name}: weights already present, skipping training")
            train_minutes = None

        if not report_path.exists():
            rc = run([py, "scripts/evaluate.py", "--weights", str(target),
                      "--imgsz", str(args.imgsz), "--out", str(report_path)])
            if rc != 0:
                rows.append({"model": weights, "name": name,
                             "innovation": innovation, "error": f"eval rc={rc}"})
                continue

        report = json.loads(report_path.read_text())

        from scripts.benchmark import bench_yolo
        from ultralytics import YOLO

        # How many epochs this model actually received. Training budgets differ
        # across the sweep (some runs early-stopped, one was interrupted), and
        # a comparison that hides that is not a comparison.
        epochs_trained = None
        csvs = sorted(Path("runs").rglob(f"*{name}*/results.csv"))
        if csvs:
            lines = [l for l in csvs[-1].read_text().splitlines() if l.strip()]
            epochs_trained = max(len(lines) - 1, 0)

        latency = bench_yolo(str(target), args.imgsz, 100)
        params = sum(p.numel() for p in YOLO(str(target)).model.parameters())

        rows.append({
            "model": weights,
            "name": name,
            "innovation": innovation,
            "parameters": params,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "microsleep_f1": report["microsleep"]["f1-score"],
            "yawning_f1": report["yawning"]["f1-score"],
            "undetected_rate": report["undetected_rate"],
            "latency_ms_mean": round(latency["mean"], 2),
            "latency_ms_p95": round(latency["p95"], 2),
            "train_minutes": round(train_minutes, 1) if train_minutes else None,
            "epochs_trained": epochs_trained,
        })
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))

    print("\n\n=== YOLO generation comparison ===")
    print("Identical data, hyperparameters and seed; scored on 3,810 raw FL3D "
          "test frames.\n")
    header = (f"| {'Model':<12} | {'Params':>9} | {'Epochs':>6} | {'Accuracy':>8} | "
              f"{'Macro-F1':>8} | {'Micro-F1':>8} | {'Latency':>9} |")
    print(header)
    print("|" + "-" * 14 + "|" + "-" * 11 + "|" + "-" * 8 + "|" + "-" * 10 +
          "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 11 + "|")
    for r in rows:
        if "error" in r:
            print(f"| {r['model']:<12} | {'-':>9} | {'-':>6} | {'FAILED':>8} | "
                  f"{'-':>8} | {'-':>8} | {'-':>9} |")
            continue
        print(f"| {r['model']:<12} | {r['parameters']:>9,} | "
              f"{str(r['epochs_trained'] or '?'):>6} | {r['accuracy']:>8.4f} | "
              f"{r['macro_f1']:>8.4f} | {r['microsleep_f1']:>8.4f} | "
              f"{r['latency_ms_mean']:>6.2f} ms |")
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
