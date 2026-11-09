"""Train the 3-class DrowSim YOLO detector.

Runs on Apple Silicon via MPS. yolo11n is the smallest model in the family and
the one the DROWSIM specification recommends: roughly 22% fewer parameters than
YOLOv8n at higher mAP.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser(description="Train DrowSim YOLO detector")
    p.add_argument("--data", default="data/yolo/drowsim.yaml")
    p.add_argument("--model", default="yolo11n.pt",
                   help="yolo11n.pt, yolov8n.pt, yolov9t.pt, yolov10n.pt - the "
                        "comparison table in the report needs one run each")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--name", default="drowsim_yolo11n")
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    if not Path(args.data).exists():
        print(f"{args.data} not found - run scripts/prepare_yolo.py first")
        return 1

    device = pick_device(args.device)
    print(f"[train] device={device} model={args.model} imgsz={args.imgsz}")

    model = YOLO(args.model)
    # project is joined onto Ultralytics' own runs_dir setting, so passing
    # "runs/detect" here yields runs/detect/runs/detect/<name>.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project="detect",
        patience=args.patience,
        # Night-time in-car footage: the driver is always upright and centred,
        # so flips and heavy geometric jitter teach the model nothing useful,
        # while brightness variation is exactly what changes between clips.
        fliplr=0.0,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        hsv_v=0.5,
        mosaic=0.3,
        plots=True,
    )

    # Ask the trainer where it actually wrote, rather than reconstructing the
    # path and silently copying nothing when the guess is wrong.
    save_dir = Path(getattr(model.trainer, "save_dir", Path("runs/detect") / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"[train] run directory: {save_dir}")
    if not best.exists():
        found = sorted(Path("runs").rglob(f"{args.name}*/weights/best.pt"))
        if found:
            best = found[-1]
            print(f"[train] located weights at {best}")
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        target = Path("models") / f"{args.name}.pt"
        shutil.copy(best, target)
        print(f"\n[train] best weights -> {target}")
        metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, device=device)
        print(f"[train] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    else:
        print(f"[train] WARNING: no best.pt found under {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
