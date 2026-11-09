"""Score the trained detector on unmodified FL3D test frames.

Detection mAP on the composites is not the headline number and should not be
quoted as one: the composite task pastes a sharp face onto a featureless blur,
so localisation is close to trivial and mAP comes out near-perfect regardless
of how well driver state is actually recognised.

This script instead runs the detector over the raw FL3D crops from the held-out
test videos and scores the predicted *class* against the ground-truth label.
That is directly comparable to the per-class table in the original thesis, and
it is the number to put in the report.

Frames where the detector finds nothing are reported separately rather than
being quietly dropped - a miss is a real failure mode, and folding it into the
class metrics would hide it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from drowsim.config import CLASSES, CLASS_TO_IDX
from scripts.prepare_yolo import choose_holdout_split

FL3D = Path("data/fl3d")
ANN = FL3D / "classification_frames" / "annotations_holdout.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/drowsim_yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-videos", type=int, default=2)
    ap.add_argument("--out", default="runs/detector_test_report.json")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"{args.weights} not found - train the detector first")
        return 1

    from ultralytics import YOLO

    holdout = json.loads(ANN.read_text())
    _, test_videos = choose_holdout_split(holdout, args.val_videos)
    print(f"[eval] test videos: {sorted(test_videos)}")

    items = [
        (FL3D / k.removeprefix("./"), CLASS_TO_IDX[v["driver_state"]])
        for k, v in holdout.items()
        if k.split("/")[2] in test_videos and v["driver_state"] in CLASS_TO_IDX
    ]
    items.sort(key=lambda t: str(t[0]))
    print(f"[eval] {len(items)} raw FL3D test frames")

    model = YOLO(args.weights)
    y_true: list[int] = []
    y_pred: list[int] = []
    undetected = Counter()

    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        results = model.predict([str(p) for p, _ in chunk], imgsz=args.imgsz,
                                conf=args.conf, verbose=False)
        for (_, truth), res in zip(chunk, results):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                undetected[CLASSES[truth]] += 1
                continue
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            y_true.append(truth)
            y_pred.append(int(cls[int(np.argmax(conf))]))

    total = len(items)
    missed = sum(undetected.values())
    print(f"\n[eval] no detection on {missed}/{total} frames ({missed / total:.2%})"
          f"  {dict(undetected)}")

    print("\n=== Per-class performance on raw FL3D test frames ===")
    print(classification_report(y_true, y_pred, target_names=list(CLASSES),
                                digits=4, zero_division=0))
    print("confusion matrix (rows = true, cols = predicted):")
    print(f"{'':<12}" + "".join(f"{c:>12}" for c in CLASSES))
    for name, row in zip(CLASSES, confusion_matrix(y_true, y_pred)):
        print(f"{name:<12}" + "".join(f"{v:>12}" for v in row))

    report = classification_report(y_true, y_pred, target_names=list(CLASSES),
                                   output_dict=True, zero_division=0)
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    report["confusion_labels"] = list(CLASSES)
    report["undetected_frames"] = missed
    report["undetected_rate"] = missed / total
    report["undetected_by_class"] = dict(undetected)
    report["total_frames"] = total
    report["test_videos"] = sorted(test_videos)
    report["weights"] = args.weights

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[eval] accuracy {report['accuracy']:.4f}  "
          f"macro-F1 {report['macro avg']['f1-score']:.4f}")
    print(f"[eval] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
