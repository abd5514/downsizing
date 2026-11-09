"""Render a held-out FL3D video into a webcam-like clip.

FL3D frames are tight crops, but the client expects a full scene, so each frame
is placed into a 640x480 canvas at a fixed position - a static dashboard camera,
not the random placement used for training. The result is a repeatable clip
containing real microsleep and yawning episodes, which is far more useful for
checking the rule engine than acting in front of a laptop.

The ground-truth label of every frame is written alongside, so a run can be
scored rather than eyeballed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

FL3D = Path("data/fl3d")
ANN = FL3D / "classification_frames" / "annotations_holdout.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="P1042762_720",
                    help="FL3D video id; P1042762_720 contains microsleep")
    ap.add_argument("--out", default="data/demo_clip.mp4")
    ap.add_argument("--labels-out", default="data/demo_clip_labels.json")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--scale", type=float, default=0.62,
                    help="face height as a fraction of frame height")
    args = ap.parse_args()

    holdout = json.loads(ANN.read_text())
    frames = sorted(
        ((k, v) for k, v in holdout.items() if k.split("/")[2] == args.video),
        key=lambda kv: int("".join(c for c in Path(kv[0]).stem if c.isdigit()) or 0),
    )[: args.max_frames]
    if not frames:
        print(f"no frames for video {args.video}")
        return 1

    W, H = 640, 480
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))

    # One fixed backdrop for the whole clip, as a real cabin camera would see.
    donor = Image.open(FL3D / frames[0][0].removeprefix("./")).convert("RGB")
    tiny = donor.resize((max(donor.width // 24, 2), max(donor.height // 24, 2)))
    backdrop = tiny.resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(28))
    backdrop = Image.eval(backdrop, lambda v: int(v * 0.3))

    labels = []
    for key, rec in frames:
        face = Image.open(FL3D / key.removeprefix("./")).convert("RGB")
        target_h = int(H * args.scale)
        target_w = max(int(face.width * target_h / face.height), 8)
        face = face.resize((target_w, target_h), Image.BILINEAR)

        canvas = backdrop.copy()
        canvas.paste(face, ((W - target_w) // 2, int(H * 0.12)))
        writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
        labels.append(rec["driver_state"])

    writer.release()
    Path(args.labels_out).write_text(json.dumps({"video": args.video,
                                                 "fps": args.fps,
                                                 "labels": labels}, indent=2))
    from collections import Counter
    print(f"wrote {args.out}: {len(labels)} frames at {args.fps} fps "
          f"({len(labels) / args.fps:.0f}s)")
    print(f"ground truth: {dict(Counter(labels))}")
    print(f"labels -> {args.labels_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
