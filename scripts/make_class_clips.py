"""Render one demonstration clip per driver state.

Each clip is cut from real labelled FL3D footage around the longest genuine run
of the target class, with a lead-in before it and a tail after. The lead-in is
not padding: the BiLSTM needs 32 frames before it will classify anything, and a
clip that opens on the event itself would show the system apparently missing it.

The frames come from held-out videos the detector never trained on, so these are
demonstrations rather than recitals.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

FL3D = Path("data/fl3d")
ANN = FL3D / "classification_frames" / "annotations_holdout.json"
W, H, FPS = 640, 480, 10


def runs_of(frames, target):
    """Contiguous stretches of `target`, as (start_index, end_index)."""
    out, start = [], None
    for i, (_, state) in enumerate(frames + [(None, None)]):
        if state == target and start is None:
            start = i
        elif state != target and start is not None:
            out.append((start, i))
            start = None
    return out


def render(frames, out_path, scale=0.62):
    backdrop_src = Image.open(FL3D / frames[0][0].removeprefix("./")).convert("RGB")
    tiny = backdrop_src.resize((max(backdrop_src.width // 24, 2),
                                max(backdrop_src.height // 24, 2)))
    backdrop = tiny.resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(28))
    backdrop = Image.eval(backdrop, lambda v: int(v * 0.3))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for key, _ in frames:
        face = Image.open(FL3D / key.removeprefix("./")).convert("RGB")
        th = int(H * scale)
        tw = max(int(face.width * th / face.height), 8)
        canvas = backdrop.copy()
        canvas.paste(face.resize((tw, th), Image.BILINEAR), ((W - tw) // 2, int(H * 0.12)))
        writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/class_clips")
    ap.add_argument("--lead-in", type=float, default=9.0,
                    help="seconds of context before the event")
    ap.add_argument("--tail", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout = json.loads(ANN.read_text())

    by_video: dict[str, list] = {}
    for k, v in holdout.items():
        by_video.setdefault(k.split("/")[2], []).append((k, v["driver_state"]))
    for vid in by_video:
        by_video[vid].sort(key=lambda kv: int("".join(c for c in Path(kv[0]).stem
                                                      if c.isdigit()) or 0))

    lead = int(args.lead_in * FPS)
    tail = int(args.tail * FPS)

    for cls in ("alert", "microsleep", "yawning"):
        best = None
        for vid, frames in by_video.items():
            for s, e in runs_of(frames, cls):
                # Need enough preceding footage to give the model its window.
                if s < lead:
                    continue
                if best is None or (e - s) > (best[2] - best[1]):
                    best = (vid, s, e)
        if best is None:
            print(f"  ! no usable run for {cls}")
            continue

        vid, s, e = best
        frames = by_video[vid]
        if cls == "alert":
            # A long alert stretch needs no event framing; take a steady slice.
            lo, hi = s, min(s + int(18 * FPS), e)
        else:
            lo, hi = max(0, s - lead), min(len(frames), e + tail)
        clip = frames[lo:hi]

        path = out_dir / f"clip_{cls}.mp4"
        render(clip, path)
        labels = Counter(st for _, st in clip)
        print(f"  {path.name}: {len(clip)} frames ({len(clip)/FPS:.1f}s) "
              f"from {vid}, ground truth {dict(labels)}")
        (out_dir / f"clip_{cls}_labels.json").write_text(
            json.dumps({"video": vid, "fps": FPS,
                        "labels": [st for _, st in clip]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
