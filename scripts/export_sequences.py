"""Run the trained YOLO detector over the dataset and emit BiLSTM training
sequences.

The point of doing it this way, rather than building sequences from ground
truth, is that the temporal model then trains on the *same distribution it will
see at inference* - including the detector's own mistakes. A BiLSTM trained on
perfect labels and deployed on noisy ones is a well-known way to get a great
table and a disappointing demo.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from drowsim.config import SEQ_LEN
from drowsim.features import empty_features, features_from_boxes

DST = Path("data/sequences")


def frames_for_split(root: Path, split: str) -> dict[str, list[tuple[int, Path, int]]]:
    """Group frames by person and keep them in capture order."""
    by_person: dict[str, list[tuple[int, Path, int]]] = defaultdict(list)
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # The "<video>crop_<idx>" images are tight-crop duplicates of frames
        # already present as composites. Including them would interleave two
        # copies of the same moment into one sequence.
        if "crop_" in img.stem:
            continue
        label_file = lbl_dir / f"{img.stem}.txt"
        if not label_file.exists():
            continue
        cls = int(label_file.read_text().split()[0])
        person, _, idx = img.stem.rpartition("_")
        by_person[person].append((int(idx), img, cls))
    for person in by_person:
        by_person[person].sort(key=lambda t: t[0])
    return by_person


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/drowsim_yolo11n.pt")
    ap.add_argument("--data", default="data/yolo")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--stride", type=int, default=4,
                    help="window hop; 1 gives maximum overlap and a big file")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"{args.weights} not found - train the detector first "
              f"(scripts/train_yolo.py)")
        return 1

    from ultralytics import YOLO

    model = YOLO(args.weights)
    root = Path(args.data)
    DST.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        by_person = frames_for_split(root, split)
        if not by_person:
            print(f"[export] {split}: no frames, skipping")
            continue

        xs, ys = [], []
        for person, frames in by_person.items():
            vectors, labels = [], []
            for start in range(0, len(frames), args.batch):
                chunk = frames[start:start + args.batch]
                images = [str(p) for _, p, _ in chunk]
                results = model.predict(images, imgsz=args.imgsz, verbose=False)
                for (_, path, cls), res in zip(chunk, results):
                    with Image.open(path) as im:
                        w, h = im.size
                    boxes = res.boxes
                    feats = (
                        features_from_boxes(boxes, w, h)
                        if boxes is not None and len(boxes)
                        else empty_features()
                    )
                    vectors.append(feats.vector)
                    labels.append(cls)

            for i in range(0, len(vectors) - SEQ_LEN + 1, args.stride):
                window = np.stack(vectors[i:i + SEQ_LEN])
                # The window is labelled by its final frame: that is the instant
                # the model is asked to classify at inference time.
                xs.append(window)
                ys.append(labels[i + SEQ_LEN - 1])

            print(f"[export] {split}/{person}: {len(frames)} frames")

        if not xs:
            print(f"[export] {split}: too few frames for a {SEQ_LEN}-frame window")
            continue
        x = np.stack(xs).astype(np.float32)
        y = np.asarray(ys, dtype=np.int64)
        np.savez_compressed(DST / f"{split}.npz", x=x, y=y)
        print(f"[export] {split}: {x.shape} -> {DST / f'{split}.npz'} "
              f"class counts {np.bincount(y, minlength=3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
