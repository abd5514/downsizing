"""Build a YOLO detection dataset from FL3D.

FL3D ships pre-cropped faces (~210x290) with a driver_state label and five
RetinaFace landmarks. There are no bounding boxes and no full frames, so a
detector cannot be trained on it directly - yet the live client must find a
face inside a 640x480 webcam frame before it can judge anything.

The fix is compositing: each FL3D crop is pasted onto a synthesised in-cabin
background at a random scale, position and brightness, and the paste rectangle
becomes the ground-truth box. One YOLO model then does both jobs - locate the
driver's face, and classify its state - which is exactly what the client
expects.

A share of the raw crops is also emitted with a full-frame box, so the model
stays accurate on tight crops and can be scored directly against the real FL3D
test frames.

Splits are video-disjoint. FL3D's own val and test come from the same five
videos, so selecting a model on val would leak into test; here the five holdout
videos are divided between val and test instead.

Caveat worth stating in the report: FL3D exposes video ids, not driver ids, and
NITYMED recorded several videos per driver. Video-disjoint is the strongest
split obtainable from this data, but it is not guaranteed person-disjoint.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from drowsim.config import CLASSES, CLASS_TO_IDX

SRC = Path("data/fl3d/classification_frames")
DST = Path("data/yolo")


def load_annotations() -> tuple[dict, dict]:
    train = json.loads((SRC / "annotations_train.json").read_text())
    holdout = json.loads((SRC / "annotations_holdout.json").read_text())
    return train, holdout


def video_of(key: str) -> str:
    # keys look like ./classification_frames/P1043127_720/frame461.jpg
    return key.split("/")[2]


def frame_index(key: str) -> int:
    stem = Path(key).stem                       # frame461
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else 0


def choose_holdout_split(holdout_ann: dict, n_val: int) -> tuple[set[str], set[str]]:
    """Divide the five holdout videos between val and test.

    A random division is not safe here: only two of the five videos contain any
    microsleep frames at all, so an unlucky draw leaves validation unable to
    measure the most safety-critical class, and model selection goes blind to
    it. Instead every division is enumerated, those that fail to cover all
    three classes on both sides are discarded, and the survivor with the
    strongest worst-case class support is chosen.

    Test support is scored first, deliberately: the test set is the number the
    report quotes and the one an examiner will scrutinise, so a thin minority
    class there is far more damaging than a thin one in validation.
    """
    import itertools

    per_video: dict[str, Counter] = defaultdict(Counter)
    for key, rec in holdout_ann.items():
        per_video[video_of(key)][rec["driver_state"]] += 1

    videos = sorted(per_video)
    best = None
    for combo in itertools.combinations(videos, n_val):
        val_counts, test_counts = Counter(), Counter()
        for v in videos:
            (val_counts if v in combo else test_counts).update(per_video[v])
        if any(val_counts[c] == 0 or test_counts[c] == 0 for c in CLASSES):
            continue
        score = (min(test_counts[c] for c in CLASSES),
                 min(val_counts[c] for c in CLASSES))
        if best is None or score > best[0]:
            best = (score, set(combo), set(videos) - set(combo), val_counts, test_counts)

    if best is None:
        raise SystemExit(
            "No division of the holdout videos covers all three classes on both "
            "sides. Reduce --val-videos or evaluate on the full holdout."
        )
    _, val_v, test_v, val_counts, test_counts = best
    print(f"[prepare] val  frames by class: {dict(val_counts)}")
    print(f"[prepare] test frames by class: {dict(test_counts)}")
    return val_v, test_v


def make_background(size: tuple[int, int], donor: Image.Image, rng: random.Random):
    """A plausible dim cabin backdrop, built by destroying an unrelated crop.

    Real in-car backgrounds are not in the dataset, so this is a stand-in: it
    gives the detector textured, non-uniform surroundings to reject rather than
    a flat colour it can separate trivially.

    The detail has to be annihilated, not merely softened. A merely blurred
    face is still a face, and an unlabelled one in the background would punish
    the model for detecting it correctly - label noise pointed straight at the
    behaviour we want.
    """
    tiny = donor.resize((max(donor.width // 24, 2), max(donor.height // 24, 2)),
                        Image.BILINEAR)
    bg = tiny.resize(size, Image.BILINEAR)
    bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(18, 34)))
    bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.15, 0.45))
    bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.5, 0.9))
    return bg


def composite(face: Image.Image, donor: Image.Image, canvas: tuple[int, int],
              rng: random.Random):
    """Paste `face` onto a synthetic background; return the image and its box."""
    cw, ch = canvas
    bg = make_background(canvas, donor, rng)

    # Face occupies 18-60% of the canvas width: at the small end it matches a
    # driver sitting back from a dashboard camera, at the large end a close
    # webcam shot.
    target_w = int(cw * rng.uniform(0.18, 0.60))
    scale = target_w / face.width
    target_h = max(int(face.height * scale), 8)
    face_r = face.resize((target_w, target_h), Image.BILINEAR)

    if rng.random() < 0.5:
        face_r = ImageEnhance.Brightness(face_r).enhance(rng.uniform(0.7, 1.3))

    max_x = max(cw - target_w, 1)
    max_y = max(ch - target_h, 1)
    # Biased towards the upper middle, where a seated driver's head actually is.
    x = int(rng.triangular(0, max_x, max_x * 0.5))
    y = int(rng.triangular(0, max_y, max_y * 0.35))
    bg.paste(face_r, (x, y))

    cx = (x + target_w / 2) / cw
    cy = (y + target_h / 2) / ch
    return bg, (cx, cy, target_w / cw, target_h / ch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--canvas", type=int, default=480)
    ap.add_argument("--train-stride", type=int, default=2,
                    help="keep every Nth training frame; consecutive video "
                         "frames are near-duplicates")
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--crop-share", type=float, default=0.25,
                    help="fraction of frames also emitted as raw tight crops")
    ap.add_argument("--val-videos", type=int, default=2,
                    help="how many of the 5 holdout videos become val")
    ap.add_argument("--only", nargs="*", default=None,
                    help="regenerate only these splits (train/val/test)")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"{SRC} not found - run scripts/get_dataset.py first")
        return 1

    rng = random.Random(args.seed)
    train_ann, holdout_ann = load_annotations()

    val_videos, test_videos = choose_holdout_split(holdout_ann, args.val_videos)
    print(f"[prepare] val videos:  {sorted(val_videos)}")
    print(f"[prepare] test videos: {sorted(test_videos)}")

    buckets: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, rec in train_ann.items():
        buckets["train"].append((key, rec))
    for key, rec in holdout_ann.items():
        split = "val" if video_of(key) in val_videos else "test"
        buckets[split].append((key, rec))

    # Frames must stay in capture order: the BiLSTM sequences are built from
    # consecutive frames later, and shuffled input would make them meaningless.
    for split in buckets:
        buckets[split].sort(key=lambda kv: (video_of(kv[0]), frame_index(kv[0])))

    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_keys = [k for items in buckets.values() for k, _ in items]
    canvas = (args.canvas, args.canvas)
    written: Counter = Counter()
    per_class: dict[str, Counter] = defaultdict(Counter)
    missing = 0

    for split, items in buckets.items():
        if args.only and split not in args.only:
            continue
        stride = args.train_stride if split == "train" else args.eval_stride
        for n, (key, rec) in enumerate(items):
            if n % stride:
                continue
            src = Path("data/fl3d") / key.removeprefix("./")
            if not src.exists():
                missing += 1
                continue
            state = rec.get("driver_state")
            if state not in CLASS_TO_IDX:
                continue
            cls = CLASS_TO_IDX[state]

            try:
                face = Image.open(src).convert("RGB")
            except Exception:
                missing += 1
                continue

            video, idx = video_of(key), frame_index(key)
            donor_key = all_keys[rng.randrange(len(all_keys))]
            donor_path = Path("data/fl3d") / donor_key.removeprefix("./")
            try:
                donor = Image.open(donor_path).convert("RGB")
            except Exception:
                donor = face

            img, box = composite(face, donor, canvas, rng)
            stem = f"{video}_{idx:06d}"
            img.save(DST / "images" / split / f"{stem}.jpg", quality=88)
            (DST / "labels" / split / f"{stem}.txt").write_text(
                f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
            )
            written[split] += 1
            per_class[split][state] += 1

            # A slice of raw crops keeps the detector honest on tight faces and
            # lets us score it against unmodified FL3D frames.
            if rng.random() < args.crop_share:
                stem_c = f"{video}crop_{idx:06d}"
                face.save(DST / "images" / split / f"{stem_c}.jpg", quality=88)
                (DST / "labels" / split / f"{stem_c}.txt").write_text(
                    f"{cls} 0.500000 0.500000 0.980000 0.980000\n"
                )
                written[split] += 1
                per_class[split][state] += 1

        print(f"[prepare] {split}: {written[split]} images  {dict(per_class[split])}")

    (DST / "drowsim.yaml").write_text(
        f"# Generated by scripts/prepare_yolo.py (seed={args.seed}, video-disjoint)\n"
        f"path: {DST.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(CLASSES)}\nnames: {list(CLASSES)}\n"
    )
    if missing:
        print(f"[prepare] {missing} frames could not be read")
    print(f"[prepare] dataset yaml -> {DST / 'drowsim.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
