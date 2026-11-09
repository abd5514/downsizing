"""Wrap each per-class recording with a title card and a caption.

One short, self-contained video per driver state, so each can be shown on its
own without narration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

SRC = Path("deliverables/class_clips")
W, H, FPS = 640, 480, 10
BG = (22, 24, 28)
FG = (238, 238, 238)
MUTED = (150, 155, 165)
OK = (110, 210, 130)
WARN = (70, 190, 240)
CRIT = (70, 70, 235)

SPEC = {
    "alert": (OK, "Alert", "Eyes open, normal mouth position", "73.0% of the dataset",
              "State holds at NORMAL throughout. No alarm, no false positive."),
    "microsleep": (CRIT, "Microsleep", "Eyes closed, involuntary lapse of consciousness",
                   "16.7% of the dataset",
                   "Eyes closed past 0.8 s -> CRITICAL, raised by the deterministic rule."),
    "yawning": (WARN, "Yawning", "Mouth wide open in a yawn gesture", "10.3% of the dataset",
                "Sustained mouth opening -> WARNING, raised by the BiLSTM on the server."),
}


def card(lines, seconds=3.5):
    frame = np.full((H, W, 3), BG, np.uint8)
    total = sum(int(40 * s) + 12 for _, s, _, _ in lines)
    y = (H - total) // 2 + 30
    for text, scale, colour, thick in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(frame, text, ((W - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thick, cv2.LINE_AA)
        y += int(40 * scale) + 12
    return [frame] * int(seconds * FPS)


def caption(frames, text, colour):
    """Drawn below the client's own header, never over its status lines."""
    out = []
    for f in frames:
        f = f.copy()
        overlay = f.copy()
        cv2.rectangle(overlay, (0, 74), (W, 104), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.8, f, 0.2, 0, f)
        cv2.putText(f, text, (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1,
                    cv2.LINE_AA)
        out.append(f)
    return out


def read(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def transcode(path):
    tool = "/usr/bin/avconvert"
    if not Path(tool).exists():
        return
    tmp = str(Path(path).with_suffix(".h264.mp4"))
    r = subprocess.run([tool, "--source", str(path), "--output", tmp,
                        "--preset", "PresetHighestQuality"],
                       capture_output=True, text=True)
    if r.returncode == 0 and Path(tmp).exists():
        shutil.move(tmp, path)
    else:
        Path(tmp).unlink(missing_ok=True)


def main() -> int:
    for cls, (colour, title, desc, share, note) in SPEC.items():
        raw = SRC / f"raw_{cls}.mp4"
        if not raw.exists():
            print(f"  ! missing {raw}")
            continue
        labels = json.loads((Path("data/class_clips") / f"clip_{cls}_labels.json").read_text())
        n_cls = sum(1 for l in labels["labels"] if l == cls)

        seq = card([
            (title, 1.4, colour, 3),
            ("", 0.35, FG, 1),
            (desc, 0.55, FG, 1),
            (share, 0.48, MUTED, 1),
            ("", 0.3, FG, 1),
            (f"{n_cls / labels['fps']:.1f}s of labelled {cls} in this clip", 0.44, MUTED, 1),
        ], 4.0)
        seq += caption(read(raw), note, colour)

        out = SRC / f"DrowSim_{cls}.mp4"
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in seq:
            writer.write(f)
        writer.release()
        transcode(out)
        print(f"  {out.name}: {len(seq) / FPS:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
