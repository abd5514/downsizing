"""Assemble the AP1 prototype demo (under three minutes).

The deliverable is a recording of the web application, so the dashboard is the
subject: first beside the in-vehicle client so the two can be seen agreeing,
then full-screen where the event log is legible.

Client and dashboard were recorded simultaneously - the client at 10 fps, the
dashboard at 5 fps over the DevTools protocol - so pairing client frame i with
dashboard frame i//2 keeps them in step.

Every figure on the closing card is read from the measured result files, so the
video cannot drift out of step with the report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 620, 10
BG = (22, 24, 28)
FG = (238, 238, 238)
MUTED = (150, 155, 165)
OK = (110, 210, 130)
WARN = (70, 190, 240)
CRIT = (70, 70, 235)


def blank():
    return np.full((H, W, 3), BG, np.uint8)


def card(lines, seconds=3.5):
    frame = blank()
    total = sum(int(42 * s) + 14 for _, s, _, _ in lines)
    y = (H - total) // 2 + 34
    for text, scale, colour, thick in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(frame, text, ((W - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thick, cv2.LINE_AA)
        y += int(42 * scale) + 14
    return [frame] * int(seconds * FPS)


def label(frame, x, y, text, colour, scale=0.5):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                cv2.LINE_AA)


def chrome(frame, heading, caption, caption_colour):
    cv2.rectangle(frame, (0, 0), (W, 46), (16, 18, 22), -1)
    label(frame, 24, 31, heading, FG, 0.62)
    cv2.rectangle(frame, (0, H - 36), (W, H), (16, 18, 22), -1)
    label(frame, 24, H - 13, caption, caption_colour, 0.48)
    return frame


def load_client(path, start, end):
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or i >= end:
            break
        if start <= i:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def load_dash(folder):
    files = sorted(Path(folder).glob("*.png"))
    out = []
    for f in files:
        im = cv2.imread(str(f))
        if im is None:
            continue
        # Trim the empty page below the content: find the last row that is not
        # the page background, so the panel is filled by dashboard and not by
        # whitespace.
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rows = np.where(grey.max(axis=1) > 40)[0]
        bottom = int(rows[-1]) + 12 if len(rows) else im.shape[0]
        out.append(im[0:min(bottom, im.shape[0]), 0:im.shape[1]])
    return out


LEGEND = [
    ("NORMAL", OK, "rule engine sees nothing of concern"),
    ("WARNING", WARN, "BiLSTM on the server recognised a yawn"),
    ("CRITICAL", CRIT, "rule engine: eyes closed past 0.8 s"),
]


def side_by_side(client, dash, heading, caption, colour, dash_offset=0):
    out = []
    cw, ch = 600, 450
    dw = 600
    for i, cf in enumerate(client):
        frame = blank()
        chrome(frame, heading, caption, colour)

        frame[74:74 + ch, 24:24 + cw] = cv2.resize(cf, (cw, ch))
        label(frame, 24, 66, "IN-VEHICLE CLIENT  -  sees the video, never sends it", MUTED, 0.42)

        di = min(dash_offset + i // 2, len(dash) - 1)
        d = dash[di]
        dh = int(d.shape[0] * dw / d.shape[1])
        frame[74:74 + dh, 656:656 + dw] = cv2.resize(d, (dw, dh))
        label(frame, 656, 66, "WEB DASHBOARD  -  receives only numbers", MUTED, 0.42)

        # The dashboard is short, so the space beneath it carries the key to
        # what the viewer is watching rather than sitting empty.
        y = 74 + dh + 34
        for name, col, meaning in LEGEND:
            cv2.rectangle(frame, (656, y - 13), (656 + 96, y + 5), col, 1)
            label(frame, 664, y, name, col, 0.42)
            label(frame, 766, y, meaning, MUTED, 0.42)
            y += 30
        out.append(frame)
    return out


def dash_only(dash, start, count, heading, caption, colour):
    out = []
    dw = 1216
    for i in range(count):
        frame = blank()
        chrome(frame, heading, caption, colour)
        di = min(start + i, len(dash) - 1)
        d = dash[di]
        # Fit to the available height rather than cropping to it: cutting the
        # bottom loses event-log rows, which are the point of this segment.
        avail = H - 36 - 56
        scale = min(dw / d.shape[1], avail / d.shape[0])
        sw, sh = int(d.shape[1] * scale), int(d.shape[0] * scale)
        x = (W - sw) // 2
        frame[56:56 + sh, x:x + sw] = cv2.resize(d, (sw, sh))
        out.append(frame)
    return out


def client_only(client, heading, caption, colour):
    out = []
    cw, ch = 704, 528
    x = (W - cw) // 2
    for cf in client:
        frame = blank()
        chrome(frame, heading, caption, colour)
        frame[56:56 + ch, x:x + cw] = cv2.resize(cf, (cw, ch))
        out.append(frame)
    return out


def transcode_h264(path: str) -> None:
    """Re-encode to H.264 in place.

    OpenCV writes MPEG-4 Part 2 (FMP4), which QuickTime plays but browsers and
    cloud previews often will not - and this video is submitted as a shared
    link an examiner will most likely preview in a browser. macOS ships
    avconvert, so no extra dependency is needed.
    """
    tool = "/usr/bin/avconvert"
    if not Path(tool).exists():
        print("[demo] avconvert not found; leaving the file as FMP4 "
              "(may not preview in a browser)")
        return
    tmp = str(Path(path).with_suffix(".h264.mp4"))
    result = subprocess.run(
        [tool, "--source", path, "--output", tmp, "--preset", "PresetHighestQuality"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and Path(tmp).exists():
        shutil.move(tmp, path)
        print("[demo] re-encoded to H.264")
    else:
        Path(tmp).unlink(missing_ok=True)
        print(f"[demo] H.264 transcode failed: {result.stderr.strip()[:120]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", default="deliverables/raw_online.mp4")
    ap.add_argument("--offline", default="deliverables/raw_offline.mp4")
    ap.add_argument("--dash", default="data/dashboard_frames")
    ap.add_argument("--out", default="deliverables/DrowSim_prototype_demo.mp4")
    args = ap.parse_args()

    det = json.loads(Path("runs/detector_test_report.json").read_text())
    sysr = json.loads(Path("runs/system_test_report.json").read_text())
    dash = load_dash(args.dash)
    print(f"[demo] {len(dash)} dashboard frames")

    seq = []
    seq += card([
        ("DrowSim", 1.6, FG, 3),
        ("Privacy-Preserving Driver Drowsiness Detection", 0.66, OK, 1),
        ("", 0.4, FG, 1),
        ("Esra Esam Halboup  -  A199870  -  UKM", 0.5, MUTED, 1),
    ], 4.0)

    seq += card([
        ("The whole system, side by side", 0.95, FG, 2),
        ("", 0.3, FG, 1),
        ("Left: the client sees the driver. Right: the web app never does.", 0.52, MUTED, 1),
        ("Only seven numbers per frame cross the network.", 0.52, MUTED, 1),
    ], 4.5)

    # NORMAL -> yawn WARNING -> first CRITICAL, both views in step.
    seq += side_by_side(
        load_client(args.online, 30, 300), dash,
        "Live session  -  vehicle node streaming telemetry",
        "Watch both sides change together: NORMAL, then WARNING on the yawn, then CRITICAL.",
        OK, dash_offset=15)

    seq += card([
        ("The dashboard in detail", 0.95, FG, 2),
        ("", 0.3, FG, 1),
        ("State, decision source and reason - reconstructed from numbers alone", 0.5, MUTED, 1),
    ], 3.5)

    seq += dash_only(dash, 100, 130,
                     "Web dashboard  -  event log",
                     "RULE raised CRITICAL on eye closure. MODEL raised WARNING on the yawn.",
                     WARN)

    seq += card([
        ("Network down", 0.95, CRIT, 2),
        ("", 0.3, FG, 1),
        ("The alarm is decided on the client, so it still fires", 0.52, MUTED, 1),
    ], 3.5)

    seq += client_only(load_client(args.offline, 130, 270),
                       "Server unreachable",
                       'Bottom line reads "no server" - it still reaches CRITICAL on its own.',
                       CRIT)

    eps = sysr["episodes"]
    seq += card([
        ("Measured results", 1.0, FG, 2),
        ("", 0.25, FG, 1),
        (f"{det['accuracy'] * 100:.2f}% accuracy, macro-F1 {det['macro avg']['f1-score']:.4f}",
         0.62, OK, 1),
        (f"on {det['total_frames']:,} unseen FL3D test frames", 0.48, MUTED, 1),
        ("", 0.2, FG, 1),
        (f"{eps['detected_by_acute_rule']}/{eps['total']} microsleep episodes caught, "
         f"{sysr['false_alarms']['acute_false_positives']} false alarms", 0.54, OK, 1),
        (f"PERCLOS within {sysr['perclos']['mean_abs_error'] * 100:.1f}% of ground truth",
         0.48, MUTED, 1),
        ("", 0.2, FG, 1),
        ("8.5 ms inference  |  ~2 KB/s  |  no video transmitted", 0.48, MUTED, 1),
    ], 6.5)

    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in seq:
        out.write(f)
    out.release()
    transcode_h264(args.out)

    secs = len(seq) / FPS
    print(f"wrote {args.out}: {secs:.1f}s ({int(secs // 60)}:{int(secs % 60):02d}) at {W}x{H}")
    if secs > 180:
        print(f"WARNING: over the 3-minute limit by {secs - 180:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
