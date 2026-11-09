"""Score the complete pipeline against a labelled clip.

Per-class frame accuracy says how good the detector is. It does not say whether
the *system* wakes a driver who falls asleep, which is the only question that
matters, and the original thesis never measured it.

This replays a clip built by make_demo_clip.py through the real client stack -
YOLO, then the rule engine - and reports:

  episode recall   how many ground-truth microsleep episodes raised CRITICAL
  alarm latency    seconds from episode onset to the alarm
  false alarms     CRITICAL raised while the driver was labelled alert
  PERCLOS error    measured PERCLOS against the true rolling closure fraction

Detection is deliberately offline and frame-accurate: the clip is decoded at a
fixed rate rather than in real time, so the numbers describe the algorithm and
not the machine's momentary load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from drowsim import config as cfg
from drowsim.features import empty_features, features_from_boxes
from drowsim.rules import RuleEngine


def episodes(labels: list[str], target: str, fps: float, min_s: float):
    """Contiguous runs of `target` at least `min_s` long, as (start, end) frames."""
    out, start = [], None
    for i, lab in enumerate(labels + [None]):
        if lab == target and start is None:
            start = i
        elif lab != target and start is not None:
            if (i - start) / fps >= min_s:
                out.append((start, i))
            start = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data/demo_clip.mp4")
    ap.add_argument("--labels", default="data/demo_clip_labels.json")
    ap.add_argument("--weights", default="models/drowsim_yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=cfg.YOLO_IMG_SIZE)
    ap.add_argument("--out", default="runs/system_test_report.json")
    args = ap.parse_args()

    meta = json.loads(Path(args.labels).read_text())
    labels, fps = meta["labels"], float(meta["fps"])

    from ultralytics import YOLO

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.clip)
    rules = RuleEngine()

    states, closed_flags, perclos_trace, acute_flags = [], [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(labels):
            break
        h, w = frame.shape[:2]
        res = model.predict(frame, imgsz=args.imgsz, conf=cfg.YOLO_CONF, verbose=False)
        boxes = res[0].boxes if res else None
        feats = (features_from_boxes(boxes, w, h)
                 if boxes is not None and len(boxes) else empty_features())
        verdict = rules.update(feats, idx / fps)
        states.append(verdict.state)
        acute_flags.append(
            verdict.state == cfg.STATE_CRITICAL
            and any(r.startswith("eyes closed") for r in verdict.reasons)
        )
        closed_flags.append(verdict.eyes_closed)
        perclos_trace.append(verdict.perclos)
        idx += 1
    cap.release()

    n = len(states)
    labels = labels[:n]
    truth_closed = np.array([lab == "microsleep" for lab in labels])
    pred_closed = np.array(closed_flags)

    tp = int((truth_closed & pred_closed).sum())
    fp = int((~truth_closed & pred_closed).sum())
    fn = int((truth_closed & ~pred_closed).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # --- episode-level: the question that actually matters ---
    # Credit is given only to the acute channel. Counting any CRITICAL state
    # would score an alarm that was already sounding from a previous episode as
    # if this one had been detected, which flatters both recall and latency.
    eps = episodes(labels, "microsleep", fps, cfg.MICROSLEEP_MIN_S)
    caught, latencies, already_critical = 0, [], 0
    for start, end in eps:
        if states[start] == cfg.STATE_CRITICAL:
            already_critical += 1
        window = range(start, min(end + int(fps * 1.0), n))
        fired = next((i for i in window if acute_flags[i]), None)
        if fired is not None:
            caught += 1
            latencies.append((fired - start) / fps)

    # --- false alarms ---
    # Only the acute channel can raise a genuine false alarm here. A PERCLOS
    # CRITICAL during an alert stretch is not an error: PERCLOS is a 60 s
    # trailing measure, and on a clip that is a quarter microsleep it is
    # *correct* for it to stay high through the driver's brief alert moments.
    # Lumping the two together would report a real system as broken.
    alert_eps = episodes(labels, "alert", fps, 3.0)
    acute_fp = sum(
        1 for s, e in alert_eps
        if any(acute_flags[i] for i in range(min(s + int(fps), e), e))
    )
    perclos_critical = sum(
        1 for s, e in alert_eps
        if any(states[i] == cfg.STATE_CRITICAL and not acute_flags[i]
               for i in range(min(s + int(fps), e), e))
    )

    # --- PERCLOS against the true rolling closure fraction ---
    win = int(cfg.PERCLOS_WINDOW_S * fps)
    true_perclos, errors = [], []
    for i in range(n):
        lo = max(0, i - win + 1)
        tp_frac = float(truth_closed[lo:i + 1].mean())
        true_perclos.append(tp_frac)
        if i >= int(cfg.PERCLOS_MIN_S * fps):
            errors.append(abs(tp_frac - perclos_trace[i]))

    report = {
        "clip": args.clip,
        "frames": n,
        "duration_s": round(n / fps, 1),
        "ground_truth": {k: labels.count(k) for k in ("alert", "microsleep", "yawning")},
        "frame_level_closure": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "episodes": {
            "total": len(eps),
            "detected_by_acute_rule": caught,
            "recall": round(caught / max(len(eps), 1), 4),
            "already_critical_at_onset": already_critical,
            "mean_alarm_latency_s": round(float(np.mean(latencies)), 3) if latencies else None,
            "max_alarm_latency_s": round(float(np.max(latencies)), 3) if latencies else None,
        },
        "false_alarms": {
            "alert_stretches": len(alert_eps),
            "acute_false_positives": acute_fp,
            "perclos_critical_during_alert": perclos_critical,
            "note": "PERCLOS CRITICAL during an alert stretch is expected on a "
                    "drowsy clip and is not counted as a false alarm",
        },
        "perclos": {
            "mean_abs_error": round(float(np.mean(errors)), 4) if errors else None,
            "max_measured": round(float(np.max(perclos_trace)), 4),
            "max_true": round(float(np.max(true_perclos)), 4),
        },
        "blinks_counted": rules.blink_count,
        "yawns_counted": rules.total_yawns,
    }

    # The per-frame traces are kept so the report figure can plot measured
    # PERCLOS against ground truth rather than restating the summary number.
    Path("runs").mkdir(exist_ok=True)
    Path("runs/perclos_trace.json").write_text(json.dumps({
        "fps": fps,
        "labels": labels,
        "states": states,
        "measured_perclos": [round(v, 5) for v in perclos_trace],
        "true_perclos": [round(v, 5) for v in true_perclos],
        "eyes_closed": [bool(v) for v in closed_flags],
    }))

    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
