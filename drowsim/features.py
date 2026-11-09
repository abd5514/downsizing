"""Turn one YOLO detection into the compact numeric vector that is all the
client is ever allowed to transmit.

The privacy claim of the whole system rests on this module: what leaves the
vehicle is these seven floats, never pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drowsim.config import FEATURE_DIM, IDX_ALERT, IDX_MICROSLEEP, IDX_YAWNING

FEATURE_NAMES = (
    "p_alert",
    "p_microsleep",
    "p_yawning",
    "bbox_aspect",
    "bbox_area",
    "center_x",
    "center_y",
)


@dataclass(frozen=True)
class FrameFeatures:
    """One frame reduced to numbers. `face_found` is False when YOLO returned
    nothing, which the rule engine treats as NO_FACE rather than as alert."""

    vector: np.ndarray          # shape (FEATURE_DIM,), float32
    face_found: bool
    p_alert: float
    p_microsleep: float
    p_yawning: float

    @property
    def closure_evidence(self) -> float:
        """Probability the eyes are closed. The rule engine keys off this."""
        return self.p_microsleep

    @property
    def yawn_evidence(self) -> float:
        return self.p_yawning


def empty_features() -> FrameFeatures:
    """Used when no face is detected: an all-zero vector keeps the BiLSTM
    sequence dense without inventing a driver state."""
    return FrameFeatures(
        vector=np.zeros(FEATURE_DIM, dtype=np.float32),
        face_found=False,
        p_alert=0.0,
        p_microsleep=0.0,
        p_yawning=0.0,
    )


def features_from_boxes(boxes, frame_w: int, frame_h: int) -> FrameFeatures:
    """Build the feature vector from an Ultralytics `Boxes` object.

    Only the highest-confidence detection is used - there is exactly one driver
    in the frame, and taking the best box avoids a passenger's face hijacking
    the signal.
    """
    if boxes is None or len(boxes) == 0:
        return empty_features()

    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    xyxy = boxes.xyxy.detach().cpu().numpy()

    best = int(np.argmax(conf))
    x1, y1, x2, y2 = xyxy[best]
    score = float(conf[best])

    # A detection model gives one class per box, not a distribution over the
    # three states. We spread the confidence over the predicted class and leave
    # the remainder on "alert", which keeps the vector on the probability
    # simplex and gives the BiLSTM a graded rather than one-hot input.
    probs = np.zeros(3, dtype=np.float32)
    predicted = int(cls[best])
    if predicted not in (IDX_ALERT, IDX_MICROSLEEP, IDX_YAWNING):
        # Weights that are not the 3-class DrowSim model (a stock COCO
        # checkpoint used for a pipeline smoke test, say). Treat the detection
        # as a face in an unknown state rather than crashing on the index.
        predicted = IDX_ALERT
    probs[predicted] = score
    if predicted != IDX_ALERT:
        probs[IDX_ALERT] = 1.0 - score
    else:
        residual = (1.0 - score) / 2.0
        probs[IDX_MICROSLEEP] = residual
        probs[IDX_YAWNING] = residual

    w = max(float(x2 - x1), 1e-6)
    h = max(float(y2 - y1), 1e-6)
    aspect = w / h
    area = (w * h) / float(max(frame_w * frame_h, 1))
    cx = ((float(x1) + float(x2)) / 2.0) / float(max(frame_w, 1))
    cy = ((float(y1) + float(y2)) / 2.0) / float(max(frame_h, 1))

    vector = np.array(
        [probs[0], probs[1], probs[2], aspect, area, cx, cy], dtype=np.float32
    )
    return FrameFeatures(
        vector=vector,
        face_found=True,
        p_alert=float(probs[IDX_ALERT]),
        p_microsleep=float(probs[IDX_MICROSLEEP]),
        p_yawning=float(probs[IDX_YAWNING]),
    )


def features_from_label(
    label: int, box_xyxy, frame_w: int, frame_h: int, confidence: float = 1.0
) -> np.ndarray:
    """Ground-truth variant used when exporting BiLSTM training sequences from
    the labelled dataset, so the sequence format matches inference exactly."""
    probs = np.zeros(3, dtype=np.float32)
    probs[label] = confidence
    if label != IDX_ALERT:
        probs[IDX_ALERT] = 1.0 - confidence

    x1, y1, x2, y2 = box_xyxy
    w = max(float(x2 - x1), 1e-6)
    h = max(float(y2 - y1), 1e-6)
    return np.array(
        [
            probs[0],
            probs[1],
            probs[2],
            w / h,
            (w * h) / float(max(frame_w * frame_h, 1)),
            ((float(x1) + float(x2)) / 2.0) / float(max(frame_w, 1)),
            ((float(y1) + float(y2)) / 2.0) / float(max(frame_h, 1)),
        ],
        dtype=np.float32,
    )
