"""Behavioural tests for the deterministic safety path."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drowsim import config as cfg
from drowsim.features import FrameFeatures
from drowsim.rules import RuleEngine

FPS = 15.0
DT = 1.0 / FPS


def feats(p_micro=0.0, p_yawn=0.0, face=True):
    p_alert = max(0.0, 1.0 - p_micro - p_yawn)
    return FrameFeatures(
        vector=np.zeros(cfg.FEATURE_DIM, dtype=np.float32),
        face_found=face,
        p_alert=p_alert,
        p_microsleep=p_micro,
        p_yawning=p_yawn,
    )


def run(engine, frames, start=0.0):
    """frames: list of (FrameFeatures, repeat_count)."""
    t = start
    verdict = None
    for f, n in frames:
        for _ in range(n):
            verdict = engine.update(f, t)
            t += DT
    return verdict, t


def test_blink_is_not_a_microsleep():
    e = RuleEngine()
    # 60s alert to fill the PERCLOS window, then a 200 ms blink.
    v, t = run(e, [(feats(0.0), int(60 * FPS))])
    v, t = run(e, [(feats(0.9), 3), (feats(0.0), 15)], start=t)
    assert v.state == cfg.STATE_NORMAL, v
    assert e.blink_count == 1, e.blink_count


def test_sustained_closure_is_critical():
    e = RuleEngine()
    _, t = run(e, [(feats(0.0), int(10 * FPS))])
    v, _ = run(e, [(feats(0.9), int(1.2 * FPS))], start=t)
    assert v.state == cfg.STATE_CRITICAL, v
    assert "eyes closed" in v.reasons[0]


def test_critical_fires_before_perclos_window_is_valid():
    """The acute channel must not wait for PERCLOS to become trustworthy."""
    e = RuleEngine()
    v, _ = run(e, [(feats(0.9), int(1.5 * FPS))])
    assert v.state == cfg.STATE_CRITICAL, v
    assert not v.perclos_valid


def test_perclos_warns_on_accumulated_closure():
    e = RuleEngine()
    # 60s window: alternate 0.6s closed / 3.4s open = 15% closure, all episodes
    # long enough to be closures but interleaved so none alone is a microsleep.
    frames = []
    for _ in range(15):
        frames.append((feats(0.9), int(0.6 * FPS)))
        frames.append((feats(0.0), int(3.4 * FPS)))
    v, _ = run(e, frames)
    assert v.perclos_valid
    assert 0.10 <= v.perclos <= 0.20, v.perclos
    assert v.state in (cfg.STATE_WARNING, cfg.STATE_CRITICAL), v


def test_no_face_is_not_normal():
    e = RuleEngine()
    _, t = run(e, [(feats(0.0), int(5 * FPS))])
    v, _ = run(e, [(feats(face=False), int(3 * FPS))], start=t)
    assert v.state == cfg.STATE_NO_FACE, v


def test_hysteresis_prevents_flicker():
    """Evidence hovering between the two thresholds must not toggle state."""
    e = RuleEngine()
    _, t = run(e, [(feats(0.0), int(5 * FPS))])
    mid = (cfg.CLOSE_ENTER + cfg.CLOSE_EXIT) / 2.0
    v, _ = run(e, [(feats(mid), int(3 * FPS))], start=t)
    assert not v.eyes_closed, "mid-band evidence should not open a closure"


def test_short_mouth_opening_is_not_a_yawn():
    e = RuleEngine()
    v, _ = run(e, [(feats(p_yawn=0.9), int(0.8 * FPS)), (feats(), int(2 * FPS))])
    assert e.total_yawns == 0, e.total_yawns


def test_long_mouth_opening_is_a_yawn():
    e = RuleEngine()
    v, _ = run(e, [(feats(p_yawn=0.9), int(2.0 * FPS)), (feats(), int(2 * FPS))])
    assert e.total_yawns == 1, e.total_yawns


def test_three_yawns_raise_a_warning():
    e = RuleEngine()
    frames = []
    for _ in range(3):
        frames.append((feats(p_yawn=0.9), int(2.0 * FPS)))
        frames.append((feats(), int(4.0 * FPS)))
    v, _ = run(e, frames)
    assert e.total_yawns == 3, e.total_yawns
    assert v.state == cfg.STATE_WARNING, v


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except AssertionError as exc:
                print(f"FAIL {name}: {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
