"""Deterministic fatigue rules - the safety-critical path.

Nothing in this module depends on a learned model. It is the part a safety
engineer can read end to end and validate by hand, and it is the part that
still works when the network is down. The client runs it locally so the driver
is alerted even with no connectivity; the server runs the identical code on the
telemetry it receives, so the two tiers can never disagree.

Two channels run in parallel, because they catch different things:

  acute   a single sustained eye closure - the microsleep actually happening
          right now. PERCLOS cannot catch this: three seconds of closure inside
          a sixty-second window scores only 5%, far below any sane threshold.

  trend   PERCLOS over a sixty-second sliding window, plus yawn rate - the
          gradual slide into fatigue that no single frame reveals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from drowsim import config as cfg
from drowsim.features import FrameFeatures


@dataclass
class RuleVerdict:
    """Everything the rule engine concluded about the current instant."""

    state: str
    reasons: list[str] = field(default_factory=list)
    perclos: float = 0.0
    perclos_valid: bool = False
    closure_duration_s: float = 0.0
    eyes_closed: bool = False
    blink_count: int = 0
    yawn_count_window: int = 0
    total_yawns: int = 0
    source: str = "RULE"

    @property
    def is_critical(self) -> bool:
        return self.state == cfg.STATE_CRITICAL


class RuleEngine:
    """Stateful evaluator. One instance per driving session.

    All timestamps are monotonic seconds supplied by the caller, so the same
    engine can be driven by a live camera or replayed over a recorded session
    at any speed.
    """

    def __init__(self, no_face_grace_s: float = 1.0) -> None:
        self.no_face_grace_s = no_face_grace_s

        self._closed = False
        self._closure_start: float | None = None
        self._last_ts: float | None = None

        # (timestamp, closed, dt) - trimmed to the PERCLOS window on each update
        self._closure_history: deque[tuple[float, bool, float]] = deque()
        self._history_span = 0.0

        self._yawning = False
        self._yawn_start: float | None = None
        self._last_yawn_end: float | None = None
        self._yawn_events: deque[float] = deque()

        self.blink_count = 0
        self.total_yawns = 0
        self.total_microsleeps = 0

        self._last_face_ts: float | None = None
        self._session_start: float | None = None

    # ------------------------------------------------------------------ api --
    def update(self, feats: FrameFeatures, ts: float) -> RuleVerdict:
        if self._session_start is None:
            self._session_start = ts
        dt = 0.0 if self._last_ts is None else max(ts - self._last_ts, 0.0)
        self._last_ts = ts

        if not feats.face_found:
            return self._handle_no_face(ts)

        self._last_face_ts = ts
        self._update_closure(feats.closure_evidence, ts, dt)
        self._update_yawn(feats.yawn_evidence, ts)
        return self._decide(ts)

    def reset(self) -> None:
        self.__init__(no_face_grace_s=self.no_face_grace_s)

    # -------------------------------------------------------------- internals --
    def _handle_no_face(self, ts: float) -> RuleVerdict:
        """A blocked or averted camera must never read as NORMAL - that is the
        single most dangerous failure mode this system has."""
        # An in-progress closure is abandoned rather than credited: we no longer
        # have evidence for it.
        self._closed = False
        self._closure_start = None
        self._yawning = False
        self._yawn_start = None

        grace_expired = (
            self._last_face_ts is None or (ts - self._last_face_ts) > self.no_face_grace_s
        )
        if grace_expired:
            return RuleVerdict(
                state=cfg.STATE_NO_FACE,
                reasons=["no face detected"],
                perclos=self._perclos(),
                perclos_valid=self._perclos_valid(),
                blink_count=self.blink_count,
                total_yawns=self.total_yawns,
                yawn_count_window=len(self._yawn_events),
            )
        # Within the grace period, hold the previous notion of the world.
        return self._decide(ts)

    def _update_closure(self, evidence: float, ts: float, dt: float) -> None:
        # Schmitt trigger: separate entry and exit thresholds stop a borderline
        # frame from toggling the state on every update.
        if self._closed:
            if evidence <= cfg.CLOSE_EXIT:
                start = ts if self._closure_start is None else self._closure_start
                duration = ts - start
                if duration <= cfg.BLINK_MAX_S:
                    self.blink_count += 1
                self._closed = False
                self._closure_start = None
        else:
            if evidence >= cfg.CLOSE_ENTER:
                self._closed = True
                self._closure_start = ts

        if dt > 0.0:
            self._closure_history.append((ts, self._closed, dt))
            self._history_span += dt
            self._trim_history(ts)

    def _trim_history(self, ts: float) -> None:
        cutoff = ts - cfg.PERCLOS_WINDOW_S
        while self._closure_history and self._closure_history[0][0] < cutoff:
            _, _, old_dt = self._closure_history.popleft()
            self._history_span -= old_dt
        self._history_span = max(self._history_span, 0.0)

    def _update_yawn(self, evidence: float, ts: float) -> None:
        if self._yawning:
            if evidence <= cfg.YAWN_EXIT:
                start = ts if self._yawn_start is None else self._yawn_start
                duration = ts - start
                refractory_ok = (
                    self._last_yawn_end is None
                    or (ts - self._last_yawn_end) >= cfg.YAWN_REFRACTORY_S
                )
                if duration >= cfg.YAWN_MIN_S and refractory_ok:
                    self.total_yawns += 1
                    self._yawn_events.append(ts)
                    self._last_yawn_end = ts
                self._yawning = False
                self._yawn_start = None
        else:
            if evidence >= cfg.YAWN_ENTER:
                self._yawning = True
                self._yawn_start = ts

        cutoff = ts - cfg.YAWN_RATE_WINDOW_S
        while self._yawn_events and self._yawn_events[0] < cutoff:
            self._yawn_events.popleft()

    def _perclos(self) -> float:
        if self._history_span <= 0.0:
            return 0.0
        closed_time = sum(dt for _, closed, dt in self._closure_history if closed)
        return closed_time / self._history_span

    def _perclos_valid(self) -> bool:
        return self._history_span >= cfg.PERCLOS_MIN_S

    def _decide(self, ts: float) -> RuleVerdict:
        closure_duration = 0.0 if self._closure_start is None else ts - self._closure_start
        perclos = self._perclos()
        perclos_valid = self._perclos_valid()
        yawns_in_window = len(self._yawn_events)

        state = cfg.STATE_NORMAL
        reasons: list[str] = []

        # --- trend channel ---
        if perclos_valid and perclos >= cfg.PERCLOS_CRITICAL:
            state = cfg.STATE_CRITICAL
            reasons.append(f"PERCLOS {perclos:.0%} over {cfg.PERCLOS_WINDOW_S:.0f}s")
        elif perclos_valid and perclos >= cfg.PERCLOS_WARN:
            state = cfg.STATE_WARNING
            reasons.append(f"PERCLOS {perclos:.0%} elevated")

        if yawns_in_window >= cfg.YAWN_RATE_WARN and state == cfg.STATE_NORMAL:
            state = cfg.STATE_WARNING
            reasons.append(
                f"{yawns_in_window} yawns in {cfg.YAWN_RATE_WINDOW_S / 60:.0f} min"
            )

        # --- acute channel, evaluated last so it always wins ---
        if self._closed and closure_duration >= cfg.MICROSLEEP_MIN_S:
            state = cfg.STATE_CRITICAL
            reasons.insert(0, f"eyes closed {closure_duration:.1f}s")

        return RuleVerdict(
            state=state,
            reasons=reasons,
            perclos=perclos,
            perclos_valid=perclos_valid,
            closure_duration_s=closure_duration,
            eyes_closed=self._closed,
            blink_count=self.blink_count,
            yawn_count_window=yawns_in_window,
            total_yawns=self.total_yawns,
        )
