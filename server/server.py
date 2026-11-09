"""DrowSim inference server.

Division of responsibility between the tiers:

  client   owns the safety-critical path. It runs the rule engine locally and
           sounds the alarm itself, so a microsleep is still caught when this
           server is unreachable.

  server   owns the learned temporal classification, the session record and
           the monitoring dashboard. It re-runs the identical rule engine on
           the telemetry it receives, so the report can show the two tiers
           reaching the same verdict independently.

Nothing is written to disk. Session state lives in memory and dies with the
process, which is what the project's no-storage privacy requirement asks for.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from drowsim import config as cfg
from drowsim.features import FrameFeatures
from drowsim.model import build_model
from drowsim.protocol import (
    HealthResponse,
    InferenceResponse,
    SessionEvent,
    TelemetryPacket,
)
from drowsim.rules import RuleEngine

MAX_EVENTS = 500
SESSION_TTL_S = 300.0

app = FastAPI(title="DrowSim Inference Server", version="2.0-yolo")
_start_time = time.monotonic()


@dataclass
class Session:
    client_id: str
    buffer: deque = field(default_factory=lambda: deque(maxlen=cfg.SEQ_LEN))
    votes: deque = field(default_factory=lambda: deque(maxlen=cfg.VOTE_WINDOW))
    rules: RuleEngine = field(default_factory=RuleEngine)
    last_seq: int = -1
    last_seen: float = 0.0
    last_state: str | None = None


class ModelHolder:
    """Loads the BiLSTM once. Absent weights are not fatal - the rule engine
    alone is still a working system, and saying so is better than pretending."""

    def __init__(self) -> None:
        self.model = None
        self.path: Path | None = None

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            print(f"[server] no BiLSTM weights at {path}; running rules-only")
            return False
        model = build_model()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()
        self.model = model
        self.path = path
        print(f"[server] BiLSTM loaded from {path} ({model.num_parameters:,} params)")
        return True

    @property
    def loaded(self) -> bool:
        return self.model is not None


models = ModelHolder()
sessions: dict[str, Session] = {}
events: deque[SessionEvent] = deque(maxlen=MAX_EVENTS)


def _reap_sessions(now: float) -> None:
    stale = [k for k, s in sessions.items() if now - s.last_seen > SESSION_TTL_S]
    for k in stale:
        del sessions[k]


def _rule_state_from_packet(session: Session, packet: TelemetryPacket) -> object:
    """Replay the received frames through the server's own rule engine."""
    verdict = None
    n = max(len(packet.features), 1)
    # Frames in a packet were captured evenly across the interval it covers.
    dt = (1.0 / packet.fps) if packet.fps > 0 else (1.0 / cfg.TARGET_FPS)
    base_ts = packet.timestamp - dt * (n - 1)

    for i, row in enumerate(packet.features):
        vec = np.asarray(row, dtype=np.float32)
        feats = FrameFeatures(
            vector=vec,
            face_found=bool(vec.any()),
            p_alert=float(vec[0]),
            p_microsleep=float(vec[1]),
            p_yawning=float(vec[2]),
        )
        verdict = session.rules.update(feats, base_ts + i * dt)
    return verdict


@app.post("/infer", response_model=InferenceResponse)
def infer(packet: TelemetryPacket) -> InferenceResponse:
    started = time.perf_counter()
    now = time.monotonic()
    _reap_sessions(now)

    if not packet.feature_width_ok():
        raise HTTPException(422, f"every feature row must be {cfg.FEATURE_DIM} wide")

    session = sessions.get(packet.session_id)
    if session is None:
        session = Session(client_id=packet.client_id)
        sessions[packet.session_id] = session

    # A gap in the sequence means we would otherwise reason over a window with
    # a hole in it. Drop the buffer and let it refill instead.
    if session.last_seq >= 0 and packet.seq != session.last_seq + 1:
        session.buffer.clear()
        session.votes.clear()
    session.last_seq = packet.seq
    session.last_seen = now

    rule_verdict = _rule_state_from_packet(session, packet)
    for row in packet.features:
        session.buffer.append(np.asarray(row, dtype=np.float32))

    confidence: list[float] = []
    model_state: str | None = None

    if models.loaded and len(session.buffer) == cfg.SEQ_LEN:
        window = np.stack(list(session.buffer))[None, ...]
        with torch.no_grad():
            logits = models.model(torch.from_numpy(window).float())
            probs = torch.softmax(logits, dim=1)[0].numpy()
        confidence = [round(float(p), 4) for p in probs]

        predicted = int(np.argmax(probs))
        # The rule engine is the sole arbiter of microsleep. A model that
        # predicts it is remapped to alert so an uncertain network can never
        # raise - or suppress - the safety verdict on its own.
        if predicted == cfg.IDX_MICROSLEEP:
            predicted = cfg.IDX_ALERT
        session.votes.append(predicted)

        voted = Counter(session.votes).most_common(1)[0][0]
        model_state = cfg.STATE_WARNING if voted == cfg.IDX_YAWNING else cfg.STATE_NORMAL

    # Resolution order: no-face, then the deterministic rule, then the model.
    if rule_verdict is not None and rule_verdict.state == cfg.STATE_NO_FACE:
        state, source, reasons = cfg.STATE_NO_FACE, "NO_FACE", ["no face detected"]
    elif rule_verdict is not None and rule_verdict.state != cfg.STATE_NORMAL:
        state, source, reasons = rule_verdict.state, "RULE", rule_verdict.reasons
    elif model_state is not None and model_state != cfg.STATE_NORMAL:
        state, source, reasons = model_state, "MODEL", ["yawning detected by model"]
    else:
        state, source, reasons = cfg.STATE_NORMAL, "MODEL" if model_state else "RULE", []

    if state != session.last_state:
        events.appendleft(
            SessionEvent(
                session_id=packet.session_id,
                client_id=packet.client_id,
                timestamp=time.time(),
                state=state,
                source=source,
                reasons=reasons,
                perclos=round(rule_verdict.perclos if rule_verdict else 0.0, 4),
            )
        )
        session.last_state = state

    return InferenceResponse(
        state=state,
        confidence=confidence,
        source=source,
        reasons=reasons,
        server_perclos=round(rule_verdict.perclos if rule_verdict else 0.0, 4),
        seq=packet.seq,
        inference_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=models.loaded,
        uptime_s=round(time.monotonic() - _start_time, 1),
        active_sessions=len(sessions),
    )


@app.get("/events")
def get_events(limit: int = 50) -> dict:
    now = time.monotonic()
    return {
        "events": [e.model_dump() for e in list(events)[:limit]],
        "sessions": [
            {
                "session_id": sid,
                "client_id": s.client_id,
                "state": s.last_state or cfg.STATE_NORMAL,
                "age_s": round(now - s.last_seen, 1),
                "online": (now - s.last_seen) < 5.0,
            }
            for sid, s in sessions.items()
        ],
    }


DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
if DASHBOARD.exists():
    app.mount("/static", StaticFiles(directory=DASHBOARD), name="static")

    @app.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(DASHBOARD / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="DrowSim inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bilstm", default="models/bilstm_attention.pt")
    args = parser.parse_args()

    models.load(args.bilstm)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
