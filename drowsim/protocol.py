"""The wire contract between the in-vehicle client and the server.

Everything that crosses the network is defined here, and it is all numbers.
No frame, crop, landmark image or encoded pixel appears in any of these models
- that is the property the privacy claim depends on, so keep it that way.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from drowsim.config import FEATURE_DIM


class TelemetryPacket(BaseModel):
    """One aggregated telemetry tick from a vehicle node."""

    client_id: str = Field(..., description="Stable pseudonymous node id")
    session_id: str = Field(..., description="New id per driving session")
    seq: int = Field(..., ge=0, description="Monotonic counter, for loss detection")
    timestamp: float = Field(..., description="Client monotonic seconds")
    fps: float = Field(..., ge=0.0)

    # Only the frames captured since the previous packet, not the whole window.
    # The server keeps the rolling buffer per session; if `seq` skips, it drops
    # the buffer and lets it refill (about two seconds) rather than reasoning
    # over a sequence with a hole in it.
    features: list[list[float]] = Field(
        ..., description="New frames since the last packet, each FEATURE_DIM wide"
    )

    face_found: bool = True

    # The client's own rule verdict travels with the packet. The server does not
    # trust it blindly - it recomputes - but recording both lets the report show
    # the two tiers agreeing.
    client_state: str
    client_reasons: list[str] = Field(default_factory=list)
    perclos: float = 0.0
    perclos_valid: bool = False
    closure_duration_s: float = 0.0
    blink_count: int = 0
    total_yawns: int = 0

    def feature_width_ok(self) -> bool:
        return all(len(row) == FEATURE_DIM for row in self.features)


class InferenceResponse(BaseModel):
    """The server's verdict for one telemetry packet."""

    state: str
    confidence: list[float] = Field(default_factory=list)
    source: str = Field(..., description="RULE, MODEL or NO_FACE")
    reasons: list[str] = Field(default_factory=list)
    server_perclos: float = 0.0
    seq: int = 0
    inference_ms: float = 0.0


class SessionEvent(BaseModel):
    """A state transition, kept in memory for the monitoring dashboard."""

    session_id: str
    client_id: str
    timestamp: float
    state: str
    source: str
    reasons: list[str] = Field(default_factory=list)
    perclos: float = 0.0


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    uptime_s: float = 0.0
    active_sessions: int = 0
