"""In-cabin audible alerts.

pygame has no wheel for Python 3.14, so the tones are synthesised once into
WAV files and handed to the platform's own player. That also keeps playback off
the capture thread - a blocking 600 ms beep would drop nine frames.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from drowsim import config as cfg

SAMPLE_RATE = 44100


def _synthesise(path: Path, freq_hz: int, duration_ms: int, volume: float = 0.6) -> Path:
    """Write a mono 16-bit sine tone with a short fade in/out (a hard edge
    produces an audible click that sounds like a fault rather than an alarm)."""
    n = int(SAMPLE_RATE * duration_ms / 1000)
    fade = max(int(SAMPLE_RATE * 0.005), 1)
    frames = bytearray()
    for i in range(n):
        env = min(i / fade, (n - i) / fade, 1.0)
        sample = volume * env * math.sin(2 * math.pi * freq_hz * i / SAMPLE_RATE)
        frames += struct.pack("<h", int(sample * 32767))

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))
    return path


def _player() -> list[str] | None:
    for cmd in (["afplay"], ["aplay", "-q"], ["paplay"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


class AlertPlayer:
    """Fires the local alarm. Deliberately owned by the client and not the
    server: the driver must still be woken when the network is gone."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._proc: subprocess.Popen | None = None
        self._last_played: dict[str, float] = {}
        self._player = _player()

        tmp = Path(tempfile.gettempdir()) / "drowsim_tones"
        tmp.mkdir(exist_ok=True)
        self.critical_wav = _synthesise(
            tmp / "critical.wav", cfg.TONE_CRITICAL_HZ, cfg.TONE_CRITICAL_MS
        )
        self.warning_wav = _synthesise(
            tmp / "warning.wav", cfg.TONE_WARNING_HZ, cfg.TONE_WARNING_MS
        )

        if self._player is None and enabled:
            print("[alerts] no audio player found; alerts will be visual only",
                  file=sys.stderr)

    def available(self) -> bool:
        return self.enabled and self._player is not None

    def play(self, state: str, now: float) -> bool:
        """Returns True if a tone was actually started."""
        if not self.available():
            return False
        if state == cfg.STATE_CRITICAL:
            wav = self.critical_wav
        elif state == cfg.STATE_WARNING:
            wav = self.warning_wav
        else:
            return False

        last = self._last_played.get(state, -1e9)
        if now - last < cfg.ALERT_REFRACTORY_S:
            return False

        # A still-sounding critical tone is never cut short by a new one.
        if self._proc is not None and self._proc.poll() is None:
            return False

        self._last_played[state] = now
        self._proc = subprocess.Popen(
            [*self._player, str(wav)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
