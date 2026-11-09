"""End-to-end check against a running server: simulate a driving session and
verify the server reaches the right verdicts from telemetry alone."""

import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drowsim import config as cfg

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"
FPS = 15.0
DT = 1.0 / FPS


def vec(p_micro=0.0, p_yawn=0.0, face=True):
    if not face:
        return [0.0] * 7
    p_alert = max(0.0, 1.0 - p_micro - p_yawn)
    return [p_alert, p_micro, p_yawn, 0.85, 0.12, 0.5, 0.45]


class Sim:
    def __init__(self):
        self.session = uuid.uuid4().hex[:12]
        self.seq = 0
        self.t = 0.0

    def send(self, rows, skip_seq=False):
        if skip_seq:
            self.seq += 5
        self.t += DT * len(rows)
        payload = {
            "client_id": "TEST_NODE", "session_id": self.session, "seq": self.seq,
            "timestamp": self.t, "fps": FPS, "features": rows,
            "face_found": any(any(r) for r in rows), "client_state": "NORMAL",
        }
        self.seq += 1
        r = requests.post(f"{BASE}/infer", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()

    def stream(self, v, seconds):
        out = None
        n = int(seconds * FPS)
        for i in range(0, n, 3):
            out = self.send([v] * min(3, n - i))
        return out


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name}{'' if cond else ': ' + detail}")
    return cond


results = []

# 1. steady alert driving -> NORMAL
s = Sim()
r = s.stream(vec(), 30)
results.append(check("steady alert is NORMAL", r["state"] == cfg.STATE_NORMAL, str(r)))

# 2. sustained closure -> CRITICAL via RULE
r = s.stream(vec(p_micro=0.9), 1.5)
results.append(check("sustained closure -> CRITICAL/RULE",
                     r["state"] == cfg.STATE_CRITICAL and r["source"] == "RULE", str(r)))

# 3. recovery
r = s.stream(vec(), 5)
results.append(check("recovers after eyes reopen",
                     r["state"] in (cfg.STATE_NORMAL, cfg.STATE_WARNING), str(r)))

# 4. blink does not trigger
s2 = Sim()
s2.stream(vec(), 25)
r = s2.send([vec(p_micro=0.9)] * 3)
r = s2.stream(vec(), 2)
results.append(check("200 ms blink is not CRITICAL", r["state"] != cfg.STATE_CRITICAL, str(r)))

# 5. no face
s3 = Sim()
s3.stream(vec(), 5)
r = s3.stream(vec(face=False), 3)
results.append(check("no face -> NO_FACE (never NORMAL)",
                     r["state"] == cfg.STATE_NO_FACE, str(r)))

# 6. sequence gap is tolerated
s4 = Sim()
s4.stream(vec(), 5)
r = s4.send([vec()] * 3, skip_seq=True)
results.append(check("sequence gap handled without error", r["state"] in
                     (cfg.STATE_NORMAL, cfg.STATE_NO_FACE), str(r)))

# 7. malformed payload rejected
bad = requests.post(f"{BASE}/infer", json={
    "client_id": "x", "session_id": "y", "seq": 0, "timestamp": 0.0, "fps": 15.0,
    "features": [[0.1, 0.2]], "client_state": "NORMAL"}, timeout=5)
results.append(check("wrong feature width rejected (422)", bad.status_code == 422,
                     f"got {bad.status_code}"))

# 8. events recorded
ev = requests.get(f"{BASE}/events", timeout=5).json()
states = {e["state"] for e in ev["events"]}
results.append(check("state transitions recorded for dashboard",
                     cfg.STATE_CRITICAL in states, str(states)))

# 9. latency
t0 = time.perf_counter()
for _ in range(20):
    s.send([vec()] * 3)
avg = (time.perf_counter() - t0) / 20 * 1000
results.append(check(f"round-trip {avg:.1f} ms < 100 ms", avg < 100, f"{avg:.1f} ms"))

print(f"\n{sum(results)} passed, {len(results) - sum(results)} failed")
sys.exit(0 if all(results) else 1)
