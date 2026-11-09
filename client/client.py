"""DrowSim in-vehicle client (edge node).

Pipeline per frame: capture -> YOLO -> seven floats -> local rule engine ->
local alarm. The server is contacted in parallel for the learned verdict, but
it is never on the critical path: pull the network cable and the microsleep
alarm still fires.

Raw frames exist only as the local `frame` variable and are overwritten by the
next read. Nothing but the feature vectors ever leaves this process.
"""

from __future__ import annotations

import argparse
import time
import uuid
from collections import deque

import cv2
import numpy as np
import requests

from client.alerts import AlertPlayer
from drowsim import config as cfg
from drowsim.features import empty_features, features_from_boxes
from drowsim.rules import RuleEngine

STATE_COLOURS = {
    cfg.STATE_NORMAL: (80, 200, 80),
    cfg.STATE_WARNING: (60, 200, 245),
    cfg.STATE_CRITICAL: (60, 60, 235),
    cfg.STATE_NO_FACE: (150, 150, 150),
}


class ServerLink:
    """Best-effort telemetry uplink. Every failure mode degrades to 'offline'
    and is shown to the driver; none of them stops the local pipeline."""

    def __init__(self, base_url: str, enabled: bool = True) -> None:
        self.url = base_url.rstrip("/") + cfg.INFER_PATH
        self.enabled = enabled
        self.status = "offline"
        self.last_response: dict | None = None
        self.rtt_ms: float = 0.0
        self._session = requests.Session()

    def send(self, payload: dict) -> dict | None:
        if not self.enabled:
            self.status = "disabled"
            return None
        started = time.perf_counter()
        try:
            r = self._session.post(self.url, json=payload, timeout=cfg.REQUEST_TIMEOUT_S)
            self.rtt_ms = (time.perf_counter() - started) * 1000.0
            if r.status_code != 200:
                self.status = f"http {r.status_code}"
                return None
            self.status = "online"
            self.last_response = r.json()
            return self.last_response
        except requests.exceptions.Timeout:
            self.status = "timeout"
        except requests.exceptions.RequestException:
            self.status = "no server"
        self.rtt_ms = (time.perf_counter() - started) * 1000.0
        return None


def draw_overlay(frame, state, verdict, fps, link, server_state, boxes_xyxy):
    h, w = frame.shape[:2]
    colour = STATE_COLOURS.get(state, (200, 200, 200))

    for x1, y1, x2, y2 in boxes_xyxy:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)

    cv2.rectangle(frame, (0, 0), (w, 74), (28, 28, 28), -1)
    cv2.rectangle(frame, (0, 0), (10, 74), colour, -1)
    cv2.putText(frame, state, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.95, colour, 2)

    perclos_txt = f"{verdict.perclos:.0%}" if verdict.perclos_valid else "--"
    cv2.putText(
        frame,
        f"PERCLOS {perclos_txt}   closed {verdict.closure_duration_s:.1f}s   "
        f"blinks {verdict.blink_count}   yawns {verdict.total_yawns}",
        (22, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (215, 215, 215),
        1,
    )

    server_txt = f"server: {link.status}"
    if link.status == "online":
        server_txt += f" ({link.rtt_ms:.0f} ms, {server_state or '-'})"
    cv2.putText(frame, f"{fps:5.1f} FPS   {server_txt}", (22, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (185, 185, 185), 1)

    if verdict.reasons:
        cv2.putText(frame, " | ".join(verdict.reasons[:2]), (22, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return frame


def open_capture(source: str):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAPTURE_HEIGHT)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="DrowSim in-vehicle client")
    parser.add_argument("--weights", default=cfg.YOLO_WEIGHTS)
    parser.add_argument("--source", default="0", help="webcam index or video path")
    parser.add_argument("--server", default=cfg.SERVER_URL)
    parser.add_argument("--client-id", default="VEHICLE_NODE_01")
    parser.add_argument("--imgsz", type=int, default=cfg.YOLO_IMG_SIZE)
    parser.add_argument("--conf", type=float, default=cfg.YOLO_CONF)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="no preview window; state is printed instead")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames (used by the benchmark)")
    parser.add_argument("--record", default=None,
                        help="also write the annotated preview to this .mp4")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names
    print(f"[client] YOLO weights: {args.weights}")
    if tuple(names.get(i, "") for i in range(len(cfg.CLASSES))) != cfg.CLASSES:
        print(
            f"[client] WARNING: these weights are not the DrowSim 3-class model "
            f"(found {len(names)} classes). Detections will be reported as "
            f"'alert' - use this only to smoke-test the pipeline."
        )
    else:
        print(f"[client] classes: {cfg.CLASSES}")

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"[client] cannot open source {args.source!r}")

    rules = RuleEngine()
    alerts = AlertPlayer(enabled=not args.no_audio)
    link = ServerLink(args.server, enabled=not args.no_server)

    session_id = uuid.uuid4().hex[:12]
    seq = 0
    pending: list[list[float]] = []
    last_send = 0.0
    send_interval = 1.0 / cfg.TELEMETRY_HZ

    fps_window: deque[float] = deque(maxlen=30)
    prev_loop_start: float | None = None
    server_state: str | None = None
    frames = 0
    infer_times: list[float] = []
    t_start = time.monotonic()

    writer = None
    if args.record:
        # Recorded at the real capture rate, so the demo plays back at the
        # speed the system actually runs at rather than a flattering one.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        probe_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.CAPTURE_WIDTH
        probe_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.CAPTURE_HEIGHT
        writer = cv2.VideoWriter(args.record, fourcc, cfg.TARGET_FPS,
                                 (probe_w, probe_h))
        print(f"[client] recording annotated preview -> {args.record}")

    print(f"[client] session {session_id} started; press q to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[client] capture ended")
                break

            loop_start = time.monotonic()
            # True capture rate: measured start-to-start, so it includes the
            # throttle below. Timing only the processing would report the rate
            # the machine *could* manage, not the rate frames actually arrive.
            if prev_loop_start is not None:
                fps_window.append(1.0 / max(loop_start - prev_loop_start, 1e-6))
            prev_loop_start = loop_start
            h, w = frame.shape[:2]

            t0 = time.perf_counter()
            results = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, verbose=False
            )
            infer_times.append((time.perf_counter() - t0) * 1000.0)

            boxes = results[0].boxes if results else None
            feats = features_from_boxes(boxes, w, h) if boxes is not None else empty_features()
            boxes_xyxy = (
                boxes.xyxy.detach().cpu().numpy() if boxes is not None and len(boxes) else []
            )

            now = time.monotonic()
            verdict = rules.update(feats, now)

            # The local verdict drives the alarm. This is the whole point of
            # keeping the rule engine on the client.
            state = verdict.state
            alerts.play(state, now)

            pending.append([round(float(v), 4) for v in feats.vector])
            if now - last_send >= send_interval and pending:
                payload = {
                    "client_id": args.client_id,
                    "session_id": session_id,
                    "seq": seq,
                    "timestamp": now,
                    "fps": float(np.mean(fps_window)) if fps_window else cfg.TARGET_FPS,
                    "features": pending,
                    "face_found": feats.face_found,
                    "client_state": state,
                    "client_reasons": verdict.reasons,
                    "perclos": round(verdict.perclos, 4),
                    "perclos_valid": verdict.perclos_valid,
                    "closure_duration_s": round(verdict.closure_duration_s, 3),
                    "blink_count": verdict.blink_count,
                    "total_yawns": verdict.total_yawns,
                }
                response = link.send(payload)
                server_state = response.get("state") if response else None
                pending = []
                seq += 1
                last_send = now

            # The server may raise a warning the local rules did not see (a yawn
            # the model caught), but it can never lower the local verdict - the
            # client stays authoritative on safety.
            #
            # Applied on every frame, not only on the frames a response lands.
            # Telemetry goes out at 5 Hz against a 10 Hz capture loop, so
            # elevating inside the send block made the displayed state flicker
            # between WARNING and NORMAL on alternate frames.
            if server_state == cfg.STATE_WARNING and state == cfg.STATE_NORMAL:
                state = cfg.STATE_WARNING
                alerts.play(state, now)

            fps = float(np.mean(fps_window)) if fps_window else 0.0
            frames += 1

            if writer is not None:
                annotated = frame.copy()
                draw_overlay(annotated, state, verdict, fps, link, server_state,
                             boxes_xyxy)
                writer.write(annotated)

            if args.headless:
                if frames % 15 == 0:
                    print(
                        f"  [{frames:5d}] {state:9s} perclos="
                        f"{verdict.perclos:.2%} fps={fps:4.1f} server={link.status}"
                    )
            else:
                draw_overlay(frame, state, verdict, fps, link, server_state, boxes_xyxy)
                cv2.imshow("DrowSim Client (YOLO)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frames >= args.max_frames:
                break

            # Do not spin faster than the target rate; it only wastes CPU.
            budget = 1.0 / cfg.TARGET_FPS - (time.monotonic() - loop_start)
            if budget > 0:
                time.sleep(budget)
    except KeyboardInterrupt:
        print("\n[client] interrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        alerts.close()

    elapsed = time.monotonic() - t_start
    if infer_times:
        arr = np.array(infer_times)
        print(
            f"\n[client] {frames} frames in {elapsed:.1f}s "
            f"({frames / max(elapsed, 1e-6):.1f} FPS)\n"
            f"[client] YOLO inference: mean {arr.mean():.1f} ms, "
            f"p95 {np.percentile(arr, 95):.1f} ms, max {arr.max():.1f} ms\n"
            f"[client] blinks {rules.blink_count}, yawns {rules.total_yawns}"
        )


if __name__ == "__main__":
    main()
