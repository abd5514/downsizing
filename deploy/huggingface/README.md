---
title: DrowSim
emoji: 🚗
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Privacy-preserving driver drowsiness monitoring dashboard
---

# DrowSim monitoring server

The server half of DrowSim, a privacy-preserving driver drowsiness detection
system. It runs the BiLSTM-Attention temporal model, re-runs the rule engine on
the telemetry it receives, and serves the monitoring dashboard.

No video reaches this server. A vehicle client runs the YOLO detector locally
and sends seven numbers per frame to `POST /infer`.

## Demo node

`DEMO_REPLAY` replays telemetry recorded from a 90-second test clip on a loop,
through the same inference path a live vehicle uses, so the dashboard shows the
system working when no vehicle is connected. It is labelled "recorded demo".

## Connecting a live vehicle client

```bash
PYTHONPATH=. python -m client.client --weights models/drowsim_yolov8n.pt \
  --server https://<this-space>.hf.space
```

Session state lives in memory only and is lost when the Space restarts or sleeps.
