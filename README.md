# DrowSim (YOLO edition)

Privacy-preserving driver drowsiness detection. The in-vehicle client sees the
video; the network sees seven floats per frame.

This is the YOLO rebuild of the original MediaPipe + BiLSTM system, with three
deliberate changes beyond swapping the vision engine:

| | Original | This build |
|---|---|---|
| Vision engine | MediaPipe Face Mesh (468 landmarks) | YOLOv8n, 3-class detector |
| Eye-closure signal | TrueEAR from 8 eyelid landmarks | `p_microsleep` from the detector |
| Per-driver calibration | 2 s baseline, assumes alert at start | none needed |
| Microsleep alarm | decided on the server | decided **on the client** |
| PERCLOS | not implemented | 60 s sliding window |
| No face in frame | classifier still emits a class | explicit `NO_FACE` state |

The calibration change is a real gain: the original conceded in its limitations
that a driver who was already tired at session start would poison the baseline.
A detector has no baseline to poison.

## Why the alarm moved to the client

The original ran the decision on an AWS instance and suppressed alerts when the
link dropped — so no internet meant no alarm, exactly when nobody is watching.
Here the rule engine runs locally and sounds the alarm itself. The server still
earns its place: it runs the learned temporal model, keeps the session record,
and drives the dashboard. Both tiers import the *same* `drowsim/rules.py`, so
they cannot drift apart.

## Detection logic

Two channels, because they catch different failures:

- **Acute** — eyes closed continuously for ≥ 0.8 s → `CRITICAL`, immediately.
  PERCLOS cannot do this: a 3 s microsleep inside a 60 s window scores 5%, far
  below any threshold worth setting.
- **Trend** — PERCLOS ≥ 12% warns, ≥ 15% is critical; three yawns in five
  minutes warns.

Blinks are separated from microsleeps by duration, not by threshold (≤ 0.4 s is
a blink). Closure uses a Schmitt trigger — enter at 0.50, leave at 0.35 — so a
borderline frame cannot make the state flicker.

The BiLSTM never decides microsleep. If it predicts that class the verdict is
remapped to alert, leaving the deterministic rule as the sole arbiter of the
safety-critical state.

## The dataset, and why the pipeline looks like this

FL3D turns out to ship **pre-cropped faces** (~210x290 px), each with a
`driver_state` label and five RetinaFace landmarks. There are no bounding boxes
and no full frames. Two consequences:

- **TrueEAR is unobtainable.** Five landmarks are the eye *centres*, nose and
  mouth corners - no eyelid contours, so an eye aspect ratio cannot be
  computed from this data at all. That is why the rule engine keys off the
  detector's `p_microsleep` instead.
- **A detector cannot be trained on it directly**, yet the client has to find a
  face inside a 640x480 webcam frame. So `prepare_yolo.py` composites each crop
  onto a synthesised dim-cabin background at a random scale and position, and
  the paste rectangle becomes the ground-truth box. A quarter of the frames are
  also emitted as raw tight crops, which keeps the model accurate on close
  faces and lets it be scored against unmodified FL3D frames.

Real in-car backgrounds are not in the dataset, so the backgrounds are
synthetic - state that as a limitation rather than glossing it.

### Splits

FL3D's own `annotations_val` and `annotations_test` are drawn from the **same
five videos**, so choosing a model on val leaks into test. This build instead
splits those five videos between val (2) and test (3), giving a genuinely
video-disjoint evaluation.

| Split | Source | Frames (FL3D) |
|---|---|---|
| train | 39 videos | 48,327 |
| val | 2 held-out videos | ~1,000 |
| test | 3 held-out videos | ~1,500 |

Class totals across the whole dataset are alert 38,954 / microsleep 8,880 /
yawning 5,497 = 53,331, which matches the thesis. The thesis's *split*
(35,735 / 8,655 / 8,941 over "30/7/7 drivers") does not match FL3D and needs
correcting.

One caveat to state plainly: FL3D exposes **video ids, not driver ids**, and
NITYMED recorded several videos per driver. Video-disjoint is the strongest
split this data supports, but it is not guaranteed person-disjoint - so the
thesis's "person-disjoint" claim cannot be made as written.

## Setup

```bash
cd ~/Desktop/side\ projects/drowsim
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Dataset (needs ~/.kaggle/kaggle.json - see the script if it is missing)
PYTHONPATH=. ./.venv/bin/python scripts/get_dataset.py
PYTHONPATH=. ./.venv/bin/python scripts/inspect_dataset.py   # confirm the layout
PYTHONPATH=. ./.venv/bin/python scripts/prepare_yolo.py      # person-disjoint split

# 2. Train the detector (MPS on Apple Silicon, ~40 epochs)
PYTHONPATH=. ./.venv/bin/python scripts/train_yolo.py --model yolo11n.pt

# 3. Temporal model, trained on the detector's own outputs
PYTHONPATH=. ./.venv/bin/python scripts/export_sequences.py
PYTHONPATH=. ./.venv/bin/python scripts/train_bilstm.py

# 4. Run it
PYTHONPATH=. ./.venv/bin/python server/server.py --port 8000        # dashboard at :8000
PYTHONPATH=. ./.venv/bin/python -m client.client --weights models/drowsim_yolo11n.pt

# 5. Numbers for the report
PYTHONPATH=. ./.venv/bin/python tests/test_rules.py
PYTHONPATH=. ./.venv/bin/python tests/test_server_integration.py http://127.0.0.1:8000
PYTHONPATH=. ./.venv/bin/python scripts/benchmark.py --weights models/drowsim_yolo11n.pt
```

For the v8/v9/v10/v11 comparison table, run step 2 once per model and record the
reported mAP and the benchmark latency for each.

Useful client flags: `--source path/to/video.mp4`, `--headless`, `--no-audio`,
`--no-server` (proves the alarm still fires with the network gone).

## Layout

```
drowsim/     config, features, rules, model, protocol   <- shared by both tiers
client/      capture, YOLO, local rule engine, alarm
server/      FastAPI inference, session state, events
dashboard/   single-page monitor, polls /events
scripts/     dataset -> training -> benchmarking
tests/       rule behaviour, server integration
```

## Privacy

What crosses the network is defined in `drowsim/protocol.py` and is entirely
numeric. Frames live in one local variable and are overwritten by the next
`cap.read()`. The server holds session state in memory only; nothing is written
to disk, and the dashboard's history dies with the process.

One honest caveat for the report: telemetry tied to a persistent `client_id` is
**pseudonymous, not anonymous**. Saying so is stronger than overclaiming.

## Measured results

All figures below were produced by this build on an Apple Silicon laptop
(MPS), not quoted from anywhere. Regenerate them with the commands in the
pipeline section.

### YOLO generation comparison

Each model trained on identical data with identical hyperparameters and seed, then
scored the same way — predicted class against ground truth on the 3,810 raw FL3D test
frames. No model failed to detect a face on any frame.

| Model | Params | Epochs | Accuracy | Macro-F1 | Microsleep F1 | Latency |
|---|---|---|---|---|---|---|
| yolov8n **(deployed)** | 3,011,433 | 21 | 0.9848 | 0.9807 | 0.9651 | 8.99 ms |
| yolov9t | 2,005,993 | 20 | 0.9816 | 0.9769 | 0.9533 | 13.61 ms |
| yolov10n | 2,708,210 | 20 | 0.9806 | 0.9755 | 0.9489 | 28.79 ms |
| yolo11n | 2,590,425 | 31 | 0.9759 | 0.9696 | 0.9373 | 8.96 ms |

**YOLOv8n was selected on the strength of this measurement, not on published
benchmarks.** It leads on every quality metric at effectively the same latency, and by
nearly three points of F1 on microsleep — the safety-critical class. The ordering is the
reverse of generation order, and YOLO11n received more training (31 epochs) than the
models that beat it, so it is not undertrained.

Two honest caveats. Each model was trained once with one seed, so the 0.9-point accuracy
spread is suggestive rather than established; the microsleep gap is wide enough to take
more seriously. And YOLOv10n's 28.79 ms is a property of its Ultralytics implementation
on Apple MPS, not of the NMS-free architecture it advertises — it should not be reported
as evidence that v10 is inherently slow.

### Detector — raw, unmodified FL3D test frames (3,810 frames, 3 held-out videos)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| alert | 0.9878 | 0.9901 | 0.9890 | 2,622 |
| microsleep | 0.9599 | 0.9704 | 0.9651 | 642 |
| yawning | 1.0000 | 0.9762 | 0.9880 | 546 |
| **accuracy** | | | **0.9848** | 3,810 |
| macro avg | 0.9826 | 0.9789 | **0.9807** | |

No detection failed on any frame (0/3,810).

Detection mAP50 on the *composite* validation set is ~0.99 — do not quote that as a
headline. A sharp face on a featureless blur makes localisation nearly trivial; the
table above is the meaningful number.

### Temporal model — BiLSTM-Attention (930 test sequences)

Accuracy 0.9667, macro-F1 0.9571, 81,412 parameters. Per-class F1: alert
0.9762, microsleep 0.9150, yawning 0.9801. Microsleep is the weakest class, as
in the original thesis — which is precisely why the deterministic rule, not this model,
decides microsleep.

### System level — 90 s labelled clip (`scripts/score_clip.py`)

The question no per-class table answers: does it actually wake the driver?

| Measure | Result |
|---|---|
| Microsleep episodes detected by the acute rule | 5 / 6 |
| Mean alarm latency from episode onset | 0.86 s (rule mandates 0.8 s dwell) |
| Max alarm latency | 1.00 s |
| **Acute false positives during alert stretches** | **0** |
| Frame-level closure P / R / F1 | 0.973 / 0.973 / 0.973 |
| PERCLOS mean absolute error vs ground truth | 0.88% |

PERCLOS tracked the true closure fraction to within 0.88 percentage points
(peak measured 28.5% against a true 30.1%). One episode in six was missed —
state that, rather than rounding it away.

### Performance (`scripts/benchmark.py`)

| Metric | Measured | Requirement |
|---|---|---|
| Mean YOLO inference @ 320 px | 8.12 ± 0.14 ms | < 100 ms |
| Max YOLO inference | 8.60 ms | < 500 ms |
| Achievable capture rate | 123 FPS (client runs at 10) | >= 10 FPS |
| Telemetry round-trip (local server) | 1.17 ms | < 500 ms |
| End-to-end alert latency | 9.29 ms | < 500 ms |
| Packet loss over 500 requests | 0.0% | < 5% |
| Peak client RSS | 418 MB | < 500 MB |
| Telemetry bandwidth | ~2 KB/s | ~1 KB/s |

Two of these deserve honesty in the report: RSS at 418 MB is close to the
500 MB ceiling, and bandwidth is double the stated ~1 KB/s target.

## Report checklist

Every measured number changed when the vision engine changed. Before submitting:

- [ ] Table 4.13 — per-class precision/recall/F1 (see Measured results above;
      raw JSON in `runs/detector_test_report.json` and `runs/bilstm_test_report.json`)
- [ ] Table 4.15 — latency and memory from `scripts/benchmark.py`
- [ ] Detector mAP from the `train_yolo.py` output (new table)
- [ ] Parameter count: the BiLSTM here is **81,412** parameters, not ~100K
- [ ] New table — the YOLO generation comparison, and why v8n was selected
- [ ] §3.x — MediaPipe/468 landmarks/TrueEAR replaced by YOLO detection
- [ ] §4.4.2 — the TrueEAR code segment no longer exists; PERCLOS replaces it
- [ ] §5.3 — drop the "2-second calibration assumes alertness" limitation
- [ ] §5.3 — drop "requires internet"; the local path now covers it
- [ ] §3.4.2 — the split is video-disjoint, not person-disjoint; fix the
      claimed 35,735/8,655/8,941 figures to the real ones
- [ ] §3.4.1 — FL3D has no bounding boxes and only 5 landmarks; say so, and
      state that composite training backgrounds are synthetic
- [ ] New limitation — synthetic backgrounds are a domain gap vs real cabins
- [ ] New limitation — 1 microsleep episode in 6 was missed on the test clip
- [ ] New section — system-level evaluation (`runs/system_test_report.json`);
      the original report has no equivalent and it is the strongest result here
- [ ] Figures 4.2–4.5 — redraw for the YOLO pipeline
