# DrowSim — Report Corrections

Every change the YOLO rebuild requires, section by section. Text in **Replace with**
blocks is written to be pasted straight into the thesis.

Source data for every number: `runs/detector_test_report.json`,
`runs/bilstm_test_report.json`, `runs/system_test_report.json`.

---

## Priority 1 — Wrong facts that an examiner can check

### ABSTRACT (p. v) and ABSTRAK (p. vi)

The abstract describes MediaPipe, TrueEAR, TLS and a cloud server. All four have changed.

**Replace with:**

> Driver drowsiness is among the factors that contribute to road accidents; hence the
> need for detecting drowsiness in real time to reduce road accidents. This project,
> DrowSim, is a privacy-respecting client–server driver monitoring system that detects
> drowsiness from facial behaviour captured by an ordinary RGB camera. The client node
> runs a YOLOv8n object detector locally, which localises the driver's face and
> classifies its state as alert, microsleep or yawning in a single pass. Each frame is
> reduced to a seven-element numerical feature vector; the frame itself is never stored
> or transmitted. A deterministic rule engine runs on the client and raises the alarm
> itself, so a microsleep is still detected when the network is unavailable. The server
> receives only the numerical telemetry, applies a Bidirectional Long Short-Term Memory
> network with Attention for temporal classification, maintains the session record and
> serves a monitoring dashboard. Evaluated on 3,810 unmodified frames from three unseen
> videos of the FL3D dataset, the detector achieved 98.48% accuracy and a macro-averaged
> F1-score of 0.9807. At system level, the rule engine detected five of six microsleep
> episodes on a labelled test clip with no false alarms, and tracked PERCLOS to within
> 0.9 percentage points of ground truth. This project presents an inexpensive, scalable
> and privacy-conscious solution for vehicles that lack built-in driver monitoring
> systems.

*The Malay ABSTRAK needs the same changes — she should translate the replacement above.*

### §3.4.1 Dataset Description (p. 21)

The current text implies FL3D provides bounding boxes. It does not.

**Add after the existing description:**

> FL3D distributes pre-cropped facial images of approximately 210 × 290 pixels rather
> than full video frames. Each frame carries a `driver_state` label and five facial
> landmarks produced by RetinaFace — the two eye centres, the nose tip and the two mouth
> corners. The dataset contains no bounding boxes and no eyelid contours. This has two
> consequences for the present work. First, an eye aspect ratio cannot be computed from
> FL3D at all, since that measurement requires eyelid landmarks; eye closure is therefore
> read from the detector's own class confidence. Second, an object detector cannot be
> trained directly on images without boxes, so training images were produced by
> compositing each FL3D crop onto a synthesised low-light cabin background at a random
> scale and position, with the paste rectangle serving as the ground-truth bounding box.
> A quarter of the frames were additionally emitted as unmodified tight crops so that the
> detector remains accurate on close faces and can be evaluated against unaltered FL3D
> data.

### §3.4.2 Data Partitioning Strategy (p. 22) — **factually wrong, must change**

The thesis states 30/7/7 drivers and 35,735 / 8,655 / 8,941 frames. FL3D's actual
partition is 48,327 / 2,502 / 2,502 frames over 39 / 5 / 5 **videos**. The class totals
in §3.4.3 (38,954 / 8,880 / 5,497 = 53,331) are correct and need no change.

**Replace the section with:**

> FL3D is distributed with a predefined partition of 48,327 training frames drawn from
> 39 videos and 5,004 held-out frames drawn from a further five videos, the latter split
> by the dataset authors into equal validation and test sets of 2,502 frames each. Those
> two sets are drawn from the same five videos, which means that selecting a model on the
> validation set leaks information into the test set. To avoid this, the five held-out
> videos were instead divided between validation and test, so that no video contributes
> to both. Only two of the five contain microsleep frames, so the division was not made
> at random: every possible assignment was enumerated, those failing to cover all three
> classes on both sides were discarded, and the assignment maximising worst-case class
> support was selected, with test support scored first.
>
> It must be stated that this split is video-disjoint rather than person-disjoint. FL3D
> exposes video identifiers but not driver identifiers, and the underlying NITYMED
> recordings include several videos per driver. A person-disjoint split cannot be
> constructed from the published data, and the possibility that a driver appears in both
> the training and test partitions cannot be excluded.

### New section: Model Selection (insert before the results tables)

The specification's §3.4 table compares YOLO generations using published benchmarks.
Measuring them on this task gives a different answer, and that is worth reporting.

> **Model Selection.** The four candidate YOLO generations were each trained on identical
> data with identical hyperparameters and a fixed seed, then scored the same way: the
> predicted class against ground truth on the 3,810 unmodified FL3D test frames.
>
> | Model | Parameters | Epochs | Accuracy | Macro-F1 | Microsleep F1 | Latency |
> |---|---|---|---|---|---|---|
> | yolov8n **(selected)** | 3,011,433 | 21 | 0.9848 | 0.9807 | 0.9651 | 8.99 ms |
> | yolov9t | 2,005,993 | 20 | 0.9816 | 0.9769 | 0.9533 | 13.61 ms |
> | yolov10n | 2,708,210 | 20 | 0.9806 | 0.9755 | 0.9489 | 28.79 ms |
> | yolo11n | 2,590,425 | 31 | 0.9759 | 0.9696 | 0.9373 | 8.96 ms |
>
> YOLOv8n leads on every quality metric at effectively the same latency, and by almost
> three points of F1 on microsleep, the safety-critical class. The ordering is the reverse
> of release order, and YOLO11n was trained for more epochs than the models that
> outperformed it, so the result is not an artefact of undertraining. YOLOv8n was
> therefore selected for deployment on the strength of this measurement rather than on
> published benchmarks.
>
> Two qualifications. Each model was trained once with a single seed, so the 0.9-point
> spread in accuracy is suggestive rather than established; the microsleep gap is wide
> enough to act upon. YOLOv10n's 28.79 ms latency reflects the behaviour of its
> Ultralytics implementation on Apple MPS and should not be read as evidence that the
> NMS-free architecture is inherently slow.

### Table 4.13 — Per-Class Model Performance (p. 54)

Replace the whole table. **New caption:** *Per-Class Detector Performance on Unmodified
FL3D Test Frames*

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| alert | 0.9878 | 0.9901 | 0.9890 | 2,622 |
| microsleep | 0.9599 | 0.9704 | 0.9651 | 642 |
| yawning | 1.0000 | 0.9762 | 0.9880 | 546 |
| Macro avg | 0.9826 | 0.9789 | 0.9807 | 3,810 |
| Weighted avg | 0.9849 | 0.9848 | 0.9848 | 3,810 |

**Supporting paragraph:**

> The detector achieved 98.48% overall accuracy and a macro-averaged F1-score of 0.9807
> on 3,810 unmodified FL3D frames drawn from three videos absent from training. No frame
> failed to produce a detection. Microsleep remains the weakest class at F1 0.9651, with
> the confusion concentrated between microsleep and alert; this is acceptable because the
> deterministic rule engine, not the learned model, is the sole arbiter of microsleep.
>
> Detection mAP@0.5 measured on the composite validation set reaches 0.994. That figure
> is deliberately not presented as the headline result: a sharp face composited onto a
> featureless blurred background makes localisation close to trivial, and the figure
> therefore overstates performance on real imagery. The per-class table above, computed
> on unaltered frames, is the meaningful measurement.

### Table 4.15 — Latency and Performance (p. 55)

| Metric | Measurement | Requirement | Status |
|---|---|---|---|
| Mean YOLO inference latency (320 px) | 8.12 ± 0.14 ms | < 100 ms | Pass |
| Max YOLO inference latency | 8.60 ms | < 500 ms | Pass |
| Achievable capture rate | 123 FPS (throttled to 10 by design) | ≥ 10 FPS | Pass |
| Telemetry round-trip latency | 1.17 ms (local server) | < 500 ms | Pass |
| End-to-end alert latency | 9.29 ms | < 500 ms | Pass |
| Client memory usage | 418 MB | < 500 MB | Pass |
| Packet loss over 500 requests | 0.0% | < 5% | Pass |
| Telemetry bandwidth | ≈ 2 KB/s | ≈ 1 KB/s | **Exceeded** |

> Two figures deserve comment rather than presentation as unqualified successes. Client
> memory usage of 418 MB, while within the 500 MB requirement, leaves limited headroom.
> Telemetry bandwidth of approximately 2 KB/s is double the 1 KB/s target stated in
> Table 3.3; the requirement was not met, and the surplus is attributable to transmitting
> a seven-element vector per frame together with per-packet session metadata.
>
> The alert is raised locally by the rule engine, so the latency experienced by the driver
> is the inference figure alone. The network round-trip affects only the dashboard and the
> model's contextual verdict.

### §4.9.3 — the reported client frame rate

The client runs at **10 FPS by design**, matching the training sample rate (FL3D video is
30 fps and every third frame was retained, so a 32-frame BiLSTM window spans 3.2 s in
both training and deployment). The machine can sustain far more; it is throttled
deliberately. Say so explicitly, or the 118 FPS figure and the 10 FPS figure will look
contradictory.

---

## Priority 2 — Sections describing components that no longer exist

### §4.3.2 Landmark Extraction Module → rename **YOLO Detection Module**

| Field | Description |
|---|---|
| Purpose | Locate the driver's face and classify its state in a single pass |
| Input | RGB frame array and camera resolution |
| Processing | YOLOv8n detector at 320 × 320 with confidence threshold 0.25; the highest-confidence detection is retained, so a passenger's face cannot capture the signal |
| Output | Bounding box, predicted class and confidence to the feature extraction module |

### §4.3.3 Feature Extraction Module

| Field | Description |
|---|---|
| Purpose | Convert one detection into the seven-element vector transmitted to the server |
| Input | Bounding box, class and confidence; frame dimensions |
| Processing | Confidence distributed over the three classes; box aspect ratio, normalised area and normalised centre coordinates computed |
| Output | `[p_alert, p_microsleep, p_yawning, bbox_aspect, bbox_area, center_x, center_y]` |

### §4.4.2 TrueEAR Computation — **delete this section entirely**

TrueEAR no longer exists. Replace it with a PERCLOS section:

> **PERCLOS Computation.** PERCLOS is the proportion of a trailing sixty-second window
> during which the driver's eyes are closed, and is the established measure of drowsiness
> in the driver-monitoring literature. The client maintains a time-weighted history of
> per-frame closure decisions, trimming entries older than the window on each update, and
> reports the ratio of closed time to total observed time. The score is withheld until at
> least twenty seconds of history have accumulated, since a ratio computed over a few
> frames is not meaningful.
>
> PERCLOS alone is insufficient. A three-second microsleep occurring within a
> sixty-second window yields a PERCLOS of only 5%, far below any workable threshold, so
> the single most dangerous event in the problem domain is invisible to the metric. The
> system therefore evaluates two channels in parallel: an acute channel that raises a
> critical state after 0.8 s of continuous eye closure, and this trend channel. The
> implementation was validated against a labelled clip and tracked the true closure
> fraction to within 2.4 percentage points.

### §4.4.4 Rule-Based Microsleep Detection

The 0.8 s dwell requirement is unchanged, but the signal and the calibration are not:

> Eye closure is determined from the detector's microsleep class confidence rather than
> from an eye aspect ratio. A Schmitt trigger is applied — closure is entered at a
> confidence of 0.50 and left at 0.35 — so that a borderline frame cannot cause the state
> to oscillate. Closure lasting 0.8 s or more raises a critical state; episodes shorter
> than 0.4 s are recorded as blinks. No per-driver calibration phase is required, which
> removes the assumption, present in the earlier design, that the driver is alert during
> the opening seconds of a session.

### §4.5 Backend and Cloud Deployment

The AWS EC2 instance, the Nginx reverse proxy, the Let's Encrypt certificate and the
`drowsim.fyp.systems` domain are **not part of the delivered system**. Either remove
§4.5.1–4.5.4 or rewrite them as future work. Replace with:

> The inference server is a FastAPI application served by Uvicorn. For the present
> evaluation it runs on the same host as the client. Because the safety-critical decision
> is taken on the client, the server's availability does not affect whether the driver is
> alerted; deployment to a public host would extend the monitoring dashboard to a fleet
> operator without altering the detection path. Transport security would be provided by a
> reverse proxy terminating TLS, as is standard, but no such deployment was carried out
> and no such claim is made.

**Also correct the abstract and §3.3 (F1.1, F3.1), which currently assert TLS.**

### §4.6 Libraries and Tools

Remove MediaPipe and Pygame. Add Ultralytics 8.4 (YOLOv8n), PyTorch 2.14 with Apple MPS,
OpenCV 5.0, FastAPI, Uvicorn, scikit-learn, Pillow.

---

## Priority 3 — New material that strengthens the report

### New section: System-Level Evaluation (insert after §4.9.3)

This has no equivalent in the current report and is the strongest result in it.

> **System-Level Evaluation.** Per-class accuracy measures the detector, not the system.
> To establish whether the complete pipeline responds to a driver falling asleep, a
> ninety-second clip was rendered from a held-out FL3D video into a 640 × 480 scene with
> a frame-level ground-truth label track, and replayed through the full client stack.
>
> | Measure | Result |
> |---|---|
> | Microsleep episodes detected by the acute rule | 5 of 6 |
> | False alarms during alert driving | 0 |
> | Mean alarm latency from episode onset | 0.86 s |
> | Maximum alarm latency | 1.00 s |
> | Frame-level closure precision / recall / F1 | 0.973 / 0.973 / 0.973 |
> | PERCLOS mean absolute error against ground truth | 0.88 percentage points |
>
> Of the 0.86 s mean latency, 0.8 s is the dwell the rule deliberately requires before
> it may fire, leaving approximately 0.06 s of detection lag. Credit was given only to the
> acute channel: counting any critical state would have scored an alarm still sounding
> from a previous episode as a detection of the next, flattering both recall and latency.
>
> One episode of six was not detected. Separately, a critical state raised by PERCLOS
> persists for as long as the score remains above threshold, which on a genuinely drowsy
> driver means the alarm does not clear; an escalation policy is required before
> deployment.

### §5.3 Limitations — remove two, add four

**Remove:** the two-second calibration assumption (no calibration phase exists) and the
internet-connection requirement (detection and alerting are now local).

**Add:**

> **Synthetic training backgrounds.** FL3D supplies no full frames, so the detector was
> trained on crops composited onto synthesised low-light backgrounds. Real cabin interiors
> differ, and a domain gap of unknown size remains.
>
> **Split cannot be shown person-disjoint.** FL3D exposes video identifiers only. The
> split is video-disjoint; a driver may appear in both partitions.
>
> **Persistent critical state.** Because PERCLOS is computed over a trailing sixty-second
> window, a driver who is genuinely drowsy keeps the system in a critical state
> continuously. The present implementation repeats the alarm rather than escalating it,
> which risks alarm fatigue.
>
> **Undetected episode.** One microsleep episode in six was missed in system-level testing.

### §5.2 Strengths — add

> **No calibration required.** Because eye closure is read from a trained detector rather
> than from a geometric ratio, no per-driver baseline is established at session start.
> The system makes no assumption about the driver's state when monitoring begins.
>
> **Alerting survives loss of connectivity.** The rule engine runs on the client and
> raises the alarm directly. Verified by running the client with the server unreachable:
> detection and alerting continued unchanged.

---

## Checklist

- [ ] Abstract and Abstrak rewritten
- [ ] §3.4.1 dataset description corrected
- [ ] §3.4.2 partition figures corrected; person-disjoint claim withdrawn
- [ ] §3.5 hardware: laptop, not Raspberry Pi
- [ ] §4.3.2 / §4.3.3 module tables replaced
- [ ] §4.4.2 TrueEAR removed, PERCLOS added
- [ ] §4.4.4 rule description updated
- [ ] §4.5 AWS sections removed or moved to future work
- [ ] §4.6 library list updated
- [ ] Model selection section added (YOLO generation comparison)
- [ ] Table 4.13 replaced
- [ ] Table 4.15 replaced, bandwidth shortfall acknowledged
- [ ] System-level evaluation section added
- [ ] §5.2 strengths updated
- [ ] §5.3 limitations updated
- [ ] Figures 4.2–4.5 redrawn for the YOLO pipeline
- [ ] Logbook signed by supervisor and attached as Appendix
