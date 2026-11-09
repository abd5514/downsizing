# CHAPTER I: INTRODUCTION

## 1.1 Overview

Road safety remains a major global concern, and driver fatigue is one of its most persistent and least visible contributors. Unlike alcohol or speeding, fatigue leaves no residue and no instrument reading; it builds gradually, erodes attention before the driver notices, and culminates in microsleeps — brief lapses of consciousness lasting a few seconds during which the vehicle is effectively unpiloted. At 90 km/h, a three-second microsleep carries a car approximately 75 metres with nobody watching the road.

Vision-based Driver Monitoring Systems (DMS) address this by observing the driver's face and inferring fatigue from behavioural signals such as eye closure, blink dynamics and yawning. The technology is mature enough to appear in production vehicles, but it remains concentrated in newer premium models. The vehicles statistically most involved in fatigue-related collisions — older cars, commercial fleets, and vehicles driven long distances at night — typically have no such system and no practical route to acquiring one.

A second obstacle is privacy. Many proposed vision-based systems transmit raw facial video to a remote server for processing. A continuous video stream of a driver's face is biometric personal data of the most sensitive kind. Transmitting and storing it creates exposure to interception, secondary use and unauthorised monitoring, and makes the system dependent on a stable network connection for a safety function that must not fail.

This project develops **DrowSim**, a privacy-preserving client–server driver monitoring system. A YOLO object detector runs locally inside the vehicle on ordinary consumer hardware, reducing each video frame to a small vector of numbers. Only those numbers are transmitted; no image data leaves the vehicle in any form. The decision that raises the alarm is taken locally, so the driver is still warned when the network is unavailable. The server receives the numerical telemetry, applies a temporal deep-learning model for contextual classification, and drives a monitoring dashboard.

## 1.2 Background Study and Motivation

The World Health Organization (2023) reports approximately 1.19 million road traffic deaths annually. Fatigue and distraction account for a substantial share. Essahraui et al. (2025) report that drowsy driving in the United States is associated with roughly 83,000 crashes, 37,000 injuries and 900 deaths each year, and these figures are widely regarded as underestimates: fatigue is difficult to establish after a collision, and self-reporting is unreliable.

The physiological mechanism makes the problem particularly difficult to self-manage. Safarov et al. (2023) describe fatigue as accumulating gradually and reducing alertness before the driver becomes subjectively aware of impairment. The microsleep — the specific event most strongly associated with fatigue-related crashes — frequently goes entirely unnoticed by the person experiencing it. A driver cannot reliably detect their own drowsiness, which is precisely why an external monitor has value.

Three factors motivate the present work:

**Coverage.** Factory-fitted DMS are available largely in newer and higher-specification vehicles. Albadawi et al. (2022) note that systems requiring specialised in-cabin hardware are costly and unsuited to large-scale deployment in mainstream or legacy vehicles. A retrofittable system built on an ordinary camera and consumer computing hardware addresses a population that current solutions do not reach.

**Privacy.** Fonseca and Ferreira (2025), in a systematic review of deep-learning drowsiness detection, identify reliance on cloud processing of raw facial video as a recurring characteristic of proposed systems, with the associated latency, connectivity dependence and privacy exposure. Beugin et al. (2022) address the broader problem of privacy-preserving protocols for smart cameras, arguing that the strongest guarantees come from architectural constraints on what data exists rather than from policy promises about how it is handled.

**Integration of privacy with temporal modelling.** Das et al. (2024) and Kong et al. (2024) demonstrate that hybrid architectures combining spatial feature extraction with temporal models such as LSTM and BiLSTM perform well on fatigue-related behaviours, since eye closure and yawning are inherently sequential phenomena. However, this line of work has generally not been integrated with privacy-preserving design. The combination — accurate temporal analysis performed on data that is never sensitive in the first place — remains underexplored.

## 1.3 Problem Statement

Despite substantial progress in automotive safety technology, existing driver drowsiness detection systems exhibit limitations that restrict their practicality, affordability and ethical acceptability.

**First, physiological sensing is accurate but intrusive.** Approaches based on electroencephalography (EEG) or electrocardiography (ECG) achieve high accuracy but require electrodes or wearable instrumentation. Albadawi et al. (2022) identify these methods as intrusive, costly and unsuitable for routine deployment. A driver cannot reasonably be expected to attach electrodes before each journey.

**Second, cloud-based vision systems create privacy exposure and connectivity dependence.** Systems that transmit raw facial video for remote processing expose continuous biometric data to interception and secondary use, and introduce latency into a safety-critical path (Fonseca & Ferreira, 2025). More seriously, they make the alarm conditional on network availability. A system that suppresses alerts when connectivity is lost fails precisely in the conditions — remote roads, night driving, long journeys — where fatigue risk is highest.

**Third, the standard fatigue metric cannot detect the most dangerous event.** PERCLOS, the proportion of time the eyes are closed over a rolling window, is the established measure of drowsiness. However, a three-second microsleep occurring within a sixty-second window produces a PERCLOS value of only 5 per cent, far below any workable alarm threshold. A system relying on PERCLOS alone is structurally incapable of responding to the single event that most directly precedes a fatigue-related collision.

**Fourth, privacy-preserving design and temporal deep learning have not been combined.** Although hybrid CNN–LSTM and BiLSTM architectures perform well (Das et al., 2024; Kong et al., 2024), their integration with privacy-by-design principles remains limited, leaving a gap for an edge-compatible, privacy-focused system that supports real-time temporal analysis without transmitting raw video.

## 1.4 Research Questions

Arising from the limitations identified above, this research addresses the following questions:

1. How can a vision-based driver drowsiness detection system recognise fatigue-related facial behaviours — eye closure, microsleep and yawning — accurately and in real time using an object detection model on consumer hardware?
2. What privacy-preserving techniques can be integrated into a driver monitoring framework so that data confidentiality is guaranteed by the system architecture rather than by policy, without reducing detection accuracy?
3. How should the detection logic be structured so that both acute microsleep events and gradual fatigue trends are detected, given that neither channel alone is sufficient?
4. Where should the decision that raises the alarm be located in a client–server architecture, such that the driver is alerted reliably even when connectivity is lost?

## 1.5 Research Objectives

This study aims to design, develop and evaluate a privacy-preserving driver drowsiness detection system. The specific objectives are:

1. **To design and train a YOLO-based detection model** capable of locating a driver's face and classifying its state as alert, microsleep or yawning in a single inference pass on consumer hardware.
2. **To design a privacy-preserving telemetry representation** that reduces each video frame to a compact numerical vector, such that no image data is transmitted or stored, and to verify this property structurally rather than by assertion.
3. **To design a dual-channel decision mechanism** combining a deterministic acute-closure rule with a PERCLOS trend measure, so that both microsleep events and gradual fatigue are detected.
4. **To design an edge-authoritative client–server architecture** in which the safety-critical decision is taken locally and the server provides temporal classification, session recording and fleet monitoring.
5. **To evaluate the proposed system** at model level, system level and performance level, establishing detection accuracy on unseen drivers, alarm reliability against labelled ground truth, and latency and resource consumption on the target hardware.

## 1.6 Research Scope

**Technical scope.** The system uses computer vision and deep learning to detect facial behaviours indicating drowsiness: eye closure, microsleep and yawning. Detection uses the YOLO family of object detectors; temporal analysis uses a Bidirectional Long Short-Term Memory network with an attention mechanism. The system is designed to run on ordinary consumer hardware with an RGB camera.

**Experimental scope.** Development and evaluation use the FL3D frame-level driver drowsiness dataset (Muc, 2022), derived from the NITYMED night-time driving recordings. The study is limited to visual facial features; physiological signals such as heart rate or EEG, and vehicle signals such as steering angle or lane position, are outside scope.

**Functional scope.** The system's output is limited to classifying driver state and raising audible and visual alerts. It does not control the vehicle, perform any driver-assistance function, or intervene in driving. It is a proof of concept intended to establish technical feasibility, not a commercial product, and no claim is made regarding regulatory certification.

**Exclusions.** Daytime and high-glare conditions are not evaluated, as the available dataset is night-time only. Drivers wearing sunglasses are outside scope, since opaque lenses prevent observation of eye state by any RGB-camera method. Deployment to a public cloud host is not undertaken.

## 1.7 Project Timeline

The project follows an incremental development model across fifteen weeks, with each phase producing a working increment rather than deferring integration to the end.

| Phase | Weeks | Activity |
|---|---|---|
| 1 | 1–3 | Project planning, title confirmation, objective definition |
| 2 | 4–6 | Literature review and identification of the research gap |
| 3 | 7–9 | System design, dataset preparation and model development |
| 4 | 10–12 | Prototype integration, testing and performance evaluation |
| 5 | 13–15 | Report preparation, presentation and submission |

*Table 1.1: Implementation schedule of the DrowSim project*

## 1.8 Structure of the Report

**Chapter I** introduces the project, establishes the problem, and states the research questions, objectives and scope.

**Chapter II** reviews the literature on driver drowsiness detection, covering physiological, vehicle-based and vision-based approaches, the established fatigue metrics, deep-learning methods, existing commercial systems, and privacy considerations, concluding with the identified research gap.

**Chapter III** presents the research methodology: the development model, the dataset and its partitioning, the detection and temporal modelling approaches, the decision logic, and the evaluation strategy.

**Chapter IV** presents the system analysis and design: requirements, architecture, module design, data design, decision logic design, interface design, and the preliminary results obtained from an initial prototype.

---

# CHAPTER II: LITERATURE REVIEW

## 2.1 Introduction

This chapter reviews existing work on driver drowsiness detection. It begins by establishing how drowsiness is defined and measured, then examines the three principal categories of detection approach and the metrics used in vision-based systems. It reviews deep-learning methods, surveys commercial driver monitoring systems, and examines privacy considerations in camera-based monitoring. The chapter concludes by identifying the gap that this project addresses.

## 2.2 Driver Drowsiness and Its Measurement

Drowsiness is a transitional state between wakefulness and sleep, characterised by reduced vigilance, slowed reaction time and impaired judgement. For detection purposes, the important property is that it manifests through observable behavioural changes before it produces a critical failure.

Three behavioural signatures dominate the literature. **Prolonged eye closure** is the most direct indicator; the microsleep, an involuntary closure lasting from a fraction of a second to several seconds, is the event of greatest safety significance. **Blink dynamics** change measurably with fatigue: blinks lengthen and blink rate alters before the driver reports feeling tired. **Yawning** is a well-established correlate, though it is a weaker signal individually, since mouth opening also accompanies speech and laughter.

A critical distinction runs through this literature and shapes the present design: **acute events and gradual trends are different phenomena requiring different detection mechanisms.** A single microsleep is an acute event demanding immediate response. Accumulating fatigue is a trend visible only over a window of time. A system designed to detect one will not reliably detect the other.

## 2.3 Categories of Detection Approach

### 2.3.1 Physiological approaches

Physiological methods measure signals correlated with the sleep–wake transition directly. EEG measures cortical activity and is generally regarded as the most accurate available indicator; ECG-derived heart rate variability and electrooculography provide alternative signals.

Albadawi et al. (2022), reviewing developments in drowsiness detection, characterise these methods as accurate but intrusive. They require electrodes or wearable sensors in contact with the driver, are sensitive to motion artefacts in a moving vehicle, and impose a preparation burden incompatible with everyday driving. Their principal research value is as a ground-truth reference against which less intrusive methods can be validated.

### 2.3.2 Vehicle-based approaches

Vehicle-based methods infer driver state from the vehicle's behaviour: steering wheel movement patterns, lane departure frequency, or variability in lane position. These require no observation of the driver and are entirely non-intrusive, which is why several manufacturers adopted them first.

Their weakness is indirection. Steering behaviour is confounded by road geometry, crosswind, traffic density and individual driving style. More fundamentally, they detect the *consequences* of impaired attention rather than impairment itself, which means detection necessarily lags the onset of danger. They are also poorly suited to detecting a brief microsleep on a straight road, where the vehicle may not deviate measurably within the duration of the event.

### 2.3.3 Vision-based approaches

Vision-based methods observe the driver directly through a camera and infer state from facial behaviour. They are non-intrusive, require only a camera, and observe the driver rather than a proxy — making them the dominant approach in recent literature and in production systems.

Albadawi et al. (2023) demonstrate real-time machine-learning drowsiness detection from visual features, and Owen and Surantha (2025) show that handcrafted feature extraction can achieve viable performance on edge computing devices, which is directly relevant to a system intended for consumer hardware. The principal limitations are sensitivity to illumination, head pose and occlusion, and — for cloud-based implementations — the privacy exposure created by transmitting facial imagery.

## 2.4 Metrics for Vision-Based Detection

### 2.4.1 Eye Aspect Ratio

The Eye Aspect Ratio (EAR) is computed from facial landmarks around the eyelid as the ratio of vertical to horizontal eye dimensions. It is computationally trivial and interpretable, which accounts for its popularity.

Two weaknesses limit it. EAR is strongly person-specific: baseline values vary with eye shape and the presence of spectacles, so a fixed threshold that works for one driver may fail for another. It is also sensitive to head pose, since pitch compresses the apparent vertical dimension and lowers EAR without any actual eye closure. Systems using EAR therefore generally require a per-driver calibration phase, which introduces its own assumption — that the driver is alert while the baseline is established.

### 2.4.2 Mouth Aspect Ratio

The Mouth Aspect Ratio (MAR) applies the same geometric principle to mouth opening and is used to detect yawning. Because speech and laughter also open the mouth, MAR requires a duration criterion: a yawn is distinguished from ordinary mouth movement by sustained opening, typically beyond one to two seconds.

### 2.4.3 PERCLOS

PERCLOS — the percentage of time the eyes are closed beyond a threshold over a rolling window, conventionally sixty seconds — is the most established measure of drowsiness in the driver-monitoring literature and is widely treated as a reference standard.

Its strength is robustness: by aggregating over a window it is insensitive to individual frame errors and captures gradual decline that no single frame reveals. Its limitation is a direct consequence of the same property. **A short microsleep is diluted by the window.** Three seconds of closure within sixty seconds yields a PERCLOS of 5 per cent, well below thresholds typically cited in the range of 12 to 15 per cent. PERCLOS is therefore necessary but not sufficient: it measures the trend correctly while being structurally blind to the acute event.

This observation directly motivates the dual-channel design adopted in this project.

!FIG:fig_2_2_perclos_dilution.png|Figure 2.1: Why PERCLOS alone cannot detect a microsleep. A three-second closure within the sixty-second window scores 5%, far below any workable threshold.|6.0

## 2.5 Deep Learning Approaches

### 2.5.1 Frame-level classification

Convolutional neural networks applied to individual frames or cropped eye regions can classify driver state with high accuracy. Albadawi et al. (2023) and Essahraui et al. (2025) report strong results from this approach.

Its limitation is the absence of temporal context. Fatigue is defined partly by duration — the difference between a blink and a microsleep is how long the eyes remain closed — and a classifier examining one frame cannot represent duration. Frame-level classification must therefore be paired with some external temporal mechanism.

### 2.5.2 Temporal models

Recurrent architectures address this directly. Soman et al. (2024) present a CNN–LSTM framework for drowsiness monitoring, and Kong et al. (2024) employ a BiLSTM within a multi-modal fatigue detection method. Das et al. (2024) apply deep learning with a U-Net-based architecture to facial movement analysis.

Bidirectional LSTMs are well suited to this task because they process a sequence in both directions, allowing the classification of a given instant to draw on frames that follow it as well as precede it — useful when the boundaries of an event such as a yawn are gradual. Attention mechanisms further allow the network to weight the most discriminative time steps rather than treating all frames in a window as equally informative.

### 2.5.3 Object detection and the YOLO family

The YOLO (You Only Look Once) family performs localisation and classification in a single forward pass, which makes it substantially faster than two-stage detectors and well suited to real-time use on constrained hardware.

The family has evolved rapidly. YOLOv8 (Jocher et al., 2023) introduced an anchor-free design with a decoupled detection head. YOLOv9 (Wang et al., 2024a) introduced Programmable Gradient Information and the GELAN backbone to improve information retention through the network. YOLOv10 (Wang et al., 2024b) introduced NMS-free training through dual label assignment, removing non-maximum suppression from the inference path. YOLO11 (Khanam & Hussain, 2024) introduced C3k2 blocks and spatial attention, reported as achieving higher accuracy with fewer parameters than YOLOv8.

For driver monitoring, object detection offers a specific advantage over landmark-based pipelines: **it distinguishes "no face detected" from "face detected, driver alert."** A landmark or classification pipeline given an obscured camera will generally still emit a state, whereas a detector returning no detection provides an unambiguous signal that monitoring has failed. Given that an obscured camera silently reporting "alert" is the most dangerous possible failure mode, this distinction is significant.

It should be noted that published comparisons between YOLO generations are conducted on general-purpose benchmarks such as COCO. Whether the ordering established there transfers to a specific narrow task is an empirical question, and this project treats it as one.

## 2.6 Review of Existing Systems

**Tesla** implements driver monitoring in production vehicles using a cabin-facing camera to assess driver attentiveness (Tesla, 2024). The system is tightly integrated with the vehicle's driver-assistance features and is unavailable outside Tesla vehicles.

**Mobileye** offers a driver monitoring system intended for integration by vehicle manufacturers, combining in-cabin sensing with the company's road-facing systems (Mobileye, 2022). It is a supplier product requiring OEM integration rather than a retrofittable solution.

**Seeing Machines** produces the Guardian system, aimed at commercial fleets and using camera-based monitoring with an operator escalation service (Seeing Machines, 2024). It is retrofittable but priced for fleet operators and involves transmitting event data to an external monitoring centre.

**Acura** implements a Driver Attention Monitor using vehicle-based signals rather than direct observation of the driver (Acura, 2024), inheriting the indirection limitations discussed in Section 2.3.2.

### 2.6.1 Comparison of existing systems

| System | Approach | Retrofittable | Local processing | Cost tier | Availability |
|---|---|---|---|---|---|
| Tesla DMS | Cabin camera, vision | No | Yes (in-vehicle) | Included with vehicle | Tesla vehicles only |
| Mobileye DMS | Cabin camera, vision | No (OEM integration) | Yes | OEM-negotiated | Via manufacturers |
| Seeing Machines Guardian | Cabin camera, vision + human escalation | Yes | Partial; events transmitted | Fleet pricing | Commercial fleets |
| Acura Driver Attention Monitor | Vehicle-based signals | No | Yes | Included with vehicle | Acura vehicles |
| **DrowSim (proposed)** | Cabin camera, vision | **Yes** | **Yes; alarm decided locally** | **Consumer hardware** | **Any vehicle** |

*Table 2.1: Comparison of existing driver monitoring systems*

The comparison shows a consistent pattern: systems offering local processing are tied to specific vehicles, while the retrofittable option transmits data externally. No reviewed system combines retrofittability, local decision-making and a guarantee that facial imagery never leaves the vehicle.

## 2.7 Privacy in Vision-Based Monitoring

A camera continuously observing a driver's face collects biometric personal data. Where such data is transmitted or stored, it becomes subject to data protection regimes including the General Data Protection Regulation in the European Union and the Personal Data Protection Act 2010 in Malaysia, and it creates practical exposure to interception, secondary use and function creep.

Beugin et al. (2022), examining privacy-preserving protocols for smart cameras and IoT devices, argue that meaningful guarantees derive from architectural constraints on what data exists and where, rather than from policy commitments about handling. Applied to driver monitoring, this suggests that the strongest available guarantee is achieved by ensuring that imagery is never transmitted at all — a property that can be verified by inspecting the transmission format, rather than trusted on assertion.

A distinction worth preserving is that between anonymity and pseudonymity. Telemetry consisting only of numerical features contains no imagery, but if it is associated with a persistent device identifier it remains linkable to an individual and is therefore pseudonymous rather than anonymous. Overstating this property would be a weakness in any system claiming privacy preservation.

## 2.8 Research Gap

The review supports four observations:

1. Vision-based detection is the most practical approach for retrofit deployment, but existing implementations either tie local processing to specific vehicles or transmit data externally.
2. Temporal deep-learning models substantially improve on frame-level classification, but this line of work has not generally been integrated with privacy-preserving architecture.
3. PERCLOS is the established fatigue metric but is structurally unable to detect short microsleep events, and systems relying on it alone inherit that blindness.
4. Systems that decide driver state on a remote server make a safety-critical function dependent on connectivity, failing in exactly the conditions where fatigue risk is greatest.

The gap addressed by this project is therefore the absence of a **retrofittable, edge-authoritative, privacy-preserving driver monitoring system that combines object detection with temporal modelling and detects both acute microsleep events and gradual fatigue trends.**

## 2.9 Conclusion

This chapter established that vision-based detection is the appropriate approach for the target deployment, that temporal modelling meaningfully improves on frame-level classification, and that the established fatigue metric requires supplementation by an acute-event channel. It identified that existing systems do not combine retrofittability with local decision-making and a structural guarantee against image transmission. Chapter III presents the methodology by which the proposed system addresses this gap.

---

# CHAPTER III: RESEARCH METHODOLOGY

## 3.1 Introduction

This chapter describes the methodology adopted to design, build and evaluate DrowSim. It sets out the development model, the dataset and how it was prepared, the approach taken to detection and temporal modelling, the design of the decision logic, and the strategy used to evaluate the system. Emphasis is placed throughout on decisions that were made empirically rather than by convention, and on constraints imposed by the data itself.

## 3.2 Research Design and Development Model

The project follows an **incremental development model**. Each increment produces a working end-to-end system rather than an isolated component, with functionality deepened at each pass. This model was chosen over a waterfall approach because the project contains a genuine technical unknown — whether an object detector can supply a reliable eye-closure signal — that is resolved only by building and measuring, not by specification.

Five increments were defined: dataset preparation; detector training; decision logic and temporal model; client–server integration; and evaluation. Integration was performed at every increment rather than deferred, on the reasoning that integration failures discovered late are the most expensive kind.

## 3.3 Dataset

### 3.3.1 Description

The study uses the **FL3D (Frame-Level Driver Drowsiness Detection)** dataset (Muc, 2022), constructed from the NITYMED night-time driving recordings. FL3D contains **53,331 labelled frames** across 44 videos, each frame assigned one of three states: alert, microsleep or yawning. The recordings were captured in real vehicles under night-time conditions with a dashboard-mounted camera, with participating drivers varying in hair colour, facial hair and use of spectacles.

!FIG:fig_3_2_dataset_samples.png|Figure 3.1: Representative FL3D frames for each class. The dataset distributes pre-cropped faces rather than full video frames.|4.6

The dataset's labelling convention is relevant to interpretation. Frames from microsleep sessions in which the eyes are closed are labelled microsleep; frames in which the mouth is open beyond approximately 2.5 cm during yawning sessions are labelled yawning; and **frames in which the driver blinks were removed and are not present in the dataset.** The last point matters: a model trained on FL3D has never observed an ordinary blink, so blink rejection must be handled by logic outside the learned model rather than by the model itself.

### 3.3.2 What FL3D provides, and what it does not

Direct inspection of the distributed dataset established two facts that materially shaped the system design and that are not evident from the dataset description alone.

**FL3D distributes pre-cropped facial images**, approximately 210 × 290 pixels, rather than full video frames. Each frame is accompanied by a state label and **five facial landmarks produced by RetinaFace**: the two eye centres, the nose tip and the two mouth corners.

Two consequences follow:

*An Eye Aspect Ratio cannot be computed from this dataset.* EAR requires landmarks tracing the eyelid contour. RetinaFace provides eye *centres*, not contours. No amount of processing recovers eyelid geometry from a single point per eye. Any method requiring EAR is therefore unavailable when working from FL3D, and eye closure must be obtained by another means.

*An object detector cannot be trained directly on the images as distributed*, because they contain no bounding boxes — the image is the face. Yet the deployed system must locate a face within a full camera frame before it can classify anything. This is addressed by compositing, described in Section 3.3.5.

### 3.3.3 Partitioning strategy

FL3D is distributed with a predefined partition: 48,327 training frames drawn from 39 videos, and 5,004 held-out frames from a further five videos, which the dataset authors divide into equal validation and test sets of 2,502 frames each.

Inspection revealed that **the distributed validation and test sets are drawn from the same five videos.** Selecting a model on the validation set therefore leaks information into the test set, since both contain the same drivers in the same sessions. To avoid this, the five held-out videos were instead divided between validation and test so that no video contributes to both.

That division was not made at random. **Only two of the five held-out videos contain any microsleep frames**, so an unlucky assignment leaves the validation set unable to measure the most safety-critical class, and model selection proceeds blind to it. Every possible assignment was therefore enumerated; those failing to cover all three classes on both sides were discarded; and the assignment maximising worst-case class support was selected, with support in the test set scored first on the reasoning that the test figure is the one reported.

An important qualification must be stated. **FL3D exposes video identifiers but not driver identifiers**, and the underlying NITYMED recordings include several videos per driver. The resulting split is therefore **video-disjoint, not person-disjoint**. A person-disjoint split cannot be constructed from the published data, and the possibility that a driver appears in both partitions cannot be excluded. Reporting the split as person-disjoint would overstate what the data supports.

### 3.3.4 Class distribution and imbalance

| Class | Frames | Proportion |
|---|---|---|
| alert | 38,954 | 73.0% |
| microsleep | 8,880 | 16.7% |
| yawning | 5,497 | 10.3% |
| **Total** | **53,331** | **100%** |

*Table 3.1: Class distribution in FL3D*

!FIG:fig_3_3_class_distribution.png|Figure 3.2: Class distribution in FL3D, approximately 7 : 1.6 : 1.|4.8

The distribution is approximately 7 : 1.6 : 1, reflecting the natural prevalence of these states in real driving. This imbalance is addressed during training using **Focal Loss** (α = 1, γ = 2), which dynamically down-weights easily classified examples and concentrates learning on difficult cases, rather than applying a fixed per-class weight as weighted cross-entropy would. Evaluation uses class-imbalance-robust metrics: per-class F1 and macro-averaged F1 rather than raw accuracy alone.

### 3.3.5 Preprocessing and composite generation

Because FL3D provides no bounding boxes, detector training images were produced by **compositing**: each FL3D crop is placed onto a synthesised low-light cabin background at a random scale, position and brightness, and the paste rectangle becomes the ground-truth bounding box. Face size varies from 18 to 60 per cent of frame width, spanning the range between a driver seated back from a dashboard camera and a close webcam view. Placement is biased towards the upper-middle region where a seated driver's head is actually found.

Backgrounds are generated by aggressive downsampling, blurring and darkening of an unrelated crop. The detail is destroyed rather than merely softened: a blurred face remains a face, and an unlabelled one in the background would penalise the model for a correct detection — label noise aimed precisely at the behaviour the system needs.

A quarter of the frames are additionally emitted as **unmodified tight crops** with a full-frame bounding box, which keeps the detector accurate on close faces and — importantly for evaluation — allows it to be scored against unaltered FL3D data.

!FIG:fig_3_4_composite.png|Figure 3.3: Composite generation. The distributed crop (left) carries no bounding box; compositing it onto a synthesised cabin background (right) derives one.|5.6

The limitation of this approach should be stated plainly: the backgrounds are synthetic, and real cabin interiors differ. A domain gap of unknown magnitude remains between training imagery and deployment conditions.

### 3.3.6 Privacy in training versus deployment

A distinction is maintained between offline research and deployed operation. FL3D is used as an offline benchmark during development, where the images necessarily exist as files. In the deployed system, no raw frame, derived feature, timestamp or log is written to persistent storage on either client or server. Frames exist only in volatile memory and are overwritten by the subsequent capture.

## 3.4 Detection Approach

### 3.4.1 Object detection rather than landmark extraction

The system uses a **YOLO object detector** trained to locate a face and classify its state in a single pass, rather than a landmark-extraction pipeline followed by geometric analysis. Three reasons support this choice.

First, as established in Section 3.3.2, EAR is not computable from the available dataset, removing the principal advantage of a landmark pipeline.

Second, detection produces an unambiguous **no-face** signal. A classifier presented with an obscured camera still emits a class; a detector returning no detection reports that monitoring has failed. Given that an obscured camera silently reporting "alert" is the most dangerous failure mode available to this system, this property has direct safety value.

Third, removing EAR removes the per-driver calibration phase that EAR-based systems require, and with it the assumption that the driver is alert during the calibration window — an assumption violated precisely when a driver begins a journey already fatigued.

### 3.4.2 Model selection by measurement

Published comparisons of YOLO generations are conducted on general-purpose benchmarks. Whether that ordering transfers to a narrow three-class task on night-time facial imagery is an empirical question, and this project treats it as one rather than adopting the newest available model by default.

Four candidates — YOLOv8n, YOLOv9t, YOLOv10n and YOLO11n — were each trained on identical data with identical hyperparameters and a fixed seed, then scored identically: predicted class against ground truth on the unmodified FL3D test frames. Results are reported in Section 4.8.2.

## 3.5 Feature Representation

Each detection is reduced to a **seven-element numerical vector**, which constitutes the entirety of what the client transmits:

`[p_alert, p_microsleep, p_yawning, bbox_aspect, bbox_area, center_x, center_y]`

The three probability terms distribute the detector's confidence across the state classes, giving the temporal model a graded rather than one-hot input. The four geometric terms — bounding box aspect ratio, normalised area, and normalised centre coordinates — capture head position and apparent distance, which carry information about head pose and nodding.

Only the highest-confidence detection in a frame is used. There is one driver, and taking the best detection prevents a passenger's face from capturing the signal.

This representation is the mechanism by which the privacy property is achieved. It is not a policy applied to image data; it is a format in which image data cannot be expressed.

## 3.6 Temporal Modelling

A **Bidirectional LSTM with additive attention** classifies sequences of thirty-two consecutive feature vectors. Bidirectionality allows the classification of a given instant to draw on subsequent frames as well as preceding ones, which suits events such as yawns whose boundaries are gradual. The attention mechanism weights the most discriminative time steps rather than treating every frame in the window as equally informative.

A methodological point governs how this model is trained. Training sequences are constructed by **running the trained detector over the dataset and recording its actual outputs**, rather than by assembling sequences from ground-truth labels. The temporal model therefore learns from the same distribution it will encounter at inference, including the detector's own errors. A temporal model trained on perfect labels and deployed on imperfect ones is a well-known route to a strong table and a disappointing system.

The window length of thirty-two frames is matched between training and deployment. FL3D video is 30 fps and every third frame is retained during preparation, giving an effective 10 Hz sample rate and a window spanning 3.2 seconds. The client is therefore throttled to 10 frames per second by design, so that a window covers the same real duration in deployment as in training.

## 3.7 Decision Logic

### 3.7.1 Dual-channel design

As established in Section 2.4.3, PERCLOS cannot detect a short microsleep. The system therefore evaluates two channels in parallel.

The **acute channel** raises a critical state when eye closure persists for 0.8 seconds or longer. Closure is determined from the detector's microsleep confidence, with a Schmitt trigger applied — closure is entered at a confidence of 0.50 and left at 0.35 — so that a borderline frame cannot cause the state to oscillate. Episodes shorter than 0.4 seconds are recorded as blinks rather than risks, distinguishing the two by duration rather than by a different threshold.

The **trend channel** computes PERCLOS over a sixty-second sliding window, warning at 12 per cent and raising a critical state at 15 per cent, with the score withheld until at least twenty seconds of history have accumulated. Yawn rate provides a corroborating trend signal: a yawn requires sustained mouth opening beyond 1.5 seconds with a refractory period preventing a single yawn being counted repeatedly, and three yawns within five minutes raises a warning.

### 3.7.2 Separation of the safety-critical path

The learned temporal model **never determines microsleep**. If it predicts that class, the verdict is remapped to alert, leaving the deterministic rule as the sole arbiter of the safety-critical state. The model's role is confined to distinguishing alert from yawning, where its temporal view is genuinely advantageous.

This separation is deliberate. The rule is a few lines of threshold logic that a safety engineer can read and validate by hand; the model is a set of learned weights that cannot be inspected in the same way. Placing the most consequential decision in the inspectable component is a design choice about accountability, not only about accuracy.

## 3.8 Evaluation Methodology

Evaluation operates at three levels, on the reasoning that per-class accuracy measures a model while the project's claims concern a system.

**Model level.** The detector is scored on the unmodified FL3D test frames — not on the synthetic composites — using per-class precision, recall and F1, macro-averaged F1, and the rate at which no detection is produced. Composite performance is reported separately and explicitly not treated as the headline result: a sharp face on a featureless blur makes localisation nearly trivial, and the resulting figure overstates real performance.

**System level.** A labelled clip is rendered from a held-out video into a full camera scene with a frame-level ground-truth track, and replayed through the complete client pipeline. Measured quantities are: the proportion of microsleep episodes that raise an alarm, the latency from episode onset to alarm, false alarms during alert driving, frame-level closure agreement, and the error of measured PERCLOS against the true closure fraction.

Two methodological cautions apply. Credit for detecting an episode is given only to the acute channel; counting any critical state would score an alarm still sounding from a previous episode as a detection of the next, flattering both recall and latency. And a PERCLOS-driven critical state during a labelled-alert stretch is not counted as a false alarm: on a clip that is one-quarter microsleep, PERCLOS remaining elevated through brief alert moments is correct behaviour, and conflating the two would make a functioning system appear broken.

**Performance level.** Inference latency, achievable frame rate, end-to-end alert latency, memory consumption, telemetry bandwidth and packet loss are measured on the target hardware, each against a stated requirement.

## 3.9 Hardware and Software

| Component | Specification |
|---|---|
| Development and deployment host | Apple Silicon laptop, macOS, Metal Performance Shaders acceleration |
| Camera | Standard RGB webcam, 640 × 480 capture |
| Detection framework | Ultralytics 8.4 (YOLO) |
| Deep learning framework | PyTorch 2.14 with MPS backend |
| Computer vision | OpenCV 5.0 |
| Server framework | FastAPI with Uvicorn |
| Evaluation | scikit-learn, NumPy |
| Language | Python 3.14 |

*Table 3.2: Development and deployment environment*

## 3.10 Conclusion

This chapter has set out the methodology, including two constraints discovered in the data that shaped the design — the impossibility of computing EAR from FL3D, and the absence of bounding boxes requiring composite training — and the reasoning behind the dual-channel decision logic and the separation of the safety-critical path from the learned model. Chapter IV presents the resulting system design.

---

# CHAPTER IV: SYSTEM ANALYSIS AND DESIGN

## 4.1 Introduction

This chapter presents the analysis and design of DrowSim. It establishes user, functional and quality requirements; describes the three-tier architecture and the reasoning behind the placement of the safety-critical decision; specifies each module, the telemetry data format and the decision state machine; describes the user interfaces; and reports preliminary results obtained from an initial prototype built to validate the design.

## 4.2 Requirements Analysis

### 4.2.1 User requirements

| ID | Requirement |
|---|---|
| U1 | The driver can start monitoring without configuration, calibration or account creation |
| U2 | The driver is warned audibly and visibly when drowsiness is detected, without needing to look at a screen |
| U3 | The driver's facial imagery never leaves the vehicle |
| U4 | The system continues to warn the driver when the network is unavailable |
| U5 | A fleet operator can view the state and history of connected vehicles remotely |
| U6 | The system runs on hardware the user already owns |

*Table 4.1: User requirements*

### 4.2.2 Functional requirements

| ID | Functional requirement |
|---|---|
| F1.1 | The client shall capture frames from an RGB camera at a minimum of 10 frames per second |
| F1.2 | The client shall initialise the detection pipeline and establish a session identifier at startup |
| F2.1 | The system shall locate the driver's face and classify its state as alert, microsleep or yawning in a single inference pass |
| F2.2 | The system shall reduce each frame to a seven-element numerical feature vector |
| F2.3 | The system shall report a distinct NO_FACE state when no face is detected |
| F3.1 | The client shall transmit only numerical feature vectors; no image data in any form |
| F3.2 | The client shall degrade gracefully on transmission failure, continuing detection and alerting |
| F4.1 | The system shall raise a critical state when eye closure persists for 0.8 seconds or longer |
| F4.2 | The system shall compute PERCLOS over a 60-second sliding window and raise warning and critical states at 12 and 15 per cent respectively |
| F4.3 | The system shall count a yawn only when mouth opening is sustained beyond 1.5 seconds |
| F4.4 | The learned temporal model shall not determine the microsleep state; the deterministic rule shall be its sole arbiter |
| F5.1 | The client shall raise an audible alert locally within 500 ms of detection, independent of server availability |
| F5.2 | The client shall display the current state visually, colour-coded by severity |
| F6.1 | Neither client nor server shall write frames, features, timestamps or logs to persistent storage |
| F6.2 | The dashboard shall display connected nodes, their current state, and a history of state transitions |

*Table 4.2: Functional requirements*

### 4.2.3 Quality requirements

| Attribute | Requirement |
|---|---|
| Accuracy | At least 85 per cent accuracy in classifying driver state on unseen drivers |
| Latency | Alert raised within 500 ms of the triggering condition |
| Throughput | At least 10 frames per second sustained on the target hardware |
| Resource use | Under 500 MB of memory on the client |
| Bandwidth | Approximately 1 KB per second of telemetry |
| Robustness | Graceful degradation on network loss, camera disconnection and absence of a face |
| Privacy | No image data transmitted; no persistent storage of any derived data |

*Table 4.3: Quality requirements*

## 4.3 System Architecture

### 4.3.1 Overview

The system comprises three tiers: an in-vehicle client, an inference server, and a monitoring dashboard.

!FIG:fig_4_1_architecture.png|Figure 4.1: System architecture. The safety-critical decision is taken on the client; the frame never crosses a tier boundary.|6.3

### 4.3.2 Placement of the safety-critical decision

The central architectural decision is that **the client, not the server, decides when to raise the alarm.**

A design in which the server classifies and the client merely reports would make the alarm conditional on connectivity. A vehicle in a tunnel, on a rural road, or in any area of poor coverage would lose its alarm — and these are the conditions in which long-distance night driving, and therefore fatigue risk, is concentrated. A driver monitoring system that fails in the circumstances where it is most needed does not meet its purpose.

The client therefore runs the complete rule engine locally and raises the alarm itself. The server retains a substantive role: it applies the temporal model, maintains the session record, and serves the dashboard. Both tiers execute the **identical rule module**, so the two cannot diverge in their interpretation of the same telemetry.

### 4.3.3 Client design

The client owns the camera, the detector and the safety path. Its loop captures a frame, runs detection, reduces the result to a feature vector, updates the rule engine, raises any required alarm, and appends the vector to a pending buffer for transmission. Transmission occurs at 5 Hz in aggregated batches rather than per frame, reducing overhead without delaying the local decision, which is taken at frame rate.

The frame variable is reused on each iteration and is never copied, written or encoded. This is the mechanism enforcing F6.1 on the client side.

### 4.3.4 Server design

The server exposes an inference endpoint accepting telemetry packets, a health endpoint, and an events endpoint serving the dashboard. It maintains per-session state: a rolling buffer of feature vectors, a vote buffer for prediction smoothing, and an instance of the rule engine.

Where a packet's sequence number does not follow its predecessor, the buffer is discarded and allowed to refill rather than reasoning across a gap. Session state is held in memory and expires after inactivity; nothing is written to disk.

### 4.3.5 Dashboard design

The dashboard is a single page polling the events endpoint. It displays connected vehicle nodes with their current state, server health including whether the temporal model is loaded, and a reverse-chronological log of state transitions showing the time, node, state, deciding component and reason.

The dashboard displays no imagery, because none is available to it. This is the privacy claim made visible: the operator sees that a driver is in a critical state without ever seeing the driver.

## 4.4 Module Design

| Module | Purpose | Input | Output |
|---|---|---|---|
| Video acquisition | Capture frames from the camera | Camera device | RGB frame array in volatile memory |
| YOLO detection | Locate the face and classify its state | RGB frame | Bounding box, class, confidence |
| Feature extraction | Reduce a detection to transmissible numbers | Box, class, confidence, frame dimensions | Seven-element feature vector |
| Rule engine | Determine driver state deterministically | Feature vector, timestamp | State, reasons, PERCLOS, counters |
| Alert generation | Warn the driver | State | Audible tone, visual overlay |
| Telemetry | Transmit numerical features | Feature vectors, session metadata | HTTP request; server verdict |
| Server inference | Apply the temporal model | Telemetry packet | State, confidences, deciding source |
| Dashboard service | Serve monitoring data | Session state, event history | JSON for the dashboard page |

*Table 4.4: Module specifications*

## 4.5 Data Design

The telemetry packet is the sole data crossing the network. Its structure constitutes the privacy guarantee and is therefore specified precisely.

```json
{
  "client_id":   "VEHICLE_NODE_01",
  "session_id":  "0f2c8a17b3d9",
  "seq":         147,
  "timestamp":   1523.84,
  "fps":         10.0,
  "features":    [[0.02, 0.95, 0.03, 0.78, 0.14, 0.51, 0.44]],
  "face_found":  true,
  "client_state": "CRITICAL",
  "client_reasons": ["eyes closed 0.9s"],
  "perclos":     0.1842,
  "blink_count": 3,
  "total_yawns": 1
}
```

*Figure 4.2: Telemetry packet structure*

Every field is numeric, boolean or a short string. There is no field in which image data could be carried. The packet is approximately 400 bytes and is sent at 5 Hz, giving roughly 2 KB per second.

A monotonic sequence number permits loss and reordering detection. The session identifier is regenerated for each driving session; the client identifier persists, which means the telemetry is **pseudonymous rather than anonymous** — a distinction stated here rather than glossed, since a persistent identifier makes the stream linkable to an individual even though it contains no imagery.

## 4.6 Decision Logic Design

The rule engine implements a state machine over four states.

| State | Condition | Deciding component |
|---|---|---|
| NO_FACE | No detection for longer than the grace period | Detector |
| CRITICAL | Eye closure sustained ≥ 0.8 s, **or** PERCLOS ≥ 15% | Rule (acute or trend) |
| WARNING | PERCLOS ≥ 12%, or ≥ 3 yawns in 5 minutes, or model reports yawning | Rule or model |
| NORMAL | None of the above | Rule |

*Table 4.5: Driver state definitions*

!FIG:fig_4_3_state_machine.png|Figure 4.3: Driver state machine. The acute closure rule bypasses WARNING entirely, so a microsleep raises CRITICAL directly.|6.3

Resolution order is NO_FACE, then the deterministic rule, then the model. The acute channel is evaluated last within the rule so that it always takes precedence: a microsleep in progress overrides any other assessment.

The model may raise a NORMAL verdict to WARNING but can never lower a state determined by the rule. This asymmetry is deliberate — the learned component can add caution but cannot remove it.

## 4.7 Interface Design

The **driver interface** is deliberately minimal, since a driver must not be induced to look away from the road. A preview window displays the current state as large colour-coded text — green for normal, amber for warning, red for critical — with PERCLOS, closure duration and counters on a single line beneath, and the detection box drawn on the face. Audible alerts differentiate severity by tone: 1500 Hz for 600 ms on a critical state, 900 Hz for 350 ms on a warning, allowing the two to be distinguished without looking. A refractory period prevents repeated alarms from overlapping. A headless mode omits the window entirely for unattended operation.

The **operator interface** is the dashboard described in Section 4.3.5.

!FIG:fig_4_7_client_interface.png|Figure 4.4: Driver interface in the CRITICAL state, showing the state, PERCLOS, counters and the detection box.|5.2

!FIG:fig_4_8_dashboard.png|Figure 4.5: Monitoring dashboard. Node state, server health and the event log are reconstructed from numerical telemetry alone; no imagery reaches this tier.|6.3

## 4.8 Initial Prototype and Preliminary Results

An initial prototype was constructed to validate the design before committing to it. The results below are preliminary and establish feasibility; a full evaluation belongs to the next phase of the project.

### 4.8.1 Detector performance

The detector was evaluated on **3,810 unmodified FL3D frames** from three held-out videos.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| alert | 0.9878 | 0.9901 | 0.9890 | 2,622 |
| microsleep | 0.9599 | 0.9704 | 0.9651 | 642 |
| yawning | 1.0000 | 0.9762 | 0.9880 | 546 |
| Macro average | 0.9826 | 0.9789 | **0.9807** | 3,810 |
| Weighted average | 0.9849 | 0.9848 | 0.9848 | 3,810 |

*Table 4.6: Detector performance on unmodified FL3D test frames*

!FIG:fig_4_4_confusion_matrix.png|Figure 4.6: Confusion matrix on the unmodified FL3D test frames. Confusion is concentrated between microsleep and alert.|4.2

Overall accuracy is **98.48 per cent** with a macro-averaged F1 of **0.9807**, exceeding the 85 per cent requirement. No frame failed to produce a detection. Microsleep remains the weakest class, with confusion concentrated between microsleep and alert — which is the expected direction, and the reason the deterministic rule rather than the model arbitrates that state.

Detection mAP on the synthetic composite validation set reaches approximately 0.99. **This figure is deliberately not presented as the headline result.** A sharp face composited onto a featureless blurred background makes localisation nearly trivial, and the figure overstates performance on real imagery. Table 4.6, computed on unaltered frames, is the meaningful measurement.

### 4.8.2 Model selection

| Model | Parameters | Epochs | Accuracy | Macro-F1 | Microsleep F1 | Latency |
|---|---|---|---|---|---|---|
| yolov8n **(selected)** | 3,011,433 | 21 | 0.9848 | 0.9807 | 0.9651 | 8.99 ms |
| yolov9t | 2,005,993 | 20 | 0.9816 | 0.9769 | 0.9533 | 13.61 ms |
| yolov10n | 2,708,210 | 20 | 0.9806 | 0.9755 | 0.9489 | 28.79 ms |
| yolo11n | 2,590,425 | 31 | 0.9759 | 0.9696 | 0.9373 | 8.96 ms |

*Table 4.7: YOLO generation comparison under identical training conditions*

!FIG:fig_4_5_model_comparison.png|Figure 4.7: YOLO generation comparison. The selected model leads on every quality metric at effectively the same latency.|6.4

**YOLOv8n leads on every quality metric at effectively the same latency**, and by almost three points of F1 on microsleep — the class of greatest safety significance. The ordering is the reverse of release order, and YOLO11n was trained for more epochs than the models that outperformed it, so the result is not an artefact of undertraining. YOLOv8n was therefore selected for deployment on the strength of this measurement rather than on published benchmarks.

Two qualifications apply. Each model was trained once with a single seed, so the 0.9-point spread in accuracy is suggestive rather than established; the microsleep gap is wide enough to act upon. YOLOv10n's latency reflects the behaviour of its Ultralytics implementation on Apple MPS and should not be read as evidence that its NMS-free architecture is inherently slow.

### 4.8.3 Temporal model performance

The BiLSTM-Attention model achieved **96.67 per cent accuracy** with a macro-averaged F1 of **0.9571** on held-out sequences, using **81,412 parameters** — small enough for CPU inference in under two milliseconds. Per-class F1 was 0.9762 for alert, 0.9150 for microsleep and 0.9801 for yawning.

### 4.8.4 System-level validation

Per-class accuracy measures a model, not a system. To establish whether the design responds to a driver falling asleep, a 90-second clip was rendered from a held-out video with a frame-level ground-truth label track and replayed through the complete client pipeline.

| Measure | Result |
|---|---|
| Microsleep episodes detected by the acute rule | 5 of 6 |
| Mean alarm latency from episode onset | 0.86 s |
| Maximum alarm latency | 1.00 s |
| **False alarms during alert driving** | **0** |
| Frame-level closure precision / recall / F1 | 0.973 / 0.973 / 0.973 |
| PERCLOS mean absolute error against ground truth | 0.88 percentage points |

*Table 4.8: System-level validation on a labelled clip*

!FIG:fig_4_6_perclos_trace.png|Figure 4.8: Measured PERCLOS against ground truth over the 90-second test clip. The bar above marks labelled microsleep frames.|6.3

Of the 0.86-second mean latency, 0.8 seconds is the dwell the rule deliberately requires before it may fire, leaving approximately **0.06 seconds of actual detection lag**. The deterministic channel produced **no false alarms** during any labelled-alert stretch, which is the result of greatest significance for a safety system. PERCLOS tracked the true closure fraction to within 0.88 percentage points, validating the implementation of the metric against ground truth rather than against expectation.

**One episode of six was not detected.** This is reported rather than omitted; it establishes that the current design does not achieve complete episode recall and identifies a specific target for the next development phase.

### 4.8.5 Performance

| Metric | Measurement | Requirement | Status |
|---|---|---|---|
| Mean inference latency (320 px) | 8.12 ± 0.14 ms | < 100 ms | Pass |
| Maximum inference latency | 8.60 ms | < 500 ms | Pass |
| Achievable frame rate | 123 FPS (throttled to 10 by design) | ≥ 10 FPS | Pass |
| Telemetry round-trip latency | 1.17 ms | < 500 ms | Pass |
| End-to-end alert latency | 9.29 ms | < 500 ms | Pass |
| Client memory usage | 418 MB | < 500 MB | Pass |
| Packet loss over 500 requests | 0.0% | < 5% | Pass |
| Telemetry bandwidth | ≈ 2 KB/s | ≈ 1 KB/s | **Not met** |

*Table 4.9: Performance measurements*

All requirements were met with the exception of telemetry bandwidth, which at approximately 2 KB/s is double the stated target. The excess arises from transmitting a seven-element vector per frame together with per-packet session metadata. This is recorded as a requirement not met rather than adjusted retrospectively.

Two figures merit comment. Client memory at 418 MB is within the 500 MB limit but leaves limited headroom. The frame rate figures require reading together: the hardware sustains approximately 123 FPS, but the client is deliberately throttled to 10 FPS so that a 32-frame temporal window spans the same real duration in deployment as in training.

Because the alarm is raised locally, the latency experienced by the driver is the inference figure alone. The network round-trip delays only the dashboard and the model's contextual verdict.

### 4.8.6 Offline operation

The client was run with the server unreachable. Detection, rule evaluation and alerting continued unchanged, with the interface reporting the loss of connectivity, and the system reached a critical state on the microsleep episodes without server involvement. This validates the architectural decision described in Section 4.3.2.

## 4.9 Design Constraints and Identified Risks

| Constraint or risk | Impact | Mitigation or status |
|---|---|---|
| FL3D is night-time only | Daylight and glare performance unknown | Acknowledged; daytime data required for the next phase |
| Synthetic composite backgrounds | Domain gap against real cabins | Real in-cabin footage required for validation |
| Split is video-disjoint, not person-disjoint | Accuracy may be optimistic | Stated explicitly; not resolvable from the published data |
| Critical state latches while PERCLOS remains elevated | Continuous alarm risks alarm fatigue | Escalation policy required before deployment |
| One episode in six undetected | Incomplete episode recall | Identified target for the next phase |
| Sunglasses defeat eye-state observation | No detection possible | Declared outside scope |
| No real-vehicle trial | Findings are from offline data and a laptop | On-road validation required |

*Table 4.10: Design constraints and identified risks*

## 4.10 Conclusion

This chapter has presented the requirements, architecture and detailed design of DrowSim, together with preliminary results from an initial prototype. The prototype achieved 98.48 per cent detection accuracy on unseen drivers, detected 5 of 6 microsleep episodes with no false alarms, and met every performance requirement except telemetry bandwidth. The architectural decision to place the safety-critical decision on the client was validated by demonstrating continued operation with the server unreachable.

The design constraints identified in Section 4.9 — particularly the night-only dataset, the synthetic training backgrounds and the latching critical state — define the agenda for the next phase of development.

---

# CONCLUSION

## 5.1 Summary of the Project

This project set out to address a specific and unmet need: driver drowsiness monitoring for the vehicles that current systems do not reach, delivered without the privacy cost that most proposed vision-based solutions carry.

The review in Chapter II established that vision-based detection is the appropriate approach for retrofit deployment, but that existing implementations force an unwelcome choice. Systems that process locally are tied to particular vehicles and unavailable to anyone else; the retrofittable option transmits data to an external monitoring service. No reviewed system combined retrofittability, local decision-making and a guarantee that facial imagery never leaves the vehicle.

DrowSim was designed to occupy that gap. A YOLO object detector runs on ordinary consumer hardware inside the vehicle, locating the driver's face and classifying its state in a single inference pass. Each frame is reduced to seven numbers, and those numbers are all that crosses the network. A deterministic rule engine runs on the client and raises the alarm itself; the server contributes a learned temporal verdict, the session record and a fleet dashboard, but is never on the safety-critical path.

An initial prototype was built to validate the design before committing to it. On 3,810 unmodified frames from three unseen videos it achieved **98.48 per cent accuracy** with a macro-averaged F1 of **0.9807**, failing to detect a face on no frame at all. At system level, replayed against a labelled clip, it caught **5 of 6 microsleep episodes with no false alarms**, at a mean alarm latency of 0.86 seconds, and tracked PERCLOS to within 0.88 percentage points of ground truth.

## 5.2 Achievement of Objectives

| Objective | Outcome |
|---|---|
| 1. Design and train a YOLO-based detector for the three driver states | Achieved. 98.48% accuracy, macro-F1 0.9807 on unseen videos, at 8.12 ms per frame |
| 2. Design a privacy-preserving telemetry representation | Achieved. Seven numeric fields per frame; the format admits no image data, so the property is structural rather than procedural |
| 3. Design a dual-channel decision mechanism | Achieved. Acute closure rule plus 60-second PERCLOS; validated at 0.88 percentage points of error against ground truth |
| 4. Design an edge-authoritative client–server architecture | Achieved. Verified by running the client with the server unreachable: detection and alerting continued unchanged |
| 5. Evaluate at model, system and performance levels | Achieved, with one qualification: all evaluation used offline data and a laptop. No on-road trial was conducted |

*Table 5.1: Achievement of research objectives*

## 5.3 Gaps Addressed

Chapter II identified four gaps. The project's contribution against each is as follows.

**Retrofittability with local processing.** Every reviewed system either tied local processing to a specific vehicle or transmitted data externally. DrowSim runs on hardware the user already owns, works in any vehicle with a camera position, and takes its decisions locally. This gap is addressed.

**Privacy integrated with temporal modelling.** Prior work applying LSTM and BiLSTM architectures to fatigue detection had not generally been combined with privacy-preserving design. DrowSim performs temporal classification on a seven-element numeric representation, demonstrating that the two are not in tension: the temporal model reached 96.67 per cent accuracy on a representation containing no imagery. This gap is addressed.

**PERCLOS blindness to acute events.** The established fatigue metric cannot detect a short microsleep, since three seconds of closure within a sixty-second window scores only 5 per cent. The dual-channel design detects the acute event through a separate deterministic rule while retaining PERCLOS for the trend it measures correctly. This gap is addressed, though not completely — one episode of six was missed.

**Connectivity dependence of the safety function.** Systems deciding driver state on a remote server fail exactly where fatigue risk concentrates. Placing the rule engine on the client resolves this, and the resolution was verified rather than assumed. This gap is addressed.

A fifth contribution emerged during the work and was not anticipated. The specification's comparison of YOLO generations, drawn from published benchmarks, recommends the newest model. Measuring all four under identical conditions on this task produced the reverse ordering, with YOLOv8n ahead of YOLO11n on every quality metric and by almost three points of F1 on the safety-critical class. The finding is modest in scope and rests on a single seed per model, but it illustrates a general point worth stating: benchmark rankings established on general-purpose datasets should be verified on the task at hand rather than inherited.

## 5.4 Limitations

Several limitations constrain what the present results establish.

**The dataset is night-time only.** FL3D derives entirely from night driving. Performance under daylight, glare and backlighting is untested and cannot be inferred from these results.

**Training backgrounds are synthetic.** Because FL3D supplies no full frames, the detector was trained on crops composited onto generated backgrounds. Real cabin interiors differ, and the size of the resulting domain gap is unknown.

**The split is video-disjoint, not person-disjoint.** FL3D exposes video identifiers but not driver identifiers, and NITYMED recorded several videos per driver. The possibility that a driver appears in both training and test partitions cannot be excluded, so the reported accuracy may be optimistic.

**Episode recall is incomplete.** One microsleep episode of six was not detected during system-level testing.

**The critical state latches.** Because PERCLOS is computed over a trailing window, a genuinely drowsy driver holds the system in a critical state continuously. The current implementation repeats the alarm rather than escalating it, which risks alarm fatigue and would need addressing before any real deployment.

**No real-vehicle validation was performed.** All evaluation used offline data and a laptop. Findings about accuracy and latency do not transfer automatically to a moving vehicle with vibration, changing illumination and a driver who is not a recording subject.

**Telemetry bandwidth exceeded its target**, at approximately 2 KB/s against a stated 1 KB/s.

## 5.5 Recommendations for Future Work

**Daytime and mixed-illumination data.** Training and evaluating on datasets spanning varied lighting — NTHU-DDD or YAWDD, or purpose-collected daytime footage — is the single highest-value extension, since it addresses the limitation that most restricts the current claims.

**Real in-cabin imagery.** Replacing synthetic composite backgrounds with genuine cabin footage would close the domain gap and allow the detector to be evaluated under the conditions it will actually meet.

**Alarm escalation policy.** A latched critical state should give way to a graduated response — escalating tone, varied interval, and eventually a recommendation to stop — designed against the alarm-fatigue literature rather than by intuition.

**Investigation of the missed episode.** One undetected episode in six is a specific, tractable target. Determining whether the failure lies in detection confidence, the dwell threshold or the hysteresis band would likely improve recall directly.

**On-road validation.** Deployment in a real vehicle, ideally with an independent drowsiness reference such as periodic Karolinska Sleepiness Scale self-reports, is required before any claim about real-world performance can be made.

**Repeated training runs.** Each model in the generation comparison was trained once. Multiple seeds would establish whether the observed differences are stable or within run-to-run variance.

**On-device temporal inference.** Quantising the BiLSTM would allow the contextual verdict to run on the client as well, removing the last functional dependence on the server and strengthening the privacy position further.

## 5.6 Concluding Remarks

The project demonstrates that accurate driver drowsiness detection does not require transmitting a driver's face. Reducing each frame to seven numbers before anything leaves the vehicle costs little in accuracy — 98.48 per cent on unseen drivers — and produces a system whose privacy property can be verified by inspecting a data format rather than trusted on assertion.

Equally, placing the decision that raises the alarm on the client rather than the server costs nothing and removes a failure mode that would otherwise appear precisely where the system is most needed. Both are architectural choices rather than algorithmic ones, and both were validated by measurement rather than argument.

The system is a proof of concept. It has not been driven, it has not seen daylight, and it misses roughly one microsleep episode in six. What it establishes is that the combination sought in Chapter II — retrofittable, locally authoritative, and structurally private — is achievable on hardware people already own, and that the remaining obstacles are ones of data coverage and field validation rather than of fundamental approach.

---

# REFERENCES

Acura. (2024). *2024 Acura TLX and MDX owner's manual: Driver attention monitor*. Honda Motor Co., Ltd. https://www.acura.com

Albadawi, Y., AlRedhaei, A., & Takruri, M. (2023). Real-time machine learning-based driver drowsiness detection using visual features. *Journal of Imaging, 9*(5), 91. https://doi.org/10.3390/jimaging9050091

Albadawi, Y., Takruri, M., & Awad, M. (2022). A review of recent developments in driver drowsiness detection systems. *Sensors, 22*(5), 2069. https://doi.org/10.3390/s22052069

Beugin, Y., Burke, Q., Hoak, B., Sheatsley, R., Pauley, E., Tan, G., Hussain, S. R., & McDaniel, P. (2022). *Privacy-preserving protocols for smart cameras and other IoT devices*. arXiv. https://arxiv.org/abs/2208.09776

Das, S., Pratihar, S., Pradhan, B., Jhaveri, R. H., & Benedetto, F. (2024). IoT-assisted automatic driver drowsiness detection through facial movement analysis using deep learning and a U-Net-based architecture. *Information, 15*(1), 30. https://doi.org/10.3390/info15010030

Essahraui, S., Lamaakal, I., El Hamly, I., Maleh, Y., Ouahbi, I., El Makkaoui, K., Filali Bouami, M., Pławiak, P., Alfarraj, O., & Abd El-Latif, A. A. (2025). Real-time driver drowsiness detection using facial analysis and machine learning techniques. *Sensors, 25*(3), 812. https://doi.org/10.3390/s25030812

Fonseca, T., & Ferreira, S. (2025). Drowsiness detection in drivers: A systematic review of deep learning-based models. *Applied Sciences, 15*(16), 9018. https://doi.org/10.3390/app15169018

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

Khanam, R., & Hussain, M. (2024). *YOLOv11: An overview of the key architectural enhancements*. arXiv. https://arxiv.org/abs/2410.17725

Kong, L., Xie, K., Niu, K., He, J., & Zhang, W. (2024). Remote photoplethysmography and motion tracking convolutional neural network with bidirectional long short-term memory: Non-invasive fatigue detection method based on multi-modal fusion. *Sensors, 24*(2), 455. https://doi.org/10.3390/s24020455

Mobileye. (2022). *Presenting the Mobileye driver monitoring system: Fusing road safety inside the cabin*. Mobileye. https://www.mobileye.com/blog/presenting-the-mobileye-driver-monitoring-system-fusing-road-safety-inside-the-cabin/

Muc, M. (2022). *Frame-level driver drowsiness detection (FL3D)* [Data set]. Kaggle. https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d

Owen, V., & Surantha, N. (2025). Computer vision-based drowsiness detection using handcrafted feature extraction for edge computing devices. *Applied Sciences, 15*(2), 638. https://doi.org/10.3390/app15020638

Safarov, F., Akhmedov, F., Abdusalomov, A. B., Nasimov, R., & Cho, Y. I. (2023). Real-time deep learning-based drowsiness detection: Leveraging computer-vision and eye-blink analyses for enhanced road safety. *Sensors, 23*(14), 6459. https://doi.org/10.3390/s23146459

Seeing Machines. (2024). *Guardian driver monitoring system*. Seeing Machines. https://seeingmachines.com/

Soman, S. P., Kumar, G. S., Nuthalapati, S. B., Zafar, S., & Abubeker, K. M. (2024). Internet of things assisted deep learning enabled driver drowsiness monitoring and alert system using CNN-LSTM framework. *Engineering Research Express, 6*(4). https://doi.org/10.1088/2631-8695/ad937b

Tesla. (2024). *Model 3 owner's manual: Driver monitoring system*. Tesla, Inc. https://www.tesla.com/ownersmanual/model3/en_eu/

Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., & Ding, G. (2024b). *YOLOv10: Real-time end-to-end object detection*. arXiv. https://arxiv.org/abs/2405.14458

Wang, C.-Y., Yeh, I.-H., & Liao, H.-Y. M. (2024a). *YOLOv9: Learning what you want to learn using programmable gradient information*. arXiv. https://arxiv.org/abs/2402.13616

World Health Organization. (2023). *Global status report on road safety 2023*. World Health Organization.
