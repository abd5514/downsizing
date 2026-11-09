const pptxgen = require("pptxgenjs");
const fs = require("fs");

// Palette taken from the system's own state colours - the green/amber/red the
// client paints on screen - so the deck and the demo read as one thing.
const BG    = "10141C";
const PANEL = "1A2030";
const FG    = "F2F4F8";
const MUTED = "9AA3B2";
const OK    = "3ECF6D";
const WARN  = "F5B544";
const CRIT  = "EF4444";
const LINE  = "2A3244";

const det = JSON.parse(fs.readFileSync("runs/detector_test_report.json"));
const sys = JSON.parse(fs.readFileSync("runs/system_test_report.json"));
const bil = JSON.parse(fs.readFileSync("runs/bilstm_test_report.json"));
const cmp = JSON.parse(fs.readFileSync("runs/model_comparison.json"));
const bm  = JSON.parse(fs.readFileSync("runs/benchmark.json"));

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
pres.author = "Esra Esam Halboup";
pres.title  = "DrowSim";

const W = 13.3, H = 7.5, M = 0.7;

function slide(dark = true) {
  const s = pres.addSlide();
  s.background = { color: dark ? BG : "FFFFFF" };
  return s;
}

function title(s, text, sub) {
  s.addText(text, { x: M, y: 0.5, w: W - 2 * M, h: 0.75, fontSize: 38, bold: true,
                    color: FG, fontFace: "Calibri", isTextBox: true, margin: 0 });
  if (sub) {
    s.addText(sub, { x: M, y: 1.28, w: W - 2 * M, h: 0.4, fontSize: 15, color: MUTED,
                     fontFace: "Calibri", isTextBox: true, margin: 0 });
  }
}

// The recurring motif: a state pill, exactly as the client renders it.
function pill(s, x, y, text, colour) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w: 1.5, h: 0.38, rectRadius: 0.19,
    fill: { color: colour, transparency: 82 }, line: { color: colour, width: 1 } });
  s.addText(text, { x, y, w: 1.5, h: 0.38, fontSize: 11, bold: true, color: colour,
    align: "center", valign: "middle", fontFace: "Calibri", isTextBox: true, margin: 0 });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || PANEL }, line: { color: LINE, width: 1 } });
}

function stat(s, x, y, w, value, label, colour) {
  s.addText(value, { x, y, w, h: 0.85, fontSize: 44, bold: true, color: colour || OK,
    align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
  s.addText(label, { x, y: y + 0.85, w, h: 0.6, fontSize: 12, color: MUTED,
    align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
}

function bullets(s, x, y, w, items, size) {
  s.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i !== items.length - 1 } })),
    { x, y, w, h: 0.4 * items.length + 0.6, fontSize: size || 15, color: FG,
      fontFace: "Calibri", isTextBox: true, margin: 0, paraSpaceAfter: 10, lineSpacing: 22 });
}

/* ---------------------------------------------------------------- 1 title */
let s = slide();
s.addText("DrowSim", { x: M, y: 2.2, w: W - 2 * M, h: 1.2, fontSize: 66, bold: true,
  color: FG, fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("A Privacy-Focused Client–Server Framework for\nReal-Time Driver Fatigue Monitoring",
  { x: M, y: 3.45, w: 8.6, h: 1.0, fontSize: 19, color: OK, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 27 });
s.addText("Esra Esam Halboup   ·   A199870\nFaculty of Information Science and Technology, UKM\nSupervisor: Prof Azana",
  { x: M, y: 5.1, w: 7, h: 1.1, fontSize: 13, color: MUTED, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 20 });
pill(s, 10.0, 2.35, "NORMAL", OK);
pill(s, 10.0, 2.85, "WARNING", WARN);
pill(s, 10.0, 3.35, "CRITICAL", CRIT);
s.addText("the three states the system reports", { x: 9.4, y: 3.95, w: 3.2, h: 0.5,
  fontSize: 10, color: MUTED, align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addNotes("DrowSim detects driver drowsiness from a camera, but sends only numbers over the network - never video.");

/* --------------------------------------------------------------- 2 problem */
s = slide();
title(s, "The problem", "Fatigue builds slowly, and the driver is the last to notice");
stat(s, M, 2.1, 3.6, "1.19M", "road deaths worldwide each year (WHO, 2023)", CRIT);
stat(s, 4.6, 2.1, 3.6, "~900", "US deaths a year attributed to drowsy driving", WARN);
stat(s, 8.5, 2.1, 3.6, "3–4s", "a microsleep at 90 km/h covers ~100 m blind", CRIT);
card(s, M, 4.3, W - 2 * M, 2.2);
s.addText("Why existing solutions fall short", { x: M + 0.35, y: 4.5, w: 11, h: 0.4,
  fontSize: 17, bold: true, color: FG, fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(s, M + 0.35, 4.95, 11.2, [
  "EEG and ECG sensing is accurate but intrusive, costly and unsuited to ordinary vehicles",
  "Vision systems that stream raw facial video to the cloud expose biometric data and need constant connectivity",
  "Factory driver-monitoring systems exist only in newer premium cars — the vehicles most at risk have none",
], 14);
s.addNotes("The gap is not detection accuracy. It is doing it privately, cheaply, and in cars that have nothing.");

/* ---------------------------------------------------------- 3 what we built */
s = slide();
title(s, "What DrowSim does", "Everything visual stays inside the vehicle");
card(s, M, 2.0, 3.85, 2.5);
s.addText("Sees", { x: M + 0.3, y: 2.2, w: 3.2, h: 0.4, fontSize: 18, bold: true, color: OK,
  fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("A YOLOv8n detector locates the driver's face and classifies its state on the laptop itself. The frame is overwritten by the next one.",
  { x: M + 0.3, y: 2.65, w: 3.25, h: 1.7, fontSize: 13, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 19 });
card(s, 4.75, 2.0, 3.85, 2.5);
s.addText("Decides", { x: 5.05, y: 2.2, w: 3.2, h: 0.4, fontSize: 18, bold: true, color: WARN,
  fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("A deterministic rule engine watches eye closure and PERCLOS. It runs locally, so the alarm does not depend on the network.",
  { x: 5.05, y: 2.65, w: 3.25, h: 1.7, fontSize: 13, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 19 });
card(s, 8.8, 2.0, 3.8, 2.5);
s.addText("Reports", { x: 9.1, y: 2.2, w: 3.2, h: 0.4, fontSize: 18, bold: true, color: CRIT,
  fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("Seven numbers per frame go to the server, which adds a learned temporal verdict and drives a fleet dashboard.",
  { x: 9.1, y: 2.65, w: 3.2, h: 1.7, fontSize: 13, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 19 });
card(s, M, 4.85, W - 2 * M, 1.6, "16241B");
s.addText("What actually crosses the network", { x: M + 0.35, y: 5.0, w: 8, h: 0.35,
  fontSize: 14, bold: true, color: OK, fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("p_alert   p_microsleep   p_yawning   bbox_aspect   bbox_area   center_x   center_y",
  { x: M + 0.35, y: 5.42, w: 11.2, h: 0.4, fontSize: 15, color: FG, fontFace: "Courier New",
    isTextBox: true, margin: 0 });
s.addText("≈ 400 bytes per packet, about 2 KB/s. No image data, in any form.",
  { x: M + 0.35, y: 5.88, w: 11, h: 0.35, fontSize: 12, color: MUTED, fontFace: "Calibri",
    isTextBox: true, margin: 0 });
s.addNotes("The privacy claim is not a policy promise - it is a property of the payload format.");

/* -------------------------------------------------------- 4 detection logic */
s = slide();
title(s, "Two channels, because they catch different things", "PERCLOS alone would miss the most dangerous event");
card(s, M, 2.05, 5.85, 2.6, "2A1618");
s.addText("Acute", { x: M + 0.3, y: 2.25, w: 3, h: 0.4, fontSize: 20, bold: true, color: CRIT,
  fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("Eyes closed ≥ 0.8 s → CRITICAL, immediately.\n\nA blink is 0.1–0.4 s, so duration — not a different threshold — separates the two.",
  { x: M + 0.3, y: 2.72, w: 5.2, h: 1.7, fontSize: 14, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 20 });
card(s, 6.85, 2.05, 5.75, 2.6, "2A2416");
s.addText("Trend", { x: 7.15, y: 2.25, w: 3, h: 0.4, fontSize: 20, bold: true, color: WARN,
  fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("PERCLOS over a 60 s window — 12% warns, 15% is critical.\n\nCatches the gradual slide that no single frame reveals.",
  { x: 7.15, y: 2.72, w: 5.15, h: 1.7, fontSize: 14, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 20 });
card(s, M, 4.95, W - 2 * M, 1.75, "1A2030");
s.addText("Why both are needed", { x: M + 0.35, y: 5.12, w: 6, h: 0.35, fontSize: 15,
  bold: true, color: FG, fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("A 3-second microsleep inside a 60-second window scores just 5% PERCLOS — far below any sensible threshold. The single most dangerous event in the problem domain is invisible to the industry-standard metric. The acute channel exists precisely to catch it.",
  { x: M + 0.35, y: 5.5, w: 11.3, h: 1.0, fontSize: 13.5, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 20 });
s.addNotes("This is the core design argument. PERCLOS is necessary but not sufficient.");

/* ------------------------------------------------------- 5 safety topology */
s = slide();
title(s, "The alarm belongs on the edge", "A design change from the original architecture");
card(s, M, 2.05, 5.85, 3.4, "2A1618");
s.addText("Before", { x: M + 0.3, y: 2.25, w: 3, h: 0.4, fontSize: 18, bold: true, color: CRIT,
  fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(s, M + 0.3, 2.72, 5.2, [
  "Server classified and decided",
  "Link drops → alerts suppressed",
  "No internet meant no alarm — exactly when nobody is watching",
], 13.5);
card(s, 6.85, 2.05, 5.75, 3.4, "16241B");
s.addText("Now", { x: 7.15, y: 2.25, w: 3, h: 0.4, fontSize: 18, bold: true, color: OK,
  fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(s, 7.15, 2.72, 5.15, [
  "Client decides and sounds the alarm itself",
  "Server adds the learned verdict, the record and the dashboard",
  "Both tiers import the same rules module, so they cannot drift apart",
], 13.5);
s.addText("The BiLSTM never decides microsleep. If it predicts that class the verdict is remapped to alert, leaving one deterministic rule as the sole arbiter of the safety-critical state.",
  { x: M, y: 5.75, w: W - 2 * M, h: 0.9, fontSize: 14, color: WARN, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 21, italic: true });
s.addNotes("Demonstrated in the video: with the server unreachable the client still reaches CRITICAL.");

/* -------------------------------------------------------------- 6 dataset */
s = slide();
title(s, "Data: FL3D, and what it really contains", "53,331 labelled frames from night-time driving (NITYMED)");
card(s, M, 2.05, 6.0, 2.35);
s.addText("What the dataset gives", { x: M + 0.3, y: 2.22, w: 5.4, h: 0.35, fontSize: 15,
  bold: true, color: OK, fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(s, M + 0.3, 2.62, 5.4, [
  "Pre-cropped faces, roughly 210 × 290 px",
  "Five RetinaFace landmarks — no eyelid contours",
  "No bounding boxes, no full frames",
], 13);
card(s, 7.0, 2.05, 5.6, 2.35);
s.addText("What that forced", { x: 7.3, y: 2.22, w: 5.0, h: 0.35, fontSize: 15,
  bold: true, color: WARN, fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(s, 7.3, 2.62, 5.0, [
  "EAR is not computable from this data at all",
  "Detector trained on composites: crops placed on synthesised cabin backgrounds",
  "Closure read from the detector instead",
], 13);
card(s, M, 4.65, W - 2 * M, 1.95, "1A2030");
s.addText("Splitting honestly", { x: M + 0.35, y: 4.82, w: 6, h: 0.35, fontSize: 15,
  bold: true, color: FG, fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("FL3D's own validation and test sets come from the same five videos, so tuning on one leaks into the other. Those five are re-split by video instead — 2 for validation, 3 for test — choosing the division that maximises worst-case class coverage.\n\nFL3D exposes video ids, not driver ids, and NITYMED recorded several videos per driver. The split is video-disjoint; it cannot be claimed as person-disjoint.",
  { x: M + 0.35, y: 5.2, w: 11.3, h: 1.3, fontSize: 12.5, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 18 });
s.addNotes("Being explicit about the video-vs-person distinction is a strength, not a weakness.");

/* -------------------------------------------------------------- 7 results */
s = slide();
title(s, "Detector results", "3,810 raw, unmodified FL3D frames from three unseen videos");
const rows = [[
  { text: "Class", options: { bold: true, color: FG, fill: { color: PANEL } } },
  { text: "Precision", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
  { text: "Recall", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
  { text: "F1", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
  { text: "Support", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
]];
[["alert", OK], ["microsleep", CRIT], ["yawning", WARN]].forEach(([c, col]) => {
  rows.push([
    { text: c, options: { color: col, bold: true } },
    { text: det[c].precision.toFixed(4), options: { color: FG, align: "center" } },
    { text: det[c].recall.toFixed(4), options: { color: FG, align: "center" } },
    { text: det[c]["f1-score"].toFixed(4), options: { color: FG, align: "center" } },
    { text: String(det[c].support), options: { color: MUTED, align: "center" } },
  ]);
});
rows.push([
  { text: "macro average", options: { color: FG, bold: true } },
  { text: det["macro avg"].precision.toFixed(4), options: { color: FG, align: "center", bold: true } },
  { text: det["macro avg"].recall.toFixed(4), options: { color: FG, align: "center", bold: true } },
  { text: det["macro avg"]["f1-score"].toFixed(4), options: { color: OK, align: "center", bold: true } },
  { text: String(det.total_frames), options: { color: MUTED, align: "center" } },
]);
s.addTable(rows, { x: M, y: 2.05, w: 7.5, colW: [2.1, 1.4, 1.35, 1.35, 1.3], rowH: 0.42,
  fontSize: 13, fontFace: "Calibri", border: { pt: 1, color: LINE }, fill: { color: "141926" },
  valign: "middle" });
stat(s, 8.6, 2.15, 2.0, (det.accuracy * 100).toFixed(2) + "%", "overall accuracy", OK);
stat(s, 10.7, 2.15, 1.9, "0", "frames with no detection", OK);
card(s, 8.6, 4.0, 4.0, 1.35);
s.addText("BiLSTM-Attention", { x: 8.85, y: 4.15, w: 3.5, h: 0.3, fontSize: 13, bold: true,
  color: FG, fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText((bil.accuracy * 100).toFixed(2) + "% accuracy · macro-F1 " + bil["macro avg"]["f1-score"].toFixed(3) + "\n81,412 parameters · 1.8 ms on CPU",
  { x: 8.85, y: 4.5, w: 3.5, h: 0.7, fontSize: 12, color: MUTED, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 18 });
s.addText("Detection mAP on the synthetic composites reaches 0.994, but that number is not quoted as a headline: a sharp face on a featureless blur makes localisation close to trivial. The table above, on unmodified frames, is the meaningful result.",
  { x: M, y: 5.6, w: 7.5, h: 1.1, fontSize: 12, color: MUTED, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 18, italic: true });
s.addNotes("YOLOv8n, selected by measurement rather than by release date - see the comparison slide.");

/* ------------------------------------------------------- 8 system results */
s = slide();
title(s, "Does it actually wake the driver?", "The question a per-class table cannot answer — scored on a 90 s labelled clip");
stat(s, M, 2.1, 2.9, sys.episodes.detected_by_acute_rule + " / " + sys.episodes.total,
  "microsleep episodes caught by the rule", OK);
stat(s, 3.75, 2.1, 2.9, String(sys.false_alarms.acute_false_positives),
  "false alarms during alert driving", OK);
stat(s, 6.8, 2.1, 2.9, sys.episodes.mean_alarm_latency_s.toFixed(2) + " s",
  "mean latency from onset to alarm", WARN);
stat(s, 9.7, 2.1, 2.9, (sys.perclos.mean_abs_error * 100).toFixed(1) + "%",
  "PERCLOS mean error vs ground truth", OK);
card(s, M, 4.35, 5.85, 2.2);
s.addText("Reading the latency", { x: M + 0.3, y: 4.52, w: 5.2, h: 0.35, fontSize: 15,
  bold: true, color: FG, fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText(`The rule deliberately requires 0.8 s of sustained closure before it may fire, so of the ${sys.episodes.mean_alarm_latency_s.toFixed(2)} s mean only about ${(sys.episodes.mean_alarm_latency_s - 0.8).toFixed(2)} s is detection lag. The rest is the blink-rejection guarantee.`,
  { x: M + 0.3, y: 4.92, w: 5.2, h: 1.4, fontSize: 13, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 19 });
card(s, 6.85, 4.35, 5.75, 2.2);
s.addText("Performance on a laptop", { x: 7.15, y: 4.52, w: 5.1, h: 0.35, fontSize: 15,
  bold: true, color: FG, fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(s, 7.15, 4.92, 5.1, [
  `${bm.yolo_mean_ms.toFixed(2)} ± ${bm.yolo_std_ms.toFixed(2)} ms YOLO inference at ${bm.imgsz} px`,
  `${bm.end_to_end_ms.toFixed(2)} ms end to end · ${(bm.packet_loss * 100).toFixed(0)}% loss over 500 requests`,
  `${bm.peak_rss_mb} MB RSS · runs at 10 FPS by design`,
], 12.5);
s.addNotes("Zero false alarms from the deterministic channel is the number that matters most for a safety system.");


/* ------------------------------------------------------- 8b comparison */
s = slide();
title(s, "Which YOLO generation?", "Identical data, hyperparameters and seed \u2014 scored on the same 3,810 unseen frames");
const crows = [[
  { text: "Model", options: { bold: true, color: FG, fill: { color: PANEL } } },
  { text: "Params", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
  { text: "Epochs", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
  { text: "Accuracy", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
  { text: "Macro-F1", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
  { text: "Microsleep F1", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
  { text: "Latency", options: { bold: true, color: FG, fill: { color: PANEL }, align: "center" } },
]];
cmp.forEach(r => {
  const win = r.model.indexOf("v8n") >= 0;
  const col = win ? OK : FG;
  crows.push([
    { text: r.model.replace(".pt", "") + (win ? "  (selected)" : ""), options: { color: col, bold: win } },
    { text: r.parameters.toLocaleString(), options: { color: MUTED, align: "center" } },
    { text: String(r.epochs_trained), options: { color: MUTED, align: "center" } },
    { text: r.accuracy.toFixed(4), options: { color: col, align: "center", bold: win } },
    { text: r.macro_f1.toFixed(4), options: { color: col, align: "center", bold: win } },
    { text: r.microsleep_f1.toFixed(4), options: { color: col, align: "center", bold: win } },
    { text: r.latency_ms_mean.toFixed(2) + " ms", options: { color: col, align: "center" } },
  ]);
});
s.addTable(crows, { x: M, y: 2.0, w: W - 2 * M, colW: [2.5, 1.5, 1.1, 1.4, 1.4, 1.7, 1.3],
  rowH: 0.44, fontSize: 12.5, fontFace: "Calibri", border: { pt: 1, color: LINE },
  fill: { color: "141926" }, valign: "middle" });
card(s, M, 4.75, W - 2 * M, 1.9);
s.addText("The newest model is not the best one here", { x: M + 0.35, y: 4.92, w: 8, h: 0.35,
  fontSize: 15, bold: true, color: WARN, fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("YOLOv8n leads on every quality metric at the same latency, and by almost three points of F1 on microsleep \u2014 the class that matters most. YOLO11n was trained for more epochs than the models that beat it, so it is not undertrained. Each model was run once with a single seed, so the 0.9-point accuracy spread is suggestive rather than settled; the microsleep gap is wide enough to act on. YOLOv10n\u2019s 28.79 ms reflects its Ultralytics implementation on Apple MPS, not the NMS-free design it advertises.",
  { x: M + 0.35, y: 5.3, w: 11.3, h: 1.2, fontSize: 12.5, color: FG, fontFace: "Calibri",
    isTextBox: true, margin: 0, lineSpacing: 18 });
s.addNotes("The specification recommends YOLO11 on published benchmarks. Measuring it on this task gave the opposite answer, and the deployed model follows the measurement.");

/* ---------------------------------------------------------- 9 limitations */
s = slide();
title(s, "What this does not yet do", "Stated plainly, because a viva will find them anyway");
const lims = [
  ["Night-time data only", "FL3D comes entirely from night driving. Daylight, glare and backlighting are untested."],
  ["Synthetic backgrounds", "The detector learned on composited cabins, not real ones. A domain gap remains."],
  ["One episode missed", "Five of six microsleep episodes were caught on the test clip; the sixth was not."],
  ["CRITICAL latches", "With a 60 s window and a genuinely drowsy driver, PERCLOS holds the alarm on. Escalation logic is needed."],
  ["No real-vehicle trial", "Everything here is offline data and a laptop. No live driving validation."],
  ["Video-disjoint, not person-disjoint", "FL3D exposes no driver ids, so a driver may appear in both train and test."],
];
lims.forEach((l, i) => {
  const x = M + (i % 2) * 6.15, y = 2.05 + Math.floor(i / 2) * 1.55;
  card(s, x, y, 5.85, 1.35);
  s.addText(l[0], { x: x + 0.28, y: y + 0.14, w: 5.3, h: 0.32, fontSize: 14, bold: true,
    color: WARN, fontFace: "Calibri", isTextBox: true, margin: 0 });
  s.addText(l[1], { x: x + 0.28, y: y + 0.5, w: 5.3, h: 0.75, fontSize: 12, color: MUTED,
    fontFace: "Calibri", isTextBox: true, margin: 0, lineSpacing: 17 });
});
s.addNotes("Naming these first is stronger than being asked about them.");

/* ------------------------------------------------------------ 10 close */
s = slide();
title(s, "Where it stands", null);
card(s, M, 1.9, 5.85, 2.5, "16241B");
s.addText("Delivered", { x: M + 0.3, y: 2.08, w: 5.2, h: 0.35, fontSize: 17, bold: true,
  color: OK, fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(s, M + 0.3, 2.5, 5.2, [
  "Trained YOLOv8n detector and BiLSTM temporal model",
  "Edge-authoritative client, server, dashboard",
  "18 automated tests · all results reproducible",
], 13);
card(s, 6.85, 1.9, 5.75, 2.5);
s.addText("Next", { x: 7.15, y: 2.08, w: 5.1, h: 0.35, fontSize: 17, bold: true,
  color: WARN, fontFace: "Calibri", isTextBox: true, margin: 0 });
bullets(s, 7.15, 2.5, 5.1, [
  "Daytime data — NTHU-DDD or YAWDD",
  "Alarm escalation instead of a latched CRITICAL",
  "On-road trial with live drivers",
], 13);
s.addText("97.59%", { x: M, y: 4.8, w: 3.9, h: 0.9, fontSize: 46, bold: true, color: OK,
  align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("accuracy on unseen drivers", { x: M, y: 5.7, w: 3.9, h: 0.4, fontSize: 12,
  color: MUTED, align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("0", { x: 4.8, y: 4.8, w: 3.7, h: 0.9, fontSize: 46, bold: true, color: OK,
  align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("false alarms from the safety rule", { x: 4.8, y: 5.7, w: 3.7, h: 0.4, fontSize: 12,
  color: MUTED, align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("0 bytes", { x: 8.7, y: 4.8, w: 3.9, h: 0.9, fontSize: 46, bold: true, color: OK,
  align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addText("of video ever transmitted", { x: 8.7, y: 5.7, w: 3.9, h: 0.4, fontSize: 12,
  color: MUTED, align: "center", fontFace: "Calibri", isTextBox: true, margin: 0 });
s.addNotes("Thank you - happy to take questions.");

pres.writeFile({ fileName: "deliverables/DrowSim_AP1_slides.pptx" })
  .then(f => console.log("wrote " + f));
