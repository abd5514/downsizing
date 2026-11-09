"""Build the AP1 report from chapters.md plus generated front matter.

Numbers in Chapter IV come from the measured result files, so the report cannot
drift out of step with the system. Rerun this after any re-evaluation.
"""

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Inches, Pt

R = Path("../../runs")
det = json.loads((R / "detector_test_report.json").read_text())
sysr = json.loads((R / "system_test_report.json").read_text())
eps = sysr["episodes"]

ABSTRACT = (
    "Driver fatigue is a persistent contributor to road traffic collisions, and the microsleep "
    "— a brief involuntary lapse of consciousness — is the event most directly implicated. "
    "Vision-based monitoring can detect fatigue from facial behaviour, but existing systems are "
    "largely confined to newer premium vehicles, and many transmit raw facial video to a remote "
    "server, exposing biometric data and making a safety-critical function dependent on the network. "
    "This project develops DrowSim, a privacy-preserving client–server driver monitoring system "
    "for vehicles without factory-fitted monitoring. A YOLO object detector runs locally on consumer "
    "hardware, locating the driver's face and classifying its state as alert, microsleep or yawning "
    "in a single inference pass. Each frame is reduced to a seven-element numerical vector; no image "
    "data is transmitted or stored, a property enforced by the transmission format rather than by "
    "policy. A deterministic rule engine runs on the client and raises the alarm locally, so the "
    "driver is warned even when connectivity is lost, while the server applies a Bidirectional LSTM "
    "with attention and serves a monitoring dashboard. Detection combines two parallel channels: an "
    "acute rule triggering on sustained eye closure, and PERCLOS over a sixty-second window, since a "
    "short microsleep is diluted below any workable PERCLOS threshold. Four YOLO generations were "
    "compared under identical conditions; YOLOv8n was selected on measurement. On "
    f"{det['total_frames']:,} unmodified frames from unseen videos of the FL3D dataset the detector "
    f"achieved {det['accuracy'] * 100:.2f} per cent accuracy and a macro-averaged F1 of "
    f"{det['macro avg']['f1-score']:.4f}. System-level testing on a labelled clip detected "
    f"{eps['detected_by_acute_rule']} of {eps['total']} microsleep episodes with no false alarms, at a "
    f"mean alarm latency of {eps['mean_alarm_latency_s']:.2f} seconds, and tracked PERCLOS to within "
    f"{sysr['perclos']['mean_abs_error'] * 100:.2f} percentage points of ground truth. The results "
    "establish the feasibility of accurate, low-cost driver monitoring that never transmits an image."
)

doc = Document()
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Inches(1.0)
    sec.top_margin = sec.bottom_margin = Inches(1.0)
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing = 1.5


def centred(text, size=12, bold=False, after=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    r.font.name = "Times New Roman"


def gap(n=1):
    for _ in range(n):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0


# ---------------------------------------------------------------- title page
# Spacing is tuned so the page holds together as one; a wrapped title pushed
# the university block onto a second page in an earlier revision.
gap(3)
centred("DROWSIM: A PRIVACY-PRESERVING CLIENT–SERVER SYSTEM", 15, True, 2)
centred("FOR REAL-TIME DRIVER DROWSINESS DETECTION USING YOLO", 15, True, 0)
gap(4)
centred("ESRA ESAM HALBOUP", 13, True, 2)
centred("A199870", 12, False, 0)
gap(6)
centred("PROJECT SUBMITTED IN PARTIAL FULFILMENT OF THE REQUIREMENTS", 11, False, 2)
centred("FOR THE DEGREE OF BACHELOR OF COMPUTER SCIENCE", 11, False, 0)
gap(4)
centred("FACULTY OF INFORMATION SCIENCE AND TECHNOLOGY", 11, False, 2)
centred("UNIVERSITI KEBANGSAAN MALAYSIA", 11, False, 2)
centred("BANGI", 11, False, 0)
gap(2)
centred("2026", 11, False, 0)
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ------------------------------------------------------------------ abstract
h = doc.add_heading("ABSTRACT", 1)
h.alignment = WD_ALIGN_PARAGRAPH.CENTER
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r = p.add_run(ABSTRACT)
r.font.size = Pt(12)
r.font.name = "Times New Roman"
assert len(ABSTRACT.split()) <= 300, f"abstract is {len(ABSTRACT.split())} words (limit 300)"

gap(2)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("[ Table of Contents — in Word: References ▸ Table of Contents "
              "▸ Automatic Table 1, then right-click ▸ Update Field ]")
r.italic = True
r.font.size = Pt(10)
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ------------------------------------------------------------------ chapters
def rich(p, text):
    for part in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*|`.+?`)", text):
        if not part:
            continue
        if part.startswith("**"):
            p.add_run(part.strip("*")).bold = True
        elif part.startswith("*"):
            p.add_run(part.strip("*")).italic = True
        elif part.startswith("`"):
            r = p.add_run(part.strip("`"))
            r.font.name = "Consolas"
            r.font.size = Pt(10)
        else:
            p.add_run(part)


src = Path("chapters.md").read_text().splitlines()
i = 0
first = True
while i < len(src):
    line = src[i]

    if line.startswith("!FIG:"):
        spec = line[5:].split("|")
        name, caption = spec[0], spec[1] if len(spec) > 1 else ""
        width = float(spec[2]) if len(spec) > 2 else 5.5
        path = Path("figures") / name
        if path.exists():
            fp = doc.add_paragraph()
            fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fp.paragraph_format.space_before = Pt(10)
            fp.paragraph_format.space_after = Pt(4)
            fp.add_run().add_picture(str(path), width=Inches(width))
            cp = doc.add_paragraph()
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.paragraph_format.space_after = Pt(12)
            cp.paragraph_format.line_spacing = 1.0
            cr = cp.add_run(caption)
            cr.italic = True
            cr.font.size = Pt(9.5)
        else:
            print(f"  ! missing figure: {path}")
        i += 1
        continue

    if line.startswith("```"):
        i += 1
        buf = []
        while i < len(src) and not src[i].startswith("```"):
            buf.append(src[i])
            i += 1
        i += 1
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run("\n".join(buf))
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
        continue

    if line.startswith("|") and i + 1 < len(src) and set(src[i + 1].replace("|", "").strip()) <= set("-: "):
        rows = []
        while i < len(src) and src[i].startswith("|"):
            cells = [c.strip() for c in src[i].strip().strip("|").split("|")]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
            i += 1
        ncol = max(len(r) for r in rows)
        t = doc.add_table(rows=len(rows), cols=ncol)
        t.style = "Table Grid"
        for ri, row in enumerate(rows):
            for ci in range(ncol):
                cell = t.cell(ri, ci)
                cell.text = ""
                cp = cell.paragraphs[0]
                cp.paragraph_format.line_spacing = 1.0
                cp.paragraph_format.space_after = Pt(2)
                raw = row[ci] if ci < len(row) else ""
                run = cp.add_run(re.sub(r"\*\*(.+?)\*\*", r"\1", raw))
                run.font.size = Pt(9.5)
                run.font.name = "Times New Roman"
                if ri == 0 or "**" in raw:
                    run.bold = True
        doc.add_paragraph()
        continue

    if line.startswith("# "):
        if not first:
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        first = False
        hh = doc.add_heading(line[2:], 1)
        hh.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif line.startswith("## "):
        doc.add_heading(line[3:], 2)
    elif line.startswith("### "):
        doc.add_heading(line[4:], 3)
    elif line.startswith("---"):
        pass
    elif line.startswith("*") and line.rstrip().endswith("*") and not line.startswith("**") and len(line) > 2:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line.strip().strip("*"))
        r.italic = True
        r.font.size = Pt(10)
    elif re.match(r"^\d+\. ", line):
        doc.add_paragraph(re.sub(r"^\d+\. ", "", line), style="List Number")
    elif line.strip():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        rich(p, line)
    i += 1

doc.save("DrowSim_Report_Ch1-4.docx")
print(f"wrote DrowSim_Report_Ch1-4.docx  (abstract {len(ABSTRACT.split())} words)")
