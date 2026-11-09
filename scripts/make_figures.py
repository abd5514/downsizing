"""Generate the report figures.

Charts follow one rule throughout: colour carries meaning, never decoration.
The model comparison highlights the selected model against neutral grey rather
than giving each model its own hue - a four-hue categorical palette fails the
normal-vision separation floor (yellow against orange, dE 13.7), and "which one
did we pick" is the question the figure exists to answer anyway.

Output is sized and weighted for print: white surface, no dark mode, and every
value that matters carries a direct label so the figures survive being
photocopied in greyscale.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = Path("runs")
OUT = Path("deliverables/report/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Validated slots from the reference palette (see scripts/validate_palette.js).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
NEUTRAL = "#b9b9b4"
CRIT, WARN, OK = "#c0392b", "#d98c00", "#1baf7a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- Figure 2.2
def perclos_dilution():
    """Why PERCLOS alone cannot see a microsleep."""
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    t = np.linspace(0, 60, 600)
    closed = ((t >= 30) & (t < 33)).astype(float)
    ax.fill_between(t, 0, closed, color=CRIT, alpha=0.85, linewidth=0)
    ax.axhline(0.05, color=BLUE, lw=2, ls="-")
    ax.axhline(0.15, color=MUTED, lw=1.4, ls="--")
    # Labels sit inside the axes; placing them beyond the right edge collided
    # with each other and clipped at the figure boundary.
    lbl = dict(fontsize=8, va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.text(1.0, 0.068, "PERCLOS over this window = 5%", color=BLUE, **lbl)
    ax.text(1.0, 0.172, "alarm threshold 15%", color=MUTED, **lbl)
    ax.annotate("3-second microsleep", xy=(31.5, 0.72), xytext=(14, 0.9),
                fontsize=8.5, color=CRIT,
                arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.2))
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.15)
    ax.set_xlabel("time within the 60-second PERCLOS window (s)")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["eyes open", "eyes closed"], fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig_2_2_perclos_dilution.png")


# ---------------------------------------------------------------- Figure 3.3
def class_distribution():
    counts = {"alert": 38954, "microsleep": 8880, "yawning": 5497}
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    names = list(counts)
    vals = [counts[k] for k in names]
    bars = ax.bar(names, vals, color=BLUE, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 700, f"{v:,}\n({v/sum(vals):.1%})",
                ha="center", fontsize=8.5, color=INK, linespacing=1.4)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_ylabel("frames")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig_3_3_class_distribution.png")


# ---------------------------------------------------------------- Figure 4.4
def confusion():
    det = json.loads((R / "detector_test_report.json").read_text())
    classes = det["confusion_labels"]
    cm = np.array(det["confusion_matrix"], dtype=float)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j]):,}\n{norm[i, j]:.1%}", ha="center",
                    va="center", fontsize=8.5, linespacing=1.4,
                    color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    save(fig, "fig_4_4_confusion_matrix.png")


# ---------------------------------------------------------------- Figure 4.5
def model_comparison():
    rows = json.loads((R / "model_comparison.json").read_text())
    rows = sorted(rows, key=lambda r: -r["accuracy"])
    names = [r["model"].replace(".pt", "") for r in rows]
    sel = ["v8n" in n for n in names]
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
               ("microsleep_f1", "Microsleep F1"), ("latency_ms_mean", "Latency (ms)")]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
    for ax, (key, label) in zip(axes, metrics):
        vals = [r[key] for r in rows]
        colours = [BLUE if s else NEUTRAL for s in sel]
        bars = ax.barh(range(len(names))[::-1], vals, color=colours, height=0.6)
        lo = min(vals) * 0.985 if key != "latency_ms_mean" else 0
        ax.set_xlim(lo, max(vals) * 1.38)
        for b, v in zip(bars, vals):
            txt = f"{v:.4f}" if key != "latency_ms_mean" else f"{v:.1f}"
            ax.text(v + (max(vals) - lo) * 0.03, b.get_y() + b.get_height() / 2,
                    txt, va="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names if ax is axes[0] else [], fontsize=8)
        ax.set_title(label, fontsize=9, color=INK, pad=6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(length=0)
    fig.legend(handles=[mpatches.Patch(color=BLUE, label="selected (YOLOv8n)"),
                        mpatches.Patch(color=NEUTRAL, label="other candidates")],
               loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig_4_5_model_comparison.png")


# ---------------------------------------------------------------- Figure 4.6
def perclos_trace():
    tr = json.loads((R / "perclos_trace.json").read_text())
    fps = tr["fps"]
    t = np.arange(len(tr["measured_perclos"])) / fps
    meas = np.array(tr["measured_perclos"]) * 100
    true = np.array(tr["true_perclos"]) * 100
    labels = tr["labels"]

    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(6.6, 3.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 5], "hspace": 0.12})

    gt = np.array([l == "microsleep" for l in labels], dtype=float)
    ax0.fill_between(t, 0, gt[:len(t)], color=CRIT, alpha=0.8, linewidth=0, step="mid")
    ax0.set_ylim(0, 1); ax0.set_yticks([])
    ax0.set_ylabel("truth", fontsize=7.5, color=MUTED, rotation=0,
                   ha="right", va="center", labelpad=14)
    ax0.spines[:].set_visible(False)
    ax0.tick_params(length=0)

    ax.plot(t, true, color=ORANGE, lw=2, label="ground truth")
    ax.plot(t, meas, color=BLUE, lw=2, label="measured by DrowSim")
    ax.axhline(15, color=MUTED, ls="--", lw=1.1)
    ax.text(t[-1] * 0.995, 15.9, "critical threshold 15%", ha="right",
            fontsize=7.5, color=MUTED)
    ax.set_xlabel("time (s)"); ax.set_ylabel("PERCLOS (%)")
    ax.set_xlim(0, t[-1]); ax.set_ylim(0, max(true.max(), meas.max()) * 1.18)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "fig_4_6_perclos_trace.png")


# ---------------------------------------------------------------- Figure 4.1
def architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, title, lines, edge=MUTED, fill="#f7f7f5"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    linewidth=1.1, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", fontsize=8.6,
                fontweight="bold", color=INK)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 7.4 - i * 3.4, ln, ha="center",
                    fontsize=7.4, color=MUTED)

    def arrow(x1, y1, x2, y2, label="", colour=MUTED, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, color=colour, lw=1.2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.4, label, ha="center",
                    fontsize=7.2, color=colour)

    ax.text(17, 49, "IN-VEHICLE CLIENT (edge)", ha="center", fontsize=8,
            color=BLUE, fontweight="bold")
    ax.text(58, 49, "SERVER", ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.text(87, 49, "OPERATOR", ha="center", fontsize=8, color=INK, fontweight="bold")

    box(2, 33, 30, 12, "Camera → YOLOv8n", ["frame held in RAM only,", "overwritten next capture"], edge=BLUE)
    box(2, 18, 30, 12, "Feature extraction", ["7 numeric fields", "no image data"], edge=BLUE)
    box(2, 3, 30, 12, "Rule engine → ALARM", ["acute closure + PERCLOS", "authoritative, works offline"], edge=CRIT, fill="#fdf2f0")

    box(40, 26, 30, 12, "Session buffer", ["32-frame window", "in memory only"])
    box(40, 11, 30, 12, "BiLSTM-Attention", ["contextual verdict", "never decides microsleep"])

    box(76, 18, 22, 14, "Dashboard", ["node states", "event log", "no imagery"])

    arrow(17, 33, 17, 30)
    arrow(17, 18, 17, 15)
    ax.add_patch(FancyArrowPatch((32, 24), (40, 31), arrowstyle="-|>",
                                 mutation_scale=11, color=BLUE, lw=1.4))
    ax.text(36, 20.5, "7 floats\nper frame", ha="center", fontsize=7.4, color=BLUE,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    arrow(55, 26, 55, 23)
    arrow(70, 17, 76, 22)

    ax.text(50, 0.2, "The frame never crosses a tier boundary.", ha="center",
            fontsize=8, color=CRIT, style="italic")
    save(fig, "fig_4_1_architecture.png")


# ---------------------------------------------------------------- Figure 4.3
def state_machine():
    """Staggered layout.

    A single row put every transition label into the narrow gaps between boxes,
    where they overlapped both the boxes and each other. Offsetting the states
    diagonally opens real space beside each arrow for its label.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")

    states = [(14, 17, "NORMAL", OK), (50, 34, "WARNING", WARN),
              (86, 51, "CRITICAL", CRIT), (86, 14, "NO_FACE", MUTED)]
    for x, y, name, col in states:
        ax.add_patch(FancyBboxPatch((x - 12, y - 4.4), 24, 8.8,
                                    boxstyle="round,pad=0.5", linewidth=1.5,
                                    edgecolor=col, facecolor="white"))
        ax.text(x, y, name, ha="center", va="center", fontsize=9,
                fontweight="bold", color=col)

    def edge(p1, p2, rad=0.0, col=MUTED):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                     color=col, lw=1.2,
                                     connectionstyle=f"arc3,rad={rad}"))

    def tag(x, y, text, col=MUTED):
        ax.text(x, y, text, ha="center", va="center", fontsize=7.4, color=col,
                linespacing=1.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.8))

    # escalation
    edge((25, 20), (39, 30), col=WARN)
    tag(25, 30, "PERCLOS ≥ 12%\nor 3 yawns / 5 min", WARN)
    edge((61, 37), (75, 47), col=CRIT)
    tag(61, 47, "PERCLOS\n≥ 15%", CRIT)

    # the acute rule bypasses WARNING entirely
    edge((16, 22), (78, 55), rad=-0.30, col=CRIT)
    tag(46, 61, "eyes closed ≥ 0.8 s  —  bypasses WARNING", CRIT)

    # recovery
    edge((39, 31), (26, 19.5))
    tag(44, 25.5, "trend clears")
    # Routed underneath rather than across: the earlier arc cut straight
    # through the WARNING box.
    edge((82, 46.4), (22, 13.5), rad=0.34)
    tag(50, 8.5, "closure ends and PERCLOS falls")

    # loss of observation - entered from any state
    edge((64, 29), (76, 18.5))
    tag(80, 31, "no face detected")

    ax.text(50, 1.0, "The acute rule is evaluated last, so a microsleep in progress "
                     "overrides every other assessment.",
            ha="center", fontsize=7.8, color=INK, style="italic")
    save(fig, "fig_4_3_state_machine.png")


if __name__ == "__main__":
    print("generating charts:")
    perclos_dilution()
    class_distribution()
    confusion()
    model_comparison()
    perclos_trace()
    architecture()
    state_machine()
    print(f"\nwritten to {OUT}")


# ------------------------------------------------------ image-based figures
def _pil():
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont


def _font(size):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def dataset_samples():
    """Real FL3D crops, three per class, so the reader sees the actual data."""
    Image, ImageDraw, _ = _pil()
    ann = json.loads(Path("data/fl3d/classification_frames/annotations_holdout.json").read_text())
    by_class = {c: [] for c in ("alert", "microsleep", "yawning")}
    for k, v in sorted(ann.items()):
        st = v["driver_state"]
        if st in by_class and len(by_class[st]) < 3:
            by_class[st].append(Path("data/fl3d") / k.removeprefix("./"))

    cw, ch, pad, lab = 150, 200, 10, 26
    sheet = Image.new("RGB", (cw * 3 + pad * 4, (ch + lab) * 3 + pad * 4), "white")
    d = ImageDraw.Draw(sheet)
    f = _font(15)
    for r, (cls, paths) in enumerate(by_class.items()):
        y = pad + r * (ch + lab + pad)
        d.text((pad, y), cls, fill=(11, 11, 11), font=f)
        for c, p in enumerate(paths):
            im = Image.open(p).convert("RGB").resize((cw, ch))
            sheet.paste(im, (pad + c * (cw + pad), y + lab))
    sheet.save(OUT / "fig_3_2_dataset_samples.png")
    print("  fig_3_2_dataset_samples.png")


def composite_illustration():
    """Left: the crop as FL3D ships it. Right: the composite used for training."""
    Image, ImageDraw, _ = _pil()
    files = sorted(Path("data/yolo/images/train").glob("*.jpg"))
    comp = next(f for f in files if "crop" not in f.name)
    lbl = (Path("data/yolo/labels/train") / f"{comp.stem}.txt").read_text().split()
    _, cx, cy, bw, bh = (float(v) for v in lbl)

    ci = Image.open(comp).convert("RGB")
    W, H = 300, 300
    ci = ci.resize((W, H))
    d = ImageDraw.Draw(ci)
    d.rectangle([(cx - bw / 2) * W, (cy - bh / 2) * H,
                 (cx + bw / 2) * W, (cy + bh / 2) * H], outline=(0, 170, 90), width=3)

    # Composites are named "<video>_<frame index>"; the video id itself
    # contains an underscore, so split from the right.
    src_stem = comp.stem.rsplit("_", 1)[0]
    crops = sorted(Path("data/fl3d/classification_frames").glob(f"{src_stem}/frame*.jpg"))
    raw = Image.open(crops[0]).convert("RGB").resize((int(W * 0.72), H))

    pad, lab = 14, 28
    sheet = Image.new("RGB", (raw.width + W + pad * 3, H + lab + pad * 2), "white")
    d2 = ImageDraw.Draw(sheet)
    f = _font(14)
    d2.text((pad, pad), "FL3D crop as distributed", fill=(11, 11, 11), font=f)
    d2.text((pad * 2 + raw.width, pad), "composited, with derived bounding box",
            fill=(11, 11, 11), font=f)
    sheet.paste(raw, (pad, pad + lab))
    sheet.paste(ci, (pad * 2 + raw.width, pad + lab))
    sheet.save(OUT / "fig_3_4_composite.png")
    print("  fig_3_4_composite.png")


def _frame_from(video, index, out_name):
    import cv2
    cap = cv2.VideoCapture(str(video))
    frame, i = None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i == index:
            frame = f
            break
        i += 1
    cap.release()
    if frame is None:
        print(f"  ! {out_name}: frame {index} not available in {video}")
        return
    cv2.imwrite(str(OUT / out_name), frame)
    print(f"  {out_name}")


def interface_figures():
    # A frame in the CRITICAL state, so the figure shows the interface doing its job.
    _frame_from("deliverables/raw_online.mp4", 230, "fig_4_7_client_interface.png")
    frames = sorted(Path("data/dashboard_frames").glob("*.png"))
    if frames:
        from PIL import Image
        im = Image.open(frames[min(180, len(frames) - 1)]).convert("RGB")
        grey = np.array(im.convert("L"))
        rows = np.where(grey.max(axis=1) > 40)[0]
        if len(rows):
            im = im.crop((0, 0, im.width, min(int(rows[-1]) + 12, im.height)))
        im.save(OUT / "fig_4_8_dashboard.png")
        print("  fig_4_8_dashboard.png")


if True:  # appended generators
    print("\ngenerating image figures:")
    dataset_samples()
    composite_illustration()
    interface_figures()
    print(f"\nall figures in {OUT}")
