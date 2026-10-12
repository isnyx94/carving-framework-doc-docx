"""Chapter 4 pilot-study package (512B): thesis-quality charts, a pipeline
diagram, and a full Excel workbook — built entirely from the already-computed
result JSONs (no retraining):

  results/new_result/fullspec_512B.json    — FFT-75 9-class, 256/8/6, 5 seeds, C2/C4
  results/new_result/p1_gate_512B.json     — P1 gate (75-class CNN) + 128-dim compound run
  results/new_result/full_pipeline_512B.json — custom doc/docx 2-class pilot (3 ablation levels)
  results/new_result/hypertune.json        — recipe sweep

Visual conventions follow the dataviz skill's validated reference palette:
categorical slots in fixed order (never cycled), entity->color mapping constant
across every chart, one hue for sequential (confusion heatmap), recessive
grid/chrome, direct labels where they earn their place, no dual axes.

Output: results/chapter4_pilot/  (ch4_*.png + CH4_PILOT_RESULTS.xlsx)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IN_DIR = "results/new_result"
OUT_DIR = "results/chapter4_pilot"

# ---- validated reference palette (dataviz skill, light mode) ----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SLOT = {  # fixed categorical order — assignment below never changes per chart
    1: "#2a78d6",  # blue
    2: "#1baf7a",  # aqua
    3: "#eda100",  # yellow
    4: "#008300",  # green
    5: "#4a3aa7",  # violet
    6: "#e34948",  # red
}
ENTITY = {
    "backbone_only": SLOT[1],
    "+C1": SLOT[2],
    "random_forest": SLOT[3],
    "xgboost": SLOT[4],
    "+C1_detection_only": SLOT[5],
    "doc": SLOT[1],
    "docx": SLOT[2],
    "IG": SLOT[1],
    "random": SLOT[6],
}
SEQ_BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
PRETTY = {"backbone_only": "Backbone only", "+C1": "+C1",
           "+C1_detection_only": "+C1 (detection only)",
           "random_forest": "Random Forest", "xgboost": "XGBoost"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "Segoe UI", "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BASELINE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _load():
    data = {}
    for key, name in (("fullspec", "fullspec_512B.json"), ("gate", "p1_gate_512B.json"),
                       ("custom", "full_pipeline_512B.json"), ("hypertune", "hypertune.json")):
        with open(os.path.join(IN_DIR, name)) as f:
            data[key] = json.load(f)
    return data


def _save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name))
    plt.close(fig)
    print(f"  wrote {name}", flush=True)


# ------------------------------------------------------------------- charts --

def chart_main_results(d):
    """Horizontal bars: macro-F1 by model (FFT-75, 9 classes)."""
    a, b = d["fullspec"]["stage_a"], d["fullspec"]["stage_b"]
    rows = [
        ("random_forest", a["random_forest"]["macro_f1"], 0.0),
        ("xgboost", a["xgboost"]["macro_f1"], 0.0),
        ("backbone_only", b["configs"]["backbone_only"]["macro_f1_mean"],
          b["configs"]["backbone_only"]["macro_f1_std"]),
        ("+C1", b["configs"]["+C1"]["macro_f1_mean"], b["configs"]["+C1"]["macro_f1_std"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(rows))[::-1]
    for yi, (name, mean, std) in zip(y, rows):
        ax.barh(yi, mean, height=0.55, color=ENTITY[name],
                 xerr=std if std else None, error_kw=dict(ecolor=INK2, capsize=3, lw=1))
        ax.text(mean + 0.012, yi, f"{mean:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[r[0]] for r in rows])
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("macro-F1 (test)")
    ax.set_title("Main results — FFT-75 compound subset, 512 B (9 classes)\n"
                  "transformers: mean ± std over 5 seeds; classical: single fit", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, "ch4_main_results.png")


def chart_perclass_doc_docx(d):
    """Grouped bars: doc-F1 / docx-F1 per model."""
    a, b = d["fullspec"]["stage_a"], d["fullspec"]["stage_b"]
    models = ["random_forest", "xgboost", "backbone_only", "+C1"]

    def f1(model, cls):
        if model in a:
            return a[model]["per_class"][cls]["f1"], 0.0
        runs = b["configs"][model]["runs"]
        vals = [r["per_class"][cls]["f1"] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(models))
    width = 0.38
    for j, cls in enumerate(("doc", "docx")):
        means, stds = zip(*[f1(m, cls) for m in models])
        ax.bar(x + (j - 0.5) * width, means, width * 0.92, yerr=stds, capsize=3,
                color=ENTITY[cls], error_kw=dict(ecolor=INK2, lw=1), label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in models], fontsize=8.5)
    ax.set_ylabel("per-class F1 (test)")
    ax.set_title("doc vs docx per-class F1 — FFT-75, 512 B", loc="left")
    ax.legend(frameon=False, title=None)
    ax.grid(axis="x", visible=False)
    _save(fig, "ch4_perclass_doc_docx.png")


def chart_docxzip_slope(d):
    """Paired slope chart: docx->zip confusion per seed, backbone -> +C1."""
    runs_b = d["fullspec"]["stage_b"]["configs"]["backbone_only"]["runs"]
    runs_c = d["fullspec"]["stage_b"]["configs"]["+C1"]["runs"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for rb, rc in zip(runs_b, runs_c):
        v0, v1 = rb["docx_zip_confusion_rate"], rc["docx_zip_confusion_rate"]
        falls = v1 < v0
        ax.plot([0, 1], [v0, v1], color=ENTITY["+C1"] if falls else MUTED,
                 lw=1.6, alpha=0.9, marker="o", ms=5,
                 markerfacecolor=SURFACE, markeredgewidth=1.6)
        ax.annotate(f"seed {rb['seed']}", (0, v0), textcoords="offset points",
                     xytext=(-8, 0), ha="right", va="center", fontsize=8, color=INK2)
    mb = np.mean([r["docx_zip_confusion_rate"] for r in runs_b])
    mc = np.mean([r["docx_zip_confusion_rate"] for r in runs_c])
    ax.plot([0, 1], [mb, mc], color=INK, lw=2.6, marker="o", ms=6, zorder=5)
    ax.annotate(f"mean {mb:.3f}", (0, mb), textcoords="offset points", xytext=(-8, 8),
                 ha="right", fontsize=9, color=INK, fontweight="bold")
    ax.annotate(f"{mc:.3f}", (1, mc), textcoords="offset points", xytext=(8, 8),
                 ha="left", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Backbone only", "+C1"])
    ax.set_xlim(-0.45, 1.3)
    ax.set_ylabel("docx→zip confusion rate")
    ax.set_title("docx→zip confusion, paired by seed — 512 B\n"
                  "falls on 4/5 seeds (paired t p=0.128 — suggestive, not significant)",
                  loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
    _save(fig, "ch4_docxzip_slope.png")


def chart_deletion_curve(d):
    dc = d["fullspec"]["stage_c"]["+C1"]["deletion_curve"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(dc["fractions"], dc["ig"], color=ENTITY["IG"], lw=2, marker="o", ms=5,
             markerfacecolor=SURFACE, markeredgewidth=1.6, label="IG-ranked deletion")
    ax.plot(dc["fractions"], dc["random"], color=ENTITY["random"], lw=2, marker="s", ms=5,
             markerfacecolor=SURFACE, markeredgewidth=1.6, label="random deletion")
    ax.annotate("faithful attribution:\nIG deletion hurts faster", xy=(0.2, dc["ig"][2]),
                 xytext=(0.33, 0.42), fontsize=8.5, color=INK2,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel("fraction of bytes deleted (by attribution rank)")
    ax.set_ylabel("accuracy on perturbed fragments")
    ax.set_title("C2 attribution faithfulness — deletion curve, +C1 @ 512 B "
                  f"(n={dc['n_fragments']})", loc="left", fontsize=10)
    ax.legend(frameon=False)
    _save(fig, "ch4_deletion_curve.png")


def chart_c4_rejection(d):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name in ("backbone_only", "+C1"):
        e = d["fullspec"]["stage_c"][name]
        xs = [r["id_kept_fraction"] for r in e["c4_rejection_curve"]]
        ys = [r["accuracy_on_kept"] for r in e["c4_rejection_curve"]]
        ax.plot(xs, ys, color=ENTITY[name], lw=2, marker="o", ms=5,
                 markerfacecolor=SURFACE, markeredgewidth=1.6,
                 label=f"{PRETTY[name]}  (AUROC {e['c4_auroc']:.3f})")
    ax.set_xlabel("fraction of in-distribution fragments kept (not abstained)")
    ax.set_ylabel("accuracy on kept fragments")
    ax.set_title("C4 open-set abstention — accuracy/coverage trade-off, 512 B\n"
                  "OOD: pdf / jpeg / random fragments; MSP after temperature scaling",
                  loc="left", fontsize=10)
    ax.legend(frameon=False, loc="lower left")
    _save(fig, "ch4_c4_rejection.png")


def chart_calibration(d):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    names = ("backbone_only", "+C1")
    x = np.arange(len(names))
    width = 0.38
    before = [d["fullspec"]["stage_c"][n]["ece_before"] for n in names]
    after = [d["fullspec"]["stage_c"][n]["ece_after"] for n in names]
    ax.bar(x - width / 2, before, width * 0.92, color=MUTED, label="before calibration")
    ax.bar(x + width / 2, after, width * 0.92,
            color=[ENTITY[n] for n in names], label="after (temperature scaled)")
    for xi, (b, a) in zip(x, zip(before, after)):
        ax.text(xi - width / 2, b + 0.0008, f"{b:.3f}", ha="center", fontsize=8.5, color=INK2)
        ax.text(xi + width / 2, a + 0.0008, f"{a:.3f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[n] for n in names])
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("C2 calibration — ECE before/after temperature scaling, 512 B", loc="left",
                  fontsize=10)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    _save(fig, "ch4_calibration.png")


def chart_confusion_matrices(d):
    labels = d["gate"]["compound"]["labels"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUES)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, name in zip(axes, ("backbone_only", "+C1")):
        cm = np.array(d["fullspec"]["stage_b"]["configs"][name]["runs"][0]["confusion_matrix"],
                       dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{PRETTY[name]} (seed 0, row-normalised)", fontsize=10)
        ax.grid(visible=False)
        for r in range(len(labels)):
            for c in range(len(labels)):
                if cm_norm[r, c] >= 0.05:
                    ax.text(c, r, f"{cm_norm[r, c]:.2f}", ha="center", va="center",
                             fontsize=6.5, color=SURFACE if cm_norm[r, c] > 0.5 else INK)
    fig.suptitle("Confusion matrices — FFT-75 compound subset, 512 B", x=0.02, ha="left",
                  fontsize=11, color=INK)
    _save(fig, "ch4_confusion_matrices.png")


def chart_custom_corpus(d):
    ab = d["custom"]["ablation"]["configs"]
    names = ["backbone_only", "+C1_detection_only", "+C1"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(names))
    width = 0.38
    macro = [ab[n]["macro_f1_mean"] for n in names]
    macro_std = [ab[n]["macro_f1_std"] for n in names]
    docx = [ab[n]["docx_f1_mean"] for n in names]
    docx_std = [ab[n]["docx_f1_std"] for n in names]
    ax.bar(x - width / 2, macro, width * 0.92, yerr=macro_std, capsize=3, color=SLOT[1],
            error_kw=dict(ecolor=INK2, lw=1), label="macro-F1")
    ax.bar(x + width / 2, docx, width * 0.92, yerr=docx_std, capsize=3, color=SLOT[2],
            error_kw=dict(ecolor=INK2, lw=1), label="docx-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[n] for n in names], fontsize=8.5)
    ax.set_ylabel("F1 (test)")
    ax.set_title("Preliminary pilot — custom 2-class corpus (877 doc / 175 docx), 512 B\n"
                  "3 ablation levels, 3 seeds, tuned recipe", loc="left", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    _save(fig, "ch4_custom_corpus.png")


def chart_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, lines, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                                      facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(x + w / 2, y + h - 4.5, title, ha="center", va="top", fontsize=9.5,
                 fontweight="bold", color=INK)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 10 - i * 4.6, line, ha="center", va="top",
                     fontsize=8, color=INK2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                       mutation_scale=14, color=INK2, lw=1.4))

    tint = {"blue": "#e6f0fc", "aqua": "#e2f6ee", "violet": "#eceafd", "yellow": "#fdf3dd"}

    box(2, 62, 22, 30, "Fragment intake", ["512 B raw bytes", "uniform random offset",
         "per-parent-file splits"], SURFACE, BASELINE)
    box(30, 55, 30, 41, "C1 compound-aware front-end",
         ["heuristic: entropy + ZIP-LFH", "CNN refiner (ambiguous only)",
          "partial raw-deflate decompress", "residual gate α per token",
          "α·raw + (1−α)·recovered"], tint["aqua"], ENTITY["+C1"])
    box(66, 62, 30, 30, "Foundation encoder (shared)",
         ["byte embedding (256-d)", "6-layer, 8-head transformer",
          "learned positions, mean-pool"], tint["blue"], ENTITY["backbone_only"])
    box(10, 8, 24, 30, "Classifier head", ["9-class softmax", "doc / docx / xls / xlsx /",
         "ppt / pptx / zip / pdf / jpg"], SURFACE, BASELINE)
    box(39, 8, 26, 30, "C2 defensible head",
         ["Integrated Gradients", "per-byte attribution", "temperature calibration",
          "ECE + deletion curve"], tint["violet"], SLOT[5])
    box(70, 8, 26, 30, "C4 open-set head",
         ["MSP abstention", "threshold sweep", "AUROC vs OOD",
          "(pdf / jpeg / random)"], tint["yellow"], SLOT[3])

    arrow(24, 77, 30, 77)
    arrow(60, 77, 66, 77)
    arrow(81, 62, 81, 40)   # encoder down to heads row
    arrow(81, 45, 22, 38)
    arrow(81, 45, 52, 38)
    arrow(81, 45, 83, 38)
    ax.text(50, 99, "DECoDE pilot pipeline — 512 B configuration (Chapter 4)",
             ha="center", fontsize=11.5, fontweight="bold", color=INK)
    _save(fig, "ch4_pipeline_diagram.png")


# -------------------------------------------------------------------- excel --

def write_excel(d):
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font

    wb = Workbook()
    bold = Font(bold=True)

    def sheet(title, header, rows):
        ws = wb.create_sheet(title)
        ws.append(header)
        for cell in ws[1]:
            cell.font = bold
        for row in rows:
            ws.append(row)
        return ws

    a, b, c = d["fullspec"]["stage_a"], d["fullspec"]["stage_b"], d["fullspec"]["stage_c"]

    sheet("Overview", ["Item", "Value"], [
        ["Chapter", "4 — Pilot study (512 B)"],
        ["Benchmark", "FFT-75 scenario 1, 512 B (fragment-level published split)"],
        ["Compound subset", "doc docx xls xlsx ppt pptx zip pdf jpg (9 classes)"],
        ["Transformer backbone", "d=256, 8 heads, 6 layers, d_ff=1024 (CarveFormer-spec)"],
        ["Seeds", "5 (paired t-test, alpha=0.05)"],
        ["Train/eval per class", "20,000 / 5,000 fragments"],
        ["Recipe", "lr=1e-4, 3 epochs, bf16 AMP (hypertuned — see Hypertune sheet)"],
        ["P1 gate", f"PASSED — FiFTy-style CNN {d['gate']['gate']['test_accuracy']:.4f} "
                     f"vs published 0.656 (band ±0.05)"],
        ["Data provenance", "fullspec_512B.json / p1_gate_512B.json / "
                              "full_pipeline_512B.json / hypertune.json"],
    ])

    rows = []
    for name in ("random_forest", "xgboost"):
        m = a[name]
        rows.append([PRETTY[name], round(m["macro_f1"], 4), None,
                      round(m["per_class"]["doc"]["f1"], 4), round(m["per_class"]["docx"]["f1"], 4),
                      round(m["docx_zip_confusion_rate"], 4), round(m["micro_accuracy"], 4)])
    for name, cfg in b["configs"].items():
        doc_f1 = np.mean([r["per_class"]["doc"]["f1"] for r in cfg["runs"]])
        acc = np.mean([r["micro_accuracy"] for r in cfg["runs"]])
        rows.append([PRETTY[name] + " (5 seeds)", round(cfg["macro_f1_mean"], 4),
                      round(cfg["macro_f1_std"], 4), round(float(doc_f1), 4),
                      round(cfg["docx_f1_mean"], 4), round(cfg["docx_zip_confusion_mean"], 4),
                      round(float(acc), 4)])
    sheet("Main Results", ["Model", "macro-F1", "std", "doc-F1", "docx-F1",
                             "docx->zip confusion", "accuracy"], rows)

    rows = []
    for name in ("random_forest", "xgboost"):
        for cls, m in a[name]["per_class"].items():
            rows.append([PRETTY[name], cls, round(m["precision"], 4), round(m["recall"], 4),
                          round(m["f1"], 4), m["support"]])
    for name, cfg in b["configs"].items():
        run0 = cfg["runs"][0]
        for cls, m in run0["per_class"].items():
            rows.append([PRETTY[name] + " (seed 0)", cls, round(m["precision"], 4),
                          round(m["recall"], 4), round(m["f1"], 4), m["support"]])
    sheet("Per-Class", ["Model", "Class", "Precision", "Recall", "F1", "Support"], rows)

    rows = []
    for name, cfg in b["configs"].items():
        for r in cfg["runs"]:
            rows.append([PRETTY[name], r["seed"], round(r["macro_f1"], 4),
                          round(r["per_class"]["doc"]["f1"], 4),
                          round(r["per_class"]["docx"]["f1"], 4),
                          round(r["docx_zip_confusion_rate"], 4), round(r["micro_accuracy"], 4)])
    sheet("Per-Seed", ["Config", "Seed", "macro-F1", "doc-F1", "docx-F1",
                        "docx->zip", "accuracy"], rows)

    runs_b = b["configs"]["backbone_only"]["runs"]
    runs_c = b["configs"]["+C1"]["runs"]
    rows = [[r0["seed"], round(r0["docx_zip_confusion_rate"], 4),
              round(r1["docx_zip_confusion_rate"], 4),
              round(r1["docx_zip_confusion_rate"] - r0["docx_zip_confusion_rate"], 4)]
             for r0, r1 in zip(runs_b, runs_c)]
    ws = sheet("docx-zip analysis", ["Seed", "Backbone", "+C1", "Delta (+C1 - backbone)"], rows)
    ws.append([])
    ws.append(["Paired t-test", "t=-1.917, p=0.128 (NOT significant; suggestive trend, 4/5 seeds fall)"])
    ws.append(["Wilcoxon", "W=2.0, p=0.188"])
    ws.append(["Macro-F1 paired t-test", f"p={b['significance']['p_value']:.4f} (null persists)"])

    rows = []
    for name, e in c.items():
        rows.append([PRETTY[name], round(e["ece_before"], 4), round(e["ece_after"], 4),
                      round(e["temperature"], 4), round(e["c4_auroc"], 4)])
    sheet("C2-C4 (seed 0)", ["Model", "ECE before", "ECE after", "Temperature",
                               "C4 AUROC (ID vs OOD)"], rows)

    dc = c["+C1"]["deletion_curve"]
    sheet("C2 Deletion Curve", ["Fraction deleted", "Accuracy (IG-ranked)", "Accuracy (random)"],
           [[f, round(ig, 4), round(rnd, 4)] for f, ig, rnd in
            zip(dc["fractions"], dc["ig"], dc["random"])])

    rows = []
    for name, e in c.items():
        for r in e["c4_rejection_curve"]:
            rows.append([PRETTY[name], r["threshold"], round(r["id_kept_fraction"], 4),
                          round(r["accuracy_on_kept"], 4) if r["accuracy_on_kept"] is not None else None,
                          round(r["ood_rejected_fraction"], 4)])
    sheet("C4 Rejection Sweep", ["Model", "Threshold", "ID kept", "Acc on kept", "OOD rejected"], rows)

    ab = d["custom"]["ablation"]["configs"]
    rows = []
    for name in ("backbone_only", "+C1_detection_only", "+C1"):
        cfg = ab[name]
        doc_f1 = np.mean([r["per_class"]["doc"]["f1"] for r in cfg["runs"]])
        rows.append([PRETTY[name], round(cfg["macro_f1_mean"], 4), round(cfg["macro_f1_std"], 4),
                      round(float(doc_f1), 4), round(cfg["docx_f1_mean"], 4)])
    ws = sheet("Custom Corpus (2-class)", ["Config", "macro-F1", "std", "doc-F1", "docx-F1"], rows)
    ws.append([])
    ws.append(["Corpus", "877 .doc / 175 .docx real files; per-parent-file splits (leak-free)"])
    ws.append(["Role in chapter", "preliminary pilot preceding the FFT-75 benchmark runs"])

    sheet("Hypertune", ["lr", "balance", "epochs", "doc-F1", "docx-F1", "min-F1",
                          "macro-F1", "geo-mean F1"],
           [[r["lr"], r["balance"], r["epochs"], round(r["doc_f1"], 4), round(r["docx_f1"], 4),
             round(r["min_f1"], 4), round(r["macro_f1"], 4), round(r.get("geo_mean_f1", 0), 4)]
            for r in d["hypertune"]])

    g = d["gate"]["gate"]
    sheet("P1 Gate", ["Item", "Value"], [
        ["Model", "FiFTy-style CNN (approximation, not byte-exact replica)"],
        ["Classes", "75 (scenario 1)"],
        ["Test accuracy", round(g["test_accuracy"], 4)],
        ["Published (Mittal et al.)", g["published_fifty_accuracy"]],
        ["Tolerance band", "±0.05"],
        ["Result", "PASSED" if g["gate_passed"] else "FAILED"],
        ["Val accuracy by epoch", ", ".join(f"{v:.4f}" for v in g["val_acc_history"])],
    ])

    ws = wb.create_sheet("Charts")
    row_anchor = 1
    for png in sorted(p for p in os.listdir(OUT_DIR) if p.endswith(".png")):
        img = XLImage(os.path.join(OUT_DIR, png))
        scale = min(1.0, 900 / img.width)
        img.width, img.height = int(img.width * scale), int(img.height * scale)
        img.anchor = f"A{row_anchor}"
        ws.add_image(img)
        row_anchor += int(img.height / 19) + 3

    del wb["Sheet"]
    wb.save(os.path.join(OUT_DIR, "CH4_PILOT_RESULTS.xlsx"))
    print("  wrote CH4_PILOT_RESULTS.xlsx", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = _load()
    print("generating charts...", flush=True)
    chart_pipeline_diagram()
    chart_main_results(d)
    chart_perclass_doc_docx(d)
    chart_docxzip_slope(d)
    chart_deletion_curve(d)
    chart_c4_rejection(d)
    chart_calibration(d)
    chart_confusion_matrices(d)
    chart_custom_corpus(d)
    print("writing excel...", flush=True)
    write_excel(d)
    print(f"\nDone -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
