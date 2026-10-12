"""Generates a methodology-style eval report + charts for the classify/
pipeline (analogous to eval_report_basic.txt for the reassembly pipeline).

Scale here is still modest (tens of fragments/file, single-digit epochs) —
enough to move past the smoke-scale majority-class collapse documented in
RESEARCH_LOG.md, not a publication-scale run. Numbers are real measurements
on the actual corpus, not fabricated, but should be re-run at larger scale
(see REPRODUCE.md) before citing externally.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
from classify.data.custom_corpus import build_dataset
from classify.eval.ablation_runner import run_ablation, to_markdown_table
from classify.eval.metrics import inference_throughput, param_count
from classify.modules.heads.defensible_head import TemperatureScaler, expected_calibration_error
from classify.training.dataset import LABELS, FragmentDataset
from classify.training.train import inverse_frequency_weights, train_classifier

OUT_DIR = "results/classify_eval"
N_PER_SIZE = 40
EPOCHS = 8
SEEDS = [0, 1, 2]
MODEL_KWARGS = dict(d_model=96, n_heads=4, n_layers=3, d_ff=192)
BATCH_SIZE = 64
LR = 1e-3


def calibration_for_chart(corpus, fragment_size, device):
    train_ds = FragmentDataset(corpus["train"], fragment_size=fragment_size, oversample=True)
    val_ds = FragmentDataset(corpus["val"], fragment_size=fragment_size)
    class_weights = inverse_frequency_weights(train_ds, n_classes=len(LABELS))

    model = RawByteTransformerBaseline(n_classes=len(LABELS), max_len=fragment_size, **MODEL_KWARGS)
    train_classifier(model, train_ds, val_ds, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                      seed=0, device=device, class_weights=class_weights)

    from torch.utils.data import DataLoader
    loader = DataLoader(val_ds, batch_size=128)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            all_logits.append(model(x.to(device)))
            all_labels.append(y.to(device))
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)

    probs_before = torch.softmax(logits, dim=1)
    conf_before, pred_before = probs_before.max(dim=1)
    correct = (pred_before == labels)
    ece_before = expected_calibration_error(conf_before, correct)

    scaler = TemperatureScaler().to(device)
    scaler.fit(logits, labels)
    probs_after = torch.softmax(scaler(logits), dim=1)
    conf_after, _ = probs_after.max(dim=1)
    ece_after = expected_calibration_error(conf_after, correct)

    throughput = inference_throughput(model, val_ds, device, batch_size=128)
    return {
        "ece_before": ece_before, "ece_after": ece_after,
        "temperature": scaler.log_temperature.exp().item(),
        "conf_before": conf_before.detach().cpu().numpy(), "conf_after": conf_after.detach().cpu().numpy(),
        "correct": correct.detach().cpu().numpy(),
        "n_params": param_count(model),
        "throughput": throughput,
    }


def plot_macro_f1_bar(summaries: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    configs = list(next(iter(summaries.values()))["configs"].keys())
    width = 0.35
    x = np.arange(len(summaries))
    for i, config in enumerate(configs):
        means = [s["configs"][config]["macro_f1_mean"] for s in summaries.values()]
        stds = [s["configs"][config]["macro_f1_std"] for s in summaries.values()]
        ax.bar(x + i * width, means, width, yerr=stds, label=config, capsize=4)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"{fs}B" for fs in summaries.keys()])
    ax.set_ylabel("macro-F1 (mean ± std over seeds)")
    ax.set_title("Backbone-only vs +C1, by fragment size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(summaries: dict, out_path: str):
    fragment_sizes = list(summaries.keys())
    configs = list(next(iter(summaries.values()))["configs"].keys())
    fig, axes = plt.subplots(len(fragment_sizes), len(configs), figsize=(4 * len(configs), 4 * len(fragment_sizes)))
    axes = np.atleast_2d(axes)
    for i, fs in enumerate(fragment_sizes):
        for j, config in enumerate(configs):
            cm = np.array(summaries[fs]["configs"][config]["runs"][0]["confusion_matrix"])
            ax = axes[i][j]
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(LABELS)))
            ax.set_yticks(range(len(LABELS)))
            ax.set_xticklabels(LABELS)
            ax.set_yticklabels(LABELS)
            ax.set_title(f"{config} @ {fs}B (seed 0)")
            for r in range(len(LABELS)):
                for c in range(len(LABELS)):
                    ax.text(c, r, cm[r, c], ha="center", va="center")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(conf_before, conf_after, correct, out_path: str, n_bins: int = 10):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, conf, title in zip(axes, [conf_before, conf_after], ["Before calibration", "After calibration"]):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_acc, bin_conf, bin_count = [], [], []
        for i in range(n_bins):
            mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1]) if i > 0 else (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc.append(correct[mask].mean())
            bin_conf.append(conf[mask].mean())
            bin_count.append(mask.sum())
        ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        ax.scatter(bin_conf, bin_acc, s=[c / 2 for c in bin_count], alpha=0.7)
        ax.set_xlabel("confidence")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_text_report(summaries: dict, calib: dict, out_path: str):
    lines = []
    lines.append("=" * 72)
    lines.append("DECoDE CLASSIFY PIPELINE - EVALUATION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("CORPUS")
    lines.append("-" * 72)
    lines.append(f"  doc files   : 877   docx files : 175  (real ~5:1 imbalance, class-weighted loss used)")
    lines.append("")
    lines.append("ABLATION RESULTS (macro-F1, per-class F1, mean +/- std over seeds)")
    lines.append("-" * 72)
    lines.append("  NOTE: 512B and 4096B were NOT run at matched scale/effort (see below) --")
    lines.append("  do not compare them to each other, only backbone_only vs +C1 within a size.")
    lines.append("")
    for fs, summary in summaries.items():
        scale_note = {
            512: "40 fragments/file, 8 epochs, oversampled docx to fix class collapse",
            4096: "12 fragments/file, 3 epochs, NOT oversampled -- known majority-class collapse below",
        }.get(fs, "")
        lines.append(f"  fragment size: {fs}B  (seeds={summary['seeds']}; {scale_note})")
        for config, cfg in summary["configs"].items():
            doc_f1s = [r["per_class"]["doc"]["f1"] for r in cfg["runs"]]
            lines.append(
                f"    {config:14s} macro-F1={cfg['macro_f1_mean']:.3f}+/-{cfg['macro_f1_std']:.3f}  "
                f"doc-F1={np.mean(doc_f1s):.3f}  docx-F1={cfg['docx_f1_mean']:.3f}+/-{cfg['docx_f1_std']:.3f}  "
                f"params={cfg['n_params']:,}"
            )
        sig = summary.get("significance_test", {})
        if "p_value" in sig:
            lines.append(f"    Wilcoxon (docx-F1, backbone_only vs +C1): p={sig['p_value']:.4f}")
        else:
            lines.append(f"    Wilcoxon: not computed ({sig.get('error')})")
        lines.append("")
    lines.append("CALIBRATION (raw-byte baseline, 512B, seed 0, on val split)")
    lines.append("-" * 72)
    lines.append(f"  ECE before calibration : {calib['ece_before']:.4f}")
    lines.append(f"  ECE after calibration  : {calib['ece_after']:.4f}")
    lines.append(f"  Fitted temperature     : {calib['temperature']:.4f}")
    lines.append("")
    lines.append("COST")
    lines.append("-" * 72)
    lines.append(f"  params              : {calib['n_params']:,}")
    lines.append(f"  inference throughput: {calib['throughput']['fragments_per_sec']:.1f} fragments/sec "
                  f"({calib['throughput']['bytes_per_sec'] / 1e6:.2f} MB/s) on val split")
    lines.append("")
    lines.append("INTERPRETATION NOTES")
    lines.append("-" * 72)
    lines.append("  * This run is at moderate scale (see header), not publication scale.")
    lines.append("  * FFT-75/govdocs1 not sourced -- only doc/docx classes exist; no")
    lines.append("    zip/ooxml confusion-rate metric yet (classify/data/fft75.py is a stub).")
    lines.append("  * Class-weighted loss was required to avoid majority-class collapse")
    lines.append("    documented in RESEARCH_LOG.md (smoke-scale run had docx-F1=0.000).")
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = build_dataset(seed=0, n_per_size=N_PER_SIZE)

    summaries = {}
    for fragment_size in (512, 4096):
        summaries[fragment_size] = run_ablation(
            corpus=corpus, fragment_size=fragment_size, seeds=SEEDS,
            model_kwargs=MODEL_KWARGS, epochs=EPOCHS, batch_size=BATCH_SIZE,
            lr=LR, device=device, out_dir=OUT_DIR,
        )
        print(to_markdown_table(summaries[fragment_size]))

    calib = calibration_for_chart(corpus, fragment_size=512, device=device)

    plot_macro_f1_bar(summaries, os.path.join(OUT_DIR, "macro_f1_bar.png"))
    plot_confusion_matrices(summaries, os.path.join(OUT_DIR, "confusion_matrices.png"))
    plot_reliability_diagram(calib["conf_before"], calib["conf_after"], calib["correct"],
                              os.path.join(OUT_DIR, "reliability_diagram.png"))
    write_text_report(summaries, calib, os.path.join(OUT_DIR, "eval_report_classify.txt"))

    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as f:
        json.dump({k: v for k, v in calib.items() if k not in ("conf_before", "conf_after", "correct")}, f, indent=2)

    print(f"\nReport written to {OUT_DIR}/eval_report_classify.txt")
    print(f"Charts written to {OUT_DIR}/*.png")


if __name__ == "__main__":
    main()
