"""Builds results/new_result/: a structured report (Sec 6.1-6.5 layout) for
the post-redesign C1 architecture (residual gating, CarveFormer-direction
backbone, CNN-refined detection, paired t-test, Integrated Gradients).

Scale note: the full CarveFormer spec (256-dim/8-head/6-layer, design Q2) is
too expensive to run at meaningful epoch/fragment counts for both 512B and
4096B in one interactive session (quadratic attention cost already caused
multi-hour runs at smaller dims in this project — see the project notes). Per the established pattern in this project (ship a
clearly-labeled moderate-scale run rather than block for hours), this uses:
  - 512B: d_model=128, n_heads=8, n_layers=4 (CarveFormer-direction, scaled down)
  - 4096B: d_model=96,  n_heads=4, n_layers=3 (further scaled down — proven
    tractable in the previous results/classify_eval run)
Numbers here are real measurements on the actual corpus, not fabricated,
but are explicitly NOT at the full Q2 spec — see computational_cost section
for the gap this leaves vs. a true CarveFormer-matched comparison.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
from classify.data.custom_corpus import build_dataset
from classify.eval.ablation_runner import run_ablation, to_markdown_table
from classify.eval.flops import estimate_transformer_flops
from classify.eval.metrics import inference_throughput, param_count
from classify.modules.c1_compound.dual_view_model import C1Classifier
from classify.modules.c1_compound.train_refiner import train_decompression_confidence_cnn
from classify.modules.heads.defensible_head import TemperatureScaler, expected_calibration_error
from classify.training.dataset import LABELS, FragmentDataset
from classify.training.train import inverse_frequency_weights, train_classifier

OUT_DIR = "results/new_result"

RUN_CONFIG = {
    # lr=3e-4, not 1e-3: a diagnostic LR sweep (classify/eval/diagnose_collapse.py,
    # the project notes) found lr=1e-3 oscillates between majority-class
    # collapse modes epoch to epoch on this backbone size, while lr=3e-4 trains
    # stably (val_acc climbs monotonically-ish to 0.75 by epoch 8).
    512: dict(n_per_size=30, epochs=8, model_kwargs=dict(d_model=128, n_heads=8, n_layers=4, d_ff=512),
              batch_size=32, lr=3e-4, refiner_epochs=3),
    4096: dict(n_per_size=10, epochs=6, model_kwargs=dict(d_model=96, n_heads=4, n_layers=3, d_ff=192),
               batch_size=16, lr=3e-4, refiner_epochs=2),
}
SEEDS = [0, 1, 2]


def calibrate_one(model_builder, corpus, fragment_size: int, device: str) -> dict:
    cfg = RUN_CONFIG[fragment_size]
    train_ds = FragmentDataset(corpus["train"], fragment_size=fragment_size, oversample=True)
    val_ds = FragmentDataset(corpus["val"], fragment_size=fragment_size)
    class_weights = inverse_frequency_weights(train_ds, n_classes=len(LABELS))

    model = model_builder().to(device)
    train_classifier(model, train_ds, val_ds, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                      lr=cfg["lr"], seed=0, device=device, class_weights=class_weights)

    loader = DataLoader(val_ds, batch_size=64)
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

    throughput = inference_throughput(model, val_ds, device, batch_size=64)
    flops = estimate_transformer_flops(
        d_model=cfg["model_kwargs"]["d_model"], n_layers=cfg["model_kwargs"]["n_layers"],
        d_ff=cfg["model_kwargs"]["d_ff"], seq_len=fragment_size,
    )
    n_params = param_count(model)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "ece_before": ece_before, "ece_after": ece_after,
        "temperature": scaler.log_temperature.exp().item(),
        "n_params": n_params,
        "flops_per_fragment": flops,
        "throughput": throughput,
        "conf_before": conf_before.detach().cpu().numpy(),
        "conf_after": conf_after.detach().cpu().numpy(),
        "correct": correct.detach().cpu().numpy(),
    }


def run_size(fragment_size: int, device: str) -> dict:
    cfg = RUN_CONFIG[fragment_size]
    corpus = build_dataset(seed=0, n_per_size=cfg["n_per_size"], sizes=(fragment_size,))

    ablation_summary = run_ablation(
        corpus=corpus, fragment_size=fragment_size, seeds=SEEDS, model_kwargs=cfg["model_kwargs"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"], device=device,
        out_dir=OUT_DIR, refiner_epochs=cfg["refiner_epochs"],
    )
    print(to_markdown_table(ablation_summary))

    refiner = train_decompression_confidence_cnn(
        corpus["train"], fragment_size=fragment_size, epochs=cfg["refiner_epochs"],
        batch_size=cfg["batch_size"], device=device,
    )
    calib_backbone = calibrate_one(
        lambda: RawByteTransformerBaseline(n_classes=len(LABELS), max_len=fragment_size, **cfg["model_kwargs"]),
        corpus, fragment_size, device,
    )
    calib_c1 = calibrate_one(
        lambda: C1Classifier(n_classes=len(LABELS), max_len=fragment_size,
                              pretrained_refiner=refiner, **cfg["model_kwargs"]),
        corpus, fragment_size, device,
    )
    return {"ablation": ablation_summary, "calib_backbone": calib_backbone, "calib_c1": calib_c1, "corpus": corpus}


def dataset_statistics_table(results: dict) -> str:
    lines = [
        "| Fragment size | Split | doc | docx | total | docx frac. deflate-labelled |",
        "|---|---|---|---|---|---|",
    ]
    for fs, r in results.items():
        corpus = r["corpus"]
        for split in ("train", "val", "test"):
            frags = [f for f in corpus[split] if f["fragment_size"] == fs]
            n_doc = sum(1 for f in frags if f["label"] == "doc")
            n_docx = sum(1 for f in frags if f["label"] == "docx")
            n_docx_deflate = sum(1 for f in frags if f["label"] == "docx" and f["is_deflate"])
            frac = n_docx_deflate / n_docx if n_docx else 0.0
            lines.append(f"| {fs}B | {split} | {n_doc} | {n_docx} | {n_doc + n_docx} | {frac:.2f} |")
    lines.append("")
    lines.append("Source corpus: 877 `.doc` (OLE2/CFBF) + 175 `.docx` (OOXML) files in this repo's "
                  "`doc/`/`docx/` directories (custom corpus, methodology §8.1). FFT-75/govdocs1 not "
                  "sourced — see `classify/data/fft75.py`; no `zip`/`ooxml` classes exist yet.")
    return "\n".join(lines)


def main_results_table(results: dict) -> str:
    lines = ["| Fragment size | Config | macro-F1 | docx-F1 | doc-F1 | params |", "|---|---|---|---|---|---|"]
    for fs, r in results.items():
        for config, cfg in r["ablation"]["configs"].items():
            doc_f1 = np.mean([run["per_class"]["doc"]["f1"] for run in cfg["runs"]])
            lines.append(
                f"| {fs}B | {config} | {cfg['macro_f1_mean']:.3f}±{cfg['macro_f1_std']:.3f} "
                f"| {cfg['docx_f1_mean']:.3f}±{cfg['docx_f1_std']:.3f} | {doc_f1:.3f} | {cfg['n_params']:,} |"
            )
    return "\n".join(lines)


def ablation_table(results: dict) -> str:
    lines = [
        "| Fragment size | Stage | macro-F1 | docx-F1 | notes |",
        "|---|---|---|---|---|",
    ]
    for fs, r in results.items():
        configs = r["ablation"]["configs"]
        lines.append(f"| {fs}B | Backbone only | {configs['backbone_only']['macro_f1_mean']:.3f} "
                      f"| {configs['backbone_only']['docx_f1_mean']:.3f} | raw bytes, no C1 |")
        lines.append(f"| {fs}B | +C1 | {configs['+C1']['macro_f1_mean']:.3f} "
                      f"| {configs['+C1']['docx_f1_mean']:.3f} | residual-gated detect+decompress |")
        lines.append(f"| {fs}B | +C1+C2 | {configs['+C1']['macro_f1_mean']:.3f} (same) "
                      f"| {configs['+C1']['docx_f1_mean']:.3f} (same) | C2 adds calibration+attribution "
                      f"post-hoc; does not change the classification decision, so accuracy metrics are "
                      f"identical to +C1 by construction |")
        sig = r["ablation"]["significance_test"]
        if "p_value" in sig:
            lines.append(f"|  |  |  |  | paired t-test p={sig['p_value']:.4f}, "
                          f"effect size={sig['effect_size']:+.4f}, significant={sig['significant']} |")
        else:
            lines.append(f"|  |  |  |  | significance test not computed: {sig.get('error')} |")
    return "\n".join(lines)


def per_class_table(results: dict) -> str:
    lines = [
        "| Fragment size | Config | Class | Precision | Recall | F1 | Support (seed 0) |",
        "|---|---|---|---|---|---|---|",
    ]
    for fs, r in results.items():
        for config, cfg in r["ablation"]["configs"].items():
            run0 = cfg["runs"][0]
            for cls in LABELS:
                m = run0["per_class"][cls]
                lines.append(f"| {fs}B | {config} | {cls} | {m['precision']:.3f} | {m['recall']:.3f} "
                              f"| {m['f1']:.3f} | {m['support']} |")
    return "\n".join(lines)


def calibration_table(results: dict) -> str:
    lines = ["| Fragment size | Config | ECE before | ECE after | Temperature |", "|---|---|---|---|---|"]
    for fs, r in results.items():
        for name, calib in (("backbone_only", r["calib_backbone"]), ("+C1", r["calib_c1"])):
            lines.append(f"| {fs}B | {name} | {calib['ece_before']:.4f} | {calib['ece_after']:.4f} "
                          f"| {calib['temperature']:.4f} |")
    return "\n".join(lines)


def cost_table(results: dict) -> str:
    lines = [
        "| Fragment size | Config | params | est. FLOPs/fragment | throughput (fragments/s) |",
        "|---|---|---|---|---|",
    ]
    for fs, r in results.items():
        for name, calib in (("backbone_only", r["calib_backbone"]), ("+C1", r["calib_c1"])):
            lines.append(f"| {fs}B | {name} | {calib['n_params']:,} | {calib['flops_per_fragment']:,} "
                          f"| {calib['throughput']['fragments_per_sec']:.1f} |")
    lines.append("")
    lines.append("FLOPs are an analytic closed-form estimate for the transformer encoder body "
                  "(see classify/eval/flops.py), not profiler-measured. C1's encoder body is the "
                  "*same* transformer as backbone_only (single shared pass over the fused embedding, "
                  "design Q2/Q3) — the FLOPs gap above, if any, comes only from the small residual "
                  "gate and CNN refiner, not a second encoder pass.")
    return "\n".join(lines)


def plot_reliability(results: dict, out_path: str):
    fig, axes = plt.subplots(len(results), 2, figsize=(10, 4 * len(results)))
    axes = np.atleast_2d(axes)
    for row, (fs, r) in enumerate(results.items()):
        for col, (name, calib) in enumerate((("backbone_only", r["calib_backbone"]), ("+C1", r["calib_c1"]))):
            ax = axes[row][col]
            ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
            for conf, marker, label in ((calib["conf_before"], "o", "before"), (calib["conf_after"], "s", "after")):
                bin_edges = np.linspace(0, 1, 11)
                accs, confs = [], []
                for i in range(10):
                    mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1]) if i > 0 else (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
                    if mask.sum() == 0:
                        continue
                    accs.append(calib["correct"][mask].mean())
                    confs.append(conf[mask].mean())
                ax.plot(confs, accs, marker=marker, label=label)
            ax.set_title(f"{name} @ {fs}B")
            ax.set_xlabel("confidence")
            ax.set_ylabel("empirical accuracy")
            ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_macro_f1(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    fragment_sizes = list(results.keys())
    configs = ["backbone_only", "+C1"]
    width = 0.35
    x = np.arange(len(fragment_sizes))
    for i, config in enumerate(configs):
        means = [results[fs]["ablation"]["configs"][config]["macro_f1_mean"] for fs in fragment_sizes]
        stds = [results[fs]["ablation"]["configs"][config]["macro_f1_std"] for fs in fragment_sizes]
        ax.bar(x + i * width, means, width, yerr=stds, label=config, capsize=4)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"{fs}B" for fs in fragment_sizes])
    ax.set_ylabel("macro-F1 (mean ± std over seeds)")
    ax.set_title("Backbone-only vs +C1 (residual gating), by fragment size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(results: dict):
    sections = []
    sections.append("# New Result — DECoDE Classify Pipeline (post-redesign architecture)\n")
    sections.append("Scale note: see module docstring in `classify/eval/new_result_report.py` — "
                     "this is NOT the full CarveFormer-spec (256/8/6) backbone at full scale; "
                     "scaled down per fragment size to stay tractable on local hardware. "
                     "Real measurements on the actual corpus, not fabricated.\n")
    sections.append("## 6.1 Main Results (C1 vs. baseline, compound-class metrics)\n")
    sections.append(main_results_table(results) + "\n")
    sections.append("## 6.2 Ablation Study (C1 components, fragment size sensitivity)\n")
    sections.append(ablation_table(results) + "\n")
    sections.append("![macro-F1 by fragment size](macro_f1_bar.png)\n")
    sections.append("## 6.3 Defensibility Analysis (Attribution maps, calibration)\n")
    sections.append(calibration_table(results) + "\n")
    sections.append("![reliability diagrams](reliability_diagram.png)\n")
    sections.append("Attribution maps: see `classify/modules/heads/defensible_head.py` "
                     "(Integrated Gradients) and `classify/eval/deletion_curve.py` "
                     "(IG-vs-random deletion faithfulness check) — not run as part of this report "
                     "(separate, fragment-level visualization, not a corpus-level table); run "
                     "`deletion_accuracy_curve()` directly for a specific model/fragment sample.\n")
    sections.append("## 6.4 Open-Set Performance (if applicable)\n")
    sections.append("Deferred (design decision) — C4 not in scope for this report.\n")
    sections.append("## 6.5 Computational Cost Analysis\n")
    sections.append(cost_table(results) + "\n")
    sections.append("## Dataset statistics\n")
    sections.append(dataset_statistics_table(results) + "\n")
    sections.append("## Per-class breakdown (compound classes)\n")
    sections.append(per_class_table(results) + "\n")

    with open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(sections))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for fragment_size in (512, 4096):
        results[fragment_size] = run_size(fragment_size, device)

    plot_macro_f1(results, os.path.join(OUT_DIR, "macro_f1_bar.png"))
    plot_reliability(results, os.path.join(OUT_DIR, "reliability_diagram.png"))
    write_report(results)

    raw = {
        fs: {
            "ablation": r["ablation"],
            "calib_backbone": {k: v for k, v in r["calib_backbone"].items() if k not in ("conf_before", "conf_after", "correct")},
            "calib_c1": {k: v for k, v in r["calib_c1"].items() if k not in ("conf_before", "conf_after", "correct")},
        }
        for fs, r in results.items()
    }
    with open(os.path.join(OUT_DIR, "raw_results.json"), "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\nReport written to {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
