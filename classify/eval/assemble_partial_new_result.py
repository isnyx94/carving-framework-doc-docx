"""Assembles results/new_result/REPORT.md from the completed 512B ablation
only — the calibration step (and 4096B entirely) was stopped after running
far longer than the ablation step itself, for reasons not fully diagnosed.
Throughput/FLOPs here use untrained models (architecture-only
properties, valid without training) so the cost section isn't left empty.
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
from classify.eval.flops import estimate_transformer_flops
from classify.eval.metrics import inference_throughput
from classify.eval.new_result_report import OUT_DIR, RUN_CONFIG
from classify.modules.c1_compound.dual_view_model import C1Classifier
from classify.training.dataset import LABELS, FragmentDataset

FRAGMENT_SIZE = 512


def dataset_statistics_table(corpus) -> str:
    lines = ["| Fragment size | Split | doc | docx | total | docx frac. deflate-labelled |",
             "|---|---|---|---|---|---|"]
    for split in ("train", "val", "test"):
        frags = [f for f in corpus[split] if f["fragment_size"] == FRAGMENT_SIZE]
        n_doc = sum(1 for f in frags if f["label"] == "doc")
        n_docx = sum(1 for f in frags if f["label"] == "docx")
        n_docx_deflate = sum(1 for f in frags if f["label"] == "docx" and f["is_deflate"])
        frac = n_docx_deflate / n_docx if n_docx else 0.0
        lines.append(f"| {FRAGMENT_SIZE}B | {split} | {n_doc} | {n_docx} | {n_doc + n_docx} | {frac:.2f} |")
    lines.append("")
    lines.append("Source corpus: 877 `.doc` + 175 `.docx` files (custom corpus, methodology §8.1). "
                  "FFT-75/govdocs1 not sourced. **4096B not run this pass — see note below.**")
    return "\n".join(lines)


def main_results_table(summary) -> str:
    lines = ["| Fragment size | Config | macro-F1 | docx-F1 | doc-F1 | params |", "|---|---|---|---|---|---|"]
    for config, cfg in summary["configs"].items():
        doc_f1 = np.mean([run["per_class"]["doc"]["f1"] for run in cfg["runs"]])
        lines.append(f"| {FRAGMENT_SIZE}B | {config} | {cfg['macro_f1_mean']:.3f}±{cfg['macro_f1_std']:.3f} "
                      f"| {cfg['docx_f1_mean']:.3f}±{cfg['docx_f1_std']:.3f} | {doc_f1:.3f} | {cfg['n_params']:,} |")
    lines.append(f"| 4096B | *(not run — stopped, see note)* | — | — | — | — |")
    return "\n".join(lines)


def ablation_table(summary) -> str:
    configs = summary["configs"]
    lines = ["| Stage | macro-F1 | docx-F1 | notes |", "|---|---|---|---|"]
    lines.append(f"| Backbone only | {configs['backbone_only']['macro_f1_mean']:.3f} "
                  f"| {configs['backbone_only']['docx_f1_mean']:.3f} | raw bytes, no C1 |")
    lines.append(f"| +C1 | {configs['+C1']['macro_f1_mean']:.3f} | {configs['+C1']['docx_f1_mean']:.3f} "
                  f"| residual-gated detect+decompress |")
    lines.append(f"| +C1+C2 | {configs['+C1']['macro_f1_mean']:.3f} (same) "
                  f"| {configs['+C1']['docx_f1_mean']:.3f} (same) "
                  f"| C2 (calibration+attribution) is post-hoc, does not change the decision |")
    sig = summary["significance_test"]
    if "p_value" in sig:
        lines.append(f"|  |  |  | paired t-test p={sig['p_value']:.4f}, effect size={sig['effect_size']:+.4f}, "
                      f"significant={sig['significant']} |")
    else:
        lines.append(f"|  |  |  | significance test not computed: {sig.get('error')} |")
    return "\n".join(lines)


def per_class_table(summary) -> str:
    lines = ["| Config | Class | Precision | Recall | F1 | Support (seed 0) |", "|---|---|---|---|---|---|"]
    for config, cfg in summary["configs"].items():
        run0 = cfg["runs"][0]
        for cls in LABELS:
            m = run0["per_class"][cls]
            lines.append(f"| {config} | {cls} | {m['precision']:.3f} | {m['recall']:.3f} "
                          f"| {m['f1']:.3f} | {m['support']} |")
    return "\n".join(lines)


def cost_table(summary, device) -> str:
    cfg = RUN_CONFIG[FRAGMENT_SIZE]["model_kwargs"]
    flops = estimate_transformer_flops(d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                                        d_ff=cfg["d_ff"], seq_len=FRAGMENT_SIZE)
    val_ds = FragmentDataset(build_dataset(seed=0, n_per_size=5, sizes=(FRAGMENT_SIZE,))["val"], FRAGMENT_SIZE)

    backbone = RawByteTransformerBaseline(n_classes=len(LABELS), max_len=FRAGMENT_SIZE, **cfg).to(device)
    c1 = C1Classifier(n_classes=len(LABELS), max_len=FRAGMENT_SIZE, **cfg).to(device)
    tp_backbone = inference_throughput(backbone, val_ds, device, batch_size=32)
    tp_c1 = inference_throughput(c1, val_ds, device, batch_size=32)

    lines = ["| Config | params | est. FLOPs/fragment | throughput (fragments/s, untrained-arch) |",
             "|---|---|---|---|"]
    for name, n_params, tp in (
        ("backbone_only", summary["configs"]["backbone_only"]["n_params"], tp_backbone),
        ("+C1", summary["configs"]["+C1"]["n_params"], tp_c1),
    ):
        lines.append(f"| {name} | {n_params:,} | {flops:,} | {tp['fragments_per_sec']:.1f} |")
    lines.append("")
    lines.append("Throughput measured on freshly-initialized (untrained) model instances — a valid "
                  "measure of architectural inference cost, since forward-pass latency doesn't depend "
                  "on learned weight values. FLOPs are analytic (classify/eval/flops.py), not "
                  "profiler-measured. **4096B cost not measured this pass.**")
    return "\n".join(lines)


def plot_macro_f1(summary, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    configs = list(summary["configs"].keys())
    means = [summary["configs"][c]["macro_f1_mean"] for c in configs]
    stds = [summary["configs"][c]["macro_f1_std"] for c in configs]
    ax.bar(configs, means, yerr=stds, capsize=4, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("macro-F1 (mean ± std over seeds)")
    ax.set_title(f"Backbone-only vs +C1 @ {FRAGMENT_SIZE}B")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(OUT_DIR, f"ablation_{FRAGMENT_SIZE}B.json")) as f:
        summary = json.load(f)

    cfg = RUN_CONFIG[FRAGMENT_SIZE]
    corpus = build_dataset(seed=0, n_per_size=cfg["n_per_size"], sizes=(FRAGMENT_SIZE,))

    plot_macro_f1(summary, os.path.join(OUT_DIR, "macro_f1_bar.png"))

    sections = [
        "# New Result — DECoDE Classify Pipeline (post-redesign architecture)\n",
        "**PARTIAL REPORT**: the calibration step (and the entire 4096B size) was stopped after "
        "running far longer than expected (>75 min past the 36-min 512B ablation, with no output "
        "and no fully diagnosed cause — see the project notes). This report covers "
        "512B ablation + architecture-only cost only. Backbone: d_model=128, n_heads=8, n_layers=4 "
        "(CarveFormer-direction, scaled down from the full 256/8/6 spec for compute-time reasons).\n",
        "**CRITICAL FINDING — total majority-class collapse, both configs**: both backbone_only "
        "and +C1 predict \"docx\" for 100% of test fragments (doc recall=0.000, docx recall=1.000), "
        "identical across all 3 seeds (std=0.000). This is the opposite collapse direction from the "
        "smaller 96-dim backbone in `results/classify_eval/` (which collapsed toward \"doc\"). Do "
        "NOT read macro-F1=0.146 here as a real C1-vs-baseline comparison — both models learned the "
        "trivial \"always predict the oversampled class\" solution. See the project notes for the "
        "likely cause (larger 128/8/4 backbone + learned positional embeddings may need a lower "
        "learning rate / more epochs / weaker oversampling than the 96-dim backbone did).\n",
        "## 6.1 Main Results (C1 vs. baseline, compound-class metrics)\n",
        main_results_table(summary) + "\n",
        "## 6.2 Ablation Study (C1 components, fragment size sensitivity)\n",
        ablation_table(summary) + "\n",
        "4096B not run this pass.\n",
        "![macro-F1](macro_f1_bar.png)\n",
        "## 6.3 Defensibility Analysis (Attribution maps, calibration)\n",
        "**Not computed this pass** (this is the step that was stopped). "
        "Integrated Gradients and deletion curves are implemented and unit-smoke-tested "
        "(classify/modules/heads/defensible_head.py, classify/eval/deletion_curve.py) but not run "
        "at corpus scale here.\n",
        "## 6.4 Open-Set Performance (if applicable)\n",
        "Deferred (design decision) — C4 not in scope.\n",
        "## 6.5 Computational Cost Analysis\n",
        cost_table(summary, device) + "\n",
        "## Dataset statistics\n",
        dataset_statistics_table(corpus) + "\n",
        "## Per-class breakdown (compound classes)\n",
        per_class_table(summary) + "\n",
    ]
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
    print(f"Partial report written to {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
