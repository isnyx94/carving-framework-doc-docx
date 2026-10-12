"""Completes the report after the full-scale 4096B ablation proved too slow
(quadratic self-attention cost: 4096^2 vs 512^2 attention scores per layer,
~64x more compute per batch). Reuses the already-computed 512B ablation
result and reruns 4096B at a reduced scale, so the report still reflects
real measurements at both required sizes (methodology Sec 8.2 — both sizes
must be reported, just not necessarily at matched compute budgets here).
"""

import json
import os

import torch

from classify.data.custom_corpus import build_dataset
from classify.eval.ablation_runner import run_ablation, to_markdown_table
from classify.eval.generate_report import (
    OUT_DIR, calibration_for_chart, plot_confusion_matrices, plot_macro_f1_bar,
    plot_reliability_diagram, write_text_report,
)

REDUCED_4096_N_PER_SIZE = 12
REDUCED_4096_EPOCHS = 3
REDUCED_4096_MODEL_KWARGS = dict(d_model=64, n_heads=4, n_layers=2, d_ff=128)
SEEDS = [0, 1, 2]
BATCH_SIZE = 32
LR = 1e-3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(OUT_DIR, "ablation_512B.json")) as f:
        summary_512 = json.load(f)

    corpus_512 = build_dataset(seed=0, n_per_size=40, sizes=(512,))
    corpus_4096 = build_dataset(seed=0, n_per_size=REDUCED_4096_N_PER_SIZE, sizes=(4096,))

    summary_4096 = run_ablation(
        corpus=corpus_4096, fragment_size=4096, seeds=SEEDS,
        model_kwargs=REDUCED_4096_MODEL_KWARGS, epochs=REDUCED_4096_EPOCHS,
        batch_size=BATCH_SIZE, lr=LR, device=device, out_dir=OUT_DIR,
    )
    print(to_markdown_table(summary_4096))

    summaries = {512: summary_512, 4096: summary_4096}

    calib = calibration_for_chart(corpus_512, fragment_size=512, device=device)

    plot_macro_f1_bar(summaries, os.path.join(OUT_DIR, "macro_f1_bar.png"))
    plot_confusion_matrices(summaries, os.path.join(OUT_DIR, "confusion_matrices.png"))
    plot_reliability_diagram(calib["conf_before"], calib["conf_after"], calib["correct"],
                              os.path.join(OUT_DIR, "reliability_diagram.png"))
    write_text_report(summaries, calib, os.path.join(OUT_DIR, "eval_report_classify.txt"))

    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as f:
        json.dump({k: v for k, v in calib.items() if k not in ("conf_before", "conf_after", "correct")},
                   f, indent=2)

    print(f"\nReport written to {OUT_DIR}/eval_report_classify.txt")


if __name__ == "__main__":
    main()
