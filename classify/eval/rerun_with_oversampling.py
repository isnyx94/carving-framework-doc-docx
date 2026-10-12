"""Rerun after diagnosing majority-class collapse (see RESEARCH_LOG.md):
class-weighted loss alone wasn't enough, especially at 4096B. Adds
oversampling of the minority (docx) class in the train split
(classify.training.dataset.FragmentDataset oversample=True, wired into
ablation_runner.run_ablation) and more epochs.
"""

import json
import os

import torch

from classify.data.custom_corpus import build_dataset
from classify.eval.ablation_runner import run_ablation, to_markdown_table
from classify.eval.generate_report import (
    BATCH_SIZE, EPOCHS, LR, MODEL_KWARGS, N_PER_SIZE, OUT_DIR, SEEDS,
    calibration_for_chart, plot_confusion_matrices, plot_macro_f1_bar,
    plot_reliability_diagram, write_text_report,
)

REDUCED_4096_N_PER_SIZE = 12
REDUCED_4096_EPOCHS = 6
REDUCED_4096_MODEL_KWARGS = dict(d_model=64, n_heads=4, n_layers=2, d_ff=128)
REDUCED_4096_BATCH_SIZE = 32


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus_512 = build_dataset(seed=0, n_per_size=N_PER_SIZE, sizes=(512,))
    summary_512 = run_ablation(
        corpus=corpus_512, fragment_size=512, seeds=SEEDS,
        model_kwargs=MODEL_KWARGS, epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr=LR, device=device, out_dir=OUT_DIR,
    )
    print(to_markdown_table(summary_512))

    corpus_4096 = build_dataset(seed=0, n_per_size=REDUCED_4096_N_PER_SIZE, sizes=(4096,))
    summary_4096 = run_ablation(
        corpus=corpus_4096, fragment_size=4096, seeds=SEEDS,
        model_kwargs=REDUCED_4096_MODEL_KWARGS, epochs=REDUCED_4096_EPOCHS,
        batch_size=REDUCED_4096_BATCH_SIZE, lr=LR, device=device, out_dir=OUT_DIR,
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
