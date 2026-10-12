"""Assembles the final report/charts from already-completed ablation runs
(results/classify_eval/ablation_512B.json, ablation_4096B.json) plus a fresh
calibration pass — avoids rerunning the slow training loops.
"""

import json
import os

import torch

from classify.data.custom_corpus import build_dataset
from classify.eval.generate_report import (
    OUT_DIR, calibration_for_chart, plot_confusion_matrices, plot_macro_f1_bar,
    plot_reliability_diagram, write_text_report,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    summaries = {}
    for fs in (512, 4096):
        with open(os.path.join(OUT_DIR, f"ablation_{fs}B.json")) as f:
            summaries[fs] = json.load(f)

    corpus_512 = build_dataset(seed=0, n_per_size=40, sizes=(512,))
    calib = calibration_for_chart(corpus_512, fragment_size=512, device=device)

    plot_macro_f1_bar(summaries, os.path.join(OUT_DIR, "macro_f1_bar.png"))
    plot_confusion_matrices(summaries, os.path.join(OUT_DIR, "confusion_matrices.png"))
    plot_reliability_diagram(calib["conf_before"], calib["conf_after"], calib["correct"],
                              os.path.join(OUT_DIR, "reliability_diagram.png"))
    write_text_report(summaries, calib, os.path.join(OUT_DIR, "eval_report_classify.txt"))

    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as f:
        json.dump({k: v for k, v in calib.items() if k not in ("conf_before", "conf_after", "correct")},
                   f, indent=2)

    print(f"Report written to {OUT_DIR}/eval_report_classify.txt")
    print(f"Charts written to {OUT_DIR}/*.png")


if __name__ == "__main__":
    main()
