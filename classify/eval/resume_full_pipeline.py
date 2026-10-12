"""Resume the full C1-C4 pipeline after the 4096B stage was interrupted: loads the completed 512B checkpoint
(full_pipeline_512B.json) instead of re-training ~1h of 512B models, runs
the 4096B stage, then writes charts + Excel + FULL_REPORT.md for both.
"""

import json
import os

import torch

from classify.eval.full_pipeline import (
    OUT_DIR, load_tuning, make_charts, run_size, write_excel, write_report,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tuning = load_tuning()

    ckpt_512 = os.path.join(OUT_DIR, "full_pipeline_512B.json")
    with open(ckpt_512) as f:
        results = {512: json.load(f)}
    print(f"Loaded 512B checkpoint from {ckpt_512}", flush=True)

    results[4096] = run_size(4096, device, tuning)
    with open(os.path.join(OUT_DIR, "full_pipeline_4096B.json"), "w") as f:
        json.dump(results[4096], f, indent=2)

    make_charts(results, OUT_DIR)
    write_excel(results, OUT_DIR)
    write_report(results)
    print(f"\nDone: {OUT_DIR}/FULL_REPORT.md, FULL_RESULTS.xlsx, chart_*.png", flush=True)


if __name__ == "__main__":
    main()
