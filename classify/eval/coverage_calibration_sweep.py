"""Coverage calibration across fragment sizes — builds the screening curve.

Chapter 10 §10.2.4 proposes that the coverage-to-gain relationship is nearly linear in
the low-coverage regime, based on two points (512B and 4096B). Testing across more
fragment sizes converts this from a speculation into a calibrated relationship that
predicts achievable gain from coverage alone, before any model is trained.

This reuses the same formats and detectors as coverage_calibration.py, just sweeping
fragment size from 256B to 8192B in doubling steps. Occupancy-model predictions are
compared to measured header rates across the range.
"""

import json
import os
import re

import numpy as np

from classify.data.fft75 import CLASS_TO_INDEX, load_fft75_arrays
from classify.eval.coverage_calibration import (
    parse_lfh,
    scan_fragment,
    predict_occupancy,
    fold_error,
)

OUT_DIR = os.environ.get("COVCAL_SWEEP_OUT", "results/coverage_calibration_sweep")
SIZES = tuple(int(s) for s in os.environ.get("COVCAL_SWEEP_SIZES", "256,512,1024,2048,4096,8192").split(","))
N_PER_CLASS = int(os.environ.get("COVCAL_SWEEP_N", "1000"))

ZIP_FAMILY = ("docx", "xlsx", "pptx", "epub", "apk", "jar", "key", "zip")
PREDICTABLE = ("docx", "xlsx", "pptx", "epub", "apk", "jar", "key")


def measure(size: int) -> dict:
    """Measure coverage and header rate at a given fragment size."""
    x, y = load_fft75_arrays(size, "test")
    per_class = {}
    for cls in ZIP_FAMILY:
        if cls not in CLASS_TO_INDEX:
            continue
        idx = np.flatnonzero(y == CLASS_TO_INDEX[cls])[:N_PER_CLASS]
        if idx.size == 0:
            continue
        n_header = n_inflate = 0
        all_parts = []
        for i in idx:
            found, ok, parts = scan_fragment(x[i].tobytes())
            n_header += int(found)
            n_inflate += int(ok)
            all_parts.extend(parts)

        rec = {
            "n": int(idx.size),
            "header_rate": n_header / idx.size,
            "coverage": n_inflate / idx.size,
            "n_headers_observed": len(all_parts),
        }
        if all_parts:
            n_deflate = sum(1 for p in all_parts if p["method"] == 8)
            rec["deflate_fraction"] = n_deflate / len(all_parts)
            sized = [p["compressed_size"] for p in all_parts if p["sizes_trustworthy"]]
            if sized:
                arr = np.asarray(sized, dtype=np.float64)
                rec["n_parts_sized"] = int(arr.size)
                rec["mean_part_size"] = float(arr.mean())
                rec["cv_part_size"] = float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0
                rec["pred_occupancy"] = predict_occupancy(size, arr)
                if rec["header_rate"] > 0:
                    rec["occupancy_fold_error"] = fold_error(
                        rec["pred_occupancy"], rec["header_rate"]
                    )
        print(f"{size:5d}B  {cls:5s}  hdr={100*rec['header_rate']:6.2f}%  "
              f"cov={100*rec['coverage']:6.2f}%  parts={rec.get('n_parts_sized',0):4d}  "
              f"pred={100*rec.get('pred_occupancy',float('nan')):6.2f}%", flush=True)
        per_class[cls] = rec
    return per_class


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"sizes": list(SIZES), "by_size": {}}
    fold_errors = {}
    for size in SIZES:
        print(f"\n=== {size}B ===", flush=True)
        results["by_size"][str(size)] = measure(size)
        by_size = results["by_size"][str(size)]
        folds = [by_size[cls].get("occupancy_fold_error") for cls in PREDICTABLE
                 if cls in by_size and by_size[cls].get("n_parts_sized", 0) >= 30
                 and by_size[cls].get("occupancy_fold_error")]
        if folds:
            fold_errors[size] = float(np.median(folds))

    # Summary: does the fold error degrade at larger fragment sizes?
    L = ["# Coverage calibration across fragment sizes", "",
         "Sweep from 256B to 8192B measuring header rate and occupancy-model prediction.",
         "Does prediction accuracy (median fold error) hold across the range?", ""]
    if fold_errors:
        L += ["## Median fold error by fragment size", ""]
        L += [f"| Size | Median fold error |", "|---|---|"]
        for size in SIZES:
            if size in fold_errors:
                L.append(f"| {size}B | {fold_errors[size]:.2f}x |")
        L.append("")
        L += ["Fold error is the ratio of prediction to measured header rate. "
              "1.0x is exact."]
    L += ["## Interpretation", "",
          "The prediction model remains valid across fragment sizes, or begins to break "
          "down at the extremes? If calibration is stable (fold error ~1.42x throughout), "
          "coverage becomes a robust pre-training screening measure across the block-size "
          "range the thesis examines."]
    with open(os.path.join(OUT_DIR, "COVERAGE_SWEEP_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT_DIR, "coverage_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nWrote {OUT_DIR}/COVERAGE_SWEEP_REPORT.md")


if __name__ == "__main__":
    main()
