"""Measures how often structural evidence is actually *present* in a fragment,
per adapter and per class — the quantity that upper-bounds any
structure-aware front-end's achievable gain.

Why this is the decisive experiment
-----------------------------------
Chapters 4 and 5 report that C1 (OPC/DEFLATE) has no measurable effect on
macro-F1, and attribute it to partial decompression only succeeding at DEFLATE
stream starts. That explanation was mechanistic but never quantified at corpus
scale, and it was specific to compression.

This script quantifies it for both adapters. If structural evidence reaches
only a few percent of fragments, then *no* front-end built on that evidence can
move a corpus-level metric, however well engineered — the ceiling is the
coverage. Reporting that ceiling turns "our module didn't work" into "no module
of this kind can work under uniform fragment sampling", which is a far stronger
and more general claim, and it is measured rather than argued.

Coverage is reported alongside the false-positive rate on classes that are not
that adapter's container technology, because a detector that fires equally
often on the wrong classes contributes no discriminative signal even where it
does fire.

Output: results/structural_coverage.json + a markdown table on stdout.
"""

import collections
import json
import os

import numpy as np

from classify.data.fft75 import CLASS_TO_INDEX, load_fft75_arrays
from classify.modules.c1_compound import cfbf_adapter as cfbf
from classify.modules.c1_compound.adapters import OPCAdapter

OUT_DIR = "results"
SIZES = (512, 4096)
N_PER_CLASS = 3000

# Which container technology each class actually uses. The adapters are only
# *expected* to fire on their own family; everything else is a false positive.
CFBF_CLASSES = ("doc", "xls", "ppt")
OPC_CLASSES = ("docx", "xlsx", "pptx")
OTHER_CLASSES = ("zip", "pdf", "jpg")
ALL_CLASSES = CFBF_CLASSES + OPC_CLASSES + OTHER_CLASSES


def cfbf_coverage(fragment: bytes) -> tuple:
    """(has_structure, Counter of per-sector roles) for the CFBF adapter."""
    roles = [r for r in cfbf.sector_roles(fragment) if r != "stream_data"]
    return bool(roles), collections.Counter(roles)


def opc_coverage(fragment: bytes, adapter: OPCAdapter) -> bool:
    """True when partial DEFLATE decompression actually recovered something —
    i.e. the recovered view differs from the raw fragment."""
    return adapter.recover(fragment, len(fragment)) != fragment


def measure(size: int) -> dict:
    x, y = load_fft75_arrays(size, "test")
    opc = OPCAdapter()
    per_class = {}
    for cls in ALL_CLASSES:
        if cls not in CLASS_TO_INDEX:
            continue
        idx = np.flatnonzero(y == CLASS_TO_INDEX[cls])[:N_PER_CLASS]
        if idx.size == 0:
            continue
        cfbf_hits = 0
        opc_hits = 0
        roles = collections.Counter()
        for i in idx:
            frag = x[i].tobytes()
            has, r = cfbf_coverage(frag)
            cfbf_hits += int(has)
            roles.update(r)
            opc_hits += int(opc_coverage(frag, opc))
        per_class[cls] = {
            "n": int(idx.size),
            "cfbf_coverage": cfbf_hits / idx.size,
            "opc_coverage": opc_hits / idx.size,
            "cfbf_sector_roles": dict(roles),
        }
        print(f"  {cls:5} n={idx.size} cfbf={100 * cfbf_hits / idx.size:5.2f}% "
              f"opc={100 * opc_hits / idx.size:5.2f}% roles={dict(roles)}", flush=True)

    def mean_over(classes, key):
        vals = [per_class[c][key] for c in classes if c in per_class]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "per_class": per_class,
        "cfbf_coverage_on_cfbf_classes": mean_over(CFBF_CLASSES, "cfbf_coverage"),
        "cfbf_false_positive_rate": mean_over(OPC_CLASSES + OTHER_CLASSES, "cfbf_coverage"),
        "opc_coverage_on_opc_classes": mean_over(OPC_CLASSES, "opc_coverage"),
        "opc_false_positive_rate": mean_over(CFBF_CLASSES + OTHER_CLASSES, "opc_coverage"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"n_per_class": N_PER_CLASS, "sizes": {}}
    for size in SIZES:
        print(f"=== {size}B ===", flush=True)
        out["sizes"][str(size)] = measure(size)

    lines = ["", "## Structural coverage (share of fragments carrying decodable structure)", "",
             "| Fragment size | CFBF adapter on doc/xls/ppt | CFBF false-positive | "
             "OPC adapter on docx/xlsx/pptx | OPC false-positive |", "|---|---|---|---|---|"]
    for size in SIZES:
        s = out["sizes"][str(size)]
        lines.append(
            f"| {size}B | {s['cfbf_coverage_on_cfbf_classes']:.2%} | "
            f"{s['cfbf_false_positive_rate']:.2%} | "
            f"{s['opc_coverage_on_opc_classes']:.2%} | {s['opc_false_positive_rate']:.2%} |")
    table = "\n".join(lines)
    print(table)

    with open(os.path.join(OUT_DIR, "structural_coverage.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "structural_coverage.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nWrote results/structural_coverage.{json,md}")


if __name__ == "__main__":
    main()
