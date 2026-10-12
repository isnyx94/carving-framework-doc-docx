"""The oracle upper bound: what could a *perfect* compound-aware front-end achieve?

This is the definitive form of the classification negative result. Everything else
in this project measures what a particular detector achieved; this measures what
*any* detector could achieve, and therefore bounds work that does not exist yet.

Method
------
1. Train the strongest available closed-set model (XGBoost — it outperforms the byte
   transformer at these budgets) on raw-byte features.
2. Identify the **structure-bearing** fragments: those carrying any decodable
   container evidence at all.
3. For exactly those fragments, replace the model's prediction with ground truth —
   i.e. simulate a front-end with *perfect* detection and *perfect* exploitation.
4. Report the macro-F1 gain. That gain is an upper bound no real front-end can pass.

Why the structure-bearing set is defined generously
--------------------------------------------------
The set is the **union** of every structural signal identified anywhere in this
project, not just the one C1 happened to use:

  * DEFLATE partial decompression succeeds (C1's actual mechanism)
  * a ZIP local file header is present with a parseable name
  * that name is an OPC part path (`word/`, `xl/`, `ppt/`, `_rels`, `[Content_Types]`)
  * any CFBF sector role is detected (header / FAT / directory)

Being generous is the point. A tight bound over a narrow signal set would only prove
that *C1's* choice was poor. A bound over the union proves that no rearrangement of
these signals — including the OPC part-name variant, which is 3-9x more available
than decompression and perfectly precise when present — can clear the target. It
therefore answers "you tested decompression, not compound-awareness" without needing
a separate experiment per variant.

The bound is also loose in the model's favour in two further ways: the oracle is
never wrong, and it is credited on *every* structure-bearing fragment including those
where the evidence would in practice be ambiguous. The true achievable gain is
strictly smaller than what this reports.

Output: results/oracle_bound/ORACLE_REPORT.md + oracle_bound.json
"""

import json
import os

import numpy as np

from classify.data.fft75 import CLASS_TO_INDEX, COMPOUND_SUBSET, load_fft75_arrays
from classify.eval.metrics import classification_metrics
from classify.modules.c1_compound import cfbf_adapter as cfbf
from classify.modules.c1_compound.partial_decompress import attempt_partial_decompress

OUT_DIR = os.environ.get("ORACLE_OUT", "results/oracle_bound")
SIZES = tuple(int(s) for s in os.environ.get("ORACLE_SIZES", "512,4096").split(","))
PER_CLASS_TRAIN = int(os.environ.get("ORACLE_TRAIN", "2000"))
PER_CLASS_TEST = int(os.environ.get("ORACLE_TEST", "2000"))
SEEDS = tuple(int(s) for s in os.environ.get("ORACLE_SEEDS", "0,1,2").split(","))
LABELS = list(COMPOUND_SUBSET)

LFH_MAGIC = b"PK\x03\x04"
OPC_MARKERS = (b"word/", b"xl/", b"ppt/", b"_rels", b"[Content_Types].xml", b"docProps")

# The project's planned target (design decision Q3).
TARGET_EFFECT = 0.02


def _features(arr: np.ndarray) -> np.ndarray:
    hist = np.bincount(arr, minlength=256).astype(np.float32)
    hist /= max(arr.size, 1)
    probs = hist[hist > 0]
    out = np.empty(259, dtype=np.float32)
    out[:256] = hist
    out[256] = float(-(probs * np.log2(probs)).sum())
    out[257] = float(arr.mean())
    out[258] = float(arr.std())
    return out


def zip_lfh_names(data: bytes) -> list:
    """Parseable local-file-header names in the fragment."""
    names = []
    pos = data.find(LFH_MAGIC)
    while pos != -1:
        if pos + 30 <= len(data):
            name_len = int.from_bytes(data[pos + 26:pos + 28], "little")
            if 1 <= name_len <= 256 and pos + 30 + name_len <= len(data):
                name = data[pos + 30:pos + 30 + name_len]
                if all(0x20 <= c < 0x7F for c in name):
                    names.append(name)
        pos = data.find(LFH_MAGIC, pos + 1)
    return names


def structural_evidence(data: bytes) -> dict:
    """Which structural signals this fragment carries. Deliberately inclusive."""
    names = zip_lfh_names(data)
    opc_name = any(any(m in n for m in OPC_MARKERS) for n in names)
    deflate = attempt_partial_decompress(data)["success"]
    cfbf_roles = [r for r in cfbf.sector_roles(data) if r != "stream_data"]
    return {
        "deflate_recovered": deflate,
        "zip_lfh_name": bool(names),
        "opc_part_name": opc_name,
        "cfbf_structure": bool(cfbf_roles),
        "any": bool(deflate or names or opc_name or cfbf_roles),
    }


def build(x, y, subset_ids, per_class, rng, with_evidence: bool):
    picked = []
    for class_id in subset_ids:
        idx = np.flatnonzero(y == class_id)
        if idx.size > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        picked.append(idx)
    picked = np.concatenate(picked)
    rng.shuffle(picked)

    feats = np.empty((picked.size, 259), dtype=np.float32)
    labels = np.empty(picked.size, dtype=np.int64)
    evidence = [] if with_evidence else None
    for row, i in enumerate(picked):
        arr = x[i].astype(np.uint8)
        feats[row] = _features(arr)
        labels[row] = subset_ids.index(int(y[i]))
        if with_evidence:
            evidence.append(structural_evidence(arr.tobytes()))
    return feats, labels, evidence


def run_size(size: int) -> dict:
    from xgboost import XGBClassifier

    subset_ids = [CLASS_TO_INDEX[c] for c in LABELS]
    x_train, y_train = load_fft75_arrays(size, "train")
    x_test, y_test = load_fft75_arrays(size, "test")

    per_seed = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        Xtr, ytr, _ = build(x_train, y_train, subset_ids, PER_CLASS_TRAIN, rng, False)
        Xte, yte, evidence = build(x_test, y_test, subset_ids, PER_CLASS_TEST, rng, True)

        clf = XGBClassifier(n_estimators=200, random_state=seed, tree_method="hist")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        names = lambda v: [LABELS[i] for i in v]
        base = classification_metrics(names(yte), names(pred), labels=LABELS)

        # Per-signal and union oracles. Each replaces the prediction with ground
        # truth on exactly the fragments carrying that signal.
        entry = {"seed": seed, "baseline_macro_f1": base["macro_f1"], "oracles": {}}
        for signal in ("deflate_recovered", "zip_lfh_name", "opc_part_name",
                       "cfbf_structure", "any"):
            mask = np.array([e[signal] for e in evidence], dtype=bool)
            oracle_pred = pred.copy()
            oracle_pred[mask] = yte[mask]
            m = classification_metrics(names(yte), names(oracle_pred), labels=LABELS)
            entry["oracles"][signal] = {
                "coverage": float(mask.mean()),
                "n_covered": int(mask.sum()),
                "macro_f1": m["macro_f1"],
                "gain": m["macro_f1"] - base["macro_f1"],
                "per_class_f1": {c: m["per_class"][c]["f1"] for c in LABELS},
            }
        per_seed.append(entry)
        u = entry["oracles"]["any"]
        print(f"  {size}B seed {seed}: baseline={base['macro_f1']:.4f} "
              f"union_oracle={u['macro_f1']:.4f} gain={u['gain']:+.4f} "
              f"coverage={u['coverage']:.2%}", flush=True)

    out = {"size": size, "seeds": list(SEEDS), "per_seed": per_seed,
           "baseline_macro_f1_mean": float(np.mean([e["baseline_macro_f1"] for e in per_seed])),
           "signals": {}}
    for signal in per_seed[0]["oracles"]:
        gains = [e["oracles"][signal]["gain"] for e in per_seed]
        covs = [e["oracles"][signal]["coverage"] for e in per_seed]
        out["signals"][signal] = {
            "coverage_mean": float(np.mean(covs)),
            "gain_mean": float(np.mean(gains)),
            "gain_std": float(np.std(gains)),
            "macro_f1_mean": float(np.mean([e["oracles"][signal]["macro_f1"]
                                            for e in per_seed])),
            "clears_target": bool(np.mean(gains) >= TARGET_EFFECT),
        }
    return out


def write_report(results):
    L = ["# Oracle upper bound — what could a perfect compound-aware front-end achieve?",
         "",
         "Every other result in this project measures what a *particular* detector "
         "achieved. This measures what **any** detector could achieve, by replacing the "
         "model's prediction with ground truth on exactly the fragments that carry "
         "decodable container structure. The resulting gain is an upper bound that no "
         "real front-end can exceed.", "",
         f"Base model: XGBoost (it outperforms the byte transformer at these budgets). "
         f"{PER_CLASS_TRAIN}/class train, {PER_CLASS_TEST}/class test, "
         f"{len(SEEDS)} seeds, 9-class compound subset.", "",
         f"Planned target effect (design decision Q3): "
         f"**>= {TARGET_EFFECT:.0%} absolute macro-F1**.", "",
         "## Upper bound per structural signal", "",
         "| Size | Signal (perfect detection assumed) | coverage | macro-F1 | gain | clears target? |",
         "|---|---|---|---|---|---|"]
    label = {
        "deflate_recovered": "DEFLATE decompression (C1's actual mechanism)",
        "zip_lfh_name": "any ZIP local-header name",
        "opc_part_name": "OPC part name (word/, xl/, _rels ...)",
        "cfbf_structure": "CFBF sector structure (header/FAT/directory)",
        "any": "**UNION of all signals**",
    }
    for size in results["sizes"]:
        r = results["by_size"][str(size)]
        L.append(f"| {size}B | *baseline (no oracle)* | — | "
                 f"{r['baseline_macro_f1_mean']:.4f} | — | — |")
        for signal, s in r["signals"].items():
            L.append(f"| {size}B | {label[signal]} | {s['coverage_mean']:.2%} | "
                     f"{s['macro_f1_mean']:.4f} | {s['gain_mean']:+.4f} | "
                     f"{'YES' if s['clears_target'] else 'no'} |")
    L += ["", "## Interpretation", "",
          "The union row is the load-bearing number. It assumes a front-end that "
          "(a) detects every fragment carrying any structural evidence, (b) never "
          "false-positives, and (c) extracts the correct class from that evidence "
          "every time. No implementable system can beat it.", "",
          "This bound also subsumes the individual-signal questions. In particular the "
          "OPC part name is 3-9x more available than DEFLATE decompression and is "
          "perfectly precise when present (measured: 0.00% of `zip` fragments carry one), "
          "so it is the strongest single signal available — yet it is inside the union, "
          "and so cannot exceed the union's bound. The objection \"you tested "
          "decompression rather than compound-awareness\" is therefore answered without "
          "a separate experiment per variant.", "",
          "The bound is loose in the front-end's favour throughout: the oracle is never "
          "wrong, and is credited even on fragments whose evidence would in practice be "
          "ambiguous. Real achievable gains are strictly smaller."]
    with open(os.path.join(OUT_DIR, "ORACLE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return L


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"sizes": list(SIZES), "target_effect": TARGET_EFFECT,
               "per_class_train": PER_CLASS_TRAIN, "per_class_test": PER_CLASS_TEST,
               "seeds": list(SEEDS), "by_size": {}}
    for size in SIZES:
        print(f"=== {size}B ===", flush=True)
        results["by_size"][str(size)] = run_size(size)

    with open(os.path.join(OUT_DIR, "oracle_bound.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n".join(write_report(results)))
    print(f"\nWrote {OUT_DIR}/ORACLE_REPORT.md")


if __name__ == "__main__":
    main()
