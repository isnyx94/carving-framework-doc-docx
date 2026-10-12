"""Does the CFBF structural decode add discriminative signal for fragment
classification? A direct test of the coverage-ceiling prediction.

`classify/eval/structural_coverage.py` measured that decodable CFBF structure is
present in only 0.94% (512B) to 1.61% (4096B) of fragments. That bounds any
front-end built on it far below the project's pre-registered >=2% absolute
macro-F1 target, so the prediction is that the CFBF adapter replicates the C1
null. A prediction stated but never tested is an untested artifact, and an
examiner is entitled to ask why the adapter is in the thesis at all — so it is
tested here.

Why XGBoost rather than the transformer
--------------------------------------
Two reasons, both practical and defensible. XGBoost *outperforms* the byte
transformer at these training budgets (0.630 vs 0.596 at 512B; 0.703 vs ~0.52 at
4096B — RESEARCH_LOG 2026-07-06 and results/ch5/classical_4096B.json), so it is
the stronger model to test an information claim against, not a weaker
substitute. And it runs on CPU in minutes, leaving the GPU to the Chapter 5 run.

The question here is *informational*: if the decoded view carries discriminative
signal the raw view lacks, a gradient-boosted model given features from both
should beat one given features from the raw view alone. If it does not, the
signal is not there to be exploited — which is a statement about the data, not
about any particular architecture.

Conditions (paired over seeds, identical splits):
  raw            — unigram/entropy features of the raw fragment
  raw + cfbf     — the same features, concatenated with those of the
                   CFBF-decoded view (`cfbf_adapter.cfbf_recovered_view`)

Output: results/cfbf_ablation/CFBF_ABLATION_REPORT.md + cfbf_ablation.json
"""

import json
import os

import numpy as np
from scipy.stats import ttest_rel

from classify.data.fft75 import CLASS_TO_INDEX, COMPOUND_SUBSET, load_fft75_arrays
from classify.eval.metrics import classification_metrics
from classify.modules.c1_compound import cfbf_adapter as cfbf

OUT_DIR = os.environ.get("CFBF_ABL_OUT", "results/cfbf_ablation")
SIZES = tuple(int(s) for s in os.environ.get("CFBF_ABL_SIZES", "512,4096").split(","))
PER_CLASS_TRAIN = int(os.environ.get("CFBF_ABL_TRAIN", "2000"))
PER_CLASS_TEST = int(os.environ.get("CFBF_ABL_TEST", "800"))
SEEDS = tuple(int(s) for s in os.environ.get("CFBF_ABL_SEEDS", "0,1,2").split(","))
LABELS = list(COMPOUND_SUBSET)
CFBF_CLASSES = ("doc", "xls", "ppt")


def _features(arr: np.ndarray) -> np.ndarray:
    """unigram-256 + entropy + mean/std — identical to fullspec_512._classical_features
    so numbers stay comparable with the existing baseline tables."""
    hist = np.bincount(arr, minlength=256).astype(np.float32)
    hist /= max(arr.size, 1)
    probs = hist[hist > 0]
    out = np.empty(259, dtype=np.float32)
    out[:256] = hist
    out[256] = float(-(probs * np.log2(probs)).sum())
    out[257] = float(arr.mean())
    out[258] = float(arr.std())
    return out


def build_matrices(x, y, subset_ids, per_class, rng) -> tuple:
    """Returns (raw_features, cfbf_features, labels). The CFBF block is computed
    from the decoded view of the same fragment, so the two conditions differ only
    in the extra columns."""
    picked = []
    for class_id in subset_ids:
        idx = np.flatnonzero(y == class_id)
        if idx.size > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        picked.append(idx)
    picked = np.concatenate(picked)
    rng.shuffle(picked)

    raw = np.empty((picked.size, 259), dtype=np.float32)
    dec = np.empty((picked.size, 259), dtype=np.float32)
    labels = np.empty(picked.size, dtype=np.int64)
    for row, i in enumerate(picked):
        arr = x[i].astype(np.uint8)
        raw[row] = _features(arr)
        decoded = cfbf.cfbf_recovered_view(arr.tobytes(), arr.size)
        dec[row] = _features(np.frombuffer(decoded, dtype=np.uint8))
        labels[row] = subset_ids.index(int(y[i]))
    return raw, dec, labels


def run_size(size: int) -> dict:
    from xgboost import XGBClassifier

    subset_ids = [CLASS_TO_INDEX[c] for c in LABELS]
    x_train, y_train = load_fft75_arrays(size, "train")
    x_test, y_test = load_fft75_arrays(size, "test")

    per_seed = {"raw": [], "raw+cfbf": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tr_raw, tr_dec, ytr = build_matrices(x_train, y_train, subset_ids,
                                             PER_CLASS_TRAIN, rng)
        te_raw, te_dec, yte = build_matrices(x_test, y_test, subset_ids,
                                            PER_CLASS_TEST, rng)
        conditions = {
            "raw": (tr_raw, te_raw),
            "raw+cfbf": (np.hstack([tr_raw, tr_dec]), np.hstack([te_raw, te_dec])),
        }
        for name, (Xtr, Xte) in conditions.items():
            clf = XGBClassifier(n_estimators=200, random_state=seed, tree_method="hist")
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            m = classification_metrics([LABELS[i] for i in yte],
                                       [LABELS[i] for i in pred], labels=LABELS)
            m["seed"] = seed
            per_seed[name].append(m)
        print(f"  {size}B seed {seed}: "
              + " | ".join(f"{n}={per_seed[n][-1]['macro_f1']:.4f}" for n in conditions),
              flush=True)

    out = {"size": size, "seeds": list(SEEDS), "conditions": {}}
    for name, runs in per_seed.items():
        out["conditions"][name] = {
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])),
            "per_class_f1": {c: float(np.mean([r["per_class"][c]["f1"] for r in runs]))
                             for c in LABELS},
            "runs": runs,
        }
    a = [r["macro_f1"] for r in per_seed["raw"]]
    b = [r["macro_f1"] for r in per_seed["raw+cfbf"]]
    diffs = np.array(b) - np.array(a)
    sig = {"effect_size": float(diffs.mean())}
    if not np.allclose(diffs, diffs[0]):
        t, p = ttest_rel(b, a)
        sig.update({"t": float(t), "p_value": float(p), "significant": bool(p < 0.05)})
    out["significance_macro_f1"] = sig

    # The targeted question: does it help the CFBF classes specifically? A
    # corpus-level null could in principle hide a gain concentrated on doc/xls/ppt.
    cfbf_gain = {}
    for c in CFBF_CLASSES:
        av = np.mean([r["per_class"][c]["f1"] for r in per_seed["raw"]])
        bv = np.mean([r["per_class"][c]["f1"] for r in per_seed["raw+cfbf"]])
        cfbf_gain[c] = {"raw": float(av), "raw+cfbf": float(bv), "delta": float(bv - av)}
    out["cfbf_class_gain"] = cfbf_gain
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"per_class_train": PER_CLASS_TRAIN, "per_class_test": PER_CLASS_TEST,
               "seeds": list(SEEDS), "sizes": list(SIZES), "by_size": {}}
    for size in SIZES:
        print(f"=== {size}B ===", flush=True)
        results["by_size"][str(size)] = run_size(size)

    L = ["# CFBF adapter ablation — does the structural decode add signal?", "",
         "Tests the coverage-ceiling prediction: structural evidence reaches only "
         "0.94-1.61% of fragments, which should make the CFBF adapter a null for "
         "classification. XGBoost is used because it *outperforms* the byte "
         "transformer at these budgets, so this is the stronger model to test an "
         "informational claim against.", "",
         f"{PER_CLASS_TRAIN}/class train, {PER_CLASS_TEST}/class test, "
         f"{len(SEEDS)} seeds, 9-class compound subset.", "",
         "| Size | Condition | macro-F1 | effect | p | significant |",
         "|---|---|---|---|---|---|"]
    for size in SIZES:
        r = results["by_size"][str(size)]
        sig = r["significance_macro_f1"]
        for name in ("raw", "raw+cfbf"):
            c = r["conditions"][name]
            extra = (f"{sig['effect_size']:+.4f} | {sig.get('p_value', float('nan')):.4g} | "
                     f"{sig.get('significant', 'n/a')}") if name == "raw+cfbf" else " | | "
            L.append(f"| {size}B | {name} | {c['macro_f1_mean']:.4f}"
                     f"±{c['macro_f1_std']:.4f} | {extra} |")
    L += ["", "## Per-class F1 on the CFBF classes (the targeted metric)", "",
          "A corpus-level null could hide a gain concentrated on doc/xls/ppt, so those "
          "classes are reported separately.", "",
          "| Size | Class | raw | raw+cfbf | delta |", "|---|---|---|---|---|"]
    for size in SIZES:
        for c, g in results["by_size"][str(size)]["cfbf_class_gain"].items():
            L.append(f"| {size}B | {c} | {g['raw']:.4f} | {g['raw+cfbf']:.4f} | "
                     f"{g['delta']:+.4f} |")
    with open(os.path.join(OUT_DIR, "CFBF_ABLATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT_DIR, "cfbf_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n".join(L))
    print(f"\nWrote {OUT_DIR}/CFBF_ABLATION_REPORT.md")


if __name__ == "__main__":
    main()
