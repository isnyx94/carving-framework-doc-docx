"""Full-spec 512B run on FFT-75 compound subset.

Stage A — classical baselines (methodology Sec 9.1 items 2-3): Random
          Forest + XGBoost on 259-dim features (unigram byte histogram +
          entropy + mean/std). Bigram histograms (65,536-dim) are
          infeasible at 180k fragments on this machine; the feature choice
          is recorded in the output.
Stage B — CarveFormer-spec backbone (d_model=256, 8 heads, 6 layers,
          d_ff=1024; design Q2) vs +C1, **5 seeds** (design Q3), bf16 AMP.
Stage C — C2 (calibration ECE + IG deletion curve) and C4 (MSP abstention
          vs OOD pdf/jpeg/random) on the seed-0 full-spec models.

All prints ASCII (cp1252 console). Per-stage JSON checkpoints.
Outputs: results/new_result/fullspec_512B.json + FULLSPEC_REPORT_512B.md.
"""

import json
import os

import numpy as np
import torch
from scipy.stats import ttest_rel

from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
from classify.data.custom_corpus import build_dataset as build_custom_corpus
from classify.data.fft75 import CLASS_TO_INDEX, COMPOUND_SUBSET, load_fft75_arrays
from classify.data.ood_corpus import OODFragmentDataset
from classify.eval.deletion_curve import deletion_accuracy_curve
from classify.eval.metrics import classification_metrics, docx_zip_confusion_rate, param_count
from classify.eval.p1_gate import predict_names
from classify.modules.c1_compound.dual_view_model import C1Classifier
from classify.modules.c1_compound.train_refiner import train_decompression_confidence_cnn
from classify.modules.heads.defensible_head import TemperatureScaler, expected_calibration_error
from classify.modules.heads.openset_head import auroc_id_vs_ood, msp_scores, rejection_curve
from classify.training.numpy_dataset import NumpyFragmentDataset
from classify.training.train import train_classifier

OUT_DIR = "results/new_result"
FRAGMENT_SIZE = 512
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 3
BATCH = 64
LR = 1e-4
TRAIN_PER_CLASS = 20000
EVAL_PER_CLASS = 5000
MODEL_KWARGS = dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024)
LABELS = list(COMPOUND_SUBSET)
DELETION_N_PER_CLASS = 25


def _datasets():
    subset_ids = [CLASS_TO_INDEX[c] for c in COMPOUND_SUBSET]
    x_train, y_train = load_fft75_arrays(FRAGMENT_SIZE, "train")
    x_val, y_val = load_fft75_arrays(FRAGMENT_SIZE, "val")
    x_test, y_test = load_fft75_arrays(FRAGMENT_SIZE, "test")
    return (NumpyFragmentDataset(x_train, y_train, subset_ids, TRAIN_PER_CLASS),
            NumpyFragmentDataset(x_val, y_val, subset_ids, EVAL_PER_CLASS),
            NumpyFragmentDataset(x_test, y_test, subset_ids, EVAL_PER_CLASS))


# ---------------------------------------------------------------- stage A --

def _classical_features(dataset) -> tuple:
    n = len(dataset)
    feats = np.empty((n, 259), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        byte_ids, label = dataset[i]
        arr = byte_ids.numpy().astype(np.uint8)
        hist = np.bincount(arr, minlength=256).astype(np.float32)
        hist /= max(len(arr), 1)
        probs = hist[hist > 0]
        entropy = float(-(probs * np.log2(probs)).sum())
        feats[i, :256] = hist
        feats[i, 256] = entropy
        feats[i, 257] = float(arr.mean())
        feats[i, 258] = float(arr.std())
        labels[i] = int(label)
    return feats, labels


def stage_a(train_ds, test_ds) -> dict:
    print("=== STAGE A: classical baselines (RF, XGBoost) ===", flush=True)
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    Xtr, ytr = _classical_features(train_ds)
    Xte, yte = _classical_features(test_ds)
    print(f"features: train {Xtr.shape}, test {Xte.shape}", flush=True)

    out = {"feature_note": "unigram-256 + entropy + mean/std (bigram infeasible at this N)"}
    for name, clf in (
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
        ("xgboost", XGBClassifier(n_estimators=200, random_state=0, tree_method="hist")),
    ):
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        y_true = [LABELS[i] for i in yte]
        y_pred = [LABELS[i] for i in pred]
        m = classification_metrics(y_true, y_pred, labels=LABELS)
        m["docx_zip_confusion_rate"] = docx_zip_confusion_rate(y_true, y_pred, LABELS)
        out[name] = m
        print(f"  {name}: macro-F1={m['macro_f1']:.4f} docx-F1={m['per_class']['docx']['f1']:.4f} "
              f"docx->zip={m['docx_zip_confusion_rate']:.4f}", flush=True)
    return out


# ---------------------------------------------------------------- stage B --

def stage_b(train_ds, val_ds, test_ds, refiner, device) -> dict:
    print("=== STAGE B: full-spec (256/8/6) backbone vs +C1, 5 seeds ===", flush=True)
    builders = {
        "backbone_only": lambda: RawByteTransformerBaseline(
            n_classes=len(LABELS), max_len=FRAGMENT_SIZE, **MODEL_KWARGS),
        "+C1": lambda: C1Classifier(
            n_classes=len(LABELS), max_len=FRAGMENT_SIZE,
            pretrained_refiner=refiner, **MODEL_KWARGS),
    }
    results = {name: [] for name in builders}
    for name, builder in builders.items():
        for seed in SEEDS:
            print(f"--- {name} seed {seed} ---", flush=True)
            model = builder()
            train_classifier(model, train_ds, val_ds, epochs=EPOCHS, batch_size=BATCH,
                              lr=LR, seed=seed, device=device, amp=True)
            y_true, y_pred = predict_names(model, test_ds, device, LABELS)
            m = classification_metrics(y_true, y_pred, labels=LABELS)
            m["docx_zip_confusion_rate"] = docx_zip_confusion_rate(y_true, y_pred, LABELS)
            m["seed"] = seed
            m["n_params"] = param_count(model)
            results[name].append(m)
            print(f"    macro-F1={m['macro_f1']:.4f} docx-F1={m['per_class']['docx']['f1']:.4f} "
                  f"docx->zip={m['docx_zip_confusion_rate']:.4f}", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            with open(os.path.join(OUT_DIR, "fullspec_512B_partial.json"), "w") as f:
                json.dump(results, f, indent=2)

    summary = {"seeds": SEEDS, "configs": {}}
    for name, runs in results.items():
        summary["configs"][name] = {
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])),
            "docx_f1_mean": float(np.mean([r["per_class"]["docx"]["f1"] for r in runs])),
            "docx_zip_confusion_mean": float(np.mean([r["docx_zip_confusion_rate"] for r in runs])),
            "n_params": runs[0]["n_params"],
            "runs": runs,
        }
    b = [r["macro_f1"] for r in results["backbone_only"]]
    c = [r["macro_f1"] for r in results["+C1"]]
    diffs = np.array(c) - np.array(b)
    summary["significance"] = {"effect_size": float(diffs.mean())}
    if not np.allclose(diffs, diffs[0]):
        stat, p = ttest_rel(c, b)
        summary["significance"].update({"test": "paired_t", "p_value": float(p),
                                          "significant": bool(p < 0.05)})
    return summary


# ---------------------------------------------------------------- stage C --

def stage_c(train_ds, val_ds, test_ds, refiner, device) -> dict:
    print("=== STAGE C: C2 + C4 on seed-0 full-spec models ===", flush=True)
    from torch.utils.data import DataLoader

    out = {}
    for name in ("backbone_only", "+C1"):
        if name == "backbone_only":
            model = RawByteTransformerBaseline(n_classes=len(LABELS), max_len=FRAGMENT_SIZE, **MODEL_KWARGS)
        else:
            model = C1Classifier(n_classes=len(LABELS), max_len=FRAGMENT_SIZE,
                                  pretrained_refiner=refiner, **MODEL_KWARGS)
        train_classifier(model, train_ds, val_ds, epochs=EPOCHS, batch_size=BATCH,
                          lr=LR, seed=0, device=device, amp=True)

        loader = DataLoader(val_ds, batch_size=256)
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits_list.append(model(xb.to(device)))
                labels_list.append(yb.to(device))
        logits, labels_t = torch.cat(logits_list), torch.cat(labels_list)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct = (pred == labels_t)
        ece_before = expected_calibration_error(conf, correct)
        scaler = TemperatureScaler().to(device)
        scaler.fit(logits, labels_t)
        conf_after, _ = torch.softmax(scaler(logits), dim=1).max(dim=1)
        ece_after = expected_calibration_error(conf_after, correct)

        ood_ds = OODFragmentDataset(fragment_size=FRAGMENT_SIZE)
        id_conf, id_pred, id_labels = msp_scores(model, test_ds, device, temperature_scaler=scaler)
        ood_conf, _, _ = msp_scores(model, ood_ds, device, temperature_scaler=scaler)

        entry = {
            "ece_before": ece_before, "ece_after": ece_after,
            "temperature": scaler.log_temperature.exp().item(),
            "c4_auroc": auroc_id_vs_ood(id_conf, ood_conf),
            "c4_rejection_curve": rejection_curve(id_conf, id_pred, id_labels, ood_conf),
        }
        if name == "+C1":
            print("    deletion curve (IG faithfulness)...", flush=True)
            entry["deletion_curve"] = deletion_accuracy_curve(
                model, test_ds, LABELS, device, n_per_class=DELETION_N_PER_CLASS)
        out[name] = entry
        print(f"  {name}: ECE {ece_before:.4f}->{ece_after:.4f}, C4 AUROC={entry['c4_auroc']:.4f}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def write_report(a: dict, b: dict, c: dict):
    L = ["# Full-spec 512B report — FFT-75 compound subset (9 classes)\n"]
    L.append(f"Backbone: 256/8/6/1024 (CarveFormer-spec, design Q2); {len(SEEDS)} seeds; "
             f"{TRAIN_PER_CLASS}/class train; lr={LR}; bf16 AMP.\n")
    L.append("## Main results (all models)\n")
    L.append("| Model | macro-F1 | docx-F1 | docx->zip |")
    L.append("|---|---|---|---|")
    for name in ("random_forest", "xgboost"):
        m = a[name]
        L.append(f"| {name} | {m['macro_f1']:.4f} | {m['per_class']['docx']['f1']:.4f} "
                  f"| {m['docx_zip_confusion_rate']:.4f} |")
    for name, cfg in b["configs"].items():
        L.append(f"| {name} (5 seeds) | {cfg['macro_f1_mean']:.4f}±{cfg['macro_f1_std']:.4f} "
                  f"| {cfg['docx_f1_mean']:.4f} | {cfg['docx_zip_confusion_mean']:.4f} |")
    sig = b["significance"]
    L.append(f"\nPaired t-test (macro-F1, +C1 vs backbone): effect={sig['effect_size']:+.4f}"
             + (f", p={sig['p_value']:.4f}, significant={sig['significant']}" if "p_value" in sig else ""))
    L.append(f"\nClassical-baseline features: {a['feature_note']}\n")
    L.append("## C2 / C4 (seed-0 models)\n")
    L.append("| Model | ECE before | ECE after | C4 AUROC |")
    L.append("|---|---|---|---|")
    for name, e in c.items():
        L.append(f"| {name} | {e['ece_before']:.4f} | {e['ece_after']:.4f} | {e['c4_auroc']:.4f} |")
    dc = c.get("+C1", {}).get("deletion_curve")
    if dc:
        L.append("\nDeletion curve (+C1, IG vs random):\n")
        L.append("| fraction | acc (IG) | acc (random) |")
        L.append("|---|---|---|")
        for f, ig, rnd in zip(dc["fractions"], dc["ig"], dc["random"]):
            L.append(f"| {f:.1f} | {ig:.3f} | {rnd:.3f} |")
    with open(os.path.join(OUT_DIR, "FULLSPEC_REPORT_512B.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds = _datasets()
    print(f"datasets: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}", flush=True)

    a = stage_a(train_ds, test_ds)
    with open(os.path.join(OUT_DIR, "fullspec_512B.json"), "w") as f:
        json.dump({"stage_a": a}, f, indent=2)

    custom = build_custom_corpus(seed=0, n_per_size=15, sizes=(FRAGMENT_SIZE,))
    refiner = train_decompression_confidence_cnn(
        custom["train"], fragment_size=FRAGMENT_SIZE, epochs=3, batch_size=64, device=device)

    b = stage_b(train_ds, val_ds, test_ds, refiner, device)
    with open(os.path.join(OUT_DIR, "fullspec_512B.json"), "w") as f:
        json.dump({"stage_a": a, "stage_b": b}, f, indent=2)

    c = stage_c(train_ds, val_ds, test_ds, refiner, device)
    with open(os.path.join(OUT_DIR, "fullspec_512B.json"), "w") as f:
        json.dump({"stage_a": a, "stage_b": b, "stage_c": c}, f, indent=2)

    write_report(a, b, c)
    print("\nDone: FULLSPEC_REPORT_512B.md + fullspec_512B.json", flush=True)


if __name__ == "__main__":
    main()
