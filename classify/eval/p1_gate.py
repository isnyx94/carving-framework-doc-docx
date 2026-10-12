"""P1 gate + first FFT-75 compound run (512B).

Stage 1 — GATE (methodology Sec 7 P1): FiFTy-style CNN trained on the full
75-class scenario-1 train split; test accuracy compared against FiFTy's
published scenario-1/512B accuracy of 65.6% (Mittal et al., TIFS). The
architecture is FiFTy-*style*, not a byte-exact replica of their TPE-tuned
model, so the gate passes within a ±5-point band. Later claims are relative
to baselines trained under our protocol, so the band's width is a
comparability statement, not a fudge.

Stage 2 — COMPOUND: backbone_only vs +C1 (full) on the 9-class compound
subset (doc/docx/xls/xlsx/ppt/pptx/zip/pdf/jpg), 3 seeds, per-class
metrics + the docx→zip confusion rate (methodology Sec 10) — computable
for the first time now that a zip class exists. C1's deflate-confidence
CNN is pretrained on the custom doc/docx corpus (which has deflate ground
truth from ZIP-LFH parsing); FFT-75 provides none.

Outputs: results/new_result/p1_gate_512B.json + FFT75_REPORT_512B.md.
"""

import json
import os

import numpy as np
import torch
from scipy.stats import ttest_rel

from classify.baselines.fifty_cnn import FiftyStyleCNN
from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
from classify.data.custom_corpus import build_dataset as build_custom_corpus
from classify.data.fft75 import CLASS_TO_INDEX, COMPOUND_SUBSET, SCENARIO1_LABELS, load_fft75_arrays
from classify.eval.metrics import classification_metrics, docx_zip_confusion_rate, param_count
from classify.modules.c1_compound.dual_view_model import C1Classifier
from classify.modules.c1_compound.train_refiner import train_decompression_confidence_cnn
from classify.training.numpy_dataset import NumpyFragmentDataset
from classify.training.train import train_classifier

OUT_DIR = "results/new_result"
FRAGMENT_SIZE = 512
PUBLISHED_FIFTY_ACC = 0.656  # scenario 1, 512B, Mittal et al. TIFS
GATE_TOLERANCE = 0.05

GATE_EPOCHS = 4  # val acc still rising at 3 (0.594/0.610/0.619 on first attempt)
GATE_BATCH = 256
GATE_LR = 1e-3  # CNN, big balanced data — the transformer-collapse concerns don't apply

COMPOUND_SEEDS = [0, 1, 2]
COMPOUND_EPOCHS = 4
COMPOUND_BATCH = 64
COMPOUND_LR = 1e-4  # tuned recipe (RESEARCH_LOG 2026-06-26)
COMPOUND_TRAIN_PER_CLASS = 20000
COMPOUND_EVAL_PER_CLASS = 5000
MODEL_KWARGS = dict(d_model=128, n_heads=8, n_layers=4, d_ff=512)


@torch.no_grad()
def predict_names(model, dataset, device, label_names, batch_size=256):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_true, y_pred = [], []
    for byte_ids, labels in loader:
        preds = model(byte_ids.to(device)).argmax(dim=1).cpu().numpy()
        y_pred.extend(label_names[p] for p in preds)
        y_true.extend(label_names[i] for i in labels.numpy())
    return y_true, y_pred


def stage1_gate(device) -> dict:
    print("=== STAGE 1: P1 gate — FiFTy-style CNN, 75 classes ===", flush=True)
    x_train, y_train = load_fft75_arrays(FRAGMENT_SIZE, "train")
    x_val, y_val = load_fft75_arrays(FRAGMENT_SIZE, "val")
    x_test, y_test = load_fft75_arrays(FRAGMENT_SIZE, "test")
    train_ds = NumpyFragmentDataset(x_train, y_train)
    val_ds = NumpyFragmentDataset(x_val, y_val)
    test_ds = NumpyFragmentDataset(x_test, y_test)
    print(f"train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}", flush=True)

    model = FiftyStyleCNN(n_classes=len(SCENARIO1_LABELS))
    history = train_classifier(model, train_ds, val_ds, epochs=GATE_EPOCHS,
                                batch_size=GATE_BATCH, lr=GATE_LR, seed=0, device=device)
    print(f"val acc per epoch: {history['val_acc']}", flush=True)

    y_true, y_pred = predict_names(model, test_ds, device, SCENARIO1_LABELS)
    test_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    gate = {
        "test_accuracy": test_acc,
        "published_fifty_accuracy": PUBLISHED_FIFTY_ACC,
        "tolerance": GATE_TOLERANCE,
        "gate_passed": abs(test_acc - PUBLISHED_FIFTY_ACC) <= GATE_TOLERANCE
                        or test_acc > PUBLISHED_FIFTY_ACC,
        "n_params": param_count(model),
        "val_acc_history": history["val_acc"],
        "note": "FiFTy-style (not byte-exact) CNN; gate band ±5 points",
    }
    print(f"GATE: test_acc={test_acc:.4f} vs published {PUBLISHED_FIFTY_ACC} "
          f"-> passed={gate['gate_passed']}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return gate


def stage2_compound(device) -> dict:
    print("=== STAGE 2: compound subset — backbone vs +C1 ===", flush=True)
    subset_ids = [CLASS_TO_INDEX[c] for c in COMPOUND_SUBSET]
    label_names = list(COMPOUND_SUBSET)

    x_train, y_train = load_fft75_arrays(FRAGMENT_SIZE, "train")
    x_val, y_val = load_fft75_arrays(FRAGMENT_SIZE, "val")
    x_test, y_test = load_fft75_arrays(FRAGMENT_SIZE, "test")
    train_ds = NumpyFragmentDataset(x_train, y_train, subset_ids, COMPOUND_TRAIN_PER_CLASS)
    val_ds = NumpyFragmentDataset(x_val, y_val, subset_ids, COMPOUND_EVAL_PER_CLASS)
    test_ds = NumpyFragmentDataset(x_test, y_test, subset_ids, COMPOUND_EVAL_PER_CLASS)
    print(f"train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}", flush=True)

    # deflate-confidence refiner: trained on the custom corpus (has deflate GT)
    custom = build_custom_corpus(seed=0, n_per_size=15, sizes=(FRAGMENT_SIZE,))
    refiner = train_decompression_confidence_cnn(
        custom["train"], fragment_size=FRAGMENT_SIZE, epochs=3, batch_size=64, device=device)

    builders = {
        "backbone_only": lambda: RawByteTransformerBaseline(
            n_classes=len(label_names), max_len=FRAGMENT_SIZE, **MODEL_KWARGS),
        "+C1": lambda: C1Classifier(
            n_classes=len(label_names), max_len=FRAGMENT_SIZE,
            pretrained_refiner=refiner, **MODEL_KWARGS),
    }

    results = {name: [] for name in builders}
    for name, builder in builders.items():
        for seed in COMPOUND_SEEDS:
            print(f"--- {name} seed {seed} ---", flush=True)
            model = builder()
            train_classifier(model, train_ds, val_ds, epochs=COMPOUND_EPOCHS,
                              batch_size=COMPOUND_BATCH, lr=COMPOUND_LR, seed=seed, device=device)
            y_true, y_pred = predict_names(model, test_ds, device, label_names)
            m = classification_metrics(y_true, y_pred, labels=label_names)
            m["docx_zip_confusion_rate"] = docx_zip_confusion_rate(y_true, y_pred, label_names)
            m["seed"] = seed
            m["n_params"] = param_count(model)
            results[name].append(m)
            print(f"    macro-F1={m['macro_f1']:.4f} docx-F1={m['per_class']['docx']['f1']:.4f} "
                  f"docx->zip={m['docx_zip_confusion_rate']:.4f}", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        # checkpoint after each config
        with open(os.path.join(OUT_DIR, "p1_gate_512B_partial.json"), "w") as f:
            json.dump(results, f, indent=2)

    summary = {"labels": label_names, "seeds": COMPOUND_SEEDS,
               "train_per_class": COMPOUND_TRAIN_PER_CLASS, "configs": {}}
    for name, runs in results.items():
        summary["configs"][name] = {
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])),
            "docx_f1_mean": float(np.mean([r["per_class"]["docx"]["f1"] for r in runs])),
            "docx_zip_confusion_mean": float(np.mean([r["docx_zip_confusion_rate"] for r in runs])),
            "runs": runs,
        }
    b = [r["macro_f1"] for r in results["backbone_only"]]
    c = [r["macro_f1"] for r in results["+C1"]]
    diffs = np.array(c) - np.array(b)
    summary["significance"] = {"effect_size": float(diffs.mean())}
    if not np.allclose(diffs, diffs[0]):
        stat, p = ttest_rel(c, b)
        summary["significance"].update({"test": "paired_t", "p_value": float(p)})
    return summary


def write_report(gate: dict, compound: dict):
    L = ["# FFT-75 512B — P1 gate + compound-subset results\n"]
    L.append("## Stage 1 — P1 baseline-reproduction gate (75 classes)\n")
    L.append(f"- FiFTy-style CNN test accuracy: **{gate['test_accuracy']:.4f}**")
    L.append(f"- Published FiFTy scenario-1/512B: {gate['published_fifty_accuracy']}")
    L.append(f"- Gate (±{gate['tolerance']:.0%} or better): "
             f"**{'PASSED' if gate['gate_passed'] else 'FAILED'}**")
    L.append(f"- {gate['note']}; params={gate['n_params']:,}\n")
    L.append("## Stage 2 — Compound subset (9 classes, 3 seeds)\n")
    L.append("| Config | macro-F1 | docx-F1 | docx→zip confusion |")
    L.append("|---|---|---|---|")
    for name, cfg in compound["configs"].items():
        L.append(f"| {name} | {cfg['macro_f1_mean']:.4f}±{cfg['macro_f1_std']:.4f} "
                  f"| {cfg['docx_f1_mean']:.4f} | {cfg['docx_zip_confusion_mean']:.4f} |")
    sig = compound["significance"]
    L.append(f"\nEffect size (macro-F1, +C1 − backbone): {sig['effect_size']:+.4f}"
             + (f", paired t-test p={sig['p_value']:.4f}" if "p_value" in sig else ""))
    L.append("\nSplit caveat: FFT-75's published split is fragment-level (see fft75.py docstring).")
    with open(os.path.join(OUT_DIR, "FFT75_REPORT_512B.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gate = stage1_gate(device)
    with open(os.path.join(OUT_DIR, "p1_gate_512B.json"), "w") as f:
        json.dump({"gate": gate}, f, indent=2)

    compound = stage2_compound(device)
    with open(os.path.join(OUT_DIR, "p1_gate_512B.json"), "w") as f:
        json.dump({"gate": gate, "compound": compound}, f, indent=2)

    write_report(gate, compound)
    print("\nDone: FFT75_REPORT_512B.md + p1_gate_512B.json", flush=True)


if __name__ == "__main__":
    main()
