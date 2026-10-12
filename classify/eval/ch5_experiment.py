"""Chapter 5 experiments: 4,096-byte condition (the front-end's predicted
best case — deflate stream starts are 8x more likely per fragment).

Protocol per thesis Sec 4.10 decisions:
  1. 4096B is the confirmation test for the docx->zip trend.
  2. C4 open-set uses HELD-OUT FFT-75 CLASSES as OOD (clean unseen-type
     test): rar, 7z, gz, png, mp3 test fragments — never seen in training.
  3. C2 deletion curves at the designed 100/class; attribution explains the
     gated-fusion path for the front-end model (fused_token_embedding).
  4. Recipe, metrics, seeds (5), and significance thresholds unchanged.
  5. XGBoost baseline retained in every table.

Stages (each checkpointed to JSON):
  gate     — FiFTy-style CNN, 75 classes, vs published 77.5% (band +/-5pp)
  probe    — times 30 training steps of the full-spec backbone, projects
             per-run cost, halves the train cap if projection exceeds
             MAX_RUN_HOURS
  compound — backbone vs +C1, 5 seeds, paired tests incl. docx->zip
  classical— RF + XGBoost on the same subset
  c2c4     — calibration, deletion curve (100/class), held-out-class OOD

All prints ASCII. Outputs: results/ch5/ (JSONs + CH5_REPORT.md).
"""

import json
import os
import time

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader

from classify.baselines.fifty_cnn import FiftyStyleCNN
from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
from classify.data.custom_corpus import build_dataset as build_custom_corpus
from classify.data.fft75 import CLASS_TO_INDEX, COMPOUND_SUBSET, SCENARIO1_LABELS, load_fft75_arrays
from classify.eval.deletion_curve import deletion_accuracy_curve
from classify.eval.metrics import classification_metrics, docx_zip_confusion_rate, param_count
from classify.eval.p1_gate import predict_names
from classify.modules.c1_compound.dual_view_model import C1Classifier
from classify.modules.c1_compound.train_refiner import train_decompression_confidence_cnn
from classify.modules.heads.defensible_head import TemperatureScaler, expected_calibration_error
from classify.modules.heads.openset_head import auroc_id_vs_ood, msp_scores, rejection_curve
from classify.training.numpy_dataset import NumpyFragmentDataset
from classify.training.train import train_classifier

OUT_DIR = "results/ch5"
FRAGMENT_SIZE = 4096
PUBLISHED_FIFTY_ACC_4096 = 0.775  # scenario 1 average, Mittal et al. abstract — verify vs TIFS table
GATE_TOLERANCE = 0.05
GATE_EPOCHS = 2
GATE_BATCH = 128
GATE_LR = 1e-3

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 2
BATCH = 8
LR = 1e-4
TRAIN_PER_CLASS = 5000
EVAL_PER_CLASS = 2000
MAX_RUN_HOURS = 2.2  # probe halves TRAIN_PER_CLASS if a run projects past this
MODEL_KWARGS = dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024)
LABELS = list(COMPOUND_SUBSET)

OOD_CLASSES = ("rar", "7z", "gz", "png", "mp3")  # held out of training entirely
OOD_PER_CLASS = 1000
DELETION_N_PER_CLASS = 100


def _ckpt(name, payload):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(payload, f, indent=2)


def stage_gate(device) -> dict:
    print("=== GATE: FiFTy-style CNN, 75 classes @ 4096B ===", flush=True)
    x_train, y_train = load_fft75_arrays(FRAGMENT_SIZE, "train")
    x_val, y_val = load_fft75_arrays(FRAGMENT_SIZE, "val")
    x_test, y_test = load_fft75_arrays(FRAGMENT_SIZE, "test")
    train_ds = NumpyFragmentDataset(x_train, y_train)
    val_ds = NumpyFragmentDataset(x_val, y_val)
    test_ds = NumpyFragmentDataset(x_test, y_test)
    print(f"train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}", flush=True)

    model = FiftyStyleCNN(n_classes=len(SCENARIO1_LABELS))
    history = train_classifier(model, train_ds, val_ds, epochs=GATE_EPOCHS,
                                batch_size=GATE_BATCH, lr=GATE_LR, seed=0, device=device, amp=True)
    print(f"val acc per epoch: {history['val_acc']}", flush=True)
    y_true, y_pred = predict_names(model, test_ds, device, SCENARIO1_LABELS)
    test_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    gate = {
        "test_accuracy": test_acc, "published": PUBLISHED_FIFTY_ACC_4096,
        "tolerance": GATE_TOLERANCE,
        "gate_passed": test_acc >= PUBLISHED_FIFTY_ACC_4096 - GATE_TOLERANCE,
        "n_params": param_count(model), "val_acc_history": history["val_acc"],
        "epochs": GATE_EPOCHS,
    }
    print(f"GATE: test_acc={test_acc:.4f} vs published {PUBLISHED_FIFTY_ACC_4096} "
          f"-> passed={gate['gate_passed']}", flush=True)
    del model
    torch.cuda.empty_cache()
    return gate


def stage_probe(train_ds, device) -> dict:
    """Times 30 optimiser steps of the full-spec backbone and projects the
    cost of one full training run; halves TRAIN_PER_CLASS if over budget."""
    global TRAIN_PER_CLASS
    print("=== PROBE: timing 30 steps of full-spec backbone ===", flush=True)
    model = RawByteTransformerBaseline(n_classes=len(LABELS), max_len=FRAGMENT_SIZE,
                                        **MODEL_KWARGS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    it = iter(loader)
    # warmup
    for _ in range(3):
        xb, yb = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = criterion(model(xb.to(device)), yb.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        xb, yb = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = criterion(model(xb.to(device)), yb.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    sec_per_step = (time.perf_counter() - t0) / 30
    steps_per_run = (TRAIN_PER_CLASS * len(LABELS) / BATCH) * EPOCHS
    projected_h = sec_per_step * steps_per_run / 3600
    probe = {"sec_per_step": sec_per_step, "projected_hours_per_run": projected_h,
              "train_per_class": TRAIN_PER_CLASS}
    print(f"probe: {sec_per_step:.3f}s/step -> {projected_h:.2f}h per training run", flush=True)
    if projected_h > MAX_RUN_HOURS:
        TRAIN_PER_CLASS //= 2
        probe["reduced_train_per_class"] = TRAIN_PER_CLASS
        print(f"over budget ({MAX_RUN_HOURS}h): TRAIN_PER_CLASS halved to {TRAIN_PER_CLASS}", flush=True)
    del model
    torch.cuda.empty_cache()
    return probe


def stage_compound(datasets, refiner, device) -> dict:
    train_ds, val_ds, test_ds = datasets
    print(f"=== COMPOUND: backbone vs +C1, {len(SEEDS)} seeds @ 4096B "
          f"(train={len(train_ds):,}) ===", flush=True)
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
            y_true, y_pred = predict_names(model, test_ds, device, LABELS, batch_size=32)
            m = classification_metrics(y_true, y_pred, labels=LABELS)
            m["docx_zip_confusion_rate"] = docx_zip_confusion_rate(y_true, y_pred, LABELS)
            m["seed"] = seed
            m["n_params"] = param_count(model)
            results[name].append(m)
            print(f"    macro-F1={m['macro_f1']:.4f} docx-F1={m['per_class']['docx']['f1']:.4f} "
                  f"docx->zip={m['docx_zip_confusion_rate']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
            _ckpt("compound_partial.json", results)

    summary = {"seeds": SEEDS, "train_per_class": TRAIN_PER_CLASS, "configs": {}}
    for name, runs in results.items():
        summary["configs"][name] = {
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])),
            "docx_f1_mean": float(np.mean([r["per_class"]["docx"]["f1"] for r in runs])),
            "docx_zip_confusion_mean": float(np.mean([r["docx_zip_confusion_rate"] for r in runs])),
            "n_params": runs[0]["n_params"], "runs": runs,
        }
    for metric, key in (("macro_f1", "macro_f1"), ("docx_zip", "docx_zip_confusion_rate")):
        b = [r[key] if key in r else r["macro_f1"] for r in results["backbone_only"]]
        c = [r[key] if key in r else r["macro_f1"] for r in results["+C1"]]
        if metric == "docx_zip":
            b = [r["docx_zip_confusion_rate"] for r in results["backbone_only"]]
            c = [r["docx_zip_confusion_rate"] for r in results["+C1"]]
        diffs = np.array(c) - np.array(b)
        sig = {"effect_size": float(diffs.mean())}
        if not np.allclose(diffs, diffs[0]):
            t, p = ttest_rel(c, b)
            sig.update({"t": float(t), "p_value": float(p), "significant": bool(p < 0.05)})
            try:
                w, wp = wilcoxon(c, b)
                sig["wilcoxon_p"] = float(wp)
            except ValueError:
                pass
        summary[f"significance_{metric}"] = sig
    return summary


def stage_classical(train_ds, test_ds) -> dict:
    print("=== CLASSICAL: RF + XGBoost @ 4096B ===", flush=True)
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from classify.eval.fullspec_512 import _classical_features

    Xtr, ytr = _classical_features(train_ds)
    Xte, yte = _classical_features(test_ds)
    out = {}
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


def stage_c2c4(datasets, refiner, device) -> dict:
    train_ds, val_ds, test_ds = datasets
    print("=== C2/C4: seed-0 models, held-out-class OOD ===", flush=True)
    x_test, y_test = load_fft75_arrays(FRAGMENT_SIZE, "test")
    ood_ids = [CLASS_TO_INDEX[c] for c in OOD_CLASSES]
    ood_ds = NumpyFragmentDataset(x_test, y_test, ood_ids, OOD_PER_CLASS)
    print(f"OOD set: {len(ood_ds):,} fragments from held-out classes {OOD_CLASSES}", flush=True)

    out = {"ood_classes": list(OOD_CLASSES)}
    for name in ("backbone_only", "+C1"):
        if name == "backbone_only":
            model = RawByteTransformerBaseline(n_classes=len(LABELS), max_len=FRAGMENT_SIZE, **MODEL_KWARGS)
        else:
            model = C1Classifier(n_classes=len(LABELS), max_len=FRAGMENT_SIZE,
                                  pretrained_refiner=refiner, **MODEL_KWARGS)
        train_classifier(model, train_ds, val_ds, epochs=EPOCHS, batch_size=BATCH,
                          lr=LR, seed=0, device=device, amp=True)

        loader = DataLoader(val_ds, batch_size=64)
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits_list.append(model(xb.to(device)))
                labels_list.append(yb.to(device))
        logits, labels_t = torch.cat(logits_list), torch.cat(labels_list)
        conf, pred = torch.softmax(logits, dim=1).max(dim=1)
        correct = (pred == labels_t)
        ece_before = expected_calibration_error(conf, correct)
        scaler = TemperatureScaler().to(device)
        scaler.fit(logits, labels_t)
        conf_after, _ = torch.softmax(scaler(logits), dim=1).max(dim=1)
        ece_after = expected_calibration_error(conf_after, correct)

        id_conf, id_pred, id_labels = msp_scores(model, test_ds, device, batch_size=32,
                                                   temperature_scaler=scaler)
        ood_conf, _, _ = msp_scores(model, ood_ds, device, batch_size=32,
                                      temperature_scaler=scaler)
        entry = {
            "ece_before": ece_before, "ece_after": ece_after,
            "temperature": scaler.log_temperature.exp().item(),
            "c4_auroc_heldout": auroc_id_vs_ood(id_conf, ood_conf),
            "c4_rejection_curve": rejection_curve(id_conf, id_pred, id_labels, ood_conf),
        }
        if name == "+C1":
            print("    deletion curve (fused-path IG, 100/class)...", flush=True)
            entry["deletion_curve"] = deletion_accuracy_curve(
                model, test_ds, LABELS, device, n_per_class=DELETION_N_PER_CLASS)
        out[name] = entry
        print(f"  {name}: ECE {ece_before:.4f}->{ece_after:.4f}, "
              f"held-out-class AUROC={entry['c4_auroc_heldout']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return out


def write_report(gate, probe, compound, classical, c2c4):
    L = ["# Chapter 5 — 4,096-byte experiments\n"]
    L.append("## Gate (75 classes)\n")
    L.append(f"FiFTy-style CNN: test acc **{gate['test_accuracy']:.4f}** vs published "
             f"{gate['published']} (band +/-{gate['tolerance']:.0%}) -> "
             f"**{'PASSED' if gate['gate_passed'] else 'FAILED'}**\n")
    L.append("## Compound subset (9 classes, 5 seeds)\n")
    L.append("| Model | macro-F1 | docx-F1 | docx->zip |")
    L.append("|---|---|---|---|")
    for name in ("random_forest", "xgboost"):
        m = classical[name]
        L.append(f"| {name} | {m['macro_f1']:.4f} | {m['per_class']['docx']['f1']:.4f} "
                  f"| {m['docx_zip_confusion_rate']:.4f} |")
    for name, cfg in compound["configs"].items():
        L.append(f"| {name} (5 seeds) | {cfg['macro_f1_mean']:.4f}+/-{cfg['macro_f1_std']:.4f} "
                  f"| {cfg['docx_f1_mean']:.4f} | {cfg['docx_zip_confusion_mean']:.4f} |")
    for metric in ("macro_f1", "docx_zip"):
        sig = compound[f"significance_{metric}"]
        L.append(f"\n{metric} paired test: effect={sig['effect_size']:+.4f}"
                 + (f", p={sig['p_value']:.4f}, significant={sig['significant']}"
                    if "p_value" in sig else " (zero variance)"))
    L.append(f"\nTrain cap: {compound['train_per_class']}/class "
             f"(probe: {probe['projected_hours_per_run']:.2f}h/run projected)\n")
    L.append("## C2 / C4 (seed 0; OOD = held-out classes "
             f"{', '.join(c2c4['ood_classes'])})\n")
    L.append("| Model | ECE before | ECE after | held-out-class AUROC |")
    L.append("|---|---|---|---|")
    for name in ("backbone_only", "+C1"):
        e = c2c4[name]
        L.append(f"| {name} | {e['ece_before']:.4f} | {e['ece_after']:.4f} "
                  f"| {e['c4_auroc_heldout']:.4f} |")
    dc = c2c4["+C1"].get("deletion_curve")
    if dc:
        L.append(f"\nDeletion curve (fused-path IG, n={dc['n_fragments']}):")
        L.append("| fraction | acc (IG) | acc (random) |")
        L.append("|---|---|---|")
        for f, ig, rnd in zip(dc["fractions"], dc["ig"], dc["random"]):
            L.append(f"| {f:.1f} | {ig:.3f} | {rnd:.3f} |")
    with open(os.path.join(OUT_DIR, "CH5_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"

    gate = stage_gate(device)
    _ckpt("gate_4096B.json", gate)

    subset_ids = [CLASS_TO_INDEX[c] for c in COMPOUND_SUBSET]
    x_train, y_train = load_fft75_arrays(FRAGMENT_SIZE, "train")
    x_val, y_val = load_fft75_arrays(FRAGMENT_SIZE, "val")
    x_test, y_test = load_fft75_arrays(FRAGMENT_SIZE, "test")
    probe_ds = NumpyFragmentDataset(x_train, y_train, subset_ids, TRAIN_PER_CLASS)
    probe = stage_probe(probe_ds, device)
    _ckpt("probe_4096B.json", probe)

    train_ds = NumpyFragmentDataset(x_train, y_train, subset_ids, TRAIN_PER_CLASS)
    val_ds = NumpyFragmentDataset(x_val, y_val, subset_ids, EVAL_PER_CLASS)
    test_ds = NumpyFragmentDataset(x_test, y_test, subset_ids, EVAL_PER_CLASS)
    datasets = (train_ds, val_ds, test_ds)

    custom = build_custom_corpus(seed=0, n_per_size=10, sizes=(FRAGMENT_SIZE,))
    refiner = train_decompression_confidence_cnn(
        custom["train"], fragment_size=FRAGMENT_SIZE, epochs=2, batch_size=16, device=device)

    classical = stage_classical(train_ds, test_ds)
    _ckpt("classical_4096B.json", classical)

    compound = stage_compound(datasets, refiner, device)
    _ckpt("compound_4096B.json", compound)

    c2c4 = stage_c2c4(datasets, refiner, device)
    _ckpt("c2c4_4096B.json", {k: v for k, v in c2c4.items()})

    write_report(gate, probe, compound, classical, c2c4)
    print("\nDone: results/ch5/CH5_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
