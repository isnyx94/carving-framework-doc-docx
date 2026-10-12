"""Resume the Chapter 5 run after an interruption. Reuses checkpointed
stages (gate/probe/classical + any compound seeds in compound_partial.json),
runs the remaining compound seeds, then c2c4, then the 4-epoch gate rerun
(the 2-epoch gate missed the band by 0.008 with val still rising).

Order puts the compound verdict first (the docx->zip confirmation test is
the headline), the gate rerun last.
"""

import json
import os

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon

import classify.eval.ch5_experiment as ch5
from classify.data.custom_corpus import build_dataset as build_custom_corpus
from classify.data.fft75 import CLASS_TO_INDEX, COMPOUND_SUBSET, load_fft75_arrays
from classify.eval.metrics import classification_metrics, docx_zip_confusion_rate, param_count
from classify.eval.p1_gate import predict_names
from classify.modules.c1_compound.train_refiner import train_decompression_confidence_cnn
from classify.training.numpy_dataset import NumpyFragmentDataset
from classify.training.train import train_classifier

OUT_DIR = ch5.OUT_DIR


def resume_compound(datasets, refiner, device) -> dict:
    train_ds, val_ds, test_ds = datasets
    partial_path = os.path.join(OUT_DIR, "compound_partial.json")
    results = {"backbone_only": [], "+C1": []}
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            saved = json.load(f)
        for name, runs in saved.items():
            results[name] = runs
    done = {(name, r["seed"]) for name, runs in results.items() for r in runs}
    print(f"resuming compound: {len(done)} of {2 * len(ch5.SEEDS)} runs already done: "
          f"{sorted(done)}", flush=True)

    from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
    from classify.modules.c1_compound.dual_view_model import C1Classifier

    builders = {
        "backbone_only": lambda: RawByteTransformerBaseline(
            n_classes=len(ch5.LABELS), max_len=ch5.FRAGMENT_SIZE, **ch5.MODEL_KWARGS),
        "+C1": lambda: C1Classifier(
            n_classes=len(ch5.LABELS), max_len=ch5.FRAGMENT_SIZE,
            pretrained_refiner=refiner, **ch5.MODEL_KWARGS),
    }
    for name, builder in builders.items():
        for seed in ch5.SEEDS:
            if (name, seed) in done:
                continue
            print(f"--- {name} seed {seed} ---", flush=True)
            model = builder()
            train_classifier(model, train_ds, val_ds, epochs=ch5.EPOCHS, batch_size=ch5.BATCH,
                              lr=ch5.LR, seed=seed, device=device, amp=True)
            y_true, y_pred = predict_names(model, test_ds, device, ch5.LABELS, batch_size=32)
            m = classification_metrics(y_true, y_pred, labels=ch5.LABELS)
            m["docx_zip_confusion_rate"] = docx_zip_confusion_rate(y_true, y_pred, ch5.LABELS)
            m["seed"] = seed
            m["n_params"] = param_count(model)
            results[name].append(m)
            print(f"    macro-F1={m['macro_f1']:.4f} docx-F1={m['per_class']['docx']['f1']:.4f} "
                  f"docx->zip={m['docx_zip_confusion_rate']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
            with open(partial_path, "w") as f:
                json.dump(results, f, indent=2)

    for name in results:
        results[name].sort(key=lambda r: r["seed"])
    summary = {"seeds": ch5.SEEDS, "train_per_class": ch5.TRAIN_PER_CLASS, "configs": {}}
    for name, runs in results.items():
        summary["configs"][name] = {
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])),
            "docx_f1_mean": float(np.mean([r["per_class"]["docx"]["f1"] for r in runs])),
            "docx_zip_confusion_mean": float(np.mean([r["docx_zip_confusion_rate"] for r in runs])),
            "n_params": runs[0]["n_params"], "runs": runs,
        }
    for metric in ("macro_f1", "docx_zip"):
        key = "macro_f1" if metric == "macro_f1" else "docx_zip_confusion_rate"
        b = [r[key] for r in results["backbone_only"]]
        c = [r[key] for r in results["+C1"]]
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


def main():
    device = "cuda"
    with open(os.path.join(OUT_DIR, "gate_4096B.json")) as f:
        gate = json.load(f)
    with open(os.path.join(OUT_DIR, "probe_4096B.json")) as f:
        probe = json.load(f)
    with open(os.path.join(OUT_DIR, "classical_4096B.json")) as f:
        classical = json.load(f)

    subset_ids = [CLASS_TO_INDEX[c] for c in COMPOUND_SUBSET]
    x_train, y_train = load_fft75_arrays(ch5.FRAGMENT_SIZE, "train")
    x_val, y_val = load_fft75_arrays(ch5.FRAGMENT_SIZE, "val")
    x_test, y_test = load_fft75_arrays(ch5.FRAGMENT_SIZE, "test")
    datasets = (
        NumpyFragmentDataset(x_train, y_train, subset_ids, ch5.TRAIN_PER_CLASS),
        NumpyFragmentDataset(x_val, y_val, subset_ids, ch5.EVAL_PER_CLASS),
        NumpyFragmentDataset(x_test, y_test, subset_ids, ch5.EVAL_PER_CLASS),
    )
    custom = build_custom_corpus(seed=0, n_per_size=10, sizes=(ch5.FRAGMENT_SIZE,))
    refiner = train_decompression_confidence_cnn(
        custom["train"], fragment_size=ch5.FRAGMENT_SIZE, epochs=2, batch_size=16, device=device)

    compound = resume_compound(datasets, refiner, device)
    ch5._ckpt("compound_4096B.json", compound)
    print("\nCOMPOUND VERDICT:", json.dumps({k: compound[k] for k in
          ("significance_macro_f1", "significance_docx_zip")}), flush=True)

    c2c4 = ch5.stage_c2c4(datasets, refiner, device)
    ch5._ckpt("c2c4_4096B.json", c2c4)

    print("=== GATE RERUN: 4 epochs ===", flush=True)
    ch5.GATE_EPOCHS = 4
    gate = ch5.stage_gate(device)
    ch5._ckpt("gate_4096B.json", gate)

    ch5.write_report(gate, probe, compound, classical, c2c4)
    print("\nDone: results/ch5/CH5_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
