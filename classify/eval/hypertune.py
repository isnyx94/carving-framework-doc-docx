"""Hyperparameter sweep to find a training recipe where BOTH doc and docx
perform (RESEARCH_LOG.md: every run so far collapsed toward one class or
the other). Objective: maximize min(doc-F1, docx-F1) on val — macro-F1 can
still hide a lopsided model; the min can't.

Sweep kept small on purpose (compute budget): lr x balancing strategy x
epochs, 512B, backbone_only (the recipe transfers to +C1 since the backbone
dominates parameter count and both share the training loop).
"""

import itertools
import json
import os

import torch
from torch.utils.data import DataLoader

from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
from classify.data.custom_corpus import build_dataset
from classify.eval.metrics import classification_metrics
from classify.training.dataset import LABELS, FragmentDataset
from classify.training.train import inverse_frequency_weights, train_classifier

OUT_PATH = "results/new_result/hypertune.json"
MODEL_KWARGS = dict(d_model=128, n_heads=8, n_layers=4, d_ff=512)

GRID = {
    "lr": [3e-4, 1e-4],
    "balance": ["oversample", "class_weights_only"],
    "epochs": [6, 10],
}


@torch.no_grad()
def val_metrics(model, val_ds, device):
    loader = DataLoader(val_ds, batch_size=64)
    model.eval()
    y_true, y_pred = [], []
    for byte_ids, labels in loader:
        preds = model(byte_ids.to(device)).argmax(dim=1).cpu().numpy()
        y_pred.extend(LABELS[p] for p in preds)
        y_true.extend(LABELS[i] for i in labels.numpy())
    return classification_metrics(y_true, y_pred, labels=LABELS)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = build_dataset(seed=0, n_per_size=10, sizes=(512,))
    val_ds = FragmentDataset(corpus["val"], fragment_size=512)

    results = []
    for lr, balance, epochs in itertools.product(GRID["lr"], GRID["balance"], GRID["epochs"]):
        oversample = balance == "oversample"
        train_ds = FragmentDataset(corpus["train"], fragment_size=512, oversample=oversample)
        class_weights = None if oversample else inverse_frequency_weights(train_ds, n_classes=len(LABELS))

        model = RawByteTransformerBaseline(n_classes=len(LABELS), max_len=512, **MODEL_KWARGS)
        train_classifier(model, train_ds, val_ds, epochs=epochs, batch_size=32, lr=lr,
                          seed=0, device=device, class_weights=class_weights)
        m = val_metrics(model, val_ds, device)
        doc_f1 = m["per_class"]["doc"]["f1"]
        docx_f1 = m["per_class"]["docx"]["f1"]
        row = {"lr": lr, "balance": balance, "epochs": epochs,
               "doc_f1": doc_f1, "docx_f1": docx_f1, "min_f1": min(doc_f1, docx_f1),
               "macro_f1": m["macro_f1"]}
        results.append(row)
        print(json.dumps(row), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    results.sort(key=lambda r: r["min_f1"], reverse=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("\nBest by min-class-F1:", json.dumps(results[0]), flush=True)


if __name__ == "__main__":
    main()
