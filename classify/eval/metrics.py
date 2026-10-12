"""Evaluation metrics (methodology Sec 10). Per-class doc/docx (+zip/ooxml
once FFT-75 is sourced) P/R/F1 and macro-F1 are the headline numbers —
overall accuracy is reported only for comparability with prior work, never
as the primary claim (Sec 0, "Average accuracy vs. collapse points").
"""

import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred, labels: list) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {"precision": p, "recall": r, "f1": f, "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    return {
        "per_class": per_class,
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "micro_accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def docx_zip_confusion_rate(y_true, y_pred, labels: list) -> float:
    """Fraction of true-docx fragments predicted as zip (methodology Sec 10,
    "targeted docx↔zip confusion rate"). Returns None if "zip" isn't a class
    in this run (e.g. the custom doc/docx-only corpus, pre-FFT-75)."""
    if "docx" not in labels or "zip" not in labels:
        return None
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    docx_mask = y_true_arr == "docx"
    if docx_mask.sum() == 0:
        return 0.0
    return float((y_pred_arr[docx_mask] == "zip").mean())


def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def inference_throughput(model: torch.nn.Module, dataset, device: str, batch_size: int = 64) -> dict:
    """Returns {fragments_per_sec, bytes_per_sec} (methodology Sec 10, "Cost")."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    n_fragments, n_bytes = 0, 0
    start = time.perf_counter()
    for byte_ids, _ in loader:
        byte_ids = byte_ids.to(device)
        model(byte_ids)
        n_fragments += byte_ids.shape[0]
        n_bytes += byte_ids.shape[0] * byte_ids.shape[1]
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "fragments_per_sec": n_fragments / elapsed,
        "bytes_per_sec": n_bytes / elapsed,
        "elapsed_sec": elapsed,
    }
