"""Open-set head (methodology Sec 6.2, C4): abstain/OOD scoring.

Method: maximum-softmax-probability (MSP) rejection (Hendrycks & Gimpel
2017) — the methodology's own suggested baseline mechanism ("max-softmax-
probability threshold"). A fragment is *abstained on* when the calibrated
softmax confidence falls below a threshold; evaluation reports AUROC for
ID-vs-OOD separation and accuracy-on-rejection (closed-set accuracy on the
fragments the model chose to keep), swept over thresholds — this sweep is
the OSCR-style operating curve.

Kept deliberately post-hoc (no training changes): it scores the logits of
an already-trained closed-set model, so it composes with backbone_only and
+C1 alike without touching the C1/C3 comparisons.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REJECTION_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


@torch.no_grad()
def msp_scores(model: nn.Module, dataset, device: str, batch_size: int = 64,
               temperature_scaler: nn.Module = None):
    """Returns (confidences, predictions, labels) as numpy arrays. Labels are
    -1 for datasets without labels (OOD fragments)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    confs, preds, labels = [], [], []
    for batch in loader:
        byte_ids, y = batch
        logits = model(byte_ids.to(device))
        if temperature_scaler is not None:
            logits = temperature_scaler(logits)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        confs.append(conf.cpu().numpy())
        preds.append(pred.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(confs), np.concatenate(preds), np.concatenate(labels)


def auroc_id_vs_ood(id_conf: np.ndarray, ood_conf: np.ndarray) -> float:
    """AUROC for 'is this fragment in-distribution' using MSP as the score."""
    from sklearn.metrics import roc_auc_score
    scores = np.concatenate([id_conf, ood_conf])
    is_id = np.concatenate([np.ones_like(id_conf), np.zeros_like(ood_conf)])
    return float(roc_auc_score(is_id, scores))


def rejection_curve(id_conf: np.ndarray, id_pred: np.ndarray, id_labels: np.ndarray,
                     ood_conf: np.ndarray, thresholds=REJECTION_THRESHOLDS) -> list:
    """For each threshold: fraction of ID kept, accuracy on kept ID
    fragments, and fraction of OOD correctly rejected (abstained on)."""
    rows = []
    for t in thresholds:
        id_keep = id_conf >= t
        ood_reject = ood_conf < t
        kept_frac = float(id_keep.mean())
        acc_on_kept = float((id_pred[id_keep] == id_labels[id_keep]).mean()) if id_keep.any() else None
        rows.append({
            "threshold": t,
            "id_kept_fraction": kept_frac,
            "accuracy_on_kept": acc_on_kept,
            "ood_rejected_fraction": float(ood_reject.mean()),
        })
    return rows
