"""Ablation runner (methodology Sec 9.2 / Sec 7 P5): backbone-only vs +C1,
multi-seed, with a paired t-test on the compound-class metric (docx-F1, the
closest available proxy to docx vs zip/ooxml confusion until FFT-75 is
sourced — see classify/data/fft75.py).

Protocol per design decision Q3: 5 seeds, paired t-test at alpha=0.05,
target effect size >=2% absolute improvement in compound-class macro-F1.
Wilcoxon signed-rank is kept as a secondary, non-parametric cross-check.

This is wired for real runs, but the model sizes / epoch counts passed in
from configs/ are smoke-scale by default (see classify/eval/run_smoke.py).
Do not read numbers produced at smoke scale as a methodology claim — scale
up epochs/data before reporting anywhere outside the project notes.
"""

import json
import os

import numpy as np
import torch
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader

from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
from classify.eval.metrics import classification_metrics, param_count
from classify.modules.c1_compound.dual_view_model import C1Classifier
from classify.modules.c1_compound.train_refiner import train_decompression_confidence_cnn
from classify.training.dataset import LABELS, FragmentDataset
from classify.training.train import inverse_frequency_weights, train_classifier

ALPHA = 0.05
MIN_EFFECT_SIZE = 0.02  # methodology Q3: target >=2% absolute macro-F1 improvement


@torch.no_grad()
def _predict_labels(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_true, y_pred = [], []
    for byte_ids, labels in loader:
        preds = model(byte_ids.to(device)).argmax(dim=1).cpu().numpy()
        y_pred.extend(LABELS[p] for p in preds)
        y_true.extend(LABELS[i] for i in labels.numpy())
    return y_true, y_pred


def run_ablation(corpus: dict, fragment_size: int, seeds: list, model_kwargs: dict,
                  epochs: int, batch_size: int, lr: float, device: str, out_dir: str,
                  refiner_epochs: int = 3, oversample: bool = True,
                  use_class_weights: bool = True,
                  include_detection_only: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    train_ds = FragmentDataset(corpus["train"], fragment_size=fragment_size, oversample=oversample)
    val_ds = FragmentDataset(corpus["val"], fragment_size=fragment_size)
    test_ds = FragmentDataset(corpus["test"], fragment_size=fragment_size)
    class_weights = (inverse_frequency_weights(train_ds, n_classes=len(LABELS))
                      if use_class_weights else None)

    # Pretrained once per fragment size (not per seed) against deflate ground
    # truth, then reused (frozen) across all +C1 seeds — see train_refiner.py
    # docstring for why this can't be trained end-to-end through the gate.
    refiner = train_decompression_confidence_cnn(
        corpus["train"], fragment_size=fragment_size, epochs=refiner_epochs,
        batch_size=batch_size, device=device,
    )

    model_builders = {
        "backbone_only": lambda: RawByteTransformerBaseline(
            n_classes=len(LABELS), max_len=fragment_size, **model_kwargs
        ),
        "+C1": lambda: C1Classifier(
            n_classes=len(LABELS), max_len=fragment_size, pretrained_refiner=refiner, **model_kwargs
        ),
    }
    if include_detection_only:
        # C1 sub-ablation (methodology Sec 9.2): detector runs but decompression
        # is skipped — isolates the detector's contribution from the decompressor's.
        model_builders["+C1_detection_only"] = lambda: C1Classifier(
            n_classes=len(LABELS), max_len=fragment_size, pretrained_refiner=refiner,
            detection_only=True, **model_kwargs
        )

    results = {config_name: [] for config_name in model_builders}
    for config_name, builder in model_builders.items():
        for seed in seeds:
            model = builder()
            train_classifier(model, train_ds, val_ds, epochs=epochs, batch_size=batch_size,
                              lr=lr, seed=seed, device=device, class_weights=class_weights)
            y_true, y_pred = _predict_labels(model, test_ds, device, batch_size)
            metrics = classification_metrics(y_true, y_pred, labels=LABELS)
            metrics["seed"] = seed
            metrics["n_params"] = param_count(model)
            results[config_name].append(metrics)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    summary = {"fragment_size": fragment_size, "seeds": seeds, "configs": {}}
    for config_name, runs in results.items():
        macro_f1s = [r["macro_f1"] for r in runs]
        docx_f1s = [r["per_class"]["docx"]["f1"] for r in runs]
        summary["configs"][config_name] = {
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
            "docx_f1_mean": float(np.mean(docx_f1s)),
            "docx_f1_std": float(np.std(docx_f1s)),
            "n_params": runs[0]["n_params"],
            "runs": runs,
        }

    backbone_macro_f1 = [r["macro_f1"] for r in results["backbone_only"]]
    c1_macro_f1 = [r["macro_f1"] for r in results["+C1"]]
    summary["significance_test"] = _significance_test(backbone_macro_f1, c1_macro_f1)

    out_path = os.path.join(out_dir, f"ablation_{fragment_size}B.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _significance_test(backbone_scores: list, c1_scores: list) -> dict:
    """Paired t-test on compound-class macro-F1 across seeds (design Q3:
    5 seeds, alpha=0.05, target effect size >=2% absolute). Needs >=2 seeds
    and non-zero variance in the paired differences or scipy raises."""
    if len(backbone_scores) < 2:
        return {"error": "need >=2 seeds for a paired t-test"}
    diffs = np.array(c1_scores) - np.array(backbone_scores)
    effect_size = float(diffs.mean())
    result = {
        "test": "paired_t_test", "alpha": ALPHA, "effect_size": effect_size,
        "effect_size_meets_target": effect_size >= MIN_EFFECT_SIZE,
    }
    if np.allclose(diffs, diffs[0]):
        result["error"] = "zero variance in paired differences across seeds"
        return result
    stat, pvalue = ttest_rel(c1_scores, backbone_scores)
    result.update({"statistic": float(stat), "p_value": float(pvalue), "significant": bool(pvalue < ALPHA)})
    try:
        w_stat, w_pvalue = wilcoxon(c1_scores, backbone_scores)
        result["wilcoxon_secondary"] = {"statistic": float(w_stat), "p_value": float(w_pvalue)}
    except ValueError:
        pass
    return result


def to_markdown_table(summary: dict) -> str:
    lines = [
        f"### Ablation @ {summary['fragment_size']}B (seeds={summary['seeds']})",
        "",
        "| Config | macro-F1 (mean±std) | docx-F1 (mean±std) | params |",
        "|---|---|---|---|",
    ]
    for name, cfg in summary["configs"].items():
        lines.append(
            f"| {name} | {cfg['macro_f1_mean']:.3f}±{cfg['macro_f1_std']:.3f} "
            f"| {cfg['docx_f1_mean']:.3f}±{cfg['docx_f1_std']:.3f} | {cfg['n_params']:,} |"
        )
    sig = summary.get("significance_test", {})
    if "p_value" in sig:
        lines.append(
            f"\nPaired t-test on macro-F1 (backbone_only vs +C1): p={sig['p_value']:.4f}, "
            f"effect size={sig['effect_size']:+.4f} (target >={MIN_EFFECT_SIZE}), "
            f"significant={sig['significant']}"
        )
    else:
        lines.append(f"\nSignificance test not computed: {sig.get('error')}")
    return "\n".join(lines)
