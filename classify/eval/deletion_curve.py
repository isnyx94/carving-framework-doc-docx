"""Deletion-curve faithfulness check for Integrated Gradients (design Q4):
on a fixed sample of fragments per class, progressively zero out the
top-k% most-important bytes (by |IG| rank) and watch accuracy degrade. A
faithful attribution should cause a *steeper* drop than a random byte
order — that comparison is what makes this a faithfulness check rather
than just a pretty curve.
"""

import random

import torch

from classify.modules.heads.defensible_head import integrated_gradients

DELETION_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)


def _sample_per_class(dataset, n_per_class: int, labels: list, seed: int = 0):
    """Works with any dataset yielding (byte_ids, label_idx) — dict-backed
    (FragmentDataset) and array-backed (NumpyFragmentDataset) alike."""
    rng = random.Random(seed)
    by_label = {i: [] for i in range(len(labels))}
    for i in range(len(dataset)):
        _, label_idx = dataset[i]
        by_label[int(label_idx)].append(i)
    sampled = []
    for _, indices in by_label.items():
        sampled.extend(rng.sample(indices, min(n_per_class, len(indices))))
    return sampled


@torch.no_grad()
def _predict_correct(model, byte_seq: torch.Tensor, label: int) -> bool:
    logits = model(byte_seq.unsqueeze(0))
    return int(logits.argmax(dim=1).item()) == label


def deletion_accuracy_curve(model, dataset, labels: list, device: str, n_per_class: int = 100,
                             fractions=DELETION_FRACTIONS, seed: int = 0) -> dict:
    """Returns {"ig": [acc@0%, acc@10%, ...], "random": [...]} over the
    fragment sample, both deleting bytes (replacing with 0) in order of
    |IG| descending (ig) vs. a random permutation (random baseline)."""
    indices = _sample_per_class(dataset, n_per_class, labels, seed=seed)
    rng = random.Random(seed)

    ig_correct = {f: 0 for f in fractions}
    random_correct = {f: 0 for f in fractions}

    for idx in indices:
        byte_seq, label = dataset[idx]
        byte_seq = byte_seq.to(device)
        seq_len = byte_seq.shape[0]

        attributions = integrated_gradients(model, byte_seq.unsqueeze(0), target_class=label, n_steps=20)
        ig_order = torch.argsort(attributions.abs().squeeze(0), descending=True).cpu().tolist()
        random_order = list(range(seq_len))
        rng.shuffle(random_order)

        for fraction in fractions:
            n_delete = int(seq_len * fraction)

            ig_perturbed = byte_seq.clone()
            ig_perturbed[ig_order[:n_delete]] = 0
            if _predict_correct(model, ig_perturbed, label):
                ig_correct[fraction] += 1

            random_perturbed = byte_seq.clone()
            random_perturbed[random_order[:n_delete]] = 0
            if _predict_correct(model, random_perturbed, label):
                random_correct[fraction] += 1

    n = len(indices)
    return {
        "fractions": list(fractions),
        "n_fragments": n,
        "ig": [ig_correct[f] / n for f in fractions],
        "random": [random_correct[f] / n for f in fractions],
    }
