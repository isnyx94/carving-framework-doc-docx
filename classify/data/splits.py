"""Per-parent-file stratified splits (methodology Sec 8.4).

Splits must never separate fragments of the same parent file across
train/val/test — that is a leakage bug, not a modelling choice. We split
on parent file path, stratified by label, using a deterministic hash so
results are reproducible across runs without persisting the split.
"""

import hashlib
from collections import defaultdict
from typing import Sequence, Tuple

SplitName = str  # "train" | "val" | "test"


def _hash_unit(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{key}|{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def split_parent_files(parent_labels: Sequence[Tuple[str, str]], seed: int,
                        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> dict:
    """parent_labels: [(parent_path, label), ...]. Returns {parent_path: split_name}.

    Stratifies by label so each split sees a proportional mix of doc/docx
    (and any other classes), while keeping every fragment of a given parent
    file in exactly one split.
    """
    assert abs(sum(ratios) - 1.0) < 1e-9
    by_label = defaultdict(list)
    for path, label in parent_labels:
        by_label[label].append(path)

    assignment = {}
    for label, paths in by_label.items():
        ranked = sorted(paths, key=lambda p: _hash_unit(f"{label}:{p}", seed))
        n = len(ranked)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        for i, path in enumerate(ranked):
            if i < n_train:
                assignment[path] = "train"
            elif i < n_train + n_val:
                assignment[path] = "val"
            else:
                assignment[path] = "test"
    return assignment


def held_out_type_split(parent_labels: Sequence[Tuple[str, str]], held_out_labels: Sequence[str]) -> dict:
    """For open-set (C4, deferred): mark parent files of held_out_labels as 'ood_test'.

    Not wired into the active pipeline yet (C4 deferred per UPGRADE_PLAN.md);
    kept here so P4 doesn't need to redesign the split contract.
    """
    held_out = set(held_out_labels)
    return {path: ("ood_test" if label in held_out else "id") for path, label in parent_labels}
