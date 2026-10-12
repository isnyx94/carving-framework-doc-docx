import random

import numpy as np
import torch
from torch.utils.data import Dataset

LABELS = ["doc", "docx"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


class FragmentDataset(Dataset):
    """Wraps a list of fragment dicts (classify.data.custom_corpus.build_dataset
    output) as byte-index tensors for the foundation transformer."""

    def __init__(self, fragments: list, fragment_size: int, oversample: bool = False, seed: int = 0,
                 labels: list = None):
        # Methodology Sec 8.2: 512B and 4096B are reported separately, never
        # averaged together — a dataset/model instance is always single-size.
        # labels: class-name list defining the index space (default: the
        # custom-corpus doc/docx pair; pass the FFT-75 subset for benchmark runs).
        self.labels = labels if labels is not None else LABELS
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.fragments = [f for f in fragments if f["fragment_size"] == fragment_size]
        self.fragment_size = fragment_size
        if oversample:
            self.fragments = self._oversample(self.fragments, seed)

    @staticmethod
    def _oversample(fragments: list, seed: int) -> list:
        """Duplicates minority-class fragments (with replacement) to match
        the majority class count. Train-split only — applying this to
        val/test would inflate metrics by repeating the same fragment.
        Addresses the real ~5:1 doc:docx imbalance (RESEARCH_LOG.md) on top
        of class-weighted loss, which alone wasn't enough to stop
        majority-class collapse at 4096B.
        """
        rng = random.Random(seed)
        by_label = {}
        for f in fragments:
            by_label.setdefault(f["label"], []).append(f)
        majority_count = max(len(v) for v in by_label.values())
        balanced = []
        for label, items in by_label.items():
            balanced.extend(items)
            deficit = majority_count - len(items)
            if deficit > 0:
                balanced.extend(rng.choices(items, k=deficit))
        rng.shuffle(balanced)
        return balanced

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        frag = self.fragments[idx]
        byte_ids = torch.from_numpy(np.frombuffer(frag["data"], dtype=np.uint8).astype(np.int64))
        label = self.label_to_idx[frag["label"]]
        return byte_ids, label
