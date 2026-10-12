"""Array-backed Dataset for FFT-75-scale training (see fft75.load_fft75_arrays).

Indexes the (N, block) uint8 array lazily — no per-fragment Python objects.
Optionally restricts to a class subset, remapping the original scenario-1
label ids to a dense [0, n_subset) index space.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyFragmentDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, subset_class_ids: list = None,
                 max_per_class: int = None, seed: int = 0):
        """subset_class_ids: original label ids to keep, in the order that
        defines the new dense index space (e.g. [CLASS_TO_INDEX[c] for c in
        COMPOUND_SUBSET]). None keeps all classes with original ids."""
        if subset_class_ids is None:
            self.indices = np.arange(len(y))
            self.remap = None
        else:
            rng = np.random.default_rng(seed)
            keep_chunks = []
            for class_id in subset_class_ids:
                rows = np.flatnonzero(y == class_id)
                if max_per_class is not None and len(rows) > max_per_class:
                    rows = rng.choice(rows, size=max_per_class, replace=False)
                keep_chunks.append(rows)
            self.indices = np.sort(np.concatenate(keep_chunks))
            self.remap = np.full(int(y.max()) + 1, -1, dtype=np.int64)
            for new_id, class_id in enumerate(subset_class_ids):
                self.remap[class_id] = new_id
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.indices[idx]
        byte_ids = torch.from_numpy(self.x[row].astype(np.int64))
        label = int(self.y[row])
        if self.remap is not None:
            label = int(self.remap[label])
        return byte_ids, label
