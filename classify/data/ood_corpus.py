"""OOD fragments for C4 open-set evaluation (methodology Sec 8.4, "hold out
entire types unseen during training").

Sources: file types present in this repo but never seen by the classifier
(trained only on doc/docx): PDF, JPEG, and uniform-random bytes. Small
corpus — enough for a preliminary C4 characterization, not a full open-set
benchmark (that needs FFT-75's held-out types; flagged in the report).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .fragmentation import sample_fragments

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OOD_FILES = {
    "pdf": ["junk.pdf", "Notis Pemberitahuan Bilangan 1-2026 Pejabat Pendaftar.pdf",
             "Technical Analysis of DOC and DOCX File Structures for File Carving.pdf"],
    "jpeg": ["noise.jpg"],
    "random": ["random.bin"],
}


class OODFragmentDataset(Dataset):
    """Yields (byte_ids, -1) — label -1 marks 'no in-distribution label'."""

    def __init__(self, fragment_size: int, seed: int = 0, n_per_size: int = 60):
        self.fragments = []
        for ood_type, names in OOD_FILES.items():
            for name in names:
                path = os.path.join(REPO_ROOT, name)
                if not os.path.isfile(path):
                    continue
                for frag in sample_fragments(path, ood_type, seed=seed,
                                              sizes=(fragment_size,), n_per_size=n_per_size):
                    self.fragments.append(frag)
        if not self.fragments:
            raise FileNotFoundError("No OOD source files found — check OOD_FILES paths")

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        frag = self.fragments[idx]
        byte_ids = torch.from_numpy(np.frombuffer(frag.data, dtype=np.uint8).astype(np.int64))
        return byte_ids, -1
