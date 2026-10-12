"""Supervised pretraining for DecompressionConfidenceCNN against the
deflate-labelled subset (methodology Sec 8.1 / classify.data.deflate_label).

The refiner is only ever called inside torch.no_grad() from
dual_view_model.detection_confidence_batch (it's a detection signal, not a
differentiable part of the classification loss), so it must be trained
separately against ground-truth is_deflate labels rather than end-to-end.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from classify.modules.c1_compound.cnn_refiner import DecompressionConfidenceCNN
from classify.training.seed import set_seed


class DeflateLabelDataset(Dataset):
    def __init__(self, fragments: list, fragment_size: int):
        self.fragments = [f for f in fragments if f["fragment_size"] == fragment_size]

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        frag = self.fragments[idx]
        byte_ids = torch.from_numpy(np.frombuffer(frag["data"], dtype=np.uint8).astype(np.int64))
        return byte_ids, float(frag["is_deflate"])


def train_decompression_confidence_cnn(fragments: list, fragment_size: int, epochs: int,
                                        batch_size: int, device: str, seed: int = 0,
                                        lr: float = 1e-3) -> DecompressionConfidenceCNN:
    set_seed(seed)
    dataset = DeflateLabelDataset(fragments, fragment_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    refiner = DecompressionConfidenceCNN().to(device)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        refiner.train()
        for byte_ids, labels in loader:
            byte_ids = byte_ids.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            preds = refiner(byte_ids)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

    for p in refiner.parameters():
        p.requires_grad_(False)
    return refiner.eval()
