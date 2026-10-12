"""Tiny 1D-CNN decompression-confidence refiner (design Q1/Q5): the
heuristic in detector.py decides the easy cases for free; this network only
runs on fragments the heuristic calls "ambiguous", outputting a scalar
P(worth attempting decompression) in [0, 1].

Deliberately small and separate from the foundation backbone — it has its
own tiny embedding, not the shared 256-dim one, so it stays cheap to run on
every ambiguous fragment without inflating the backbone-identical
comparison (design Q3).
"""

import torch
from torch import nn

REFINER_EMBED_DIM = 16


class DecompressionConfidenceCNN(nn.Module):
    def __init__(self, embed_dim: int = REFINER_EMBED_DIM, channels: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, channels, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, byte_seq: torch.Tensor) -> torch.Tensor:
        """byte_seq: (B, L) int64 byte values. Returns (B,) confidence in [0, 1]."""
        x = self.embedding(byte_seq).transpose(1, 2)  # (B, embed_dim, L)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, channels)
        return torch.sigmoid(self.fc(x)).squeeze(-1)
