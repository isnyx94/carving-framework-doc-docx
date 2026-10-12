"""FiFTy-style 1D CNN baseline (methodology Sec 9.1 item 3; Mittal, Korus,
Memon, IEEE TIFS 2020/2021).

Architecture follows the paper's design family — trainable byte embedding
into stacked Conv1D + maxpool blocks, global pooling, dense head — but is
NOT a byte-exact replica of their TPE-tuned per-scenario hyperparameters,
so the P1 gate uses a tolerance band (published scenario-1/512B accuracy:
65.6%) rather than exact-match. Label as "FiFTy-style", never "FiFTy", in
any table.
"""

from torch import nn


class FiftyStyleCNN(nn.Module):
    def __init__(self, n_classes: int, embed_dim: int = 32, channels: int = 128,
                 kernel_size: int = 11, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.features = nn.Sequential(
            nn.Conv1d(embed_dim, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, byte_seq):
        x = self.embedding(byte_seq).transpose(1, 2)  # (B, embed, L)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)
