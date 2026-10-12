"""Baseline #4 (methodology Sec 9.1): the foundation backbone on raw bytes,
with no C1 front-end. This is the critical internal control — C1's measured
effect (P2) is defined relative to this model, same backbone, same seed
protocol, same fragment size.
"""

from torch import nn

from classify.modules.foundation.byte_transformer import (
    DEFAULT_D_FF, DEFAULT_D_MODEL, DEFAULT_N_HEADS, DEFAULT_N_LAYERS, ByteTransformerEncoder,
)
from classify.modules.heads.classifier_head import ClassifierHead


class RawByteTransformerBaseline(nn.Module):
    def __init__(self, n_classes: int, max_len: int, d_model: int = DEFAULT_D_MODEL,
                 n_heads: int = DEFAULT_N_HEADS, n_layers: int = DEFAULT_N_LAYERS,
                 d_ff: int = DEFAULT_D_FF, dropout: float = 0.1):
        super().__init__()
        self.encoder = ByteTransformerEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, max_len=max_len, dropout=dropout,
        )
        self.head = ClassifierHead(d_model, n_classes)

    def forward(self, byte_seq):
        pooled = self.encoder(byte_seq)
        return self.head(pooled)
