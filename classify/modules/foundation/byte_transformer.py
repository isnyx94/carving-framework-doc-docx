"""Foundation backbone (methodology Sec 6.2): byte embedding + transformer encoder.

This is the reproducible baseline, not the contribution. Sized to match
CarveFormer (2025) for a fair backbone-controlled comparison per design
decision Q2/Q3: 6 layers, 8 heads, 256-dim embedding, learned (not
sinusoidal) positional embeddings, trained separately per fragment size.

C1 (modules/c1_compound) reuses this *same* encoder instance/class for both
configs (backbone_only and +C1) — only the input embedding it's fed differs
(raw bytes vs. residual-gated fusion of raw+recovered). Per design decision
Q3 ("Keep the backbone identical"), forward() and encode_embeddings() must
stay the single entry points so no second backbone definition can drift
from this one.
"""

import torch
from torch import nn

VOCAB_SIZE = 256  # one token per byte value
DEFAULT_D_MODEL = 256
DEFAULT_N_HEADS = 8
DEFAULT_N_LAYERS = 6
DEFAULT_D_FF = 1024


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)


class ByteTransformerEncoder(nn.Module):
    """Maps a (batch, seq_len) tensor of byte values [0, 255] to a
    (batch, d_model) pooled representation via mean-pooling over positions.

    forward() is the raw-byte path (backbone_only baseline). encode_embeddings()
    takes pre-computed token embeddings directly — this is the hook C1 uses to
    feed a fused (residual-gated) embedding through the identical encoder body
    without redefining it.
    """

    def __init__(self, d_model: int = DEFAULT_D_MODEL, n_heads: int = DEFAULT_N_HEADS,
                 n_layers: int = DEFAULT_N_LAYERS, d_ff: int = DEFAULT_D_FF,
                 max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_encoding = LearnedPositionalEmbedding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d_model = d_model

    def encode_embeddings(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(token_embeddings)
        x = self.encoder(x)
        return x.mean(dim=1)

    def forward(self, byte_seq: torch.Tensor) -> torch.Tensor:
        return self.encode_embeddings(self.embedding(byte_seq))
