"""C1 classifier (methodology Sec 6.2 / design decisions Q1, Q2, Q5): the
contribution.

Detection (Q5, "Both"): the cheap heuristic in detector.py decides the easy
cases for free (very low entropy -> definitely not compressed; high entropy
+ ZIP LFH marker -> definitely compressed); only "ambiguous" fragments pay
for a forward pass through the tiny CNN refiner. detection_only=True skips
decompression entirely (recovered_view = raw_view), isolating the
detector's own contribution from the decompressor's (Sec 9.2 ablation).

Fusion (Q2, "Residual Gating", option B): rather than concatenating two
pooled vectors (which doubles head width and lets the model just learn two
separate "lanes"), a per-token gate alpha in [0, 1] blends the raw and
recovered token embeddings *before* the single shared transformer body:
    fused[i] = alpha[i] * raw_embed[i] + (1 - alpha[i]) * decomp_embed[i]
This keeps the sequence length and per-token width identical to the
backbone_only path, and the gate can learn to fall back to alpha=1 (ignore
the recovered view) when decompression failed or is garbage — exactly the
"plug-and-play, degrades gracefully" property called for in Q2.

Backbone identity (Q3): both configs instantiate the *same*
ByteTransformerEncoder class with the same d_model/n_heads/n_layers; C1 only
changes what's fed into encode_embeddings(), via the shared embedding layer
(self.dual_encoder.encoder.embedding) used for both views, so no second
embedding table is learned to drift from the baseline's.
"""

import numpy as np
import torch
from torch import nn

from classify.modules.c1_compound.cnn_refiner import DecompressionConfidenceCNN
from classify.modules.c1_compound.detector import heuristic_decision
from classify.modules.c1_compound.partial_decompress import attempt_partial_decompress
from classify.modules.foundation.byte_transformer import ByteTransformerEncoder

CNN_CONFIDENCE_THRESHOLD = 0.5


def detection_confidence_batch(raw_byte_batch: torch.Tensor, refiner: nn.Module) -> torch.Tensor:
    """Returns (B,) confidence in [0, 1] that each fragment is worth
    decompressing: heuristic resolves the easy cases for free, the CNN
    refiner only runs (as a single batched forward pass) on the ambiguous
    ones."""
    arr = raw_byte_batch.cpu().numpy().astype(np.uint8)
    confidence = torch.zeros(arr.shape[0])
    ambiguous_idx = []
    for i in range(arr.shape[0]):
        decision = heuristic_decision(arr[i].tobytes())
        if decision == "compressed":
            confidence[i] = 1.0
        elif decision == "not_compressed":
            confidence[i] = 0.0
        else:
            ambiguous_idx.append(i)

    if ambiguous_idx and refiner is not None:
        idx_tensor = torch.tensor(ambiguous_idx, device=raw_byte_batch.device)
        with torch.no_grad():
            refined = refiner(raw_byte_batch[idx_tensor])
        confidence[idx_tensor] = refined.float().cpu()  # .float(): refiner may run under bf16 autocast
    return confidence


def build_recovered_view(raw_bytes: bytes, target_len: int, should_attempt: bool) -> bytes:
    if not should_attempt:
        return raw_bytes
    result = attempt_partial_decompress(raw_bytes)
    recovered = result["output"]
    if not result["success"] or len(recovered) == 0:
        recovered = raw_bytes
    if len(recovered) < target_len:
        recovered = recovered + raw_bytes[len(recovered):target_len]
    return recovered[:target_len]


def recovered_view_batch(raw_byte_batch: torch.Tensor, attempt_mask: torch.Tensor) -> torch.Tensor:
    """raw_byte_batch: (B, L) int64. attempt_mask: (B,) bool, whether to try
    decompression for each fragment (False short-circuits to raw bytes,
    saving the zlib call when the detector already said "not compressed")."""
    out = np.empty(raw_byte_batch.shape, dtype=np.int64)
    arr = raw_byte_batch.cpu().numpy().astype(np.uint8)
    mask = attempt_mask.cpu().numpy()
    for i in range(arr.shape[0]):
        raw_bytes = arr[i].tobytes()
        recovered = build_recovered_view(raw_bytes, target_len=arr.shape[1], should_attempt=bool(mask[i]))
        out[i] = np.frombuffer(recovered, dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(out)


class ResidualGate(nn.Module):
    """Per-token gate alpha = sigmoid(Linear([raw_embed, decomp_embed, confidence]))."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_fc = nn.Linear(2 * d_model + 1, 1)

    def forward(self, raw_embed: torch.Tensor, decomp_embed: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        # raw_embed, decomp_embed: (B, L, D); confidence: (B,)
        conf_broadcast = confidence.view(-1, 1, 1).expand(-1, raw_embed.size(1), 1).to(raw_embed.device)
        gate_input = torch.cat([raw_embed, decomp_embed, conf_broadcast], dim=-1)
        return torch.sigmoid(self.gate_fc(gate_input))  # (B, L, 1)


class C1Classifier(nn.Module):
    def __init__(self, n_classes: int, max_len: int, d_model: int = None,
                 n_heads: int = None, n_layers: int = None, d_ff: int = None,
                 dropout: float = 0.1, detection_only: bool = False,
                 pretrained_refiner: nn.Module = None):
        super().__init__()
        encoder_kwargs = dict(max_len=max_len, dropout=dropout)
        for name, value in [("d_model", d_model), ("n_heads", n_heads),
                             ("n_layers", n_layers), ("d_ff", d_ff)]:
            if value is not None:
                encoder_kwargs[name] = value
        self.encoder = ByteTransformerEncoder(**encoder_kwargs)
        # Pretrained against deflate ground truth (train_refiner.py) — frozen,
        # since it's only ever called under no_grad (see detection_confidence_batch).
        self.refiner = pretrained_refiner if pretrained_refiner is not None else DecompressionConfidenceCNN()
        self.gate = ResidualGate(self.encoder.d_model)
        self.detection_only = detection_only
        self.head = nn.Linear(self.encoder.d_model, n_classes)

    def fused_token_embedding(self, raw_view: torch.Tensor) -> torch.Tensor:
        """The gated fusion of raw and recovered token embeddings — the actual
        input to the shared encoder. Exposed so attribution (Integrated
        Gradients) can explain the true decision path rather than the
        raw-view path only (Chapter 4 limitation, fixed for Chapter 5)."""
        confidence = detection_confidence_batch(raw_view, self.refiner).to(raw_view.device)

        if self.detection_only:
            recovered_view = raw_view
        else:
            attempt_mask = confidence >= CNN_CONFIDENCE_THRESHOLD
            recovered_view = recovered_view_batch(raw_view, attempt_mask).to(raw_view.device)

        raw_embed = self.encoder.embedding(raw_view)
        decomp_embed = self.encoder.embedding(recovered_view)
        alpha = self.gate(raw_embed, decomp_embed, confidence)
        return alpha * raw_embed + (1 - alpha) * decomp_embed

    def forward(self, raw_view: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder.encode_embeddings(self.fused_token_embedding(raw_view))
        return self.head(pooled)
