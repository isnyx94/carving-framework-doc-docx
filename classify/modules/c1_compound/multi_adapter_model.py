"""Multi-adapter compound front-end: the framework core (methodology Sec 6.2,
generalised from the single-adapter C1Classifier in dual_view_model.py).

C1Classifier hard-wires one recovery path (DEFLATE decompression) and blends
it with the raw view through a per-token sigmoid gate. This model takes a
*list* of adapters (adapters.py) and blends raw + N recovered views through a
per-token softmax over views — the natural generalisation, and exactly
equivalent in expressive power to the sigmoid gate when N = 1.

    fused[i] = sum_k  w[i,k] * embed_k[i],    sum_k w[i,k] = 1

Design invariants carried over unchanged:
  * Backbone identity (Q3) — one ByteTransformerEncoder, one embedding table
    shared by every view, so nothing can drift from the backbone_only control.
  * Graceful degradation (Q2) — an adapter whose recovery failed returns the
    raw bytes, and the gate can drive its weight to zero.
  * Length preservation — every view is (B, L), so sequence length and
    per-token width match the baseline exactly.

`adapter_names=("opc",)` reproduces the Chapter 4/5 C1 configuration;
`("cfbf",)` isolates the new binary-compound adapter; `("opc", "cfbf")` is the
full two-adapter framework. Those three plus backbone_only are the ablation
levels in eval/cfbf_ablation.py.
"""

import numpy as np
import torch
from torch import nn

from classify.modules.c1_compound.adapters import build_adapters
from classify.modules.c1_compound.cnn_refiner import DecompressionConfidenceCNN
from classify.modules.c1_compound.dual_view_model import detection_confidence_batch
from classify.modules.foundation.byte_transformer import ByteTransformerEncoder


class ViewGate(nn.Module):
    """Per-token softmax over the raw view and each adapter's recovered view.

    Conditioned on all view embeddings plus the per-adapter confidences, so the
    gate can learn "trust the CFBF decode only when the detector was sure" —
    the confidence is an input, never a hard switch.
    """

    def __init__(self, d_model: int, n_adapters: int):
        super().__init__()
        self.n_views = n_adapters + 1
        self.gate_fc = nn.Linear(self.n_views * d_model + n_adapters, self.n_views)

    def forward(self, view_embeds: list, confidences: torch.Tensor) -> torch.Tensor:
        # view_embeds: list of (B, L, D), raw first. confidences: (B, n_adapters)
        b, seq_len, _ = view_embeds[0].shape
        conf = confidences.to(view_embeds[0].device).view(b, 1, -1).expand(-1, seq_len, -1)
        gate_input = torch.cat(view_embeds + [conf], dim=-1)
        return torch.softmax(self.gate_fc(gate_input), dim=-1)  # (B, L, n_views)


def adapter_recovered_batch(raw_byte_batch: torch.Tensor, adapter,
                            attempt_mask: torch.Tensor) -> torch.Tensor:
    """Applies one adapter's recovery to every fragment in the batch.

    attempt_mask short-circuits fragments the detector already rejected, so a
    batch of pure stream data costs no recovery work at all.
    """
    arr = raw_byte_batch.cpu().numpy().astype(np.uint8)
    mask = attempt_mask.cpu().numpy()
    out = np.empty(arr.shape, dtype=np.int64)
    target_len = arr.shape[1]
    for i in range(arr.shape[0]):
        raw_bytes = arr[i].tobytes()
        if mask[i]:
            recovered = adapter.recover(raw_bytes, target_len)
        else:
            recovered = raw_bytes
        out[i] = np.frombuffer(recovered, dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(out)


def adapter_confidence_batch(raw_byte_batch: torch.Tensor, adapter) -> torch.Tensor:
    arr = raw_byte_batch.cpu().numpy().astype(np.uint8)
    conf = torch.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        conf[i] = adapter.confidence(arr[i].tobytes())
    return conf


CONFIDENCE_THRESHOLD = 0.5


class MultiAdapterClassifier(nn.Module):
    def __init__(self, n_classes: int, max_len: int, adapter_names=("opc", "cfbf"),
                 d_model: int = None, n_heads: int = None, n_layers: int = None,
                 d_ff: int = None, dropout: float = 0.1,
                 detection_only: bool = False, pretrained_refiner: nn.Module = None):
        super().__init__()
        encoder_kwargs = dict(max_len=max_len, dropout=dropout)
        for name, value in [("d_model", d_model), ("n_heads", n_heads),
                            ("n_layers", n_layers), ("d_ff", d_ff)]:
            if value is not None:
                encoder_kwargs[name] = value
        self.encoder = ByteTransformerEncoder(**encoder_kwargs)
        self.adapter_names = tuple(adapter_names)
        self.adapters = build_adapters(self.adapter_names)
        # Only the OPC adapter has a learned detector (the CNN refiner, pretrained
        # against deflate ground truth). CFBF detection is exact structural
        # parsing, so it needs no learned component.
        self.refiner = pretrained_refiner if pretrained_refiner is not None else DecompressionConfidenceCNN()
        self.gate = ViewGate(self.encoder.d_model, len(self.adapters))
        self.detection_only = detection_only
        self.head = nn.Linear(self.encoder.d_model, n_classes)

    def _confidences(self, raw_view: torch.Tensor) -> torch.Tensor:
        cols = []
        for adapter in self.adapters:
            if adapter.name == "opc":
                cols.append(detection_confidence_batch(raw_view, self.refiner))
            else:
                cols.append(adapter_confidence_batch(raw_view, adapter))
        return torch.stack(cols, dim=1)  # (B, n_adapters)

    def fused_token_embedding(self, raw_view: torch.Tensor) -> torch.Tensor:
        """Gated fusion of the raw view and every adapter's recovered view —
        the encoder's actual input. Exposed for Integrated Gradients so
        attribution explains the real decision path (Chapter 5 protocol)."""
        confidences = self._confidences(raw_view)
        raw_embed = self.encoder.embedding(raw_view)
        view_embeds = [raw_embed]
        for k, adapter in enumerate(self.adapters):
            if self.detection_only:
                view_embeds.append(raw_embed)
                continue
            attempt = confidences[:, k] >= CONFIDENCE_THRESHOLD
            recovered = adapter_recovered_batch(raw_view, adapter, attempt).to(raw_view.device)
            view_embeds.append(self.encoder.embedding(recovered))
        weights = self.gate(view_embeds, confidences)  # (B, L, n_views)
        stacked = torch.stack(view_embeds, dim=-1)     # (B, L, D, n_views)
        return (stacked * weights.unsqueeze(-2)).sum(dim=-1)

    def forward(self, raw_view: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder.encode_embeddings(self.fused_token_embedding(raw_view))
        return self.head(pooled)
