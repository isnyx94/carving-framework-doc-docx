"""Defensible head (methodology Sec 6.2, C2): per-byte attribution +
confidence calibration, so a doc/docx decision can be inspected rather than
trusted blindly (methodology Sec 4, "Explainability for evidentiary use").

Attribution method (design Q4): Integrated Gradients (Sundararajan et al.
2017) — the IEEE TIFS standard for forensic explainability, theoretically
sound (satisfies completeness: attributions sum to logit(x) - logit(baseline)),
and works directly on byte embeddings. Both model classes expose
`encoder.embedding` and `encoder.encode_embeddings(...)` (see
byte_transformer.py) so this single implementation covers backbone_only and
+C1 without duplicating the forward path.
"""

import torch
from torch import nn


def integrated_gradients(model: nn.Module, byte_seq: torch.Tensor, target_class: int,
                          n_steps: int = 50, baseline: torch.Tensor = None) -> torch.Tensor:
    """Returns (B, L) per-byte attribution scores. baseline defaults to the
    zero embedding (Sundararajan et al.'s recommended "absence of signal"
    baseline for embeddings).

    For models exposing fused_token_embedding (C1Classifier), attribution
    interpolates the residual-gated fused embedding — the encoder's actual
    input — so the explanation covers the full decision path including the
    recovered view and the gate (alpha and the recovered view are held fixed
    at their values for the input fragment). For plain backbones it
    interpolates the raw byte embedding as before.
    """
    model.eval()
    if hasattr(model, "fused_token_embedding"):
        with torch.no_grad():
            input_embed = model.fused_token_embedding(byte_seq).detach()
    else:
        input_embed = model.encoder.embedding(byte_seq).detach()
    baseline_embed = torch.zeros_like(input_embed) if baseline is None else baseline

    total_grad = torch.zeros_like(input_embed)
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interpolated = (baseline_embed + alpha * (input_embed - baseline_embed)).requires_grad_(True)
        pooled = model.encoder.encode_embeddings(interpolated)
        logits = model.head(pooled)
        score = logits[:, target_class].sum()
        grad = torch.autograd.grad(score, interpolated)[0]
        total_grad += grad

    avg_grad = total_grad / n_steps
    attributions = (input_embed - baseline_embed) * avg_grad
    return attributions.sum(dim=-1)  # (B, L)


class TemperatureScaler(nn.Module):
    """Single-parameter calibration (Guo et al. 2017): rescales logits by a
    learned scalar T so reported confidence matches empirical accuracy
    (methodology Sec 6.2, "Calibration"). Fit post-hoc on held-out (val) logits.
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature)
        return logits / temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, n_steps: int = 200):
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=n_steps)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


def expected_calibration_error(confidences: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    """ECE (Guo et al. 2017): |empirical accuracy - mean confidence| per bin,
    weighted by bin occupancy."""
    confidences = confidences.detach().cpu()
    correct = correct.detach().cpu().float()
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean().item()
        bin_conf = confidences[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(bin_acc - bin_conf)
    return ece
