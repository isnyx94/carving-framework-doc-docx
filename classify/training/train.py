import contextlib

import torch
from torch import nn
from torch.utils.data import DataLoader

from classify.training.seed import set_seed


def inverse_frequency_weights(dataset, n_classes: int) -> torch.Tensor:
    """Per-class weights for CrossEntropyLoss (counteracts the real ~5:1
    doc:docx imbalance in the custom corpus, see RESEARCH_LOG.md P1 finding)."""
    counts = torch.zeros(n_classes)
    for _, label in dataset:
        counts[label] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (n_classes * counts)
    return weights


def _autocast(device: str, amp: bool):
    # bf16 autocast: safe without a GradScaler (unlike fp16) and fast on
    # Ada-generation GPUs. No-op when amp=False or on CPU.
    if amp and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_classifier(model: nn.Module, train_ds, val_ds, epochs: int, batch_size: int,
                      lr: float, seed: int, device: str, class_weights: torch.Tensor = None,
                      amp: bool = False) -> dict:
    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for byte_ids, labels in train_loader:
            byte_ids, labels = byte_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            with _autocast(device, amp):
                logits = model(byte_ids)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        history["train_loss"].append(total_loss / max(len(train_ds), 1))
        history["val_acc"].append(evaluate_accuracy(model, val_loader, device, amp=amp))
    return history


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str, amp: bool = False) -> float:
    model.eval()
    correct, total = 0, 0
    for byte_ids, labels in loader:
        byte_ids, labels = byte_ids.to(device), labels.to(device)
        with _autocast(device, amp):
            preds = model(byte_ids).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / max(total, 1)
