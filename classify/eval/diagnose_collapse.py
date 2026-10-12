"""Diagnostic sweep for the 128/8/4-backbone majority-class collapse found
in results/new_result/. Tries lower learning
rates and tracks per-epoch prediction distribution (not just final val_acc)
to see whether the model is escaping collapse late, or never moving.
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader

from classify.baselines.raw_byte_transformer import RawByteTransformerBaseline
from classify.data.custom_corpus import build_dataset
from classify.training.dataset import LABELS, FragmentDataset
from classify.training.train import inverse_frequency_weights, train_classifier


@torch.no_grad()
def pred_distribution(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    counts = Counter()
    for byte_ids, _ in loader:
        preds = model(byte_ids.to(device)).argmax(dim=1).cpu().numpy()
        for p in preds:
            counts[LABELS[p]] += 1
    return dict(counts)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset(seed=0, n_per_size=10, sizes=(512,))
    train_ds = FragmentDataset(ds["train"], fragment_size=512, oversample=True)
    val_ds = FragmentDataset(ds["val"], fragment_size=512)
    class_weights = inverse_frequency_weights(train_ds, n_classes=2)
    print("train class counts:", Counter(f["label"] for f in train_ds.fragments))

    for lr in (1e-3, 3e-4, 1e-4):
        model = RawByteTransformerBaseline(n_classes=2, max_len=512, d_model=128, n_heads=8,
                                            n_layers=4, d_ff=512).to(device)
        hist = train_classifier(model, train_ds, val_ds, epochs=8, batch_size=32, lr=lr,
                                 seed=0, device=device, class_weights=class_weights)
        dist = pred_distribution(model, val_ds, device)
        print(f"lr={lr}: val_acc_per_epoch={[round(a, 3) for a in hist['val_acc']]} final_pred_dist={dist}")


if __name__ == "__main__":
    main()
