import os
import tempfile

import numpy as np
import pytest

from classify.data.fft75 import (
    CLASS_TO_INDEX, FFT75NotAvailable, SCENARIO1_LABELS, load_fft75,
)
from classify.training.dataset import FragmentDataset


def _make_fixture(root: str, fragment_size: int, n_per_class: int = 4):
    """Synthetic scenario-1-shaped npz files: a few classes, tiny N."""
    class_indices = [CLASS_TO_INDEX[c] for c in ("doc", "docx", "zip", "pdf")]
    size_dir = os.path.join(root, str(fragment_size), "extracted")
    os.makedirs(size_dir)
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        x = rng.integers(0, 256, size=(n_per_class * len(class_indices), fragment_size), dtype=np.uint8)
        y = np.repeat(class_indices, n_per_class)
        np.savez_compressed(os.path.join(size_dir, f"{split}.npz"), x=x, y=y)


def test_labels_sanity():
    assert len(SCENARIO1_LABELS) == 75
    assert len(set(SCENARIO1_LABELS)) == 75
    # spot-check the indices the methodology depends on (from labels.json)
    assert SCENARIO1_LABELS[44] == "doc"
    assert SCENARIO1_LABELS[45] == "docx"
    assert SCENARIO1_LABELS[39] == "zip"
    assert SCENARIO1_LABELS[54] == "pdf"
    assert SCENARIO1_LABELS[0] == "jpg"


def test_load_subset_and_shapes():
    with tempfile.TemporaryDirectory() as root:
        _make_fixture(root, fragment_size=512)
        data = load_fft75(512, subset=("doc", "docx", "zip"), root_dir=root)
        assert set(data.keys()) == {"train", "val", "test"}
        labels_seen = {f["label"] for f in data["train"]}
        assert labels_seen == {"doc", "docx", "zip"}  # pdf filtered out
        for f in data["train"]:
            assert len(f["data"]) == 512
            assert f["fragment_size"] == 512


def test_max_per_class_cap_deterministic():
    with tempfile.TemporaryDirectory() as root:
        _make_fixture(root, fragment_size=512, n_per_class=10)
        a = load_fft75(512, subset=("doc",), root_dir=root, max_per_class=3, seed=7)
        b = load_fft75(512, subset=("doc",), root_dir=root, max_per_class=3, seed=7)
        assert len(a["train"]) == 3
        assert [f["data"] for f in a["train"]] == [f["data"] for f in b["train"]]


def test_wrong_block_size_rejected():
    with tempfile.TemporaryDirectory() as root:
        _make_fixture(root, fragment_size=512)
        # point the 4096 loader at 512-sized data by copying the dir name
        os.rename(os.path.join(root, "512"), os.path.join(root, "4096"))
        with pytest.raises(FFT75NotAvailable, match="block size"):
            load_fft75(4096, subset=("doc",), root_dir=root)


def test_missing_data_raises_with_instructions():
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(FFT75NotAvailable, match="ieee-dataport"):
            load_fft75(512, root_dir=root)


def test_fragment_dataset_integration():
    with tempfile.TemporaryDirectory() as root:
        _make_fixture(root, fragment_size=512)
        data = load_fft75(512, subset=("doc", "docx", "zip", "pdf"), root_dir=root)
        labels = ["doc", "docx", "zip", "pdf"]
        ds = FragmentDataset(data["train"], fragment_size=512, labels=labels)
        byte_ids, label_idx = ds[0]
        assert byte_ids.shape == (512,)
        assert 0 <= label_idx < len(labels)