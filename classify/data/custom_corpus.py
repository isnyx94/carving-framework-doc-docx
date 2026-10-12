"""Custom doc/docx corpus (methodology Sec 8.1): the 877 .doc + 175 .docx
files already present in this repo's doc/ and docx/ directories.
"""

import os

from .deflate_label import deflate_byte_ranges, fragment_is_deflate
from .fragmentation import FRAGMENT_SIZES, enumerate_corpus, sample_fragments
from .splits import split_parent_files

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOTS = {
    "doc": os.path.join(REPO_ROOT, "doc"),
    "docx": os.path.join(REPO_ROOT, "docx"),
}


def build_dataset(seed: int, n_per_size: int = 10, hardest: bool = False,
                   roots: dict = None, sizes=FRAGMENT_SIZES) -> dict:
    """Returns {"train": [...], "val": [...], "test": [...]} of dicts with
    keys: parent_path, label, fragment_size, offset, hardest, data, is_deflate.

    `is_deflate` is derived from each parent file's actual container type, not
    from its class label. The earlier version computed DEFLATE ranges only when
    label == "docx", on the assumption that a `doc`-labelled file is OLE2/CFBF
    and therefore never DEFLATE-compressed. That assumption is false for this
    corpus: doc/ holds ZIP archives named *.doc (classify/tests/
    test_corpus_integrity.py), so every one of their compressed fragments was
    labelled is_deflate=False and the DecompressionConfidenceCNN refiner was
    trained against those wrong labels — a detector taught to say "not
    compressed" about compressed bytes, in precisely the ambiguous band C1
    defers to it for.

    deflate_byte_ranges() now returns [] for non-ZIP parents by itself, so a
    genuine CFBF file still yields all-False labels, which was the original
    intent (Sec 8.1: the detector should also correctly say *no* on OLE2).
    """
    roots = roots or DEFAULT_ROOTS
    parent_labels = enumerate_corpus(roots)
    if not parent_labels:
        raise FileNotFoundError(f"No files found under {roots!r}")

    split_assignment = split_parent_files(parent_labels, seed=seed)

    out = {"train": [], "val": [], "test": []}
    for parent_path, label in parent_labels:
        deflate_ranges = deflate_byte_ranges(parent_path)
        split_name = split_assignment[parent_path]
        for frag in sample_fragments(parent_path, label, seed=seed, sizes=sizes,
                                      n_per_size=n_per_size, hardest=hardest):
            out[split_name].append({
                "parent_path": frag.parent_path,
                "label": frag.label,
                "fragment_size": frag.fragment_size,
                "offset": frag.offset,
                "hardest": frag.hardest,
                "data": frag.data,
                "is_deflate": fragment_is_deflate(frag.offset, frag.fragment_size, deflate_ranges),
            })
    return out
