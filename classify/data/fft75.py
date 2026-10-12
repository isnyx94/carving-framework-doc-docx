"""FFT-75 loader (methodology Sec 8.1, primary benchmark).

Dataset: FiFTy / FFT-75 (Mittal, Korus, Memon — IEEE TIFS 2020/2021),
scenario 1 (75 classes), 512B and 4096B variants, distributed as
train/val/test .npz archives with arrays x (uint8 blocks, shape
(N, block_size)) and y (int labels, shape (N,)).

Source (login-gated, free IEEE account):
https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset
Download 512_1.tar.gz (3.18 GB) and optionally 4k_1.tar.gz (24.18 GB),
extract under data/fft75/512/ and data/fft75/4096/ — the loader searches
recursively for train.npz / val.npz / test.npz, so the inner tarball
layout doesn't matter. See FFT75_DOWNLOAD.md.

Label order below is copied verbatim from the FiFTy repo's
fifty/utilities/labels.json scenario "1" entry (the order the models and
dataset actually use). NOTE: it differs from the README table (jpg is
index 0 here, not 11) — trust labels.json, and never hardcode indices;
use CLASS_TO_INDEX.

Split caveat (document in any write-up): FFT-75's published train/val/test
split is at *fragment* level, not parent-file level, so it does not meet
this project's own no-parent-leakage rule (Sec 8.4). We use the published
split anyway for cross-paper comparability — that is the entire point of a
shared benchmark — but per-file-split claims must come from the custom
corpus, not FFT-75.
"""

import os

import numpy as np

# Verbatim from fifty/utilities/labels.json, key "1".
SCENARIO1_LABELS = [
    "jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf",
    "rw2", "3fr", "tiff", "heic", "bmp", "gif", "png", "ai", "eps", "psd",
    "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar", "msi",
    "dmg", "7z", "bz2", "deb", "gz", "pkg", "rar", "rpm", "xz", "zip",
    "exe", "mach-o", "elf", "dll", "doc", "docx", "key", "ppt", "pptx",
    "xls", "xlsx", "djvu", "epub", "mobi", "pdf", "md", "rtf", "txt",
    "tex", "json", "html", "xml", "log", "csv", "aiff", "flac", "m4a",
    "mp3", "ogg", "wav", "wma", "pcap", "ttf", "dwg", "sqlite",
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(SCENARIO1_LABELS)}

# The methodology's compound/confusable subset (Sec 4): doc/docx and their
# confusable neighbours.
COMPOUND_SUBSET = ("doc", "docx", "xls", "xlsx", "ppt", "pptx", "zip", "pdf", "jpg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(REPO_ROOT, "data", "fft75")


class FFT75NotAvailable(RuntimeError):
    pass


def _find_npz(root_dir: str, split: str) -> str:
    for dirpath, _, filenames in os.walk(root_dir):
        if f"{split}.npz" in filenames:
            return os.path.join(dirpath, f"{split}.npz")
    raise FFT75NotAvailable(
        f"{split}.npz not found under {root_dir!r}. Download scenario-1 FFT-75 from "
        f"https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset "
        f"(free IEEE login) and extract there — see FFT75_DOWNLOAD.md."
    )


def load_fft75_arrays(fragment_size: int, split: str, root_dir: str = None):
    """Memory-lean path for benchmark-scale runs: returns (x, y) numpy arrays
    (uint8 (N, block), int labels (N,)).

    Prefers raw .npy files (written once by classify.data.convert_fft75_npy),
    loaded with mmap_mode='r' — random access with no RAM residency, which
    matters at 4096B where the train split is ~25 GB decompressed. Falls back
    to full npz decompression when the .npy files are absent (fine at 512B).
    """
    size_dir = os.path.join(root_dir or DEFAULT_ROOT, str(fragment_size))
    x_npy = os.path.join(size_dir, f"{split}_x.npy")
    y_npy = os.path.join(size_dir, f"{split}_y.npy")
    if os.path.exists(x_npy) and os.path.exists(y_npy):
        x = np.load(x_npy, mmap_mode="r")
        y = np.load(y_npy)  # labels are small; keep in RAM
    else:
        path = _find_npz(size_dir, split)
        with np.load(path) as npz:
            x, y = npz["x"], npz["y"]
    if x.shape[1] != fragment_size:
        raise FFT75NotAvailable(
            f"{split} data has block size {x.shape[1]}, expected {fragment_size}")
    return x, y


def load_fft75(fragment_size: int, subset=COMPOUND_SUBSET, root_dir: str = None,
                splits=("train", "val", "test"), max_per_class: int = None,
                seed: int = 0) -> dict:
    """Returns {"train": [...], "val": [...], "test": [...]} of fragment dicts
    compatible with classify.training.dataset.FragmentDataset:
    keys data/label/fragment_size/parent_path/offset/hardest/is_deflate.

    subset: class names to keep (None = all 75).
    max_per_class: optional per-class-per-split cap (deterministic subsample,
    for compute-bounded runs) — record it in any reported result.
    """
    size_dir = os.path.join(root_dir or DEFAULT_ROOT, str(fragment_size))
    keep = None if subset is None else {CLASS_TO_INDEX[name] for name in subset}
    rng = np.random.default_rng(seed)

    out = {}
    for split in splits:
        path = _find_npz(size_dir, split)
        with np.load(path) as npz:
            x, y = npz["x"], npz["y"]
        if x.shape[1] != fragment_size:
            raise FFT75NotAvailable(
                f"{path} has block size {x.shape[1]}, expected {fragment_size} — "
                f"wrong scenario tarball extracted into {size_dir!r}?"
            )
        fragments = []
        for class_idx in (sorted(keep) if keep is not None else range(len(SCENARIO1_LABELS))):
            rows = np.flatnonzero(y == class_idx)
            if max_per_class is not None and len(rows) > max_per_class:
                rows = rng.choice(rows, size=max_per_class, replace=False)
                rows.sort()
            label = SCENARIO1_LABELS[class_idx]
            for r in rows:
                fragments.append({
                    "data": x[r].astype(np.uint8).tobytes(),
                    "label": label,
                    "fragment_size": fragment_size,
                    "parent_path": f"fft75/{split}",  # fragment-level split; no parent identity
                    "offset": -1,
                    "hardest": False,
                    "is_deflate": False,  # unknown for FFT-75; do not use its deflate labels
                })
        out[split] = fragments
    return out
