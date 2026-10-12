"""One-time conversion of FFT-75 npz archives to raw .npy files.

The 4096B train split decompresses to ~25 GB; loading it from .npz costs
minutes of decompression and full RAM residency on every run. Raw .npy
files load instantly via np.load(mmap_mode='r') with random access and no
RAM residency, which is what NumpyFragmentDataset needs. Run once per
block size:  python -m classify.data.convert_fft75_npy 4096
"""

import os
import sys

import numpy as np

from classify.data.fft75 import DEFAULT_ROOT, _find_npz


def convert(fragment_size: int):
    size_dir = os.path.join(DEFAULT_ROOT, str(fragment_size))
    for split in ("train", "val", "test"):
        x_path = os.path.join(size_dir, f"{split}_x.npy")
        y_path = os.path.join(size_dir, f"{split}_y.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"[{fragment_size}B] {split}: already converted", flush=True)
            continue
        npz_path = _find_npz(size_dir, split)
        print(f"[{fragment_size}B] {split}: decompressing {npz_path} ...", flush=True)
        with np.load(npz_path) as npz:
            x, y = npz["x"], npz["y"]
            print(f"    x {x.shape} {x.dtype}, y {y.shape} {y.dtype}; writing .npy ...", flush=True)
            np.save(x_path, x)
            np.save(y_path, y)
        print(f"    wrote {x_path} ({os.path.getsize(x_path) / 1e9:.1f} GB)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    convert(int(sys.argv[1]) if len(sys.argv) > 1 else 4096)
