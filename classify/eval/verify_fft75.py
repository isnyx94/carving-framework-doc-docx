"""Post-download verification for FFT-75 (see FFT75_DOWNLOAD.md).

Loads whichever block sizes are present, prints per-split class counts for
the compound subset, and confirms the loader's contract end-to-end
(FragmentDataset integration included).
"""

from collections import Counter

from classify.data.fft75 import COMPOUND_SUBSET, FFT75NotAvailable, load_fft75
from classify.training.dataset import FragmentDataset


def main():
    any_ready = False
    for fragment_size in (512, 4096):
        try:
            # cap per class so verification stays fast even on the full data
            data = load_fft75(fragment_size, subset=COMPOUND_SUBSET, max_per_class=5000)
        except FFT75NotAvailable as e:
            print(f"[{fragment_size}B] NOT AVAILABLE: {e}\n")
            continue

        print(f"[{fragment_size}B] loaded (counts capped at 5000/class for this check):")
        for split, fragments in data.items():
            counts = Counter(f["label"] for f in fragments)
            pretty = ", ".join(f"{k}={counts[k]}" for k in COMPOUND_SUBSET if k in counts)
            print(f"  {split:5s}: {len(fragments):7d} fragments  ({pretty})")

        ds = FragmentDataset(data["test"], fragment_size=fragment_size, labels=list(COMPOUND_SUBSET))
        byte_ids, label_idx = ds[0]
        assert byte_ids.shape[0] == fragment_size
        print(f"  FragmentDataset integration OK (test[0]: {byte_ids.shape[0]} bytes, "
              f"label={ds.labels[label_idx]})\n")
        any_ready = True

    print("READY — proceed to the P1 baseline-reproduction gate." if any_ready
          else "No FFT-75 data found. Follow FFT75_DOWNLOAD.md.")


if __name__ == "__main__":
    main()
