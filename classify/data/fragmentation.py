"""Fragment sampling for the DECoDE classifier (methodology Sec 8.2).

Fragments are sampled uniformly at random from file bodies, at fixed sizes
(512B / 4096B, reported separately — never averaged), with an optional
"hardest" variant that excludes the leading 512B signature region.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Iterator, Sequence

FRAGMENT_SIZES = (512, 4096)
SIGNATURE_REGION = 512


@dataclass(frozen=True)
class Fragment:
    parent_path: str
    label: str
    fragment_size: int
    offset: int
    hardest: bool
    data: bytes


def _seeded_rng_offsets(parent_path: str, seed: int, file_len: int, frag_size: int,
                         n: int, hardest: bool):
    import random
    # Per-file deterministic seed so re-running with the same global seed
    # reproduces the same offsets without storing them.
    digest = hashlib.sha256(f"{parent_path}|{seed}|{frag_size}|{hardest}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))

    lo = SIGNATURE_REGION if hardest else 0
    hi = file_len - frag_size
    if hi <= lo:
        return []
    return sorted(rng.randint(lo, hi) for _ in range(n))


def sample_fragments(parent_path: str, label: str, seed: int,
                      sizes: Sequence[int] = FRAGMENT_SIZES,
                      n_per_size: int = 10,
                      hardest: bool = False) -> Iterator[Fragment]:
    """Sample fragments from one parent file. Caller controls dedup across runs via `seed`."""
    file_len = os.path.getsize(parent_path)
    with open(parent_path, "rb") as f:
        raw = f.read()

    for frag_size in sizes:
        if file_len < frag_size:
            continue
        offsets = _seeded_rng_offsets(parent_path, seed, file_len, frag_size, n_per_size, hardest)
        for offset in offsets:
            yield Fragment(
                parent_path=parent_path,
                label=label,
                fragment_size=frag_size,
                offset=offset,
                hardest=hardest,
                data=raw[offset:offset + frag_size],
            )


def enumerate_corpus(root_dirs: dict) -> list:
    """root_dirs: {label: directory_path}. Returns [(path, label), ...]."""
    out = []
    for label, directory in root_dirs.items():
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                out.append((path, label))
    return out
