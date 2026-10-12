"""Deflate/ZIP-OOXML structural detector for partial fragments (methodology
Sec 6.2, C1 step 1). Entropy alone cannot distinguish DEFLATE from other
high-entropy data (encrypted, already-compressed media) — it is combined
with the ZIP local-file-header marker, which is the only structural cue
available without full-file context.
"""

import math

import numpy as np

ZIP_LFH_MAGIC = b"PK\x03\x04"
HIGH_ENTROPY_THRESHOLD = 6.5  # bits/byte; deflate output is close to 8.0
LOW_ENTROPY_THRESHOLD = 4.0  # confidently not compressed below this


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


def find_zip_local_header(data: bytes) -> int:
    """Returns the offset of a ZIP LFH magic within the fragment, or -1."""
    return data.find(ZIP_LFH_MAGIC)


def deflate_likely(data: bytes) -> dict:
    """Returns a dict with the entropy score and structural marker info —
    callers (C1 dual-view encoder, eval harness) decide the threshold so the
    detector's raw signal stays inspectable rather than collapsed to a bool
    too early."""
    entropy = shannon_entropy(data)
    lfh_offset = find_zip_local_header(data)
    return {
        "entropy": entropy,
        "high_entropy": entropy >= HIGH_ENTROPY_THRESHOLD,
        "zip_lfh_offset": lfh_offset,
        "has_zip_lfh": lfh_offset >= 0,
    }


def heuristic_decision(data: bytes) -> str:
    """Quick decision from entropy + structural marker alone (design Q5):
    confidently decide the easy cases, defer the rest to the CNN refiner
    rather than guessing — this is what makes "detection-only vs
    detection+decompression" (methodology Sec 9.2) a meaningful ablation
    instead of a coin flip on ambiguous fragments.

    Returns one of "compressed", "not_compressed", "ambiguous".
    """
    signal = deflate_likely(data)
    if signal["entropy"] < LOW_ENTROPY_THRESHOLD:
        return "not_compressed"
    if signal["high_entropy"] and signal["has_zip_lfh"]:
        return "compressed"
    return "ambiguous"
