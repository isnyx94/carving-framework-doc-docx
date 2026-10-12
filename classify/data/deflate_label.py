"""Deflate-likely labelling for fragments (methodology Sec 8.1, "deflate-labelled subset").

Ground truth for "is this fragment deflate-compressed data" is derived from
the parent file's ZIP local-file-header layout (reusing the existing
zip_tokenizer LFH scan), not from attempting decompression — decompression
success/failure is itself measured by C1 (Sec 6.2), so it must not be used
to produce the label it's being evaluated against.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zip_tokenizer import tokenize_zip  # noqa: E402


ZIP_MAGIC = b"PK\x03\x04"
CFBF_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def is_zip_container(path: str) -> bool:
    """Whether the file is actually a ZIP archive, by magic bytes.

    Deliberately content-based, not extension-based. The repo's doc/ directory
    contains ZIP archives named *.doc; trusting the extension made every one of
    their DEFLATE fragments carry is_deflate=False, and the
    DecompressionConfidenceCNN was then trained on those wrong labels (see
    classify/tests/test_corpus_integrity.py).
    """
    with open(path, "rb") as f:
        return f.read(4) == ZIP_MAGIC


def is_cfbf_container(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == CFBF_MAGIC


def deflate_byte_ranges(path: str) -> list:
    """Return [(data_start, data_end), ...] for DEFLATEd ZIP entries in `path`.

    compression_method == 8 is DEFLATE per the ZIP spec; method 0 is STORED
    (no compression) and is excluded. Returns [] for non-ZIP files, so callers
    can pass any parent file without pre-checking its type.
    """
    if not is_zip_container(path):
        return []
    with open(path, "rb") as f:
        raw = f.read()
    ranges = []
    for entry in tokenize_zip(raw):
        if entry["type"] != "ZIP_LFH":
            continue
        meta = entry["metadata"]
        if meta.get("compression_method") == 8:
            ranges.append((meta["data_offset"], meta["data_end"]))
    return ranges


def fragment_is_deflate(offset: int, frag_size: int, deflate_ranges: list) -> bool:
    """A fragment counts as deflate-labelled if it overlaps a DEFLATE entry's
    byte range at all (matches the methodology's fragment-level, not
    byte-level, granularity)."""
    frag_end = offset + frag_size
    return any(offset < end and frag_end > start for start, end in deflate_ranges)
