"""Whole-file signature sanity check (methodology Sec 9.1 item 1).

Explicitly out of scope as a contribution (methodology Sec 4: whole-file
identification is a baseline only, not a finding) — this exists purely to
confirm the corpus's parent files are what their directory says they are,
not to be reported as a fragment-classification result.
"""

OLE2_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # CFBF magic, used by legacy .doc
ZIP_SIGNATURE = b"PK\x03\x04"  # OOXML (.docx) is a ZIP container


def whole_file_type(path: str) -> str:
    with open(path, "rb") as f:
        header = f.read(8)
    if header.startswith(OLE2_SIGNATURE):
        return "doc"
    if header.startswith(ZIP_SIGNATURE):
        return "docx"
    return "unknown"


def sanity_check_corpus(parent_labels: list) -> dict:
    """parent_labels: [(path, expected_label), ...]. Returns mismatch report."""
    mismatches = []
    for path, expected in parent_labels:
        detected = whole_file_type(path)
        if detected != expected:
            mismatches.append((path, expected, detected))
    return {"n_checked": len(parent_labels), "n_mismatch": len(mismatches), "mismatches": mismatches}
