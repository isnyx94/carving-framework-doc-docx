"""Corpus-integrity guards.

Discovered 2026-07-30 while validating the CFBF adapter: the repo's doc/
directory does not contain a single genuine OLE2/CFBF file. All 877 entries are
ZIP archives (175 named *.doc, 527 named *.raw) or non-document binary data
(175 *.raw). docx/ is 175 genuine ZIP files.

Two consequences, both of which these tests pin down:

1. The "custom doc/docx corpus" pilot was never a doc-vs-docx experiment. The
   class labelled `doc` is ZIP content, the same container as `docx`, which is
   the likely reason docx-F1 sat at ~0.28 no matter how the model was trained.
   The FFT-75 results are unaffected — those use the public benchmark's real
   doc/docx fragments.

2. deflate_label assigned is_deflate=False to every `doc` fragment on the
   assumption that .doc means CFBF (which is never DEFLATE-compressed). Since
   those files are actually ZIP, that label was systematically wrong for
   compressed fragments, and the DecompressionConfidenceCNN refiner was trained
   against it — a detector taught to answer "not compressed" for compressed
   data, which is exactly the ambiguous band C1 relies on it for.

These tests assert *content-derived* facts, never extensions, so the same class
of error cannot recur silently.
"""

import glob
import os

import pytest

from classify.data.custom_corpus import DEFAULT_ROOTS
from classify.modules.c1_compound.cfbf_adapter import CFBF_SIGNATURE

ZIP_MAGIC = b"PK\x03\x04"


def _magic(path, n=8):
    with open(path, "rb") as f:
        return f.read(n)


def _all_files(root):
    return [p for p in glob.glob(os.path.join(root, "*")) if os.path.isfile(p)]


def test_deflate_labels_follow_content_not_extension():
    """The core fix: a fragment's is_deflate label must be derived from what the
    parent file actually contains, so a ZIP file named *.doc is labelled as the
    ZIP it is."""
    from classify.data.deflate_label import deflate_byte_ranges, is_zip_container

    doc_root = DEFAULT_ROOTS["doc"]
    if not os.path.isdir(doc_root):
        pytest.skip("doc/ corpus not present")
    zip_named_doc = [p for p in _all_files(doc_root)
                     if _magic(p, 4) == ZIP_MAGIC and p.lower().endswith(".doc")]
    if not zip_named_doc:
        pytest.skip("no ZIP-content .doc files present")

    path = zip_named_doc[0]
    assert is_zip_container(path), "content check must see through the .doc extension"
    ranges = deflate_byte_ranges(path)
    assert ranges, (
        f"{path} is a ZIP archive, so it must yield DEFLATE ranges regardless of "
        "its .doc extension — labelling it as CFBF is what broke the refiner")


def test_corpus_label_matches_container_technology():
    """Fails loudly if a class directory's files are not the container the class
    name claims. Skips (rather than fails) when a directory is absent, so the
    suite still runs on a fresh clone."""
    findings = []
    for label, root in DEFAULT_ROOTS.items():
        if not os.path.isdir(root):
            continue
        files = _all_files(root)
        if not files:
            continue
        n_cfbf = sum(1 for p in files if _magic(p) == CFBF_SIGNATURE)
        n_zip = sum(1 for p in files if _magic(p, 4) == ZIP_MAGIC)
        findings.append((label, len(files), n_cfbf, n_zip))

    if not findings:
        pytest.skip("no custom corpus present")

    for label, total, n_cfbf, n_zip in findings:
        if label == "doc":
            # Documents the known-bad state. When a genuine CFBF corpus is added
            # this assertion flips to n_cfbf > 0 and the pilot can be rerun.
            assert n_cfbf == 0, (
                f"doc/ now contains {n_cfbf} genuine CFBF files — the custom-corpus "
                "pilot and the refiner labels can and should be regenerated; update "
                "this test and see test_corpus_integrity docstring")
        if label == "docx":
            assert n_zip == total, f"docx/ should be all ZIP, got {n_zip}/{total}"


def test_no_cfbf_available_in_custom_corpus_is_explicit():
    """The CFBF adapter cannot be evaluated on the custom corpus. Anything that
    needs real OLE2 data must use FFT-75's doc/xls/ppt classes instead; this
    test exists so that constraint is discoverable from the test suite rather
    than only from the research log."""
    roots = [r for r in DEFAULT_ROOTS.values() if os.path.isdir(r)]
    if not roots:
        pytest.skip("no custom corpus present")
    total_cfbf = sum(1 for root in roots for p in _all_files(root)
                     if _magic(p) == CFBF_SIGNATURE)
    assert total_cfbf == 0, (
        "genuine CFBF files appeared in the custom corpus — CFBF evaluation no "
        "longer has to rely solely on FFT-75")
