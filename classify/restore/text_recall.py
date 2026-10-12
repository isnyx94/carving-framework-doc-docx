"""Text-recall metrics for restoration.

THESIS_SPINE_v2 Sec 7.3 named text recall as the secondary metric that "exposes
the current [PLACEHOLDER] problem and forces real data restoration, which your
title literally demands". It was never implemented; the reassembly track scored
only clustering agreement (ARI / V-measure), which is why a pipeline whose
output text was the literal string "[PHASE 3 RECONSTRUCTED DOCUMENT]" could
still post an ARI of 0.27. Measuring recovered *text* makes that impossible.

Token-multiset scoring, not sequence alignment: partial DEFLATE recovery yields
contiguous prose with a truncated tail, so word-level recall against the
original is the quantity of forensic interest ("how much of the document did we
get back"). Multiset (not set) counting keeps repeated words honest.
"""

import re
import zipfile
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list:
    return WORD_RE.findall(text.lower())


def reference_text(docx_path: str) -> str:
    """Ground-truth visible text of an intact .docx, from its text-bearing
    parts. Uses zipfile on the original file — the oracle, never available to
    the restorer."""
    from classify.restore.docx_restore import _is_text_part, strip_xml_text

    chunks = []
    with zipfile.ZipFile(docx_path) as z:
        for name in z.namelist():
            if _is_text_part(name):
                chunks.append(strip_xml_text(z.read(name)))
    return " ".join(chunks)


def text_scores(recovered: str, reference: str) -> dict:
    """Multiset precision / recall / F1 over word tokens.

    recall    — share of the original document's words that came back. The
                headline restoration number.
    precision — share of emitted words that are genuinely from the document;
                catches a restorer that pads output with garbage to inflate
                recall.
    """
    ref = Counter(tokenize(reference))
    rec = Counter(tokenize(recovered))
    n_ref = sum(ref.values())
    n_rec = sum(rec.values())
    overlap = sum((ref & rec).values())

    recall = overlap / n_ref if n_ref else 0.0
    precision = overlap / n_rec if n_rec else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "n_reference_tokens": n_ref, "n_recovered_tokens": n_rec,
        "n_overlap_tokens": overlap,
    }


def aggregate(per_document: list) -> dict:
    """Corpus-level summary. Reports both the macro mean (each document counts
    equally) and the token-weighted micro figure (long documents count more) —
    they diverge when recovery succeeds mainly on small files, which is a real
    failure mode worth seeing."""
    if not per_document:
        return {"n_documents": 0, "macro_recall": 0.0, "micro_recall": 0.0,
                "macro_precision": 0.0, "macro_f1": 0.0, "n_fully_recovered": 0}
    n = len(per_document)
    total_ref = sum(d["n_reference_tokens"] for d in per_document)
    total_overlap = sum(d["n_overlap_tokens"] for d in per_document)
    return {
        "n_documents": n,
        "macro_recall": sum(d["recall"] for d in per_document) / n,
        "macro_precision": sum(d["precision"] for d in per_document) / n,
        "macro_f1": sum(d["f1"] for d in per_document) / n,
        "micro_recall": (total_overlap / total_ref) if total_ref else 0.0,
        "n_fully_recovered": sum(1 for d in per_document if d["recall"] >= 0.99),
        "n_zero_recovered": sum(1 for d in per_document if d["recall"] == 0.0),
    }
