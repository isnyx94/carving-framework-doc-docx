"""Why did decompress-then-classify work for PDF but not for OOXML?

The most diagnostic experiment the literature suggests. Nguyen, Tran, Ma and Sharma
(2014, 2015) reported that detecting DEFLATE data inside a compound-file fragment,
decompressing it, and only then classifying, improves fragment classification — on
**PDF**. The same architecture is a measured null in this thesis for OOXML and CFBF.

This thesis attributes its null to *structural coverage*: the evidence the front-end
needs is present in only a fraction of a percent of fragments. That explanation makes a
falsifiable prediction — **PDF coverage should be materially higher** — because PDF
embeds many small independently compressed object streams, whereas an OOXML package
holds a few large ones (typically ~21 parts, of which one dominates).

If PDF coverage is high, the divergence is explained and neither result is wrong. If PDF
coverage is comparably low, the coverage explanation is **falsified** and the null needs
a different account. Either outcome is worth having; the second would be more important.

Fairness of the comparison
--------------------------
Each format is given its own correct detection procedure rather than one applied to
both, because a mechanism that fails only because it was misapplied proves nothing:

  * **OOXML** — locate a ZIP local file header, compute the payload offset from its
    declared name and extra-field lengths, and inflate as **raw** DEFLATE (RFC 1951,
    wbits=-15), which is what ZIP stores.
  * **PDF** — locate the `stream` keyword, skip the EOL that the specification requires
    to follow it, and inflate as **zlib-wrapped** data (RFC 1950, wbits=15), which is
    what `/FlateDecode` normally stores. A raw-DEFLATE fallback is also attempted, since
    some producers emit unwrapped streams.

`endstream` contains the substring `stream`, so matches preceded by `end` are excluded;
counting them would inflate PDF's apparent stream density.

Output: results/pdf_coverage/PDF_COVERAGE_REPORT.md + pdf_coverage.json
"""

import json
import os
import zlib

import numpy as np

from classify.data.fft75 import CLASS_TO_INDEX, load_fft75_arrays

OUT_DIR = os.environ.get("PDFCOV_OUT", "results/pdf_coverage")
SIZES = tuple(int(s) for s in os.environ.get("PDFCOV_SIZES", "512,4096").split(","))
N_PER_CLASS = int(os.environ.get("PDFCOV_N", "3000"))

OOXML_CLASSES = ("docx", "xlsx", "pptx")
PDF_CLASSES = ("pdf",)
CONTROL_CLASSES = ("zip", "jpg", "doc")

LFH_MAGIC = b"PK\x03\x04"
PDF_STREAM_KW = b"stream"
PDF_MARKERS = (b"obj", b"endobj", b"stream", b"endstream", b"/FlateDecode", b"xref")

MIN_USEFUL_OUTPUT = 16  # bytes; below this a "success" carries no classifiable content


# --------------------------------------------------------------------------
# OOXML: ZIP local file header -> raw DEFLATE
# --------------------------------------------------------------------------

def ooxml_deflate_recovered(data: bytes) -> bool:
    """True when a ZIP local header is present and its payload inflates."""
    pos = data.find(LFH_MAGIC)
    while pos != -1:
        if pos + 30 <= len(data):
            name_len = int.from_bytes(data[pos + 26:pos + 28], "little")
            extra_len = int.from_bytes(data[pos + 28:pos + 30], "little")
            start = pos + 30 + name_len + extra_len
            if 0 < name_len <= 512 and extra_len <= 4096 and start < len(data):
                try:
                    out = zlib.decompressobj(wbits=-15).decompress(data[start:])
                    if len(out) >= MIN_USEFUL_OUTPUT:
                        return True
                except zlib.error:
                    pass
        pos = data.find(LFH_MAGIC, pos + 1)
    return False


# --------------------------------------------------------------------------
# PDF: `stream` keyword -> zlib (with raw fallback)
# --------------------------------------------------------------------------

def pdf_stream_starts(data: bytes) -> list:
    """Payload offsets following each `stream` keyword, excluding `endstream`."""
    starts = []
    pos = data.find(PDF_STREAM_KW)
    while pos != -1:
        if not (pos >= 3 and data[pos - 3:pos] == b"end"):
            q = pos + len(PDF_STREAM_KW)
            if data[q:q + 2] == b"\r\n":
                q += 2
            elif data[q:q + 1] in (b"\n", b"\r"):
                q += 1
            starts.append(q)
        pos = data.find(PDF_STREAM_KW, pos + 1)
    return starts


def pdf_deflate_recovered(data: bytes) -> bool:
    """True when a PDF stream start is present and its payload inflates."""
    for start in pdf_stream_starts(data):
        if start >= len(data):
            continue
        payload = data[start:]
        for wbits in (15, -15):  # zlib-wrapped first (normal /FlateDecode), then raw
            try:
                out = zlib.decompressobj(wbits=wbits).decompress(payload)
                if len(out) >= MIN_USEFUL_OUTPUT:
                    return True
            except zlib.error:
                continue
    return False


def pdf_structural_marker(data: bytes) -> bool:
    return any(m in data for m in PDF_MARKERS)


# --------------------------------------------------------------------------

def measure(size: int) -> dict:
    x, y = load_fft75_arrays(size, "test")
    per_class = {}
    for cls in OOXML_CLASSES + PDF_CLASSES + CONTROL_CLASSES:
        if cls not in CLASS_TO_INDEX:
            continue
        idx = np.flatnonzero(y == CLASS_TO_INDEX[cls])[:N_PER_CLASS]
        if idx.size == 0:
            continue
        ooxml_hits = pdf_hits = marker_hits = 0
        for i in idx:
            frag = x[i].tobytes()
            ooxml_hits += int(ooxml_deflate_recovered(frag))
            pdf_hits += int(pdf_deflate_recovered(frag))
            marker_hits += int(pdf_structural_marker(frag))
        per_class[cls] = {
            "n": int(idx.size),
            "ooxml_mechanism_coverage": ooxml_hits / idx.size,
            "pdf_mechanism_coverage": pdf_hits / idx.size,
            "pdf_structural_marker_rate": marker_hits / idx.size,
        }
        print(f"  {cls:5} n={idx.size} ooxml_mech={100 * ooxml_hits / idx.size:5.2f}% "
              f"pdf_mech={100 * pdf_hits / idx.size:5.2f}% "
              f"pdf_markers={100 * marker_hits / idx.size:5.2f}%", flush=True)

    def mean_over(classes, key):
        vals = [per_class[c][key] for c in classes if c in per_class]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "per_class": per_class,
        # Each format measured with its own mechanism — the like-for-like comparison.
        "ooxml_coverage_own_mechanism": mean_over(OOXML_CLASSES, "ooxml_mechanism_coverage"),
        "pdf_coverage_own_mechanism": mean_over(PDF_CLASSES, "pdf_mechanism_coverage"),
        # False-positive checks: each mechanism on classes it does not target.
        "ooxml_mechanism_on_pdf": mean_over(PDF_CLASSES, "ooxml_mechanism_coverage"),
        "pdf_mechanism_on_ooxml": mean_over(OOXML_CLASSES, "pdf_mechanism_coverage"),
        "pdf_mechanism_on_controls": mean_over(CONTROL_CLASSES, "pdf_mechanism_coverage"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"n_per_class": N_PER_CLASS, "sizes": list(SIZES), "by_size": {}}
    for size in SIZES:
        print(f"=== {size}B ===", flush=True)
        results["by_size"][str(size)] = measure(size)

    L = ["# PDF versus OOXML structural coverage — why the PDF precedent diverges", "",
         "Nguyen, Tran, Ma and Sharma (2014, 2015) report that decompress-then-classify "
         "improves fragment classification for **PDF**. The same architecture is a "
         "measured null in this thesis for OOXML and CFBF. This thesis attributes its "
         "null to structural coverage, which predicts that PDF coverage should be "
         "materially higher.", "",
         "Each format is measured with its **own** detection mechanism: ZIP local header "
         "plus raw DEFLATE for OOXML, `stream` keyword plus zlib for PDF. A mechanism "
         "that failed only through misapplication would prove nothing.", "",
         "| Fragment size | OOXML coverage (own mechanism) | PDF coverage (own mechanism) | ratio |",
         "|---|---|---|---|"]
    for size in SIZES:
        s = results["by_size"][str(size)]
        o = s["ooxml_coverage_own_mechanism"]
        p = s["pdf_coverage_own_mechanism"]
        ratio = f"{p / o:.1f}x" if o > 0 else "n/a"
        L.append(f"| {size}B | {o:.2%} | {p:.2%} | **{ratio}** |")

    L += ["", "## Per class", "",
          "| Size | Class | OOXML mechanism | PDF mechanism | PDF structural markers |",
          "|---|---|---|---|---|"]
    for size in SIZES:
        for cls, v in results["by_size"][str(size)]["per_class"].items():
            L.append(f"| {size}B | {cls} | {v['ooxml_mechanism_coverage']:.2%} | "
                     f"{v['pdf_mechanism_coverage']:.2%} | "
                     f"{v['pdf_structural_marker_rate']:.2%} |")

    L += ["", "## Interpretation", "",
          "The coverage explanation predicts a high ratio. A ratio near 1 would falsify "
          "it, and the null reported in Chapter 6 would then require a different account "
          "— which would be the more consequential outcome of the two.", "",
          "Note that this measures **availability of the evidence**, not the accuracy any "
          "classifier achieves with it. A high PDF coverage figure would establish that "
          "the mechanism has room to help for PDF; it would not by itself reproduce the "
          "published PDF result, which would require running their pipeline."]
    with open(os.path.join(OUT_DIR, "PDF_COVERAGE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT_DIR, "pdf_coverage.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n".join(L))
    print(f"\nWrote {OUT_DIR}/PDF_COVERAGE_REPORT.md")


if __name__ == "__main__":
    main()
