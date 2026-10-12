"""CFBF adapter tests. The structural detector is heuristic, so the tests that
matter are the ones run against *real* .doc files with olefile as an
independent oracle — synthetic fixtures can only confirm the field offsets.

olefile is a test-only dependency; classify/ never imports it at inference
time (see cfbf_adapter.py module docstring).
"""

import glob
import os

import numpy as np
import pytest

from classify.modules.c1_compound import cfbf_adapter as cfbf
from classify.modules.c1_compound.adapters import CFBFAdapter, OPCAdapter, build_adapters

DOC_DIR = "doc"
DOCX_DIR = "docx"


def _real_cfbf_paths(limit=8):
    """Files that are *actually* CFBF, by magic — not by extension.

    The repo's doc/ directory is misleading: all 877 entries are ZIP or random
    data and none is a real OLE2 container (see test_corpus_integrity.py). So
    these tests must filter on content and skip cleanly rather than silently
    "passing" against renamed DOCX files.
    """
    paths = []
    for path in sorted(glob.glob(os.path.join(DOC_DIR, "*.doc"))):
        with open(path, "rb") as f:
            if f.read(8) == cfbf.CFBF_SIGNATURE:
                paths.append(path)
        if len(paths) >= limit:
            break
    if not paths:
        pytest.skip("no genuine CFBF files in doc/ — see test_corpus_integrity.py")
    return paths


def _fft75_class_fragments(cls, size=512, limit=3000):
    """Real fragments of a real CFBF class from the public benchmark — the only
    genuine OLE2 data available in this project."""
    np_mod = pytest.importorskip("numpy")
    try:
        from classify.data.fft75 import CLASS_TO_INDEX, load_fft75_arrays
        x, y = load_fft75_arrays(size, "test")
    except Exception as exc:  # dataset not downloaded / converted
        pytest.skip(f"FFT-75 {size}B not available: {exc}")
    idx = np_mod.flatnonzero(y == CLASS_TO_INDEX[cls])[:limit]
    if idx.size == 0:
        pytest.skip(f"no {cls} fragments in FFT-75 {size}B test split")
    return [x[i].tobytes() for i in idx]


def _fragments(path, size=512, limit=None):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // size
    if limit is not None:
        n = min(n, limit)
    return [data[i * size:(i + 1) * size] for i in range(n)]


# --------------------------------------------------------------------------
# field-offset correctness (synthetic, exact)
# --------------------------------------------------------------------------

def _make_dir_entry(name: str, obj_type: int = 2, start_sector: int = 7,
                    size: int = 4096) -> bytes:
    entry = bytearray(128)
    encoded = name.encode("utf-16-le") + b"\x00\x00"
    entry[0:len(encoded)] = encoded
    entry[cfbf.OFF_NAME_LEN:cfbf.OFF_NAME_LEN + 2] = len(encoded).to_bytes(2, "little")
    entry[cfbf.OFF_OBJ_TYPE] = obj_type
    entry[cfbf.OFF_COLOR] = 1
    entry[cfbf.OFF_START_SECTOR:cfbf.OFF_START_SECTOR + 4] = start_sector.to_bytes(4, "little")
    entry[cfbf.OFF_STREAM_SIZE:cfbf.OFF_STREAM_SIZE + 8] = size.to_bytes(8, "little")
    return bytes(entry)


def test_decode_entry_name_roundtrip():
    entry = _make_dir_entry("WordDocument")
    assert cfbf.decode_entry_name(entry) == b"WordDocument"


def test_decode_entry_name_strips_office_control_prefix():
    entry = _make_dir_entry("\x05SummaryInformation")
    assert cfbf.decode_entry_name(entry) == b"SummaryInformation"


def test_directory_sector_detected_and_names_surface_as_ascii():
    sector = (_make_dir_entry("Root Entry", obj_type=5)
              + _make_dir_entry("WordDocument")
              + _make_dir_entry("1Table")
              + _make_dir_entry("Data"))
    assert cfbf.structural_role(sector) == "directory"
    decoded = cfbf.cfbf_recovered_view(sector)
    assert len(decoded) == len(sector)
    # The whole point of the decode: names become plain ASCII tokens.
    assert b"WordDocument" in decoded
    assert b"1Table" in decoded
    # ...and they were NOT plain ASCII before it.
    assert b"WordDocument" not in sector


def test_application_hint_separates_cfbf_hosts():
    doc = cfbf.cfbf_signals(_make_dir_entry("WordDocument") * 4)
    xls = cfbf.cfbf_signals(_make_dir_entry("Workbook") * 4)
    ppt = cfbf.cfbf_signals(_make_dir_entry("PowerPoint Document") * 4)
    assert doc["application_hint"] == "doc"
    assert xls["application_hint"] == "xls"
    assert ppt["application_hint"] == "ppt"


def test_fat_delta_encoding_collapses_contiguous_chain():
    chain = np.arange(1, 129, dtype="<u4").tobytes()  # v[i] = i+1, fully contiguous
    decoded = cfbf.decode_fat_sector(chain)
    vals = np.frombuffer(decoded, dtype="<u4")
    assert vals[0] == 0          # first delta is 0 by definition
    assert np.all(vals[1:] == 1)  # every subsequent step is +1


def test_fat_special_values_pass_through_verbatim():
    raw = np.array([5, 6, cfbf.ENDOFCHAIN, cfbf.FREESECT, cfbf.FATSECT] * 4,
                   dtype="<u4").tobytes()
    vals = np.frombuffer(cfbf.decode_fat_sector(raw), dtype="<u4")
    for i, v in enumerate(np.frombuffer(raw, dtype="<u4")):
        if v >= cfbf.SPECIAL_MIN:
            assert vals[i] == v, "special markers must stay recognisable"


def test_fat_sector_detected_from_chain_structure():
    chain = np.concatenate([
        np.arange(1, 100, dtype="<u4"),
        np.array([cfbf.ENDOFCHAIN] * 29, dtype="<u4"),
    ]).tobytes()
    assert cfbf.structural_role(chain) == "fat"


def test_all_transforms_preserve_length():
    for size in (512, 4096):
        for payload in (os.urandom(size),
                        _make_dir_entry("WordDocument") * (size // 128),
                        np.arange(size // 4, dtype="<u4").tobytes()):
            assert len(cfbf.cfbf_recovered_view(payload, size)) == size


# --------------------------------------------------------------------------
# graceful degradation / false positives
# --------------------------------------------------------------------------

def test_random_bytes_are_stream_data_and_pass_through_unchanged():
    payload = os.urandom(4096)
    assert cfbf.structural_role(payload) == "stream_data"
    assert cfbf.cfbf_recovered_view(payload) == payload
    assert cfbf.recovery_confidence(payload) == 0.0


def test_compressed_docx_fragments_not_misread_as_cfbf_structure():
    """A DOCX fragment is high-entropy DEFLATE; misreading it as a directory or
    FAT sector would scramble it into a misleading view. Fires on the real
    corpus, which is the case that actually matters."""
    paths = sorted(glob.glob(os.path.join(DOCX_DIR, "*.docx")))[:5]
    if not paths:
        pytest.skip("no .docx corpus present")
    misread = 0
    total = 0
    for path in paths:
        for frag in _fragments(path, 512, limit=60):
            total += 1
            if cfbf.structural_role(frag) in ("directory", "header"):
                misread += 1
    assert total > 0
    assert misread / total < 0.02, f"CFBF false-positive rate too high: {misread}/{total}"


# --------------------------------------------------------------------------
# real .doc corpus, olefile as oracle
# --------------------------------------------------------------------------

def test_real_cfbf_header_fragment_detected():
    for path in _real_cfbf_paths():
        first = _fragments(path, 512, limit=1)[0]
        assert cfbf.structural_role(first) == "header", f"{path} sector 0 not read as header"
        assert cfbf.recovery_confidence(first) == 1.0


def test_recovered_stream_names_match_olefile_listing():
    """The strong test: names carved out of raw fragments, with no container
    context, must be names olefile reports for the intact file."""
    olefile = pytest.importorskip("olefile")
    checked = 0
    for path in _real_cfbf_paths():
        if not olefile.isOleFile(path):
            continue
        with olefile.OleFileIO(path) as ole:
            truth = set()
            for entry in ole.listdir(streams=True, storages=True):
                truth.add(entry[-1].lstrip("\x01\x02\x03\x04\x05").encode())
            truth.add(b"Root Entry")

        found = set()
        for frag in _fragments(path, 512):
            signals = cfbf.cfbf_signals(frag)
            if signals["directory"]["valid_fraction"] >= cfbf.DIR_VALID_FRACTION_THRESHOLD:
                found.update(n for n in signals["directory"]["stream_names"] if n)

        if not found:
            continue
        checked += 1
        spurious = {n for n in found if n not in truth}
        assert not spurious, f"{path}: carved names absent from olefile listing: {spurious}"
    if checked == 0:
        pytest.skip("no directory sectors recovered from the sampled CFBF files")


def test_cfbf_detector_is_precise_on_real_benchmark_fragments():
    """Precision is the property this adapter must have: a false "directory"
    call rewrites genuine stream bytes into a misleading view, which is worse
    than not firing. Measured on real non-CFBF benchmark fragments, where the
    correct answer is always "no structure".

    Guards the two false-positive sources found during development: all-zero
    sectors parsing as unallocated directory entries, and 0xFF padding runs
    parsing as FREESECT-filled FAT sectors (9% of pptx fragments before the
    degeneracy check).
    """
    for cls in ("docx", "zip", "jpg"):
        frags = _fft75_class_fragments(cls, 512, limit=1500)
        fired = sum(1 for f in frags if any(r != "stream_data" for r in cfbf.sector_roles(f)))
        assert fired / len(frags) < 0.01, (
            f"CFBF detector false-positive rate on {cls} too high: {fired}/{len(frags)}")


def test_cfbf_coverage_exceeds_opc_but_stays_low():
    """Records the measured coverage relationship that the thesis argument rests
    on (classify/eval/structural_coverage.py, results/structural_coverage.json).

    Two facts, both asserted so a regression in either is caught:
      1. CFBF structure reaches MORE fragments than DEFLATE recovery does —
         the design premise, since CFBF stores structure uncompressed.
      2. It is still only a small percentage. This is the coverage ceiling that
         caps any structure-aware front-end's achievable gain, and it is why
         the C1 null is expected to replicate for CFBF rather than be fixed by
         it. An assertion, not a comment, because the whole Chapter-6 argument
         breaks if a future change makes coverage large without anyone noticing.
    """
    cfbf_ad, opc_ad = CFBFAdapter(), OPCAdapter()
    doc_frags = _fft75_class_fragments("doc", 4096, limit=2000)
    cfbf_hits = sum(1 for f in doc_frags if cfbf_ad.confidence(f) > 0)
    opc_hits = sum(1 for f in doc_frags if opc_ad.recover(f, len(f)) != f)
    coverage = cfbf_hits / len(doc_frags)
    assert cfbf_hits > opc_hits, f"cfbf={cfbf_hits} opc={opc_hits} of {len(doc_frags)}"
    assert coverage < 0.10, (
        f"CFBF coverage unexpectedly high ({coverage:.2%}) — if this is real, the "
        "coverage-ceiling argument in Chapter 6 needs revisiting, not this assertion")


# --------------------------------------------------------------------------
# adapter interface
# --------------------------------------------------------------------------

def test_build_adapters_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_adapters(("opc", "nonesuch"))


def test_adapters_preserve_length_and_are_total():
    payloads = [os.urandom(512), b"PK\x03\x04" + os.urandom(508), bytes(512)]
    for adapter in build_adapters(("opc", "cfbf")):
        for payload in payloads:
            out = adapter.recover(payload, 512)
            assert len(out) == 512
            assert 0.0 <= adapter.confidence(payload) <= 1.0
