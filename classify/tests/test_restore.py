"""Restoration pipeline tests.

The invariant that matters most is the ablation floor: B2 must never recover
less than B0, because the whole experiment reads the paired difference as the
value of the container metadata. If B2 could lose, a negative difference would
be indistinguishable from "the metadata doesn't help".

The CRC tests matter second: crc_verified is reported as a *proof* of byte-exact
reassembly, so a false positive there would be a false claim in the thesis.
"""

import os
import zipfile
import zlib

import pytest

from classify.restore import docx_restore as dr
from classify.restore.fragmenter import SEVERITIES, build_image, corpus_paths
from classify.restore.text_recall import aggregate, reference_text, text_scores, tokenize

DOCX_DIR = "docx"


def _paths(limit=4):
    paths = corpus_paths(DOCX_DIR, limit=limit)
    if not paths:
        pytest.skip("no .docx corpus present")
    return paths


# --------------------------------------------------------------------------
# local file header parsing
# --------------------------------------------------------------------------

def test_parse_local_header_matches_zipfile_metadata():
    """zipfile is the oracle: our hand-rolled header parse must agree with it on
    method, sizes and CRC for a real .docx."""
    path = _paths(1)[0]
    with open(path, "rb") as f:
        raw = f.read()
    with zipfile.ZipFile(path) as z:
        infos = {i.filename: i for i in z.infolist()}

    checked = 0
    for lfh in dr.find_local_headers(raw, block_size=512):
        info = infos.get(lfh.name)
        if info is None:
            continue
        assert lfh.method == info.compress_type
        assert lfh.compressed_size == info.compress_size
        assert lfh.uncompressed_size == info.file_size
        assert lfh.crc32 == info.CRC
        checked += 1
    assert checked > 0, "no headers cross-checked against zipfile"


def test_false_headers_rejected():
    """PK\\x03\\x04 occurs by chance inside compressed data; acting on a bogus
    header wastes the whole restoration attempt."""
    assert dr.parse_local_header(b"PK\x03\x04" + b"\x00" * 8, 0) is None  # truncated
    assert dr.parse_local_header(b"NOTAHEADER" + b"\x00" * 40, 0) is None
    # Valid magic but an impossible compression method.
    bogus = bytearray(b"PK\x03\x04" + b"\x00" * 60)
    bogus[8:10] = (99).to_bytes(2, "little")
    assert dr.parse_local_header(bytes(bogus), 0) is None


def test_data_descriptor_flag_marks_sizes_untrustworthy():
    """General-purpose bit 3 means sizes/CRC are zero in the header and stored
    after the payload. Treating those zeros as real would make every
    completeness test and CRC check meaningless."""
    lfh = dr.LocalFileHeader(0, 0, dr.METHOD_DEFLATE, 100, 200, "x.xml", 30,
                             crc32=0x1234, flags=dr.FLAG_DATA_DESCRIPTOR)
    assert not lfh.sizes_trustworthy
    assert not dr.crc_verified(b"anything", lfh)

    ok = dr.LocalFileHeader(0, 0, dr.METHOD_DEFLATE, 100, 200, "x.xml", 30,
                            crc32=0x1234, flags=0)
    assert ok.sizes_trustworthy


# --------------------------------------------------------------------------
# CRC verification — reported as proof, so false positives are unacceptable
# --------------------------------------------------------------------------

def test_crc_verified_accepts_exact_and_rejects_corrupt():
    payload = b"<w:t>hello world</w:t>" * 20
    lfh = dr.LocalFileHeader(0, 0, dr.METHOD_DEFLATE, 50, len(payload), "word/document.xml",
                             30, crc32=zlib.crc32(payload) & 0xFFFFFFFF, flags=0)
    assert dr.crc_verified(payload, lfh)
    assert not dr.crc_verified(payload[:-1], lfh)          # wrong length
    assert not dr.crc_verified(b"x" + payload[1:], lfh)     # right length, wrong bytes


def test_crc_verified_requires_nonzero_crc():
    payload = b"abc"
    lfh = dr.LocalFileHeader(0, 0, dr.METHOD_DEFLATE, 3, 3, "x", 30, crc32=0, flags=0)
    assert not dr.crc_verified(payload, lfh)


# --------------------------------------------------------------------------
# plausibility gate
# --------------------------------------------------------------------------

def test_text_plausibility_separates_xml_from_binary():
    xml = b"<w:p><w:r><w:t>ordinary document text</w:t></w:r></w:p>"
    assert dr.text_plausibility(xml) >= dr.PLAUSIBILITY_FLOOR
    assert dr.text_plausibility(bytes(range(256)) * 4) < dr.PLAUSIBILITY_FLOOR
    assert dr.text_plausibility(b"") == 0.0


def test_text_plausibility_tolerates_truncated_multibyte_tail():
    """Partial inflation cuts mid-character; that must not be treated as binary."""
    text = "normal ascii prose here and some unicode: éèê".encode("utf-8")
    assert dr.text_plausibility(text[:-1]) >= dr.PLAUSIBILITY_FLOOR


# --------------------------------------------------------------------------
# text extraction
# --------------------------------------------------------------------------

def test_strip_xml_text_handles_truncated_xml():
    full = b"<w:p><w:t>alpha</w:t><w:t>beta</w:t></w:p>"
    assert "alpha" in dr.strip_xml_text(full) and "beta" in dr.strip_xml_text(full)
    truncated = b"<w:p><w:t>alpha</w:t><w:t>bet"
    assert "alpha" in dr.strip_xml_text(truncated)  # keeps what survived


def test_reference_text_nonempty_for_real_corpus():
    for path in _paths(3):
        assert tokenize(reference_text(path)), f"{path} yielded no reference tokens"


# --------------------------------------------------------------------------
# the ablation floor invariant
# --------------------------------------------------------------------------

@pytest.mark.parametrize("severity", sorted(SEVERITIES))
def test_structural_never_recovers_less_than_contiguous(severity):
    """The load-bearing invariant: B2 contains B0's contiguous read as a floor,
    so the paired difference isolates the metadata's contribution rather than
    comparing two unrelated algorithms."""
    paths = _paths(4)
    img = build_image(paths, severity, seed=0)
    data = bytes(img.data)
    for lfh in dr.find_local_headers(data, img.block_size):
        if not dr._is_text_part(lfh.name) or lfh.compressed_size == 0:
            continue
        b0 = dr.restore_contiguous(data, lfh)
        b2 = dr.restore_structural(data, lfh, img.block_size, jump_budget=64)
        assert b2.bytes_recovered >= b0.bytes_recovered, (
            f"{lfh.name} at {severity}: B2 recovered {b2.bytes_recovered} < "
            f"B0 {b0.bytes_recovered}")


def test_unfragmented_image_recovers_full_text():
    """Sanity anchor: with one fragment per file and no noise, restoration must
    recover essentially everything. If this drops, the pipeline is broken
    independently of any fragmentation question."""
    paths = _paths(4)
    image = bytearray()
    truth = []
    from classify.restore.fragmenter import BlockTruth, FragmentedImage
    for path in paths:
        with open(path, "rb") as f:
            payload = f.read()
        for off in range(0, len(payload), 512):
            chunk = payload[off:off + 512]
            image += chunk + bytes(512 - len(chunk))
            truth.append(BlockTruth(path, off))
    img = FragmentedImage(data=image, truth=truth, block_size=512)

    from classify.eval.restore_experiment import restore_by_document
    result = restore_by_document(img, paths, "structural")
    for path in paths:
        score = text_scores(result["text"].get(path, ""), reference_text(path))
        assert score["recall"] > 0.95, f"{os.path.basename(path)} recall {score['recall']:.3f}"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_text_scores_multiset_semantics():
    assert text_scores("a a b", "a a b")["recall"] == 1.0
    # Repeated words counted as a multiset: one recovered "a" of two is half.
    half = text_scores("a", "a a")
    assert half["recall"] == 0.5
    assert text_scores("", "a b")["recall"] == 0.0
    assert text_scores("a b", "")["recall"] == 0.0


def test_aggregate_flags_zero_and_full_recovery():
    docs = [text_scores("a b", "a b"), text_scores("", "c d")]
    agg = aggregate(docs)
    assert agg["n_documents"] == 2
    assert agg["n_fully_recovered"] == 1
    assert agg["n_zero_recovered"] == 1
    assert agg["macro_recall"] == pytest.approx(0.5)


def test_realistic_model_has_the_properties_it_claims():
    """The realistic model's justification *is* these three properties, so they are
    asserted rather than described. If any drifts, the sensitivity analysis in the
    thesis stops meaning what it says.
    """
    from classify.restore.fragmenter import (
        REALISTIC_SEVERITIES, build_image_realistic, fragmentation_stats,
    )
    paths = _paths(12)
    prev_runs = 0.0
    for severity in ("basic", "moderate", "severe"):
        stats = fragmentation_stats(build_image_realistic(paths, severity, seed=0))

        # 1. Fragments appear in increasing file-offset order (allocators append
        #    forward). This is the property uniform shuffling destroys.
        assert stats["frac_successive_in_order"] == 1.0, severity

        # 2. Run counts stay in a realistic range: most files contiguous, and
        #    bifragmentation dominant among the rest. An interleaving model with a
        #    fixed small chunk size gave 47-419 runs/file, which is a different
        #    regime rather than a harder one.
        assert 1.0 <= stats["mean_runs_per_file"] < 8.0, (
            f"{severity}: {stats['mean_runs_per_file']:.1f} runs/file is not a "
            "realistic fragmentation regime")

        # 3. Severity is monotone in fragmentation, so tiers remain comparable
        #    with the shuffled model's tiers.
        assert stats["mean_runs_per_file"] >= prev_runs, severity
        prev_runs = stats["mean_runs_per_file"]

    assert set(REALISTIC_SEVERITIES) == set(SEVERITIES), (
        "both models must expose the same severity tiers to be comparable")


def test_realistic_model_places_no_noise_blocks():
    """Gaps are filled with other documents' real data, not random bytes. That is
    deliberately *harder* than noise: a wrong continuation is genuine compressed
    document data, the exact case the plausibility gate cannot catch."""
    from classify.restore.fragmenter import build_image_realistic
    img = build_image_realistic(_paths(6), "severe", seed=0)
    assert all(t.source != "NOISE" for t in img.truth)


def test_fragmenter_ground_truth_covers_every_block():
    paths = _paths(3)
    img = build_image(paths, "moderate", seed=1)
    assert img.n_blocks == len(img.data) // img.block_size
    sources = {t.source for t in img.truth}
    assert sources - {"NOISE"}, "no real source blocks recorded"
    for path in paths:
        assert img.blocks_of(path), f"{path} contributed no blocks"
