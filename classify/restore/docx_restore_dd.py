"""Extension of the OOXML restorers to ZIP parts written with a data descriptor.

Some OOXML writers (LibreOffice among them) set general-purpose bit 3, so the local file
header carries zero sizes and a zero CRC and the real values follow the payload in a data
descriptor. The frozen restorers in `docx_restore.py` skip such parts, because their
compressed size is zero. This module adds the smallest change that handles them, without
editing the frozen code:

- B0-dd: inflate forward from the payload offset until the DEFLATE stream reports its own
  end (DEFLATE is self-terminating, so no declared size is needed).
- B2-dd: if that contiguous read reaches the end of the stream, return it. Otherwise run the
  `restore_structural` search with the unknown sizes replaced by upper bounds (4 MiB of
  compressed input, 16 MiB of output), so the search continues until the stream ends, no
  candidate extends it, a bound is reached, or the jump budget is spent. The output bound is
  needed because, without a declared size, a wrong continuation that decodes into long
  back-references of earlier XML can otherwise grow without limit. Without a declared size or CRC, B2-dd cannot label a part complete or verified.

Added 2026-10-07 as an exploratory extension, after the analysis plan of that date; it is
not part of the pre-specified experiments.
"""
import copy
import zlib

from classify.restore import docx_restore as dr

SIZE_CAP = 4 << 20    # upper bound used in place of an unknown compressed size (largest test part: 4.5 MB inflated)
OUT_CAP = 16 << 20    # stop a reassembly whose inflated output exceeds this (unknown declared size)
_UNKNOWN = 1 << 40    # stands in for an unknown uncompressed size


# A copy of docx_restore.restore_structural with the CRC checks removed. For data-descriptor
# parts the header carries no CRC, so every CRC check in the original returns False; removing
# them (and the whole-output copy each check needed) changes no result, only the run time.
from classify.restore.docx_restore import (RestoredPart, PLAUSIBILITY_FLOOR, METHOD_STORED,
                                           crc_verified, LocalFileHeader)


# --- Faster equivalents of two helpers (identical results; the originals are quadratic on
# long garbage output, which unbounded data-descriptor parts can produce). ---------------
import re as _re
_OPEN = _re.compile(r"<w:t[^>]*>")
_PRINT = bytes(range(0x20, 0x7F)) + b"\t\n\r"


def strip_xml_text(xml_bytes: bytes) -> str:
    """Same output as docx_restore.strip_xml_text, in linear time."""
    text = xml_bytes.decode("utf-8", errors="ignore")
    runs, pos = [], 0
    while True:
        m = _OPEN.search(text, pos)
        if not m:
            break
        e = text.find("</w:t>", m.end())
        if e < 0:
            break
        runs.append(text[m.end():e]); pos = e + 6
    if runs:
        return " ".join(runs)
    return _re.sub(r"<[^>]*>", " ", text)


def text_plausibility(data: bytes) -> float:
    """Same output as docx_restore.text_plausibility, counting with bytes.translate."""
    if not data:
        return 0.0
    data = bytes(data)
    ratio = (len(data) - len(data.translate(None, _PRINT))) / len(data)
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.start < len(data) - 4:
            ratio *= 0.5
    return ratio


def restore_contiguous(image, lfh):
    """Same as docx_restore.restore_contiguous, using the faster text helper."""
    payload = bytes(image[lfh.data_offset:lfh.data_offset + max(lfh.compressed_size, 1)])
    if lfh.method == METHOD_STORED:
        raw = payload[:lfh.uncompressed_size]
        return RestoredPart(lfh.name, strip_xml_text(raw), len(raw), lfh.uncompressed_size, [],
                            len(raw) >= lfh.uncompressed_size, crc_verified(raw, lfh))
    out, _ = dr._inflate_forward(payload)
    return RestoredPart(lfh.name, strip_xml_text(out), len(out), lfh.uncompressed_size, [],
                        len(out) >= lfh.uncompressed_size, crc_verified(out, lfh))


def _restore_structural_nocrc(image, lfh: LocalFileHeader, block_size: int = 512,
                       search_window: int = None, max_steps: int = None,
                       max_jumps: int = 32, jump_budget: int = 400) -> RestoredPart:
    """Reassemble the part's DEFLATE stream using the header's declared sizes and
    the decompressor as a correctness oracle.

    Structured so it can never do worse than B0, which is what makes the
    ablation interpretable:

      1. Read contiguously first, exactly as B0 does.
      2. Compare the inflated length against the header's declared
         *uncompressed* size. This is the step B0 cannot take — a
         content-agnostic carver has no way to know its output stopped early,
         so it silently reports a truncated document as a whole one.
      3. Only if the part is incomplete, search for continuation blocks: append
         the candidate block that advances inflation furthest, bounded by the
         declared compressed size, and keep the best result seen.

    The declared sizes do two jobs here — completeness test and search bound.
    Without them there is no principled stopping point, which is precisely the
    container metadata the structural view contributes.
    """
    image = bytes(image)
    if lfh.method == METHOD_STORED:
        return restore_contiguous(image, lfh)

    # Step 1-2: contiguous attempt plus the completeness check B0 lacks.
    contiguous = restore_contiguous(image, lfh)
    if contiguous.complete or lfh.uncompressed_size == 0:
        contiguous.blocks_used = [lfh.data_offset // block_size]
        return contiguous

    n_blocks = len(image) // block_size
    start_block = lfh.data_offset // block_size

    # The contiguous read is the floor this must never fall below.
    fallback_out_len = contiguous.bytes_recovered
    fallback_text = contiguous.text

    pool = list(range(n_blocks)) if search_window is None else list(range(
        max(0, start_block - search_window), min(n_blocks, start_block + search_window)))

    # Each beam entry carries a *live* decompressor. Testing a candidate is then
    # state.copy() plus one block of input, which costs O(block_size) instead of
    # re-inflating the whole growing prefix — without this, scanning the full
    # image per jump is unaffordable, and a narrow window cannot work because
    # shuffled runs place a continuation anywhere in the image.
    head = zlib.decompressobj(wbits=-15)
    try:
        out0 = head.decompress(image[lfh.data_offset:(start_block + 1) * block_size])
    except zlib.error:
        return contiguous
    consumed0 = (start_block + 1) * block_size - lfh.data_offset

    # Step budget must cover the stream's own length; a flat cap silently
    # truncated every part longer than `max_steps` blocks (a 133-block stream
    # against a 64-step limit could never complete).
    n_stream_blocks = -(-lfh.compressed_size // block_size)
    step_budget = min(max_steps, n_stream_blocks + 8) if max_steps else n_stream_blocks + 8

    state, out, used, consumed = head, bytearray(out0), [start_block], consumed0
    used_set = {start_block}  # membership is hot; a list made it O(n) per candidate
    best_out, best_used = bytearray(out0), [start_block]
    jumps_spent = 0

    for _ in range(step_budget):
        if consumed >= lfh.compressed_size:
            break
        adjacent = used[-1] + 1

        # Fast path: the run continues. Inside a fragmentation run consecutive
        # blocks are consecutive file bytes, so this resolves most steps with a
        # single trial and no scan.
        advanced = False
        if 0 <= adjacent < n_blocks and adjacent not in used_set:
            trial = state.copy()
            try:
                new = trial.decompress(image[adjacent * block_size:(adjacent + 1) * block_size])
            except zlib.error:
                new = b""
            if new and text_plausibility(new) >= PLAUSIBILITY_FLOOR and len(out) + len(new) <= OUT_CAP:
                state, out = trial, out + new
                used.append(adjacent)
                used_set.add(adjacent)
                consumed += block_size
                advanced = True

        if advanced:
            if len(out) > len(best_out):
                best_out, best_used = bytearray(out), list(used)
            continue

        # Slow path: the run ended, so find where the stream jumped to. Scanned
        # at most `max_jumps` times per part — an unbounded rescan every step is
        # what made whole-image search unaffordable.
        if jumps_spent >= max_jumps:
            break
        jumps_spent += 1
        best_score, best_pick = 0.0, None
        # Nearest-first, capped at `jump_budget` trials. Testing a candidate
        # costs a decompressor state copy (~1 ms), so an uncapped scan of the
        # whole image runs to ~35k copies per part and is unaffordable; the cap
        # is the search budget and is reported with the results. Nearest-first
        # is the standard carver prior and costs nothing to impose.
        ranked = sorted((b for b in pool if b not in used_set and b != adjacent),
                        key=lambda b: abs(b - used[-1]))[:jump_budget]
        for b in ranked:
            trial = state.copy()
            try:
                new = trial.decompress(image[b * block_size:(b + 1) * block_size])
            except zlib.error:
                continue
            if not new:
                continue
            quality = text_plausibility(new)
            if quality < PLAUSIBILITY_FLOOR:
                continue
            score = len(new) * quality
            if score > best_score:
                best_score, best_pick = score, (trial, new, b)

        if best_pick is None or len(out) + len(best_pick[1]) > OUT_CAP:
            break  # nothing extends the stream (or the output cap is reached)
        state, out, b = best_pick[0], out + best_pick[1], best_pick[2]
        used.append(b)
        used_set.add(b)
        consumed += block_size
        if len(out) > len(best_out):
            best_out, best_used = bytearray(out), list(used)

    if len(best_out) >= fallback_out_len:
        text, recovered, used = strip_xml_text(bytes(best_out)), len(best_out), best_used
        verified = crc_verified(bytes(best_out), lfh)
    else:
        text, recovered, used = fallback_text, fallback_out_len, [start_block]
        verified = contiguous.crc_verified
    return RestoredPart(lfh.name, text, recovered, lfh.uncompressed_size, used,
                        recovered >= lfh.uncompressed_size > 0, verified)



def needs_dd(lfh) -> bool:
    return bool(lfh.flags & dr.FLAG_DATA_DESCRIPTOR) and lfh.compressed_size == 0


def _bounded(lfh, image_len):
    b = copy.copy(lfh)
    b.compressed_size = max(1, min(SIZE_CAP, image_len - lfh.data_offset))
    b.uncompressed_size = _UNKNOWN
    return b


def _inflate_to_end(payload: bytes):
    d = zlib.decompressobj(wbits=-15)
    try:
        out = d.decompress(payload)
    except zlib.error:
        return b"", False
    return out, d.eof


def restore_contiguous_any(image, lfh):
    """B0, or B0-dd for a data-descriptor part."""
    if not needs_dd(lfh):
        return dr.restore_contiguous(image, lfh)
    b = _bounded(lfh, len(image))
    out, eof = _inflate_to_end(bytes(image[b.data_offset:b.data_offset + b.compressed_size]))
    out = out[:OUT_CAP]
    return dr.RestoredPart(lfh.name, strip_xml_text(out), len(out), 0, [], eof, False)


def restore_structural_any(image, lfh, block_size=512, **kw):
    """B2, or B2-dd for a data-descriptor part."""
    if not needs_dd(lfh):
        return dr.restore_structural(image, lfh, block_size, **kw)
    first = restore_contiguous_any(image, lfh)
    if first.complete:
        return first
    part = _restore_structural_nocrc(image, _bounded(lfh, len(image)), block_size, **kw)
    part.bytes_expected, part.complete, part.crc_verified = 0, False, False
    return part


def is_restorable_part(lfh) -> bool:
    return dr._is_text_part(lfh.name) and (lfh.compressed_size > 0 or needs_dd(lfh))
