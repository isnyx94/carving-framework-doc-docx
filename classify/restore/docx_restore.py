"""OPC/DOCX text restoration from a fragmented image — the "data restoration"
the thesis title names.

Everything the classification track does stops at deciding a fragment's *type*.
This module goes the rest of the way: locate a compressed OPC part in a
fragmented image, reassemble enough of its DEFLATE stream to inflate, and
return the document text that comes out. Text is the deliverable a forensic
examiner actually needs, and it is measurable against the original (see
text_recall.py), so restoration quality becomes a number rather than a claim.

Two restorers, forming the ablation:

  B0  restore_contiguous   — content-agnostic. Finds a ZIP local file header,
                             inflates forward from the data offset, keeps
                             whatever comes out before the stream breaks. This
                             is what a signature-plus-adjacency carver can do.

  B2  restore_structural   — structure-aware. Parses the local file header, so
                             it knows the part's *name* and its exact
                             compressed size, then uses the decompressor's own
                             progress as an oracle to select continuation
                             blocks from elsewhere in the image: the block that
                             lets inflation advance furthest is the block that
                             really followed. The declared compressed size
                             tells it when to stop, which is what stops it
                             wandering into unrelated data.

The B2 oracle is the contribution: DEFLATE is self-checking enough that a wrong
continuation block almost always fails immediately, so "which block extends the
stream" is a far stronger reassembly signal than byte-level adjacency. It needs
no ground truth at restore time — only the container's own metadata.
"""

import re
import zlib
from dataclasses import dataclass, field

LFH_MAGIC = b"PK\x03\x04"
LFH_FIXED_LEN = 30
METHOD_DEFLATE = 8
METHOD_STORED = 0

# Parts worth restoring for text purposes. word/document.xml holds the body;
# the header/footer parts hold the rest of the visible prose.
TEXT_PARTS = ("word/document.xml", "word/header", "word/footer", "word/footnotes")


FLAG_DATA_DESCRIPTOR = 0x08  # sizes and CRC live after the payload, not in the header


@dataclass
class LocalFileHeader:
    block_index: int
    offset_in_image: int
    method: int
    compressed_size: int
    uncompressed_size: int
    name: str
    data_offset: int  # absolute offset of the compressed payload in the image
    crc32: int = 0
    flags: int = 0

    @property
    def sizes_trustworthy(self) -> bool:
        """General-purpose bit 3 means the writer deferred crc/sizes to a data
        descriptor *after* the payload, leaving zeros in the header. Word does
        not do this, but third-party OOXML writers do, and treating those zeros
        as real sizes would make every completeness test and CRC check
        meaningless."""
        return not (self.flags & FLAG_DATA_DESCRIPTOR) and self.uncompressed_size > 0


@dataclass
class RestoredPart:
    name: str
    text: str
    bytes_recovered: int
    bytes_expected: int
    blocks_used: list = field(default_factory=list)
    complete: bool = False
    crc_verified: bool = False  # byte-exact against the header's CRC32


def parse_local_header(image: bytes, offset: int) -> LocalFileHeader:
    """Parse a ZIP local file header at `offset`, or return None if the fields
    are not self-consistent. Validation matters: PK\\x03\\x04 occurs by chance
    inside compressed data, and acting on a false header wastes the whole
    restoration attempt."""
    if offset + LFH_FIXED_LEN > len(image):
        return None
    if image[offset:offset + 4] != LFH_MAGIC:
        return None
    flags = int.from_bytes(image[offset + 6:offset + 8], "little")
    method = int.from_bytes(image[offset + 8:offset + 10], "little")
    crc32_declared = int.from_bytes(image[offset + 14:offset + 18], "little")
    csize = int.from_bytes(image[offset + 18:offset + 22], "little")
    usize = int.from_bytes(image[offset + 22:offset + 26], "little")
    name_len = int.from_bytes(image[offset + 26:offset + 28], "little")
    extra_len = int.from_bytes(image[offset + 28:offset + 30], "little")

    if method not in (METHOD_DEFLATE, METHOD_STORED):
        return None
    if not 1 <= name_len <= 512 or extra_len > 4096:
        return None
    name_bytes = image[offset + LFH_FIXED_LEN:offset + LFH_FIXED_LEN + name_len]
    if len(name_bytes) < name_len:
        return None
    try:
        name = name_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # OPC part names are printable paths; anything else means a false header.
    if not name or any(ord(c) < 0x20 for c in name):
        return None
    return LocalFileHeader(
        block_index=-1, offset_in_image=offset, method=method,
        compressed_size=csize, uncompressed_size=usize, name=name,
        data_offset=offset + LFH_FIXED_LEN + name_len + extra_len,
        crc32=crc32_declared, flags=flags,
    )


def find_local_headers(image: bytes, block_size: int = 512) -> list:
    """All validated local file headers in the image, in offset order."""
    found = []
    pos = image.find(LFH_MAGIC)
    while pos != -1:
        lfh = parse_local_header(image, pos)
        if lfh is not None:
            lfh.block_index = pos // block_size
            found.append(lfh)
        pos = image.find(LFH_MAGIC, pos + 1)
    return found


def strip_xml_text(xml_bytes: bytes) -> str:
    """Visible text from (possibly truncated) WordprocessingML.

    Tolerates a cut-off tail, because partially inflated XML is the normal case
    here and discarding it would throw away exactly the partial recovery this
    experiment is measuring.
    """
    text = xml_bytes.decode("utf-8", errors="ignore")
    # Word puts body text in <w:t> runs; prefer them when present.
    runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", text, flags=re.DOTALL)
    if runs:
        return " ".join(runs)
    return re.sub(r"<[^>]*>", " ", text)


def _inflate_forward(payload: bytes) -> tuple:
    """Inflate as much of a raw DEFLATE stream as possible.

    Returns (output, consumed). A truncated stream raises no error — the
    decompressor simply stops needing input — so partial output is normal and
    is returned rather than discarded.
    """
    d = zlib.decompressobj(wbits=-15)
    try:
        out = d.decompress(payload)
    except zlib.error:
        return b"", 0
    consumed = len(payload) - len(d.unused_data) - len(d.unconsumed_tail)
    return out, consumed


# --------------------------------------------------------------------------
# B0 — content-agnostic
# --------------------------------------------------------------------------

def restore_contiguous(image: bytes, lfh: LocalFileHeader) -> RestoredPart:
    """Inflate straight forward from the payload offset, taking the image as if
    it were contiguous. Whatever the stream yields before it breaks is the
    recovery — this is the content-agnostic carver's ceiling."""
    payload = bytes(image[lfh.data_offset:lfh.data_offset + max(lfh.compressed_size, 1)])
    if lfh.method == METHOD_STORED:
        raw = payload[:lfh.uncompressed_size]
        return RestoredPart(lfh.name, strip_xml_text(raw), len(raw),
                            lfh.uncompressed_size, [], len(raw) >= lfh.uncompressed_size,
                            crc_verified(raw, lfh))
    out, _ = _inflate_forward(payload)
    return RestoredPart(lfh.name, strip_xml_text(out), len(out), lfh.uncompressed_size,
                        [], len(out) >= lfh.uncompressed_size, crc_verified(out, lfh))


# --------------------------------------------------------------------------
# B2 — structure-aware, decompressor-guided reassembly
# --------------------------------------------------------------------------

def _try_extend(prefix: bytes, candidate: bytes) -> int:
    """How many output bytes inflating `prefix + candidate` produces. Used to
    rank candidate continuation blocks: the true successor lets the stream
    advance, a wrong one stalls almost immediately."""
    out, _ = _inflate_forward(prefix + candidate)
    return len(out)


def crc_verified(output: bytes, lfh: LocalFileHeader) -> bool:
    """Exact confirmation that a reassembled stream is byte-correct.

    This is the strongest oracle available and it costs nothing: the ZIP local
    file header already carries the CRC32 of the uncompressed part. When it
    matches, the reassembly is provably right and no further search is needed —
    which turns a heuristic "longest output wins" into a decision with a
    certificate.
    """
    if not lfh.sizes_trustworthy or lfh.crc32 == 0:
        return False
    if len(output) != lfh.uncompressed_size:
        return False
    return zlib.crc32(output) & 0xFFFFFFFF == lfh.crc32


# Text-bearing OPC parts are UTF-8 XML, so genuine output is almost entirely
# printable. Garbage from a wrongly chosen continuation block usually is not.
PLAUSIBILITY_FLOOR = 0.85


def text_plausibility(data: bytes) -> float:
    """Share of bytes consistent with UTF-8 XML text.

    The length-maximising oracle alone is not safe: a wrong block that happens
    to inflate can yield *more* bytes than the correct partial stream, and the
    restorer would then prefer garbage (measured: 23 of 200 documents regressed
    against the contiguous baseline for exactly this reason). For a partial
    stream the CRC cannot help — it covers the whole part — so plausibility of
    the decoded bytes is the available substitute.
    """
    if not data:
        return 0.0
    printable = sum(1 for b in data if 0x20 <= b < 0x7F or b in (0x09, 0x0A, 0x0D))
    ratio = printable / len(data)
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A truncated multibyte sequence at the very end is expected and fine;
        # an early failure means this is not text.
        if exc.start < len(data) - 4:
            ratio *= 0.5
    return ratio


def restore_structural(image, lfh: LocalFileHeader, block_size: int = 512,
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
            if new and text_plausibility(new) >= PLAUSIBILITY_FLOOR:
                state, out = trial, out + new
                used.append(adjacent)
                used_set.add(adjacent)
                consumed += block_size
                advanced = True
                if crc_verified(bytes(out), lfh):
                    return RestoredPart(lfh.name, strip_xml_text(bytes(out)), len(out),
                                        lfh.uncompressed_size, used, True, True)

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
            merged = out + new
            if crc_verified(bytes(merged), lfh):
                # Proof of a byte-exact reassembly; nothing can improve on it.
                return RestoredPart(lfh.name, strip_xml_text(bytes(merged)), len(merged),
                                    lfh.uncompressed_size, used + [b], True, True)
            quality = text_plausibility(new)
            if quality < PLAUSIBILITY_FLOOR:
                continue
            score = len(new) * quality
            if score > best_score:
                best_score, best_pick = score, (trial, merged, b)

        if best_pick is None:
            break  # nothing extends the stream: recovery is as complete as it gets
        state, out, b = best_pick[0], best_pick[1], best_pick[2]
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


# --------------------------------------------------------------------------
# whole-image drivers
# --------------------------------------------------------------------------

def _is_text_part(name: str) -> bool:
    return any(name.startswith(p) or name == p for p in TEXT_PARTS)


def restore_image(image, mode: str = "structural", block_size: int = 512,
                  text_parts_only: bool = True, search_window: int = None) -> list:
    """Restore every text-bearing OPC part found in the image.

    mode: "contiguous" (B0) or "structural" (B2).
    """
    image = bytes(image)
    headers = find_local_headers(image, block_size)
    restorer = {"contiguous": restore_contiguous, "structural": restore_structural}[mode]

    out = []
    for lfh in headers:
        if text_parts_only and not _is_text_part(lfh.name):
            continue
        if lfh.compressed_size == 0:
            continue
        if mode == "structural":
            out.append(restorer(image, lfh, block_size, search_window=search_window))
        else:
            out.append(restorer(image, lfh))
    return out
