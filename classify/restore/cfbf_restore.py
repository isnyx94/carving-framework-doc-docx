"""CFBF / OLE2 (.doc) text restoration — the binary-compound counterpart of
docx_restore.py, and the second instantiation the framework claim needs.

The structural asymmetry that makes this more than a repeat of the DOCX chapter
-----------------------------------------------------------------------------
In OPC/DOCX a part's compressed bytes are *contiguous* within the file, so a
linear read from the local file header recovers the whole part whenever the file
itself is unfragmented. That is why B0 reaches 0.997 recall at basic severity
under the realistic fragmentation model: with nothing broken, there is nothing
for structure-awareness to fix.

CFBF is different in kind. A stream's sectors form a **FAT linked list** and may
be stored in any order *inside an intact file*. A linear read is therefore wrong
even at zero disk fragmentation. Two further consequences:

  * B0 cannot know where the WordDocument stream ends, so it runs on into other
    streams (SummaryInformation, formatting tables, embedded objects) and then
    into neighbouring files in the image — which costs precision, not just
    recall.
  * Text in CFBF is stored **uncompressed**, so any surviving sector yields
    readable text. DEFLATE is all-or-nothing from the stream start. The two
    formats should therefore show opposite partial-recovery profiles under the
    same framework, which is a substantive RQ2 finding rather than a repetition.

What the two conditions do
--------------------------
  B0 `restore_contiguous_doc`  — locate the CFBF signature, read forward over a
      bounded span, extract every text run found. A content-agnostic carver's
      ceiling.
  B2 `restore_structural_doc`  — parse the header, walk the FAT to the directory
      chain, read directory entries to find the WordDocument entry's starting
      sector and *declared size*, follow that stream's FAT chain, and extract
      text from those sectors only.

Text extraction is deliberately shared between the reference and both conditions
(`extract_word_text`). It is a heuristic printable-run extractor, not a full
[MS-DOC] piece-table implementation, so it does not reproduce document order
perfectly. That is acceptable *because it is applied identically to ground truth
and to recovered bytes*: the comparison isolates sector recovery, which is the
research question, rather than the quality of a Word text parser. Stated as a
limitation rather than hidden.
"""

import re

CFBF_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
HEADER_LEN = 512
DIR_ENTRY_SIZE = 128
FREESECT = 0xFFFFFFFF
ENDOFCHAIN = 0xFFFFFFFE
MAX_CHAIN = 1 << 20  # guards against a corrupt FAT producing an endless walk

# Header field offsets ([MS-CFB]).
OFF_SECTOR_SHIFT = 0x1E
OFF_MINI_SHIFT = 0x20
OFF_N_FAT_SECTORS = 0x2C
OFF_FIRST_DIR = 0x30
OFF_MINI_CUTOFF = 0x38
OFF_FIRST_DIFAT = 0x44
OFF_N_DIFAT = 0x48
OFF_DIFAT_ARRAY = 0x4C
N_DIFAT_IN_HEADER = 109

# Directory entry field offsets.
E_NAME_LEN = 0x40
E_OBJ_TYPE = 0x42
E_START_SECTOR = 0x74
E_STREAM_SIZE = 0x78

TARGET_STREAM = b"WordDocument"

# A run must be this long to count as text. Short printable runs in a .doc are
# overwhelmingly style names, field codes and GUID fragments rather than prose.
MIN_UTF16_RUN = 12
MIN_ASCII_RUN = 20

_UTF16_RUN = re.compile(rb"(?:[\x20-\x7e\x09\x0a\x0d]\x00){%d,}" % MIN_UTF16_RUN)
_ASCII_RUN = re.compile(rb"[\x20-\x7e\x09\x0a\x0d]{%d,}" % MIN_ASCII_RUN)


def extract_word_text(stream: bytes) -> str:
    """Readable text from (possibly partial) WordDocument stream bytes.

    Handles both storage forms Word uses: UTF-16LE (printable bytes alternating
    with NULs) and 8-bit cp1252. Applied identically to reference and recovered
    bytes — see module docstring.
    """
    parts = []
    for match in _UTF16_RUN.findall(stream):
        parts.append(match.decode("utf-16-le", errors="ignore"))
    for match in _ASCII_RUN.findall(stream):
        parts.append(match.decode("cp1252", errors="ignore"))
    return " ".join(parts)


def reference_doc_text(path: str) -> str:
    """Ground-truth text of an intact .doc, via olefile (test/eval oracle only —
    never used by the restorers)."""
    import olefile

    with olefile.OleFileIO(path) as ole:
        if not ole.exists("WordDocument"):
            return ""
        stream = ole.openstream("WordDocument").read()
    return extract_word_text(stream)


# --------------------------------------------------------------------------
# structure parsing
# --------------------------------------------------------------------------

def _u32(data: bytes, off: int) -> int:
    return int.from_bytes(data[off:off + 4], "little")


def parse_cfbf_header(image: bytes, offset: int) -> dict:
    """Parse a CFBF header at `offset`, or None if the fields are inconsistent."""
    if offset + HEADER_LEN > len(image) or image[offset:offset + 8] != CFBF_SIGNATURE:
        return None
    sector_shift = int.from_bytes(image[offset + OFF_SECTOR_SHIFT:
                                       offset + OFF_SECTOR_SHIFT + 2], "little")
    if sector_shift not in (9, 12):  # 512B (v3) or 4096B (v4)
        return None
    sector_size = 1 << sector_shift
    difat = [_u32(image, offset + OFF_DIFAT_ARRAY + 4 * i) for i in range(N_DIFAT_IN_HEADER)]
    return {
        "offset": offset,
        "sector_size": sector_size,
        "n_fat_sectors": _u32(image, offset + OFF_N_FAT_SECTORS),
        "first_dir_sector": _u32(image, offset + OFF_FIRST_DIR),
        "mini_cutoff": _u32(image, offset + OFF_MINI_CUTOFF),
        "difat": [s for s in difat if s != FREESECT],
    }


def find_cfbf_headers(image: bytes, block_size: int = 512) -> list:
    """Every validated CFBF header in the image, with its block index."""
    found = []
    pos = image.find(CFBF_SIGNATURE)
    while pos != -1:
        header = parse_cfbf_header(image, pos)
        if header is not None:
            header["block_index"] = pos // block_size
            found.append(header)
        pos = image.find(CFBF_SIGNATURE, pos + 1)
    return found


def _sector_bytes(image: bytes, header: dict, sector: int) -> bytes:
    """Bytes of file-sector `sector`, mapped assuming the container is laid out
    contiguously from its header.

    This mapping is exactly what disk fragmentation breaks, and its failure is
    the phenomenon under study — not a bug. Both conditions share it, so neither
    is advantaged by the assumption.
    """
    start = header["offset"] + HEADER_LEN + sector * header["sector_size"]
    if start < 0 or start >= len(image):
        return b""
    return image[start:start + header["sector_size"]]


def read_fat(image: bytes, header: dict) -> list:
    """The FAT as a flat list of next-sector pointers, assembled from the DIFAT."""
    fat = []
    for fat_sector in header["difat"][:max(header["n_fat_sectors"], 0) or None]:
        data = _sector_bytes(image, header, fat_sector)
        if len(data) < 4:
            break
        fat.extend(_u32(data, i) for i in range(0, len(data) - 3, 4))
    return fat


def follow_chain(fat: list, start: int, max_sectors: int = MAX_CHAIN) -> list:
    """Walk a FAT chain from `start`, stopping at ENDOFCHAIN, a bad pointer, or a
    revisited sector (a corrupt FAT in a damaged image can form a cycle)."""
    chain = []
    seen = set()
    sector = start
    while (0 <= sector < len(fat) and sector not in seen
           and len(chain) < max_sectors):
        chain.append(sector)
        seen.add(sector)
        sector = fat[sector]
        if sector >= ENDOFCHAIN:
            break
    return chain


def find_stream_entry(image: bytes, header: dict, fat: list,
                      name: bytes = TARGET_STREAM) -> dict:
    """Locate a named stream's directory entry by walking the directory chain."""
    for dir_sector in follow_chain(fat, header["first_dir_sector"]):
        data = _sector_bytes(image, header, dir_sector)
        for start in range(0, max(0, len(data) - DIR_ENTRY_SIZE + 1), DIR_ENTRY_SIZE):
            entry = data[start:start + DIR_ENTRY_SIZE]
            name_len = int.from_bytes(entry[E_NAME_LEN:E_NAME_LEN + 2], "little")
            if not 2 <= name_len <= 64:
                continue
            entry_name = bytes(entry[:name_len - 2:2])
            if entry_name != name:
                continue
            return {
                "name": entry_name,
                "start_sector": _u32(entry, E_START_SECTOR),
                "size": int.from_bytes(entry[E_STREAM_SIZE:E_STREAM_SIZE + 8], "little"),
                "obj_type": entry[E_OBJ_TYPE],
            }
    return None


# --------------------------------------------------------------------------
# the two conditions
# --------------------------------------------------------------------------

# Bounded span for the content-agnostic read. A carver with no size information
# must guess how far to read; this is generous, since it is well above the median
# .doc size in the corpus.
B0_SPAN_BYTES = 1 << 21


class RestoredDoc:
    def __init__(self, text: str, bytes_recovered: int, bytes_expected: int,
                 sectors_used: int = 0, structure_parsed: bool = False):
        self.text = text
        self.bytes_recovered = bytes_recovered
        self.bytes_expected = bytes_expected
        self.sectors_used = sectors_used
        self.structure_parsed = structure_parsed


def restore_contiguous_doc(image, header: dict, span: int = B0_SPAN_BYTES) -> RestoredDoc:
    """B0: read forward from the container signature and take every text run.

    With no directory to consult this cannot know where WordDocument ends, so it
    also sweeps up other streams' text and, past the container, whatever file
    follows it in the image. That is the content-agnostic ceiling, and its cost
    lands on precision as much as recall.
    """
    start = header["offset"]
    payload = bytes(image[start:start + span])
    return RestoredDoc(extract_word_text(payload), len(payload), 0)


def restore_structural_doc(image, header: dict) -> RestoredDoc:
    """B2: parse header -> FAT -> directory -> WordDocument entry, follow that
    stream's FAT chain, and extract text from those sectors only, bounded by the
    entry's declared size.

    **There is no floor guarantee here, unlike the OPC restorer, and the reason is
    a property of the format rather than of this implementation.** ZIP stores a
    CRC32 per entry, so `docx_restore.restore_structural` can *verify* a
    reassembly and keep the contiguous baseline whenever verification fails —
    which is what makes B2 >= B0 provable there. CFBF stores no checksum for a
    stream at all: only a declared size in the directory entry. So a
    structure-aware restorer must trust its FAT-chain mapping with no way to
    detect that fragmentation has invalidated it, and cannot tell which of two
    candidate outputs is better without ground truth it does not have.

    The measured consequence is a precision/recall trade rather than a strict
    improvement: following the chain reads *only* WordDocument (much higher
    precision) but, when the sector mapping breaks, reads the wrong sectors and
    loses recall it cannot notice. That asymmetry between the two containers is a
    substantive RQ2 finding — the abstraction transfers, the strength of the
    benefit depends on whether the format offers an integrity check.

    Falls back to the contiguous read only when the structure cannot be parsed at
    all, or parses to no usable text.
    """
    image = bytes(image)
    fat = read_fat(image, header)
    if not fat:
        result = restore_contiguous_doc(image, header)
        return result

    entry = find_stream_entry(image, header, fat)
    if entry is None or entry["size"] == 0:
        return restore_contiguous_doc(image, header)

    chain = follow_chain(fat, entry["start_sector"])
    if not chain:
        return restore_contiguous_doc(image, header)

    collected = bytearray()
    for sector in chain:
        collected += _sector_bytes(image, header, sector)
        if len(collected) >= entry["size"]:
            break
    stream = bytes(collected[:entry["size"]])

    text = extract_word_text(stream)
    if not text:
        # Structure parsed but yielded nothing usable: fall back rather than
        # return empty.
        return restore_contiguous_doc(image, header)
    return RestoredDoc(text, len(stream), entry["size"], len(chain), True)
