"""CFBF / OLE2 structural adapter — the binary-compound counterpart of the
OPC/DEFLATE front-end in detector.py + partial_decompress.py.

Why this exists
---------------
The OPC adapter's recovery step is *decompression*, and it only succeeds when
a fragment happens to begin at a DEFLATE stream start. That is rare under
uniform fragment sampling, which is the measured cause of the C1 null result. CFBF has no such precondition: its structural evidence is
*stored*, not compressed, so every fragment that overlaps a FAT sector, a
directory sector, or the header carries directly decodable structure.

The recovery step here is therefore *decoding*, not decompressing:

  FAT / DIFAT sector  -> delta-encode the uint32 chain. A contiguous sector
                         chain (v[i] = v[i-1] + 1) collapses to a run of
                         `01 00 00 00`, i.e. near-zero entropy, instead of a
                         dense array of large distinct integers.
  Directory sector    -> transliterate each 128-byte entry's UTF-16LE stream
                         name to ASCII and drop CLSID/timestamp noise. The
                         raw view forces the model to learn "W\\0o\\0r\\0d\\0";
                         the decoded view hands it the literal token
                         "WordDocument".
  Header sector       -> canonicalise the fixed fields, zero the reserved and
                         mostly-constant regions.
  anything else       -> raw passthrough (graceful degradation, same contract
                         as build_recovered_view in dual_view_model.py).

Every transform is length-preserving, so the decoded view goes through the
*same* embedding table and encoder as the raw view — the backbone-identity
requirement (design Q3) is unchanged.

Discriminative claim
--------------------
doc, xls and ppt are all CFBF containers; they differ mainly in their stream
*names* ("WordDocument"/"1Table" vs "Workbook" vs "PowerPoint Document").
Those names are exactly what the directory decode surfaces, so this adapter
targets the doc/xls/ppt block of the confusion matrix, where the raw-byte
backbone is currently weakest (ppt F1 ~0.25 at 4096B, results/ch5/
compound_partial.json seed 0).

Field offsets follow [MS-CFB]; olefile is used only as a test oracle
(classify/tests/test_cfbf.py), never at inference time.
"""

import numpy as np

CFBF_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
FIB_MAGIC = b"\xec\xa5"  # 0xA5EC little-endian, at a WordDocument stream start

# [MS-CFB] special FAT entry values.
FREESECT = 0xFFFFFFFF
ENDOFCHAIN = 0xFFFFFFFE
FATSECT = 0xFFFFFFFD
DIFSECT = 0xFFFFFFFC
SPECIAL_MIN = 0xFFFFFFF0  # 0xFFFFFFF0..FF are reserved/special

SECTOR_SIZE = 512  # CFBF v3 sector size (v4 uses 4096; v3 is what real .doc uses)
DIR_ENTRY_SIZE = 128
NAME_FIELD_BYTES = 64  # 32 UTF-16LE code units
OBJ_TYPE_VALID = (0, 1, 2, 5)  # unallocated, storage, stream, root

# Offsets within a 128-byte directory entry.
OFF_NAME_LEN = 0x40
OFF_OBJ_TYPE = 0x42
OFF_COLOR = 0x43
OFF_LEFT_SIB = 0x44
OFF_RIGHT_SIB = 0x48
OFF_CHILD = 0x4C
OFF_START_SECTOR = 0x74
OFF_STREAM_SIZE = 0x78

# Stream names that identify the host application of a CFBF container.
DOC_STREAMS = (b"WordDocument", b"1Table", b"0Table")
XLS_STREAMS = (b"Workbook", b"Book")
PPT_STREAMS = (b"PowerPoint Document", b"Current User", b"Pictures")
GENERIC_STREAMS = (b"Root Entry", b"Data", b"ObjectPool", b"CompObj",
                   b"SummaryInformation", b"DocumentSummaryInformation")

_ROLE_STREAM_DATA = "stream_data"
_ROLE_HEADER = "header"
_ROLE_FAT = "fat"
_ROLE_DIRECTORY = "directory"


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def _u32_array(data: bytes) -> np.ndarray:
    n = len(data) // 4
    if n == 0:
        return np.zeros(0, dtype=np.uint32)
    return np.frombuffer(data[: n * 4], dtype="<u4")


def fat_sector_score(data: bytes) -> dict:
    """How much this fragment looks like a FAT/DIFAT sector.

    Two independent cues, kept separate so the caller can inspect them:
    `special_fraction` (ENDOFCHAIN/FREESECT/FATSECT markers, which appear in
    essentially every real FAT) and `successor_fraction` (v[i+1] == v[i]+1,
    the signature of contiguously allocated stream chains).
    """
    vals = _u32_array(data)
    if vals.size < 8:
        return {"special_fraction": 0.0, "successor_fraction": 0.0,
                "plausible_fraction": 0.0, "n_entries": int(vals.size)}
    special = vals >= SPECIAL_MIN
    # A sector index must address somewhere inside a plausible container; real
    # .doc files are far below 2^28 sectors, so huge non-special values are a
    # strong sign this is not a FAT.
    plausible = special | (vals < (1 << 28))
    wide = vals.astype(np.int64)
    successor = (wide[1:] == wide[:-1] + 1) & ~special[1:] & ~special[:-1]
    # Degeneracy guard. A sector of uniform 0xFF bytes reads as an all-FREESECT
    # FAT, and 0xFF padding is common in unrelated formats (measured: 9% of
    # pptx fragments misdetected before this check). Such a sector also has
    # nothing to gain from decoding, since special values pass through
    # verbatim — so rejecting it costs no true positives.
    dominant = float(np.max(np.unique(vals, return_counts=True)[1]) / vals.size)
    return {
        "special_fraction": float(special.mean()),
        "successor_fraction": float(successor.mean()) if successor.size else 0.0,
        "plausible_fraction": float(plausible.mean()),
        "dominant_fraction": dominant,
        "n_entries": int(vals.size),
    }


def _entry_looks_valid(entry: bytes) -> bool:
    if len(entry) < DIR_ENTRY_SIZE:
        return False
    name_len = int.from_bytes(entry[OFF_NAME_LEN:OFF_NAME_LEN + 2], "little")
    obj_type = entry[OFF_OBJ_TYPE]
    color = entry[OFF_COLOR]
    if obj_type not in OBJ_TYPE_VALID or color not in (0, 1):
        return False
    if obj_type == 0:
        # Unallocated entry: [MS-CFB] requires the whole entry be zeroed apart
        # from the sibling/child pointers, so an all-zero name is the tell.
        return name_len == 0 and entry[:NAME_FIELD_BYTES] == b"\x00" * NAME_FIELD_BYTES
    if not 2 <= name_len <= NAME_FIELD_BYTES or name_len % 2 != 0:
        return False
    # Name must be UTF-16LE: high bytes zero for the ASCII/control range that
    # every real Office stream name uses.
    name = entry[: name_len]
    if name[-2:] != b"\x00\x00":  # terminating null code unit
        return False
    highs = name[1:name_len:2]
    return all(b == 0 for b in highs)


def directory_sector_score(data: bytes, entry_offset: int = 0) -> dict:
    """Fraction of 128-byte slots at `entry_offset` that parse as directory
    entries, plus which application's stream names were seen."""
    valid = 0
    total = 0
    names = []
    for start in range(entry_offset, len(data) - DIR_ENTRY_SIZE + 1, DIR_ENTRY_SIZE):
        entry = data[start:start + DIR_ENTRY_SIZE]
        total += 1
        if _entry_looks_valid(entry):
            valid += 1
            name = decode_entry_name(entry)
            if name:
                names.append(name)
    return {
        # A sector of pure zero bytes parses as 100% "unallocated" entries, and
        # zero-filled sectors are common in many unrelated formats — so a
        # directory call additionally requires at least one *named* entry.
        # Without this, valid_fraction alone made zero padding look like a
        # directory sector (measured false positives on docx/jpg fragments).
        "valid_fraction": (valid / total) if total and names else 0.0,
        "n_slots": total,
        "n_valid": valid,
        "n_named": len(names),
        "stream_names": names,
    }


def best_directory_alignment(data: bytes) -> tuple:
    """FFT-75 blocks are cut at fixed multiples of the block size, so a 512B
    CFBF v3 fragment is usually sector-aligned and entry_offset 0 works. This
    does not assume it: it tries every 4-byte alignment inside one entry and
    returns the best-scoring one."""
    best = (0, directory_sector_score(data, 0))
    for offset in range(4, DIR_ENTRY_SIZE, 4):
        score = directory_sector_score(data, offset)
        if score["valid_fraction"] > best[1]["valid_fraction"]:
            best = (offset, score)
    return best


def decode_entry_name(entry: bytes) -> bytes:
    """UTF-16LE stream name -> ASCII bytes. Returns b"" when unparseable.

    The leading 0x01-0x05 control prefixes Office uses ("\\x05Summary-
    Information") are stripped so the token matches its documented name.
    """
    name_len = int.from_bytes(entry[OFF_NAME_LEN:OFF_NAME_LEN + 2], "little")
    if not 2 <= name_len <= NAME_FIELD_BYTES:
        return b""
    raw = entry[: name_len - 2]  # drop terminating null code unit
    ascii_bytes = bytes(raw[0::2])
    return ascii_bytes.lstrip(b"\x00\x01\x02\x03\x04\x05")


def application_hint(stream_names) -> str:
    """Maps recovered stream names to the CFBF host application. This is a
    *diagnostic* signal for the thesis narrative and for tests; the model is
    never given this label, it only ever sees the decoded bytes."""
    joined = b" ".join(stream_names)
    for markers, app in ((DOC_STREAMS, "doc"), (XLS_STREAMS, "xls"), (PPT_STREAMS, "ppt")):
        if any(m in joined for m in markers):
            return app
    return "unknown"


def cfbf_signals(data: bytes) -> dict:
    """All raw structural signals, uncollapsed — mirrors detector.deflate_likely
    so thresholds stay the caller's decision."""
    fat = fat_sector_score(data)
    dir_offset, dir_score = best_directory_alignment(data)
    has_header = data.startswith(CFBF_SIGNATURE) or CFBF_SIGNATURE in data[:512]
    return {
        "has_cfbf_header": has_header,
        "header_offset": data.find(CFBF_SIGNATURE),
        "has_fib_magic": data[:2] == FIB_MAGIC or FIB_MAGIC in data[:16],
        "fat": fat,
        "directory": dir_score,
        "directory_entry_offset": dir_offset,
        "application_hint": application_hint(dir_score["stream_names"]),
    }


# Thresholds. Deliberately conservative: a false "directory" call would
# scramble real stream bytes into a misleading decoded view, whereas a missed
# call merely falls back to raw passthrough (the safe direction).
DIR_VALID_FRACTION_THRESHOLD = 0.5
FAT_SPECIAL_FRACTION_THRESHOLD = 0.20
FAT_SUCCESSOR_FRACTION_THRESHOLD = 0.50
FAT_PLAUSIBLE_FRACTION_THRESHOLD = 0.95
FAT_DOMINANT_FRACTION_CEILING = 0.90  # reject single-value (e.g. all-0xFF) sectors


def structural_role(data: bytes, signals: dict = None) -> str:
    """Which CFBF structure this fragment carries: header / directory / fat /
    stream_data. Checked most-specific first."""
    s = signals if signals is not None else cfbf_signals(data)
    if s["has_cfbf_header"] and s["header_offset"] == 0:
        return _ROLE_HEADER
    if s["directory"]["valid_fraction"] >= DIR_VALID_FRACTION_THRESHOLD:
        return _ROLE_DIRECTORY
    fat = s["fat"]
    if (fat["plausible_fraction"] >= FAT_PLAUSIBLE_FRACTION_THRESHOLD
            and fat["dominant_fraction"] < FAT_DOMINANT_FRACTION_CEILING
            and (fat["special_fraction"] >= FAT_SPECIAL_FRACTION_THRESHOLD
                 or fat["successor_fraction"] >= FAT_SUCCESSOR_FRACTION_THRESHOLD)):
        return _ROLE_FAT
    return _ROLE_STREAM_DATA


def _sector_confidence(sector: bytes, signals: dict = None) -> float:
    s = signals if signals is not None else cfbf_signals(sector)
    role = structural_role(sector, s)
    if role == _ROLE_HEADER:
        return 1.0
    if role == _ROLE_DIRECTORY:
        return float(min(1.0, s["directory"]["valid_fraction"]))
    if role == _ROLE_FAT:
        return float(min(1.0, 0.5 * (s["fat"]["special_fraction"]
                                     + s["fat"]["successor_fraction"])))
    return 0.0


def recovery_confidence(data: bytes, signals: dict = None,
                        sector_size: int = SECTOR_SIZE) -> float:
    """[0, 1] estimate that decoding will expose structure the raw view hides.

    Mirrors detection_confidence_batch's contract in dual_view_model.py: the
    residual gate consumes this, so it must be a graded signal rather than a
    bool. Computed as the *maximum* over sectors, not the mean — one decodable
    directory sector in an otherwise opaque 4096B fragment is exactly the case
    worth acting on, and averaging would bury it.
    """
    if len(data) <= sector_size:
        return _sector_confidence(data, signals)
    best = 0.0
    for start in range(0, len(data), sector_size):
        sector = data[start:start + sector_size]
        if len(sector) < 64:
            continue
        best = max(best, _sector_confidence(sector))
        if best >= 1.0:
            break
    return best


# --------------------------------------------------------------------------
# recovery (decode-then-classify)
# --------------------------------------------------------------------------

def decode_fat_sector(data: bytes) -> bytes:
    """Delta-encode the uint32 chain; keep special markers verbatim.

    A contiguous chain becomes `01 00 00 00` repeated — the regularity the
    raw view buries in large distinct integers. Special values (ENDOFCHAIN,
    FREESECT, FATSECT, DIFSECT) are passed through unchanged so they stay
    recognisable as the constants they are.
    """
    vals = _u32_array(data)
    if vals.size == 0:
        return data
    wide = vals.astype(np.int64)
    prev = np.concatenate([wide[:1], wide[:-1]])
    delta = ((wide - prev) & 0xFFFFFFFF).astype("<u4")
    special = vals >= SPECIAL_MIN
    delta[special] = vals[special]
    tail = data[vals.size * 4:]
    return delta.tobytes() + tail


def decode_directory_sector(data: bytes, entry_offset: int = 0) -> bytes:
    """Rewrite each 128-byte entry into a canonical, ASCII-named form.

    Layout of the decoded entry (length preserved at 128 bytes):
        0x00..0x1F  ASCII stream name, null-padded
        0x20        object type      0x21  colour flag
        0x24..0x2F  left / right sibling and child IDs (tree structure)
        0x30..0x33  starting sector location
        0x34..0x3B  stream size
        0x3C..0x7F  zeroed (was CLSID, timestamps, state bits — noise for
                    format identification, and the dominant entropy source
                    in the raw entry)
    Slots that do not parse are copied through untouched rather than zeroed,
    so a misdetected sector degrades to (mostly) the raw view.
    """
    out = bytearray(data)
    for start in range(entry_offset, len(data) - DIR_ENTRY_SIZE + 1, DIR_ENTRY_SIZE):
        entry = data[start:start + DIR_ENTRY_SIZE]
        if not _entry_looks_valid(entry):
            continue
        canonical = bytearray(DIR_ENTRY_SIZE)
        name = decode_entry_name(entry)[:32]
        canonical[0:len(name)] = name
        canonical[0x20] = entry[OFF_OBJ_TYPE]
        canonical[0x21] = entry[OFF_COLOR]
        canonical[0x24:0x28] = entry[OFF_LEFT_SIB:OFF_LEFT_SIB + 4]
        canonical[0x28:0x2C] = entry[OFF_RIGHT_SIB:OFF_RIGHT_SIB + 4]
        canonical[0x2C:0x30] = entry[OFF_CHILD:OFF_CHILD + 4]
        canonical[0x30:0x34] = entry[OFF_START_SECTOR:OFF_START_SECTOR + 4]
        canonical[0x34:0x3C] = entry[OFF_STREAM_SIZE:OFF_STREAM_SIZE + 8]
        out[start:start + DIR_ENTRY_SIZE] = canonical
    return bytes(out)


def decode_header_sector(data: bytes) -> bytes:
    """Keep the signature and the fields that vary between real containers
    (versions, sector shifts, FAT/directory pointers, the first DIFAT
    entries); zero the CLSID and reserved regions."""
    out = bytearray(len(data))
    end = min(len(data), 512)
    out[0:8] = data[0:8]           # signature
    out[0x18:0x24] = data[0x18:0x24]  # minor/major version, byte order, shifts
    out[0x28:0x4C] = data[0x28:0x4C]  # sector counts and first-sector pointers
    difat_end = min(end, 0x200)
    out[0x4C:difat_end] = data[0x4C:difat_end]  # DIFAT[0..108]
    if len(data) > 512:
        # A 4096B fragment starting at the header also contains the first
        # FAT/directory sectors; decode the remainder as a FAT chain, which is
        # what sector 1 of a v3 container almost always is.
        out[512:] = decode_fat_sector(data[512:])
    return bytes(out)


def _decode_one_sector(sector: bytes) -> tuple:
    """Decode a single sector according to its own role. Returns
    (decoded_bytes, role)."""
    signals = cfbf_signals(sector)
    role = structural_role(sector, signals)
    if role == _ROLE_HEADER:
        return decode_header_sector(sector), role
    if role == _ROLE_DIRECTORY:
        return decode_directory_sector(sector, signals["directory_entry_offset"]), role
    if role == _ROLE_FAT:
        return decode_fat_sector(sector), role
    return sector, role


def sector_roles(data: bytes, sector_size: int = SECTOR_SIZE) -> list:
    """Per-sector structural roles across the fragment.

    A CFBF container is a sector array, so a 4096B fragment is a *mixture* of
    eight independent 512B sectors — typically one FAT or directory sector
    surrounded by stream data. Judging the fragment as a whole dilutes that
    one structured sector below every threshold and reports "stream_data",
    which is why detection has to be per-sector.
    """
    roles = []
    for start in range(0, len(data), sector_size):
        sector = data[start:start + sector_size]
        if len(sector) < 64:  # too short to judge; treat as stream data
            roles.append(_ROLE_STREAM_DATA)
            continue
        roles.append(structural_role(sector))
    return roles


def cfbf_recovered_view(data: bytes, target_len: int = None,
                        sector_size: int = SECTOR_SIZE) -> bytes:
    """The adapter's recovery step, applied sector by sector.

    Same contract as dual_view_model.build_recovered_view: always returns
    exactly target_len bytes, and any sector with no decodable structure is
    copied through untouched, so a fragment of pure stream data is returned
    byte-identical to its input.
    """
    target_len = target_len if target_len is not None else len(data)
    out = bytearray()
    for start in range(0, len(data), sector_size):
        sector = data[start:start + sector_size]
        if len(sector) < 64:
            out += sector
            continue
        decoded, _ = _decode_one_sector(sector)
        out += decoded[:len(sector)]
    decoded = bytes(out)
    if len(decoded) < target_len:
        decoded = decoded + data[len(decoded):target_len]
    return decoded[:target_len]
