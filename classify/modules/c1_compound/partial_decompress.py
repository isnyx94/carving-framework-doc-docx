"""Partial decompression (methodology Sec 6.2, C1 step 2).

ZIP/OOXML DEFLATE streams have no zlib header (RFC 1951 raw deflate, ZIP
compression method 8), so this uses zlib's raw-deflate mode (wbits=-15) —
not utils.partial_inflate, which assumes a zlib-wrapped stream (RFC 1950)
and would reject every real OOXML fragment.

Decompression genuinely requires the bitstream from the start of the
DEFLATE block; a fragment sampled from the interior of a compressed stream
cannot be decompressed without the preceding bytes (methodology A3 — this
is measured, not assumed). Honest failure is the expected outcome for most
interior fragments; this function reports it rather than silently
returning empty bytes as if nothing were wrong.
"""

import zlib


def attempt_partial_decompress(data: bytes, max_output: int = 1 << 16) -> dict:
    decompressor = zlib.decompressobj(wbits=-15)
    try:
        out = decompressor.decompress(data, max_output)
    except zlib.error as e:
        return {"success": False, "output": b"", "bytes_consumed": 0, "error": str(e)}
    bytes_consumed = len(data) - len(decompressor.unused_data) - len(decompressor.unconsumed_tail)
    return {
        "success": len(out) > 0,
        "output": out,
        "bytes_consumed": bytes_consumed,
    }
