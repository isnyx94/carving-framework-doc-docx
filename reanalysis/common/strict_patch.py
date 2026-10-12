"""Corrected text extraction for OOXML scoring.

The original extractor used the pattern <w:t[^>]*>(.*?)</w:t>, which also matches <w:tab>, <w:tabs>,
<w:tbl...>, <w:tc> and <w:tr>, so markup was captured as text. The corrected opening pattern
<w:t(?:\\s[^>]*)?> matches only the <w:t> run element. Restorer decisions never use the extracted text
(they use byte lengths, the CRC and byte-level plausibility), so the replacement changes scoring only.
apply() rebinds the function in every module that uses it; it must run before scripts that import
the function by name are imported."""
import re as _re
import classify.restore.docx_restore as dr
import classify.restore.docx_restore_dd as dd
_OPEN = _re.compile(r"<w:t(?:\s[^>]*)?>")

def strict_strip_xml_text(xml_bytes: bytes) -> str:
    """Visible text of (possibly truncated) WordprocessingML: the contents of <w:t> runs only."""
    text = xml_bytes.decode("utf-8", errors="ignore")
    runs, pos = [], 0
    while True:
        m = _OPEN.search(text, pos)
        if not m: break
        e = text.find("</w:t>", m.end())
        if e < 0: break
        runs.append(text[m.end():e]); pos = e + 6
    if runs: return " ".join(runs)
    return _re.sub(r"<[^>]*>", " ", text)

def apply():
    dr.strip_xml_text = strict_strip_xml_text
    dd.strip_xml_text = strict_strip_xml_text
    try:
        import classify.eval.photorec_baseline as pb; pb.strip_xml_text = strict_strip_xml_text
    except Exception: pass
    s = b'<w:p><w:tabs><w:tab w:val="clear" w:pos="4320"/></w:tabs><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space="preserve"> world</w:t></w:r><w:tbl><w:tr><w:tc><w:p><w:r><w:t>cell</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:p>'
    assert dr.strip_xml_text(s) == "Hello  world cell"
    from classify.restore.text_recall import reference_text  # imports strip_xml_text at call time
