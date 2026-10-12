"""E4: can a restored OOXML document be rebuilt into a file that opens?
For each document, the restored word/document.xml bytes (B0 and B2) are packaged into a minimal
.docx and tested: (a) XML well-formed; (b) the rebuilt .docx opens with python-docx; (c) after
lxml recovery of a damaged XML, the repaired .docx opens.
Usage: python e4_openable.py <model> <severity> <seed> <out.jsonl>   (development set, files 1-60)"""
import sys, os, io, json, zipfile
from collections import defaultdict
sys.path.insert(0, "."); os.chdir(".")
import classify.restore.docx_restore as dr
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import reference_text
from lxml import etree
import docx
model, sev, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
paths = corpus_paths("docx", limit=60); refs = {p: reference_text(p) for p in paths}
img = {"shuffled": build_image, "realistic": build_image_realistic}[model](paths, sev, seed=seed, block_size=512)
data = bytes(img.data)
_captured = []
_orig = dr.strip_xml_text
def _spy(b):
    _captured.append(bytes(b)); return _orig(b)
dr.strip_xml_text = _spy
CT = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
RELS = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
def opens(xml):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CT); z.writestr("_rels/.rels", RELS); z.writestr("word/document.xml", xml)
    buf.seek(0)
    try:
        d = docx.Document(buf); _ = [p.text for p in d.paragraphs]; return True
    except Exception: return False
def wellformed(xml):
    try: etree.fromstring(xml); return True
    except Exception: return False
def repaired(xml):
    try:
        root = etree.fromstring(xml, etree.XMLParser(recover=True, huge_tree=True))
        if root is None: return None
        return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    except Exception: return None
res = {}
for m in ("B0", "B2"):
    best = {}
    for lfh in dr.find_local_headers(data, 512):
        if lfh.name != "word/document.xml" or lfh.compressed_size == 0: continue
        if not 0 <= lfh.block_index < len(img.truth): continue
        src = img.truth[lfh.block_index].source
        if src == "NOISE": continue
        _captured.clear()
        part = dr.restore_contiguous(data, lfh) if m == "B0" else dr.restore_structural(data, lfh, 512)
        xml = _captured[-1] if _captured else b""
        if src not in best or len(xml) > len(best[src][0]): best[src] = (xml, part.complete, part.crc_verified)
    res[m] = best
with open(out, "a") as fh:
    for p in paths:
        if not refs[p].strip(): continue
        r = {"model": model, "severity": sev, "seed": seed, "document": os.path.basename(p)}
        for m in res:
            xml, comp, crc = res[m].get(p, (b"", False, False))
            wf = wellformed(xml) if xml else False
            rep = repaired(xml) if xml and not wf else (xml if wf else None)
            r[m + "_wellformed"] = wf; r[m + "_opens"] = opens(xml) if wf else False
            r[m + "_opens_repaired"] = opens(rep) if rep else False
            r[m + "_crc"] = crc
        fh.write(json.dumps(r) + "\n")
print(model, sev, seed, "done", flush=True)
