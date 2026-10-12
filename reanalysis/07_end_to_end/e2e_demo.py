"""End-to-end demonstration of the framework on one mixed image (public files only).
Targets: 3 NapierOne DOCX (Word) and 2 govdocs1 DOC; non-targets: 2 PDF and 1 JPEG from govdocs1.
Realistic fragmentation model, moderate severity, seed 0, 512-byte blocks. Restorers unchanged; corrected text rule."""
import sys, os, json, collections
sys.path.insert(0, "."); os.chdir(".")
sys.path.insert(0, __import__("os").path.abspath("./reanalysis/common"))
import strict_patch; strict_patch.apply()
import classify.restore.docx_restore as dr
from classify.restore import cfbf_restore as cr
from classify.restore.fragmenter import build_image_realistic, fragmentation_stats
from classify.restore.text_recall import reference_text, text_scores
docx = [f"data/napierone_docx/extracted/{n}-docx.docx" for n in ("0003", "0014", "0092")]
doc = ["data/govdocs1/doc_real/000002.doc", "data/govdocs1/doc_real/000151.doc"]
other = ["data/e2e_demo/000009.pdf", "data/e2e_demo/000010.pdf", "data/e2e_demo/000107.jpg"]
paths = docx + doc + other
img = build_image_realistic(paths, "moderate", seed=0, block_size=512)
data = bytes(img.data); B = 512
src = lambda bi: os.path.basename(img.truth[bi].source) if 0 <= bi < len(img.truth) else "?"
runs = collections.Counter(); prev = None
for t in img.truth:
    if t.source != prev: runs[os.path.basename(t.source)] += 1; prev = t.source
out = {"image_blocks": len(img.truth), "image_bytes": len(data), "runs_per_file": dict(runs)}
# stage 1: signature scan
lfh = dr.find_local_headers(data, B)
cf = cr.find_cfbf_headers(data, B)
out["zip_headers"] = len(lfh); out["zip_text_headers"] = sum(dr._is_text_part(h.name) and h.compressed_size > 0 for h in lfh)
out["zip_headers_by_source"] = dict(collections.Counter(src(h.block_index) for h in lfh))
out["cfbf_headers"] = len(cf); out["cfbf_by_source"] = dict(collections.Counter(src(h["block_index"]) for h in cf))
# stage 2-3: restore
rec = {m: collections.defaultdict(list) for m in ("B0", "B2")}; parts = []
for h in lfh:
    if not dr._is_text_part(h.name) or h.compressed_size == 0: continue
    s = src(h.block_index)
    p0 = dr.restore_contiguous(data, h); p2 = dr.restore_structural(data, h, B, search_window=None)
    rec["B0"][s].append(p0.text); rec["B2"][s].append(p2.text)
    parts.append({"source": s, "part": h.name, "B0_complete": p0.complete, "B2_complete": p2.complete, "B0_crc": p0.crc_verified, "B2_crc": p2.crc_verified})
for h in cf:
    s = src(h["block_index"])
    if img.truth[h["block_index"]].file_offset != 0: continue
    rec["B0"][s].append(cr.restore_contiguous_doc(data, h).text); rec["B2"][s].append(cr.restore_structural_doc(data, h).text)
out["parts"] = parts
# stage 4: score against intact originals
res = []
for p in docx + doc:
    b = os.path.basename(p); ref = reference_text(p) if p.endswith(".docx") else cr.reference_doc_text(p)
    row = {"file": b, "type": "DOCX" if p.endswith(".docx") else "DOC", "fragments": runs[b]}
    for m in ("B0", "B2"):
        s = text_scores(" ".join(rec[m].get(b, [])), ref); row[m] = {"recall": s["recall"], "precision": s["precision"]}
    res.append(row)
out["documents"] = res
os.makedirs("writing/v24/e2e_output", exist_ok=True)
for m in ("B0", "B2"):
    for b, txt in rec[m].items(): open(f"writing/v24/e2e_output/{b}.{m}.txt", "w").write(" ".join(txt))
json.dump(out, open("writing/v24/e2e_demo.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "parts"}, indent=1, default=str))
for p in parts: print(p)
