"""Worked recovery examples for the appendix, from public corpora only (NapierOne DOCX, govdocs1 .doc).
Restorers unchanged. Output: examples.json"""
import sys, os, json, glob, re
sys.path.insert(0, "."); os.chdir(".")
from collections import defaultdict
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import reference_text, text_scores, tokenize
from classify.eval.restore_experiment import restore_by_document
from classify.restore import cfbf_restore as cr
out = {}
# --- OOXML (NapierOne), shuffled-moderate, seed 0
paths = sorted(glob.glob("data/napierone_docx/extracted/*.docx"))
refs = {p: reference_text(p) for p in paths}
img = build_image(paths, "moderate", seed=0, block_size=512)
res = {m: restore_by_document(img, paths, mode)["text"] for m, mode in (("B0", "contiguous"), ("B2", "structural"))}
rows = []
for p in paths:
    s0 = text_scores(res["B0"].get(p, ""), refs[p]); s2 = text_scores(res["B2"].get(p, ""), refs[p])
    rows.append({"doc": os.path.basename(p), "ntok": s0["n_reference_tokens"], "B0": s0, "B2": s2,
                 "ref": refs[p], "t0": res["B0"].get(p, ""), "t2": res["B2"].get(p, "")})
json.dump(rows, open("/tmp/napier_example_rows.json", "w"))
out["ooxml"] = {"corpus": "NapierOne DOCX-tiny", "model": "shuffled", "severity": "moderate", "seed": 0}
json.dump(out, open("writing/appendix_v20/examples_meta.json", "w"), indent=1)
print("ooxml done")
# --- CFBF (govdocs1), realistic-basic, seed 0
cpaths = corpus_paths(os.path.join("data", "govdocs1", "doc_real"), limit=40, pattern=".doc")
crefs = {p: cr.reference_doc_text(p) for p in cpaths}
cimg = build_image_realistic(cpaths, "basic", seed=0, block_size=512)
def crestore(img, fn):
    data = bytes(img.data); rec = defaultdict(list)
    for h in cr.find_cfbf_headers(data, img.block_size):
        i = h["block_index"]
        if not 0 <= i < len(img.truth): continue
        src = img.truth[i].source
        if src == "NOISE" or img.truth[i].file_offset != 0: continue
        rec[src].append(fn(data, h).text)
    return {k: " ".join(v) for k, v in rec.items()}
c0 = crestore(cimg, cr.restore_contiguous_doc); c2 = crestore(cimg, cr.restore_structural_doc)
crows = []
for p in cpaths:
    if not crefs[p].strip(): continue
    s0 = text_scores(c0.get(p, ""), crefs[p]); s2 = text_scores(c2.get(p, ""), crefs[p])
    crows.append({"doc": os.path.basename(p), "ntok": s0["n_reference_tokens"], "B0": s0, "B2": s2,
                  "ref": crefs[p], "t0": c0.get(p, ""), "t2": c2.get(p, "")})
json.dump(crows, open("/tmp/cfbf_example_rows.json", "w"))
print("cfbf done")
