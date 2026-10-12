"""CFBF restoration re-run with per-document scores (recall, precision, cross-document
control) and a B0 span-sensitivity condition. Thesis code reused unchanged.
Usage: python run_cfbf.py <model> <out.json>"""
import sys, os, json
sys.path.insert(0, "."); os.chdir(".")
from collections import defaultdict
from classify.restore import cfbf_restore as cr
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import text_scores
model, out = sys.argv[1], sys.argv[2]
builder = {"shuffled": build_image, "realistic": build_image_realistic}[model]
paths = corpus_paths(os.path.join("data", "govdocs1", "doc_real"), limit=40, pattern=".doc")
refs = {p: cr.reference_doc_text(p) for p in paths}
kept = [p for p in paths if refs[p].strip()]
MODES = {"B0": lambda d, h: cr.restore_contiguous_doc(d, h),
         "B0_256K": lambda d, h: cr.restore_contiguous_doc(d, h, span=1 << 18),
         "B2": lambda d, h: cr.restore_structural_doc(d, h)}
rows = []
def restore(img, m):
    data = bytes(img.data); rec = defaultdict(list)
    for h in cr.find_cfbf_headers(data, img.block_size):
        i = h["block_index"]
        if not 0 <= i < len(img.truth): continue
        src = img.truth[i].source
        if src == "NOISE" or img.truth[i].file_offset != 0: continue
        rec[src].append(MODES[m](data, h).text)
    return {k: " ".join(v) for k, v in rec.items()}
# intact control per document
for p in kept:
    data = open(p, "rb").read(); h = cr.find_cfbf_headers(data, 512)
    if not h: continue
    row = {"model": "intact", "severity": "intact", "seed": -1, "document": os.path.basename(p)}
    for m in MODES:
        s = text_scores(MODES[m](data, h[0]).text, refs[p]); row[m+"_recall"] = s["recall"]; row[m+"_precision"] = s["precision"]
    rows.append(row)
for sev in ("basic", "moderate", "severe"):
    for seed in range(5):
        img = builder(paths, sev, seed=seed, block_size=512)
        res = {m: restore(img, m) for m in MODES}
        for i, p in enumerate(kept):
            other = kept[(i + 1) % len(kept)]
            row = {"model": model, "severity": sev, "seed": seed, "document": os.path.basename(p)}
            for m in MODES:
                s = text_scores(res[m].get(p, ""), refs[p])
                row[m+"_recall"] = s["recall"]; row[m+"_precision"] = s["precision"]
                row[m+"_xdoc_recall"] = text_scores(res[m].get(p, ""), refs[other])["recall"]
            rows.append(row)
        print(model, sev, seed, flush=True)
json.dump({"model": model, "n_files": len(paths), "n_kept": len(kept), "rows": rows}, open(out, "w"))
