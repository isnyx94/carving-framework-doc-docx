"""CFBF restoration on all 109 genuine govdocs1 .doc files (thesis code unchanged).
Usage: python run_cfbf109.py <model> <severity|intact> <out.jsonl>"""
import sys, os, json
sys.path.insert(0, "."); os.chdir(".")
from collections import defaultdict
from classify.restore import cfbf_restore as cr
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import text_scores
model, sev, out = sys.argv[1], sys.argv[2], sys.argv[3]
paths = corpus_paths(os.path.join("data", "govdocs1", "doc_real"), pattern=".doc")
refs = {p: cr.reference_doc_text(p) for p in paths}; kept = [p for p in paths if refs[p].strip()]
MODES = {"B0": lambda d, h: cr.restore_contiguous_doc(d, h), "B2": lambda d, h: cr.restore_structural_doc(d, h)}
def restore(img, m):
    data = bytes(img.data); rec = defaultdict(list)
    for h in cr.find_cfbf_headers(data, img.block_size):
        i = h["block_index"]
        if not 0 <= i < len(img.truth): continue
        src = img.truth[i].source
        if src == "NOISE" or img.truth[i].file_offset != 0: continue
        rec[src].append(MODES[m](data, h).text)
    return {k: " ".join(v) for k, v in rec.items()}
rows = []
if sev == "intact":
    for p in kept:
        data = open(p, "rb").read(); h = cr.find_cfbf_headers(data, 512)
        if not h: continue
        r = {"model": "intact", "severity": "intact", "seed": -1, "document": os.path.basename(p)}
        for m in MODES:
            s = text_scores(MODES[m](data, h[0]).text, refs[p]); r[m + "_recall"] = s["recall"]; r[m + "_precision"] = s["precision"]
        rows.append(r)
else:
    b = {"shuffled": build_image, "realistic": build_image_realistic}[model]
    for seed in range(5):
        img = b(paths, sev, seed=seed, block_size=512); res = {m: restore(img, m) for m in MODES}
        for p in kept:
            r = {"model": model, "severity": sev, "seed": seed, "document": os.path.basename(p)}
            for m in MODES:
                s = text_scores(res[m].get(p, ""), refs[p]); r[m + "_recall"] = s["recall"]; r[m + "_precision"] = s["precision"]
            rows.append(r)
with open(out, "a") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(model, sev, len(rows), "kept", len(kept), "of", len(paths), flush=True)
