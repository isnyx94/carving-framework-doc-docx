"""CFBF sensitivity analysis: the main CFBF reference text is produced by the same printable-run extractor
that reads the restorers' output. Here every output is also scored against an independent reference, the
document text exported by LibreOffice (which parses the Word piece table). Usage: python ... <model> <out.jsonl>"""
import sys, os, json
from collections import defaultdict
sys.path.insert(0, "."); os.chdir(".")
from classify.restore import cfbf_restore as cr
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import text_scores
model, out = sys.argv[1], sys.argv[2]
SEVS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["basic", "moderate", "severe"]
LO = "./work/lotxt/"
paths = corpus_paths(os.path.join("data", "govdocs1", "doc_real"), pattern=".doc")
ref_h = {p: cr.reference_doc_text(p) for p in paths}
ref_lo = {p: open(LO + os.path.basename(p)[:-4] + ".txt", encoding="utf-8", errors="ignore").read() for p in paths}
kept = [p for p in paths if ref_h[p].strip() and ref_lo[p].strip()]
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
if model == "intact":
    for p in kept:
        data = open(p, "rb").read(); h = cr.find_cfbf_headers(data, 512)
        if not h: continue
        r = {"model": "intact", "severity": "intact", "seed": -1, "document": os.path.basename(p)}
        for m in MODES:
            t = MODES[m](data, h[0]).text
            for rn, ref in (("h", ref_h[p]), ("lo", ref_lo[p])):
                s = text_scores(t, ref); r[f"{m}_{rn}_recall"] = s["recall"]; r[f"{m}_{rn}_precision"] = s["precision"]
        rows.append(r)
else:
    b = {"shuffled": build_image, "realistic": build_image_realistic}[model]
    for sev in SEVS:
        for seed in range(5):
            img = b(paths, sev, seed=seed, block_size=512); res = {m: restore(img, m) for m in MODES}
            for p in kept:
                r = {"model": model, "severity": sev, "seed": seed, "document": os.path.basename(p)}
                for m in MODES:
                    for rn, ref in (("h", ref_h[p]), ("lo", ref_lo[p])):
                        s = text_scores(res[m].get(p, ""), ref); r[f"{m}_{rn}_recall"] = s["recall"]; r[f"{m}_{rn}_precision"] = s["precision"]
                rows.append(r)
with open(out, "a") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(model, len(rows), "kept", len(kept), flush=True)
