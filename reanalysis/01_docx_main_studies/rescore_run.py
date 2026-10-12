"""Re-scoring run with the corrected OOXML text extractor (strict_patch). Restorers unchanged.
Usage: python rescore_run.py <set> <model> <severity> <seed> <modes comma> <out.jsonl>
sets: dev (first 60 python-docx files), heldout (next 60), napier (NapierOne DOCX), lo (LibreOffice-written)
modes: B0, B2, B2nocrc, B0dd, B2dd.  JUMP_BUDGET env sets B2's jump budget (default 400)."""
import sys, os, json, time, glob
sys.path.insert(0, "."); os.chdir(".")
sys.path.insert(0, __import__("os").path.abspath("./reanalysis/common"))
import strict_patch; strict_patch.apply()
from collections import defaultdict, Counter
import classify.restore.docx_restore as dr
import classify.restore.docx_restore_dd as dd
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import reference_text, text_scores, tokenize
S, model, sev, seed, modes, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(","), sys.argv[6]
JUMP = int(os.environ.get("JUMP_BUDGET", "400"))
if S == "dev": paths = corpus_paths("docx")[:60]
elif S == "heldout": paths = corpus_paths("docx")[60:120]
elif S == "napier": paths = sorted(glob.glob("data/napierone_docx/extracted/*.docx"))
elif S == "pilot25": paths = corpus_paths("docx")[:25]
elif S == "pilot20": paths = corpus_paths("docx")[:20]
elif S == "lo": paths = [p for p in corpus_paths(os.path.join("data", "govdocs1", "docx_lo"), pattern=".docx") if p.endswith(".docx")]
refs = {p: reference_text(p) for p in paths}
kept = [p for p in paths if tokenize(refs[p])]
REAL_CRC = dr.crc_verified
img = {"shuffled": build_image, "realistic": build_image_realistic}[model](paths, sev, seed=seed, block_size=512)
data = bytes(img.data)
def run(mode):
    rec = defaultdict(list); st = defaultdict(Counter)
    for lfh in dr.find_local_headers(data, img.block_size):
        if mode in ("B0", "B2", "B2nocrc"):
            if not dr._is_text_part(lfh.name) or lfh.compressed_size == 0: continue
        else:
            if not dd.is_restorable_part(lfh): continue
        if not 0 <= lfh.block_index < len(img.truth): continue
        src = img.truth[lfh.block_index].source
        if src == "NOISE": continue
        if mode == "B0": part = dr.restore_contiguous(data, lfh)
        elif mode in ("B2", "B2nocrc"):
            dr.crc_verified = REAL_CRC if mode == "B2" else (lambda o, l: False)
            try: part = dr.restore_structural(data, lfh, img.block_size, search_window=None, jump_budget=JUMP)
            finally: dr.crc_verified = REAL_CRC
        elif mode == "B0dd": part = dd.restore_contiguous_any(data, lfh)
        else: part = dd.restore_structural_any(data, lfh, img.block_size)
        rec[src].append(part.text)
        st[src]["parts"] += 1; st[src]["complete"] += int(part.complete); st[src]["crc_verified"] += int(part.crc_verified)
    return {k: " ".join(v) for k, v in rec.items()}, st
def sh(t, n=5):
    t = tokenize(t); return Counter(tuple(t[i:i+n]) for i in range(max(0, len(t)-n+1)))
res = {}; parts = {}; timing = {}
for m in modes:
    t0 = time.time(); res[m], s = run(m); timing[m] = time.time() - t0
    parts[m] = dict(sum(s.values(), Counter()))
rows = []
for i, p in enumerate(kept):
    other = kept[(i + 1) % len(kept)]
    row = {"document": os.path.basename(p)}
    for m in modes:
        txt = res[m].get(p, ""); s = text_scores(txt, refs[p])
        row[m] = [s["recall"], s["precision"], s["n_reference_tokens"], s["n_recovered_tokens"], s["n_overlap_tokens"]]
        a, b = sh(txt), sh(refs[p]); row[m + "_5g"] = (sum((a & b).values()) / sum(b.values())) if b else 0.0
        row[m + "_x"] = text_scores(txt, refs[other])["recall"]
    rows.append(row)
with open(out, "a") as fh:
    fh.write(json.dumps({"set": S, "model": model, "severity": sev, "seed": seed, "modes": modes, "jump": JUMP,
                         "n_paths": len(paths), "n_kept": len(kept), "rows": rows, "parts": parts, "timing": timing}) + "\n")
print(S, model, sev, seed, modes, {k: round(v, 1) for k, v in timing.items()}, flush=True)
