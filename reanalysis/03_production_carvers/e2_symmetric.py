"""E2: external carvers vs B0/B2 on the 2026-10-06 synthetic images, every condition scored with BOTH
position-based attribution (ground-truth block of the carve/header start) and best-match attribution.
Usage: python e2_symmetric.py <ooxml|cfbf> <model> <severity> <seed> <out.jsonl>"""
import sys, os, json, glob, re, shutil, subprocess, zipfile
from collections import defaultdict, Counter
PHD = "."; sys.path.insert(0, PHD); os.chdir(PHD)
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import reference_text, text_scores, tokenize
from classify.restore.docx_restore import find_local_headers, restore_structural, restore_contiguous, _is_text_part, strip_xml_text
from classify.restore import cfbf_restore as cr
T = "./work/tools"
track, model, sev, seed, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
def bm(texts, refs):
    refc = {p: Counter(tokenize(r)) for p, r in refs.items()}; refn = {p: sum(c.values()) for p, c in refc.items()}
    a = {p: [] for p in refs}
    for t in texts:
        tc = Counter(tokenize(t))
        if not tc: continue
        bp, br = None, -1.0
        for p, rc in refc.items():
            r = sum((rc & tc).values()) / refn[p] if refn[p] else 0.0
            if r > br: bp, br = p, r
        if bp is not None and br > 0: a[bp].append(t)
    return {p: " ".join(v) for p, v in a.items()}
if track == "ooxml":
    paths = corpus_paths("docx", limit=20); refs = {p: reference_text(p) for p in paths}; fam, fty = "zip", "zip"
else:
    paths = corpus_paths("data/govdocs1/doc_real", limit=40, pattern=".doc"); refs = {p: cr.reference_doc_text(p) for p in paths}; fam, fty = "doc", "ole"
refs = {p: r for p, r in refs.items() if r.strip()}
img = {"shuffled": build_image, "realistic": build_image_realistic}[model](paths, sev, seed=seed, block_size=512)
data = bytes(img.data)
def cell(pos):
    i = pos // 512
    return img.truth[i] if 0 <= i < len(img.truth) else None
outs = {}
for m in ("B0", "B2"):
    pos, texts = defaultdict(list), []
    if track == "ooxml":
        for lfh in find_local_headers(data, 512):
            if not _is_text_part(lfh.name) or lfh.compressed_size == 0: continue
            part = restore_structural(data, lfh, 512) if m == "B2" else restore_contiguous(data, lfh)
            texts.append(part.text); c = cell(lfh.offset_in_image)
            if c is not None and c.source != "NOISE": pos[c.source].append(part.text)
    else:
        for h in cr.find_cfbf_headers(data, 512):
            d = (cr.restore_structural_doc(data, h) if m == "B2" else cr.restore_contiguous_doc(data, h)).text
            texts.append(d); c = cell(h["offset"])
            if c is not None and c.source != "NOISE" and c.file_offset == 0: pos[c.source].append(d)
    outs[m] = {"pos": {k: " ".join(v) for k, v in pos.items()}, "bm": bm(texts, refs)}
scr = f"/tmp/e2_{track}_{model}_{sev}_{seed}"; shutil.rmtree(scr, ignore_errors=True); os.makedirs(scr)
ip = scr + "/img.dd"; open(ip, "wb").write(data)
def text_of(f):
    try:
        if track == "ooxml":
            with zipfile.ZipFile(f) as z: return " ".join(strip_xml_text(z.read(n)) for n in z.namelist() if _is_text_part(n))
        return cr.extract_word_text(open(f, "rb").read())
    except Exception: return ""
carves = {}
for name, opts in (("PR_default", ""), ("PR_keep", "options,keep_corrupted_file,")):
    d = scr + "/" + name; os.makedirs(d)
    subprocess.run([T + "/usr/bin/photorec", "/log", "/d", d + "/recup", "/cmd", ip, f"partition_none,{opts}fileopt,everything,disable,{fam},enable,search"], capture_output=True, timeout=900, cwd=scr)
    carves[name] = [(int(re.match(r"[fbt](\d+)", os.path.basename(f)).group(1)) * 512, f) for f in glob.glob(d + "/**/*", recursive=True)
                    if os.path.isfile(f) and re.match(r"[fbt]\d+", os.path.basename(f)) and not f.endswith((".log", ".xml"))]
subprocess.run([T + "/usr/bin/foremost", "-c", T + "/etc/foremost.conf", "-q", "-t", fty, "-b", "512", "-i", ip, "-o", scr + "/FM"], capture_output=True, timeout=900)
carves["FM"] = [(int(re.match(r"(\d+)", os.path.basename(f)).group(1)) * 512, f) for f in glob.glob(scr + "/FM/*/*") if re.match(r"\d+", os.path.basename(f))]
for name, fl in carves.items():
    pos, texts = defaultdict(list), []
    for o, f in fl:
        t = text_of(f); texts.append(t); c = cell(o)
        if c is not None and c.source != "NOISE" and c.file_offset == 0: pos[c.source].append(t)
    outs[name] = {"pos": {k: " ".join(v) for k, v in pos.items()}, "bm": bm(texts, refs)}
shutil.rmtree(scr, ignore_errors=True)
row = {"track": track, "model": model, "severity": sev, "seed": seed}
for m, v in outs.items():
    for att in ("pos", "bm"):
        sc = [text_scores(v[att].get(p, ""), r) for p, r in refs.items()]
        row[f"{m}_{att}_recall"] = sum(s["recall"] for s in sc) / len(sc); row[f"{m}_{att}_precision"] = sum(s["precision"] for s in sc) / len(sc)
open(out, "a").write(json.dumps(row) + "\n")
print(track, model, sev, seed, {k: round(v, 3) for k, v in row.items() if k.endswith("recall")}, flush=True)
