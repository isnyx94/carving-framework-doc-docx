"""External carvers (PhotoRec 7.1 Linux build, Foremost 1.5.7) on the thesis's synthetic images.
Usage: python run_external.py <track: ooxml|cfbf> <model> <severity> <seed> <out.jsonl>"""
import sys, os, json, shutil, subprocess, zipfile, glob
sys.path.insert(0, "."); os.chdir(".")
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import reference_text, text_scores, tokenize
from classify.restore.docx_restore import _is_text_part, strip_xml_text
from classify.restore import cfbf_restore as cr
from classify.eval.photorec_baseline import attribute_by_best_match
track, model, sev, seed, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
PR, FM = "/tmp/tools/usr/bin/photorec", "/tmp/tools/usr/bin/foremost"
SCR = f"/tmp/ext_{track}_{model}_{sev}_{seed}"; shutil.rmtree(SCR, ignore_errors=True); os.makedirs(SCR)
builder = {"shuffled": build_image, "realistic": build_image_realistic}[model]
if track == "ooxml":
    paths = corpus_paths("docx", limit=20); refs = {p: reference_text(p) for p in paths}
    pr_family, fm_type = "zip", "zip"
else:
    paths = corpus_paths("data/govdocs1/doc_real", limit=40, pattern=".doc"); refs = {p: cr.reference_doc_text(p) for p in paths}
    pr_family, fm_type = "doc", "ole"
refs = {p: r for p, r in refs.items() if r.strip()}
def text_of(f):
    if track == "ooxml":
        if not zipfile.is_zipfile(f): return ""
        try:
            with zipfile.ZipFile(f) as z:
                return " ".join(strip_xml_text(z.read(n)) for n in z.namelist() if _is_text_part(n))
        except Exception: return ""
    # CFBF: lenient — heuristic text extraction from the carved bytes (same extractor as the reference)
    try: return cr.extract_word_text(open(f, "rb").read())
    except Exception: return ""
def collect(d):
    return [p for p in glob.glob(d + "/**/*", recursive=True) if os.path.isfile(p) and not p.endswith((".log", ".txt"))]
ip = SCR + "/img.dd"
if sev == "contiguous":
    with open(ip, "wb") as fh:
        for p in paths:
            b = open(p, "rb").read(); fh.write(b.ljust(-(-len(b) // 512) * 512, b"\0"))
else:
    img = builder(paths, sev, seed=seed, block_size=512); open(ip, "wb").write(bytes(img.data))
res = {}
for name, opts in [("photorec_default", ""), ("photorec_keep_corrupted", "options,keep_corrupted_file,")]:
    d = SCR + "/" + name; os.makedirs(d, exist_ok=True)
    cmd = [PR, "/log", "/d", d + "/recup", "/cmd", ip, f"partition_none,{opts}fileopt,everything,disable,{pr_family},enable,search"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=SCR)
    res[name] = [p for p in glob.glob(d + "/**/*", recursive=True) if os.path.isfile(p) and not p.endswith((".log", ".xml"))]
d = SCR + "/foremost"
subprocess.run([FM, "-c", "/tmp/tools/etc/foremost.conf", "-q", "-t", fm_type, "-b", "512", "-i", ip, "-o", d], capture_output=True, timeout=900)
res["foremost"] = [p for p in glob.glob(d + "/**/*", recursive=True) if os.path.isfile(p) and not p.endswith("audit.txt")]
rows = []
summary = {}
for tool, files in res.items():
    texts = [text_of(f) for f in files]
    assigned = attribute_by_best_match(texts, refs)
    sc = [text_scores(assigned.get(p, ""), r) for p, r in refs.items()]
    summary[tool] = {"carved": len(files), "readable": sum(1 for t in texts if tokenize(t)),
                     "macro_recall": sum(s["recall"] for s in sc) / len(sc),
                     "macro_precision": sum(s["precision"] for s in sc) / len(sc),
                     "n_zero": sum(1 for s in sc if s["recall"] == 0), "n_docs": len(sc)}
open(out, "a").write(json.dumps({"track": track, "model": model, "severity": sev, "seed": seed, "summary": summary}) + "\n")
print(track, model, sev, seed, {k: (v["carved"], v["readable"], round(v["macro_recall"], 3)) for k, v in summary.items()}, flush=True)
shutil.rmtree(SCR, ignore_errors=True)
