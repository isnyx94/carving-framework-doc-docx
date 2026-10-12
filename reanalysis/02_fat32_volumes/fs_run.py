"""E1: run B0, B2, PhotoRec (default, keep-corrupted) and Foremost on one FAT32 image built by fs_build.py.
Usage: python fs_run.py <workdir> <out.jsonl> [stages: rest,tools]"""
import sys, os, json, glob, re, shutil, subprocess
from collections import defaultdict, namedtuple
PHD = "."; sys.path.insert(0, PHD); os.chdir(PHD)
TOOLS = "./work/tools"
from classify.restore.text_recall import reference_text, text_scores, tokenize
from classify.restore import cfbf_restore as cr
from classify.restore.docx_restore import find_local_headers, restore_structural, restore_contiguous, _is_text_part, strip_xml_text
from collections import Counter
def attribute_by_best_match(texts, references):
    # Same rule as classify.eval.photorec_baseline.attribute_by_best_match (first reference with the
    # highest token recall wins; recall must be > 0), with each text tokenised once for speed.
    refc = {p: Counter(tokenize(r)) for p, r in references.items()}
    refn = {p: sum(c.values()) for p, c in refc.items()}
    assigned = {p: [] for p in references}
    for text in texts:
        tc = Counter(tokenize(text))
        if not tc: continue
        best_p, best_r = None, -1.0
        for p, rc in refc.items():
            r = sum((rc & tc).values()) / refn[p] if refn[p] else 0.0
            if r > best_r: best_p, best_r = p, r
        if best_p is not None and best_r > 0: assigned[best_p].append(text)
    return {p: " ".join(c) for p, c in assigned.items()}
wd, out = sys.argv[1], sys.argv[2]
stages = sys.argv[3].split(",") if len(sys.argv) > 3 else ["rest", "tools"]
T = json.load(open(os.path.join(wd, "truth.json")))
raw = open(os.path.join(wd, "img.dd"), "rb").read()
off, CL = T["data_off"], T["cluster"]
data = raw[off:]
owner = {int(k): v for k, v in T["owner"].items()}
Cell = namedtuple("Cell", "source file_offset")
truth = [Cell(*owner[i + 2]) if (i + 2) in owner else Cell("NOISE", -1) for i in range(len(data) // CL)]
track = T["track"]; targets = T["targets"]
refs = {p: (reference_text(p) if track == "ooxml" else cr.reference_doc_text(p)) for p in targets}
def src_of(pos):  # position in the data region -> (source, offset in file)
    i = pos // CL
    return truth[i] if 0 <= i < len(truth) else Cell("NOISE", -1)
res = {}
if "rest" in stages:
    for mode in ("B0", "B2"):
        rec = defaultdict(list)
        if track == "ooxml":
            for lfh in find_local_headers(data, CL):
                if not _is_text_part(lfh.name) or lfh.compressed_size == 0: continue
                s = src_of(lfh.offset_in_image).source
                if s == "NOISE": continue
                part = restore_structural(data, lfh, CL) if mode == "B2" else restore_contiguous(data, lfh)
                rec[s].append(part.text)
        else:
            for h in cr.find_cfbf_headers(data, CL):
                c = truth[h["block_index"]] if h["block_index"] < len(truth) else Cell("NOISE", -1)
                if c.source == "NOISE" or c.file_offset != 0: continue
                d = cr.restore_structural_doc(data, h) if mode == "B2" else cr.restore_contiguous_doc(data, h)
                rec[c.source].append(d.text)
        res[mode] = {"pos": {k: " ".join(v) for k, v in rec.items()}}
if "tools" in stages:
    PR, FM = TOOLS + "/usr/bin/photorec", TOOLS + "/usr/bin/foremost"
    scr = os.path.join(wd, "carve"); shutil.rmtree(scr, ignore_errors=True); os.makedirs(scr)
    ip = os.path.join(wd, "img.dd"); fam, fty = ("zip", "zip") if track == "ooxml" else ("doc", "ole")
    def text_of(f):
        try:
            if track == "ooxml":
                import zipfile
                with zipfile.ZipFile(f) as z:
                    return " ".join(strip_xml_text(z.read(n)) for n in z.namelist() if _is_text_part(n))
            return cr.extract_word_text(open(f, "rb").read())
        except Exception: return ""
    carves = {}
    for name, opts in (("PR_default", ""), ("PR_keep", "options,keep_corrupted_file,")):
        d = os.path.join(scr, name); os.makedirs(d)
        # PhotoRec as an examiner would use it on a FAT volume: file system detected, unallocated space only.
        subprocess.run([PR, "/log", "/d", d + "/recup", "/cmd", ip, f"{opts}fileopt,everything,disable,{fam},enable,freespace,search"],
                       capture_output=True, timeout=900, cwd=scr)
        fl = []
        for f in glob.glob(d + "/**/*", recursive=True):
            m = re.match(r"[fbt](\d+)", os.path.basename(f))
            if os.path.isfile(f) and m and not f.endswith((".log", ".xml")): fl.append((int(m.group(1)) * 512, f))
        carves[name] = fl
    d = os.path.join(scr, "FM")
    subprocess.run([FM, "-c", TOOLS + "/etc/foremost.conf", "-q", "-t", fty, "-b", "512", "-i", ip, "-o", d], capture_output=True, timeout=900)
    fl = []
    for f in glob.glob(d + "/*/*"):
        m = re.match(r"(\d+)", os.path.basename(f))
        if m: fl.append((int(m.group(1)) * 512, f))
    carves["FM"] = fl
    for name, fl in carves.items():
        pos, texts = defaultdict(list), []
        for o, f in fl:
            t = text_of(f); texts.append(t)
            c = src_of(o - off)
            if c.source != "NOISE" and c.file_offset == 0: pos[c.source].append(t)
        res[name] = {"pos": {k: " ".join(v) for k, v in pos.items()}, "bm": attribute_by_best_match(texts, refs), "n": len(fl)}
    shutil.rmtree(scr, ignore_errors=True)
rows = []
for p in targets:
    if not refs[p].strip(): continue
    r = {"track": track, "scenario": T["scenario"], "fillers": T.get("fillers", "L"), "seed": T["seed"],
         "document": os.path.basename(p), "runs": T["runs_per_target"][p]}
    for m, v in res.items():
        for att in ("pos", "bm"):
            if att in v:
                s = text_scores(v[att].get(p, ""), refs[p]); r[f"{m}_{att}_recall"] = s["recall"]; r[f"{m}_{att}_precision"] = s["precision"]
    rows.append(r)
with open(out, "a") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
print(os.path.basename(wd), stages, len(rows), {m: round(sum(r.get(m + "_pos_recall", 0) for r in rows) / len(rows), 3) for m in res}, flush=True)
