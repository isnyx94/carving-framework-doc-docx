"""Structural coverage on the thesis's own (non-FFT-75) files, with the same detectors as the oracle,
compared with FFT-75 test fragments of the same classes. Usage: python own_coverage.py <size> <out.json>"""
import sys, os, glob, json, random
sys.path.insert(0, "."); os.chdir(".")
import numpy as np
import types, importlib
src = open("classify/eval/oracle_upper_bound.py").read()
start = src.index("LFH_MAGIC = "); end = src.index("def build(")
ns = {}
exec("import numpy as np, os\nfrom classify.data.fft75 import COMPOUND_SUBSET\nfrom classify.modules.c1_compound import cfbf_adapter as cfbf\nfrom classify.modules.c1_compound.partial_decompress import attempt_partial_decompress\n" + src[start:end], ns)
structural_evidence = ns["structural_evidence"]
size = int(sys.argv[1]); out = sys.argv[2]; ONLY = sys.argv[3]; N = 3000
SETS = {"docx (local OOXML corpus)": sorted(glob.glob("docx/*.docx")),
        "doc (govdocs1 thread 000)": sorted(glob.glob("data/govdocs1/doc_real/*.doc")),
        "xls (govdocs1 thread 000)": sorted(glob.glob("/tmp/gd/**/*.xls", recursive=True)),
        "ppt (govdocs1 thread 000)": sorted(glob.glob("/tmp/gd/**/*.ppt", recursive=True))}
KEYS = ("deflate_recovered", "zip_lfh_name", "opc_part_name", "cfbf_structure", "any")
res = {}
for name, files in SETS.items():
    rng = random.Random(1000 + size)
    blocks = []
    for f in files:
        n = os.path.getsize(f) // size
        blocks += [(f, i) for i in range(n)]
    if name.split()[0] != ONLY: continue
    sample = sorted(rng.sample(blocks, min(N, len(blocks))))
    cnt = dict.fromkeys(KEYS, 0); cache = {}
    for f, i in sample:
        if f not in cache:
            cache.clear(); cache[f] = open(f, "rb").read()
        frag = cache[f][i * size:(i + 1) * size]
        ev = structural_evidence(frag)
        for k in KEYS: cnt[k] += int(ev[k])
    res[name] = {"n_files": len(files), "n_blocks_total": len(blocks), "n_sampled": len(sample), **{k: cnt[k] / len(sample) for k in KEYS}}
    print(size, name, res[name], flush=True)
    open(out, "a").write(json.dumps({"size": size, "set": name, **res[name]}) + "\n")
# FFT-75 comparison, same detectors, same classes
from classify.data.fft75 import CLASS_TO_INDEX, load_fft75_arrays
if ONLY.startswith("fft_"): x, y = load_fft75_arrays(size, "test")
for cls in ([ONLY[4:]] if ONLY.startswith("fft_") else []):
    idx = np.flatnonzero(y == CLASS_TO_INDEX[cls])[:N]
    cnt = dict.fromkeys(KEYS, 0)
    for i in idx:
        ev = structural_evidence(x[i].tobytes())
        for k in KEYS: cnt[k] += int(ev[k])
    res[f"{cls} (FFT-75)"] = {"n_sampled": int(idx.size), **{k: cnt[k] / idx.size for k in KEYS}}
    print(size, cls, res[f"{cls} (FFT-75)"], flush=True)
    open(out, "a").write(json.dumps({"size": size, "set": f"{cls} (FFT-75)", **res[f"{cls} (FFT-75)"]}) + "\n")
