"""E1: build a genuine FAT32 image (4 KiB clusters) with mtools' allocator, then delete the targets.
Usage: python fs_build.py <ooxml|cfbf> <D|O> <seed> <workdir> [L|S]
L = whole filler files (large gaps); S = fillers cut into 8-128 KiB pieces of the same files (many small gaps).
Writes <workdir>/img.dd and <workdir>/truth.json (cluster -> owner, data offset, targets)."""
import sys, os, json, random, subprocess, re, glob, math
PHD = "."; sys.path.insert(0, PHD); os.chdir(PHD)
TOOLS = "./work/tools/usr/bin"
ENV = dict(os.environ, PATH=TOOLS + ":" + os.environ["PATH"], MTOOLS_SKIP_CHECK="1")
track, scen, seed, wd = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
FS = sys.argv[5] if len(sys.argv) > 5 else "L"
os.makedirs(wd, exist_ok=True)
IMG = os.path.join(wd, "img.dd"); CL = 4096; SIZE_MIB = 272
def mt(*a):
    r = subprocess.run(list(a), env=ENV, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(" ".join(a) + "\n" + r.stderr)
    return r.stdout
def chain(name):
    out = mt("mshowfat", "-i", IMG, "::/" + name)
    cl = []
    for a, b in re.findall(r"<(\d+)(?:-(\d+))?>", out):
        a = int(a); b = int(b) if b else a; cl += list(range(a, b + 1))
    return cl
from classify.restore.text_recall import reference_text
from classify.restore import cfbf_restore as cr
from classify.restore.fragmenter import corpus_paths
rng = random.Random(seed * 7919 + (0 if track == "ooxml" else 1) * 104729 + (0 if scen == "D" else 1) * 13)
if track == "ooxml":
    pool = [p for p in corpus_paths("docx")[:120] if os.path.getsize(p) <= 2 << 20 and reference_text(p).strip()]
    ext = "DOCX"
else:
    pool = [p for p in corpus_paths(os.path.join("data", "govdocs1", "doc_real"), pattern=".doc")
            if os.path.getsize(p) <= 2 << 20 and cr.reference_doc_text(p).strip()]
    ext = "DOC"
targets = rng.sample(pool, 30)
fillers = sorted(glob.glob("./work/gd/**/*.*", recursive=True))
fillers = [f for f in fillers if os.path.getsize(f) > 0]
if FS == "S":
    pdir = os.path.join(wd, "pieces"); os.makedirs(pdir, exist_ok=True); pieces = []
    prng = random.Random(seed + 555); tot = 0
    for f in sorted(fillers, key=lambda x: prng.random()):
        b = open(f, "rb").read(); o = 0
        while o < len(b) and tot < (SIZE_MIB << 20):
            n = prng.randint(8, 128) * 1024; q = os.path.join(pdir, f"p{len(pieces):06d}")
            open(q, "wb").write(b[o:o + n]); pieces.append(q); o += n; tot += min(n, len(b) - o + n)
        if tot >= (SIZE_MIB << 20): break
    fillers = pieces
open(IMG, "wb").truncate(SIZE_MIB << 20)
mt("mformat", "-i", IMG, "-F", "-c", "8", "::")
bs = open(IMG, "rb").read(512)
reserved = int.from_bytes(bs[14:16], "little"); nfats = bs[16]; fatsz = int.from_bytes(bs[36:40], "little")
total_sectors = int.from_bytes(bs[32:36], "little")
data_off = (reserved + nfats * fatsz) * 512
n_clusters = (total_sectors * 512 - data_off) // CL
free = n_clusters - 1  # root directory
def ncl(p): return max(1, math.ceil(os.path.getsize(p) / CL))
half = len(targets) // 2; t1, t2 = targets[:half], targets[half:]
order, fill_names, k = [], [], 0
fpool = fillers[:]; rng.shuffle(fpool)
reserve = 64  # leave only a small tail, so later writes wrap into the gaps left by deletion
budget = free - reserve - sum(ncl(p) for p in t1)
fi = 0
while True:
    f = fpool[fi % len(fpool)]; c = ncl(f)
    if c > budget: break
    budget -= c; order.append(("F", f)); fi += 1
    if fi > 20000: break
for t in t1: order.insert(rng.randrange(len(order) + 1), ("T", t))
names = {}
for i, (kind, p) in enumerate(order):
    nm = (f"T{targets.index(p):03d}.{ext}" if kind == "T" else f"F{i:05d}.BIN")
    mt("mcopy", "-i", IMG, p, "::/" + nm); names[nm] = p
    if kind == "F": fill_names.append(nm)
dele = rng.sample(fill_names, len(fill_names) // 2)
for j in range(0, len(dele), 50): mt("mdel", "-i", IMG, *["::/" + d for d in dele[j:j + 50]])
for p in t2:
    nm = f"T{targets.index(p):03d}.{ext}"; mt("mcopy", "-i", IMG, p, "::/" + nm); names[nm] = p
tchains = {nm: chain(nm) for nm in names if nm.startswith("T")}
mt("mdel", "-i", IMG, *["::/" + nm for nm in tchains])
owner = {}
for nm, cl in tchains.items():
    for j, c in enumerate(cl): owner[c] = (names[nm], j * CL)
over = []
if scen == "O":
    freed = sum(len(cl) for cl in tchains.values()); need = freed // 4; got = 0; i = 0
    while got < need:
        f = fpool[(fi + i) % len(fpool)]; nm = f"O{i:05d}.BIN"; mt("mcopy", "-i", IMG, f, "::/" + nm)
        cl = chain(nm); got += len(cl); i += 1
        for c in cl:
            if c in owner: over.append(c); owner.pop(c)
frag = {names[nm]: sum(1 for a, b in zip(cl, cl[1:]) if b != a + 1) + 1 for nm, cl in tchains.items()}
json.dump({"track": track, "scenario": scen, "fillers": FS, "seed": seed, "data_off": data_off, "cluster": CL,
           "n_clusters": n_clusters, "targets": targets, "runs_per_target": frag,
           "overwritten_clusters": len(over), "target_clusters": sum(len(c) for c in tchains.values()),
           "owner": {str(c): v for c, v in owner.items()}}, open(os.path.join(wd, "truth.json"), "w"))
print(track, scen, FS, seed, "fillers", len(fill_names), "fragmented", sum(v > 1 for v in frag.values()), "/", len(frag),
      "overwritten", len(over), flush=True)
