"""Boiko-style evaluation on NapierOne DOCX (corrected text rule; restorers unchanged).
Setting S (Boiko & Moskalenko, 2023): each document alone, cut into 4,096-byte clusters, all clusters shuffled.
Setting M (Boiko, Moskalenko & Sloma, 2025, without the classifier): the clusters of all 100 documents shuffled together.
Usage: python boiko_style.py <S|M> <seed> <out.jsonl>"""
import sys, os, json, glob, random, zipfile
sys.path.insert(0, "."); os.chdir(".")
sys.path.insert(0, __import__("os").path.abspath("./reanalysis/common"))
import strict_patch; strict_patch.apply()
import classify.restore.docx_restore as dr
from classify.restore.fragmenter import FragmentedImage, BlockTruth
from classify.restore.text_recall import text_scores, tokenize
from collections import defaultdict
B = 4096
mode, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
paths = sorted(glob.glob("data/napierone_docx/extracted/*.docx"))
def refs(p):
    z = zipfile.ZipFile(p); allt, main = [], ""
    for n in z.namelist():
        if dr._is_text_part(n):
            s = dr.strip_xml_text(z.read(n)); allt.append(s)
            if n == "word/document.xml": main = s
    return " ".join(allt), main
R = {p: refs(p) for p in paths}
def clusters(p):
    d = open(p, "rb").read(); n = (len(d) + B - 1) // B
    return [(p, i, d[i * B:(i + 1) * B].ljust(B, b"\0")) for i in range(n)]
def image(items):
    img = FragmentedImage(block_size=B)
    for p, i, chunk in items:
        img.data += chunk; img.truth.append(BlockTruth(p, i * B))
    return img
def restore(img):
    data = bytes(img.data); res = defaultdict(lambda: {"B0": {}, "B2": {}})
    for lfh in dr.find_local_headers(data, B):
        if not dr._is_text_part(lfh.name) or lfh.compressed_size == 0: continue
        src = img.truth[lfh.block_index].source
        res[src]["B0"][lfh.name] = dr.restore_contiguous(data, lfh)
        res[src]["B2"][lfh.name] = dr.restore_structural(data, lfh, B, search_window=None)
    return res
rng = random.Random(seed); rows = []
groups = [[p] for p in paths] if mode == "S" else [paths]
for g in groups:
    items = [c for p in g for c in clusters(p)]; rng.shuffle(items)
    res = restore(image(items))
    for p in g:
        all_ref, main_ref = R[p]
        if not tokenize(all_ref): continue
        row = {"document": os.path.basename(p), "clusters": len(clusters(p))}
        for m in ("B0", "B2"):
            parts = res[p][m]
            txt = " ".join(x.text for x in parts.values())
            main = parts.get("word/document.xml")
            row[m] = {"recall": text_scores(txt, all_ref)["recall"], "precision": text_scores(txt, all_ref)["precision"],
                      "main_recall": text_scores(main.text if main else "", main_ref)["recall"] if tokenize(main_ref) else None,
                      "main_crc": bool(main and main.crc_verified), "parts": len(parts), "parts_crc": sum(int(x.crc_verified) for x in parts.values())}
        rows.append(row)
with open(out, "a") as fh: fh.write(json.dumps({"mode": mode, "seed": seed, "rows": rows}) + "\n")
print(mode, seed, len(rows))
