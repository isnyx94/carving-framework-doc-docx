"""Worked recovery examples (public corpora only), scored with the corrected text rule. Output: /tmp/ex_*.json"""
import sys, os, json, glob
sys.path.insert(0, "."); os.chdir(".")
sys.path.insert(0, __import__("os").path.abspath("./reanalysis/common"))
import strict_patch; strict_patch.apply()
from classify.restore.fragmenter import build_image
from classify.restore.text_recall import reference_text, text_scores
from classify.eval.restore_experiment import restore_by_document
paths = sorted(glob.glob("data/napierone_docx/extracted/*.docx"))
refs = {p: reference_text(p) for p in paths}
img = build_image(paths, "moderate", seed=0, block_size=512)
res = {m: restore_by_document(img, paths, mode)["text"] for m, mode in (("B0", "contiguous"), ("B2", "structural"))}
rows = []
for p in paths:
    s0 = text_scores(res["B0"].get(p, ""), refs[p]); s2 = text_scores(res["B2"].get(p, ""), refs[p])
    rows.append({"doc": os.path.basename(p), "ntok": s0["n_reference_tokens"], "B0": s0, "B2": s2, "ref": refs[p], "t0": res["B0"].get(p, ""), "t2": res["B2"].get(p, "")})
json.dump(rows, open("/tmp/ex_napier_strict.json", "w")); print("done")
