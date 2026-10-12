"""Probe: score the same restorations with the original extractor and a strict <w:t> extractor.
Restorer decisions do not use the extracted text, so only scoring changes. One seed per cell."""
import sys, re, json, numpy as np
sys.path.insert(0, ".")
import classify.restore.docx_restore as dr
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
import classify.restore.text_recall as tr
from classify.eval.restore_experiment import restore_by_document
ORIG = dr.strip_xml_text
STRICT = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)
def strict(b):
    t = b.decode("utf-8", "ignore"); r = STRICT.findall(t)
    return " ".join(r) if r else re.sub(r"<[^>]*>", " ", t)
model, sev, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
paths = corpus_paths("docx")[:60]
B = {"shuffled": build_image, "realistic": build_image_realistic}[model]
img = B(paths, sev, seed=seed, block_size=512)
out = {}
for name, fn in (("orig", ORIG), ("strict", strict)):
    dr.strip_xml_text = fn
    refs = {p: tr.reference_text(p) for p in paths}
    keep = [p for p in paths if tr.tokenize(refs[p])]
    res = {m: restore_by_document(img, paths, mode)["text"] for m, mode in (("B0", "contiguous"), ("B2", "structural"))}
    r0 = np.array([tr.text_scores(res["B0"].get(p, ""), refs[p])["recall"] for p in keep])
    r2 = np.array([tr.text_scores(res["B2"].get(p, ""), refs[p])["recall"] for p in keep])
    out[name] = dict(n=len(keep), B0=r0.mean(), B2=r2.mean(), eff=(r2 - r0).mean(), better=int((r2 > r0 + 1e-12).sum()), worse=int((r2 < r0 - 1e-12).sum()))
dr.strip_xml_text = ORIG
print(model, sev, seed, json.dumps(out, default=float))
