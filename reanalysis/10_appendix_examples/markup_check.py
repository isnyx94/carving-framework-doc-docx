"""Check: how much of the OOXML reference text is markup captured by the <w:t[^>]*> pattern
(which also matches <w:tab>, <w:tabs>, <w:tbl...>, <w:tc>, <w:tr>)."""
import sys, glob, re, zipfile, json, statistics as st
sys.path.insert(0, ".")
from classify.restore.docx_restore import _is_text_part, strip_xml_text
from classify.restore.fragmenter import corpus_paths
from classify.restore.text_recall import tokenize
from collections import Counter
STRICT = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)
def strict_text(b):
    t = b.decode("utf-8", "ignore"); runs = STRICT.findall(t)
    return " ".join(runs) if runs else re.sub(r"<[^>]*>", " ", t)
def check(paths, label):
    fr = []; affected = 0
    for p in paths:
        try: z = zipfile.ZipFile(p)
        except Exception: continue
        a = Counter(); b = Counter()
        for n in z.namelist():
            if _is_text_part(n):
                x = z.read(n); a.update(tokenize(strip_xml_text(x))); b.update(tokenize(strict_text(x)))
        na = sum(a.values())
        if na == 0: continue
        extra = sum((a - b).values()); fr.append(extra / na); affected += extra > 0
    return {"corpus": label, "n": len(fr), "docs_affected": affected, "median_extra_share": st.median(fr), "mean_extra_share": st.mean(fr), "max": max(fr)}
res = [check(corpus_paths("docx")[:60], "python-docx dev (first 60)"),
       check(corpus_paths("docx")[60:120], "python-docx held-out"),
       check(sorted(glob.glob("data/napierone_docx/extracted/*.docx")), "NapierOne (Word)"),
       check(sorted(glob.glob("data/govdocs1/docx_lo/*.docx")), "LibreOffice")]
for r in res: print(r)
json.dump(res, open("writing/appendix_v20/markup_check.json", "w"), indent=1)
