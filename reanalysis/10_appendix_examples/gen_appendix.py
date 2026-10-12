"""Builds Appendix A (supporting data, calculations, recovery examples) as markdown."""
import json, re, collections, numpy as np, sys
sys.path.insert(0, __import__("os").path.abspath("./reanalysis/common"))
from rescore_analyse import load
W = "./writing/"
os_path = lambda f: W + f
L = ["# Appendix A — Supporting Data, Calculations and Recovery Examples", "", "---", ""]
def P(*a): L.extend(a)
P("This appendix gives the per-document data behind the restoration results of Chapter 5, the volume-level figures of the FAT32 study, the full list of statistical tests corrected together in Section 5.12.11, worked calculations for the main statistical and modelling steps, and worked recovery examples. The examples use only public documents (NapierOne and govdocs1); no text from the local OOXML corpus is reproduced (Chapter 3, Section 3.4.7). Document identifiers of the local corpus are the first eight characters of their hash-derived file names.", "")
SEV = ("basic", "moderate", "severe")
def perdoc(tag, a, b, k=0):
    J = load(tag); out = collections.defaultdict(dict)
    for (m, s, seed), j in J.items():
        for r in j["rows"]:
            out[(m, s)].setdefault(r["document"], [[], []]); out[(m, s)][r["document"]][0].append(r[a][k]); out[(m, s)][r["document"]][1].append(r[b][k])
    return out
tno = [0]
def table(caption, header, rows, source):
    tno[0] += 1
    P(f"**Table A.{tno[0]}**: {caption}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header))
    for r in rows: P("| " + " | ".join(r) + " |")
    P("", f"*Source: {source}*", "")
def doc_tables(tag, a, b, label, idf, k=0, what="recall", src=""):
    D = perdoc(tag, a, b, k)
    for m in ("shuffled", "realistic"):
        docs = sorted(D[(m, "basic")])
        rows = []
        for d in docs:
            r = [idf(d)]
            for s in SEV:
                v = D[(m, s)][d]; r += [f"{np.mean(v[0]):.3f}", f"{np.mean(v[1]):.3f}"]
            rows.append(r)
        table(f"{label}: per-document mean {what}, {m} model ({len(docs)} documents; mean over the fragmentations of each cell)",
              ["Document", "Basic B0", "Basic B2", "Moderate B0", "Moderate B2", "Severe B0", "Severe B2"], rows, src)
P("## A.1 Per-document results of the restoration studies", "")
P("Tables A.1–A.6 give, for every document, the mean recall (OOXML) or precision (CFBF) of B0 and B2 in each cell. The document-level tests of Chapter 5 are computed from these values.", "")
doc_tables("dev", "B0", "B2", "OOXML development set", lambda d: d[:8], src="`writing/rescore/out_dev.jsonl` (ten fragmentations per cell).")
doc_tables("napier", "B0", "B2", "Word-written OOXML (NapierOne)", lambda d: d.replace("-docx.docx", ""), src="`writing/rescore/out_napier.jsonl` (ten fragmentations per cell).")
# CFBF 39-document precision
for m in ("shuffled", "realistic"):
    d = json.load(open(W + f"reanalysis_2026-09-30/cfbf_{m}.json"))["rows"]
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in d:
        if r["severity"] in SEV: g[r["document"]][r["severity"]].append((r["B0_precision"], r["B2_precision"]))
    rows = []
    for doc in sorted(g):
        r = [doc.replace(".doc", "")]
        for s in SEV:
            v = g[doc][s]; r += [f"{np.mean([x[0] for x in v]):.3f}", f"{np.mean([x[1] for x in v]):.3f}"]
        rows.append(r)
    table(f"CFBF sample: per-document mean precision, {m} model (39 documents; mean over five fragmentations)",
          ["Document", "Basic B0", "Basic B2", "Moderate B0", "Moderate B2", "Severe B0", "Severe B2"], rows, f"`writing/reanalysis_2026-09-30/cfbf_{m}.json`.")
# FAT32 volumes
P("## A.2 Constructed FAT32 volumes", "")
F = json.load(open(W + "rescore/fs_summary_rescored.json"))
rows = []
for tr in ("ooxml", "cfbf"):
    for c in F[tr]:
        rows.append([tr.upper(), {"D": "Deletion", "O": "Deletion + reuse"}[c["scenario"]], {"L": "large gaps", "S": "small gaps"}[c["fillers"]],
                     "10", str(c["n_obs"]), str(c["n_docs"]), f"{100*c['frag_share']:.1f} %", f"{c['mean_runs_fragmented']:.2f}"])
table("Composition of the constructed FAT32 volumes", ["Targets", "Scenario", "Fillers", "Volumes", "Target document-volumes", "Distinct documents", "Fragmented", "Mean fragments of a fragmented target"], rows,
      "`writing/rescore/fs_summary_rescored.json` (OOXML) and `writing/reanalysis_2026-10-07/fs_summary.json` (CFBF); each volume is 272 MiB with 4,096-byte clusters and 30 target documents.")
# all 63 tests
P("## A.3 The tests corrected together across the thesis", "")
G = json.load(open(W + "rescore/global_multiplicity_v3.json"))
rows = [[t["family"], t["cell"], f"{t['effect']:+.4f}", f"{t['p']:.2g}", f"{t['holm_global']:.2g}", f"{t['bh_global']:.2g}"] for t in G]
table("All 63 tests with their raw, global Holm and global Benjamini–Hochberg p-values", ["Family", "Cell", "Effect", "Raw p", "Global Holm p", "Global BH p"], rows, "`writing/rescore/global_multiplicity_v3.json`.")
open("/tmp/appendix_part1.md", "w").write("\n".join(L) + "\n")
print(len(L), "lines; tables", tno[0])
