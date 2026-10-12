"""Thesis-wide multiplicity check: every restoration and classification hypothesis test reported with a
p-value is collected, and Holm (family-wise) and Benjamini-Hochberg (false discovery rate) corrections are
applied across all of them at once. v3: OOXML families from the re-scored runs (corrected text extractor)."""
import json, numpy as np, collections
from scipy import stats
W = "./writing/"
tests = []
def add(fam, cell, p, eff): tests.append({"family": fam, "cell": cell, "p": float(p), "effect": float(eff)})
# classification
add("classification", "512 B macro-F1", 0.666, 0.0011); add("classification", "4,096 B macro-F1", 0.200, 0.0118)
r = json.load(open(W + "reanalysis_2026-09-30/reanalysis_results.json"))
RS = json.load(open(W + "rescore/rescore_summary.json"))
for c in RS["dev"]: add("OOXML dev recall", f'{c["model"]}-{c["severity"]}', c["p"], c["eff"])
for c in r["cfbf_doclevel"]:
    if c["p_t_doc"] is not None and not (isinstance(c["p_t_doc"], float) and np.isnan(c["p_t_doc"])):
        add("CFBF-39 recall", f'{c["model"]}-{c["severity"]}', c["p_t_doc"], c["effect_recall"])
# CFBF-39 precision from per-document rows
for model in ("shuffled", "realistic"):
    d = json.load(open(W + f"reanalysis_2026-09-30/cfbf_{model}.json"))
    rows = d["rows"]; by = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in rows:
        if x.get("severity") in (None, "intact"): continue
        by[x["severity"]][x["document"]].append(x)
    for sev, docs in by.items():
        dd = np.array([np.mean([y["B2_precision"] for y in v]) - np.mean([y["B0_precision"] for y in v]) for v in docs.values()])
        add("CFBF-39 precision", f"{model}-{sev}", stats.ttest_1samp(dd, 0).pvalue, dd.mean())
n = json.load(open(W + "reanalysis_2026-10-06/new_results.json"))
for c in RS["heldout"]: add("OOXML held-out recall", f'{c["model"]}-{c["severity"]}', c["p"], c["eff"])
for c in n["cfbf108"]: add("CFBF-108 recall", f'{c["model"]}-{c["severity"]}', c["p"], c["eff"])
rows = [json.loads(l) for l in open(W + "reanalysis_2026-10-06/cfbf109.jsonl")]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for x in rows:
    if x["model"] == "intact": continue
    by[(x["model"], x["severity"])][x["document"]].append(x)
for (m, s), docs in by.items():
    dd = np.array([np.mean([y["B2_precision"] for y in v]) - np.mean([y["B0_precision"] for y in v]) for v in docs.values()])
    add("CFBF-108 precision", f"{m}-{s}", stats.ttest_1samp(dd, 0).pvalue, dd.mean())
fs = json.load(open(W + "reanalysis_2026-10-07/fs_summary.json"))
fsr = json.load(open(W + "rescore/fs_summary_rescored.json"))
for c in fsr["ooxml"]: add("FAT32 OOXML recall", f'{c["scenario"]}-{c["fillers"]}', c["p_recall"], c["eff_recall"])
for c in fs["cfbf"]: add("FAT32 CFBF precision", f'{c["scenario"]}-{c["fillers"]}', c["p_precision"], c["eff_precision"])
for c in RS["lo"]: add("LibreOffice OOXML recall (exploratory)", f'{c["model"]}-{c["severity"]}', c["p"], c["eff"])
for c in json.load(open(W + "reanalysis_2026-10-07/e4_summary.json")): add("Openable after repair", f'{c["model"]}-{c["severity"]}', c["p"], c["eff"])
for c in RS["napier"]: add("NapierOne (Word) OOXML recall", f'{c["model"]}-{c["severity"]}', c["p"], c["eff"])
p = np.array([t["p"] for t in tests]); m = len(p)
order = np.argsort(p); holm = np.empty(m); run = 0
for k, i in enumerate(order): run = max(run, min(1, (m - k) * p[i])); holm[i] = run
bh = np.empty(m); prev = 1
for k in range(m - 1, -1, -1):
    i = order[k]; prev = min(prev, p[i] * m / (k + 1)); bh[i] = prev
for t, h, b in zip(tests, holm, bh): t["holm_global"], t["bh_global"] = float(h), float(b)
json.dump(tests, open(W + "rescore/global_multiplicity_v3.json", "w"), indent=1)
print("tests:", m)
fam = collections.OrderedDict()
for t in tests: fam.setdefault(t["family"], []).append(t)
for f, ts in fam.items():
    print(f"{f:40s} n={len(ts)} raw<.05={sum(t['p']<.05 for t in ts)} holm<.05={sum(t['holm_global']<.05 for t in ts)} bh<.05={sum(t['bh_global']<.05 for t in ts)}")
for t in tests:
    if t["p"] < .05 and t["holm_global"] >= .05: print("  loses under global Holm:", t["family"], t["cell"], round(t["p"], 4), round(t["holm_global"], 4), "BH", round(t["bh_global"], 4))
