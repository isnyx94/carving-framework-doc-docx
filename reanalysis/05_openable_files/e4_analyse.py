import json, numpy as np, collections
from scipy import stats
D = "./writing/reanalysis_2026-10-07"
rows = {}
for l in open(D + "/e4.jsonl"):
    r = json.loads(l); rows[(r["model"], r["severity"], r["seed"], r["document"])] = r
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0]*len(ps); run = 0
    for k, i in enumerate(idx): run = max(run, min(1, (len(ps)-k)*ps[i])); out[i] = run
    return out
rng = np.random.default_rng(0); cells = []
for mo in ("shuffled", "realistic"):
    for sv in ("basic", "moderate", "severe"):
        per = collections.defaultdict(list)
        for k, r in rows.items():
            if k[:2] == (mo, sv): per[k[3]].append(r)
        dm = lambda f: np.array([np.mean([float(x[f]) for x in v]) for v in per.values()])
        c = {"model": mo, "severity": sv, "n_docs": len(per)}
        for m in ("B0", "B2"):
            for f in ("wellformed", "opens", "opens_repaired", "crc"): c[f"{m}_{f}"] = float(dm(f"{m}_{f}").mean())
        d = dm("B2_opens_repaired") - dm("B0_opens_repaired")
        b = rng.choice(d, (10000, len(d))).mean(1)
        c["eff"] = float(d.mean()); c["ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        c["p"] = float(stats.ttest_1samp(d, 0).pvalue) if np.any(d) else 1.0
        d2 = dm("B2_opens") - dm("B0_opens"); c["eff_strict"] = float(d2.mean())
        c["p_strict"] = float(stats.ttest_1samp(d2, 0).pvalue) if np.any(d2) else 1.0
        cells.append(c)
for c, h in zip(cells, holm([c["p"] for c in cells])): c["holm"] = h
for c, h in zip(cells, holm([c["p_strict"] for c in cells])): c["holm_strict"] = h
json.dump(cells, open(D + "/e4_summary.json", "w"), indent=1)
for c in cells:
    print(c["model"], c["severity"], c["n_docs"], "opens strict B0 %.3f B2 %.3f (eff %+.3f holm %.2g) | repaired B0 %.3f B2 %.3f eff %+.3f [%.3f,%.3f] holm %.2g | wf %.3f/%.3f crc %.3f/%.3f" % (
        c["B0_opens"], c["B2_opens"], c["eff_strict"], c["holm_strict"], c["B0_opens_repaired"], c["B2_opens_repaired"], c["eff"], *c["ci"], c["holm"],
        c["B0_wellformed"], c["B2_wellformed"], c["B0_crc"], c["B2_crc"]))
