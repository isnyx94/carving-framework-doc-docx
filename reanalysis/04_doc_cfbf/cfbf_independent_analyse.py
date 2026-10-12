import json, numpy as np, collections, glob
from scipy import stats
rows = [json.loads(l) for f in glob.glob("cfbf_lo_*.jsonl") for l in open(f)]
rng = np.random.default_rng(0); out = []
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows: by[(r["model"], r["severity"])][r["document"]].append(r)
for key in sorted(by):
    docs = by[key]; c = {"model": key[0], "severity": key[1], "n_docs": len(docs)}
    for ref in ("h", "lo"):
        for met in ("recall", "precision"):
            b0 = np.array([np.mean([x[f"B0_{ref}_{met}"] for x in v]) for v in docs.values()])
            b2 = np.array([np.mean([x[f"B2_{ref}_{met}"] for x in v]) for v in docs.values()])
            d = b2 - b0; bs = rng.choice(d, (10000, len(d))).mean(1)
            c[f"{ref}_{met}"] = {"B0": float(b0.mean()), "B2": float(b2.mean()), "eff": float(d.mean()),
                                 "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                 "p": float(stats.ttest_1samp(d, 0).pvalue) if np.any(d) else 1.0}
    out.append(c)
    print(key, "precision B0/B2 heuristic %.3f/%.3f eff %+.3f | LibreOffice-text %.3f/%.3f eff %+.3f %s" % (
        c["h_precision"]["B0"], c["h_precision"]["B2"], c["h_precision"]["eff"], c["lo_precision"]["B0"], c["lo_precision"]["B2"], c["lo_precision"]["eff"], [round(x, 3) for x in c["lo_precision"]["ci"]]),
        "| recall heur %.3f/%.3f, LO %.3f/%.3f eff %+.4f" % (c["h_recall"]["B0"], c["h_recall"]["B2"], c["lo_recall"]["B0"], c["lo_recall"]["B2"], c["lo_recall"]["eff"]))
json.dump(out, open("cfbf_independent_reference.json", "w"), indent=1)
