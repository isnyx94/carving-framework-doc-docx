"""Re-scored version (OOXML rows from the corrected extractor).
FAT32 study, sensitivity analysis: documents appear in different numbers of volumes, so the effect is
re-estimated with every document-volume observation weighted equally (i.e. documents weighted by the number
of volumes they appear in), with a document-cluster bootstrap for the confidence interval and p-value."""
import json, numpy as np, collections
W = "./writing/"
rows = {}
for l in open(W + "reanalysis_2026-10-07/fs_results.jsonl"):
    r = json.loads(l)
    if r["track"] == "cfbf": rows[(r["track"], r["scenario"], r["fillers"], r["seed"], r["document"])] = r
for l in open(W + "rescore/out_fs.jsonl"):
    r = json.loads(l); rows[(r["track"], r["scenario"], r["fillers"], r["seed"], r["document"])] = r
rng = np.random.default_rng(0); out = []
for tr, met in (("ooxml", "recall"), ("cfbf", "precision"), ("cfbf", "recall")):
    for sc in ("D", "O"):
        for fz in ("L", "S"):
            per = collections.defaultdict(list)
            for k, r in rows.items():
                if k[:3] == (tr, sc, fz): per[r["document"]].append(r[f"B2_pos_{met}"] - r[f"B0_pos_{met}"])
            docs = list(per.values()); n = np.array([len(v) for v in docs]); s = np.array([sum(v) for v in docs])
            eff = s.sum() / n.sum(); unweighted = float(np.mean([np.mean(v) for v in docs]))
            idx = rng.integers(0, len(docs), (10000, len(docs)))
            boot = s[idx].sum(1) / n[idx].sum(1)
            # p-value: bootstrap of the centred statistic
            p = float(min(1, 2 * min(np.mean(boot - eff + 0 <= -eff + 0) if eff > 0 else 1, np.mean(boot - eff >= -eff) if eff < 0 else 1))) if eff != 0 else 1.0
            c = {"track": tr, "metric": met, "scenario": sc, "fillers": fz, "n_docs": len(docs), "n_obs": int(n.sum()),
                 "appearances_min": int(n.min()), "appearances_max": int(n.max()), "effect_weighted": float(eff),
                 "ci_weighted": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                 "p_boot": p, "effect_unweighted": unweighted}
            out.append(c)
            print(tr, met, sc, fz, "docs", len(docs), "appear", n.min(), "-", n.max(), "weighted %+.4f [%.4f, %.4f] p %.3g | unweighted %+.4f" % (eff, *c["ci_weighted"], p, unweighted))
json.dump(out, open(W + "rescore/fat32_weighted_rescored.json", "w"), indent=1)
