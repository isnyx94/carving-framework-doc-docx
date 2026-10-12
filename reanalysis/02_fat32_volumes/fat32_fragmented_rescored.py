"""FAT32 OOXML, fragmented document-volumes only (runs > 1), re-scored: per-document mean B2 - B0 recall."""
import json, numpy as np, collections
from scipy import stats
rows = [json.loads(l) for l in open("./writing/rescore/out_fs.jsonl")]
fr = [r for r in rows if r["runs"] > 1]
per = collections.defaultdict(list)
for r in fr: per[r["document"]].append((r["B0_pos_recall"], r["B2_pos_recall"]))
d0 = np.array([np.mean([a for a, b in v]) for v in per.values()]); d2 = np.array([np.mean([b for a, b in v]) for v in per.values()])
dd = d2 - d0; rng = np.random.default_rng(0); i = rng.integers(0, len(dd), (10000, len(dd)))
out = {"n_obs": len(fr), "n_docs": len(dd), "B0": float(d0.mean()), "B2": float(d2.mean()), "eff": float(dd.mean()),
       "ci": [float(x) for x in np.percentile(dd[i].mean(1), [2.5, 97.5])], "p": float(stats.ttest_rel(d2, d0).pvalue),
       "better": int((dd > 1e-12).sum()), "worse": int((dd < -1e-12).sum()),
       "PRk": float(np.mean([r["PR_keep_pos_recall"] for r in fr]))}
json.dump(out, open("fat32_fragmented_rescored.json", "w"), indent=1); print(out)
