import json, numpy as np
from scipy import stats
out = {}
for mode in ("S", "M"):
    J = [json.loads(l) for l in open(f"boiko_{mode}.jsonl")]
    rows = [r for j in J for r in j["rows"]]
    res = {"observations": len(rows), "seeds": len(J), "multi_cluster_share": float(np.mean([r["clusters"] > 1 for r in rows]))}
    for m in ("B0", "B2"):
        rec = np.array([r[m]["recall"] for r in rows]); mr = np.array([r[m]["main_recall"] for r in rows if r[m]["main_recall"] is not None])
        res[m] = {"mean_recall": float(rec.mean()), "mean_precision": float(np.mean([r[m]["precision"] for r in rows])),
                  "full_text": float(np.mean(rec >= 0.999)), "partial_text": float(np.mean((rec > 0) & (rec < 0.999))), "no_text": float(np.mean(rec == 0)),
                  "main_text_full": float(np.mean(mr >= 0.999)), "main_part_crc": float(np.mean([r[m]["main_crc"] for r in rows])),
                  "all_text_parts_crc": float(np.mean([r[m]["parts"] > 0 and r[m]["parts_crc"] == r[m]["parts"] for r in rows]))}
    docs = sorted({r["document"] for r in rows})
    d = np.array([np.mean([r["B2"]["recall"] - r["B0"]["recall"] for r in rows if r["document"] == x]) for x in docs])
    res["effect"] = float(d.mean()); res["p"] = float(stats.ttest_1samp(d, 0).pvalue); res["better"] = int((d > 1e-12).sum()); res["worse"] = int((d < -1e-12).sum())
    out[mode] = res
    print(mode, json.dumps(res, indent=0))
json.dump(out, open("boiko_summary.json", "w"), indent=1)
