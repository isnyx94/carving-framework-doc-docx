"""FAT32 analysis with OOXML rows re-scored by the corrected extractor (CFBF rows unchanged)."""
import json, numpy as np, collections
from scipy import stats
D = "./writing/reanalysis_2026-10-07"
rows = {}
R = "./writing/rescore"
for l in open(D + "/fs_results.jsonl"):
    r = json.loads(l)
    if r["track"] == "cfbf": rows[(r["track"], r["scenario"], r["fillers"], r["seed"], r["document"])] = r
for l in open(R + "/out_fs.jsonl"):
    r = json.loads(l); rows[(r["track"], r["scenario"], r["fillers"], r["seed"], r["document"])] = r
tools = {}
for l in open(D + "/fs_tools2.jsonl"):
    r = json.loads(l)
    if r["track"] == "cfbf": tools[(r["track"], r["scenario"], r["fillers"], r["seed"], r["document"])] = r
for l in open(R + "/out_fs.jsonl"):
    r = json.loads(l); tools[(r["track"], r["scenario"], r["fillers"], r["seed"], r["document"])] = r
mism = 0
for k, r in rows.items():
    t = tools[k]; mism += t["runs"] != r["runs"]
    for kk, v in t.items():
        if kk.startswith(("PR_", "FM_")): r[kk] = v
print("rebuild mismatches in fragmentation:", mism, "of", len(rows))
rows = list(rows.values())
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); run = 0
    for k, i in enumerate(idx):
        run = max(run, min(1, (m - k) * ps[i])); out[i] = run
    return out
def boot(d, rng=np.random.default_rng(0)):
    d = np.asarray(d); b = rng.choice(d, (10000, len(d))).mean(1); return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
res = {}
for tr in ("ooxml", "cfbf"):
    cells = []
    for sc in ("D", "O"):
        for fz in ("L", "S"):
            rs = [r for r in rows if r["track"] == tr and r["scenario"] == sc and r["fillers"] == fz]
            per = collections.defaultdict(list)
            for r in rs: per[r["document"]].append(r)
            def dm(key): return np.array([np.mean([x[key] for x in v]) for v in per.values()])
            c = {"track": tr, "scenario": sc, "fillers": fz, "n_obs": len(rs), "n_docs": len(per),
                 "frag_share": float(np.mean([r["runs"] > 1 for r in rs])), "mean_runs_fragmented": float(np.mean([r["runs"] for r in rs if r["runs"] > 1] or [1]))}
            for m in ("B0", "B2", "PR_default", "PR_keep", "FM"):
                for met in ("recall", "precision"):
                    c[f"{m}_{met}"] = float(dm(f"{m}_pos_{met}").mean())
                    c[f"{m}_{met}_bm"] = float(dm(f"{m}_bm_{met}").mean()) if f"{m}_bm_{met}" in rs[0] else float("nan")
            fr = [r for r in rs if r["runs"] > 1]
            for m in ("B0", "B2", "PR_keep"):
                for met in ("recall", "precision"):
                    c[f"frag_{m}_{met}"] = float(np.mean([r[f"{m}_pos_{met}"] for r in fr])) if fr else None
            c["n_frag_obs"] = len(fr)
            for met in ("recall", "precision"):
                d = dm(f"B2_pos_{met}") - dm(f"B0_pos_{met}")
                c[f"eff_{met}"] = float(d.mean()); c[f"ci_{met}"] = boot(d)
                c[f"p_{met}"] = float(stats.ttest_1samp(d, 0).pvalue) if np.any(d != 0) else 1.0
                c[f"better_{met}"] = int((d > 0).sum()); c[f"worse_{met}"] = int((d < 0).sum())
            cells.append(c)
    for met in ("recall", "precision"):
        for c, h in zip(cells, holm([c[f"p_{met}"] for c in cells])): c[f"holm_{met}"] = h
    res[tr] = cells
json.dump(res, open(R + "/fs_summary_rescored.json", "w"), indent=1)
for tr, cells in res.items():
    for c in cells:
        print(tr, c["scenario"], c["fillers"], "docs", c["n_docs"], "obs", c["n_obs"], "frag%", round(100 * c["frag_share"], 1), "fragobs", c["n_frag_obs"])
        print("   R  B0 %.3f B2 %.3f PRd %.3f PRk %.3f FM %.3f | eff %+.4f %s holm %.3g b/w %d/%d" % (c["B0_recall"], c["B2_recall"], c["PR_default_recall"], c["PR_keep_recall"], c["FM_recall"], c["eff_recall"], [round(x, 4) for x in c["ci_recall"]], c["holm_recall"], c["better_recall"], c["worse_recall"]))
        print("   P  B0 %.3f B2 %.3f PRd %.3f PRk %.3f FM %.3f | eff %+.4f %s holm %.3g" % (c["B0_precision"], c["B2_precision"], c["PR_default_precision"], c["PR_keep_precision"], c["FM_precision"], c["eff_precision"], [round(x, 4) for x in c["ci_precision"]], c["holm_precision"]))
        print("   frag-only R B0 %s B2 %s PRk %s | P B0 %s B2 %s PRk %s" % tuple(None if c[k] is None else round(c[k], 3) for k in ["frag_B0_recall", "frag_B2_recall", "frag_PR_keep_recall", "frag_B0_precision", "frag_B2_precision", "frag_PR_keep_precision"]))
        print("   bestmatch R B0 %.3f B2 %.3f PRk %.3f FM %.3f" % (c["B0_recall_bm"], c["B2_recall_bm"], c["PR_keep_recall_bm"], c["FM_recall_bm"]))
