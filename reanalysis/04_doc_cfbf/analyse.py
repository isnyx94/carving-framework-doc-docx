import json, numpy as np, collections
from scipy import stats
rng = np.random.default_rng(20260930)
P = "./results/"
def boot_ci(x, B=10000):
    x = np.asarray(x); idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1); return np.percentile(m, [2.5, 97.5])
def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); adj = np.empty_like(ps); run = 0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj
def exact_wilcoxon(d):
    d = np.asarray(d); d = d[d != 0]
    if len(d) == 0: return float('nan'), 0
    try: p = stats.wilcoxon(d, method='exact').pvalue
    except Exception: p = stats.wilcoxon(d).pvalue
    return float(p), len(d)
out = {}
# ---------- OOXML document-level re-analysis (stored 10-seed per-document data) ----------
ox = []
for model, f in [("shuffled", "restoration_scaled/restore_results.json"), ("realistic", "restoration_realistic/restore_results.json")]:
    d = json.load(open(P + f))
    for sev in ("basic", "moderate", "severe"):
        pdoc = d["by_severity"][sev]["per_document"]
        c, s = pdoc["contiguous"], pdoc["structural"]
        docs = sorted({x["document"] for x in c})
        def per_doc(rows, key): 
            g = collections.defaultdict(list)
            for x in rows: g[x["document"]].append(x[key])
            return np.array([np.mean(g[k]) for k in docs])
        r0, r2 = per_doc(c, "recall"), per_doc(s, "recall")
        p0, p2 = per_doc(c, "precision"), per_doc(s, "precision")
        dr, dp = r2 - r0, p2 - p0
        obs = np.array([b["recall"] - a["recall"] for a, b in zip(c, s)])
        worse_obs = [(a["document"], a["seed"], round(b["recall"] - a["recall"], 4)) for a, b in zip(c, s) if b["recall"] < a["recall"] - 1e-12]
        t = stats.ttest_rel(r2, r0); wp, nz = exact_wilcoxon(dr)
        nb, nw = int((dr > 1e-12).sum()), int((dr < -1e-12).sum())
        sign_p = stats.binomtest(nb, nb + nw).pvalue if nb + nw else float('nan')
        ci_r, ci_p = boot_ci(dr), boot_ci(dp)
        B0m = r0.mean()
        ox.append(dict(model=model, severity=sev, n_docs=len(docs), n_obs=len(c),
            B0_recall=r0.mean(), B2_recall=r2.mean(), effect_recall=dr.mean(), ci_recall=list(ci_r),
            B0_prec=p0.mean(), B2_prec=p2.mean(), effect_prec=dp.mean(), ci_prec=list(ci_p),
            t=float(t.statistic), p_t_doc=float(t.pvalue), p_wilcoxon_exact_doc=wp, n_nonzero_docs=nz,
            docs_better=nb, docs_worse=nw, p_sign_doc=float(sign_p),
            p_t_obs_original=float(stats.ttest_rel([x["recall"] for x in s], [x["recall"] for x in c]).pvalue),
            obs_better=int((obs > 1e-12).sum()), obs_worse=int((obs < -1e-12).sum()), worse_obs=worse_obs,
            headroom_gain=(dr.mean() / (1 - B0m)) if B0m < 1 else float('nan')))
adj = holm([r["p_t_doc"] for r in ox])
for r, a in zip(ox, adj): r["p_t_doc_holm"] = float(a)
out["ooxml_doclevel"] = ox
# ---------- OOXML CRC ablation (rerun, seeds 0-4 unless more) ----------
abl = []
for model in ("shuffled", "realistic"):
    for sev in ("basic", "moderate", "severe"):
        recs = [json.loads(l) for l in open(f"ooxml_{model}_{sev}.jsonl") if l.strip()]
        rows = [r for rec in recs for r in rec["rows"]]
        parts = collections.defaultdict(collections.Counter)
        for rec in recs:
            for m, v in rec["parts"].items(): parts[m].update(v)
        tim = collections.Counter()
        for rec in recs: tim.update(rec["timing_s"])
        diff = np.array([r["B2_recall"] - r["B2noCRC_recall"] for r in rows])
        diffp = np.array([r["B2_precision"] - r["B2noCRC_precision"] for r in rows])
        abl.append(dict(model=model, severity=sev, seeds=sorted({r["seed"] for r in rows}), n_obs=len(rows),
            B0_recall=np.mean([r["B0_recall"] for r in rows]), B2_recall=np.mean([r["B2_recall"] for r in rows]),
            B2noCRC_recall=np.mean([r["B2noCRC_recall"] for r in rows]),
            max_abs_recall_diff_B2_vs_noCRC=float(np.abs(diff).max()), max_abs_prec_diff=float(np.abs(diffp).max()),
            n_obs_differing=int((np.abs(diff) > 1e-12).sum()),
            parts=dict({m: dict(v) for m, v in parts.items()}),
            B0_5gram=np.mean([r["B0_5gram_recall"] for r in rows]), B2_5gram=np.mean([r["B2_5gram_recall"] for r in rows]),
            B0_xdoc=np.mean([r["B0_xdoc_recall"] for r in rows]), B2_xdoc=np.mean([r["B2_xdoc_recall"] for r in rows]),
            B2_xdoc_nonempty=np.mean([r["B2_xdoc_recall"] for r in rows if r["B2_ntok"] > 0]),
            timing_s_per_image=dict({m: v / len(recs) for m, v in tim.items()})))
out["ooxml_crc_ablation"] = abl
# ---------- CFBF document-level ----------
cf = []; cf_raw = {}
for model in ("shuffled", "realistic"):
    d = json.load(open(f"cfbf_{model}.json")); cf_raw[model] = d
    for sev in ("basic", "moderate", "severe"):
        rows = [r for r in d["rows"] if r["severity"] == sev]
        docs = sorted({r["document"] for r in rows})
        g = lambda key: np.array([np.mean([r[key] for r in rows if r["document"] == k]) for k in docs])
        r0, r2, p0, p2, r0s, p0s = g("B0_recall"), g("B2_recall"), g("B0_precision"), g("B2_precision"), g("B0_256K_recall"), g("B0_256K_precision")
        dr, dp = r2 - r0, p2 - p0
        if np.allclose(dr, 0): pt = float('nan')
        else: pt = float(stats.ttest_rel(r2, r0).pvalue)
        wp, nz = exact_wilcoxon(dr)
        nb, nw = int((dr > 1e-12).sum()), int((dr < -1e-12).sum())
        cf.append(dict(model=model, severity=sev, n_docs=len(docs), n_obs=len(rows),
            B0_recall=r0.mean(), B2_recall=r2.mean(), effect_recall=dr.mean(), ci_recall=list(boot_ci(dr)),
            B0_prec=p0.mean(), B2_prec=p2.mean(), effect_prec=dp.mean(), ci_prec=list(boot_ci(dp)),
            p_t_doc=pt, p_wilcoxon_exact_doc=wp, n_nonzero_docs=nz, docs_better=nb, docs_worse=nw,
            p_sign_doc=float(stats.binomtest(nb, nb + nw).pvalue) if nb + nw else float('nan'),
            B0_256K_recall=r0s.mean(), B0_256K_prec=p0s.mean(),
            B0_xdoc=np.mean([r["B0_xdoc_recall"] for r in rows]), B2_xdoc=np.mean([r["B2_xdoc_recall"] for r in rows])))
ps = [r["p_t_doc"] for r in cf if not np.isnan(r["p_t_doc"])]
adj = iter(holm(ps))
for r in cf: r["p_t_doc_holm"] = float(next(adj)) if not np.isnan(r["p_t_doc"]) else float('nan')
intact = [r for r in cf_raw["shuffled"]["rows"] if r["severity"] == "intact"]
out["cfbf_intact"] = {k: float(np.mean([r[k] for r in intact])) for k in intact[0] if k.endswith(("recall", "precision"))}
out["cfbf_intact"]["n_docs"] = len(intact)
out["cfbf_intact"]["zero_recall_docs"] = [r["document"] for r in intact if r["B2_recall"] == 0]
out["cfbf_doclevel"] = cf
out["cfbf_n_files"] = cf_raw["shuffled"]["n_files"]; out["cfbf_n_kept"] = cf_raw["shuffled"]["n_kept"]
# ---------- classification 4096-B per-seed CI/TOST ----------
d = np.array([0.0375, 0.0067, -0.0094, 0.0075, 0.0165])
se = d.std(ddof=1) / np.sqrt(5); tcrit = stats.t.ppf(0.975, 4)
ci = [d.mean() - tcrit * se, d.mean() + tcrit * se]
p_lower = 1 - stats.t.cdf((d.mean() + 0.02) / se, 4); p_upper = stats.t.cdf((d.mean() - 0.02) / se, 4)
out["cls4096"] = dict(mean=d.mean(), ci95=ci, tost_p=max(p_lower, p_upper), t_p=float(stats.ttest_1samp(d, 0).pvalue),
                      ci90=[d.mean() - stats.t.ppf(0.95, 4) * se, d.mean() + stats.t.ppf(0.95, 4) * se])
json.dump(out, open("reanalysis_results.json", "w"), indent=1, default=float)
def fmt(x): return f"{x:.4f}" if isinstance(x, float) else str(x)
print("OOXML doc-level")
for r in ox: print(r["model"], r["severity"], r["n_docs"], "eff", fmt(r["effect_recall"]), "CI", [round(v, 4) for v in r["ci_recall"]], "p_t", f"{r['p_t_doc']:.2g}", "holm", f"{r['p_t_doc_holm']:.2g}", "wilc", f"{r['p_wilcoxon_exact_doc']:.2g}", "b/w docs", r["docs_better"], r["docs_worse"], "obs b/w", r["obs_better"], r["obs_worse"], "prec eff", fmt(r["effect_prec"]), [round(v, 4) for v in r["ci_prec"]], "headroom", round(r["headroom_gain"], 3))
print("CRC ablation")
for r in abl: print(r["model"], r["severity"], r["seeds"], "B2", fmt(r["B2_recall"]), "noCRC", fmt(r["B2noCRC_recall"]), "maxdiff", r["max_abs_recall_diff_B2_vs_noCRC"], r["max_abs_prec_diff"], "parts", r["parts"], "5g", fmt(r["B0_5gram"]), fmt(r["B2_5gram"]), "xdoc", fmt(r["B0_xdoc"]), fmt(r["B2_xdoc"]), "time", {k: round(v, 1) for k, v in r["timing_s_per_image"].items()})
print("CFBF doc-level", out["cfbf_n_files"], out["cfbf_n_kept"], out["cfbf_intact"])
for r in cf: print(r["model"], r["severity"], r["n_docs"], "recall", fmt(r["B0_recall"]), fmt(r["B2_recall"]), "eff", fmt(r["effect_recall"]), [round(v, 4) for v in r["ci_recall"]], "p", f"{r['p_t_doc']:.3g}", "holm", f"{r['p_t_doc_holm']:.3g}", "wilc", f"{r['p_wilcoxon_exact_doc']:.3g}", "n0", r["n_nonzero_docs"], "b/w", r["docs_better"], r["docs_worse"], "prec", fmt(r["B0_prec"]), fmt(r["B2_prec"]), [round(v, 4) for v in r["ci_prec"]], "B0_256K", fmt(r["B0_256K_recall"]), fmt(r["B0_256K_prec"]), "xdoc", fmt(r["B0_xdoc"]), fmt(r["B2_xdoc"]))
print("cls4096", out["cls4096"])
