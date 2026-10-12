"""Document-level analysis of the re-scored runs (same tests as the original analyses)."""
import json, sys, numpy as np, collections
from scipy import stats
rng = np.random.default_rng(20261008)
def boot(x): x = np.asarray(x); i = rng.integers(0, len(x), (10000, len(x))); return [float(v) for v in np.percentile(x[i].mean(1), [2.5, 97.5])]
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); a = np.empty_like(ps); run = 0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); a[i] = min(1, run)
    return a
def exw(d):
    d = np.asarray(d); d = d[np.abs(d) > 1e-12]
    if len(d) == 0: return float("nan")
    try: return float(stats.wilcoxon(d, method="exact").pvalue)
    except Exception: return float(stats.wilcoxon(d).pvalue)
def load(tag):
    J = {}
    for l in open(f"out_{tag}.jsonl"):
        if l.strip(): j = json.loads(l); J[(j["model"], j["severity"], j["seed"])] = j
    return J
def analyse(tag, a="B0", b="B2", seeds=None):
    J = load(tag); out = []
    for model in ("shuffled", "realistic"):
        for sev in ("basic", "moderate", "severe"):
            js = [J[k] for k in sorted(J) if k[0] == model and k[1] == sev and (seeds is None or k[2] in seeds)]
            if not js: continue
            rows = [(j["seed"], r) for j in js for r in j["rows"]]
            docs = sorted({r["document"] for _, r in rows})
            g = lambda m, i: np.array([np.mean([r[m][i] for _, r in rows if r["document"] == d]) for d in docs])
            r0, r2, p0, p2 = g(a, 0), g(b, 0), g(a, 1), g(b, 1); dr, dp = r2 - r0, p2 - p0
            pt = float(stats.ttest_rel(r2, r0).pvalue) if not np.allclose(dr, 0) else 1.0
            ptp = float(stats.ttest_rel(p2, p0).pvalue) if not np.allclose(dp, 0) else 1.0
            micro = lambda m: sum(r[m][4] for _, r in rows) / max(1, sum(r[m][2] for _, r in rows))
            obs = np.array([r[b][0] - r[a][0] for _, r in rows])
            c = dict(model=model, severity=sev, n_docs=len(docs), n_obs=len(rows), seeds=len(js),
                     B0=float(r0.mean()), B2=float(r2.mean()), eff=float(dr.mean()), ci=boot(dr), p=pt, w=exw(dr),
                     better=int((dr > 1e-12).sum()), worse=int((dr < -1e-12).sum()),
                     P0=float(p0.mean()), P2=float(p2.mean()), effp=float(dp.mean()), cip=boot(dp), pp=ptp,
                     micro0=micro(a), micro2=micro(b),
                     zero0=sum(1 for _, r in rows if r[a][0] == 0), zero2=sum(1 for _, r in rows if r[b][0] == 0),
                     g5_0=float(np.mean([r[a + "_5g"] for _, r in rows])), g5_2=float(np.mean([r[b + "_5g"] for _, r in rows])),
                     x0=float(np.mean([r[a + "_x"] for _, r in rows])), x2=float(np.mean([r[b + "_x"] for _, r in rows])),
                     obs_better=int((obs > 1e-12).sum()), obs_worse=int((obs < -1e-12).sum()), p_obs=float(stats.ttest_rel([r[b][0] for _, r in rows],[r[a][0] for _, r in rows]).pvalue) if not np.allclose(obs,0) else 1.0, Pmacro0=float(np.mean([r[a][1] for _, r in rows])), Pmacro2=float(np.mean([r[b][1] for _, r in rows])),
                     headroom=float(dr.mean() / (1 - r0.mean())) if r0.mean() < 1 else float("nan"),
                     crc0=sum(j["parts"].get(a, {}).get("crc_verified", 0) for j in js), crc2=sum(j["parts"].get(b, {}).get("crc_verified", 0) for j in js),
                     comp0=sum(j["parts"].get(a, {}).get("complete", 0) for j in js), comp2=sum(j["parts"].get(b, {}).get("complete", 0) for j in js),
                     nparts=sum(j["parts"].get(a, {}).get("parts", 0) for j in js))
            if all("B2nocrc" in j["modes"] for j in js):
                c["nocrc"] = float(np.mean([r["B2nocrc"][0] for _, r in rows])); c["b2_seed04"] = float(np.mean([r["B2"][0] for _, r in rows]))
                c["nocrc_diff_obs"] = sum(1 for _, r in rows if abs(r["B2nocrc"][0] - r["B2"][0]) > 1e-12)
                c["nocrc_maxdiff"] = max(abs(r["B2nocrc"][0] - r["B2"][0]) for _, r in rows)
            out.append(c)
    h = holm([c["p"] for c in out])
    for c, x in zip(out, h): c["holm"] = float(x)
    return out
if __name__ == "__main__":
    res = {}
    for tag, kw in (("dev", {}), ("heldout", {}), ("napier", {}), ("lo", {"a": "B0dd", "b": "B2dd"})):
        try: res[tag] = analyse(tag, **kw)
        except FileNotFoundError: continue
    res["crc_ablation"] = analyse("dev", seeds={0, 1, 2, 3, 4})
    try:
        J = {}
        for jb in (100, 1600):
            js = load(f"jump{jb}"); J[jb] = float(np.mean([r["B2"][0] for j in js.values() for r in j["rows"]]))
        res["jump"] = J
    except FileNotFoundError: pass
    json.dump(res, open("rescore_summary.json", "w"), indent=1)
    for tag in ("dev", "heldout", "napier", "lo", "crc_ablation"):
        if tag not in res: continue
        print("==", tag)
        for c in res[tag]:
            print(f"{c['model']:9} {c['severity']:8} n={c['n_docs']} B0={c['B0']:.4f} B2={c['B2']:.4f} eff={c['eff']:+.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] p={c['p']:.2g} holm={c['holm']:.2g} b/w={c['better']}/{c['worse']} P {c['P0']:.3f}->{c['P2']:.3f} ({c['effp']:+.3f}) zero {c['zero0']}->{c['zero2']} micro {c['micro0']:.3f}->{c['micro2']:.3f}" + (f" noCRC={c['nocrc']:.4f} B2={c['b2_seed04']:.4f} diffobs={c['nocrc_diff_obs']}" if 'nocrc' in c else ""))
    print("jump", res.get("jump"))
