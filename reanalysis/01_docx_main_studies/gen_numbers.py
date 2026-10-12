"""Collect every re-scored number needed for the thesis text into numbers.json."""
import json, numpy as np, collections
from rescore_analyse import load
S = json.load(open("rescore_summary.json"))
N = {"summary": S}
# CRC ablation table (seeds 0-4)
J = load("dev"); tab = []
for model in ("shuffled", "realistic"):
    for sev in ("basic", "moderate", "severe"):
        js = [J[(model, sev, k)] for k in range(5)]
        rows = [r for j in js for r in j["rows"]]
        f = lambda m, i: float(np.mean([r[m][i] for r in rows]))
        P = lambda m, k: sum(j["parts"][m].get(k, 0) for j in js)
        tab.append(dict(model=model, severity=sev, obs=len(rows), B0=f("B0", 0), B2=f("B2", 0), noCRC=f("B2nocrc", 0),
            maxdiff=max(abs(r["B2"][0] - r["B2nocrc"][0]) for r in rows), ndiff=sum(1 for r in rows if abs(r["B2"][0] - r["B2nocrc"][0]) > 1e-12),
            parts=P("B0", "parts"), comp=(P("B0", "complete"), P("B2", "complete"), P("B2nocrc", "complete")),
            crc=(P("B0", "crc_verified"), P("B2", "crc_verified"), P("B2nocrc", "crc_verified")),
            g5=(float(np.mean([r["B0_5g"] for r in rows])), float(np.mean([r["B2_5g"] for r in rows]))),
            x=(float(np.mean([r["B0_x"] for r in rows])), float(np.mean([r["B2_x"] for r in rows]))),
            t=(float(np.mean([j["timing"]["B0"] for j in js])), float(np.mean([j["timing"]["B2"] for j in js])))))
N["crc_table"] = tab
# parts over all 10 seeds (development set)
pt = []
for model in ("shuffled", "realistic"):
    for sev in ("basic", "moderate", "severe"):
        js = [J[(model, sev, k)] for k in range(10)]
        P = lambda m, k: sum(j["parts"][m].get(k, 0) for j in js)
        pt.append(dict(model=model, severity=sev, parts=(P("B0", "parts"), P("B2", "parts")), comp=(P("B0", "complete"), P("B2", "complete")), crc=(P("B0", "crc_verified"), P("B2", "crc_verified"))))
N["parts10"] = pt
# external carvers on synthetic OOXML images (position attribution and best match)
E = [json.loads(l) for l in open("out_e2.jsonl")]
ext = []
for model in ("shuffled", "realistic"):
    for sev in ("basic", "moderate", "severe"):
        es = [e for e in E if e["model"] == model and e["severity"] == sev]
        g = lambda k: float(np.mean([e[k] for e in es]))
        ext.append({"model": model, "severity": sev, **{k: g(k) for k in es[0] if k.endswith(("recall", "precision"))}})
N["ext"] = ext
N["fs"] = json.load(open("fs_summary_rescored.json"))
N["fs_frag"] = json.load(open("fat32_fragmented_rescored.json"))
N["fs_weighted"] = json.load(open("fat32_weighted_rescored.json"))
json.dump(N, open("numbers.json", "w"), indent=1, default=float)
for t in tab: print(t)
for p in pt: print(p)
for e in ext: print({k: round(v, 3) if isinstance(v, float) else v for k, v in e.items() if "recall" in k or k in ("model", "severity")})
