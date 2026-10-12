import json, numpy as np, collections
from scipy import stats
rng = np.random.default_rng(20261007)
def boot(x): x=np.asarray(x); i=rng.integers(0,len(x),(10000,len(x))); m=x[i].mean(1); return np.percentile(m,[2.5,97.5])
def holm(ps):
    ps=np.asarray(ps,float); o=np.argsort(ps); a=np.empty_like(ps); run=0
    for r,i in enumerate(o): run=max(run,(len(ps)-r)*ps[i]); a[i]=min(1,run)
    return a
def exw(d):
    d=np.asarray(d); d=d[np.abs(d)>1e-12]
    if len(d)==0: return float('nan')
    try: return float(stats.wilcoxon(d,method='exact').pvalue)
    except Exception: return float(stats.wilcoxon(d).pvalue)
out={}
# held-out OOXML
H='./work/heldout/'
ox=[]
for model in ('shuffled','realistic'):
    for sev in ('basic','moderate','severe'):
        rows=[r for l in open(H+f'heldout_{model}_{sev}.jsonl') for r in json.loads(l)['rows']]
        docs=sorted({r['document'] for r in rows})
        g=lambda k: np.array([np.mean([r[k] for r in rows if r['document']==d]) for d in docs])
        r0,r2,p0,p2=g('B0_recall'),g('B2_recall'),g('B0_precision'),g('B2_precision'); dr=r2-r0; dp=p2-p0
        pt=float(stats.ttest_rel(r2,r0).pvalue) if not np.allclose(dr,0) else float('nan')
        ox.append(dict(model=model,severity=sev,n_docs=len(docs),n_obs=len(rows),B0=r0.mean(),B2=r2.mean(),eff=dr.mean(),ci=list(boot(dr)),p=pt,w=exw(dr),better=int((dr>1e-12).sum()),worse=int((dr<-1e-12).sum()),P0=p0.mean(),P2=p2.mean(),effp=dp.mean(),cip=list(boot(dp))))
ps=[r['p'] for r in ox if r['p']==r['p']]; it=iter(holm(ps))
for r in ox: r['holm']=float(next(it)) if r['p']==r['p'] else float('nan')
out['ooxml_heldout']=ox
# CFBF 108
rows=[json.loads(l) for l in open('./work/cfbf109/cfbf109.jsonl')]
cf=[]
for model in ('shuffled','realistic'):
    for sev in ('basic','moderate','severe'):
        rr=[r for r in rows if r['model']==model and r['severity']==sev]
        docs=sorted({r['document'] for r in rr})
        g=lambda k: np.array([np.mean([r[k] for r in rr if r['document']==d]) for d in docs])
        r0,r2,p0,p2=g('B0_recall'),g('B2_recall'),g('B0_precision'),g('B2_precision'); dr=r2-r0; dp=p2-p0
        pt=float(stats.ttest_rel(r2,r0).pvalue) if not np.allclose(dr,0) else float('nan')
        cf.append(dict(model=model,severity=sev,n_docs=len(docs),B0=r0.mean(),B2=r2.mean(),eff=dr.mean(),ci=list(boot(dr)),p=pt,better=int((dr>1e-12).sum()),worse=int((dr<-1e-12).sum()),P0=p0.mean(),P2=p2.mean(),effp=dp.mean(),cip=list(boot(dp))))
ps=[r['p'] for r in cf if r['p']==r['p']]; it=iter(holm(ps))
for r in cf: r['holm']=float(next(it)) if r['p']==r['p'] else float('nan')
it=[r for r in rows if r['severity']=='intact']
out['cfbf108_intact']={k:float(np.mean([r[k] for r in it])) for k in ('B0_recall','B0_precision','B2_recall','B2_precision')}; out['cfbf108_intact']['n']=len(it)
out['cfbf108']=cf
json.dump(out,open('./work/cfbf109/new_results.json','w'),indent=1,default=float)
for k in ('ooxml_heldout','cfbf108'):
    print(k)
    for r in out[k]: print(f"  {r['model']:9} {r['severity']:8} n={r['n_docs']} B0={r['B0']:.4f} B2={r['B2']:.4f} eff={r['eff']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.2g} holm={r['holm']:.2g} b/w={r['better']}/{r['worse']} P0={r['P0']:.4f} P2={r['P2']:.4f} effp={r['effp']:+.4f} [{r['cip'][0]:+.4f},{r['cip'][1]:+.4f}]")
print(out['cfbf108_intact'])
