"""Re-run E2 (external carvers vs B0/B2 on synthetic OOXML images, both attributions) under the corrected extractor."""
import os, sys, json, time, runpy, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); R = "./writing/rescore"; os.makedirs(R, exist_ok=True); budget = float(sys.argv[1]); t0 = time.time()
D7 = "./writing/reanalysis_2026-10-07"
OUT = R + "/out_e2.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l); done.add((d["model"], d["severity"], d["seed"]))
jobs = [(m, s, k) for m in ("shuffled", "realistic") for s in ("basic", "moderate", "severe") for k in range(3)]
WRAP = HERE + "/_e2_wrap.py"
open(WRAP, "w").write("import sys, runpy\nsys.path.insert(0, '.'); sys.path.insert(0, '%s')\nimport strict_patch; strict_patch.apply()\nsys.argv = ['e2_symmetric.py'] + sys.argv[1:]\nrunpy.run_path('%s/e2_symmetric.py', run_name='__main__')\n" % (os.path.abspath('./reanalysis/common'), HERE))
pend = [j for j in jobs if j not in done]; run = []
while pend or run:
    run = [p for p in run if p.poll() is None]
    while pend and len(run) < 2 and time.time() - t0 < budget - 50:
        m, s, k = pend.pop(0)
        run.append(subprocess.Popen([sys.executable, WRAP, "ooxml", m, s, str(k), OUT], stdout=open(R + "/log_e2.txt", "a"), stderr=subprocess.STDOUT))
    if not run: break
    time.sleep(2)
print("remaining", len(pend))
