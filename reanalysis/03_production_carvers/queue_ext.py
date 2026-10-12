import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); R = "./writing/reanalysis_2026-10-06"; os.makedirs(R, exist_ok=True); budget = float(sys.argv[1])
jobs = [(t, m, s, k) for t in ("ooxml", "cfbf") for m in ("shuffled", "realistic") for s in ("basic", "moderate", "severe") for k in range(3)]
done = set()
if os.path.exists(R + "/ext.jsonl"):
    for l in open(R + "/ext.jsonl"):
        d = json.loads(l); done.add((d["track"], d["model"], d["severity"], d["seed"]))
pending = [j for j in jobs if j not in done]; print("pending", len(pending), flush=True)
t0 = time.time(); running = []
while pending or running:
    running = [p for p in running if p.poll() is None]
    while pending and len(running) < 2 and time.time() - t0 < budget - 60:
        t, m, s, k = pending.pop(0)
        running.append(subprocess.Popen([sys.executable, HERE + "/run_external.py", t, m, s, str(k), R + "/ext.jsonl"], stdout=open(R + "/log_ext.txt", "a"), stderr=subprocess.STDOUT))
    if not running: break
    time.sleep(2)
print("elapsed", round(time.time() - t0), "remaining", len(pending))
