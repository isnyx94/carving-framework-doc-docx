import os, sys, json, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); R = "./writing/reanalysis_2026-10-06"; os.makedirs(R, exist_ok=True); budget = float(sys.argv[1])
jobs = [(sz, c) for sz in (512, 4096) for c in ("docx", "doc", "xls", "ppt", "fft_docx", "fft_doc", "fft_xls", "fft_ppt", "fft_xlsx", "fft_pptx")]
done = set()
if os.path.exists(R + "/cov.jsonl"):
    for l in open(R + "/cov.jsonl"):
        d = json.loads(l); s = d["set"]; key = ("fft_" + s.split()[0]) if "FFT-75" in s else s.split()[0]; done.add((d["size"], key))
pend = [j for j in jobs if j not in done]; print("pending", len(pend)); t0 = time.time(); run = []
while pend or run:
    run = [p for p in run if p.poll() is None]
    while pend and len(run) < 2 and time.time() - t0 < budget - 70:
        sz, c = pend.pop(0); run.append(subprocess.Popen([sys.executable, HERE + "/own_coverage.py", str(sz), R + "/cov.jsonl", c], stdout=open(R + "/log.txt", "a"), stderr=subprocess.STDOUT))
    if not run: break
    time.sleep(2)
print("elapsed", round(time.time() - t0), "left", len(pend))
