import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); R = "./writing/rescore"; os.makedirs(R, exist_ok=True); budget = float(sys.argv[1]); t0 = time.time()
done = {l.split()[0] for l in open(R + "/fs_rescore_done.txt")} if os.path.exists(R + "/fs_rescore_done.txt") else set()
keys = [f"ooxml_{sc}_{fz}_{s}" for s in range(10) for sc in ("D", "O") for fz in ("L", "S")]
for k in keys:
    if k in done: continue
    if time.time() - t0 > budget: break
    subprocess.run([sys.executable, HERE + "/fs_rescore.py", k, R + "/out_fs.jsonl"], check=True, stdout=open(R + "/log_fs.txt", "a"), stderr=subprocess.STDOUT)
done = {l.split()[0] for l in open(R + "/fs_rescore_done.txt")}
print("remaining", len([k for k in keys if k not in done]))
