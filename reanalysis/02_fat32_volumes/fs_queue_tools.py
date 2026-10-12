"""Run E1 jobs until a time budget is used. Usage: python fs_queue.py <seconds>"""
import sys, os, time, subprocess, shutil
D = "./writing/reanalysis_2026-10-07"
OUT = D + "/fs_tools2.jsonl"
import fcntl; DONE = D + "/fs_tools2_done.txt"
budget = float(sys.argv[1]); t0 = time.time()
shard = int(sys.argv[2]) if len(sys.argv) > 2 else -1
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
jobs = [(tr, sc, fz, s) for s in range(10) for tr in ("ooxml", "cfbf") for sc in ("D", "O") for fz in ("L", "S")]
for ji, (tr, sc, fz, s) in enumerate(jobs):
    if shard >= 0 and ji % 2 != shard: continue
    key = f"{tr}_{sc}_{fz}_{s}"
    if key in done: continue
    if time.time() - t0 > budget: break
    wd = f"/tmp/fsq/{key}"
    subprocess.run(["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_build.py"), tr, sc, str(s), wd, fz], check=True)
    subprocess.run(["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_run.py"), wd, OUT, "tools"], check=True)
    shutil.copy(wd + "/truth.json", f"{D}/fs_truth/{key}.json") if os.path.isdir(D + "/fs_truth") else None
    shutil.rmtree(wd, ignore_errors=True)
    open(DONE, "a").write(key + "\n")
print("remaining", len([1 for j in jobs if "_".join(map(str, j)) not in set(open(DONE).read().split())]))
