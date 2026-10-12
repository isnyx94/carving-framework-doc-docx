import sys, os, time, subprocess
D = "./writing/reanalysis_2026-10-07"
OUT, DONE = D + "/e2b.jsonl", D + "/e2b_done.txt"
budget, shard = float(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
jobs = [(t, m, s, k) for t in ("ooxml", "cfbf") for m in ("shuffled", "realistic") for s in ("basic", "moderate", "severe") for k in range(3)]
for i, j in enumerate(jobs):
    key = "_".join(map(str, j))
    if i % 2 != shard or key in done: continue
    if time.time() - t0 > budget: break
    subprocess.run(["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_symmetric.py"), *map(str, j), OUT], check=True)
    open(DONE, "a").write(key + "\n")
