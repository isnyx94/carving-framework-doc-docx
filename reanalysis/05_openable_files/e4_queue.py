import sys, os, time, subprocess
D = "./writing/reanalysis_2026-10-07"
OUT, DONE = D + "/e4.jsonl", D + "/e4_done.txt"
budget, shard = float(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
jobs = [(m, s, k) for k in range(5) for m in ("shuffled", "realistic") for s in ("basic", "moderate", "severe")]
pend = [j for j in jobs if "_".join(map(str, j)) not in done]
for i, j in enumerate(pend):
    if i % 2 != shard: continue
    if time.time() - t0 > budget: break
    subprocess.run(["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "e4_openable.py"), *map(str, j), OUT], check=True)
    open(DONE, "a").write("_".join(map(str, j)) + "\n")
