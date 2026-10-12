"""Checkpointed queue: runs pending jobs, NPAR at a time, within a wall-clock budget (seconds)."""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); R = "./writing/rescore"; os.makedirs(R, exist_ok=True); budget = float(sys.argv[1]); which = sys.argv[2:]  # job groups
NPAR = int(os.environ.get("NPAR", "2"))
SEV = ("severe", "moderate", "basic"); MOD = ("shuffled", "realistic")
def jobs():
    J = []
    if "dev" in which:      J += [("dev", m, s, k, "B0,B2,B2nocrc" if k < 5 else "B0,B2", 400, "dev") for s in SEV for m in MOD for k in range(10)]
    if "heldout" in which:  J += [("heldout", m, s, k, "B0,B2", 400, "heldout") for s in SEV for m in MOD for k in range(10)]
    if "jump" in which:     J += [("dev", "shuffled", "severe", k, "B2", jb, f"jump{jb}") for jb in (100, 1600) for k in range(5)]
    if "napier" in which:   J += [("napier", m, s, k, "B0,B2", 400, "napier") for s in SEV for m in MOD for k in range(10)]
    if "pilot" in which:
        J += [("pilot25", "shuffled", s, k, "B0,B2", 400, "pilot25") for s in SEV for k in range(3)]
        J += [("pilot25", "shuffled", "severe", k, "B0,B2", 400, "power25") for k in range(10)]
        J += [("pilot20", "realistic", s, k, "B0,B2", 400, "smoke20") for s in SEV for k in range(3)]
    if "lo" in which:       J += [("lo", m, s, k, "B0dd,B2dd", 400, "lo") for s in SEV for m in MOD for k in range(10)]
    return J
def done(tag):
    f = f"{R}/out_{tag}.jsonl"
    if not os.path.exists(f): return set()
    return {(j["set"], j["model"], j["severity"], j["seed"]) for j in map(json.loads, filter(str.strip, open(f)))}
pend = [j for j in jobs() if (j[0], j[1], j[2], j[3]) not in done(j[6])]
print("pending", len(pend), flush=True)
est = {"severe": 45, "moderate": 35, "basic": 30}
t0 = time.time(); run = []
while pend or run:
    run = [p for p in run if p.poll() is None]
    while pend and len(run) < NPAR and time.time() - t0 + est[pend[0][2]] * (1.4 if pend[0][0] == "napier" else (0.6 if pend[0][0] == "lo" else 1)) < budget:
        S, m, s, k, modes, jb, tag = pend.pop(0)
        env = dict(os.environ, JUMP_BUDGET=str(jb))
        run.append(subprocess.Popen([sys.executable, f"{HERE}/rescore_run.py", S, m, s, str(k), modes, f"{R}/out_{tag}.jsonl"], env=env,
                                    stdout=open(f"{R}/log_{tag}.txt", "a"), stderr=subprocess.STDOUT))
    if not run: break
    time.sleep(2)
print("elapsed", round(time.time() - t0), "remaining", len(pend))
