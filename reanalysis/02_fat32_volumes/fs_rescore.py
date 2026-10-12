"""Re-score one constructed FAT32 OOXML volume with the corrected extractor: rebuild the volume with
fs_build.py (deterministic in its seed), check it against the stored ground truth, then run fs_run.py
(restorers and tools) under strict_patch. Usage: python fs_rescore.py <key> <out.jsonl>"""
import sys, os, json, runpy, subprocess, shutil
D7 = "./writing/reanalysis_2026-10-07"
sys.path.insert(0, "."); sys.path.insert(0, __import__("os").path.abspath("./reanalysis/common"))
key, out = sys.argv[1], sys.argv[2]
tr, sc, fz, s = key.split("_")
wd = f"/tmp/fsr/{key}"; shutil.rmtree(wd, ignore_errors=True)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_build.py"), tr, sc, s, wd, fz], check=True, stdout=subprocess.DEVNULL)
new = json.load(open(wd + "/truth.json")); old = json.load(open(f"{D7}/fs_truth/{key}.json"))
same = new["owner"] == old["owner"] and new["targets"] == old["targets"]
os.chdir(".")
import strict_patch; strict_patch.apply()
sys.argv = ["fs_run.py", wd, out, "rest,tools"]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fs_run.py"), run_name="__main__")
open("./writing/rescore/fs_rescore_done.txt", "a").write(f"{key} identical_volume={same}\n")
shutil.rmtree(wd, ignore_errors=True)
