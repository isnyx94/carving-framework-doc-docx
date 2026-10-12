import sys, os, json
sys.path.insert(0, "."); os.chdir(".")
from classify.restore.fragmenter import build_image, build_image_realistic, corpus_paths
from classify.restore.text_recall import reference_text, text_scores
from classify.eval.restore_experiment import restore_by_document
model, sev = sys.argv[1], sys.argv[2]
paths = corpus_paths("docx", limit=20); refs = {p: reference_text(p) for p in paths}; refs = {p: r for p, r in refs.items() if r.strip()}
b = {"shuffled": build_image, "realistic": build_image_realistic}[model]
for seed in range(3):
    img = b(paths, sev, seed=seed, block_size=512)
    out = {m: restore_by_document(img, paths, m)["text"] for m in ("contiguous", "structural")}
    for m in out:
        sc = [text_scores(out[m].get(p, ""), r) for p, r in refs.items()]
        open("./work/ext/b0b2_20.jsonl", "a").write(json.dumps({"model": model, "severity": sev, "seed": seed, "cond": m,
            "recall": sum(s["recall"] for s in sc) / len(sc), "precision": sum(s["precision"] for s in sc) / len(sc)}) + "\n")
print(model, sev, "done", flush=True)
