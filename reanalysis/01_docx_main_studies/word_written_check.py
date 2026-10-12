"""Check on Word-written OOXML: the two Microsoft Word documents available (the author's own
proposal and progress report) are inspected for data descriptors and restored under both models."""
import sys, os, json, numpy as np
sys.path.insert(0, "."); 
import classify.restore.docx_restore as dr
from classify.restore.fragmenter import build_image, build_image_realistic
from classify.restore.text_recall import reference_text, text_scores
from classify.eval.restore_experiment import restore_by_document
U = "./data/napierone/"
paths = [U + "LU4_Updated_PCM21002_Nor Ika Shahirah_2026.docx", U + "PCM21002_Nor Ika Shahirah_apd.docx"]
info = []
for p in paths:
    d = open(p, "rb").read(); hs = dr.find_local_headers(d, 512)
    info.append({"parts": len(hs), "bit3": sum(1 for h in hs if h.flags & 8), "size": len(d),
                 "text_parts": [h.name for h in hs if dr._is_text_part(h.name)]})
refs = {p: reference_text(p) for p in paths}
out = {"files": info, "cells": []}
for model, b in (("shuffled", build_image), ("realistic", build_image_realistic)):
    for sev in ("basic", "moderate", "severe"):
        acc = {"B0": [], "B2": []}
        for seed in range(10):
            img = b(paths, sev, seed=seed, block_size=512)
            for m, mode in (("B0", "contiguous"), ("B2", "structural")):
                t = restore_by_document(img, paths, mode)["text"]
                acc[m] += [text_scores(t.get(p, ""), refs[p])["recall"] for p in paths]
        out["cells"].append({"model": model, "severity": sev, "B0": float(np.mean(acc["B0"])), "B2": float(np.mean(acc["B2"]))})
        print(model, sev, round(np.mean(acc["B0"]), 3), round(np.mean(acc["B2"]), 3), flush=True)
json.dump(out, open("writing/reanalysis_2026-10-08/word_written_check.json", "w"), indent=1)
print(info)
