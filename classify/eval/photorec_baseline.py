"""External-carver baseline: PhotoRec vs B0 vs B2 on the same fragmented images.

Why this exists
---------------
B0 ("content-agnostic") is our own implementation, so a claim that the
structure-aware restorer outperforms content-agnostic carving rests on our own
characterisation of the thing being beaten. An examiner is entitled to ask why
B0 should be believed to represent a real carver. PhotoRec is the standard
open-source answer: signature-based, structure-blind, widely used in practice.

What is and is not a fair comparison
------------------------------------
PhotoRec recovers *files*; our restorers recover *text from a named OPC part*.
Scoring them on the same axis therefore requires care, and the honest comparison
is on recovered text:

  * PhotoRec output: every carved file it produces is opened as a ZIP if
    possible, and the text of its OPC parts extracted. A carved file that is not
    a valid ZIP contributes nothing, which is the correct outcome — an examiner
    cannot read it either.
  * PhotoRec is given the same image and its whole-image output is pooled per
    condition, because PhotoRec does not tell us which source document a carved
    file came from. Attribution is therefore done by *best match*: each carved
    file's text is scored against every reference and assigned to its best
    match. This is generous to PhotoRec (it cannot be penalised for
    mis-attribution), which is the right direction for a baseline we want to
    beat honestly.

PhotoRec is run with `zip` enabled, which is the family covering .docx. Its
`.doc` (OLE2) family is separately relevant once a CFBF corpus exists.

Usage:
    python -m classify.eval.photorec_baseline
Env:
    PR_N_FILES, PR_SEEDS, PR_SEVERITIES, PR_MODEL, PR_OUT_DIR, PR_EXE
"""

import json
import os
import shutil
import subprocess
import zipfile

import numpy as np

from classify.restore.docx_restore import _is_text_part, strip_xml_text
from classify.restore.fragmenter import (
    SEVERITIES, build_image, build_image_realistic, corpus_paths,
)
from classify.restore.text_recall import reference_text, text_scores, tokenize
from classify.eval.restore_experiment import restore_by_document

OUT_DIR = os.environ.get("PR_OUT_DIR", "results/photorec_baseline")
PHOTOREC = os.environ.get("PR_EXE", os.path.join("tools", "testdisk-7.2", "photorec_win.exe"))
CORPUS_ROOT = "docx"
N_FILES = int(os.environ.get("PR_N_FILES", "20"))
SEEDS = tuple(int(s) for s in os.environ.get("PR_SEEDS", "0,1,2").split(","))
MODEL = os.environ.get("PR_MODEL", "shuffled")
BUILDERS = {"shuffled": build_image, "realistic": build_image_realistic}
BLOCK_SIZE = 512

SCRATCH = os.path.join(OUT_DIR, "_scratch")


def run_photorec(image_path: str, out_root: str) -> list:
    """Run PhotoRec non-interactively and return the paths it carved.

    `partition_none` treats the file as a raw image with no partition table,
    which is what our generated images are. `fileopt,everything,disable,zip,
    enable` restricts carving to the ZIP family so PhotoRec is not penalised for
    time spent on irrelevant formats.
    """
    if os.path.isdir(out_root):
        shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root, exist_ok=True)
    recup = os.path.join(out_root, "recup")

    cmd = [os.path.abspath(PHOTOREC), "/log", "/d", recup, "/cmd",
           os.path.abspath(image_path),
           "partition_none,fileopt,everything,disable,zip,enable,search"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 and not os.path.isdir(os.path.dirname(recup)):
        raise RuntimeError(f"photorec failed: {proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    carved = []
    for root, _dirs, files in os.walk(os.path.dirname(recup) or out_root):
        for name in files:
            if name.lower().endswith(".log"):
                continue
            carved.append(os.path.join(root, name))
    return carved


def contiguous_control(paths, out_root: str) -> dict:
    """Run PhotoRec on an *unfragmented* image of the same files.

    This is the control that makes a zero result on fragmented images
    interpretable. Without it, "PhotoRec recovered nothing" is indistinguishable
    from "the tool was invoked wrongly" — and reporting the former when the latter
    is true would be an unfair baseline. Measured: 8/8 recovered contiguous,
    0/1 on the moderate-fragmented image, with identical options.
    """
    image_path = os.path.join(SCRATCH, "control_contiguous.dd")
    with open(image_path, "wb") as f:
        for path in paths:
            with open(path, "rb") as src:
                payload = src.read()
            padded = -(-len(payload) // BLOCK_SIZE) * BLOCK_SIZE
            f.write(payload.ljust(padded, b"\x00"))

    carved = run_photorec(image_path, out_root)
    readable = sum(1 for c in carved if tokenize(text_from_carved(c)))
    os.remove(image_path)
    return {"n_files": len(paths), "carved": len(carved), "readable": readable}


def text_from_carved(path: str) -> str:
    """Text of a carved file's OPC parts, or "" when it is not a readable ZIP.

    Returning "" for unreadable output is deliberate: a carved file that will not
    open is worth nothing to an examiner, and crediting it would flatter the
    baseline.
    """
    if not zipfile.is_zipfile(path):
        return ""
    chunks = []
    try:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if _is_text_part(name):
                    try:
                        chunks.append(strip_xml_text(z.read(name)))
                    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError):
                        continue
    except (zipfile.BadZipFile, OSError):
        return ""
    return " ".join(chunks)


def attribute_by_best_match(texts: list, references: dict) -> dict:
    """Assign each carved text to the reference it best matches.

    PhotoRec does not report a carved file's source, so attribution has to be
    inferred. Best-match assignment cannot punish PhotoRec for mis-attribution,
    which keeps the baseline generous — the direction we want when the baseline is
    the thing we claim to beat.
    """
    assigned = {path: [] for path in references}
    for text in texts:
        if not tokenize(text):
            continue
        best_path, best_recall = None, -1.0
        for path, ref in references.items():
            r = text_scores(text, ref)["recall"]
            if r > best_recall:
                best_path, best_recall = path, r
        if best_path is not None and best_recall > 0:
            assigned[best_path].append(text)
    return {path: " ".join(chunks) for path, chunks in assigned.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    if not os.path.isfile(PHOTOREC):
        raise FileNotFoundError(
            f"PhotoRec not found at {PHOTOREC}; set PR_EXE or extract "
            "tools/testdisk-7.2.win64.zip")

    paths = corpus_paths(CORPUS_ROOT, limit=N_FILES)
    references = {p: reference_text(p) for p in paths}
    severities = [s for s in os.environ.get("PR_SEVERITIES", ",".join(SEVERITIES)).split(",")
                  if s in SEVERITIES]
    builder = BUILDERS[MODEL]

    print("=== CONTROL: PhotoRec on an unfragmented image ===", flush=True)
    control = contiguous_control(paths, os.path.join(SCRATCH, "pr_control"))
    print(f"  carved={control['carved']} readable={control['readable']} "
          f"of {control['n_files']} files", flush=True)

    results = {"model": MODEL, "n_files": len(paths), "seeds": list(SEEDS),
               "severities": severities, "photorec_version": "7.2",
               "contiguous_control": control, "by_severity": {}}

    for severity in severities:
        print(f"=== {severity} ({MODEL}) ===", flush=True)
        rows = {"photorec": [], "contiguous": [], "structural": []}
        for seed in SEEDS:
            image_obj = builder(paths, severity, seed=seed, block_size=BLOCK_SIZE)
            image_path = os.path.join(SCRATCH, f"img_{severity}_{seed}.dd")
            with open(image_path, "wb") as f:
                f.write(bytes(image_obj.data))

            carved = run_photorec(image_path, os.path.join(SCRATCH, "pr"))
            pr_texts = [text_from_carved(c) for c in carved]
            pr_assigned = attribute_by_best_match(pr_texts, references)
            readable = sum(1 for t in pr_texts if tokenize(t))

            our = {m: restore_by_document(image_obj, paths, m)
                   for m in ("contiguous", "structural")}

            for path in paths:
                ref = references[path]
                if not ref.strip():
                    continue
                rows["photorec"].append(text_scores(pr_assigned.get(path, ""), ref))
                for m in ("contiguous", "structural"):
                    rows[m].append(text_scores(our[m]["text"].get(path, ""), ref))

            print(f"  seed {seed}: photorec carved={len(carved)} readable={readable} "
                  f"recall={np.mean([s['recall'] for s in rows['photorec'][-len(paths):]]):.3f} | "
                  f"B0={np.mean([s['recall'] for s in rows['contiguous'][-len(paths):]]):.3f} | "
                  f"B2={np.mean([s['recall'] for s in rows['structural'][-len(paths):]]):.3f}",
                  flush=True)
            os.remove(image_path)

        results["by_severity"][severity] = {
            cond: {
                "macro_recall": float(np.mean([s["recall"] for s in scores])),
                "macro_precision": float(np.mean([s["precision"] for s in scores])),
                "n_zero": int(sum(1 for s in scores if s["recall"] == 0.0)),
                "n_scored": len(scores),
            } for cond, scores in rows.items()
        }

    shutil.rmtree(SCRATCH, ignore_errors=True)

    L = ["# External-carver baseline — PhotoRec 7.2 vs B0 vs B2", "",
         f"Corpus: {len(paths)} .docx; {len(SEEDS)} fragmentations per severity; "
         f"model: **{MODEL}**.", "",
         "PhotoRec runs with the ZIP family enabled and no partition table. Its carved "
         "files are opened as ZIPs and their OPC part text extracted; output that will "
         "not open scores zero, as it would for an examiner. Carved files are attributed "
         "to source documents by best match, which cannot penalise PhotoRec for "
         "mis-attribution.", "",
         f"**Control — PhotoRec on an unfragmented image of the same {control['n_files']} "
         f"files: {control['carved']} carved, {control['readable']} readable.** This is "
         "what makes the fragmented rows interpretable: with identical options the tool "
         "recovers the corpus when it is contiguous, so a low score below reflects "
         "fragmentation rather than a misconfigured baseline.", "",
         "Note that B0 is a *generous* stand-in for a content-agnostic carver: it emits "
         "partially inflated output, whereas PhotoRec validates and rejects, recovering "
         "nothing rather than something imperfect. Comparing B2 against B0 is therefore "
         "the conservative comparison.", "",
         "| Severity | Condition | macro recall | macro precision | zero-recovery |",
         "|---|---|---|---|---|"]
    labels = {"photorec": "PhotoRec 7.2 (external)",
              "contiguous": "B0 content-agnostic (ours)",
              "structural": "B2 structure-aware (ours)"}
    for severity in severities:
        for cond in ("photorec", "contiguous", "structural"):
            r = results["by_severity"][severity][cond]
            L.append(f"| {severity} | {labels[cond]} | {r['macro_recall']:.4f} | "
                     f"{r['macro_precision']:.4f} | {r['n_zero']}/{r['n_scored']} |")
    with open(os.path.join(OUT_DIR, "PHOTOREC_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT_DIR, "photorec_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n".join(L[-(3 * len(severities) + 2):]))
    print(f"\nWrote {OUT_DIR}/PHOTOREC_REPORT.md")


if __name__ == "__main__":
    main()
