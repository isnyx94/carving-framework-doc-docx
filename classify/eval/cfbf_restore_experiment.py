"""Restoration experiment for CFBF / .doc — the cross-format half of RQ2.

Mirrors classify/eval/restore_experiment.py exactly (same fragmentation models,
same severities, same per-document scoring, same paired statistics) so the two
container technologies are directly comparable. Only the restorers differ, which
is the point: one framework, two structurally unrelated compound formats.

Corpus: genuine OLE2 .doc files from govdocs1 (`data/govdocs1/doc_real/`). The
repo's own doc/ directory contains no CFBF files at all — see
classify/tests/test_corpus_integrity.py — so this experiment is only possible with
the external corpus.

An intact-container control is reported alongside the fragmented conditions,
because for CFBF it is informative in its own right: a stream's sectors form a FAT
linked list and can be out of order *inside an undamaged file*, so B0's linear
read is already wrong at zero disk fragmentation. On DOCX the equivalent control
is a tie (B0 reaches 0.997), which is precisely the contrast RQ2 asks about.

Output: results/restoration_cfbf/CFBF_RESTORE_REPORT.md + cfbf_restore_results.json
"""

import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from classify.restore import cfbf_restore as cr
from classify.restore.fragmenter import (
    SEVERITIES, build_image, build_image_realistic, corpus_paths,
)
from classify.restore.text_recall import aggregate, text_scores

OUT_DIR = os.environ.get("CFBF_OUT_DIR", "results/restoration_cfbf")
CORPUS_ROOT = os.environ.get("CFBF_CORPUS", os.path.join("data", "govdocs1", "doc_real"))
N_FILES = int(os.environ.get("CFBF_N_FILES", "40"))
SEEDS = tuple(int(s) for s in os.environ.get("CFBF_SEEDS", "0,1,2,3,4").split(","))
MODEL = os.environ.get("CFBF_MODEL", "shuffled")
BLOCK_SIZE = 512
BUILDERS = {"shuffled": build_image, "realistic": build_image_realistic}
MODES = ("contiguous", "structural")


def restore_by_document(image_obj, mode: str) -> dict:
    """Restore every CFBF container found, grouped by the document its header
    actually came from. Attribution is evaluation-only bookkeeping; the restorers
    see nothing but the raw image."""
    data = bytes(image_obj.data)
    recovered = defaultdict(list)
    stats = defaultdict(lambda: {"containers": 0, "structure_parsed": 0, "sectors": 0})

    for header in cr.find_cfbf_headers(data, image_obj.block_size):
        idx = header["block_index"]
        if not 0 <= idx < len(image_obj.truth):
            continue
        source = image_obj.truth[idx].source
        if source == "NOISE":
            continue
        # Only a header sitting at its true file start is a real container start;
        # a signature elsewhere is a coincidence inside other data.
        if image_obj.truth[idx].file_offset != 0:
            continue

        if mode == "structural":
            out = cr.restore_structural_doc(data, header)
        else:
            out = cr.restore_contiguous_doc(data, header)
        recovered[source].append(out.text)
        stats[source]["containers"] += 1
        stats[source]["structure_parsed"] += int(getattr(out, "structure_parsed", False))
        stats[source]["sectors"] += getattr(out, "sectors_used", 0)

    return {"text": {k: " ".join(v) for k, v in recovered.items()}, "stats": dict(stats)}


def intact_control(paths, references) -> dict:
    """Each file alone, unfragmented. For CFBF this is a real result, not a
    formality: FAT-ordered sectors mean a linear read can be wrong even here."""
    rows = {m: [] for m in MODES}
    for path in paths:
        with open(path, "rb") as f:
            image = f.read()
        headers = cr.find_cfbf_headers(image, BLOCK_SIZE)
        if not headers:
            continue
        ref = references[path]
        if not ref.strip():
            continue
        b0 = cr.restore_contiguous_doc(image, headers[0])
        b2 = cr.restore_structural_doc(image, headers[0])
        rows["contiguous"].append(text_scores(b0.text, ref))
        rows["structural"].append(text_scores(b2.text, ref))
    return {m: aggregate(rows[m]) for m in MODES}


def run_severity(paths, references, severity: str) -> dict:
    per_doc = {m: [] for m in MODES}
    paired = {m: [] for m in MODES}
    part_stats = {m: {"containers": 0, "structure_parsed": 0} for m in MODES}
    builder = BUILDERS[MODEL]

    for seed in SEEDS:
        image_obj = builder(paths, severity, seed=seed, block_size=BLOCK_SIZE)
        results = {m: restore_by_document(image_obj, m) for m in MODES}
        for m in MODES:
            for st in results[m]["stats"].values():
                part_stats[m]["containers"] += st["containers"]
                part_stats[m]["structure_parsed"] += st["structure_parsed"]
        for path in paths:
            ref = references[path]
            if not ref.strip():
                continue
            for m in MODES:
                score = text_scores(results[m]["text"].get(path, ""), ref)
                score["document"] = os.path.basename(path)
                per_doc[m].append(score)
                paired[m].append(score["recall"])
        print(f"  {severity} seed {seed}: "
              + " | ".join(f"{m}={np.mean([s['recall'] for s in per_doc[m][-len(paths):]]):.3f}"
                           for m in MODES), flush=True)

    out = {"severity": severity,
           "conditions": {m: aggregate(per_doc[m]) for m in MODES},
           "part_stats": part_stats}
    b0 = np.array(paired["contiguous"])
    b2 = np.array(paired["structural"])
    diffs = b2 - b0
    sig = {"effect_size_mean_recall": float(diffs.mean()), "n_paired": int(diffs.size),
           "n_better": int((diffs > 0).sum()), "n_worse": int((diffs < 0).sum())}
    if diffs.size and not np.allclose(diffs, 0):
        t, p = ttest_rel(b2, b0)
        sig.update({"t": float(t), "p_value": float(p), "significant": bool(p < 0.05)})
        try:
            _, wp = wilcoxon(b2, b0)
            sig["wilcoxon_p"] = float(wp)
        except ValueError:
            pass
    out["significance_recall"] = sig
    return out


def write_report(results):
    L = ["# CFBF (.doc) restoration — structure-aware vs content-agnostic", "",
         f"Corpus: {results['n_files']} genuine OLE2 .doc files from govdocs1; "
         f"{len(SEEDS)} fragmentations per severity; model **{results['model']}**.", "",
         "The repo's own `doc/` directory contains no CFBF files (all ZIP or random "
         "data), so this experiment depends on the external govdocs1 corpus.", "",
         "## Intact-container control (no disk fragmentation)", "",
         "| Condition | macro recall | macro precision |", "|---|---|---|"]
    for m in MODES:
        c = results["intact_control"][m]
        label = "B0 content-agnostic" if m == "contiguous" else "B2 structure-aware"
        L.append(f"| {label} | {c['macro_recall']:.4f} | {c['macro_precision']:.4f} |")
    L += ["", "This control is a *result* for CFBF, not a formality. A stream's sectors "
          "form a FAT linked list and may be stored out of order inside an undamaged "
          "file, and a linear read cannot know where the stream ends — so it also "
          "collects other streams' text. On DOCX the equivalent control is a tie.", "",
          "## Text recall by fragmentation severity", "",
          "| Severity | Condition | macro recall | macro precision | zero-recovery |",
          "|---|---|---|---|---|"]
    for severity in results["severities"]:
        for m in MODES:
            a = results["by_severity"][severity]["conditions"][m]
            label = "B0 content-agnostic" if m == "contiguous" else "B2 structure-aware"
            L.append(f"| {severity} | {label} | {a['macro_recall']:.4f} | "
                     f"{a['macro_precision']:.4f} | "
                     f"{a['n_zero_recovered']}/{a['n_documents']} |")
    L += ["", "## Paired significance (B2 − B0, per-document recall)", "",
          "| Severity | effect | better | worse | t-test p | Wilcoxon p | significant |",
          "|---|---|---|---|---|---|---|"]
    for severity in results["severities"]:
        s = results["by_severity"][severity]["significance_recall"]
        L.append(f"| {severity} | {s['effect_size_mean_recall']:+.4f} | {s['n_better']} | "
                 f"{s['n_worse']} | {s.get('p_value', float('nan')):.4g} | "
                 f"{s.get('wilcoxon_p', float('nan')):.4g} | "
                 f"{s.get('significant', 'n/a')} |")
    L += ["", "## Note on text extraction", "",
          "Text is extracted by the same heuristic printable-run extractor for the "
          "reference and for both conditions. It is not a full [MS-DOC] piece-table "
          "implementation and does not reproduce document order exactly. Applying it "
          "identically on both sides keeps the comparison a test of *sector recovery*, "
          "which is the research question, rather than of a Word text parser."]
    with open(os.path.join(OUT_DIR, "CFBF_RESTORE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = corpus_paths(CORPUS_ROOT, limit=N_FILES, pattern=".doc")
    if not paths:
        raise FileNotFoundError(
            f"no .doc files under {CORPUS_ROOT}; extract govdocs1 first")
    print(f"corpus: {len(paths)} genuine CFBF .doc files", flush=True)

    references = {p: cr.reference_doc_text(p) for p in paths}
    print(f"  reference corpus: ~{sum(len(r.split()) for r in references.values()):,} tokens",
          flush=True)

    print("=== intact-container control ===", flush=True)
    control = intact_control(paths, references)
    for m in MODES:
        print(f"  {m}: recall={control[m]['macro_recall']:.4f} "
              f"precision={control[m]['macro_precision']:.4f}", flush=True)

    severities = [s for s in os.environ.get("CFBF_SEVERITIES", ",".join(SEVERITIES)).split(",")
                  if s in SEVERITIES]
    results = {"n_files": len(paths), "model": MODEL, "seeds": list(SEEDS),
               "severities": severities, "intact_control": control, "by_severity": {}}
    for severity in severities:
        print(f"=== {severity} ===", flush=True)
        results["by_severity"][severity] = run_severity(paths, references, severity)

    with open(os.path.join(OUT_DIR, "cfbf_restore_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_report(results)

    print("\n--- summary (macro text recall) ---")
    for severity in severities:
        s = results["by_severity"][severity]
        sig = s["significance_recall"]
        print(f"  {severity:9} B0={s['conditions']['contiguous']['macro_recall']:.4f}  "
              f"B2={s['conditions']['structural']['macro_recall']:.4f}  "
              f"effect={sig['effect_size_mean_recall']:+.4f}  "
              f"p={sig.get('p_value', float('nan')):.4g}")
    print(f"\nWrote {OUT_DIR}/CFBF_RESTORE_REPORT.md")


if __name__ == "__main__":
    main()
