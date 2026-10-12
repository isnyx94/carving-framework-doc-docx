"""Restoration experiment: structure-aware vs content-agnostic text recovery
from fragmented images, across three fragmentation severities.

This is the experiment the thesis title asks for and the repo did not have.
The reassembly track scored cluster agreement (ARI/V-measure) on blocks and
reported ARI 0.27 while the document it "restored" contained the literal text
"[PHASE 3 RECONSTRUCTED DOCUMENT]"; the classification track stops at deciding a
fragment's type. Neither produced recovered content. Here the dependent variable
is the document text itself.

Design
------
Conditions (per severity, paired on identical fragmentations):
  B0  contiguous  — signature scan + inflate forward. The content-agnostic
                    carver's ceiling.
  B2  structural  — B0, plus the container's declared sizes used to detect
                    truncation and bound a decompressor-guided search for
                    continuation blocks.

B2 is constructed to include B0's result as a floor, so any difference isolates
the contribution of the container metadata rather than reflecting two unrelated
algorithms.

Scoring is strictly per source document. Pooling every document's text into one
bag would let a token recovered from document A count as a hit against document
B's reference and inflate recall; attribution uses the fragmenter's block ground
truth, which the restorers never see.

Statistics follow the protocol already used elsewhere in the project: paired
Wilcoxon signed-rank plus a paired t-test over per-document recall, with the
effect size reported alongside p (THESIS_SPINE_v2 Sec 7.4).

Output: results/restoration/RESTORE_REPORT.md + restore_results.json
"""

import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from classify.restore.docx_restore import find_local_headers, restore_contiguous, restore_structural
from classify.restore.fragmenter import (
    SEVERITIES, build_image, build_image_realistic, corpus_paths, fragmentation_stats,
)
from classify.restore.text_recall import aggregate, reference_text, text_scores

# "shuffled" = runs scattered uniformly across the image (worst case).
# "realistic" = interleaved-write locality, mostly-contiguous files, bifragmentation
# dominant. Both must be reported: the shuffled model flatters a global search, and
# the realistic one is the sensitivity check that says whether the effect survives.
MODEL = os.environ.get("RESTORE_MODEL", "shuffled")
BUILDERS = {"shuffled": build_image, "realistic": build_image_realistic}

OUT_DIR = os.environ.get("RESTORE_OUT_DIR", "results/restoration")
CORPUS_ROOT = "docx"
N_FILES = int(os.environ.get("RESTORE_N_FILES", "25"))
# More seeds buy statistical power on the severe condition, where the effect is
# largest but per-fragmentation variance is high. Overridable so a power run
# needs no code edit: RESTORE_SEEDS=0,1,...,9 RESTORE_SEVERITIES=severe
SEEDS = tuple(int(s) for s in os.environ.get("RESTORE_SEEDS", "0,1,2").split(","))
BLOCK_SIZE = 512
# None = scan the whole image. A bounded window cannot work: the fragmenter
# shuffles runs globally, so a stream's continuation lands anywhere. Affordable
# because candidate testing reuses live decompressor state (docx_restore.py).
SEARCH_WINDOW = None
MODES = ("contiguous", "structural")


def restore_by_document(image_obj, paths, mode: str) -> dict:
    """Restore every text part and group the recovered text by the document the
    part's header actually came from.

    Attribution is evaluation-only bookkeeping: `mode`'s restorer receives just
    the raw image, so nothing about ground truth leaks into recovery.
    """
    data = bytes(image_obj.data)
    recovered = defaultdict(list)
    stats = defaultdict(lambda: {"parts": 0, "complete": 0, "crc_verified": 0,
                                 "blocks_searched": 0})

    for lfh in find_local_headers(data, image_obj.block_size):
        from classify.restore.docx_restore import _is_text_part
        if not _is_text_part(lfh.name) or lfh.compressed_size == 0:
            continue
        if not 0 <= lfh.block_index < len(image_obj.truth):
            continue
        source = image_obj.truth[lfh.block_index].source
        if source == "NOISE":
            continue  # a header signature inside noise; nothing to attribute it to

        if mode == "structural":
            part = restore_structural(data, lfh, image_obj.block_size,
                                      search_window=SEARCH_WINDOW)
        else:
            part = restore_contiguous(data, lfh)
        recovered[source].append(part.text)
        stats[source]["parts"] += 1
        stats[source]["complete"] += int(part.complete)
        stats[source]["crc_verified"] += int(part.crc_verified)
        stats[source]["blocks_searched"] += len(part.blocks_used)

    return {"text": {k: " ".join(v) for k, v in recovered.items()}, "stats": dict(stats)}


def run_severity(paths, severity: str, references: dict) -> dict:
    """One severity level across all seeds; returns per-condition aggregates and
    the paired per-document recall vectors used for significance testing."""
    per_doc = {mode: [] for mode in MODES}
    paired = {mode: [] for mode in MODES}  # aligned per (seed, document)

    part_stats = {mode: {"parts": 0, "complete": 0, "crc_verified": 0} for mode in MODES}
    frag_stats = []
    builder = BUILDERS[MODEL]
    for seed in SEEDS:
        image_obj = builder(paths, severity, seed=seed, block_size=BLOCK_SIZE)
        frag_stats.append(fragmentation_stats(image_obj))
        results = {mode: restore_by_document(image_obj, paths, mode) for mode in MODES}
        for mode in MODES:
            for st in results[mode]["stats"].values():
                for k in part_stats[mode]:
                    part_stats[mode][k] += st[k]
        for path in paths:
            ref = references[path]
            if not ref.strip():
                continue
            for mode in MODES:
                score = text_scores(results[mode]["text"].get(path, ""), ref)
                score["document"] = os.path.basename(path)
                score["seed"] = seed
                per_doc[mode].append(score)
                paired[mode].append(score["recall"])
        print(f"  {severity} seed {seed}: "
              + ", ".join(f"{m}={np.mean([s['recall'] for s in per_doc[m][-len(paths):]]):.3f}"
                          for m in MODES), flush=True)

    out = {"severity": severity, "config": SEVERITIES[severity],
           "conditions": {m: aggregate(per_doc[m]) for m in MODES},
           "part_stats": part_stats,
           # What the model actually produced, so the two models are compared on
           # measured fragmentation rather than on their parameter names.
           "fragmentation": {
               k: float(np.mean([f[k] for f in frag_stats]))
               for k in ("mean_runs_per_file", "frac_fragmented", "mean_gap_blocks",
                         "frac_successive_in_order")
           },
           "per_document": {m: per_doc[m] for m in MODES}}

    b0 = np.array(paired["contiguous"])
    b2 = np.array(paired["structural"])
    diffs = b2 - b0
    sig = {"effect_size_mean_recall": float(diffs.mean()),
           "n_paired": int(diffs.size),
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


def write_report(results: dict):
    L = ["# Restoration experiment — structure-aware vs content-agnostic text recovery", "",
         f"Corpus: {results['n_files']} real .docx files from `{CORPUS_ROOT}/`; "
         f"{len(SEEDS)} fragmentations per severity; {BLOCK_SIZE}B blocks.",
         f"Fragmentation model: **{results['model']}**.", "",
         "## Fragmentation actually produced", "",
         "| Severity | runs/file | fragmented | mean gap (blocks) | successive in order |",
         "|---|---|---|---|---|"]
    for severity in results["severities"]:
        f = results["by_severity"][severity]["fragmentation"]
        L.append(f"| {severity} | {f['mean_runs_per_file']:.2f} | "
                 f"{f['frac_fragmented']:.0%} | {f['mean_gap_blocks']:.0f} | "
                 f"{f['frac_successive_in_order']:.0%} |")
    L += ["",
         "Dependent variable is recovered **document text**, scored per source "
         "document against the intact original (token-multiset recall).", "",
         "## Text recall by fragmentation severity", "",
         "| Severity | Condition | macro recall | micro recall | precision | "
         "fully recovered | zero recovered |", "|---|---|---|---|---|---|---|"]
    for severity in results["severities"]:
        s = results["by_severity"][severity]
        for mode in MODES:
            a = s["conditions"][mode]
            label = "B0 content-agnostic" if mode == "contiguous" else "B2 structure-aware"
            L.append(f"| {severity} | {label} | {a['macro_recall']:.4f} | "
                     f"{a['micro_recall']:.4f} | {a['macro_precision']:.4f} | "
                     f"{a['n_fully_recovered']}/{a['n_documents']} | "
                     f"{a['n_zero_recovered']}/{a['n_documents']} |")

    L += ["", "## Part-level reassembly (CRC32-certified correctness)", "",
          "`crc verified` is the share of parts whose reassembly matches the CRC32 "
          "stored in the ZIP local file header — a *proof* of byte-exact recovery, "
          "not an estimate. `complete` only means the expected length was reached, "
          "which a wrong continuation block can also achieve.", "",
          "| Severity | Condition | parts | complete | crc verified |",
          "|---|---|---|---|---|"]
    for severity in results["severities"]:
        ps = results["by_severity"][severity]["part_stats"]
        for mode in MODES:
            s = ps[mode]
            label = "B0 content-agnostic" if mode == "contiguous" else "B2 structure-aware"
            n = s["parts"] or 1
            L.append(f"| {severity} | {label} | {s['parts']} | "
                     f"{s['complete']} ({s['complete'] / n:.1%}) | "
                     f"{s['crc_verified']} ({s['crc_verified'] / n:.1%}) |")

    L += ["", "## Paired significance (B2 − B0, per-document recall)", "",
          "| Severity | effect | better | worse | t-test p | Wilcoxon p | significant |",
          "|---|---|---|---|---|---|---|"]
    for severity in results["severities"]:
        sig = results["by_severity"][severity]["significance_recall"]
        L.append(f"| {severity} | {sig['effect_size_mean_recall']:+.4f} | "
                 f"{sig['n_better']} | {sig['n_worse']} | "
                 f"{sig.get('p_value', float('nan')):.4g} | "
                 f"{sig.get('wilcoxon_p', float('nan')):.4g} | "
                 f"{sig.get('significant', 'n/a')} |")

    L += ["", "## Notes", "",
          "* B2 contains B0's contiguous read as a floor, so it cannot score lower "
          "by construction; the paired difference isolates the value of the "
          "container's declared sizes (truncation detection + bounded search).",
          "* Precision below 1.0 reflects partially inflated XML whose tail is cut "
          "mid-token, not fabricated content.",
          "* Recall is bounded above by how much of each document's text lives in "
          "parts whose local file header survived in a locatable block; a header "
          "landing inside a noise block makes its part unrecoverable by either "
          "condition."]
    with open(os.path.join(OUT_DIR, "RESTORE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = corpus_paths(CORPUS_ROOT, limit=N_FILES)
    if not paths:
        raise FileNotFoundError(f"no .docx files under {CORPUS_ROOT}/")
    print(f"corpus: {len(paths)} documents", flush=True)

    print("extracting reference text from intact originals...", flush=True)
    references = {p: reference_text(p) for p in paths}
    n_tokens = sum(len(r.split()) for r in references.values())
    print(f"  reference corpus: ~{n_tokens:,} tokens", flush=True)

    severities = [s for s in os.environ.get("RESTORE_SEVERITIES", ",".join(SEVERITIES)).split(",")
                  if s in SEVERITIES]
    results = {"n_files": len(paths), "seeds": list(SEEDS), "block_size": BLOCK_SIZE,
               "search_window": SEARCH_WINDOW, "model": MODEL, "severities": severities,
               "by_severity": {}}
    for severity in severities:
        print(f"=== {severity} ===", flush=True)
        results["by_severity"][severity] = run_severity(paths, severity, references)

    with open(os.path.join(OUT_DIR, "restore_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_report(results)

    print("\n--- summary (macro text recall) ---")
    for severity in severities:
        s = results["by_severity"][severity]
        b0 = s["conditions"]["contiguous"]["macro_recall"]
        b2 = s["conditions"]["structural"]["macro_recall"]
        sig = s["significance_recall"]
        print(f"  {severity:9} B0={b0:.4f}  B2={b2:.4f}  effect={sig['effect_size_mean_recall']:+.4f}"
              f"  p={sig.get('p_value', float('nan')):.4g}")
    print(f"\nWrote {OUT_DIR}/RESTORE_REPORT.md")


if __name__ == "__main__":
    main()
