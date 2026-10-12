"""Export every result JSON in the project to thesis-ready CSV tables.

Written to be re-run: runs still in flight (Chapter 5 compound, the 4096B CFBF
ablation) are simply skipped with a note, so running this again after they finish
picks them up without any edit. Nothing is computed here — this only reshapes
what the experiment scripts already wrote, so a CSV can never disagree with its
source JSON.

Outputs to results/csv/:
    00_all_results_long.csv     every headline number in tidy long form
    01..13_*.csv                one table per thesis table
    INDEX.md                    what each file is, and which JSON it came from

Long format (00_) is the one to write from: filter it in Excel by `table` or
`metric` rather than hunting through the wide files.
"""

import csv
import json
import os

RESULTS = "results"
OUT_DIR = os.path.join(RESULTS, "csv")

# Where each table's numbers come from, recorded in INDEX.md for traceability.
SOURCES = {}
LONG_ROWS = []
MISSING = []


def _load(path):
    full = os.path.join(RESULTS, path) if not path.startswith(RESULTS) else path
    if not os.path.isfile(full):
        MISSING.append(path)
        return None
    with open(full) as f:
        return json.load(f)


def _write(name, header, rows, source, note=""):
    if not rows:
        return
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    SOURCES[name] = (source, note, len(rows))
    print(f"  {name:38} {len(rows):5} rows")


def _long(table, experiment, condition, metric, value, unit="", note=""):
    LONG_ROWS.append([table, experiment, condition, metric, value, unit, note])


def fmt_p(p) -> str:
    """p-value as it should appear in prose. Raw floats like
    2.144405679657344e-06 are error-prone to transcribe by hand, so a
    ready-to-paste string travels alongside the full-precision value."""
    if p == "" or p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return str(p)
    if p < 1e-4:
        exponent = 0
        mantissa = p
        while mantissa < 1:
            mantissa *= 10
            exponent += 1
        return f"p < 0.0001 ({mantissa:.1f}e-{exponent})"
    if p < 0.001:
        return f"p = {p:.5f}"
    return f"p = {p:.3f}"


CONDITION_LABEL = {"contiguous": "B0 content-agnostic", "structural": "B2 structure-aware",
                   "photorec": "PhotoRec 7.2 (external)"}


# --------------------------------------------------------------------------
# restoration
# --------------------------------------------------------------------------

def restoration_tables():
    """Both formats, both fragmentation models, in one comparable shape."""
    specs = [
        ("docx", "shuffled", "restoration_scaled/restore_results.json"),
        ("docx", "realistic", "restoration_realistic/restore_results.json"),
        ("docx", "shuffled (severe power run)", "restoration_power/restore_results.json"),
        ("doc (CFBF)", "shuffled", "restoration_cfbf/cfbf_restore_results.json"),
        ("doc (CFBF)", "realistic", "restoration_cfbf_realistic/cfbf_restore_results.json"),
    ]
    main, sig, part, control = [], [], [], []

    for track, model, path in specs:
        data = _load(path)
        if data is None:
            continue
        src = path

        if "intact_control" in data:
            for cond, c in data["intact_control"].items():
                control.append([track, model, "intact (no fragmentation)",
                                CONDITION_LABEL.get(cond, cond),
                                round(c["macro_recall"], 4), round(c["macro_precision"], 4),
                                round(c["macro_f1"], 4), c["n_documents"]])
                _long("restoration_control", f"{track}/{model}", CONDITION_LABEL.get(cond, cond),
                      "macro_recall", round(c["macro_recall"], 4),
                      note="intact container, no disk fragmentation")
                _long("restoration_control", f"{track}/{model}", CONDITION_LABEL.get(cond, cond),
                      "macro_precision", round(c["macro_precision"], 4),
                      note="intact container, no disk fragmentation")

        for severity in data.get("severities", []):
            block = data["by_severity"].get(severity)
            if not block:
                continue
            for cond, a in block["conditions"].items():
                label = CONDITION_LABEL.get(cond, cond)
                main.append([track, model, severity, label,
                             round(a["macro_recall"], 4), round(a["micro_recall"], 4),
                             round(a["macro_precision"], 4), round(a["macro_f1"], 4),
                             a["n_fully_recovered"], a["n_zero_recovered"], a["n_documents"]])
                for metric in ("macro_recall", "micro_recall", "macro_precision"):
                    _long("restoration_main", f"{track}/{model}/{severity}", label,
                          metric, round(a[metric], 4))
                _long("restoration_main", f"{track}/{model}/{severity}", label,
                      "zero_recovery_documents",
                      f"{a['n_zero_recovered']}/{a['n_documents']}")

            s = block.get("significance_recall", {})
            if s:
                sig.append([track, model, severity,
                            round(s.get("effect_size_mean_recall", 0), 4),
                            s.get("n_better", ""), s.get("n_worse", ""),
                            s.get("n_paired", ""),
                            s.get("p_value", ""), fmt_p(s.get("p_value", "")),
                            s.get("wilcoxon_p", ""), fmt_p(s.get("wilcoxon_p", "")),
                            s.get("significant", "")])
                _long("restoration_significance", f"{track}/{model}/{severity}",
                      "B2 - B0", "effect_size_recall",
                      round(s.get("effect_size_mean_recall", 0), 4))
                if "p_value" in s:
                    _long("restoration_significance", f"{track}/{model}/{severity}",
                          "B2 - B0", "paired_t_p", s["p_value"])
                    _long("restoration_significance", f"{track}/{model}/{severity}",
                          "B2 - B0", "wilcoxon_p", s.get("wilcoxon_p", ""))

            for cond, ps in (block.get("part_stats") or {}).items():
                label = CONDITION_LABEL.get(cond, cond)
                if "parts" in ps:
                    part.append([track, model, severity, label, ps["parts"],
                                 ps.get("complete", ""), ps.get("crc_verified", "")])
                    _long("restoration_part_level", f"{track}/{model}/{severity}", label,
                          "crc_verified_parts",
                          f"{ps.get('crc_verified', '')}/{ps['parts']}",
                          note="CRC32 match = proof of byte-exact reassembly")
                elif "containers" in ps:
                    part.append([track, model, severity, label, ps["containers"],
                                 ps.get("structure_parsed", ""), "n/a (CFBF has no CRC)"])

        SOURCES.setdefault("_restoration", (src, "", 0))

    _write("01_restoration_main.csv",
           ["track", "fragmentation_model", "severity", "condition", "macro_recall",
            "micro_recall", "macro_precision", "macro_f1", "n_fully_recovered",
            "n_zero_recovery", "n_documents"],
           main, "results/restoration_*/",
           "Primary restoration table. micro_recall flat vs macro = breadth not volume.")
    _write("02_restoration_significance.csv",
           ["track", "fragmentation_model", "severity", "effect_size_recall", "n_better",
            "n_worse", "n_paired", "paired_t_p", "paired_t_p_for_prose", "wilcoxon_p",
            "wilcoxon_p_for_prose", "significant"],
           sig, "results/restoration_*/",
           "Paired per-document tests, B2 - B0. *_for_prose columns are "
           "ready-to-paste; the raw floats are kept for full precision.")
    _write("03_restoration_part_level.csv",
           ["track", "fragmentation_model", "severity", "condition",
            "parts_or_containers", "complete_or_structure_parsed", "crc_verified"],
           part, "results/restoration_*/",
           "crc_verified is a proof; 'complete' only means expected length reached. "
           "CFBF has no checksum, so no equivalent exists there.")
    _write("04_restoration_intact_control.csv",
           ["track", "fragmentation_model", "condition_set", "condition", "macro_recall",
            "macro_precision", "macro_f1", "n_documents"],
           control, "results/restoration_cfbf/",
           "For CFBF this is a result (FAT-ordered sectors); for DOCX it is a tie.")


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classification_tables():
    main, per_class, sig, gates, calib, deletion = [], [], [], [], [], []

    def add_model(experiment, size, name, m, seeds="1", status="current"):
        main.append([experiment, size, name, seeds, status,
                     round(m["macro_f1"], 4), "",
                     round(m["per_class"]["docx"]["f1"], 4) if "docx" in m["per_class"] else "",
                     round(m.get("docx_zip_confusion_rate") or 0, 4),
                     round(m.get("micro_accuracy", 0), 4)])
        _long("classification_main", f"{experiment}/{size}B", name, "macro_f1",
              round(m["macro_f1"], 4))
        for cls, v in m["per_class"].items():
            per_class.append([experiment, size, name, cls, round(v["precision"], 4),
                              round(v["recall"], 4), round(v["f1"], 4), v.get("support", "")])

    # 512B full-spec (CarveFormer-spec backbone, 5 seeds).
    # The 2026-07-06 run was archived pending regeneration, because its `+C1` arm used a
    # refiner trained on 61.3% wrong labels. Its classical baselines and `backbone_only`
    # arm use no refiner and remain valid, so the archive is read as a fallback and rows
    # are tagged by status rather than discarded — withdrawing one arm should not delete
    # the results that sit beside it.
    fs = _load("new_result/fullspec_512B.json")
    fs_status = "current"
    if fs is None:
        fs = _load("new_result/superseded_confounded_refiner/fullspec_512B.json")
        fs_status = "SUPERSEDED-pending-rerun"
    if fs:
        for name in ("random_forest", "xgboost"):
            if name in fs.get("stage_a", {}):
                # classical baselines use no refiner -> unaffected by the defect
                add_model("fullspec_512B", 512, name, fs["stage_a"][name],
                          status="valid (no refiner)")
        for name, cfg in fs.get("stage_b", {}).get("configs", {}).items():
            row_status = ("valid (no refiner)" if name == "backbone_only"
                          else fs_status)
            main.append(["fullspec_512B", 512, name,
                         len(fs["stage_b"].get("seeds", [])), row_status,
                         round(cfg["macro_f1_mean"], 4), round(cfg["macro_f1_std"], 4),
                         round(cfg.get("docx_f1_mean", 0), 4),
                         round(cfg.get("docx_zip_confusion_mean", 0), 4), ""])
            _long("classification_main", "fullspec_512B/512B", name, "macro_f1",
                  f"{cfg['macro_f1_mean']:.4f}+/-{cfg['macro_f1_std']:.4f}")
        s = fs.get("stage_b", {}).get("significance", {})
        if s:
            sig.append(["fullspec_512B", 512, "+C1 vs backbone", "macro_f1",
                        round(s.get("effect_size", 0), 4), s.get("p_value", ""),
                        s.get("significant", "")])
        for name, c in fs.get("stage_c", {}).items():
            calib.append(["fullspec_512B", 512, name, round(c["ece_before"], 4),
                          round(c["ece_after"], 4), round(c.get("temperature", 0), 4),
                          round(c.get("c4_auroc", 0), 4), "mixed OOD (see Ch4 4.7 caveat)"])
            _long("calibration", f"fullspec_512B/512B", name, "ECE_before",
                  round(c["ece_before"], 4))
            _long("calibration", f"fullspec_512B/512B", name, "ECE_after",
                  round(c["ece_after"], 4))
            _long("open_set", f"fullspec_512B/512B", name, "MSP_AUROC",
                  round(c.get("c4_auroc", 0), 4), note="OOD set mismatched — Ch4 4.7")
            dc = c.get("deletion_curve")
            if dc:
                for frac, ig, rnd in zip(dc["fractions"], dc["ig"], dc["random"]):
                    deletion.append(["fullspec_512B", 512, name, frac,
                                     round(ig, 4), round(rnd, 4)])

    # 4096B classical baselines
    cl = _load("ch5/classical_4096B.json")
    if cl:
        for name, m in cl.items():
            add_model("ch5_4096B", 4096, name, m, status="current")

    # 4096B transformer arms. The completed run carries aggregates and significance;
    # `compound_partial.json` is the in-flight checkpoint and is only read as a fallback
    # so the table still populates while a run is going.
    comp = _load("ch5/compound_4096B.json")
    if comp:
        n_seeds = len(comp.get("seeds", []))
        for name, cfg in comp["configs"].items():
            main.append(["ch5_4096B", 4096, name, n_seeds, "current",
                         round(cfg["macro_f1_mean"], 4), round(cfg["macro_f1_std"], 4),
                         round(cfg.get("docx_f1_mean", 0), 4),
                         round(cfg.get("docx_zip_confusion_mean", 0), 4), ""])
            _long("classification_main", "ch5_4096B/4096B", name, "macro_f1",
                  f"{cfg['macro_f1_mean']:.4f}+/-{cfg['macro_f1_std']:.4f}")
            for r in cfg["runs"]:
                main.append(["ch5_4096B (per-seed)", 4096, f"{name} seed{r['seed']}", 1,
                             "current", round(r["macro_f1"], 4), "",
                             round(r["per_class"]["docx"]["f1"], 4),
                             round(r.get("docx_zip_confusion_rate") or 0, 4),
                             round(r.get("micro_accuracy", 0), 4)])
                for cls, v in r["per_class"].items():
                    per_class.append(["ch5_4096B", 4096, f"{name} seed{r['seed']}", cls,
                                      round(v["precision"], 4), round(v["recall"], 4),
                                      round(v["f1"], 4), v.get("support", "")])
        for key, metric, note in (
                ("significance_macro_f1", "macro_f1",
                 "below the pre-registered +0.02 threshold and not significant"),
                ("significance_docx_zip", "docx_to_zip_confusion",
                 "POSITIVE effect = confusion INCREASED, i.e. the targeted mechanism "
                 "moved the wrong way")):
            s = comp.get(key)
            if s:
                sig.append(["ch5_4096B", 4096, "+C1 vs backbone", metric,
                            round(s.get("effect_size", 0), 4), s.get("p_value", ""),
                            s.get("significant", "")])
                _long("classification_significance", "ch5_4096B/4096B",
                      "+C1 vs backbone", metric, round(s.get("effect_size", 0), 4),
                      note=note)
    else:
        cp = _load("ch5/compound_partial.json")
        if cp:
            for name, runs in cp.items():
                for r in runs:
                    main.append(["ch5_4096B (per-seed, RUN IN FLIGHT)", 4096,
                                 f"{name} seed{r['seed']}", 1, "partial",
                                 round(r["macro_f1"], 4), "",
                                 round(r["per_class"]["docx"]["f1"], 4),
                                 round(r.get("docx_zip_confusion_rate") or 0, 4),
                                 round(r.get("micro_accuracy", 0), 4)])

    # reproduction gates
    g512 = _load("new_result/p1_gate_512B.json")
    if g512 and "gate" in g512:
        g = g512["gate"]
        gates.append([512, round(g["test_accuracy"], 4),
                      g.get("published_fifty_accuracy", ""), g.get("tolerance", ""),
                      g.get("gate_passed", ""), g.get("n_params", ""),
                      len(g.get("val_acc_history", []))])
        _long("gate", "P1 reproduction/512B", "FiFTy-style CNN", "test_accuracy",
              round(g["test_accuracy"], 4),
              note=f"published {g.get('published_fifty_accuracy')}, passed={g.get('gate_passed')}")
    g4096 = _load("ch5/gate_4096B.json")
    if g4096:
        gates.append([4096, round(g4096["test_accuracy"], 4), g4096.get("published", ""),
                      g4096.get("tolerance", ""), g4096.get("gate_passed", ""),
                      g4096.get("n_params", ""), g4096.get("epochs", "")])
        _long("gate", "P1 reproduction/4096B", "FiFTy-style CNN", "test_accuracy",
              round(g4096["test_accuracy"], 4),
              note=f"published {g4096.get('published')}, passed={g4096.get('gate_passed')} "
                   f"({g4096.get('epochs')} epochs)")

    _write("05_classification_main.csv",
           ["experiment", "fragment_size", "model", "n_seeds", "status", "macro_f1",
            "macro_f1_std", "docx_f1", "docx_to_zip_confusion", "micro_accuracy"],
           main, "results/new_result/, results/ch5/",
           "XGBoost beats the transformer at both sizes — include in the comparison table.")
    _write("06_classification_per_class.csv",
           ["experiment", "fragment_size", "model", "class", "precision", "recall", "f1",
            "support"],
           per_class, "results/new_result/, results/ch5/", "")
    _write("07_classification_significance.csv",
           ["experiment", "fragment_size", "comparison", "metric", "effect_size",
            "p_value", "significant"],
           sig, "results/new_result/fullspec_512B.json, results/ch5/compound_4096B.json",
           "The C1 null at both fragment sizes. Read the docx_to_zip row carefully: a "
           "POSITIVE effect there means confusion INCREASED, so the one mechanism the "
           "front-end was designed to fix moved the wrong way at 4096B.")
    _write("08_reproduction_gates.csv",
           ["fragment_size", "test_accuracy", "published", "tolerance", "gate_passed",
            "n_params", "epochs_or_history_len"],
           gates, "results/new_result/p1_gate_512B.json, results/ch5/gate_4096B.json",
           "4096B gate failed narrowly at 2 epochs; a 4-epoch rerun is queued.")
    _write("09_calibration_openset.csv",
           ["experiment", "fragment_size", "model", "ece_before", "ece_after",
            "temperature", "msp_auroc", "ood_note"],
           calib, "results/new_result/fullspec_512B.json", "")
    _write("10_deletion_curve.csv",
           ["experiment", "fragment_size", "model", "fraction_deleted", "accuracy_ig",
            "accuracy_random"],
           deletion, "results/new_result/fullspec_512B.json",
           "IG faithfulness: IG-ranked deletion should hurt more than random.")


# --------------------------------------------------------------------------
# coverage ceiling, PhotoRec, CFBF ablation
# --------------------------------------------------------------------------

def coverage_table():
    data = _load("structural_coverage.json")
    if not data:
        return
    rows, summary = [], []
    for size, block in data["sizes"].items():
        for cls, v in block["per_class"].items():
            rows.append([size, cls, round(v["cfbf_coverage"], 5),
                         round(v["opc_coverage"], 5), v["n"]])
        summary.append([size,
                        round(block["cfbf_coverage_on_cfbf_classes"], 5),
                        round(block["cfbf_false_positive_rate"], 5),
                        round(block["opc_coverage_on_opc_classes"], 5),
                        round(block["opc_false_positive_rate"], 5)])
        _long("structural_coverage", f"{size}B", "CFBF adapter",
              "coverage_on_own_classes",
              round(block["cfbf_coverage_on_cfbf_classes"], 5),
              note="upper bound on any structure-aware front-end's gain")
        _long("structural_coverage", f"{size}B", "OPC adapter",
              "coverage_on_own_classes", round(block["opc_coverage_on_opc_classes"], 5))
        _long("structural_coverage", f"{size}B", "OPC adapter",
              "false_positive_rate", round(block["opc_false_positive_rate"], 5),
              note="exceeds its own coverage: recovery is anti-discriminative")
    _write("11a_structural_coverage_per_class.csv",
           ["fragment_size", "class", "cfbf_coverage", "opc_coverage", "n_fragments"],
           rows, "results/structural_coverage.json", "")
    _write("11b_structural_coverage_summary.csv",
           ["fragment_size", "cfbf_coverage_own_classes", "cfbf_false_positive_rate",
            "opc_coverage_own_classes", "opc_false_positive_rate"],
           summary, "results/structural_coverage.json",
           "The coverage ceiling: 1-2% caps any structure-aware front-end, well "
           "below the pre-registered >=2% macro-F1 target.")


def photorec_table():
    data = _load("photorec_baseline/photorec_results.json")
    if not data:
        return
    rows = []
    ctrl = data.get("contiguous_control", {})
    for severity in data.get("severities", []):
        for cond, r in data["by_severity"][severity].items():
            label = CONDITION_LABEL.get(cond, cond)
            rows.append([severity, label, round(r["macro_recall"], 4),
                         round(r["macro_precision"], 4),
                         f"{r['n_zero']}/{r['n_scored']}"])
            _long("photorec_baseline", severity, label, "macro_recall",
                  round(r["macro_recall"], 4))
    _write("12_photorec_baseline.csv",
           ["severity", "condition", "macro_recall", "macro_precision", "zero_recovery"],
           rows, "results/photorec_baseline/photorec_results.json",
           f"CONTROL (mandatory when citing): on an unfragmented image of the same "
           f"{ctrl.get('n_files', '?')} files PhotoRec carved {ctrl.get('carved', '?')}, "
           f"{ctrl.get('readable', '?')} readable. Without this the zeros read as "
           "operator error.")
    _long("photorec_baseline", "contiguous control", "PhotoRec 7.2",
          "files_carved_readable",
          f"{ctrl.get('carved', '?')}/{ctrl.get('readable', '?')} of {ctrl.get('n_files', '?')}",
          note="proves the tool was correctly invoked")


def cfbf_ablation_table():
    data = _load("cfbf_ablation/cfbf_ablation.json")
    if not data:
        return
    rows, cls_rows = [], []
    for size, block in data["by_size"].items():
        sig = block.get("significance_macro_f1", {})
        for name, c in block["conditions"].items():
            rows.append([size, name, len(block.get("seeds", [])),
                         round(c["macro_f1_mean"], 4), round(c["macro_f1_std"], 4),
                         round(sig.get("effect_size", 0), 4) if name == "raw+cfbf" else "",
                         sig.get("p_value", "") if name == "raw+cfbf" else "",
                         sig.get("significant", "") if name == "raw+cfbf" else ""])
            _long("cfbf_ablation", f"{size}B", name, "macro_f1",
                  round(c["macro_f1_mean"], 4))
        for cls, g in block.get("cfbf_class_gain", {}).items():
            cls_rows.append([size, cls, round(g["raw"], 4), round(g["raw+cfbf"], 4),
                             round(g["delta"], 4)])
    _write("13a_cfbf_adapter_ablation.csv",
           ["fragment_size", "condition", "n_seeds", "macro_f1", "macro_f1_std",
            "effect_size", "p_value", "significant"],
           rows, "results/cfbf_ablation/cfbf_ablation.json",
           "Tests the coverage-ceiling prediction with XGBoost (stronger than the "
           "transformer at these budgets).")
    _write("13b_cfbf_adapter_perclass.csv",
           ["fragment_size", "class", "f1_raw", "f1_raw_plus_cfbf", "delta"],
           cls_rows, "results/cfbf_ablation/cfbf_ablation.json",
           "doc/xls/ppt are the targeted classes; a corpus null could hide a gain here.")


def oracle_table():
    """The upper bound on any compound-aware front-end — the definitive form of the
    classification negative."""
    data = _load("oracle_bound/oracle_bound.json")
    if not data:
        return
    rows = []
    label = {
        "deflate_recovered": "DEFLATE decompression (C1's mechanism)",
        "zip_lfh_name": "any ZIP local-header name",
        "opc_part_name": "OPC part name",
        "cfbf_structure": "CFBF sector structure",
        "any": "UNION of all signals",
    }
    target = data.get("target_effect", 0.02)
    for size in data["sizes"]:
        block = data["by_size"][str(size)]
        rows.append([size, "baseline (no oracle)", "",
                     round(block["baseline_macro_f1_mean"], 4), "", "", ""])
        for signal, s in block["signals"].items():
            rows.append([size, label.get(signal, signal),
                         round(s["coverage_mean"], 5),
                         round(s["macro_f1_mean"], 4),
                         round(s["gain_mean"], 4), round(s["gain_std"], 4),
                         s["clears_target"]])
            _long("oracle_upper_bound", f"{size}B", label.get(signal, signal),
                  "max_achievable_macro_f1_gain", round(s["gain_mean"], 4),
                  note=f"perfect detection assumed; target {target}")
    _write("14_oracle_upper_bound.csv",
           ["fragment_size", "signal_with_perfect_detection", "coverage",
            "macro_f1", "gain", "gain_std", "clears_2pct_target"],
           rows, "results/oracle_bound/oracle_bound.json",
           f"Upper bound no real front-end can exceed. The UNION row is load-bearing: "
           f"even perfect detection over every structural signal gains far less than "
           f"the pre-registered {target:.0%} target, so the classification null is a "
           f"property of the data, not of any architecture.")


def pdf_coverage_table():
    """PDF vs OOXML coverage — explains why the published PDF precedent diverges."""
    data = _load("pdf_coverage/pdf_coverage.json")
    if not data:
        return
    rows, summary = [], []
    for size in data["sizes"]:
        block = data["by_size"][str(size)]
        for cls, v in block["per_class"].items():
            rows.append([size, cls, round(v["ooxml_mechanism_coverage"], 5),
                         round(v["pdf_mechanism_coverage"], 5),
                         round(v["pdf_structural_marker_rate"], 5), v["n"]])
        o = block["ooxml_coverage_own_mechanism"]
        p = block["pdf_coverage_own_mechanism"]
        summary.append([size, round(o, 5), round(p, 5),
                        round(p / o, 2) if o else "",
                        round(block["pdf_mechanism_on_ooxml"], 5),
                        round(block["ooxml_mechanism_on_pdf"], 5)])
        _long("pdf_coverage", f"{size}B", "OOXML (own mechanism)", "coverage", round(o, 5))
        _long("pdf_coverage", f"{size}B", "PDF (own mechanism)", "coverage", round(p, 5),
              note="explains why decompress-then-classify works for PDF but not OOXML")
    _write("15a_pdf_vs_ooxml_coverage_summary.csv",
           ["fragment_size", "ooxml_coverage_own_mechanism", "pdf_coverage_own_mechanism",
            "ratio", "pdf_mechanism_on_ooxml_FP", "ooxml_mechanism_on_pdf_FP"],
           summary, "results/pdf_coverage/pdf_coverage.json",
           "Confirms the coverage explanation against a published positive result in "
           "another format: PDF carries 7.2x OOXML's coverage at 4096B. Cross-mechanism "
           "false positives are 0.00%, so the gap is not a detector-permissiveness "
           "artifact.")
    _write("15b_pdf_vs_ooxml_coverage_per_class.csv",
           ["fragment_size", "class", "ooxml_mechanism", "pdf_mechanism",
            "pdf_structural_markers", "n_fragments"],
           rows, "results/pdf_coverage/pdf_coverage.json", "")


def coverage_calibration_table():
    """Coverage across ZIP-family formats — granularity isolated, container held fixed."""
    data = _load("coverage_calibration/coverage_calibration.json")
    if not data:
        return
    src = "results/coverage_calibration/coverage_calibration.json"
    measured, predicted = [], []
    for size in data["sizes"]:
        block = data["by_size"][str(size)]
        for cls, v in block.items():
            if cls.startswith("_"):
                continue
            measured.append([
                size, cls, round(v["header_rate"], 5), round(v["coverage"], 5),
                round(v.get("deflate_fraction", ""), 4) if "deflate_fraction" in v else "",
                round(v["mean_part_size"], 1) if "mean_part_size" in v else "",
                round(v["cv_part_size"], 3) if "cv_part_size" in v else "",
                v.get("n_parts_sized", 0), v["n"]])
            _long("coverage_calibration", f"{size}B", cls, "coverage",
                  round(v["coverage"], 5))
            if v.get("sufficient_sample"):
                predicted.append([
                    size, cls, round(v["header_rate"], 5),
                    round(v["pred_scattered"], 5), round(v["scattered_fold_error"], 3),
                    round(v["pred_occupancy"], 5), round(v["occupancy_fold_error"], 3),
                    round(v["cv_part_size"], 3), v["n_parts_sized"]])
        if "_median_fold_occupancy" in block:
            _long("coverage_calibration", f"{size}B", "occupancy model",
                  "median_fold_error", round(block["_median_fold_occupancy"], 3),
                  note="vs scattered model "
                       f"{block['_median_fold_scattered']:.2f}x — clustering matters")
    _write("16a_coverage_calibration_measured.csv",
           ["fragment_size", "format", "header_rate", "coverage", "deflate_fraction",
            "mean_part_size_bytes", "cv_part_size", "n_parts_sized", "n_fragments"],
           measured, src,
           "All ZIP containers measured with ONE detector, so container technology is "
           "held constant and only content granularity varies. Coverage spans 0.10%-62.40% "
           "at 4096B (624x). Note EPUB at 18.57%: the ceiling is NOT a property of "
           "compound documents as a class. Controls rtf/doc/txt read 0.00%.")
    _write("16b_coverage_calibration_predicted.csv",
           ["fragment_size", "format", "header_rate_measured", "pred_scattered",
            "scattered_fold_error", "pred_occupancy", "occupancy_fold_error",
            "cv_part_size", "n_parts_sized"],
           predicted, src,
           "Coverage predicted from part sizes alone. The occupancy model drops median "
           "fold error 2.29x -> 1.42x at 4096B by not assuming boundaries are scattered. "
           "xlsx withheld at both sizes (4 and 12 parts observed — undersampled).")


def write_index():
    L = ["# Result CSV index", "",
         "Generated by `python -m classify.eval.export_results_csv`. Re-run it after any "
         "experiment finishes — runs still in flight are skipped, not faked.", "",
         "**Write from `00_all_results_long.csv`**: one row per number, filterable by "
         "`table` or `metric`, so you do not have to hunt across the wide files.", "",
         "| file | rows | source | note |", "|---|---|---|---|"]
    for name, (source, note, n) in sorted(SOURCES.items()):
        if name.startswith("_"):
            continue
        L.append(f"| `{name}` | {n} | `{source}` | {note} |")
    if MISSING:
        L += ["", "## Not yet available (run still in flight or not started)", ""]
        L += [f"- `{m}`" for m in sorted(set(MISSING))]
    L += ["", "## Caveats that must travel with these numbers", "",
          "1. **micro vs macro recall** — where micro is flat while macro rises, the gain "
          "is *breadth* (fewer total failures), not more total text. Do not write "
          "\"recovered more text\" on its own.",
          "2. **`crc_verified` vs `complete`** — only the former is proof. B2 raises "
          "`complete` without raising `crc_verified`, so its extra completions are "
          "unverified.",
          "3. **PhotoRec's zeros require the contiguous control** (table 12) or they look "
          "like operator error.",
          "4. **CFBF has no checksum**, so no CRC column exists there and B2 has no "
          "provable floor — hence a precision-for-recall trade rather than a strict win.",
          "5. **Chapter 4's custom-corpus results are invalid** (that corpus was never "
          "doc-vs-docx) and `+C1` numbers predating 2026-07-30 used a refiner trained on "
          "61% wrong labels. Neither is in these CSVs except where labelled.",
          "6. **Fragmentation is synthetic** in both models; no real disk image was used.",
          "7. **The coverage ceiling is format-specific, not container-specific** "
          "(table 16a). EPUB is a ZIP-based document container reaching 18.57% at 4096B. "
          "Do not write that compound documents in general have low coverage.",
          ]
    with open(os.path.join(OUT_DIR, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"exporting to {OUT_DIR}/")
    restoration_tables()
    classification_tables()
    coverage_table()
    photorec_table()
    cfbf_ablation_table()
    oracle_table()
    pdf_coverage_table()
    coverage_calibration_table()

    _write("00_all_results_long.csv",
           ["table", "experiment", "condition", "metric", "value", "unit", "note"],
           LONG_ROWS, "all of the above", "Tidy long form — write from this one.")
    write_index()

    print(f"\n{len(SOURCES) - (1 if '_restoration' in SOURCES else 0)} CSV files + INDEX.md")
    if MISSING:
        print(f"skipped {len(set(MISSING))} not-yet-available source(s): "
              f"{', '.join(sorted(set(MISSING)))}")


if __name__ == "__main__":
    main()
