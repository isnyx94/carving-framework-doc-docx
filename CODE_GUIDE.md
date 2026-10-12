# Code guide

This file explains every folder and script in the repository: what it does, how important it is, and which part of the thesis it supports. Section, table and figure numbers refer to the thesis *Developing a Carving Framework for Data Restoration of Composite File-Structure Document Types (DOC and DOCX)*. Items D1–D41 refer to Appendix A, Table A.1, and items C1–C9 to Table A.10.

**Importance**

- **Core**: the framework itself. Start here.
- **Main result**: produces a result reported in Chapters 4–5.
- **Supporting**: a check, a sensitivity analysis or a helper.
- **Figure**: draws a thesis figure from stored results.

## 1. Repository layout

```
classify/          Core framework: restorers, fragmenter, text metric, adapters, classifiers (Python package)
reanalysis/        One folder per study; each script runs an experiment or analyses its results
  common/          Helpers shared by the studies (corrected text extraction, loading results)
figures/           Scripts that draw the thesis figures from the stored results
CODE_GUIDE.md      This file
README.md          Short overview, installation and data sources
```

## 2. How to run

1. Install the dependencies: `pip install -r requirements.txt -r requirements-followup.txt`.
2. Run every script **from the repository root**. Paths are relative to the root, and imports assume `classify/` is on the path.
3. Get the input data and put it where the scripts expect it:
   - `data/govdocs1/` for the govdocs1 thread 000 `.doc` files;
   - `data/govdocs1/docx_lo/` for those files converted to `.docx` by LibreOffice;
   - `data/napierone_docx/extracted/` for the NapierOne DOCX files;
   - FFT-75 as described in `README.md`.

   The python-docx corpus of the main study cannot be shared (third-party personal data).
4. Scripts read and write results under `writing/…` and `results/…`. These are the same paths as in the separately deposited results package, so you can place that package at the root to re-analyse without re-running the experiments.
5. Every OOXML score in the thesis uses the corrected text extraction in `reanalysis/common/strict_patch.py`, which counts only `<w:t>` runs (thesis Section 3.7.2). Run scripts load it automatically.

## 3. `classify/` — the core framework

### 3.1 Restoration (Chapters 3–5, restoration studies)

| File | What it does | Importance | Thesis |
|---|---|---|---|
| `restore/docx_restore.py` | DOCX restorers B0 (contiguous inflate from each ZIP local header) and B2 (completeness check, adjacent-block fast path, nearest-first decoder-guided search, plausibility floor 0.85, CRC early stop, per-part byte floor) | **Core** | §3.3.3, Fig 3.2, Table 3.2; C5 |
| `restore/docx_restore_dd.py` | B0-dd and B2-dd: the same restorers for ZIP parts whose sizes are deferred to a data descriptor (e.g. LibreOffice) | Core (exploratory extension) | §3.3.3, §5.12.9; C8 |
| `restore/cfbf_restore.py` | DOC restorers: B0 (2 MiB span read) and B2 (FAT/DIFAT → directory → `WordDocument` stream selection) | **Core** | §3.3.3, §5.15; C6 |
| `restore/fragmenter.py` | Builds fragmented images (shuffled and realistic models, three severities, seeded) with ground truth and fragmentation statistics | **Core** | §3.6, Table 3.9; C1 |
| `restore/text_recall.py` | Text metric: tokenisation, multiset recall and precision, reference text | **Core** | §3.7.2; C2 |
| `eval/restore_experiment.py` | DOCX experiment driver and harness | **Core** | §3.2, §4.3; C3 |
| `eval/cfbf_restore_experiment.py` | DOC experiment driver, same contract as the DOCX driver | **Core** | §3.2, §4.4; C7 |
| `eval/photorec_baseline.py` | Runs PhotoRec on the same images and attributes carved files to documents | Main result | §5.12.4; C4 |
| `modules/c1_compound/adapters.py` | Container-adapter interface (OPC and CFBF) | **Core** | §3.2; C9 |
| `tests/` | Tests for the fragmenter, restorers, CRC checks, adapter contracts, splits and corpus integrity | Supporting | §3.10 |

### 3.2 Fragment classification (Chapter 4 §4.2; Chapter 5 §§5.2–5.10)

| Folder / file | What it does | Importance | Thesis |
|---|---|---|---|
| `modules/foundation/byte_transformer.py` | Byte-level transformer backbone | Core | §3.3.1 |
| `modules/c1_compound/` (detector, partial_decompress, cfbf_adapter, multi_adapter_model, dual_view_model, cnn_refiner, train_refiner) | Compound-aware front-end: structural detectors and partial decompression for OPC and CFBF, and the refiner | Core | §3.3.2 |
| `modules/heads/` | Classifier, attribution (Integrated Gradients) and open-set heads | Core | §3.3, §§5.4–5.5 |
| `baselines/` | FiFTy-style CNN, n-gram classical models, raw-byte transformer, signature sanity check | Main result | §5.2, §5.7.1 |
| `data/` | FFT-75 loader, fragment sampling, per-parent-file splits, local corpus and open-set data | Core | §3.4 |
| `training/` | Training loop, datasets, seeding | Core | §3.8 |
| `eval/p1_gate.py`, `eval/fullspec_512.py`, `eval/full_pipeline.py` | 512-byte reproduction gate and full ablation runs (backbone vs compound-aware) | Main result | §3.9.2, §5.3; D6–D8 |
| `eval/ch5_experiment.py` | 4,096-byte replication | Main result | §3.9.3, §5.7; D2, D6 |
| `eval/hypertune.py`, `eval/chapter4_pilot.py` | Training-recipe sweep and classification pilot | Supporting | §3.9.1; D1 |
| `eval/structural_coverage.py`, `eval/coverage_calibration*.py`, `eval/pdf_coverage_comparison.py` | How often structural evidence is present in a fragment; coverage calibration; PDF comparison | Main result | §5.8; D11–D14, D18–D20 |
| `eval/oracle_upper_bound.py` | Coverage-restricted oracle bound on the possible gain | Main result | §5.8.5; D15–D16 |
| `eval/cfbf_adapter_ablation.py` | Whether CFBF structural decoding adds classification signal | Main result | §5.15.7; D17 |
| `eval/deletion_curve.py`, `eval/metrics.py`, `eval/flops.py` | Attribution faithfulness, metrics and cost | Supporting | §§5.4, 5.6; D9–D10 |
| `eval/export_results_csv.py` | Converts every result JSON into the `results/csv/` tables used in the thesis | Supporting | §3.10 |
| `eval/ablation_runner.py`, `run_smoke.py`, `run_with_retry.py`, `resume_full_pipeline.py`, `ch5_resume.py`, `rerun_with_oversampling.py`, `diagnose_collapse.py`, `assemble_*`, `generate_report.py`, `new_result_report.py`, `finish_report.py`, `verify_fft75.py` | Run management, diagnostics, resumable runs and report assembly | Supporting | — |

## 4. `reanalysis/` — the studies

| Folder | Script | What it does | Importance | Thesis | Output |
|---|---|---|---|---|---|
| `common/` | `strict_patch.py` | Corrected OOXML text extraction (`<w:t>` runs only), applied to every OOXML score | **Core** | §3.7.2 | — |
| | `rescore_analyse.py` | Loads run files; document-level tests (paired t, Holm, Wilcoxon, bootstrap) | **Core** | §3.8, §5.12 | D21 |
| `01_docx_main_studies/` | `rescore_run.py` | Runs B0/B2 (and B2 without CRC, B0-dd/B2-dd) on one image: development, held-out, NapierOne (Word), LibreOffice, pilots and search budget | **Main result** | §§3.9.5, 3.9.7, 5.12.1–5.12.3, 5.12.5–5.12.7, 5.12.9 | D3–D5, D21, D27–D28 |
| | `rescore_queue.py` | Runs all jobs of a study in parallel, with checkpoints | Supporting | — | — |
| | `gen_numbers.py` | Collects the numbers quoted in the thesis text | Supporting | Ch5 | D25 |
| | `napierone_inspect.py`, `word_written_check.py` | Checks writer, data descriptors and text of the Word-written files | Supporting | §3.4.5, §5.12.9 | — |
| `02_fat32_volumes/` | `fs_build.py`, `fs_run.py` | Builds FAT32 volumes with mtools (4 KiB clusters, deletion and reuse) and runs B0, B2, PhotoRec and Foremost | **Main result** | §§4.5, 5.12.8, 5.15.8 | D37 |
| | `fs_queue.py`, `fs_queue_tools.py` | Job queues | Supporting | — | — |
| | `fs_analyse.py` | Volume-level analysis (DOC rows) | Main result | §5.15.8 | D37 |
| | `fs_rescore.py`, `fs_rescore_queue.py`, `fs_analyse_rescored.py` | Rebuilds each DOCX volume identically and re-scores it with the corrected extraction | **Main result** | §5.12.8 | D26 |
| | `fat32_weighted_rescored.py`, `fat32_fragmented_rescored.py` | Weighting check, and analysis of fragmented documents only | Supporting | §5.12.11, §5.12.8 | D31 |
| `03_production_carvers/` | `run_external.py`, `queue_ext.py` | Runs PhotoRec 7.1 and Foremost 1.5.7 on synthetic images with contiguous positive controls | **Main result** | §§5.12.4, 5.15.6 | D36 |
| | `e2_symmetric.py`, `e2_queue.py`, `e2_rescore_queue.py`, `ext_contiguous_wrap.py` | Scores every condition with both attributions, under the corrected extraction | Main result | §5.12.4, §5.15.8 | D24, D39 |
| | `b0b2_20.py` | B0/B2 on the same 20-file images, for comparison | Supporting | §5.12.4 | — |
| `04_doc_cfbf/` | `run_cfbf.py`, `analyse.py` | DOC study on 39 documents with per-document scores, and document-level analysis | **Main result** | §§5.15.1–5.15.4 | D32–D34 |
| | `run_cfbf109.py`, `analyse_heldout_cfbf108.py` | Whole-corpus confirmation (108 documents) | **Main result** | §5.15.5 | D35 |
| | `cfbf_independent_reference.py`, `cfbf_independent_analyse.py` | Re-scores against LibreOffice's exported text | Main result | §5.15.9 | D38 |
| `05_openable_files/` | `e4_openable.py`, `e4_queue.py`, `e4_analyse.py` | Rebuilds restored text parts into `.docx` files and tests whether they open | Main result | §5.12.10 | D29 |
| | `e4_libreoffice_build.py`, `lo_open_test.py` | The same test in LibreOffice | Supporting | §5.12.10 | — |
| `06_prior_art_setting/` | `boiko_style.py`, `boiko_analyse.py` | B0/B2 on NapierOne with 4 KiB clusters shuffled uniformly (settings S and M) | **Main result** | §5.12.12, Table 5.29 | D40 |
| `07_end_to_end/` | `e2e_demo.py` | Whole pipeline on one mixed image (DOCX, DOC, PDF, JPEG) | Main result | §4.6, Table 4.3 | D41 |
| `08_structural_coverage/` | `own_coverage.py`, `q.py` | Structural coverage on the local files with the FFT-75 detectors | Supporting | §5.8, Table 5.11 | D13 |
| `09_global_multiplicity/` | `global_multiplicity_v3.py` | Corrects all 63 tests together (Holm and Benjamini–Hochberg) | Supporting | §5.12.11, Table 5.28 | D30 |
| `10_appendix_examples/` | `examples.py`, `examples_strict.py`, `gen_appendix.py` | Worked recovery examples (public files only) and Appendix A tables | Supporting | Appendix A | — |
| | `markup_check.py`, `markup_impact_probe.py` | Checks that found and measured the text-extraction defect | Supporting | §3.7.2 | — |

## 5. `figures/`

Each script writes PNG files to `output_figures/`.

| Script | Figures | Thesis |
|---|---|---|
| `make_figures_classification.py` | Classification, coverage and oracle figures | Chapter 5, §§5.2–5.8 |
| `make_pilot_figs.py` | Pilot-study and search-budget figures | §3.9, Figs 3.6–3.7 |
| `make_ch4_figs.py` | Algorithm boxes, workflow diagrams and brief results | Chapter 4 |
| `make_ch5_figs.py`, `make_figures_v20.py` | Restoration results (recall by severity, effects, CRC ablation, FAT32, writers) | §§5.12–5.15 |
| `make_figures_v13.py`, `make_figures_v15.py` | Earlier diagrams and held-out, carver and FAT32 charts | Chapter 3 diagrams; §§5.12.4, 5.12.7 |
| `fig_boiko.py`, `fig_e2e.py` | Prior-art comparison and end-to-end pipeline | Fig 5.22, Fig 4.12 |


