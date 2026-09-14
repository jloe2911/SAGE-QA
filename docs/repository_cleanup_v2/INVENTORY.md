# SAGE-QA repository cleanup v2: inventory

Audit snapshot: 2026-09-11 18:10:43 +02:00  
Repository: `C:\Users\julie\github\PhD\SAGE-QA`  
Git: `main` at `25be348f4cf885e258f65d5a28bfc62abbb94158`  
Mode: non-destructive inspection; only this audit directory was created.

## Executive result

The checkout is not ready for physical cleanup. The canonical retrieval artifacts are complete and frozen, but `outputs/final_results/final_manuscript_test_end_to_end/` is still an in-progress answer-generation bundle. Its manifest status is `preflight_complete_no_api_calls`; `predictions.jsonl` is not listed in that manifest and was changing during this audit. The directory must be protected as a whole, and no prediction count or hash is endorsed here.

The reviewer-facing default should be:

`generator_d_frozen_v1 -> question-candidate DistilBERT cross-encoder -> Text-Chain (text) / Proof (ontology) -> support aggregation -> support-grounded answer generation`

The old GNN pipeline remains a labelled historical/original-paper architecture and graph ablation. It must not remain the README default.

## Snapshot totals and limitations

| Measure | Snapshot |
|---|---:|
| Accessible files | 62,489 |
| Accessible bytes | 41,910,391,680 |
| GiB | 39.032 |
| Git-tracked paths | 193 |
| First-party Python files parsed | 134 |
| First-party test files found | 38 |
| Tests under `tests/` | 36 |
| Tracked tests under `tests/` | 13 |
| Git worktree entries | 3 untracked, 0 tracked modifications |

The totals are lower bounds. Windows denied traversal of 12 pytest scratch directories: `.test_tmp_adaptive_k/`, eight `.tmp/pytest*` directories, `.tmp_test_answer_generation/`, `pytest-of-julie/`, and `outputs/audits/final_test_evaluation_audit/pytest_tmp/`. `outputs/final_results/final_manuscript_test_end_to_end/predictions.jsonl` was also growing, so `outputs/` and repository byte totals are a point-in-time snapshot.

The three untracked files are thesis-critical, not clutter:

- `evaluation/export_manuscript_retrieval_results.py`
- `generation/run_final_manuscript_answer_generation.py`
- `tests/test_final_manuscript_answer_generation.py`

No `.env` contents were inspected or reproduced.

## Top-level inventory

The complete measured table is in `DIRECTORY_SIZES.csv`. The largest top-level roots are:

| Path | Files | Bytes | Role |
|---|---:|---:|---|
| `data/` | 181 | 27,703,723,438 | Frozen Generator D, original processed data, raw inputs |
| `.venv/` | 57,074 | 5,778,488,083 | Re-creatable local environment; retain until freeze/environment capture |
| `.git/` | 2,843 | 2,620,334,539 | Repository history; never manually prune objects |
| `third_party/` | 524 | 1,919,105,185 | GNN-RAG source plus generated inputs/results |
| `checkpoints/` | 417 | 1,874,943,938 | Final, baseline, original-paper, and development models |
| `outputs/` | 922 | approximately 1,807,979,190 | Frozen results, live results, diagnostics, historical outputs |
| `.worktrees/` | 145 | 112,370,538 | Detached original-evaluation compatibility worktree |

## Canonical thesis-final inventory

The following are the primary release roots and must not be modified:

| Artifact | Files | Bytes | Status/evidence |
|---|---:|---:|---|
| `data/production_generator_d_v1/` | 66 | 16,906,792,497 | `generator_d_frozen_v1`; 66-entry lineage aggregate `27a4c149181e96df1fa9aca581eaeee17e354755f6a034ce93f2021b496296e5` |
| `experiments/cross_encoder_reranking_dev_v1/` | 9 | 120,804 | DEV protocol, frozen experiment code and A40 launchers; includes re-creatable `__pycache__` |
| `outputs/development_runs/question_candidate_cross_encoder_v1/` | 15 | 357,739,103 | Frozen DEV cross-encoder experiment/checkpoint bundle |
| `outputs/development_runs/question_candidate_cross_encoder_v1_adaptive_input/` | 3 | 17,232,753 | Frozen adaptive-policy input |
| `outputs/development_runs/question_candidate_cross_encoder_v1_cross_encoder_adaptive_k/` | 16 | 6,774,977 | Cross-encoder adaptive policy |
| `outputs/development_runs/question_candidate_cross_encoder_v1_final_sageqa_adaptive_k/` | 16 | 6,838,227 | Final SAGE-QA adaptive policy |
| `outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40/` | 6 | 15,587,433 | `complete_frozen`; all 4 manifest members re-hashed successfully |
| `outputs/final_results/manuscript_retrieval_results/` | 8 | 46,907 | `complete_frozen_artifact_only_export`; all 7 members and 14 sources re-hashed successfully |
| `outputs/final_results/final_manuscript_test_end_to_end/` | 4 at snapshot | approximately 27.6 MB and growing | Protected in progress; only the two preflight-manifest members re-hashed successfully |

The corpus also has a distinct 30-retrieval-split-only hash, `fb77a4f30019d297ef6dbed22dec83bc86bedeaed5281fa51d2f1e7a8c33b076`. It is not interchangeable with the 66-entry aggregate.

## Historical and supporting inventory

- Original-paper processed data: ten dataset roots under `data/`, 8,848,381,850 bytes.
- Original-paper GraphSAGE checkpoints: ten `checkpoints/gnn_subgraph_ranker_*_full/` roots, 182,312,415 bytes, all referenced by `experiments/run_experiments.py`.
- Original-paper results: `outputs/full_results/`, 276,686,666 bytes.
- Frozen hard-pair graph-ablation artifacts: `checkpoints/production_generator_d_v2_hard_pair/`, the three hard-pair policy roots, and the hard-pair retrieval/end-to-end bundles. These are no longer the canonical final architecture but remain thesis-grade historical/ablation evidence.
- Clean GNN-RAG baseline: `checkpoints/production_generator_d_v1_gnn_rag_clean/` and `outputs/final_results/production_generator_d_v1_test_baselines/`. Keep until final manuscript generation, evaluation, and lineage are frozen.
- Shared active source: candidate construction/contracts, model modules, adaptive aggregation, readers, evaluators, and final export/generation runners.
- Development evidence: rejected proof-gate, RRF, static-hard, corrected-listwise, disagreement, closure, coverage, and bottleneck studies. Archive with status; do not erase negative results.

## Source, test, and quality audit

Static AST parsing covered 134 first-party Python files in `data_processing/`, `evaluation/`, `generation/`, `experiments/`, `models/`, `training/`, `utils/`, `scripts/`, and `tests/`. There were no syntax errors.

The 134 files break down as: 11 data-processing, 56 evaluation, 6 generation, 5 experiments, 9 models, 7 training, 4 utilities, and 36 tests. Two additional tests live outside `tests/`: `evaluation/test_generator_d_production_smoke.py` and `experiments/cross_encoder_reranking_dev_v1/test_experiment.py`.

Eighteen tests import missing first-party modules and cannot be collected successfully from this checkout without restoration or archival:

- `tests/test_2wiki_heldout_a3.py`
- `tests/test_2wiki_hybrid_candidate_retrieval_audit.py`
- `tests/test_candidate_construction_audit.py`
- `tests/test_compare_retrieval_runs.py`
- `tests/test_create_gnn_dev_sample.py`
- `tests/test_familyowl_a4_proof_projection.py`
- `tests/test_familyowl_architecture_extensions.py`
- `tests/test_familyowl_semantic_first_reranking.py`
- `tests/test_familyowl_semantic_labels.py`
- `tests/test_final_architecture.py`
- `tests/test_gnn_rag_alignment.py`
- `tests/test_prepare_2wiki_reader_comparison.py`
- `tests/test_probe_familyowl_compactness.py`
- `tests/test_probe_familyowl_representations.py`
- `tests/test_symbolic_candidate_diagnostics.py`

Some files have multiple missing imports; the missing modules total 18 distinct import statements. Of the 36 tests in `tests/`, 13 are tracked, 22 historical tests are ignored through repository-local `.git/info/exclude`, and `tests/test_final_manuscript_answer_generation.py` is visible as an untracked final-pipeline test.

Ruff 0.15.1 was run read-only with `--no-cache --select F401`. It reported 41 unused imports; no fixes were applied. Notable final-path findings include unused imports in `evaluation/audit_final_test_evaluation.py`, `evaluation/export_manuscript_retrieval_results.py`, `evaluation/run_cross_encoder_test_retrieval.py`, `generation/run_final_manuscript_answer_generation.py`, and `generation/run_final_manuscript_baselines.py`. These are quality tasks, not evidence that the files are obsolete.

No tests, imports, training, inference, answer generation, or evaluation were run. Prior pytest results affected by Windows temp ACLs are not treated as current suite evidence.

## Exact duplicates and large files

The SHA-256 duplicate scan excluded `.git/`, `.venv/`, and `.worktrees/`. It found 108 groups and 887,816,341 bytes of theoretical redundancy if one copy per group were retained. This is not safely removable space. The largest groups are detailed in `DUPLICATE_AND_SUPERSEDED_CODE.md`; most copies are intentional GNN-RAG self-contained inputs/results, raw benchmark shards, or frozen output-local checkpoint copies.

The top 20 files and directories appear in `DATA_OUTPUT_CHECKPOINT_AUDIT.md`. The two largest files are the Generator D HotpotQA and 2Wiki training retrieval JSONL files (3.86 GB and 3.43 GB). They are protected, not cleanup targets.

## Reference and entry-point findings

- Root `README.md` still makes `experiments/run_experiments.py` the main runner and does not expose the canonical cross-encoder pipeline.
- Final cross-encoder orchestration is spread across experiment launchers plus `evaluation/preflight_cross_encoder_a40_test.py`, `evaluation/prepare_cross_encoder_adaptive_dev.py`, `evaluation/run_cross_encoder_test_retrieval.py`, `evaluation/export_manuscript_retrieval_results.py`, and `generation/run_final_manuscript_answer_generation.py`.
- Generator D is hardcoded in at least 21 current source/launcher locations. A physical move requires a config/path-compatibility change first.
- V2 adaptive aggregation imports helpers from V1; V1 is not currently safe to archive independently.
- Frozen lineage contains absolute paths. Preserve frozen bytes and add a release-side relative-path map rather than rewriting results.
- No top-level `retrieval/` or `symbolic/` directory exists. Retrieval logic currently lives in `data_processing/`, `evaluation/`, `models/`, and `training/`; symbolic logic is mainly in `models/` and `evaluation/`. The proposed structure avoids a disruptive split before freeze.

## Counts used in the final report

`FILE_CLASSIFICATION.csv` uses explicit logical path units. Counts refer to those path units, not every nested member of a directory:

| Classification | Path units |
|---|---:|
| SAFE_DELETE | 28 |
| SAFE_ARCHIVE | 32 |
| KEEP_FINAL | 38 |
| KEEP_LEGACY | 42 |
| NEEDS_REVIEW | 36 |

Known conditionally removable bytes are 5,781,065,147 (5.384 GiB): the local environment plus measured caches. This excludes 12 unreadable scratch roots of unknown size. Nothing should be removed until the in-progress manuscript run exits and its environment/freeze records are complete.

Estimated manifest-preserving archiveable bytes are 1,002,367,860 (0.934 GiB), covering six development output/checkpoint roots and 26 superseded diagnostic source files. Archival means relocation only after reference updates and hash verification, never deletion.
