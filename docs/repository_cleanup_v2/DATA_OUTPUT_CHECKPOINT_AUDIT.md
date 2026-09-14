# Data, output, and checkpoint safety audit

All actions are proposals only. “Archive” means hash-manifested relocation after consumers are updated. It never means delete.

## Large artifact roots

| Path | Bytes | Files | Purpose/status | Referenced by | Recommended destination/action |
|---|---:|---:|---|---|---|
| `data/production_generator_d_v1/` | 16,906,792,497 | 66 | Canonical frozen Generator D corpus | 21+ current sources/launchers; final retrieval/generation | Keep at current path through freeze; publish external `thesis_final/generator_d_frozen_v1` bundle and relative-path manifest. |
| Ten original processed `data/<dataset>/` roots | 8,848,381,850 | 65 | Original-paper inputs | `experiments/run_experiments.py` | Keep legacy; later map logically to `paper_original/processed` without rewriting frozen files. |
| `data/raw/` | 1,948,468,991 | 46 | Raw benchmark/ontology sources | Data builders | Keep checksums and acquisition metadata; redistribute only after license review. |
| `third_party/GNN-RAG/` | 1,900,266,512 | 496 | Baseline source plus generated data/checkpoints/results | Clean GNN-RAG pipeline | Keep code/patch; review generated 1.67 GB for deduplication only after fresh-clone baseline reproduction. |
| `checkpoints/development/` | 781,219,644 | 124 | Development/submission checkpoints | Tests, scripts, `artifacts/manifest.json` | SAFE_ARCHIVE after per-subtree status/manifest index. |
| `outputs/final_results/production_generator_d_v1_test_baselines/` | 559,972,632 | 124 | Clean lexical/GNN-RAG baseline plus failed attempts | Manuscript exporter and answer baselines | KEEP_LEGACY until final answer bundle freezes; then split COMPLETE from FAILED/INCOMPLETE logically. |
| `outputs/development_runs/` | 478,863,041 | 215 | Canonical cross-encoder DEV plus older DEV studies | Cross-encoder TEST runner and diagnostics | Keep four cross-encoder roots (388,585,060 bytes); archive the remaining 90,277,981 bytes by status. |
| `checkpoints/production_generator_d_v1_gnn_rag_clean/` | 367,179,545 | 193 | Clean GNN-RAG baseline checkpoints/adapters | Baseline runner/final answers | KEEP_LEGACY baseline bundle. |
| `outputs/full_results/` | 276,686,666 | 333 | Original-paper results | README/analysis | KEEP_LEGACY under paper-original release index. |
| Ten `checkpoints/gnn_subgraph_ranker_*_full/` | 182,312,415 | 30 | Original-paper GraphSAGE models | `experiments/run_experiments.py` | KEEP_LEGACY; manifest as `paper_original`. |
| `checkpoints/production_generator_d_v1/` | 181,578,675 | 30 | Pre-hard-pair clean GraphSAGE models | Diagnostics/baseline protections | NEEDS_REVIEW after final freeze and reference map. |
| `checkpoints/production_generator_d_v2_hard_pair/` | 181,577,929 | 30 | Frozen graph-ablation models | Frozen hard-pair retrieval lineage | KEEP_LEGACY immutable thesis ablation. |
| `checkpoints/production_generator_d_static_hard_v1/` | 181,075,730 | 10 | Rejected static-hard models | Static-hard diagnostics | NEEDS_REVIEW; likely archive as rejected negative result. |
| `outputs/final_results/final_manuscript_baselines_test_end_to_end/` | 195,347,127 | 9 | Frozen lexical/GNN-RAG/full-context answers | Manuscript comparison | KEEP_LEGACY immutable baseline/reference bundle. |
| `.worktrees/sageqa-original-eval-compat/` | 112,370,538 | 145 | Detached compatibility source | Git worktree metadata | NEEDS_REVIEW; tag/archive authoritative commit before any Git-native removal. |
| `outputs/development_diagnostics/` | 73,900,581 | 33 | Final bottleneck and reader diagnostics | Reports/scripts | SAFE_ARCHIVE as thesis diagnostics with manifests. |
| Hard-pair retrieval + end-to-end outputs | 74,873,319 | 12 | Frozen graph-ablation retrieval and 16,996 answers | Evaluation audit/lineage | KEEP_LEGACY immutable. |
| `dist/sageqa-submission-artifacts.zip` | 52,118,416 | 1 | Separate A0/A3 submission generation | `artifacts/manifest.json` | NEEDS_REVIEW; classify manuscript generation before deduplication. |
| `artifacts/step_01_context_to_kg/` | 36,463,705 | 64 | Wikidata cache and early architecture artifacts | Artifact manifest | NEEDS_REVIEW for license/privacy/costly-query provenance. |
| `outputs/diagnostics/` | 27,395,576 | 86 | Rejected/exploratory diagnostics | One-off evaluation scripts | SAFE_ARCHIVE by verdict. |
| `outputs/final_results/final_manuscript_test_end_to_end/` | approximately 27.6 MB and growing | 4 at snapshot | Canonical answer generation; preflight only | Final generation runner | KEEP_FINAL; do not hash/inspect predictions as final until freeze. |
| `outputs/quarantine/` | 20,047,392 | 11 | Aborted run, explicitly non-evidence | Quarantine metadata | SAFE_ARCHIVE as `ABORTED`. |
| `outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40/` | 15,587,433 | 6 | Canonical frozen TEST retrieval | Manuscript export/final generation | KEEP_FINAL immutable; 4/4 member hashes verified. |
| `outputs/final_results/manuscript_retrieval_results/` | 46,907 | 8 | Canonical frozen manuscript tables | Manuscript/export | KEEP_FINAL immutable; 21/21 source/member checks verified. |
| `experiments/cross_encoder_reranking_dev_v1/` | 120,804 | 9 | Canonical DEV protocol and launchers | Final retrieval | KEEP_FINAL source/protocol; delete only its bytecode cache later. |

## Top 20 largest directories

Recursive sizes overlap parent/child rows.

| Rank | Path | Bytes | Files |
|---:|---|---:|---:|
| 1 | `data/` | 27,703,723,438 | 181 |
| 2 | `data/production_generator_d_v1/` | 16,906,792,497 | 66 |
| 3 | `.venv/` | 5,778,488,083 | 57,074 |
| 4 | `.venv/Lib/site-packages/` | 5,743,194,378 | 56,447 |
| 5 | `data/production_generator_d_v1/HotpotQA/` | 5,404,032,767 | 5 |
| 6 | `data/production_generator_d_v1/2WikiMultiHopQA/` | 4,839,010,618 | 5 |
| 7 | `.venv/Lib/site-packages/torch/` | 4,681,261,896 | 12,695 |
| 8 | `.venv/Lib/site-packages/torch/lib/` | 4,570,241,692 | 59 |
| 9 | `data/HotpotQA/` | 3,463,148,993 | 8 |
| 10 | `data/2WikiMultiHopQA/` | 2,858,237,107 | 8 |
| 11 | `.git/` | 2,620,334,539 | 2,843 |
| 12 | `.git/objects/` | 2,620,233,531 | 2,762 |
| 13 | `data/raw/` | 1,948,468,991 | 46 |
| 14 | `third_party/` | 1,919,105,185 | 524 |
| 15 | `third_party/GNN-RAG/` | 1,900,266,512 | 496 |
| 16 | `checkpoints/` | 1,874,943,938 | 417 |
| 17 | `outputs/` | approximately 1,807,979,190 | 922 |
| 18 | `data/production_generator_d_v1/FamilyOWL_2hop/` | 1,321,427,519 | 7 |
| 19 | `data/production_generator_d_v1/FamilyOWL_1hop/` | 1,319,890,441 | 7 |
| 20 | `third_party/GNN-RAG/gnn/` | 1,310,692,205 | 246 |

## Top 20 largest files

| Rank | Path | Bytes | Action |
|---:|---|---:|---|
| 1 | `data/production_generator_d_v1/HotpotQA/train_subgraph_retrieval.jsonl` | 3,858,164,948 | KEEP_FINAL |
| 2 | `data/production_generator_d_v1/2WikiMultiHopQA/train_subgraph_retrieval.jsonl` | 3,425,043,706 | KEEP_FINAL |
| 3 | `data/HotpotQA/train_subgraph_retrieval.jsonl` | 2,317,847,984 | KEEP_LEGACY |
| 4 | `data/2WikiMultiHopQA/train_subgraph_retrieval.jsonl` | 1,861,301,492 | KEEP_LEGACY |
| 5 | `.venv/Lib/site-packages/torch/lib/torch_cuda.dll` | 957,268,480 | SAFE_DELETE only with whole environment and preconditions |
| 6 | `data/production_generator_d_v1/HotpotQA/test_subgraph_retrieval.jsonl` | 908,811,609 | KEEP_FINAL |
| 7 | `data/production_generator_d_v1/FamilyOWL_1hop/train_subgraph_retrieval.jsonl` | 854,643,861 | KEEP_FINAL |
| 8 | `data/production_generator_d_v1/FamilyOWL_2hop/train_subgraph_retrieval.jsonl` | 852,589,868 | KEEP_FINAL |
| 9 | `data/production_generator_d_v1/2WikiMultiHopQA/test_subgraph_retrieval.jsonl` | 830,289,872 | KEEP_FINAL |
| 10 | `data/HotpotQA/test_subgraph_retrieval.jsonl` | 749,400,810 | KEEP_LEGACY |
| 11 | `data/production_generator_d_v1/OWL2Bench_2hop/train_subgraph_retrieval.jsonl` | 745,667,030 | KEEP_FINAL |
| 12 | `data/production_generator_d_v1/OWL2Bench_1hop/train_subgraph_retrieval.jsonl` | 688,987,625 | KEEP_FINAL |
| 13 | `data/2WikiMultiHopQA/test_subgraph_retrieval.jsonl` | 662,236,278 | KEEP_LEGACY |
| 14 | `.venv/Lib/site-packages/torch/lib/dnnl.lib` | 653,534,016 | SAFE_DELETE only with whole environment and preconditions |
| 15 | `data/production_generator_d_v1/HotpotQA/dev_subgraph_retrieval.jsonl` | 636,884,836 | KEEP_FINAL |
| 16 | `.venv/Lib/site-packages/torch/lib/cudnn_engines_precompiled64_9.dll` | 588,910,632 | SAFE_DELETE only with whole environment and preconditions |
| 17 | `data/production_generator_d_v1/2WikiMultiHopQA/dev_subgraph_retrieval.jsonl` | 583,470,041 | KEEP_FINAL |
| 18 | `.venv/Lib/site-packages/torch/lib/cublasLt64_12.dll` | 472,802,816 | SAFE_DELETE only with whole environment and preconditions |
| 19 | `data/HotpotQA/dev_subgraph_retrieval.jsonl` | 382,601,331 | KEEP_LEGACY |
| 20 | `data/FamilyOWL_1hop/train_subgraph_retrieval.jsonl` | 357,458,046 | KEEP_LEGACY |

## Safety conclusions

- Do not place raw data, checkpoints, outputs, or `.env` in ordinary Git wholesale.
- Use ordinary Git for source, tests, documentation, configs, and small manifests/metrics.
- Publish `thesis-final`, `thesis-baselines`, and `paper-original` as separate checksummed release/archive bundles.
- Keep frozen output-local copies until a release verifier proves that deduplication does not break self-contained reproduction.
- Never infer safe deletion from duplicate hashes alone.
