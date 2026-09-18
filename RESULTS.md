# SAGE-QA results

Exact values, cohorts, hashes, and lineage are stored in the linked frozen outputs and
release manifests.

## 1. Published-paper results

- Protocol: historical `experiments/run_experiments.py` workflow over six reported settings.
- Results: `outputs/full_results/`.
- Manifest: `release_manifests/paper_original/index.yaml`.
- Source caveat: `dbdbb507` is the strongest candidate, but the exact published-result/source binding is not fully proven.

## 2. Thesis Chapter 7 main results

- Protocol: Generator D, DistilBERT cross-encoder, Text-Chain/Proof, adaptive aggregation, and the support-grounded reader over ten settings.
- Retrieval: `outputs/final_results/manuscript_retrieval_results_hard_pair_v2/`.
- End-to-end results: `outputs/final_results/final_manuscript_test_end_to_end/`.
- Manifest: `release_manifests/thesis_final/index.yaml`.
- Denominators: 4,249 answer rows; 3,509 rows with defined, non-empty gold support for Support and Joint metrics; equal-dataset macro is primary.

## 3. Thesis Chapter 7 baselines

- Methods: Lexical Subgraph k=1, clean GNN-RAG k=1, and full context.
- Results: `outputs/final_results/final_manuscript_baselines_test_end_to_end/` and the successful children under `outputs/final_results/production_generator_d_v1_test_baselines/`.
- Manifest: `release_manifests/thesis_baselines/index.yaml`.

## 4. Thesis Chapter 7 graph ablation

- Protocol: hard-pair GraphSAGE with frozen GNN-only and SAGE-QA adaptive policies.
- Results: `outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/` and `outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end/`.
- Manifest: `release_manifests/thesis_graph_ablation/index.yaml`.
- Boundary: this is a graph-model ablation, not the final cross-encoder checkpoint.

## 5. Thesis Chapter 7 complete-support reference

- Preferred term: **complete ground-truth support reference condition**.
- Repository artifact name: `gold_support_complete_oracle`.
- Results: `outputs/final_results/gold_support_complete_oracle/`.
- Manifest: `release_manifests/oracle/index.yaml`.
- Cohort: all 3,509 support-bearing rows, using the order-preserving union of persisted gold-support alternatives.

## 6. Thesis Chapter 7 stage-wise analysis

- Results: `outputs/final_results/final_manuscript_stagewise_error_analysis/`.
- Producing analysis: `evaluation/analyze_stagewise_test_errors.py`.
- Contract test: `tests/test_stagewise_test_errors.py`.
- Authority: `release_manifests/reported_results_mapping.yaml`.
