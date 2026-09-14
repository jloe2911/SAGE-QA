# SAGE-QA result map

This document maps scientific result generations to their authoritative artifacts. It is
not a duplicate metric dump. Exact values, hashes, cohorts, and lineage are defined by the
linked frozen outputs and machine-readable release indexes.

## 1. Final thesis results

| Field | Value |
|---|---|
| Protocol | Final thesis protocol: Generator D, DistilBERT cross-encoder, Text-Chain/Proof, adaptive support, support-grounded reader |
| Artifact root | `outputs/final_results/final_manuscript_test_end_to_end/` |
| Retrieval root | `outputs/final_results/manuscript_retrieval_results_hard_pair_v2/` |
| Release manifest | `release_manifests/thesis_final/index.yaml` |
| Status | Complete frozen; canonical thesis results |
| Thesis/manuscript | Yes |
| Caveat | Answer denominator 4,249; Support/Joint denominator 3,509; equal-dataset macro is primary. Historical V1 and hard-pair-v2 retrieval lineage remain distinct. |

## 2. Thesis baselines

| Field | Value |
|---|---|
| Protocol | Lexical k=1, clean GNN-RAG k=1, and gold-free full context on the frozen ten-dataset TEST cohort |
| Artifact root | `outputs/final_results/final_manuscript_baselines_test_end_to_end/` |
| Component root | `outputs/final_results/production_generator_d_v1_test_baselines/` |
| Release manifest | `release_manifests/thesis_baselines/index.yaml` |
| Status | Complete frozen thesis baselines |
| Thesis/manuscript | Yes, as comparisons rather than the main method |
| Caveat | Failed native attempts in the component root are development records, not baseline results. |

## 3. Thesis graph ablation

| Field | Value |
|---|---|
| Protocol | Hard-pair GraphSAGE with frozen GNN-only and SAGE-QA adaptive policies |
| Artifact roots | `outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/`; `outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end/` |
| Release manifest | `release_manifests/thesis_graph_ablation/index.yaml` |
| Status | Complete frozen TEST evaluation |
| Thesis/manuscript | Yes, as graph-model ablation |
| Caveat | This is a GraphSAGE lineage, not the final cross-encoder checkpoint. |

## 4. Original published results

| Field | Value |
|---|---|
| Protocol | Historical `experiments/run_experiments.py` workflow |
| Artifact root | `outputs/full_results/` |
| Release manifest | `release_manifests/paper_original/index.yaml` |
| Status | Frozen historical results; exact authoritative source revision unresolved |
| Thesis/manuscript | Published-paper results; not final-thesis results |
| Caveat | `dbdbb507` is the strongest source candidate but remains `candidate_not_fully_proven`; ignored assets lack a commit-bound top-level manifest. |

## 5. Superseded thesis results

| Field | Value |
|---|---|
| Protocol | Earlier Generator-D-v1 GraphSAGE and adaptive-policy thesis stages, plus separately classified A0/A3 submission work |
| Artifact roots | `outputs/final_results/production_generator_d_v1_test_retrieval/`; `outputs/final_results/production_generator_d_v1_answer_generation/`; `outputs/final_results/manuscript_retrieval_results/` |
| Release manifest | `release_manifests/thesis_superseded/index.yaml` |
| Status | Valid, frozen, historical, and superseded |
| Thesis/manuscript | Retained for provenance; not the canonical final-thesis result |
| Caveat | Do not subtract unmatched cohorts or substitute the historical V1 export for hard-pair-v2 retrieval. |

## 6. Complete Gold Support oracle and provenance correction

| Field | Value |
|---|---|
| Protocol | Order-preserving union of all persisted gold-support alternatives followed by support-grounded answer generation |
| Artifact roots | `outputs/final_results/gold_support_complete_oracle/`; `outputs/final_results/provenance_correction_v1/` |
| Release manifest | `release_manifests/oracle/index.yaml` |
| Status | Complete frozen post-publication oracle; additive provenance correction |
| Thesis/manuscript | Post-publication analysis; does not rewrite historical thesis/manuscript results |
| Caveat | The complete oracle has 3,509 support-bearing rows. It is not the historical Gold Support condition embedded in the frozen 16,256-row final-thesis prediction bundle. |

Development-only, rejected, diagnostic, exploratory, aborted, and incomplete work is
indexed separately in `release_manifests/development_archive/index.yaml`. See
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) and
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the reviewer-facing artifact and protocol
maps.
