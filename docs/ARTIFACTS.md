# Reviewer artifact map

Only artifacts required for the published paper or Thesis Chapter 7 are listed here.
Machine-readable paths and hashes remain authoritative in `release_manifests/`.

## Published paper

| Required component | Expected path | Authority |
|---|---|---|
| Six processed datasets | `data/{HotpotQA,2WikiMultiHopQA,FamilyOWL_1hop,FamilyOWL_2hop,owl2bench_1hop,owl2bench_2hop}/` | `release_manifests/paper_original/index.yaml` |
| Six GraphSAGE checkpoints | `checkpoints/gnn_subgraph_ranker_*_full/` for the six reported settings | `release_manifests/paper_original/index.yaml` |
| Raw benchmark/ontology inputs | `data/raw/{hotpot_qa,2WikiMultihopQA,family,owl2bench}/` | `release_manifests/protection_rules.yaml` |
| Frozen paper results | Six reported children and aggregate files under `outputs/full_results/` | `release_manifests/paper_original/index.yaml` |
| Historical source and evaluators | `experiments/run_experiments.py`, historical generation/evaluation modules, and vendored GNN-RAG | `release_manifests/reported_results_mapping.yaml` |

## Thesis Chapter 7

| Result family | Required component/path | Authority |
|---|---|---|
| Main method | `data/production_generator_d_v1/`, final cross-encoder checkpoint and adaptive policies, `outputs/final_results/manuscript_retrieval_results_hard_pair_v2/`, `outputs/final_results/final_manuscript_test_end_to_end/` | `release_manifests/thesis_final/index.yaml` |
| Baselines | `checkpoints/production_generator_d_v1_gnn_rag_clean/`, successful lexical/GNN-RAG retrieval roots, `outputs/final_results/final_manuscript_baselines_test_end_to_end/` | `release_manifests/thesis_baselines/index.yaml` |
| Graph ablation | hard-pair checkpoint, DEV policies, TEST retrieval/end-to-end results, and final TEST audit | `release_manifests/thesis_graph_ablation/index.yaml` |
| Complete-support reference | `outputs/final_results/gold_support_complete_oracle/` | `release_manifests/oracle/index.yaml` |
| Stage-wise analysis | `outputs/final_results/final_manuscript_stagewise_error_analysis/` | `release_manifests/reported_results_mapping.yaml` |

The complete-support artifact supplies the reported **complete ground-truth support
reference condition**. Its repository artifact name is `gold_support_complete_oracle`.

## REVIEWER_REPRODUCTION_BLOCKER

No verified DOI, archive URL, or model-hosting URL is currently bound to these required
external bundles. The release indexes record `externally_archived: unknown`, and the
previous availability audit recorded `external_location: pending` /
`CURRENTLY_NOT_PUBLICLY_BOUND`.

The exact publication units still needed are:

1. `paper_original_artifacts`
   - six processed paper datasets;
   - six paper GraphSAGE checkpoints;
   - required raw inputs;
   - frozen `outputs/full_results/` results; and
   - an environment/hosted-reader provenance record with checksums and extraction paths.
2. `thesis_ch7_artifacts`
   - the 66-member Generator D corpus;
   - pinned DistilBERT base snapshot;
   - final cross-encoder, hard-pair GraphSAGE, and clean GNN-RAG checkpoints;
   - adaptive policies;
   - frozen main, baseline, graph-ablation, complete-support, and stage-wise outputs; and
   - GNN-RAG-specific model/data/configuration assets required beyond the vendored source.

Until both bundles have verified locations, a fresh clone supports source inspection but
not complete artifact verification or end-to-end reproduction. Hosted-reader answer
generation additionally requires provider credentials and availability; frozen answers
can be verified locally without an API call.
