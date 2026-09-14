# Keep legacy

Legacy artifacts remain scientifically useful for original-paper reproduction, baselines, and graph-ablation comparisons. They must be clearly labelled so reviewers do not mistake them for the canonical final thesis pipeline.

## Original paper

- Ten original processed dataset roots under `data/`: HotpotQA, 2WikiMultiHopQA, FamilyOWL 1/2-hop, OWL2Bench 1/2-hop, and Pizza 100/250 1/2-hop. Combined size: 8,848,381,850 bytes.
- Ten `checkpoints/gnn_subgraph_ranker_*_full/` roots. Combined size: 182,312,415 bytes. All ten are referenced by `experiments/run_experiments.py`.
- `outputs/full_results/` (276,686,666 bytes).
- `experiments/run_experiments.py`, retained behind an explicit “Original paper — historical/leakage-affected protocol” wrapper.
- Original/shared support files such as `experiments/run_matched_original_a0_a3.py`, `evaluation/collect_final_results.py`, and the dataset-level LLM readers must remain until accepted-paper source lineage is pinned.

The public artifact index must state which original-paper inputs/results were leakage-affected and must not compare them as if generated under the final gold-free protocol.

## Thesis baselines

- `checkpoints/production_generator_d_v1_gnn_rag_clean/` (367,179,545 bytes).
- `outputs/final_results/production_generator_d_v1_test_baselines/` (559,972,632 bytes), including clean lexical/GNN-RAG data plus explicitly failed/incomplete native-run attempts.
- `outputs/final_results/final_manuscript_baselines_test_end_to_end/` (195,347,127 bytes). Full context must be labelled a reference/upper-context condition, not a deployable retrieval method.
- `third_party/GNN-RAG/` source and `patches/GNN-RAG-local-changes.patch`/`.diff`, pending an upstream commit/license decision.

Keep successful and failed baseline records distinct. Failed directories are not evidence of final scores, but their status/manifests document the path to the valid run.

## Graph-based thesis ablation

- `checkpoints/production_generator_d_v2_hard_pair/`.
- Hard-pair adaptive-k policy and k-sensitivity roots under `outputs/development_runs/`.
- `outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/`.
- `outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end/`.
- `outputs/audits/final_test_evaluation_audit/`, including its meaningful zero-byte finding sets.

These are complete frozen thesis artifacts, but the cross-encoder architecture now supersedes them as the final/default pipeline. Present them as a graph ablation/historical thesis stage.

## Earlier clean V1 results

- `outputs/final_results/production_generator_d_v1_test_retrieval/`.
- `outputs/final_results/production_generator_d_v1_answer_generation/`.

These are superseded, not disposable. Retain in a versioned historical index until the manuscript-to-artifact map confirms they are no longer cited.

## Raw sources

Keep `data/raw/` or replace it only with license-compliant acquisition instructions and exact hashes. The identical Hotpot parquet shards under distractor/fullwiki packaging are not safe deletion candidates until upstream dataset structure and licensing are confirmed.

## Required labels

Every legacy artifact must carry:

- protocol: `paper_original`, `thesis_baseline`, `thesis_graph_ablation`, or `thesis_superseded`;
- status: `COMPLETE`, `FAILED`, `INCOMPLETE`, `ABORTED`, or `REJECTED`;
- leakage scope;
- split and gold-access policy;
- producing source commit/file hashes;
- manifest hash and external archive location.
