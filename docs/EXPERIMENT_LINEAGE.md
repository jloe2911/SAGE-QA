# Experiment lineage

The table links reported result families to the narrowest surviving executable and
artifact evidence. Hashes refer to original bytes unless a row explicitly says
`derived overlay`.

| Manuscript result family | Source/configuration | Frozen predictions or supports | Evaluator/finalizer | Recorded evidence and status |
|---|---|---|---|---|
| Published-paper retrieval and k sensitivity | `dbdbb507`; `experiments/run_experiments.py`; six paper datasets/checkpoints | `outputs/full_results/{dataset}/`; aggregate k export | `evaluation/eval_gnn_subgraph_retriever.py`, `evaluation/evaluate_owl_qa_predictions.py` | 18 data/checkpoint/result tree digests in the Phase 2 gate; historical source/result binding remains unverified |
| Published-paper end to end | Same historical runner and hosted reader | `outputs/full_results/full_pipeline_results.{csv,json}` | paper evaluators plus `evaluation/collect_final_results.py` | Surviving local results; no contemporaneous top-level manifest or provider snapshot |
| Thesis main cross-encoder/SAGE-QA | preservation baseline `884480b`; Generator D v1; DistilBERT; frozen adaptive policies | TEST retrieval `0251c2d2...`; reader predictions `03e391ed...` | `evaluation/export_manuscript_retrieval_results.py`, `evaluation/finalize_final_manuscript_end_to_end.py` | Complete frozen baseline indexes; 4,249 answer and 3,509 support rows |
| Thesis lexical/GNN-RAG/full-context baselines | `thesis_baselines` index and method-specific configurations | lexical `f7f12d47...`; GNN-RAG `6b343bd8...`; baseline predictions `2fec835e...` | `evaluation/finalize_final_manuscript_end_to_end.py` | Frozen and hash-indexed |
| Thesis hard-pair-v2 graph ablation | hard-pair checkpoint/policies; run began from `20c3ca8` with recorded dirty-state lineage | retrieval `1575240a...`; answers `751d9aac...` | production retrieval/answer evaluators | Frozen and hash-indexed; retained separately from optimized results |
| Thesis optimized SAGE-QA | `d26cb5a` + `75010ef`; `experiments/symbolic_coefficients_original_v1/` | retained selected supports `08e493b2...`; answers `712fce04...` | official overlay verifier; end-to-end finalizer | Downstream bytes verified; raw ranking `770ef934...` and derived DEV policy bundle missing |
| Thesis optimized retrieval overlay | `release_manifests/thesis_optimized/canonical_retrieval_overlay.json` | selected supports from frozen reader inputs | `evaluation/verify_canonical_retrieval_overlay.py` using `best_support_scores` | Independently reconstructed metric view, not recovered original ranking bytes |
| Thesis stagewise analysis | standard and optimized analysis scripts | per-example diagnostic outputs; optimized manifest `a8d6cf73...` | `classify_support`, including reasoner-validated alternatives | Diagnostic only; not official Support F1 |

## Evidence classification

| Class | Meaning | Current examples |
|---|---|---|
| Recovered and hash-verified | Original retained bytes match a recorded SHA-256/tree digest | thesis predictions/checkpoints; optimized reader inputs/answers; paper data/checkpoint/result trees |
| Independently reconstructed | Deterministic result recomputed from retained frozen downstream evidence | optimized canonical retrieval overlay |
| Missing raw artifact | Expected hash survives, bytes do not | optimized TEST ranking and selected coefficient/policy/DEV-reference bundle |
| Unverified historical provenance | Association is strong but no contemporaneous manifest closes it | `dbdbb507` as the exact published-paper source; historical provider/environment identity |

Machine-readable details are in `release_manifests/reported_results_mapping.yaml`, the
five existing thesis/paper indexes, and `release_manifests/thesis_optimized/index.yaml`.
Names such as `final`, `original`, or `v1` are not provenance evidence by themselves.
