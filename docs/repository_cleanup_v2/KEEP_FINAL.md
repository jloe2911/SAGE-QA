# Keep final

These artifacts define or directly support the canonical leakage-free thesis pipeline. They must remain immutable where frozen and must be the default in future documentation.

## Frozen data, model, policy, and result roots

1. `data/production_generator_d_v1/` — frozen Generator D candidate corpus. Preserve the 66-entry aggregate hash `27a4c149181e96df1fa9aca581eaeee17e354755f6a034ce93f2021b496296e5` and the distinct 30-split hash `fb77a4f30019d297ef6dbed22dec83bc86bedeaed5281fa51d2f1e7a8c33b076` with explicit scopes.
2. `experiments/cross_encoder_reranking_dev_v1/` — protocol, frozen experiment implementation, test, and A40 launchers. Its `__pycache__` is excluded.
3. `outputs/development_runs/question_candidate_cross_encoder_v1/` — frozen DEV experiment/model.
4. `outputs/development_runs/question_candidate_cross_encoder_v1_adaptive_input/` — policy input.
5. `outputs/development_runs/question_candidate_cross_encoder_v1_cross_encoder_adaptive_k/` — cross-encoder policy.
6. `outputs/development_runs/question_candidate_cross_encoder_v1_final_sageqa_adaptive_k/` — final Text-Chain/Proof policy.
7. `outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40/` — final frozen TEST retrieval. Manifest status `complete_frozen`; 4/4 member hashes verified in this audit.
8. `outputs/final_results/manuscript_retrieval_results/` — canonical harmonized retrieval results. Manifest status `complete_frozen_artifact_only_export`; 21/21 member/source checks verified.
9. `outputs/final_results/final_manuscript_test_end_to_end/` — canonical answer-generation area. It is KEEP_FINAL but not yet a completed result. At snapshot its manifest was only `preflight_complete_no_api_calls`, while predictions were present and changing.

## Canonical orchestration files

- `evaluation/preflight_cross_encoder_a40_test.py`
- `evaluation/prepare_cross_encoder_adaptive_dev.py`
- `evaluation/run_cross_encoder_test_retrieval.py`
- `evaluation/export_manuscript_retrieval_results.py`
- `generation/run_final_manuscript_answer_generation.py`
- `evaluation/adaptive_support_aggregation_v2.py`
- `evaluation/fit_production_dev_adaptive_k.py`

`evaluation/export_manuscript_retrieval_results.py` and `generation/run_final_manuscript_answer_generation.py` are currently untracked. They require allowlisted source review and commitment; they are not deletion candidates.

## Shared implementation required by the final path

- `data_processing/evidence_graph_candidates.py`
- `data_processing/retrieval_contracts.py`
- `models/familyowl_atomic_selector_v2.py`
- `models/familyowl_compiler_v2.py`
- `models/twowiki_sentence_compiler_v1.py`
- `models/symbolic_composer.py`
- `models/symbolic_executor.py`
- `evaluation/adaptive_support_aggregation.py` — despite its V1 name, V2 imports shared identity/fixed-k helpers from it.

Readers and answer evaluators invoked by the final generation runner must be added to its final lineage hash manifest after completion. Do not infer their canonicality solely from filenames; use the runner’s actual imports and final lineage record.

## Supported tests currently tracked

The 13 tracked tests under `tests/` are:

- `test_adaptive_support_aggregation.py`
- `test_adaptive_support_aggregation_v2.py`
- `test_final_manuscript_baselines.py`
- `test_full_context_progressive_beam.py`
- `test_gold_free_retrieval.py`
- `test_hard_pair_reservation.py`
- `test_listwise_soft_target_loss.py`
- `test_ontology_atomic_pool_experiment.py`
- `test_prepare_production_gnn_rag_clean.py`
- `test_production_test_answer_generation.py`
- `test_size_balanced_candidate_composer.py`
- `test_static_hard_negative_refinement.py`
- `test_unified_evidence_graph_candidates.py`

Also KEEP_FINAL and track `tests/test_final_manuscript_answer_generation.py`, which is currently untracked.

After the final freeze, define a smaller canonical CI set focused on:

- gold-free candidate/input construction;
- frozen selection identity and manifest verification;
- cross-encoder model snapshot portability;
- Text-Chain/Proof routing and support aggregation;
- main-thread-only prediction persistence and exact resume keys;
- freeze-before-gold-join enforcement;
- evaluator parity and per-example/aggregate consistency.

## Protection rules

- Never rewrite frozen files to remove absolute paths; add a relative-path release map beside them.
- Never substitute the 66-entry Generator D hash for the 30-split baseline hash.
- Never expose 2Wiki gold evidence, answers, or gold units to selection/generation.
- Do not endorse the final answer bundle until prediction row counts, generation freeze, metrics, per-example output, lineage, audit, and self-excluding manifest are complete and hash-valid.
