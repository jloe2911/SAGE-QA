# Phase 7C final removal families

Status: planning only. No file was backed up, copied, moved, restored, deleted, or scientifically modified in Phase 7C.

The sole retention authority is `release_manifests/reported_results_mapping.yaml`. A family is removable only when no member lies on a reported PAPER or THESIS Chapter 7 dependency path. Every future physical batch still requires a new literal allowlist, reference audit, per-file hashes, external backup, isolated restore test, exact removal, narrow metadata update, and post-change gates.

## Closure conclusions

- PAPER contains six settings: HotpotQA, 2WikiMultiHopQA, Family 1/2-hop, and OWL2Bench 1/2-hop. Paper-era Pizza is not in the reported paper result set.
- THESIS contains all ten settings through the frozen Generator-D corpus, not through the old paper-era Pizza roots.
- The canonical retrieval export is `outputs/final_results/manuscript_retrieval_results_hard_pair_v2/`. The V1 export is superseded.
- The successful lexical and GNN-RAG children under `production_generator_d_v1_test_baselines/` are retained; failed and incomplete sibling attempts are not.
- `gold_support_complete_oracle/` is the reported complete-ground-truth-support condition: 3,509 examples, equal-dataset macro Answer EM 0.789303 and F1 0.896707.
- `evaluation/analyze_stagewise_test_errors.py`, `tests/test_stagewise_test_errors.py`, and `outputs/final_results/final_manuscript_stagewise_error_analysis/` are final Chapter 7 analysis, not development material.
- `outputs/development_diagnostics/final_old_vs_cross_encoder_analysis/comparison.json` is retained because the canonical hard-pair-v2 exporter reads it at runtime. The other six files in that diagnostic root are outside the result closure.

## Family inventory

Counts and bytes are current accessible-file measurements. ACL-blocked scratch paths are excluded. Sizes are planning estimates, not deletion authorization.

| Order | Family | File count | Bytes | PAPER reason | THESIS reason | External backup | Risk |
|---:|---|---:|---:|---|---|---|---|
| 1 | `ABORTED_LOCAL_GTX_TEST` | 1 | 499 | Post-paper | Aborted before evaluation; no predictions, metrics, or scientific decision | Optional preservation record | Low |
| 2 | `FAILED_INCOMPLETE_GNN_RAG_ATTEMPTS` | 11 | 104,017,264 | Not paper output | Successful `gnn_rag/` sibling is canonical; failed/incomplete attempts do not produce reported scores | Required | Low after protection-rule narrowing |
| 3 | `REJECTED_CANDIDATE_BUILDERS` | 93 | 16,952,614 | Post-paper development | Beam union, atomic-pool, query-local, size-balanced, semantic-sufficiency, and related experiments are not reported methods | Required | Low to medium |
| 4 | `REJECTED_SYMBOLIC_ALTERNATIVES` | 5 source files plus paired diagnostic inputs | 149,425 source bytes plus paired outputs | Not paper methods | RRF, proof gate, disagreement, complementarity, and cross-branch alternatives are not the final Text-Chain/Proof rule | Required | Medium; remove readers and their inputs together |
| 5 | `V1_DIAGNOSTIC_FAMILIES` | 39 | 22,369,220 | Post-paper | Causal retrieval, representation, mechanism decision, retrieval failure, symbolic-mechanism, and training-signal diagnostics do not produce a reported Chapter 7 result | Required | Medium; preserve the separate canonical exporter snapshot |
| 6 | `BOTTLENECK_AND_READER_DIAGNOSTICS` | 32 | 73,675,000 | Post-paper | Not the reported stage-wise analysis; six non-runtime files from `final_old_vs_cross_encoder_analysis/` are removable but `comparison.json` stays | Required | Medium |
| 7 | `LISTWISE` | 21 | 20,269,719 | Not the paper ranker | Chapter 7 specifies pairwise margin ranking/BCE, not listwise training | Required | Medium |
| 8 | `STATIC_HARD` | 29 | 190,141,429 | Not the paper checkpoint family | Rejected static-hard refinement is not the final reserved hard-pair lineage | Required | Medium |
| 9 | `MISC_DEVELOPMENT_CHECKPOINTS` | 53 | 293,754,255 | Not any of the six paper GraphSAGE checkpoints | Smoke, workbench, feasibility, walkthrough, and alternative FamilyOWL checkpoints do not support a reported result | Required | Medium |
| 10 | `A0_A3_SUBMISSION_AND_REBUILT_GNN` | 142 | 557,905,929 | Not the reported paper six-setting workflow | Current Chapter 7 does not report A0/A3 or rebuilt-GNN submission families | Required | Medium to high; includes costly-query cache and submission archive |
| 11 | `EARLY_V1_PIPELINE` | 122 | 256,052,847 | Post-paper | Replaced by hard-pair GNN plus final Cross-Encoder; exact `comparison.json` runtime snapshot remains retained | Required | High; current indexes and historical references must be redirected narrowly |
| 12 | `ORIGINAL_PAPER_UNREPORTED_PIZZA` | 166 | 947,432,081 | Pizza rows were commented out and are not reported PAPER results | Thesis Pizza uses `data/production_generator_d_v1/`, final Cross-Encoder, hard-pair, and clean baseline lineages | Required | High; currently protected and asserted by stale paper metadata/tests |
| 13 | `ABORTED_HARD_PAIR_QUARANTINE` | 11 | 20,047,392 | Post-paper | Aborted run is not the successful reported hard-pair family | Required | Medium |
| 14 | `RECREATABLE_BYTECODE` | 5 | 116,898 | No scientific content | No scientific content | No | Low; delete only after processes exit |
| 15 | `LOCAL_ENVIRONMENT` | 57,074 | 5,778,488,083 | Reinstallable environment | Reinstallable environment | No | Operational; never remove during a scientific batch |

The twelve measured scientific `BACKUP_AND_REMOVE` groups total 2,502,767,175 bytes (about 2.33 GiB), before the 499-byte aborted-local root and before any unresolved raw-source or worktree decision. `DELETE_RECREATABLE` is deliberately separate from scientific archival.

## Literal family paths

### 1. ABORTED_LOCAL_GTX_TEST

- `outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test/`

This contains only a 499-byte README stating that the local GTX attempt was aborted before evaluation and preserved no partial predictions. It is the safest next family once a literal one-root batch is authorized.

### 2. FAILED_INCOMPLETE_GNN_RAG_ATTEMPTS

- `outputs/final_results/production_generator_d_v1_test_baselines/gnn_rag_failed_clean_pre_prediction_20260830T131316/`
- `outputs/final_results/production_generator_d_v1_test_baselines/gnn_rag_incomplete_legacy_20260830T120036/`

Keep the successful siblings `lexical_subgraph/` and `gnn_rag/`. The current protection rule covers their common parent, so a later batch must first receive explicit authorization to narrow that rule without touching either successful child.

### 3. REJECTED_CANDIDATE_BUILDERS

- `outputs/development_runs/current_beam_union_v1/`
- `outputs/development_runs/current_clean_candidate_coverability_v1/`
- `outputs/development_runs/full_context_progressive_beam_v1/`
- `outputs/development_runs/generator_d_pair_supervision_v1/`
- `outputs/development_runs/ontology_atomic_pool_graph_expansion_v1/`
- `outputs/development_runs/ontology_semantic_sufficiency_v1/`
- `outputs/development_runs/protected_query_anchor_comparison_v1/`
- `outputs/development_runs/query_local_candidate_closure_v1/`
- `outputs/development_runs/size_balanced_structural_composer_v1/`
- `outputs/development_runs/unified_ontology_reachability_diagnostic_v1/`
- paired comparison/diagnostic CLIs under `evaluation/`
- `tests/test_full_context_progressive_beam.py`
- `tests/test_ontology_atomic_pool_experiment.py`
- `tests/test_size_balanced_candidate_composer.py`

The retained `data_processing/evidence_graph_candidates.py` and `tests/test_unified_evidence_graph_candidates.py` are excluded because they remain part of the Generator-D reproducibility contract.

### 4. REJECTED_SYMBOLIC_ALTERNATIVES

- `evaluation/evaluate_disagreement_union_dev.py`
- `evaluation/evaluate_generator_d_cross_branch_completion_dev.py`
- `evaluation/evaluate_ontology_proof_gate_dev.py`
- `evaluation/evaluate_production_dev_complementarity.py`
- `evaluation/evaluate_rrf_reranker_dev.py`
- any paired output roots already indexed as disagreement, proof-gate, complementarity, RRF, or cross-branch development results

The final `sageqa_text_chain_adjustment`, `sageqa_proof_adjustment`, and `compute_adjusted_score` implementation in `training/train_gnn_subgraph_retriever.py` remains retained.

### 5. V1_DIAGNOSTIC_FAMILIES

- `outputs/diagnostics/production_generator_d_v1_causal_retrieval_diagnosis/`
- `outputs/diagnostics/production_generator_d_v1_gnn_representation_diagnosis/`
- `outputs/diagnostics/production_generator_d_v1_mechanism_decision/`
- `outputs/diagnostics/production_generator_d_v1_retrieval_failure_audit/`
- `outputs/diagnostics/production_generator_d_v1_symbolic_mechanism_dev/`
- `outputs/diagnostics/production_generator_d_v1_training_signal_audit/`
- their producing diagnostic source files

These roots are consumed only by other rejected diagnostic/proof-gate CLIs. None is read by a retained reported-result evaluator.

### 6. BOTTLENECK_AND_READER_DIAGNOSTICS

- `outputs/development_diagnostics/final_bottleneck_analysis/`
- `outputs/development_diagnostics/final_bottleneck_reader_diagnostic/`
- all files in `outputs/development_diagnostics/final_old_vs_cross_encoder_analysis/` except `comparison.json`
- `generation/run_dev_reader_diagnostic.py`

The reported stage-wise family is a different root and remains retained.

### 7. LISTWISE

- `checkpoints/development/production_generator_d_v1_listwise_corrected_dev/`
- `outputs/development_runs/production_generator_d_v1_listwise_corrected_dev/`
- `outputs/diagnostics/production_generator_d_v1_listwise_corrected_dev/`
- `outputs/diagnostics/production_generator_d_v1_listwise_dev/`
- `experiments/run_corrected_listwise_dev_grid.py`
- `evaluation/audit_corrected_listwise_loss_scale.py`
- `evaluation/summarize_corrected_listwise_dev.py`
- `evaluation/verify_listwise_zero_baseline.py`
- `tests/test_listwise_soft_target_loss.py`

### 8. STATIC_HARD

- `checkpoints/production_generator_d_static_hard_v1/`
- `outputs/final_model_development/`
- `training/run_static_hard_negative_refinement.py`
- `evaluation/audit_static_hard_negative_refinement.py`
- `evaluation/evaluate_static_hard_refinement_dev.py`
- `evaluation/finalize_static_hard_refinement_dev.py`
- `tests/test_static_hard_negative_refinement.py`

### 9. MISC_DEVELOPMENT_CHECKPOINTS

- `checkpoints/development/2wiki_hybrid_feasibility_v1_seed42/`
- `checkpoints/development/gnn_familyowl_1hop_canonical_headlr/`
- `checkpoints/development/gnn_familyowl_1hop_canonical_v1/`
- `checkpoints/development/gnn_familyowl_1hop_exact_contrastive/`
- `checkpoints/development/gnn_familyowl_1hop_exact_proof_features/`
- `checkpoints/development/gnn_familyowl_1hop_nl_only_exact/`
- `checkpoints/development/gnn_familyowl_1hop_smoke/`
- `checkpoints/development/gnn_rag_walkthrough/`
- `checkpoints/development/pre_run_production_sanity_20260829/`
- `checkpoints/development/smoke_family_hybrid/`
- `checkpoints/development/workbench/`

The empty `checkpoints/development/debug_pizza_100_2hop/` is recreatable directory state, not a scientific payload.

### 10. A0_A3_SUBMISSION_AND_REBUILT_GNN

- `checkpoints/development/gnn_rebuilt/`
- `artifacts/manifest.json`
- `artifacts/step_01_context_to_kg/`
- `dist/sageqa-submission-artifacts.zip`
- `experiments/2wiki_a0_a3_heldout.json`
- `experiments/familyowl_a3_reader_test.json`
- `experiments/familyowl_a3_symbolic_reranking_frozen.json`
- `experiments/run_matched_original_a0_a3.py`
- `experiments/sageqa_original_a0_a3_matched_v1.json`
- `tests/test_2wiki_heldout_a3.py`
- `tests/test_run_matched_original_a0_a3.py`
- `tests/test_submission_readiness.py`
- other untracked tests whose imported A0/A3 implementation files are already absent, including the legacy `tests/test_symbolic_executor.py` path rather than the current reader-local deterministic proof implementation

Because the Wikidata cache may contain costly external-query provenance and redistribution-sensitive material, it requires an external preservation and license/privacy review before removal. That concern affects handling, not scientific KEEP status.

### 11. EARLY_V1_PIPELINE

- `checkpoints/production_generator_d_v1/`
- `outputs/development_runs/production_generator_d_v1_adaptive_k/`
- `outputs/development_runs/production_generator_d_v1_gnn_adaptive_k/`
- `outputs/development_runs/production_generator_d_v1_k_sensitivity/`
- `outputs/final_results/production_generator_d_v1_test_retrieval/`
- `outputs/final_results/production_generator_d_v1_answer_generation/`
- `outputs/final_results/manuscript_retrieval_results/`
- `release_manifests/thesis_superseded/index.yaml` after it has been converted into a retained removal/provenance record rather than a live bundle index

Do not include `outputs/development_diagnostics/final_old_vs_cross_encoder_analysis/comparison.json`; it is still a runtime input to the canonical exporter. Its embedded V1 values are a frozen validation snapshot and do not require keeping the entire V1 experiment zoo.

### 12. ORIGINAL_PAPER_UNREPORTED_PIZZA

- `data/pizza_100_1hop/`
- `data/pizza_100_2hop/`
- `data/pizza_250_1hop/`
- `data/pizza_250_2hop/`
- `data/raw/pizza_100/`
- `data/raw/pizza_250/`
- `data/raw/pizza_100.zip`
- `data/raw/pizza_250.zip`
- `checkpoints/gnn_subgraph_ranker_pizza_100_1hop_full/`
- `checkpoints/gnn_subgraph_ranker_pizza_100_2hop_full/`
- `checkpoints/gnn_subgraph_ranker_pizza_250_1hop_full/`
- `checkpoints/gnn_subgraph_ranker_pizza_250_2hop_full/`
- `outputs/full_results/pizza_100_1hop/`
- `outputs/full_results/pizza_100_2hop/`
- `outputs/full_results/pizza_250_1hop/`
- `outputs/full_results/pizza_250_2hop/`

Blockers before any physical batch: explicitly approved edits to `release_manifests/protection_rules.yaml`, `release_manifests/paper_original/index.yaml`, and `tests/reproducibility/test_original_paper_contracts.py`; confirmation that no external paper appendix actively reports Pizza; and a full per-file manifest/restore test.

### 13. ABORTED_HARD_PAIR_QUARANTINE

- `outputs/quarantine/aborted_20260909_sageqa_v2_hard_pair/`

This is not the successful final hard-pair checkpoint/result lineage.

### 14-15. DELETE_RECREATABLE

- first-party `__pycache__/` roots, including the five currently measured files
- `.venv/`
- empty scratch/output directories after process exit and ACL verification

These must never be mixed into a scientific backup/removal batch. `.venv/` was explicitly untouched in Phase 7C.

## Source closure summary

`KEEP_PAPER` centers on `experiments/run_experiments.py`, the original data/checkpoint readers, GraphSAGE trainer/evaluator, paper error analysis, dataset readers/evaluators, and GNN-RAG adapters.

`KEEP_THESIS` centers on the Cross-Encoder experiment and A40 runners, hard-pair trainer/launcher/audit, adaptive aggregation and policy fitting, canonical retrieval exporter, final reader/finalizers, successful baseline pipeline, complete-support oracle, and final stage-wise analysis.

`KEEP_BOTH` includes shared GraphSAGE architecture/training, text and ontology readers/evaluators, retrieval contracts, and GNN-RAG implementation/adapters.

The diagnostic, static-hard, listwise, A0/A3, rejected symbolic, and superseded V1 source files listed above are `BACKUP_AND_REMOVE`. A CLI is not retained merely because it remains executable.

Additional first-party source dispositions outside those explicit families are:

- `BACKUP_AND_REMOVE` with `EARLY_V1_PIPELINE`: `evaluation/evaluate_adaptive_support_aggregation.py`.
- `BACKUP_AND_REMOVE` as unreported baseline CLIs: `evaluation/eval_axiom_retriever_baseline.py` and `evaluation/eval_subgraph_baselines.py`.
- `BACKUP_AND_REMOVE` with `A0_A3_SUBMISSION_AND_REBUILT_GNN`: `models/dataset_adapters.py`, `models/familyowl_atomic_selector_v2.py`, `models/familyowl_compiler_v2.py`, `models/reasoning_ir.py`, `models/symbolic_executor.py`, `models/twowiki_sentence_compiler_v1.py`, and `training/learn_symbolic_composer_weights.py`.
- `BACKUP_AND_REMOVE` with miscellaneous development: `training/run_gnn_rag_walkthrough.py` and `utils/tokenizer.py`.
- `KEEP_REPRODUCIBILITY`: package initializers, `evaluation/check_thesis_final_paths.py`, `evaluation/validate_gold_free_candidate_pools.py`, `evaluation/test_generator_d_production_smoke.py`, and `utils/model_loader.py`.

Together with the exact retained-source records in `final_repository_keep.yaml`, these lists classify every currently inventoried first-party Python/shell source path. Cache bytecode beneath source roots remains `DELETE_RECREATABLE` rather than inheriting scientific KEEP.

## Test closure summary

Retain the reproducibility suite, adaptive aggregation tests, final answer/baseline tests, gold-free and complete-support contracts, hard-pair reservation test, clean GNN-RAG preparation test, production reader test, stage-wise test, evaluator/GNN tests, and current Generator-D/GNN-RAG integration tests. Deterministic ontology proof execution is covered by the retained reader, production-generation, and stage-wise contracts; the separate legacy symbolic-executor test belongs to the superseded A0/A3 family.

Remove tests only with their complete rejected families. Stale untracked tests importing already-absent implementation modules are not current reproduction contracts. `tests/reproducibility/test_original_paper_contracts.py` is retained but requires a future explicitly authorized scope correction because it still asserts all four unreported paper-era Pizza roots.

Exact `BACKUP_AND_REMOVE` test paths are:

- A0/A3 and superseded symbolic: `tests/test_2wiki_heldout_a3.py`, `tests/test_2wiki_hybrid_candidate_retrieval_audit.py`, `tests/test_2wiki_sentence_compiler_v1.py`, `tests/test_candidate_construction_audit.py`, `tests/test_familyowl_a4_proof_projection.py`, `tests/test_familyowl_architecture_extensions.py`, `tests/test_familyowl_semantic_first_reranking.py`, `tests/test_familyowl_semantic_labels.py`, `tests/test_final_architecture.py`, `tests/test_run_matched_original_a0_a3.py`, `tests/test_submission_readiness.py`, `tests/test_symbolic_candidate_diagnostics.py`, and `tests/test_symbolic_executor.py`.
- Removed or missing development helpers: `tests/test_compare_retrieval_runs.py`, `tests/test_create_gnn_dev_sample.py`, `tests/test_gnn_rag_alignment.py`, `tests/test_prepare_2wiki_reader_comparison.py`, `tests/test_probe_familyowl_compactness.py`, and `tests/test_probe_familyowl_representations.py`.
- Rejected candidate families: `tests/test_full_context_progressive_beam.py`, `tests/test_ontology_atomic_pool_experiment.py`, and `tests/test_size_balanced_candidate_composer.py`.
- Rejected objectives: `tests/test_listwise_soft_target_loss.py` and `tests/test_static_hard_negative_refinement.py`.

All other currently inventoried test paths appear as retained files/roots in `final_repository_keep.yaml`, including the experiment-local Cross-Encoder test through its retained source root.

## Checkpoint and output summary

Checkpoint KEEP is limited to six original-paper GraphSAGE roots, the ten-setting hard-pair root, the clean GNN-RAG root, and the final DistilBERT checkpoint inside the Cross-Encoder DEV bundle. All other checkpoint roots are removable or review-required as listed above.

Output KEEP is limited to the six paper-setting result roots and aggregate files; final Cross-Encoder DEV/policies/A40 TEST; hard-pair policies/training/TEST/e2e/audit; successful lexical/GNN-RAG/full-context bundles; canonical hard-pair retrieval export; canonical final answer bundle; complete-support/provenance roots; and final stage-wise analysis. No failed baseline or rejected experiment zoo is retained.

## Review-required items

1. `chapters/7_reasoner.tex` is absent from the local PhD workspace. The user-supplied result mapping was used as authority; a later provenance package should add the exact chapter source/hash.
2. `dbdbb507...` remains only the strongest original-paper source candidate. Ignored artifacts, environment, and hosted-reader lineage are not commit-bound.
3. `data/raw/family*` and `data/raw/owl2bench*` are not runtime dependencies, but exact processed-data regeneration and redistribution requirements are not manifest-bound.
4. `.worktrees/sageqa-original-eval-compat/` may contain unindexed compatibility evidence and was not classified for removal.
5. Current protection/index/test metadata overstates PAPER scope by including Pizza. It is a blocker to removal, not evidence that Pizza is scientifically retained.

## Projected final size and root count

The current accessible scientific working tree is 33,448,720,679 bytes when `.git`, `.venv`, `.worktrees`, and temp/ACL scratch are excluded. Subtracting only the measured scientific removal families yields a conservative projected 30,945,953,504 bytes (about 28.82 GiB). This is an upper bound because review-required raw/worktree payloads and smaller stale files remain unresolved.

`final_repository_keep.yaml` contains 136 logical retained path records for the current closure. This is not a physical top-level directory count because several entries intentionally retain exact files beneath otherwise removable families.

## Required removal order

1. Resolve the named review items and approve a literal metadata-migration allowlist where protection rules conflict with Phase 7C.
2. Back up and remove the one-file aborted local GTX attempt.
3. Back up and remove failed/incomplete GNN-RAG attempts while preserving successful siblings.
4. Remove rejected candidate, symbolic, and V1 diagnostic families in complete source/output/test units.
5. Remove bottleneck diagnostics except the retained `comparison.json` runtime input.
6. Remove listwise and static-hard families.
7. Remove miscellaneous development checkpoints and A0/A3 submission/rebuilt-GNN material.
8. Remove the early V1 pipeline only after all historical/live references are classified and redirected to retained provenance records.
9. Remove paper-era Pizza only after protection/index/test scope migration and external backup/restore proof.
10. Remove aborted quarantine evidence.
11. Handle bytecode and `.venv` separately as recreatable local state, never as a scientific batch.

No step above is authorized by this document. Each physical batch needs a new user-approved literal allowlist.
