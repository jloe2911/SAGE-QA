# Reproducibility test baseline

This baseline validates source and artifact contracts only. It does not train, rerank,
generate answers, call an API, alter evaluation semantics, or rewrite frozen artifacts.
The thesis-final source revision is `884480be53c554edae70d3b2d8e781847590aa69`.
The original-paper source status is `candidate_not_fully_proven` at
`dbdbb50708bdc6c686ef82518ec71c1d1bf55985`.

## Existing-test classification matrix

Every test file present at the Phase 3 start is included below. `Temp` means writes are
restricted to pytest fixtures. None of these tests requires network access. Torch tests
listed as CPU do not require a GPU.

| Generation | Test files | Dependencies | GPU | Writes | Frozen inputs | Routine level |
|---|---|---|---|---|---|---|
| THESIS_FINAL_TEST | `evaluation/test_generator_d_production_smoke.py`; `experiments/cross_encoder_reranking_dev_v1/test_experiment.py`; `tests/test_adaptive_support_aggregation.py`; `tests/test_adaptive_support_aggregation_v2.py`; `tests/test_final_manuscript_answer_generation.py`; `tests/test_gold_free_retrieval.py`; `tests/test_unified_evidence_graph_candidates.py` | Present | No | Temp in adaptive-v2 | Some | FAST focused / STANDARD |
| Oracle | `tests/test_gold_support_complete_oracle.py` | Present | No | No | Contract only | FAST focused / STANDARD |
| Thesis baseline | `tests/test_final_manuscript_baselines.py`; `tests/test_full_context_progressive_beam.py`; `tests/test_prepare_production_gnn_rag_clean.py` | Present | No | Temp | Some | STANDARD |
| Graph ablation | `tests/test_hard_pair_reservation.py`; `tests/test_production_test_answer_generation.py` | Present; CPU Torch | No | Temp in the broader generation fixture | Policy contracts | STANDARD focused |
| PAPER_ORIGINAL_TEST | `tests/test_evaluate_owl_qa_predictions.py`; `tests/test_gnn_rag_alignment.py`; `tests/test_gnn_subgraph_retriever.py`; `tests/test_run_matched_original_a0_a3.py` | Evaluator and matched runner present; GNN adapter/API dependencies stale | No | Temp | Historical/local | FULL; two archival failures excluded |
| Thesis superseded | `tests/test_2wiki_heldout_a3.py`; `tests/test_2wiki_sentence_compiler_v1.py`; `tests/test_final_architecture.py`; `tests/test_prepare_2wiki_reader_comparison.py`; `tests/test_submission_readiness.py`; `tests/test_symbolic_executor.py` | Three files fail collection; compiler 2/4 pass, readiness 1/4, executor 8/9 | No | No | A0/A3 bundle | Archival |
| Development, supported | `tests/test_ontology_atomic_pool_experiment.py`; `tests/test_size_balanced_candidate_composer.py`; `tests/test_static_hard_negative_refinement.py`; `tests/test_unified_kg_pipeline.py` | Present; CPU Torch in static-hard | No | No/Temp | No | FULL diagnostic |
| Development/superseded, broken | `tests/test_2wiki_hybrid_candidate_retrieval_audit.py`; `tests/test_candidate_construction_audit.py`; `tests/test_compare_retrieval_runs.py`; `tests/test_create_gnn_dev_sample.py`; `tests/test_familyowl_a4_proof_projection.py`; `tests/test_familyowl_architecture_extensions.py`; `tests/test_familyowl_semantic_first_reranking.py`; `tests/test_familyowl_semantic_labels.py`; `tests/test_listwise_soft_target_loss.py`; `tests/test_probe_familyowl_compactness.py`; `tests/test_probe_familyowl_representations.py`; `tests/test_symbolic_candidate_diagnostics.py` | Paired modules absent; listwise expectation disagrees with current disabled-listwise implementation | No | Temp only in sample test | No | Archival; excluded |

The unfiltered collection fails on `test_2wiki_heldout_a3.py`,
`test_final_architecture.py`, `test_gnn_rag_alignment.py`,
`test_gnn_subgraph_retriever.py`, `test_prepare_2wiki_reader_comparison.py`, and the eleven
collection-failing files in the final matrix row: 16 collection failures total. These are
recorded rather than repaired through current thesis code.

`tests/test_listwise_soft_target_loss.py` collects, but its explicit-KL assertion currently
expects twice the implemented value. The final hard-pair training contract keeps listwise
disabled, so this test is excluded as superseded rather than changing production loss code
or a frozen expectation.

An archival execution audit also records six runtime failures: two sentence-compiler tests
(an absent development sample manifest and a historical hash mismatch), three submission
readiness tests (removed notebooks/root documents), and one symbolic-executor explanation
wording assertion. The other 17 tests in that focused archival audit pass.

## Isolated Windows commands

Each command uses a fresh directory under the writable, ignored
`.tmp/reproducibility/` root and disables pytest's cache provider. It does not use or
clean any blocked pytest scratch directory or protected scientific root.

### FAST

```powershell
$base = New-Item -ItemType Directory -Force -Path .tmp/reproducibility
$bt = Join-Path $base.FullName ("sageqa-repro-fast-" + [guid]::NewGuid())
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $bt `
  tests/reproducibility --ignore=tests/reproducibility/test_original_paper_contracts.py
```

This checks release-index resolution, protected roots, the 66-entry Generator D manifest
identity and member sizes, small Generator D member hashes, critical main/oracle hashes,
manifest consistency, thesis-final routing and aggregation, the gold firewall, and oracle
separation. It does not hash the multi-gigabyte Generator D members.

### STANDARD

```powershell
$base = New-Item -ItemType Directory -Force -Path .tmp/reproducibility
$bt = Join-Path $base.FullName ("sageqa-repro-standard-" + [guid]::NewGuid())
$standard = @(
  "tests/reproducibility",
  "tests/test_adaptive_support_aggregation.py",
  "tests/test_adaptive_support_aggregation_v2.py",
  "tests/test_final_manuscript_answer_generation.py::test_frozen_configuration_set_is_exact",
  "tests/test_final_manuscript_answer_generation.py::test_cache_key_reuses_identical_inputs_across_method_labels",
  "tests/test_final_manuscript_answer_generation.py::test_persisted_preflight_has_required_exact_counts_and_no_calls",
  "tests/test_final_manuscript_answer_generation.py::test_persisted_reader_input_bundle_matches_freeze",
  "tests/test_final_manuscript_answer_generation.py::test_reader_input_bundle_excludes_cross_encoder_adaptive_and_empty_gold_support",
  "tests/test_gold_free_retrieval.py",
  "tests/test_gold_support_complete_oracle.py",
  "tests/test_final_manuscript_baselines.py",
  "tests/test_full_context_progressive_beam.py",
  "tests/test_prepare_production_gnn_rag_clean.py",
  "tests/test_hard_pair_reservation.py",
  "tests/test_production_test_answer_generation.py::test_generate_cli_defaults_to_eight_workers",
  "tests/test_production_test_answer_generation.py::test_default_ontology_deterministic_proof_is_identical_to_existing_generator",
  "tests/test_production_test_answer_generation.py::test_existing_answer_normalization_and_evaluation_semantics_are_preserved",
  "tests/test_unified_evidence_graph_candidates.py",
  "evaluation/test_generator_d_production_smoke.py",
  "experiments/cross_encoder_reranking_dev_v1/test_experiment.py"
)
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $bt `
  --ignore=tests/reproducibility/test_original_paper_contracts.py $standard
```

STANDARD adds supported thesis-final, baseline, graph-ablation, and import-level tests.

### FULL

```powershell
$base = New-Item -ItemType Directory -Force -Path .tmp/reproducibility
$bt = Join-Path $base.FullName ("sageqa-repro-full-" + [guid]::NewGuid())
$env:SAGEQA_FULL_ARTIFACT_HASHES = "1"
$full = @(
  "tests/reproducibility",
  "tests/test_adaptive_support_aggregation.py",
  "tests/test_adaptive_support_aggregation_v2.py",
  "tests/test_final_manuscript_answer_generation.py::test_frozen_configuration_set_is_exact",
  "tests/test_final_manuscript_answer_generation.py::test_cache_key_reuses_identical_inputs_across_method_labels",
  "tests/test_final_manuscript_answer_generation.py::test_persisted_preflight_has_required_exact_counts_and_no_calls",
  "tests/test_final_manuscript_answer_generation.py::test_persisted_reader_input_bundle_matches_freeze",
  "tests/test_final_manuscript_answer_generation.py::test_reader_input_bundle_excludes_cross_encoder_adaptive_and_empty_gold_support",
  "tests/test_gold_free_retrieval.py",
  "tests/test_gold_support_complete_oracle.py",
  "tests/test_final_manuscript_baselines.py",
  "tests/test_full_context_progressive_beam.py",
  "tests/test_prepare_production_gnn_rag_clean.py",
  "tests/test_hard_pair_reservation.py",
  "tests/test_production_test_answer_generation.py::test_generate_cli_defaults_to_eight_workers",
  "tests/test_production_test_answer_generation.py::test_default_ontology_deterministic_proof_is_identical_to_existing_generator",
  "tests/test_production_test_answer_generation.py::test_existing_answer_normalization_and_evaluation_semantics_are_preserved",
  "tests/test_unified_evidence_graph_candidates.py",
  "evaluation/test_generator_d_production_smoke.py",
  "experiments/cross_encoder_reranking_dev_v1/test_experiment.py",
  "tests/test_evaluate_owl_qa_predictions.py",
  "tests/test_run_matched_original_a0_a3.py",
  "tests/test_ontology_atomic_pool_experiment.py",
  "tests/test_size_balanced_candidate_composer.py",
  "tests/test_static_hard_negative_refinement.py",
  "tests/test_unified_kg_pipeline.py"
)
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $bt $full
Remove-Item Env:SAGEQA_FULL_ARTIFACT_HASHES
```

FULL adds original-paper source/worktree contracts, supported historical/development tests,
and byte-hashes all 66 Generator D members (about 16.9 GB). It still excludes the 16 known
archival collection failures and never invokes generation or an external service.

## `artifacts/manifest.json`

Classification: `THESIS_SUPERSEDED`. It is the manifest packaged at the root of
`dist/sageqa-submission-artifacts.zip`, matches the `sageqa_a0_a3_reader_v1` frozen
configuration, and is consumed by the older submission-readiness/reproduction path. Its
members are three development GraphSAGE checkpoints and three A0/A3 reader/policy outputs.
It has no path overlap with Generator D, thesis-final/oracle roots, or original-paper roots.
The manifest and its artifacts remain unchanged.

## Original-paper boundary

The historical contract checks Git objects at `dbdbb507`, the original runner, processed
dataset/checkpoint roots, `outputs/full_results`, evaluator/reader modules, and the optional
detached compatibility worktree. Exact reproduction remains unproven because there is no
commit-bound manifest for the ignored data, checkpoints, results, environment, or hosted
reader. Historical checks do not import or route through thesis-final validation logic.
