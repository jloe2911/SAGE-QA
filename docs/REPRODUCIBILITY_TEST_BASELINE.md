# Reproducibility test baseline

This baseline validates source and artifact contracts only. It does not train, rerank,
generate answers, call an API, alter evaluation semantics, or rewrite frozen artifacts.
The scientific thesis-source baseline is `884480be53c554edae70d3b2d8e781847590aa69`.
The original-paper source status is `candidate_not_fully_proven` at
`dbdbb50708bdc6c686ef82518ec71c1d1bf55985`.

The retained baselines are FAST `27 passed, 1 skipped`, STANDARD
`119 passed, 1 skipped`, original-paper supported `4 passed, 1 skipped`, and stage-wise
`6 passed`. The unfiltered retained collection contains `142 tests`.

## Final retained-test matrix

`Temp` means writes are restricted to pytest fixtures. None of these tests requires
network access. Torch tests listed as CPU do not require a GPU.

| Generation | Test files | Dependencies | GPU | Writes | Frozen inputs | Routine level |
|---|---|---|---|---|---|---|
| THESIS_FINAL_TEST | `evaluation/test_generator_d_production_smoke.py`; `experiments/cross_encoder_reranking_dev_v1/test_experiment.py`; `tests/test_adaptive_support_aggregation.py`; `tests/test_adaptive_support_aggregation_v2.py`; `tests/test_final_manuscript_answer_generation.py`; `tests/test_gold_free_retrieval.py`; `tests/test_unified_evidence_graph_candidates.py` | Present | No | Temp in adaptive-v2 | Some | FAST focused / STANDARD |
| Oracle | `tests/test_gold_support_complete_oracle.py` | Present | No | No | Contract only | FAST focused / STANDARD |
| Thesis baseline | `tests/test_final_manuscript_baselines.py`; `tests/test_prepare_production_gnn_rag_clean.py` | Present | No | Temp | Some | STANDARD |
| Graph ablation | `tests/test_hard_pair_reservation.py`; `tests/test_production_test_answer_generation.py` | Present; CPU Torch | No | Temp in the broader generation fixture | Policy contracts | STANDARD focused |
| PAPER_ORIGINAL_TEST | `tests/reproducibility/test_original_paper_contracts.py`; `tests/test_evaluate_owl_qa_predictions.py` | Present | No | Temp | Historical/local | FULL / focused paper gate |
| Shared integration | `tests/test_unified_kg_pipeline.py` | Present | No | Temp | No | FULL |

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
  "tests/test_prepare_production_gnn_rag_clean.py",
  "tests/test_hard_pair_reservation.py",
  "tests/test_production_test_answer_generation.py::test_generate_cli_defaults_to_eight_workers",
  "tests/test_production_test_answer_generation.py::test_default_ontology_deterministic_proof_is_identical_to_existing_generator",
  "tests/test_production_test_answer_generation.py::test_existing_answer_normalization_and_evaluation_semantics_are_preserved",
  "tests/test_unified_evidence_graph_candidates.py",
  "evaluation/test_generator_d_production_smoke.py",
  "experiments/cross_encoder_reranking_dev_v1/test_experiment.py",
  "tests/test_evaluate_owl_qa_predictions.py",
  "tests/test_unified_kg_pipeline.py"
)
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $bt $full
Remove-Item Env:SAGEQA_FULL_ARTIFACT_HASHES
```

FULL adds retained original-paper/shared integration contracts and byte-hashes all 66
Generator D members (about 16.9 GB). It never invokes generation or an external service.

## Original-paper boundary

The historical contract checks Git objects at `dbdbb507`, the original runner, processed
dataset/checkpoint roots, `outputs/full_results`, and evaluator/reader modules. Exact
reproduction remains unproven because there is no commit-bound manifest for the ignored
data, checkpoints, results, environment, or hosted reader. Historical checks do not
import or route through thesis-final validation logic.
