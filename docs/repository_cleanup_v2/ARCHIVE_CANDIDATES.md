# Archive candidates

SAFE_ARCHIVE means preserve content under a clearer status/location after manifest and reference updates. No move is authorized or performed.

## Artifact roots

| Current path | Bytes | Proposed class/destination | Required handling |
|---|---:|---|---|
| `checkpoints/development/` | 781,219,644 | `development_archive/checkpoints/` | Classify submission-frozen vs smoke/workbench first; preserve manifests. |
| `outputs/development_runs/` excluding four cross-encoder roots | 90,277,981 | `outputs/development_archive/{accepted_intermediate,rejected,exploratory}/` | Keep hard-pair policies together with graph-ablation lineage. |
| `outputs/development_diagnostics/` | 73,900,581 | `outputs/development_archive/diagnostics/` | Preserve final bottleneck conclusions and reader diagnostic freeze. |
| `outputs/diagnostics/` | 27,395,576 | `outputs/development_archive/{rejected,exploratory}/` | Pair each directory with its deciding report/script. |
| `outputs/final_model_development/` | 9,020,751 | `outputs/development_archive/rejected/static_hard/` | Preserve rejection status. |
| `outputs/quarantine/` | 20,047,392 | `outputs/development_archive/aborted/` | Preserve non-evidence/quarantine semantics. |

Total artifact-root estimate: 1,001,861,925 bytes.

## Superseded diagnostic source files

The following 26 files total 505,935 bytes. Move them only after imports, tests, docs, and launcher references are updated:

- `evaluation/audit_corrected_listwise_loss_scale.py`
- `evaluation/audit_hard_pair_reservation.py`
- `evaluation/audit_static_hard_negative_refinement.py`
- `evaluation/compare_current_beam_union.py`
- `evaluation/compare_full_context_progressive_beam.py`
- `evaluation/compare_ontology_atomic_pool_builders.py`
- `evaluation/compare_query_local_candidate_closure.py`
- `evaluation/compare_size_balanced_candidate_composer.py`
- `evaluation/compare_unified_evidence_graph_candidates.py`
- `evaluation/diagnose_current_clean_candidate_coverability.py`
- `evaluation/diagnose_generator_d_pair_supervision.py`
- `evaluation/diagnose_production_generator_d_causal_retrieval.py`
- `evaluation/diagnose_symbolic_mechanism_dev.py`
- `evaluation/diagnose_symbolic_validity_dev.py`
- `evaluation/diagnose_unified_ontology_reachability.py`
- `evaluation/evaluate_disagreement_union_dev.py`
- `evaluation/evaluate_generator_d_cross_branch_completion_dev.py`
- `evaluation/evaluate_ontology_proof_gate_dev.py`
- `evaluation/evaluate_production_dev_complementarity.py`
- `evaluation/evaluate_rrf_reranker_dev.py`
- `evaluation/evaluate_static_hard_refinement_dev.py`
- `evaluation/finalize_static_hard_refinement_dev.py`
- `evaluation/score_production_generator_d_causal_pairs.py`
- `evaluation/summarize_corrected_listwise_dev.py`
- `experiments/run_corrected_listwise_dev_grid.py`
- `training/run_static_hard_negative_refinement.py`

Combined archive estimate: 1,002,367,860 bytes (0.934 GiB).

## Archive manifest requirements

Each archive operation should record `source_path`, `destination_path`, protocol label, status (`REJECTED`, `ABORTED`, `SUPERSEDED`, `DIAGNOSTIC`, or `ACCEPTED_INTERMEDIATE`), byte/file counts, per-file SHA-256, source commit, direct references, and the post-move verification result. Keep a compatibility index at the old logical name when frozen lineage uses it.
