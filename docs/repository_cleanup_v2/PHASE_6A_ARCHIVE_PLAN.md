# Phase 6A development archive plan

## Decision and scope

No archive operation was executed. This plan is based on checkout `06c832022b39e149bf847a1109319992e9852690` on 2026-09-14. Git was clean before the three allowlisted planning files were written. No protected artifact, frozen manifest, source, test, config, ignore rule, or existing documentation was changed.

The normalized per-path authority is `PHASE_6A_ARCHIVE_DECISIONS.csv`. It contains 130 candidates: 55 artifact directories, 33 Python sources, 15 configs, one notebook, and 26 non-canonical or archival tests. Seven protected development-named roots are deliberately outside that candidate denominator and are listed below.

## Totals

| Recommendation | Paths | Bytes | Meaning |
|---|---:|---:|---|
| `KEEP_VISIBLE` | 26 | 33,566,232 | Active/shared, historical reproduction, or frozen-lineage material stays at its current path. |
| `ARCHIVE_INTERNAL` | 98 | 349,541,737 | Preserve under a status-oriented internal archive after a separately approved manifest batch. |
| `ARCHIVE_RELEASE` | 0 | 0 | No whole candidate is presently suitable for a public release archive without additional lineage/dependency decisions. |
| `DELETE_LATER` | 2 | 0 | Two empty, unreferenced directories only; deletion still requires a future literal allowlist. |
| `NEEDS_REVIEW` | 4 | 689,149,293 | Heterogeneous or frozen-referenced checkpoint/baseline payloads require human decisions. |
| **Total** | **130** | **1,072,257,262** | Lower-risk clarity is preferred over disk saving. |

Moving all `ARCHIVE_INTERNAL` artifact directories would relocate 45 of the 55 inspected artifact children (81.8%) and 348,932,036 artifact bytes. It could leave `outputs/diagnostics/`, `outputs/final_model_development/`, and `outputs/quarantine/` with no retained child after the separately approved empty-directory disposition. Source/config/test archival adds only 609,701 bytes but has substantially higher import and documentation risk.

The reference scan found at least 44 distinct files that would need path/import/documentation review across the full proposed program: 32 Python files and 12 documentation, manifest, or configuration files. This is a file count, not an occurrence count; it excludes references contained only inside the moved payload itself.

## Evidence profiles used by the decision table

All artifact roots are ignored/untracked because `outputs/` and `checkpoints/` are ignored. Tracked source has a Git-blob hash but is not thereby a frozen scientific artifact.

| Profile | Producing and consuming evidence | Manifest / release / thesis evidence | Self-contained and recreatable | Scientific meaning |
|---|---|---|---|---|
| P01 | Training or walkthrough CLI; no live external consumer found except named paired tools where recorded | No root manifest; no thesis/manuscript citation found | Checkpoint plus metrics is not independently recreatable without code/config/data | Yes; exploratory/rejected model evidence |
| P02 | No files and no inbound reference | No manifest or lineage reference | Empty directory is trivially recreatable | No preserved scientific content |
| P03 | Historical A0/A3 symbolic workflow | `artifacts/manifest.json` hash-locks three descendants; release classification is thesis-superseded | Heterogeneous and not self-contained | Yes; **BLOCKED_BY_FROZEN_REFERENCE** |
| P04 | Multiple workbench experiments | Mentioned only in cleanup documentation | Heterogeneous; no root manifest | Yes; needs subrun-level ownership |
| P05 | Named evaluation/diagnostic CLI produces the directory; some later diagnostics consume it | No complete root manifest unless stated; not directly cited in thesis/result map | Usually paired with current source and frozen inputs but not self-contained | Yes; negative or intermediate evidence |
| P06 | V1 retrieval/policy evaluators consume these roots | `release_manifests/thesis_superseded/index.yaml` is frozen; two roots have complete per-file hashes | Self-contained policies; reproducible only with frozen inputs | Yes; valid superseded thesis lineage |
| P07 | `generation/run_dev_reader_diagnostic.py` or output-local analysis source | Complete output hash manifest; DEV diagnostic not cited as canonical thesis result | Substantially self-contained result; regeneration may require provider or frozen inputs | Yes |
| P08 | Consumed by `evaluation/export_manuscript_retrieval_results.py` | Upstream comparison for a frozen manuscript export | Not independently recreatable without frozen inputs | Yes; must stay visible |
| P09 | Paired one-off diagnostic source when present | No root manifest and no frozen/release inbound reference found | Result is readable but generally not independently regenerable | Yes; diagnostic or negative result |
| P10 | Static-hard and hard-negative refinement scripts | Protocol/invariance/artifact manifests as noted; not a canonical release root | Result and decision are mostly self-describing | Yes; rejected/aborted result |
| P11 | Aborted hard-pair training | Quarantine manifest explicitly forbids scientific interpretation | Self-contained chronology but not a valid result | Yes; provenance only |
| P12/P13 | Historical native GNN-RAG attempts | Inside the baseline component parent but not baseline evidence; no own manifest | Duplicate payload plus logs; partial self-containment | Yes; payload-retention policy unresolved |
| S01 | Standalone development CLI; outbound imports are current shared training/data/evaluation modules; inbound refs are paired CLIs/tests/docs | No release or frozen-manifest ownership | Git-recoverable at the source revision; output recreation may be expensive | Yes |
| S02 | Imported by `training/run_sageqa_v2_final.py` | Graph-ablation ownership | Active dependency | Yes; keep visible |
| S03 | Imported by `tests/test_full_context_progressive_beam.py` in STANDARD | Thesis-baseline ownership | Active tested helper | Yes; keep visible |
| S04 | Rejected FamilyOWL selector/compiler; paired archival test | Classified development in hidden/untracked index | Source is locally complete but its evaluator is missing | Yes |
| S05 | Shared symbolic IR/adapters/compiler/executor used across historical configs/tests | Classified thesis-superseded and required to interpret frozen historical configurations | Not safely separable yet | Yes |
| S06 | Produces the GNN-RAG walkthrough checkpoints/logs | Development index only | Paired source and outputs; upstream environment still required | Yes |
| C01 | Development evaluator/test config | Hidden/untracked classification; no frozen release index | Some paired evaluators are absent | Yes |
| C02 | Historical A0/A3 or compiler config | Thesis-superseded classification; architecture identity is used by `artifacts/manifest.json` | Requires historical modules/assets | Yes; keep visible |
| N01 | Notebook-local exploratory analysis | No inbound ref outside cleanup docs | Environment/data dependent | Yes |
| T01 | Paper-original contract tests; outbound imports identify historical evaluator/GNN/runner dependencies | Paper reproduction role; no thesis-final use | Two have stale/missing dependencies | Yes; retain and restore dependencies |
| T02 | Thesis-superseded tests | Historical configs/modules; known collection/runtime failures documented in test baseline | Not currently reproducible as a suite | Yes; keep failures explicit |
| T03 | Supported development tests paired with development source | FULL diagnostic set | Reproducible locally when dependencies exist | Yes; archive only with source |
| T04 | Broken development tests; outbound imports name missing historical modules, except listwise mismatch | Archival test baseline documents exact failures | Not currently reproducible; restore identified source rather than rewrite against thesis code | Yes |

For a source row, “outbound references” means its import graph; for an artifact row it means embedded paths/manifests. Exact direct inbound files found during the scan are reflected in the profile and blocker columns; the Phase 6B manifest for each batch must enumerate every occurrence before execution.

## Exact reference and move-impact ledger

The following are the non-empty external references found by literal basename/path search over Python, shell, test, config, documentation, release YAML, and manifest JSON. A decision-table candidate not listed here had no external hit in that search. Cleanup-audit self-references are omitted from this ledger but remain among the 44 update-review files.

| Candidate | External producer / consumer / lineage reference | Required future update or decision |
|---|---|---|
| `checkpoints/development/gnn_rebuilt/` | `artifacts/manifest.json` hash-locks three `familyowl_semantic_pairwise_seed*` checkpoints | Frozen historical manifest cannot be rewritten; blocked. |
| `checkpoints/development/production_generator_d_v1_listwise_corrected_dev/` | corrected-listwise grid, audit, summarizer, and `verify_listwise_zero_baseline.py` | Move only with paired outputs and four source path updates. |
| `outputs/development_runs/current_beam_union_v1/` | `compare_current_beam_union.py` | Update CLI default/output path. |
| `.../current_clean_candidate_coverability_v1/` | `diagnose_current_clean_candidate_coverability.py` | Update CLI path. |
| `.../full_context_progressive_beam_v1/` | `compare_full_context_progressive_beam.py` | Update helper default; keep helper visible for STANDARD. |
| `.../generator_d_pair_supervision_v1/` | `diagnose_generator_d_pair_supervision.py` | Update CLI path. |
| `.../ontology_atomic_pool_graph_expansion_v1/` | `compare_ontology_atomic_pool_builders.py` | Move source/test/output together. |
| `.../ontology_semantic_sufficiency_v1/` | `posthoc_ontology_semantic_sufficiency.py` | Update CLI path. |
| `.../production_generator_d_v1_adaptive_k/` | six evaluation CLIs and frozen `release_manifests/thesis_superseded/index.yaml` | Keep visible; blocked by frozen reference. |
| `.../production_generator_d_v1_gnn_adaptive_k/` | three evaluation CLIs and frozen `release_manifests/thesis_superseded/index.yaml` | Keep visible; blocked by frozen reference. |
| `.../production_generator_d_v1_k_sensitivity/` | ten diagnostic/fitting/verifier CLIs | Keep visible as shared frozen-lineage input. |
| `.../production_generator_d_v1_listwise_corrected_dev/` | corrected-listwise grid, verifier, audit, and summarizer | Move as one aborted batch with checkpoint/log/diagnostic components. |
| `.../protected_query_anchor_comparison_v1/` | unified-evidence comparison and semantic-sufficiency posthoc | Update both paths. |
| `.../query_local_candidate_closure_v1/` | `evidence_graph_candidates.py` plus two diagnostic CLIs | Update three paths; do not move the active shared builder. |
| `.../size_balanced_structural_composer_v1/` | `compare_size_balanced_candidate_composer.py` | Move with source/test. |
| `.../unified_ontology_reachability_diagnostic_v1/` | `diagnose_unified_ontology_reachability.py` | Update CLI path. |
| `outputs/development_diagnostics/final_bottleneck_analysis/` | `run_dev_reader_diagnostic.py` and prior cleanup planning records | Update producer and non-frozen docs only. |
| `.../final_bottleneck_reader_diagnostic/` | `run_dev_reader_diagnostic.py` | Update producer path. |
| `.../final_old_vs_cross_encoder_analysis/` | `export_manuscript_retrieval_results.py` | Keep visible as an upstream frozen-export input. |
| `outputs/diagnostics/production_generator_d_v1_causal_retrieval_diagnosis/` | causal scorer/evaluator and static-hard/symbolic diagnostics | Move only after four source paths are updated. |
| `.../production_generator_d_v1_complementarity_dev/` | complementarity evaluator | Update CLI path. |
| `.../production_generator_d_v1_cross_branch_completion_dev/` | cross-branch evaluator | Update CLI path. |
| `.../production_generator_d_v1_disagreement_union_dev/` | disagreement evaluator | Update CLI path. |
| `.../production_generator_d_v1_listwise_corrected_dev/` | corrected-listwise grid/verifier/audit/summarizer | Same homogeneous aborted batch as checkpoint/log output. |
| `.../production_generator_d_v1_mechanism_decision/` | cross-branch evaluator | Update input path. |
| `.../production_generator_d_v1_ontology_proof_gate_dev/` | proof-gate evaluator | Update CLI path. |
| `.../production_generator_d_v1_rrf_dev/` | RRF evaluator | Update CLI path. |
| `.../production_generator_d_v1_symbolic_mechanism_dev/` | symbolic-mechanism, symbolic-validity, and proof-gate CLIs | Update three paths. |
| `.../production_generator_d_v1_symbolic_validity_diagnosis/` | symbolic-validity CLI | Update path. |
| `outputs/final_model_development/production_generator_d_static_hard_v1*` | static-hard trainer, auditor, evaluator, and finalizer | Treat all three roots and paired source/test as one rejected family. |
| `outputs/diagnostics/cross_repo_methodology_transfer_audit/` | none | Pilot requires only non-frozen archive-index/docs updates. |

Functional Python inbound references beyond cleanup documentation are also exact: `audit_hard_pair_reservation.py` is imported by `training/run_sageqa_v2_final.py`; `compare_full_context_progressive_beam.py` by its test and `compare_current_beam_union.py`; `compare_ontology_atomic_pool_builders.py`, `compare_size_balanced_candidate_composer.py`, and `run_static_hard_negative_refinement.py` by their paired tests; `compare_query_local_candidate_closure.py` by the pair-supervision diagnostic; and `compare_unified_evidence_graph_candidates.py` by query-closure, pair-supervision, reachability, and semantic-sufficiency CLIs. Other S01 CLIs have no functional inbound import and remain scientifically owned as standalone entry points.

Outbound import groups are: candidate-building CLIs -> `data.build_subgraph_training_data`, text/2Wiki builders, `retrieval_contracts`, `evidence_graph_candidates`, `text_kg_constructor`, and `validate_gold_free_candidate_pools`; ranking diagnostics -> `train_gnn_subgraph_retriever`, `run_production_dev_k_sensitivity`, and GNN model/evaluator helpers; static-hard source -> the active GNN trainer/model plus adaptive-support helpers; rejected FamilyOWL modules -> `dataset_adapters` and `reasoning_ir`; historical symbolic modules -> each other in the IR/adapter/compiler/executor chain. Those outbound dependencies stay visible; archiving a CLI does not authorize moving them.

## DO_NOT_ARCHIVE_DESPITE_NAME

- `outputs/development_runs/question_candidate_cross_encoder_v1/` — protected final-thesis checkpoint and DEV bundle.
- `outputs/development_runs/question_candidate_cross_encoder_v1_adaptive_input/` — protected frozen policy input.
- `outputs/development_runs/question_candidate_cross_encoder_v1_cross_encoder_adaptive_k/` — protected final-thesis policy.
- `outputs/development_runs/question_candidate_cross_encoder_v1_final_sageqa_adaptive_k/` — protected final SAGE-QA policy.
- `outputs/development_runs/production_generator_d_v2_hard_pair_gnn_only_adaptive_k/`, `..._k_sensitivity_merged/`, and `..._sageqa_final_adaptive_k/` — protected graph-ablation lineage.
- `outputs/development_runs/production_generator_d_v1_adaptive_k/`, `..._gnn_adaptive_k/`, and `..._k_sensitivity/` — valid frozen thesis-superseded lineage; the first two are named by the frozen release index.
- `outputs/development_diagnostics/final_old_vs_cross_encoder_analysis/` — live input to the frozen manuscript retrieval exporter.
- `outputs/final_results/production_generator_d_v1_test_baselines/gnn_rag/` — valid baseline result with ten output-local checkpoint copies required for self-contained runs.
- The five zero-byte JSONL findings under `outputs/audits/final_test_evaluation_audit/` — explicit zero-result scientific findings, not empty garbage.
- `evaluation/audit_hard_pair_reservation.py` — imported by the active graph-ablation runner.
- `evaluation/compare_full_context_progressive_beam.py` — imported by a STANDARD baseline test.
- `models/dataset_adapters.py`, `models/reasoning_ir.py`, `models/symbolic_executor.py`, and `models/twowiki_sentence_compiler_v1.py` — shared historical source required by thesis-superseded tests/configs.
- The two failed/incomplete GNN-RAG directories — not final evidence, but they stay pending a payload-retention decision because their logs establish chronology.

## Source ownership findings

Of 33 audited Python files: 27 are `DEVELOPMENT_ONLY` and proposed for `ARCHIVE_INTERNAL`; one is `GRAPH_ABLATION` (`audit_hard_pair_reservation.py`); one is `BASELINE` (`compare_full_context_progressive_beam.py`); and four are `SUPERSEDED_ONLY` shared symbolic sources. No audited file was assigned `THESIS_FINAL`, `ORACLE`, or `PAPER_HISTORICAL`. Standalone CLIs were treated as owned source despite zero Python inbound imports. The source rows identify the exact ownership and destination.

The archive order must respect source dependencies: outputs first, then paired tests/configs, then leaf CLIs, and only then shared development helpers. No source move is safe if it would make STANDARD fail or require a compatibility wrapper (wrappers are outside this phase and would need separate approval).

## Test ownership findings

The 26 non-canonical/archival tests are classified individually in the CSV:

- 4 paper-original tests: `KEEP_AND_RESTORE_DEPENDENCY`.
- 6 thesis-superseded tests: 3 `KEEP_AND_RESTORE_DEPENDENCY` and 3 `KEEP_AS_DOCUMENTED_FAILURE`.
- 4 supported development tests: `ARCHIVE_WITH_SOURCE` in their eventual homogeneous source batches.
- 12 broken development tests: 10 `KEEP_AND_RESTORE_DEPENDENCY`, 1 `ARCHIVE_WITH_SOURCE`, and 1 `KEEP_AS_DOCUMENTED_FAILURE`. The exact row-level actions are authoritative; notably the listwise KL mismatch must not be rewritten against current training code.

The broken-test action is future intent only. No module was restored, no expectation changed, and no test was moved.

## Failed and incomplete run policy

- `FAILED`: execution reached a scientifically interpretable failure condition. Preserve logs, exact inputs/identity, and the failure statement; archive if non-canonical.
- `INCOMPLETE`: expected outputs are missing and no success claim is valid. Preserve when it explains chronology; do not present it as final evidence.
- `ABORTED`: deliberately stopped before completion (for safety, compute, methodology, or operator decision). Preserve the stop reason and partial-output warning.
- `REJECTED`: completed enough to support a negative scientific decision. Preserve results, decision criteria, producing source, and manifests.
- `EXPLORATORY`: hypothesis-generating work without confirmatory standing. Archive internally unless it later receives a formal lineage role.
- `SUPERSEDED`: previously valid work replaced by a later generation. Preserve complete provenance and never equate it with failure.

Policy A (informative failed/rejected) is preserved or archived; B (partial attempt superseded by success) is normally archived away from final evidence; C (runtime garbage with no scientific content) alone may be `DELETE_LATER`; D (explicit zero-result audits) is always preserved. The two zero-byte directories meet C on present evidence. Zero-byte scientific files do not.

## Proposed archive structure

```text
archive/
  development/
    accepted_intermediate/
    diagnostics/
    exploratory/
    rejected/
  superseded/
  aborted/
  failed_runs/
```

Within a status directory, retain only shallow functional suffixes such as `source/`, `tests/`, `config/`, `checkpoints/`, or the original run basename. Status communicates the archival decision; the existing scientific protocol remains in the move manifest.

## Proposed move batches

1. `batch_01_cross_repo_methodology_transfer_audit`: seven-file rejected diagnostic; exact pilot manifest supplied; no exact external inbound reference.
2. Other small, unmanifested rejected diagnostics under `outputs/diagnostics/`, one directory per manifest.
3. Manifested static-hard rejected outputs under `outputs/final_model_development/` together with verified source references.
4. The quarantined aborted hard-pair run, preserving its quarantine manifest verbatim.
5. Complete development diagnostics, excluding `final_old_vs_cross_encoder_analysis`.
6. Rejected/exploratory `outputs/development_runs` roots, excluding every protected or frozen-lineage root.
7. Small development checkpoints, one homogeneous family at a time.
8. Development configs/tests/source, dependency-leaf batches only after import and FULL/archival-test decisions.
9. `gnn_rebuilt`, workbench, and failed/incomplete baseline payloads only after human ownership decisions; they are not presently executable batches.

Prefer one manifest per homogeneous status and ownership group. Every manifest uses the schema exemplified by `archive_manifests/batch_01_cross_repo_methodology_transfer_audit.yaml`, has `execute: false`, records literal source/destination, expected type/count/bytes, per-file SHA-256, reference updates, rollback paths, scientific class, and status.

## Exact Phase 6B pilot

The single safest future batch is `outputs/diagnostics/cross_repo_methodology_transfer_audit/` to `archive/development/rejected/cross_repo_methodology_transfer_audit/`. It is 7 files / 51,049 bytes, is ignored/untracked, has no exact inbound reference outside itself, has no frozen or release-manifest reference, contains a clear negative decision, and now has complete per-file SHA-256s in the planning manifest. It cannot affect FAST or STANDARD by import. Phase 6B must still approve the two literal paths and satisfy every readiness gate.

## Frozen-reference blockers

- `checkpoints/development/gnn_rebuilt/`: three descendant checkpoints are byte/hash-bound by `artifacts/manifest.json`; moving the root would require rewriting that historical manifest. `BLOCKED_BY_FROZEN_REFERENCE`.
- `outputs/development_runs/production_generator_d_v1_adaptive_k/` and `..._gnn_adaptive_k/`: named by frozen `release_manifests/thesis_superseded/index.yaml` and hash-manifested locally. `BLOCKED_BY_FROZEN_REFERENCE`.
- The seven protected cross-encoder/hard-pair DEV roots are excluded from archive candidacy by `release_manifests/protection_rules.yaml`.
- Any batch that discovers an output-local or release manifest with the old literal path becomes blocked; frozen files are not rewritten.

## Phase 6B readiness gates

Every future batch must satisfy all of these immediately before execution:

1. Git status is clean.
2. Phase 6A and all prior phases are committed.
3. FAST passes in an isolated writable temp root.
4. STANDARD passes where the moved paths can affect supported code/tests.
5. Original-paper contracts pass for any source/test/layout change.
6. Every source file SHA-256 is captured.
7. The literal destination is absent.
8. All Python, shell, test, config, manifest, documentation, and ignore/exclude references are enumerated.
9. Frozen references are checked; any required frozen edit blocks the move.
10. A rollback operation with literal source/destination is generated.
11. The user approves the exact literal source and destination paths.
12. After moving, source file count is zero.
13. Destination file count, total bytes, and every SHA-256 equal the source snapshot.
14. A post-move reference search has no unintended old-path hit.
15. Relevant tests pass after the move.

Additional gates: no concurrent writer, destination tracking policy decided, backup/restore location established for ignored artifacts, and `execute` remains false until the separately authorized executor validates the manifest.

## Remaining human decisions and readiness conclusion

Human decisions remain for: public-vs-internal retention of negative results; whether `gnn_rebuilt` may ever move despite the A0/A3 manifest; subrun ownership inside `workbench`; full-payload versus log/manifest-only retention for the two GNN-RAG failures; whether ignored historical sources/tests will be tracked; and whether an externally archived release copy exists.

**Phase 6B is not yet safe to begin.** The pilot is technically well scoped, but the Phase 6A files must be reviewed/committed, FAST/STANDARD/original-paper gates must be run immediately before movement, the destination/tracking and rollback locations must be approved, and literal move authorization has not been given.

## Phase 6A validation

- Decision CSV: 130 rows; recommendation counts and byte sums exactly match the totals above.
- Pilot manifest: YAML parsed; `execute` is Boolean false; 7 files and 51,049 bytes match; every current SHA-256 matches.
- FAST: 28 passed, 1 expected FULL-only hash skip.
- STANDARD: 123 passed, 1 expected FULL-only hash skip.
- Original-paper contracts plus supported paper tests: 10 passed.
- `git diff --check`: passed.

These results validate the plan snapshot; they do not replace the mandatory immediate pre-move rerun after the plan is reviewed and committed.

Final `git diff --stat` and `git status --short` are reported in the task handoff after validation.
