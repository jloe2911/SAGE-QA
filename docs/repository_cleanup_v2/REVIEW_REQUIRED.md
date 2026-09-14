# Manual review required

No operation should be generated for these items until the named decision is recorded.

| Item | Evidence | Decision needed |
|---|---|---|
| `.worktrees/sageqa-original-eval-compat/` | 112,370,538 bytes; registered detached compatibility checkout | Is its commit the authoritative original evaluator? Tag/archive it before any `git worktree remove`. |
| `artifacts/step_01_context_to_kg/wikidata_cache/` | Approximately 36.5 MB and may encode costly external queries | Determine protocol role, privacy, and redistribution/license status. |
| `dist/sageqa-submission-artifacts.zip` | 52,118,416 bytes; SHA-256 recorded in the prior audit; separate A0/A3 generation | Identify the exact manuscript/submission it supports and whether a durable external copy exists. |
| `checkpoints/production_generator_d_v1/` | 181,578,675 bytes; referenced by diagnostics/baseline protections | Keep as thesis development, archive, or include in graph-ablation bundle? Redirect references first. |
| `checkpoints/production_generator_d_static_hard_v1/` | 181,075,730 bytes; rejected mechanism | Decide whether negative-result reproducibility warrants checkpoint publication or only manifest/metrics retention. |
| Generated `third_party/GNN-RAG/{gnn/data,gnn/checkpoint,llm/results}` | About 1.67 GB; contains major exact-duplicate groups | Decide canonical copy and license; prove clean download/adapter flow before deduplication. |
| Failed/incomplete GNN-RAG native-run directories | Explicit failure chronology inside a final-results parent | Retain full payload or archive only logs/manifests? Never merge with successful results. |
| `.tmp/distilbert_portability_preflight/` | 4,322-byte preflight manifest, no live reference found | Copy into canonical cross-encoder provenance or discard after verifying equivalent frozen preflight evidence? |
| 23 ignored/untracked tests | 18 missing-import statements in the historical subset; one final test is untracked | Select supported final/original tests; restore missing modules or archive each historical test. |
| `.git/objects/` | 2,620,233,531 bytes | After tags/backups/worktree decision, measure reachability and use Git-native maintenance only. |
| `.env` | Local ignored credential/config file; values not inspected | Keep local, rotate if necessary, and ensure no release/manifest captures it. |
| Frozen absolute paths | Present in lineage/per-example records | Preserve bytes; define release-side relative mappings rather than edits. |

## Broken/dead entry-point review

Static parsing found no syntax failures, but these first-party modules imported by tests are absent:

- `evaluation.evaluate_2wiki_a0_a3_heldout`
- `evaluation.audit_2wiki_hybrid_candidate_retrieval`
- `evaluation.audit_familyowl_candidate_construction`
- `evaluation.compare_retrieval_runs`
- `data_processing.create_gnn_dev_sample`
- `evaluation.evaluate_familyowl_a4_proof_projection`
- `evaluation.evaluate_familyowl_architecture_extensions`
- `evaluation.evaluate_familyowl_semantic_first_reranking`
- `evaluation.audit_familyowl_semantic_labels`
- `evaluation.familyowl_semantic_labels`
- `evaluation.run_final_architecture_walkthroughs`
- `data_processing.gnn_rag_adapter_utils`
- `evaluation.convert_gnn_rag_retrieval`
- `evaluation.prepare_2wiki_reader_comparison`
- `evaluation.probe_familyowl_compactness`
- `evaluation.probe_familyowl_representations`
- `evaluation.diagnose_symbolic_candidates`
- `evaluation.evaluate_familyowl_symbolic_reranking`

The affected tests are historical/development-heavy. Do not restore guessed implementations. Recover their source from an identified commit or archive the tests with that commit reference.

## Quality review

Ruff 0.15.1 reported 41 unused imports. Fix them only in a separate source-quality change after final code hashes are frozen. Unused imports are not sufficient evidence for deletion or behavioral deadness.

## Documentation review

The root README still presents the original runner as canonical. A human should approve:

1. the exact final architecture wording;
2. leakage-affected artifact labels;
3. whether hard-pair results are called “graph ablation” or “previous thesis pipeline”;
4. public availability/licensing of raw benchmarks and Wikidata caches;
5. DOI/GitHub Release layout and which large bundles are public.
