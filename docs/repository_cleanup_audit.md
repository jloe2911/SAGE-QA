# SAGE-QA repository cleanup audit

**Audit date:** 2026-09-10 (Europe/Luxembourg)  
**Mode:** read-only repository inspection, except for creation of this report  
**Repository root:** `C:\Users\julie\github\PhD\SAGE-QA`  
**Current Git branch/commit:** `main`, `20c3ca8aedf1cb8656395e90dccc1adbd18bdde4`

## Executive decision

Do not clean, reorganize, package, or publish the repository yet. The final
manuscript baseline process was still active during this audit. At 18:46 local
time, two Python processes started at 18:28 were present and
`outputs/final_results/final_manuscript_baselines_test_end_to_end/predictions.jsonl`
was still growing (14,266,114 bytes and 3,903 complete JSONL rows at an earlier
count). That directory has a valid input freeze but does **not** yet have a
generation freeze, metrics, lineage metadata, evaluation audit, or a final
artifact manifest. Its sizes and hashes in this report are therefore only a
transient snapshot and must not be treated as final results.

The already completed hard-pair thesis retrieval and GNN/SAGE end-to-end runs
are internally well protected: their manifests were rechecked during this
audit and all ten listed files matched their recorded sizes and SHA-256 hashes.
The independent final-evaluation audit also records zero per-example and zero
aggregate-metric mismatches. These artifacts must remain byte-for-byte intact.

The public-release structure should make the leakage-free Generator D plus
hard-pair pipeline the default, while retaining a clearly labelled
`paper_original` protocol. The original-paper checkpoints, processed inputs,
and results are historical reproduction artifacts, not cleanup waste.

The current checkout is not publication-ready:

- Git reports 118 worktree entries: 115 modified tracked files and 3 untracked
  files.
- The three untracked files are thesis-critical:
  `evaluation/audit_final_test_evaluation.py`,
  `generation/run_final_manuscript_baselines.py`, and
  `generation/run_production_test_answer_generation.py`.
- `tests/`, `checkpoints/`, `outputs/`, and the Generator D corpus are ignored
  by `.gitignore`; no tests are currently tracked.
- The root README still presents the original, leakage-affected pipeline as the
  main reproduction route and does not identify the hard-pair thesis pipeline.
- There is no `configs/` directory and no standalone Generator D corpus
  manifest at the corpus root.
- The working environment differs from `requirements.txt`: the frozen run used
  `torch==2.6.0+cu124`, while the file requests `torch==2.11.0`. Several direct
  runtime dependencies are not declared.

## Audit scope and limitations

The recursive inventory found 62,286 accessible files totalling
**41,308,716,312 bytes (38.47 GiB)** at the initial snapshot. This includes
`.git`, `.venv`, and a nested worktree; it is not the size of a proposed public
release. The count is a lower bound because Windows denied traversal of:

- `.test_tmp_adaptive_k/`
- `.tmp/pytest-static-hard/`
- `.tmp/pytest_answer_generation/`
- `.tmp/pytest_answer_generation_final/`
- `.tmp/pytest_answer_generation_resume/`
- `.tmp_test_answer_generation/`
- `pytest-of-julie/`
- `outputs/audits/final_test_evaluation_audit/pytest_tmp/`

The live baseline also continued to grow after the initial size snapshot. No
API was called, no production experiment or test was run, and no existing
artifact was written, moved, renamed, or deleted.

The audit used current filesystem contents, Git metadata, manifests, source
references, and saved run lineage. Age alone was not used as evidence that a
file is obsolete. Recommendations marked “safe” are still for execution only
after the live baseline is complete and frozen.

## 1. Recursive inventory and classification

| Class | Current paths | Assessment |
|---|---|---|
| A. Final-thesis source | `data_processing/evidence_graph_candidates.py`, `data_processing/retrieval_contracts.py`, current text/ontology builders, `training/train_gnn_subgraph_retriever.py`, `training/run_sageqa_v2_final.py`, `scripts/run_sageqa_hard_pair_v2_all.sh`, `evaluation/run_production_dev_k_sensitivity.py`, `evaluation/fit_production_dev_adaptive_k.py`, `evaluation/run_production_test_retrieval.py`, `generation/run_production_test_answer_generation.py`, `generation/run_final_manuscript_baselines.py`, `evaluation/audit_final_test_evaluation.py` | Required. Several are modified and the last three Python entry points are untracked. Freeze exact code hashes and commit an allowlisted snapshot before release. |
| B. Original-paper-only source | Primarily `experiments/run_experiments.py` and its original data-building/orchestration behavior; historical GNN-RAG conversion/finalization entry points | Retain under a tagged historical source snapshot or `legacy/paper_original/`. Current builders have evolved, so current HEAD alone is not evidence of byte-identical paper reproduction. |
| C. Shared source | `models/gnn_subgraph_retriever.py`, `models/symbolic_composer.py`, `utils/`, answer generators, Hotpot/OWL evaluators, export utilities, and portions of the builders | Keep in normal package locations. Document which commit and CLI apply to each protocol. |
| D. Thesis-final checkpoints | `checkpoints/production_generator_d_v2_hard_pair/` (10 GraphSAGE models, 181,577,929 bytes); clean GNN-RAG baseline checkpoints under `checkpoints/production_generator_d_v1_gnn_rag_clean/` (367,179,545 bytes) | Preserve. The hard-pair hashes are embedded in frozen retrieval lineage. GNN-RAG is needed for final baseline reproduction. |
| E. Original-paper checkpoints | Ten `checkpoints/gnn_subgraph_ranker_*_full/` directories (182,312,415 bytes) | Preserve as `paper_original`; all ten are referenced by `experiments/run_experiments.py`. |
| F. Frozen final results | Hard-pair retrieval and end-to-end directories under `outputs/final_results/`; adaptive policies under `outputs/development_runs/production_generator_d_v2_hard_pair_*`; evaluation audit under `outputs/audits/final_test_evaluation_audit/` | Keep immutable. The manuscript-baseline directory is protected but still in progress, not frozen final. |
| G. Development diagnostics/ablations | `outputs/development_runs/` (90,277,981 bytes), `outputs/diagnostics/` (27,395,576 bytes), `outputs/final_model_development/` (9,020,751 bytes), `checkpoints/development/` (781,219,644 bytes), and associated `compare_*`, `diagnose_*`, `audit_*`, `evaluate_*_dev`, corrected-listwise, RRF, disagreement, proof-gate, and static-hard scripts | Archive with decision/status metadata. Do not delete: several directories record rejected mechanisms and scientific negative results. |
| H. Raw/processed datasets | `data/raw/` (1,948,468,991 bytes); original processed roots `data/{HotpotQA,2WikiMultiHopQA,FamilyOWL_*,owl2bench_*,pizza_*}` (8,848,381,850 bytes); Generator D `data/production_generator_d_v1/` (16,906,792,497 bytes) | Separate raw, `paper_original/processed`, and `thesis_final/generator_d_frozen_v1`. Preserve licenses and source URLs. |
| I. Temporary/cache/generated | `.venv/` (5,778,488,083 bytes), `.ruff_cache/` (9,159), first-party `__pycache__/` (2,868,465), root pytest scratch directories, and an empty `.deepeval/` | Re-creatable. Remove only after the live process exits and environment capture is complete. Never classify KG/LLM caches as disposable without a provenance check. |
| J. Obsolete/superseded | V1 adaptive aggregation/evaluation beside V2; rejected corrected-listwise, static-hard, RRF, disagreement-union, and ontology-proof-gate workflows; failed/incomplete GNN-RAG run directories | Superseded for the recommended pipeline, but scientifically meaningful. Move to a development archive with explicit `REJECTED`, `ABORTED`, or `SUPERSEDED` labels; do not delete by age/name alone. |
| K. Duplicates | GNN-RAG final checkpoints copied into `outputs/final_results/.../native_runs/*/checkpoint`; generated adapters in both checkpoint/output trees; submission bundle `dist/sageqa-submission-artifacts.zip` duplicates selected development artifacts | Deduplicate only after a content-addressed release bundle exists and all consumers have been redirected. The output-local copies currently make frozen runs self-contained. |
| L. Purpose not safely determined | `.worktrees/sageqa-original-eval-compat/`, `artifacts/step_01_context_to_kg/wikidata_cache/`, the old A0/A3 submission bundle and manifest, and some experimental checkpoint families | Requires human classification. They may encode historical evaluation compatibility or costly external-query provenance. |

### Source grouping recommendation

Keep the small stable modules in their functional directories. Put only
orchestrators in `scripts/`; do not physically split shared Python modules into
two copies. Protocol separation should be expressed by versioned configs,
manifests, and top-level wrappers:

- final thesis wrappers: hard-pair training, adaptive fitting, frozen TEST
  retrieval, answer generation/evaluation, and final baselines;
- original-paper wrapper: the historical `run_experiments.py` behavior pinned
  to the accepted-paper commit and original processed data/checkpoints;
- development archive: diagnostic and rejected-mechanism entry points, with a
  machine-readable status index.

### Duplicate/superseded implementations

- `evaluation/adaptive_support_aggregation.py` and
  `evaluation/evaluate_adaptive_support_aggregation.py` are V1. The final
  retrieval imports `adaptive_support_aggregation_v2.py`, and current fitting
  uses `evaluate_adaptive_support_aggregation_v2.py`. V1 must be archived, not
  deleted, until the original/development workflows are proven independent.
- `checkpoints/production_generator_d_v1/` is the clean Generator D predecessor
  of the hard-pair checkpoints. It remains referenced by multiple diagnostic
  scripts and is explicitly protected by the baseline runner. It is a thesis
  development checkpoint set, not an original-paper set and not presently safe
  to remove.
- `checkpoints/production_generator_d_static_hard_v1/` and its output records
  are superseded/rejected thesis-development evidence. Archive together.
- `outputs/final_results/production_generator_d_v1_test_baselines/` contains a
  completed clean lexical/GNN-RAG baseline plus explicitly named failed and
  incomplete attempts. The completed portion is an input to the live final
  manuscript baseline. Preserve the entire directory until the final baseline
  bundle is frozen; later separate completed provenance from failed attempts.
- `evaluation/test_generator_d_production_smoke.py` is a test placed under
  `evaluation/`; after freeze it should move to `tests/` only with import and CI
  updates.

## 2. Frozen thesis artifacts to protect

### 2.1 Generator D frozen corpus

**Path:** `data/production_generator_d_v1/`  
**Size:** 16,906,792,497 bytes  
**Datasets:** HotpotQA, 2WikiMultiHopQA, FamilyOWL 1/2 hop, Pizza 100/250 1/2
hop, OWL2Bench 1/2 hop.

There is no standalone root `corpus_manifest.json`. The complete manifest is
embedded as 66 `(relative path, size, SHA-256)` entries in
`outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/lineage_metadata.json`.
Its aggregate hash is:

`27a4c149181e96df1fa9aca581eaeee17e354755f6a034ce93f2021b496296e5`

The clean baseline runner also pins a distinct **30 retrieval-split-files-only**
manifest hash:

`fb77a4f30019d297ef6dbed22dec83bc86bedeaed5281fa51d2f1e7a8c33b076`

These hashes have different declared scopes and must not be substituted for
one another. After the baseline freezes, copy the 66-entry manifest into a
standalone, self-describing corpus manifest without changing the corpus.

### 2.2 Hard-pair GraphSAGE checkpoints

All models are directly referenced by the frozen retrieval lineage and by the
hard-pair launcher/retrieval code. Each directory also contains
`best_dev_metrics.json` and `best_dev_samples.json`, which should travel with
the model.

| Dataset | Checkpoint path | Bytes | SHA-256 |
|---|---|---:|---|
| 2WikiMultiHopQA | `checkpoints/production_generator_d_v2_hard_pair/2wiki/best_model.pt` | 18,107,829 | `734c9f39bd0908c4a59ff60eeeb12a0a1e56b7a238fe90ed51a576751a806172` |
| HotpotQA | `checkpoints/production_generator_d_v2_hard_pair/hotpotqa/best_model.pt` | 18,107,829 | `7aeb4e4d4c4780f43313e23f041f2d798fc46250772c5b55bf7060a885571fda` |
| FamilyOWL 1-hop | `checkpoints/production_generator_d_v2_hard_pair/familyowl_1hop/best_model.pt` | 18,107,829 | `94901a2339b418bb316bb04c5846bb50aeffac160297805576f1b7721ded0874` |
| FamilyOWL 2-hop | `checkpoints/production_generator_d_v2_hard_pair/familyowl_2hop/best_model.pt` | 18,107,829 | `e632824b63ba8ca4d010dd40696990bdeddf2f706f3c8db70cd33ab217912217` |
| Pizza 100 1-hop | `checkpoints/production_generator_d_v2_hard_pair/pizza_100_1hop/best_model.pt` | 18,107,829 | `5ad2d2e37a1bd3bde6ddc211bb5661721469f2a8f655f4c7d630b2555dc757ab` |
| Pizza 100 2-hop | `checkpoints/production_generator_d_v2_hard_pair/pizza_100_2hop/best_model.pt` | 18,107,829 | `b9e971fdccdab773d268890d39033299d2be89da6cfd7fbb0c22d7be06e920e1` |
| Pizza 250 1-hop | `checkpoints/production_generator_d_v2_hard_pair/pizza_250_1hop/best_model.pt` | 18,107,829 | `13e957365a7eeb8264efd8ee7d63145edba26be8c7ad43f54fcac921c6779dfd` |
| Pizza 250 2-hop | `checkpoints/production_generator_d_v2_hard_pair/pizza_250_2hop/best_model.pt` | 18,107,829 | `2270b5c743e8491576cd6854a718336caae85da23dedc3ce9b08234a32a8e136` |
| OWL2Bench 1-hop | `checkpoints/production_generator_d_v2_hard_pair/OWL2Bench_1hop/best_model.pt` | 18,107,829 | `8b6dc2f4071189f77dfb648ab36591f5fb041ce3f4f13e5877a42e6cd72cf9fa` |
| OWL2Bench 2-hop | `checkpoints/production_generator_d_v2_hard_pair/OWL2Bench_2hop/best_model.pt` | 18,107,829 | `18f02564c64b60117c6030b0de588c6b6d3f38e78411566cfc9a389e8ae42c7f` |

### 2.3 Frozen adaptive policies

Protect these directories as part of the thesis checkpoint bundle:

- `outputs/development_runs/production_generator_d_v2_hard_pair_gnn_only_adaptive_k/`
  (6,964,793 bytes), artifact-manifest hash
  `2fdf70d8982ad547f7caed856460c94202e84ebcd1864e9599420c59cdfcb8e8`.
- `outputs/development_runs/production_generator_d_v2_hard_pair_sageqa_final_adaptive_k/`
  (6,975,064 bytes), artifact-manifest hash
  `fafe3a6f96e27d82e6512f15e05f531f08119457c86b0733439c629ca00be1ba`.
- `outputs/development_runs/production_generator_d_v2_hard_pair_k_sensitivity_merged/`
  (20,363,435 bytes), which supplies the persisted score lineage used by the
  fitted policies.

The retrieval lineage enumerates the hash of every policy JSONL, JSON, joblib,
and summary file. Preserve the joblib files together with the exact
`scikit-learn==1.8.0` environment; do not regenerate them for packaging.

### 2.4 Frozen TEST retrieval

**Directory:**
`outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/`  
**Size:** 19,099,915 bytes  
**Status:** `complete_frozen_one_time_test_retrieval_evaluation`

Manifest SHA-256:
`7c8cfe13fc2ee18f52e5df856f5753dce5eb26bbb6d6fc13e5295fb126a816d9`.

| Manifest member | Bytes | SHA-256 | Rechecked |
|---|---:|---|---|
| `metrics.json` | 27,939 | `6a4a4bf2f667d976fdd6c4fc917d6276dfcb078dd71f92731b0ac86af81576a6` | yes |
| `per_example_test_retrieval.jsonl` | 19,017,982 | `1575240aa4e2f0cf53e34b793d91181b3f75b87c06ef9849da13506488fc12c5` | yes |
| `summary.md` | 8,605 | `f4ab2f9bd44cd1e067c979b4323559203aa7cce758d92751c5c028e8eb94418c` | yes |
| `lineage_metadata.json` | 44,599 | `75ac71599b1bbfd4ed22cb120cbd192e5a2f213b05cabd6f9ed02a24e0ae0423` | yes |

`artifact_manifest.json` intentionally excludes itself from its member hashes.

### 2.5 Frozen GNN/SAGE predictions, metrics, generation freeze, and lineage

**Directory:**
`outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end/`  
**Size:** 55,773,404 bytes  
**Status:** `complete_frozen`  
**Configurations:** `gnn_only` and `sageqa_final`, each at `k1` and `adaptive`  
**Predictions:** 16,996  
**Reader:** `gpt-4.1-mini`

Manifest SHA-256:
`3e70c14c0bee45852cb1c25da7245afa4b2cbc6c79402e0db17dad3c3c785ee0`.

| Manifest member | Bytes | SHA-256 | Rechecked |
|---|---:|---|---|
| `predictions.jsonl` | 30,503,763 | `751d9aac2ee19f0fe5c9191bb4c6df3c9160790c1fdea32d41a4c08d07406b4b` | yes |
| `generation_freeze.json` | 1,494 | `45dc3b3581d99e508dd11f9b3699f27126c6cd75cb1e0d0b06b958019e796b1c` | yes |
| `metrics.json` | 14,615 | `3215b1ffe7e32083fad891e67d4fc4a8d557c64390c38c1f90d2b9e5f6b5d9c1` | yes |
| `per_example_end_to_end.jsonl` | 25,236,991 | `be1e00d01887bacccb2a21171bc44349d44e374f50efac2ad84a9ff41fd56f81` | yes |
| `summary.md` | 5,153 | `ae04083a22d7ee492325f3669a802487957a2a5ddf5e141473cd483dd209848f` | yes |
| `lineage_metadata.json` | 10,259 | `68298723572f4961cf1406eac3572ca9c7117afeef36a2759077672cb03dfc25` | yes |

The generation freeze directly binds the predictions to retrieval SHA-256
`1575240a...c12c5` and records code hashes for the production runner, both
readers, and both evaluators. The prediction/support identity must not be
reconstructed from another artifact.

### 2.6 Final evaluation audit

**Directory:** `outputs/audits/final_test_evaluation_audit/`  
**Scientific verdict:** keep frozen metrics unchanged; zero score mismatches  
**Limitation:** the audit is diagnostic-only and its focused pytest attempt
ended with 3 passed and 13 environment/temp-directory setup errors. That test
result is not a repository-wide pass.

Protect at minimum:

- `audit_summary.json`: 102,077 bytes,
  `d3120acf5d977b4fd1c691b9fb976d3374e931964d3e7f97b0616feaa48ef6e3`.
- `verdict.md`: 9,312 bytes,
  `d85b3709dc028774b84fa5dddd959b757315c2a571ccd7288fc08e4aa313c91f`.
- `report.md`: 3,158 bytes,
  `a7e3f8ffbad1905b6ed94ef34e7371eb950645c7367dcbb8c0c434743f0fe1a0`.
- Five zero-byte JSONL finding sets, each with the standard empty-file SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Do not remove the zero-byte files: they are explicit empty finding sets, not
accidental clutter. A top-level audit manifest should be added after the
baseline freezes.

### 2.7 Protected but not yet final manuscript baseline

Protect the complete directory
`outputs/final_results/final_manuscript_baselines_test_end_to_end/` while it is
running. The stable input-side records at the audit snapshot are:

- `frozen_reader_inputs.jsonl`: 93,186,077 bytes, SHA-256
  `7096b39451e1f29c96ab121cb93fb1b84ebb2a634c92a6caf816503bdb2caa33`.
- `input_freeze.json`: 7,796 bytes, SHA-256
  `d66d20be9276d85a43b075a730b1902cf1e3aebb1151d1951aa037cca52cd467`.

The input freeze covers 4,249 examples and 12,747 reader inputs for lexical
k=1, GNN-RAG k=1, and full context. `predictions.jsonl` was incomplete and
actively changing; no hash for it is endorsed here.

## 3. Original-paper artifacts to preserve

The following ten checkpoint directories are referenced directly by the
`DATASETS` table in `experiments/run_experiments.py`; moving them without a
compatibility config/wrapper will break original-paper commands.

| Dataset | Checkpoint | Directory bytes | Model SHA-256 |
|---|---|---:|---|
| 2WikiMultiHopQA | `checkpoints/gnn_subgraph_ranker_2wiki_full/` | 18,181,222 | `a11ed8e3c5402a071eb4469de9f9f488b0078e93befcdee9b478674afee74a7b` |
| HotpotQA | `checkpoints/gnn_subgraph_ranker_hotpotqa_full/` | 18,175,209 | `101067c2fa5823036c2005bb5afb9d69239b937d11e1d190e7652e216ddcc09c` |
| FamilyOWL 1-hop | `checkpoints/gnn_subgraph_ranker_familyowl_1hop_full/` | 18,822,243 | `3ae6ed7132dc77ce77e1315bf82960b6445302b9418134530c33f520664e7184` |
| FamilyOWL 2-hop | `checkpoints/gnn_subgraph_ranker_familyowl_2hop_full/` | 18,152,815 | `697e29e2222949861dc391e8645978cd93b3f73c3bf334b2baae6f7742fe9d99` |
| Pizza 100 1-hop | `checkpoints/gnn_subgraph_ranker_pizza_100_1hop_full/` | 18,168,714 | `46f29429c2fe673472a6e4a29a1ac219f67e2937da702bffb14ba0d5951b18ba` |
| Pizza 100 2-hop | `checkpoints/gnn_subgraph_ranker_pizza_100_2hop_full/` | 18,167,579 | `ea76b8f06416dfc9c7f74304ced548c2a5c2511eb093ff18ad4dd85c5d7bea49` |
| Pizza 250 1-hop | `checkpoints/gnn_subgraph_ranker_pizza_250_1hop_full/` | 18,172,946 | `f799c51fa984fbffa13fdfcb50a118a058afc4f7c5841b92eff2db69d7830db6` |
| Pizza 250 2-hop | `checkpoints/gnn_subgraph_ranker_pizza_250_2hop_full/` | 18,158,095 | `d3ed4b1fc3e39772fbcb749ef317b3ed080ac3de59b28605a5968c6f89a0eeda` |
| OWL2Bench 1-hop | `checkpoints/gnn_subgraph_ranker_owl2bench_1hop_full/` | 18,158,196 | `3c7be6b74fe6af52ba7c03f43f30eb9cfb807a369f5eb850d2f4145a411a9e09` |
| OWL2Bench 2-hop | `checkpoints/gnn_subgraph_ranker_owl2bench_2hop_full/` | 18,155,396 | `2b17be6cf41f04aa6dd559e477044b29385487202b9ef2de08b309450a56b56d` |

Each directory also contains its `best_dev_metrics.json` and
`best_dev_samples.json`; keep those with the model. Preserve alongside:

- the ten original processed dataset roots under `data/` (8,848,381,850
  bytes);
- `outputs/full_results/` (276,686,666 bytes), including
  `full_pipeline_results.csv` SHA-256
  `df3c6292d71341ee9e8873528e908a0dea846cb8f29d6fa08542f41dc6b62473`
  and `full_pipeline_results.json` SHA-256
  `8c81730ad8fa273c03d41d6e91dbe275f493d10d27aa4006ea6683f74a186e7c`;
- the exact accepted-paper source commit/configuration and reader outputs;
- the local GNN-RAG fork/patch version used for those results.

Four checkpoint directories configured for the exploratory OpenAI/context-KG
variants are absent:

- `checkpoints/gnn_subgraph_ranker_hotpotqa_openai_gpt41mini_full`
- `checkpoints/gnn_subgraph_ranker_2wiki_openai_gpt41mini_full`
- `checkpoints/gnn_subgraph_ranker_hotpotqa_contextkg_v2_full`
- `checkpoints/gnn_subgraph_ranker_2wiki_contextkg_v2_full`

They are not part of the ten standard original-paper dataset entries. Decide
whether those config entries should be documented as unsupported exploratory
variants or supplied as additional artifacts.

## 4. Code-reference and movement impact audit

### Hardcoded current-path consumers

The following are direct breakpoints for a physical reorganization:

- `experiments/run_experiments.py` hardcodes original dataset, checkpoint,
  output, and gold paths for every dataset.
- `evaluation/run_production_test_retrieval.py` defaults to Generator D,
  `checkpoints/production_generator_d_v1`, V1 policy directories, and V1 final
  output. The hard-pair run overrides these on the command line; its exact
  command is saved in lineage.
- `scripts/run_sageqa_hard_pair_v2_all.sh` hardcodes the Generator D data root,
  hard-pair checkpoint root, and final training-log root.
- `generation/run_production_test_answer_generation.py` hardcodes the frozen
  hard-pair retrieval and end-to-end directories.
- `generation/run_final_manuscript_baselines.py` hardcodes Generator D, clean
  lexical/GNN-RAG inputs, hard-pair artifacts, and its live output directory.
- `evaluation/audit_final_test_evaluation.py` hardcodes the hard-pair frozen
  results, retrieval JSONL, and audit output.
- `evaluation/run_production_test_baselines.py` explicitly protects
  `checkpoints/production_generator_d_v1` and the V1 completed retrieval
  directory.
- V1/ablation diagnostic scripts refer directly to V1 checkpoints and
  development output directories.

No final-thesis wrapper is linked from README/REPRODUCE/RESULTS. A move must be
implemented as one atomic change: add configs/path resolution, update every
consumer and test, retain a legacy compatibility wrapper, and verify hashes
before deleting old locations. Do not move frozen artifacts merely to obtain a
prettier tree; a release index can map logical names to existing immutable
paths.

### Stored absolute paths

The final lineage and per-example artifacts contain absolute Windows paths,
including `C:\Users\julie\github\PhD\SAGE-QA`. This is useful provenance but
not portable. Do not rewrite frozen files. Instead, add a release-side logical
path map and teach verification code to match by relative path plus SHA-256.
The local GNN-RAG patch also contains upstream `/home/...` and
`/export/scratch/...` paths; those should be documented as upstream historical
context or removed from executable defaults in a separate source change.

### Tests and missing references

`.gitignore` ignores the entire `tests/` tree, so none of the current tests are
tracked. Several tests also import first-party modules that are absent from the
current filesystem listing (for example `data_processing.create_gnn_dev_sample`,
`evaluation.compare_retrieval_runs`, and multiple older A0/A3 evaluation
modules). A reviewer cannot currently obtain a coherent test suite from a
clone. Build an explicit test allowlist and either restore each required module
or archive the corresponding test with its historical protocol.

No test was run in this audit. Existing saved evidence must be reported
accurately: the answer-generation test attempt ended `2 passed, 7 errors` from
Windows temp-directory permissions, and the later evaluation-audit attempt
ended `3 passed, 13 errors` at temp setup/cleanup. These are partial,
environment-limited results, not full-suite success.

## 5. Proposed reviewer-ready structure

This is a logical target, not an instruction to move files before the final
baseline freeze:

```text
SAGE-QA/
|-- README.md
|-- CITATION.cff
|-- LICENSE
|-- pyproject.toml
|-- requirements/
|   |-- base-lock.txt
|   |-- dev-lock.txt
|   `-- frozen-20260910-win-cu124.txt
|-- configs/
|   |-- thesis_final/
|   |   |-- generator_d.yaml
|   |   |-- hard_pair_training.yaml
|   |   |-- adaptive_k.yaml
|   |   |-- test_retrieval.yaml
|   |   |-- answer_generation.yaml
|   |   `-- final_baselines.yaml
|   `-- paper_original/
|       `-- paper_pipeline.yaml
|-- data/
|   |-- README.md
|   |-- raw/                         # downloads/licensed archives
|   |-- thesis_final/
|   |   `-- generator_d_frozen_v1/  # external artifact, manifest in Git
|   `-- paper_original/
|       `-- processed/               # external artifact, manifest in Git
|-- data_processing/                 # shared builders/contracts
|-- models/                          # shared model implementations
|-- training/                        # reusable training modules
|-- retrieval/                       # optional thin public API/wrappers
|-- generation/                      # readers and generation core
|-- evaluation/                      # evaluators and integrity checks
|-- scripts/
|   |-- thesis_final/
|   `-- paper_original/
|-- tests/
|   |-- shared/
|   |-- thesis_final/
|   `-- paper_original/
|-- checkpoints/
|   |-- README.md
|   |-- thesis_final/
|   |   |-- hard_pair_graphsage/     # external bundle plus manifest
|   |   |-- adaptive_k/              # external bundle plus manifest
|   |   `-- gnn_rag_baseline/        # external bundle plus manifest
|   `-- paper_original/
|       `-- graphsage/                # external bundle plus manifest
|-- outputs/
|   |-- README.md
|   |-- thesis_final/
|   |   |-- retrieval/
|   |   |-- end_to_end/
|   |   |-- baselines/
|   |   `-- evaluation_audit/
|   |-- paper_original/
|   |   `-- full_results/
|   `-- development_archive/
|       |-- accepted_intermediate/
|       |-- rejected/
|       |-- aborted/
|       `-- exploratory/
|-- docs/
|   |-- REPRODUCIBILITY.md
|   |-- EXPERIMENTS.md
|   |-- DATA_LEAKAGE_REVISION.md
|   |-- ARTIFACTS.md
|   `-- repository_cleanup_audit.md
|-- manifests/
|   |-- thesis_final/
|   |-- paper_original/
|   `-- release_index.json
|-- third_party/
|   `-- GNN-RAG/                     # pinned source only
`-- legacy/
    `-- README.md                    # compatibility/deprecation map
```

Prefer a release index and configurable roots over duplicating shared code.
Frozen artifacts may retain their current physical paths in the first public
release if moving would invalidate lineage; the proposed tree can initially be
implemented as logical documentation plus download destinations.

## 6. Documentation plan

### `README.md`

- State in the first screen that the recommended pipeline is leakage-free
  Generator D + hard-pair GraphSAGE + frozen adaptive policy.
- Present two clearly separated reproduction tracks: **Final thesis
  (recommended)** and **Original paper (legacy protocol; historical
  reproducibility)**.
- Link the leakage revision, artifact manifest, experiments matrix, and exact
  reproduction commands.
- Mark which commands require GPU, hosted LLM API access, raw-data downloads,
  or externally hosted artifacts.
- Replace the current main-command emphasis on `run_experiments.py`; retain
  that command only in the original-paper section.
- Include a “frozen vs reproducible” legend: immutable saved result,
  deterministic local reconstruction, hosted-model reproduction, and
  exploratory/development artifact.

### `docs/REPRODUCIBILITY.md`

- Give clean-clone PowerShell and Bash instructions.
- Pin the exact Python/CUDA/PyTorch environment used for frozen training and
  inference and document CPU limitations.
- Provide commands in dependency order: acquire/verify corpus, acquire/verify
  checkpoint/policy bundles, verify manifests, reproduce retrieval, generate
  answers, evaluate after freeze, and verify original paper.
- Include expected output paths, row counts, hashes, estimated runtime, GPU
  memory/hardware notes, resume semantics, and offline Hugging Face flags.
- Explain that exact hosted-reader reproduction uses frozen predictions;
  rerunning `gpt-4.1-mini` may drift and requires explicit API authorization.
- Include Windows writable-temp guidance for pytest and UTF-8 settings.

### `docs/EXPERIMENTS.md`

- One row per experiment generation with protocol ID, status, data generation,
  checkpoint family, split use, gold-access boundary, output directory, and
  manuscript role.
- Mark hard-pair as accepted thesis final; mark corrected-listwise, static-hard,
  RRF, disagreement-union, and ontology proof gate with their recorded
  rejected/aborted/superseded status.
- Separate completed, partial, aborted, and exploratory runs. Never summarize a
  partial checkpoint/output as a result.

### `docs/DATA_LEAKAGE_REVISION.md`

Use neutral scientific wording along these lines:

> Review of the original experimental protocol identified target-information
> leakage in candidate/evidence construction. Gold support/evidence fields
> could influence candidate pools or GNN-RAG graph/target preparation before
> retrieval predictions were fixed. Consequently, results that depend on those
> candidate pools and the checkpoints trained from them are retained as the
> original-paper protocol and are not used as the recommended thesis results.
>
> The thesis protocol introduces Generator D candidate construction and clean
> retrieval contracts. At TEST time, answer fields, supporting facts, 2Wiki
> `evidences`, gold units, and derived labels are excluded from candidate
> construction, ranking, adaptive-k selection, and answer generation.
> Retrieval/support selections and reader predictions are frozen before a
> separate evaluation stage opens the permitted gold fields. Manifests bind the
> corpus, checkpoints, policies, source files, predictions, and metrics.
>
> Original-paper artifacts remain available solely to reproduce the published
> protocol. Thesis artifacts are explicitly labelled `thesis_final` and are the
> default route for new reproduction.

Also enumerate scope precisely: original processed candidate data, the ten
original checkpoints, and downstream retrieval/reader results are affected by
the original construction protocol; the issue is not hidden or described as
an evaluator failure. Describe Generator D's TRAIN/DEV labeling separately
from its gold-free TEST construction and post-freeze gold join.

### `docs/ARTIFACTS.md`

- Provide a single table of logical artifact ID, protocol, path/download URL,
  byte size, SHA-256, manifest schema, producer commit, and required-for field.
- Include both corpus-manifest scopes and all original/thesis checkpoint
  hashes.
- State storage location, retention policy, license, and whether an artifact
  contains raw benchmark text or hosted-model output.
- Provide a verifier command that never rewrites artifacts.

### `data/README.md`

- Separate raw downloads, original-paper processed data, and Generator D.
- Give source URLs/versions/licenses, exact extraction layout, split semantics,
  and checksum verification.
- Explain 2Wiki validation-as-evaluation usage and the gold-field firewall.
- Document generated caches, which are costly/reusable, and which can be
  discarded.

### `outputs/README.md`

- Define `paper_original`, `thesis_final`, and `development_archive`.
- Define statuses `COMPLETE_FROZEN`, `COMPLETE_DIAGNOSTIC`, `PARTIAL`,
  `ABORTED`, `REJECTED`, and `EXPLORATORY`.
- Identify the canonical metrics/summary files and evaluation populations.
- State that failed/incomplete baseline directories are non-evidence and that
  the current manuscript-baseline path becomes final only after its complete
  freeze/manifest/audit gate.

## 7. Reproducibility gaps

### A. Final thesis retrieval

Available: frozen Generator D corpus, ten hard-pair models, two adaptive policy
bundles, frozen TEST retrieval, hashes, exact saved command, Python/Torch/device
lineage, and gold-boundary declarations.

Missing or inadequate:

- standalone downloadable corpus/checkpoint/policy manifests and URLs;
- committed final runners/tests/configs;
- a clean-clone command using the hard-pair paths rather than V1 defaults;
- exact training environment lock, CUDA/driver/GPU model and memory notes;
- one portable root/path mapping for Windows and Linux;
- a documented seed matrix and deterministic-kernel policy;
- a release verifier that checks all bundles before importing/loading models.

### B. Final thesis answer generation/evaluation

Available: frozen retrieval, 16,996 frozen GNN/SAGE predictions, generation
freeze, code hashes, metrics, per-example outputs, lineage, manifest, and an
independent diagnostic audit.

Missing or inadequate:

- the runner and audit script are untracked;
- API/provider requirements are not fully pinned beyond the recorded model
  label `gpt-4.1-mini`; endpoint/provider deployment identity is mutable;
- no exact environment lock for `openai==1.66.3` and all transitives;
- no public instructions for generate-versus-evaluate phases, safe resume, and
  the post-freeze gold join;
- no manifest for the evaluation-audit directory;
- no successful complete focused test record due Windows temp ACL failures.

### C. Original paper results

Available: original processed data roots, ten referenced checkpoints, full
per-method outputs/aggregates, current historical runner, raw archives, and a
local GNN-RAG patch.

Missing or inadequate:

- an identified accepted-paper Git tag/commit and exact source archive;
- a manifest binding original data, checkpoints, outputs, prompts, configs,
  and evaluator versions;
- neutral protocol labelling and leakage-scope documentation;
- exact hosted reader/provider version and cached-output manifest;
- unambiguous dataset versions and acquisition dates;
- confirmation whether the detached
  `.worktrees/sageqa-original-eval-compat` commit
  `ffee42cced77134614f1615b391c67661ab963d7` is the intended compatibility
  snapshot;
- decisions for the four configured-but-missing exploratory checkpoints.

### Dependency and environment gaps

`requirements.txt` pins several packages but does not reproduce the frozen
environment. The active environment includes at least:

- `torch==2.6.0+cu124` (frozen lineage), not requested `torch==2.11.0`;
- `datasets==4.8.5`;
- `pyarrow==23.0.1`;
- `numpy==2.4.4`;
- `networkx==3.6.1`;
- `joblib==1.5.3`;
- `openai==1.66.3`;
- `huggingface-hub==0.36.2`;
- `transformers==4.49.0`;
- `scikit-learn==1.8.0`.

Declare direct dependencies rather than relying on transitive installation.
Keep a human-readable minimal requirements file and generate platform-specific
locked environments for frozen reproduction. Document the known incompatibility
of the system `huggingface-hub==1.14.0` with eager Transformers imports and the
repository `.venv` used for the frozen run.

## 8. Large-file and GitHub strategy

| Artifact class | Current size | Recommendation | Reason |
|---|---:|---|---|
| Source, configs, docs, tests, JSON manifests, small aggregate metrics | small | ordinary Git | Reviewable diffs and permanent protocol history. |
| Hard-pair checkpoints | 181,577,929 bytes total; ~18.1 MB/model | versioned GitHub Release bundle plus DOI-backed archive; Git LFS only if binary history in branches is required | Each file fits GitHub's hard limit, but ordinary Git history is inappropriate. Release checksum is stable for reviewers. |
| Original-paper checkpoints | 182,312,415 bytes total | separate `paper-original` Release asset plus DOI-backed archive | Must be retained without confusing it with thesis defaults. |
| Clean GNN-RAG checkpoint/adapter tree | 367,179,545 bytes | thesis-baselines Release asset; consider excluding reproducible adapters if separately manifest-bound | Required baseline provenance but too bulky for ordinary Git. |
| All checkpoints | 1,874,943,938 bytes | do not add wholesale to Git; split final/original/development bundles | Most are development artifacts. |
| Generator D corpus | 16,906,792,497 bytes | external archived storage (Zenodo/institutional/object store) with DOI/version plus download script and manifest | Too large for ordinary Git or practical LFS distribution. |
| Original processed datasets | 8,848,381,850 bytes | separate historical archive or deterministic rebuild instructions plus manifest | Needed for original protocol; keep separate from thesis data. |
| Raw text benchmarks | part of 1,948,468,991-byte raw tree | download instructions and checksums only, subject to upstream licenses | Avoid redistributing third-party benchmark data without license review. |
| Four ontology ZIPs | 82,536,157 bytes total and currently tracked | ordinary Git is technically possible; Git LFS or Release is cleaner after license review | `family.zip` is 57 MB and already creates a heavy clone. |
| Frozen hard-pair retrieval/end-to-end/audit | ~74.99 MB | GitHub Release plus small in-Git summaries/manifests; DOI archive preferred | Immutable research record with per-example files. |
| Final manuscript baselines | live/incomplete | decide only after complete freeze | Do not package a partial prediction file. |
| `outputs/full_results/` | 276,686,666 bytes | original-paper Release bundle | Historical results must remain available but clearly labelled. |
| `third_party/GNN-RAG/` | 1,900,266,512 bytes | keep pinned source/patch in Git; externalize generated data/models after license review | Source and generated payload are currently mixed. |
| `dist/sageqa-submission-artifacts.zip` | 52,118,416 bytes | keep until its A0/A3 role is classified; then release/DOI, not duplicate ordinary Git | SHA-256 `9a4bc68150094165b0fbfa30dcb3dc3bbaa4bcb11da5d7f73f32b4c8128a23be`. |

Do not use Git LFS as the only preservation mechanism. Publish checksummed
release assets and mirror thesis-critical and original-paper bundles in durable
archived storage. Keep manifests and download/verification scripts in ordinary
Git.

## 9. Cleanup candidates

All actions below are proposals for **after** the final baseline is frozen,
backed up, and independently verified.

### SAFE TO REMOVE

| Exact path | Reason | Referenced? | Recoverable space |
|---|---|---|---:|
| `.ruff_cache/` | Re-creatable linter cache | ignored; no workflow dependency | 9,159 bytes |
| `data/__pycache__/` | Python bytecode | no; imports use source | 43,701 bytes |
| `data_processing/__pycache__/` | Python bytecode | no | 238,728 bytes |
| `evaluation/__pycache__/` | Python bytecode, including audit compile output | no scientific dependency | 1,366,535 bytes |
| `experiments/__pycache__/` | Python bytecode | no | 90,568 bytes |
| `generation/__pycache__/` | Python bytecode | no | 145,250 bytes |
| `models/__pycache__/` | Python bytecode | no | 145,605 bytes |
| `tests/__pycache__/` | Python bytecode | no | 702,614 bytes |
| `training/__pycache__/` | Python bytecode | no | 119,169 bytes |
| `utils/__pycache__/` | Python bytecode | no | 16,295 bytes |
| `.deepeval/` | Empty local tool-state directory | no reference found | 0 bytes |
| `.test_tmp_adaptive_k/` | Pytest scratch | no source reference; ACL blocked measurement | unknown |
| `.tmp/pytest-static-hard/` | Pytest scratch | no source reference; ACL blocked measurement | unknown |
| `.tmp/pytest_answer_generation/` | Pytest scratch | no source reference; ACL blocked measurement | unknown |
| `.tmp/pytest_answer_generation_final/` | Pytest scratch | no source reference; ACL blocked measurement | unknown |
| `.tmp/pytest_answer_generation_resume/` | Pytest scratch | no source reference; ACL blocked measurement | unknown |
| `.tmp_test_answer_generation/` | Pytest scratch | no source reference; ACL blocked measurement | unknown |
| `pytest-of-julie/` | Pytest scratch | no source reference; ACL blocked measurement | unknown |
| `outputs/audits/final_test_evaluation_audit/pytest_tmp/` | Failed audit-test scratch only | audit verdict documents it; not a scientific artifact | unknown |

Known safe cache recovery excluding unreadable scratch is **2,877,624 bytes
(2.74 MiB)**. After the running baseline exits and a complete environment lock
is saved and tested, `.venv/` is also re-creatable and could recover
5,778,488,083 bytes (5.38 GiB), but it must not be removed while either live
Python process exists.

Do not remove the five zero-byte audit JSONL files, KG/LLM caches, failed-run
manifests, or quarantine records under this category.

### SAFE TO MOVE/ARCHIVE

These are safe to *classify and relocate with manifest-preserving path updates*,
not safe to delete:

| Current path | Proposed destination/class | Size | Reference impact |
|---|---|---:|---|
| Ten `data/<original dataset>/` roots | `data/paper_original/processed/` | 8,848,381,850 bytes | Breaks `run_experiments.py`; add legacy config/wrapper first. |
| Ten `checkpoints/gnn_subgraph_ranker_*_full/` roots | `checkpoints/paper_original/graphsage/` | 182,312,415 bytes | Directly referenced by `run_experiments.py`. |
| `outputs/full_results/` | `outputs/paper_original/full_results/` | 276,686,666 bytes | README and error-analysis examples refer to it. |
| `checkpoints/development/` | external `development_archive/checkpoints/` | 781,219,644 bytes | Some tests/scripts and `artifacts/manifest.json` reference subsets. |
| `outputs/development_runs/` excluding the three frozen hard-pair policy directories | `outputs/development_archive/accepted_intermediate/` or status-specific subdirectories | up to 90,277,981 bytes | Many diagnostics use hardcoded paths; create an index and retain frozen policy paths. |
| `outputs/diagnostics/` excluding final evaluation audit | `outputs/development_archive/{rejected,exploratory}/` | 27,395,576 bytes | Associated scripts contain direct defaults. |
| `outputs/final_model_development/` | `outputs/development_archive/rejected/` | 9,020,751 bytes | Static-hard scripts refer to it. |
| `outputs/quarantine/aborted_20260909_sageqa_v2_hard_pair/` | `outputs/development_archive/aborted/` | 20,047,392 bytes | Keep quarantine manifest and non-evidence status. |
| V1 adaptive aggregation/evaluation and rejected-mechanism scripts | `legacy/development/` | source only | Update/import-test every consumer before moving. |

### KEEP

- Every artifact in Section 2, including the live manuscript-baseline
  directory until it has its own final freeze and audit.
- Every original-paper artifact in Section 3.
- `checkpoints/production_generator_d_v1_gnn_rag_clean/` and the completed
  clean lexical/GNN-RAG baseline outputs used by final manuscript baselines.
- `outputs/final_runs/production_generator_d_v2_hard_pair/` training logs
  (7,359,825 bytes) as checkpoint lineage.
- All raw-source archives and download provenance pending license review.
- `third_party/GNN-RAG` source and `patches/GNN-RAG-local-changes.*` until a
  reproducible pinned fork/commit replaces them.
- `artifacts/manifest.json` and its referenced A0/A3 bundle until that distinct
  submission generation is classified.
- `.env` only as a local ignored file; never copy it into a release, report, or
  archive. The audit did not inspect or disclose its values.

### REQUIRES HUMAN DECISION

| Path/item | Why a decision is required |
|---|---|
| `.worktrees/sageqa-original-eval-compat/` (112,370,538 bytes) | It is a registered detached worktree at commit `ffee42c...`; decide whether this is the required original evaluation source before pruning it. |
| `artifacts/step_01_context_to_kg/wikidata_cache/` (about 36.5 MB with manifest) | May be costly external-query provenance and may contain redistributable third-party response data; inspect license/privacy and protocol role. |
| `dist/sageqa-submission-artifacts.zip` (52,118,416 bytes) | It packages a separate frozen A0/A3 architecture generation, not the hard-pair thesis or the obvious ten-dataset paper bundle. Keep until its thesis/manuscript role is resolved. |
| `checkpoints/production_generator_d_v1/` (181,578,675 bytes) | Superseded by hard-pair for final thesis, but referenced by diagnostics and protected by baseline code. Archive only after references are redirected. |
| `checkpoints/production_generator_d_static_hard_v1/` (181,075,730 bytes) | Rejected/superseded development set; retain if negative-result reproducibility is part of reviewer materials. |
| Remaining `checkpoints/development/*` | Some are smoke/workbench runs, some are frozen submission artifacts. Classify using manifests before selecting a release subset. |
| `outputs/final_results/production_generator_d_v1_answer_generation/` (22,108,978 bytes) | Earlier clean V1 answer run, superseded for final thesis but potentially useful protocol history; not safe to delete without manuscript mapping. |
| Failed/incomplete GNN-RAG native-run directories under `outputs/final_results/production_generator_d_v1_test_baselines/` | Scientifically non-final but document failure chronology. Archive or retain based on reviewer transparency policy. |
| Generated data under `third_party/GNN-RAG/gnn/data/` (~940 MB) and output-local native adapters | Potential duplicates, but exact self-contained baseline reproduction and upstream licensing must be resolved first. |
| 2.44 GiB of loose `.git/objects` | Never delete manually. After backups/tags and worktree decisions, measure reachability and use normal Git maintenance; recovery is not yet estimable. |

## 10. Largest directories and files

Directory sizes are recursive and overlap when a child is listed beneath a
parent.

### Largest 20 directories

| Rank | Path | Bytes | GiB |
|---:|---|---:|---:|
| 1 | `data/` | 27,703,723,438 | 25.801 |
| 2 | `data/production_generator_d_v1/` | 16,906,792,497 | 15.746 |
| 3 | `.venv/` | 5,778,488,083 | 5.382 |
| 4 | `.venv/Lib/` | 5,743,194,378 | 5.349 |
| 5 | `.venv/Lib/site-packages/` | 5,743,194,378 | 5.349 |
| 6 | `data/production_generator_d_v1/HotpotQA/` | 5,404,032,767 | 5.033 |
| 7 | `data/production_generator_d_v1/2WikiMultiHopQA/` | 4,839,010,618 | 4.507 |
| 8 | `.venv/Lib/site-packages/torch/` | 4,681,261,896 | 4.360 |
| 9 | `.venv/Lib/site-packages/torch/lib/` | 4,570,241,692 | 4.256 |
| 10 | `data/HotpotQA/` | 3,463,148,993 | 3.225 |
| 11 | `data/2WikiMultiHopQA/` | 2,858,237,107 | 2.662 |
| 12 | `.git/` | 2,620,057,944 | 2.440 |
| 13 | `.git/objects/` | 2,619,965,044 | 2.440 |
| 14 | `data/raw/` | 1,948,468,991 | 1.815 |
| 15 | `third_party/` | 1,919,105,185 | 1.787 |
| 16 | `third_party/GNN-RAG/` | 1,900,266,512 | 1.770 |
| 17 | `checkpoints/` | 1,874,943,938 | 1.746 |
| 18 | `data/production_generator_d_v1/FamilyOWL_2hop/` | 1,321,427,519 | 1.231 |
| 19 | `data/production_generator_d_v1/FamilyOWL_1hop/` | 1,319,890,441 | 1.229 |
| 20 | `third_party/GNN-RAG/gnn/` | 1,310,692,205 | 1.221 |

### Largest 20 files

| Rank | Path | Bytes |
|---:|---|---:|
| 1 | `data/production_generator_d_v1/HotpotQA/train_subgraph_retrieval.jsonl` | 3,858,164,948 |
| 2 | `data/production_generator_d_v1/2WikiMultiHopQA/train_subgraph_retrieval.jsonl` | 3,425,043,706 |
| 3 | `data/HotpotQA/train_subgraph_retrieval.jsonl` | 2,317,847,984 |
| 4 | `data/2WikiMultiHopQA/train_subgraph_retrieval.jsonl` | 1,861,301,492 |
| 5 | `.venv/Lib/site-packages/torch/lib/torch_cuda.dll` | 957,268,480 |
| 6 | `data/production_generator_d_v1/HotpotQA/test_subgraph_retrieval.jsonl` | 908,811,609 |
| 7 | `data/production_generator_d_v1/FamilyOWL_1hop/train_subgraph_retrieval.jsonl` | 854,643,861 |
| 8 | `data/production_generator_d_v1/FamilyOWL_2hop/train_subgraph_retrieval.jsonl` | 852,589,868 |
| 9 | `data/production_generator_d_v1/2WikiMultiHopQA/test_subgraph_retrieval.jsonl` | 830,289,872 |
| 10 | `data/HotpotQA/test_subgraph_retrieval.jsonl` | 749,400,810 |
| 11 | `data/production_generator_d_v1/OWL2Bench_2hop/train_subgraph_retrieval.jsonl` | 745,667,030 |
| 12 | `data/production_generator_d_v1/OWL2Bench_1hop/train_subgraph_retrieval.jsonl` | 688,987,625 |
| 13 | `data/2WikiMultiHopQA/test_subgraph_retrieval.jsonl` | 662,236,278 |
| 14 | `.venv/Lib/site-packages/torch/lib/dnnl.lib` | 653,534,016 |
| 15 | `data/production_generator_d_v1/HotpotQA/dev_subgraph_retrieval.jsonl` | 636,884,836 |
| 16 | `.venv/Lib/site-packages/torch/lib/cudnn_engines_precompiled64_9.dll` | 588,910,632 |
| 17 | `data/production_generator_d_v1/2WikiMultiHopQA/dev_subgraph_retrieval.jsonl` | 583,470,041 |
| 18 | `.venv/Lib/site-packages/torch/lib/cublasLt64_12.dll` | 472,802,816 |
| 19 | `data/HotpotQA/dev_subgraph_retrieval.jsonl` | 382,601,331 |
| 20 | `data/FamilyOWL_1hop/train_subgraph_retrieval.jsonl` | 357,458,046 |

## 11. Exact recommended cleanup sequence after the final baseline freezes

1. **Wait for completion.** Confirm both Python processes have exited. Do not
   stop them for cleanup. Record final row counts, status, stderr/stdout, and
   completion timestamp.
2. **Freeze the manuscript baselines.** Produce their generation/prediction
   freeze, post-freeze gold evaluation, metrics, per-example output, summary,
   lineage, and self-excluding artifact manifest. Do not refit retrieval,
   adaptive-k, or any reader input.
3. **Run a read-only integrity/audit gate.** Recompute every final-baseline
   member hash and independently reproduce aggregate metrics. Resolve test temp
   ACLs in a demonstrably writable location, and record the actual final test
   outcome without changing scientific artifacts.
4. **Take two immutable backups.** Snapshot the complete current repository
   (including ignored artifacts and Git metadata) and separately copy the final
   scientific bundles. Generate a top-level checksum manifest for each copy and
   verify both copies before any move.
5. **Resolve source lineage.** Identify and tag the exact original-paper source
   commit; decide the role of detached commit `ffee42c...`; capture the exact
   hard-pair/final-baseline source as an allowlisted commit. Do not bulk-commit
   the current 115-file modification set.
6. **Create in-Git manifests/configs/docs first.** Add the release index,
   standalone Generator D manifest, original checkpoint/result manifest,
   thesis checkpoint/policy manifest, locked environments, protocol configs,
   and the documentation set described above. Ensure `.env` and secrets remain
   excluded.
7. **Publish external artifacts before redirecting paths.** Upload separate
   checksummed `thesis-final`, `thesis-baselines`, `paper-original`, and optional
   `development-archive` bundles to a GitHub Release and durable DOI-backed
   storage. Test download and hash verification from a fresh directory.
8. **Introduce configurable roots and compatibility wrappers.** Replace
   hardcoded paths with config/repository-relative resolution while preserving
   frozen file bytes. Maintain the original-paper command through a legacy
   wrapper. Test both Windows and Linux command forms.
9. **Move only in small protocol-specific batches.** First original data,
   checkpoints, and results; then thesis artifacts if still beneficial; then
   development/rejected/aborted outputs. After each batch, search code/docs/
   configs/tests for the old paths and run focused integrity and import tests.
10. **Repair the public test surface.** Stop ignoring all of `tests/`; explicitly
    track the supported thesis, original, and shared tests. Archive tests whose
    modules are absent. Obtain a complete recorded pass in the locked clean
    environment.
11. **Remove only verified disposable local state.** Delete the exact cache/temp
    paths in `SAFE TO REMOVE`. Recreate and validate the environment before
    optionally removing `.venv`. Never manually delete `.git/objects`.
12. **Perform Git maintenance only after lineage decisions.** Remove a nested
    worktree through normal Git worktree commands only if its commit is tagged
    and archived. Then run normal Git reachability/garbage-collection checks
    and remeasure repository size.
13. **Fresh-clone reviewer rehearsal.** On both PowerShell and Bash where
    feasible: clone, install from lock, download/verify each selected bundle,
    run code-only tests, verify frozen artifacts, reproduce thesis retrieval
    without gold/API access, verify saved end-to-end metrics, and reproduce the
    original-paper aggregate from its historical bundle.
14. **Release gate.** Compare the fresh-clone outputs to the release manifests,
    check links/licenses/citation metadata, confirm no secret or restricted raw
    data is staged, and only then make the repository/release public.

## Final summary

1. **Current repository size:** at least 41,308,716,312 bytes (38.47 GiB),
   including Git, local environment, and worktrees; live outputs were growing.
2. **Largest items:** dominated by Generator D (15.75 GiB), the local venv
   (5.38 GiB), original processed text data, Git objects (2.44 GiB), raw data,
   GNN-RAG, and checkpoints. Exact top-20 tables are above.
3. **Original-paper checkpoints:** ten `gnn_subgraph_ranker_*_full` directories,
   182,312,415 bytes total, all referenced and hash-listed above.
4. **Thesis checkpoints:** ten hard-pair GraphSAGE models plus adaptive-policy
   bundles and clean GNN-RAG baseline checkpoints.
5. **Frozen thesis artifacts:** 66-file Generator D corpus manifest; hard-pair
   models; adaptive policies; completed retrieval; 16,996 frozen GNN/SAGE
   predictions; generation freeze; metrics; lineage; and diagnostic evaluation
   audit. The manuscript baselines are protected but not yet final.
6. **Proposed tree:** separates `paper_original`, `thesis_final`, and
   `development_archive` across configs, data, checkpoints, outputs, manifests,
   scripts, tests, and documentation.
7. **Safe removal:** only re-creatable caches/temp roots, with 2.74 MiB measured
   plus unknown ACL-blocked scratch; `.venv` adds 5.38 GiB only after process
   exit and environment capture.
8. **Human decisions:** historical worktree/commit, A0/A3 submission bundle,
   Wikidata cache, superseded checkpoint families, failed-run retention,
   GNN-RAG generated payload, and original source tag.
9. **Main gaps:** untracked final runners, ignored/untracked tests and
   artifacts, dirty source tree, missing configs/manifests/docs, mismatched and
   incomplete dependency pins, absolute stored paths, missing original source
   tag, and incomplete final baselines.
10. **Cleanup order:** freeze and audit the live baseline first; back up and
    manifest; resolve lineage; publish verified external bundles; make paths
    configurable; move in small tested batches; remove caches last; rehearse a
    fresh-clone release before publication.
