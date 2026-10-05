# Reproducibility

This guide separates source verification, frozen-output verification, protocol replay,
and full regeneration. A successful lower level must not be reported as a higher one.

## Fresh-clone table reproduction

Copy the separately supplied `paper_table_artifacts.zip` and
`thesis_table_artifacts.zip` into `release/reviewer_artifacts/`. Their expected package
hashes and complete member inventories are tracked under
`release_manifests/reviewer_artifacts/`. Then run:

```powershell
python scripts/restore_reviewer_artifacts.py release/reviewer_artifacts/paper_table_artifacts.zip
python scripts/restore_reviewer_artifacts.py release/reviewer_artifacts/thesis_table_artifacts.zip
python scripts/reproduce_reported_tables.py --output-dir reproduced_tables
```

No API, GPU, model download, dataset download, or checkpoint is used. The output
manifest records the exact reproduced files. This is the acceptance path for the
reported paper and final-thesis tables.

## Common source-only verification

From a fresh checkout with Python 3.12:

```powershell
python -m pip install pytest==9.0.2 PyYAML==6.0.2
python -m compileall -q data data_processing evaluation experiments generation models training utils
python -m pytest -q -p no:cacheprovider tests/ci/test_reviewer_contracts.py
```

This checks tracked source, portable paths, release-index schemas, recorded source
hashes, protocol separation, and the distinction between official and diagnostic
scoring. It does not require or validate external artifacts.

## Path A: original published paper

### Scientific boundary

The strongest source candidate is commit
`dbdbb50708bdc6c686ef82518ec71c1d1bf55985`, tagged
`conference-submission`. Exact publication source/result binding is unverified because
the ignored datasets, checkpoints, results, historical environment, and hosted-reader
state were not captured in a contemporaneous commit-bound manifest.

The historical pipeline used gold-informed candidate generation on TEST: text builders
could insert/prioritize gold support, and ontology construction used SPARQL-derived
information in candidate selection/scoring/features. This path reproduces the published
historical protocol; it is not the thesis gold-firewall protocol.

### Currently supported

The two source-only original-paper contract tests run in a fresh clone:

```powershell
python -m pytest -q -p no:cacheprovider `
  tests/reproducibility/test_original_paper_contracts.py::test_original_paper_candidate_and_historical_modules_exist_in_git `
  tests/reproducibility/test_original_paper_contracts.py::test_original_paper_index_fails_closed_on_unproven_lineage
```

After the compact paper table package is restored, the table reproducer operates only
on frozen paper result files. The larger full-pipeline package remains specified by
`release_manifests/archive_specs/paper_original.yaml`.

If that larger package is later restored, run the complete contract file:

```powershell
python -m pytest -q -p no:cacheprovider `
  --basetemp .tmp/reproducibility/paper `
  tests/reproducibility/test_original_paper_contracts.py
```

Scientific execution is described by `experiments/run_experiments.py`. It requires the
restored datasets/checkpoints, GPU resources for practical runs, and a hosted reader.
Because provider state was not frozen, a new run is a protocol reproduction, not a
promise of byte-identical answers.

### Blocked

- No public URL exists for the larger training/re-execution package.
- The historical environment and provider/model snapshot are incomplete.
- The candidate source revision is not an authoritative publication binding.

## Path B: revised thesis experiments

### Scientific boundary

The thesis path freezes TEST rankings and reader inputs before answer/support gold is
opened. It contains two separately retained result layers:

1. The earlier hard-pair-v2 baseline/export under
   `outputs/final_results/manuscript_retrieval_results_hard_pair_v2/`.
2. The later optimized symbolic-coefficient results indexed by
   `release_manifests/thesis_optimized/index.yaml`.

The optimized overlay does not overwrite the earlier export. It recomputes official
Support P/R/F1 from the retained optimized selected supports and the canonical annotated
references.

### Frozen-output verification after artifact restore

The compact thesis table package is sufficient for `scripts/reproduce_reported_tables.py`.
For broader frozen-output and pipeline contracts, restore `thesis_final_artifacts` using
`release_manifests/archive_specs/thesis_final.yaml`, then run:

```powershell
python evaluation/check_thesis_final_paths.py
python -m pytest -q -p no:cacheprovider `
  --basetemp .tmp/reproducibility/thesis `
  tests/reproducibility --ignore=tests/reproducibility/test_original_paper_contracts.py
python evaluation/verify_canonical_retrieval_overlay.py --asset-root .
```

Set `SAGEQA_FULL_ARTIFACT_HASHES=1` only for the documented full 66-member corpus hash
pass. The exact FAST/STANDARD/FULL boundaries are in
`docs/REPRODUCIBILITY_TEST_BASELINE.md`.

### Optimized correction: available and blocked capabilities

Available after restoring the retained downstream bundle:

- verify 8,498 frozen reader-input rows over 4,249 questions;
- recompute official Support P/R/F1 on 3,509 support-bearing examples;
- match every optimized per-dataset and equal-dataset-macro Support F1 cell;
- verify frozen answer, end-to-end metric, and stagewise diagnostic hashes.

Blocked upstream steps:

- byte-compare the missing optimized TEST ranking file (`770ef934...`);
- byte-compare the missing selected coefficients (`9cc3d4b8...`), adaptive-policy
  manifest (`45f771a3...`), and derived DEV comparison (`0c08b16f...`);
- prove the original GPU handoff contents, which were not recovered.

A future optimization rerun from the verified DEV inputs would be a new protocol
reproduction. It must not be labeled recovery of those missing frozen bytes.

## Environments

`requirements.txt` records the current direct environment but is not a complete
historical lock; `openai` remains a range and CUDA/hardware builds are not frozen.
Vendored GNN-RAG uses a separate legacy dependency stack. Archival publication should
therefore include an environment inventory for each path, model identifiers, CUDA/driver
metadata where relevant, and hosted-reader identity separately from source code.
