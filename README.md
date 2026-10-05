# SAGE-QA

SAGE-QA is a retrieval-augmented question-answering system for multi-hop text and
OWL-style benchmarks. This repository preserves two scientifically distinct research
versions. They share infrastructure, but their candidate-generation and evaluation
protocols must not be interchanged.

| Reproduction path | Scientific status | What a fresh clone can do now |
|---|---|---|
| [Original published paper](docs/REPRODUCIBILITY.md#path-a-original-published-paper) | Historical, gold-informed candidate generation; source revision is the strongest candidate, not proven authoritative | Inspect and test source contracts; full replay is blocked until the unpublished artifact package is restored |
| [Revised thesis experiments](docs/REPRODUCIBILITY.md#path-b-revised-thesis-experiments) | TEST retrieval and reader inputs frozen before gold evaluation; includes hard-pair-v2 and the later optimized SAGE-QA correction | Inspect and test source/manifests; verify frozen outputs after restoring the unpublished thesis package |

## Reviewer quick start: reproduce both result-table families

The review package supplies these two files separately from Git:

- `paper_table_artifacts.zip` — SHA-256
  `7e776628d2988222ce0f7770bda868d94ee49fec6ffc705e7b30b1bea76a1b48`
- `thesis_table_artifacts.zip` — SHA-256
  `18d9faf7b483f59a2ba6367e28db20d2ef7071bd6bb8ca06da4464376f7e7ea1`

Place them under `release/reviewer_artifacts/` in a fresh clone, then run:

```powershell
python scripts/restore_reviewer_artifacts.py release/reviewer_artifacts/paper_table_artifacts.zip
python scripts/restore_reviewer_artifacts.py release/reviewer_artifacts/thesis_table_artifacts.zip
python scripts/reproduce_reported_tables.py --output-dir reproduced_tables
```

The restore commands validate every file against the embedded per-file manifest. The
reproduction command materializes the hash-verified frozen paper tables, independently
reruns the thesis retrieval and end-to-end finalizers,
requires byte-identical agreement with the retained hard-pair-v2 tables, and replays the
optimized overlay from the frozen selected supports. Outputs and hashes are written to
`reproduced_tables/REPRODUCTION_MANIFEST.json`.

## Source-only verification

These commands use only tracked files and do not train models, call an API, or claim
artifact reproduction:

```powershell
python -m pip install pytest==9.0.2 PyYAML==6.0.2
python -m compileall -q data data_processing evaluation experiments generation models training utils
python -m pytest -q -p no:cacheprovider tests/ci/test_reviewer_contracts.py
python -m pytest -q -p no:cacheprovider `
  tests/reproducibility/test_original_paper_contracts.py::test_original_paper_candidate_and_historical_modules_exist_in_git `
  tests/reproducibility/test_original_paper_contracts.py::test_original_paper_index_fails_closed_on_unproven_lineage
```

Full model training and hosted-reader regeneration remain outside this table-replay
acceptance path. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for that boundary.

## Evidence states

- **Recovered and hash-verified:** original bytes are present and match recorded hashes.
- **Independently reconstructed result:** a metric was recomputed from retained frozen
  supports; this is not recovery of its missing upstream artifact.
- **Missing raw artifact:** only the expected path/hash or downstream binding survives.
- **Unverified historical provenance:** surviving files are associated by paths and
  protocol, but no contemporaneous manifest proves the exact source/result binding.

The optimized retrieval overlay is an independently reconstructed metric view. It
matches the retained selected supports and canonical annotated-reference evaluator, but
the original optimized TEST ranking file with SHA-256 `770ef934...` remains missing.

## Documentation map

- [Reproduction procedures](docs/REPRODUCIBILITY.md)
- [Experiment lineage and hashes](docs/EXPERIMENT_LINEAGE.md)
- [Official and diagnostic evaluation](docs/EVALUATION_PROTOCOL.md)
- [Archival package specifications](docs/ARCHIVAL_PACKAGES.md)
- [Phase 2 evidence gate](docs/PHASE2_REPRODUCIBILITY_GATE.md)
- [Machine-readable release manifests](release_manifests/)

## Availability boundary

The two compact table-reproduction packages are built and hash-bound but supplied
separately because frozen predictions do not belong in normal Git history. The larger
training/checkpoint archives remain specifications only. Hosted-reader regeneration is
nondeterministic and requires external credentials; table reproduction and verification
of frozen answers do not.

Citation metadata is in [CITATION.cff](CITATION.cff). First-party code is MIT-licensed;
the vendored GNN-RAG components retain their upstream licensing constraints.
