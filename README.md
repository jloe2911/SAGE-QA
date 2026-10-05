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

Download the two frozen artifact packages from the [SAGE-QA reproducibility release]([RELEASE_URL](https://github.com/jloe2911/SAGE-QA/releases/tag/v1.0-reproducible)):

- `paper_table_artifacts.zip`  
  SHA-256: `7e776628d2988222ce0f7770bda868d94ee49fec6ffc705e7b30b1bea76a1b48`

- `thesis_table_artifacts.zip`  
  SHA-256: `18d9faf7b483f59a2ba6367e28db20d2ef7071bd6bb8ca06da4464376f7e7ea1`

Place both files under:

```text
release/reviewer_artifacts/
```

Then run:

```bash
python scripts/restore_reviewer_artifacts.py release/reviewer_artifacts/paper_table_artifacts.zip
python scripts/restore_reviewer_artifacts.py release/reviewer_artifacts/thesis_table_artifacts.zip
python scripts/reproduce_reported_tables.py --output-dir reproduced_tables
```

The restore commands validate every file against the embedded per-file manifests.

The reproduction command:

- reproduces the reported original-paper tables from the frozen historical outputs;
- regenerates and verifies the hard-pair-v2 thesis retrieval and end-to-end tables;
- verifies the optimized thesis Support P/R/F1 overlay from the retained frozen selected supports; and
- writes hashes of the reproduced outputs to `reproduced_tables/REPRODUCTION_MANIFEST.json`.

The original-paper path reproduces the historical gold-informed candidate-generation protocol. The thesis path corresponds to the revised leakage-free evaluation. Full model retraining and hosted-reader regeneration are documented separately in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
