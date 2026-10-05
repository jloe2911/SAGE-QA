# Archival package specifications

Two compact table-reproduction packages have been built and verified locally. They are
supplied separately from Git and have not been uploaded or assigned a public URL:

| Package | Files | Uncompressed | ZIP size | SHA-256 |
|---|---:|---:|---:|---|
| `paper_table_artifacts.zip` | 209 | 246,499,230 B | 24,797,197 B | `7e776628d2988222ce0f7770bda868d94ee49fec6ffc705e7b30b1bea76a1b48` |
| `thesis_table_artifacts.zip` | 82 | 507,644,993 B | 48,926,558 B | `18d9faf7b483f59a2ba6367e28db20d2ef7071bd6bb8ca06da4464376f7e7ea1` |

Complete per-file manifests are tracked in `release_manifests/reviewer_artifacts/` and
embedded in each ZIP. These packages close table reproduction only.
`release_manifests/reviewer_artifacts/acceptance_reproduction.json` records the 24 files
produced by the successful fresh-clone acceptance run and their hashes.

The larger full-pipeline archival specifications remain:

- `release_manifests/archive_specs/paper_original.yaml`
- `release_manifests/archive_specs/thesis_final.yaml`

## Original publication package

`paper_original_artifacts` must contain the six processed datasets, raw benchmark/
ontology sources needed by the runner, six historical GraphSAGE checkpoints, frozen
paper predictions/results, evaluation scripts or a bound source archive, a per-file
SHA-256 manifest, and the best available environment/provider record. The accessible
local roots total approximately 10,239,231,608 bytes (9.53 GiB) before archive overhead
or compression.

The 18 processed-data/checkpoint/result directory identities were hash-verified in
Phase 2. This does not prove that `dbdbb507` produced them, so the package must preserve
`candidate_source_revision_unresolved` rather than minting an authoritative claim.

## Final thesis package

`thesis_final_artifacts` must retain Generator D v1, the DistilBERT/cross-encoder and
GraphSAGE/GNN-RAG checkpoints, adaptive policies, hard-pair-v2 baseline exports,
baseline/oracle/main outputs, and the optimized downstream answer/stagewise bundles.
The listed accessible roots total approximately 18,756,382,015 bytes (17.46 GiB) before
archive overhead or compression.

The optimized sub-bundle is intentionally incomplete upstream. Preserve the recorded
hashes for the missing TEST ranking and selected DEV coefficient/policy artifacts. The
retained selected supports already permit official metric replay, but not candidate-
ranking or score reconstruction.

## Full-pipeline publication gate

Before publication, generate a sorted per-file SHA-256 inventory, compute the final
archive SHA-256, extract each package into a clean checkout, run its required contracts,
compare recorded tables, inspect licensing/redistribution constraints, and only then bind
an immutable URL/DOI. Until those steps pass, `download_url` and `archive_sha256` remain
null and the status remains `specification_only_not_published`.
