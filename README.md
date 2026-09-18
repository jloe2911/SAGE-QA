# SAGE-QA

SAGE-QA is a retrieval-augmented question-answering system for multi-hop reasoning over
text benchmarks and OWL-style datasets. This repository preserves several scientific
generations. They share code and data conventions, but they are not interchangeable
protocols.

## Repository status

The repository contains:

- the original published SAGE-QA implementation and results;
- the final thesis pipeline and frozen results;
- thesis baselines and a graph-model ablation;
- external preservation records for scientifically meaningful superseded and development experiments; and
- a post-publication complete Gold Support oracle and provenance correction.

The machine-readable indexes in [`release_manifests/`](release_manifests/) define these
logical generations without moving or rewriting their physical artifacts. See
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for the public artifact map and
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for protocol boundaries.

## Final thesis pipeline

The default current architecture is:

```text
Frozen Generator D candidates
  -> DistilBERT question-candidate cross-encoder
  -> Text-Chain (text) / Proof reranking (ontology)
  -> adaptive support aggregation
  -> support-grounded answer generation
```

Retrieval, reranking, support selection, and reader inputs are frozen before evaluation
gold is joined. The final answer cohort has 4,249 examples; Support and Joint metrics use
the 3,509 examples with defined, non-empty gold support. The primary aggregate is an
equal-dataset macro. This is the final thesis protocol, not a reconstruction of the
published-paper architecture.

## Published-paper implementation

The historical entry point is [`experiments/run_experiments.py`](experiments/run_experiments.py).
It is preserved for historical reproduction and must not be silently routed through the
final thesis logic. Its strongest source-revision candidate is
`dbdbb50708bdc6c686ef82518ec71c1d1bf55985` (`dbdbb507`), but its status remains
`candidate_not_fully_proven`: the ignored processed data, checkpoints, result bytes,
environment, and hosted-reader lineage are not formally bound to that revision.

Historical methodological choices are therefore documented and preserved rather than
retrospectively changed. See
[`docs/ORIGINAL_PAPER_SOURCE_REVISIONS.md`](docs/ORIGINAL_PAPER_SOURCE_REVISIONS.md).

## Results and artifacts

| Logical category | Meaning |
|---|---|
| `thesis_final` | Canonical final-thesis retrieval and end-to-end results |
| `paper_original` | Published/original-paper protocol and historical results |
| `thesis_baselines` | Frozen lexical, clean GNN-RAG, and full-context comparisons |
| `thesis_graph_ablation` | Frozen hard-pair GraphSAGE ablation lineage |
| `thesis_superseded` | Earlier thesis experiments externally preserved for provenance |
| `oracle` | Complete Gold Support oracle and additive provenance correction |
| `development_archive` | External preservation records for rejected, diagnostic, exploratory, incomplete, or DEV-only work |

[`RESULTS.md`](RESULTS.md) maps result groups to their authoritative roots and manifests.
The release indexes, rather than this summary, are authoritative for exact paths and
hashes.

## Quick verification

These checks are local, read-only, and make no model or API calls:

```powershell
python evaluation/check_thesis_final_paths.py

$base = New-Item -ItemType Directory -Force -Path .tmp/reproducibility
$bt = Join-Path $base.FullName ("sageqa-repro-fast-" + [guid]::NewGuid())
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $bt `
  tests/reproducibility --ignore=tests/reproducibility/test_original_paper_contracts.py
```

The first command reports missing external assets explicitly. FAST validates the release
indexes, protected roots, selected frozen hashes, routing, the gold firewall, and oracle
separation. It does not hash all 16.9 GB of Generator D members. See
[`docs/REPRODUCIBILITY_TEST_BASELINE.md`](docs/REPRODUCIBILITY_TEST_BASELINE.md) for
STANDARD and FULL.

## Reproduction

[`REPRODUCE.md`](REPRODUCE.md) separates:

1. verification of already frozen thesis-final results;
2. reproduction of thesis-final retrieval and answer-generation stages; and
3. historical original-paper reproduction.

A fresh clone does not contain all scientific assets. Git-ignored `data/`, `checkpoints/`,
and `outputs/` content, the pinned DistilBERT snapshot, and some baseline-specific assets
must be restored separately. No public location is claimed where one has not been bound
and verified.

## Repository structure

See [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md) for the current physical
layout and [`docs/PATH_PORTABILITY.md`](docs/PATH_PORTABILITY.md) for active-source path
resolution. The logical indexes do not authorize physical moves or archival.

## Text benchmark representation

HotpotQA and 2WikiMultiHopQA use sentence evidence units for scored support and optional KG
triples as graph context. [`TEXT_BENCHMARK_RETRIEVAL.md`](TEXT_BENCHMARK_RETRIEVAL.md) is
the specialized schema and worked-example note.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Unverified publication
identifiers, dates, and release versions are intentionally omitted.

## License and third-party code

First-party repository code is released under the MIT License; see [`LICENSE`](LICENSE).
The adapted GNN-RAG baseline under [`third_party/GNN-RAG/`](third_party/GNN-RAG/) is
third-party code. Its upstream provenance and local changes are described in
[`patches/GNN-RAG-local-changes.md`](patches/GNN-RAG-local-changes.md) and the accompanying
patch. The repository's MIT license must not be read as a relicensing statement for
third-party components; consult their upstream terms before redistribution or use.
