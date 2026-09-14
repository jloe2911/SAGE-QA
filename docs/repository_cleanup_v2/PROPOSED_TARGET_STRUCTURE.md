# Proposed target structure

## Smallest safe structure

The repository already has useful functional package boundaries. Before the final answer bundle freezes, prefer a logical release index over physical moves.

```text
README.md
REPRODUCIBILITY.md
RESULTS.md
configs/
  thesis_final/
  paper_original/
docs/
  DATA_LEAKAGE_REVISION.md
  EXPERIMENTS.md
  ARTIFACTS.md
  REPOSITORY_STRUCTURE.md
  archive_notes/
data_processing/
evaluation/
generation/
experiments/
  cross_encoder_reranking_dev_v1/
models/
training/
tests/
third_party/
scripts/
  reproduce_thesis_retrieval.*
  reproduce_thesis_answers.*
  reproduce_paper_original.*
data/
  README.md
  production_generator_d_v1/       # physical path retained initially
checkpoints/
  README.md                         # logical paper_original/thesis_baselines map
outputs/
  README.md                         # logical final/legacy/diagnostic map
release_manifests/
  thesis_final/
  thesis_baselines/
  paper_original/
  development_archive/
```

Do not create duplicate `retrieval/` and `symbolic/` packages merely to match an aesthetic target. Current retrieval code crosses `data_processing/`, `models/`, `training/`, and `evaluation/`; symbolic code is shared by final and historical protocols. Moving it now would multiply import and hash risk.

## Logical artifact map

| Logical class | Current physical roots | Public role |
|---|---|---|
| `thesis_final` | Generator D; cross-encoder DEV/model/policies; frozen A40 retrieval; manuscript retrieval export; final answer bundle | Default architecture and reproduction path |
| `thesis_baselines` | Clean lexical/GNN-RAG/full-context inputs/results | Manuscript comparisons; full-context clearly labelled reference/upper-context condition |
| `thesis_graph_ablation` | Hard-pair GraphSAGE checkpoints/policies/retrieval/answers/audit | Historical graph ablation, not default |
| `paper_original` | Original processed data, ten original checkpoints, `outputs/full_results`, historical runner | Accepted-paper reproduction with leakage-scope warning |
| `development_archive` | Diagnostics, rejected/aborted experiments, development checkpoints | Scientific history and negative results |

## Documentation set

- `README.md`: what SAGE-QA is; the final architecture; one canonical quick start; links to leakage revision, reproduction, artifacts, and original-paper history.
- `REPRODUCIBILITY.md`: environment locks; download/verify; frozen generate/evaluate boundary; resume semantics; Windows/Linux commands; no-gold boundary.
- `docs/DATA_LEAKAGE_REVISION.md`: original leakage mechanism, affected artifacts, Generator D correction, gold-free invariants, and which claims changed.
- `docs/EXPERIMENTS.md`: table of FINAL, BASELINE, GRAPH_ABLATION, ORIGINAL_PAPER, REJECTED, ABORTED, EXPLORATORY runs.
- `docs/ARTIFACTS.md`: logical-to-physical path map, hashes, sizes, status, external URLs/DOIs, licenses.
- `docs/REPOSITORY_STRUCTURE.md`: package ownership and why physical artifact paths may differ from logical release names.
- `docs/archive_notes/`: one status note per moved historical group.

## README canonical flow

The first executable path shown should be:

1. verify/download `generator_d_frozen_v1`;
2. verify the frozen DistilBERT cross-encoder and adaptive policies;
3. reproduce or verify frozen TEST retrieval;
4. freeze support-grounded reader inputs before any gold join;
5. generate/resume predictions through the single-writer runner;
6. evaluate only after predictions are frozen;
7. reproduce manuscript tables from frozen artifacts.

The original `experiments/run_experiments.py` command should move to a clearly labelled “Original paper (historical protocol)” section.

## Physical move policy

Physical `paper_original/`, `thesis_final/`, and `development_archive/` directories are optional. If used, moves must be staged one protocol at a time with:

- pre-move checksums and immutable backup;
- repository-relative config roots;
- compatibility wrappers for old commands;
- full reference search for old paths;
- post-move checksum equality;
- focused import/test verification;
- fresh-clone release verification.

Until those gates pass, keep current physical paths and expose the clean structure through manifests and documentation.
