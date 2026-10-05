# Reproducing SAGE-QA

The canonical reviewer instructions are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). This compatibility entry point is
retained so older links do not route reviewers to stale commands.

The repository contains two distinct paths:

- the original published-paper protocol, with historically gold-informed TEST candidate
  construction and unverified exact source/result provenance;
- the revised thesis protocol, whose TEST rankings and reader inputs were frozen before
  evaluation gold was joined.

From a fresh clone, restore the two separately supplied compact packages and run
`scripts/reproduce_reported_tables.py` as documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Larger full-pipeline reproduction
still requires the unpublished packages specified in
[release_manifests/archive_specs](release_manifests/archive_specs/).
