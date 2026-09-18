# Current repository structure

This is the final physical layout after Phase 8 on 2026-09-18.

| Current path | Current role |
|---|---|
| `data/` | Six reported-paper processed datasets, retained raw runtime inputs, and frozen Generator D under `data/production_generator_d_v1/` |
| `checkpoints/` | Six reported-paper GraphSAGE roots, thesis hard-pair GraphSAGE, and clean GNN-RAG baseline |
| `experiments/` | Historical paper runner and final cross-encoder protocol/runner |
| `evaluation/` | Retrieval, aggregation, audits, exporters, and finalizers |
| `generation/` | Historical and final-thesis answer-generation entry points |
| `models/` | Retained shared model and symbolic-composer code |
| `outputs/full_results/` | Six reported-paper result roots and aggregate files |
| `outputs/final_results/` | Frozen thesis, baseline, graph-ablation, oracle, and correction results |
| `outputs/development_runs/` | Only retained cross-encoder and hard-pair policy/runtime inputs, plus one exact Generator D validation snapshot |
| `outputs/audits/` | Read-only scientific audits, including final TEST evaluation audit |
| `release_manifests/` | Logical indexes, path sidecar, hidden-material classification, and protection rules created by this phase |
| `docs/` | Audit and reviewer-facing documentation |
| `third_party/GNN-RAG/` | Vendored upstream baseline with local patch documentation |

Git ignore/exclude rules still hide large scientific roots and historical exact local-exclude entries. `release_manifests/hidden_untracked_classification.yaml` is retained as a pre-Phase-8 snapshot; it is not a live-path index.

The compatibility worktree was audited as redundant and removed on 2026-09-18; `ffee42c` remains recoverable from the preserved `post-submission-development` references. The local `.venv/` remains outside cleanup scope. Three root pytest scratch directories remain ACL-blocked and are reported as operational blockers rather than scientific repository content.
