# Current repository structure

This is the physical layout observed on 2026-09-14. It is not a proposed future tree.

| Current path | Current role |
|---|---|
| `data/` | Original processed datasets, raw sources, development data, and frozen Generator D under `data/production_generator_d_v1/` |
| `checkpoints/` | Original-paper GraphSAGE, thesis GraphSAGE, clean GNN-RAG baseline, and development checkpoints; ignored by Git |
| `experiments/` | Historical runner, cross-encoder protocol/runner, and ignored experiment configurations |
| `evaluation/` | Retrieval, aggregation, audits, exporters, and finalizers |
| `generation/` | Historical and final-thesis answer-generation entry points |
| `models/` | Tracked active model code plus ignored symbolic/development modules |
| `outputs/full_results/` | Historical published-paper result layout |
| `outputs/final_results/` | Frozen thesis, baseline, graph-ablation, oracle, and correction results |
| `outputs/development_runs/`, `outputs/diagnostics/`, `outputs/final_model_development/` | accepted intermediate, rejected, exploratory, and diagnostic results |
| `outputs/audits/` | Read-only scientific audits, including final TEST evaluation audit |
| `release_manifests/` | Logical indexes, path sidecar, hidden-material classification, and protection rules created by this phase |
| `docs/` | Audit and reviewer-facing documentation |
| `third_party/GNN-RAG/` | Vendored upstream baseline with local patch documentation |
| `.worktrees/sageqa-original-eval-compat/` | Clean detached local worktree at post-submission revision `ffee42c`; ignored and not a release bundle |

Git ignore/exclude rules currently hide large scientific roots plus 206 exact local-exclude entries. Of those exact entries, 122 exist and 84 are stale/absent. `release_manifests/hidden_untracked_classification.yaml` classifies the entire local-exclude set without changing `.git/info/exclude` or Git tracking.

The repository also contains local environments, caches, logs, notebooks, distribution output, and ACL-blocked pytest scratch directories. Those are current physical facts; this phase does not delete or relocate them.
