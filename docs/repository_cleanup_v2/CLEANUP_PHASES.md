# Cleanup phases and proposed Phase 2 operation plan

## Phase 0 — current hard stop

Do not delete, move, rename, train, infer, generate, evaluate, stop processes, or rewrite frozen artifacts. Protect `outputs/final_results/final_manuscript_test_end_to_end/` as an in-progress final bundle.

## Phase 1 — complete and freeze the final answer bundle

This is a scientific workflow, not cleanup, and requires its own authorization if not already running:

1. Confirm generation exits normally; record logs and final row counts.
2. Create/verify generation freeze and prediction hash.
3. Freeze predictions before any gold join.
4. Evaluate from the frozen predictions only.
5. Produce metrics, per-example output, summary, lineage, audit, and a self-excluding artifact manifest.
6. Recompute every member hash and verify cohort/aggregate parity.

## Phase 2 — preservation and release index

1. Take two immutable backups: full checkout including ignored artifacts/Git metadata, and selected scientific bundles.
2. Generate and verify manifests for both backups.
3. Pin the original-paper source commit and decide the detached compatibility worktree.
4. Commit an explicit allowlist of final source/tests/manifests; do not use `git add -A`.
5. Add logical artifact indexes for `thesis_final`, `thesis_baselines`, `thesis_graph_ablation`, `paper_original`, and `development_archive`.
6. Add environment locks and platform notes.
7. Publish and verify separate checksummed release/DOI assets before changing local paths.

## Phase 3 — path abstraction and documentation

1. Add repository-relative/configurable data, checkpoint, policy, and output roots.
2. Keep compatibility wrappers for original-paper commands.
3. Rewrite README so the cross-encoder pipeline is first/default.
4. Add `REPRODUCIBILITY.md`, leakage revision, experiments, artifacts, and repository-structure documents.
5. Do not rewrite frozen lineage to remove absolute paths; add a sidecar path map.

## Phase 4 — archive in small batches

Recommended order:

1. diagnostic outputs plus their scripts;
2. rejected/aborted outputs;
3. development checkpoints after manifest classification;
4. original-paper data/checkpoints/results only if physical movement remains worthwhile;
5. duplicate GNN-RAG payload only after fresh-clone verification.

For each batch: verify source hashes, execute allowlisted moves, search all code/docs/configs/tests for old paths, run focused import/tests, verify destination hashes, and retain a rollback manifest.

## Phase 5 — test and quality repair

1. Track the approved test allowlist.
2. Restore missing historical modules from a pinned commit or archive their tests.
3. Resolve Windows temp ACLs using a demonstrated writable isolated base temp.
4. Run canonical, baseline, and original compatibility test groups separately.
5. Remove unused imports in a separate reviewable source change; update final source hashes if scientifically relevant.

## Phase 6 — exact deletions

Start with small caches only. `.venv/` is a separate later operation after a replacement environment passes. Never combine cache deletion with artifact movement or Git maintenance.

## Proposed Phase 2 command/operation plan

The commands below are a review template, not authorization to run them.

```powershell
# Read-only preflight
git status --short
git rev-parse HEAD
git worktree list --porcelain
git ls-files

# For each approved operation manifest:
# 1. resolve every target with Resolve-Path -LiteralPath
# 2. verify it remains under C:\Users\julie\github\PhD\SAGE-QA
# 3. compare current type, size, count, and SHA-256 with expected values
# 4. abort the entire batch on any mismatch

# Backup and release-copy commands must use explicit literal source/destination
# paths approved by the user. After copying, Get-FileHash -Algorithm SHA256
# must match the source manifest before any move/delete is considered.

# Git staging must be allowlisted, for example individual docs/source paths.
# Never use: git add -A, git clean, git reset, wildcard Remove-Item, or manual
# deletion inside .git/objects.
```

Machine-readable proposed operations and gates are in `cleanup_plan.json`. All operations have `execute: false`.

## Release gate

From clean Windows and Linux checkouts where feasible:

- install from locks;
- download and verify each selected bundle;
- run code-only tests;
- reproduce/verify cross-encoder TEST retrieval without gold leakage;
- validate frozen answer metrics without calling an API;
- reproduce original-paper aggregates from the historical bundle;
- check links, licenses, citation metadata, secret exclusions, and explicit status labels.

Only after this gate should repository publication or destructive cleanup be considered.
