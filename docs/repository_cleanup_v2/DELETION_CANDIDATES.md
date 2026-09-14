# Deletion candidates

No deletion is authorized or performed. SAFE_DELETE means technically re-creatable local state, subject to the stated gates.

| Candidate | Units | Known bytes | Why safe in principle | Required gate |
|---|---:|---:|---|---|
| `.venv/` | 1 | 5,778,488,083 | Re-creatable dependency environment | Final generation process exited; exact environment lock captured; clean replacement environment verified. |
| `.ruff_cache/` | 1 | 9,529 | Re-creatable lint cache | Final process exit. |
| Eleven first-party `__pycache__/` roots | 11 | 2,567,535 | Bytecode regenerated from source | Final process exit; delete exact roots only. |
| `.deepeval/` | 1 | 0 | Empty local tool state | Confirm still empty. |
| `outputs/.test_scratch_adaptive_k/`, `outputs/test_runs/` | 2 | 0 | Empty test roots | Confirm still empty. |
| Twelve ACL-blocked pytest scratch roots | 12 | unknown | Test temporary state; no scientific reference found | Resolve/inspect ownership and exact paths; verify no manifest member; final process exit. |

Known conditional recovery is 5,781,065,147 bytes (5.384 GiB), plus unknown scratch-directory contents. Only 2,577,064 measured bytes are caches independent of the environment.

## Explicit non-candidates

- `.env`: exclude from publication, but do not delete as part of repository cleanup.
- `.git/objects/`: never delete manually.
- duplicate GNN-RAG inputs/checkpoints/results: not safe until consumers and self-contained bundle requirements are resolved.
- zero-byte audit finding JSONL files: meaningful explicit empty finding sets.
- failed/aborted manifests and quarantine records: scientific status evidence.
- `outputs/final_results/final_manuscript_test_end_to_end/`: protected in-progress final artifact.
- any frozen corpus, prediction, checkpoint, metric, lineage, freeze, or manifest file.

## Proposed Phase 2 delete commands

Do not run these now. Phase 2 should first materialize an allowlisted JSON operation file with exact absolute targets, expected file counts/sizes, and pre-deletion hashes. A reviewer should approve that file. Execution should use one PowerShell process and `Remove-Item -LiteralPath` for each verified target; do not use globs or cross-shell path construction. `.venv/` must be a separate, later approval from small caches/scratch.
