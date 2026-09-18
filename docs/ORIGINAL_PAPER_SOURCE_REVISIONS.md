# Original-paper source revision audit

Audit date: 2026-09-14. This report is an index only; no historical source was changed.

## Findings

The current thesis-final source revision is `884480be53c554edae70d3b2d8e781847590aa69` (`main`). It is not the published-paper implementation: later commits added Generator D, the cross-encoder, hard-pair GraphSAGE, adaptive support, finalization, and the complete Gold Support oracle.

The strongest original-paper candidate is `dbdbb50708bdc6c686ef82518ec71c1d1bf55985`, because:

- it is tagged `conference-submission`;
- its commit date is 2026-06-05 and its README says the repository contains the manuscript experiment code;
- the README's main command uses `experiments/run_experiments.py` over the six manuscript datasets and writes `outputs/full_results/`;
- the runner names the original processed roots and GraphSAGE checkpoint roots preserved in this checkout; and
- it is the merge base of the later `post-submission-development` branch.

This is persuasive repository evidence, but not formal proof that `dbdbb507` is the exact source used for every published number. The commit does not contain the ignored processed datasets, checkpoints, `outputs/full_results/`, environment lock, hosted-reader snapshot, or a paper artifact manifest binding those bytes to the tag. Accordingly, this phase records `dbdbb507` as `candidate_not_authoritative` and does not create or move a tag.

`ffee42cced77134614f1615b391c67661ab963d7` is not an original-paper candidate. It is the tip of `post-submission-development`, is explicitly titled "Continue development after conference submission", and descends from `dbdbb507`. The former clean detached worktree at `.worktrees/sageqa-original-eval-compat/` was audited as redundant and removed on 2026-09-18. The revision remains recoverable from both `post-submission-development` and `origin/post-submission-development`; that reachability does not show that it produced the paper. Relative to `dbdbb507`, it changes the runner and shared builders, evaluators, generators, model, and trainer; adds development datasets and comparison tooling; and removes `TEXT_BENCHMARK_RETRIEVAL.md`. It can help compatibility investigation but must not be labeled published source.

## Runner differences

Between `dbdbb507` and `ffee42c`, `experiments/run_experiments.py` changes by 24 additions and 147 deletions. Material changes include treating 2Wiki as text rather than KG, removing current-schema checks and isolated development-run arguments, reverting current GraphSAGE training flags, and narrowing KG/OWL dispatch. Between `dbdbb507` and current HEAD the runner changes again (56 additions, 67 deletions), including gold-free retrieval validation and later safety/formatting changes. Current HEAD therefore cannot silently stand in for the paper runner.

## Reproduction verdict

- `dbdbb507`: source-level paper workflow candidate; exact published-pipeline reproduction is **unproven** until the paper commit and ignored artifact/environment lineage are formally bound.
- `ffee42c`: post-submission compatibility/development revision, recoverable from the preserved local and remote-tracking branches after removal of its redundant compatibility worktree; **not** authoritative published source.
- `884480b`: authoritative source revision for the indexed final-thesis state in this phase; **not** the paper source.

The authoritative original-paper revision remains unresolved because the exact published-result/source binding is not fully proven. Removing the redundant compatibility worktree does not resolve or worsen that evidentiary gap; `dbdbb507` remains the strongest candidate.
