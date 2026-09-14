# Cross-repository methodology transfer audit

## Scope and decision

This was one read-only transfer audit of `C:\Users\julie\github\PhD\NeSyGR-QA` and `C:\Users\julie\github\PhD\nesy-reasoner`. No source repository was modified, no model was trained, no DEV or TEST experiment was run, and no inference or tuning was launched. The only writes are the seven requested audit files in this directory.

**Final recommendation: A. KEEP CURRENT FINAL DESIGN. Run the already-implemented hard-pair SAGE-QA training unchanged.**

Neither repository contains a mechanism that clears all six recommendation conditions. The closest match is `nesy-reasoner`'s complete-graph versus required-edge-deletion training with a fixed counterfactual margin. It addresses supervision exposure, but its controlled evidence is only five FamilyOWL cases, the resulting V2 still failed its retrieval gate, and the V3 relation-aware extension regressed. SAGE-QA v2 targets the measured candidate-level failure more directly while retaining the frozen candidate pool, GraphSAGE architecture, objective, budgets, and inference path.

## Audit anchors

- NeSyGR-QA checkout: `515fcd184ad617440bd3d6a53726657b7f479506`; the worktree is heavily dirty, so current code and accepted frozen artifacts are not assumed to have identical lineage.
- nesy-reasoner checkout: `7214161a1f0eb2418ae8426e10b9e607db8429d8`.
- Pending SAGE-QA v2: `training/train_gnn_subgraph_retriever.py::select_reserved_hard_pair` and `sample_weighted_margin_pairs`; frozen runner settings are recorded in `training/run_sageqa_v2_final.py` lines 42-47.
- The prior recorded TRAIN-only invariance gate reports 7,604 eligible examples and 100% reserved-candidate and reserved-pair inclusion. This is memory-derived protocol context: the gate JSON was not present in the current checkout, and this audit did not rerun it.

## Strongest repository evidence and why it does not transfer now

| Mechanism | Repository evidence | Direct SAGE-QA mapping | Decision |
|---|---|---|---|
| All-positive x all-negative pairwise evidence loss | Family k=3 F1 rose 0.3263 to 0.5298, but 2Wiki fell 0.4229 to 0.3714 and both datasets lost k=10 F1/complete coverage | D, but atomic evidence ranking rather than complete-support candidates | Reject; weaker and less targeted than reserved hard pairs |
| Shared semantic listwise evidence loss | Family k=3 F1 0.6842 and 2Wiki 0.4457, but k=10 regressed on both; result labels itself mixed and not a loss-only ablation | C/D | Reject; mixed, changes loss/encoder workflow, and contradicts the frozen listwise-zero protocol |
| Typed-chain lexicographic top-10 reranking | On 40 2Wiki examples, k=3 F1 rose 0.4229 to 0.6829 with 25/15/0 improved/tied/worsened; fixed top-10 membership | E | Do not transfer; narrow text relation vocabulary, small reused development cohort, and SAGE-QA's additive reranker has already survived broader matched diagnostics |
| Typed proof plus relation-witness fallback | On the same 40 2Wiki examples, F1 rose 0.6583 to 0.7904, 14/26/0 paired outcomes | E/F | Do not transfer; explanation construction is not candidate-ranking supervision and is text-specific |
| Query-faithful ontology proof constructor | Frozen matched ontology macro F1 0.554623 versus SAGE-QA 0.337629 on 903 evidence-evaluable questions | E/F | Unsafe as a general transfer: it consumes the exact formal query/target; it is not a gold-free replacement for SAGE-QA ranking |
| Learned question-to-triple linker | Top-12 working graph had recall 1.0 and complete proof coverage 5/5; direct selected-subgraph F1 0.420 and complete proofs 2/5 | B/C | Insufficient: tiny ontology-only evidence and atomic-edge admission does not solve support-set ranking |
| Required-edge deletion plus necessity margin | V1 to V2 on five Family cases: proof-sensitive 1/5 to 4/5, explainer F1 0.18 to 0.373, complete sufficient retrieval 1/5 to 2/5 | D | Closest candidate, but still failed retrieval and does not outperform the already-gated, candidate-level hard-pair reservation scientifically |
| QA-GNN typed relational attention | V3 regressed versus V2: complete proof recall 0.4 to 0.2 and explainer F1 0.373 to 0.1 | C | Reject |
| GNNExplainer / GraphMask | Multiple controlled negative results; direct learned links outperformed GNNExplainer 0.420 to 0.080 F1, and NeSyGR-QA explainer diagnostics failed their gates | F | Reject |

## Special-search findings

No validated, transferable implementation was found for a training curriculum, diversity-preserving candidate pruning, multi-branch candidate composition, support-set hard-negative mining beyond the mechanisms above, or cap-free evidence scoring demonstrated on the relevant SAGE-QA failure. Both repositories contain question-conditioned pruning and path/proof logic, but their evidence is either negative, narrow, oracle-target-dependent, or at a different granularity.

## Bottom line

The measured SAGE-QA bottleneck is not missing atomic evidence supervision, mean-pooling collapse, or a demonstrably defective symbolic term. It is rare exposure of the strongest complete-versus-partial candidate comparison. The pending v2 change is the only audited mechanism that directly guarantees that comparison for every eligible TRAIN example without altering inference or adding a tuning surface. There is no scientifically supported reason to delay or change the final run.
