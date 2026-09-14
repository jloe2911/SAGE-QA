# Final recommendation

## A. KEEP CURRENT FINAL DESIGN

**Run the already-implemented hard-pair SAGE-QA training unchanged.**

No audited mechanism satisfies all six strict recommendation conditions. In particular, none has stronger evidence that it fixes SAGE-QA's measured candidate-level complete-versus-hard-partial ranking failure without tuning or protocol expansion.

The closest mechanism, `nesy-reasoner`'s required-proof-edge deletion margin, is not preferable: it was demonstrated on only five FamilyOWL cases, still failed its retrieval gate, acts on artificial atomic-edge deletions rather than real support-set competitors, and requires both an objective and architecture change. Relation-aware attention then regressed. NeSyGR-QA's pairwise/listwise objectives were mixed, its constrained search and explainers were negative, and its positive symbolic mechanisms are narrow post-ranking methods or depend on an exact formal-query target.

The pending hard-pair reservation is already the least-assumptive intervention that directly fixes the observed exposure defect. It guarantees one real complete-versus-strongest-partial comparison per eligible TRAIN example while preserving all frozen budgets, loss definitions, architecture, Generator D, reranking, aggregation, and inference behavior.

Therefore:

- do not replace any SAGE-QA mechanism;
- do not add a second mechanism to v2;
- do not change the other-machine command;
- do not tune after training or DEV results appear;
- proceed with the frozen ten-dataset hard-pair run.
