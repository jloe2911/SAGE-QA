# Comparison with pending SAGE-QA hard-pair v2

## Reference modification

The frozen pending modification keeps production v1 unchanged except for one deterministic reservation per eligible TRAIN example:

- highest-target complete candidate;
- strongest gold-free pre-ranked incomplete candidate, with irrelevant fallback only when no partial exists;
- the ordered margin pair appears exactly once;
- candidate and pair budgets, RNG behavior, Generator D, GraphSAGE, frozen BERT, BCE/margin objective, `max_pairs=512`, reranker, and aggregation remain unchanged.

Implementation: `training/train_gnn_subgraph_retriever.py::select_reserved_hard_pair` (line 874) and `sample_weighted_margin_pairs` (line 964). Frozen runner parameters: `training/run_sageqa_v2_final.py:42-47`. The prior recorded TRAIN-only gate covered 7,604 eligible examples with 100% candidate/pair inclusion; this is memory-derived protocol context because the gate JSON was not present in the current checkout, and it was not rerun.

## Serious candidates

| Candidate | More fundamental bottleneck? | Stronger evidence? | Subsumes reservation? | Combine now? | Decision |
|---|---|---|---|---|---|
| nesy-reasoner required-edge deletion + necessity margin | No. Same exposure family, but atomic proof-edge deletion rather than measured complete-vs-hard-partial support-set ranking | No. Five Family TEST cases; partial improvement but retrieval gate failed | No. It does not guarantee exposure to the strongest real partial candidate | No. Would add a new objective and architecture change | Leave v2 unchanged |
| NeSyGR-QA all-pairs pairwise loss | No. SAGE-QA already has margin pairs; the diagnosed defect is pair admission, not absence of a pairwise loss | No. Mixed Family/2Wiki and k=10 regressions | No | No | Leave v2 unchanged |
| NeSyGR-QA shared listwise semantic model | No measured listwise or representation bottleneck; SAGE-QA's corrected listwise and pooling diagnoses do not support it | No. Mixed workflow result and both k=10 regressions | No | No | Leave v2 unchanged |
| nesy-reasoner learned question-to-triple linker | No. Atomic candidate admission differs from support-set ranking; SAGE-QA atomic availability is already generally high | No. Five ontology cases and DEV-selected top-12 | No | No | Leave v2 unchanged |
| nesy-reasoner typed relational attention | No; representation collapse was not supported | No; V3 regressed sharply versus V2 | No | No | Leave v2 unchanged |
| NeSyGR-QA typed-chain reranker / typed proof | No. It acts after atomic ranking on a narrow text relation vocabulary and does not improve exposure to complete support candidates | Positive but only 40 reused 2Wiki DEV examples; not stronger than ten-dataset SAGE diagnostics | No | No. Would replace/stack a reranker after RRF, disagreement union, and Proof gate were rejected | Leave v2 unchanged |
| Exact formal-query ontology proof construction | It can solve ontology proof extraction when an exact target is available, but that is a different problem | Strong ontology result, but target-conditioned and not gold-free as a general transfer | No | No | Leave v2 unchanged |
| GNNExplainer / GraphMask | No; post-hoc explanation is downstream of ranking | No; repeated negative results | No | No | Leave v2 unchanged |

## Why v2 remains the scientifically preferred intervention

1. It is the only mechanism aligned with the measured failure statistic: strongest relevant direct-pair exposure of 3.70% per replayed update and global complete-vs-partial pair coverage of 1.44%.
2. It operates at the actual decision granularity: complete and partial support-set candidates.
3. It changes no inference-time behavior and creates no new hyperparameter.
4. Its implementation invariant has already been checked over all ten TRAIN datasets.
5. Competing mechanisms either failed their own gates, have small/narrow evidence, require oracle targets, or attack bottlenecks that SAGE-QA diagnostics did not support.

## Command implication

No command change is justified. Use the already-frozen `training/run_sageqa_v2_final.py` workflow on the other machine. Do not add flags, losses, attention modules, pruners, rerankers, or explainers before that run.
