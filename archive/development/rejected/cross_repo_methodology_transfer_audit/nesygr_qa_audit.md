# NeSyGR-QA audit

## Repository state and pipeline

Audited checkout: `515fcd184ad617440bd3d6a53726657b7f479506`. The checkout contains many staged/untracked manuscript and experiment files and modified sources. Frozen artifacts are therefore treated as evidence tied to their recorded manifests, not automatically to the current source bytes.

The active methodology is atomic-evidence retrieval rather than SAGE-QA support-set candidate ranking:

1. `src/graph/builder.py::FullGraphBuilder` / `FamilyGraphBuilder` or `src/graph/sentence_evidence.py::TwoWikiSentenceGraphBuilder` constructs source graphs.
2. `src/retrieval/candidate_graph.py::CandidateGraphBuilder` and `src/graph/sentence_evidence.py::TwoWikiSentenceCandidateBuilder` use lexical question anchors and complete two-hop expansion. No learned pruning or diversity objective is present.
3. `src/learning/incidence_gnn.py::IncidenceEvidenceGNN` reifies evidence and terms, adds typed directed incidence roles and a question node, performs two mean-aggregation layers, and emits one logit per atomic evidence unit.
4. Training uses balanced pointwise softplus in `src/learning/train_evidence_ranker.py::_balanced_loss`. Separate controlled experiments implement all-pairs pairwise logistic and multi-positive listwise losses.
5. Text symbolic work reorders a frozen prefix and/or constructs typed proofs; ontology work can execute an exact formal-query target.

## Graph construction and candidate admission

`CandidateGraphBuilder.build` (`src/retrieval/candidate_graph.py:137-329`) scores subject, predicate, and object labels against the question, adds context-title anchors, and retains every edge reachable on a deterministic shortest chain of at most two edges. With no anchor it returns the full graph. `TwoWikiSentenceCandidateBuilder.build` (`src/graph/sentence_evidence.py:328-425`) scores sentence/entity/title surfaces, performs an undirected two-edge expansion, and likewise falls back to the full graph.

Leakage classification: **SAFE**. Both accept `ModelInput` and source graphs; gold is joined later. Transfer value is low: these are fixed lexical two-hop expansions without a learned admission scorer, diversity reservation, or a demonstrated solution to SAGE-QA's 320-cap behavior. They could also enlarge rather than control the pool.

## GNN and representation

`IncidenceEvidenceGNN` (`src/learning/incidence_gnn.py`) uses evidence, term, and question nodes; eight directed incidence roles; additive role embeddings; shared mean message aggregation; and one evidence-node classifier. The semantic experiments replace lexical hashes with frozen MiniLM embeddings but do not add candidate-aware support-set attention.

The Family semantic-pointwise artifact (`artifacts/family_semantic_pointwise_rq2_rq3/20260817T074226650608Z/results.json`) achieved k=3 F1 0.659649 but made the downstream explanation F1 worse, 0.877193 to 0.850877. This does not support replacing SAGE-QA's representation, particularly because SAGE-QA's matched mean-pooling-collapse diagnostic was negative.

Leakage classification: **SAFE** when gold is restricted to TRAIN labels. Relation to SAGE-QA: **C**. Confidence preferable: **LOW**.

## Objectives and negatives

### All-pairs pairwise logistic

`src/learning/pairwise_ranking_experiment.py::pairwise_logistic_loss` forms every positive x non-gold pair in each candidate graph and applies `mean(softplus(-(s_pos-s_neg)))`; `_train_pairwise` performs no negative sampling (`:31-48`, `:363-432`).

Evidence: on the frozen 40-example development cohorts, Family k=3 F1 improved 0.326316 to 0.529825, but 2Wiki k=3 fell 0.422857 to 0.371429. k=10 F1 and complete recovery fell on both datasets. This is a proper pointwise-baseline comparison but is mixed and atomic-edge-level.

Leakage classification: **SAFE** for TRAIN-only label construction. Relation: **D**. It neither mines nor reserves the strongest complete-vs-partial support-set competitor. It is weaker than SAGE-QA v2 and would replace the frozen loss workflow. Do not transfer.

### Multi-positive listwise

`src/learning/shared_semantic_listwise_experiment.py::multi_positive_listwise_loss` computes `logsumexp(all logits) - mean(positive logits)` (`:203-219`) using all present gold evidence. Its saved result explicitly says the shared workflow is not a loss-only ablation.

Evidence: Family k=3 F1 improved 0.326316 to 0.684211 and 2Wiki 0.422857 to 0.445714, but k=10 F1 fell by 0.008772 and 0.019643, respectively; decision `B/mixed`. It changes encoder/workflow as well as loss and conflicts with SAGE-QA's frozen `listwise_weight=0` after the corrected-loss audit.

Leakage classification: **SAFE** for TRAIN-only labels. Relation: **C/D**. Requires tuning/change of objective: **YES**. Confidence preferable: **LOW**.

No curriculum or hard-negative miner was found. The pairwise experiment uses all negatives; the listwise experiment uses the full eligible list.

## Text symbolic mechanisms

### Typed-chain lexicographic reranking

`src/retrieval/top10_symbolic_semantic_reranker.py::rerank_frozen_top10` (`:172-294`) rejects gold/answer/support fields, parses a fixed relation chain, extracts source-backed typed facts, and sorts lexicographically by complete-chain membership, continuation compatibility, required-relation compatibility, title compatibility, original rank/score, and two frozen similarities. It preserves the exact input pool.

Evidence: `artifacts/2wiki_top10_top3_reranking/20260817T134142142061Z/metrics.json`, 40 development examples, k=3 F1 0.422857 to 0.682857, complete 5 to 19, and 25/15/0 paired improved/tied/worsened. The same report shows no top-10 change by construction.

Leakage classification: **SAFE** in the saved ranking phase. Relation: **E**. Compatibility: **text only**. It is not preferable now because the evidence is a small reused 2Wiki cohort with a fixed 17-relation extractor, while SAGE-QA's additive reranker has broad ten-dataset evidence and no isolated one-sided term defect. Adding it would create a new reranker and tuning/selection decision after the production protocol was frozen.

### Typed proof plus mandatory relation-witness fallback

`src/explanation/typed_relation_proof.py::{extract_typed_facts, search_typed_proof, construct_typed_relation_proof}` builds a deterministic source-backed relation proof over ten sentences and abstains on unsupported, incomplete, or ambiguous chains. The accepted artifact uses a mandatory saved witness fallback.

Evidence: `artifacts/typed_relation_proof_layer/20260817T130041221776Z/aggregate_metrics.json`, 40 examples, baseline F1 0.658333 to 0.790417, complete recovery 47.5% to 67.5%, 14/26/0 improved/tied/worsened; 27 typed proofs and 13 fallbacks. This is credible for that explanation task, not evidence that it improves candidate-set ranking or SAGE-QA aggregation.

Leakage classification: **SAFE** for construction as recorded; evaluation gold is joined later. Relation: **E/F**. Compatibility: **text only**. Confidence preferable: **LOW**.

### Constrained explanation subset search

`src/explanation/constrained_search.py::ConstrainedExplanationSearch` exhausts bounded subsets when feasible and otherwise uses a fixed beam; it selects the smallest valid subset, then aggregate frozen score, rank sum, and stable IDs. It rejects gold/evaluation-bearing input fields.

Evidence: `artifacts/constrained_explanation_diagnostic/20260815T115637473055Z/metrics.json`. Family tied the current method exactly. 2Wiki F1 fell from 0.580595 to 0.452500 and complete recovery from 32.5% to 15.0%. Relation: **F**. Leakage: **SAFE**. Reject.

## Ontology symbolic constructor

`src/explanation/ontology_query.py::{parse_formal_query, construct_query_faithful_proof}` parses an exact SPARQL `ASK`/`SELECT` target, grounds its subject/predicate/object against available statements, materializes selected OWL/RDFS rules, and returns an asserted-evidence proof.

Evidence is strong but task-conditioned: frozen matched macro F1 0.554623 versus SAGE-QA 0.337629 on 903 evidence-evaluable questions, delta +0.216994, as indexed by `reproducibility/manuscript/FROZEN_RESULTS.md`. This result concerns exact formal-query execution and compact proof construction.

Leakage classification: **UNSAFE for direct transfer** because the exact formal query/target is a gold benchmark field, not a question-only inference signal. Removing it requires a new question-to-formal-query mechanism whose accuracy is not established here. Relation: **E/F**. Compatibility: **ontology only**. It cannot justify changing final SAGE-QA training.

## Explanation models

NeSyGR-QA contains faithful adapters for GNNExplainer and GraphMask, but the saved gates are negative. GNNExplainer's best dropped-gold comparison was 45.5% and the decision was `C/unstable-unfaithful`; GraphMask's best was 36.4% with insufficient control concentration and decision `B/different but insufficient`. These are diagnostics, not supported inference-time selectors. Leakage: **SAFE** for inference masks; relation **F**; reject.

## Mechanisms not found with supporting evidence

No validated multi-branch candidate composer, diversity-preserving pool pruner, candidate-aware support-set attention, hard-negative curriculum/miner, or cap-free support-set scorer was found. Existing path logic is symbolic post-ranking logic, not Generator-D candidate composition.
