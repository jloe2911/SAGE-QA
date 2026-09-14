# nesy-reasoner audit

## Repository state and pipeline

Audited checkout: `7214161a1f0eb2418ae8426e10b9e607db8429d8`.

The repository explores a QA-GNN-style pipeline over reified atomic RDF/text-extracted triples: context-only graph adaptation, question-grounded working-graph construction, typed/query-conditioned GNN reasoning, post-hoc explanation, connected subgraph selection, and symbolic verification. Its strongest experiments are deliberately small feasibility studies, usually 100 TRAIN / 20 DEV / five TEST Family examples or 12/4/4 non-held-out 2Wiki cases.

## Graph construction and leakage boundary

- `nesy_reasoner/adapters/family.py::FamilyGraphAdapter` parses RDF context. Gold answers, SPARQL, and minimum explanations remain outside model input.
- `nesy_reasoner/adapters/two_wiki.py::TwoWikiGraphAdapter` is context-only but its transparent extraction baseline has poor complete-proof coverage; the frozen V1 2Wiki gate passed only 1/5 TEST cases.
- Historical `auto` / `llm_with_provided` data paths can use provided benchmark evidence and are **UNSAFE** for transfer. Only context-only paths are admissible.

## Question-conditioned graph pruning

`nesy_reasoner/methods/qa_gnn_components.py::QAGNNWorkingGraphBuilder` (`:11-76`) scores question overlap with readable subject/relation/object terms, expands two relational hops, ranks by distance/overlap/edge ID, and caps at 128 edges. It explicitly receives no answer, query, or gold explanation.

Leakage: **SAFE**. Relation: **B**. Compatibility: **both**. Evidence: Family fixed architecture has working-graph candidate coverage 1.0 on five cases, but 2Wiki context extraction coverage is 0.2 and no ablation shows this pruning improves over SAGE-QA admission. The fixed cap reproduces rather than solves a cap/admission problem. Transfer requires a new cap and grounding protocol: tuning **YES**, frozen protocol invalidated **YES**, confidence **LOW**.

## Learned question-to-triple linker

`nesy_reasoner/learning/incidence_gnn.py::EvidenceSupervisedRelationalGNN` uses typed relational attention, node semantic types, question features, and a lexical prior to emit one logit per atomic triple. `nesy_reasoner/experiments/family_question_link_gnnexplainer.py::{train_linker,_working_cases}` trains with balanced positive/negative softplus and retains the top 12 learned links as a compact graph, adding explicit question-to-triple edges.

Evidence: on the same five Family cases, linker top-12 recall and complete-proof coverage were 1.0/5-of-5; the direct-link connected selection achieved F1 0.420, complete proofs 2/5, and symbolic sufficiency 2/5. GNNExplainer then degraded F1 to 0.080 and complete proofs to 0/5.

Leakage: **SAFE** if proof labels are TRAIN-only; gold is absent at inference. Relation: **B/C**. Compatibility: implemented for ontology; conceptually both. This is atomic-edge selection, not support-set scoring, and its only positive evidence is five cases with a DEV-selected top-12 cutoff. It requires architecture and admission-policy changes and tuning. Confidence preferable: **LOW**.

## Complete-versus-required-edge counterfactual training

`nesy_reasoner/experiments/family_small.py::_support_variants` (`:47-129`) creates a positive complete graph and one negative per required proof edge by replacement (V1) or deletion (V2/V3). `train_reasoner` (`:230-349`) uses balanced graph-level BCE and, for V2/V3, `relu(margin - full_logit + deletion_logit)` with fixed margin 1.0. `QAGNNProofNecessityModel` removes the direct question/context classifier shortcut and scores only an attention-pooled evidence graph.

Original purpose: force graph-sufficiency predictions to depend causally on every required proof edge so a post-hoc explainer has a meaningful target.

Evidence: on identical five Family TEST IDs, V1 to V2 increased positive required-edge-deletion behavior from 1/5 to 4/5, GNNExplainer F1 0.18 to 0.373333, and complete/sufficient retrieval 1/5 to 2/5. It nevertheless failed the predeclared retrieval gate, exact proof retrieval remained 0/5, and selected-DEV margin satisfaction was only 0.263.

Leakage: **SAFE** only as TRAIN supervision. Relation: **D**, with secondary **C**. Compatibility: ontology as demonstrated; conceptually both if trustworthy gold TRAIN supports exist. Compute **MEDIUM**, tuning **YES** for transfer, frozen protocol invalidated **YES**.

Direct comparison to SAGE-QA v2: it addresses the same broad exposure problem but at the wrong granularity. It fabricates one deletion per gold atomic proof edge, changes the graph-sufficiency architecture and loss, and was not validated on candidate support sets. SAGE-QA v2 instead guarantees the actually diagnosed complete-candidate versus strongest gold-free partial-candidate comparison without new loss, architecture, budget, or inference behavior. It neither has stronger evidence nor subsumes reserved candidate-level pairs. Do not combine before the final run.

## Relation-aware/query-conditioned attention and candidate-aware pooling

`nesy_reasoner/learning/incidence_gnn.py::QAGNNRelationalAttention` conditions messages on incidence direction, evidence semantic type, source/target node types, and the question. `QAGNNRelationalProofNecessityModel` adds typed attention and attention-pools evidence nodes against the question. `QAGNNRelationalCandidateScorer` additionally scores answer candidates from question, context node, and pooled graph.

Evidence is negative:

- Fixed V3 changed only typed ABox/TBox/extracted node types plus relational attention and regressed versus V2: complete proof recall/sufficiency 0.4 to 0.2, positive-deletion examples 0.8 to 0.6, and explainer F1 0.373333 to 0.1.
- The 2Wiki answer-conditioned feasibility result selected zero correct answer candidates on four non-held-out test cases and had retrieval F1 0.0.

Leakage: relational attention itself **SAFE**; answer-candidate training is **ADAPTABLE** but is an answer-ranking task and includes gold-answer labels during training. Relation: **C**. Compatibility: conceptually both. SAGE-QA's representation-collapse test was already negative, so this does not address a measured need. Reject.

## Question-rooted connected aggregation

`nesy_reasoner/methods/question_rooted.py::QuestionRootedSubgraphBuilder` (`:14-94`) greedily grows the highest-saliency edge incident to question-grounded/reached entities, permits multiple question-rooted branches for comparison questions, and stops at a maximum edge count or relative threshold.

Leakage: **SAFE**. Relation: **F**. Compatibility: both. Evidence is weak/negative: in the 2Wiki 12/4/4 feasibility study, the GNNExplainer path obtained F1 0.083333 and no complete proof; even direct GNN selection obtained F1 0.391667 and complete proof rate 0.25. The Family learned-link study recovered only 2/5 complete proofs despite 5/5 availability. It requires selecting max edges and threshold. Reject.

## GNNExplainer and GraphMask

`nesy_reasoner/explainers/pyg_gnn_explainer.py` maps PyG node/edge masks back to reified triples. `nesy_reasoner/explainers/graphmask.py` learns question/triple gates to preserve a frozen model prediction under a sparsity penalty.

Evidence is consistently negative for proof retrieval:

- Direct learned link scores versus GNNExplainer: F1 0.420 versus 0.080; complete proofs 2/5 versus 0/5.
- After stronger verified counterfactual training, GraphMask F1 was 0.0 and GNNExplainer F1 0.133333; neither recovered a complete proof.
- V1 and V2 fixed architectures both failed explainer retrieval gates.

Leakage: **SAFE** at inference, although training and checkpoint selection must remain gold-free. Relation: **F**. Compute **HIGH** for per-example GNNExplainer and **MEDIUM/HIGH** for GraphMask. Tuning **YES**. Reject.

## Symbolic completion and verifier

`nesy_reasoner/symbolic/completion.py::complete_minimal_proof` combines retrieved ABox units with available TBox rules, verifies a supplied target, and performs leave-one-out minimization. `nesy_reasoner/symbolic/verifier.py::verify_candidate` forward-chains inverse, subproperty, domain, and range rules.

This is useful as an evaluation/verifier component but **UNSAFE as a direct gold-free SAGE-QA inference replacement** when the supplied target triple comes from a benchmark SPARQL/query answer. A question-derived target parser would be a new unvalidated mechanism. SAGE-QA has already rejected its own ontology Proof gate on clean DEV, so no transfer is justified.

## Mechanisms absent or unsupported

No demonstrated multi-branch candidate composer, diversity-preserving admission, support-set hard-negative miner/curriculum, or cap-free graph scorer was found. The repository's same-size V1 corruptions and V2 deletions are constructed from reference proof edges, not mined gold-free partial support candidates.
