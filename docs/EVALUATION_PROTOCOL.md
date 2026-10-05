# Evaluation protocol

## Official scoring

Official retrieval and end-to-end Support F1 use the canonical annotated-reference
contract implemented by `best_support_scores` in
`evaluation/evaluate_owl_qa_predictions.py`:

1. deduplicate predicted and reference support as sets;
2. compute precision, recall, exact match, and harmonic-mean F1;
3. when annotations contain alternatives, select the alternative with highest F1;
4. score only examples with defined, non-empty support annotations;
5. report unweighted means within each dataset and use the equal mean of ten dataset
   means as the primary thesis aggregate.

The thesis answer population is 4,249. Support and Joint metrics use the 3,509 eligible
examples; 740 ontology examples without defined/non-empty support are excluded. Joint F1
is computed from joint precision and recall, not by multiplying Answer F1 by Support F1.

`evaluation/verify_canonical_retrieval_overlay.py` applies this contract to the optimized
selected supports retained in `frozen_reader_inputs.jsonl`. It validates the source
hashes, all per-dataset cells, and the persisted end-to-end Support F1 cells. The check
does not recreate ranking scores, candidate order, or the missing raw TEST selection.

## Diagnostic scoring

`evaluation/analyze_stagewise_test_errors.py` and
`evaluation/analyze_optimized_stagewise_test_errors.py` classify failure stages. Their
support classification may accept reasoner-validated alternative proofs. That expanded
semantic classification is useful for diagnosis but is not an official annotated-
reference Support F1 and must not replace manuscript retrieval/end-to-end cells.

| Question | Official metric | Stagewise diagnostic |
|---|---|---|
| Did selected support match an annotated reference alternative? | Yes | Included as one signal |
| May an unannotated reasoner-validated proof count as sufficient? | No | Yes |
| Used for manuscript Support/Joint F1? | Yes | No |
| Used to localize retrieval/reasoning/reader failure? | No | Yes |

## Protocol separation

The published-paper protocol historically allowed gold-informed TEST candidate
construction. The revised thesis protocol freezes candidate rankings, selected supports,
and reader inputs before evaluation gold is joined. Paper outputs remain valid historical
observations of their published protocol, but they must not be relabeled as gold-free or
routed through the thesis evaluator to manufacture comparability.
