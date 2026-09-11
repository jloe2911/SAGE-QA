from experiments.cross_encoder_reranking_dev_v1.run_experiment import (
    candidate_identity,
    select_training_pairs,
    serialize_candidate,
)


def row(units, *, exact=False, complete=False, f1=0.0):
    return {
        "subgraph_units": units,
        "exact_match_any_gold": exact,
        "contains_any_gold_explanation": complete,
        "best_set_f1_to_gold": f1,
    }


def test_text_serialization_preserves_identity_and_order():
    value = serialize_candidate(
        "Who?",
        ["SENT::Page B::7::Second.", "SENT::Page A::2::First."],
        "text",
    )
    assert value == (
        "Question:\nWho?\n\nCandidate evidence:\n"
        "[1] Title: Page B\nSentence index: 7\nText: Second.\n"
        "[2] Title: Page A\nSentence index: 2\nText: First."
    )


def test_ontology_serialization_is_faithful():
    value = serialize_candidate("Is A B?", ["A subClassOf B", "x rdf:type A"], "ontology")
    assert value.endswith("[1] A subClassOf B\n[2] x rdf:type A")


def test_pairs_are_within_example_deterministic_and_ordered_by_target():
    rows = [
        row(["complete"], complete=True),
        row(["partial"], f1=0.5),
        row(["irrelevant"]),
    ]
    first = select_training_pairs("example", rows, max_pairs=8)
    second = select_training_pairs("example", rows, max_pairs=8)
    assert [(candidate_identity(a), candidate_identity(b)) for a, b in first] == [
        (candidate_identity(a), candidate_identity(b)) for a, b in second
    ]
    assert len(first) == 3
    targets = {(a["subgraph_units"][0], b["subgraph_units"][0]) for a, b in first}
    assert ("complete", "partial") in targets
    assert ("complete", "irrelevant") in targets
    assert ("partial", "irrelevant") in targets
