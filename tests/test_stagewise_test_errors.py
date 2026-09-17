from __future__ import annotations

import json
from pathlib import Path

from evaluation.analyze_stagewise_test_errors import aggregate, classify_support


TEXT_ARGS = {"domain": "text", "metadata": {}, "gold_answer": "answer"}


def test_eight_set_answer_cells_are_distinguished() -> None:
    gold = [["a", "b"]]
    cases = [
        (["a", "b"], "exact"),
        (["a", "b", "c"], "complete_nonminimal"),
        (["a"], "incomplete"),
        (["z"], "disjoint"),
    ]
    rows = []
    for support, outcome in cases:
        result = classify_support(support, gold, **TEXT_ARGS)
        assert result["retrieval_outcome"] == outcome
        rows.append({**result, "answer_correct": True})
        rows.append({**result, "answer_correct": False})
    stats = aggregate(rows)
    assert all(value == 1 for value in stats["cells"].values())


def test_any_complete_reference_alternative_is_success() -> None:
    gold = [["a", "b"], ["x"]]
    exact = classify_support(["x"], gold, **TEXT_ARGS)
    nonminimal = classify_support(["x", "z"], gold, **TEXT_ARGS)
    assert exact["retrieval_outcome"] == "exact"
    assert exact["reference_index"] == 1
    assert nonminimal["retrieval_outcome"] == "complete_nonminimal"
    assert nonminimal["missing_units"] == 0
    assert nonminimal["additional_units"] == 1


def test_reference_distance_prefers_fewer_missing_then_additional() -> None:
    result = classify_support(["a", "x"], [["a", "b", "c"], ["x", "y"]], **TEXT_ARGS)
    assert result["retrieval_outcome"] == "incomplete"
    assert result["reference_index"] == 1
    assert result["missing_units"] == 1
    assert result["additional_units"] == 1


def test_reasoner_validated_alternative_proof_is_not_failure() -> None:
    metadata = {
        "question": "Is Alice a person?",
        "sparql_query": (
            "ASK WHERE { <http://example.org/alice> "
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            "<http://example.org/Person> . }"
        ),
    }
    exact = classify_support(
        ["alice rdf:type Person"],
        [["unrelated rdf:type Person"]],
        domain="ontology",
        metadata=metadata,
        gold_answer="TRUE",
    )
    nonminimal = classify_support(
        ["alice rdf:type Person", "noise relatedTo noise"],
        [["unrelated rdf:type Person"]],
        domain="ontology",
        metadata=metadata,
        gold_answer="TRUE",
    )
    assert exact["retrieval_outcome"] == "exact"
    assert exact["reasoner_validated_alternative"] is True
    assert nonminimal["retrieval_outcome"] == "complete_nonminimal"
    assert nonminimal["additional_units"] == 1


def test_reasoner_does_not_rescue_false_or_select_queries() -> None:
    ask = {
        "question": "Is Alice a person?",
        "sparql_query": (
            "ASK WHERE { <http://example.org/alice> "
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            "<http://example.org/Person> . }"
        ),
    }
    result = classify_support(
        ["alice rdf:type Person"],
        [["unrelated rdf:type Person"]],
        domain="ontology",
        metadata=ask,
        gold_answer="FALSE",
    )
    assert result["retrieval_outcome"] == "disjoint"


def test_persisted_analysis_reconciles_manuscript_claims() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "outputs/final_results/final_manuscript_stagewise_error_analysis/summary.json").read_text(
            encoding="utf-8"
        )
    )
    adaptive = summary["overall"]["final_sageqa/adaptive"]
    assert adaptive["n"] == 3509
    assert sum(adaptive["counts"].values()) == 3509
    assert adaptive["counts"] == {
        "exact": 1118,
        "complete_nonminimal": 851,
        "incomplete": 1293,
        "disjoint": 247,
    }
    assert adaptive["answer_correct_without_complete_count"] == 536

    transitions = summary["paired_transitions"]["sageqa_k1_to_adaptive"]
    assert transitions["incomplete_to_complete_nonminimal"] + transitions["disjoint_to_complete_nonminimal"] == 113
    assert transitions["incomplete_to_exact"] == 34
    assert transitions["exact_to_complete_nonminimal"] == 129

    routing = summary["ontology_routing"]["final_sageqa/adaptive"]
    assert routing["ask_deterministic_proof"] == {
        "n": 515,
        "answer_errors": 0,
        "complete_support_n": 515,
        "complete_support_answer_errors": 0,
        "answer_failure_given_complete_support": 0.0,
    }
    assert routing["ask_llm_no_proof"]["complete_support_answer_errors"] == 0
    assert routing["select_llm"]["complete_support_answer_errors"] == 250

    ontology_hop = summary["ontology_by_hop"]["final_sageqa/adaptive"]
    assert ontology_hop["1hop"]["n"] == 751
    assert ontology_hop["2hop"]["n"] == 758
