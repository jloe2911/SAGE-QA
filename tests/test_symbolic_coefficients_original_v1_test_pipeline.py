import json
from pathlib import Path

import pytest

from experiments.symbolic_coefficients_original_v1.original_symbolic_features import (
    ProofCoefficients,
    TextCoefficients,
)
from experiments.symbolic_coefficients_original_v1.test_pipeline import (
    _legacy_gold_explanations,
    candidate_identity,
    rank_selected,
    safe_inference_row,
    serialize_candidate,
    validate_candidate_fixture,
)


def test_selected_coefficients_drive_text_ranking_without_gold():
    rows = [
        {
            "dataset": "HotpotQA",
            "question": "Who is Alice?",
            "subgraph_units": ["SENT::Alice::0::Alice is a researcher."],
            "graph_context_units": [],
        },
        {
            "dataset": "HotpotQA",
            "question": "Who is Alice?",
            "subgraph_units": ["SENT::Other::0::Nothing relevant."],
            "graph_context_units": [],
        },
    ]
    ranked = rank_selected(
        rows,
        [0.0, 0.0],
        "text",
        TextCoefficients(question_title=0.5),
        ProofCoefficients(),
    )
    assert ranked[0]["candidate_identity"] == candidate_identity(rows[0])
    assert "gold" not in {key.lower() for key in ranked[0]}


def test_safe_projection_removes_dev_labels_and_serialization_is_frozen():
    raw = {
        "example_id": "e",
        "dataset": "HotpotQA",
        "question": "Q?",
        "subgraph_units": ["SENT::T::2::Evidence."],
        "gold_support_units": ["secret"],
        "label": 1,
    }
    projected = safe_inference_row(raw)
    assert set(projected) == {"example_id", "dataset", "question", "subgraph_units"}
    assert serialize_candidate("Q?", projected["subgraph_units"], "text") == (
        "Question:\nQ?\n\nCandidate evidence:\n"
        "[1] Title: T\nSentence index: 2\nText: Evidence."
    )


def test_dev_fixture_validation_never_requires_test():
    path = Path("tests/.symbolic_test_pipeline_dev_fixture.jsonl")
    rows = [
        {"example_id": "dev-1", "dataset": "HotpotQA", "question": "Q", "subgraph_units": ["a"], "label": 0},
        {"example_id": "dev-1", "dataset": "HotpotQA", "question": "Q", "subgraph_units": ["b"], "label": 1},
        {"example_id": "dev-2", "dataset": "HotpotQA", "question": "R", "subgraph_units": ["c"], "gold_support_units": ["c"]},
    ]
    try:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        assert validate_candidate_fixture(path, expected_examples=2, expected_rows=3) == (2, 3)
        with pytest.raises(ValueError, match="count mismatch"):
            validate_candidate_fixture(path, expected_examples=3, expected_rows=3)
    finally:
        path.unlink(missing_ok=True)


def test_v100_gold_reader_accepts_only_legacy_explanation_fields():
    qa = {"Explanations": [["b owl:inverseOf a", "TAG:M", "p Domain C"]], "answer_explanations": [["new-schema"]]}
    assert _legacy_gold_explanations(qa) == [["InverseObjectProperties(a,b)", "p domain C"]]
    assert _legacy_gold_explanations({"Minimum Explanation": ["fallback"]}) == [["fallback"]]
    assert _legacy_gold_explanations({"answer_explanations": [["new-schema"]]}) == []
