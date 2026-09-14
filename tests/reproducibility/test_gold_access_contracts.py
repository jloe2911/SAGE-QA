from __future__ import annotations

import inspect
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from data_processing.build_2wiki_subgraph_dataset import generate_candidates
from data_processing.retrieval_contracts import clean_text_retrieval_input
from data_processing.text_kg_constructor import KGConstructionConfig
from evaluation import run_production_test_retrieval as retrieval
from generation import run_final_manuscript_answer_generation as final_runner
from generation import run_gold_support_complete_oracle as complete_oracle
from generation import run_production_test_answer_generation as production


def _candidate_payload(example: dict) -> dict:
    return generate_candidates(
        retrieval_example=clean_text_retrieval_input(example),
        split_name="test",
        max_sentences_per_example=8,
        max_subgraph_size=2,
        max_candidates_per_question=32,
        kg_config=KGConstructionConfig(backend="context_only"),
        llm_kg_constructor=None,
        kg_cache=None,
        kg_cache_path=None,
        kg_cache_lock=None,
        seed=42,
    )


def test_2wiki_selection_is_invariant_to_answer_and_gold_evidence():
    base = {
        "id": "wiki-contract-1",
        "question": "Who is Alice's grandparent?",
        "answer": "Carol",
        "supporting_facts": [["Alice", 0], ["Bob", 0]],
        "evidences": [["Alice", "parent", "Bob"], ["Bob", "parent", "Carol"]],
        "context": [
            ["Alice", ["Alice's parent is Bob."]],
            ["Bob", ["Bob's parent is Carol."]],
        ],
    }
    changed = {
        **base,
        "answer": "FORBIDDEN",
        "supporting_facts": [["Alice", 99]],
        "evidences": [["FORBIDDEN", "gold", "ONLY"]],
    }
    first, second = _candidate_payload(base), _candidate_payload(changed)
    for field in ("sentence_pool", "graph_context_units", "candidates"):
        assert first[field] == second[field]
    assert first["gold_available_during_candidate_generation"] is False


def test_retrieval_candidate_loader_rejects_gold_derived_fields():
    payload = (
        '{"example_id":"x","subgraph_units":["u"],'
        '"gold_explanations":[["secret"]],"gold_available_during_candidate_generation":false}\n'
    )
    with patch.object(Path, "open", return_value=StringIO(payload)):
        with pytest.raises(ValueError, match="gold-derived fields"):
            list(retrieval.grouped_jsonl(Path("synthetic-candidate.jsonl")))


def test_generation_metadata_rejects_answers_and_gold_support():
    base = {"example_id": "x", "dataset": "HotpotQA", "question": "Q?"}
    for field in production.FORBIDDEN_GENERATION_FIELDS:
        with pytest.raises(ValueError, match="prohibited generation fields"):
            production._validate_candidate_metadata({**base, field: "secret"}, "HotpotQA")


def test_generation_is_gold_answer_blind_until_frozen_evaluation():
    generation_source = inspect.getsource(final_runner.generate_phase)
    evaluation_source = inspect.getsource(final_runner.evaluate_phase)
    assert "load_gold_answers" not in generation_source
    assert '"gold_answers_opened": False' in generation_source
    assert "load_gold_answers" in evaluation_source
    assert "Prediction population must contain 16,256 rows before gold access" in evaluation_source
    assert "Prediction support differs from frozen reader input" in evaluation_source


def test_oracle_gold_access_is_namespaced_away_from_main_generation():
    assert complete_oracle.DEFAULT_OUTPUT != final_runner.DEFAULT_OUTPUT
    assert complete_oracle.CONDITION == "gold_support_complete/oracle"
    assert final_runner.DEFAULT_OUTPUT.name == "final_manuscript_test_end_to_end"
    assert "run_gold_support_complete_oracle" not in Path(final_runner.__file__).read_text(
        encoding="utf-8"
    )
