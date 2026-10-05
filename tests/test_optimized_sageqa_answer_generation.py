from __future__ import annotations

import json
from pathlib import Path

import pytest

from generation import run_optimized_sageqa_answer_generation as runner


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _input(*, setting="k1", support=None, sparql_query=None):
    support = ["SENT::Page::0::evidence"] if support is None else support
    metadata = {
        "example_id": "HotpotQA__test__x",
        "dataset": "HotpotQA",
        "question": "Question?",
        "answer_type": "OPEN",
        "task_type": None,
        "sparql_query": sparql_query,
    }
    row = {
        "dataset": "HotpotQA",
        "domain": "text",
        "example_id": metadata["example_id"],
        "method": runner.METHOD,
        "setting": setting,
        "question": metadata["question"],
        "metadata": metadata,
        "selected_support": support,
        "support_sha256": runner.production.canonical_hash(support),
        "model": runner.MODEL_NAME,
    }
    return row


def test_scope_is_exactly_optimized_k1_and_adaptive():
    assert runner.CONFIGURATIONS == (
        ("sageqa_optimized", "k1"),
        ("sageqa_optimized", "adaptive"),
    )
    assert runner.EXPECTED_EXAMPLES == 4249


def test_cache_reuses_method_label_but_preserves_ordered_support_and_reader_metadata():
    base = _input()
    same = {**base, "method": "historical", "setting": "other"}
    assert runner.reader_cache_key(base) == runner.reader_cache_key(same)

    reversed_support = list(reversed(["first", "second"]))
    ordered = _input(support=["first", "second"])
    reversed_row = _input(support=reversed_support)
    assert runner.reader_cache_key(ordered) != runner.reader_cache_key(reversed_row)

    ontology = {
        **base,
        "domain": "ontology",
        "metadata": {**base["metadata"], "sparql_query": "ASK WHERE { <a> <b> <c> }"},
    }
    changed_query = {
        **ontology,
        "metadata": {**ontology["metadata"], "sparql_query": "ASK WHERE { <x> <b> <c> }"},
    }
    assert runner.reader_cache_key(ontology) != runner.reader_cache_key(changed_query)


def test_historical_cache_requires_complete_hash_locked_freeze(tmp_path):
    root = tmp_path / "history"
    prediction = {
        **_input(),
        "generation_status": "complete",
        "predicted_answer": "answer",
        "explanation": "why",
        "raw_response": "{}",
        "answer_source": "gpt-4.1-mini_reader",
    }
    _write_jsonl(root / "predictions.jsonl", [prediction])
    runner.production.write_json(
        root / "generation_freeze.json",
        {
            "status": "complete_frozen",
            "model": runner.MODEL_NAME,
            "prediction_count": 1,
            "predictions_sha256": runner.production.sha256(root / "predictions.jsonl"),
        },
    )
    cache, lineage, duplicates = runner._load_historical_cache([root])
    assert runner.reader_cache_key(prediction) in cache
    assert lineage[0]["prediction_count"] == 1
    assert duplicates == 0

    with (root / "predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="changed after freeze"):
        runner._load_historical_cache([root])


def test_generation_reuses_history_without_calling_readers(tmp_path):
    output, history = tmp_path / "output", tmp_path / "history"
    inputs = [_input(setting="k1"), _input(setting="adaptive")]
    for row in inputs:
        row["reader_cache_contract"] = runner.reader_cache_contract(row)
        row["reader_cache_key_sha256"] = runner.reader_cache_key(row)
    _write_jsonl(output / "frozen_reader_inputs.jsonl", inputs)
    runner.production.write_json(
        output / "preflight_report.json",
        {"frozen_reader_inputs_sha256": runner.production.sha256(output / "frozen_reader_inputs.jsonl"),
         "source_lineage": {"prediction_sha256": "optimized"},
         "configurations": [{"method": runner.METHOD, "setting": setting} for setting in runner.SETTINGS]},
    )
    historical_prediction = {
        **_input(),
        "generation_status": "complete",
        "predicted_answer": "answer",
        "explanation": "why",
        "raw_response": "{}",
        "answer_source": "gpt-4.1-mini_reader",
    }
    _write_jsonl(history / "predictions.jsonl", [historical_prediction])
    runner.production.write_json(
        history / "generation_freeze.json",
        {"status": "complete_frozen", "model": runner.MODEL_NAME, "prediction_count": 1,
         "predictions_sha256": runner.production.sha256(history / "predictions.jsonl")},
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reader must not run for a historical input match")

    freeze = runner.generate_phase(
        output,
        historical_roots=[history],
        text_reader=forbidden,
        ontology_reader=forbidden,
    )
    assert freeze["prediction_count"] == 2
    predictions = runner.production.load_jsonl(output / "predictions.jsonl")
    assert {row["setting"] for row in predictions} == {"k1", "adaptive"}
    assert all(row["historical_reuse_source"] for row in predictions)


def test_evaluation_cannot_touch_gold_without_complete_prediction_freeze(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.evaluate_phase(tmp_path)


def test_persisted_preflight_is_gold_free_and_self_consistent():
    report_path = runner.DEFAULT_OUTPUT / "preflight_report.json"
    if not report_path.is_file():
        pytest.skip("Run the local optimized preflight to materialize persisted evidence")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete_gold_free_no_api_calls"
    assert report["examples"] == 4249
    assert report["prediction_rows"] == 8498
    assert report["test_gold_opened"] is False
    assert report["gold_bearing_retrieval_files_opened"] == []
    assert report["openai_calls_made"] == 0
    assert runner.production.sha256(runner.DEFAULT_OUTPUT / "frozen_reader_inputs.jsonl") == report[
        "frozen_reader_inputs_sha256"
    ]
