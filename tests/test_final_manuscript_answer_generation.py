from __future__ import annotations

import json
from collections import Counter

import pytest

from generation import run_final_manuscript_answer_generation as runner


def test_frozen_configuration_set_is_exact():
    assert runner.CONFIGURATIONS == (
        ("cross_encoder", "k1"),
        ("final_sageqa", "k1"),
        ("final_sageqa", "adaptive"),
        ("gold_support", "oracle"),
    )


def test_cache_key_reuses_identical_inputs_across_method_labels():
    base = {
        "dataset": "HotpotQA", "domain": "text", "example_id": "x",
        "question": "q", "support_sha256": "abc", "method": "cross_encoder",
        "setting": "k1",
    }
    other = {**base, "method": "final_sageqa", "setting": "adaptive"}
    assert runner._input_key(base) == runner._input_key(other)


def test_persisted_preflight_has_required_exact_counts_and_no_calls():
    report = json.loads((runner.DEFAULT_OUTPUT / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["answer_evaluation_examples"] == 4249
    assert report["support_joint_evaluation_examples"] == 3509
    assert report["new_conditions"] == 4
    assert report["raw_prediction_rows"] == 16256
    assert report["unique_reader_inputs"] == 8284
    assert report["deterministic_proof_first_inputs"] == 1169
    assert report["expected_openai_calls"] == 7115
    assert report["gold_support_answer_examples"] == 3509
    assert report["gold_support_undefined_support_examples_excluded"] == 740
    assert report["openai_calls_made"] == 0
    assert report["old_metrics_denominator_recomputation_required"] is False


def test_persisted_reader_input_bundle_matches_freeze():
    report = json.loads((runner.DEFAULT_OUTPUT / "preflight_report.json").read_text(encoding="utf-8"))
    assert runner.production.sha256(runner.DEFAULT_OUTPUT / "frozen_reader_inputs.jsonl") == report[
        "frozen_reader_inputs_sha256"
    ]


def test_reader_input_bundle_excludes_cross_encoder_adaptive_and_empty_gold_support():
    rows = runner.production.load_jsonl(runner.DEFAULT_OUTPUT / "frozen_reader_inputs.jsonl")
    counts = Counter((row["method"], row["setting"]) for row in rows)
    assert counts == {
        ("cross_encoder", "k1"): 4249,
        ("final_sageqa", "k1"): 4249,
        ("final_sageqa", "adaptive"): 4249,
        ("gold_support", "oracle"): 3509,
    }
    assert all(row["selected_support"] for row in rows if row["method"] == "gold_support")


def test_evaluation_requires_generation_freeze(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.evaluate_phase(tmp_path)
