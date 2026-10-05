import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.symbolic_coefficients_original_v1.original_symbolic_features import (
    ORIGINAL_PROOF_COEFFICIENTS,
    ORIGINAL_TEXT_COEFFICIENTS,
    candidate_identity,
    proof_features,
    score_proof_features,
    score_text_features,
    text_features,
)
from experiments.symbolic_coefficients_original_v1.optimize import (
    coordinate_search,
    evaluate_coefficients,
    load_domain_cache,
)
from experiments.symbolic_coefficients_original_v1.prepare_dev_cache import (
    assert_dev_only_path,
    build_cached_example,
    canonical_json,
    sha256,
    validate_cache,
    validate_inputs,
    write_json,
    write_jsonl,
)


def text_row(units=None):
    return {
        "dataset": "HotpotQA",
        "question": "Who is younger Alice or Bob?",
        "subgraph_units": units or [
            "SENT::Alice::0::Alice met Bob in 2001.",
            "SENT::Bob::0::Bob was born later than Alice.",
            "SENT::Other::1::Other mentions Alice.",
        ],
        "graph_context_units": ["KG::Alice::knows::Bob"],
    }


def test_original_text_coefficients_reproduce_locked_formula_and_ranking():
    row = text_row()
    features = text_features(row)
    expected = (
        .030 * features["question_title"] + .020 * features["question_overlap"]
        + .020 * features["cross_page"] + .025 * features["bridge_overlap"]
        + .030 * features["kg_connectivity"] + .020 * features["comparison"]
        - .002 * features["is_size_three"] - .004 * features["is_size_four"]
        - .010 * features["size_beyond_four"] - .010 * features["duplicate_page"]
    )
    assert score_text_features(features) == pytest.approx(expected)
    weak = text_features(text_row(["SENT::Other::0::Nothing relevant."]))
    candidates = [(0.0 + score_text_features(features), "a"), (0.0 + score_text_features(weak), "b")]
    assert min(candidates, key=lambda item: (-item[0], item[1]))[1] == "a"


def test_original_proof_coefficients_reproduce_locked_nested_formula():
    row = {
        "dataset": "FamilyOWL_1hop",
        "sparql_query": "ASK { <http://x#alice> <http://x#hasParent> <http://x#bob> }",
        "subgraph_units": ["alice hasParent bob", "InverseObjectProperties(hasParent,isParentOf)", "isParentOf domain Person"],
        "subgraph_size": 3,
    }
    features = proof_features(row, proof_success=True)
    compact = .025 * features["query_property_match"] + .015 * features["query_entity_coverage"] + .020 * features["fact_rule_mix"] - .006 * features["oversize"] - .004 * features["extra_schema"]
    expected = .180 * features["proof_success"] + .030 * features["query_coverage"] + .020 * features["fact_rule_mix"] + .350 * compact - .010 * features["oversize"]
    assert score_proof_features(features) == pytest.approx(expected)


def synthetic_cache(domain="text"):
    feature = {key: 0.0 for key in text_features(text_row())}
    feature["question_title"] = 1.0
    return [{
        "split": "dev", "dataset": "D", "domain": domain, "example_id": "e",
        "gold_explanations": [["gold"]],
        "candidates": [
            {"candidate_identity": "gold-id", "cross_encoder_logit": 0.0, "subgraph_units": ["gold"], "features": feature},
            {"candidate_identity": "bad-id", "cross_encoder_logit": 0.02, "subgraph_units": ["bad"], "features": {**feature, "question_title": 0.0}},
        ],
    }]


def test_changing_coefficients_preserves_identities_and_logits():
    rows = synthetic_cache()
    before = [(c["candidate_identity"], c["cross_encoder_logit"]) for c in rows[0]["candidates"]]
    evaluate_coefficients(rows, "text", replace(ORIGINAL_TEXT_COEFFICIENTS, question_title=.060))
    after = [(c["candidate_identity"], c["cross_encoder_logit"]) for c in rows[0]["candidates"]]
    assert after == before


def test_optimizer_is_deterministic_and_retains_original_without_improvement():
    first = coordinate_search(synthetic_cache(), "text")
    second = coordinate_search(synthetic_cache(), "text")
    assert canonical_json(first) == canonical_json(second)
    flat = synthetic_cache()
    for candidate in flat[0]["candidates"]:
        candidate["cross_encoder_logit"] = 0.0
        candidate["features"] = {key: 0.0 for key in candidate["features"]}
    result = coordinate_search(flat, "text")
    assert result["retained_original_no_improvement"] is True
    assert result["selected_coefficients"] == vars(ORIGINAL_TEXT_COEFFICIENTS)


def test_text_and_ontology_cache_loading_is_independent():
    path = Path("tests/.symbolic_dev_cache_independence.jsonl")
    try:
        text = synthetic_cache("text")[0]
        ontology = {**text, "domain": "ontology", "dataset": "O", "example_id": "o"}
        write_jsonl(path, [text, ontology])
        assert [row["example_id"] for row in load_domain_cache(path, "text")] == ["e"]
        assert [row["example_id"] for row in load_domain_cache(path, "ontology")] == ["o"]
    finally:
        path.unlink(missing_ok=True)


def test_test_paths_are_rejected_before_open():
    with pytest.raises(ValueError, match="TEST path"):
        assert_dev_only_path(Path("data/HotpotQA/test_subgraph_retrieval.jsonl"))
    with pytest.raises(ValueError, match="TEST path"):
        assert_dev_only_path(Path("outputs/test/predictions.jsonl"))
    assert_dev_only_path(Path("outputs/development_runs/dev_predictions.jsonl"))


def test_candidate_identity_is_order_sensitive_and_stable():
    first = candidate_identity({"subgraph_units": ["a", "b"]})
    assert first == candidate_identity({"subgraph_units": ["a", "b"]})
    assert first != candidate_identity({"subgraph_units": ["b", "a"]})


def test_cache_join_validates_every_example_and_candidate_identity():
    row = {**text_row(["SENT::Alice::0::Alice."]), "gold_support_units": ["SENT::Alice::0::Alice."]}
    identity = candidate_identity(row)
    prediction = {
        "dataset": "HotpotQA", "domain": "text",
        "cross_encoder_order": [identity], "cross_encoder_scores": [1.25],
    }
    cached = build_cached_example("HotpotQA", "text", "example", [row], prediction)
    assert cached["candidates"][0]["candidate_identity"] == identity
    assert cached["candidates"][0]["cross_encoder_logit"] == 1.25
    with pytest.raises(ValueError, match="candidate identity mismatch"):
        build_cached_example(
            "HotpotQA", "text", "example", [row],
            {**prediction, "cross_encoder_order": ["wrong"]},
        )


def test_hash_and_manifest_validation_and_reproducible_json():
    cache = Path("tests/.symbolic_dev_feature_cache.jsonl")
    manifest = Path("tests/.symbolic_dev_cache_manifest.json")
    left, right = Path("tests/.symbolic_left.json"), Path("tests/.symbolic_right.json")
    try:
        row = {**synthetic_cache()[0], "schema_version": "sageqa_symbolic_coefficients_original_v1_cache"}
        write_jsonl(cache, [row])
        value = {"split": "dev", "test_accessed": False, "cache_sha256": sha256(cache), "datasets": {"D": 1}}
        write_json(manifest, value)
        assert validate_cache(cache, manifest)["valid"] is True
        write_json(left, {"b": 2, "a": 1})
        write_json(right, {"a": 1, "b": 2})
        assert left.read_bytes() == right.read_bytes()
        cache.write_text(cache.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="hash mismatch"):
            validate_cache(cache, manifest)
    finally:
        for path in (cache, manifest, left, right):
            path.unlink(missing_ok=True)


def test_required_frozen_input_hashes_are_enforced(monkeypatch):
    predictions = Path("tests/.symbolic_dev_predictions.jsonl")
    freeze = Path("tests/.symbolic_dev_prediction_freeze.json")
    candidate = Path("tests/.symbolic_dev_candidates.jsonl")
    source = Path("tests/.symbolic_dev_source_manifest.json")
    try:
        predictions.write_text('{"example_id":"e"}\n', encoding="utf-8", newline="\n")
        candidate.write_text('{"example_id":"e"}\n', encoding="utf-8", newline="\n")
        checkpoint_hash = "checkpoint"
        write_json(freeze, {"split": "dev", "test_accessed": False, "predictions_sha256": sha256(predictions), "checkpoint_sha256": checkpoint_hash})
        write_json(source, {"input_sha256": {"data/D/dev_subgraph_retrieval.jsonl": sha256(candidate)}})
        monkeypatch.setattr("experiments.symbolic_coefficients_original_v1.prepare_dev_cache.EXPECTED_PREDICTIONS_SHA256", sha256(predictions))
        monkeypatch.setattr("experiments.symbolic_coefficients_original_v1.prepare_dev_cache.EXPECTED_PREDICTION_FREEZE_SHA256", sha256(freeze))
        monkeypatch.setattr("experiments.symbolic_coefficients_original_v1.prepare_dev_cache.EXPECTED_SOURCE_MANIFEST_SHA256", sha256(source))
        monkeypatch.setattr("experiments.symbolic_coefficients_original_v1.prepare_dev_cache.EXPECTED_CHECKPOINT_SHA256", checkpoint_hash)
        monkeypatch.setattr(
            "experiments.symbolic_coefficients_original_v1.prepare_dev_cache.expected_inputs",
            lambda data_root: {"D": candidate},
        )
        assert validate_inputs(predictions, freeze, Path("data/dev"), source)["predictions_sha256"] == sha256(predictions)
        candidate.write_text('{"example_id":"changed"}\n', encoding="utf-8", newline="\n")
        with pytest.raises(ValueError, match="candidate hash mismatch"):
            validate_inputs(predictions, freeze, Path("data/dev"), source)
    finally:
        for path in (predictions, freeze, candidate, source):
            path.unlink(missing_ok=True)
