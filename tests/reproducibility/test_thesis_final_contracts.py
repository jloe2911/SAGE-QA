from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from evaluation import finalize_final_manuscript_end_to_end as finalizer
from evaluation import run_production_test_retrieval as retrieval
from evaluation.adaptive_support_aggregation import fixed_k_support_aggregate
from generation import run_final_manuscript_answer_generation as runner


def test_thesis_final_entry_points_import_and_resolve_frozen_policy(repo_root: Path):
    index = yaml.safe_load(
        (repo_root / "release_manifests/thesis_final/index.yaml").read_text(encoding="utf-8")
    )["bundles"][0]
    freeze = json.loads(runner.DEFAULT_SELECTION_FREEZE.read_text(encoding="utf-8"))

    assert callable(runner.preflight_phase)
    assert callable(runner.generate_phase)
    assert callable(runner.evaluate_phase)
    assert callable(retrieval.score_and_freeze)
    assert callable(finalizer.validate_freeze_and_new_rows)
    assert index["source_revision"] == "884480be53c554edae70d3b2d8e781847590aa69"
    assert index["checkpoint_root"] == (
        "outputs/development_runs/question_candidate_cross_encoder_v1/checkpoint_final"
    )
    assert freeze["checkpoint_path"] == (
        "outputs/development_runs/question_candidate_cross_encoder_v1/checkpoint_final/"
        "model.safetensors"
    )
    assert freeze["gold_accessed_before_freeze"] is False
    for policy_root in index["policy_root"]:
        assert (repo_root / policy_root / "adaptive_k_config.json").is_file()


def test_text_and_ontology_datasets_route_to_distinct_final_logic():
    routes = {name: (domain, mode) for name, _data, _checkpoint, domain, mode, _gold in retrieval.DATASETS}
    assert routes["HotpotQA"] == ("text", "sageqa_text_chain")
    assert routes["2WikiMultiHopQA"] == ("text", "sageqa_text_chain")
    assert len(routes) == 10
    assert all(
        route == ("ontology", "sageqa_proof")
        for dataset, route in routes.items()
        if dataset not in {"HotpotQA", "2WikiMultiHopQA"}
    )


def test_support_aggregation_and_identity_are_deterministic():
    candidates = [
        {"rank": 1, "subgraph_units": ["A", "B"]},
        {"rank": 2, "subgraph_units": ["B", "C"]},
        {"rank": 3, "subgraph_units": ["D"]},
    ]
    first = fixed_k_support_aggregate(candidates, k=3)
    second = fixed_k_support_aggregate(candidates, k=3)
    assert first == second
    assert first["support_units"] == ["A", "B", "C", "D"]
    assert runner.production.canonical_hash(first["support_units"]) == (
        "503f248273fbb6c6e85a2ff9ee75283178b009f3ca92b4c98a72958393bccd1a"
    )


def test_persisted_preflight_consumes_the_frozen_retrieval_bundle_without_calls(repo_root: Path):
    report = json.loads((runner.DEFAULT_OUTPUT / "preflight_report.json").read_text(encoding="utf-8"))
    freeze = json.loads((runner.DEFAULT_OUTPUT / "generation_freeze.json").read_text(encoding="utf-8"))
    inputs = runner.production.load_jsonl(runner.DEFAULT_OUTPUT / "frozen_reader_inputs.jsonl")
    counts = Counter((row["method"], row["setting"]) for row in inputs)

    assert report["status"] == "complete_no_api_calls"
    assert report["openai_calls_made"] == 0
    assert report["source_paths"]["frozen_selections"].replace("\\", "/") == str(
        runner.DEFAULT_SELECTIONS.relative_to(repo_root)
    ).replace("\\", "/")
    assert freeze["status"] == "complete_frozen"
    assert freeze["gold_answers_opened"] is False
    assert counts == {
        ("cross_encoder", "k1"): 4249,
        ("final_sageqa", "k1"): 4249,
        ("final_sageqa", "adaptive"): 4249,
        ("gold_support", "oracle"): 3509,
    }


def test_main_pipeline_does_not_import_or_configure_the_complete_oracle():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "run_gold_support_complete_oracle" not in source
    assert "gold_support_complete" not in source
    assert ("gold_support_complete", "oracle") not in runner.CONFIGURATIONS
    assert runner.DEFAULT_OUTPUT.name == "final_manuscript_test_end_to_end"
