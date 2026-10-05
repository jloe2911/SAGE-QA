from __future__ import annotations

import json
from pathlib import Path

from evaluation import analyze_optimized_stagewise_test_errors as optimized


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/final_results/symbolic_coefficients_original_v1_stagewise_error_analysis"


def _summary():
    return json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))


def test_scope_and_frozen_hashes_are_exact() -> None:
    assert optimized.CONDITIONS == (
        ("sageqa_optimized", "k1", "Optimized SAGE-QA ($k=1$)"),
        ("sageqa_optimized", "adaptive", "Optimized SAGE-QA (adaptive)"),
    )
    summary = _summary()
    assert summary["denominators"] == {
        "answer_questions": 4249,
        "support_bearing_questions_per_method": 3509,
        "recomputed_sageqa_rows": 7018,
    }
    assert summary["regression_checks"]["saved_answer_score_mismatches"] == 0
    assert summary["source_hashes"]["optimized_answer_predictions.jsonl"] == (
        "712fce04ed8d25bbee55e34a3d71927b70af252101a0c742c62a537cdc8c4dc6"
    )
    assert summary["source_hashes"]["optimized_retrieval_predictions.jsonl"] == (
        "770ef934fca43279b875a5c711765d7771f600fdcfd382590070d2d166293264"
    )


def test_optimized_overall_cells_and_transitions_are_fixed() -> None:
    summary = _summary()
    assert summary["overall"]["sageqa_optimized/k1"]["counts"] == {
        "exact": 1238,
        "complete_nonminimal": 616,
        "incomplete": 1337,
        "disjoint": 318,
    }
    assert summary["overall"]["sageqa_optimized/adaptive"]["counts"] == {
        "exact": 1123,
        "complete_nonminimal": 908,
        "incomplete": 1253,
        "disjoint": 225,
    }
    transitions = summary["paired_transitions"]["optimized_k1_to_adaptive"]
    assert transitions["incomplete_to_complete_nonminimal"] == 115
    assert transitions["disjoint_to_complete_nonminimal"] == 19
    assert transitions["incomplete_to_exact"] == 43
    assert transitions["exact_to_complete_nonminimal"] == 158


def test_principal_failure_mode_statistics_are_fixed() -> None:
    summary = _summary()
    adaptive = summary["overall"]["sageqa_optimized/adaptive"]
    assert adaptive["answer_correct_without_complete_count"] == 512
    routing = summary["ontology_routing"]["sageqa_optimized/adaptive"]
    assert routing["ask_deterministic_proof"] == {
        "n": 532,
        "answer_errors": 0,
        "complete_support_n": 532,
        "complete_support_answer_errors": 0,
        "answer_failure_given_complete_support": 0.0,
    }
    assert routing["ask_llm_no_proof"]["complete_support_answer_errors"] == 0
    assert routing["select_llm"]["complete_support_answer_errors"] == 257
    hops = summary["ontology_by_hop"]["sageqa_optimized/adaptive"]
    assert hops["1hop"]["n"] == 751
    assert hops["2hop"]["n"] == 758


def test_historical_cross_encoder_rows_are_byte_identical() -> None:
    historical_path = (
        ROOT / "outputs/final_results/final_manuscript_stagewise_error_analysis/stagewise_table.tex"
    )
    current_path = OUTPUT / "stagewise_table.tex"
    old_rows = [
        line for line in historical_path.read_text(encoding="utf-8").splitlines()
        if " & Cross-Encoder " in line
    ]
    new_rows = [
        line for line in current_path.read_text(encoding="utf-8").splitlines()
        if " & Cross-Encoder " in line
    ]
    assert len(old_rows) == len(new_rows) == 10
    assert new_rows == old_rows


def test_derived_manifest_covers_every_output() -> None:
    manifest = json.loads((OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["historical_cross_encoder_recomputed"] is False
    assert manifest["prediction_or_retrieval_generation_performed"] is False
    files = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    assert set(manifest["derived_files"]) == files - {"artifact_manifest.json"}
    for name, details in manifest["derived_files"].items():
        assert optimized.historical.sha256(OUTPUT / name) == details["sha256"]
