from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from generation import run_final_manuscript_answer_generation as historical_runner
from generation import run_gold_support_complete_oracle as complete_runner


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_complete_and_correction_logical_ids_are_distinct(repo_root: Path):
    thesis = yaml.safe_load(
        (repo_root / "release_manifests/thesis_final/index.yaml").read_text(encoding="utf-8")
    )["bundles"][0]
    oracle = yaml.safe_load(
        (repo_root / "release_manifests/oracle/index.yaml").read_text(encoding="utf-8")
    )["bundles"][0]
    correction = _json(
        repo_root / "outputs/final_results/provenance_correction_v1/provenance_manifest.json"
    )
    logical_ids = {
        f'{thesis["logical_name"]}::gold_support/oracle',
        oracle["logical_name"],
        correction["schema_version"],
    }
    assert len(logical_ids) == 3
    assert oracle["logical_name"] == "gold_support_complete_oracle"
    assert correction["schema_version"] == "final_provenance_correction_manifest_v1"


def test_oracle_roots_and_runners_cannot_overwrite_historical_result(repo_root: Path):
    correction_root = repo_root / "outputs/final_results/provenance_correction_v1"
    roots = {historical_runner.DEFAULT_OUTPUT, complete_runner.DEFAULT_OUTPUT, correction_root}
    assert len(roots) == 3
    assert historical_runner.DEFAULT_OUTPUT.name == "final_manuscript_test_end_to_end"
    assert complete_runner.DEFAULT_OUTPUT.name == "gold_support_complete_oracle"
    assert complete_runner.METHOD == "gold_support_complete"
    assert complete_runner.DEFAULT_OUTPUT not in historical_runner.DEFAULT_OUTPUT.parents
    assert historical_runner.DEFAULT_OUTPUT not in complete_runner.DEFAULT_OUTPUT.parents


def test_main_results_cannot_silently_substitute_complete_oracle(repo_root: Path):
    main_root = repo_root / "outputs/final_results/final_manuscript_test_end_to_end"
    oracle_root = repo_root / "outputs/final_results/gold_support_complete_oracle"
    main_manifest = _json(main_root / "artifact_manifest.json")
    oracle_manifest = _json(oracle_root / "artifact_manifest.json")
    main_predictions = main_manifest["frozen_inputs"]["predictions.jsonl"]
    complete_predictions = oracle_manifest["files"]["predictions.jsonl"]["sha256"]
    assert main_predictions != complete_predictions
    assert _sha256(main_root / "predictions.jsonl") == main_predictions
    assert _sha256(oracle_root / "predictions.jsonl") == complete_predictions
    assert "gold_support_complete" not in json.dumps(main_manifest)
    assert main_manifest["verification"]["prediction_counts_by_condition"]["gold_support/oracle"] == 3509


def test_provenance_correction_references_the_intended_oracle_bundle(repo_root: Path):
    correction = _json(
        repo_root / "outputs/final_results/provenance_correction_v1/provenance_manifest.json"
    )
    reference = correction["corrected_gold_support_manifest"]
    path = repo_root / reference["path"].replace("\\", "/")
    assert path == complete_runner.DEFAULT_OUTPUT / "artifact_manifest.json"
    assert _sha256(path) == reference["sha256"]
    assert correction["protected_hashes_unchanged_and_manifest_verified"] is True
    assert correction["main_predictions_regenerated"] is False
