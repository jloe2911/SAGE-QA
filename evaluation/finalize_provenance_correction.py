"""Write a fail-closed manifest for the final provenance correction pass."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation import run_production_test_answer_generation as production
from utils.paths import (
    complete_oracle_root,
    cross_encoder_test_root,
    final_answer_root,
    manuscript_retrieval_root,
    outputs_root,
    provenance_correction_root,
    repo_display_path,
    repo_root,
)


ROOT = repo_root()
ORACLE_ROOT = complete_oracle_root()
GNN_ROOT = manuscript_retrieval_root()
HISTORICAL_GNN_ROOT = outputs_root() / "final_results/manuscript_retrieval_results"
SELECTION_ROOT = cross_encoder_test_root()
ANSWER_ROOT = final_answer_root()
OUTPUT = provenance_correction_root()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_hash(manifest: dict[str, Any], filename: str) -> str:
    files = manifest.get("files")
    if files is None:
        for section in ("frozen_inputs", "derived_files"):
            entry = manifest.get(section, {}).get(filename)
            if entry is not None:
                return str(entry["sha256"] if isinstance(entry, dict) else entry)
        raise KeyError(filename)
    if isinstance(files, dict):
        entry = files[filename]
        return str(entry["sha256"] if isinstance(entry, dict) else entry)
    for entry in files:
        if Path(entry["path"]).name == filename:
            return str(entry["sha256"])
    raise KeyError(filename)


def locked(path: Path, expected: str) -> str:
    actual = production.sha256(path)
    if actual != expected:
        raise ValueError(f"Protected/frozen hash mismatch: {path}")
    return actual


def main() -> dict[str, Any]:
    selection_freeze = read_json(SELECTION_ROOT / "generation_freeze.json")
    selection_hash = locked(
        SELECTION_ROOT / "test_predictions_frozen.jsonl",
        str(selection_freeze["predictions_sha256"]),
    )
    answer_freeze = read_json(ANSWER_ROOT / "generation_freeze.json")
    answer_hash = locked(
        ANSWER_ROOT / "predictions.jsonl", str(answer_freeze["predictions_sha256"])
    )
    answer_manifest = read_json(ANSWER_ROOT / "artifact_manifest.json")
    metrics_hash = locked(
        ANSWER_ROOT / "metrics.json", manifest_hash(answer_manifest, "metrics.json")
    )
    canonical_hash = locked(
        ANSWER_ROOT / "canonical_metrics.json",
        manifest_hash(answer_manifest, "canonical_metrics.json"),
    )

    oracle_manifest_path = ORACLE_ROOT / "artifact_manifest.json"
    oracle_manifest = read_json(oracle_manifest_path)
    if oracle_manifest.get("status") != "complete_frozen":
        raise ValueError("Corrected Gold Support oracle is not complete/frozen")
    for filename, entry in oracle_manifest["files"].items():
        locked(ORACLE_ROOT / filename, str(entry["sha256"] if isinstance(entry, dict) else entry))

    gnn_manifest_path = GNN_ROOT / "artifact_manifest.json"
    gnn_manifest = read_json(gnn_manifest_path)
    if gnn_manifest.get("gnn_lineage") != "production_generator_d_v2_hard_pair_test_retrieval":
        raise ValueError("Canonical GNN export is not hard-pair-v2")
    for filename, entry in gnn_manifest["files"].items():
        locked(GNN_ROOT / filename, str(entry["sha256"] if isinstance(entry, dict) else entry))
    if not (HISTORICAL_GNN_ROOT / "artifact_manifest.json").is_file():
        raise ValueError("Historical canonical GNN export was not preserved")

    protected = {
        "cross_encoder_frozen_predictions": {
            "path": repo_display_path(SELECTION_ROOT / "test_predictions_frozen.jsonl"),
            "sha256": selection_hash,
            "condition_path": "methods.cross_encoder",
        },
        "final_sageqa_k1_frozen_predictions": {
            "path": repo_display_path(SELECTION_ROOT / "test_predictions_frozen.jsonl"),
            "sha256": selection_hash,
            "condition_path": "methods.final_sageqa.k1",
        },
        "final_sageqa_adaptive_frozen_predictions": {
            "path": repo_display_path(SELECTION_ROOT / "test_predictions_frozen.jsonl"),
            "sha256": selection_hash,
            "condition_path": "methods.final_sageqa.adaptive",
        },
        "main_reader_predictions": {
            "path": repo_display_path(ANSWER_ROOT / "predictions.jsonl"),
            "sha256": answer_hash,
        },
        "main_reader_metrics": {
            "path": repo_display_path(ANSWER_ROOT / "metrics.json"),
            "sha256": metrics_hash,
        },
        "main_canonical_metrics": {
            "path": repo_display_path(ANSWER_ROOT / "canonical_metrics.json"),
            "sha256": canonical_hash,
        },
    }
    result = {
        "schema_version": "final_provenance_correction_manifest_v1",
        "status": "complete_verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corrected_gold_support_manifest": {
            "path": repo_display_path(oracle_manifest_path),
            "sha256": production.sha256(oracle_manifest_path),
        },
        "hard_pair_v2_gnn_manifest": {
            "path": repo_display_path(gnn_manifest_path),
            "sha256": production.sha256(gnn_manifest_path),
        },
        "historical_gnn_manifest_preserved": {
            "path": repo_display_path(HISTORICAL_GNN_ROOT / "artifact_manifest.json"),
            "sha256": production.sha256(HISTORICAL_GNN_ROOT / "artifact_manifest.json"),
        },
        "protected_main_artifacts": protected,
        "protected_hashes_unchanged_and_manifest_verified": True,
        "main_predictions_regenerated": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    production.write_json(OUTPUT / "provenance_manifest.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
