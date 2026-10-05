from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _yaml(relative: str):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def _sha256_text_lf(relative: str) -> str:
    content = (ROOT / relative).read_text(encoding="utf-8")
    return hashlib.sha256(content.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _is_relative(value: str) -> bool:
    return not PurePosixPath(value).is_absolute() and not PureWindowsPath(value).is_absolute()


def test_publication_and_thesis_protocols_remain_distinct() -> None:
    paper = _yaml("release_manifests/paper_original/index.yaml")["bundles"][0]
    thesis = _yaml("release_manifests/thesis_final/index.yaml")["bundles"][0]
    assert paper["source_revision"] == "dbdbb50708bdc6c686ef82518ec71c1d1bf55985"
    assert paper["status"] == "candidate_source_revision_unresolved"
    assert "historical" in paper["gold_access_policy"]
    assert thesis["source_revision"] == "884480be53c554edae70d3b2d8e781847590aa69"
    assert thesis["gold_access_policy"] == "predictions and reader inputs frozen before evaluation gold join"
    assert paper["producing_entry_point"] != thesis["producing_entry_point"]


def test_optimized_index_source_hashes_and_statuses() -> None:
    index = _yaml("release_manifests/thesis_optimized/index.yaml")
    assert index["scientific_status"] == {
        "completed": True,
        "raw_test_selection_bundle_recovered": False,
        "downstream_outputs_hash_verified": True,
        "regeneration_performed_in_phase2": False,
    }
    assert index["git_text_hash_mode"] == "sha256_utf8_lf"
    for record in index["git_sources"][1:]:
        assert _sha256_text_lf(record["path"]) == record["sha256"]
    overlay = index["canonical_retrieval_overlay"]
    assert _sha256_text_lf(overlay["path"]) == overlay["sha256"]


def test_canonical_overlay_matches_recorded_end_to_end_support_cells() -> None:
    overlay = json.loads(
        (ROOT / "release_manifests/thesis_optimized/canonical_retrieval_overlay.json").read_text(
            encoding="utf-8"
        )
    )
    assert overlay["status"] == "verified_derived_overlay_raw_selection_bundle_missing"
    assert overlay["cohort"] == {
        "definition": "examples with defined, non-empty ground-truth support",
        "included": 3509,
        "excluded": 740,
        "prediction_population": 4249,
    }
    expected = {"k1": 0.6976593565548725, "adaptive": 0.697455332849901}
    for setting, f1 in expected.items():
        record = overlay["methods"]["sageqa_optimized"][setting]
        assert record["equal_dataset_macro"]["f1"] == f1
        assert abs(
            sum(cell["f1"] for cell in record["by_dataset"].values()) / 10 - f1
        ) < 1e-12


def test_archival_specs_are_fail_closed_and_portable() -> None:
    for relative in (
        "release_manifests/archive_specs/paper_original.yaml",
        "release_manifests/archive_specs/thesis_final.yaml",
    ):
        spec = _yaml(relative)
        assert spec["publication_status"] == "specification_only_not_published"
        assert spec["download_url"] is None
        assert spec["archive_sha256"] is None
        assert spec["estimated_uncompressed_bytes"] > 0
        for component in spec["components"]:
            assert _is_relative(component["path"]), component["path"]
            assert component["status"] in {
                "present_locally_hash_inventory_required",
                "recovered_hash_verified",
                "verified_derived_result",
                "missing_recorded_hash_preserved",
                "unverified_historical_provenance",
                "tracked_source",
            }


def test_official_and_diagnostic_scoring_are_explicitly_separate() -> None:
    protocol = (ROOT / "docs/EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    diagnostic = (ROOT / "evaluation/analyze_optimized_stagewise_test_errors.py").read_text(
        encoding="utf-8"
    )
    verifier = (ROOT / "evaluation/verify_canonical_retrieval_overlay.py").read_text(
        encoding="utf-8"
    )
    assert "Official scoring" in protocol and "Diagnostic scoring" in protocol
    assert "reasoner-validated" in protocol
    assert "classify_support" in diagnostic
    assert "best_support_scores" in verifier
