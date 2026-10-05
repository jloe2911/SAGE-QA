from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import build_reviewer_artifacts as builder
from scripts import restore_reviewer_artifacts as restorer


def test_safe_relative_rejects_traversal_and_absolute_paths() -> None:
    for value in ("../escape", "a/../../escape", "/absolute"):
        with pytest.raises(ValueError):
            restorer.safe_relative(value)
    assert restorer.safe_relative("outputs/example.json") == Path("outputs/example.json")


def test_restore_validates_manifest_and_payload(tmp_path: Path) -> None:
    payload = b"frozen-result\n"
    relative = "outputs/final_results/example.json"
    manifest = {
        "schema_version": "sageqa_reviewer_table_artifacts_v1",
        "package": "fixture.zip",
        "lineage": "test",
        "purpose": "test",
        "file_count": 1,
        "uncompressed_bytes": len(payload),
        "files": [
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "limitations": [],
    }
    package = tmp_path / "fixture.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(relative, payload)
        archive.writestr(restorer.MANIFEST_NAME, json.dumps(manifest))
    target = tmp_path / "checkout"
    first = restorer.restore(package, target)
    second = restorer.restore(package, target)
    assert first["restored"] == 1
    assert second["already_present_and_verified"] == 1
    assert (target / relative).read_bytes() == payload


def test_package_scopes_preserve_two_distinct_lineages() -> None:
    assert builder.PACKAGES["paper"]["lineage"] != builder.PACKAGES["thesis"]["lineage"]
    assert builder.PAPER_PATHS == ("outputs/full_results",)
    assert any("symbolic_coefficients_original_v1" in path for path in builder.THESIS_PATHS)
    assert not any("native_runs" in path for path in builder.THESIS_PATHS)


def test_built_package_manifests_are_hash_bound_and_not_claimed_public() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "paper_table_artifacts.json": (
            "paper_table_artifacts.zip",
            "7e776628d2988222ce0f7770bda868d94ee49fec6ffc705e7b30b1bea76a1b48",
            209,
        ),
        "thesis_table_artifacts.json": (
            "thesis_table_artifacts.zip",
            "18d9faf7b483f59a2ba6367e28db20d2ef7071bd6bb8ca06da4464376f7e7ea1",
            82,
        ),
    }
    for name, (package, digest, count) in expected.items():
        manifest = json.loads(
            (root / "release_manifests/reviewer_artifacts" / name).read_text(encoding="utf-8")
        )
        assert manifest["package"] == package
        assert manifest["archive_sha256"] == digest
        assert manifest["file_count"] == count == len(manifest["files"])
        assert manifest["publication_status"] == "local_supplied_artifact_not_uploaded"
        assert manifest["download_url"] is None


def test_fresh_clone_acceptance_manifest_records_both_table_lineages() -> None:
    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (
            root
            / "release_manifests/reviewer_artifacts/acceptance_reproduction.json"
        ).read_text(encoding="utf-8")
    )
    assert report["reports"]["paper"]["status"] == (
        "paper_reported_tables_materialized_from_verified_frozen_outputs"
    )
    thesis = report["reports"]["thesis"]
    assert thesis["status"] == "thesis_tables_reproduced_and_matched_frozen_outputs"
    assert thesis["hard_pair_v2_products_byte_identical"] is True
    assert thesis["optimized_overlay"]["checked_setting_rows"] == 7018
    assert thesis["optimized_overlay"]["raw_selection_artifact_recovered"] is False
    assert len(report["files"]) == 24
