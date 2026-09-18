from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest
import yaml


MAIN_PREDICTIONS_SHA256 = "03e391ed697fb25a07439ec7664544cd6582d2a2dffc868bb9f972c1ae9236d3"
COMPLETE_ORACLE_SHA256 = "27dcdbd6c4cdac8388502a24d6d2d0cb6ce422a95b8cac09a9c364cd45630dae"
GENERATOR_D_MANIFEST_SHA256 = "27a4c149181e96df1fa9aca581eaeee17e354755f6a034ce93f2021b496296e5"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(value: str) -> bool:
    return not PurePosixPath(value).is_absolute() and not PureWindowsPath(value).is_absolute()


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def _resolve_declared_path(repo_root: Path, bundle_root: Path, value: str) -> Path:
    normalized = value.replace("\\", "/")
    return repo_root / normalized if "/" in normalized else bundle_root / normalized


def test_protected_roots_and_release_indexes_resolve(repo_root: Path):
    protection = _yaml(repo_root / "release_manifests/protection_rules.yaml")
    for category, roots in protection["protected_roots"].items():
        assert roots, category
        for relative in roots:
            assert _relative(relative), relative
            assert (repo_root / relative).exists(), relative

    indexes = sorted((repo_root / "release_manifests").glob("*/index.yaml"))
    assert {path.parent.name for path in indexes} == {
        "oracle",
        "paper_original",
        "thesis_baselines",
        "thesis_final",
        "thesis_graph_ablation",
    }
    for index_path in indexes:
        for bundle in _yaml(index_path)["bundles"]:
            paths_may_be_absent = bundle.get("status") in {
                "indexed_not_physically_archived",
                "externally_backed_up_and_removed",
            }
            for field in ("data_root", "checkpoint_root", "policy_root", "result_root"):
                for relative in _values(bundle.get(field)):
                    assert _relative(relative), (index_path, field, relative)
                    if "*" in relative:
                        assert paths_may_be_absent or list(repo_root.glob(relative)), relative
                    elif not paths_may_be_absent:
                        assert (repo_root / relative).exists(), relative
            for field in ("producing_entry_point", "evaluator_finalizer", "manifest_file"):
                for relative in _values(bundle.get(field)):
                    assert _relative(relative), (index_path, field, relative)
                    if not paths_may_be_absent:
                        assert (repo_root / relative).is_file(), relative


def test_generator_d_manifest_identity_member_sizes_and_small_hashes(repo_root: Path):
    lineage_path = repo_root / (
        "outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/"
        "lineage_metadata.json"
    )
    lineage = _json(lineage_path)
    members = lineage["corpus_manifest"]
    canonical = json.dumps(members, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    release = _yaml(repo_root / "release_manifests/thesis_final/index.yaml")["bundles"][0]

    assert lineage["generator_d_version"] == "generator_d_frozen_v1"
    assert len(members) == 66
    assert manifest_hash == lineage["corpus_manifest_sha256"]
    assert manifest_hash == release["known_sha256"]["generator_d_66_entry_aggregate"]
    assert manifest_hash == GENERATOR_D_MANIFEST_SHA256

    corpus_root = repo_root / "data/production_generator_d_v1"
    for member in members:
        path = corpus_root / member["path"]
        assert path.is_file(), member["path"]
        assert path.stat().st_size == member["size_bytes"], member["path"]
        if member["size_bytes"] <= 2_000_000:
            assert _sha256(path) == member["sha256"], member["path"]


@pytest.mark.skipif(
    os.environ.get("SAGEQA_FULL_ARTIFACT_HASHES") != "1",
    reason="Set SAGEQA_FULL_ARTIFACT_HASHES=1 only for the documented FULL suite.",
)
def test_generator_d_all_66_member_hashes(repo_root: Path):
    lineage = _json(
        repo_root
        / "outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/lineage_metadata.json"
    )
    corpus_root = repo_root / "data/production_generator_d_v1"
    for member in lineage["corpus_manifest"]:
        assert _sha256(corpus_root / member["path"]) == member["sha256"], member["path"]


def _verify_main_manifest(repo_root: Path) -> None:
    bundle_root = repo_root / "outputs/final_results/final_manuscript_test_end_to_end"
    manifest = _json(bundle_root / "artifact_manifest.json")
    for section in ("frozen_inputs", "derived_files"):
        for relative, expected in manifest[section].items():
            path = _resolve_declared_path(repo_root, bundle_root, relative)
            assert path.is_file(), relative
            assert _sha256(path) == expected, relative
    assert manifest["verification"]["predictions_sha256"] == MAIN_PREDICTIONS_SHA256


def _verify_oracle_manifest(repo_root: Path) -> None:
    bundle_root = repo_root / "outputs/final_results/gold_support_complete_oracle"
    manifest_path = bundle_root / "artifact_manifest.json"
    manifest = _json(manifest_path)
    for relative, record in manifest["files"].items():
        path = bundle_root / relative
        assert path.is_file(), relative
        assert path.stat().st_size == record["size_bytes"], relative
        assert _sha256(path) == record["sha256"], relative
    oracle_index = _yaml(repo_root / "release_manifests/oracle/index.yaml")["bundles"][0]
    assert _sha256(manifest_path) == oracle_index["known_sha256"]["oracle_manifest"]


def test_critical_prediction_hashes_and_manifests_are_consistent(repo_root: Path):
    main = repo_root / "outputs/final_results/final_manuscript_test_end_to_end/predictions.jsonl"
    oracle = repo_root / "outputs/final_results/gold_support_complete_oracle/predictions.jsonl"
    assert _sha256(main) == MAIN_PREDICTIONS_SHA256
    assert _sha256(oracle) == COMPLETE_ORACLE_SHA256
    _verify_main_manifest(repo_root)
    _verify_oracle_manifest(repo_root)


def test_integrity_verification_is_read_only(repo_root: Path):
    protected = (
        repo_root / "outputs/final_results/final_manuscript_test_end_to_end/predictions.jsonl",
        repo_root / "outputs/final_results/final_manuscript_test_end_to_end/artifact_manifest.json",
        repo_root / "outputs/final_results/gold_support_complete_oracle/predictions.jsonl",
        repo_root / "outputs/final_results/gold_support_complete_oracle/artifact_manifest.json",
    )
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in protected}
    _verify_main_manifest(repo_root)
    _verify_oracle_manifest(repo_root)
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in protected}
    assert after == before
