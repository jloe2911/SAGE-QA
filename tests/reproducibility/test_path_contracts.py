from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.check_thesis_final_paths import LOGICAL_BUNDLE, PROTOCOL, resolution_report
from utils import paths


PATH_ENV = (
    "SAGEQA_REPO_ROOT",
    "SAGEQA_DATA_ROOT",
    "SAGEQA_CHECKPOINTS_ROOT",
    "SAGEQA_OUTPUTS_ROOT",
)


@pytest.fixture(autouse=True)
def clean_path_environment(monkeypatch: pytest.MonkeyPatch):
    for name in PATH_ENV:
        monkeypatch.delenv(name, raising=False)


def test_repository_root_and_defaults_are_independent_of_cwd(repo_root: Path):
    expected = repo_root.resolve()
    original_cwd = Path.cwd()
    try:
        os.chdir(repo_root.parent)
        assert paths.repo_root() == expected
        assert paths.generator_d_root() == expected / "data/production_generator_d_v1"
        assert paths.cross_encoder_checkpoint() == expected / "outputs/development_runs/question_candidate_cross_encoder_v1/checkpoint_final"
        assert paths.manuscript_retrieval_root() == expected / "outputs/final_results/manuscript_retrieval_results_hard_pair_v2"
        assert paths.final_answer_root() == expected / "outputs/final_results/final_manuscript_test_end_to_end"
    finally:
        os.chdir(original_cwd)


def test_canonical_defaults_resolve_existing_protected_assets(repo_root: Path):
    assert paths.generator_d_root().is_dir()
    assert paths.cross_encoder_checkpoint().is_dir()
    assert (paths.checkpoints_root() / "production_generator_d_v2_hard_pair").is_dir()
    assert paths.cross_encoder_test_root().is_dir()
    assert paths.manuscript_retrieval_root().is_dir()
    assert paths.final_answer_root().is_dir()


def test_environment_overrides_change_paths_not_logical_identity(repo_root: Path, monkeypatch: pytest.MonkeyPatch):
    data, checkpoints, outputs = repo_root / "checkpoints", repo_root / "data", repo_root / "artifacts"
    monkeypatch.setenv("SAGEQA_DATA_ROOT", str(data))
    monkeypatch.setenv("SAGEQA_CHECKPOINTS_ROOT", str(checkpoints))
    monkeypatch.setenv("SAGEQA_OUTPUTS_ROOT", str(outputs))
    assert paths.generator_d_root() == data / "production_generator_d_v1"
    assert paths.checkpoints_root() == checkpoints
    assert paths.final_answer_root() == outputs / "final_results/final_manuscript_test_end_to_end"
    assert PROTOCOL == "thesis_final"
    assert LOGICAL_BUNDLE == "thesis_final_generator_d_cross_encoder_adaptive_answers"


def test_invalid_override_fails_without_fallback(repo_root: Path, monkeypatch: pytest.MonkeyPatch):
    missing = repo_root / ".tmp/reproducibility/does-not-exist-path-contract"
    monkeypatch.setenv("SAGEQA_DATA_ROOT", str(missing))
    with pytest.raises(FileNotFoundError, match="SAGEQA_DATA_ROOT"):
        paths.data_root()


def test_relative_cli_paths_resolve_against_repository(repo_root: Path):
    assert paths.repo_path_arg("outputs/example") == (repo_root / "outputs/example").resolve()
    assert paths.repo_display_path(repo_root / "outputs/example") == str(Path("outputs/example"))
    assert paths.repo_display_path(repo_root.parent) == str(repo_root.parent.resolve())


def test_importing_path_utility_creates_nothing(repo_root: Path):
    roots = (repo_root / "data", repo_root / "checkpoints", repo_root / "outputs")
    before = {root: {path.name for path in root.iterdir()} for root in roots}
    env = os.environ.copy()
    env.update({"SAGEQA_DATA_ROOT": str(roots[0]), "SAGEQA_CHECKPOINTS_ROOT": str(roots[1]), "SAGEQA_OUTPUTS_ROOT": str(roots[2])})
    subprocess.run([sys.executable, "-c", "import utils.paths"], cwd=repo_root, env=env, check=True)
    after = {root: {path.name for path in root.iterdir()} for root in roots}
    assert after == before


def test_frozen_path_map_is_descriptive_not_a_runtime_input(repo_root: Path):
    source = (repo_root / "utils/paths.py").read_text(encoding="utf-8")
    assert "frozen_path_map.yaml" in source
    assert "read_text" not in source
    assert "yaml.safe_load" not in source
    frozen_map = repo_root / "release_manifests/path_maps/frozen_path_map.yaml"
    assert hashlib.sha256(frozen_map.read_bytes()).hexdigest() == "fc6bb290e72c7a39d783deeb3a8092df5e52c5e16aaaae2cd2308d02d6774b77"


def test_resolution_check_is_read_only_and_complete(repo_root: Path):
    watched = (repo_root / "release_manifests/thesis_final/index.yaml", paths.final_answer_root() / "artifact_manifest.json")
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in watched}
    report = resolution_report()
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in watched}
    assert after == before
    assert report["mutated_files"] == 0
    assert all(report[name]["exists"] for name in ("repository_root", "generator_d_root", "cross_encoder_checkpoint", "frozen_test_selection", "retrieval_export", "answer_bundle"))
    assert all(item["exists"] for item in report["adaptive_policy"].values())


def test_original_paper_runner_does_not_import_active_path_layer(repo_root: Path):
    runner = repo_root / "experiments/run_experiments.py"
    assert "utils.paths" not in runner.read_text(encoding="utf-8")
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "experiments/run_experiments.py"], cwd=repo_root)
    assert result.returncode == 0
