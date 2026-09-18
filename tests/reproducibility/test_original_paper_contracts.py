from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml


ORIGINAL_PAPER_COMMIT_CANDIDATE = "dbdbb50708bdc6c686ef82518ec71c1d1bf55985"
ORIGINAL_PAPER_SOURCE_STATUS = "candidate_not_fully_proven"
HISTORICAL_ENTRY_POINT = "experiments/run_experiments.py"
HISTORICAL_MODULES = (
    "evaluation/hotpot_official_eval.py",
    "evaluation/evaluate_owl_qa_predictions.py",
    "evaluation/collect_final_results.py",
    "generation/generate_hotpot_answers_with_llm.py",
    "generation/generate_owl_answers_with_llm.py",
)
REPORTED_RESULT_ROOTS = [
    "outputs/full_results/HotpotQA",
    "outputs/full_results/2WikiMultiHopQA",
    "outputs/full_results/FamilyOWL_1hop",
    "outputs/full_results/FamilyOWL_2hop",
    "outputs/full_results/OWL2Bench_1hop",
    "outputs/full_results/OWL2Bench_2hop",
    "outputs/full_results/full_pipeline_results.csv",
    "outputs/full_results/full_pipeline_results.json",
    "outputs/full_results/sageqa_retrieval_prf_deduplicated_union_k1_k2_k3_k5.csv",
]


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=check, capture_output=True, text=True
    )


def test_original_paper_candidate_and_historical_modules_exist_in_git(repo_root: Path):
    assert ORIGINAL_PAPER_SOURCE_STATUS == "candidate_not_fully_proven"
    _git(repo_root, "cat-file", "-e", f"{ORIGINAL_PAPER_COMMIT_CANDIDATE}^{{commit}}")
    for relative in (HISTORICAL_ENTRY_POINT, *HISTORICAL_MODULES):
        _git(repo_root, "cat-file", "-e", f"{ORIGINAL_PAPER_COMMIT_CANDIDATE}:{relative}")


def test_historical_runner_records_original_roots_without_thesis_routing(repo_root: Path):
    source = _git(
        repo_root, "show", f"{ORIGINAL_PAPER_COMMIT_CANDIDATE}:{HISTORICAL_ENTRY_POINT}"
    ).stdout
    data_roots = (
        "data/HotpotQA",
        "data/2WikiMultiHopQA",
        "data/FamilyOWL_1hop",
        "data/FamilyOWL_2hop",
        "data/OWL2Bench_1hop",
        "data/OWL2Bench_2hop",
    )
    checkpoint_roots = (
        "checkpoints/gnn_subgraph_ranker_hotpotqa_full",
        "checkpoints/gnn_subgraph_ranker_2wiki_full",
        "checkpoints/gnn_subgraph_ranker_familyowl_1hop_full",
        "checkpoints/gnn_subgraph_ranker_familyowl_2hop_full",
        "checkpoints/gnn_subgraph_ranker_owl2bench_1hop_full",
        "checkpoints/gnn_subgraph_ranker_owl2bench_2hop_full",
    )
    for relative in (*data_roots, *checkpoint_roots, "outputs/full_results"):
        assert relative in source
        assert (repo_root / relative).exists()
    assert "production_generator_d_v1" not in source
    assert "question_candidate_cross_encoder_v1" not in source


def test_original_paper_index_fails_closed_on_unproven_lineage(repo_root: Path):
    index = yaml.safe_load(
        (repo_root / "release_manifests/paper_original/index.yaml").read_text(encoding="utf-8")
    )["bundles"][0]
    assert index["source_revision"] == ORIGINAL_PAPER_COMMIT_CANDIDATE
    assert index["status"] == "candidate_source_revision_unresolved"
    assert index["known_sha256"] == {}
    assert index["externally_archived"] == "unknown"
    assert index["producing_entry_point"] == HISTORICAL_ENTRY_POINT
    assert index["result_root"] == REPORTED_RESULT_ROOTS
    assert "not yet formally proven" in index["notes"]


def test_detached_compatibility_worktree_is_post_submission_only(repo_root: Path):
    worktree = repo_root / ".worktrees/sageqa-original-eval-compat"
    if not worktree.exists():
        pytest.skip("Optional local compatibility worktree is absent")
    head = _git(worktree, "rev-parse", "HEAD").stdout.strip()
    assert head != ORIGINAL_PAPER_COMMIT_CANDIDATE
    assert _git(
        worktree,
        "merge-base",
        "--is-ancestor",
        ORIGINAL_PAPER_COMMIT_CANDIDATE,
        head,
        check=False,
    ).returncode == 0
