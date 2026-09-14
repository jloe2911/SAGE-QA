"""Repository-rooted runtime paths for active SAGE-QA workflows.

Frozen manifests and metadata are historical records.  This module resolves
active paths only and deliberately does not read ``frozen_path_map.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path


_SOURCE_REPO_ROOT = Path(__file__).resolve().parents[1]


def _directory_override(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} must name an existing directory: {path}")
    return path


def repo_root() -> Path:
    """Return the checkout root, independent of the process working directory."""

    root = _directory_override("SAGEQA_REPO_ROOT") or _SOURCE_REPO_ROOT
    required = (root / "pyproject.toml", root / "release_manifests")
    if not required[0].is_file() or not required[1].is_dir():
        raise FileNotFoundError(
            f"SAGEQA_REPO_ROOT is not a SAGE-QA checkout (missing pyproject.toml or "
            f"release_manifests): {root}"
        )
    return root


def data_root() -> Path:
    return _directory_override("SAGEQA_DATA_ROOT") or repo_root() / "data"


def checkpoints_root() -> Path:
    return _directory_override("SAGEQA_CHECKPOINTS_ROOT") or repo_root() / "checkpoints"


def outputs_root() -> Path:
    return _directory_override("SAGEQA_OUTPUTS_ROOT") or repo_root() / "outputs"


def generator_d_root() -> Path:
    return data_root() / "production_generator_d_v1"


def cross_encoder_root() -> Path:
    return outputs_root() / "development_runs/question_candidate_cross_encoder_v1"


def cross_encoder_checkpoint() -> Path:
    return cross_encoder_root() / "checkpoint_final"


def cross_encoder_policy_root() -> Path:
    return outputs_root() / (
        "development_runs/question_candidate_cross_encoder_v1_cross_encoder_adaptive_k"
    )


def final_sageqa_policy_root() -> Path:
    return outputs_root() / (
        "development_runs/question_candidate_cross_encoder_v1_final_sageqa_adaptive_k"
    )


def cross_encoder_test_root() -> Path:
    return outputs_root() / "final_results/question_candidate_cross_encoder_v1_adaptive_test_a40"


def hard_pair_test_retrieval_root() -> Path:
    return outputs_root() / "final_results/production_generator_d_v2_hard_pair_test_retrieval"


def manuscript_retrieval_root() -> Path:
    return outputs_root() / "final_results/manuscript_retrieval_results_hard_pair_v2"


def final_answer_root() -> Path:
    return outputs_root() / "final_results/final_manuscript_test_end_to_end"


def baseline_retrieval_root() -> Path:
    return outputs_root() / "final_results/production_generator_d_v1_test_baselines"


def baseline_answer_root() -> Path:
    return outputs_root() / "final_results/final_manuscript_baselines_test_end_to_end"


def hard_pair_checkpoint_root() -> Path:
    return checkpoints_root() / "production_generator_d_v2_hard_pair"


def hard_pair_answer_root() -> Path:
    return outputs_root() / "final_results/production_generator_d_v2_hard_pair_test_end_to_end"


def complete_oracle_root() -> Path:
    return outputs_root() / "final_results/gold_support_complete_oracle"


def provenance_correction_root() -> Path:
    return outputs_root() / "final_results/provenance_correction_v1"


def repo_display_path(path: Path) -> str:
    """Keep current-checkout metadata repository-relative when possible."""

    resolved = path.resolve()
    root = repo_root().resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def repo_path_arg(value: str) -> Path:
    """Resolve an existing CLI path spelling relative to the checkout root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()
