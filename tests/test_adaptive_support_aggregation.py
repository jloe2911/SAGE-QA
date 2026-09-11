from collections.abc import Mapping
from typing import Any

import pytest

from evaluation.adaptive_support_aggregation import (
    adaptive_support_aggregate,
    fixed_k_support_aggregate,
)
from evaluation.evaluate_owl_qa_predictions import get_top_support_units as evaluated_support
from generation.generate_owl_answers_with_llm import get_top_support_units as generated_support


def candidate(score: float, units: list[str], rank: int) -> dict[str, Any]:
    return {"rank": rank, "adjusted_score": score, "subgraph_units": units}


def test_completely_redundant_candidate_has_zero_novelty_and_stops():
    result = adaptive_support_aggregate(
        [candidate(1.0, ["a", "b"], 1), candidate(0.9, ["b", "a"], 2)],
        tau=0.01,
    )

    assert result["decisions"][0]["marginal_novelty"] == 0.0
    assert result["selected_k"] == 1
    assert result["stopping_point"] == 2


def test_completely_novel_candidate_has_unit_novelty():
    result = adaptive_support_aggregate(
        [candidate(1.0, ["a"], 1), candidate(0.9, ["b", "c"], 2)], tau=0.0
    )

    assert result["decisions"][0]["marginal_novelty"] == 1.0
    assert result["selected_k"] == 2


def test_identical_scores_have_unit_relative_strength():
    result = adaptive_support_aggregate(
        [candidate(0.5, ["a"], 1), candidate(0.5, ["b"], 2)], tau=0.5
    )

    assert result["identical_scores"] is True
    assert result["decisions"][0]["relative_strength"] == 1.0


def test_very_small_score_variance_is_stable():
    result = adaptive_support_aggregate(
        [
            candidate(1.0, ["a"], 1),
            candidate(1.0 - 1e-14, ["b"], 2),
            candidate(1.0 - 2e-14, ["c"], 3),
        ],
        tau=0.9,
        epsilon=1e-12,
    )

    assert result["identical_scores"] is True
    assert all(step["relative_strength"] == 1.0 for step in result["decisions"])


def test_always_selects_at_least_top_one():
    result = adaptive_support_aggregate([candidate(1.0, ["a"], 1)], tau=1.0)

    assert result["selected_k"] == 1
    assert result["support_units"] == ["a"]


def test_never_selects_more_than_five():
    candidates = [candidate(1.0, [str(index)], index) for index in range(1, 7)]

    result = adaptive_support_aggregate(candidates, tau=0.0)

    assert result["selected_k"] == 5
    assert result["support_units"] == ["1", "2", "3", "4", "5"]


def test_early_stopping_does_not_consider_later_candidate():
    result = adaptive_support_aggregate(
        [
            candidate(1.0, ["a"], 1),
            candidate(0.9, ["a"], 2),
            candidate(0.8, ["new"], 3),
        ],
        tau=0.01,
    )

    assert result["selected_k"] == 1
    assert len(result["decisions"]) == 1
    assert "new" not in result["support_units"]


def test_deduplicates_within_and_across_candidates():
    result = fixed_k_support_aggregate(
        [candidate(1.0, ["a", "a", "b"], 1), candidate(0.9, ["b", "c"], 2)],
        k=2,
    )

    assert result["support_units"] == ["a", "b", "c"]
    assert result["final_support_size"] == 3


class GoldGuardCandidate(Mapping[str, Any]):
    def __init__(self, allowed: dict[str, Any]):
        self.allowed = allowed

    def __getitem__(self, key: str) -> Any:
        if key not in self.allowed:
            raise AssertionError(f"Adaptive policy accessed forbidden field: {key}")
        return self.allowed[key]

    def __iter__(self):
        return iter(self.allowed)

    def __len__(self) -> int:
        return len(self.allowed)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in {"rank", "adjusted_score", "subgraph_units"}:
            raise AssertionError(f"Adaptive policy accessed forbidden field: {key}")
        return self.allowed.get(key, default)


def test_inference_policy_does_not_access_ground_truth_fields():
    candidates = [
        GoldGuardCandidate(
            {"rank": 1, "adjusted_score": 1.0, "subgraph_units": ["a"]}
        ),
        GoldGuardCandidate(
            {"rank": 2, "adjusted_score": 0.9, "subgraph_units": ["b"]}
        ),
    ]

    result = adaptive_support_aggregate(candidates, tau=0.0)

    assert result["selected_k"] == 2


@pytest.mark.parametrize("support_function", [generated_support, evaluated_support])
def test_owl_pipeline_requires_explicit_adaptive_mode(support_function):
    item = {
        "top5": [
            candidate(1.0, ["a"], 1),
            candidate(0.9, ["a"], 2),
            candidate(0.8, ["new"], 3),
        ]
    }

    assert support_function(item, top_k=3) == ["a", "new"]
    assert support_function(
        item, top_k=3, aggregation_mode="adaptive", adaptive_tau=0.01
    ) == ["a"]


@pytest.mark.parametrize("invalid_k", [0, 6])
def test_enforces_adaptive_budget_bounds(invalid_k: int):
    with pytest.raises(ValueError, match="k_max"):
        adaptive_support_aggregate(
            [candidate(1.0, ["a"], 1)], tau=0.5, k_max=invalid_k
        )
