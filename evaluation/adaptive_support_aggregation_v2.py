"""Learned, rank-aware adaptive support aggregation after symbolic reranking.

Inference is intentionally isolated from supervision.  The public aggregation
function accepts only final ranked candidates and an explicitly selected domain
policy; gold support, answers, dataset names, hops, and split labels are not
part of its interface.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from evaluation.adaptive_support_aggregation import (
    DEFAULT_EPSILON,
    MAX_ADAPTIVE_K,
    canonical_evidence_key,
)


CONTINUOUS_FEATURES = (
    "current_adjusted_score",
    "next_adjusted_score",
    "relative_to_top_next",
    "normalized_consecutive_gap",
    "cumulative_support_size",
    "next_candidate_size",
    "new_evidence_units",
    "novelty",
    "overlap",
)
FEATURE_ORDER = (*CONTINUOUS_FEATURES, "decision_rank")
DOMAINS = ("text", "ontology")


def _unique_pairs(
    units: Sequence[Any], evidence_key: Callable[[Any], Hashable]
) -> list[tuple[Hashable, Any]]:
    pairs: list[tuple[Hashable, Any]] = []
    seen: set[Hashable] = set()
    for unit in units:
        key = evidence_key(unit)
        if key not in seen:
            seen.add(key)
            pairs.append((key, unit))
    return pairs


def validate_ranked_candidates(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    k_max: int = MAX_ADAPTIVE_K,
    epsilon: float = DEFAULT_EPSILON,
    score_key: str = "adjusted_score",
) -> list[Mapping[str, Any]]:
    if not 1 <= k_max <= MAX_ADAPTIVE_K:
        raise ValueError(f"k_max must be in [1, {MAX_ADAPTIVE_K}], got {k_max}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not ranked_candidates:
        raise ValueError("At least one ranked candidate is required")

    candidates = list(ranked_candidates[:k_max])
    scores = [float(candidate[score_key]) for candidate in candidates]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("All candidate scores must be finite")
    if any(left + epsilon < right for left, right in zip(scores, scores[1:])):
        raise ValueError("Candidates must be ordered by non-increasing final score")
    return candidates


def effective_score_scale(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    epsilon: float = DEFAULT_EPSILON,
    score_key: str = "adjusted_score",
) -> float:
    scores = [float(candidate[score_key]) for candidate in ranked_candidates]
    mean_score = sum(scores) / len(scores)
    population_std = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / len(scores))
    return max(population_std, epsilon)


def compute_decision_features(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    decision_rank: int,
    epsilon: float = DEFAULT_EPSILON,
    score_key: str = "adjusted_score",
    units_key: str = "subgraph_units",
    evidence_key: Callable[[Any], Hashable] = canonical_evidence_key,
) -> dict[str, float]:
    """Compute the fixed v2 feature set for STOP/CONTINUE after prefix ``k``."""

    candidates = validate_ranked_candidates(ranked_candidates, epsilon=epsilon, score_key=score_key)
    if not 1 <= decision_rank < len(candidates):
        raise ValueError("decision_rank must identify a prefix with an available next candidate")

    scores = [float(candidate[score_key]) for candidate in candidates]
    score_scale = effective_score_scale(candidates, epsilon=epsilon, score_key=score_key)

    prefix_pairs: list[tuple[Hashable, Any]] = []
    prefix_keys: set[Hashable] = set()
    for candidate in candidates[:decision_rank]:
        for key, unit in _unique_pairs(list(candidate.get(units_key, []) or []), evidence_key):
            if key not in prefix_keys:
                prefix_keys.add(key)
                prefix_pairs.append((key, unit))

    next_pairs = _unique_pairs(
        list(candidates[decision_rank].get(units_key, []) or []), evidence_key
    )
    next_keys = {key for key, _ in next_pairs}
    new_count = len(next_keys - prefix_keys)
    overlap_count = len(next_keys & prefix_keys)
    next_size = len(next_keys)
    next_score = scores[decision_rank]
    current_score = scores[decision_rank - 1]

    return {
        "current_adjusted_score": current_score,
        "next_adjusted_score": next_score,
        "relative_to_top_next": math.exp((next_score - scores[0]) / score_scale),
        "normalized_consecutive_gap": (current_score - next_score) / score_scale,
        "cumulative_support_size": float(len(prefix_pairs)),
        "next_candidate_size": float(next_size),
        "new_evidence_units": float(new_count),
        "novelty": new_count / next_size if next_size else 0.0,
        "overlap": overlap_count / next_size if next_size else 0.0,
        "decision_rank": float(decision_rank),
    }


def feature_vector(features: Mapping[str, float]) -> list[float]:
    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise ValueError(f"Missing adaptive-v2 features: {missing}")
    return [float(features[name]) for name in FEATURE_ORDER]


@dataclass(frozen=True)
class AdaptiveV2Policy:
    domain: str
    scaler: Any
    model: Any
    threshold: float

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

    def continue_probability(self, features: Mapping[str, float]) -> float:
        vector = feature_vector(features)
        continuous = self.scaler.transform([vector[: len(CONTINUOUS_FEATURES)]])[0]
        model_vector = [*continuous.tolist(), vector[-1]]
        probability = float(self.model.predict_proba([model_vector])[0][1])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Model returned invalid probability: {probability}")
        return probability


def load_domain_policy(policy_dir: Path | str, *, domain: str) -> AdaptiveV2Policy:
    """Load one policy by explicit domain; no dataset-to-domain inference occurs here."""

    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    root = Path(policy_dir)
    thresholds = joblib.load(root / "chosen_thresholds.joblib")
    return AdaptiveV2Policy(
        domain=domain,
        scaler=joblib.load(root / f"{domain}_scaler.joblib"),
        model=joblib.load(root / f"{domain}_model.joblib"),
        threshold=float(thresholds[domain]),
    )


def adaptive_v2_support_aggregate(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    policy: AdaptiveV2Policy,
    k_max: int = MAX_ADAPTIVE_K,
    epsilon: float = DEFAULT_EPSILON,
    score_key: str = "adjusted_score",
    units_key: str = "subgraph_units",
    evidence_key: Callable[[Any], Hashable] = canonical_evidence_key,
) -> dict[str, Any]:
    """Sequentially select a ranked prefix using a frozen adaptive-v2 policy."""

    candidates = validate_ranked_candidates(
        ranked_candidates, k_max=k_max, epsilon=epsilon, score_key=score_key
    )
    scores = [float(candidate[score_key]) for candidate in candidates]
    score_scale = effective_score_scale(candidates, epsilon=epsilon, score_key=score_key)

    selected_pairs = _unique_pairs(list(candidates[0].get(units_key, []) or []), evidence_key)
    selected_keys = {key for key, _ in selected_pairs}
    support_units = [unit for _, unit in selected_pairs]
    decisions: list[dict[str, Any]] = []
    selected_k = 1
    stop_reason = "candidate_list_exhausted" if len(candidates) == 1 else "model_stop"

    for decision_rank in range(1, len(candidates)):
        features = compute_decision_features(
            candidates,
            decision_rank=decision_rank,
            epsilon=epsilon,
            score_key=score_key,
            units_key=units_key,
            evidence_key=evidence_key,
        )
        probability = policy.continue_probability(features)
        should_continue = probability >= policy.threshold
        decisions.append(
            {
                "decision_rank": decision_rank,
                "current_candidate_rank": int(
                    candidates[decision_rank - 1].get("rank", decision_rank)
                ),
                "next_candidate_rank": int(
                    candidates[decision_rank].get("rank", decision_rank + 1)
                ),
                "candidate_adjusted_scores": scores,
                "score_scale": score_scale,
                "R_next": features["relative_to_top_next"],
                "G_k": features["normalized_consecutive_gap"],
                "cumulative_support_size": int(features["cumulative_support_size"]),
                "next_candidate_size": int(features["next_candidate_size"]),
                "new_evidence_units": int(features["new_evidence_units"]),
                "novelty": features["novelty"],
                "overlap": features["overlap"],
                "predicted_continue_probability": probability,
                "threshold": policy.threshold,
                "decision": "CONTINUE" if should_continue else "STOP",
            }
        )
        if not should_continue:
            stop_reason = "model_stop"
            break

        selected_k = decision_rank + 1
        for key, unit in _unique_pairs(
            list(candidates[decision_rank].get(units_key, []) or []), evidence_key
        ):
            if key not in selected_keys:
                selected_keys.add(key)
                support_units.append(unit)

        if selected_k == k_max:
            stop_reason = "k_max_reached"
        elif selected_k == len(candidates):
            stop_reason = "candidate_list_exhausted"

    for decision in decisions:
        decision["final_selected_k"] = selected_k
        decision["final_support_size"] = len(support_units)

    return {
        "mode": "adaptive_v2",
        "domain": policy.domain,
        "threshold": policy.threshold,
        "k_max": k_max,
        "epsilon": epsilon,
        "score_key": score_key,
        "candidate_adjusted_scores": scores,
        "score_scale": score_scale,
        "selected_k": selected_k,
        "support_units": support_units,
        "final_support_size": len(support_units),
        "decisions": decisions,
        "stop_reason": stop_reason,
    }
