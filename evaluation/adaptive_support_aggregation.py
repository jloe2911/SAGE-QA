"""Generic post-ranking adaptive support aggregation.

The policy deliberately accepts only ranked candidate scores and candidate
evidence units.  Gold answers, gold support, answer type, and split labels are
not part of its interface.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any


MAX_ADAPTIVE_K = 5
DEFAULT_EPSILON = 1e-12


def canonical_evidence_key(unit: Any) -> Hashable:
    """Return a deterministic hashable identity for an evidence unit.

    Ontology and current text artifacts use strings, which pass through
    unchanged.  The recursive cases keep the helper reusable for future
    structured evidence representations.
    """

    if isinstance(unit, Mapping):
        return tuple(
            sorted((str(key), canonical_evidence_key(value)) for key, value in unit.items())
        )
    if isinstance(unit, Sequence) and not isinstance(unit, (str, bytes, bytearray)):
        return tuple(canonical_evidence_key(value) for value in unit)
    if not isinstance(unit, Hashable):
        raise TypeError(f"Evidence unit is not hashable or canonically supported: {unit!r}")
    return unit


def _deduplicated_units(
    units: Sequence[Any], evidence_key: Callable[[Any], Hashable]
) -> list[tuple[Hashable, Any]]:
    unique: list[tuple[Hashable, Any]] = []
    seen: set[Hashable] = set()
    for unit in units:
        key = evidence_key(unit)
        if key not in seen:
            seen.add(key)
            unique.append((key, unit))
    return unique


def _validate_budget(k_max: int) -> None:
    if not 1 <= k_max <= MAX_ADAPTIVE_K:
        raise ValueError(f"k_max must be in [1, {MAX_ADAPTIVE_K}], got {k_max}")


def adaptive_support_aggregate(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    tau: float,
    k_max: int = MAX_ADAPTIVE_K,
    epsilon: float = DEFAULT_EPSILON,
    score_key: str = "adjusted_score",
    units_key: str = "subgraph_units",
    evidence_key: Callable[[Any], Hashable] = canonical_evidence_key,
) -> dict[str, Any]:
    """Select and aggregate a ranked prefix with the adaptive stopping rule.

    Candidate order must already be the final post-reranking order.  The
    function never re-scores or re-ranks candidates.
    """

    _validate_budget(k_max)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
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

    mean_score = sum(scores) / len(scores)
    score_scale = math.sqrt(
        sum((score - mean_score) ** 2 for score in scores) / len(scores)
    )
    identical_scores = score_scale <= epsilon

    first_units = _deduplicated_units(
        list(candidates[0].get(units_key, []) or []), evidence_key
    )
    selected_keys = {key for key, _ in first_units}
    selected_support = [unit for _, unit in first_units]
    selected_indices = [1]
    decisions: list[dict[str, Any]] = []
    stopping_point: int | None = None

    for candidate_index, (candidate, score) in enumerate(
        zip(candidates[1:], scores[1:]), start=2
    ):
        candidate_units = _deduplicated_units(
            list(candidate.get(units_key, []) or []), evidence_key
        )
        new_units = [(key, unit) for key, unit in candidate_units if key not in selected_keys]
        novelty = len(new_units) / len(candidate_units) if candidate_units else 0.0
        relative_strength = (
            1.0
            if identical_scores
            else math.exp((score - scores[0]) / max(score_scale, epsilon))
        )
        utility = relative_strength * novelty
        include = utility >= tau

        decisions.append(
            {
                "candidate_index": candidate_index,
                "rank": int(candidate.get("rank", candidate_index)),
                "score": score,
                "relative_strength": relative_strength,
                "candidate_support_size": len(candidate_units),
                "new_evidence_count": len(new_units),
                "R_i": relative_strength,
                "N_i": novelty,
                "U_i": utility,
                "marginal_novelty": novelty,
                "marginal_utility": utility,
                "included": include,
            }
        )

        if not include:
            stopping_point = candidate_index
            break

        selected_indices.append(candidate_index)
        for key, unit in new_units:
            selected_keys.add(key)
            selected_support.append(unit)

    if stopping_point is not None:
        stop_reason = "utility_below_tau"
    elif len(candidates) < k_max:
        stop_reason = "candidate_list_exhausted"
    else:
        stop_reason = "k_max_reached"

    return {
        "mode": "adaptive",
        "tau": tau,
        "k_max": k_max,
        "epsilon": epsilon,
        "score_key": score_key,
        "candidate_scores": scores,
        "score_scale": score_scale,
        "identical_scores": identical_scores,
        "selected_candidate_indices": selected_indices,
        "selected_k": len(selected_indices),
        "support_units": selected_support,
        "final_support_size": len(selected_support),
        "decisions": decisions,
        "stopping_point": stopping_point,
        "stop_reason": stop_reason,
    }


def fixed_k_support_aggregate(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    k: int,
    units_key: str = "subgraph_units",
    evidence_key: Callable[[Any], Hashable] = canonical_evidence_key,
) -> dict[str, Any]:
    """Return the existing ordered, deduplicated fixed-k support union."""

    _validate_budget(k)
    if not ranked_candidates:
        raise ValueError("At least one ranked candidate is required")

    selected = list(ranked_candidates[:k])
    support: list[Any] = []
    seen: set[Hashable] = set()
    for candidate in selected:
        for key, unit in _deduplicated_units(
            list(candidate.get(units_key, []) or []), evidence_key
        ):
            if key not in seen:
                seen.add(key)
                support.append(unit)

    return {
        "mode": "fixed",
        "requested_k": k,
        "selected_k": len(selected),
        "support_units": support,
        "final_support_size": len(support),
    }
