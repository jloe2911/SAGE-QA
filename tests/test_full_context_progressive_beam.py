from __future__ import annotations

import inspect

from evaluation.compare_full_context_progressive_beam import (
    diagnostics,
    ontology_beam_candidates,
    progressive_beam,
    text_beam_candidates,
)


def test_progressive_beam_retains_fixed_width_at_each_depth() -> None:
    universe = ["a", "b", "c", "d"]

    def score(indices: tuple[int, ...]) -> float:
        return float(sum(index + 1 for index in indices))

    first = progressive_beam(universe, max_depth=3, beam_width=2, score=score)
    second = progressive_beam(universe, max_depth=3, beam_width=2, score=score)
    assert first == second
    assert [len(candidate) for candidate in first] == [1, 1, 2, 2, 3, 3]
    assert len({tuple(candidate) for candidate in first}) == len(first)


def test_posthoc_diagnostics_separate_retention_and_combination_misses() -> None:
    retained_miss = diagnostics(
        [["a"], ["c"]], [["a", "b"]], ["a", "b", "c"], max_depth=2
    )
    combined_miss = diagnostics(
        [["a"], ["b"]], [["a", "b"]], ["a", "b", "c"], max_depth=2
    )
    assert retained_miss["failure_category"] == "useful_evidence_not_retained"
    assert combined_miss["failure_category"] == "useful_units_retained_but_not_combined"


def test_candidate_generators_cannot_accept_gold_fields() -> None:
    forbidden = {"answer", "supporting_facts", "evidences", "gold", "label"}
    for function in (text_beam_candidates, ontology_beam_candidates):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)

