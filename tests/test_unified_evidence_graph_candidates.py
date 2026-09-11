from data_processing.evidence_graph_candidates import (
    EvidenceGraph,
    EvidenceUnit,
    ProgressiveExpansionConfig,
    progressive_connected_supports,
    query_local_candidate_closure,
)


def _graph() -> EvidenceGraph:
    units = tuple(EvidenceUnit(f"u{index}", f"unit {index}") for index in range(7))
    adjacency = (
        frozenset({1, 2}),
        frozenset({0, 3}),
        frozenset({0, 4}),
        frozenset({1, 5}),
        frozenset({2, 6}),
        frozenset({3}),
        frozenset({4}),
    )
    return EvidenceGraph(units, adjacency, (1.0, 0.9, 0.8, 0.3, 0.2, 0.1, 0.0))


def test_progressive_search_is_deterministic_connected_and_bounded():
    config = ProgressiveExpansionConfig(
        max_support_size=4,
        max_candidates=24,
        max_seeds=3,
        max_first_hops_per_seed=2,
        max_neighbors_per_state=3,
        per_branch_width=2,
    )
    first = progressive_connected_supports(_graph(), config)
    second = progressive_connected_supports(_graph(), config)

    assert first == second
    assert len(first.candidates) <= config.max_candidates
    assert {len(candidate) for candidate in first.candidates} >= {1, 2, 3, 4}

    index = {unit.unit_id: position for position, unit in enumerate(_graph().units)}
    for candidate in first.candidates:
        pending = {index[unit_id] for unit_id in candidate}
        reached = {pending.pop()}
        while pending:
            newly_reached = {
                node
                for node in pending
                if any(node in _graph().adjacency[parent] for parent in reached)
            }
            assert newly_reached
            reached.update(newly_reached)
            pending.difference_update(newly_reached)


def test_progressive_search_preserves_multiple_seeds_and_first_hop_branches():
    result = progressive_connected_supports(
        _graph(),
        ProgressiveExpansionConfig(
            max_support_size=3,
            max_candidates=30,
            max_seeds=2,
            max_first_hops_per_seed=2,
            max_neighbors_per_state=3,
            per_branch_width=1,
        ),
    )

    assert result.seed_unit_ids == ("u0", "u1")
    assert ("u0", "u1") in result.candidates
    assert ("u0", "u2") in result.candidates
    assert any("u3" in candidate for candidate in result.candidates)


def test_empty_graph_returns_no_candidates():
    graph = EvidenceGraph((), (), ())
    result = progressive_connected_supports(graph)
    assert result.candidates == ()
    assert result.explored_unit_ids == ()


def test_query_anchor_component_is_admitted_below_global_seed_cutoff():
    base = _graph()
    graph = EvidenceGraph(
        base.units,
        base.adjacency,
        base.query_scores,
        (False, False, False, False, False, True, False),
    )
    common = dict(
        max_support_size=3,
        max_candidates=12,
        max_seeds=1,
        max_first_hops_per_seed=1,
        max_neighbors_per_state=1,
        per_branch_width=1,
    )
    v1 = progressive_connected_supports(
        graph, ProgressiveExpansionConfig(**common, protect_query_anchors=False)
    )
    protected = progressive_connected_supports(
        graph, ProgressiveExpansionConfig(**common, protect_query_anchors=True)
    )

    assert v1.seed_unit_ids == ("u0",)
    assert "u5" not in v1.explored_unit_ids
    assert protected.seed_unit_ids == ("u0", "u5")
    assert protected.protected_anchor_unit_ids == ("u5",)
    assert protected.protected_component_count == 1
    assert ("u3", "u5") in protected.candidates


def test_connected_query_anchors_share_one_protected_component():
    base = _graph()
    graph = EvidenceGraph(
        base.units,
        base.adjacency,
        base.query_scores,
        (True, True, False, False, False, False, False),
    )
    result = progressive_connected_supports(
        graph,
        ProgressiveExpansionConfig(
            max_support_size=2,
            max_candidates=8,
            max_seeds=1,
            max_first_hops_per_seed=1,
            max_neighbors_per_state=1,
            per_branch_width=1,
        ),
    )

    assert result.protected_component_count == 1
    assert result.protected_anchor_unit_ids == ("u0",)
    assert ("u0", "u1") in result.candidates


def test_query_local_closure_materializes_every_anchor_pair_and_triple_before_cap():
    base = _graph()
    graph = EvidenceGraph(
        base.units,
        base.adjacency,
        base.query_scores,
        (True, False, True, False, False, False, False),
    )
    result = query_local_candidate_closure(
        graph,
        ProgressiveExpansionConfig(max_candidates=100, max_support_size=6),
    )
    pre_cap = set(result.pre_cap_candidate_indices)

    assert result.query_anchor_unit_ids == ("u0", "u2")
    assert {(0,), (2,), (0, 1), (0, 2), (2, 4)} <= pre_cap
    assert {(0, 1, 2), (0, 1, 3), (0, 2, 4), (2, 4, 6)} <= pre_cap
    assert len(pre_cap) == len(result.pre_cap_candidate_indices)


def test_query_local_cap_is_deterministic_and_preserves_anchor_and_size_diversity():
    units = tuple(EvidenceUnit(f"u{index}", f"unit {index}") for index in range(8))
    adjacency = (
        frozenset({1, 2}),
        frozenset({0, 3}),
        frozenset({0}),
        frozenset({1}),
        frozenset({5, 6}),
        frozenset({4, 7}),
        frozenset({4}),
        frozenset({5}),
    )
    graph = EvidenceGraph(
        units,
        adjacency,
        (1.0, 0.8, 0.7, 0.1, 0.9, 0.6, 0.5, 0.0),
        (True, False, False, False, True, False, False, False),
    )
    config = ProgressiveExpansionConfig(max_candidates=6, max_support_size=6)
    first = query_local_candidate_closure(graph, config)
    second = query_local_candidate_closure(graph, config)

    assert first == second
    assert len(first.candidates) == 6
    assert {len(candidate) for candidate in first.candidates} == {1, 2, 3}
    assert {"u0", "u4"} <= set().union(*(set(candidate) for candidate in first.candidates))
    assert first.local_candidate_count_before_cap > len(first.candidates)
    assert len(first.pre_cap_candidate_indices) > len(first.candidate_indices)


def test_query_local_closure_uses_remaining_budget_for_progressive_expansion():
    base = _graph()
    graph = EvidenceGraph(
        base.units,
        base.adjacency,
        base.query_scores,
        (True, False, False, False, False, False, False),
    )
    result = query_local_candidate_closure(
        graph,
        ProgressiveExpansionConfig(
            max_candidates=24,
            max_support_size=4,
            max_seeds=2,
            max_first_hops_per_seed=2,
            max_neighbors_per_state=3,
            per_branch_width=2,
        ),
    )

    assert result.progressive_candidate_count > 0
    assert any(len(candidate) == 4 for candidate in result.candidates)
