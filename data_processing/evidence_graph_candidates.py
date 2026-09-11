"""Generic leakage-free candidate construction over evidence-unit graphs.

The graph adapters are responsible only for producing evidence units, undirected
connectivity, and query-relevance scores.  The progressive constructor below is
modality agnostic: it never inspects evidence text, ontology syntax, answers,
labels, or reference supports.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


GENERATOR_D_VERSION = "generator_d_frozen_v1"
GENERATOR_D_VALIDATION_REPORT = (
    "outputs/development_runs/query_local_candidate_closure_v1/comparison.json"
)
GENERATOR_D_VALIDATION_SHA256 = "680EFCC8BB6AFAADCD0DD357F592331F8ED987C29635A866FEA555779C32EE5E"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CAPITALIZED_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9'-]*)(?:\s+[A-Z][A-Za-z0-9'-]*)*")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "other",
    "over",
    "same",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "were",
    "whose",
}


@dataclass(frozen=True)
class EvidenceUnit:
    """One addressable item of evidence in its native representation."""

    unit_id: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class EvidenceGraph:
    """A deterministic undirected graph over unique EvidenceUnits."""

    units: tuple[EvidenceUnit, ...]
    adjacency: tuple[frozenset[int], ...]
    query_scores: tuple[float, ...]
    query_anchors: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        size = len(self.units)
        if len(self.adjacency) != size or len(self.query_scores) != size:
            raise ValueError("units, adjacency, and query_scores must have equal length")
        if self.query_anchors and len(self.query_anchors) != size:
            raise ValueError("query_anchors must be empty or match the number of units")
        if len({unit.unit_id for unit in self.units}) != size:
            raise ValueError("EvidenceUnit IDs must be unique")
        for index, neighbors in enumerate(self.adjacency):
            if index in neighbors:
                raise ValueError("EvidenceGraph cannot contain self loops")
            if any(neighbor < 0 or neighbor >= size for neighbor in neighbors):
                raise ValueError("EvidenceGraph adjacency index out of range")
            if any(index not in self.adjacency[neighbor] for neighbor in neighbors):
                raise ValueError("EvidenceGraph adjacency must be symmetric")


@dataclass(frozen=True)
class ProgressiveExpansionConfig:
    """One fixed, deterministic search configuration for every modality."""

    max_support_size: int = 6
    max_candidates: int = 320
    max_seeds: int = 16
    max_first_hops_per_seed: int = 8
    max_neighbors_per_state: int = 12
    per_branch_width: int = 2
    protect_query_anchors: bool = True

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool):
                continue
            if value <= 0:
                raise ValueError(f"{name} must be positive")


# Frozen from the development-only Generator D comparison. Production callers
# use this singleton directly; dataset CLI options may validate it, but cannot
# tune or replace it.
GENERATOR_D_CONFIG = ProgressiveExpansionConfig(
    max_support_size=6,
    max_candidates=512,
    max_seeds=32,
    max_first_hops_per_seed=6,
    max_neighbors_per_state=12,
    per_branch_width=2,
    protect_query_anchors=True,
)


@dataclass(frozen=True)
class _State:
    indices: tuple[int, ...]
    seed: int
    branch: int
    score: float
    protected_component: int | None = None


@dataclass(frozen=True)
class ProgressiveExpansionResult:
    candidates: tuple[tuple[str, ...], ...]
    candidate_indices: tuple[tuple[int, ...], ...]
    seed_unit_ids: tuple[str, ...]
    explored_unit_ids: tuple[str, ...]
    generated_state_count: int
    protected_anchor_unit_ids: tuple[str, ...] = ()
    protected_component_count: int = 0


@dataclass(frozen=True)
class QueryLocalClosureResult:
    """Candidates from query-local closure followed by progressive expansion."""

    candidates: tuple[tuple[str, ...], ...]
    candidate_indices: tuple[tuple[int, ...], ...]
    pre_cap_candidate_indices: tuple[tuple[int, ...], ...]
    query_anchor_unit_ids: tuple[str, ...]
    explored_unit_ids: tuple[str, ...]
    local_candidate_count_before_cap: int
    retained_local_candidate_count: int
    progressive_candidate_count: int
    generated_state_count: int
    query_anchor_component_count: int


def frozen_generator_d_config() -> dict[str, int | bool]:
    """Return a serializable copy of the immutable production configuration."""
    return dict(vars(GENERATOR_D_CONFIG))


def frozen_generator_d_lineage() -> dict[str, str]:
    return {
        "version": GENERATOR_D_VERSION,
        "validation_report": GENERATOR_D_VALIDATION_REPORT,
        "validation_report_sha256": GENERATOR_D_VALIDATION_SHA256,
    }


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(str(text))
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _make_graph(
    units: Sequence[EvidenceUnit],
    adjacency: Sequence[set[int]],
    scores: Sequence[float],
    anchors: Sequence[bool],
) -> EvidenceGraph:
    return EvidenceGraph(
        units=tuple(units),
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        query_scores=tuple(float(score) for score in scores),
        query_anchors=tuple(bool(anchor) for anchor in anchors),
    )


def build_text_evidence_graph(
    sentence_records: Sequence[Mapping[str, Any]], question: str
) -> EvidenceGraph:
    """Adapt context sentences to Generator D's validated text graph."""
    records_by_unit = {str(record["unit"]): record for record in sentence_records}
    ordered = [records_by_unit[key] for key in sorted(records_by_unit)]
    units = [
        EvidenceUnit(
            unit_id=str(record["unit"]),
            content=str(record.get("sentence", "")),
            metadata={
                "title": str(record.get("title", "")),
                "sentence_index": int(record.get("sent_idx", -1)),
            },
        )
        for record in ordered
    ]
    adjacency = [set() for _ in units]
    question_tokens = _tokens(question)
    title_tokens = [_tokens(str(unit.metadata["title"])) for unit in units]
    sentence_tokens = [_tokens(unit.content) for unit in units]
    named_entities = [
        {
            " ".join(token.lower() for token in _TOKEN_RE.findall(match.group(0)))
            for match in _CAPITALIZED_RE.finditer(unit.content)
            if _TOKEN_RE.findall(match.group(0))
        }
        for unit in units
    ]

    def connect(left: int, right: int) -> None:
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)

    by_title: dict[str, list[int]] = collections.defaultdict(list)
    token_index: dict[str, list[int]] = collections.defaultdict(list)
    entity_index: dict[str, list[int]] = collections.defaultdict(list)
    bridge_index: dict[str, list[int]] = collections.defaultdict(list)
    for index, unit in enumerate(units):
        by_title[str(unit.metadata["title"]).casefold()].append(index)
        for token in title_tokens[index]:
            token_index[f"title:{token}"].append(index)
        for entity in named_entities[index]:
            entity_index[entity].append(index)
        for token in sentence_tokens[index]:
            if len(token) >= 4:
                bridge_index[token].append(index)

    for indices in by_title.values():
        indices.sort(key=lambda index: (int(units[index].metadata["sentence_index"]), index))
        for left, right in zip(indices, indices[1:]):
            connect(left, right)
    for index, tokens in enumerate(sentence_tokens):
        for token in tokens:
            for title_index in token_index.get(f"title:{token}", ()):
                connect(index, title_index)
    for index_group in entity_index.values():
        if 1 < len(index_group) <= 12:
            for position, left in enumerate(index_group):
                for right in index_group[position + 1 :]:
                    connect(left, right)
    for index_group in bridge_index.values():
        if 1 < len(index_group) <= 8:
            for position, left in enumerate(index_group):
                for right in index_group[position + 1 :]:
                    connect(left, right)

    denominator = max(len(question_tokens), 1)
    scores: list[float] = []
    anchors: list[bool] = []
    for index in range(len(units)):
        visible_tokens = sentence_tokens[index] | title_tokens[index]
        content_overlap = len(question_tokens & visible_tokens) / denominator
        title_overlap = len(question_tokens & title_tokens[index]) / denominator
        scores.append(content_overlap + 0.25 * title_overlap + 0.01 * min(len(adjacency[index]), 8))
        anchors.append(bool(question_tokens & visible_tokens))
    return _make_graph(units, adjacency, scores, anchors)


def build_ontology_evidence_graph(
    context_axioms: Sequence[str],
    *,
    unit_signature: Callable[[str], tuple[set[str], set[str]]],
    query_entities: set[str],
    query_properties: set[str],
    score_unit: Callable[[str, set[str], set[str], int], float],
) -> EvidenceGraph:
    """Adapt ontology facts/axioms to Generator D's validated ontology graph."""
    ordered_axioms = sorted(set(context_axioms))
    units = [EvidenceUnit(unit_id=axiom, content=axiom) for axiom in ordered_axioms]
    adjacency = [set() for _ in units]
    signatures = [unit_signature(unit.unit_id) for unit in units]
    inverted: dict[str, list[int]] = collections.defaultdict(list)
    for index, (entities, properties) in enumerate(signatures):
        for term in entities | properties:
            inverted[str(term).casefold()].append(index)
    for indices in inverted.values():
        if len(indices) > 64:
            continue
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
    scores = [
        score_unit(unit.unit_id, query_entities, query_properties, len(adjacency[index]))
        for index, unit in enumerate(units)
    ]
    anchors = [
        bool((entities & query_entities) or (properties & query_properties))
        for entities, properties in signatures
    ]
    return _make_graph(units, adjacency, scores, anchors)


def _state_score(indices: tuple[int, ...], graph: EvidenceGraph, seed: int) -> float:
    relevance = sum(graph.query_scores[index] for index in indices) / len(indices)
    internal_edges = sum(
        right in graph.adjacency[left]
        for position, left in enumerate(indices)
        for right in indices[position + 1 :]
    )
    frontier = set().union(*(graph.adjacency[index] for index in indices))
    frontier.difference_update(indices)
    # Relevance dominates; small structural terms prefer coherent expandable
    # supports without imposing a modality-specific proof template.
    return (
        relevance
        + 0.025 * internal_edges
        + 0.002 * min(len(frontier), 10)
        + 0.0001 * graph.query_scores[seed]
        - 0.004 * max(0, len(indices) - 3)
    )


def _rank_indices(indices: Iterable[int], graph: EvidenceGraph) -> list[int]:
    return sorted(
        set(indices),
        key=lambda index: (-graph.query_scores[index], graph.units[index].unit_id),
    )


def _query_anchor_components(graph: EvidenceGraph) -> tuple[tuple[int, ...], ...]:
    """Return deterministic components in the query-anchor-induced graph."""
    anchors = {index for index, anchored in enumerate(graph.query_anchors) if anchored}
    components: list[tuple[int, ...]] = []
    while anchors:
        start = min(anchors, key=lambda index: graph.units[index].unit_id)
        pending = [start]
        reached = {start}
        anchors.remove(start)
        while pending:
            current = pending.pop()
            neighbors = graph.adjacency[current] & anchors
            for neighbor in sorted(neighbors, key=lambda index: graph.units[index].unit_id):
                anchors.remove(neighbor)
                reached.add(neighbor)
                pending.append(neighbor)
        components.append(tuple(_rank_indices(reached, graph)))
    return tuple(components)


def _full_component_ids(graph: EvidenceGraph) -> tuple[int, ...]:
    """Return stable full-graph component IDs for diversity allocation."""
    component_ids = [-1] * len(graph.units)
    remaining = set(range(len(graph.units)))
    component = 0
    while remaining:
        start = min(remaining, key=lambda index: graph.units[index].unit_id)
        pending = [start]
        remaining.remove(start)
        component_ids[start] = component
        while pending:
            current = pending.pop()
            for neighbor in sorted(
                graph.adjacency[current] & remaining,
                key=lambda index: graph.units[index].unit_id,
            ):
                remaining.remove(neighbor)
                component_ids[neighbor] = component
                pending.append(neighbor)
        component += 1
    return tuple(component_ids)


def _interleave_anchor_components(
    anchors: Sequence[int], component_ids: Sequence[int], graph: EvidenceGraph
) -> list[int]:
    """Interleave anchors so no large graph component monopolizes the cap."""
    by_component: dict[int, list[int]] = {}
    for anchor in anchors:
        by_component.setdefault(component_ids[anchor], []).append(anchor)
    for component_anchors in by_component.values():
        component_anchors.sort(key=lambda index: graph.units[index].unit_id)

    ordered: list[int] = []
    offset = 0
    components = sorted(by_component)
    while True:
        added = False
        for component in components:
            component_anchors = by_component[component]
            if offset < len(component_anchors):
                ordered.append(component_anchors[offset])
                added = True
        if not added:
            return ordered
        offset += 1


def _allocate_query_local_candidates(
    lanes: Mapping[int, Mapping[int, Sequence[_State]]],
    anchors: Sequence[int],
    component_ids: Sequence[int],
    graph: EvidenceGraph,
    capacity: int,
) -> list[_State]:
    """Round-robin across components, anchors, and candidate sizes.

    Query relevance orders candidates only within an anchor-and-size lane.  The
    outer allocation never compares scores across anchors or components.
    """
    if capacity <= 0:
        return []
    ordered_anchors = _interleave_anchor_components(anchors, component_ids, graph)
    positions = {(anchor, size): 0 for anchor in ordered_anchors for size in (1, 2, 3)}
    selected: list[_State] = []
    selected_indices: set[tuple[int, ...]] = set()
    while len(selected) < capacity:
        added = False
        for anchor in ordered_anchors:
            for size in (1, 2, 3):
                lane = lanes.get(anchor, {}).get(size, ())
                position_key = (anchor, size)
                position = positions[position_key]
                while position < len(lane) and lane[position].indices in selected_indices:
                    position += 1
                positions[position_key] = position
                if position >= len(lane):
                    continue
                state = lane[position]
                positions[position_key] = position + 1
                selected.append(state)
                selected_indices.add(state.indices)
                added = True
                if len(selected) >= capacity:
                    return selected
        if not added:
            return selected
    return selected


def _select_diverse_candidates(
    states: Sequence[_State], graph: EvidenceGraph, config: ProgressiveExpansionConfig
) -> list[_State]:
    """Reserve capacity by support size, seed, and first-hop branch."""
    unique: dict[tuple[int, ...], _State] = {}
    for state in states:
        previous = unique.get(state.indices)
        if previous is None or (
            state.protected_component is None,
            -state.score,
            graph.units[state.seed].unit_id,
            graph.units[state.branch].unit_id,
        ) < (
            previous.protected_component is None,
            -previous.score,
            graph.units[previous.seed].unit_id,
            graph.units[previous.branch].unit_id,
        ):
            unique[state.indices] = state

    buckets: dict[int, dict[tuple[int, int], list[_State]]] = {}
    for state in unique.values():
        buckets.setdefault(len(state.indices), {}).setdefault(
            (state.seed, state.branch), []
        ).append(state)
    for lanes in buckets.values():
        for lane_states in lanes.values():
            lane_states.sort(key=lambda state: (-state.score, state.indices))

    protected = sorted(
        (state for state in unique.values() if state.protected_component is not None),
        key=lambda state: (
            # Preserve a connected first expansion for every component first,
            # then its singleton and successively longer path states.
            0 if len(state.indices) == 2 else 1,
            len(state.indices),
            int(state.protected_component),
            -state.score,
            state.indices,
        ),
    )[: config.max_candidates]
    selected: list[_State] = list(protected)
    selected_indices: set[tuple[int, ...]] = {state.indices for state in selected}
    size_count = max(1, min(config.max_support_size, len(graph.units)))
    base_quota = config.max_candidates // size_count
    remainder = config.max_candidates % size_count

    def take_round_robin(size: int, quota: int) -> None:
        lanes = buckets.get(size, {})
        ordered_lanes = sorted(
            lanes,
            key=lambda lane: (
                graph.units[lane[0]].unit_id,
                graph.units[lane[1]].unit_id,
            ),
        )
        offset = 0
        while len([state for state in selected if len(state.indices) == size]) < quota:
            added = False
            for lane in ordered_lanes:
                lane_states = lanes[lane]
                if offset >= len(lane_states):
                    continue
                state = lane_states[offset]
                if state.indices not in selected_indices:
                    selected.append(state)
                    selected_indices.add(state.indices)
                    added = True
                    if len([item for item in selected if len(item.indices) == size]) >= quota:
                        break
            if not added:
                break
            offset += 1

    for size in range(1, size_count + 1):
        take_round_robin(size, base_quota + int(size <= remainder))

    if len(selected) < config.max_candidates:
        remaining = sorted(
            (state for state in unique.values() if state.indices not in selected_indices),
            key=lambda state: (
                -state.score,
                len(state.indices),
                graph.units[state.seed].unit_id,
                graph.units[state.branch].unit_id,
                state.indices,
            ),
        )
        selected.extend(remaining[: config.max_candidates - len(selected)])
    return selected[: config.max_candidates]


def progressive_connected_supports(
    graph: EvidenceGraph,
    config: ProgressiveExpansionConfig = ProgressiveExpansionConfig(),
) -> ProgressiveExpansionResult:
    """Generate connected supports with generic query-anchor path protection.

    Query anchors come solely from the evidence adapter's query-visible signals.
    One representative of every anchor-induced component enters the frontier,
    even below the global seed cutoff.  Its strongest connected path is carried
    through each reachable depth before the unchanged lane search and final
    diversity allocation use the remaining capacity.
    """
    if not graph.units:
        return ProgressiveExpansionResult((), (), (), (), 0)

    max_size = min(config.max_support_size, len(graph.units))
    anchor_components = _query_anchor_components(graph) if config.protect_query_anchors else ()
    protected_seeds = tuple(component[0] for component in anchor_components)
    global_seeds = _rank_indices(range(len(graph.units)), graph)[: config.max_seeds]
    seeds = list(global_seeds)
    for seed in protected_seeds:
        if seed not in seeds:
            seeds.append(seed)
    protected_by_seed = {seed: component for component, seed in enumerate(protected_seeds)}
    all_states: list[_State] = []
    explored = set(seeds)
    frontier: list[_State] = []

    for seed in seeds:
        singleton = (seed,)
        state = _State(
            singleton,
            seed,
            seed,
            _state_score(singleton, graph, seed),
            protected_by_seed.get(seed),
        )
        all_states.append(state)
        frontier.append(state)

    for target_size in range(2, max_size + 1):
        by_lane: dict[tuple[int, int], dict[tuple[int, ...], _State]] = {}
        protected_expansions: dict[int, _State] = {}
        for state in frontier:
            selected = set(state.indices)
            neighbors = set().union(*(graph.adjacency[index] for index in state.indices))
            all_ranked_neighbors = _rank_indices(neighbors - selected, graph)
            ranked_neighbors = all_ranked_neighbors[: config.max_neighbors_per_state]
            if len(state.indices) == 1:
                ranked_neighbors = ranked_neighbors[: config.max_first_hops_per_seed]
            for neighbor in ranked_neighbors:
                indices = tuple(sorted((*state.indices, neighbor)))
                branch = neighbor if len(state.indices) == 1 else state.branch
                expanded = _State(
                    indices,
                    state.seed,
                    branch,
                    _state_score(indices, graph, state.seed),
                )
                lane = (expanded.seed, expanded.branch)
                prior = by_lane.setdefault(lane, {}).get(indices)
                if prior is None or (-expanded.score, expanded.indices) < (
                    -prior.score,
                    prior.indices,
                ):
                    by_lane[lane][indices] = expanded
                explored.add(neighbor)

            # Carry exactly one strongest connected path state per protected
            # anchor component outside ordinary neighbor and branch truncation.
            if state.protected_component is not None:
                for neighbor in all_ranked_neighbors:
                    indices = tuple(sorted((*state.indices, neighbor)))
                    branch = neighbor if len(state.indices) == 1 else state.branch
                    expanded = _State(
                        indices,
                        state.seed,
                        branch,
                        _state_score(indices, graph, state.seed),
                        state.protected_component,
                    )
                    prior = protected_expansions.get(state.protected_component)
                    if prior is None or (-expanded.score, expanded.indices) < (
                        -prior.score,
                        prior.indices,
                    ):
                        protected_expansions[state.protected_component] = expanded
                    explored.add(neighbor)

        frontier = []
        for lane in sorted(
            by_lane,
            key=lambda item: (
                graph.units[item[0]].unit_id,
                graph.units[item[1]].unit_id,
            ),
        ):
            retained = sorted(
                by_lane[lane].values(), key=lambda state: (-state.score, state.indices)
            )[: config.per_branch_width]
            frontier.extend(retained)
        for expanded in protected_expansions.values():
            if not any(
                state.indices == expanded.indices
                and state.seed == expanded.seed
                and state.branch == expanded.branch
                for state in frontier
            ):
                frontier.append(expanded)
            else:
                frontier = [
                    expanded
                    if state.indices == expanded.indices
                    and state.seed == expanded.seed
                    and state.branch == expanded.branch
                    else state
                    for state in frontier
                ]
        all_states.extend(frontier)
        if not frontier:
            break

    selected = _select_diverse_candidates(all_states, graph, config)
    candidate_indices = tuple(state.indices for state in selected)
    candidates = tuple(
        tuple(graph.units[index].unit_id for index in indices) for indices in candidate_indices
    )
    return ProgressiveExpansionResult(
        candidates=candidates,
        candidate_indices=candidate_indices,
        seed_unit_ids=tuple(graph.units[index].unit_id for index in seeds),
        explored_unit_ids=tuple(
            graph.units[index].unit_id
            for index in sorted(explored, key=lambda i: graph.units[i].unit_id)
        ),
        generated_state_count=len(all_states),
        protected_anchor_unit_ids=tuple(graph.units[index].unit_id for index in protected_seeds),
        protected_component_count=len(anchor_components),
    )


def query_local_candidate_closure(
    graph: EvidenceGraph,
    config: ProgressiveExpansionConfig = ProgressiveExpansionConfig(),
) -> QueryLocalClosureResult:
    """Materialize query-local size-1/2/3 closure before global pruning.

    This constructor is deliberately modality agnostic.  It consumes only the
    shared graph, its query-visible anchor flags, and its existing query scores.
    Every anchor singleton, every anchor-neighbor pair, and every connected
    one-step extension of those pairs is materialized before capacity is
    allocated.  Any remaining capacity is filled by the existing progressive
    expansion mechanism through ``config.max_support_size``.
    """
    if not graph.units:
        return QueryLocalClosureResult((), (), (), (), (), 0, 0, 0, 0, 0)

    anchors = _rank_indices(
        (index for index, anchored in enumerate(graph.query_anchors) if anchored), graph
    )
    component_ids = _full_component_ids(graph)
    lanes: dict[int, dict[int, list[_State]]] = {}
    local_unique: set[tuple[int, ...]] = set()
    explored = set(anchors)
    generated_local_states = 0

    for anchor in anchors:
        by_size: dict[int, dict[tuple[int, ...], _State]] = {
            1: {},
            2: {},
            3: {},
        }
        singleton = (anchor,)
        by_size[1][singleton] = _State(
            singleton,
            anchor,
            anchor,
            _state_score(singleton, graph, anchor),
            component_ids[anchor],
        )
        generated_local_states += 1
        explored.add(anchor)

        for neighbor in sorted(
            graph.adjacency[anchor], key=lambda index: graph.units[index].unit_id
        ):
            pair = tuple(sorted((anchor, neighbor)))
            by_size[2][pair] = _State(
                pair,
                anchor,
                neighbor,
                _state_score(pair, graph, anchor),
                component_ids[anchor],
            )
            generated_local_states += 1
            explored.add(neighbor)
            frontier = (graph.adjacency[anchor] | graph.adjacency[neighbor]) - set(pair)
            for extension in sorted(frontier, key=lambda index: graph.units[index].unit_id):
                triple = tuple(sorted((*pair, extension)))
                prior = by_size[3].get(triple)
                state = _State(
                    triple,
                    anchor,
                    neighbor,
                    _state_score(triple, graph, anchor),
                    component_ids[anchor],
                )
                if prior is None or (-state.score, state.indices) < (
                    -prior.score,
                    prior.indices,
                ):
                    by_size[3][triple] = state
                generated_local_states += 1
                explored.add(extension)

        lanes[anchor] = {
            size: sorted(states.values(), key=lambda state: (-state.score, state.indices))
            for size, states in by_size.items()
        }
        local_unique.update(state.indices for states in lanes[anchor].values() for state in states)

    retained_local = _allocate_query_local_candidates(
        lanes,
        anchors,
        component_ids,
        graph,
        min(config.max_candidates, len(local_unique)),
    )
    selected = list(retained_local)
    selected_indices = {state.indices for state in selected}

    progressive = progressive_connected_supports(graph, config)
    progressive_states = [
        _State(
            indices,
            indices[0],
            indices[0],
            _state_score(indices, graph, indices[0]),
        )
        for indices in progressive.candidate_indices
    ]
    for state in progressive_states:
        if len(selected) >= config.max_candidates:
            break
        if state.indices not in selected_indices:
            selected.append(state)
            selected_indices.add(state.indices)

    pre_cap = sorted(
        local_unique | {state.indices for state in progressive_states},
        key=lambda indices: (len(indices), indices),
    )
    candidate_indices = tuple(state.indices for state in selected)
    candidates = tuple(
        tuple(graph.units[index].unit_id for index in indices) for indices in candidate_indices
    )
    return QueryLocalClosureResult(
        candidates=candidates,
        candidate_indices=candidate_indices,
        pre_cap_candidate_indices=tuple(pre_cap),
        query_anchor_unit_ids=tuple(graph.units[index].unit_id for index in anchors),
        explored_unit_ids=tuple(
            graph.units[index].unit_id
            for index in sorted(
                explored
                | {index for indices in progressive.candidate_indices for index in indices},
                key=lambda index: graph.units[index].unit_id,
            )
        ),
        local_candidate_count_before_cap=len(local_unique),
        retained_local_candidate_count=len(retained_local),
        progressive_candidate_count=len(selected) - len(retained_local),
        generated_state_count=generated_local_states + progressive.generated_state_count,
        query_anchor_component_count=len({component_ids[anchor] for anchor in anchors}),
    )


def generate_generator_d_candidates(graph: EvidenceGraph) -> QueryLocalClosureResult:
    """Run the immutable production Generator D implementation."""
    return query_local_candidate_closure(graph, GENERATOR_D_CONFIG)
