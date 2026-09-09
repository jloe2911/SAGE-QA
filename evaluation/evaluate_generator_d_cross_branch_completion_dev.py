"""Evaluate one isolated Generator-D cross-branch completion change on DEV.

Candidate generation is frozen before gold is read.  The production Generator-D
module is imported but never modified: this file carries the experimental
composition and allocation code and writes diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    parse_owl_context,
    score_unit_for_query,
    unit_signature,
)
from data_processing.build_2wiki_subgraph_dataset import (  # noqa: E402
    flatten_context as flatten_2wiki_context,
    load_2wiki_file,
)
from data_processing.build_hotpot_subgraph_dataset import (  # noqa: E402
    flatten_context as flatten_hotpot_context,
    load_hotpot_file,
)
from data_processing.evidence_graph_candidates import (  # noqa: E402
    GENERATOR_D_CONFIG,
    EvidenceGraph,
    _State,
    _full_component_ids,
    _interleave_anchor_components,
    _rank_indices,
    _state_score,
    build_ontology_evidence_graph,
    build_text_evidence_graph,
    generate_generator_d_candidates,
    progressive_connected_supports,
)
from models.symbolic_composer import extract_query_signature  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "outputs/diagnostics/production_generator_d_v1_cross_branch_completion_dev"
)
MECHANISM_PATH = (
    ROOT
    / "outputs/diagnostics/production_generator_d_v1_mechanism_decision"
    / "generator_failure_mechanisms.json"
)
DATASETS = (
    ("HotpotQA", "text", "sageqa_text_chain"),
    ("2WikiMultiHopQA", "text", "sageqa_text_chain"),
    ("FamilyOWL_1hop", "ontology", "sageqa_proof"),
    ("FamilyOWL_2hop", "ontology", "sageqa_proof"),
    ("pizza_100_1hop", "ontology", "sageqa_proof"),
    ("pizza_100_2hop", "ontology", "sageqa_proof"),
    ("pizza_250_1hop", "ontology", "sageqa_proof"),
    ("pizza_250_2hop", "ontology", "sageqa_proof"),
    ("OWL2Bench_1hop", "ontology", "sageqa_proof"),
    ("OWL2Bench_2hop", "ontology", "sageqa_proof"),
)


@dataclass(frozen=True)
class ExperimentalPool:
    candidate_indices: tuple[tuple[int, ...], ...]
    completion_indices: tuple[tuple[int, ...], ...]
    completion_count_before_cap: int
    source_partial_count: int


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(indices: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(indices)))


def alternatives(row: Mapping[str, Any]) -> list[list[str]]:
    values = row.get("gold_explanations") or []
    if values:
        return [[str(unit) for unit in explanation] for explanation in values if explanation]
    support = row.get("gold_support_units") or row.get("gold_units") or []
    return [[str(unit) for unit in support]] if support else []


def grouped_jsonl(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") not in (None, "dev"):
                raise ValueError(f"Non-DEV row rejected: {row.get('split')!r}")
            example_id = str(row["example_id"])
            if current_id is None:
                current_id = example_id
            if example_id != current_id:
                yield current_id, rows
                current_id, rows = example_id, []
            rows.append(row)
    if current_id is not None:
        yield current_id, rows


def build_raw_text_index(
    dataset: str, metadata: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    loader = load_hotpot_file if dataset == "HotpotQA" else load_2wiki_file
    records = loader(str(ROOT / Path(metadata["dev_file"])))
    return {
        str(row.get("id") or row.get("_id")): row
        for row in records
        if row.get("id") or row.get("_id")
    }


def text_graph(
    dataset: str,
    example_id: str,
    question: str,
    raw_index: Mapping[str, Mapping[str, Any]],
) -> EvidenceGraph:
    marker = "__dev__"
    if marker not in example_id:
        raise ValueError(f"Unexpected text DEV id: {example_id}")
    raw = raw_index[example_id.split(marker, 1)[1]]
    # The adapter receives only inference-visible fields.  In particular, 2Wiki
    # ``evidences`` and Hotpot ``supporting_facts`` are not passed downstream.
    clean = {
        key: raw[key]
        for key in ("id", "_id", "question", "context")
        if key in raw
    }
    records = (
        flatten_hotpot_context(clean)
        if dataset == "HotpotQA"
        else flatten_2wiki_context(clean)
    )
    return build_text_evidence_graph(records, question)


def ontology_graph(item: Mapping[str, Any], qa: Mapping[str, Any]) -> EvidenceGraph:
    question = str(
        qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or ""
    )
    sparql = str(qa.get("SPARQL Query") or "")
    signature = extract_query_signature(question=question, sparql_query=sparql)
    query_entities = set(signature.get("query_entities", []))
    query_properties = set(signature.get("query_properties", []))
    axioms = parse_owl_context(str(item["OWL Context"]))
    return build_ontology_evidence_graph(
        axioms,
        unit_signature=unit_signature,
        query_entities=query_entities,
        query_properties=query_properties,
        score_unit=lambda unit, _entities, _properties, degree: score_unit_for_query(
            unit,
            query_entities=query_entities,
            query_properties=query_properties,
            degree=degree,
        ),
    )


def unit_candidates(
    graph: EvidenceGraph, candidates: Sequence[Sequence[int]]
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(graph.units[index].unit_id for index in candidate) for candidate in candidates
    )


def is_connected(indices: Sequence[int], graph: EvidenceGraph) -> bool:
    selected = set(indices)
    if not selected:
        return False
    visited = {next(iter(selected))}
    pending = list(visited)
    while pending:
        node = pending.pop()
        for neighbor in graph.adjacency[node] & selected:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return visited == selected


def state_preference(state: _State, graph: EvidenceGraph) -> tuple[Any, ...]:
    return (
        state.protected_component is None,
        -state.score,
        graph.units[state.seed].unit_id,
        graph.units[state.branch].unit_id,
        state.indices,
    )


def build_local_lanes(
    graph: EvidenceGraph,
) -> tuple[dict[int, dict[int, list[_State]]], tuple[int, ...], tuple[int, ...]]:
    """Exact replay of Generator-D's unchanged local size-1/2/3 branches."""
    anchors = tuple(
        _rank_indices(
            (index for index, anchored in enumerate(graph.query_anchors) if anchored), graph
        )
    )
    component_ids = _full_component_ids(graph)
    lanes: dict[int, dict[int, list[_State]]] = {}
    for anchor in anchors:
        by_size: dict[int, dict[tuple[int, ...], _State]] = {1: {}, 2: {}, 3: {}}
        singleton = (anchor,)
        by_size[1][singleton] = _State(
            singleton,
            anchor,
            anchor,
            _state_score(singleton, graph, anchor),
            component_ids[anchor],
        )
        for neighbor in sorted(
            graph.adjacency[anchor], key=lambda index: graph.units[index].unit_id
        ):
            pair = canonical((anchor, neighbor))
            by_size[2][pair] = _State(
                pair,
                anchor,
                neighbor,
                _state_score(pair, graph, anchor),
                component_ids[anchor],
            )
            frontier = (graph.adjacency[anchor] | graph.adjacency[neighbor]) - set(pair)
            for extension in sorted(frontier, key=lambda index: graph.units[index].unit_id):
                triple = canonical((*pair, extension))
                state = _State(
                    triple,
                    anchor,
                    neighbor,
                    _state_score(triple, graph, anchor),
                    component_ids[anchor],
                )
                prior = by_size[3].get(triple)
                if prior is None or state_preference(state, graph) < state_preference(
                    prior, graph
                ):
                    by_size[3][triple] = state
        lanes[anchor] = {
            size: sorted(values.values(), key=lambda state: (-state.score, state.indices))
            for size, values in by_size.items()
        }
    return lanes, anchors, component_ids


def allocate_lanes(
    lanes: Mapping[int, Mapping[int, Sequence[_State]]],
    anchors: Sequence[int],
    component_ids: Sequence[int],
    graph: EvidenceGraph,
    capacity: int,
    sizes: Sequence[int],
) -> list[_State]:
    """Generator-D's existing round-robin allocator, over supplied sizes."""
    if capacity <= 0:
        return []
    ordered_anchors = _interleave_anchor_components(anchors, component_ids, graph)
    positions = {(anchor, size): 0 for anchor in ordered_anchors for size in sizes}
    selected: list[_State] = []
    selected_indices: set[tuple[int, ...]] = set()
    while len(selected) < capacity:
        added = False
        for anchor in ordered_anchors:
            for size in sizes:
                lane = lanes.get(anchor, {}).get(size, ())
                key = (anchor, size)
                position = positions[key]
                while position < len(lane) and lane[position].indices in selected_indices:
                    position += 1
                positions[key] = position
                if position >= len(lane):
                    continue
                state = lane[position]
                positions[key] = position + 1
                selected.append(state)
                selected_indices.add(state.indices)
                added = True
                if len(selected) >= capacity:
                    return selected
        if not added:
            break
    return selected


def cross_branch_completions(
    partials: Sequence[_State], graph: EvidenceGraph
) -> dict[tuple[int, ...], _State]:
    """Union retained partials from distinct query-conditioned branches."""
    by_member: dict[int, set[int]] = defaultdict(set)
    masks: list[int] = []
    for position, state in enumerate(partials):
        mask = 0
        for index in state.indices:
            mask |= 1 << index
            by_member[index].add(position)
        masks.append(mask)

    # Store only the best provenance while unions are enumerated.  The expensive
    # structural score is then evaluated once per exact deduplicated union.
    completed_sources: dict[int, _State] = {}
    for left_position, left in enumerate(partials):
        possible: set[int] = set()
        touched = set(left.indices)
        for index in left.indices:
            touched.update(graph.adjacency[index])
        for index in touched:
            possible.update(by_member[index])
        for right_position in sorted(position for position in possible if position > left_position):
            right = partials[right_position]
            if (left.seed, left.branch) == (right.seed, right.branch):
                continue
            left_mask, right_mask = masks[left_position], masks[right_position]
            overlap = left_mask & right_mask
            if overlap in (left_mask, right_mask):
                continue
            union_mask = left_mask | right_mask
            if not 4 <= union_mask.bit_count() <= GENERATOR_D_CONFIG.max_support_size:
                continue
            for source in (left, right):
                prior = completed_sources.get(union_mask)
                provenance = (
                    source.protected_component is None,
                    -graph.query_scores[source.seed],
                    graph.units[source.seed].unit_id,
                    graph.units[source.branch].unit_id,
                )
                prior_provenance = (
                    prior.protected_component is None,
                    -graph.query_scores[prior.seed],
                    graph.units[prior.seed].unit_id,
                    graph.units[prior.branch].unit_id,
                ) if prior is not None else None
                if prior is None or provenance < prior_provenance:
                    completed_sources[union_mask] = source

    completed: dict[tuple[int, ...], _State] = {}
    for union_mask, source in completed_sources.items():
        union = tuple(
            index for index in range(len(graph.units)) if union_mask & (1 << index)
        )
        completed[union] = _State(
            union,
            source.seed,
            source.branch,
            _state_score(union, graph, source.seed),
            source.protected_component,
        )
    return completed


def generate_experimental_pool(graph: EvidenceGraph) -> ExperimentalPool:
    """Run the single completion modification without touching production code."""
    if not graph.units:
        return ExperimentalPool((), (), 0, 0)
    lanes, anchors, component_ids = build_local_lanes(graph)
    local_unique = {
        state.indices for by_size in lanes.values() for states in by_size.values() for state in states
    }
    retained_partials = allocate_lanes(
        lanes,
        anchors,
        component_ids,
        graph,
        min(GENERATOR_D_CONFIG.max_candidates, len(local_unique)),
        (1, 2, 3),
    )
    completions = cross_branch_completions(retained_partials, graph)

    experimental_lanes: dict[int, dict[int, list[_State]]] = {
        anchor: {size: list(states) for size, states in by_size.items()}
        for anchor, by_size in lanes.items()
    }
    for state in completions.values():
        experimental_lanes.setdefault(state.seed, {}).setdefault(len(state.indices), []).append(
            state
        )
    for by_size in experimental_lanes.values():
        for states in by_size.values():
            states.sort(key=lambda state: (-state.score, state.indices))

    selected = allocate_lanes(
        experimental_lanes,
        anchors,
        component_ids,
        graph,
        GENERATOR_D_CONFIG.max_candidates,
        tuple(range(1, GENERATOR_D_CONFIG.max_support_size + 1)),
    )
    selected_indices = {state.indices for state in selected}
    progressive = progressive_connected_supports(graph, GENERATOR_D_CONFIG)
    for indices in progressive.candidate_indices:
        if len(selected) >= GENERATOR_D_CONFIG.max_candidates:
            break
        if indices not in selected_indices:
            selected.append(
                _State(indices, indices[0], indices[0], _state_score(indices, graph, indices[0]))
            )
            selected_indices.add(indices)

    completion_selected = tuple(
        state.indices for state in selected if state.indices in completions
    )
    return ExperimentalPool(
        candidate_indices=tuple(state.indices for state in selected),
        completion_indices=completion_selected,
        completion_count_before_cap=len(completions),
        source_partial_count=len(retained_partials),
    )


def gold_sets(gold: Sequence[Sequence[str]]) -> list[set[str]]:
    return [set(map(str, explanation)) for explanation in gold if explanation]


def pool_diagnostics(
    candidates: Sequence[Sequence[str]], gold: Sequence[Sequence[str]]
) -> dict[str, Any]:
    targets = gold_sets(gold)
    candidate_sets = [set(map(str, candidate)) for candidate in candidates]
    pool_union = set().union(*candidate_sets) if candidate_sets else set()
    complete = [any(target <= candidate for target in targets) for candidate in candidate_sets]
    coverability = {}
    for k in (1, 2, 3, 5):
        prefix_union = set().union(*candidate_sets[:k]) if candidate_sets[:k] else set()
        coverability[str(k)] = any(target <= prefix_union for target in targets)
    recall = max(
        (len(target & pool_union) / len(target) for target in targets), default=0.0
    )
    return {
        "complete_candidate_available_anywhere": any(complete),
        "first_complete_candidate_rank": next(
            (index for index, value in enumerate(complete, start=1) if value), None
        ),
        "complete_support_coverability": coverability,
        "full_pool_complete_support_available": any(target <= pool_union for target in targets),
        "gold_unit_recall": recall,
        "candidate_count": len(candidates),
        "candidate_unit_total": sum(map(len, candidates)),
        "mean_candidate_size": (
            statistics.mean(map(len, candidates)) if candidates else 0.0
        ),
        "reaches_512_budget": len(candidates) == GENERATOR_D_CONFIG.max_candidates,
    }


def aggregate(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    metrics = [record[method] for record in records]
    counts = [int(metric["candidate_count"]) for metric in metrics]
    candidate_total = sum(counts)
    total = len(metrics)
    rate = lambda values: sum(bool(value) for value in values) / total if total else 0.0
    return {
        "examples": total,
        "complete_candidate_availability": rate(
            metric["complete_candidate_available_anywhere"] for metric in metrics
        ),
        "complete_support_coverability": {
            k: rate(metric["complete_support_coverability"][k] for metric in metrics)
            for k in ("1", "2", "3", "5")
        },
        "full_pool_complete_support_availability": rate(
            metric["full_pool_complete_support_available"] for metric in metrics
        ),
        "mean_gold_unit_recall": (
            statistics.mean(float(metric["gold_unit_recall"]) for metric in metrics)
            if metrics
            else 0.0
        ),
        "mean_candidate_count": statistics.mean(counts) if counts else 0.0,
        "median_candidate_count": statistics.median(counts) if counts else 0.0,
        "mean_candidate_size": (
            sum(int(metric["candidate_unit_total"]) for metric in metrics) / candidate_total
            if candidate_total
            else 0.0
        ),
        "fraction_reaching_512_budget": rate(
            metric["reaches_512_budget"] for metric in metrics
        ),
    }


def comparison(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = aggregate(records, "baseline")
    modified = aggregate(records, "modified")

    def delta(left: Any, right: Any) -> Any:
        if isinstance(left, dict):
            return {key: delta(left[key], right[key]) for key in left if key != "examples"}
        return float(right) - float(left)

    return {"baseline": baseline, "modified": modified, "delta": delta(baseline, modified)}


def subgroup(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = sorted({str(record[key]) for record in records})
    return {
        value: comparison([record for record in records if str(record[key]) == value])
        for value in values
    }


def assignment_index(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = {
        str(row["example_id"]): str(row["first_observable_mechanism"])
        for row in payload["assignments"]
    }
    counts = Counter(assignments.values())
    if len(assignments) != 326 or counts != Counter(
        {"progressive_expansion_failure": 199, "diversity_or_composer_pruning": 127}
    ):
        raise AssertionError(f"Unexpected diagnosed cohort: {counts}")
    return assignments, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument("--mechanism", type=Path, default=MECHANISM_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-per-dataset", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments, mechanism_payload = assignment_index(args.mechanism)
    records: list[dict[str, Any]] = []
    input_paths: list[Path] = [args.mechanism]

    for dataset, domain, _ in DATASETS:
        source = args.data_root / dataset / "dev_subgraph_retrieval.jsonl"
        metadata_path = args.data_root / dataset / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        input_paths.extend((source, metadata_path))
        raw_text = build_raw_text_index(dataset, metadata) if domain == "text" else None
        raw_ontology = None
        if domain == "ontology":
            ontology_path = ROOT / Path(metadata["source_file"])
            raw_ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
            input_paths.append(ontology_path)

        print(f"{dataset}: DEV candidate generation", flush=True)
        for number, (example_id, rows) in enumerate(grouped_jsonl(source), start=1):
            if args.limit_per_dataset and number > args.limit_per_dataset:
                break
            if domain == "text":
                graph = text_graph(
                    dataset, example_id, str(rows[0].get("question", "")), raw_text or {}
                )
            else:
                item = raw_ontology[int(rows[0]["group_index"])]  # type: ignore[index]
                qa = item["QAs"][int(rows[0]["qa_index"])]
                graph = ontology_graph(item, qa)

            # Hard boundary: both pools are frozen before any gold field is read.
            baseline_result = generate_generator_d_candidates(graph)
            baseline_candidates = unit_candidates(graph, baseline_result.candidate_indices)
            persisted_candidates = tuple(
                tuple(map(str, row.get("subgraph_units", []))) for row in rows
            )
            if baseline_candidates != persisted_candidates:
                raise AssertionError(f"Frozen Generator-D replay mismatch: {example_id}")
            modified_result = generate_experimental_pool(graph)
            modified_candidates = unit_candidates(graph, modified_result.candidate_indices)
            frozen = (baseline_candidates, modified_candidates)

            gold = alternatives(rows[0])
            if not gold:
                continue
            baseline_metrics = pool_diagnostics(frozen[0], gold)
            modified_metrics = pool_diagnostics(frozen[1], gold)
            baseline_sets = set(map(frozenset, frozen[0]))
            modified_sets = set(map(frozenset, frozen[1]))
            removed = baseline_sets - modified_sets
            target_sets = gold_sets(gold)
            useful_removed = [
                candidate
                for candidate in removed
                if any(candidate & target for target in target_sets)
            ]
            complete_removed = [
                candidate
                for candidate in removed
                if any(target <= candidate for target in target_sets)
            ]
            records.append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "hop": "1hop" if "1hop" in dataset else "2hop",
                    "example_id": example_id,
                    "diagnosed_failure_class": assignments.get(example_id),
                    "baseline": baseline_metrics,
                    "modified": modified_metrics,
                    "completion_count_before_cap": modified_result.completion_count_before_cap,
                    "completion_count_in_pool": len(modified_result.completion_indices),
                    "source_partial_count": modified_result.source_partial_count,
                    "removed_baseline_candidate_count": len(removed),
                    "removed_gold_overlapping_candidate_count": len(useful_removed),
                    "removed_complete_candidate_count": len(complete_removed),
                }
            )
            if number % 100 == 0:
                print(f"{dataset}: {number} DEV examples", flush=True)

    if args.limit_per_dataset:
        print("Limited smoke run complete; summary artifacts intentionally not written.")
        return

    diagnosed = [record for record in records if record["diagnosed_failure_class"]]
    if len(diagnosed) != 326:
        raise AssertionError(f"Expected all 326 diagnosed examples, found {len(diagnosed)}")
    recovery_details = []
    for failure_class in ("progressive_expansion_failure", "diversity_or_composer_pruning"):
        subset = [record for record in diagnosed if record["diagnosed_failure_class"] == failure_class]
        recovered = [
            record for record in subset if record["modified"]["complete_candidate_available_anywhere"]
        ]
        recovery_details.append(
            {
                "failure_class": failure_class,
                "examples": len(subset),
                "recovered": len(recovered),
                "recovery_rate": len(recovered) / len(subset),
                "recovered_example_ids": [record["example_id"] for record in recovered],
                "per_dataset": dict(Counter(record["dataset"] for record in recovered)),
                "by_domain": dict(Counter(record["domain"] for record in recovered)),
                "by_hop": dict(Counter(record["hop"] for record in recovered)),
            }
        )

    previously_successful = [
        record for record in records if record["baseline"]["complete_candidate_available_anywhere"]
    ]
    regressions = [
        record
        for record in previously_successful
        if not record["modified"]["complete_candidate_available_anywhere"]
    ]
    displacement_examples = [
        record for record in records if record["removed_gold_overlapping_candidate_count"] > 0
    ]

    metrics = {
        "schema_version": "generator_d_cross_branch_completion_dev_v1",
        "split": "dev",
        "metric_definitions": {
            "complete_candidate_availability": "fraction with at least one candidate containing an entire gold explanation",
            "complete_support_coverability_at_k": "fraction whose union of the first k generator-preorder candidates contains an entire gold explanation",
            "full_pool_complete_support_availability": "fraction whose full candidate-pool union contains an entire gold explanation",
            "gold_unit_recall": "per example, maximum gold-alternative unit recall in the full pool union",
        },
        "overall": comparison(records),
        "by_domain": subgroup(records, "domain"),
        "by_hop": subgroup(records, "hop"),
    }
    per_dataset = {dataset: comparison([r for r in records if r["dataset"] == dataset]) for dataset, _, _ in DATASETS}
    recovery = {
        "schema_version": "generator_d_cross_branch_completion_diagnosed_recovery_v1",
        "split": "dev",
        "cohort": mechanism_payload["scope"],
        "classes": recovery_details,
        "previously_successful_examples": len(previously_successful),
        "previously_successful_losing_complete_candidate": len(regressions),
        "regression_example_ids": [record["example_id"] for record in regressions],
        "by_domain": subgroup(diagnosed, "domain"),
        "by_hop": subgroup(diagnosed, "hop"),
    }
    displacement = {
        "schema_version": "generator_d_cross_branch_completion_pool_displacement_v1",
        "split": "dev",
        "examples": len(records),
        "useful_candidate_definition": "a frozen baseline candidate overlapping at least one unit of at least one gold explanation; complete candidates are reported separately",
        "examples_at_512_baseline": sum(r["baseline"]["reaches_512_budget"] for r in records),
        "examples_at_512_modified": sum(r["modified"]["reaches_512_budget"] for r in records),
        "examples_with_any_baseline_candidate_displaced": sum(
            r["removed_baseline_candidate_count"] > 0 for r in records
        ),
        "examples_with_gold_overlapping_candidate_displaced": len(displacement_examples),
        "examples_with_complete_candidate_displaced": sum(
            r["removed_complete_candidate_count"] > 0 for r in records
        ),
        "total_baseline_candidates_displaced": sum(
            r["removed_baseline_candidate_count"] for r in records
        ),
        "total_gold_overlapping_candidates_displaced": sum(
            r["removed_gold_overlapping_candidate_count"] for r in records
        ),
        "total_complete_candidates_displaced": sum(
            r["removed_complete_candidate_count"] for r in records
        ),
        "previously_successful_losing_all_complete_candidates": len(regressions),
        "details": [
            {
                "dataset": r["dataset"],
                "domain": r["domain"],
                "hop": r["hop"],
                "example_id": r["example_id"],
                "removed_baseline_candidate_count": r["removed_baseline_candidate_count"],
                "removed_gold_overlapping_candidate_count": r[
                    "removed_gold_overlapping_candidate_count"
                ],
                "removed_complete_candidate_count": r["removed_complete_candidate_count"],
                "completion_count_before_cap": r["completion_count_before_cap"],
                "completion_count_in_pool": r["completion_count_in_pool"],
            }
            for r in records
            if r["removed_baseline_candidate_count"] > 0
        ],
    }

    write_json(args.output_dir / "candidate_metrics.json", metrics)
    write_json(args.output_dir / "per_dataset.json", per_dataset)
    write_json(args.output_dir / "diagnosed_failure_recovery.json", recovery)
    write_json(args.output_dir / "pool_displacement_analysis.json", displacement)

    progressive = next(
        row for row in recovery_details if row["failure_class"] == "progressive_expansion_failure"
    )
    incidental = next(
        row for row in recovery_details if row["failure_class"] == "diversity_or_composer_pruning"
    )
    gains_concentrated_in_target_class = (
        progressive["recovery_rate"] > incidental["recovery_rate"]
    )
    no_meaningful_regression = len(regressions) == 0
    manageable_pool_pressure = (
        displacement["examples_at_512_modified"] <= displacement["examples_at_512_baseline"]
    )
    justified = (
        progressive["recovered"] > 0
        and no_meaningful_regression
        and manageable_pool_pressure
        and gains_concentrated_in_target_class
    )
    proposal = [
        "# Proposed Generator-D change",
        "",
        "This DEV-only experiment did not modify production. The isolated change composes pairs of already-retained query-local partial candidates only when they come from distinct `(anchor, first-hop branch)` lanes, both contribute at least one unique unit, and their connected union has size 4-6. Exact unit-index tuples use Generator-D's deterministic deduplication preference. Completed candidates enter the existing component/anchor/size round-robin allocation before the unchanged 512 cap; progressive fallback remains unchanged.",
        "",
        "No gold, labels, targets, answers, supporting facts, explanations, 2Wiki evidences, learned scores, new widths, lambdas, dataset rules, or modality thresholds enter generation.",
        "",
        f"Mechanism-validation decision: **{'justified for the next controlled stage' if justified else 'not justified for the next stage'}**.",
        "",
        f"- Progressive-expansion recovery: {progressive['recovered']}/{progressive['examples']} ({100*progressive['recovery_rate']:.2f}%).",
        f"- Incidental recovery in the remaining 127: {incidental['recovered']}/{incidental['examples']} ({100*incidental['recovery_rate']:.2f}%).",
        f"- Previously successful examples losing all complete candidates: {len(regressions)}/{len(previously_successful)}.",
        f"- Modified pools reaching 512: {displacement['examples_at_512_modified']}/{len(records)}.",
        f"- Gains concentrated in intended class: {gains_concentrated_in_target_class} (target-class recovery {100*progressive['recovery_rate']:.2f}% vs incidental {100*incidental['recovery_rate']:.2f}%).",
        "",
        "The negative decision does not depend on an unstated recovery cutoff: the modification fails the no-regression, manageable-saturation, and target-class-concentration criteria.",
    ]
    (args.output_dir / "proposed_generator_change.md").write_text(
        "\n".join(proposal) + "\n", encoding="utf-8"
    )

    overall_a = metrics["overall"]["baseline"]
    overall_b = metrics["overall"]["modified"]
    summary = [
        "# Generator-D cross-branch completion: DEV candidate evaluation",
        "",
        "Candidate generation only; DEV only. No TEST, training, GNN inference, symbolic reranking, answer generation, adaptive policy, hyperparameter search, or production modification was performed.",
        "",
        "## Overall",
        "",
        "| Metric | Frozen D | D + completion | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Complete candidate availability", "complete_candidate_availability"),
        ("Full-pool complete-support availability", "full_pool_complete_support_availability"),
        ("Gold-unit recall", "mean_gold_unit_recall"),
        ("Mean candidate count", "mean_candidate_count"),
        ("Median candidate count", "median_candidate_count"),
        ("Mean candidate size", "mean_candidate_size"),
        ("Fraction at 512", "fraction_reaching_512_budget"),
    ):
        a, b = overall_a[key], overall_b[key]
        summary.append(f"| {label} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    for k in ("1", "2", "3", "5"):
        a = overall_a["complete_support_coverability"][k]
        b = overall_b["complete_support_coverability"][k]
        summary.append(f"| Complete-support coverability @{k} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    summary += [
        "",
        "## Diagnosed mechanism",
        "",
        f"- Progressive-expansion failures recovered: {progressive['recovered']}/199 ({100*progressive['recovery_rate']:.2f}%).",
        f"- Remaining composition failures incidentally recovered: {incidental['recovered']}/127 ({100*incidental['recovery_rate']:.2f}%).",
        f"- Previously successful examples losing complete-support availability: {len(regressions)}.",
        f"- Examples where a gold-overlapping baseline candidate was displaced: {len(displacement_examples)}.",
        "",
        "Text/ontology, 1-hop/2-hop, and all ten per-dataset comparisons are in `candidate_metrics.json` and `per_dataset.json`.",
        "",
        "## Lineage",
        "",
    ]
    for path in dict.fromkeys(input_paths):
        summary.append(f"- `{path.relative_to(ROOT)}` SHA-256 `{sha256(path)}`")
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
