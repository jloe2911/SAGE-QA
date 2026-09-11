"""One dev-only post-generation diagnostic of unified ontology reachability.

The current-clean and unified generators are run unchanged with the fixed
comparison configuration.  Gold explanations are read only after both outputs
and the instrumented unified search trace are frozen.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    extract_query_signature,
    generate_ontology_candidates,
    get_gold_explanations,
    parse_owl_context,
    unit_signature,
)
from data_processing.evidence_graph_candidates import (  # noqa: E402
    EvidenceGraph,
    _State,
    _rank_indices,
    _select_diverse_candidates,
    _state_score,
    progressive_connected_supports,
)
from data_processing.retrieval_contracts import (  # noqa: E402
    git_provenance,
    sha256_file,
    write_json,
)
from evaluation.compare_unified_evidence_graph_candidates import (  # noqa: E402
    DISPLAY_NAMES,
    UNIFIED_CONFIG,
    _canonical,
    build_ontology_evidence_graph,
)
from evaluation.validate_gold_free_candidate_pools import (  # noqa: E402
    ONTO,
    ONTO_BASE,
    ROOT,
    onto_rows,
)

COHORTS = ("Family_2hop", "Pizza_100_2hop", "Pizza_250_2hop", "OWL2Bench_2hop")
SEARCH_EDGE_DEPTH = UNIFIED_CONFIG.max_support_size - 1


def _distances(
    graph: EvidenceGraph, sources: Iterable[int]
) -> tuple[list[int | None], list[int | None]]:
    distance: list[int | None] = [None] * len(graph.units)
    parent: list[int | None] = [None] * len(graph.units)
    queue = collections.deque()
    for source in sorted(set(sources)):
        distance[source] = 0
        queue.append(source)
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph.adjacency[node]):
            if distance[neighbor] is None:
                distance[neighbor] = int(distance[node]) + 1
                parent[neighbor] = node
                queue.append(neighbor)
    return distance, parent


def _path_to(index: int, distance: Sequence[int | None], parent: Sequence[int | None]) -> list[int]:
    if distance[index] is None:
        return []
    path = [index]
    while parent[path[-1]] is not None:
        path.append(int(parent[path[-1]]))
    return list(reversed(path))


def _trace_search(graph: EvidenceGraph) -> dict[str, Any]:
    """Replay the fixed search and retain pre/post-pruning states."""
    config = UNIFIED_CONFIG
    max_size = min(config.max_support_size, len(graph.units))
    seeds = _rank_indices(range(len(graph.units)), graph)[: config.max_seeds]
    all_states: list[_State] = []
    explored = set(seeds)
    frontier: list[_State] = []
    retained_by_size: dict[int, list[_State]] = {}
    generated_by_size: dict[int, list[dict[str, Any]]] = {}

    for seed in seeds:
        state = _State((seed,), seed, seed, _state_score((seed,), graph, seed))
        all_states.append(state)
        frontier.append(state)
    retained_by_size[1] = list(frontier)

    for target_size in range(2, max_size + 1):
        by_lane: dict[tuple[int, int], dict[tuple[int, ...], _State]] = {}
        generated: list[dict[str, Any]] = []
        for state in frontier:
            selected = set(state.indices)
            neighbors = set().union(*(graph.adjacency[index] for index in state.indices))
            all_ranked = _rank_indices(neighbors - selected, graph)
            cap = config.max_neighbors_per_state
            if len(state.indices) == 1:
                cap = min(cap, config.max_first_hops_per_seed)
            ranked = all_ranked[:cap]
            for neighbor in ranked:
                indices = tuple(sorted((*state.indices, neighbor)))
                branch = neighbor if len(state.indices) == 1 else state.branch
                expanded = _State(
                    indices, state.seed, branch, _state_score(indices, graph, state.seed)
                )
                lane = (expanded.seed, expanded.branch)
                prior = by_lane.setdefault(lane, {}).get(indices)
                if prior is None or (-expanded.score, expanded.indices) < (
                    -prior.score,
                    prior.indices,
                ):
                    by_lane[lane][indices] = expanded
                generated.append(
                    {
                        "state": expanded,
                        "parent": state,
                        "added": neighbor,
                        "neighbor_rank": all_ranked.index(neighbor) + 1,
                        "neighbor_cap": cap,
                    }
                )
                explored.add(neighbor)
        generated_by_size[target_size] = generated
        frontier = []
        for lane in sorted(
            by_lane, key=lambda x: (graph.units[x[0]].unit_id, graph.units[x[1]].unit_id)
        ):
            retained = sorted(
                by_lane[lane].values(), key=lambda state: (-state.score, state.indices)
            )[: config.per_branch_width]
            frontier.extend(retained)
        retained_by_size[target_size] = list(frontier)
        all_states.extend(frontier)
        if not frontier:
            break

    selected = _select_diverse_candidates(all_states, graph, config)
    replay = tuple(tuple(graph.units[i].unit_id for i in state.indices) for state in selected)
    actual = progressive_connected_supports(graph, config)
    if replay != actual.candidates or explored != {
        i for i, u in enumerate(graph.units) if u.unit_id in actual.explored_unit_ids
    }:
        raise AssertionError("Instrumented unified search does not reproduce the generator")
    return {
        "seeds": seeds,
        "explored": explored,
        "retained_by_size": retained_by_size,
        "generated_by_size": generated_by_size,
        "selected": selected,
        "actual": actual,
    }


def _query_matches(graph: EvidenceGraph, question: str, sparql: str) -> list[int]:
    signature = extract_query_signature(question=question, sparql_query=sparql)
    entities = set(signature.get("query_entities", []))
    properties = set(signature.get("query_properties", []))
    return [
        index
        for index, unit in enumerate(graph.units)
        if (unit_signature(unit.unit_id)[0] & entities)
        or (unit_signature(unit.unit_id)[1] & properties)
    ]


def _structural_kinds(unit: str) -> set[str]:
    kinds: set[str] = set()
    if " SubClassOf " in unit:
        kinds.add("subclass/subproperty")
    if unit.startswith("SubObjectPropertyOf("):
        kinds.add("subclass/subproperty")
    if " domain " in unit or " range " in unit:
        kinds.add("domain/range")
    if unit.startswith("InverseObjectProperties("):
        kinds.add("inverse")
    if unit.startswith("ObjectPropertyChain("):
        kinds.add("property chain")
    if any(
        token in unit
        for token in ("Restriction", " rdf:first ", " rdf:rest ", "someValuesFrom", "allValuesFrom")
    ):
        kinds.add("RDF-list/restriction structure")
    return kinds


def _class_terms(unit: str) -> set[str]:
    parts = unit.split()
    if len(parts) != 3:
        return set()
    if parts[1] == "rdf:type":
        return {parts[2]}
    if parts[1] == "SubClassOf":
        return {parts[0], parts[2]}
    if parts[1] in {"domain", "range"}:
        return {parts[2]}
    return set()


def _edge_mechanisms(left: str, right: str) -> list[str]:
    left_entities, left_properties = unit_signature(left)
    right_entities, right_properties = unit_signature(right)
    mechanisms = _structural_kinds(left) | _structural_kinds(right)
    shared_entities = left_entities & right_entities
    if shared_entities:
        if shared_entities & (_class_terms(left) | _class_terms(right)):
            mechanisms.add("shared class")
        if shared_entities - (_class_terms(left) | _class_terms(right)):
            mechanisms.add("shared entity")
    if left_properties & right_properties:
        mechanisms.add("shared property")
    if (left_entities & right_properties) or (left_properties & right_entities):
        mechanisms.add("shared property")
    return sorted(mechanisms or {"other"})


def _current_selection_trace(
    context: Sequence[str], question: str, sparql: str
) -> dict[str, dict[str, Any]]:
    """Replay only the current clean gold-free atomic selection stages."""
    signature = extract_query_signature(question=question, sparql_query=sparql)
    query_entities = set(signature.get("query_entities", []))
    query_properties = set(signature.get("query_properties", []))
    signatures = [unit_signature(unit) for unit in context]
    tokens = [entities | properties for entities, properties in signatures]
    token_index: dict[str, set[int]] = collections.defaultdict(set)
    direct = []
    for index, (entities, properties) in enumerate(signatures):
        for token in tokens[index]:
            token_index[token].add(index)
        score = 2 * len(entities & query_entities) + len(properties & query_properties)
        if score:
            direct.append((score, index))
    direct.sort(key=lambda x: (-x[0], context[x[1]]))
    selected: list[int] = []
    selected_set: set[int] = set()
    frontier: list[int] = []
    trace: dict[str, dict[str, Any]] = {}
    for score, index in direct:
        if index not in selected_set and len(selected) < ONTO_BASE["atomic_budget"]:
            selected.append(index)
            selected_set.add(index)
            frontier.append(index)
            trace[context[index]] = {
                "stage": "formal-query seed",
                "round": 0,
                "direct_score": score,
            }
    round_number = 0
    while frontier and len(selected) < ONTO_BASE["atomic_budget"]:
        round_number += 1
        previous = list(frontier)
        overlap: dict[int, int] = collections.defaultdict(int)
        for index in previous:
            for token in tokens[index]:
                for neighbor in token_index[token]:
                    if neighbor not in selected_set:
                        overlap[neighbor] += 1
        if not overlap:
            break
        ranked = sorted(
            overlap,
            key=lambda index: (
                -overlap[index],
                -len(tokens[index] & (query_entities | query_properties)),
                context[index],
            ),
        )
        frontier = []
        for index in ranked:
            if len(selected) >= ONTO_BASE["atomic_budget"]:
                break
            selected.append(index)
            selected_set.add(index)
            frontier.append(index)
            parents = [p for p in previous if tokens[p] & tokens[index]]
            trace[context[index]] = {
                "stage": "formal-query seed expansion",
                "round": round_number,
                "shared_terms": sorted(set().union(*(tokens[p] & tokens[index] for p in parents))),
                "parent_units": [context[p] for p in parents[:8]],
                "edge_mechanisms": sorted(
                    set().union(*(_edge_mechanisms(context[p], context[index]) for p in parents))
                ),
            }
    return trace


def _diagnose_path_loss(
    graph: EvidenceGraph, trace: Mapping[str, Any], path: Sequence[int]
) -> dict[str, Any]:
    if len(path) < 2:
        return {"stage": "none", "detail": "target is itself a retained seed"}
    branch = path[1]
    for step in range(1, len(path)):
        parent_indices = tuple(sorted(path[:step]))
        next_index = path[step]
        parent_states = [
            state
            for state in trace["retained_by_size"].get(step, [])
            if state.indices == parent_indices
            and state.seed == path[0]
            and (step == 1 or state.branch == branch)
        ]
        if not parent_states:
            generated = [
                record
                for record in trace["generated_by_size"].get(step, [])
                if record["state"].indices == parent_indices and record["state"].seed == path[0]
            ]
            return {
                "stage": "per_branch_width",
                "path_step": step,
                "detail": "required prefix was generated but not retained"
                if generated
                else "required prefix was absent from the retained frontier",
            }
        state = parent_states[0]
        selected = set(state.indices)
        ranked = _rank_indices(
            set().union(*(graph.adjacency[i] for i in state.indices)) - selected, graph
        )
        cap = (
            UNIFIED_CONFIG.max_first_hops_per_seed
            if step == 1
            else UNIFIED_CONFIG.max_neighbors_per_state
        )
        if next_index not in ranked[:cap]:
            return {
                "stage": "max_first_hops_per_seed" if step == 1 else "max_neighbors_per_state",
                "path_step": step,
                "required_neighbor_rank": ranked.index(next_index) + 1
                if next_index in ranked
                else None,
                "limit": cap,
            }
    return {
        "stage": "other",
        "detail": "shortest prefix survived; another connected-state ordering caused the loss",
    }


def _unit_diagnosis(
    unit: str,
    graph: EvidenceGraph,
    trace: Mapping[str, Any],
    query_indices: Sequence[int],
    current_trace: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = [item.unit_id for item in graph.units]
    if unit not in ids:
        return {
            "gold_unit": unit,
            "classification": "A",
            "reason": "not represented as an EvidenceUnit",
            "distance_from_query_matching_unit": None,
            "distance_from_retained_seed": None,
            "current_clean_signal": current_trace.get(unit),
            "structural_kinds": sorted(_structural_kinds(unit)),
        }
    index = ids.index(unit)
    query_distance, query_parent = _distances(graph, query_indices)
    seed_distance, seed_parent = _distances(graph, trace["seeds"])
    output_union = {i for state in trace["selected"] for i in state.indices}
    seed_rank = _rank_indices(range(len(graph.units)), graph).index(index) + 1
    base = {
        "gold_unit": unit,
        "unit_index": index,
        "unified_seed_rank": seed_rank,
        "retained_as_seed": index in trace["seeds"],
        "distance_from_query_matching_unit": query_distance[index],
        "distance_from_retained_seed": seed_distance[index],
        "shortest_path_from_query_matching_unit": [
            graph.units[i].unit_id for i in _path_to(index, query_distance, query_parent)
        ],
        "shortest_path_from_retained_seed": [
            graph.units[i].unit_id for i in _path_to(index, seed_distance, seed_parent)
        ],
        "current_clean_signal": current_trace.get(unit),
        "structural_kinds": sorted(_structural_kinds(unit)),
    }
    if query_distance[index] is None:
        return {
            **base,
            "classification": "B",
            "reason": "disconnected from every query-matching EvidenceUnit",
        }
    if index in trace["explored"] and index not in output_union:
        retained_with_target = [
            state
            for states in trace["retained_by_size"].values()
            for state in states
            if index in state.indices
        ]
        stage = "final_candidate_selection" if retained_with_target else "per_branch_width"
        return {
            **base,
            "classification": "F",
            "reason": "reached during search but absent from every output candidate",
            "loss": {"stage": stage},
        }
    if seed_distance[index] is not None and seed_distance[index] > SEARCH_EDGE_DEPTH:
        return {
            **base,
            "classification": "C",
            "reason": "nearest retained seed is beyond the configured expansion depth",
        }
    if seed_distance[index] is None:
        return {
            **base,
            "classification": "D",
            "reason": "query-connected component has no retained seed",
            "loss": {"stage": "max_seeds", "limit": UNIFIED_CONFIG.max_seeds},
        }
    if index not in trace["explored"] and seed_distance[index] <= SEARCH_EDGE_DEPTH:
        path = _path_to(index, seed_distance, seed_parent)
        return {
            **base,
            "classification": "E",
            "reason": "reachable from a retained seed within depth but pruned",
            "loss": _diagnose_path_loss(graph, trace, path),
        }
    return {**base, "classification": "G", "reason": "other unit-level loss"}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {output_dir}")
    output_dir.mkdir(parents=True)
    all_details: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": "unified_ontology_reachability_diagnostic_v1",
        "development_only": True,
        "cohorts": list(COHORTS),
        "parameters_changed": False,
        "generator_modified": False,
        "training_run": False,
        "test_data_used": False,
        "parameter_grid_run": False,
        "gold_available_during_generation": False,
        "gold_join_boundary": "after current, unified, and instrumented trace outputs are frozen",
        "current_config": dict(ONTO_BASE),
        "unified_config": asdict(UNIFIED_CONFIG),
        "classification_priority": ["A", "B", "F", "C", "D", "E", "G"],
        "git": git_provenance(ROOT),
        "datasets": {},
    }
    for name in COHORTS:
        dataset_details: list[dict[str, Any]] = []
        dev_count = 0
        for group_index, qa_index, item, qa in onto_rows(name):
            dev_count += 1
            question = str(
                qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or ""
            )
            sparql = str(qa.get("SPARQL Query") or "")
            owl_context = str(item.get("OWL Context") or "")
            current_generated = generate_ontology_candidates(
                question=question,
                sparql_query=sparql,
                owl_context=owl_context,
                max_subgraph_size=ONTO_BASE["max_size"],
                min_subgraph_size=1,
                max_context_units=ONTO_BASE["atomic_budget"],
                candidate_beam_width=ONTO_BASE["beam_width"],
                max_candidate_subgraphs=ONTO_BASE["candidate_cap"],
            )
            current = _canonical(current_generated["candidate_subgraphs"])
            context = parse_owl_context(owl_context)
            graph = build_ontology_evidence_graph(context, question, sparql)
            search_trace = _trace_search(graph)
            unified = _canonical(search_trace["actual"].candidates)
            # Gold boundary begins here.
            golds = [tuple(dict.fromkeys(gold)) for gold in get_gold_explanations(qa) if gold]
            current_covering = [
                (gold_index, gold, candidate)
                for gold_index, gold in enumerate(golds)
                for candidate in current
                if set(gold) <= set(candidate)
            ]
            unified_complete = any(
                set(gold) <= set(candidate) for gold in golds for candidate in unified
            )
            if not current_covering or unified_complete:
                continue
            example_id = f"{name}__g{group_index}__q{qa_index}"
            unified_union = (
                set().union(*(set(candidate) for candidate in unified)) if unified else set()
            )
            query_indices = _query_matches(graph, question, sparql)
            current_trace = _current_selection_trace(context, question, sparql)
            explanation_records = []
            unique_gold: dict[int, tuple[str, ...]] = {}
            for gold_index, gold, _ in current_covering:
                unique_gold[gold_index] = gold
            for gold_index, gold in sorted(unique_gold.items()):
                covering_candidates = [
                    candidate for candidate in current if set(gold) <= set(candidate)
                ]
                chosen_candidate = min(
                    covering_candidates, key=lambda candidate: (len(candidate), candidate)
                )
                missing = [unit for unit in gold if unit not in unified_union]
                diagnoses = [
                    _unit_diagnosis(unit, graph, search_trace, query_indices, current_trace)
                    for unit in missing
                ]
                explanation_records.append(
                    {
                        "gold_explanation_index": gold_index,
                        "gold_explanation": list(gold),
                        "current_covering_candidate": list(chosen_candidate),
                        "missing_from_unified_candidate_union": missing,
                        "unit_diagnoses": diagnoses,
                        "support_level_classification": None if missing else "G",
                        "support_level_reason": None
                        if missing
                        else "all required units were preserved individually but never co-composed in one output candidate",
                        "current_clean_mechanisms": sorted(
                            set().union(
                                *(
                                    set((current_trace.get(unit) or {}).get("edge_mechanisms", []))
                                    | (
                                        {"formal-query seed expansion"}
                                        if (current_trace.get(unit) or {}).get("stage")
                                        == "formal-query seed expansion"
                                        else set()
                                    )
                                    | _structural_kinds(unit)
                                    for unit in gold
                                )
                            )
                        ),
                    }
                )
            detail = {
                "dataset": DISPLAY_NAMES[name],
                "example_id": example_id,
                "question": question,
                "sparql_query": sparql,
                "context_evidence_unit_count": len(graph.units),
                "query_matching_evidence_unit_count": len(query_indices),
                "retained_seed_count": len(search_trace["seeds"]),
                "current_candidate_count": len(current),
                "unified_candidate_count": len(unified),
                "unified_explored_unit_count": len(search_trace["explored"]),
                "unified_output_union_unit_count": len(
                    {i for state in search_trace["selected"] for i in state.indices}
                ),
                "current_covered_gold_explanations": explanation_records,
            }
            dataset_details.append(detail)
            all_details.append(detail)

        classification_events: dict[tuple[str, str], str] = {}
        mechanism_cases: collections.Counter[str] = collections.Counter()
        support_g_cases: set[str] = set()
        loss_stages: collections.Counter[str] = collections.Counter()
        for detail in dataset_details:
            case_mechanisms: set[str] = set()
            for explanation in detail["current_covered_gold_explanations"]:
                case_mechanisms.update(explanation["current_clean_mechanisms"])
                if explanation["support_level_classification"] == "G":
                    support_g_cases.add(detail["example_id"])
                for diagnosis in explanation["unit_diagnoses"]:
                    classification_events[(detail["example_id"], diagnosis["gold_unit"])] = (
                        diagnosis["classification"]
                    )
                    if diagnosis.get("loss"):
                        loss_stages[str(diagnosis["loss"].get("stage"))] += 1
                    signal = diagnosis.get("current_clean_signal") or {}
                    case_mechanisms.update(signal.get("edge_mechanisms", []))
                    if signal.get("stage") == "formal-query seed expansion":
                        case_mechanisms.add("formal-query seed expansion")
            mechanism_cases.update(case_mechanisms)
        classifications = collections.Counter(classification_events.values())
        if support_g_cases:
            classifications["G"] += len(support_g_cases)
        display = DISPLAY_NAMES[name]
        summary["datasets"][display] = {
            "development_examples": dev_count,
            "current_clean_solved_unified_missed_examples": len(dataset_details),
            "unique_missed_gold_units": len(classification_events),
            "support_composition_G_examples": len(support_g_cases),
            "classification_counts": dict(sorted(classifications.items())),
            "classification_case_counts": {
                label: len(
                    {
                        example_id
                        for (example_id, _), value in classification_events.items()
                        if value == label
                    }
                )
                + (len(support_g_cases) if label == "G" else 0)
                for label in "ABCDEFG"
            },
            "loss_stage_counts": dict(sorted(loss_stages.items())),
            "current_clean_structural_mechanism_case_counts": dict(sorted(mechanism_cases.items())),
            "source_file": str(ONTO[name].relative_to(ROOT)),
            "source_sha256": sha256_file(ONTO[name]),
        }
        _write_jsonl(output_dir / name / "details.jsonl", dataset_details)
    _write_jsonl(output_dir / "all_details.jsonl", all_details)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/development_runs/unified_ontology_reachability_diagnostic_v1",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
