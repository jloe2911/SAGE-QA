"""Causal DEV-only diagnosis of the frozen production Generator-D pipeline.

This evaluator is intentionally post hoc.  It rebuilds evidence graphs from
the raw DEV inputs to perform oracle structural checks, but it never writes to
the production corpus, changes Generator D, trains a model, reads TEST, or
generates answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (
    parse_owl_context,
    score_unit_for_query,
    unit_signature,
)
from data_processing.build_2wiki_subgraph_dataset import (
    flatten_context as flatten_2wiki_context,
    load_2wiki_file,
)
from data_processing.build_hotpot_subgraph_dataset import (
    flatten_context as flatten_hotpot_context,
    load_hotpot_file,
)
from data_processing.evidence_graph_candidates import (
    EvidenceGraph,
    build_ontology_evidence_graph,
    build_text_evidence_graph,
    generate_generator_d_candidates,
)
from data_processing.retrieval_contracts import cap_inference_candidate_rows
from models.symbolic_composer import extract_query_signature
from training.train_gnn_subgraph_retriever import (
    fact_rule_mix_symbolic,
    query_entity_coverage,
    query_property_match,
    sageqa_compact_adjustment,
    sageqa_proof_adjustment,
    sageqa_text_chain_adjustment,
    schema_count,
)


ROOT = Path(__file__).resolve().parents[1]
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
MAX_SUPPORT_SIZE = 6
ADMISSION_CAP = 320
INF = 10**9


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_set(units: Iterable[Any]) -> frozenset[str]:
    return frozenset(str(unit) for unit in units)


def candidate_id(row: Mapping[str, Any]) -> str:
    explicit = row.get("row_id")
    if explicit is not None:
        return str(explicit)
    return json.dumps(
        [int(row.get("generation_rank", -1)), sorted(unit_set(row.get("subgraph_units", [])))],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def alternatives(row: Mapping[str, Any]) -> list[list[str]]:
    values = row.get("gold_explanations") or []
    if values:
        return [[str(unit) for unit in alt] for alt in values if alt]
    support = row.get("gold_support_units") or row.get("gold_units") or []
    return [[str(unit) for unit in support]] if support else []


def contains(candidate: Iterable[Any], gold: Sequence[Sequence[Any]]) -> bool:
    candidate_keys = unit_set(candidate)
    return any(unit_set(alt) <= candidate_keys for alt in gold)


def grouped_jsonl(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_id = str(row["example_id"])
            if current_id is None:
                current_id = example_id
            if example_id != current_id:
                yield current_id, rows
                current_id, rows = example_id, []
            rows.append(row)
    if current_id is not None:
        yield current_id, rows


def load_rankings(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(f"Non-DEV ranking row rejected: {row.get('split')!r}")
            result[str(row["example_id"])][str(row["method"])] = row
    return dict(result)


def load_adaptive(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") not in (None, "dev"):
                raise ValueError("Non-DEV adaptive row rejected")
            result[str(row["example_id"])] = row
    return result


def build_raw_text_index(dataset: str, metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    path = metadata["dev_file"]
    loader = load_hotpot_file if dataset == "HotpotQA" else load_2wiki_file
    records = loader(str(ROOT / Path(path)))
    result = {}
    for row in records:
        raw_id = str(row.get("id") or row.get("_id") or "")
        if raw_id:
            result[raw_id] = row
    return result


def raw_text_id(example_id: str) -> str:
    marker = "__dev__"
    if marker not in example_id:
        raise ValueError(f"Unexpected text DEV id: {example_id}")
    return example_id.split(marker, 1)[1]


def text_graph(
    dataset: str, example_id: str, question: str, raw_index: Mapping[str, Mapping[str, Any]]
) -> EvidenceGraph:
    raw = raw_index[raw_text_id(example_id)]
    records = flatten_hotpot_context(raw) if dataset == "HotpotQA" else flatten_2wiki_context(raw)
    return build_text_evidence_graph(records, question)


def ontology_graph(item: Mapping[str, Any], qa: Mapping[str, Any]) -> EvidenceGraph:
    question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
    sparql = str(qa.get("SPARQL Query") or "")
    signature = extract_query_signature(question=question, sparql_query=sparql)
    entities = set(signature.get("query_entities", []))
    properties = set(signature.get("query_properties", []))
    axioms = parse_owl_context(str(item["OWL Context"]))
    return build_ontology_evidence_graph(
        axioms,
        unit_signature=unit_signature,
        query_entities=entities,
        query_properties=properties,
        score_unit=lambda unit, _entities, _properties, degree: score_unit_for_query(
            unit, query_entities=entities, query_properties=properties, degree=degree
        ),
    )


def connected_components(graph: EvidenceGraph) -> list[int]:
    component = [-1] * len(graph.units)
    current = 0
    for start in range(len(graph.units)):
        if component[start] >= 0:
            continue
        component[start] = current
        pending = [start]
        while pending:
            node = pending.pop()
            for neighbor in graph.adjacency[node]:
                if component[neighbor] < 0:
                    component[neighbor] = current
                    pending.append(neighbor)
        current += 1
    return component


def exact_steiner_size(graph: EvidenceGraph, terminals: Sequence[int]) -> int | None:
    """Exact minimum connected vertex count containing all terminals.

    Dreyfus-Wagner dynamic programming on the unweighted evidence graph.  The
    returned number counts evidence units (vertices), matching max_support_size.
    """
    terminals = tuple(dict.fromkeys(terminals))
    if not terminals:
        return 0
    if len(terminals) == 1:
        return 1
    n = len(graph.units)
    masks = 1 << len(terminals)
    dp = [[INF] * n for _ in range(masks)]
    for bit, terminal in enumerate(terminals):
        dp[1 << bit][terminal] = 0
    for mask in range(1, masks):
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub < other:
                for node in range(n):
                    value = dp[sub][node] + dp[other][node]
                    if value < dp[mask][node]:
                        dp[mask][node] = value
            sub = (sub - 1) & mask
        queue = deque(node for node, value in enumerate(dp[mask]) if value < INF)
        in_queue = [value < INF for value in dp[mask]]
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            next_value = dp[mask][node] + 1
            for neighbor in graph.adjacency[node]:
                if next_value < dp[mask][neighbor]:
                    dp[mask][neighbor] = next_value
                    if not in_queue[neighbor]:
                        queue.append(neighbor)
                        in_queue[neighbor] = True
    edges = min(dp[masks - 1])
    return None if edges >= INF else edges + 1


def structural_alternatives(
    graph: EvidenceGraph, gold: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    index = {unit.unit_id: position for position, unit in enumerate(graph.units)}
    components = connected_components(graph)
    output = []
    for alt_index, alt in enumerate(gold):
        missing = [unit for unit in alt if unit not in index]
        terminals = [index[unit] for unit in alt if unit in index]
        same_component = bool(terminals) and len({components[node] for node in terminals}) == 1
        minimum = exact_steiner_size(graph, terminals) if not missing and same_component else None
        output.append(
            {
                "gold_explanation_index": alt_index,
                "gold_support_size": len(unit_set(alt)),
                "missing_units": missing,
                "all_units_present": not missing,
                "same_connected_component": same_component if not missing else False,
                "minimum_connected_superset_size": minimum,
                "bridge_units_needed": None
                if minimum is None
                else max(0, minimum - len(unit_set(alt))),
                "representable_with_max_support_6": minimum is not None
                and minimum <= MAX_SUPPORT_SIZE,
            }
        )
    return output


def candidate_category(row: Mapping[str, Any], gold: Sequence[Sequence[str]]) -> str:
    units = unit_set(row.get("subgraph_units", []))
    gold_sets = [unit_set(alt) for alt in gold]
    if any(units == alt for alt in gold_sets):
        return "exact"
    if any(alt <= units for alt in gold_sets):
        return "complete-superset"
    if any(units & alt for alt in gold_sets):
        return "partial"
    return "irrelevant"


def symbolic_terms(row: Mapping[str, Any], mode: str) -> dict[str, float]:
    shaped = dict(row)
    if mode == "sageqa_text_chain":
        return {"text_chain_adjustment": float(sageqa_text_chain_adjustment(shaped))}
    compact = float(sageqa_compact_adjustment(shaped))
    proof = float(sageqa_proof_adjustment(shaped))
    return {
        "proof_adjustment_total": proof,
        "compact_adjustment": compact,
        "query_property_match": float(query_property_match(shaped)),
        "query_entity_coverage": float(query_entity_coverage(shaped)),
        "fact_rule_mix": float(fact_rule_mix_symbolic(shaped)),
        "schema_count": float(schema_count(shaped)),
    }


def pair_feature_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    a = [float(value) for value in left.get("symbolic_features", [])]
    b = [float(value) for value in right.get("symbolic_features", [])]
    length = max(len(a), len(b))
    a += [0.0] * (length - len(a))
    b += [0.0] * (length - len(b))
    l1 = sum(abs(x - y) for x, y in zip(a, b))
    return {
        "same_subgraph_size": len(left.get("subgraph_units", []))
        == len(right.get("subgraph_units", [])),
        "symbolic_feature_l1": l1,
        "same_symbolic_features": l1 <= 1e-12,
        "same_pre_rank_score": math.isclose(
            float(left.get("candidate_pre_rank_score", 0.0)),
            float(right.get("candidate_pre_rank_score", 0.0)),
            abs_tol=1e-12,
        ),
        "nearly_indistinguishable_hand_features": l1 <= 0.05,
    }


def minimum_candidate_cover(
    candidates: Sequence[Mapping[str, Any]], gold: Sequence[Sequence[str]], max_k: int = 5
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for alt_index, alt in enumerate(gold):
        target = unit_set(alt)
        masks = []
        for row in candidates:
            masks.append(unit_set(row.get("subgraph_units", [])) & target)
        dp: dict[frozenset[str], tuple[int, tuple[int, ...]]] = {frozenset(): (0, ())}
        for rank, contribution in enumerate(masks, start=1):
            for covered, (count, ranks) in list(dp.items()):
                merged = covered | contribution
                proposal = (count + 1, ranks + (rank,))
                if merged not in dp or proposal < dp[merged]:
                    dp[merged] = proposal
        if target in dp:
            count, ranks = dp[target]
            record = {
                "gold_explanation_index": alt_index,
                "minimum_candidates_needed": count,
                "candidate_ranks": list(ranks),
                "minimum_achievable_k": max(ranks),
                "support_in_top2": max(ranks) <= 2,
                "support_in_top3": max(ranks) <= 3,
                "support_in_top5": max(ranks) <= 5,
            }
            if best is None or (count, max(ranks), ranks) < (
                best["minimum_candidates_needed"],
                best["minimum_achievable_k"],
                tuple(best["candidate_ranks"]),
            ):
                best = record
    return best or {
        "gold_explanation_index": None,
        "minimum_candidates_needed": None,
        "candidate_ranks": [],
        "minimum_achievable_k": None,
        "support_in_top2": False,
        "support_in_top3": False,
        "support_in_top5": False,
    }


def search_failure_reason(
    graph: EvidenceGraph,
    generated: Any,
    structural: Sequence[Mapping[str, Any]],
    gold: Sequence[Sequence[str]],
) -> dict[str, Any]:
    unit_ids = [unit.unit_id for unit in graph.units]
    index = {unit: i for i, unit in enumerate(unit_ids)}
    anchors = set(generated.query_anchor_unit_ids)
    explored = set(generated.explored_unit_ids)
    viable = [row for row in structural if row["representable_with_max_support_6"]]
    best = min(
        viable,
        key=lambda row: (row["minimum_connected_superset_size"], row["gold_explanation_index"]),
    )
    alt = gold[int(best["gold_explanation_index"])]
    missing_explored = [unit for unit in alt if unit not in explored]
    gold_anchors = sorted(set(alt) & anchors)
    min_size = int(best["minimum_connected_superset_size"])
    pre_cap_complete = any(
        contains((unit_ids[node] for node in indices), [alt])
        for indices in generated.pre_cap_candidate_indices
    )
    if pre_cap_complete:
        cause = "diversity/composer stage"
    elif missing_explored and not gold_anchors:
        cause = "seed/anchor never selected"
    elif min_size > 3:
        cause = "progressive expansion"
    elif gold_anchors:
        cause = "diversity/composer stage"
    else:
        cause = "generation ordering"
    return {
        "primary_identifiable_cause": cause,
        "minimum_connected_superset_size": min_size,
        "gold_anchor_units": gold_anchors,
        "gold_units_not_explored": missing_explored,
        "complete_candidate_in_pre_cap_union": pre_cap_complete,
        "local_candidate_count_before_cap": int(generated.local_candidate_count_before_cap),
        "retained_local_candidate_count": int(generated.retained_local_candidate_count),
        "progressive_candidate_count": int(generated.progressive_candidate_count),
        "generated_state_count": int(generated.generated_state_count),
        "pool_budget_reached": len(generated.candidates) >= 512,
        "candidate_size_bias_risk": min_size >= 4,
        "note": "Cause is assigned by deterministic replay observability; internal discarded progressive states are not persisted, so first-hop, neighbor, and per-branch truncations cannot always be uniquely separated.",
    }


def overlap_redundancy(rows: Sequence[Mapping[str, Any]], ranks: Sequence[int]) -> dict[str, Any]:
    selected = [unit_set(rows[rank - 1].get("subgraph_units", [])) for rank in ranks]
    if len(selected) < 2:
        return {"pairwise_jaccard_mean": 0.0, "redundant_unit_count": 0}
    similarities = []
    counts = Counter(unit for values in selected for unit in values)
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            union = selected[left] | selected[right]
            similarities.append(
                len(selected[left] & selected[right]) / len(union) if union else 0.0
            )
    return {
        "pairwise_jaccard_mean": statistics.mean(similarities),
        "redundant_unit_count": sum(count - 1 for count in counts.values() if count > 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument(
        "--rankings",
        type=Path,
        default=ROOT
        / "outputs/development_runs/production_generator_d_v1_k_sensitivity/per_example_rankings.jsonl",
    )
    parser.add_argument(
        "--adaptive",
        type=Path,
        default=ROOT
        / "outputs/development_runs/production_generator_d_v1_adaptive_k/per_example_adaptive_dev.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/diagnostics/production_generator_d_v1_causal_retrieval_diagnosis",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rankings = load_rankings(args.rankings)
    adaptive = load_adaptive(args.adaptive)
    all_details: list[dict[str, Any]] = []
    constructability: list[dict[str, Any]] = []
    search_failures: list[dict[str, Any]] = []
    cap_analysis: list[dict[str, Any]] = []
    gnn_analysis: list[dict[str, Any]] = []
    symbolic_analysis: list[dict[str, Any]] = []
    aggregation_analysis: list[dict[str, Any]] = []

    for dataset, domain, final_mode in DATASETS:
        print(f"{dataset}: loading frozen DEV pool", flush=True)
        metadata_path = args.data_root / dataset / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_text = build_raw_text_index(dataset, metadata) if domain == "text" else None
        raw_ontology = None
        if domain == "ontology":
            source_path = ROOT / Path(metadata["source_file"])
            raw_ontology = json.loads(source_path.read_text(encoding="utf-8"))

        for example_number, (example_id, rows) in enumerate(
            grouped_jsonl(args.data_root / dataset / "dev_subgraph_retrieval.jsonl"), start=1
        ):
            gold = alternatives(rows[0])
            if not gold:
                continue
            ranking_pair = rankings.get(example_id, {})
            gnn_record = ranking_pair.get("gnn_only")
            final_record = ranking_pair.get("sageqa_final")
            if not gnn_record or not final_record:
                raise ValueError(f"Missing frozen ranking for {example_id}")
            gnn_top = gnn_record["ranked_candidates"][0]
            final_top = final_record["ranked_candidates"][0]
            if contains(final_top["subgraph_units"], gold):
                continue

            if domain == "text":
                graph = text_graph(
                    dataset, example_id, str(rows[0].get("question", "")), raw_text or {}
                )
            else:
                group_index = int(rows[0]["group_index"])
                qa_index = int(rows[0]["qa_index"])
                item = raw_ontology[group_index]  # type: ignore[index]
                qa = item["QAs"][qa_index]
                graph = ontology_graph(item, qa)

            graph_ids = [unit.unit_id for unit in graph.units]
            graph_id_set = set(graph_ids)
            structural = structural_alternatives(graph, gold)
            stage1_present = any(row["all_units_present"] for row in structural)
            representable = any(row["representable_with_max_support_6"] for row in structural)
            generated = generate_generator_d_candidates(graph)
            persisted_candidates = [row["subgraph_units"] for row in rows]
            replay_digest = hashlib.sha256(
                json.dumps(
                    [list(candidate) for candidate in generated.candidates],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            persisted_digest = hashlib.sha256(
                json.dumps(persisted_candidates, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            replay_matches = [
                list(candidate) for candidate in generated.candidates
            ] == persisted_candidates
            if not replay_matches:
                raise AssertionError(f"Generator-D replay mismatch: {example_id}")

            complete_indices = [
                i for i, row in enumerate(rows) if contains(row["subgraph_units"], gold)
            ]
            admitted = list(cap_inference_candidate_rows(rows, max_candidates=ADMISSION_CAP))
            admitted_complete = [row for row in admitted if contains(row["subgraph_units"], gold)]
            full_union_complete = contains(
                (unit for row in rows for unit in row["subgraph_units"]), gold
            )
            admitted_union_complete = contains(
                (unit for row in admitted for unit in row["subgraph_units"]), gold
            )

            if not stage1_present:
                taxonomy = "1_atomic_evidence_unavailable"
            elif not representable:
                taxonomy = (
                    "7_true_multi_candidate_aggregation_stopping_failure"
                    if admitted_union_complete
                    else "2_structurally_not_representable_as_one_candidate"
                )
            elif not complete_indices:
                taxonomy = "3_generator_d_search_composition_failure"
            elif not admitted_complete:
                taxonomy = "4_generated_but_excluded_by_320_admission_order"
            elif not contains(gnn_top["subgraph_units"], gold):
                taxonomy = "5_gnn_misranking"
            elif not contains(final_top["subgraph_units"], gold):
                taxonomy = "6_symbolic_reranking_degradation"
            else:
                raise AssertionError(f"Unclassified failure: {example_id}")

            base = {
                "dataset": dataset,
                "domain": domain,
                "example_id": example_id,
                "taxonomy": taxonomy,
                "gold_explanations": gold,
                "gold_support_sizes": [len(unit_set(alt)) for alt in gold],
                "evidence_graph_units": len(graph.units),
                "all_gold_units_in_graph_for_any_explanation": stage1_present,
                "structurally_representable_for_any_explanation": representable,
                "complete_candidate_in_512_pool": bool(complete_indices),
                "complete_candidate_in_admitted_320": bool(admitted_complete),
                "full_pool_union_complete": full_union_complete,
                "admitted_union_complete": admitted_union_complete,
                "generator_replay_matches_persisted_pool": replay_matches,
                "generator_replay_sha256": replay_digest,
                "persisted_pool_sha256": persisted_digest,
            }
            all_details.append(base)
            constructability.append({**base, "alternatives": structural})

            if taxonomy == "3_generator_d_search_composition_failure":
                search_failures.append(
                    {**base, **search_failure_reason(graph, generated, structural, gold)}
                )

            if complete_indices:
                complete_rows = [rows[index] for index in complete_indices]
                complete_generation_ranks = [
                    int(row.get("generation_rank", index))
                    for index, row in zip(complete_indices, complete_rows)
                ]
                complete_pre_scores = [
                    float(row.get("candidate_pre_rank_score", 0.0)) for row in complete_rows
                ]
                admitted_positions = {
                    candidate_id(row): position for position, row in enumerate(admitted, start=1)
                }
                cap_row = {
                    **base,
                    "complete_candidate_count": len(complete_rows),
                    "best_complete_generation_rank_zero_based": min(complete_generation_ranks),
                    "complete_generation_ranks_zero_based": complete_generation_ranks,
                    "complete_candidate_sizes": [
                        len(row["subgraph_units"]) for row in complete_rows
                    ],
                    "complete_candidate_pre_rank_scores": complete_pre_scores,
                    "best_complete_admission_position": min(
                        (
                            admitted_positions[candidate_id(row)]
                            for row in complete_rows
                            if candidate_id(row) in admitted_positions
                        ),
                        default=None,
                    ),
                    "complete_only_after_admission_320": not admitted_complete,
                    "admission_cutoff_pre_rank_score": float(
                        admitted[-1].get("candidate_pre_rank_score", 0.0)
                    ),
                    "best_complete_pre_rank_minus_cutoff": max(complete_pre_scores)
                    - float(admitted[-1].get("candidate_pre_rank_score", 0.0)),
                    "complete_query_anchor_coverage": [
                        len(set(row["subgraph_units"]) & set(generated.query_anchor_unit_ids))
                        / max(len(set(generated.query_anchor_unit_ids)), 1)
                        for row in complete_rows
                    ],
                    "complete_symbolic_features": [
                        row.get("symbolic_features", []) for row in complete_rows
                    ],
                }
                cap_analysis.append(cap_row)

            gnn_complete = contains(gnn_top["subgraph_units"], gold)
            final_complete = contains(final_top["subgraph_units"], gold)
            if admitted_complete:
                # The persisted artifact contains exact scores for the first five.
                # Pick the highest-ranked complete candidate visible there; the
                # full-score supplement can later replace this conservative view.
                visible_complete = [
                    candidate
                    for candidate in gnn_record["ranked_candidates"]
                    if contains(candidate["subgraph_units"], gold)
                ]
                complete_source = visible_complete[0] if visible_complete else None
                top_row = next(
                    row
                    for row in admitted
                    if unit_set(row["subgraph_units"]) == unit_set(gnn_top["subgraph_units"])
                )
                complete_row = None
                if complete_source is not None:
                    complete_row = next(
                        row
                        for row in admitted
                        if unit_set(row["subgraph_units"])
                        == unit_set(complete_source["subgraph_units"])
                    )
                pair = {
                    **base,
                    "gnn_top1_complete": gnn_complete,
                    "rank1_candidate": {
                        "score": float(gnn_top["score"]),
                        "subgraph_units": gnn_top["subgraph_units"],
                        "candidate_size": len(gnn_top["subgraph_units"]),
                        "generator_pre_rank_score": float(
                            top_row.get("candidate_pre_rank_score", 0.0)
                        ),
                        "rank_target": float(top_row.get("rank_target", 0.0)),
                        "category": candidate_category(top_row, gold),
                        "symbolic_features": top_row.get("symbolic_features", []),
                    },
                    "best_complete_candidate": None,
                    "complete_candidate_visible_in_persisted_top5": complete_source is not None,
                }
                if complete_source is not None and complete_row is not None:
                    pair["best_complete_candidate"] = {
                        "score": float(complete_source["score"]),
                        "subgraph_units": complete_source["subgraph_units"],
                        "candidate_size": len(complete_source["subgraph_units"]),
                        "generator_pre_rank_score": float(
                            complete_row.get("candidate_pre_rank_score", 0.0)
                        ),
                        "rank_target": float(complete_row.get("rank_target", 0.0)),
                        "category": candidate_category(complete_row, gold),
                        "symbolic_features": complete_row.get("symbolic_features", []),
                    }
                    pair["top1_minus_complete_score_gap"] = float(gnn_top["score"]) - float(
                        complete_source["score"]
                    )
                    pair["complete_has_strictly_better_target"] = float(
                        complete_row.get("rank_target", 0.0)
                    ) > float(top_row.get("rank_target", 0.0))
                    pair["representation_comparison"] = pair_feature_distance(top_row, complete_row)
                gnn_analysis.append(pair)

            if gnn_complete != final_complete:
                effect = "fixed" if final_complete else "harmed"
            else:
                effect = "unchanged"
            if effect != "unchanged" or taxonomy == "6_symbolic_reranking_degradation":
                gnn_top_row = next(
                    row
                    for row in admitted
                    if unit_set(row["subgraph_units"]) == unit_set(gnn_top["subgraph_units"])
                )
                final_top_row = next(
                    row
                    for row in admitted
                    if unit_set(row["subgraph_units"]) == unit_set(final_top["subgraph_units"])
                )
                symbolic_analysis.append(
                    {
                        **base,
                        "effect": effect,
                        "mode": final_mode,
                        "gnn_top1": {
                            "complete": gnn_complete,
                            "neural_score": float(gnn_top["score"]),
                            "final_score": next(
                                (
                                    float(c["adjusted_score"])
                                    for c in final_record["ranked_candidates"]
                                    if unit_set(c["subgraph_units"])
                                    == unit_set(gnn_top["subgraph_units"])
                                ),
                                None,
                            ),
                            "symbolic_terms": symbolic_terms(gnn_top_row, final_mode),
                            "subgraph_units": gnn_top["subgraph_units"],
                        },
                        "final_top1": {
                            "complete": final_complete,
                            "neural_score": next(
                                (
                                    float(c["score"])
                                    for c in gnn_record["ranked_candidates"]
                                    if unit_set(c["subgraph_units"])
                                    == unit_set(final_top["subgraph_units"])
                                ),
                                None,
                            ),
                            "final_score": float(final_top["adjusted_score"]),
                            "symbolic_terms": symbolic_terms(final_top_row, final_mode),
                            "subgraph_units": final_top["subgraph_units"],
                        },
                    }
                )

            if admitted_union_complete and not admitted_complete:
                final_ranked_rows = []
                for candidate in final_record["ranked_candidates"]:
                    match = next(
                        row
                        for row in admitted
                        if unit_set(row["subgraph_units"]) == unit_set(candidate["subgraph_units"])
                    )
                    final_ranked_rows.append(match)
                cover = minimum_candidate_cover(final_ranked_rows, gold, max_k=5)
                adaptive_row = adaptive.get(example_id, {})
                selected_k = adaptive_row.get("selected_k") or adaptive_row.get(
                    "prediction", {}
                ).get("selected_k")
                ranks = cover.get("candidate_ranks", [])
                aggregation_analysis.append(
                    {
                        **base,
                        **cover,
                        **overlap_redundancy(final_ranked_rows, ranks),
                        "adaptive_selected_k": selected_k,
                        "stops_too_early": bool(
                            selected_k
                            and cover.get("minimum_achievable_k")
                            and int(selected_k) < int(cover["minimum_achievable_k"])
                        ),
                        "aggregation_subtype": (
                            "G1_inherent_multi_candidate_requirement"
                            if not representable
                            else "G2_candidate_construction_failure_masquerading_as_aggregation"
                        ),
                    }
                )

            if example_number % 100 == 0:
                print(f"{dataset}: {example_number} DEV examples scanned", flush=True)

    taxonomy_counts = Counter(row["taxonomy"] for row in all_details)
    per_dataset: dict[str, Any] = {}
    for dataset, _, _ in DATASETS:
        subset = [row for row in all_details if row["dataset"] == dataset]
        counts = Counter(row["taxonomy"] for row in subset)
        per_dataset[dataset] = {
            "failures": len(subset),
            "counts": dict(sorted(counts.items())),
            "percentages": {
                key: 100.0 * value / len(subset) for key, value in sorted(counts.items())
            },
        }
    taxonomy_output = {
        "schema_version": "production_generator_d_v1_causal_failure_taxonomy_v1",
        "split": "dev",
        "failure_definition": "frozen SAGE-QA final rank-1 does not contain a complete gold explanation",
        "failures": len(all_details),
        "counts": dict(sorted(taxonomy_counts.items())),
        "percentages": {
            key: 100.0 * value / len(all_details) for key, value in sorted(taxonomy_counts.items())
        },
        "assignments": all_details,
        "taxonomy_disambiguation": "Structurally non-singleton-representable cases are assigned to stage 7 when the admitted candidate union can assemble a gold explanation (G1); otherwise they remain stage 2. This preserves the requested true-aggregation class while avoiding double counting.",
    }

    write_json(args.output_dir / "causal_failure_taxonomy.json", taxonomy_output)
    write_json(args.output_dir / "per_dataset_taxonomy.json", per_dataset)
    write_json(
        args.output_dir / "constructability_analysis.json",
        {"split": "dev", "examples": constructability},
    )
    write_json(
        args.output_dir / "generator_search_failures.json",
        {"split": "dev", "examples": search_failures},
    )
    write_json(
        args.output_dir / "cap_admission_analysis.json", {"split": "dev", "examples": cap_analysis}
    )
    write_json(
        args.output_dir / "gnn_ranking_analysis.json",
        {
            "split": "dev",
            "examples": gnn_analysis,
            "limitation": "Scores come from the frozen persisted top-five artifact in this structural pass. A complete candidate below rank five is marked unavailable in the pair detail and is not treated as evidence about target alignment.",
        },
    )
    write_json(
        args.output_dir / "symbolic_reranking_analysis.json",
        {"split": "dev", "examples": symbolic_analysis},
    )
    write_json(
        args.output_dir / "true_aggregation_analysis.json",
        {"split": "dev", "examples": aggregation_analysis},
    )

    inputs = [args.rankings, args.adaptive] + [
        args.data_root / dataset / "dev_subgraph_retrieval.jsonl" for dataset, _, _ in DATASETS
    ]
    summary_lines = [
        "# Production Generator D causal retrieval diagnosis",
        "",
        "This is a DEV-only post-hoc diagnosis. No TEST rows, training, answer generation, hyperparameter search, candidate regeneration, or production modification was performed.",
        "",
        "## Mutually exclusive first-causal-stage taxonomy",
        "",
        "| Stage | Count | Percent of failures |",
        "|---|---:|---:|",
    ]
    for key, count in sorted(taxonomy_counts.items()):
        summary_lines.append(f"| {key} | {count} | {100.0 * count / len(all_details):.2f}% |")
    summary_lines += [
        "",
        f"Total frozen final-rank-1 DEV failures: {len(all_details)}.",
        "",
        "The mechanistic decision is written separately in `recommended_mechanism.md` after the exact count artifacts are validated.",
        "",
        "## Lineage",
        "",
    ]
    for path in inputs:
        summary_lines.append(f"- `{path.relative_to(ROOT)}` SHA-256 `{sha256(path)}`")
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (args.output_dir / "recommended_mechanism.md").write_text(
        "# Recommended mechanism\n\nPending validation of the causal decomposition artifacts. No intervention is recommended until those checks pass.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
