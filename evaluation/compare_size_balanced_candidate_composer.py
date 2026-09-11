"""Isolated dev-only comparison of current and size-balanced composers.

Candidate atomic pools are constructed once by the clean builders and reused
unchanged by both composers. Gold annotations are joined only after both
candidate sets have been frozen.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    beam_connected_subgraphs,
    build_split_map,
    build_unit_adjacency,
    extract_query_signature,
    generate_ontology_candidates,
    get_gold_explanations,
    parse_owl_context,
    score_subgraph_indices,
    score_unit_for_query,
)
from data_processing.build_2wiki_subgraph_dataset import (  # noqa: E402
    candidate_pre_rank_score as pre_rank_2wiki,
    generate_candidate_subgraphs as compose_2wiki_current,
    generate_candidates as generate_2wiki,
    get_gold_support_units as gold_2wiki,
    normalize_2wiki_record,
    parse_sentence_unit as parse_2wiki_unit,
    token_set as token_set_2wiki,
)
from data_processing.build_hotpot_subgraph_dataset import (  # noqa: E402
    candidate_pre_rank_score as pre_rank_hotpot,
    generate_candidate_subgraphs as compose_hotpot_current,
    generate_candidates as generate_hotpot,
    get_gold_support_units as gold_hotpot,
    normalize_hotpot_record,
    parse_sentence_unit as parse_hotpot_unit,
    token_set as token_set_hotpot,
)
from data_processing.retrieval_contracts import (  # noqa: E402
    clean_text_retrieval_input,
    sha256_file,
    write_json,
)
from data_processing.text_kg_constructor import KGConstructionConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEXT = {
    "2Wiki": {
        "path": ROOT / "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
        "atomic_budget": 30,
        "max_size": 4,
        "candidate_cap": 512,
    },
    "HotpotQA": {
        "path": ROOT / "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
        "atomic_budget": 30,
        "max_size": 3,
        "candidate_cap": 512,
    },
}
ONTOLOGY = {
    "Pizza 100 2-hop": ROOT / "data/raw/pizza_100/pizza_100_2hop.json",
    "OWL2Bench 2-hop": ROOT / "data/raw/owl2bench/OWL2Bench_2hop.json",
}
ONTOLOGY_CONFIG = {
    "atomic_budget": 40,
    "max_size": 6,
    "beam_width": 96,
    "candidate_cap": 320,
}
RESERVED_FRACTION = 0.5
MEANINGFUL_ABSOLUTE_PP = 2.0


def _canonical(units: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(units)))


def _reserve_by_size(
    *,
    buckets: Mapping[int, Sequence[tuple[Any, tuple[str, ...]]]],
    current_order: Sequence[tuple[str, ...]],
    max_size: int,
    budget: int,
) -> list[list[str]]:
    """Reserve half the cap evenly across sizes 2..max_size.

    Reserved candidates use the supplied bucket order. Remaining and unused
    slots use the current composer's stable order, so the change is confined
    to fixed-budget allocation.
    """
    if budget <= 0:
        return []
    larger_sizes = list(range(2, max_size + 1))
    reserved_total = min(budget, int(math.floor(budget * RESERVED_FRACTION)))
    base, remainder = divmod(reserved_total, max(len(larger_sizes), 1))
    quotas = {
        size: base + int(index < remainder) for index, size in enumerate(reversed(larger_sizes))
    }

    selected: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for size in larger_sizes:
        for _, candidate in buckets.get(size, ())[: quotas[size]]:
            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)

    # Current-order fill also deterministically redistributes unavailable
    # reserved slots without changing the total candidate budget.
    for candidate in current_order:
        if len(selected) >= budget:
            break
        if candidate not in seen:
            seen.add(candidate)
            selected.append(candidate)

    # A reserved candidate may already occur early in current order. If that
    # causes a shortfall, use the remaining ranked bucket candidates.
    if len(selected) < budget:
        for size in range(1, max_size + 1):
            for _, candidate in buckets.get(size, ()):
                if len(selected) >= budget:
                    break
                if candidate not in seen:
                    seen.add(candidate)
                    selected.append(candidate)
    return [list(candidate) for candidate in selected]


def compose_text_balanced(
    *,
    sentence_pool: Sequence[str],
    question: str,
    max_size: int,
    budget: int,
    seed: int,
    pre_rank: Callable[[str, list[str]], float],
    token_setter: Callable[[str], set[str]] | None = None,
    unit_parser: Callable[[str], tuple[str, int, str]] | None = None,
) -> list[list[str]]:
    """Run the existing combination/sampling logic, then balance its cap."""
    rng = random.Random(seed)
    buckets: dict[int, list[tuple[float, tuple[str, ...]]]] = {}
    current_order: list[tuple[str, ...]] = []
    sample_limit = budget * 4
    question_tokens = token_setter(question) if token_setter else set()
    unit_tokens = {unit: token_setter(unit) for unit in sentence_pool} if token_setter else {}
    title_tokens = (
        {unit: token_setter(unit_parser(unit)[0]) for unit in sentence_pool}
        if token_setter and unit_parser
        else {}
    )
    titles = {unit: unit_parser(unit)[0] for unit in sentence_pool} if unit_parser else {}

    def score(candidate: tuple[str, ...]) -> float:
        if not token_setter or not unit_parser:
            return pre_rank(question, list(candidate))
        candidate_tokens: set[str] = set()
        candidate_title_tokens: set[str] = set()
        candidate_titles: set[str] = set()
        for unit in candidate:
            candidate_tokens.update(unit_tokens[unit])
            candidate_title_tokens.update(title_tokens[unit])
            candidate_titles.add(titles[unit])
        denominator = max(len(question_tokens), 1)
        return (
            len(question_tokens & candidate_tokens) / denominator
            + 0.2 * len(question_tokens & candidate_title_tokens) / denominator
            + 0.05 * float(len(candidate_titles) > 1)
            - 0.005 * len(candidate)
        )

    for size in range(1, max_size + 1):
        if size == 1:
            combinations = [(unit,) for unit in sentence_pool]
        else:
            combinations = list(itertools.combinations(sentence_pool, size))
            if len(combinations) > sample_limit:
                combinations = rng.sample(combinations, sample_limit)
        candidates = [_canonical(combo) for combo in combinations]
        # Deduplication is normally a no-op, but mirrors the current composer.
        candidates = list(dict.fromkeys(candidate for candidate in candidates if candidate))
        current_order.extend(candidates)
        ranked = sorted(
            ((score(candidate), candidate) for candidate in candidates),
            key=lambda item: (-item[0], item[1]),
        )
        buckets[size] = ranked
    return _reserve_by_size(
        buckets=buckets,
        current_order=current_order,
        max_size=max_size,
        budget=budget,
    )


def compose_ontology_balanced(
    *,
    candidate_units: list[str],
    question: str,
    sparql_query: str,
    min_size: int,
    max_size: int,
    beam_width: int,
    budget: int,
) -> list[list[str]]:
    """Run the existing connected beam and allocate its fixed cap by size."""
    if not candidate_units or max_size < min_size:
        return []
    signature = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities = set(signature.get("query_entities", []))
    query_properties = set(signature.get("query_properties", []))
    adjacency = build_unit_adjacency(candidate_units)
    unit_scores = [
        score_unit_for_query(
            unit,
            query_entities=query_entities,
            query_properties=query_properties,
            degree=len(adjacency[index]),
        )
        for index, unit in enumerate(candidate_units)
    ]
    max_size = min(max_size, len(candidate_units))
    seed_indices = sorted(
        range(len(candidate_units)),
        key=lambda index: (-unit_scores[index], candidate_units[index]),
    )
    frontier = [(index,) for index in seed_indices[: max(beam_width, budget)]]
    seen_indices = set(frontier)
    scored: list[tuple[float, tuple[int, ...]]] = []
    for size in range(1, max_size + 1):
        if size >= min_size:
            scored.extend(
                (
                    score_subgraph_indices(combo, candidate_units, adjacency, unit_scores),
                    combo,
                )
                for combo in frontier
            )
        if size == max_size:
            break
        expansions: dict[tuple[int, ...], float] = {}
        for combo in frontier:
            neighbors: set[int] = set()
            for index in combo:
                neighbors.update(adjacency[index])
            for next_index in neighbors - set(combo):
                expanded = tuple(sorted((*combo, next_index)))
                if expanded in seen_indices:
                    continue
                seen_indices.add(expanded)
                expansions[expanded] = score_subgraph_indices(
                    expanded, candidate_units, adjacency, unit_scores
                )
        if not expansions:
            break
        frontier = [
            combo
            for combo, _ in sorted(expansions.items(), key=lambda item: (-item[1], item[0]))[
                :beam_width
            ]
        ]

    globally_ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    current_order = [
        tuple(candidate_units[index] for index in combo) for _, combo in globally_ranked
    ]
    buckets: dict[int, list[tuple[float, tuple[str, ...]]]] = collections.defaultdict(list)
    for score, combo in globally_ranked:
        candidate = tuple(candidate_units[index] for index in combo)
        buckets[len(candidate)].append((score, candidate))
    return _reserve_by_size(
        buckets=buckets,
        current_order=current_order,
        max_size=max_size,
        budget=budget,
    )


def _complete(candidates: Sequence[Sequence[str]], gold_sets: Sequence[Sequence[str]]) -> bool:
    candidate_sets = [set(candidate) for candidate in candidates]
    return any(set(gold) <= candidate for gold in gold_sets if gold for candidate in candidate_sets)


def _all_gold_in_pool(pool: Sequence[str], gold_sets: Sequence[Sequence[str]]) -> bool:
    pool_set = set(pool)
    return any(set(gold) <= pool_set for gold in gold_sets if gold)


def _method_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    counts = [int(row[f"{method}_count"]) for row in rows]
    sizes = [size for row in rows for size in row[f"{method}_sizes"]]
    covered = sum(bool(row[f"{method}_complete"]) for row in rows)
    conditioned = [row for row in rows if row["all_gold_in_atomic_pool"]]
    conditioned_covered = sum(bool(row[f"{method}_complete"]) for row in conditioned)
    return {
        "complete_candidate_coverage": {
            "count": covered,
            "rate": covered / len(rows) if rows else 0.0,
        },
        "complete_candidate_coverage_conditioned_on_all_gold_in_atomic_pool": {
            "denominator": len(conditioned),
            "count": conditioned_covered,
            "rate": conditioned_covered / len(conditioned) if conditioned else 0.0,
        },
        "candidate_count": {
            "mean": statistics.fmean(counts) if counts else 0.0,
            "median": statistics.median(counts) if counts else 0.0,
            "min": min(counts, default=0),
            "max": max(counts, default=0),
        },
        "candidate_size_distribution": {
            str(size): count for size, count in sorted(collections.Counter(sizes).items())
        },
        "runtime_seconds": sum(float(row[f"{method}_runtime_seconds"]) for row in rows),
    }


def _condition_summary(name: str, rows: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    atomic_count = sum(bool(row["all_gold_in_atomic_pool"]) for row in rows)
    current = _method_summary(rows, "current")
    experimental = _method_summary(rows, "experimental")
    current_rate = current["complete_candidate_coverage"]["rate"]
    experimental_rate = experimental["complete_candidate_coverage"]["rate"]
    delta_pp = 100.0 * (experimental_rate - current_rate)
    additional = (
        experimental["complete_candidate_coverage"]["count"]
        - current["complete_candidate_coverage"]["count"]
    )
    return {
        "condition": name,
        "development_examples": len(rows),
        "source_file": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "all_gold_in_atomic_pool": {
            "count": atomic_count,
            "rate": atomic_count / len(rows) if rows else 0.0,
        },
        "current_clean_composer": current,
        "experimental_size_balanced_composer": experimental,
        "complete_candidate_coverage_delta_percentage_points": delta_pp,
        "additional_complete_examples": additional,
        "meaningful_improvement": delta_pp >= MEANINGFUL_ABSOLUTE_PP and additional >= 1,
    }


def evaluate_text(name: str, config: Mapping[str, Any], limit: int, seed: int) -> dict[str, Any]:
    path = Path(config["path"])
    normalize = normalize_2wiki_record if name == "2Wiki" else normalize_hotpot_record
    generator = generate_2wiki if name == "2Wiki" else generate_hotpot
    gold_fn = gold_2wiki if name == "2Wiki" else gold_hotpot
    current_fn = compose_2wiki_current if name == "2Wiki" else compose_hotpot_current
    pre_rank = pre_rank_2wiki if name == "2Wiki" else pre_rank_hotpot
    token_setter = token_set_2wiki if name == "2Wiki" else token_set_hotpot
    unit_parser = parse_2wiki_unit if name == "2Wiki" else parse_hotpot_unit
    records = [normalize(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    records.sort(key=lambda row: str(row.get("_id") or row.get("id") or ""))
    if limit > 0:
        records = records[:limit]
    kg_config = KGConstructionConfig(backend="context_only", max_triples=64)
    details: list[dict[str, Any]] = []
    for index, example in enumerate(records):
        generated = generator(
            retrieval_example=clean_text_retrieval_input(example),
            split_name="development_composer_experiment",
            max_sentences_per_example=int(config["atomic_budget"]),
            max_subgraph_size=int(config["max_size"]),
            max_candidates_per_question=int(config["candidate_cap"]),
            kg_config=kg_config,
            llm_kg_constructor=None,
            kg_cache=None,
            kg_cache_path=None,
            kg_cache_lock=None,
            seed=seed + index,
        )
        current_started = time.perf_counter()
        if name == "2Wiki":
            current = current_fn(
                sentence_pool=generated["sentence_pool"],
                max_subgraph_size=int(config["max_size"]),
                max_candidates_per_question=int(config["candidate_cap"]),
                seed=seed + index,
            )
        else:
            current = current_fn(
                example=clean_text_retrieval_input(example),
                sentence_pool=generated["sentence_pool"],
                max_subgraph_size=int(config["max_size"]),
                max_candidates_per_question=int(config["candidate_cap"]),
                seed=seed + index,
            )
        current_runtime = time.perf_counter() - current_started
        if current != generated["candidates"]:
            raise AssertionError(f"Current composer replay mismatch for {generated['example_id']}")
        experimental_started = time.perf_counter()
        experimental = compose_text_balanced(
            sentence_pool=generated["sentence_pool"],
            question=generated["question"],
            max_size=int(config["max_size"]),
            budget=int(config["candidate_cap"]),
            seed=seed + index,
            pre_rank=pre_rank,
            token_setter=token_setter,
            unit_parser=unit_parser,
        )
        experimental_runtime = time.perf_counter() - experimental_started
        gold_sets = [gold_fn(example, generated["sent_lookup"])]
        details.append(
            {
                "all_gold_in_atomic_pool": _all_gold_in_pool(generated["sentence_pool"], gold_sets),
                "current_complete": _complete(current, gold_sets),
                "experimental_complete": _complete(experimental, gold_sets),
                "current_count": len(current),
                "experimental_count": len(experimental),
                "current_sizes": [len(candidate) for candidate in current],
                "experimental_sizes": [len(candidate) for candidate in experimental],
                "current_runtime_seconds": current_runtime,
                "experimental_runtime_seconds": experimental_runtime,
            }
        )
    return _condition_summary(name, details, path)


def _ontology_dev_rows(path: Path) -> list[tuple[int, int, dict, dict]]:
    groups = json.loads(path.read_text(encoding="utf-8"))
    split = build_split_map(groups, 0.65, 0.1)
    return [
        (group_index, qa_index, item, qa)
        for group_index, item in enumerate(groups)
        if split[group_index] == "dev"
        for qa_index, qa in enumerate(item.get("QAs", []))
        if get_gold_explanations(qa)
    ]


def evaluate_ontology(name: str, path: Path, limit: int) -> dict[str, Any]:
    config = ONTOLOGY_CONFIG
    rows = _ontology_dev_rows(path)
    if limit > 0:
        rows = rows[:limit]
    details: list[dict[str, Any]] = []
    for group_index, qa_index, item, qa in rows:
        del group_index, qa_index
        question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
        sparql_query = str(qa.get("SPARQL Query") or "")
        owl_context = str(item.get("OWL Context") or "")
        generated = generate_ontology_candidates(
            question=question,
            sparql_query=sparql_query,
            owl_context=owl_context,
            max_subgraph_size=config["max_size"],
            min_subgraph_size=1,
            max_context_units=config["atomic_budget"],
            candidate_beam_width=config["beam_width"],
            max_candidate_subgraphs=config["candidate_cap"],
        )
        current_started = time.perf_counter()
        current_set = beam_connected_subgraphs(
            candidate_units=generated["candidate_units"],
            question=question,
            sparql_query=sparql_query,
            min_subgraph_size=1,
            max_subgraph_size=config["max_size"],
            beam_width=config["beam_width"],
            max_candidate_subgraphs=config["candidate_cap"],
        )
        current = sorted(current_set, key=lambda candidate: (len(candidate), candidate))
        current_runtime = time.perf_counter() - current_started
        if current != generated["candidate_subgraphs"]:
            raise AssertionError(f"Current ontology composer replay mismatch for {name}")
        experimental_started = time.perf_counter()
        experimental = compose_ontology_balanced(
            candidate_units=generated["candidate_units"],
            question=question,
            sparql_query=sparql_query,
            min_size=1,
            max_size=config["max_size"],
            beam_width=config["beam_width"],
            budget=config["candidate_cap"],
        )
        experimental_runtime = time.perf_counter() - experimental_started
        gold_sets = get_gold_explanations(qa)
        details.append(
            {
                "all_gold_in_atomic_pool": _all_gold_in_pool(
                    generated["candidate_units"], gold_sets
                ),
                "current_complete": _complete(current, gold_sets),
                "experimental_complete": _complete(experimental, gold_sets),
                "current_count": len(current),
                "experimental_count": len(experimental),
                "current_sizes": [len(candidate) for candidate in current],
                "experimental_sizes": [len(candidate) for candidate in experimental],
                "current_runtime_seconds": current_runtime,
                "experimental_runtime_seconds": experimental_runtime,
            }
        )
    return _condition_summary(name, details, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    conditions = [
        evaluate_text(name, config, args.limit, args.seed) for name, config in TEXT.items()
    ]
    conditions.extend(evaluate_ontology(name, path, args.limit) for name, path in ONTOLOGY.items())
    meaningful_count = sum(condition["meaningful_improvement"] for condition in conditions)
    report = {
        "experiment": "size_balanced_structural_composer_v1",
        "development_only": True,
        "seed": args.seed,
        "limit_per_condition": args.limit,
        "candidate_budgets": {"text": 512, "ontology": 320},
        "atomic_pool_budgets_unchanged": {"text": 30, "ontology": 40},
        "scheme": {
            "reserved_fraction": RESERVED_FRACTION,
            "reserved_sizes": "2..max_candidate_size",
            "within_size_order": "existing gold-free pre-rank for text; existing connected-beam score for ontology",
            "unused_slot_policy": "deterministic current-order fill",
        },
        "meaningful_improvement_rule": {
            "minimum_absolute_percentage_points": MEANINGFUL_ABSOLUTE_PP,
            "minimum_additional_complete_examples": 1,
        },
        "conditions": conditions,
        "conditions_with_meaningful_improvement": meaningful_count,
        "recommendation": (
            "consider experimental composer" if meaningful_count >= 3 else "keep current composer"
        ),
    }
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug-only prefix limit per condition; zero evaluates the full development sets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/development_runs/size_balanced_structural_composer_v1/comparison.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
