"""Compare two gold-free ontology atomic-pool builders on development data.

This is an isolated development experiment. Both approaches use the same
``beam_connected_subgraphs`` candidate composer and fixed candidate budgets.
Gold explanations are read only after candidate generation has finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    beam_connected_subgraphs,
    build_split_map,
    generate_ontology_candidates,
    get_gold_explanations,
    parse_owl_context,
    unit_signature,
)
from data_processing.retrieval_contracts import (  # noqa: E402
    git_provenance,
    sha256_file,
    write_json,
)
from models.symbolic_composer import extract_query_signature  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "Pizza_100_2hop": ROOT / "data/raw/pizza_100/pizza_100_2hop.json",
    "Pizza_250_2hop": ROOT / "data/raw/pizza_250/pizza_250_2hop.json",
    "OWL2Bench_2hop": ROOT / "data/raw/owl2bench/OWL2Bench_2hop.json",
}
CONFIG = {
    "atomic_pool_budget": 40,
    "expansion_hops": 2,
    "min_subgraph_size": 1,
    "max_subgraph_size": 6,
    "candidate_beam_width": 96,
    "candidate_budget": 320,
}
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
TOKEN = re.compile(r"[A-Za-z]+|\d+")


def text_tokens(text: str) -> set[str]:
    """Match the deterministic tokenization used by the reference builder."""
    expanded = CAMEL_BOUNDARY.sub(" ", str(text).replace("_", " ").replace("-", " "))
    return {token.lower() for token in TOKEN.findall(expanded)}


def graph_expansion_atomic_pool(
    axioms: list[str],
    *,
    question: str,
    sparql_query: str,
    max_context_units: int,
    hops: int = 2,
) -> list[str]:
    """Select a bounded question/query-grounded relational neighborhood.

    Units are nodes in the working graph. Shared entities *or properties* form
    connectivity, so an ABox assertion can reach schema axioms about its
    relation. No fallback is used when grounding finds no seed: arbitrary
    ontology order is not treated as question grounding.
    """
    if max_context_units <= 0 or not axioms:
        return []

    query_signature = extract_query_signature(
        question=question, sparql_query=sparql_query
    )
    query_entities = set(query_signature.get("query_entities", []))
    query_properties = set(query_signature.get("query_properties", []))
    question_terms = text_tokens(question)

    signatures = [unit_signature(axiom) for axiom in axioms]
    terms_by_index = [entities | properties for entities, properties in signatures]
    term_to_indices: dict[str, list[int]] = defaultdict(list)
    formal_scores: dict[int, int] = {}
    lexical_scores: dict[int, float] = {}

    for index, ((entities, properties), terms) in enumerate(
        zip(signatures, terms_by_index)
    ):
        for term in terms:
            term_to_indices[term].append(index)
        formal_scores[index] = (
            2 * len(entities & query_entities)
            + len(properties & query_properties)
        )
        readable_tokens = text_tokens(" ".join(sorted(terms)))
        lexical_scores[index] = len(question_terms & readable_tokens) / max(
            len(question_terms), 1
        )

    seeds = {
        index
        for index in range(len(axioms))
        if formal_scores[index] > 0 or lexical_scores[index] > 0.0
    }
    if not seeds:
        return []

    distance = {index: 0 for index in seeds}
    frontier = set(seeds)
    for depth in range(1, hops + 1):
        next_frontier: set[int] = set()
        for index in frontier:
            for term in terms_by_index[index]:
                for neighbor in term_to_indices[term]:
                    if neighbor not in distance:
                        distance[neighbor] = depth
                        next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    ranked = sorted(
        distance,
        key=lambda index: (
            distance[index],
            -formal_scores[index],
            -lexical_scores[index],
            axioms[index],
        ),
    )[:max_context_units]
    return [axioms[index] for index in ranked]


def generate_graph_expansion_candidates(
    *, question: str, sparql_query: str, owl_context: str, config: Mapping[str, int]
) -> dict[str, Any]:
    """Gold-free experimental generation with the production composer."""
    axioms = parse_owl_context(owl_context)
    candidate_units = graph_expansion_atomic_pool(
        axioms,
        question=question,
        sparql_query=sparql_query,
        max_context_units=config["atomic_pool_budget"],
        hops=config["expansion_hops"],
    )
    candidate_subgraphs = beam_connected_subgraphs(
        candidate_units=candidate_units,
        question=question,
        sparql_query=sparql_query,
        min_subgraph_size=config["min_subgraph_size"],
        max_subgraph_size=config["max_subgraph_size"],
        beam_width=config["candidate_beam_width"],
        max_candidate_subgraphs=config["candidate_budget"],
    )
    return {
        "candidate_units": candidate_units,
        "candidate_subgraphs": sorted(candidate_subgraphs, key=lambda row: (len(row), row)),
        "gold_available_during_candidate_generation": False,
    }


def development_rows(path: Path) -> list[dict[str, Any]]:
    groups = json.loads(path.read_text(encoding="utf-8"))
    split_by_group = build_split_map(groups, train_ratio=0.65, dev_ratio=0.1)
    return [
        {
            "group_index": group_index,
            "qa_index": qa_index,
            "item": item,
            "qa": qa,
        }
        for group_index, item in enumerate(groups)
        if split_by_group[group_index] == "dev"
        for qa_index, qa in enumerate(item.get("QAs", []))
        if get_gold_explanations(qa)
    ]


def clean_generation_input(row: Mapping[str, Any]) -> dict[str, str]:
    qa = row["qa"]
    return {
        "question": str(
            qa.get("NL Question")
            or qa.get("ABS Question")
            or qa.get("Task ID")
            or qa.get("SPARQL Query")
            or ""
        ),
        "sparql_query": str(qa.get("SPARQL Query") or ""),
        "owl_context": str(row["item"].get("OWL Context") or ""),
    }


def generate_approach(
    approach: str, rows: list[dict[str, Any]], config: Mapping[str, int]
) -> tuple[list[dict[str, Any]], float]:
    """Generate every candidate set before returning to the gold-bearing rows."""
    generated_rows = []
    started = time.perf_counter()
    for row in rows:
        clean = clean_generation_input(row)
        if approach == "current_clean":
            generated = generate_ontology_candidates(
                **clean,
                max_subgraph_size=config["max_subgraph_size"],
                min_subgraph_size=config["min_subgraph_size"],
                max_context_units=config["atomic_pool_budget"],
                candidate_beam_width=config["candidate_beam_width"],
                max_candidate_subgraphs=config["candidate_budget"],
            )
        elif approach == "graph_expansion_2hop":
            generated = generate_graph_expansion_candidates(**clean, config=config)
        else:
            raise ValueError(f"Unknown approach: {approach}")
        generated_rows.append(generated)
    return generated_rows, time.perf_counter() - started


def attach_gold_diagnostics(
    dataset: str,
    source_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Post-generation gold join used only for aggregate coverage diagnostics."""
    details = []
    for source, generated in zip(source_rows, generated_rows):
        golds = [set(gold) for gold in get_gold_explanations(source["qa"]) if gold]
        pool = set(generated["candidate_units"])
        candidates = [set(candidate) for candidate in generated["candidate_subgraphs"]]
        recalls = [len(pool & gold) / len(gold) for gold in golds]
        details.append(
            {
                "example_id": (
                    f"{dataset}__g{source['group_index']}__q{source['qa_index']}"
                ),
                "atomic_pool_size": len(pool),
                "candidate_count": len(candidates),
                "atomic_pool_gold_recall": max(recalls, default=0.0),
                "complete_gold_explanation_in_pool": any(gold <= pool for gold in golds),
                "complete_candidate": any(
                    gold <= candidate for gold in golds for candidate in candidates
                ),
                "zero_candidates": not candidates,
            }
        )
    return details


def summarize(details: list[dict[str, Any]], runtime_seconds: float) -> dict[str, Any]:
    count = len(details)
    pct = lambda key: 100.0 * sum(bool(row[key]) for row in details) / max(count, 1)
    mean = lambda key: statistics.fmean(row[key] for row in details) if details else 0.0
    return {
        "examples": count,
        "atomic_pool_gold_recall": mean("atomic_pool_gold_recall"),
        "complete_gold_explanation_in_pool_pct": pct(
            "complete_gold_explanation_in_pool"
        ),
        "complete_candidate_coverage_pct": pct("complete_candidate"),
        "zero_candidate_rate_pct": pct("zero_candidates"),
        "average_atomic_pool_size": mean("atomic_pool_size"),
        "average_candidate_count": mean("candidate_count"),
        "runtime_seconds": runtime_seconds,
        "runtime_seconds_per_example": runtime_seconds / max(count, 1),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "ontology_atomic_pool_comparison_v1",
        "development_only": True,
        "datasets": list(DATASETS),
        "config": dict(CONFIG),
        "only_intended_change": "atomic_pool_construction",
        "candidate_composer": "beam_connected_subgraphs",
        "gold_available_during_candidate_generation": False,
        "gold_join_boundary": "after_all_candidates_generated_per_approach_dataset",
        "answer_available_during_candidate_generation": False,
        "prohibited_actions_performed": {
            "gnn_training": False,
            "text_dataset_access": False,
            "adaptive_k": False,
            "answer_generation": False,
            "test_evaluation": False,
            "parameter_grid": False,
            "llm_or_api_calls": False,
        },
        "git": git_provenance(ROOT),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "results": {},
        "sources": {},
    }
    for dataset, source_path in DATASETS.items():
        rows = development_rows(source_path)
        report["sources"][dataset] = {
            "path": str(source_path.relative_to(ROOT)),
            "sha256": sha256_file(source_path),
            "selection": "all gold-labeled QAs in deterministic group-stratified development split",
        }
        report["results"][dataset] = {}
        for approach in ("current_clean", "graph_expansion_2hop"):
            generated, elapsed = generate_approach(approach, rows, CONFIG)
            details = attach_gold_diagnostics(dataset, rows, generated)
            report["results"][dataset][approach] = summarize(details, elapsed)
            write_jsonl(output_dir / dataset / f"{approach}_details.jsonl", details)
    write_json(output_dir / "comparison.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "outputs/development_runs/ontology_atomic_pool_graph_expansion_v1",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
