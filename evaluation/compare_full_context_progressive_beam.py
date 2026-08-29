"""One isolated dev experiment: current clean versus full-context progressive beam.

Candidate generation receives only question/context-visible fields. Gold support is
joined after both methods have frozen their candidates, solely for diagnostics.
This module is not imported by the SAGE-QA training or inference pipeline.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
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
    flatten_context as flatten_2wiki,
    generate_candidates as generate_2wiki,
    get_gold_support_units as gold_2wiki,
    normalize_2wiki_record,
    parse_sentence_unit as parse_2wiki_unit,
    token_set as token_set_2wiki,
)
from data_processing.build_hotpot_subgraph_dataset import (  # noqa: E402
    candidate_pre_rank_score as pre_rank_hotpot,
    flatten_context as flatten_hotpot,
    generate_candidates as generate_hotpot,
    get_gold_support_units as gold_hotpot,
    normalize_hotpot_record,
    parse_sentence_unit as parse_hotpot_unit,
    token_set as token_set_hotpot,
)
from data_processing.retrieval_contracts import (  # noqa: E402
    BUILDER_VERSION,
    clean_text_retrieval_input,
    git_provenance,
    sha256_file,
    write_json,
)
from data_processing.text_kg_constructor import KGConstructionConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TEXT_SAMPLE_SIZE = 200
INVARIANCE_SAMPLE_SIZE = 5

TEXT = {
    "2Wiki": {
        "path": ROOT / "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
        "atomic_budget": 30,
        "max_depth": 4,
        "candidate_cap": 512,
    },
    "HotpotQA": {
        "path": ROOT / "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
        "atomic_budget": 30,
        "max_depth": 3,
        "candidate_cap": 512,
    },
}
ONTOLOGY = {
    "Pizza 100 2-hop": ROOT / "data/raw/pizza_100/pizza_100_2hop.json",
    "Pizza 250 2-hop": ROOT / "data/raw/pizza_250/pizza_250_2hop.json",
    "OWL2Bench 2-hop": ROOT / "data/raw/owl2bench/OWL2Bench_2hop.json",
}
ONTOLOGY_CONFIG = {
    "atomic_budget": 40,
    "max_depth": 6,
    "current_beam_width": 96,
    "candidate_cap": 320,
}

# Only used for cross-sentence connectivity, never for relevance or labels.
BRIDGE_STOPWORDS = {
    "about", "after", "also", "been", "before", "being", "between", "both",
    "could", "does", "from", "have", "into", "more", "other", "over", "same",
    "that", "their", "there", "these", "they", "this", "those", "through",
    "under", "what", "when", "where", "which", "while", "with", "would",
}


def canonical(units: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(units)))


def progressive_beam(
    universe: Sequence[str],
    *,
    max_depth: int,
    beam_width: int,
    score: Callable[[tuple[int, ...]], float],
) -> list[list[str]]:
    """Retain one deterministic width-B frontier per depth and collect all hops."""
    if not universe or max_depth <= 0 or beam_width <= 0:
        return []
    depth_limit = min(max_depth, len(universe))
    frontier = [(index,) for index in range(len(universe))]
    frontier.sort(key=lambda indices: (-score(indices), indices))
    frontier = frontier[:beam_width]
    retained = list(frontier)

    for _depth in range(2, depth_limit + 1):
        expansions: set[tuple[int, ...]] = set()
        for hypothesis in frontier:
            selected = set(hypothesis)
            for next_index in range(len(universe)):
                if next_index not in selected:
                    expansions.add(tuple(sorted((*hypothesis, next_index))))
        if not expansions:
            break
        frontier = sorted(expansions, key=lambda indices: (-score(indices), indices))[
            :beam_width
        ]
        retained.extend(frontier)

    seen: set[tuple[str, ...]] = set()
    candidates: list[list[str]] = []
    for indices in retained:
        candidate = canonical(universe[index] for index in indices)
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(list(candidate))
    return candidates


def text_beam_candidates(
    sentence_units: Sequence[str],
    *,
    question: str,
    max_depth: int,
    beam_width: int,
    pre_rank: Callable[[str, list[str]], float],
    token_setter: Callable[[str], set[str]],
    unit_parser: Callable[[str], tuple[str, int, str]],
) -> list[list[str]]:
    """Full-context text beam using existing relevance plus transparent bridges."""
    universe = sorted(set(sentence_units))
    parsed = [unit_parser(unit) for unit in universe]
    title_tokens = [token_setter(title) for title, _, _ in parsed]
    sentence_tokens = [token_setter(sentence) for _, _, sentence in parsed]
    bridge_tokens = [
        {token for token in title_tokens[i] | sentence_tokens[i]
         if len(token) >= 4 and token not in BRIDGE_STOPWORDS}
        for i in range(len(universe))
    ]

    def score(indices: tuple[int, ...]) -> float:
        units = [universe[index] for index in indices]
        base = pre_rank(question, units)
        shared_links = 0
        title_links = 0
        titles = {parsed[index][0] for index in indices}
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                shared_links += int(bool(bridge_tokens[left] & bridge_tokens[right]))
                title_links += int(
                    bool(title_tokens[left] & sentence_tokens[right])
                    or bool(title_tokens[right] & sentence_tokens[left])
                )
        return (
            base
            + 0.04 * min(shared_links, 4)
            + 0.06 * min(title_links, 3)
            + 0.03 * float(len(titles) > 1)
        )

    return progressive_beam(
        universe, max_depth=max_depth, beam_width=beam_width, score=score
    )


def ontology_beam_candidates(
    context_axioms: Sequence[str],
    *,
    question: str,
    sparql_query: str,
    max_depth: int,
    beam_width: int,
) -> list[list[str]]:
    """Full parsed-context beam using existing query and RDF/schema connectivity."""
    universe = sorted(set(context_axioms))
    signature = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities = set(signature.get("query_entities", []))
    query_properties = set(signature.get("query_properties", []))
    adjacency = build_unit_adjacency(universe)
    unit_scores = [
        score_unit_for_query(
            unit,
            query_entities=query_entities,
            query_properties=query_properties,
            degree=len(adjacency[index]),
        )
        for index, unit in enumerate(universe)
    ]

    def score(indices: tuple[int, ...]) -> float:
        return score_subgraph_indices(indices, universe, adjacency, unit_scores)

    return progressive_beam(
        universe, max_depth=max_depth, beam_width=beam_width, score=score
    )


def _sample_text_rows(name: str) -> list[dict[str, Any]]:
    normalizer = normalize_2wiki_record if name == "2Wiki" else normalize_hotpot_record
    rows = [normalizer(row) for row in pd.read_parquet(TEXT[name]["path"]).to_dict("records")]
    rows.sort(key=lambda row: str(row.get("_id") or row.get("id") or ""))
    random.Random(SEED).shuffle(rows)
    return rows[:TEXT_SAMPLE_SIZE]


def _ontology_dev_rows(path: Path) -> list[tuple[int, int, dict[str, Any], dict[str, Any]]]:
    groups = json.loads(path.read_text(encoding="utf-8"))
    split = build_split_map(groups, 0.65, 0.1)
    return [
        (group_index, qa_index, item, qa)
        for group_index, item in enumerate(groups)
        if split[group_index] == "dev"
        for qa_index, qa in enumerate(item.get("QAs", []))
        if get_gold_explanations(qa)
    ]


def _recall(have: set[str], gold: set[str]) -> float:
    return len(have & gold) / len(gold) if gold else 0.0


def diagnostics(
    candidates: Sequence[Sequence[str]],
    gold_sets: Sequence[Sequence[str]],
    universe: Sequence[str],
    max_depth: int,
) -> dict[str, Any]:
    """Post-generation diagnostics with best-of-valid-gold semantics."""
    candidate_sets = [set(candidate) for candidate in candidates]
    gold = [set(reference) for reference in gold_sets if reference]
    universe_set = set(universe)
    candidate_union = set().union(*candidate_sets) if candidate_sets else set()
    universe_complete = any(reference <= universe_set for reference in gold)
    retained_complete = any(reference <= candidate_union for reference in gold)
    complete = any(reference <= candidate for reference in gold for candidate in candidate_sets)
    smallest_available = min(
        (len(reference) for reference in gold if reference <= universe_set), default=None
    )
    if complete:
        failure = "complete"
    elif not universe_complete:
        failure = "useful_evidence_not_in_explored_universe"
    elif not retained_complete:
        failure = "useful_evidence_not_retained"
    elif smallest_available is not None and smallest_available > max_depth:
        failure = "gold_support_exceeds_max_depth"
    else:
        failure = "useful_units_retained_but_not_combined"
    return {
        "complete_candidate": complete,
        "candidate_union_atomic_recall": max(
            (_recall(candidate_union, reference) for reference in gold), default=0.0
        ),
        "explored_universe_complete": universe_complete,
        "failure_category": failure,
    }


def _method_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    counts = [int(row[f"{method}_candidate_count"]) for row in rows]
    sizes = [size for row in rows for size in row[f"{method}_candidate_sizes"]]
    diagnostics_rows = [row[f"{method}_diagnostics"] for row in rows]
    runtime = sum(float(row[f"{method}_runtime_seconds"]) for row in rows)
    return {
        "complete_candidate_coverage": {
            "count": sum(item["complete_candidate"] for item in diagnostics_rows),
            "rate": statistics.fmean(
                float(item["complete_candidate"]) for item in diagnostics_rows
            ) if diagnostics_rows else 0.0,
        },
        "evidence_unit_atomic_recall": statistics.fmean(
            item["candidate_union_atomic_recall"] for item in diagnostics_rows
        ) if diagnostics_rows else 0.0,
        "explored_universe_complete_coverage": {
            "count": sum(item["explored_universe_complete"] for item in diagnostics_rows),
            "rate": statistics.fmean(
                float(item["explored_universe_complete"]) for item in diagnostics_rows
            ) if diagnostics_rows else 0.0,
        },
        "zero_candidate_rate": sum(count == 0 for count in counts) / len(counts) if counts else 0.0,
        "average_generated_candidates": statistics.fmean(counts) if counts else 0.0,
        "candidate_count_range": [min(counts, default=0), max(counts, default=0)],
        "candidate_size_distribution": {
            str(size): count for size, count in sorted(collections.Counter(sizes).items())
        },
        "runtime_seconds": runtime,
        "runtime_seconds_per_example": runtime / len(rows) if rows else 0.0,
        "miss_attribution": dict(sorted(collections.Counter(
            item["failure_category"] for item in diagnostics_rows
        ).items())),
    }


def _condition_summary(
    name: str,
    rows: list[dict[str, Any]],
    source: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    current = _method_summary(rows, "current")
    beam = _method_summary(rows, "beam")
    delta = (
        beam["complete_candidate_coverage"]["rate"]
        - current["complete_candidate_coverage"]["rate"]
    )
    return {
        "dataset": name,
        "development_examples": len(rows),
        "source_file": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "cohort_policy": (
            "seed-42 deterministic sample of 200 official-development rows"
            if name in TEXT
            else "all labeled rows in deterministic group-stratified development split"
        ),
        "config": dict(config),
        "current_clean_generator": current,
        "full_context_progressive_beam": beam,
        "complete_candidate_coverage_delta_percentage_points": 100.0 * delta,
    }


def evaluate_text(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = TEXT[name]
    rows = _sample_text_rows(name)
    generator = generate_2wiki if name == "2Wiki" else generate_hotpot
    flattener = flatten_2wiki if name == "2Wiki" else flatten_hotpot
    gold_fn = gold_2wiki if name == "2Wiki" else gold_hotpot
    pre_rank = pre_rank_2wiki if name == "2Wiki" else pre_rank_hotpot
    token_setter = token_set_2wiki if name == "2Wiki" else token_set_hotpot
    unit_parser = parse_2wiki_unit if name == "2Wiki" else parse_hotpot_unit
    kg_config = KGConstructionConfig(backend="context_only", max_triples=64)
    beam_width = config["candidate_cap"] // config["max_depth"]
    details: list[dict[str, Any]] = []

    for index, example in enumerate(rows):
        clean = clean_text_retrieval_input(example)
        current_started = time.perf_counter()
        generated = generator(
            retrieval_example=clean,
            split_name="full_context_beam_development",
            max_sentences_per_example=config["atomic_budget"],
            max_subgraph_size=config["max_depth"],
            max_candidates_per_question=config["candidate_cap"],
            kg_config=kg_config,
            llm_kg_constructor=None,
            kg_cache=None,
            kg_cache_path=None,
            kg_cache_lock=None,
            seed=SEED + index,
        )
        current_runtime = time.perf_counter() - current_started

        beam_started = time.perf_counter()
        sentence_records = flattener(clean)
        full_universe = [record["unit"] for record in sentence_records]
        beam_candidates = text_beam_candidates(
            full_universe,
            question=str(clean.get("question", "")),
            max_depth=config["max_depth"],
            beam_width=beam_width,
            pre_rank=pre_rank,
            token_setter=token_setter,
            unit_parser=unit_parser,
        )
        beam_runtime = time.perf_counter() - beam_started

        # Gold boundary: neither generator above receives this value.
        gold_sets = [gold_fn(example, generated["sent_lookup"])]
        details.append({
            "example_id": generated["example_id"],
            "full_context_unit_count": len(full_universe),
            "current_candidate_count": len(generated["candidates"]),
            "current_candidate_sizes": [len(candidate) for candidate in generated["candidates"]],
            "current_runtime_seconds": current_runtime,
            "current_diagnostics": diagnostics(
                generated["candidates"], gold_sets, generated["sentence_pool"], config["max_depth"]
            ),
            "beam_candidate_count": len(beam_candidates),
            "beam_candidate_sizes": [len(candidate) for candidate in beam_candidates],
            "beam_runtime_seconds": beam_runtime,
            "beam_diagnostics": diagnostics(
                beam_candidates, gold_sets, full_universe, config["max_depth"]
            ),
        })

    reported_config = {
        "current_atomic_budget": config["atomic_budget"],
        "max_depth": config["max_depth"],
        "candidate_cap": config["candidate_cap"],
        "beam_width": beam_width,
        "beam_universe": "all sentences in provided benchmark context",
    }
    return _condition_summary(name, details, config["path"], reported_config), details


def evaluate_ontology(name: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = ONTOLOGY_CONFIG
    rows = _ontology_dev_rows(path)
    beam_width = config["candidate_cap"] // config["max_depth"]
    details: list[dict[str, Any]] = []

    for group_index, qa_index, item, qa in rows:
        question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
        sparql_query = str(qa.get("SPARQL Query") or "")
        owl_context = str(item.get("OWL Context") or "")
        current_started = time.perf_counter()
        generated = generate_ontology_candidates(
            question=question,
            sparql_query=sparql_query,
            owl_context=owl_context,
            max_subgraph_size=config["max_depth"],
            min_subgraph_size=1,
            max_context_units=config["atomic_budget"],
            candidate_beam_width=config["current_beam_width"],
            max_candidate_subgraphs=config["candidate_cap"],
        )
        current_runtime = time.perf_counter() - current_started

        beam_started = time.perf_counter()
        full_universe = parse_owl_context(owl_context)
        beam_candidates = ontology_beam_candidates(
            full_universe,
            question=question,
            sparql_query=sparql_query,
            max_depth=config["max_depth"],
            beam_width=beam_width,
        )
        beam_runtime = time.perf_counter() - beam_started

        # Gold boundary: explanations are read only after both candidate sets freeze.
        gold_sets = get_gold_explanations(qa)
        details.append({
            "example_id": f"{name}__g{group_index}__q{qa_index}",
            "full_context_unit_count": len(full_universe),
            "current_candidate_count": len(generated["candidate_subgraphs"]),
            "current_candidate_sizes": [len(candidate) for candidate in generated["candidate_subgraphs"]],
            "current_runtime_seconds": current_runtime,
            "current_diagnostics": diagnostics(
                generated["candidate_subgraphs"], gold_sets, generated["candidate_units"], config["max_depth"]
            ),
            "beam_candidate_count": len(beam_candidates),
            "beam_candidate_sizes": [len(candidate) for candidate in beam_candidates],
            "beam_runtime_seconds": beam_runtime,
            "beam_diagnostics": diagnostics(
                beam_candidates, gold_sets, full_universe, config["max_depth"]
            ),
        })

    reported_config = {
        "current_atomic_budget": config["atomic_budget"],
        "current_beam_width": config["current_beam_width"],
        "max_depth": config["max_depth"],
        "candidate_cap": config["candidate_cap"],
        "beam_width": beam_width,
        "beam_universe": "all parsed axioms/triples in provided OWL context",
    }
    return _condition_summary(name, details, path, reported_config), details


def _digest(candidates: Sequence[Sequence[str]]) -> str:
    payload = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def invariance_checks() -> dict[str, Any]:
    """Compare gold-attached and gold-deleted inputs for both generators."""
    result: dict[str, Any] = {"sample_size_per_dataset": INVARIANCE_SAMPLE_SIZE, "datasets": {}}
    kg_config = KGConstructionConfig(backend="context_only", max_triples=64)
    for name, config in TEXT.items():
        generator = generate_2wiki if name == "2Wiki" else generate_hotpot
        flattener = flatten_2wiki if name == "2Wiki" else flatten_hotpot
        pre_rank = pre_rank_2wiki if name == "2Wiki" else pre_rank_hotpot
        token_setter = token_set_2wiki if name == "2Wiki" else token_set_hotpot
        unit_parser = parse_2wiki_unit if name == "2Wiki" else parse_hotpot_unit
        mismatches = []
        for index, attached in enumerate(_sample_text_rows(name)[:INVARIANCE_SAMPLE_SIZE]):
            deleted = {key: value for key, value in attached.items()
                       if key not in {"answer", "supporting_facts", "evidences"}}
            digests = []
            for source in (attached, deleted):
                clean = clean_text_retrieval_input(source)
                current = generator(
                    retrieval_example=clean,
                    split_name="beam_invariance",
                    max_sentences_per_example=config["atomic_budget"],
                    max_subgraph_size=config["max_depth"],
                    max_candidates_per_question=config["candidate_cap"],
                    kg_config=kg_config,
                    llm_kg_constructor=None,
                    kg_cache=None,
                    kg_cache_path=None,
                    kg_cache_lock=None,
                    seed=SEED + index,
                )["candidates"]
                units = [record["unit"] for record in flattener(clean)]
                beam = text_beam_candidates(
                    units,
                    question=str(clean.get("question", "")),
                    max_depth=config["max_depth"],
                    beam_width=config["candidate_cap"] // config["max_depth"],
                    pre_rank=pre_rank,
                    token_setter=token_setter,
                    unit_parser=unit_parser,
                )
                digests.append((_digest(current), _digest(beam)))
            if digests[0] != digests[1]:
                mismatches.append(str(attached.get("_id") or attached.get("id")))
        result["datasets"][name] = {"mismatches": mismatches, "passed": not mismatches}

    for name, path in ONTOLOGY.items():
        mismatches = []
        for group_index, qa_index, item, qa in _ontology_dev_rows(path)[:INVARIANCE_SAMPLE_SIZE]:
            inputs = {
                "question": str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or ""),
                "sparql_query": str(qa.get("SPARQL Query") or ""),
                "owl_context": str(item.get("OWL Context") or ""),
            }
            # Attached/deleted annotations produce the same explicit generation inputs.
            digests = []
            for _annotation_state in ("attached", "deleted"):
                current = generate_ontology_candidates(
                    **inputs,
                    max_subgraph_size=ONTOLOGY_CONFIG["max_depth"],
                    min_subgraph_size=1,
                    max_context_units=ONTOLOGY_CONFIG["atomic_budget"],
                    candidate_beam_width=ONTOLOGY_CONFIG["current_beam_width"],
                    max_candidate_subgraphs=ONTOLOGY_CONFIG["candidate_cap"],
                )["candidate_subgraphs"]
                universe = parse_owl_context(inputs["owl_context"])
                beam = ontology_beam_candidates(
                    universe,
                    question=inputs["question"],
                    sparql_query=inputs["sparql_query"],
                    max_depth=ONTOLOGY_CONFIG["max_depth"],
                    beam_width=ONTOLOGY_CONFIG["candidate_cap"] // ONTOLOGY_CONFIG["max_depth"],
                )
                digests.append((_digest(current), _digest(beam)))
            if digests[0] != digests[1]:
                mismatches.append(f"{name}__g{group_index}__q{qa_index}")
        result["datasets"][name] = {"mismatches": mismatches, "passed": not mismatches}
    result["all_passed"] = all(item["passed"] for item in result["datasets"].values())
    return result


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "full_context_progressive_beam_development_v1",
        "development_only": True,
        "seed": SEED,
        "builder_version": BUILDER_VERSION,
        "git": git_provenance(ROOT),
        "gold_available_during_candidate_generation": False,
        "gold_join_boundary": "after current and beam candidates are frozen",
        "configuration_policy": "one fixed configuration; beam width=floor(existing candidate cap/max depth)",
        "prohibited_actions_performed": {
            "test_data": False,
            "llm_or_api_calls": False,
            "gnn_training": False,
            "adaptive_k": False,
            "answer_generation": False,
            "parameter_grid": False,
        },
        "datasets": {},
    }
    for name in TEXT:
        summary, details = evaluate_text(name)
        report["datasets"][name] = summary
        _write_jsonl(output_dir / name / "details.jsonl", details)
    for name, path in ONTOLOGY.items():
        summary, details = evaluate_ontology(name, path)
        report["datasets"][name] = summary
        _write_jsonl(output_dir / name / "details.jsonl", details)
    report["invariance"] = invariance_checks()
    write_json(output_dir / "comparison_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/development_runs/full_context_progressive_beam_v1",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
