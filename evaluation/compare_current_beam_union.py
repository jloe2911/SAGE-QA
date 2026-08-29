"""Isolated development diagnostic for current/beam candidate complementarity.

Candidate generation, union deduplication, pre-ranking, and capping are completed
before gold support is read. This module is not imported by the training or final
inference pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
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
    cap_inference_candidate_rows,
    clean_text_retrieval_input,
    git_provenance,
    sha256_file,
    write_json,
)
from data_processing.text_kg_constructor import KGConstructionConfig  # noqa: E402
from evaluation.compare_full_context_progressive_beam import (  # noqa: E402
    ONTOLOGY,
    ONTOLOGY_CONFIG,
    ROOT,
    SEED,
    TEXT,
    _ontology_dev_rows,
    _sample_text_rows,
    ontology_beam_candidates,
    text_beam_candidates,
)


def canonical(candidate: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(candidate)))


def deduplicate(candidates: Sequence[Sequence[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    result: list[list[str]] = []
    for candidate in candidates:
        key = canonical(candidate)
        if key and key not in seen:
            seen.add(key)
            result.append(list(key))
    return result


def union_candidates(
    current: Sequence[Sequence[str]], beam: Sequence[Sequence[str]]
) -> list[list[str]]:
    return deduplicate([*current, *beam])


def cap_union(
    candidates: Sequence[Sequence[str]],
    *,
    budget: int,
    score: Callable[[list[str]], float],
) -> list[list[str]]:
    rows = [
        {
            "subgraph_units": list(candidate),
            "candidate_pre_rank_score": score(list(candidate)),
            "generation_rank": rank,
        }
        for rank, candidate in enumerate(candidates)
    ]
    return [
        list(row["subgraph_units"])
        for row in cap_inference_candidate_rows(rows, max_candidates=budget)
    ]


def complete(candidates: Sequence[Sequence[str]], gold_sets: Sequence[Sequence[str]]) -> bool:
    candidate_sets = [set(candidate) for candidate in candidates]
    references = [set(reference) for reference in gold_sets if reference]
    return any(reference <= candidate for reference in references for candidate in candidate_sets)


def digest(candidates: Sequence[Sequence[str]]) -> str:
    payload = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize(name: str, source: Path, rows: Sequence[Mapping[str, Any]], budget: int) -> dict[str, Any]:
    total = len(rows)

    def method(key: str) -> dict[str, Any]:
        solved = sum(bool(row[f"{key}_complete"]) for row in rows)
        counts = [int(row[f"{key}_candidate_count"]) for row in rows]
        return {
            "complete_candidate_coverage": {
                "count": solved,
                "total": total,
                "rate": solved / total if total else 0.0,
            },
            "average_candidate_count_after_deduplication": (
                statistics.fmean(counts) if counts else 0.0
            ),
            "candidate_count_range": [min(counts, default=0), max(counts, default=0)],
        }

    both = sum(bool(row["current_complete"] and row["beam_complete"]) for row in rows)
    current_only = sum(bool(row["current_complete"] and not row["beam_complete"]) for row in rows)
    beam_only = sum(bool(row["beam_complete"] and not row["current_complete"]) for row in rows)
    neither = total - both - current_only - beam_only
    return {
        "dataset": name,
        "development_examples": total,
        "source_file": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "candidate_budget": budget,
        "A_current": method("current"),
        "B_progressive_beam": method("beam"),
        "C_uncapped_deduplicated_union": method("union"),
        "complementarity": {
            "solved_by_both": both,
            "solved_only_by_current": current_only,
            "solved_only_by_beam": beam_only,
            "solved_by_neither": neither,
        },
        "C_gold_free_capped_union": method("capped_union"),
    }


def evaluate_text(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = TEXT[name]
    generator = generate_2wiki if name == "2Wiki" else generate_hotpot
    flattener = flatten_2wiki if name == "2Wiki" else flatten_hotpot
    gold_fn = gold_2wiki if name == "2Wiki" else gold_hotpot
    pre_rank = pre_rank_2wiki if name == "2Wiki" else pre_rank_hotpot
    token_setter = token_set_2wiki if name == "2Wiki" else token_set_hotpot
    unit_parser = parse_2wiki_unit if name == "2Wiki" else parse_hotpot_unit
    kg_config = KGConstructionConfig(backend="context_only", max_triples=64)
    beam_width = config["candidate_cap"] // config["max_depth"]
    details: list[dict[str, Any]] = []

    for index, example in enumerate(_sample_text_rows(name)):
        clean = clean_text_retrieval_input(example)
        generated = generator(
            retrieval_example=clean,
            split_name="candidate_union_development",
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
        current = deduplicate(generated["candidates"])
        full_universe = [record["unit"] for record in flattener(clean)]
        beam = deduplicate(text_beam_candidates(
            full_universe,
            question=str(clean.get("question", "")),
            max_depth=config["max_depth"],
            beam_width=beam_width,
            pre_rank=pre_rank,
            token_setter=token_setter,
            unit_parser=unit_parser,
        ))
        union = union_candidates(current, beam)
        capped = cap_union(
            union,
            budget=config["candidate_cap"],
            score=lambda candidate: pre_rank(str(clean.get("question", "")), candidate),
        )

        # Gold boundary: generation, unioning, scoring, and capping are frozen above.
        gold_sets = [gold_fn(example, generated["sent_lookup"])]
        details.append(_detail_row(generated["example_id"], current, beam, union, capped, gold_sets))

    return summarize(name, config["path"], details, config["candidate_cap"]), details


def _ontology_scorer(
    *, question: str, sparql_query: str, universe: list[str]
) -> Callable[[list[str]], float]:
    adjacency = build_unit_adjacency(universe)
    signature = extract_query_signature(question=question, sparql_query=sparql_query)
    entities = set(signature.get("query_entities", []))
    properties = set(signature.get("query_properties", []))
    unit_scores = [
        score_unit_for_query(
            unit,
            query_entities=entities,
            query_properties=properties,
            degree=len(adjacency[index]),
        )
        for index, unit in enumerate(universe)
    ]
    index_by_unit = {unit: index for index, unit in enumerate(universe)}
    def score(candidate: list[str]) -> float:
        indices = tuple(sorted(index_by_unit[unit] for unit in candidate))
        return score_subgraph_indices(indices, universe, adjacency, unit_scores)

    return score


def evaluate_ontology(name: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = ONTOLOGY_CONFIG
    beam_width = config["candidate_cap"] // config["max_depth"]
    details: list[dict[str, Any]] = []
    for group_index, qa_index, item, qa in _ontology_dev_rows(path):
        question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
        sparql_query = str(qa.get("SPARQL Query") or "")
        owl_context = str(item.get("OWL Context") or "")
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
        current = deduplicate(generated["candidate_subgraphs"])
        full_universe = sorted(set(parse_owl_context(owl_context)))
        beam = deduplicate(ontology_beam_candidates(
            full_universe,
            question=question,
            sparql_query=sparql_query,
            max_depth=config["max_depth"],
            beam_width=beam_width,
        ))
        union = union_candidates(current, beam)
        pre_rank = _ontology_scorer(
            question=question,
            sparql_query=sparql_query,
            universe=full_universe,
        )
        capped = cap_union(
            union,
            budget=config["candidate_cap"],
            score=pre_rank,
        )

        # Gold boundary: generation, unioning, scoring, and capping are frozen above.
        gold_sets = get_gold_explanations(qa)
        example_id = f"{name}__g{group_index}__q{qa_index}"
        details.append(_detail_row(example_id, current, beam, union, capped, gold_sets))

    return summarize(name, path, details, config["candidate_cap"]), details


def _detail_row(
    example_id: str,
    current: list[list[str]],
    beam: list[list[str]],
    union: list[list[str]],
    capped: list[list[str]],
    gold_sets: Sequence[Sequence[str]],
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "current_candidate_count": len(current),
        "beam_candidate_count": len(beam),
        "union_candidate_count": len(union),
        "capped_union_candidate_count": len(capped),
        "current_complete": complete(current, gold_sets),
        "beam_complete": complete(beam, gold_sets),
        "union_complete": complete(union, gold_sets),
        "capped_union_complete": complete(capped, gold_sets),
        "candidate_digests": {
            "current": digest(current),
            "beam": digest(beam),
            "union": digest(union),
            "capped_union": digest(capped),
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "current_beam_union_development_v1",
        "development_only": True,
        "seed": SEED,
        "builder_version": BUILDER_VERSION,
        "git": git_provenance(ROOT),
        "gold_available_during_candidate_generation_or_capping": False,
        "gold_join_boundary": "after A, B, uncapped C, and capped C are frozen",
        "deduplication_key": "sorted unique subgraph_units",
        "capped_union_selection": (
            "existing candidate_pre_rank_score followed by existing deterministic "
            "cap_inference_candidate_rows tie-breaking"
        ),
        "candidate_budgets": {"text": 512, "ontology": 320},
        "prohibited_actions_performed": {
            "test_data": False,
            "llm_or_api_calls": False,
            "training": False,
            "adaptive_k": False,
            "answer_generation": False,
            "tuning": False,
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
    write_json(output_dir / "comparison_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/development_runs/current_beam_union_v1",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
