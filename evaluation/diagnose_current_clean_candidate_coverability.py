"""Post-hoc union coverability for the current clean development candidates.

Candidate generation is completed before benchmark support annotations are read.
This diagnostic does not import or use the alternative progressive-beam generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    generate_ontology_candidates,
    get_gold_explanations,
)
from data_processing.build_2wiki_subgraph_dataset import (  # noqa: E402
    generate_candidates as generate_2wiki_candidates,
    get_gold_support_units as get_2wiki_gold_support,
)
from data_processing.build_hotpot_subgraph_dataset import (  # noqa: E402
    generate_candidates as generate_hotpot_candidates,
    get_gold_support_units as get_hotpot_gold_support,
)
from data_processing.retrieval_contracts import (  # noqa: E402
    BUILDER_VERSION,
    clean_text_retrieval_input,
    git_provenance,
    sha256_file,
    write_json,
)
from data_processing.text_kg_constructor import KGConstructionConfig  # noqa: E402
from evaluation.validate_gold_free_candidate_pools import (  # noqa: E402
    ONTO,
    ONTO_BASE,
    ROOT,
    TEXT,
    TEXT_BASE,
    onto_rows,
    text_rows,
)

DISPLAY_NAMES = {
    "2WikiMultiHopQA": "2Wiki",
    "HotpotQA": "HotpotQA",
    "Family_1hop": "Family 1-hop",
    "Family_2hop": "Family 2-hop",
    "Pizza_100_1hop": "Pizza 100 1-hop",
    "Pizza_100_2hop": "Pizza 100 2-hop",
    "Pizza_250_1hop": "Pizza 250 1-hop",
    "Pizza_250_2hop": "Pizza 250 2-hop",
    "OWL2Bench_1hop": "OWL2Bench 1-hop",
    "OWL2Bench_2hop": "OWL2Bench 2-hop",
}


def _canonical_candidates(candidates: Iterable[Iterable[str]]) -> list[tuple[str, ...]]:
    """Deduplicate candidates by their sorted unique evidence units."""
    return sorted(
        {tuple(sorted(set(candidate))) for candidate in candidates if candidate},
        key=lambda candidate: (len(candidate), candidate),
    )


def _candidate_digest(candidates: Sequence[Sequence[str]]) -> str:
    payload = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _maximal_masks(masks: Iterable[int]) -> tuple[int, ...]:
    """Drop zero, duplicate, and subset-dominated masks."""
    unique = sorted(set(masks) - {0}, key=lambda mask: (-mask.bit_count(), mask))
    maximal: list[int] = []
    for mask in unique:
        if not any(mask | kept == kept for kept in maximal):
            maximal.append(mask)
    return tuple(maximal)


def _minimum_cover(mask_values: Sequence[int], full_mask: int) -> int | None:
    """Return the exact minimum candidate count using a 2^|gold| state space."""
    if not full_mask:
        return None
    masks = _maximal_masks(mask_values)
    if not masks:
        return None
    union_mask = 0
    for mask in masks:
        union_mask |= mask
    if union_mask != full_mask:
        return None

    reached = {0}
    frontier = {0}
    depth = 0
    while frontier:
        depth += 1
        next_frontier = {
            state | mask for state in frontier for mask in masks if (state | mask) not in reached
        }
        if full_mask in next_frontier:
            return depth
        reached.update(next_frontier)
        frontier = next_frontier
    return None


def coverability(
    candidates: Sequence[Sequence[str]], gold_explanations: Sequence[Sequence[str]]
) -> dict[str, Any]:
    candidate_sets = [set(candidate) for candidate in candidates]
    gold_sets = [tuple(dict.fromkeys(gold)) for gold in gold_explanations if gold]
    if not gold_sets:
        raise ValueError("Coverability requires at least one non-empty gold support")

    explanation_results = []
    for gold in gold_sets:
        bit_by_unit = {unit: 1 << index for index, unit in enumerate(gold)}
        full_mask = (1 << len(bit_by_unit)) - 1
        masks = []
        for candidate in candidate_sets:
            mask = 0
            for unit in candidate:
                mask |= bit_by_unit.get(unit, 0)
            masks.append(mask)
        explanation_results.append(
            {
                "gold_unit_count": len(bit_by_unit),
                "deduplicated_nonzero_mask_count": len(set(masks) - {0}),
                "minimum_candidates": _minimum_cover(masks, full_mask),
            }
        )

    possible = [
        result["minimum_candidates"]
        for result in explanation_results
        if result["minimum_candidates"] is not None
    ]
    minimum = min(possible, default=None)
    return {
        "gold_explanation_count": len(gold_sets),
        "minimum_candidates": minimum,
        "all_candidate_union_covers_gold": minimum is not None,
        "explanations": explanation_results,
    }


def _rate(count: int, total: int) -> dict[str, Any]:
    return {
        "count": count,
        "total": total,
        "rate": count / total if total else 0.0,
        "percent": 100.0 * count / total if total else 0.0,
    }


def summarize(
    *,
    name: str,
    source: Path,
    config: Mapping[str, int],
    rows: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    total = len(rows)
    minimums = [row["minimum_candidates"] for row in rows]
    impossible = sum(value is None for value in minimums)

    def within(limit: int) -> int:
        return sum(value is not None and value <= limit for value in minimums)

    distribution_counts = {
        "1": sum(value == 1 for value in minimums),
        "2": sum(value == 2 for value in minimums),
        "3": sum(value == 3 for value in minimums),
        "4-5": sum(value is not None and 4 <= value <= 5 for value in minimums),
        ">5 / impossible": sum(value is None or value > 5 for value in minimums),
    }
    distribution = {
        key: {**_rate(count, total), "total": total} for key, count in distribution_counts.items()
    }
    candidate_counts = [int(row["candidate_count"]) for row in rows]
    summary = {
        "dataset": DISPLAY_NAMES[name],
        "development_examples": total,
        "source_file": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "generation_config": dict(config),
        "complete_candidate_coverage_oracle_at_1": _rate(within(1), total),
        "oracle_support_coverage_at_2": _rate(within(2), total),
        "oracle_support_coverage_at_3": _rate(within(3), total),
        "oracle_support_coverage_at_5": _rate(within(5), total),
        "all_candidate_union_coverage": _rate(total - impossible, total),
        "minimum_candidates_distribution": distribution,
        "minimum_candidates_over_5_count": sum(
            value is not None and value > 5 for value in minimums
        ),
        "impossible_count": impossible,
        "candidate_count": {
            "mean": statistics.fmean(candidate_counts) if candidate_counts else 0.0,
            "minimum": min(candidate_counts, default=0),
            "maximum": max(candidate_counts, default=0),
        },
        "candidate_set_digest": hashlib.sha256(
            "".join(str(row["candidate_digest"]) for row in rows).encode("ascii")
        ).hexdigest(),
        "runtime_seconds": elapsed_seconds,
    }
    metric_counts = [within(1), within(2), within(3), within(5), total - impossible]
    if metric_counts != sorted(metric_counts):
        raise AssertionError(f"Non-monotonic coverability metrics for {name}: {metric_counts}")
    if sum(distribution_counts.values()) != total:
        raise AssertionError(f"Minimum-cover distribution does not sum to N for {name}")
    return summary


def evaluate_text(
    name: str, sample_size: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    examples = text_rows(name, sample_size, seed)
    config = TEXT_BASE[name]
    generator = (
        generate_2wiki_candidates if name == "2WikiMultiHopQA" else generate_hotpot_candidates
    )
    gold_fn = get_2wiki_gold_support if name == "2WikiMultiHopQA" else get_hotpot_gold_support
    kg_config = KGConstructionConfig(backend="context_only", max_triples=64)
    details = []
    started = time.perf_counter()
    for index, example in enumerate(examples):
        clean = clean_text_retrieval_input(example)
        generated = generator(
            retrieval_example=clean,
            split_name="current_clean_coverability_development",
            max_sentences_per_example=config["atomic_budget"],
            max_subgraph_size=config["max_size"],
            max_candidates_per_question=config["candidate_cap"],
            kg_config=kg_config,
            llm_kg_constructor=None,
            kg_cache=None,
            kg_cache_path=None,
            kg_cache_lock=None,
            seed=seed + index,
        )
        candidates = _canonical_candidates(generated["candidates"])

        # Gold boundary: the complete candidate set is frozen above. For text,
        # this reads benchmark supporting_facts only; normalized 2Wiki rows have
        # already discarded evidences.
        gold_support = gold_fn(example, generated["sent_lookup"])
        result = coverability(candidates, [gold_support])
        details.append(
            {
                "example_id": generated["example_id"],
                "candidate_count": len(candidates),
                "candidate_digest": _candidate_digest(candidates),
                "gold_source": "supporting_facts",
                **result,
            }
        )
    return (
        summarize(
            name=name,
            source=TEXT[name],
            config=config,
            rows=details,
            elapsed_seconds=time.perf_counter() - started,
        ),
        details,
    )


def evaluate_ontology(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = ONTO_BASE
    details = []
    started = time.perf_counter()
    for group_index, qa_index, item, qa in onto_rows(name):
        question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
        generated = generate_ontology_candidates(
            question=question,
            sparql_query=str(qa.get("SPARQL Query") or ""),
            owl_context=str(item.get("OWL Context") or ""),
            max_subgraph_size=config["max_size"],
            min_subgraph_size=1,
            max_context_units=config["atomic_budget"],
            candidate_beam_width=config["beam_width"],
            max_candidate_subgraphs=config["candidate_cap"],
        )
        candidates = _canonical_candidates(generated["candidate_subgraphs"])

        # Gold boundary: the current clean candidates are frozen above.
        gold_explanations = get_gold_explanations(qa)
        result = coverability(candidates, gold_explanations)
        details.append(
            {
                "example_id": f"{name}__g{group_index}__q{qa_index}",
                "candidate_count": len(candidates),
                "candidate_digest": _candidate_digest(candidates),
                "gold_source": "Explanations_or_Minimum_Explanation",
                **result,
            }
        )
    return (
        summarize(
            name=name,
            source=ONTO[name],
            config=config,
            rows=details,
            elapsed_seconds=time.perf_counter() - started,
        ),
        details,
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(output_dir: Path, text_sample_size: int, seed: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output_dir}")
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "current_clean_candidate_coverability_v1",
        "development_only": True,
        "seed": seed,
        "text_sample_size": text_sample_size,
        "builder_version": BUILDER_VERSION,
        "git": git_provenance(ROOT),
        "candidate_source": "current clean generator with baseline audit configuration",
        "alternative_progressive_beam_generator_used": False,
        "gold_available_during_candidate_generation": False,
        "gold_join_boundary": "after each complete candidate set is frozen",
        "text_gold_source": "benchmark supporting_facts only",
        "2wiki_evidences_used": False,
        "ontology_multiple_explanation_semantics": "success if any valid explanation is covered",
        "cover_algorithm": "deduplicated candidate-intersection bitmasks with subset dominance and 2^|gold| reachability",
        "prohibited_actions_performed": {
            "test_data": False,
            "training": False,
            "pipeline_modification": False,
            "candidate_generation_modification": False,
            "llm_or_api_calls": False,
            "commit_or_push": False,
        },
        "datasets": {},
    }
    for name in TEXT:
        summary, details = evaluate_text(name, text_sample_size, seed)
        report["datasets"][DISPLAY_NAMES[name]] = summary
        _write_jsonl(output_dir / name / "details.jsonl", details)
    for name in ONTO:
        summary, details = evaluate_ontology(name)
        report["datasets"][DISPLAY_NAMES[name]] = summary
        _write_jsonl(output_dir / name / "details.jsonl", details)
    write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/development_runs/current_clean_candidate_coverability_v1",
    )
    args = parser.parse_args()
    report = run(args.output_dir, args.text_sample_size, args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
