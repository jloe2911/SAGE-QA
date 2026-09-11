"""Run the final A/C/D unified candidate-generation experiment on DEV only."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    generate_ontology_candidates,
    get_gold_explanations,
    parse_owl_context,
)
from data_processing.build_2wiki_subgraph_dataset import (  # noqa: E402
    flatten_context as flatten_2wiki,
    generate_candidates as generate_2wiki,
    get_gold_support_units as gold_2wiki,
)
from data_processing.build_hotpot_subgraph_dataset import (  # noqa: E402
    flatten_context as flatten_hotpot,
    generate_candidates as generate_hotpot,
    get_gold_support_units as gold_hotpot,
)
from data_processing.evidence_graph_candidates import (  # noqa: E402
    EvidenceGraph,
    ProgressiveExpansionConfig,
    progressive_connected_supports,
    query_local_candidate_closure,
)
from data_processing.retrieval_contracts import (  # noqa: E402
    BUILDER_VERSION,
    clean_text_retrieval_input,
    git_provenance,
    sha256_file,
    write_json,
)
from data_processing.text_kg_constructor import KGConstructionConfig  # noqa: E402
from evaluation.compare_unified_evidence_graph_candidates import (  # noqa: E402
    DISPLAY_NAMES,
    PROTECTED_CONFIG,
    _canonical,
    _digest,
    _method_summary,
    build_ontology_evidence_graph,
    build_text_evidence_graph,
    post_gold_diagnostics,
)
from evaluation.validate_gold_free_candidate_pools import (  # noqa: E402
    ONTO,
    ONTO_BASE,
    ROOT,
    TEXT,
    TEXT_BASE,
    onto_rows,
    text_rows,
)

SEED = 42
TEXT_SAMPLE_SIZE = 200
D_CONFIG = ProgressiveExpansionConfig(
    max_support_size=6,
    max_candidates=512,
    max_seeds=32,
    max_first_hops_per_seed=6,
    max_neighbors_per_state=12,
    per_branch_width=2,
    protect_query_anchors=True,
)


def _unit_candidates(
    graph: EvidenceGraph, candidates: Iterable[Sequence[int]]
) -> list[tuple[str, ...]]:
    return [tuple(graph.units[index].unit_id for index in candidate) for candidate in candidates]


def _d_failure_class(
    final_candidates: Sequence[Sequence[str]],
    pre_cap_candidates: Sequence[Sequence[str]],
    gold_explanations: Sequence[Sequence[str]],
) -> str | None:
    """Classify union-complete D examples that still lack one complete candidate."""
    final_sets = [set(candidate) for candidate in final_candidates]
    final_union = set().union(*final_sets) if final_sets else set()
    gold_sets = [set(gold) for gold in gold_explanations if gold]
    relevant = [gold for gold in gold_sets if gold <= final_union]
    if not relevant or any(gold <= candidate for gold in relevant for candidate in final_sets):
        return None

    pre_cap_sets = [set(candidate) for candidate in pre_cap_candidates]
    if any(gold <= candidate for gold in relevant for candidate in pre_cap_sets):
        return "candidate_existed_before_final_cap_but_was_removed"
    if min(len(gold) for gold in relevant) >= 3:
        return "required_units_require_size_ge_3"
    if not any(
        len(candidate) == 2 and gold <= candidate for gold in relevant for candidate in pre_cap_sets
    ):
        return "required_units_never_together_in_connected_size_2_candidate"
    return "other"


def _detail(
    *,
    example_id: str,
    current: Sequence[Sequence[str]],
    protected: Sequence[Sequence[str]],
    closure: Sequence[Sequence[str]],
    closure_pre_cap: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    current_reached: Iterable[str],
    protected_reached: Iterable[str],
    closure_reached: Iterable[str],
    current_max_size: int,
    current_seconds: float,
    protected_seconds: float,
    closure_seconds: float,
    protected_result: Any,
    closure_result: Any,
) -> dict[str, Any]:
    diagnostics = {
        "current": post_gold_diagnostics(
            current, gold, reached_units=current_reached, max_support_size=current_max_size
        ),
        "protected": post_gold_diagnostics(
            protected,
            gold,
            reached_units=protected_reached,
            max_support_size=PROTECTED_CONFIG.max_support_size,
        ),
        "closure": post_gold_diagnostics(
            closure,
            gold,
            reached_units=closure_reached,
            max_support_size=D_CONFIG.max_support_size,
        ),
    }
    row: dict[str, Any] = {"example_id": example_id}
    for prefix, candidates, seconds in (
        ("current", current, current_seconds),
        ("protected", protected, protected_seconds),
        ("closure", closure, closure_seconds),
    ):
        row.update(
            {
                f"{prefix}_candidate_count": len(candidates),
                f"{prefix}_candidate_sizes": [len(candidate) for candidate in candidates],
                f"{prefix}_candidate_digest": _digest(candidates),
                f"{prefix}_runtime_seconds": seconds,
            }
        )
        row.update({f"{prefix}_{key}": value for key, value in diagnostics[prefix].items()})
    row.update(
        {
            "protected_anchor_count": len(protected_result.protected_anchor_unit_ids),
            "protected_component_count": protected_result.protected_component_count,
            "protected_generated_state_count": protected_result.generated_state_count,
            "closure_query_anchor_count": len(closure_result.query_anchor_unit_ids),
            "closure_query_anchor_component_count": closure_result.query_anchor_component_count,
            "closure_local_candidate_count_before_cap": closure_result.local_candidate_count_before_cap,
            "closure_retained_local_candidate_count": closure_result.retained_local_candidate_count,
            "closure_progressive_candidate_count": closure_result.progressive_candidate_count,
            "closure_pre_cap_candidate_count": len(closure_result.pre_cap_candidate_indices),
            "closure_generated_state_count": closure_result.generated_state_count,
            "closure_union_complete_but_not_at_1_class": _d_failure_class(
                closure, closure_pre_cap, gold
            ),
        }
    )
    return row


def _invariance_summary(
    examples: int, current: list[str], protected: list[str], closure: list[str]
) -> dict[str, Any]:
    return {
        "examples": examples,
        "A_current_mismatches": current,
        "C_protected_mismatches": protected,
        "D_query_local_closure_mismatches": closure,
    }


def evaluate_text(
    name: str, sample_size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = text_rows(name, sample_size, seed)
    cfg = TEXT_BASE[name]
    generator = generate_2wiki if name == "2WikiMultiHopQA" else generate_hotpot
    flattener = flatten_2wiki if name == "2WikiMultiHopQA" else flatten_hotpot
    gold_fn = gold_2wiki if name == "2WikiMultiHopQA" else gold_hotpot
    kg = KGConstructionConfig(backend="context_only", max_triples=64)
    rows: list[dict[str, Any]] = []
    mismatches: dict[str, list[str]] = {"current": [], "protected": [], "closure": []}
    for index, example in enumerate(examples):
        clean = clean_text_retrieval_input(example)
        started = time.perf_counter()
        generated = generator(
            retrieval_example=clean,
            split_name="query_local_closure_development",
            max_sentences_per_example=cfg["atomic_budget"],
            max_subgraph_size=cfg["max_size"],
            max_candidates_per_question=cfg["candidate_cap"],
            kg_config=kg,
            llm_kg_constructor=None,
            kg_cache=None,
            kg_cache_path=None,
            kg_cache_lock=None,
            seed=seed + index,
        )
        current_seconds = time.perf_counter() - started
        current = _canonical(generated["candidates"])

        started = time.perf_counter()
        graph = build_text_evidence_graph(flattener(clean), str(clean.get("question", "")))
        graph_seconds = time.perf_counter() - started
        started = time.perf_counter()
        protected_result = progressive_connected_supports(graph, PROTECTED_CONFIG)
        protected_seconds = graph_seconds + time.perf_counter() - started
        protected = _canonical(protected_result.candidates)
        started = time.perf_counter()
        closure_result = query_local_candidate_closure(graph, D_CONFIG)
        closure_seconds = graph_seconds + time.perf_counter() - started
        closure = _canonical(closure_result.candidates)
        closure_pre_cap = _canonical(
            _unit_candidates(graph, closure_result.pre_cap_candidate_indices)
        )

        # Hard gold boundary: supports are accessed only after A/C/D are frozen.
        gold = [gold_fn(example, generated["sent_lookup"])]
        rows.append(
            _detail(
                example_id=generated["example_id"],
                current=current,
                protected=protected,
                closure=closure,
                closure_pre_cap=closure_pre_cap,
                gold=gold,
                current_reached=generated["sentence_pool"],
                protected_reached=protected_result.explored_unit_ids,
                closure_reached=closure_result.explored_unit_ids,
                current_max_size=int(cfg["max_size"]),
                current_seconds=current_seconds,
                protected_seconds=protected_seconds,
                closure_seconds=closure_seconds,
                protected_result=protected_result,
                closure_result=closure_result,
            )
        )

        deleted = {
            key: value
            for key, value in example.items()
            if key not in {"answer", "supporting_facts", "evidences"}
        }
        deleted_clean = clean_text_retrieval_input(deleted)
        deleted_current = generator(
            retrieval_example=deleted_clean,
            split_name="query_local_closure_development",
            max_sentences_per_example=cfg["atomic_budget"],
            max_subgraph_size=cfg["max_size"],
            max_candidates_per_question=cfg["candidate_cap"],
            kg_config=kg,
            llm_kg_constructor=None,
            kg_cache=None,
            kg_cache_path=None,
            kg_cache_lock=None,
            seed=seed + index,
        )
        if any(
            generated[field] != deleted_current[field]
            for field in ("sentence_pool", "graph_context_units", "candidates")
        ):
            mismatches["current"].append(generated["example_id"])
        deleted_graph = build_text_evidence_graph(
            flattener(deleted_clean), str(deleted_clean.get("question", ""))
        )
        if (
            protected_result.candidates
            != progressive_connected_supports(deleted_graph, PROTECTED_CONFIG).candidates
        ):
            mismatches["protected"].append(generated["example_id"])
        if (
            closure_result.candidates
            != query_local_candidate_closure(deleted_graph, D_CONFIG).candidates
        ):
            mismatches["closure"].append(generated["example_id"])
    return rows, _invariance_summary(
        len(rows), mismatches["current"], mismatches["protected"], mismatches["closure"]
    )


def evaluate_ontology(name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = ONTO_BASE
    rows: list[dict[str, Any]] = []
    mismatches: dict[str, list[str]] = {"current": [], "protected": [], "closure": []}
    for group_index, qa_index, item, qa in onto_rows(name):
        question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
        sparql = str(qa.get("SPARQL Query") or "")
        owl_context = str(item.get("OWL Context") or "")
        started = time.perf_counter()
        generated = generate_ontology_candidates(
            question=question,
            sparql_query=sparql,
            owl_context=owl_context,
            max_subgraph_size=cfg["max_size"],
            min_subgraph_size=1,
            max_context_units=cfg["atomic_budget"],
            candidate_beam_width=cfg["beam_width"],
            max_candidate_subgraphs=cfg["candidate_cap"],
        )
        current_seconds = time.perf_counter() - started
        current = _canonical(generated["candidate_subgraphs"])

        context_axioms = parse_owl_context(owl_context)
        started = time.perf_counter()
        graph = build_ontology_evidence_graph(context_axioms, question, sparql)
        graph_seconds = time.perf_counter() - started
        started = time.perf_counter()
        protected_result = progressive_connected_supports(graph, PROTECTED_CONFIG)
        protected_seconds = graph_seconds + time.perf_counter() - started
        protected = _canonical(protected_result.candidates)
        started = time.perf_counter()
        closure_result = query_local_candidate_closure(graph, D_CONFIG)
        closure_seconds = graph_seconds + time.perf_counter() - started
        closure = _canonical(closure_result.candidates)
        closure_pre_cap = _canonical(
            _unit_candidates(graph, closure_result.pre_cap_candidate_indices)
        )

        # Hard gold boundary: explanations are accessed only after A/C/D freeze.
        gold = get_gold_explanations(qa)
        example_id = f"{name}__g{group_index}__q{qa_index}"
        rows.append(
            _detail(
                example_id=example_id,
                current=current,
                protected=protected,
                closure=closure,
                closure_pre_cap=closure_pre_cap,
                gold=gold,
                current_reached=generated["candidate_units"],
                protected_reached=protected_result.explored_unit_ids,
                closure_reached=closure_result.explored_unit_ids,
                current_max_size=int(cfg["max_size"]),
                current_seconds=current_seconds,
                protected_seconds=protected_seconds,
                closure_seconds=closure_seconds,
                protected_result=protected_result,
                closure_result=closure_result,
            )
        )

        deleted_generated = generate_ontology_candidates(
            question=question,
            sparql_query=sparql,
            owl_context=owl_context,
            max_subgraph_size=cfg["max_size"],
            min_subgraph_size=1,
            max_context_units=cfg["atomic_budget"],
            candidate_beam_width=cfg["beam_width"],
            max_candidate_subgraphs=cfg["candidate_cap"],
        )
        if any(
            generated[field] != deleted_generated[field]
            for field in ("candidate_units", "candidate_subgraphs", "unit_scores")
        ):
            mismatches["current"].append(example_id)
        deleted_graph = build_ontology_evidence_graph(context_axioms, question, sparql)
        if (
            protected_result.candidates
            != progressive_connected_supports(deleted_graph, PROTECTED_CONFIG).candidates
        ):
            mismatches["protected"].append(example_id)
        if (
            closure_result.candidates
            != query_local_candidate_closure(deleted_graph, D_CONFIG).candidates
        ):
            mismatches["closure"].append(example_id)
    return rows, _invariance_summary(
        len(rows), mismatches["current"], mismatches["protected"], mismatches["closure"]
    )


def _summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    summary = _method_summary(rows, prefix)
    sizes = collections.Counter(size for row in rows for size in row[f"{prefix}_candidate_sizes"])
    total = sum(sizes.values())
    summary["candidate_size_proportions"] = {
        str(size): count / total for size, count in sorted(sizes.items())
    }
    return summary


def _closure_failure_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = collections.Counter(
        str(row["closure_union_complete_but_not_at_1_class"])
        for row in rows
        if row.get("closure_union_complete_but_not_at_1_class")
    )
    eligible = sum(classes.values())
    return {
        "eligible_examples": eligible,
        "counts": dict(sorted(classes.items())),
        "rates_within_eligible": {key: value / eligible for key, value in sorted(classes.items())}
        if eligible
        else {},
        "example_ids": {
            key: [
                str(row["example_id"])
                for row in rows
                if row.get("closure_union_complete_but_not_at_1_class") == key
            ]
            for key in sorted(classes)
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(output_dir: Path, text_sample_size: int, seed: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output_dir}")
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "query_local_candidate_closure_comparison_v1",
        "experiment": "D: query-local candidate closure + progressive expansion",
        "development_only": True,
        "seed": seed,
        "builder_version": BUILDER_VERSION,
        "git": git_provenance(ROOT),
        "implementation_files": {
            "shared_generator": {
                "path": "data_processing/evidence_graph_candidates.py",
                "sha256": sha256_file(ROOT / "data_processing/evidence_graph_candidates.py"),
            },
            "experiment_driver": {
                "path": "evaluation/compare_query_local_candidate_closure.py",
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "common_representation": "EvidenceUnit -> EvidenceGraph -> unified candidate generation -> downstream SAGE-QA",
        "A": "current clean generator",
        "C": "protected query-anchor progressive generator",
        "D": "every query-anchor singleton, connected anchor-neighbor pair, and connected one-step triple before cap; deterministic component/anchor/size round-robin; remaining budget uses C progressive expansion",
        "C_config": vars(PROTECTED_CONFIG),
        "D_config": vars(D_CONFIG),
        "same_D_algorithm_and_config_all_datasets": True,
        "gold_available_during_candidate_generation": False,
        "gold_join_boundary": "after A, C, and D freeze candidates per example",
        "2wiki_evidences_used_for_generation": False,
        "parameter_grid_run": False,
        "test_data_used": False,
        "training_run": False,
        "adaptive_k_run": False,
        "answer_generation_run": False,
        "datasets": {},
        "invariance": {},
    }
    for name in TEXT:
        rows, invariance = evaluate_text(name, text_sample_size, seed)
        display = DISPLAY_NAMES[name]
        report["datasets"][display] = {
            "development_examples": len(rows),
            "source_file": str(TEXT[name].relative_to(ROOT)),
            "source_sha256": sha256_file(TEXT[name]),
            "cohort_policy": f"existing deterministic seed-{seed} sample of official development",
            "A_current_clean_generator": _summary(rows, "current"),
            "C_protected_query_anchors": _summary(rows, "protected"),
            "D_query_local_candidate_closure": _summary(rows, "closure"),
            "D_union_complete_but_not_at_1": _closure_failure_summary(rows),
        }
        report["invariance"][display] = invariance
        _write_jsonl(output_dir / name / "details.jsonl", rows)
    for name in ONTO:
        rows, invariance = evaluate_ontology(name)
        display = DISPLAY_NAMES[name]
        report["datasets"][display] = {
            "development_examples": len(rows),
            "source_file": str(ONTO[name].relative_to(ROOT)),
            "source_sha256": sha256_file(ONTO[name]),
            "cohort_policy": "all labeled QAs in deterministic group-stratified development split",
            "A_current_clean_generator": _summary(rows, "current"),
            "C_protected_query_anchors": _summary(rows, "protected"),
            "D_query_local_candidate_closure": _summary(rows, "closure"),
            "D_union_complete_but_not_at_1": _closure_failure_summary(rows),
        }
        report["invariance"][display] = invariance
        _write_jsonl(output_dir / name / "details.jsonl", rows)
    report["all_invariance_checks_passed"] = all(
        not value["A_current_mismatches"]
        and not value["C_protected_mismatches"]
        and not value["D_query_local_closure_mismatches"]
        for value in report["invariance"].values()
    )
    write_json(output_dir / "comparison.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-sample-size", type=int, default=TEXT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/development_runs/query_local_candidate_closure_v1",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.text_sample_size, args.seed), indent=2))


if __name__ == "__main__":
    main()
