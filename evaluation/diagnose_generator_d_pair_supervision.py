"""Audit trainer-facing supervision for frozen Generator D top-2 covers.

Development only. Generator D is rerun in memory solely to recover candidate
contents, and every candidate list must match the digest saved by the frozen
query-local closure experiment before gold is accessed.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import get_gold_explanations, parse_owl_context, set_scores
from data_processing.build_2wiki_subgraph_dataset import (
    flatten_context as flatten_2wiki,
    generate_candidates as generate_2wiki,
    get_gold_support_units as gold_2wiki,
)
from data_processing.build_hotpot_subgraph_dataset import (
    flatten_context as flatten_hotpot,
    generate_candidates as generate_hotpot,
    get_gold_support_units as gold_hotpot,
)
from data_processing.evidence_graph_candidates import query_local_candidate_closure
from data_processing.retrieval_contracts import clean_text_retrieval_input, sha256_file, write_json
from data_processing.text_kg_constructor import KGConstructionConfig
from evaluation.compare_query_local_candidate_closure import D_CONFIG, _unit_candidates
from evaluation.compare_unified_evidence_graph_candidates import (
    DISPLAY_NAMES,
    _canonical,
    _digest,
    build_ontology_evidence_graph,
    build_text_evidence_graph,
)
from evaluation.validate_gold_free_candidate_pools import ONTO, ROOT, TEXT, onto_rows, text_rows
FROZEN_DIR = ROOT / "outputs/development_runs/query_local_candidate_closure_v1"
DATASET_ORDER = ["2WikiMultiHopQA", "HotpotQA", *ONTO]
TRAINER_PATH = ROOT / "training/train_gnn_subgraph_retriever.py"


def _load_target_functions() -> tuple[Any, Any]:
    """Load only the active target functions, avoiding trainer dependencies."""
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"), filename=str(TRAINER_PATH))
    wanted = {"ranking_target", "binary_label_from_target"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    if {node.name for node in definitions} != wanted:
        raise AssertionError("Could not locate active trainer target definitions")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {}
    exec(compile(ast.fix_missing_locations(module), str(TRAINER_PATH), "exec"), namespace)
    return namespace["ranking_target"], namespace["binary_label_from_target"]


ranking_target, binary_label_from_target = _load_target_functions()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(candidate: Sequence[str], golds: Sequence[Sequence[str]]) -> dict[str, Any]:
    scores = set_scores(list(candidate), [list(gold) for gold in golds])
    target = ranking_target(scores)
    return {
        "label": binary_label_from_target(target),
        "rank_target": target,
        "best_set_f1_to_gold": float(scores["best_set_f1_to_gold"]),
        "candidate_size": len(set(candidate)),
    }


def _covering_pairs(
    candidates: Sequence[Sequence[str]], golds: Sequence[Sequence[str]]
) -> list[tuple[int, int, int]]:
    pairs: set[tuple[int, int, int]] = set()
    candidate_sets = [set(candidate) for candidate in candidates]
    for gold_index, gold in enumerate(golds):
        gold_set = set(gold)
        if not gold_set:
            continue
        masks = []
        bits = {unit: 1 << index for index, unit in enumerate(sorted(gold_set))}
        full = (1 << len(bits)) - 1
        for candidate in candidate_sets:
            masks.append(sum(bits.get(unit, 0) for unit in candidate))
        useful = [index for index, mask in enumerate(masks) if mask]
        for offset, left in enumerate(useful):
            if masks[left] == full:
                continue
            for right in useful[offset + 1 :]:
                if masks[right] != full and masks[left] | masks[right] == full:
                    pairs.add((left, right, gold_index))
    return sorted(pairs)


def _pair_detail(
    example_id: str,
    candidates: Sequence[Sequence[str]],
    golds: Sequence[Sequence[str]],
    frozen_row: Mapping[str, Any],
) -> dict[str, Any] | None:
    at_1 = any(set(gold) <= set(candidate) for gold in golds for candidate in candidates if gold)
    pairs = _covering_pairs(candidates, golds)
    at_2 = at_1 or bool(pairs)
    if at_1 != bool(frozen_row["closure_oracle_support_coverage_at_1"]):
        raise AssertionError(f"@1 mismatch for {example_id}")
    if at_2 != bool(frozen_row["closure_oracle_support_coverage_at_2"]):
        raise AssertionError(f"@2 mismatch for {example_id}")
    if at_1 or not pairs:
        return None

    target_cache = {index: _target(candidate, golds) for index, candidate in enumerate(candidates)}
    # Canonical candidate order is size then unit tuple. Select the first
    # cardinality-minimal pair deterministically; do not optimize by target.
    left, right, gold_index = pairs[0]
    selected_gold = set(golds[gold_index])
    union = set(candidates[left]) | set(candidates[right])
    intersection = len(union & selected_gold)
    precision = intersection / len(union)
    recall = intersection / len(selected_gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    all_pairs_both_nonzero = [
        target_cache[i]["rank_target"] > 0 and target_cache[j]["rank_target"] > 0
        for i, j, _ in pairs
    ]
    return {
        "example_id": example_id,
        "selected_gold_index": gold_index,
        "covering_pair_count": len(pairs),
        "at_least_one_pair_both_nonzero": any(all_pairs_both_nonzero),
        "all_pairs_have_both_nonzero": all(all_pairs_both_nonzero),
        "candidate_1": {
            "candidate_index": left,
            "units": list(candidates[left]),
            **target_cache[left],
        },
        "candidate_2": {
            "candidate_index": right,
            "units": list(candidates[right]),
            **target_cache[right],
        },
        "union": {
            "size": len(union),
            "selected_gold_size": len(selected_gold),
            "precision_to_covered_gold": precision,
            "recall_to_covered_gold": recall,
            "f1_to_covered_gold": f1,
            "best_set_f1_to_any_gold": float(set_scores(list(union), [list(g) for g in golds])["best_set_f1_to_gold"]),
        },
    }


def _distribution(values: Iterable[float]) -> dict[str, int]:
    return {
        format(value, ".6g"): count
        for value, count in sorted(collections.Counter(round(float(v), 12) for v in values).items())
    }


def _summary(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    members = [detail[key] for detail in details for key in ("candidate_1", "candidate_2")]
    targets = [float(member["rank_target"]) for member in members]
    labels = [int(member["label"]) for member in members]
    count = len(details)
    pct = lambda numerator, denominator: 100.0 * numerator / denominator if denominator else 0.0
    both = sum(bool(detail["at_least_one_pair_both_nonzero"]) for detail in details)
    return {
        "at_1_false_at_2_true_examples": count,
        "selected_pair_members": len(members),
        "selected_pair_member_rank_target": {
            "mean": statistics.fmean(targets) if targets else None,
            "median": statistics.median(targets) if targets else None,
            "minimum": min(targets) if targets else None,
            "percentage_gt_0": pct(sum(value > 0 for value in targets), len(targets)),
            "percentage_label_1": pct(sum(labels), len(labels)),
            "distribution": _distribution(targets),
        },
        "percentage_examples_with_at_least_one_pair_both_nonzero": pct(both, count),
        "percentage_examples_where_one_or_both_necessary_candidates_zero": pct(count - both, count),
        "near_zero_diagnostics": {
            "percentage_members_rank_target_le_0_05": pct(sum(0 < value <= 0.05 for value in targets), len(targets)),
            "percentage_members_rank_target_le_0_10": pct(sum(0 < value <= 0.10 for value in targets), len(targets)),
        },
        "selected_pair_union_overlap_distribution": {
            "precision_to_covered_gold": _distribution(detail["union"]["precision_to_covered_gold"] for detail in details),
            "recall_to_covered_gold": _distribution(detail["union"]["recall_to_covered_gold"] for detail in details),
            "f1_to_covered_gold": _distribution(detail["union"]["f1_to_covered_gold"] for detail in details),
            "best_set_f1_to_any_gold": _distribution(detail["union"]["best_set_f1_to_any_gold"] for detail in details),
        },
    }


def _audit_dataset(name: str, frozen_rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    if name in TEXT:
        examples = text_rows(name, 200, 42)
        generator = generate_2wiki if name == "2WikiMultiHopQA" else generate_hotpot
        flattener = flatten_2wiki if name == "2WikiMultiHopQA" else flatten_hotpot
        gold_fn = gold_2wiki if name == "2WikiMultiHopQA" else gold_hotpot
        cfg = {"2WikiMultiHopQA": (30, 4), "HotpotQA": (30, 3)}[name]
        kg = KGConstructionConfig(backend="context_only", max_triples=64)
        for index, example in enumerate(examples):
            clean = clean_text_retrieval_input(example)
            generated = generator(
                retrieval_example=clean,
                split_name="query_local_closure_development",
                max_sentences_per_example=cfg[0],
                max_subgraph_size=cfg[1],
                max_candidates_per_question=512,
                kg_config=kg,
                llm_kg_constructor=None,
                kg_cache=None,
                kg_cache_path=None,
                kg_cache_lock=None,
                seed=42 + index,
            )
            graph = build_text_evidence_graph(flattener(clean), str(clean.get("question", "")))
            candidates = _canonical(query_local_candidate_closure(graph, D_CONFIG).candidates)
            example_id = generated["example_id"]
            frozen = frozen_rows[example_id]
            if _digest(candidates) != frozen["closure_candidate_digest"]:
                raise AssertionError(f"Frozen candidate digest mismatch for {example_id}")
            # Gold is first accessed only after digest validation above.
            golds = [gold_fn(example, generated["sent_lookup"])]
            detail = _pair_detail(example_id, candidates, golds, frozen)
            if detail:
                details.append(detail)
        return details

    for group_index, qa_index, item, qa in onto_rows(name):
        question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
        sparql = str(qa.get("SPARQL Query") or "")
        graph = build_ontology_evidence_graph(parse_owl_context(str(item.get("OWL Context") or "")), question, sparql)
        candidates = _canonical(query_local_candidate_closure(graph, D_CONFIG).candidates)
        example_id = f"{name}__g{group_index}__q{qa_index}"
        frozen = frozen_rows[example_id]
        if _digest(candidates) != frozen["closure_candidate_digest"]:
            raise AssertionError(f"Frozen candidate digest mismatch for {example_id}")
        # Gold is first accessed only after digest validation above.
        golds = get_gold_explanations(qa)
        detail = _pair_detail(example_id, candidates, golds, frozen)
        if detail:
            details.append(detail)
    return details


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {output_dir}")
    frozen_report_path = FROZEN_DIR / "comparison.json"
    frozen_report = json.loads(frozen_report_path.read_text(encoding="utf-8"))
    if frozen_report.get("experiment") != "D: query-local candidate closure + progressive expansion":
        raise AssertionError("Unexpected frozen Generator D experiment")
    for gate in ("test_data_used", "training_run", "adaptive_k_run", "answer_generation_run"):
        if frozen_report.get(gate):
            raise AssertionError(f"Frozen run violates gate: {gate}")
    if not frozen_report.get("all_invariance_checks_passed"):
        raise AssertionError("Frozen D invariance checks did not pass")
    for metadata in frozen_report["implementation_files"].values():
        if _file_digest(ROOT / metadata["path"]) != metadata["sha256"]:
            raise AssertionError(f"Implementation hash drift: {metadata['path']}")

    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "generator_d_pair_supervision_diagnostic_v1",
        "development_only": True,
        "frozen_generator_d_report": str(frozen_report_path.relative_to(ROOT)),
        "frozen_generator_d_report_sha256": sha256_file(frozen_report_path),
        "candidate_recovery": "in-memory deterministic D rerun; exact per-example saved digest required before gold access",
        "pair_selection": "first covering pair in canonical candidate order (size then unit tuple); no target optimization",
        "trainer_target_definition": "ranking_target and binary_label_from_target AST-loaded unchanged from active training/train_gnn_subgraph_retriever.py",
        "trainer_source_sha256": sha256_file(TRAINER_PATH),
        "test_data_used": False,
        "training_run": False,
        "parameters_tuned": False,
        "adaptive_k_run": False,
        "answer_generation_run": False,
        "datasets": {},
    }
    for name in DATASET_ORDER:
        frozen_rows = {
            row["example_id"]: row
            for row in _read_jsonl(FROZEN_DIR / name / "details.jsonl")
        }
        details = _audit_dataset(name, frozen_rows)
        display_name = DISPLAY_NAMES[name]
        dataset_dir = output_dir / name
        dataset_dir.mkdir()
        with (dataset_dir / "covering_pairs.jsonl").open("x", encoding="utf-8") as handle:
            for detail in details:
                handle.write(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n")
        report["datasets"][display_name] = _summary(details)
    write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/development_runs/generator_d_pair_supervision_v1",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
