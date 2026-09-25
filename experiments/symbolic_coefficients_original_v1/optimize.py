"""Deterministic DEV-only coordinate search for original symbolic coefficients."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.symbolic_coefficients_original_v1.original_symbolic_features import (
    ORIGINAL_PROOF_COEFFICIENTS,
    ORIGINAL_TEXT_COEFFICIENTS,
    ProofCoefficients,
    TextCoefficients,
    coefficient_dict,
    score_proof_features,
    score_text_features,
)
from experiments.symbolic_coefficients_original_v1.prepare_dev_cache import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256,
    validate_cache,
    write_json,
    write_jsonl,
)


SEED = 42
IMPROVEMENT_EPSILON = 1e-12


def _grid(stop: int, step: int, scale: int = 1000) -> tuple[float, ...]:
    return tuple(value / scale for value in range(0, stop + step, step))


TEXT_GRIDS: dict[str, tuple[float, ...]] = {
    "question_title": _grid(60, 5),
    "question_overlap": _grid(40, 5),
    "cross_page": _grid(40, 5),
    "bridge_overlap": _grid(50, 5),
    "kg_connectivity": _grid(60, 5),
    "comparison": _grid(40, 5),
    "size_penalty_coefficient": _grid(20, 2),
    "duplicate_page_penalty": _grid(20, 2),
}
PROOF_GRIDS: dict[str, tuple[float, ...]] = {
    "proof_bonus": _grid(300, 10),
    "query_coverage_bonus": _grid(60, 5),
    "schema_mix_bonus": _grid(40, 5),
    "compact_weight": _grid(700, 50),
    "extra_unit_penalty": _grid(20, 2),
}


def original_search_parameters(domain: str) -> dict[str, float]:
    if domain == "text":
        values = coefficient_dict(ORIGINAL_TEXT_COEFFICIENTS)
        return {
            "question_title": values["question_title"],
            "question_overlap": values["question_overlap"],
            "cross_page": values["cross_page"],
            "bridge_overlap": values["bridge_overlap"],
            "kg_connectivity": values["kg_connectivity"],
            "comparison": values["comparison"],
            "size_penalty_coefficient": values["size_beyond_four_penalty"],
            "duplicate_page_penalty": values["duplicate_page_penalty"],
        }
    values = coefficient_dict(ORIGINAL_PROOF_COEFFICIENTS)
    return {key: values[key] for key in PROOF_GRIDS}


def expand_search_parameters(domain: str, values: Mapping[str, float]) -> TextCoefficients | ProofCoefficients:
    if domain == "text":
        size = float(values["size_penalty_coefficient"])
        return TextCoefficients(
            question_title=float(values["question_title"]),
            question_overlap=float(values["question_overlap"]),
            cross_page=float(values["cross_page"]),
            bridge_overlap=float(values["bridge_overlap"]),
            kg_connectivity=float(values["kg_connectivity"]),
            comparison=float(values["comparison"]),
            size_three_penalty=0.2 * size,
            size_four_penalty=0.4 * size,
            size_beyond_four_penalty=size,
            duplicate_page_penalty=float(values["duplicate_page_penalty"]),
        )
    return replace(ORIGINAL_PROOF_COEFFICIENTS, **{key: float(value) for key, value in values.items()})


def _support_f1(predicted: Sequence[str], alternatives: Sequence[Sequence[str]]) -> float:
    predicted_set = set(predicted)
    best = 0.0
    for gold in alternatives:
        gold_set = set(gold)
        overlap = len(predicted_set & gold_set)
        precision = overlap / len(predicted_set) if predicted_set else 0.0
        recall = overlap / len(gold_set) if gold_set else 0.0
        value = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        best = max(best, value)
    return best


def load_domain_cache(cache_path: Path, domain: str) -> list[dict[str, Any]]:
    if domain not in {"text", "ontology"}:
        raise ValueError("domain must be text or ontology")
    rows = [row for row in read_jsonl(cache_path) if row.get("domain") == domain]
    if not rows or any(row.get("split") != "dev" for row in rows):
        raise ValueError(f"No valid DEV cache rows for {domain}")
    return rows


def evaluate_coefficients(rows: Sequence[Mapping[str, Any]], domain: str, coefficients: TextCoefficients | ProofCoefficients) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = {}
    for row in rows:
        scored = []
        for candidate in row["candidates"]:
            adjustment = (
                score_text_features(candidate["features"], coefficients)
                if domain == "text"
                else score_proof_features(candidate["features"], coefficients)
            )
            scored.append((float(candidate["cross_encoder_logit"]) + adjustment, str(candidate["candidate_identity"]), candidate))
        top = min(scored, key=lambda item: (-item[0], item[1]))[2]
        by_dataset.setdefault(str(row["dataset"]), []).append(_support_f1(top["subgraph_units"], row["gold_explanations"]))
    dataset_f1 = {dataset: sum(values) / len(values) for dataset, values in sorted(by_dataset.items())}
    return {
        "objective": sum(dataset_f1.values()) / len(dataset_f1),
        "dataset_support_f1_at_1": dataset_f1,
        "datasets": len(dataset_f1),
        "examples": len(rows),
    }


def _distance(values: Mapping[str, float], originals: Mapping[str, float]) -> float:
    return sum(abs(values[key] - original) / original for key, original in originals.items())


def _selection_key(
    metrics: Mapping[str, Any],
    values: Mapping[str, float],
    baseline: Mapping[str, Any],
    originals: Mapping[str, float],
    order: Sequence[str],
) -> tuple[Any, ...]:
    deltas = [
        float(metrics["dataset_support_f1_at_1"][dataset])
        - float(baseline["dataset_support_f1_at_1"][dataset])
        for dataset in sorted(baseline["dataset_support_f1_at_1"])
    ]
    return (
        -float(metrics["objective"]),
        sum(delta < -IMPROVEMENT_EPSILON for delta in deltas),
        -min(deltas),
        _distance(values, originals),
        tuple(values[key] for key in order),
    )


def coordinate_search(rows: Sequence[Mapping[str, Any]], domain: str) -> dict[str, Any]:
    grids = TEXT_GRIDS if domain == "text" else PROOF_GRIDS
    originals = original_search_parameters(domain)
    parameter_order = list(grids)
    baseline = evaluate_coefficients(rows, domain, expand_search_parameters(domain, originals))
    doubled = {
        key: min(grids[key], key=lambda candidate: (abs(candidate - 2.0 * value), candidate))
        for key, value in originals.items()
    }
    starts = (
        ("original", originals),
        ("zero_symbolic", {key: 0.0 for key in parameter_order}),
        ("doubled_original_clipped", doubled),
    )
    trace: list[dict[str, Any]] = []
    completed: list[tuple[dict[str, Any], dict[str, float], str, int]] = []
    for start_name, start_values in starts:
        values = dict(start_values)
        current_metrics = evaluate_coefficients(rows, domain, expand_search_parameters(domain, values))
        passes = 0
        for pass_index in range(1, 11):
            passes = pass_index
            changed = False
            for parameter in parameter_order:
                trials = []
                for value in grids[parameter]:
                    trial_values = {**values, parameter: value}
                    metrics = evaluate_coefficients(rows, domain, expand_search_parameters(domain, trial_values))
                    trials.append((metrics, trial_values))
                    trace.append({
                        "start": start_name,
                        "pass": pass_index,
                        "parameter": parameter,
                        "value": value,
                        "parameters": trial_values,
                        "metrics": metrics,
                    })
                best_metrics, best_values = min(
                    trials,
                    key=lambda item: _selection_key(item[0], item[1], baseline, originals, parameter_order),
                )
                changed |= best_values[parameter] != values[parameter]
                values, current_metrics = best_values, best_metrics
            if not changed:
                break
        completed.append((current_metrics, values, start_name, passes))
    current_metrics, values, winning_start, pass_index = min(
        completed,
        key=lambda item: _selection_key(item[0], item[1], baseline, originals, parameter_order),
    )
    improved = current_metrics["objective"] > baseline["objective"] + IMPROVEMENT_EPSILON
    if not improved:
        values = originals
        current_metrics = baseline
    return {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_optimization",
        "domain": domain,
        "split": "dev",
        "seed": SEED,
        "objective_definition": "unweighted mean of dataset-level mean DEV Support F1 at k=1",
        "parameter_order": parameter_order,
        "grids": {key: list(value) for key, value in grids.items()},
        "starts": [name for name, _ in starts],
        "maximum_passes_per_start": 10,
        "winning_start": winning_start,
        "original_search_parameters": originals,
        "original_coefficients": coefficient_dict(expand_search_parameters(domain, originals)),
        "original_metrics": baseline,
        "selected_search_parameters": values,
        "selected_coefficients": coefficient_dict(expand_search_parameters(domain, values)),
        "selected_metrics": current_metrics,
        "improved": improved,
        "retained_original_no_improvement": not improved,
        "winning_start_passes": pass_index,
        "trace": trace,
    }


def run_search(cache: Path, cache_manifest: Path, domain: str, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    validate_cache(cache, cache_manifest)
    result = coordinate_search(load_domain_cache(cache, domain), domain)
    result["input_cache_sha256"] = sha256(cache)
    write_json(output_dir / f"{domain}_optimization.json", result)
    write_jsonl(output_dir / f"{domain}_optimization_trace.jsonl", result.pop("trace"))
    # Restore the in-memory return while keeping the primary JSON compact.
    result["trace_sha256"] = sha256(output_dir / f"{domain}_optimization_trace.jsonl")
    write_json(output_dir / f"{domain}_optimization_manifest.json", {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_optimization_manifest",
        "domain": domain,
        "split": "dev",
        "test_accessed": False,
        "cache_sha256": sha256(cache),
        "result_sha256": sha256(output_dir / f"{domain}_optimization.json"),
        "trace_sha256": result["trace_sha256"],
        "code_sha256": {
            "optimize.py": sha256(Path(__file__)),
            "original_symbolic_features.py": sha256(Path(__file__).with_name("original_symbolic_features.py")),
            "PROTOCOL.md": sha256(Path(__file__).with_name("PROTOCOL.md")),
        },
    })
    return result


def compare(text_result: Path, ontology_result: Path, output: Path) -> dict[str, Any]:
    text = read_json(text_result)
    ontology = read_json(ontology_result)
    if text.get("domain") != "text" or ontology.get("domain") != "ontology":
        raise ValueError("Comparison requires independent text and ontology results")
    report = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_comparison",
        "split": "dev",
        "test_accessed": False,
        "domains": {
            domain: {
                "original_objective": value["original_metrics"]["objective"],
                "selected_objective": value["selected_metrics"]["objective"],
                "delta": value["selected_metrics"]["objective"] - value["original_metrics"]["objective"],
                "improved": value["improved"],
                "retained_original_no_improvement": value["retained_original_no_improvement"],
            }
            for domain, value in (("text", text), ("ontology", ontology))
        },
        "input_sha256": {"text": sha256(text_result), "ontology": sha256(ontology_result)},
    }
    write_json(output, report)
    return report


def freeze(text_result: Path, ontology_result: Path, cache_manifest: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    text, ontology, cache = read_json(text_result), read_json(ontology_result), read_json(cache_manifest)
    if text.get("domain") != "text" or ontology.get("domain") != "ontology":
        raise ValueError("Freeze requires independent text and ontology results")
    selected = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_selected",
        "status": "frozen_after_dev_selection",
        "split": "dev",
        "seed": SEED,
        "text": text["selected_coefficients"],
        "ontology": ontology["selected_coefficients"],
    }
    write_json(output_dir / "selected_coefficients.json", selected)
    text_file = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_text_selected",
        "status": "frozen_after_dev_selection",
        "split": "dev",
        "coefficients": text["selected_coefficients"],
    }
    proof_file = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_proof_selected",
        "status": "frozen_after_dev_selection",
        "split": "dev",
        "coefficients": ontology["selected_coefficients"],
    }
    write_json(output_dir / "text_chain_coefficients.json", text_file)
    write_json(output_dir / "proof_coefficients.json", proof_file)
    manifest = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_freeze_manifest",
        "status": "frozen_after_dev_selection",
        "test_accessed": False,
        "objective": "unweighted dataset-macro DEV Support F1 at k=1",
        "inputs_sha256": {
            "text_optimization": sha256(text_result),
            "ontology_optimization": sha256(ontology_result),
            "cache_manifest": sha256(cache_manifest),
            "cache": cache["cache_sha256"],
        },
        "selected_coefficients_sha256": sha256(output_dir / "selected_coefficients.json"),
        "coefficient_files_sha256": {
            "text_chain_coefficients.json": sha256(output_dir / "text_chain_coefficients.json"),
            "proof_coefficients.json": sha256(output_dir / "proof_coefficients.json"),
        },
    }
    write_json(output_dir / "freeze_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--cache", type=Path, required=True)
    search.add_argument("--cache-manifest", type=Path, required=True)
    search.add_argument("--domain", choices=("text", "ontology"), required=True)
    search.add_argument("--output-dir", type=Path, required=True)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--text-result", type=Path, required=True)
    comparison.add_argument("--ontology-result", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    freezing = subparsers.add_parser("freeze")
    freezing.add_argument("--text-result", type=Path, required=True)
    freezing.add_argument("--ontology-result", type=Path, required=True)
    freezing.add_argument("--cache-manifest", type=Path, required=True)
    freezing.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "search":
        result = run_search(args.cache, args.cache_manifest, args.domain, args.output_dir)
    elif args.command == "compare":
        result = compare(args.text_result, args.ontology_result, args.output)
    else:
        result = freeze(args.text_result, args.ontology_result, args.cache_manifest, args.output_dir)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
