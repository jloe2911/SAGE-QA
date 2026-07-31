import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


HEADLINE_GNN_METRICS = (
    "hit@1",
    "exact_hit@1",
    "best_set_f1@1",
    "hit@3",
    "exact_hit@3",
    "best_set_f1@3",
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def resolve_retrieval_path(path: Path, split: str) -> Path:
    if path.is_file():
        return path
    candidate = path / f"{split}_subgraph_retrieval.jsonl"
    if not candidate.exists():
        raise FileNotFoundError(f"Retrieval split not found: {candidate}")
    return candidate


def load_metadata(retrieval_path: Path) -> Tuple[Path | None, Dict[str, Any]]:
    metadata_path = retrieval_path.parent / "metadata.json"
    if not metadata_path.exists():
        return None, {}
    with metadata_path.open("r", encoding="utf-8-sig") as handle:
        return metadata_path, json.load(handle)


def summarize_retrieval(
    retrieval_path: Path,
    split: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    metadata_path, metadata = load_metadata(retrieval_path)
    examples: Dict[str, Dict[str, Any]] = {}
    construction_methods = Counter()
    total_candidate_size = 0
    rows = 0

    with retrieval_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {retrieval_path} line {line_number}: {exc}"
                ) from exc

            example_id = str(row.get("example_id") or "")
            if not example_id:
                raise ValueError(
                    f"Missing example_id in {retrieval_path} line {line_number}"
                )

            state = examples.setdefault(
                example_id,
                {
                    "candidate_count": 0,
                    "positive_available": 0.0,
                    "exact_available": 0.0,
                    "contains_available": 0.0,
                    "oracle_best_f1": 0.0,
                    "oracle_best_jaccard": 0.0,
                    "gold_kg_coverage": None,
                    "gold_context_coverage": None,
                    "construction_method": "",
                    "uses_legacy_support_annotations": 0.0,
                },
            )
            state["candidate_count"] += 1
            state["positive_available"] = max(
                state["positive_available"], float(bool(row.get("label", 0)))
            )
            state["exact_available"] = max(
                state["exact_available"],
                float(bool(row.get("exact_match_any_gold", False))),
            )
            state["contains_available"] = max(
                state["contains_available"],
                float(bool(row.get("contains_any_gold_explanation", False))),
            )
            state["oracle_best_f1"] = max(
                state["oracle_best_f1"],
                float(row.get("best_set_f1_to_gold", row.get("rank_target", 0.0))),
            )
            state["oracle_best_jaccard"] = max(
                state["oracle_best_jaccard"],
                float(row.get("best_jaccard_to_gold", 0.0)),
            )

            for coverage_field in ("gold_kg_coverage", "gold_context_coverage"):
                if coverage_field in row:
                    state[coverage_field] = float(row[coverage_field])

            method = str(row.get("kg_construction_method") or "")
            if method:
                state["construction_method"] = method
            if row.get("raw_supporting_facts") or row.get("gold_support_units"):
                state["uses_legacy_support_annotations"] = 1.0

            rows += 1
            total_candidate_size += int(
                row.get("subgraph_size", len(row.get("subgraph_units", [])))
            )

    for state in examples.values():
        method = state["construction_method"]
        if method:
            construction_methods[method] += 1

    observed_examples = len(examples)
    expected_value = metadata.get(f"{split}_examples")
    expected_examples = (
        int(expected_value)
        if isinstance(expected_value, (int, float)) and expected_value >= 0
        else observed_examples
    )
    denominator = max(expected_examples, observed_examples, 1)

    def total(metric: str) -> float:
        return sum(float(state[metric]) for state in examples.values())

    def coverage_mean(field: str) -> float | None:
        values = [
            float(state[field])
            for state in examples.values()
            if state[field] is not None
        ]
        if not values:
            return None
        # Expected-but-unobserved examples correspond to zero emitted candidates.
        return sum(values) / denominator

    summary = {
        "retrieval_path": str(retrieval_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "schema_version": metadata.get("schema_version"),
        "candidate_composer": metadata.get("candidate_composer"),
        "row_materializer": metadata.get("row_materializer"),
        "kg_construction_backend": metadata.get("kg_construction_backend"),
        "rows": rows,
        "observed_examples": observed_examples,
        "expected_examples": expected_examples,
        "inferred_zero_candidate_examples": max(
            expected_examples - observed_examples, 0
        ),
        "mean_candidates_per_expected_example": rows / denominator,
        "mean_candidate_path_size": (total_candidate_size / rows if rows else 0.0),
        "positive_candidate_rate": total("positive_available") / denominator,
        "exact_candidate_rate": total("exact_available") / denominator,
        "contains_gold_candidate_rate": total("contains_available") / denominator,
        "mean_oracle_best_f1": total("oracle_best_f1") / denominator,
        "mean_oracle_best_jaccard": total("oracle_best_jaccard") / denominator,
        "mean_gold_kg_coverage": coverage_mean("gold_kg_coverage"),
        "mean_gold_context_coverage": coverage_mean("gold_context_coverage"),
        "legacy_support_annotation_rate": (
            total("uses_legacy_support_annotations") / denominator
        ),
        "kg_construction_methods": dict(sorted(construction_methods.items())),
    }
    return summary, examples


def paired_example_deltas(
    baseline: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    shared_ids = sorted(set(baseline) & set(current))
    metrics = (
        "candidate_count",
        "positive_available",
        "exact_available",
        "contains_available",
        "oracle_best_f1",
        "oracle_best_jaccard",
        "gold_kg_coverage",
        "gold_context_coverage",
    )
    deltas: Dict[str, float | None] = {}
    for metric in metrics:
        values = []
        for example_id in shared_ids:
            old = baseline[example_id].get(metric)
            new = current[example_id].get(metric)
            if old is not None and new is not None:
                values.append(float(new) - float(old))
        deltas[f"mean_delta_{metric}"] = mean(values) if values else None

    return {
        "shared_examples": len(shared_ids),
        "baseline_only_examples": len(set(baseline) - set(current)),
        "current_only_examples": len(set(current) - set(baseline)),
        **deltas,
    }


def load_gnn_metrics(path: Path | None) -> Tuple[Path | None, Dict[str, float]]:
    if path is None:
        return None, {}
    metrics_path = path / "test_metrics.json" if path.is_dir() else path
    if not metrics_path.exists():
        raise FileNotFoundError(f"GNN test metrics not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    metrics = {
        key: float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    return metrics_path, metrics


def numeric_deltas(
    baseline: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, float]:
    return {
        key: float(current[key]) - float(baseline[key])
        for key in sorted(set(baseline) & set(current))
        if isinstance(baseline[key], (int, float))
        and not isinstance(baseline[key], bool)
        and isinstance(current[key], (int, float))
        and not isinstance(current[key], bool)
    }


def leakage_warnings(label: str, summary: Dict[str, Any]) -> list[str]:
    methods = summary.get("kg_construction_methods", {})
    provided = {
        method: count
        for method, count in methods.items()
        if "provided" in method.casefold()
    }
    backend = str(summary.get("kg_construction_backend") or "")
    legacy_support_rate = float(summary.get("legacy_support_annotation_rate") or 0.0)
    warnings = []
    if provided or "provided" in backend.casefold():
        warnings.append(
            f"{label} contains gold-assisted KG construction "
            f"(backend={backend!r}, methods={provided}). Treat its metrics as "
            "legacy rather than leakage-free performance."
        )
    if legacy_support_rate > 0:
        warnings.append(
            f"{label} uses sentence-level supporting-fact annotations for "
            f"{legacy_support_rate:.1%} of observed examples. This is not the "
            "current triple-level 2Wiki task."
        )
    return warnings


def comparison_rows(report: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for scope, baseline, current in (
        (
            "candidate_summary",
            report["baseline"]["candidate_summary"],
            report["current"]["candidate_summary"],
        ),
        (
            "gnn_test",
            report["baseline"]["gnn_metrics"],
            report["current"]["gnn_metrics"],
        ),
    ):
        for metric in sorted(set(baseline) & set(current)):
            old = baseline[metric]
            new = current[metric]
            if (
                isinstance(old, (int, float))
                and not isinstance(old, bool)
                and isinstance(new, (int, float))
                and not isinstance(new, bool)
            ):
                rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "baseline": old,
                        "current": new,
                        "delta": float(new) - float(old),
                    }
                )
    return rows


def write_report(
    output_dir: Path,
    report: Dict[str, Any],
    overwrite: bool,
) -> Tuple[Path, Path]:
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing comparison files: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scope", "metric", "baseline", "current", "delta"),
        )
        writer.writeheader()
        writer.writerows(comparison_rows(report))
    return json_path, csv_path


def print_report(report: Dict[str, Any]) -> None:
    print("\nCandidate construction and oracle ceiling")
    print(f"{'metric':38} {'baseline':>12} {'current':>12} {'delta':>12}")
    candidate_metrics = (
        "observed_examples",
        "inferred_zero_candidate_examples",
        "mean_candidates_per_expected_example",
        "mean_candidate_path_size",
        "positive_candidate_rate",
        "exact_candidate_rate",
        "mean_oracle_best_f1",
        "mean_oracle_best_jaccard",
        "mean_gold_kg_coverage",
        "mean_gold_context_coverage",
        "legacy_support_annotation_rate",
    )
    old_summary = report["baseline"]["candidate_summary"]
    new_summary = report["current"]["candidate_summary"]
    for metric in candidate_metrics:
        old = old_summary.get(metric)
        new = new_summary.get(metric)
        if old is None and new is None:
            continue
        delta = (
            float(new) - float(old)
            if isinstance(old, (int, float)) and isinstance(new, (int, float))
            else None
        )
        print(
            f"{metric:38} {format_value(old):>12} "
            f"{format_value(new):>12} {format_value(delta):>12}"
        )

    old_gnn = report["baseline"]["gnn_metrics"]
    new_gnn = report["current"]["gnn_metrics"]
    if old_gnn or new_gnn:
        print("\nGNN test metrics")
        print(f"{'metric':38} {'baseline':>12} {'current':>12} {'delta':>12}")
        for metric in HEADLINE_GNN_METRICS:
            old = old_gnn.get(metric)
            new = new_gnn.get(metric)
            if old is None and new is None:
                continue
            delta = (
                float(new) - float(old) if old is not None and new is not None else None
            )
            print(
                f"{metric:38} {format_value(old):>12} "
                f"{format_value(new):>12} {format_value(delta):>12}"
            )

    paired = report["deltas"]["paired_examples"]
    print(
        "\nPaired examples: "
        f"{paired['shared_examples']} shared, "
        f"{paired['baseline_only_examples']} baseline-only, "
        f"{paired['current_only_examples']} current-only"
    )
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    baseline_path = resolve_retrieval_path(Path(args.baseline_data), args.split)
    current_path = resolve_retrieval_path(Path(args.current_data), args.split)
    baseline_summary, baseline_examples = summarize_retrieval(baseline_path, args.split)
    current_summary, current_examples = summarize_retrieval(current_path, args.split)
    baseline_gnn_path, baseline_gnn = load_gnn_metrics(
        Path(args.baseline_gnn) if args.baseline_gnn else None
    )
    current_gnn_path, current_gnn = load_gnn_metrics(
        Path(args.current_gnn) if args.current_gnn else None
    )

    warnings = [
        *leakage_warnings("baseline", baseline_summary),
        *leakage_warnings("current", current_summary),
    ]
    return {
        "split": args.split,
        "baseline": {
            "label": args.baseline_label,
            "candidate_summary": baseline_summary,
            "gnn_metrics_path": (str(baseline_gnn_path) if baseline_gnn_path else None),
            "gnn_metrics": baseline_gnn,
        },
        "current": {
            "label": args.current_label,
            "candidate_summary": current_summary,
            "gnn_metrics_path": str(current_gnn_path) if current_gnn_path else None,
            "gnn_metrics": current_gnn,
        },
        "deltas": {
            "candidate_summary": numeric_deltas(baseline_summary, current_summary),
            "paired_examples": paired_example_deltas(
                baseline_examples, current_examples
            ),
            "gnn_metrics": numeric_deltas(baseline_gnn, current_gnn),
        },
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate construction, oracle ceilings, and optional GNN "
            "test metrics between two versioned retrieval runs."
        )
    )
    parser.add_argument("--baseline-data", required=True)
    parser.add_argument("--current-data", required=True)
    parser.add_argument("--baseline-gnn")
    parser.add_argument("--current-gnn")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of comparison.json and comparison.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    print_report(report)
    json_path, csv_path = write_report(
        Path(args.output_dir), report, overwrite=args.overwrite
    )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
