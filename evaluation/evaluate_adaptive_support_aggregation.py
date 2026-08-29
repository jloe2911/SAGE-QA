"""Tune one pooled development tau and evaluate frozen ontology rankings.

This utility is retrieval-only.  It consumes persisted final SAGE-QA ``top5``
details and never invokes candidate generation, a GNN, symbolic reranking, an
answer reader, or an external API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.adaptive_support_aggregation import (  # noqa: E402
    MAX_ADAPTIVE_K,
    adaptive_support_aggregate,
    fixed_k_support_aggregate,
)


CONDITIONS = (
    "FamilyOWL_1hop",
    "FamilyOWL_2hop",
    "OWL2Bench_1hop",
    "OWL2Bench_2hop",
)
FIXED_K_VALUES = (1, 2, 3, 5)
DEFAULT_TAU_GRID = tuple(index / 20 for index in range(20))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_candidates(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = sorted(
        (dict(candidate) for candidate in item.get("top5", []) or []),
        key=lambda candidate: int(candidate.get("rank", 999)),
    )[:MAX_ADAPTIVE_K]
    if not candidates:
        raise ValueError(f"{item.get('example_id', '<unknown>')}: missing final top5")
    for candidate in candidates:
        if "adjusted_score" not in candidate:
            raise ValueError(
                f"{item.get('example_id', '<unknown>')}: candidate lacks adjusted_score"
            )
    return candidates


def best_support_scores(
    predicted: Sequence[Any], gold_explanations: Sequence[Sequence[Any]]
) -> dict[str, float]:
    predicted_set = set(predicted)
    best = {"support_precision": 0.0, "support_recall": 0.0, "support_f1": 0.0}
    for gold in gold_explanations:
        gold_set = set(gold)
        if not gold_set:
            continue
        intersection = len(predicted_set & gold_set)
        precision = intersection / len(predicted_set) if predicted_set else 0.0
        recall = intersection / len(gold_set)
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        if f1 > best["support_f1"]:
            best = {
                "support_precision": precision,
                "support_recall": recall,
                "support_f1": f1,
            }
    return best


def summarize_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty decision list")
    distribution = Counter(int(row["selected_k"]) for row in rows)
    return {
        "examples": len(rows),
        "support_precision": sum(float(row["support_precision"]) for row in rows)
        / len(rows),
        "support_recall": sum(float(row["support_recall"]) for row in rows) / len(rows),
        "support_f1": sum(float(row["support_f1"]) for row in rows) / len(rows),
        "average_selected_k": sum(int(row["selected_k"]) for row in rows) / len(rows),
        "selected_k_distribution": {
            str(k): distribution.get(k, 0) for k in range(1, MAX_ADAPTIVE_K + 1)
        },
        "average_unique_evidence_units": sum(
            int(row["final_support_size"]) for row in rows
        )
        / len(rows),
    }


def evaluate_items(
    items: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    split: str,
    mode: str,
    tau: float | None = None,
    fixed_k: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions = []
    for item in items:
        candidates = final_candidates(item)
        if mode == "adaptive":
            if tau is None:
                raise ValueError("Adaptive evaluation requires tau")
            aggregation = adaptive_support_aggregate(candidates, tau=tau)
        elif mode == "fixed":
            if fixed_k is None:
                raise ValueError("Fixed evaluation requires fixed_k")
            aggregation = fixed_k_support_aggregate(candidates, k=fixed_k)
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

        gold_explanations = item.get("gold_explanations", []) or []
        if not gold_explanations:
            raise ValueError(f"{item.get('example_id')}: support evaluation lacks gold")
        scores = best_support_scores(aggregation["support_units"], gold_explanations)
        decision = {
            "example_id": str(item.get("example_id") or ""),
            "condition": condition,
            "split": split,
            **aggregation,
            **scores,
        }
        decisions.append(decision)
    return summarize_decisions(decisions), decisions


def tune_common_tau(
    dev_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    tau_grid: Sequence[float] = DEFAULT_TAU_GRID,
) -> tuple[float, list[dict[str, Any]]]:
    table = []
    for tau in tau_grid:
        pooled = []
        by_condition = {}
        for condition in CONDITIONS:
            metrics, decisions = evaluate_items(
                dev_by_condition[condition],
                condition=condition,
                split="dev",
                mode="adaptive",
                tau=float(tau),
            )
            by_condition[condition] = metrics
            pooled.extend(decisions)
        table.append(
            {
                "tau": float(tau),
                **summarize_decisions(pooled),
                "by_condition": by_condition,
            }
        )

    selected = max(
        table,
        key=lambda row: (
            float(row["support_f1"]),
            -float(row["average_unique_evidence_units"]),
            -float(row["average_selected_k"]),
            -float(row["tau"]),
        ),
    )
    return float(selected["tau"]), table


def legacy_best_candidate_metrics(
    items: Sequence[Mapping[str, Any]], k: int
) -> dict[str, float]:
    per_example = []
    for item in items:
        candidates = final_candidates(item)[:k]
        gold = item.get("gold_explanations", []) or []
        candidate_scores = [
            best_support_scores(candidate.get("subgraph_units", []) or [], gold)
            for candidate in candidates
        ]
        per_example.append(max(candidate_scores, key=lambda row: row["support_f1"]))
    return {
        "support_precision": sum(row["support_precision"] for row in per_example)
        / len(per_example),
        "support_recall": sum(row["support_recall"] for row in per_example)
        / len(per_example),
        "support_f1": sum(row["support_f1"] for row in per_example) / len(per_example),
    }


def numeric_regression(actual: float, expected: float, tolerance: float = 1e-12) -> dict[str, Any]:
    delta = actual - expected
    return {
        "actual": actual,
        "expected": expected,
        "absolute_delta": abs(delta),
        "matched": abs(delta) <= tolerance,
        "tolerance": tolerance,
    }


def build_regression_report(
    condition: str,
    items: Sequence[Mapping[str, Any]],
    fixed_metrics: Mapping[int, Mapping[str, Any]],
    method_dir: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {"condition": condition}

    manuscript_path = method_dir / "metrics_top3_gpt_4_1_mini.json"
    manuscript = read_json(manuscript_path)["metrics"]
    fixed_k3_checks = {
        metric: numeric_regression(
            float(fixed_metrics[3][metric]), float(manuscript[persisted_key])
        )
        for metric, persisted_key in (
            ("support_precision", "sp_prec"),
            ("support_recall", "sp_recall"),
            ("support_f1", "sp_f1"),
        )
    }
    report["fixed_k3_aggregated_support"] = {
        "source": str(manuscript_path),
        "metric_note": "Persisted sp_* metrics are the deduplicated top-3 support union.",
        **fixed_k3_checks,
    }

    sensitivity_path = method_dir / "test_metrics.json"
    sensitivity = read_json(sensitivity_path)
    legacy = {}
    for k in (1, 3, 5):
        actual = legacy_best_candidate_metrics(items, k)
        legacy[str(k)] = {
            metric: numeric_regression(
                actual[metric], float(sensitivity[f"best_{persisted_name}@{k}"])
            )
            for metric, persisted_name in (
                ("support_precision", "precision"),
                ("support_recall", "recall"),
                ("support_f1", "set_f1"),
            )
        }
    legacy["2"] = {
        "status": "no_persisted_reference",
        "note": "The existing k-sensitivity artifact stores only k=1,3,5.",
    }
    report["legacy_best_candidate_k_sensitivity"] = {
        "source": str(sensitivity_path),
        "metric_note": "These are best-candidate-within-k diagnostics, not aggregated support.",
        "by_k": legacy,
    }
    return report


def compact_tradeoff_summary(
    aggregate: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    summary = {}
    for condition, systems in aggregate.items():
        adaptive = systems["adaptive"]
        summary[condition] = {}
        for baseline in ("fixed_k1", "fixed_k3"):
            fixed = systems[baseline]
            summary[condition][f"adaptive_vs_{baseline}"] = {
                "support_precision_delta": adaptive["support_precision"]
                - fixed["support_precision"],
                "support_recall_delta": adaptive["support_recall"]
                - fixed["support_recall"],
                "support_f1_delta": adaptive["support_f1"] - fixed["support_f1"],
                "average_selected_k_delta": adaptive["average_selected_k"]
                - fixed["average_selected_k"],
                "average_unique_evidence_units_delta": adaptive[
                    "average_unique_evidence_units"
                ]
                - fixed["average_unique_evidence_units"],
            }
    return summary


def pooled_system_metrics(
    aggregate: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, Any]]:
    pooled = {}
    system_names = next(iter(aggregate.values())).keys()
    for system_name in system_names:
        condition_metrics = [systems[system_name] for systems in aggregate.values()]
        total_examples = sum(int(metrics["examples"]) for metrics in condition_metrics)
        distribution = {
            str(k): sum(
                int(metrics["selected_k_distribution"][str(k)])
                for metrics in condition_metrics
            )
            for k in range(1, MAX_ADAPTIVE_K + 1)
        }
        pooled[system_name] = {
            "examples": total_examples,
            **{
                metric_name: sum(
                    float(metrics[metric_name]) * int(metrics["examples"])
                    for metrics in condition_metrics
                )
                / total_examples
                for metric_name in (
                    "support_precision",
                    "support_recall",
                    "support_f1",
                    "average_selected_k",
                    "average_unique_evidence_units",
                )
            },
            "selected_k_distribution": distribution,
        }
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/full_results"))
    parser.add_argument("--method", default="gnn_sageqa_proof")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/adaptive_support_aggregation_v1")
    )
    args = parser.parse_args()

    dev_paths = {
        condition: args.input_root / condition / args.method / "dev_details.json"
        for condition in CONDITIONS
    }
    dev_by_condition = {condition: read_json(path) for condition, path in dev_paths.items()}
    selected_tau, tuning_table = tune_common_tau(dev_by_condition)

    tuning_artifact = {
        "selection_scope": "pooled development support examples only",
        "conditions": list(CONDITIONS),
        "tau_grid": list(DEFAULT_TAU_GRID),
        "policy": {
            "score_field": "adjusted_score",
            "score_scale": "population standard deviation over the available final top-5",
            "epsilon": 1e-12,
            "k_max": MAX_ADAPTIVE_K,
            "evidence_identity": "exact stored evidence-unit identity",
        },
        "selection_rule": (
            "highest pooled development Support F1; then fewer average unique evidence "
            "units; then smaller average k; then smaller tau as deterministic final tie-break"
        ),
        "selected_tau": selected_tau,
        "input_files": {
            condition: {"path": str(path), "sha256": sha256(path)}
            for condition, path in dev_paths.items()
        },
        "table": tuning_table,
    }
    tuning_path = args.output_dir / "dev_tuning_table.json"
    write_json(tuning_path, tuning_artifact)
    write_json(
        args.output_dir / "chosen_tau.json",
        {
            "selected_tau": selected_tau,
            "selection_scope": "pooled development support examples only",
            "selection_rule": tuning_artifact["selection_rule"],
            "policy": tuning_artifact["policy"],
            "dev_tuning_table": str(tuning_path),
            "dev_tuning_table_sha256": sha256(tuning_path),
            "dev_input_sha256": {
                condition: sha256(path) for condition, path in dev_paths.items()
            },
        },
    )

    for condition, items in dev_by_condition.items():
        _, decisions = evaluate_items(
            items,
            condition=condition,
            split="dev",
            mode="adaptive",
            tau=selected_tau,
        )
        write_jsonl(args.output_dir / "dev" / f"{condition}_adaptive_decisions.jsonl", decisions)

    # Test data are deliberately loaded only after the development selection is frozen above.
    test_paths = {
        condition: args.input_root / condition / args.method / "test_details.json"
        for condition in CONDITIONS
    }
    test_by_condition = {condition: read_json(path) for condition, path in test_paths.items()}

    aggregate = {}
    regressions = {}
    for condition, items in test_by_condition.items():
        systems = {}
        fixed_metrics = {}
        for k in FIXED_K_VALUES:
            metrics, _ = evaluate_items(
                items, condition=condition, split="test", mode="fixed", fixed_k=k
            )
            fixed_metrics[k] = metrics
            systems[f"fixed_k{k}"] = metrics

        adaptive_metrics, adaptive_decisions = evaluate_items(
            items,
            condition=condition,
            split="test",
            mode="adaptive",
            tau=selected_tau,
        )
        systems["adaptive"] = adaptive_metrics
        aggregate[condition] = systems
        write_jsonl(
            args.output_dir / "test" / f"{condition}_adaptive_decisions.jsonl",
            adaptive_decisions,
        )
        method_dir = args.input_root / condition / args.method
        regressions[condition] = build_regression_report(
            condition, items, fixed_metrics, method_dir
        )

    overall = pooled_system_metrics(aggregate)
    comparison_scope = {**aggregate, "overall_pooled": overall}
    result = {
        "selected_tau": selected_tau,
        "test_inputs": {
            condition: {"path": str(path), "sha256": sha256(path)}
            for condition, path in test_paths.items()
        },
        "by_condition": aggregate,
        "overall_pooled": overall,
        "tradeoff_summary": compact_tradeoff_summary(comparison_scope),
    }
    write_json(args.output_dir / "aggregate_test_metrics.json", result)
    write_json(
        args.output_dir / "fixed_k_regression.json",
        {
            "selected_tau": selected_tau,
            "regressions": regressions,
        },
    )
    write_json(
        args.output_dir / "comparison_fixed_vs_adaptive.json",
        {
            "selected_tau": selected_tau,
            "by_condition": aggregate,
            "overall_pooled": overall,
            "tradeoff_summary": result["tradeoff_summary"],
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
