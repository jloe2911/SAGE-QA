"""Summarize the frozen corrected-listwise DEV grid without refitting policies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.adaptive_support_aggregation import canonical_evidence_key
from evaluation.adaptive_support_aggregation_v2 import (
    adaptive_v2_support_aggregate,
    load_domain_policy,
)


WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.2)
K_VALUES = (1, 2, 3, 5)
DATASETS = (
    "HotpotQA",
    "2WikiMultiHopQA",
    "FamilyOWL_1hop",
    "FamilyOWL_2hop",
    "pizza_100_1hop",
    "pizza_100_2hop",
    "pizza_250_1hop",
    "pizza_250_2hop",
    "OWL2Bench_1hop",
    "OWL2Bench_2hop",
)


def slug(weight: float) -> str:
    return "w_" + str(weight).replace(".", "p")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("split") != "dev":
                    raise ValueError("Only DEV records are permitted")
                records.append(row)
    expected = len(DATASETS) * 2
    if len({(row["dataset"], row["method"]) for row in records}) != expected:
        raise ValueError(f"Incomplete dataset/method coverage in {path}")
    return records


def support_scores(
    predicted: Sequence[Any], alternatives: Sequence[Sequence[Any]]
) -> dict[str, float]:
    predicted_keys = {canonical_evidence_key(value) for value in predicted}
    rows = []
    containment = False
    for alternative in alternatives:
        gold = {canonical_evidence_key(value) for value in alternative}
        overlap = len(predicted_keys & gold)
        precision = overlap / len(predicted_keys) if predicted_keys else 0.0
        recall = overlap / len(gold) if gold else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append({"precision": precision, "recall": recall, "f1": f1})
        containment = containment or bool(gold and gold <= predicted_keys)
    best = max(
        rows, key=lambda row: row["f1"], default={"precision": 0.0, "recall": 0.0, "f1": 0.0}
    )
    return {**best, "complete_support_containment": float(containment)}


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    names = ("precision", "recall", "f1", "complete_support_containment")
    return {name: statistics.fmean(float(row[name]) for row in rows) for name in names}


def evaluate_record(record: Mapping[str, Any], policy_dir: Path) -> dict[str, Any]:
    fixed = {}
    for prefix in record["prefix_evaluation"]:
        k = int(prefix["k"])
        if k in K_VALUES:
            fixed[str(k)] = support_scores(
                prefix["retrieved_evidence_units"], record["gold_explanations"]
            )
    policy = load_domain_policy(policy_dir, domain=record["domain"])
    aggregate = adaptive_v2_support_aggregate(record["ranked_candidates"], policy=policy)
    adaptive = {
        **support_scores(aggregate["support_units"], record["gold_explanations"]),
        "selected_k": int(aggregate["selected_k"]),
    }
    return {"fixed": fixed, "adaptive": adaptive}


def summarize_weight(
    records: Sequence[Mapping[str, Any]], policy_dirs: Mapping[str, Path]
) -> dict[str, Any]:
    evaluated = []
    for record in records:
        result = evaluate_record(record, policy_dirs[record["method"]])
        evaluated.append({**record, "evaluation": result})

    per_dataset = {}
    for dataset in DATASETS:
        per_dataset[dataset] = {}
        for method in ("gnn_only", "sageqa_final"):
            rows = [
                row for row in evaluated if row["dataset"] == dataset and row["method"] == method
            ]
            fixed = {
                str(k): mean_metrics([row["evaluation"]["fixed"][str(k)] for row in rows])
                for k in K_VALUES
            }
            adaptive_rows = [row["evaluation"]["adaptive"] for row in rows]
            adaptive = {
                **mean_metrics(adaptive_rows),
                "mean_k": statistics.fmean(row["selected_k"] for row in adaptive_rows),
            }
            per_dataset[dataset][method] = {"fixed": fixed, "adaptive": adaptive}

    macro = {}
    for method in ("gnn_only", "sageqa_final"):
        macro[method] = {
            "fixed": {
                str(k): {
                    metric: statistics.fmean(
                        per_dataset[d][method]["fixed"][str(k)][metric] for d in DATASETS
                    )
                    for metric in ("precision", "recall", "f1", "complete_support_containment")
                }
                for k in K_VALUES
            },
            "adaptive": {
                metric: statistics.fmean(
                    per_dataset[d][method]["adaptive"][metric] for d in DATASETS
                )
                for metric in (
                    "precision",
                    "recall",
                    "f1",
                    "complete_support_containment",
                    "mean_k",
                )
            },
        }
    return {"per_dataset": per_dataset, "macro": macro, "records": evaluated}


def nested_delta(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in current.items():
        if isinstance(value, Mapping):
            result[key] = nested_delta(value, baseline[key])
        elif isinstance(value, (int, float)):
            result[key] = float(value) - float(baseline[key])
    return result


def cohort_analysis(by_weight: Mapping[float, Mapping[str, Any]]) -> dict[str, Any]:
    baseline_records = {
        row["example_id"]: row
        for row in by_weight[0.0]["records"]
        if row["method"] == "sageqa_final"
    }
    cohort = {
        example_id: row
        for example_id, row in baseline_records.items()
        if not row["ranking_diagnostic"]["top1_complete"]
        and row["ranking_diagnostic"]["best_complete_rank"] is not None
    }
    near_ties = {
        example_id
        for example_id, row in cohort.items()
        if row["ranking_diagnostic"]["top1_minus_best_complete_score_gap"] <= 0.02
    }
    result = {
        "baseline_scoreable_complete_candidate_misrankings": len(cohort),
        "baseline_near_ties_gap_le_0_02": len(near_ties),
        "weights": {},
    }
    for weight, summary in by_weight.items():
        current = {
            row["example_id"]: row for row in summary["records"] if row["method"] == "sageqa_final"
        }
        rows = []
        for example_id, base in cohort.items():
            now = current[example_id]
            base_rank = int(base["ranking_diagnostic"]["best_complete_rank"])
            now_rank = now["ranking_diagnostic"]["best_complete_rank"]
            rows.append((example_id, base, now, base_rank, now_rank))

        ranks = [float(row[4]) for row in rows if row[4] is not None]
        gaps = [
            float(row[2]["ranking_diagnostic"]["top1_minus_best_complete_score_gap"])
            for row in rows
            if row[4] is not None
        ]
        result["weights"][str(weight)] = {
            "corrected_to_rank_1": sum(row[4] == 1 for row in rows),
            "best_complete_rank_le_2": sum(row[4] is not None and row[4] <= 2 for row in rows),
            "best_complete_rank_le_3": sum(row[4] is not None and row[4] <= 3 for row in rows),
            "best_complete_rank_le_5": sum(row[4] is not None and row[4] <= 5 for row in rows),
            "newly_moved_into_top_2": sum(
                row[3] > 2 and row[4] is not None and row[4] <= 2 for row in rows
            ),
            "newly_moved_into_top_3": sum(
                row[3] > 3 and row[4] is not None and row[4] <= 3 for row in rows
            ),
            "newly_moved_into_top_5": sum(
                row[3] > 5 and row[4] is not None and row[4] <= 5 for row in rows
            ),
            "made_worse": sum(row[4] is None or row[4] > row[3] for row in rows),
            "mean_best_complete_candidate_rank": statistics.fmean(ranks) if ranks else None,
            "mean_score_gap": statistics.fmean(gaps) if gaps else None,
            "near_ties_corrected_to_rank_1": sum(
                example_id in near_ties and now_rank == 1 for example_id, _, _, _, now_rank in rows
            ),
            "near_ties_made_worse": sum(
                example_id in near_ties and (now_rank is None or now_rank > base_rank)
                for example_id, _, _, base_rank, now_rank in rows
            ),
        }
    return result


def group_effects(summary: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    groups = {
        "1hop": [d for d in DATASETS if "1hop" in d],
        "2hop": [d for d in DATASETS if "2hop" in d or d in {"HotpotQA", "2WikiMultiHopQA"}],
        "text": ["HotpotQA", "2WikiMultiHopQA"],
        "ontology": [d for d in DATASETS if d not in {"HotpotQA", "2WikiMultiHopQA"}],
    }
    effects = {}
    for group, datasets in groups.items():
        current_rows = [summary["per_dataset"][d]["sageqa_final"]["adaptive"] for d in datasets]
        baseline_rows = [baseline["per_dataset"][d]["sageqa_final"]["adaptive"] for d in datasets]
        effects[group] = {
            metric: statistics.fmean(row[metric] for row in current_rows)
            - statistics.fmean(row[metric] for row in baseline_rows)
            for metric in ("precision", "recall", "f1", "complete_support_containment", "mean_k")
        }
    return effects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_listwise_corrected_dev"),
    )
    parser.add_argument(
        "--sageqa-policy-dir",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_adaptive_k"),
    )
    parser.add_argument(
        "--gnn-policy-dir",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_gnn_adaptive_k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/production_generator_d_v1_listwise_corrected_dev"),
    )
    args = parser.parse_args()

    reproduction_path = args.output_dir / "baseline_reproduction.json"
    if not reproduction_path.is_file():
        raise FileNotFoundError("Fresh weight-zero baseline reproduction gate has not been run")
    reproduction = read_json(reproduction_path)
    if (
        reproduction.get("status") != "passed"
        or reproduction.get("safe_to_interpret_nonzero_weights") is not True
    ):
        raise RuntimeError(
            "Fresh weight-zero baseline did not reproduce production; stop before grid interpretation"
        )

    policy_dirs = {"gnn_only": args.gnn_policy_dir, "sageqa_final": args.sageqa_policy_dir}
    by_weight = {}
    for weight in WEIGHTS:
        records = read_records(args.run_root / slug(weight) / "per_example_rankings.jsonl")
        by_weight[weight] = summarize_weight(records, policy_dirs)

    baseline = by_weight[0.0]
    per_dataset = {
        str(weight): {
            dataset: {
                "metrics": summary["per_dataset"][dataset],
                "delta_vs_weight_0": nested_delta(
                    summary["per_dataset"][dataset], baseline["per_dataset"][dataset]
                ),
            }
            for dataset in DATASETS
        }
        for weight, summary in by_weight.items()
    }
    candidate_bias = {
        str(weight): {
            method: statistics.fmean(
                row["ranking_diagnostic"]["candidate_size_score_pearson"]
                for row in summary["records"]
                if row["method"] == method
            )
            for method in ("gnn_only", "sageqa_final")
        }
        for weight, summary in by_weight.items()
    }
    grid_rows = []
    for weight, summary in by_weight.items():
        operational = summary["macro"]["sageqa_final"]["adaptive"]
        dataset_deltas = [
            summary["per_dataset"][d]["sageqa_final"]["adaptive"]["f1"]
            - baseline["per_dataset"][d]["sageqa_final"]["adaptive"]["f1"]
            for d in DATASETS
        ]
        grid_rows.append(
            {
                "weight": weight,
                "macro": summary["macro"],
                "operational_selection_metrics": operational,
                "datasets_improved_f1": sum(delta > 0 for delta in dataset_deltas),
                "datasets_harmed_f1": sum(delta < 0 for delta in dataset_deltas),
                "minimum_dataset_f1_delta": min(dataset_deltas),
                "group_effects_vs_weight_0": group_effects(summary, baseline),
            }
        )

    selected = max(
        grid_rows,
        key=lambda row: (
            row["operational_selection_metrics"]["f1"],
            row["operational_selection_metrics"]["recall"],
            row["operational_selection_metrics"]["complete_support_containment"],
            row["operational_selection_metrics"]["precision"],
            row["minimum_dataset_f1_delta"],
            -row["weight"],
        ),
    )
    selected_weight = float(selected["weight"])
    replacement = (
        selected_weight > 0.0
        and selected["operational_selection_metrics"]["f1"]
        > grid_rows[0]["operational_selection_metrics"]["f1"]
    )

    metrics = {
        "schema_version": "production_generator_d_v1_corrected_listwise_dev_metrics_v1",
        "split": "dev",
        "weights": grid_rows,
        "selection_scope": "macro across 10 datasets for final SAGE-QA under the existing frozen adaptive policy",
        "candidate_size_score_pearson": candidate_bias,
    }
    ranking = cohort_analysis(by_weight)
    selected_config = {
        "schema_version": "production_generator_d_v1_corrected_listwise_selected_dev_configuration_v1",
        "split": "dev",
        "selected_listwise_weight": selected_weight,
        "selection_order": [
            "macro F1",
            "macro recall",
            "complete-support containment",
            "precision",
            "worst per-dataset F1 delta",
            "smaller weight",
        ],
        "selected_metrics": selected["operational_selection_metrics"],
        "production_replacement_justified": replacement,
        "production_modified": False,
        "next_justified_experiment_if_not_replaced": "512-candidate graph regime"
        if not replacement
        else None,
        "adaptive_thresholds_refit": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "per_dataset.json").write_text(
        json.dumps(per_dataset, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ranking_failure_analysis.json").write_text(
        json.dumps(ranking, indent=2), encoding="utf-8"
    )
    (args.output_dir / "selected_dev_configuration.json").write_text(
        json.dumps(selected_config, indent=2), encoding="utf-8"
    )

    lines = [
        "# Corrected listwise DEV experiment",
        "",
        f"Selected DEV-only weight: `{selected_weight}`.",
        f"Production replacement justified: `{str(replacement).lower()}`.",
        "Selection used final SAGE-QA macro metrics under the existing frozen adaptive policy; thresholds were not refit.",
        "",
        "| Weight | Macro precision | Macro recall | Macro F1 | Complete containment | Mean k | Improved datasets | Harmed datasets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grid_rows:
        value = row["operational_selection_metrics"]
        lines.append(
            f"| {row['weight']} | {value['precision']:.6f} | {value['recall']:.6f} | {value['f1']:.6f} | "
            f"{value['complete_support_containment']:.6f} | {value['mean_k']:.4f} | "
            f"{row['datasets_improved_f1']} | {row['datasets_harmed_f1']} |"
        )
    lines.extend(
        [
            "",
            f"The fixed baseline diagnostic recovered {ranking['baseline_scoreable_complete_candidate_misrankings']} scoreable complete-candidate misrankings and {ranking['baseline_near_ties_gap_le_0_02']} near ties.",
            "",
            "No production checkpoint was replaced, no adaptive threshold was refit, and evaluation stopped at DEV.",
        ]
    )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(selected_config, indent=2))


if __name__ == "__main__":
    main()
