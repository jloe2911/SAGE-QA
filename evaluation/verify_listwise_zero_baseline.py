"""Hard gate: verify the fresh weight-zero DEV control reproduces production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch


DATASETS = (
    ("HotpotQA", "hotpotqa"),
    ("2WikiMultiHopQA", "2wiki"),
    ("FamilyOWL_1hop", "familyowl_1hop"),
    ("FamilyOWL_2hop", "familyowl_2hop"),
    ("pizza_100_1hop", "pizza_100_1hop"),
    ("pizza_100_2hop", "pizza_100_2hop"),
    ("pizza_250_1hop", "pizza_250_1hop"),
    ("pizza_250_2hop", "pizza_250_2hop"),
    ("OWL2Bench_1hop", "OWL2Bench_1hop"),
    ("OWL2Bench_2hop", "OWL2Bench_2hop"),
)
METRIC_ATOL = 1e-12
SCORE_ATOL = 1e-7
STATE_ATOL = 1e-7
STATE_RTOL = 1e-6
CONFIG_KEYS = (
    "model_name",
    "node_symbolic_dim",
    "subgraph_symbolic_dim",
    "gnn_hidden_dim",
    "gnn_layers",
    "classifier_hidden_dim",
    "freeze_encoder",
    "training_objective",
    "ranking_margin",
    "ranking_weight",
    "bce_weight",
    "listwise_weight",
    "max_pairs",
    "score_mode",
    "size_penalty",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("split") != "dev":
                    raise ValueError("Baseline gate accepts DEV records only")
                key = (row["dataset"], row["method"], row["example_id"])
                if key in rows:
                    raise ValueError(f"Duplicate ranking key: {key}")
                rows[key] = row
    return rows


def numeric_differences(original: Any, fresh: Any, path: str = "") -> list[dict[str, Any]]:
    failures = []
    if isinstance(original, Mapping) and isinstance(fresh, Mapping):
        if set(original) != set(fresh):
            return [{"path": path, "reason": "mapping_keys_differ"}]
        for key in original:
            failures.extend(numeric_differences(original[key], fresh[key], f"{path}.{key}"))
    elif isinstance(original, list) and isinstance(fresh, list):
        if len(original) != len(fresh):
            return [{"path": path, "reason": "list_lengths_differ"}]
        for index, (left, right) in enumerate(zip(original, fresh)):
            failures.extend(numeric_differences(left, right, f"{path}[{index}]"))
    elif isinstance(original, (int, float)) and isinstance(fresh, (int, float)):
        difference = abs(float(original) - float(fresh))
        if difference > METRIC_ATOL:
            failures.append(
                {"path": path, "original": original, "fresh": fresh, "abs_diff": difference}
            )
    elif original != fresh:
        failures.append({"path": path, "original": original, "fresh": fresh})
    return failures


def metric_view(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row["dataset"],
            "domain": row["domain"],
            "dev_examples": row["dev_examples"],
            "methods": row["methods"],
        }
        for row in metrics["datasets"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-run",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_k_sensitivity"),
    )
    parser.add_argument(
        "--fresh-run",
        type=Path,
        default=Path(
            "outputs/development_runs/production_generator_d_v1_listwise_corrected_dev/w_0p0"
        ),
    )
    parser.add_argument(
        "--original-checkpoints", type=Path, default=Path("checkpoints/production_generator_d_v1")
    )
    parser.add_argument(
        "--fresh-checkpoints",
        type=Path,
        default=Path(
            "checkpoints/development/production_generator_d_v1_listwise_corrected_dev/w_0p0"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/diagnostics/production_generator_d_v1_listwise_corrected_dev/baseline_reproduction.json"
        ),
    )
    args = parser.parse_args()

    metric_failures = numeric_differences(
        metric_view(read_json(args.original_run / "metrics.json")),
        metric_view(read_json(args.fresh_run / "metrics.json")),
        "datasets",
    )

    original_rankings = read_jsonl(args.original_run / "per_example_rankings.jsonl")
    fresh_rankings = read_jsonl(args.fresh_run / "per_example_rankings.jsonl")
    ranking_failures = []
    maximum_score_difference = 0.0
    if set(original_rankings) != set(fresh_rankings):
        ranking_failures.append({"reason": "ranking_record_keys_differ"})
    else:
        for key in original_rankings:
            original_candidates = original_rankings[key]["ranked_candidates"]
            fresh_candidates = fresh_rankings[key]["ranked_candidates"]
            if len(original_candidates) != len(fresh_candidates):
                ranking_failures.append({"key": key, "reason": "top5_lengths_differ"})
                continue
            for rank, (original, fresh) in enumerate(
                zip(original_candidates, fresh_candidates), start=1
            ):
                if original["subgraph_units"] != fresh["subgraph_units"]:
                    ranking_failures.append(
                        {"key": key, "rank": rank, "reason": "candidate_identity_differs"}
                    )
                    break
                for score_key in ("score", "adjusted_score"):
                    difference = abs(float(original[score_key]) - float(fresh[score_key]))
                    maximum_score_difference = max(maximum_score_difference, difference)
                    if difference > SCORE_ATOL:
                        ranking_failures.append(
                            {
                                "key": key,
                                "rank": rank,
                                "score_key": score_key,
                                "abs_diff": difference,
                                "reason": "score_tolerance_exceeded",
                            }
                        )
                        break

    checkpoint_rows = []
    checkpoint_failures = []
    maximum_state_difference = 0.0
    for dataset, directory in DATASETS:
        original = torch.load(
            args.original_checkpoints / directory / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        fresh = torch.load(
            args.fresh_checkpoints / directory / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        config_equal = all(original.get(key) == fresh.get(key) for key in CONFIG_KEYS)
        if not config_equal:
            checkpoint_failures.append({"dataset": dataset, "reason": "checkpoint_config_differs"})
        original_state = original["model_state_dict"]
        fresh_state = fresh["model_state_dict"]
        if set(original_state) != set(fresh_state):
            checkpoint_failures.append({"dataset": dataset, "reason": "state_keys_differ"})
            continue
        dataset_max = 0.0
        all_close = True
        for name in original_state:
            left = original_state[name]
            right = fresh_state[name]
            difference = float((left - right).abs().max()) if left.numel() else 0.0
            dataset_max = max(dataset_max, difference)
            all_close = all_close and torch.allclose(left, right, atol=STATE_ATOL, rtol=STATE_RTOL)
        maximum_state_difference = max(maximum_state_difference, dataset_max)
        if not all_close:
            checkpoint_failures.append(
                {
                    "dataset": dataset,
                    "reason": "state_tolerance_exceeded",
                    "max_abs_diff": dataset_max,
                }
            )
        checkpoint_rows.append(
            {
                "dataset": dataset,
                "config_equal": config_equal,
                "state_allclose": all_close,
                "max_abs_diff": dataset_max,
            }
        )

    passed = not metric_failures and not ranking_failures and not checkpoint_failures
    result = {
        "schema_version": "production_generator_d_v1_listwise_zero_baseline_reproduction_v1",
        "status": "passed" if passed else "failed_stop_before_nonzero_interpretation",
        "predeclared_tolerances": {
            "metric_absolute": METRIC_ATOL,
            "candidate_identity": "exact top-5 units and order for every DEV example/method",
            "score_absolute": SCORE_ATOL,
            "state_absolute": STATE_ATOL,
            "state_relative": STATE_RTOL,
        },
        "metric_failures": metric_failures[:100],
        "ranking_failures": ranking_failures[:100],
        "checkpoint_failures": checkpoint_failures,
        "checkpoint_comparison": checkpoint_rows,
        "maximum_top5_score_abs_diff": maximum_score_difference,
        "maximum_state_abs_diff": maximum_state_difference,
        "safe_to_interpret_nonzero_weights": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
