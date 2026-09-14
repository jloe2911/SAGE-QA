"""Materialize frozen cross-encoder DEV rankings for adaptive-k fitting.

This adapter never scores candidates. It verifies the frozen DEV prediction
hash, joins candidate identities back to the frozen DEV candidate corpus, and
writes the canonical ranking-artifact schema consumed by
``fit_production_dev_adaptive_k.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing.retrieval_contracts import cap_inference_candidate_rows  # noqa: E402
from evaluation.run_production_dev_k_sensitivity import (  # noqa: E402
    DATASETS,
    evaluate_ranking,
    grouped_jsonl,
    sha256,
)
from training.train_gnn_subgraph_retriever import gold_explanations_from_row  # noqa: E402


ALLOWED_K = (1, 2, 3, 5)
METHOD_SOURCES = {
    "cross_encoder": ("cross_encoder_order", "cross_encoder_scores", "cross_encoder"),
    "final_sageqa": (
        "cross_encoder_symbolic_order",
        "cross_encoder_symbolic_scores",
        "cross_encoder_plus_text_chain_or_proof",
    ),
}


def candidate_identity(row: Mapping[str, Any]) -> str:
    units = [str(unit) for unit in row.get("subgraph_units", []) or []]
    payload = json.dumps(units, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in result:
                raise ValueError(f"{path}:{line_number}: missing or duplicate example_id")
            result[example_id] = row
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def materialize(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = args.cross_encoder_dir / "dev_predictions_frozen.jsonl"
    freeze_path = args.cross_encoder_dir / "prediction_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("split") != "dev" or freeze.get("test_accessed") is not False:
        raise ValueError("Prediction freeze does not establish a DEV-only source")
    if sha256(prediction_path) != freeze.get("predictions_sha256"):
        raise ValueError("Frozen DEV prediction hash mismatch")
    predictions = load_predictions(prediction_path)

    sums: dict[str, dict[str, dict[str, float]]] = {
        method: {str(k): defaultdict(float) for k in ALLOWED_K}
        for method in METHOD_SOURCES
    }
    counts: dict[str, int] = defaultdict(int)
    dataset_metadata: dict[str, Any] = {}
    ranking_path = args.output_dir / "per_example_rankings.jsonl"
    with ranking_path.open("w", encoding="utf-8", newline="\n") as output:
        for dataset, data_dir, _, domain, _ in DATASETS:
            dev_path = args.data_root / data_dir / "dev_subgraph_retrieval.jsonl"
            if not dev_path.is_file() or "test" in dev_path.name.lower():
                raise FileNotFoundError(f"DEV input missing or invalid: {dev_path}")
            row_count = 0
            for example_id, source_rows in grouped_jsonl(dev_path):
                row_count += len(source_rows)
                prediction = predictions.pop(example_id, None)
                if prediction is None:
                    raise ValueError(f"Missing frozen prediction: {example_id}")
                if prediction.get("dataset") != dataset or prediction.get("domain") != domain:
                    raise ValueError(f"Dataset/domain mismatch: {example_id}")
                admitted = cap_inference_candidate_rows(source_rows, max_candidates=320)
                by_identity = {candidate_identity(row): row for row in admitted}
                if len(by_identity) != len(admitted):
                    raise ValueError(f"Duplicate candidate identity: {example_id}")
                gold = gold_explanations_from_row(dict(admitted[0]))
                if not gold:
                    raise ValueError(f"Missing DEV gold: {example_id}")

                for method, (order_key, score_key, score_mode) in METHOD_SOURCES.items():
                    order = list(prediction[order_key])
                    scores = [float(value) for value in prediction[score_key]]
                    if len(order) != len(admitted) or len(scores) != len(order):
                        raise ValueError(f"Candidate-count mismatch: {example_id}/{method}")
                    if set(order) != set(by_identity) or len(set(order)) != len(order):
                        raise ValueError(f"Candidate-identity mismatch: {example_id}/{method}")
                    if any(left < right for left, right in zip(scores, scores[1:])):
                        raise ValueError(f"Scores are not non-increasing: {example_id}/{method}")
                    ranked = []
                    for rank, (identity, score) in enumerate(zip(order, scores), 1):
                        item = dict(by_identity[identity])
                        item["score"] = score
                        item["adjusted_score"] = score
                        item["rank"] = rank
                        ranked.append(item)
                    by_k, prefixes = evaluate_ranking(ranked, gold)
                    for k, values in by_k.items():
                        for metric, value in values.items():
                            sums[method][k][metric] += float(value)
                    output.write(
                        json.dumps(
                            {
                                "dataset": dataset,
                                "domain": domain,
                                "split": "dev",
                                "example_id": example_id,
                                "method": method,
                                "score_mode": score_mode,
                                "gold_explanations": gold,
                                "ranked_candidates": [
                                    {
                                        "rank": rank,
                                        "score": float(item["score"]),
                                        "adjusted_score": float(item["adjusted_score"]),
                                        "subgraph_size": int(
                                            item.get("subgraph_size", len(item.get("subgraph_units", [])))
                                        ),
                                        "subgraph_units": list(item.get("subgraph_units", [])),
                                    }
                                    for rank, item in enumerate(ranked[:5], 1)
                                ],
                                "prefix_evaluation": prefixes,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                counts[dataset] += 1
            dataset_metadata[dataset] = {
                "dev_path": str(dev_path),
                "dev_sha256": sha256(dev_path),
                "dev_candidate_rows": row_count,
                "dev_examples": counts[dataset],
            }
    if predictions:
        raise ValueError(f"Unmatched frozen predictions: {len(predictions)}")

    total_examples = sum(counts.values())
    metrics = {
        "schema_version": "cross_encoder_adaptive_dev_input_v1",
        "split": "dev",
        "k_values": list(ALLOWED_K),
        "methods": {
            method: {
                k: {metric: value / total_examples for metric, value in values.items()}
                for k, values in by_k.items()
            }
            for method, by_k in sums.items()
        },
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(
        args.output_dir / "checkpoint_metadata.json",
        {
            "schema_version": "cross_encoder_adaptive_dev_input_metadata_v1",
            "status": "adaptive_k_ready",
            "split": "dev",
            "test_rows_read": 0,
            "test_gold_accessed": False,
            "adaptive_k_ready": True,
            "prediction_freeze_path": str(freeze_path),
            "prediction_freeze_sha256": sha256(freeze_path),
            "frozen_predictions_path": str(prediction_path),
            "frozen_predictions_sha256": sha256(prediction_path),
            "checkpoint_sha256": freeze.get("checkpoint_sha256"),
            "datasets": dataset_metadata,
            "candidate_generation_rerun": False,
            "model_training_run": False,
            "symbolic_reranking_recomputed": False,
        },
    )
    print(json.dumps({"complete": True, "dev_examples": total_examples, "methods": list(METHOD_SOURCES)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-encoder-dir",
        type=Path,
        default=Path("outputs/development_runs/question_candidate_cross_encoder_v1"),
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/production_generator_d_v1")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    materialize(parse_args())
