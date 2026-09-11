"""One-time frozen TEST retrieval for cross-encoder k=1 and adaptive-k."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.retrieval_contracts import cap_inference_candidate_rows  # noqa: E402
from evaluation.adaptive_support_aggregation import fixed_k_support_aggregate  # noqa: E402
from evaluation.run_production_dev_k_sensitivity import grouped_jsonl  # noqa: E402
from evaluation.run_production_test_retrieval import (  # noqa: E402
    ALLOWED_K,
    DATASETS,
    adaptive_select,
    best_scores,
    load_policy_bundle,
    ontology_gold,
    sha256,
    summarize_setting,
    text_gold,
)
from experiments.cross_encoder_reranking_dev_v1.run_experiment import (  # noqa: E402
    MAX_INFERENCE_CANDIDATES,
    MAX_LENGTH,
    candidate_identity,
    safe_inference_row,
    score_texts,
    serialize_candidate,
)
from training.train_gnn_subgraph_retriever import compute_adjusted_score  # noqa: E402


METHODS = {
    "cross_encoder": "neural",
    "final_sageqa": None,
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def fixed_select(ranked: list[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = fixed_k_support_aggregate(ranked, k=1)
    return {
        "selected_k": 1,
        "retrieved_evidence_units": list(aggregate["support_units"]),
        "retrieved_evidence_count": int(aggregate["final_support_size"]),
    }


def generate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "test_predictions_frozen.jsonl"
    freeze_path = args.output_dir / "generation_freeze.json"
    if output_path.exists() or freeze_path.exists():
        raise FileExistsError("Refusing to overwrite an existing one-time TEST prediction freeze")

    policies = {}
    policy_hashes = {}
    for method, policy_dir in (
        ("cross_encoder", args.cross_encoder_policy_dir),
        ("final_sageqa", args.final_sageqa_policy_dir),
    ):
        policies[method], _, policy_hashes[method] = load_policy_bundle(policy_dir, method)

    checkpoint_path = args.cross_encoder_dir / "checkpoint_final"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path, local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    input_hashes = {}
    dataset_counts = {}

    for dataset, data_dir, _, domain, final_mode, _ in DATASETS:
        test_path = args.data_root / data_dir / "test_subgraph_retrieval.jsonl"
        if not test_path.is_file() or "test" not in test_path.name.lower():
            raise FileNotFoundError(f"Missing frozen TEST candidates: {test_path}")
        input_hashes[str(test_path)] = sha256(test_path)
        count = 0
        for example_id, raw_rows in grouped_jsonl(test_path):
            admitted_source = cap_inference_candidate_rows(
                raw_rows, max_candidates=MAX_INFERENCE_CANDIDATES
            )
            rows = [safe_inference_row(row) for row in admitted_source]
            prohibited = {
                "gold_explanations",
                "gold_units",
                "gold_support_units",
                "supporting_facts",
                "evidences",
                "answer",
                "answers",
                "label",
                "rank_target",
            }
            if any(prohibited & set(row) for row in rows):
                raise ValueError(f"Gold-bearing field reached inference: {example_id}")
            question = str(rows[0].get("question") or rows[0].get("abs_question") or example_id)
            identities = [candidate_identity(row) for row in rows]
            if len(identities) != len(set(identities)):
                raise ValueError(f"Duplicate candidate identity: {example_id}")
            texts = [serialize_candidate(question, row.get("subgraph_units", []), domain) for row in rows]
            scores = score_texts(texts, tokenizer, model, device)
            methods = {}
            for method, mode in METHODS.items():
                selected_mode = final_mode if mode is None else mode
                adjusted = [
                    score
                    if selected_mode == "neural"
                    else compute_adjusted_score(
                        row, score, score_mode=selected_mode, size_penalty=0.01
                    )
                    for row, score in zip(rows, scores)
                ]
                order = sorted(
                    range(len(rows)), key=lambda index: (-adjusted[index], identities[index])
                )
                ranked = []
                for rank, index in enumerate(order[:5], 1):
                    item = dict(rows[index])
                    item["score"] = float(scores[index])
                    item["adjusted_score"] = float(adjusted[index])
                    item["rank"] = rank
                    ranked.append(item)
                methods[method] = {
                    "score_mode": selected_mode,
                    "k1": fixed_select(ranked),
                    "adaptive": adaptive_select(ranked, policies[method][domain]),
                }
            predictions.append(
                {"dataset": dataset, "domain": domain, "example_id": example_id, "methods": methods}
            )
            count += 1
            if count % 100 == 0:
                print(f"{dataset}: froze {count} TEST predictions", flush=True)
        dataset_counts[dataset] = count

    write_jsonl(output_path, predictions)
    checkpoint_file = checkpoint_path / "model.safetensors"
    write_json(
        freeze_path,
        {
            "schema_version": "cross_encoder_adaptive_test_prediction_freeze_v1",
            "status": "complete_frozen_before_gold_join",
            "split": "test",
            "predictions_path": str(output_path),
            "predictions_sha256": sha256(output_path),
            "prediction_examples": len(predictions),
            "datasets": dataset_counts,
            "test_candidate_input_sha256": input_hashes,
            "checkpoint_path": str(checkpoint_file),
            "checkpoint_sha256": sha256(checkpoint_file),
            "adaptive_policy_artifact_hashes": policy_hashes,
            "allowed_k": list(ALLOWED_K),
            "gold_accessed_before_freeze": False,
            "prediction_seconds": time.perf_counter() - started,
            "device": str(device),
        },
    )
    print(json.dumps({"phase": "generate", "complete": True, "predictions": len(predictions)}))


def evaluate(args: argparse.Namespace) -> None:
    prediction_path = args.output_dir / "test_predictions_frozen.jsonl"
    freeze_path = args.output_dir / "generation_freeze.json"
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError("Refusing a second TEST evaluation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "complete_frozen_before_gold_join":
        raise ValueError("TEST predictions are not completely frozen")
    if sha256(prediction_path) != freeze.get("predictions_sha256"):
        raise ValueError("TEST prediction freeze hash mismatch")
    predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    if len(predictions) != int(freeze.get("prediction_examples", -1)):
        raise ValueError("Frozen TEST prediction count mismatch")

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_dataset[str(row["dataset"])].append(row)
    scored_rows = []
    dataset_results = []
    for dataset, _, _, domain, _, gold_source in DATASETS:
        rows = by_dataset[dataset]
        ids = {str(row["example_id"]) for row in rows}
        source = Path(gold_source)
        gold = text_gold(source, dataset, ids) if domain == "text" else ontology_gold(source, ids)
        if set(gold) != ids:
            raise ValueError(f"Incomplete TEST gold join: {dataset}")
        method_rows = {method: {setting: [] for setting in ("k1", "adaptive")} for method in METHODS}
        for row in rows:
            example_id = str(row["example_id"])
            for method in METHODS:
                for setting in ("k1", "adaptive"):
                    prediction = row["methods"][method][setting]
                    values = {
                        "dataset": dataset,
                        "domain": domain,
                        "example_id": example_id,
                        "method": method,
                        "setting": setting,
                        "selected_k": int(prediction["selected_k"]),
                        **best_scores(prediction["retrieved_evidence_units"], gold[example_id]),
                    }
                    method_rows[method][setting].append(values)
                    scored_rows.append(values)
        dataset_results.append(
            {
                "dataset": dataset,
                "domain": domain,
                "examples": len(rows),
                "methods": {
                    method: {
                        "k1": summarize_setting(method_rows[method]["k1"]),
                        "adaptive": summarize_setting(method_rows[method]["adaptive"], adaptive=True),
                        "delta_f1_adaptive_minus_k1": (
                            summarize_setting(method_rows[method]["adaptive"])["f1"]
                            - summarize_setting(method_rows[method]["k1"])["f1"]
                        ),
                    }
                    for method in METHODS
                },
            }
        )

    overall = {}
    for method in METHODS:
        k1 = [row for row in scored_rows if row["method"] == method and row["setting"] == "k1"]
        adaptive = [
            row for row in scored_rows if row["method"] == method and row["setting"] == "adaptive"
        ]
        overall[method] = {
            "k1": summarize_setting(k1),
            "adaptive": summarize_setting(adaptive, adaptive=True),
        }
        overall[method]["delta_f1_adaptive_minus_k1"] = (
            overall[method]["adaptive"]["f1"] - overall[method]["k1"]["f1"]
        )
    metrics = {
        "schema_version": "cross_encoder_adaptive_test_retrieval_v1",
        "status": "complete_one_time_frozen_test_evaluation",
        "split": "test",
        "metric_aggregation": "unweighted mean over examples",
        "allowed_k": list(ALLOWED_K),
        "overall": overall,
        "datasets": dataset_results,
        "prediction_freeze_sha256": sha256(freeze_path),
        "test_informed_tuning": False,
    }
    write_jsonl(args.output_dir / "per_example_test_retrieval.jsonl", scored_rows)
    write_json(metrics_path, metrics)
    artifact_files = [prediction_path, freeze_path, args.output_dir / "per_example_test_retrieval.jsonl", metrics_path]
    write_json(
        args.output_dir / "artifact_manifest.json",
        {"status": "complete_frozen", "files": {path.name: sha256(path) for path in artifact_files}},
    )
    print(json.dumps({"phase": "evaluate", "complete": True, "overall": overall}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("generate", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=Path("data/production_generator_d_v1"))
    parser.add_argument(
        "--cross-encoder-dir",
        type=Path,
        default=Path("outputs/development_runs/question_candidate_cross_encoder_v1"),
    )
    parser.add_argument("--cross-encoder-policy-dir", type=Path, required=True)
    parser.add_argument("--final-sageqa-policy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.phase == "generate":
        generate(parsed)
    else:
        evaluate(parsed)
