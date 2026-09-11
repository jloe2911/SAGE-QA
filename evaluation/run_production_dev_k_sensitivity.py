"""DEV-only fixed-k retrieval evaluation for frozen production Generator D models."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.adaptive_support_aggregation import (
    canonical_evidence_key,
    fixed_k_support_aggregate,
)
from evaluation.eval_gnn_subgraph_retriever import load_gnn_checkpoint
from models.gnn_subgraph_retriever import GNNSubgraphRetriever
from training.train_gnn_subgraph_retriever import (
    compute_adjusted_score,
    encode_example_graph,
    prepare_examples,
    score_candidate_rows,
)


K_VALUES = (1, 2, 3, 5)
DATASETS = (
    ("HotpotQA", "HotpotQA", "hotpotqa", "text", "sageqa_text_chain"),
    ("2WikiMultiHopQA", "2WikiMultiHopQA", "2wiki", "text", "sageqa_text_chain"),
    ("FamilyOWL_1hop", "FamilyOWL_1hop", "familyowl_1hop", "ontology", "sageqa_proof"),
    ("FamilyOWL_2hop", "FamilyOWL_2hop", "familyowl_2hop", "ontology", "sageqa_proof"),
    ("pizza_100_1hop", "pizza_100_1hop", "pizza_100_1hop", "ontology", "sageqa_proof"),
    ("pizza_100_2hop", "pizza_100_2hop", "pizza_100_2hop", "ontology", "sageqa_proof"),
    ("pizza_250_1hop", "pizza_250_1hop", "pizza_250_1hop", "ontology", "sageqa_proof"),
    ("pizza_250_2hop", "pizza_250_2hop", "pizza_250_2hop", "ontology", "sageqa_proof"),
    ("OWL2Bench_1hop", "OWL2Bench_1hop", "OWL2Bench_1hop", "ontology", "sageqa_proof"),
    ("OWL2Bench_2hop", "OWL2Bench_2hop", "OWL2Bench_2hop", "ontology", "sageqa_proof"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def grouped_jsonl(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    """Stream a candidate JSONL that is required to be contiguous by example."""
    seen: set[str] = set()
    current_id: str | None = None
    current_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id:
                raise ValueError(f"{path}:{line_number}: missing example_id")
            if current_id is None:
                current_id = example_id
            if example_id != current_id:
                if example_id in seen:
                    raise ValueError(f"{path}: non-contiguous example {example_id!r}")
                seen.add(current_id)
                yield current_id, current_rows
                current_id = example_id
                current_rows = []
            current_rows.append(row)
    if current_id is not None:
        yield current_id, current_rows


def evidence_scores(predicted: Sequence[Any], gold: Sequence[Any]) -> dict[str, float]:
    predicted_keys = {canonical_evidence_key(unit) for unit in predicted}
    gold_keys = {canonical_evidence_key(unit) for unit in gold}
    overlap = len(predicted_keys & gold_keys)
    precision = overlap / len(predicted_keys) if predicted_keys else 0.0
    recall = overlap / len(gold_keys) if gold_keys else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def best_evidence_scores(
    predicted: Sequence[Any], gold_explanations: Sequence[Sequence[Any]]
) -> dict[str, float]:
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for gold in gold_explanations:
        current = evidence_scores(predicted, gold)
        if current["f1"] > best["f1"]:
            best = current
    return best


def ranked_view(rows: Sequence[Mapping[str, Any]], score_mode: str) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        item = dict(row)
        item["adjusted_score"] = compute_adjusted_score(
            item, float(item["score"]), score_mode=score_mode, size_penalty=0.01
        )
        ranked.append(item)
    ranked.sort(key=lambda item: float(item["adjusted_score"]), reverse=True)
    return ranked


def candidate_record(row: Mapping[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": float(row["score"]),
        "adjusted_score": float(row["adjusted_score"]),
        "subgraph_size": int(row["subgraph_size"]),
        "subgraph_units": list(row["subgraph_units"]),
    }


def evaluate_ranking(
    ranked: Sequence[Mapping[str, Any]], gold_explanations: Sequence[Sequence[Any]]
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    by_k: dict[str, dict[str, float]] = {}
    prefixes = []
    for k in K_VALUES:
        aggregate = fixed_k_support_aggregate(ranked, k=k)
        scores = best_evidence_scores(aggregate["support_units"], gold_explanations)
        by_k[str(k)] = scores
        prefixes.append(
            {
                "k": k,
                "selected_k": int(aggregate["selected_k"]),
                "retrieved_evidence_units": list(aggregate["support_units"]),
                "retrieved_evidence_count": int(aggregate["final_support_size"]),
                **scores,
            }
        )
    return by_k, prefixes


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = load_gnn_checkpoint(str(checkpoint_path), device)
    model_name = checkpoint.get("model_name", "google/bert_uncased_L-2_H-128_A-2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = GNNSubgraphRetriever(
        model_name=model_name,
        node_symbolic_dim=checkpoint.get("node_symbolic_dim", 8),
        subgraph_symbolic_dim=checkpoint.get("subgraph_symbolic_dim", 8),
        gnn_hidden_dim=checkpoint.get("gnn_hidden_dim", 128),
        gnn_layers=checkpoint.get("gnn_layers", 2),
        classifier_hidden_dim=checkpoint.get("classifier_hidden_dim", 128),
        dropout=0.1,
        freeze_encoder=checkpoint.get("freeze_encoder", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, tokenizer, model


def aggregate_metrics(
    sums: Mapping[str, Mapping[str, Mapping[str, float]]], examples: int
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        method: {
            k: {metric: value / examples for metric, value in metric_sums.items()}
            for k, metric_sums in by_k.items()
        }
        for method, by_k in sums.items()
    }


def saturation(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    f1 = {k: float(metrics[str(k)]["f1"]) for k in (2, 3, 5)}
    spread = max(f1.values()) - min(f1.values())
    return {
        "definition": "absolute F1 spread across k={2,3,5} <= 0.01",
        "saturated": spread <= 0.01,
        "f1_spread": spread,
    }


def _pearson_size_score(ranked: Sequence[Mapping[str, Any]]) -> float:
    sizes = [float(row.get("subgraph_size", len(row.get("subgraph_units", [])))) for row in ranked]
    scores = [float(row["adjusted_score"]) for row in ranked]
    if len(sizes) < 2:
        return 0.0
    mean_size = sum(sizes) / len(sizes)
    mean_score = sum(scores) / len(scores)
    numerator = sum((size - mean_size) * (score - mean_score) for size, score in zip(sizes, scores))
    size_ss = sum((size - mean_size) ** 2 for size in sizes)
    score_ss = sum((score - mean_score) ** 2 for score in scores)
    denominator = (size_ss * score_ss) ** 0.5
    return numerator / denominator if denominator else 0.0


def ranking_diagnostic(ranked: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [
        (rank, row)
        for rank, row in enumerate(ranked, start=1)
        if bool(row.get("contains_any_gold_explanation"))
    ]
    best_rank, best = complete[0] if complete else (None, None)
    top = ranked[0]
    return {
        "candidate_count": len(ranked),
        "top1_complete": bool(top.get("contains_any_gold_explanation")),
        "top1_adjusted_score": float(top["adjusted_score"]),
        "top1_subgraph_size": int(top.get("subgraph_size", len(top.get("subgraph_units", [])))),
        "best_complete_rank": best_rank,
        "best_complete_adjusted_score": float(best["adjusted_score"]) if best is not None else None,
        "best_complete_subgraph_size": (
            int(best.get("subgraph_size", len(best.get("subgraph_units", []))))
            if best is not None
            else None
        ),
        "top1_minus_best_complete_score_gap": (
            float(top["adjusted_score"]) - float(best["adjusted_score"])
            if best is not None
            else None
        ),
        "candidate_size_score_pearson": _pearson_size_score(ranked),
    }


def summary_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Production Generator D — DEV retrieval k sensitivity",
        "",
        "All values are macro-averaged retrieval metrics over the deduplicated top-k evidence union.",
        "",
        "| Dataset | Method | k | Precision | Recall | F1 | ΔF1 vs k=1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in metrics["datasets"]:
        for method in ("gnn_only", "sageqa_final"):
            base = dataset["methods"][method]["metrics"]["1"]["f1"]
            for k in K_VALUES:
                row = dataset["methods"][method]["metrics"][str(k)]
                lines.append(
                    f"| {dataset['dataset']} | {method} | {k} | {row['precision']:.6f} | "
                    f"{row['recall']:.6f} | {row['f1']:.6f} | {row['f1'] - base:+.6f} |"
                )
    lines.extend(
        [
            "",
            "## Descriptive selection",
            "",
            "| Dataset | Method | Best observed DEV k | Best F1 | k=2/3/5 saturated |",
            "|---|---|---:|---:|---|",
        ]
    )
    for dataset in metrics["datasets"]:
        for method in ("gnn_only", "sageqa_final"):
            result = dataset["methods"][method]
            lines.append(
                f"| {dataset['dataset']} | {method} | {result['best_dev_k_by_f1']} | "
                f"{result['best_dev_f1']:.6f} | {str(result['saturation_k_2_3_5']['saturated']).lower()} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit = git_output("rev-parse", "HEAD")
    status_before = git_output("status", "--short")

    input_paths = []
    checkpoint_paths = []
    for _, data_dir, checkpoint_dir, _, _ in DATASETS:
        dev_path = args.data_root / data_dir / "dev_subgraph_retrieval.jsonl"
        checkpoint_path = args.checkpoint_root / checkpoint_dir / "best_model.pt"
        if not dev_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing production input: {dev_path} or {checkpoint_path}")
        input_paths.append(dev_path)
        checkpoint_paths.append(checkpoint_path)
    before_hashes = {str(path): sha256(path) for path in input_paths + checkpoint_paths}

    metadata: dict[str, Any] = {
        "schema_version": "production_generator_d_dev_k_sensitivity_v1",
        "split": "dev",
        "k_values": list(K_VALUES),
        "data_root": str(args.data_root),
        "checkpoint_root": str(args.checkpoint_root),
        "code_commit_hash": commit,
        "git_status_before": status_before.splitlines(),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "candidate_generation_rerun": False,
        "adaptive_k_applied": False,
        "answer_generation_invoked": False,
        "accessed_data_files": [],
        "datasets": {},
    }
    output: dict[str, Any] = {
        "schema_version": "production_generator_d_dev_k_sensitivity_v1",
        "split": "dev",
        "k_values": list(K_VALUES),
        "saturation_definition": "absolute F1 spread across k={2,3,5} <= 0.01",
        "datasets": [],
    }
    ranking_path = args.output_dir / "per_example_rankings.jsonl"

    with ranking_path.open("w", encoding="utf-8") as ranking_handle:
        for dataset, data_dir, checkpoint_dir, domain, final_mode in DATASETS:
            dev_path = args.data_root / data_dir / "dev_subgraph_retrieval.jsonl"
            checkpoint_path = args.checkpoint_root / checkpoint_dir / "best_model.pt"
            if "test" in dev_path.name.lower() or "test" in str(dev_path.parent).lower():
                raise AssertionError(f"TEST path rejected: {dev_path}")
            metadata["accessed_data_files"].append(str(dev_path))
            checkpoint, tokenizer, model = load_model(checkpoint_path, device)
            method_sums = {
                method: {str(k): defaultdict(float) for k in K_VALUES}
                for method in ("gnn_only", "sageqa_final")
            }
            example_count = 0
            row_count = 0
            for example_id, raw_rows in grouped_jsonl(dev_path):
                row_count += len(raw_rows)
                prepared = prepare_examples(raw_rows, candidate_selection="inference")
                if len(prepared) != 1 or prepared[0]["example_id"] != example_id:
                    raise ValueError(f"Unexpected preparation result for {example_id}")
                example = prepared[0]
                encoded = encode_example_graph(
                    model, tokenizer, example, device, max_length=args.max_length
                )
                scored = []
                with torch.no_grad():
                    for start in range(
                        0, len(example["candidate_rows"]), args.candidate_batch_size
                    ):
                        batch = example["candidate_rows"][start : start + args.candidate_batch_size]
                        probabilities = (
                            score_candidate_rows(model, encoded, batch, device)["probs"]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        for row, probability in zip(batch, probabilities):
                            item = dict(row)
                            item["score"] = float(probability)
                            scored.append(item)
                gold = list(scored[0].get("gold_explanations", []) or [])
                if not gold:
                    support = list(scored[0].get("gold_support_units", []) or [])
                    gold = [support] if support else []
                if not gold:
                    raise ValueError(f"DEV example has no gold support: {example_id}")

                rankings = {
                    "gnn_only": ranked_view(scored, "neural"),
                    "sageqa_final": ranked_view(scored, final_mode),
                }
                for method, ranked in rankings.items():
                    by_k, prefixes = evaluate_ranking(ranked, gold)
                    for k, values in by_k.items():
                        for metric, value in values.items():
                            method_sums[method][k][metric] += value
                    record = {
                        "dataset": dataset,
                        "domain": domain,
                        "split": "dev",
                        "example_id": example_id,
                        "method": method,
                        "score_mode": "neural" if method == "gnn_only" else final_mode,
                        "gold_explanations": gold,
                        "ranked_candidates": [
                            candidate_record(row, rank)
                            for rank, row in enumerate(ranked[:5], start=1)
                        ],
                        "ranking_diagnostic": ranking_diagnostic(ranked),
                        "prefix_evaluation": prefixes,
                    }
                    ranking_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                example_count += 1
                if example_count % 50 == 0:
                    print(f"{dataset}: {example_count} DEV examples", flush=True)

            averaged = aggregate_metrics(method_sums, example_count)
            methods = {}
            for method in ("gnn_only", "sageqa_final"):
                observed = averaged[method]
                best_k = min(K_VALUES, key=lambda k: (-observed[str(k)]["f1"], k))
                methods[method] = {
                    "score_mode": "neural" if method == "gnn_only" else final_mode,
                    "metrics": observed,
                    "best_dev_k_by_f1": best_k,
                    "best_dev_f1": observed[str(best_k)]["f1"],
                    "delta_f1_k1_to_k2": observed["2"]["f1"] - observed["1"]["f1"],
                    "delta_f1_k1_to_k3": observed["3"]["f1"] - observed["1"]["f1"],
                    "saturation_k_2_3_5": saturation(observed),
                }
            output["datasets"].append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "dev_examples": example_count,
                    "methods": methods,
                }
            )
            metadata["datasets"][dataset] = {
                "dev_path": str(dev_path),
                "dev_sha256": before_hashes[str(dev_path)],
                "dev_candidate_rows": row_count,
                "dev_examples": example_count,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": before_hashes[str(checkpoint_path)],
                "checkpoint_model_name": checkpoint.get("model_name"),
                "checkpoint_training_objective": checkpoint.get("training_objective"),
                "checkpoint_score_mode": checkpoint.get("score_mode"),
                "final_reranking_score_mode": final_mode,
                "inference_candidate_cap": 320,
            }
            del model, tokenizer, checkpoint

    after_hashes = {str(path): sha256(path) for path in input_paths + checkpoint_paths}
    metadata["input_hashes_unchanged"] = before_hashes == after_hashes
    metadata["generator_d_inputs_unchanged"] = all(
        before_hashes[str(path)] == after_hashes[str(path)] for path in input_paths
    )
    metadata["checkpoints_unchanged"] = all(
        before_hashes[str(path)] == after_hashes[str(path)] for path in checkpoint_paths
    )
    metadata["test_rows_read"] = 0
    metadata["test_gold_accessed"] = False
    metadata["adaptive_k_ready"] = True
    metadata["adaptive_k_readiness_fields"] = [
        "example_id",
        "domain",
        "method",
        "score_mode",
        "gold_explanations",
        "ranked_candidates.rank",
        "ranked_candidates.score",
        "ranked_candidates.adjusted_score",
        "ranked_candidates.subgraph_units",
        "prefix_evaluation",
    ]
    metadata["git_status_after"] = git_output("status", "--short").splitlines()
    json_dump(args.output_dir / "metrics.json", output)
    json_dump(args.output_dir / "checkpoint_metadata.json", metadata)
    (args.output_dir / "summary.md").write_text(summary_markdown(output), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "complete": True}))


if __name__ == "__main__":
    main()
