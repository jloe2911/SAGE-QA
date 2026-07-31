import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.gnn_subgraph_retriever import GNNSubgraphRetriever
from training.train_gnn_subgraph_retriever import (
    load_jsonl,
    prepare_examples,
    evaluate,
)
from utils.eval_splits import split_support_metrics


def load_gnn_checkpoint(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # New checkpoint format from train_gnn_subgraph_retriever.py
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint

    # Fallback if a raw state_dict was saved
    return {
        "model_state_dict": checkpoint,
        "model_name": "google/bert_uncased_L-2_H-128_A-2",
        "node_symbolic_dim": 8,
        "subgraph_symbolic_dim": 8,
        "gnn_hidden_dim": 128,
        "gnn_layers": 2,
        "classifier_hidden_dim": 128,
        "architecture_version": 1,
        "freeze_encoder": False,
        "score_mode": "neural",
        "size_penalty": 0.01,
    }


def print_samples(details, n=3):
    print("\nSample predictions:")
    for item in details[:n]:
        print(json.dumps(item, indent=2, ensure_ascii=False))


def summarize_failure_modes(details):
    counts = Counter(item.get("failure_mode", "unknown") for item in details)
    failures = [
        {
            "example_id": item["example_id"],
            "question": item["question"],
            "failure_mode": item.get("failure_mode", "unknown"),
            "oracle_f1": item.get("candidate_oracle_f1"),
            "best_exact_rank": item.get("best_exact_rank"),
            "best_entailing_rank": item.get("best_entailing_rank"),
            "top1_f1": item.get("top1_best_set_f1_to_gold"),
            "top1_query_entailed": item.get("top1_query_entailed"),
            "top1_units": item.get("top1_subgraph_units", []),
        }
        for item in details
        if item.get("failure_mode") != "pass_exact"
    ]
    return {
        "counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def evaluate_file(
    split_name: str,
    path: str,
    model,
    tokenizer,
    device,
    max_length: int,
    candidate_batch_size: int,
    score_mode: str,
    size_penalty: float,
    max_examples: int = 0,
    source_name: str | None = None,
):
    rows = load_jsonl(
        path,
        source_name=source_name,
        max_examples=max_examples,
    )
    # Evaluation must rank the complete materialized candidate pool. Reusing
    # training-time negative subsampling makes metrics stochastic and can hide
    # the model's actual highest-scoring false positives.
    examples = prepare_examples(rows, subsample_candidates=False)

    criterion = nn.BCEWithLogitsLoss()

    metrics, details = evaluate(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        bce_criterion=criterion,
        max_length=max_length,
        candidate_batch_size=candidate_batch_size,
        score_mode=score_mode,
        size_penalty=size_penalty,
    )

    print(f"\n=== {split_name} ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print_samples(details, n=3)

    split_metrics = split_support_metrics(details)
    failure_summary = summarize_failure_modes(details)

    print(f"\n=== {split_name} SPLIT METRICS ===")
    print(json.dumps(split_metrics, indent=2, ensure_ascii=False))
    print(f"\n=== {split_name} FAILURE DIAGNOSTICS ===")
    print(json.dumps(failure_summary, indent=2, ensure_ascii=False))

    return metrics, details, split_metrics, failure_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        type=str,
        default="data/train_subgraph_retrieval.jsonl",
    )
    parser.add_argument(
        "--dev-path",
        type=str,
        default="data/dev_subgraph_retrieval.jsonl",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/test_subgraph_retrieval.jsonl",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/gnn_subgraph_retriever/best_model.pt",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Overrides the model name saved in the checkpoint.",
    )

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--candidate-batch-size", type=int, default=64)

    parser.add_argument(
        "--score-mode",
        type=str,
        default=None,
        choices=[
            "neural",
            "minimality_adjusted",
            "completeness_adjusted",
            "sageqa_compact",
            "sageqa_text_chain",
            "sageqa_proof",
        ],
        help="Overrides checkpoint score mode.",
    )
    parser.add_argument(
        "--size-penalty",
        type=float,
        default=None,
        help="Overrides checkpoint size penalty.",
    )

    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-dev-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--source-name", type=str, default=None)

    parser.add_argument("--save-details", action="store_true")
    parser.add_argument(
        "--details-dir",
        type=str,
        default="outputs/gnn_subgraph_eval",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_gnn_checkpoint(str(checkpoint_path), device=device)

    model_name = args.model_name or checkpoint.get(
        "model_name",
        "google/bert_uncased_L-2_H-128_A-2",
    )

    score_mode = args.score_mode or checkpoint.get("score_mode", "neural")
    size_penalty = (
        args.size_penalty
        if args.size_penalty is not None
        else checkpoint.get("size_penalty", 0.01)
    )

    print(f"Model name: {model_name}")
    print(f"Score mode: {score_mode}")
    print(f"Size penalty: {size_penalty}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = GNNSubgraphRetriever(
        model_name=model_name,
        node_symbolic_dim=checkpoint.get("node_symbolic_dim", 8),
        subgraph_symbolic_dim=checkpoint.get("subgraph_symbolic_dim", 8),
        gnn_hidden_dim=checkpoint.get("gnn_hidden_dim", 128),
        gnn_layers=checkpoint.get("gnn_layers", 2),
        classifier_hidden_dim=checkpoint.get("classifier_hidden_dim", 128),
        dropout=0.1,
        freeze_encoder=checkpoint.get("freeze_encoder", False),
        architecture_version=checkpoint.get("architecture_version", 1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_results = {}

    split_specs = [
        ("TRAIN", args.train_path, args.max_train_examples),
        ("DEV", args.dev_path, args.max_dev_examples),
        ("TEST", args.test_path, args.max_test_examples),
    ]

    for split_name, path, max_examples in split_specs:
        if not Path(path).exists():
            print(f"[WARN] Skipping missing split file: {path}")
            continue

        metrics, details, split_metrics, failure_summary = evaluate_file(
            split_name=split_name,
            path=path,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=args.max_length,
            candidate_batch_size=args.candidate_batch_size,
            score_mode=score_mode,
            size_penalty=size_penalty,
            max_examples=max_examples,
            source_name=args.source_name,
        )

        all_results[split_name.lower()] = {
            "metrics": metrics,
            "details": details,
            "split_metrics": split_metrics,
            "failure_summary": failure_summary,
        }

    if args.save_details:
        out_dir = Path(args.details_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, result in all_results.items():
            with open(
                out_dir / f"{split_name}_metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result["metrics"], f, indent=2, ensure_ascii=False)

            with open(
                out_dir / f"{split_name}_split_metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result["split_metrics"], f, indent=2, ensure_ascii=False)

            with open(
                out_dir / f"{split_name}_details.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result["details"], f, indent=2, ensure_ascii=False)

            with open(
                out_dir / f"{split_name}_failure_summary.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    result["failure_summary"],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        # Convenience copy for collect_final_results.py
        if "test" in all_results:
            with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    all_results["test"]["split_metrics"],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        print(f"\nSaved evaluation details to: {out_dir}")


if __name__ == "__main__":
    main()
