"""Audit production-objective loss magnitudes on frozen Generator-D DEV data."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_gnn_subgraph_retriever import (
    encode_example_graph,
    listwise_soft_target_loss,
    pairwise_ranking_loss,
    prepare_examples,
    score_candidate_rows,
)
from evaluation.run_production_dev_k_sensitivity import grouped_jsonl, load_model


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


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def train_label_counts(path: Path) -> tuple[int, int]:
    positive = 0
    total = 0
    for row in read_jsonl(path):
        total += 1
        positive += int(float(row["label"]) > 0.0)
    return positive, total - positive


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/production_generator_d_v1"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/production_generator_d_v1")
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--candidate-batch-size", type=int, default=256)
    parser.add_argument("--max-dev-examples-per-dataset", type=int, default=25)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "outputs/diagnostics/production_generator_d_v1_listwise_corrected_dev/loss_scale_audit.json"
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_losses = defaultdict(list)
    per_dataset = {}
    for dataset, checkpoint_dir in DATASETS:
        train_path = args.data_root / dataset / "train_subgraph_retrieval.jsonl"
        dev_path = args.data_root / dataset / "dev_subgraph_retrieval.jsonl"
        checkpoint_path = args.checkpoint_root / checkpoint_dir / "best_model.pt"
        positive, negative = train_label_counts(train_path)
        pos_weight = negative / max(positive, 1)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        _, tokenizer, model = load_model(checkpoint_path, device)

        dataset_losses = defaultdict(list)
        for example_index, (example_id, raw_rows) in enumerate(grouped_jsonl(dev_path)):
            if example_index >= args.max_dev_examples_per_dataset:
                break
            prepared = prepare_examples(raw_rows, candidate_selection="inference")
            if len(prepared) != 1:
                raise ValueError(f"Unexpected preparation result for {dataset}/{example_id}")
            example = prepared[0]
            encoded = encode_example_graph(model, tokenizer, example, device, args.max_length)
            logits = []
            with torch.no_grad():
                for start in range(0, len(example["candidate_rows"]), args.candidate_batch_size):
                    batch = example["candidate_rows"][start : start + args.candidate_batch_size]
                    logits.append(score_candidate_rows(model, encoded, batch, device)["logits"])
            score_tensor = torch.cat(logits)
            target_tensor = torch.tensor(
                [float(row["rank_target"]) for row in example["candidate_rows"]],
                dtype=torch.float,
                device=device,
            )
            label_tensor = torch.tensor(
                [float(row["label"]) for row in example["candidate_rows"]],
                dtype=torch.float,
                device=device,
            )
            random.seed(42)
            losses = {
                "margin": float(pairwise_ranking_loss(score_tensor, target_tensor, 0.2, 512)),
                "bce": float(bce(score_tensor, label_tensor)),
                "listwise": float(listwise_soft_target_loss(score_tensor, target_tensor, 0.1)),
                "candidate_count": len(example["candidate_rows"]),
            }
            for name, value in losses.items():
                dataset_losses[name].append(value)
                all_losses[name].append(value)

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        per_dataset[dataset] = {
            "train_positive_rows": positive,
            "train_negative_rows": negative,
            "train_derived_bce_pos_weight": pos_weight,
            **{name: distribution(values) for name, values in dataset_losses.items()},
        }
        print(f"{dataset}: audited {len(dataset_losses['listwise'])} DEV examples", flush=True)

    summary = {name: distribution(values) for name, values in all_losses.items()}
    weighted_reference = {
        "2.0_x_margin_mean": 2.0 * summary["margin"]["mean"],
        "0.2_x_bce_mean": 0.2 * summary["bce"]["mean"],
        **{
            f"{weight}_x_listwise_mean": weight * summary["listwise"]["mean"]
            for weight in (0.1, 0.25, 0.5, 1.0)
        },
    }
    result = {
        "schema_version": "production_generator_d_v1_corrected_listwise_loss_scale_v1",
        "split_inputs": [
            "train labels for BCE class weight",
            "dev candidate scores and supervision",
        ],
        "temperature": 0.1,
        "ranking_margin": 0.2,
        "max_pairs": 512,
        "fixed_weights": {"ranking": 2.0, "bce": 0.2},
        "sampling": {
            "method": "first N prepared DEV examples in frozen file order",
            "max_examples_per_dataset": args.max_dev_examples_per_dataset,
            "purpose": "loss-scale audit only; no retrieval metrics computed",
        },
        "summary": summary,
        "weighted_reference": weighted_reference,
        "per_dataset": per_dataset,
        "retrieval_metrics_used_for_grid_choice": False,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output_path), "weighted_reference": weighted_reference}, indent=2
        )
    )


if __name__ == "__main__":
    main()
