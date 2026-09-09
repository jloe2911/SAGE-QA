"""Exact GNN pair scoring supplement for the causal Generator-D DEV diagnosis.

Scores the matched gold-free 320-candidate admission on TRAIN and DEV using
the frozen production checkpoints.  It never trains, reads TEST, regenerates
candidates, changes checkpoints, or invokes answer generation.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.eval_gnn_subgraph_retriever import load_gnn_checkpoint
from data_processing.retrieval_contracts import cap_inference_candidate_rows
from models.gnn_subgraph_retriever import GNNSubgraphRetriever
from training.train_gnn_subgraph_retriever import (
    encode_example_graph,
    prepare_examples,
    score_candidate_rows,
)


ROOT = Path(__file__).resolve().parents[1]
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def grouped_jsonl(path: Path):
    current = None
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_id = str(row["example_id"])
            if current is None:
                current = example_id
            if example_id != current:
                yield current, rows
                current, rows = example_id, []
            rows.append(row)
    if current is not None:
        yield current, rows


def load_model(path: Path, device: torch.device):
    checkpoint = load_gnn_checkpoint(str(path), device)
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
    return tokenizer, model


def score_rows(model, tokenizer, example, device, batch_size: int, max_length: int):
    encoded = encode_example_graph(model, tokenizer, example, device, max_length=max_length)
    output = []
    with torch.no_grad():
        for start in range(0, len(example["candidate_rows"]), batch_size):
            batch = example["candidate_rows"][start : start + batch_size]
            scores = score_candidate_rows(model, encoded, batch, device)["probs"].detach().cpu().tolist()
            for row, score in zip(batch, scores):
                shaped = dict(row)
                shaped["score"] = float(score)
                output.append(shaped)
    return output


def pair_accuracy(better: Sequence[float], worse: Sequence[float]) -> tuple[float | None, int]:
    if not better or not worse:
        return None, 0
    ordered = sorted(float(value) for value in worse)
    credit = 0.0
    for value in better:
        left = bisect.bisect_left(ordered, value)
        right = bisect.bisect_right(ordered, value)
        credit += left + 0.5 * (right - left)
    pairs = len(better) * len(worse)
    return credit / pairs, pairs


def candidate_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        target = float(row.get("rank_target", 0.0))
        score = float(row["score"])
        exact = bool(row.get("exact_match_any_gold"))
        complete = bool(row.get("contains_any_gold_explanation")) or target >= 0.9
        if complete:
            groups["complete"].append(score)
            groups["exact" if exact else "complete_superset"].append(score)
        elif target > 0.0:
            groups["partial"].append(score)
        else:
            groups["irrelevant"].append(score)
    return groups


def update_pair_stats(stats: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    groups = candidate_groups(rows)
    comparisons = {
        "complete_vs_partial": (groups["complete"], groups["partial"]),
        "exact_complete_vs_complete_superset": (groups["exact"], groups["complete_superset"]),
        "complete_vs_irrelevant": (groups["complete"], groups["irrelevant"]),
    }
    for name, (better, worse) in comparisons.items():
        accuracy, pairs = pair_accuracy(better, worse)
        if accuracy is None:
            continue
        item = stats[name]
        item["examples_with_comparison"] += 1
        item["pairs"] += pairs
        item["pair_credit"] += accuracy * pairs
        item["example_accuracy_sum"] += accuracy


def finalize_pair_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name, item in stats.items():
        examples = int(item["examples_with_comparison"])
        pairs = int(item["pairs"])
        result[name] = {
            "examples_with_comparison": examples,
            "pairs": pairs,
            "pair_weighted_accuracy": item["pair_credit"] / pairs if pairs else None,
            "example_macro_accuracy": item["example_accuracy_sum"] / examples if examples else None,
            "tie_credit": 0.5,
        }
    return result


def feature_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    a = [float(value) for value in left.get("symbolic_features", [])]
    b = [float(value) for value in right.get("symbolic_features", [])]
    length = max(len(a), len(b))
    a += [0.0] * (length - len(a))
    b += [0.0] * (length - len(b))
    l1 = sum(abs(x - y) for x, y in zip(a, b))
    return {
        "symbolic_feature_l1": l1,
        "same_symbolic_features": l1 <= 1e-12,
        "same_subgraph_size": len(left["subgraph_units"]) == len(right["subgraph_units"]),
        "nearly_indistinguishable_hand_features": l1 <= 0.05,
    }


def detail_row(dataset: str, example_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: (-float(row["score"]), int(row.get("generation_rank", 2**31 - 1))))
    top = ranked[0]
    complete = [row for row in ranked if bool(row.get("contains_any_gold_explanation")) or float(row.get("rank_target", 0.0)) >= 0.9]
    if not complete:
        raise AssertionError(f"Expected admitted complete candidate: {example_id}")
    best = complete[0]
    best_rank = ranked.index(best) + 1

    def shaped(row: Mapping[str, Any]) -> dict[str, Any]:
        exact = bool(row.get("exact_match_any_gold"))
        sufficient = bool(row.get("contains_any_gold_explanation")) or float(row.get("rank_target", 0.0)) >= 0.9
        target = float(row.get("rank_target", 0.0))
        category = "exact" if exact else "complete-superset" if sufficient else "partial" if target > 0 else "irrelevant"
        return {
            "score": float(row["score"]),
            "subgraph_units": row["subgraph_units"],
            "candidate_size": len(row["subgraph_units"]),
            "generator_pre_rank_score": float(row.get("candidate_pre_rank_score", 0.0)),
            "generation_rank": int(row.get("generation_rank", -1)),
            "rank_target": target,
            "category": category,
            "symbolic_features": row.get("symbolic_features", []),
        }

    return {
        "dataset": dataset,
        "example_id": example_id,
        "rank1_candidate": shaped(top),
        "best_complete_candidate": shaped(best),
        "best_complete_gnn_rank": best_rank,
        "top1_minus_complete_score_gap": float(top["score"]) - float(best["score"]),
        "complete_has_strictly_better_target": float(best.get("rank_target", 0.0)) > float(top.get("rank_target", 0.0)),
        "representation_comparison": feature_distance(top, best),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/production_generator_d_v1")
    parser.add_argument(
        "--diagnosis-dir",
        type=Path,
        default=ROOT / "outputs/diagnostics/production_generator_d_v1_causal_retrieval_diagnosis",
    )
    parser.add_argument("--candidate-batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    taxonomy = json.loads((args.diagnosis_dir / "causal_failure_taxonomy.json").read_text(encoding="utf-8"))
    detail_ids = {
        row["example_id"]
        for row in taxonomy["assignments"]
        if row["taxonomy"] in {"5_gnn_misranking", "6_symbolic_reranking_degradation"}
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    details = []
    by_split_dataset: dict[str, Any] = {split: {} for split in ("train", "dev")}
    pooled = {
        split: defaultdict(lambda: {"examples_with_comparison": 0, "pairs": 0, "pair_credit": 0.0, "example_accuracy_sum": 0.0})
        for split in ("train", "dev")
    }

    for dataset, checkpoint_dir in DATASETS:
        print(f"{dataset}: loading frozen checkpoint", flush=True)
        tokenizer, model = load_model(args.checkpoint_root / checkpoint_dir / "best_model.pt", device)
        for split in ("train", "dev"):
            stats = defaultdict(lambda: {"examples_with_comparison": 0, "pairs": 0, "pair_credit": 0.0, "example_accuracy_sum": 0.0})
            count = 0
            for example_id, raw_rows in grouped_jsonl(args.data_root / dataset / f"{split}_subgraph_retrieval.jsonl"):
                prepared = prepare_examples(raw_rows, candidate_selection="inference")
                if len(prepared) != 1:
                    raise ValueError(f"Unexpected prepared example count: {example_id}")
                scored = score_rows(model, tokenizer, prepared[0], device, args.candidate_batch_size, args.max_length)
                admitted_lookup = {
                    tuple(sorted(str(unit) for unit in row.get("subgraph_units", []))): row
                    for row in cap_inference_candidate_rows(raw_rows, max_candidates=320)
                }
                for row in scored:
                    source = admitted_lookup[tuple(sorted(str(unit) for unit in row.get("subgraph_units", [])))]
                    row["candidate_pre_rank_score"] = float(source.get("candidate_pre_rank_score", 0.0))
                    row["generation_rank"] = int(source.get("generation_rank", -1))
                update_pair_stats(stats, scored)
                update_pair_stats(pooled[split], scored)
                if split == "dev" and example_id in detail_ids:
                    details.append(detail_row(dataset, example_id, scored))
                count += 1
                if count % 100 == 0:
                    print(f"{dataset} {split}: {count}", flush=True)
            by_split_dataset[split][dataset] = {
                "examples": count,
                "candidate_selection": "matched gold-free inference admission cap 320",
                "ordering_accuracy": finalize_pair_stats(stats),
            }
        del model, tokenizer

    expected = len(detail_ids)
    if len(details) != expected:
        raise AssertionError(f"pair detail mismatch: {len(details)} != {expected}")
    output = {
        "schema_version": "production_generator_d_v1_causal_gnn_pair_scoring_v1",
        "split_scope": ["train", "dev"],
        "test_rows_read": 0,
        "training_run": False,
        "model_inference_only": True,
        "candidate_generation_rerun": False,
        "answer_generation_run": False,
        "candidate_admission": "actual cap_inference_candidate_rows via prepare_examples(candidate_selection='inference'), max 320",
        "pair_definitions": {
            "complete_vs_partial": "contains a complete explanation / rank_target >= 0.9 versus 0 < rank_target < 0.9",
            "exact_complete_vs_complete_superset": "exact complete candidate versus sufficient non-exact complete superset",
            "complete_vs_irrelevant": "contains a complete explanation / rank_target >= 0.9 versus rank_target == 0",
        },
        "pooled": {split: finalize_pair_stats(pooled[split]) for split in ("train", "dev")},
        "by_split_dataset": by_split_dataset,
        "dev_failure_pairs": sorted(details, key=lambda row: (row["dataset"], row["example_id"])),
    }
    write_json(args.diagnosis_dir / "gnn_ranking_analysis.json", output)


if __name__ == "__main__":
    main()
