"""One frozen DEV evaluation of the final static-hard refinement."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.adaptive_support_aggregation_v2 import adaptive_v2_support_aggregate, load_domain_policy
from evaluation.run_production_dev_k_sensitivity import (
    DATASETS, K_VALUES, best_evidence_scores, candidate_record, evaluate_ranking,
    grouped_jsonl, load_model, ranked_view,
)
from training.train_gnn_subgraph_retriever import encode_example_graph, prepare_examples, select_reserved_hard_pair


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score_all_once(model, encoded, rows, device):
    with torch.no_grad():
        nodes = model.encode_graph(encoded["node_text_embeddings"], encoded["node_symbolic_features"], encoded["edge_index"])
        pooled = torch.stack([
            model.pool_subgraph(nodes, torch.tensor(row["subgraph_node_ids"], dtype=torch.long, device=device))
            for row in rows
        ])
        symbolic = torch.tensor([row["symbolic_features"] for row in rows], dtype=torch.float, device=device)
        symbolic_repr = model.subgraph_feature_projection(symbolic)
        query = encoded["query_embedding"].unsqueeze(0).expand(len(rows), -1)
        logits = model.classifier(torch.cat([query, pooled, symbolic_repr], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits).cpu().tolist()


def mean_metrics(rows):
    n = len(rows)
    return {key: sum(float(row[key]) for row in rows) / n for key in ("precision", "recall", "f1")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/production_generator_d_static_hard_v1")
    parser.add_argument("--v1-rankings", type=Path, default=ROOT / "outputs/development_runs/production_generator_d_v1_k_sensitivity/per_example_rankings.jsonl")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "outputs/diagnostics/production_generator_d_v1_causal_retrieval_diagnosis/causal_failure_taxonomy.json")
    parser.add_argument("--adaptive-policy", type=Path, default=ROOT / "outputs/development_runs/production_generator_d_v1_adaptive_k")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/final_model_development/production_generator_d_static_hard_v1_dev")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = {domain: load_domain_policy(args.adaptive_policy, domain=domain) for domain in ("text", "ontology")}
    records = []
    per_dataset = []
    ranking_path = args.output_dir / "per_example_rankings.jsonl"
    with ranking_path.open("x", encoding="utf-8") as handle:
        for dataset, data_dir, checkpoint_dir, domain, final_mode in DATASETS:
            checkpoint, tokenizer, model = load_model(args.checkpoint_root / checkpoint_dir / "best_model.pt", device)
            method_rows = {method: [] for method in ("gnn_only", "sageqa_final", "adaptive_sageqa")}
            ordering = {"eligible_complete_vs_partial": 0, "positive_strictly_above_partial": 0}
            containment = {"gnn_only": 0, "sageqa_final": 0}
            examples = 0
            for example_id, raw_rows in grouped_jsonl(args.data_root / data_dir / "dev_subgraph_retrieval.jsonl"):
                prepared = prepare_examples(raw_rows, candidate_selection="inference")
                if len(prepared) != 1:
                    raise ValueError(example_id)
                example = prepared[0]
                with torch.no_grad():
                    encoded = encode_example_graph(model, tokenizer, example, device, max_length=128)
                probabilities = score_all_once(model, encoded, example["candidate_rows"], device)
                scored = []
                for row, probability in zip(example["candidate_rows"], probabilities):
                    item = dict(row); item["score"] = float(probability); scored.append(item)
                gold = list(scored[0].get("gold_explanations", []) or [])
                if not gold:
                    support = list(scored[0].get("gold_support_units", []) or []); gold = [support] if support else []
                rankings = {"gnn_only": ranked_view(scored, "neural"), "sageqa_final": ranked_view(scored, final_mode)}
                pair = select_reserved_hard_pair(scored)
                if pair is not None and float(pair[1]["rank_target"]) > 0.0:
                    ordering["eligible_complete_vs_partial"] += 1
                    ordering["positive_strictly_above_partial"] += int(float(pair[0]["score"]) > float(pair[1]["score"]))
                for method, ranked in rankings.items():
                    by_k, _ = evaluate_ranking(ranked, gold)
                    values = by_k["1"]
                    method_rows[method].append(values)
                    containment[method] += int(bool(ranked[0].get("contains_any_gold_explanation")))
                adaptive = adaptive_v2_support_aggregate(rankings["sageqa_final"], policy=policies[domain])
                adaptive_values = best_evidence_scores(adaptive["support_units"], gold)
                method_rows["adaptive_sageqa"].append(adaptive_values)
                record = {
                    "dataset": dataset, "domain": domain, "split": "dev", "example_id": example_id,
                    "gold_explanations": gold,
                    "gnn_only": {"metrics_k1": method_rows["gnn_only"][-1], "top1_complete": bool(rankings["gnn_only"][0].get("contains_any_gold_explanation")), "top5": [candidate_record(row, rank) for rank, row in enumerate(rankings["gnn_only"][:5], 1)]},
                    "sageqa_final": {"metrics_k1": method_rows["sageqa_final"][-1], "top1_complete": bool(rankings["sageqa_final"][0].get("contains_any_gold_explanation")), "top5": [candidate_record(row, rank) for rank, row in enumerate(rankings["sageqa_final"][:5], 1)]},
                    "adaptive_sageqa": {"selected_k": adaptive["selected_k"], "metrics": adaptive_values},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record); examples += 1
            per_dataset.append({
                "dataset": dataset, "domain": domain, "examples": examples,
                "gnn_k1": mean_metrics(method_rows["gnn_only"]),
                "sageqa_k1": mean_metrics(method_rows["sageqa_final"]),
                "adaptive_sageqa": mean_metrics(method_rows["adaptive_sageqa"]),
                "complete_support_containment": {key: value / examples for key, value in containment.items()},
                "complete_vs_hard_partial_ordering": {**ordering, "accuracy": ordering["positive_strictly_above_partial"] / max(1, ordering["eligible_complete_vs_partial"])},
            })
    baseline = {}
    with args.v1_rankings.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["method"] in {"gnn_only", "sageqa_final"}:
                baseline[(row["example_id"], row["method"])] = row
    diagnosed = {
        item["example_id"] for item in json.loads(args.taxonomy.read_text(encoding="utf-8"))["assignments"]
        if item["taxonomy"] == "5_gnn_misranking"
    }
    corrected = worsened = unchanged = 0
    for record in records:
        if record["example_id"] not in diagnosed:
            continue
        if record["sageqa_final"]["top1_complete"]:
            corrected += 1
        old = baseline[(record["example_id"], "sageqa_final")]["prefix_evaluation"][0]["f1"]
        new = record["sageqa_final"]["metrics_k1"]["f1"]
        if new < old - 1e-12: worsened += 1
        elif abs(new - old) <= 1e-12: unchanged += 1
    macro = {}
    for method in ("gnn_k1", "sageqa_k1", "adaptive_sageqa"):
        macro[method] = {metric: sum(row[method][metric] for row in per_dataset) / len(per_dataset) for metric in ("precision", "recall", "f1")}
    report = {
        "schema_version": "sageqa_static_hard_refinement_dev_v1", "split": "dev", "status": "complete",
        "per_dataset": per_dataset, "macro": macro,
        "diagnosed_443": {"expected": 443, "matched": len(diagnosed), "corrected_to_complete_sageqa_top1": corrected, "worsened_sageqa_k1_f1": worsened, "unchanged_sageqa_k1_f1": unchanged},
        "adaptive_policy_reused_unchanged": str(args.adaptive_policy),
        "test_accessed": False, "answer_generation_run": False,
    }
    if len(diagnosed) != 443:
        raise AssertionError(len(diagnosed))
    write_json(args.output_dir / "metrics.json", report)
    print(json.dumps({"status": "complete", "macro": macro, "diagnosed_443": report["diagnosed_443"]}))


if __name__ == "__main__":
    main()
