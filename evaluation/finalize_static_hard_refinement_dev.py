"""Matched v1 comparison and immutable retain/reject decision for static-hard DEV."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.adaptive_support_aggregation_v2 import (
    adaptive_v2_support_aggregate,
    load_domain_policy,
)
from evaluation.adaptive_support_aggregation import canonical_evidence_key
from evaluation.run_production_dev_k_sensitivity import best_evidence_scores
from training.run_static_hard_negative_refinement import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]
NEW_DIR = ROOT / "outputs/final_model_development/production_generator_d_static_hard_v1_dev"
V1_METRICS = ROOT / "outputs/development_runs/production_generator_d_v1_k_sensitivity/metrics.json"
V1_RANKINGS = (
    ROOT
    / "outputs/development_runs/production_generator_d_v1_k_sensitivity/per_example_rankings.jsonl"
)
POLICY_DIR = ROOT / "outputs/development_runs/production_generator_d_v1_adaptive_k"


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def top1_complete(row):
    predicted = {
        canonical_evidence_key(unit) for unit in row["ranked_candidates"][0]["subgraph_units"]
    }
    return any(
        {canonical_evidence_key(unit) for unit in gold} <= predicted
        for gold in row["gold_explanations"]
    )


def main() -> None:
    refined = json.loads((NEW_DIR / "metrics.json").read_text(encoding="utf-8"))
    baseline = json.loads(V1_METRICS.read_text(encoding="utf-8"))
    old_macro = {}
    for method in ("gnn_only", "sageqa_final"):
        old_macro[method] = {
            metric: sum(
                row["methods"][method]["metrics"]["1"][metric] for row in baseline["datasets"]
            )
            / 10
            for metric in ("precision", "recall", "f1")
        }
    policies = {
        domain: load_domain_policy(POLICY_DIR, domain=domain) for domain in ("text", "ontology")
    }
    old_adaptive_rows = []
    old_containment = defaultdict(lambda: {"gnn_only": [0, 0], "sageqa_final": [0, 0]})
    with V1_RANKINGS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            method = row["method"]
            if method in old_containment[row["dataset"]]:
                old_containment[row["dataset"]][method][0] += int(top1_complete(row))
                old_containment[row["dataset"]][method][1] += 1
            if method != "sageqa_final":
                continue
            adaptive = adaptive_v2_support_aggregate(
                row["ranked_candidates"], policy=policies[row["domain"]]
            )
            old_adaptive_rows.append(
                best_evidence_scores(adaptive["support_units"], row["gold_explanations"])
            )
    old_macro["adaptive_sageqa_frozen_full_policy"] = {
        metric: mean(old_adaptive_rows, metric) for metric in ("precision", "recall", "f1")
    }
    new_macro = refined["macro"]
    comparison = {
        "gnn_k1": {
            metric: {
                "v1": old_macro["gnn_only"][metric],
                "refined": new_macro["gnn_k1"][metric],
                "delta": new_macro["gnn_k1"][metric] - old_macro["gnn_only"][metric],
            }
            for metric in ("precision", "recall", "f1")
        },
        "sageqa_k1": {
            metric: {
                "v1": old_macro["sageqa_final"][metric],
                "refined": new_macro["sageqa_k1"][metric],
                "delta": new_macro["sageqa_k1"][metric] - old_macro["sageqa_final"][metric],
            }
            for metric in ("precision", "recall", "f1")
        },
        "adaptive_sageqa": {
            metric: {
                "v1": old_macro["adaptive_sageqa_frozen_full_policy"][metric],
                "refined": new_macro["adaptive_sageqa"][metric],
                "delta": new_macro["adaptive_sageqa"][metric]
                - old_macro["adaptive_sageqa_frozen_full_policy"][metric],
            }
            for metric in ("precision", "recall", "f1")
        },
    }
    per_dataset_containment = {}
    for row in refined["per_dataset"]:
        dataset = row["dataset"]
        per_dataset_containment[dataset] = {}
        for method in ("gnn_only", "sageqa_final"):
            count, total = old_containment[dataset][method]
            new_value = row["complete_support_containment"][method]
            per_dataset_containment[dataset][method] = {
                "v1": count / total,
                "refined": new_value,
                "delta": new_value - count / total,
            }
    order_correct = sum(
        row["complete_vs_hard_partial_ordering"]["positive_strictly_above_partial"]
        for row in refined["per_dataset"]
    )
    order_total = sum(
        row["complete_vs_hard_partial_ordering"]["eligible_complete_vs_partial"]
        for row in refined["per_dataset"]
    )
    accepted = (
        comparison["gnn_k1"]["f1"]["delta"] >= 0 and comparison["sageqa_k1"]["f1"]["delta"] >= 0
    )
    decision = {
        "schema_version": "sageqa_static_hard_refinement_dev_decision_v1",
        "status": "complete_final_method_no_further_search",
        "decision": "retain_refined" if accepted else "retain_clean_v1",
        "acceptance_rule": "refined macro GNN k1 F1 and macro SAGE-QA k1 F1 must both be non-decreasing versus clean v1",
        "comparison": comparison,
        "complete_support_containment": per_dataset_containment,
        "complete_vs_hard_partial_ordering": {
            "correct": order_correct,
            "eligible": order_total,
            "accuracy": order_correct / order_total,
        },
        "diagnosed_443": refined["diagnosed_443"],
        "test_authorized_by_dev_decision": accepted,
        "test_run": False,
        "answer_generation_run": False,
        "note": "Refined checkpoints are retained as rejected development artifacts; clean production v1 remains the selected retrieval model."
        if not accepted
        else "Refined checkpoints are accepted and may be frozen for one TEST run.",
    }
    write_json(NEW_DIR / "decision.json", decision)
    summary = [
        "# Static-hard refinement DEV decision",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        "| Method | v1 macro F1 | refined macro F1 | delta |",
        "|---|---:|---:|---:|",
    ]
    for method in ("gnn_k1", "sageqa_k1", "adaptive_sageqa"):
        item = comparison[method]["f1"]
        summary.append(
            f"| {method} | {item['v1']:.6f} | {item['refined']:.6f} | {item['delta']:+.6f} |"
        )
    summary.extend(
        [
            "",
            f"Complete-vs-hard-partial ordering: {order_correct}/{order_total} ({order_correct / order_total:.2%}).",
            f"Diagnosed 443: {refined['diagnosed_443']}.",
            "",
            "No TEST or answer generation was run.",
        ]
    )
    (NEW_DIR / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    files = [
        NEW_DIR / name
        for name in ("metrics.json", "decision.json", "summary.md", "per_example_rankings.jsonl")
    ]
    write_json(
        NEW_DIR / "artifact_manifest.json",
        {
            "schema_version": "sageqa_static_hard_refinement_dev_manifest_v1",
            "files": {
                path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in files
            },
        },
    )
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "comparison": comparison,
                "ordering": decision["complete_vs_hard_partial_ordering"],
            }
        )
    )


if __name__ == "__main__":
    main()
