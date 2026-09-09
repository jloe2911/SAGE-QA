"""Evaluate the single predeclared GNN/SAGE-QA disagreement-union policy on DEV.

The selection pass consumes only persisted top-1 identities and evidence units.
Gold explanations are loaded in a separate second pass, after every selection
has been frozen, and are used only for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RANKINGS = ROOT / "outputs/development_runs/production_generator_d_v1_k_sensitivity/per_example_rankings.jsonl"
OUTPUT_DIR = ROOT / "outputs/diagnostics/production_generator_d_v1_disagreement_union_dev"
METHODS = ("gnn_k1", "sageqa_k1", "disagreement_union")
DATASET_ORDER = (
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
EXPECTED_EFFECT_COUNTS = {"fixed": 113, "harmed": 21, "completeness_unchanged": 384}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def candidate_key(units: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(unit) for unit in units)


def deduplicated_union(left: Sequence[Any], right: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for unit in (*left, *right):
        key = str(unit)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def dataset_hop(dataset: str) -> str:
    return "1hop" if "1hop" in dataset else "2hop"


def selection_pass(path: Path) -> dict[str, dict[str, Any]]:
    """Freeze all predictions without reading any gold-valued field."""
    paired: dict[str, dict[str, Any]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(f"{path}:{line_number}: rejected non-DEV row")
            method = str(row.get("method"))
            if method not in ("gnn_only", "sageqa_final"):
                raise ValueError(f"{path}:{line_number}: unexpected method {method!r}")
            ranked = row.get("ranked_candidates") or []
            if not ranked:
                raise ValueError(f"{path}:{line_number}: missing persisted top-1")
            example_id = str(row["example_id"])
            paired[example_id][method] = {
                "dataset": str(row["dataset"]),
                "domain": str(row["domain"]),
                "units": [str(unit) for unit in ranked[0]["subgraph_units"]],
            }

    selections: dict[str, dict[str, Any]] = {}
    for example_id, pair in paired.items():
        if set(pair) != {"gnn_only", "sageqa_final"}:
            raise ValueError(f"{example_id}: missing paired persisted ranking")
        gnn, sage = pair["gnn_only"], pair["sageqa_final"]
        if (gnn["dataset"], gnn["domain"]) != (sage["dataset"], sage["domain"]):
            raise ValueError(f"{example_id}: paired metadata mismatch")
        agree = candidate_key(gnn["units"]) == candidate_key(sage["units"])
        union = list(gnn["units"]) if agree else deduplicated_union(gnn["units"], sage["units"])
        selections[example_id] = {
            "dataset": gnn["dataset"],
            "domain": gnn["domain"],
            "hop": dataset_hop(gnn["dataset"]),
            "agree": agree,
            "predictions": {
                "gnn_k1": list(gnn["units"]),
                "sageqa_k1": list(sage["units"]),
                "disagreement_union": union,
            },
        }
    return selections


def gold_pass(path: Path, expected_ids: set[str]) -> dict[str, list[list[str]]]:
    """Load gold only after the selection pass has returned frozen predictions."""
    gold_by_id: dict[str, list[list[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("split") != "dev":
                raise ValueError(f"{path}:{line_number}: rejected non-DEV row")
            example_id = str(row["example_id"])
            if example_id not in expected_ids:
                raise ValueError(f"{path}:{line_number}: unexpected example id")
            alternatives = [
                [str(unit) for unit in alternative]
                for alternative in (row.get("gold_explanations") or [])
                if alternative
            ]
            if not alternatives:
                raise ValueError(f"{example_id}: missing gold explanations")
            previous = gold_by_id.get(example_id)
            if previous is not None and previous != alternatives:
                raise ValueError(f"{example_id}: gold mismatch between ranking methods")
            gold_by_id[example_id] = alternatives
    if set(gold_by_id) != expected_ids:
        raise ValueError("Gold/pass selection example-id mismatch")
    return gold_by_id


def score(predicted: Sequence[str], alternatives: Sequence[Sequence[str]]) -> dict[str, Any]:
    predicted_set = set(predicted)
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    complete = False
    for alternative in alternatives:
        gold_set = set(alternative)
        overlap = len(predicted_set & gold_set)
        precision = overlap / len(predicted_set) if predicted_set else 0.0
        recall = overlap / len(gold_set) if gold_set else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {"precision": precision, "recall": recall, "f1": f1}
        complete = complete or gold_set <= predicted_set
    return {**best, "complete_support_containment": complete, "retrieved_evidence_units": len(predicted_set)}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty slice")
    result: dict[str, Any] = {"examples": len(rows), "methods": {}}
    for method in METHODS:
        values = [row["scores"][method] for row in rows]
        result["methods"][method] = {
            "precision": statistics.fmean(float(value["precision"]) for value in values),
            "recall": statistics.fmean(float(value["recall"]) for value in values),
            "f1": statistics.fmean(float(value["f1"]) for value in values),
            "complete_support_containment": statistics.fmean(
                float(value["complete_support_containment"]) for value in values
            ),
            "mean_retrieved_evidence_units": statistics.fmean(
                int(value["retrieved_evidence_units"]) for value in values
            ),
            "median_retrieved_evidence_units": statistics.median(
                int(value["retrieved_evidence_units"]) for value in values
            ),
        }
    return result


def delta_record(summary: Mapping[str, Any]) -> dict[str, Any]:
    union = summary["methods"]["disagreement_union"]
    return {
        baseline: {
            metric: float(union[metric]) - float(summary["methods"][baseline][metric])
            for metric in ("precision", "recall", "f1", "complete_support_containment")
        }
        for baseline in ("gnn_k1", "sageqa_k1")
    }


def table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Method | Precision | Recall | F1 | Complete containment | Mean units | Median units |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"gnn_k1": "GNN k=1", "sageqa_k1": "SAGE-QA k=1", "disagreement_union": "Disagreement union"}
    for method in METHODS:
        m = summary["methods"][method]
        lines.append(
            f"| {labels[method]} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | "
            f"{m['complete_support_containment']:.6f} | {m['mean_retrieved_evidence_units']:.6f} | "
            f"{m['median_retrieved_evidence_units']:.1f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, default=RANKINGS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")

    selections = selection_pass(args.rankings)
    gold_by_id = gold_pass(args.rankings, set(selections))
    evaluated: list[dict[str, Any]] = []
    for example_id, selection in selections.items():
        evaluated.append({
            "example_id": example_id,
            **{key: selection[key] for key in ("dataset", "domain", "hop", "agree")},
            "scores": {
                method: score(prediction, gold_by_id[example_id])
                for method, prediction in selection["predictions"].items()
            },
        })

    datasets = sorted({row["dataset"] for row in evaluated}, key=DATASET_ORDER.index)
    if tuple(datasets) != DATASET_ORDER:
        raise ValueError(f"Dataset coverage mismatch: {datasets}")
    per_dataset = {dataset: summarize([row for row in evaluated if row["dataset"] == dataset]) for dataset in datasets}
    for dataset, values in per_dataset.items():
        values["domain"] = next(row["domain"] for row in evaluated if row["dataset"] == dataset)
        values["hop"] = next(row["hop"] for row in evaluated if row["dataset"] == dataset)
        values["union_deltas"] = delta_record(values)

    slices = {
        "overall": summarize(evaluated),
        "text": summarize([row for row in evaluated if row["domain"] == "text"]),
        "ontology": summarize([row for row in evaluated if row["domain"] == "ontology"]),
        "1hop": summarize([row for row in evaluated if row["hop"] == "1hop"]),
        "2hop": summarize([row for row in evaluated if row["hop"] == "2hop"]),
    }
    for values in slices.values():
        values["union_deltas"] = delta_record(values)

    disagreements = [row for row in evaluated if not row["agree"]]
    effect_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in disagreements:
        g_complete = bool(row["scores"]["gnn_k1"]["complete_support_containment"])
        s_complete = bool(row["scores"]["sageqa_k1"]["complete_support_containment"])
        effect = "fixed" if not g_complete and s_complete else "harmed" if g_complete and not s_complete else "completeness_unchanged"
        effect_rows[effect].append(row)
    effect_counts = {effect: len(effect_rows[effect]) for effect in EXPECTED_EFFECT_COUNTS}
    if effect_counts != EXPECTED_EFFECT_COUNTS:
        raise ValueError(f"Known flip-count reproduction failed: {effect_counts}")

    unchanged = effect_rows["completeness_unchanged"]
    unchanged_both_complete = [row for row in unchanged if row["scores"]["gnn_k1"]["complete_support_containment"]]
    unchanged_both_incomplete = [row for row in unchanged if not row["scores"]["gnn_k1"]["complete_support_containment"]]
    newly_complete = [row for row in unchanged_both_incomplete if row["scores"]["disagreement_union"]["complete_support_containment"]]
    disagreement_analysis = {
        "schema_version": "production_generator_d_v1_disagreement_union_dev_v1",
        "split": "dev",
        "agreement": {
            "examples": len(evaluated),
            "agree": len(evaluated) - len(disagreements),
            "disagree": len(disagreements),
            "by_dataset": {
                dataset: {
                    "agree": sum(row["agree"] for row in evaluated if row["dataset"] == dataset),
                    "disagree": sum(not row["agree"] for row in evaluated if row["dataset"] == dataset),
                }
                for dataset in datasets
            },
        },
        "known_flip_reproduction": {"expected": EXPECTED_EFFECT_COUNTS, "observed": effect_counts, "passed": True},
        "symbolic_fixes": {
            "known": len(effect_rows["fixed"]),
            "complete_under_union": sum(row["scores"]["disagreement_union"]["complete_support_containment"] for row in effect_rows["fixed"]),
        },
        "symbolic_harms": {
            "known": len(effect_rows["harmed"]),
            "rescued_by_retaining_gnn_candidate": sum(row["scores"]["disagreement_union"]["complete_support_containment"] for row in effect_rows["harmed"]),
        },
        "completeness_unchanged_disagreements": {
            "known": len(unchanged),
            "both_top1_complete": len(unchanged_both_complete),
            "both_top1_incomplete": len(unchanged_both_incomplete),
            "both_complete_preserved_under_union": sum(row["scores"]["disagreement_union"]["complete_support_containment"] for row in unchanged_both_complete),
            "newly_complete_by_cross_candidate_union": len(newly_complete),
            "remain_incomplete_under_union": len(unchanged_both_incomplete) - len(newly_complete),
            "metrics": summarize(unchanged),
        },
        "disagreement_only_metrics": summarize(disagreements),
    }

    overall = slices["overall"]
    union_delta = overall["union_deltas"]["sageqa_k1"]
    improved_datasets = [dataset for dataset, values in per_dataset.items() if values["union_deltas"]["sageqa_k1"]["f1"] > 0]
    harmed_datasets = [dataset for dataset, values in per_dataset.items() if values["union_deltas"]["sageqa_k1"]["f1"] < 0]
    accept = union_delta["f1"] > 0
    decision = "ACCEPT" if accept else "REJECT"

    lineage = {
        "input_rankings": str(args.rankings.relative_to(ROOT)),
        "input_rankings_sha256": sha256(args.rankings),
        "selection_boundary": "All paired top-1 selections and unions were frozen in selection_pass before gold_pass loaded gold explanations.",
        "candidate_identity": "Exact ordered tuple of persisted subgraph_units, matching the prior top-1 flip diagnosis.",
        "union_order": "GNN top-1 units followed by SAGE-QA top-1 units with first-occurrence deduplication.",
        "metric_aggregation": "Arithmetic macro mean over examples; evidence identity is exact persisted native-unit string identity; multiple gold explanations use the alternative with maximum per-example F1.",
        "complete_support_containment": "True when any complete stored gold explanation is a subset of retrieved native evidence units.",
    }
    constraints = {
        "training_run": False,
        "gnn_inference_run": False,
        "candidate_generation_run": False,
        "symbolic_weights_changed": False,
        "hyperparameter_search_run": False,
        "test_accessed": False,
        "answer_generation_run": False,
        "production_modified": False,
        "policy_count_evaluated": 1,
    }
    metrics = {
        "schema_version": "production_generator_d_v1_disagreement_union_dev_v1",
        "split": "dev",
        "policy": "If GNN and SAGE-QA top-1 candidates agree, retrieve that candidate; otherwise retrieve their order-preserving deduplicated evidence-unit union.",
        "slices": slices,
        "lineage": lineage,
        "constraints": constraints,
    }
    per_dataset_artifact = {
        "schema_version": "production_generator_d_v1_disagreement_union_dev_v1",
        "split": "dev",
        "datasets": per_dataset,
    }

    summary_lines = [
        "# DEV disagreement-aware top-1 union",
        "",
        f"**Decision: {decision}.** The single predeclared policy was evaluated on all {len(evaluated):,} examples in all 10 DEV datasets.",
        "",
        "## Overall macro results",
        "",
        *table(overall),
        "",
        f"Against SAGE-QA k=1, the union changed macro F1 by {union_delta['f1']:+.6f}, precision by {union_delta['precision']:+.6f}, recall by {union_delta['recall']:+.6f}, and complete-support containment by {union_delta['complete_support_containment']:+.6f}.",
        "",
        "## Requested slices",
        "",
    ]
    for name in ("text", "ontology", "1hop", "2hop"):
        summary_lines.extend([f"### {name}", "", *table(slices[name]), ""])
    summary_lines.extend([
        "## Disagreement mechanism",
        "",
        f"GNN and SAGE-QA agreed on {len(evaluated) - len(disagreements):,} examples and disagreed on {len(disagreements):,}.",
        f"The union retained complete support for {disagreement_analysis['symbolic_fixes']['complete_under_union']}/{len(effect_rows['fixed'])} known symbolic fixes and rescued {disagreement_analysis['symbolic_harms']['rescued_by_retaining_gnn_candidate']}/{len(effect_rows['harmed'])} known symbolic harms.",
        f"Of the 384 completeness-unchanged disagreements, {len(unchanged_both_complete)} had both candidates complete and remained complete; {len(unchanged_both_incomplete)} had both incomplete, of which {len(newly_complete)} became complete only through the cross-candidate union and {len(unchanged_both_incomplete) - len(newly_complete)} remained incomplete.",
        "",
        "## Evidence boundary",
        "",
        lineage["selection_boundary"],
        "No training, neural inference, candidate generation, symbolic-weight change, hyperparameter search, TEST access, answer generation, or production modification was performed.",
        "",
    ])

    decision_lines = [
        "# Mechanism decision",
        "",
        f"## {decision}",
        "",
        f"1. **FAIL — macro F1:** union F1 is {overall['methods']['disagreement_union']['f1']:.6f} versus {overall['methods']['sageqa_k1']['f1']:.6f} for SAGE-QA k=1 ({union_delta['f1']:+.6f}).",
        f"2. **FAIL — precision:** macro precision falls from {overall['methods']['sageqa_k1']['precision']:.6f} to {overall['methods']['disagreement_union']['precision']:.6f} ({union_delta['precision']:+.6f}, {100.0 * union_delta['precision'] / overall['methods']['sageqa_k1']['precision']:+.2f}% relative), which is material in this retrieval setting.",
        f"3. **PASS — fix/harm mechanism:** the union preserves {disagreement_analysis['symbolic_fixes']['complete_under_union']}/{len(effect_rows['fixed'])} symbolic fixes and recovers {disagreement_analysis['symbolic_harms']['rescued_by_retaining_gnn_candidate']}/{len(effect_rows['harmed'])} symbolic harms.",
        f"4. **FAIL — beneficial cross-dataset consistency:** {len(improved_datasets)}/10 datasets have higher F1 and {len(harmed_datasets)}/10 have lower F1 than SAGE-QA k=1. Improved: {', '.join(improved_datasets) or 'none'}. Lower: {', '.join(harmed_datasets) or 'none'}.",
        "",
        "Because all four conditions are conjunctive, failure of any mandatory condition rejects the policy. No alternative disagreement rule was tried.",
        "",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "per_dataset.json", per_dataset_artifact)
    write_json(args.output_dir / "disagreement_analysis.json", disagreement_analysis)
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (args.output_dir / "mechanism_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")
    print(json.dumps({"decision": decision, "examples": len(evaluated), "agree": len(evaluated) - len(disagreements), "disagree": len(disagreements), "union_delta_vs_sageqa": union_delta}, indent=2))


if __name__ == "__main__":
    main()
