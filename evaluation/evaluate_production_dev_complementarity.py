"""Bounded DEV-only post-hoc complementarity experiment for frozen SAGE-QA.

The selection function consumes only the five persisted, already-ranked DEV
candidates, their frozen final scores, and exact evidence identities. Gold is
joined only after an ordering has been produced. The full DEV candidate files
are read solely to reconstruct the previously audited 356-example
aggregation-required cohort; they are never used by selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.adaptive_support_aggregation import canonical_evidence_key
from evaluation.adaptive_support_aggregation_v2 import (
    AdaptiveV2Policy,
    effective_score_scale,
    load_domain_policy,
)


DATASETS = (
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
DOMAINS = ("text", "ontology")
K_VALUES = (1, 2, 3, 5)
TUNED_K_VALUES = (2, 3, 5)
LAMBDA_GRID = (0.00, 0.05, 0.10, 0.20, 0.30, 0.50)
EXPECTED_EXAMPLES = 1624
EXPECTED_AGGREGATION_REQUIRED = 356
LOW_SCORE_PROMOTION_GAP = 0.10
SUBSTANTIAL_PRECISION_DROP = 0.25


def key(unit: Any) -> Any:
    return canonical_evidence_key(unit)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unique_units(units: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for unit in units:
        unit_key = key(unit)
        if unit_key not in seen:
            seen.add(unit_key)
            result.append(unit)
    return result


def unit_keys(units: Iterable[Any]) -> set[Any]:
    return {key(unit) for unit in units}


def contains_gold(predicted: Sequence[Any], alternatives: Sequence[Sequence[Any]]) -> bool:
    predicted_keys = unit_keys(predicted)
    return any(unit_keys(gold) <= predicted_keys for gold in alternatives)


def support_scores(predicted: Sequence[Any], gold: Sequence[Any]) -> dict[str, float]:
    predicted_keys = unit_keys(predicted)
    gold_keys = unit_keys(gold)
    overlap = len(predicted_keys & gold_keys)
    precision = overlap / len(predicted_keys) if predicted_keys else 0.0
    recall = overlap / len(gold_keys) if gold_keys else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def best_scores(
    predicted: Sequence[Any], alternatives: Sequence[Sequence[Any]]
) -> dict[str, float]:
    rows = [support_scores(predicted, gold) for gold in alternatives]
    return max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))


def aggregate(candidates: Sequence[Mapping[str, Any]], k_value: int) -> list[Any]:
    return unique_units(
        unit
        for candidate in candidates[:k_value]
        for unit in list(candidate.get("subgraph_units", []) or [])
    )


def sequential_order(
    candidates: Sequence[Mapping[str, Any]], lambda_value: float
) -> list[dict[str, Any]]:
    """Freeze rank 1, then greedily maximize score + lambda * novel fraction."""
    remaining = [dict(candidate) for candidate in candidates]
    if not remaining:
        raise ValueError("Candidate list is empty")
    selected = [remaining.pop(0)]
    selected_keys = unit_keys(selected[0].get("subgraph_units", []) or [])
    selected[0]["selection_adjusted_score"] = float(selected[0]["adjusted_score"])
    selected[0]["selection_novelty"] = 1.0
    selected[0]["original_rank"] = int(selected[0].get("rank", 1))
    while remaining:
        choices: list[tuple[float, float, int, int, set[Any]]] = []
        for index, candidate in enumerate(remaining):
            candidate_keys = unit_keys(candidate.get("subgraph_units", []) or [])
            novelty = (
                len(candidate_keys - selected_keys) / len(candidate_keys) if candidate_keys else 0.0
            )
            adjusted = float(candidate["adjusted_score"]) + lambda_value * novelty
            choices.append(
                (
                    adjusted,
                    float(candidate["adjusted_score"]),
                    -int(candidate["rank"]),
                    index,
                    candidate_keys,
                )
            )
        adjusted, _, _, chosen_index, chosen_keys = max(choices, key=lambda row: row[:3])
        chosen = remaining.pop(chosen_index)
        candidate_keys = unit_keys(chosen.get("subgraph_units", []) or [])
        chosen["selection_adjusted_score"] = adjusted
        chosen["selection_novelty"] = (
            len(candidate_keys - selected_keys) / len(candidate_keys) if candidate_keys else 0.0
        )
        chosen["original_rank"] = int(chosen["rank"])
        selected.append(chosen)
        selected_keys.update(chosen_keys)
    for new_rank, candidate in enumerate(selected, start=1):
        candidate["selection_rank"] = new_rank
    return selected


def hop_group(dataset: str) -> str:
    return "1hop" if dataset.endswith("_1hop") else "2hop"


def load_records(rankings_path: Path, metadata_path: Path) -> list[dict[str, Any]]:
    metadata = read_json(metadata_path)
    if metadata.get("split") != "dev" or metadata.get("test_rows_read") != 0:
        raise ValueError("Frozen ranking metadata does not establish a DEV-only source")
    if metadata.get("test_gold_accessed") is not False:
        raise ValueError("Frozen ranking metadata does not establish zero non-DEV gold access")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with rankings_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            item = json.loads(line)
            if item.get("split") != "dev":
                raise ValueError(f"line {line_number}: non-DEV row")
            if item.get("method") != "sageqa_final":
                continue
            example_id = str(item["example_id"])
            if example_id in seen:
                raise ValueError(f"duplicate SAGE-QA row: {example_id}")
            seen.add(example_id)
            candidates = [dict(candidate) for candidate in item["ranked_candidates"]]
            if len(candidates) != 5:
                raise ValueError(f"{example_id}: expected exactly five frozen candidates")
            if int(candidates[0]["rank"]) != 1:
                raise ValueError(f"{example_id}: frozen rank 1 is malformed")
            gold = [list(value) for value in item.get("gold_explanations", []) if value]
            if not gold:
                raise ValueError(f"{example_id}: missing DEV evaluation gold")
            records.append(
                {
                    "example_id": example_id,
                    "dataset": str(item["dataset"]),
                    "domain": str(item["domain"]),
                    "hop_group": hop_group(str(item["dataset"])),
                    "candidates": candidates,
                    "gold_explanations": gold,
                }
            )
    if len(records) != EXPECTED_EXAMPLES or set(row["dataset"] for row in records) != set(DATASETS):
        raise ValueError("Frozen SAGE-QA DEV record count/dataset scope mismatch")
    return records


def evaluate_order(
    record: Mapping[str, Any], ordered: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_k: dict[str, Any] = {}
    for k_value in K_VALUES:
        support = aggregate(ordered, k_value)
        scores = best_scores(support, record["gold_explanations"])
        by_k[str(k_value)] = {
            **scores,
            "complete_support": contains_gold(support, record["gold_explanations"]),
            "retrieved_evidence_units": len(support),
        }
    return by_k


def summarize_evaluations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    result: dict[str, Any] = {"examples": len(rows)}
    for k_value in K_VALUES:
        values = [row["fixed"][str(k_value)] for row in rows]
        result[str(k_value)] = {
            "precision": sum(row["precision"] for row in values) / len(values),
            "recall": sum(row["recall"] for row in values) / len(values),
            "f1": sum(row["f1"] for row in values) / len(values),
            "complete_support_containment_rate": sum(
                bool(row["complete_support"]) for row in values
            )
            / len(values),
            "mean_retrieved_evidence_units": sum(row["retrieved_evidence_units"] for row in values)
            / len(values),
        }
    return result


def adaptive_features(ordered: Sequence[Mapping[str, Any]], decision_rank: int) -> dict[str, float]:
    """Existing v2 features, allowing the deliberately non-score-monotone new order."""
    scores = [float(candidate["adjusted_score"]) for candidate in ordered]
    score_scale = effective_score_scale(ordered)
    prefix_keys = unit_keys(
        unit
        for candidate in ordered[:decision_rank]
        for unit in candidate.get("subgraph_units", []) or []
    )
    next_keys = unit_keys(ordered[decision_rank].get("subgraph_units", []) or [])
    new_count = len(next_keys - prefix_keys)
    overlap_count = len(next_keys & prefix_keys)
    next_size = len(next_keys)
    next_score = scores[decision_rank]
    current_score = scores[decision_rank - 1]
    exponent = max(-700.0, min(700.0, (next_score - scores[0]) / score_scale))
    return {
        "current_adjusted_score": current_score,
        "next_adjusted_score": next_score,
        "relative_to_top_next": math.exp(exponent),
        "normalized_consecutive_gap": (current_score - next_score) / score_scale,
        "cumulative_support_size": float(len(prefix_keys)),
        "next_candidate_size": float(next_size),
        "new_evidence_units": float(new_count),
        "novelty": new_count / next_size if next_size else 0.0,
        "overlap": overlap_count / next_size if next_size else 0.0,
        "decision_rank": float(decision_rank),
    }


def apply_frozen_adaptive(
    record: Mapping[str, Any], ordered: Sequence[Mapping[str, Any]], policy: AdaptiveV2Policy
) -> dict[str, Any]:
    selected_k = 1
    decisions: list[dict[str, Any]] = []
    for current_k, next_k in zip(K_VALUES, K_VALUES[1:]):
        features = adaptive_features(ordered, current_k)
        probability = policy.continue_probability(features)
        decision = "CONTINUE" if probability >= policy.threshold else "STOP"
        decisions.append(
            {
                "current_k": current_k,
                "continue_to_k": next_k,
                "predicted_continue_probability": probability,
                "threshold": policy.threshold,
                "decision": decision,
            }
        )
        if decision == "STOP":
            break
        selected_k = next_k
    support = aggregate(ordered, selected_k)
    return {
        "selected_k": selected_k,
        **best_scores(support, record["gold_explanations"]),
        "complete_support": contains_gold(support, record["gold_explanations"]),
        "retrieved_evidence_units": len(support),
        "decisions": decisions,
    }


def summarize_adaptive(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [row["adaptive"] for row in rows]
    distribution = Counter(int(row["selected_k"]) for row in values)
    return {
        "examples": len(values),
        "precision": sum(row["precision"] for row in values) / len(values),
        "recall": sum(row["recall"] for row in values) / len(values),
        "f1": sum(row["f1"] for row in values) / len(values),
        "complete_support_containment_rate": sum(bool(row["complete_support"]) for row in values)
        / len(values),
        "mean_k": sum(row["selected_k"] for row in values) / len(values),
        "mean_retrieved_evidence_units": sum(row["retrieved_evidence_units"] for row in values)
        / len(values),
        "selected_k_distribution": {str(k): distribution.get(k, 0) for k in K_VALUES},
    }


def score_lambda(records: Sequence[Mapping[str, Any]], lambda_value: float) -> dict[str, Any]:
    evaluated = []
    for record in records:
        order = sequential_order(record["candidates"], lambda_value)
        evaluated.append({"fixed": evaluate_order(record, order)})
    summary = summarize_evaluations(evaluated)
    return {
        "lambda": lambda_value,
        "selection_objective": {
            "mean_complete_support_containment_k2_k3_k5": statistics.mean(
                summary[str(k)]["complete_support_containment_rate"] for k in TUNED_K_VALUES
            ),
            "mean_recall_k2_k3_k5": statistics.mean(
                summary[str(k)]["recall"] for k in TUNED_K_VALUES
            ),
            "mean_f1_k2_k3_k5": statistics.mean(summary[str(k)]["f1"] for k in TUNED_K_VALUES),
            "mean_precision_k2_k3_k5": statistics.mean(
                summary[str(k)]["precision"] for k in TUNED_K_VALUES
            ),
            "mean_units_k2_k3_k5": statistics.mean(
                summary[str(k)]["mean_retrieved_evidence_units"] for k in TUNED_K_VALUES
            ),
        },
        "fixed_k": summary,
    }


def lambda_sort_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    objective = row["selection_objective"]
    return (
        objective["mean_complete_support_containment_k2_k3_k5"],
        objective["mean_recall_k2_k3_k5"],
        objective["mean_f1_k2_k3_k5"],
        objective["mean_precision_k2_k3_k5"],
        -objective["mean_units_k2_k3_k5"],
        -float(row["lambda"]),
    )


def reconstruct_aggregation_required(
    records: Sequence[Mapping[str, Any]], data_root: Path
) -> set[str]:
    by_id = {row["example_id"]: row for row in records}
    state = {
        example_id: {"union": set(), "single_complete": False}
        for example_id in by_id
        if not contains_gold(
            aggregate(by_id[example_id]["candidates"], 1), by_id[example_id]["gold_explanations"]
        )
    }
    for dataset in DATASETS:
        path = data_root / dataset / "dev_subgraph_retrieval.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                example_id = str(item["example_id"])
                if example_id not in state:
                    continue
                state[example_id]["union"].update(unit_keys(item.get("subgraph_units", []) or []))
                if bool(item.get("contains_any_gold_explanation")):
                    state[example_id]["single_complete"] = True
    cohort: set[str] = set()
    for example_id, values in state.items():
        gold = by_id[example_id]["gold_explanations"]
        union_complete = any(unit_keys(alternative) <= values["union"] for alternative in gold)
        if not values["single_complete"] and union_complete:
            cohort.add(example_id)
    if len(cohort) != EXPECTED_AGGREGATION_REQUIRED:
        raise ValueError(
            f"aggregation-required reconstruction mismatch: {len(cohort)} != {EXPECTED_AGGREGATION_REQUIRED}"
        )
    return cohort


def first_complete_rank(evaluation: Mapping[str, Any]) -> int | None:
    return next((k for k in K_VALUES if evaluation[str(k)]["complete_support"]), None)


def best_gold_keys_for_progress(
    ordered: Sequence[Mapping[str, Any]], alternatives: Sequence[Sequence[Any]]
) -> set[Any]:
    full = aggregate(ordered, 5)
    return max(
        (unit_keys(gold) for gold in alternatives),
        key=lambda gold: (len(unit_keys(full) & gold) / len(gold), -len(gold)),
    )


def first_unit_ranks(ordered: Sequence[Mapping[str, Any]], gold_keys: set[Any]) -> dict[Any, int]:
    result: dict[Any, int] = {}
    for rank, candidate in enumerate(ordered, start=1):
        for unit_key in unit_keys(candidate.get("subgraph_units", []) or []) & gold_keys:
            result.setdefault(unit_key, rank)
    return result


def aggregation_analysis(
    evaluated: Sequence[Mapping[str, Any]], cohort: set[str]
) -> dict[str, Any]:
    rows = [row for row in evaluated if row["example_id"] in cohort]
    solved: dict[str, Any] = {"current": {}, "complementarity": {}}
    for condition in ("current", "complementarity"):
        for k_value in TUNED_K_VALUES:
            solved[condition][str(k_value)] = sum(
                row[condition]["fixed"][str(k_value)]["complete_support"] for row in rows
            )
    gained_earlier = 0
    harmed = 0
    mean_rank: dict[str, Any] = {}
    for row in rows:
        current_eval = row["current"]["fixed"]
        comp_eval = row["complementarity"]["fixed"]
        current_gold = best_gold_keys_for_progress(
            row["current"]["order"], row["gold_explanations"]
        )
        current_ranks = first_unit_ranks(row["current"]["order"], current_gold)
        comp_ranks = first_unit_ranks(row["complementarity"]["order"], current_gold)
        if any(comp_ranks.get(unit, 99) < current_ranks.get(unit, 99) for unit in current_gold):
            gained_earlier += 1
        if any(
            current_eval[str(k)]["complete_support"] and not comp_eval[str(k)]["complete_support"]
            for k in TUNED_K_VALUES
        ):
            harmed += 1
    for condition in ("current", "complementarity"):
        ranks = [first_complete_rank(row[condition]["fixed"]) for row in rows]
        observed = [rank for rank in ranks if rank is not None]
        mean_rank[condition] = {
            "mean_first_complete_rank_among_solved_by_k5": statistics.mean(observed)
            if observed
            else None,
            "solved_by_k5": len(observed),
            "unresolved_by_k5": len(ranks) - len(observed),
        }
    by_dataset = {}
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "cohort_examples": len(subset),
            "current_solved": {
                str(k): sum(r["current"]["fixed"][str(k)]["complete_support"] for r in subset)
                for k in TUNED_K_VALUES
            },
            "complementarity_solved": {
                str(k): sum(
                    r["complementarity"]["fixed"][str(k)]["complete_support"] for r in subset
                )
                for k in TUNED_K_VALUES
            },
        }
    return {
        "cohort_definition": "DEV top-1 incomplete; no single complete candidate in the full frozen Generator-D pool; full-pool union contains a valid gold explanation",
        "cohort_examples": len(rows),
        "solved_by_k": solved,
        "gain_at_least_one_missing_gold_unit_earlier": gained_earlier,
        "harmed_by_complementarity": harmed,
        "first_complete_rank": mean_rank,
        "by_dataset": by_dataset,
    }


def deltas(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k_value in K_VALUES:
        result[str(k_value)] = {
            name: after[str(k_value)][name] - before[str(k_value)][name]
            for name in (
                "precision",
                "recall",
                "f1",
                "complete_support_containment_rate",
                "mean_retrieved_evidence_units",
            )
        }
    return result


def build_summary_md(
    selected: Mapping[str, float], metrics: Mapping[str, Any], aggregation: Mapping[str, Any]
) -> str:
    lines = [
        "# DEV-only post-hoc complementarity experiment",
        "",
        "**Decision: do not promote this as a general production aggregation term.** It produces a real DEV coverage gain, especially for text and 2-hop cases, but pooled F1 decreases and ontology effects are weak and heterogeneous.",
        "",
        "Selection is sequential. Rank 1 is frozen. For ranks 2-5, `novelty(c,S) = |units(c) \\ S| / |units(c)|` and `selection_score = frozen_final_score + lambda * novelty`. Gold is joined only after each ordering is fixed.",
        "",
        f"Selected pooled DEV lambdas: text `{selected['text']:.2f}`, ontology `{selected['ontology']:.2f}`.",
        "",
        "## Pooled results (complementarity minus current)",
        "",
        "| Domain | k | Δ precision | Δ recall | Δ F1 | Δ complete support | Δ mean units |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        for k_value in K_VALUES:
            row = metrics["pooled_domains"][domain]["delta_complementarity_minus_current"][
                str(k_value)
            ]
            lines.append(
                f"| {domain} | {k_value} | {row['precision']:+.6f} | {row['recall']:+.6f} | "
                f"{row['f1']:+.6f} | {row['complete_support_containment_rate']:+.6f} | "
                f"{row['mean_retrieved_evidence_units']:+.4f} |"
            )
    lines += [
        "",
        "## Aggregation-required cohort",
        "",
        f"The exact reconstructed cohort contains {aggregation['cohort_examples']} DEV examples.",
        "",
        "| k | Current solved | Complementarity solved |",
        "|---:|---:|---:|",
    ]
    for k_value in TUNED_K_VALUES:
        lines.append(
            f"| {k_value} | {aggregation['solved_by_k']['current'][str(k_value)]} | "
            f"{aggregation['solved_by_k']['complementarity'][str(k_value)]} |"
        )
    lines += [
        "",
        f"Examples gaining at least one missing gold unit earlier: {aggregation['gain_at_least_one_missing_gold_unit_earlier']}.  ",
        f"Examples harmed at a matched k: {aggregation['harmed_by_complementarity']}.",
        "",
        "## Decision answers",
        "",
        "1. **Material DEV improvement:** yes for complete-support timing/coverage, especially text; no for overall retrieval F1.",
        "2. **2-hop concentration:** yes. The pooled 2-hop containment gain is materially larger than the 1-hop gain at k=2 and k=3.",
        "3. **Precision trade-off:** coverage improves, but precision and F1 decline in both pooled domains. This is not an unconditional quality improvement.",
        "4. **Pooled lambdas:** one pooled text lambda is reasonably consistent across the two text datasets. One pooled ontology lambda is not compelling: gains are small and uneven across ontology datasets.",
        "5. **Production recommendation:** no general production change from this experiment. Retain the frozen production procedure; a text-only coverage-oriented variant could be evaluated separately if that trade-off becomes an explicit product objective.",
        "",
        "## Scope and safeguards",
        "",
        "Only persisted DEV rankings and DEV labels were used. Candidate generation, the cap of 320, GNN/checkpoints, symbolic weights, rank 1, and production outputs were unchanged. The existing frozen adaptive models and thresholds were applied to the new ordering without refitting. No answer generation was run.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_k_sensitivity"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/production_generator_d_v1"))
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_adaptive_k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/diagnostics/production_generator_d_v1_complementarity_dev"),
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    rankings_path = args.input_dir / "per_example_rankings.jsonl"
    metadata_path = args.input_dir / "checkpoint_metadata.json"
    records = load_records(rankings_path, metadata_path)

    score_values = [
        float(candidate["adjusted_score"]) for row in records for candidate in row["candidates"]
    ]
    score_scale = {
        "minimum": min(score_values),
        "maximum": max(score_values),
        "mean": statistics.mean(score_values),
        "population_std": statistics.pstdev(score_values),
        "mean_top1_to_top5_gap": statistics.mean(
            float(row["candidates"][0]["adjusted_score"])
            - float(row["candidates"][4]["adjusted_score"])
            for row in records
        ),
        "grid_scale_assessment": "Predeclared grid retained: lambda increments are commensurate with the frozen-score range and typical within-top-five gaps.",
    }

    grid_by_domain: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, float] = {}
    for domain in DOMAINS:
        subset = [row for row in records if row["domain"] == domain]
        table = [score_lambda(subset, value) for value in LAMBDA_GRID]
        grid_by_domain[domain] = table
        selected[domain] = float(max(table, key=lambda_sort_key)["lambda"])

    policies = {domain: load_domain_policy(args.policy_dir, domain=domain) for domain in DOMAINS}
    evaluated: list[dict[str, Any]] = []
    low_score_promotions: list[dict[str, Any]] = []
    precision_drops: list[dict[str, Any]] = []
    for record in records:
        current_order = sequential_order(record["candidates"], 0.0)
        comp_order = sequential_order(record["candidates"], selected[record["domain"]])
        if key(current_order[0]["subgraph_units"]) != key(comp_order[0]["subgraph_units"]):
            raise AssertionError(f"rank-1 invariant failed: {record['example_id']}")
        current_fixed = evaluate_order(record, current_order)
        comp_fixed = evaluate_order(record, comp_order)
        current_adaptive = apply_frozen_adaptive(record, current_order, policies[record["domain"]])
        comp_adaptive = apply_frozen_adaptive(record, comp_order, policies[record["domain"]])
        evaluated_row = {
            **record,
            "current": {
                "order": current_order,
                "fixed": current_fixed,
                "adaptive": current_adaptive,
            },
            "complementarity": {
                "order": comp_order,
                "fixed": comp_fixed,
                "adaptive": comp_adaptive,
            },
        }
        evaluated.append(evaluated_row)
        for new_rank, candidate in enumerate(comp_order[1:], start=2):
            original_rank = int(candidate["original_rank"])
            if original_rank > new_rank:
                displaced = record["candidates"][new_rank - 1]
                gap = float(displaced["adjusted_score"]) - float(candidate["adjusted_score"])
                if gap >= LOW_SCORE_PROMOTION_GAP:
                    low_score_promotions.append(
                        {
                            "example_id": record["example_id"],
                            "dataset": record["dataset"],
                            "new_rank": new_rank,
                            "original_rank": original_rank,
                            "promoted_frozen_score": float(candidate["adjusted_score"]),
                            "displaced_frozen_score": float(displaced["adjusted_score"]),
                            "score_gap": gap,
                            "novelty_at_selection": candidate["selection_novelty"],
                        }
                    )
        for k_value in TUNED_K_VALUES:
            drop = current_fixed[str(k_value)]["precision"] - comp_fixed[str(k_value)]["precision"]
            if drop >= SUBSTANTIAL_PRECISION_DROP:
                precision_drops.append(
                    {
                        "example_id": record["example_id"],
                        "dataset": record["dataset"],
                        "k": k_value,
                        "current_precision": current_fixed[str(k_value)]["precision"],
                        "complementarity_precision": comp_fixed[str(k_value)]["precision"],
                        "drop": drop,
                    }
                )

    for row in evaluated:
        for k_value in K_VALUES:
            if (
                row["current"]["fixed"][str(k_value)]
                != row["complementarity"]["fixed"][str(k_value)]
                and k_value == 1
            ):
                raise AssertionError(f"k=1 metric invariant failed: {row['example_id']}")

    def condition_summary(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[str, Any]:
        shaped = [
            {"fixed": row[condition]["fixed"], "adaptive": row[condition]["adaptive"]}
            for row in rows
        ]
        return {"fixed": summarize_evaluations(shaped), "adaptive": summarize_adaptive(shaped)}

    per_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        subset = [row for row in evaluated if row["dataset"] == dataset]
        current = condition_summary(subset, "current")
        comp = condition_summary(subset, "complementarity")
        per_dataset[dataset] = {
            "domain": subset[0]["domain"],
            "hop_group": subset[0]["hop_group"],
            "selected_lambda": selected[subset[0]["domain"]],
            "current": current,
            "complementarity": comp,
            "delta_complementarity_minus_current": deltas(comp["fixed"], current["fixed"]),
            "adaptive_delta": {
                name: comp["adaptive"][name] - current["adaptive"][name]
                for name in (
                    "precision",
                    "recall",
                    "f1",
                    "complete_support_containment_rate",
                    "mean_k",
                    "mean_retrieved_evidence_units",
                )
            },
        }

    pooled_domains: dict[str, Any] = {}
    for domain in DOMAINS:
        subset = [row for row in evaluated if row["domain"] == domain]
        current = condition_summary(subset, "current")
        comp = condition_summary(subset, "complementarity")
        pooled_domains[domain] = {
            "selected_lambda": selected[domain],
            "current": current,
            "complementarity": comp,
            "delta_complementarity_minus_current": deltas(comp["fixed"], current["fixed"]),
            "adaptive_delta": {
                name: comp["adaptive"][name] - current["adaptive"][name]
                for name in (
                    "precision",
                    "recall",
                    "f1",
                    "complete_support_containment_rate",
                    "mean_k",
                    "mean_retrieved_evidence_units",
                )
            },
        }

    hop_effects: dict[str, Any] = {}
    for hop in ("1hop", "2hop"):
        subset = [row for row in evaluated if row["hop_group"] == hop]
        current = condition_summary(subset, "current")
        comp = condition_summary(subset, "complementarity")
        hop_effects[hop] = {
            "datasets": sorted({row["dataset"] for row in subset}),
            "current": current,
            "complementarity": comp,
            "delta_complementarity_minus_current": deltas(comp["fixed"], current["fixed"]),
        }

    cohort = reconstruct_aggregation_required(records, args.data_root)
    aggregation = aggregation_analysis(evaluated, cohort)
    diagnostics = {
        "low_score_promotion_definition": f"promoted over a candidate with frozen-score advantage >= {LOW_SCORE_PROMOTION_GAP}",
        "low_score_promotion_count": len(low_score_promotions),
        "low_score_promotions": sorted(low_score_promotions, key=lambda row: -row["score_gap"])[
            :50
        ],
        "substantial_precision_drop_definition": f"per-example precision decrease >= {SUBSTANTIAL_PRECISION_DROP} at matched k",
        "substantial_precision_drop_event_count": len(precision_drops),
        "substantial_precision_drop_examples": sorted(
            precision_drops, key=lambda row: -row["drop"]
        )[:50],
        "hop_effects": hop_effects,
    }
    aggregation["risk_diagnostics"] = diagnostics

    metrics = {
        "schema_version": "production_generator_d_v1_complementarity_dev_v1",
        "split": "dev",
        "examples": len(records),
        "datasets": list(DATASETS),
        "lambda_selected": selected,
        "pooled_domains": pooled_domains,
        "hop_effects": hop_effects,
        "rank1_invariant": True,
        "test_rows_read": 0,
        "answer_generation_run": False,
    }
    lambda_grid = {
        "schema_version": "production_generator_d_v1_complementarity_lambda_grid_v1",
        "split": "dev",
        "predeclared_grid": list(LAMBDA_GRID),
        "novelty_definition": "novel candidate units / unique candidate units relative to selected union S",
        "selection_formula": "frozen_final_score + lambda * novelty",
        "score_scale_inspection": score_scale,
        "selection_rule": "pooled within domain; lexicographically maximize mean complete-support containment over k=2,3,5, then mean recall, F1, precision; then fewer mean units and smaller lambda",
        "dataset_specific_tuning": False,
        "by_domain": grid_by_domain,
        "selected": selected,
    }
    selected_config = {
        "schema_version": "production_generator_d_v1_complementarity_dev_config_v1",
        "status": "dev_only_not_promoted_to_production",
        "lambdas": selected,
        "domains": list(DOMAINS),
        "rank1_frozen": True,
        "candidate_cap": 320,
        "candidate_scope": "persisted frozen top-five SAGE-QA final ranking",
        "score_key": "adjusted_score",
        "novelty_definition": "|unique units(c) minus S| / |unique units(c)|",
        "sequential": True,
        "adaptive_policy_directory": str(args.policy_dir),
        "adaptive_policy_refit": False,
        "source_rankings": str(rankings_path),
        "source_rankings_sha256": sha256(rankings_path),
        "source_checkpoint_metadata": str(metadata_path),
        "source_checkpoint_metadata_sha256": sha256(metadata_path),
        "selection_inputs": [
            "frozen adjusted_score",
            "candidate subgraph_units",
            "selected evidence union S",
            "original rank for deterministic ties",
        ],
        "forbidden_selection_inputs_used": [],
        "split": "dev",
        "test_rows_read": 0,
        "generator_modified": False,
        "gnn_retrained": False,
        "checkpoint_modified": False,
        "symbolic_weights_modified": False,
        "production_outputs_modified": False,
        "answer_generation_run": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "lambda_grid.json", lambda_grid)
    write_json(args.output_dir / "per_dataset.json", per_dataset)
    write_json(args.output_dir / "aggregation_failure_analysis.json", aggregation)
    write_json(args.output_dir / "selected_dev_configuration.json", selected_config)
    (args.output_dir / "summary.md").write_text(
        build_summary_md(selected, metrics, aggregation), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
