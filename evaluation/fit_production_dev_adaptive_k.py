"""Fit and freeze adaptive-v2 from clean production Generator-D DEV rankings.

This command fits from only the selected method's persisted DEV rankings and
the ranking artifact's two metadata files. It does not load raw datasets,
checkpoints, TEST data, generation code, training code, or symbolic reranking
code. The fitted policies preserve the existing adaptive-v2
feature/model/calibration family while restricting selected depths to
``{1, 2, 3, 5}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.adaptive_support_aggregation import (  # noqa: E402
    DEFAULT_EPSILON,
    canonical_evidence_key,
    fixed_k_support_aggregate,
)
from evaluation.adaptive_support_aggregation_v2 import (  # noqa: E402
    CONTINUOUS_FEATURES,
    FEATURE_ORDER,
    AdaptiveV2Policy,
    compute_decision_features,
)
from evaluation.evaluate_adaptive_support_aggregation_v2 import (  # noqa: E402
    RANDOM_STATE,
    THRESHOLD_GRID,
    classification_metrics,
    fit_model,
    grouped_oof,
    model_description,
)


ALLOWED_K = (1, 2, 3, 5)
DOMAINS = ("text", "ontology")
INPUT_FILENAMES = (
    "per_example_rankings.jsonl",
    "metrics.json",
    "checkpoint_metadata.json",
)
RANKING_METHODS = (
    "gnn_only",
    "sageqa_final",
    "cross_encoder",
    "final_sageqa",
)


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def load_clean_final_records(
    input_dir: Path, ranking_method: str = "gnn_only"
) -> list[dict[str, Any]]:
    if ranking_method not in RANKING_METHODS:
        raise ValueError(f"ranking_method must be one of {RANKING_METHODS}, got {ranking_method!r}")
    path = input_dir / "per_example_rankings.jsonl"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            item = json.loads(line)
            if item.get("split") != "dev":
                raise ValueError(f"line {line_number}: non-DEV split encountered")
            if item.get("method") != ranking_method:
                continue
            example_id = str(item.get("example_id") or "")
            if not example_id or example_id in seen:
                raise ValueError(f"line {line_number}: missing or duplicate example_id")
            seen.add(example_id)
            candidates = list(item.get("ranked_candidates", []) or [])
            if len(candidates) < max(ALLOWED_K):
                raise ValueError(f"{example_id}: fewer than five ranked candidates")
            scores = [float(candidate["adjusted_score"]) for candidate in candidates[:5]]
            if any(left + DEFAULT_EPSILON < right for left, right in zip(scores, scores[1:])):
                raise ValueError(f"{example_id}: final scores are not non-increasing")
            gold = [list(value) for value in item.get("gold_explanations", []) if value]
            if not gold:
                raise ValueError(f"{example_id}: missing DEV gold explanations")
            records.append(
                {
                    "example_id": example_id,
                    "group_id": example_id,
                    "dataset": str(item["dataset"]),
                    "domain": str(item["domain"]),
                    "score_mode": str(item["score_mode"]),
                    "candidates": candidates[:5],
                    "gold_explanations": gold,
                    "persisted_prefix_evaluation": item["prefix_evaluation"],
                }
            )
    if not records:
        raise ValueError(f"No {ranking_method} DEV records found")
    if {record["domain"] for record in records} != set(DOMAINS):
        raise ValueError("Expected both pooled text and ontology DEV records")
    return records


def support_scores(predicted: Sequence[Any], gold: Sequence[Any]) -> dict[str, float]:
    predicted_keys = {canonical_evidence_key(value) for value in predicted}
    gold_keys = {canonical_evidence_key(value) for value in gold}
    overlap = len(predicted_keys & gold_keys)
    precision = overlap / len(predicted_keys) if predicted_keys else 0.0
    recall = overlap / len(gold_keys) if gold_keys else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def best_support_scores(
    predicted: Sequence[Any], alternatives: Sequence[Sequence[Any]]
) -> dict[str, float]:
    values = [support_scores(predicted, gold) for gold in alternatives]
    return max(
        values, key=lambda row: row["f1"], default={"precision": 0.0, "recall": 0.0, "f1": 0.0}
    )


def prefix(record: Mapping[str, Any], k: int) -> dict[str, Any]:
    aggregate = fixed_k_support_aggregate(record["candidates"], k=k)
    return {
        "selected_k": int(aggregate["selected_k"]),
        "final_support_size": int(aggregate["final_support_size"]),
        "support_units": aggregate["support_units"],
        **best_support_scores(aggregate["support_units"], record["gold_explanations"]),
    }


def oracle_k(record: Mapping[str, Any]) -> int:
    rows = [(k, prefix(record, k)) for k in ALLOWED_K]
    return min(
        rows,
        key=lambda item: (-item[1]["f1"], item[1]["final_support_size"], item[0]),
    )[0]


def build_decision_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        best_k = oracle_k(record)
        for current_k, next_k in zip(ALLOWED_K, ALLOWED_K[1:]):
            features = compute_decision_features(record["candidates"], decision_rank=current_k)
            rows.append(
                {
                    "example_id": record["example_id"],
                    "group_id": record["group_id"],
                    "condition": record["dataset"],
                    "domain": record["domain"],
                    "decision_rank": current_k,
                    "current_k": current_k,
                    "next_allowed_k": next_k,
                    "oracle_best_k": best_k,
                    "oracle_decision": "CONTINUE" if best_k > current_k else "STOP",
                    "label_continue": int(best_k > current_k),
                    "features": features,
                }
            )
    return rows


def select_with_probabilities(
    record: Mapping[str, Any], probabilities: Mapping[tuple[str, int], float], threshold: float
) -> dict[str, Any]:
    selected_k = 1
    decisions: list[dict[str, Any]] = []
    for current_k, next_k in zip(ALLOWED_K, ALLOWED_K[1:]):
        probability = float(probabilities[(record["example_id"], current_k)])
        decision = "CONTINUE" if probability >= threshold else "STOP"
        decisions.append(
            {
                "current_k": current_k,
                "next_allowed_k": next_k,
                "predicted_continue_probability": probability,
                "threshold": threshold,
                "decision": decision,
            }
        )
        if decision == "STOP":
            break
        selected_k = next_k
    aggregate = fixed_k_support_aggregate(record["candidates"], k=selected_k)
    return {
        "selected_k": selected_k,
        "final_support_size": int(aggregate["final_support_size"]),
        "support_units": aggregate["support_units"],
        "decisions": decisions,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    distribution = Counter(int(row["selected_k"]) for row in rows)
    return {
        "examples": count,
        "precision": sum(float(row["precision"]) for row in rows) / count,
        "recall": sum(float(row["recall"]) for row in rows) / count,
        "f1": sum(float(row["f1"]) for row in rows) / count,
        "mean_selected_k": sum(int(row["selected_k"]) for row in rows) / count,
        "median_selected_k": statistics.median(int(row["selected_k"]) for row in rows),
        "selected_k_distribution": {
            str(k): {
                "count": distribution.get(k, 0),
                "percentage": 100.0 * distribution.get(k, 0) / count,
            }
            for k in ALLOWED_K
        },
        "mean_deduplicated_support_units": sum(int(row["final_support_size"]) for row in rows)
        / count,
    }


def evaluate_fixed(records: Sequence[Mapping[str, Any]], k: int) -> dict[str, Any]:
    rows = []
    for record in records:
        result = prefix(record, k)
        rows.append({"selected_k": k, **result})
    return summarize(rows)


def simulate_threshold(
    records: Sequence[Mapping[str, Any]], oof_rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    probabilities = {
        (str(row["example_id"]), int(row["current_k"])): float(row["oof_continue_probability"])
        for row in oof_rows
    }
    evaluated = []
    for record in records:
        selected = select_with_probabilities(record, probabilities, threshold)
        evaluated.append(
            {
                "selected_k": selected["selected_k"],
                "final_support_size": selected["final_support_size"],
                **best_support_scores(selected["support_units"], record["gold_explanations"]),
            }
        )
    summary = summarize(evaluated)
    classification = classification_metrics(oof_rows, threshold)
    return {
        "threshold": threshold,
        "support_precision": summary["precision"],
        "support_recall": summary["recall"],
        "support_f1": summary["f1"],
        "average_selected_k": summary["mean_selected_k"],
        "average_deduplicated_support_units": summary["mean_deduplicated_support_units"],
        "selected_k_distribution": {
            k: value["count"] for k, value in summary["selected_k_distribution"].items()
        },
        "continue_precision": classification["continue_precision"],
    }


def fit(args: argparse.Namespace) -> None:
    input_dir = args.input_dir
    output_dir = args.output_dir
    ranking_method = args.ranking_method
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected_existing = [path for path in output_dir.iterdir()]
    if unexpected_existing and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite only to "
            "replace a prior fit in this exact directory"
        )

    input_paths = [input_dir / name for name in INPUT_FILENAMES]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input artifacts: {missing}")
    input_hashes_before = {path.name: sha256(path) for path in input_paths}
    source_metrics = read_json(input_dir / "metrics.json")
    source_metadata = read_json(input_dir / "checkpoint_metadata.json")
    if source_metadata.get("test_rows_read") != 0 or source_metadata.get("test_gold_accessed"):
        raise ValueError("Source metadata does not establish a DEV-only input artifact")
    if source_metadata.get("adaptive_k_ready") is not True:
        raise ValueError("Source ranking artifact is not marked adaptive-k ready")
    if tuple(source_metrics.get("k_values", [])) != ALLOWED_K:
        raise ValueError("Source fixed-k evaluation does not exactly match allowed depths")

    records = load_clean_final_records(input_dir, ranking_method)
    decision_rows = build_decision_rows(records)
    thresholds: dict[str, float] = {}
    oof_by_domain: dict[str, list[dict[str, Any]]] = {}
    cv_by_domain: dict[str, Any] = {}
    sweeps: dict[str, Any] = {}
    model_descriptions: dict[str, Any] = {}

    for domain in DOMAINS:
        domain_records = [record for record in records if record["domain"] == domain]
        domain_rows = [row for row in decision_rows if row["domain"] == domain]
        oof_rows, cv_info = grouped_oof(domain_rows)
        fixed_k3 = evaluate_fixed(domain_records, 3)
        table = [simulate_threshold(domain_records, oof_rows, value) for value in THRESHOLD_GRID]
        chosen = min(
            table,
            key=lambda row: (
                -float(row["support_f1"]),
                float(row["average_selected_k"]),
                float(row["average_deduplicated_support_units"]),
                -float(row["support_precision"]),
                float(row["threshold"]),
            ),
        )
        threshold = float(chosen["threshold"])
        thresholds[domain] = threshold
        for row in oof_rows:
            row["threshold"] = threshold
            row["predicted_decision"] = (
                "CONTINUE" if row["oof_continue_probability"] >= threshold else "STOP"
            )
        oof_by_domain[domain] = oof_rows
        cv_by_domain[domain] = cv_info
        sweeps[domain] = {
            "scope": "pooled development out-of-fold probabilities only",
            "threshold_grid": list(THRESHOLD_GRID),
            "selection_rule": (
                "maximize pooled macro retrieval F1; ties: smaller mean selected k, "
                "fewer mean deduplicated support units, higher retrieval precision, "
                "then smaller threshold"
            ),
            "legacy_selection_rule_not_reused": (
                "The legacy compactness objective subject to 99% of fixed-k3 recall "
                "was incompatible with this run's explicit retrieval-F1 target."
            ),
            "fixed_k3_reference": fixed_k3,
            "chosen": chosen,
            "full_table": table,
        }
        scaler, model = fit_model(domain_rows)
        joblib.dump(scaler, output_dir / f"{domain}_scaler.joblib")
        joblib.dump(model, output_dir / f"{domain}_model.joblib")
        model_descriptions[domain] = model_description(domain, scaler, model, domain_rows)

    joblib.dump(thresholds, output_dir / "chosen_thresholds.joblib")
    for domain in DOMAINS:
        write_jsonl(output_dir / f"{domain}_oof_predictions.jsonl", oof_by_domain[domain])
        write_json(output_dir / f"{domain}_threshold_sweep.json", sweeps[domain])

    per_example: list[dict[str, Any]] = []
    for record in records:
        domain = record["domain"]
        probability_lookup = {
            (str(row["example_id"]), int(row["current_k"])): float(row["oof_continue_probability"])
            for row in oof_by_domain[domain]
        }
        selected = select_with_probabilities(record, probability_lookup, thresholds[domain])
        scores = best_support_scores(selected["support_units"], record["gold_explanations"])
        per_example.append(
            {
                "dataset": record["dataset"],
                "domain": domain,
                "split": "dev",
                "example_id": record["example_id"],
                "method": f"{ranking_method}_adaptive_v2_oof",
                "score_mode": record["score_mode"],
                "selected_k": selected["selected_k"],
                "final_support_size": selected["final_support_size"],
                "retrieved_evidence_units": selected["support_units"],
                **scores,
                "decisions": selected["decisions"],
            }
        )
    write_jsonl(output_dir / "per_example_adaptive_dev.jsonl", per_example)

    datasets = sorted({record["dataset"] for record in records})
    dataset_metrics: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_records = [record for record in records if record["dataset"] == dataset]
        adaptive_rows = [row for row in per_example if row["dataset"] == dataset]
        adaptive = summarize(adaptive_rows)
        fixed = {str(k): evaluate_fixed(dataset_records, k) for k in ALLOWED_K}
        domain = dataset_records[0]["domain"]
        domain_records = [record for record in records if record["domain"] == domain]
        domain_fixed = {k: evaluate_fixed(domain_records, k) for k in ALLOWED_K}
        globally_fixed_k = min(ALLOWED_K, key=lambda k: (-domain_fixed[k]["f1"], k))
        dataset_metrics.append(
            {
                "dataset": dataset,
                "domain": domain,
                "examples": len(dataset_records),
                "adaptive_oof": adaptive,
                "fixed": fixed,
                "delta_f1_adaptive_minus_k1": adaptive["f1"] - fixed["1"]["f1"],
                "delta_f1_adaptive_minus_k3": adaptive["f1"] - fixed["3"]["f1"],
                "policy_scope_best_globally_fixed_k": globally_fixed_k,
                "policy_scope_best_globally_fixed_f1_on_this_dataset": fixed[str(globally_fixed_k)][
                    "f1"
                ],
                "adaptive_improves_over_policy_scope_best_globally_fixed_k": (
                    adaptive["f1"] > fixed[str(globally_fixed_k)]["f1"]
                ),
            }
        )

    domain_metrics: dict[str, Any] = {}
    for domain in DOMAINS:
        domain_records = [record for record in records if record["domain"] == domain]
        adaptive_rows = [row for row in per_example if row["domain"] == domain]
        fixed = {str(k): evaluate_fixed(domain_records, k) for k in ALLOWED_K}
        best_k = min(ALLOWED_K, key=lambda k: (-fixed[str(k)]["f1"], k))
        adaptive = summarize(adaptive_rows)
        domain_metrics[domain] = {
            "threshold": thresholds[domain],
            "adaptive_oof": adaptive,
            "fixed": fixed,
            "best_globally_fixed_k": best_k,
            "best_globally_fixed_f1": fixed[str(best_k)]["f1"],
            "adaptive_delta_f1_vs_best_globally_fixed": adaptive["f1"] - fixed[str(best_k)]["f1"],
            "adaptive_improves_over_best_globally_fixed_k": adaptive["f1"]
            > fixed[str(best_k)]["f1"],
        }

    metrics = {
        "schema_version": "production_generator_d_adaptive_k_dev_v1",
        "status": "frozen_production_policy_after_clean_dev",
        "evaluation_estimate": "grouped out-of-fold DEV predictions",
        "metric_aggregation": "macro mean over examples",
        "ranking_method": ranking_method,
        "score_source": "persisted adjusted_score values from the selected ranking records",
        "allowed_k": list(ALLOWED_K),
        "policy_scope": "separate pooled text and ontology policies; no dataset-specific thresholds",
        "domains": domain_metrics,
        "datasets": dataset_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)

    fitted_files = [
        output_dir / f"{domain}_{kind}.joblib" for domain in DOMAINS for kind in ("scaler", "model")
    ] + [output_dir / "chosen_thresholds.joblib"]
    config = {
        "schema_version": "production_generator_d_adaptive_k_config_v1",
        "status": f"frozen_production_{ranking_method}_adaptive_k_policy",
        "source_policy_family": "adaptive_v2 sequential STOP/CONTINUE",
        "ranking_method": ranking_method,
        "score_key": "persisted adjusted_score",
        "allowed_k": list(ALLOWED_K),
        "transitions": [
            {"current_k": current, "continue_to_k": next_k, "feature_decision_rank": current}
            for current, next_k in zip(ALLOWED_K, ALLOWED_K[1:])
        ],
        "k3_to_k5_compatibility_note": (
            "The existing next-candidate features at decision rank 3 determine whether "
            "to continue from prefix 3 directly to allowed prefix 5; k=4 is never selected."
        ),
        "domains": list(DOMAINS),
        "dataset_specific_thresholds": False,
        "thresholds": thresholds,
        "feature_order": list(FEATURE_ORDER),
        "continuous_features_standardized": list(CONTINUOUS_FEATURES),
        "decision_rank_standardized": False,
        "model": {
            "class": "sklearn.linear_model.LogisticRegression",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_selection": sweeps["text"]["selection_rule"],
        "legacy_threshold_selection_departure": sweeps["text"]["legacy_selection_rule_not_reused"],
        "oracle_target": (
            "highest per-example DEV retrieval F1 among allowed k; ties by fewer "
            "deduplicated evidence units, then smaller k"
        ),
        "calibration": "5-fold grouped OOF by example_id, pooled separately by modality",
        "epsilon": DEFAULT_EPSILON,
        "inference_feature_contract": (
            "final ranked adjusted scores, rank position, and exact deduplicated candidate "
            "evidence identities only; no gold, answer, dataset, hop, proof-label, or split feature"
        ),
        "fitted_artifact_sha256": {path.name: sha256(path) for path in fitted_files},
        "full_dev_model_parameters": model_descriptions,
    }
    write_json(output_dir / "adaptive_k_config.json", config)
    write_json(
        output_dir / "grouped_cv_metadata.json",
        {"grouping": "example_id; no decision-row random split", "domains": cv_by_domain},
    )

    input_hashes_after = {path.name: sha256(path) for path in input_paths}
    lineage = {
        "schema_version": "production_generator_d_adaptive_k_lineage_v1",
        "status": f"frozen_production_{ranking_method}_adaptive_k_policy_after_clean_dev",
        "current_code_commit_hash": git_output("rev-parse", "HEAD"),
        "source_code_commit_hash": source_metadata.get("code_commit_hash"),
        "git_status_after": git_output("status", "--short").splitlines(),
        "python": platform.python_version(),
        "source_directory": str(input_dir),
        "source_artifact_sha256": input_hashes_before,
        "source_artifacts_unchanged": input_hashes_before == input_hashes_after,
        "source_checkpoint_metadata": source_metadata,
        "source_metrics_schema_version": source_metrics.get("schema_version"),
        "corpus_lineage": {
            dataset: {
                "dev_path": values.get("dev_path"),
                "dev_sha256": values.get("dev_sha256"),
                "checkpoint_path": values.get("checkpoint_path"),
                "checkpoint_sha256": values.get("checkpoint_sha256"),
                "checkpoint_training_objective": values.get("checkpoint_training_objective"),
                "final_reranking_score_mode": values.get("final_reranking_score_mode"),
            }
            for dataset, values in source_metadata.get("datasets", {}).items()
        },
        "input_scope": "three persisted clean DEV output artifacts for fitting",
        "accessed_files": [str(path) for path in input_paths],
        "fit_process_test_files_opened": 0,
        "test_scores_labels_answers_or_gold_used": False,
        "candidate_generation_rerun": False,
        "gnn_retrained": False,
        "gnn_objective_changed": False,
        "gnn_parameters_changed": False,
        "text_chain_parameters_changed": False,
        "proof_reranking_parameters_changed": False,
        "answer_generation_run": False,
        "historical_threshold_values_required_or_reused": False,
        "fitting_split": "dev only",
        "fitting_method": "grouped OOF calibration plus final refit on all clean DEV",
        "test_application_run": False,
    }
    write_json(output_dir / "lineage_checkpoint_metadata.json", lineage)

    display_method = {
        "gnn_only": "GNN-only",
        "sageqa_final": "SAGE-QA final",
        "cross_encoder": "cross-encoder",
        "final_sageqa": "final SAGE-QA",
    }[ranking_method]
    lines = [
        f"# Frozen production {display_method} adaptive-k policy (clean DEV)",
        "",
        f"This policy is fitted only from the `{ranking_method}` records and their persisted "
        "`adjusted_score` values in the clean production Generator-D DEV ranking artifact. "
        "Reported adaptive metrics are grouped out-of-fold DEV estimates; the fit process "
        "did not open TEST.",
        "",
        "## Rule",
        "",
        "Separate pooled text and ontology adaptive-v2 logistic policies make sequential "
        "STOP/CONTINUE decisions over allowed depths 1, 2, 3, and 5. The 3-to-5 transition "
        "uses the existing decision-rank-3 next-candidate features; depth 4 is never selected.",
        "",
        "## Frozen thresholds",
        "",
        f"- Text: `{thresholds['text']:.2f}`",
        f"- Ontology: `{thresholds['ontology']:.2f}`",
        "",
        "## DEV results",
        "",
        "| Dataset | P | R | F1 | Mean k | Median k | k=1 | k=2 | k=3 | k=5 | ΔF1 vs k=1 | ΔF1 vs k=3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_metrics:
        adaptive = row["adaptive_oof"]
        dist = adaptive["selected_k_distribution"]
        lines.append(
            f"| {row['dataset']} | {adaptive['precision']:.6f} | {adaptive['recall']:.6f} | "
            f"{adaptive['f1']:.6f} | {adaptive['mean_selected_k']:.4f} | "
            f"{adaptive['median_selected_k']:.1f} | {dist['1']['percentage']:.2f}% | "
            f"{dist['2']['percentage']:.2f}% | {dist['3']['percentage']:.2f}% | "
            f"{dist['5']['percentage']:.2f}% | {row['delta_f1_adaptive_minus_k1']:+.6f} | "
            f"{row['delta_f1_adaptive_minus_k3']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"## {display_method} fixed-depth DEV baselines",
            "",
            "| Dataset | k | P | R | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_metrics:
        for k in ALLOWED_K:
            fixed = row["fixed"][str(k)]
            lines.append(
                f"| {row['dataset']} | {k} | {fixed['precision']:.6f} | "
                f"{fixed['recall']:.6f} | {fixed['f1']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Pooled DEV comparison",
            "",
            "| Domain | Adaptive F1 | Best pooled fixed k | Best pooled fixed F1 | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for domain in DOMAINS:
        row = domain_metrics[domain]
        lines.append(
            f"| {domain} | {row['adaptive_oof']['f1']:.6f} | "
            f"{row['best_globally_fixed_k']} | {row['best_globally_fixed_f1']:.6f} | "
            f"{row['adaptive_delta_f1_vs_best_globally_fixed']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Freeze boundary",
            "",
            "The scaler/model parameters were refitted on all clean DEV after grouped-OOF threshold "
            "selection. The fit process opened no TEST file and used no candidate generation, GNN "
            "training, reranker modification, or answer generation. This directory is frozen for a "
            "later one-time TEST evaluation.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    output_files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    write_json(
        output_dir / "artifact_manifest.json",
        {
            "status": f"frozen_production_{ranking_method}_adaptive_k_policy",
            "ranking_method": ranking_method,
            "files": {path.name: sha256(path) for path in output_files},
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "thresholds": thresholds, "complete": True}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_k_sensitivity"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_gnn_adaptive_k"),
    )
    parser.add_argument(
        "--ranking-method",
        choices=RANKING_METHODS,
        default="gnn_only",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    fit(parse_args())
