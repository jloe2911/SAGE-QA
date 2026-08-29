"""Fit, freeze, and evaluate the development-calibrated adaptive-v2 policy.

The ``dev`` and ``test`` phases are deliberately separate commands.  The dev
phase uses only persisted development details, writes out-of-fold calibration
and frozen model artifacts, and never opens a test file.  The test phase first
verifies and loads those frozen artifacts and only then reads held-out details.
No candidate generation, GNN training, symbolic reranking, answer generation,
or external API call occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.adaptive_support_aggregation import (  # noqa: E402
    DEFAULT_EPSILON,
    MAX_ADAPTIVE_K,
    adaptive_support_aggregate,
    canonical_evidence_key,
    fixed_k_support_aggregate,
)
from evaluation.adaptive_support_aggregation_v2 import (  # noqa: E402
    CONTINUOUS_FEATURES,
    FEATURE_ORDER,
    AdaptiveV2Policy,
    adaptive_v2_support_aggregate,
    compute_decision_features,
    load_domain_policy,
    validate_ranked_candidates,
)


RANDOM_STATE = 42
THRESHOLD_GRID = tuple(index / 100 for index in range(101))
V1_POLICY_DIR = Path("outputs/adaptive_support_aggregation_v1")

CONDITION_CONFIG = {
    "HotpotQA": {"domain": "text", "method": "gnn_sageqa_text_chain"},
    "2WikiMultiHopQA": {"domain": "text", "method": "gnn_sageqa_text_chain"},
    "FamilyOWL_1hop": {"domain": "ontology", "method": "gnn_sageqa_proof"},
    "FamilyOWL_2hop": {"domain": "ontology", "method": "gnn_sageqa_proof"},
    "pizza_100_1hop": {"domain": "ontology", "method": "gnn_sageqa_proof"},
    "pizza_100_2hop": {"domain": "ontology", "method": "gnn_sageqa_proof"},
    "pizza_250_1hop": {"domain": "ontology", "method": "gnn_sageqa_proof"},
    "pizza_250_2hop": {"domain": "ontology", "method": "gnn_sageqa_proof"},
    "OWL2Bench_1hop": {"domain": "ontology", "method": "gnn_sageqa_proof"},
    "OWL2Bench_2hop": {"domain": "ontology", "method": "gnn_sageqa_proof"},
}
DOMAIN_CONDITIONS = {
    domain: tuple(
        condition
        for condition, config in CONDITION_CONFIG.items()
        if config["domain"] == domain
    )
    for domain in ("text", "ontology")
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detail_path(input_root: Path, condition: str, split: str) -> Path:
    method = CONDITION_CONFIG[condition]["method"]
    return input_root / condition / method / f"{split}_details.json"


def final_candidates(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = sorted(
        list(item.get("top5", []) or []),
        key=lambda candidate: int(candidate.get("rank", 999)),
    )[:MAX_ADAPTIVE_K]
    return validate_ranked_candidates(candidates)


def gold_explanations(item: Mapping[str, Any]) -> list[list[Any]]:
    alternatives = item.get("gold_explanations", []) or []
    if alternatives:
        return [list(explanation) for explanation in alternatives if explanation]
    text_support = item.get("gold_support_units", []) or []
    if text_support:
        return [list(text_support)]
    raise ValueError(f"{item.get('example_id', '<unknown>')}: missing gold support")


def support_scores(predicted: Sequence[Any], gold: Sequence[Any]) -> dict[str, float]:
    predicted_keys = {canonical_evidence_key(unit) for unit in predicted}
    gold_keys = {canonical_evidence_key(unit) for unit in gold}
    overlap = len(predicted_keys & gold_keys)
    precision = overlap / len(predicted_keys) if predicted_keys else 0.0
    recall = overlap / len(gold_keys) if gold_keys else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
    }


def best_support_scores(
    predicted: Sequence[Any], alternatives: Sequence[Sequence[Any]]
) -> dict[str, float]:
    best = {"support_precision": 0.0, "support_recall": 0.0, "support_f1": 0.0}
    for gold in alternatives:
        current = support_scores(predicted, gold)
        if current["support_f1"] > best["support_f1"]:
            best = current
    return best


def example_record(
    item: Mapping[str, Any], *, condition: str, domain: str
) -> dict[str, Any]:
    example_id = str(item.get("example_id") or "")
    if not example_id:
        raise ValueError(f"{condition}: empty example_id")
    return {
        "example_id": example_id,
        "group_id": example_id,
        "condition": condition,
        "domain": domain,
        "candidates": final_candidates(item),
        "gold_explanations": gold_explanations(item),
    }


def load_split(input_root: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for condition, config in CONDITION_CONFIG.items():
        path = detail_path(input_root, condition, split)
        hashes[condition] = sha256(path)
        items = read_json(path)
        records.extend(
            example_record(item, condition=condition, domain=config["domain"])
            for item in items
        )
    return records, hashes


def prefix_evaluation(record: Mapping[str, Any], k: int) -> dict[str, Any]:
    aggregate = fixed_k_support_aggregate(record["candidates"], k=k)
    return {
        "selected_k": int(aggregate["selected_k"]),
        "final_support_size": int(aggregate["final_support_size"]),
        **best_support_scores(aggregate["support_units"], record["gold_explanations"]),
    }


def oracle_for_example(record: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = []
    for k in range(1, min(MAX_ADAPTIVE_K, len(record["candidates"])) + 1):
        metrics = prefix_evaluation(record, k)
        prefixes.append({"k": k, **metrics})
    oracle = min(
        prefixes,
        key=lambda row: (
            -float(row["support_f1"]),
            int(row["final_support_size"]),
            int(row["k"]),
        ),
    )
    return {
        "example_id": record["example_id"],
        "condition": record["condition"],
        "domain": record["domain"],
        "oracle_best_k": int(oracle["k"]),
        "prefix_metrics": prefixes,
    }


def k_distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(int(row[key]) for row in rows)
    return {str(k): counts.get(k, 0) for k in range(1, MAX_ADAPTIVE_K + 1)}


def build_oracles(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oracle_rows = [oracle_for_example(record) for record in records]
    by_condition = {
        condition: k_distribution(
            [row for row in oracle_rows if row["condition"] == condition],
            "oracle_best_k",
        )
        for condition in CONDITION_CONFIG
    }
    pooled = {
        domain: k_distribution(
            [row for row in oracle_rows if row["domain"] == domain],
            "oracle_best_k",
        )
        for domain in ("text", "ontology")
    }
    lookup = {row["example_id"]: row for row in oracle_rows}
    if len(lookup) != len(oracle_rows):
        raise ValueError("Development example_id values must be globally unique")
    return oracle_rows, {"by_dataset": by_condition, "pooled": pooled, "lookup": lookup}


def build_decision_rows(
    records: Sequence[Mapping[str, Any]], oracle_lookup: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        oracle_k = int(oracle_lookup[record["example_id"]]["oracle_best_k"])
        for decision_rank in range(1, min(MAX_ADAPTIVE_K, len(record["candidates"]))):
            features = compute_decision_features(
                record["candidates"], decision_rank=decision_rank
            )
            rows.append(
                {
                    "example_id": record["example_id"],
                    "group_id": record["group_id"],
                    "condition": record["condition"],
                    "domain": record["domain"],
                    "decision_rank": decision_rank,
                    "oracle_decision": "CONTINUE" if decision_rank < oracle_k else "STOP",
                    "label_continue": int(decision_rank < oracle_k),
                    "features": features,
                }
            )
    return rows


def fit_model(rows: Sequence[Mapping[str, Any]]) -> tuple[StandardScaler, LogisticRegression]:
    x = np.asarray(
        [[float(row["features"][name]) for name in FEATURE_ORDER] for row in rows],
        dtype=float,
    )
    y = np.asarray([int(row["label_continue"]) for row in rows], dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("LogisticRegression requires both STOP and CONTINUE rows")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x[:, : len(CONTINUOUS_FEATURES)])
    model_x = np.column_stack([scaled, x[:, -1]])
    model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear",
        max_iter=1000,
    )
    model.fit(model_x, y)
    return scaler, model


def predict_rows(
    rows: Sequence[Mapping[str, Any]], scaler: StandardScaler, model: LogisticRegression
) -> np.ndarray:
    x = np.asarray(
        [[float(row["features"][name]) for name in FEATURE_ORDER] for row in rows],
        dtype=float,
    )
    scaled = scaler.transform(x[:, : len(CONTINUOUS_FEATURES)])
    model_x = np.column_stack([scaled, x[:, -1]])
    return model.predict_proba(model_x)[:, 1]


def grouped_oof(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["group_id"])].append(index)
    group_ids = sorted(groups)
    group_labels = np.asarray(
        [max(int(rows[index]["label_continue"]) for index in groups[group]) for group in group_ids]
    )
    positive_groups = int(group_labels.sum())
    negative_groups = len(group_labels) - positive_groups

    if min(positive_groups, negative_groups) >= 5:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(group_ids, group_labels))
        strategy = "5-fold StratifiedKFold over unique example_id groups by any-CONTINUE"
    else:
        n_splits = min(5, len(group_ids))
        group_splitter = GroupKFold(n_splits=n_splits)
        dummy = np.zeros(len(group_ids))
        splits = list(group_splitter.split(dummy, groups=group_ids))
        strategy = f"{n_splits}-fold GroupKFold over unique example_id groups"

    probabilities = np.full(len(rows), np.nan, dtype=float)
    folds: list[dict[str, Any]] = []
    for fold_index, (train_group_indexes, valid_group_indexes) in enumerate(splits):
        train_groups = {group_ids[index] for index in train_group_indexes}
        valid_groups = {group_ids[index] for index in valid_group_indexes}
        if train_groups & valid_groups:
            raise AssertionError("Grouped validation leaked an example_id across folds")
        train_rows = [row for row in rows if row["group_id"] in train_groups]
        valid_indexes = [
            index for index, row in enumerate(rows) if row["group_id"] in valid_groups
        ]
        valid_rows = [rows[index] for index in valid_indexes]
        scaler, model = fit_model(train_rows)
        probabilities[valid_indexes] = predict_rows(valid_rows, scaler, model)
        folds.append(
            {
                "fold": fold_index,
                "train_examples": len(train_groups),
                "validation_examples": len(valid_groups),
                "train_decision_rows": len(train_rows),
                "validation_decision_rows": len(valid_rows),
                "train_continue": sum(int(row["label_continue"]) for row in train_rows),
                "validation_continue": sum(
                    int(row["label_continue"]) for row in valid_rows
                ),
                "validation_example_ids": sorted(valid_groups),
            }
        )
    if np.isnan(probabilities).any():
        raise AssertionError("Every development decision row must receive one OOF probability")
    output = [
        dict(row, oof_continue_probability=float(probabilities[index]))
        for index, row in enumerate(rows)
    ]
    return output, {
        "strategy": strategy,
        "folds": folds,
        "unique_examples": len(group_ids),
        "positive_example_groups": positive_groups,
        "negative_example_groups": negative_groups,
    }


def classification_metrics(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    y = np.asarray([int(row["label_continue"]) for row in rows], dtype=int)
    probabilities = np.asarray(
        [float(row["oof_continue_probability"]) for row in rows], dtype=float
    )
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return {
        "decision_rows": len(rows),
        "stop_count": int((y == 0).sum()),
        "continue_count": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, probabilities)) if int(y.sum()) else None,
        "continue_precision": float(precision),
        "continue_recall": float(recall),
        "continue_f1": float(f1),
        "confusion_matrix": {
            "labels": ["STOP", "CONTINUE"],
            "matrix": matrix.astype(int).tolist(),
        },
        "classification_threshold": threshold,
    }


def summarize_example_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero examples")
    return {
        "examples": len(rows),
        "support_precision": sum(float(row["support_precision"]) for row in rows) / len(rows),
        "support_recall": sum(float(row["support_recall"]) for row in rows) / len(rows),
        "support_f1": sum(float(row["support_f1"]) for row in rows) / len(rows),
        "average_selected_k": sum(int(row["selected_k"]) for row in rows) / len(rows),
        "selected_k_distribution": k_distribution(rows, "selected_k"),
        "average_deduplicated_support_units": sum(
            int(row["final_support_size"]) for row in rows
        )
        / len(rows),
    }


def evaluate_aggregations(
    records: Sequence[Mapping[str, Any]], aggregations: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows = []
    for record in records:
        aggregate = aggregations[record["example_id"]]
        rows.append(
            {
                "example_id": record["example_id"],
                "condition": record["condition"],
                "selected_k": aggregate["selected_k"],
                "final_support_size": aggregate["final_support_size"],
                **best_support_scores(
                    aggregate["support_units"], record["gold_explanations"]
                ),
            }
        )
    return summarize_example_metrics(rows)


def fixed_metrics(records: Sequence[Mapping[str, Any]], k: int) -> dict[str, Any]:
    aggregates = {
        record["example_id"]: fixed_k_support_aggregate(record["candidates"], k=k)
        for record in records
    }
    return evaluate_aggregations(records, aggregates)


def simulate_oof_threshold(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    probability = {
        (str(row["example_id"]), int(row["decision_rank"])): float(
            row["oof_continue_probability"]
        )
        for row in rows
    }
    aggregates = {}
    for record in records:
        selected_k = 1
        for decision_rank in range(1, min(MAX_ADAPTIVE_K, len(record["candidates"]))):
            if probability[(record["example_id"], decision_rank)] < threshold:
                break
            selected_k = decision_rank + 1
        aggregates[record["example_id"]] = fixed_k_support_aggregate(
            record["candidates"], k=selected_k
        )
    return {
        "threshold": threshold,
        **evaluate_aggregations(records, aggregates),
        "continue_precision": classification_metrics(rows, threshold)["continue_precision"],
    }


def mark_pareto(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in table:
        dominated = any(
            other is not row
            and float(other["support_precision"]) >= float(row["support_precision"])
            and float(other["support_recall"]) >= float(row["support_recall"])
            and float(other["average_deduplicated_support_units"])
            <= float(row["average_deduplicated_support_units"])
            and (
                float(other["support_precision"]) > float(row["support_precision"])
                or float(other["support_recall"]) > float(row["support_recall"])
                or float(other["average_deduplicated_support_units"])
                < float(row["average_deduplicated_support_units"])
            )
            for other in table
        )
        row["pareto_optimal"] = not dominated
    return table


def choose_threshold(
    table: Sequence[Mapping[str, Any]], fixed_k3_recall: float
) -> tuple[dict[str, Any], bool]:
    required_recall = 0.99 * fixed_k3_recall
    feasible = [row for row in table if float(row["support_recall"]) >= required_recall]
    if feasible:
        chosen = min(
            feasible,
            key=lambda row: (
                float(row["average_deduplicated_support_units"]),
                -float(row["support_f1"]),
                float(row["average_selected_k"]),
                -float(row["continue_precision"]),
                float(row["threshold"]),
            ),
        )
        return dict(chosen), True
    chosen = min(
        table,
        key=lambda row: (
            -float(row["support_recall"]),
            -float(row["support_f1"]),
            float(row["average_selected_k"]),
            -float(row["continue_precision"]),
            float(row["threshold"]),
        ),
    )
    return dict(chosen), False


def model_description(
    domain: str,
    scaler: StandardScaler,
    model: LogisticRegression,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "domain": domain,
        "feature_order": list(FEATURE_ORDER),
        "continuous_features_standardized": list(CONTINUOUS_FEATURES),
        "decision_rank_standardized": False,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients_in_model_feature_order": model.coef_[0].tolist(),
        "coefficient_by_feature": dict(zip(FEATURE_ORDER, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0]),
        "classes": model.classes_.astype(int).tolist(),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "solver": "liblinear",
        "training_examples": len({str(row["example_id"]) for row in rows}),
        "training_decision_rows": len(rows),
        "training_stop_rows": sum(1 - int(row["label_continue"]) for row in rows),
        "training_continue_rows": sum(int(row["label_continue"]) for row in rows),
        "training_counts_by_condition": {
            condition: {
                "examples": len(
                    {
                        str(row["example_id"])
                        for row in rows
                        if row["condition"] == condition
                    }
                ),
                "decision_rows": sum(row["condition"] == condition for row in rows),
            }
            for condition in DOMAIN_CONDITIONS[domain]
        },
    }


def dev_phase(input_root: Path, output_dir: Path) -> None:
    records, input_hashes = load_split(input_root, "dev")
    oracle_rows, oracle_info = build_oracles(records)
    decision_rows = build_decision_rows(records, oracle_info["lookup"])

    write_json(
        output_dir / "oracle_best_k_dev.json",
        {
            "scope": "development only",
            "oracle_definition": (
                "highest per-example Support F1; then fewest deduplicated evidence "
                "units; then smallest k"
            ),
            "evidence_semantics": {
                "text": "exact stored gold_support_units and deduplicated SENT evidence identity",
                "ontology": "best-matching stored gold_explanations and exact evidence identity",
            },
            "input_sha256": input_hashes,
            "distributions": {
                "by_dataset": oracle_info["by_dataset"],
                "pooled_text": oracle_info["pooled"]["text"],
                "pooled_ontology": oracle_info["pooled"]["ontology"],
            },
            "by_example": oracle_rows,
        },
    )

    chosen_thresholds: dict[str, float] = {}
    cv_results: dict[str, Any] = {}
    model_descriptions: dict[str, Any] = {}
    dev_tradeoff: dict[str, Any] = {}

    for domain in ("text", "ontology"):
        domain_records = [record for record in records if record["domain"] == domain]
        domain_rows = [row for row in decision_rows if row["domain"] == domain]
        oof_rows, cv_info = grouped_oof(domain_rows)
        fixed_k1 = fixed_metrics(domain_records, 1)
        fixed_k3 = fixed_metrics(domain_records, 3)
        table = mark_pareto(
            [
                simulate_oof_threshold(domain_records, oof_rows, threshold)
                for threshold in THRESHOLD_GRID
            ]
        )
        chosen, constraint_satisfied = choose_threshold(
            table, float(fixed_k3["support_recall"])
        )
        threshold = float(chosen["threshold"])
        chosen_thresholds[domain] = threshold

        for row in oof_rows:
            row["threshold"] = threshold
            row["predicted_decision"] = (
                "CONTINUE"
                if float(row["oof_continue_probability"]) >= threshold
                else "STOP"
            )
        write_jsonl(output_dir / f"{domain}_oof_predictions.jsonl", oof_rows)

        by_rank = {
            str(rank): classification_metrics(
                [row for row in oof_rows if int(row["decision_rank"]) == rank],
                threshold,
            )
            for rank in range(1, MAX_ADAPTIVE_K)
        }
        by_dataset = {
            condition: classification_metrics(
                [row for row in oof_rows if row["condition"] == condition],
                threshold,
            )
            for condition in DOMAIN_CONDITIONS[domain]
        }
        cv_results[domain] = {
            **cv_info,
            "pooled": classification_metrics(oof_rows, threshold),
            "by_decision_rank": by_rank,
            "by_dataset": by_dataset,
        }

        sweep_artifact = {
            "domain": domain,
            "scope": "pooled development out-of-fold probabilities only",
            "threshold_grid": list(THRESHOLD_GRID),
            "fixed_k1": fixed_k1,
            "fixed_k3": fixed_k3,
            "required_recall": 0.99 * float(fixed_k3["support_recall"]),
            "recall_constraint_satisfied": constraint_satisfied,
            "selection_rule": (
                "minimize average deduplicated support units subject to at least 99% "
                "of fixed-k3 recall; ties: higher F1, smaller average k, higher "
                "CONTINUE precision, then smaller threshold"
            ),
            "chosen_threshold": threshold,
            "chosen_row": chosen,
            "full_table": table,
            "pareto_frontier": [row for row in table if row["pareto_optimal"]],
        }
        write_json(output_dir / f"{domain}_threshold_sweep.json", sweep_artifact)
        dev_tradeoff[domain] = {
            "fixed_k1": fixed_k1,
            "fixed_k3": fixed_k3,
            "adaptive_v2_oof": chosen,
        }

        scaler, model = fit_model(domain_rows)
        joblib.dump(scaler, output_dir / f"{domain}_scaler.joblib")
        joblib.dump(model, output_dir / f"{domain}_model.joblib")
        model_descriptions[domain] = model_description(domain, scaler, model, domain_rows)

    joblib.dump(chosen_thresholds, output_dir / "chosen_thresholds.joblib")
    model_files = [
        output_dir / f"{domain}_{kind}.joblib"
        for domain in ("text", "ontology")
        for kind in ("scaler", "model")
    ] + [output_dir / "chosen_thresholds.joblib"]
    artifact_hashes = {path.name: sha256(path) for path in model_files}
    write_json(
        output_dir / "model_coefficients_and_features.json",
        {
            "model_class": "sklearn.linear_model.LogisticRegression",
            "feature_order": list(FEATURE_ORDER),
            "domains": model_descriptions,
        },
    )
    write_json(
        output_dir / "grouped_cv_results.json",
        {
            "grouping": "example_id; no decision-row random split",
            "domains": cv_results,
        },
    )
    write_json(output_dir / "development_fixed_vs_adaptive.json", dev_tradeoff)
    write_json(
        output_dir / "chosen_thresholds.json",
        {
            "status": "frozen_after_development",
            "thresholds": chosen_thresholds,
            "feature_order": list(FEATURE_ORDER),
            "epsilon": DEFAULT_EPSILON,
            "k_max": MAX_ADAPTIVE_K,
            "random_state": RANDOM_STATE,
            "development_input_sha256": input_hashes,
            "fitted_artifact_sha256": artifact_hashes,
        },
    )


def verify_frozen_policy(output_dir: Path) -> dict[str, Any]:
    manifest = read_json(output_dir / "chosen_thresholds.json")
    if manifest.get("status") != "frozen_after_development":
        raise ValueError("Adaptive-v2 policy is not marked frozen after development")
    for filename, expected in manifest["fitted_artifact_sha256"].items():
        actual = sha256(output_dir / filename)
        if actual != expected:
            raise ValueError(f"Frozen policy hash mismatch for {filename}")
    return manifest


def pooled_metrics(
    by_condition: Mapping[str, Mapping[str, Mapping[str, Any]]], conditions: Sequence[str]
) -> dict[str, Any]:
    systems = next(iter(by_condition.values())).keys()
    output: dict[str, Any] = {}
    for system in systems:
        entries = [by_condition[condition][system] for condition in conditions]
        total = sum(int(entry["examples"]) for entry in entries)
        output[system] = {
            "examples": total,
            **{
                metric: sum(float(entry[metric]) * int(entry["examples"]) for entry in entries)
                / total
                for metric in (
                    "support_precision",
                    "support_recall",
                    "support_f1",
                    "average_selected_k",
                    "average_deduplicated_support_units",
                )
            },
            "selected_k_distribution": {
                str(k): sum(
                    int(entry["selected_k_distribution"][str(k)]) for entry in entries
                )
                for k in range(1, MAX_ADAPTIVE_K + 1)
            },
        }
    return output


def regression_check(actual: float, expected: float, tolerance: float = 1e-12) -> dict[str, Any]:
    return {
        "actual": actual,
        "expected": expected,
        "absolute_delta": abs(actual - expected),
        "tolerance": tolerance,
        "matched": abs(actual - expected) <= tolerance,
    }


def fixed_regression(
    input_root: Path,
    condition: str,
    systems: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    method_dir = input_root / condition / CONDITION_CONFIG[condition]["method"]
    sensitivity_path = method_dir / "test_metrics.json"
    sensitivity = read_json(sensitivity_path)
    fixed1 = systems["fixed_k1"]
    k1 = {
        "support_f1": regression_check(
            float(fixed1["support_f1"]), float(sensitivity["best_set_f1@1"])
        )
    }
    if CONDITION_CONFIG[condition]["domain"] == "ontology":
        k1.update(
            {
                "support_precision": regression_check(
                    float(fixed1["support_precision"]), float(sensitivity["best_precision@1"])
                ),
                "support_recall": regression_check(
                    float(fixed1["support_recall"]), float(sensitivity["best_recall@1"])
                ),
            }
        )
    else:
        k1["precision_recall_note"] = (
            "Text test_metrics persists zeros for best_precision@1/best_recall@1; "
            "only the valid best_set_f1@1 sensitivity value is checked."
        )

    if CONDITION_CONFIG[condition]["domain"] == "text":
        expected = {
            "support_precision": float(sensitivity["precision@3"]),
            "support_recall": float(sensitivity["recall@3"]),
            "support_f1": float(sensitivity["set_f1@3"]),
        }
        k3_source = sensitivity_path
    else:
        k3_source = method_dir / "metrics_top3_gpt_4_1_mini.json"
        persisted = read_json(k3_source)["metrics"]
        expected = {
            "support_precision": float(persisted["sp_prec"]),
            "support_recall": float(persisted["sp_recall"]),
            "support_f1": float(persisted["sp_f1"]),
        }
    fixed3 = systems["fixed_k3"]
    return {
        "fixed_k1_sensitivity": {"source": str(sensitivity_path), **k1},
        "fixed_k3_manuscript_or_thesis": {
            "source": str(k3_source),
            **{
                metric: regression_check(float(fixed3[metric]), expected[metric])
                for metric in expected
            },
        },
    }


def test_phase(input_root: Path, output_dir: Path, v1_policy_dir: Path) -> None:
    manifest = verify_frozen_policy(output_dir)
    policies = {
        domain: load_domain_policy(output_dir, domain=domain)
        for domain in ("text", "ontology")
    }
    v1_manifest = read_json(v1_policy_dir / "chosen_tau.json")
    v1_tau = float(v1_manifest["selected_tau"])

    # Test files are first opened here, after all v2 settings and thresholds are frozen.
    records, test_hashes = load_split(input_root, "test")
    by_condition: dict[str, dict[str, Any]] = {}
    decision_audit_rows: list[dict[str, Any]] = []
    regressions: dict[str, Any] = {}

    for condition, config in CONDITION_CONFIG.items():
        condition_records = [record for record in records if record["condition"] == condition]
        system_aggregates: dict[str, dict[str, Mapping[str, Any]]] = {
            "fixed_k1": {},
            "fixed_k3": {},
            "adaptive_v1": {},
            "adaptive_v2": {},
        }
        for record in condition_records:
            example_id = record["example_id"]
            system_aggregates["fixed_k1"][example_id] = fixed_k_support_aggregate(
                record["candidates"], k=1
            )
            system_aggregates["fixed_k3"][example_id] = fixed_k_support_aggregate(
                record["candidates"], k=3
            )
            system_aggregates["adaptive_v1"][example_id] = adaptive_support_aggregate(
                record["candidates"], tau=v1_tau
            )
            v2 = adaptive_v2_support_aggregate(
                record["candidates"], policy=policies[config["domain"]]
            )
            system_aggregates["adaptive_v2"][example_id] = v2
            for decision in v2["decisions"]:
                decision_audit_rows.append(
                    {
                        "example_id": example_id,
                        "condition": condition,
                        "domain_configuration": config["domain"],
                        **decision,
                    }
                )

        systems = {
            system: evaluate_aggregations(condition_records, aggregates)
            for system, aggregates in system_aggregates.items()
        }
        by_condition[condition] = systems
        regressions[condition] = fixed_regression(input_root, condition, systems)

    original_v1 = read_json(v1_policy_dir / "aggregate_test_metrics.json")
    v1_regression = {}
    for condition, persisted_systems in original_v1["by_condition"].items():
        current = by_condition[condition]["adaptive_v1"]
        persisted = persisted_systems["adaptive"]
        v1_regression[condition] = {
            metric: regression_check(float(current[metric]), float(persisted[metric]))
            for metric in (
                "support_precision",
                "support_recall",
                "support_f1",
                "average_selected_k",
            )
        }
        v1_regression[condition]["average_deduplicated_support_units"] = regression_check(
            float(current["average_deduplicated_support_units"]),
            float(persisted["average_unique_evidence_units"]),
        )

    pooled = {
        "text": pooled_metrics(by_condition, DOMAIN_CONDITIONS["text"]),
        "ontology": pooled_metrics(by_condition, DOMAIN_CONDITIONS["ontology"]),
        "all": pooled_metrics(by_condition, tuple(CONDITION_CONFIG)),
    }
    write_json(
        output_dir / "test_metrics_by_dataset.json",
        {
            "policy_status": manifest["status"],
            "test_input_sha256": test_hashes,
            "adaptive_v1_tau": v1_tau,
            "by_dataset": by_condition,
        },
    )
    write_json(output_dir / "pooled_test_metrics.json", pooled)
    write_json(
        output_dir / "fixed_vs_adaptive_comparison.json",
        {
            "by_dataset": by_condition,
            "pooled": pooled,
            "adaptive_v1_calibration_note": (
                "Existing v1 tau was frozen on pooled FamilyOWL and OWL2Bench "
                "development only; it is applied unchanged to all conditions for comparison."
            ),
        },
    )
    write_jsonl(output_dir / "adaptive_v2_test_decisions.jsonl", decision_audit_rows)
    write_json(
        output_dir / "regression_results.json",
        {
            "fixed_default_unchanged": True,
            "adaptive_v2_opt_in": True,
            "fixed_by_condition": regressions,
            "adaptive_v1_original_four_conditions": v1_regression,
            "v1_policy_source_sha256": {
                "chosen_tau.json": sha256(v1_policy_dir / "chosen_tau.json"),
                "aggregate_test_metrics.json": sha256(
                    v1_policy_dir / "aggregate_test_metrics.json"
                ),
                "adaptive_support_aggregation.py": sha256(
                    Path("evaluation/adaptive_support_aggregation.py")
                ),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "test"), required=True)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/full_results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adaptive_support_aggregation_v2"),
    )
    parser.add_argument("--v1-policy-dir", type=Path, default=V1_POLICY_DIR)
    args = parser.parse_args()

    if args.phase == "dev":
        dev_phase(args.input_root, args.output_dir)
    else:
        test_phase(args.input_root, args.output_dir, args.v1_policy_dir)


if __name__ == "__main__":
    main()
