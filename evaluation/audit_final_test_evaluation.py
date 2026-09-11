"""Read-only diagnostic audit of frozen production TEST evaluation artifacts.

This script never generates answers, changes retrieval, or writes into the
frozen end-to-end directory.  It independently recomputes the persisted scores
and writes diagnostics under ``outputs/audits/final_test_evaluation_audit``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_owl_qa_predictions import (  # noqa: E402
    answer_set_scores,
    best_support_scores,
    normalize_answer as normalize_ontology_answer,
    split_answer_items,
)
from evaluation.hotpot_official_eval import (  # noqa: E402
    exact_match_score,
    f1_score,
    normalize_answer as normalize_text_answer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN = ROOT / "outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end"
DEFAULT_RETRIEVAL = (
    ROOT
    / "outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/per_example_test_retrieval.jsonl"
)
DEFAULT_OUTPUT = ROOT / "outputs/audits/final_test_evaluation_audit"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def current_answer_scores(
    domain: str, prediction: str, gold: str
) -> tuple[float, float, float, float]:
    if domain == "ontology":
        return tuple(float(value) for value in answer_set_scores(prediction, gold))
    f1, precision, recall = f1_score(prediction, gold)
    return float(exact_match_score(prediction, gold)), float(f1), float(precision), float(recall)


def general_normalize(value: str) -> str:
    """Hotpot/SQuAD normalization without benchmark-specific boolean aliases."""
    value = str(value).lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def bool_after_general_normalization(value: str) -> str:
    normalized = general_normalize(value)
    if normalized in {"true", "yes", "1", "entailed", "correct"}:
        return "yes"
    if normalized in {"false", "no", "0", "not entailed", "incorrect"}:
        return "no"
    return normalized


def boolean_diagnostic_scores(prediction: str, gold: str) -> tuple[float, float, float, float]:
    pred = bool_after_general_normalization(prediction)
    target = bool_after_general_normalization(gold)
    em = float(pred == target)
    if pred in {"yes", "no"} or target in {"yes", "no"}:
        value = 1.0 if pred == target else 0.0
        return em, value, value, value
    return current_answer_scores("ontology", prediction, gold)


def text_native_identity(unit: str) -> tuple[str, int] | None:
    if not str(unit).startswith("SENT::"):
        return None
    parts = str(unit).split("::", 3)
    if len(parts) != 4:
        return None
    try:
        return parts[1], int(parts[2])
    except ValueError:
        return None


def canonical_ontology_unit(unit: str) -> str:
    """Conservative serialization diagnostic, not an asserted metric policy."""
    value = str(unit).strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"<([^<>]+)>", r"\1", value)
    value = re.sub(r"https?://[^\s#>]+#([^\s>]+)", r"\1", value)
    return value


def canonical_support_unit(domain: str, unit: str) -> Any:
    if domain == "text":
        return text_native_identity(unit) or ("RAW", str(unit))
    return canonical_ontology_unit(unit)


def score_sets(predicted: Sequence[Any], gold: Sequence[Any]) -> dict[str, float]:
    pred_set, gold_set = set(predicted), set(gold)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"em": float(pred_set == gold_set), "f1": f1, "prec": precision, "recall": recall}


def best_canonical_support(
    domain: str, predicted: Sequence[str], alternatives: Sequence[Sequence[str]]
) -> dict[str, Any]:
    pred = [canonical_support_unit(domain, unit) for unit in predicted]
    scored = []
    for index, gold in enumerate(alternatives):
        canonical_gold = [canonical_support_unit(domain, unit) for unit in gold]
        scored.append((score_sets(pred, canonical_gold), index, canonical_gold))
    if not scored:
        return {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0, "best_index": None}
    score, index, _ = max(scored, key=lambda item: item[0]["f1"])
    return {**score, "best_index": index}


def support_failure_reason(predicted: Sequence[str], alternatives: Sequence[Sequence[str]]) -> str:
    pred = set(predicted)
    gold_sets = [set(gold) for gold in alternatives]
    if any(pred == gold for gold in gold_sets):
        return "exact"
    if any(gold and gold < pred for gold in gold_sets):
        return "complete_but_nonminimal"
    overlaps = [(len(pred & gold), gold) for gold in gold_sets]
    overlap, gold = max(overlaps, default=(0, set()), key=lambda item: item[0])
    if overlap == 0:
        return "missing_all_gold_evidence"
    if pred < gold:
        return "missing_gold_evidence_only"
    return "genuine_partial_support"


def normalized_answer(domain: str, value: str) -> Any:
    return split_answer_items(value) if domain == "ontology" else normalize_text_answer(value)


def aggregate(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, float | None]:
    result = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = fmean(values) if values else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    freeze = json.loads((args.frozen_dir / "generation_freeze.json").read_text(encoding="utf-8"))
    persisted_metrics = json.loads((args.frozen_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = load_jsonl(args.frozen_dir / "per_example_end_to_end.jsonl")
    retrieval_rows = load_jsonl(args.retrieval)
    retrieval = {row["example_id"]: row for row in retrieval_rows}

    integrity = {
        "prediction_hash_expected": freeze["predictions_sha256"],
        "prediction_hash_actual": sha256(args.frozen_dir / "predictions.jsonl"),
        "retrieval_hash_expected": freeze["retrieval_sha256"],
        "retrieval_hash_actual": sha256(args.retrieval),
        "prediction_count_expected": freeze["prediction_count"],
        "prediction_count_actual": len(load_jsonl(args.frozen_dir / "predictions.jsonl")),
        "per_example_count": len(rows),
    }
    integrity["hashes_match"] = (
        integrity["prediction_hash_expected"] == integrity["prediction_hash_actual"]
        and integrity["retrieval_hash_expected"] == integrity["retrieval_hash_actual"]
    )

    audited_rows = []
    score_mismatches = []
    boolean_changed = []
    support_canonical_changed = []
    duplicate_rows = []
    tie_rows = []
    failure_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for row in rows:
        dataset, domain = str(row["dataset"]), str(row["domain"])
        method, setting = str(row["method"]), str(row["setting"])
        prediction, gold = str(row["predicted_answer"]), str(row["gold_answer"])
        answer = current_answer_scores(domain, prediction, gold)
        for field, actual in zip(
            ("answer_em", "answer_f1", "answer_precision", "answer_recall"), answer
        ):
            if not close(row[field], actual):
                score_mismatches.append(
                    {
                        "example_id": row["example_id"],
                        "field": field,
                        "persisted": row[field],
                        "recomputed": actual,
                    }
                )

        answer_diag = answer
        bool_pred = bool_after_general_normalization(prediction)
        bool_gold = bool_after_general_normalization(gold)
        if bool_pred in {"yes", "no"} or bool_gold in {"yes", "no"}:
            answer_diag = boolean_diagnostic_scores(prediction, gold)
            if any(not close(a, b) for a, b in zip(answer, answer_diag)):
                boolean_changed.append(
                    {
                        "example_id": row["example_id"],
                        "dataset": dataset,
                        "method": method,
                        "setting": setting,
                        "prediction": prediction,
                        "gold": gold,
                        "current_normalized_prediction": normalized_answer(domain, prediction),
                        "current_normalized_gold": normalized_answer(domain, gold),
                        "diagnostic_normalized_prediction": bool_pred,
                        "diagnostic_normalized_gold": bool_gold,
                        "current_scores": dict(zip(("em", "f1", "precision", "recall"), answer)),
                        "diagnostic_scores": dict(
                            zip(("em", "f1", "precision", "recall"), answer_diag)
                        ),
                    }
                )

        support = list(row["selected_support"])
        alternatives = retrieval[row["example_id"]].get("gold_explanations") or []
        support_diag = None
        failure_reason = None
        if row["support_evaluable"]:
            current_support = best_support_scores(support, alternatives)
            for field, current_field in (
                ("support_em", "em"),
                ("support_f1", "f1"),
                ("support_precision", "prec"),
                ("support_recall", "recall"),
            ):
                if not close(row[field], current_support[current_field]):
                    score_mismatches.append(
                        {
                            "example_id": row["example_id"],
                            "field": field,
                            "persisted": row[field],
                            "recomputed": current_support[current_field],
                        }
                    )
            support_diag = best_canonical_support(domain, support, alternatives)
            if any(
                not close(current_support[name], support_diag[name])
                for name in ("em", "f1", "prec", "recall")
            ):
                support_canonical_changed.append(
                    {
                        "example_id": row["example_id"],
                        "dataset": dataset,
                        "method": method,
                        "setting": setting,
                        "selected_support": support,
                        "gold_explanations": alternatives,
                        "current": {
                            name: current_support[name] for name in ("em", "f1", "prec", "recall")
                        },
                        "canonical_diagnostic": {
                            name: support_diag[name] for name in ("em", "f1", "prec", "recall")
                        },
                    }
                )
            failure_reason = support_failure_reason(support, alternatives)
            failure_counts[(dataset, method, setting)][failure_reason] += 1
            if len(support) != len(set(support)) or any(
                len(gold_alt) != len(set(gold_alt)) for gold_alt in alternatives
            ):
                duplicate_rows.append(
                    {
                        "example_id": row["example_id"],
                        "dataset": dataset,
                        "method": method,
                        "setting": setting,
                    }
                )

            scored_alternatives = [score_sets(support, gold_alt) for gold_alt in alternatives]
            if scored_alternatives:
                best_f1 = max(score["f1"] for score in scored_alternatives)
                ties = [score for score in scored_alternatives if close(score["f1"], best_f1)]
                if len(ties) > 1 and len({(score["prec"], score["recall"]) for score in ties}) > 1:
                    tie_rows.append(
                        {
                            "example_id": row["example_id"],
                            "dataset": dataset,
                            "method": method,
                            "setting": setting,
                            "best_f1": best_f1,
                            "tied_precision_recall": sorted(
                                {(score["prec"], score["recall"]) for score in ties}
                            ),
                        }
                    )

        if support_diag is None:
            joint_diag = (None, None, None, None)
        else:
            joint_precision = answer_diag[2] * support_diag["prec"]
            joint_recall = answer_diag[3] * support_diag["recall"]
            joint_f1 = (
                0.0
                if joint_precision + joint_recall == 0
                else 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
            )
            joint_diag = (
                answer_diag[0] * support_diag["em"],
                joint_f1,
                joint_precision,
                joint_recall,
            )
            for field, recomputed in zip(
                ("joint_em", "joint_f1", "joint_precision", "joint_recall"), joint_diag
            ):
                current_support = best_support_scores(support, alternatives)
                current_joint_precision = answer[2] * current_support["prec"]
                current_joint_recall = answer[3] * current_support["recall"]
                current_joint_f1 = (
                    0.0
                    if current_joint_precision + current_joint_recall == 0
                    else 2
                    * current_joint_precision
                    * current_joint_recall
                    / (current_joint_precision + current_joint_recall)
                )
                current_joint = (
                    answer[0] * current_support["em"],
                    current_joint_f1,
                    current_joint_precision,
                    current_joint_recall,
                )
                expected = current_joint[
                    ("joint_em", "joint_f1", "joint_precision", "joint_recall").index(field)
                ]
                if not close(row[field], expected):
                    score_mismatches.append(
                        {
                            "example_id": row["example_id"],
                            "field": field,
                            "persisted": row[field],
                            "recomputed": expected,
                        }
                    )

        audited_rows.append(
            {
                **row,
                "current_normalized_prediction": normalized_answer(domain, prediction),
                "current_normalized_gold": normalized_answer(domain, gold),
                "diagnostic_answer_em": answer_diag[0],
                "diagnostic_answer_f1": answer_diag[1],
                "diagnostic_answer_precision": answer_diag[2],
                "diagnostic_answer_recall": answer_diag[3],
                "diagnostic_support_em": None if support_diag is None else support_diag["em"],
                "diagnostic_support_f1": None if support_diag is None else support_diag["f1"],
                "diagnostic_support_precision": None
                if support_diag is None
                else support_diag["prec"],
                "diagnostic_support_recall": None
                if support_diag is None
                else support_diag["recall"],
                "diagnostic_joint_em": joint_diag[0],
                "diagnostic_joint_f1": joint_diag[1],
                "diagnostic_joint_precision": joint_diag[2],
                "diagnostic_joint_recall": joint_diag[3],
                "support_failure_reason": failure_reason,
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in audited_rows:
        grouped[(row["dataset"], row["method"], row["setting"])].append(row)

    metric_checks = []
    current_vs_diagnostic: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    metric_fields = ("answer_em", "answer_f1", "support_em", "support_f1", "joint_em", "joint_f1")
    diagnostic_fields = tuple("diagnostic_" + field for field in metric_fields)
    for (dataset, method, setting), group in sorted(grouped.items()):
        current = aggregate(group, metric_fields)
        diagnostic = aggregate(group, diagnostic_fields)
        persisted = persisted_metrics["by_dataset"][dataset][method][setting]
        for field in metric_fields:
            metric_checks.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "setting": setting,
                    "field": field,
                    "persisted": persisted[field],
                    "recomputed": current[field],
                    "matches": close(persisted[field], current[field]),
                }
            )
        current_vs_diagnostic[dataset][method][setting] = {
            "current": current,
            "diagnostic": {field: diagnostic["diagnostic_" + field] for field in metric_fields},
            "delta": {
                field: None
                if current[field] is None
                else diagnostic["diagnostic_" + field] - current[field]
                for field in metric_fields
            },
        }

    report = {
        "scope": "DIAGNOSTIC ONLY; frozen predictions, retrieval, and metrics were read-only",
        "integrity": integrity,
        "populations": {"retrieval_examples": len(retrieval_rows), "evaluation_rows": len(rows)},
        "reproduction": {
            "per_example_score_mismatches": len(score_mismatches),
            "aggregate_metric_mismatches": sum(not item["matches"] for item in metric_checks),
        },
        "answer_diagnostics": {
            "boolean_normalization_changed_rows": len(boolean_changed),
            "boolean_normalization_changed_unique_examples": len(
                {item["example_id"] for item in boolean_changed}
            ),
        },
        "support_diagnostics": {
            "canonicalization_changed_rows": len(support_canonical_changed),
            "canonicalization_changed_unique_examples": len(
                {item["example_id"] for item in support_canonical_changed}
            ),
            "duplicate_rows": len(duplicate_rows),
            "multi_gold_best_f1_tie_with_different_precision_recall_rows": len(tie_rows),
            "failure_counts_by_dataset_method_setting": {
                f"{dataset}|{method}|{setting}": dict(counts)
                for (dataset, method, setting), counts in sorted(failure_counts.items())
            },
        },
        "metric_reproduction_checks": metric_checks,
        "current_vs_diagnostic": current_vs_diagnostic,
    }

    write_json(args.output_dir / "audit_summary.json", report)
    write_jsonl(args.output_dir / "boolean_normalization_affected.jsonl", boolean_changed)
    write_jsonl(
        args.output_dir / "support_canonicalization_affected.jsonl", support_canonical_changed
    )
    write_jsonl(args.output_dir / "duplicate_support_rows.jsonl", duplicate_rows)
    write_jsonl(args.output_dir / "multi_gold_tie_rows.jsonl", tie_rows)
    write_jsonl(args.output_dir / "score_reproduction_mismatches.jsonl", score_mismatches)

    lines = [
        "# Final TEST evaluation audit (DIAGNOSTIC ONLY)",
        "",
        "Frozen predictions, retrieval, and official metrics were read-only.",
        "",
        f"- Frozen hashes match: {integrity['hashes_match']}",
        f"- Per-example score mismatches: {len(score_mismatches)}",
        f"- Aggregate metric mismatches: {sum(not item['matches'] for item in metric_checks)}",
        f"- Boolean normalization changed rows: {len(boolean_changed)} ({len({item['example_id'] for item in boolean_changed})} unique examples)",
        f"- Native/canonical support identity changed rows: {len(support_canonical_changed)}",
        f"- Duplicate support rows: {len(duplicate_rows)}",
        f"- Multi-gold F1 tie rows with unequal precision/recall: {len(tie_rows)}",
        "",
        "## Support EM failure counts",
        "",
        "| Dataset | Method | Setting | Exact | Complete + extra | Missing only | Partial | Missing all |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, counts in sorted(failure_counts.items()):
        dataset, method, setting = key
        lines.append(
            f"| {dataset} | {method} | {setting} | {counts['exact']} | {counts['complete_but_nonminimal']} | "
            f"{counts['missing_gold_evidence_only']} | {counts['genuine_partial_support']} | {counts['missing_all_gold_evidence']} |"
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "hashes_match": integrity["hashes_match"],
                "score_mismatches": len(score_mismatches),
                "boolean_changed": len(boolean_changed),
                "support_canonical_changed": len(support_canonical_changed),
                "duplicate_rows": len(duplicate_rows),
                "tie_rows": len(tie_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
