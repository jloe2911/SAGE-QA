"""Finalize frozen TEST end-to-end manuscript results without generation.

This script consumes three already-frozen metric bundles, validates the new
prediction freeze and cohort denominators, recomputes all aggregate values from
dataset-level metrics, and writes presentation artifacts.  It never imports or
calls an OpenAI client.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/final_results/final_manuscript_test_end_to_end"
NEW_METRICS = OUTPUT / "metrics.json"
NEW_PER_EXAMPLE = OUTPUT / "per_example_end_to_end.jsonl"
OLD_METRICS = ROOT / "outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end/metrics.json"
BASELINE_METRICS = ROOT / "outputs/final_results/final_manuscript_baselines_test_end_to_end/metrics.json"
PREDICTIONS = OUTPUT / "predictions.jsonl"
READER_INPUTS = OUTPUT / "frozen_reader_inputs.jsonl"
GENERATION_FREEZE = OUTPUT / "generation_freeze.json"
PREFLIGHT = OUTPUT / "preflight_report.json"

PREDICTIONS_SHA256 = "03e391ed697fb25a07439ec7664544cd6582d2a2dffc868bb9f972c1ae9236d3"
READER_INPUTS_SHA256 = "07624fa96c4d5f91303a75a5f2c3492b2edfa46d4ee1edc9d6b7d46be9d529a2"
METRICS = ("answer_em", "answer_f1", "support_em", "support_f1", "joint_em", "joint_f1")
DATASETS = (
    "HotpotQA", "2WikiMultiHopQA", "FamilyOWL_1hop", "FamilyOWL_2hop",
    "pizza_100_1hop", "pizza_100_2hop", "pizza_250_1hop", "pizza_250_2hop",
    "OWL2Bench_1hop", "OWL2Bench_2hop",
)
CONDITIONS = (
    ("lexical_subgraph/k1", "Lexical Subgraph k=1", "baseline", False),
    ("gnn_rag/k1", "GNN-RAG k=1", "baseline", False),
    ("gnn/k1", "GNN k=1", "old", False),
    ("cross_encoder/k1", "Cross-Encoder k=1", "new", False),
    ("final_sageqa/k1", "Final SAGE-QA k=1", "new", False),
    ("final_sageqa/adaptive", "Final SAGE-QA adaptive", "new", False),
    ("full_context/full", "Full Context", "baseline", True),
    ("gold_support/oracle", "Gold Support", "new", True),
    ("old_gnn_sageqa/k1", "Old GNN-based SAGE-QA k=1", "old", False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_row(bundle: Mapping[str, Any], dataset: str, key: str, source: str) -> Mapping[str, Any]:
    method, setting = key.split("/", 1)
    if source == "new":
        return bundle["by_dataset"][dataset][method][setting]
    if source == "baseline":
        return bundle["by_dataset"][dataset][method]
    mapped = "gnn_only" if method == "gnn" else "sageqa_final"
    return bundle["by_dataset"][dataset][mapped][setting]


def pooled(dataset_rows: list[Mapping[str, Any]], answer_only: bool) -> dict[str, Any]:
    answer_n = sum(int(row["answer_examples"]) for row in dataset_rows)
    result: dict[str, Any] = {
        "answer_examples": answer_n,
        "answer_em": sum(float(row["answer_em"]) * int(row["answer_examples"]) for row in dataset_rows) / answer_n,
        "answer_f1": sum(float(row["answer_f1"]) * int(row["answer_examples"]) for row in dataset_rows) / answer_n,
    }
    if not answer_only:
        support_n = sum(int(row["support_examples"]) for row in dataset_rows)
        result["support_examples"] = support_n
        for metric in METRICS[2:]:
            result[metric] = sum(float(row[metric]) * int(row["support_examples"]) for row in dataset_rows) / support_n
    return result


def validate_freeze_and_new_rows() -> dict[str, Any]:
    if sha256(PREDICTIONS) != PREDICTIONS_SHA256 or sha256(READER_INPUTS) != READER_INPUTS_SHA256:
        raise ValueError("Frozen prediction or reader-input SHA-256 mismatch")
    freeze = load_json(GENERATION_FREEZE)
    if freeze.get("status") != "complete_frozen" or freeze.get("prediction_count") != 16256:
        raise ValueError("Generation freeze is not complete_frozen with 16,256 predictions")
    if freeze.get("gold_answers_opened") is not False:
        raise ValueError("Generation freeze does not attest gold_answers_opened=false")
    predictions = load_jsonl(PREDICTIONS)
    counts = Counter((str(row["method"]), str(row["setting"])) for row in predictions)
    expected = {
        ("cross_encoder", "k1"): 4249,
        ("final_sageqa", "k1"): 4249,
        ("final_sageqa", "adaptive"): 4249,
        ("gold_support", "oracle"): 3509,
    }
    if len(predictions) != 16256 or counts != expected:
        raise ValueError(f"Unexpected prediction counts: {counts}")
    scored = load_jsonl(NEW_PER_EXAMPLE)
    scored_counts = Counter((str(row["method"]), str(row["setting"])) for row in scored)
    if scored_counts != expected:
        raise ValueError(f"Unexpected scored counts: {scored_counts}")
    for method, setting in (("cross_encoder", "k1"), ("final_sageqa", "k1"), ("final_sageqa", "adaptive")):
        rows = [row for row in scored if (row["method"], row["setting"]) == (method, setting)]
        eligible = [row for row in rows if row["support_evaluable"]]
        ineligible = [row for row in rows if not row["support_evaluable"]]
        if len(eligible) != 3509 or len(ineligible) != 740:
            raise ValueError(f"Wrong support cohort for {method}/{setting}")
        if any("support_f1" in row or "joint_f1" in row for row in ineligible):
            raise ValueError(f"Support-ineligible rows were scored for {method}/{setting}")
    gold_rows = [row for row in scored if row["method"] == "gold_support"]
    if len(gold_rows) != 3509 or any(any(metric in row for metric in METRICS[2:]) for row in gold_rows):
        raise ValueError("Gold Support must contain 3,509 answer-only rows")
    preflight = load_json(PREFLIGHT)
    if preflight.get("openai_calls_made") != 0:
        raise ValueError("Preflight does not attest zero OpenAI calls")
    return {
        "prediction_count": len(predictions),
        "prediction_counts_by_condition": {f"{m}/{s}": n for (m, s), n in sorted(counts.items())},
        "answer_denominator": 4249,
        "support_joint_denominator": 3509,
        "excluded_undefined_or_empty_support": 740,
        "gold_support_answer_denominator": 3509,
        "support_ineligible_rows_with_support_metrics": 0,
        "openai_calls_during_preflight": 0,
        "openai_calls_during_evaluation_and_finalization": 0,
    }


def format_cell(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.2f}"


def latex_table(rows: list[dict[str, Any]], include: set[str], caption: str) -> str:
    lines = [
        "% Values are percentages; equal-dataset macro over the ten TEST datasets.",
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Method & Ans. EM & Ans. F1 & Supp. EM & Supp. F1 & Joint EM & Joint F1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["condition"] not in include:
            continue
        macro = row["equal_dataset_macro"]
        values = " & ".join(format_cell(macro.get(metric)) for metric in METRICS)
        label = row["label"].replace("_", "\\_")
        lines.append(f"{label} & {values} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", "\\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    audit = validate_freeze_and_new_rows()
    bundles = {"new": load_json(NEW_METRICS), "old": load_json(OLD_METRICS), "baseline": load_json(BASELINE_METRICS)}
    conditions: list[dict[str, Any]] = []
    per_dataset_rows: list[dict[str, Any]] = []
    for key, label, source, answer_only in CONDITIONS:
        rows = [dict(source_row(bundles[source], dataset, key, source)) for dataset in DATASETS]
        answer_n = sum(int(row["answer_examples"]) for row in rows)
        expected_answer = 3509 if key == "gold_support/oracle" else 4249
        if answer_n != expected_answer:
            raise ValueError(f"Wrong answer denominator for {key}: {answer_n}")
        if not answer_only and sum(int(row["support_examples"]) for row in rows) != 3509:
            raise ValueError(f"Wrong support denominator for {key}")
        fields = METRICS[:2] if answer_only else METRICS
        macro = {metric: fmean(float(row[metric]) for row in rows) for metric in fields}
        secondary = pooled(rows, answer_only)
        condition = {
            "condition": key, "label": label, "source_bundle": source,
            "reporting_scope": "answer_only" if answer_only else "answer_support_joint",
            "answer_examples": answer_n,
            "support_joint_examples": None if answer_only else 3509,
            "equal_dataset_macro": macro,
            "pooled_per_example": secondary,
        }
        conditions.append(condition)
        for dataset, row in zip(DATASETS, rows):
            per_dataset_rows.append({
                "dataset": dataset, "condition": key, "label": label,
                "answer_examples": row["answer_examples"],
                "support_joint_examples": row.get("support_examples"),
                **{metric: row.get(metric) for metric in METRICS},
            })

    # Cross-check the new evaluator's stored macros against this independent recomputation.
    new_metrics = bundles["new"]
    for condition in conditions:
        key = condition["condition"]
        if key not in new_metrics.get("primary_equal_dataset_macro", {}):
            continue
        for metric, value in condition["equal_dataset_macro"].items():
            stored = float(new_metrics["primary_equal_dataset_macro"][key][metric])
            if abs(stored - float(value)) > 1e-15:
                raise ValueError(f"Independent macro mismatch: {key}/{metric}")

    canonical = {
        "schema_version": "final_manuscript_canonical_end_to_end_v1",
        "status": "complete_frozen_evaluation_only",
        "primary_aggregation": "equal_dataset_macro_unweighted_over_10_datasets",
        "secondary_aggregation": "pooled_per_example",
        "cohorts": {
            "answer": {"examples": 4249},
            "support_and_joint": {"examples": 3509, "excludes_undefined_or_empty_gold_support": 740},
            "gold_support_answer_only": {"examples": 3509},
        },
        "conditions": conditions,
        "verification": {
            **audit,
            "predictions_sha256": sha256(PREDICTIONS),
            "reader_inputs_sha256": sha256(READER_INPUTS),
            "generation_status": "complete_frozen",
            "gold_answers_opened_during_generation": False,
            "independent_equal_dataset_macro_mismatches": 0,
        },
    }
    write_json(OUTPUT / "canonical_metrics.json", canonical)

    with (OUTPUT / "per_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ("dataset", "condition", "label", "answer_examples", "support_joint_examples", *METRICS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_dataset_rows)

    main_keys = {key for key, _, _, _ in CONDITIONS if key != "old_gnn_sageqa/k1"}
    ablation_keys = {"gnn/k1", "old_gnn_sageqa/k1", "cross_encoder/k1", "final_sageqa/k1", "final_sageqa/adaptive"}
    (OUTPUT / "main_table.tex").write_text(latex_table(
        conditions, main_keys, "Frozen TEST end-to-end results. Equal-dataset macro is primary; Gold Support and Full Context are answer-only."
    ), encoding="utf-8")
    (OUTPUT / "ablation_table.tex").write_text(latex_table(
        conditions, ablation_keys, "Frozen TEST ranking and symbolic ablation. The old SAGE-QA row uses the retained GNN-based ranking."
    ), encoding="utf-8")

    protocol = """# Frozen manuscript end-to-end evaluation protocol

Evaluation begins only after validating the 16,256-row prediction freeze and its SHA-256. No answer generation, retrieval, reranking, adaptive-k recomputation, parameter tuning, or API call occurs in this phase. Gold answers and gold support are joined only after the prediction population is frozen.

Answer EM/F1 use all 4,249 TEST examples for retrieval-based systems and Full Context. Support and Joint EM/F1 use exactly the 3,509 examples with defined, non-empty gold support. The other 740 examples are excluded from Support and Joint and receive no Support or Joint score. Gold Support is evaluated for Answer EM/F1 only on those 3,509 examples; Support and Joint are intentionally not reported.

Per-dataset values are arithmetic means over eligible examples in each dataset. The primary aggregate is the unweighted arithmetic mean of the ten dataset-level values. Pooled per-example values are secondary diagnostics. Support EM is exact-set match; Support F1 uses the best matching persisted gold-support alternative. Joint EM is Answer EM times Support EM, and Joint F1 is formed from the products of answer/support precision and recall under the audited evaluator semantics.

Lexical Subgraph k=1, GNN-RAG k=1, GNN k=1, the retained old GNN-based SAGE-QA k=1 ablation, and Full Context are reused from their frozen evaluated bundles without regeneration. Full Context is answer-only.
"""
    (OUTPUT / "evaluation_protocol.md").write_text(protocol, encoding="utf-8")

    lookup = {row["condition"]: row for row in conditions}
    summary_lines = [
        "# Final manuscript TEST end-to-end results", "",
        "Status: **complete frozen evaluation-only**. Primary values below are equal-dataset macros over ten datasets, shown as percentages.", "",
        "| Method | Answer EM | Answer F1 | Support EM | Support F1 | Joint EM | Joint F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        if row["condition"] == "old_gnn_sageqa/k1":
            continue
        values = [format_cell(row["equal_dataset_macro"].get(metric)) for metric in METRICS]
        summary_lines.append(f"| {row['label']} | " + " | ".join(values) + " |")
    summary_lines.extend([
        "", "The old GNN-based SAGE-QA k=1 result is retained in `ablation_table.tex` and `canonical_metrics.json` only.", "",
        "Verification: prediction SHA-256 and reader-input SHA-256 matched; condition counts were 4,249 / 4,249 / 4,249 / 3,509; Support and Joint denominators were 3,509; 740 support-undefined/empty examples had no Support or Joint metric; independent equal-dataset macro mismatches were zero; evaluation/finalization made zero API calls.", "",
    ])
    (OUTPUT / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    output_names = ("canonical_metrics.json", "per_dataset.csv", "main_table.tex", "ablation_table.tex", "evaluation_protocol.md", "SUMMARY.md", "metrics.json", "per_example_end_to_end.jsonl")
    manifest = {
        "schema_version": "final_manuscript_end_to_end_artifact_manifest_v1",
        "status": "complete_frozen_evaluation_only",
        "frozen_inputs": {
            "predictions.jsonl": sha256(PREDICTIONS),
            "frozen_reader_inputs.jsonl": sha256(READER_INPUTS),
            "generation_freeze.json": sha256(GENERATION_FREEZE),
            "preflight_report.json": sha256(PREFLIGHT),
            str(OLD_METRICS.relative_to(ROOT)): sha256(OLD_METRICS),
            str(BASELINE_METRICS.relative_to(ROOT)): sha256(BASELINE_METRICS),
        },
        "derived_files": {name: sha256(OUTPUT / name) for name in output_names},
        "verification": canonical["verification"],
    }
    write_json(OUTPUT / "artifact_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "conditions": len(conditions), "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
