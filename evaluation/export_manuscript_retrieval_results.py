"""Export canonical manuscript TEST retrieval results from frozen artifacts only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.prepare_familyowl_gnn_rag import unit_node
from data_processing.prepare_text_gnn_rag import evidence_node
from utils.paths import (
    manuscript_retrieval_root,
    outputs_root,
    repo_display_path,
    repo_path_arg,
    repo_root,
)


ROOT = repo_root()
DEFAULT_OUTPUT = manuscript_retrieval_root()
# The previous canonical export remains in manuscript_retrieval_results.  This
# version deliberately aligns the GNN and GNN-based SAGE rows with the frozen
# hard-pair-v2 end-to-end lineage.
OLD_ROOT = outputs_root() / "final_results/production_generator_d_v2_hard_pair_test_retrieval"
LEXICAL_ROOT = outputs_root() / "final_results/production_generator_d_v1_test_baselines/lexical_subgraph"
GNN_RAG_ROOT = outputs_root() / "final_results/production_generator_d_v1_test_baselines/gnn_rag"
CE_ROOT = outputs_root() / "final_results/question_candidate_cross_encoder_v1_adaptive_test_a40"
DIAGNOSTIC_ROOT = outputs_root() / "development_diagnostics/final_old_vs_cross_encoder_analysis"

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
DISPLAY = {
    "HotpotQA": "HotpotQA",
    "2WikiMultiHopQA": "2WikiMultiHopQA",
    "FamilyOWL_1hop": "FamilyOWL 1-hop",
    "FamilyOWL_2hop": "FamilyOWL 2-hop",
    "pizza_100_1hop": "Pizza-100 1-hop",
    "pizza_100_2hop": "Pizza-100 2-hop",
    "pizza_250_1hop": "Pizza-250 1-hop",
    "pizza_250_2hop": "Pizza-250 2-hop",
    "OWL2Bench_1hop": "OWL2Bench 1-hop",
    "OWL2Bench_2hop": "OWL2Bench 2-hop",
}
METHOD_LABELS = {
    "lexical": "Lexical Subgraph",
    "gnn_rag": "GNN-RAG top-1 EvidenceUnit",
    "old_gnn": "GNN (hard-pair v2)",
    "old_sage": "GNN-based SAGE (hard-pair v2)",
    "cross_encoder": "Cross-Encoder",
    "final_sageqa": "Final SAGE-QA",
}
SETTINGS = {
    "lexical": ("k1",),
    "gnn_rag": ("k1",),
    "old_gnn": ("k1", "adaptive"),
    "old_sage": ("k1", "adaptive"),
    "cross_encoder": ("k1", "adaptive"),
    "final_sageqa": ("k1", "adaptive"),
}
SOURCE_PATHS = {
    "old_per_example": OLD_ROOT / "per_example_test_retrieval.jsonl",
    "old_manifest": OLD_ROOT / "artifact_manifest.json",
    "lexical_per_example": LEXICAL_ROOT / "per_example_retrieval.jsonl",
    "lexical_manifest": LEXICAL_ROOT / "artifact_manifest.json",
    "gnn_rag_predictions": GNN_RAG_ROOT / "predictions_frozen.jsonl",
    "gnn_rag_freeze": GNN_RAG_ROOT / "prediction_freeze_manifest.json",
    "cross_encoder_per_example": CE_ROOT / "per_example_test_retrieval.jsonl",
    "cross_encoder_manifest": CE_ROOT / "artifact_manifest.json",
    "prior_diagnostic": DIAGNOSTIC_ROOT / "comparison.json",
}
EXCLUSION_REASON = (
    "ground-truth support is undefined or empty, so retrieval precision, recall, "
    "and F1 are undefined"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_manifest(manifest_path: Path, required: Sequence[str]) -> dict[str, str]:
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Invalid manifest: {manifest_path}")
    hashes: dict[str, str] = {}
    for name in required:
        entry = files.get(name)
        if entry is None:
            raise ValueError(f"Manifest does not lock {name}: {manifest_path}")
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        target = manifest_path.parent / name
        actual = sha256(target)
        if actual != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {target}")
        if isinstance(entry, dict) and target.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Frozen artifact size mismatch: {target}")
        hashes[repo_display_path(target)] = actual
    hashes[repo_display_path(manifest_path)] = sha256(manifest_path)
    return hashes


def score(predicted: Sequence[Any], golds: Sequence[Sequence[Any]]) -> dict[str, float]:
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    predicted_set = set(map(str, predicted))
    for gold in golds:
        gold_set = set(map(str, gold))
        overlap = len(predicted_set & gold_set)
        precision = overlap / len(predicted_set) if predicted_set else 0.0
        recall = overlap / len(gold_set) if gold_set else 0.0
        f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
        current = {"precision": precision, "recall": recall, "f1": f1}
        if current["f1"] > best["f1"]:
            best = current
    return best


def mean_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in ("precision", "recall", "f1")
    }
    if all("selected_k" in row for row in rows):
        selected = [int(row["selected_k"]) for row in rows]
        result["mean_selected_k"] = statistics.fmean(selected)
        result["selected_k_distribution"] = dict(sorted(Counter(selected).items()))
    return result


def aggregate(scored: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_dataset: list[dict[str, Any]] = []
    overall: dict[str, Any] = {}
    for method, settings in SETTINGS.items():
        overall[method] = {}
        for setting in settings:
            selected = [r for r in scored if r["method"] == method and r["setting"] == setting]
            dataset_values = []
            for dataset in DATASET_ORDER:
                rows = [r for r in selected if r["dataset"] == dataset]
                values = mean_metrics(rows)
                dataset_values.append(values)
                per_dataset.append(
                    {
                        "dataset": dataset,
                        "display_name": DISPLAY[dataset],
                        "domain": rows[0]["domain"],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "setting": setting,
                        "examples": len(rows),
                        **values,
                    }
                )
            macro = {
                metric: statistics.fmean(row[metric] for row in dataset_values)
                for metric in ("precision", "recall", "f1")
            }
            if all("mean_selected_k" in row for row in dataset_values):
                macro["mean_selected_k"] = statistics.fmean(
                    row["mean_selected_k"] for row in dataset_values
                )
            overall[method][setting] = {
                "equal_dataset_macro": macro,
                "pooled_per_example_mean": mean_metrics(selected),
            }
    return per_dataset, overall


def tex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def metric_text(values: Mapping[str, Any]) -> str:
    return " & ".join(f"{float(values[key]):.4f}" for key in ("precision", "recall", "f1"))


def main_table(per_dataset: Sequence[Mapping[str, Any]], overall: Mapping[str, Any]) -> str:
    methods = ("lexical", "gnn_rag", "old_gnn", "cross_encoder", "final_sageqa")
    lines = [
        "% Canonical support-bearing TEST cohort; equal-dataset macro is the primary aggregate.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Dataset & Method & P & R & F1 \\\\",
        "\\midrule",
    ]
    for dataset in (*DATASET_ORDER, "__macro__"):
        for method in methods:
            if dataset == "__macro__":
                values = overall[method]["k1"]["equal_dataset_macro"]
                dataset_name = "Equal-dataset macro"
            else:
                values = next(
                    r
                    for r in per_dataset
                    if r["dataset"] == dataset and r["method"] == method and r["setting"] == "k1"
                )
                dataset_name = DISPLAY[dataset]
            lines.append(
                f"{tex_escape(dataset_name)} & {tex_escape(METHOD_LABELS[method])} & "
                f"{metric_text(values)} \\\\"
            )
        if dataset != "__macro__":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def ablation_table(per_dataset: Sequence[Mapping[str, Any]], overall: Mapping[str, Any]) -> str:
    methods = ("old_gnn", "old_sage", "cross_encoder", "final_sageqa")
    lines = [
        "% Delta F1 is adaptive minus k=1 on the same support-bearing cohort.",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Dataset & Method & Setting & P & R & F1 & $\\Delta$F1 & Mean $k$ \\\\",
        "\\midrule",
    ]
    for dataset in (*DATASET_ORDER, "__macro__"):
        for method in methods:
            if dataset == "__macro__":
                k1 = overall[method]["k1"]["equal_dataset_macro"]
                adaptive = overall[method]["adaptive"]["equal_dataset_macro"]
                dataset_name = "Equal-dataset macro"
            else:
                k1 = next(r for r in per_dataset if r["dataset"] == dataset and r["method"] == method and r["setting"] == "k1")
                adaptive = next(r for r in per_dataset if r["dataset"] == dataset and r["method"] == method and r["setting"] == "adaptive")
                dataset_name = DISPLAY[dataset]
            delta = float(adaptive["f1"]) - float(k1["f1"])
            for setting, values in (("k=1", k1), ("adaptive", adaptive)):
                delta_text = f"{delta:+.4f}" if setting == "adaptive" else "--"
                mean_k = float(values.get("mean_selected_k", 1.0))
                lines.append(
                    f"{tex_escape(dataset_name)} & {tex_escape(METHOD_LABELS[method])} & {setting} & "
                    f"{metric_text(values)} & {delta_text} & {mean_k:.3f} \\\\"
                )
        if dataset != "__macro__":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def export(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    source_hashes: dict[str, str] = {}
    source_hashes.update(validate_manifest(SOURCE_PATHS["old_manifest"], ["per_example_test_retrieval.jsonl", "metrics.json"]))
    source_hashes.update(validate_manifest(SOURCE_PATHS["lexical_manifest"], ["per_example_retrieval.jsonl", "metrics.json"]))
    source_hashes.update(validate_manifest(SOURCE_PATHS["cross_encoder_manifest"], ["per_example_test_retrieval.jsonl", "metrics.json", "test_predictions_frozen.jsonl", "generation_freeze.json"]))
    gnn_freeze = read_json(SOURCE_PATHS["gnn_rag_freeze"])
    if gnn_freeze.get("status") != "predictions_frozen_gold_unopened":
        raise ValueError("GNN-RAG prediction freeze is not gold-unopened")
    gnn_hash = sha256(SOURCE_PATHS["gnn_rag_predictions"])
    if gnn_hash != gnn_freeze.get("prediction_freeze_sha256"):
        raise ValueError("GNN-RAG frozen prediction hash mismatch")
    source_hashes[repo_display_path(SOURCE_PATHS["gnn_rag_predictions"])] = gnn_hash
    source_hashes[repo_display_path(SOURCE_PATHS["gnn_rag_freeze"])] = sha256(SOURCE_PATHS["gnn_rag_freeze"])
    source_hashes[repo_display_path(SOURCE_PATHS["prior_diagnostic"])] = sha256(SOURCE_PATHS["prior_diagnostic"])

    old_rows = read_jsonl(SOURCE_PATHS["old_per_example"])
    lexical_rows = read_jsonl(SOURCE_PATHS["lexical_per_example"])
    gnn_rows = read_jsonl(SOURCE_PATHS["gnn_rag_predictions"])
    ce_rows = read_jsonl(SOURCE_PATHS["cross_encoder_per_example"])
    old_by_id = {str(row["example_id"]): row for row in old_rows}
    lexical_by_id = {str(row["example_id"]): row for row in lexical_rows}
    gnn_by_id = {str(row["example_id"]): row for row in gnn_rows}
    population = set(old_by_id)
    if len(population) != 4249 or set(lexical_by_id) != population or set(gnn_by_id) != population:
        raise ValueError("Frozen TEST populations are not the expected matched 4,249 IDs")
    eligible = {example_id for example_id, row in old_by_id.items() if row.get("evaluation_eligible") is True}
    excluded = population - eligible
    if len(eligible) != 3509 or len(excluded) != 740:
        raise ValueError(f"Unexpected retrieval cohort: included={len(eligible)}, excluded={len(excluded)}")
    for example_id in excluded:
        if old_by_id[example_id].get("gold_explanations"):
            raise ValueError(f"Excluded example has non-empty support: {example_id}")
    for example_id in eligible:
        if not old_by_id[example_id].get("gold_explanations"):
            raise ValueError(f"Eligible example lacks support: {example_id}")

    scored: list[dict[str, Any]] = []
    for example_id in sorted(eligible):
        old = old_by_id[example_id]
        dataset, domain = str(old["dataset"]), str(old["domain"])
        base = {"dataset": dataset, "domain": domain, "example_id": example_id}
        lexical = lexical_by_id[example_id]
        if lexical.get("evaluation_eligible") is not True:
            raise ValueError(f"Lexical eligibility mismatch: {example_id}")
        lexical_values = lexical["settings"]["k1"]
        scored.append({**base, "method": "lexical", "setting": "k1", "selected_k": 1, **{k: float(lexical_values[k]) for k in ("precision", "recall", "f1")}})

        gnn = gnn_by_id[example_id]
        ranked = gnn.get("native_ranked_entities", []) or []
        mapped = gnn.get("retrieved_evidence_units", []) or []
        node_fn = evidence_node if domain == "text" else unit_node
        if not ranked or not mapped or int(ranked[0]["rank"]) != 1 or str(ranked[0]["entity"]) != node_fn(str(mapped[0])):
            raise ValueError(f"GNN-RAG rank-1 mapping invariant failed: {example_id}")
        scored.append({**base, "method": "gnn_rag", "setting": "k1", "selected_k": 1, **score([mapped[0]], old["gold_explanations"])})

        for source_method, method in (("gnn_only", "old_gnn"), ("sageqa_final", "old_sage")):
            for setting in ("k1", "adaptive"):
                values = old["methods"][source_method][setting]
                scored.append({**base, "method": method, "setting": setting, "selected_k": int(values["selected_k"]), **{k: float(values[k]) for k in ("precision", "recall", "f1")}})

    ce_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in ce_rows:
        key = (str(row["example_id"]), str(row["method"]), str(row["setting"]))
        if key in ce_index:
            raise ValueError(f"Duplicate Cross-Encoder score row: {key}")
        ce_index[key] = row
    for example_id in sorted(eligible):
        old = old_by_id[example_id]
        base = {"dataset": str(old["dataset"]), "domain": str(old["domain"]), "example_id": example_id}
        for source_method, method in (("cross_encoder", "cross_encoder"), ("final_sageqa", "final_sageqa")):
            for setting in ("k1", "adaptive"):
                row = ce_index[(example_id, source_method, setting)]
                scored.append({**base, "method": method, "setting": setting, "selected_k": int(row["selected_k"]), **{k: float(row[k]) for k in ("precision", "recall", "f1")}})

    per_dataset, overall = aggregate(scored)
    prior = read_json(SOURCE_PATHS["prior_diagnostic"])
    expected = prior["headline"]["test_k1_equal_dataset_macro"]
    # The prior diagnostic intentionally locks the historical v1 GNN lineage,
    # so only lineage-independent rows may be compared with it here.
    aliases = {"cross_encoder": "cross_encoder", "cross_encoder_symbolic": "final_sageqa"}
    for prior_name, method in aliases.items():
        actual = overall[method]["k1"]["equal_dataset_macro"]["f1"]
        if abs(actual - float(expected[prior_name])) > 1e-12:
            raise ValueError(f"Independent recomputation disagrees with prior diagnostic: {method}")

    cohort_rows = []
    for dataset in DATASET_ORDER:
        included_count = sum(old_by_id[x]["dataset"] == dataset for x in eligible)
        excluded_count = sum(old_by_id[x]["dataset"] == dataset for x in excluded)
        domain = next(str(r["domain"]) for r in old_rows if r["dataset"] == dataset)
        cohort_rows.append({"dataset": dataset, "display_name": DISPLAY[dataset], "domain": domain, "included_support_bearing": included_count, "excluded_no_gold_support": excluded_count, "total_predictions": included_count + excluded_count, "exclusion_reason": EXCLUSION_REASON})

    comparisons = {
        "cross_encoder_minus_old_gnn": {},
        "final_sageqa_minus_old_gnn_based_sage": {},
        "final_sageqa_minus_cross_encoder_symbolic_contribution": {},
        "adaptive_minus_k1": {},
    }
    for aggregation in ("equal_dataset_macro", "pooled_per_example_mean"):
        value = lambda method, setting="k1": overall[method][setting][aggregation]["f1"]
        comparisons["cross_encoder_minus_old_gnn"][aggregation] = value("cross_encoder") - value("old_gnn")
        comparisons["final_sageqa_minus_old_gnn_based_sage"][aggregation] = value("final_sageqa") - value("old_sage")
        comparisons["final_sageqa_minus_cross_encoder_symbolic_contribution"][aggregation] = value("final_sageqa") - value("cross_encoder")
        comparisons["adaptive_minus_k1"][aggregation] = {
            method: value(method, "adaptive") - value(method, "k1")
            for method in ("old_gnn", "old_sage", "cross_encoder", "final_sageqa")
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = {
        "schema_version": "canonical_manuscript_retrieval_results_hard_pair_v2_v1",
        "status": "complete_frozen_artifact_only_export",
        "split": "test",
        "cohort": {"definition": "examples with defined, non-empty ground-truth support", "included": len(eligible), "excluded": len(excluded), "prediction_population": len(population), "exclusion_reason": EXCLUSION_REASON},
        "aggregation": {"primary": "equal_dataset_macro: unweighted mean of ten dataset-level per-example means", "secondary": "pooled_per_example_mean: unweighted mean over all 3,509 eligible examples"},
        "methods": overall,
        "comparisons": comparisons,
        "diagnostic_preserved": {"path": repo_display_path(CE_ROOT / "metrics.json"), "examples": 4249, "manuscript_comparison": False},
        "gnn_lineage": "production_generator_d_v2_hard_pair_test_retrieval",
        "historical_canonical_export_preserved": "outputs/final_results/manuscript_retrieval_results",
        "source_sha256": source_hashes,
        "inference_invoked": False,
        "predictions_regenerated": False,
        "test_informed_tuning": False,
    }
    write_json(output_dir / "canonical_metrics.json", canonical)

    with (output_dir / "per_dataset.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "display_name", "domain", "method", "method_label", "setting", "examples", "precision", "recall", "f1", "mean_selected_k"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_dataset)
        for method, settings in SETTINGS.items():
            for setting in settings:
                for aggregation, display_name in (("equal_dataset_macro", "Equal-dataset macro"), ("pooled_per_example_mean", "Pooled per-example mean")):
                    values = overall[method][setting][aggregation]
                    writer.writerow({"dataset": f"__{aggregation}__", "display_name": display_name, "domain": "all", "method": method, "method_label": METHOD_LABELS[method], "setting": setting, "examples": len(DATASET_ORDER) if aggregation == "equal_dataset_macro" else len(eligible), **values})

    with (output_dir / "cohort_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cohort_rows[0]))
        writer.writeheader()
        writer.writerows(cohort_rows)
        writer.writerow({"dataset": "__total__", "display_name": "Total", "domain": "all", "included_support_bearing": len(eligible), "excluded_no_gold_support": len(excluded), "total_predictions": len(population), "exclusion_reason": EXCLUSION_REASON})

    (output_dir / "main_table.tex").write_text(main_table(per_dataset, overall), encoding="utf-8")
    (output_dir / "ablation_table.tex").write_text(ablation_table(per_dataset, overall), encoding="utf-8")
    comparison_lines = [
        "# Canonical matched-TEST retrieval comparisons",
        "",
        "All values use the 3,509-example support-bearing cohort. Equal-dataset macro is primary; pooled per-example mean is secondary.",
        "",
        f"- Cross-Encoder minus old GNN (k=1): **{comparisons['cross_encoder_minus_old_gnn']['equal_dataset_macro']:+.4f}** macro F1; {comparisons['cross_encoder_minus_old_gnn']['pooled_per_example_mean']:+.4f} pooled F1.",
        f"- Final SAGE-QA minus old GNN-based SAGE (k=1): **{comparisons['final_sageqa_minus_old_gnn_based_sage']['equal_dataset_macro']:+.4f}** macro F1; {comparisons['final_sageqa_minus_old_gnn_based_sage']['pooled_per_example_mean']:+.4f} pooled F1.",
        f"- Final SAGE-QA minus Cross-Encoder (symbolic contribution, k=1): **{comparisons['final_sageqa_minus_cross_encoder_symbolic_contribution']['equal_dataset_macro']:+.4f}** macro F1; {comparisons['final_sageqa_minus_cross_encoder_symbolic_contribution']['pooled_per_example_mean']:+.4f} pooled F1.",
        "",
        "## Adaptive minus k=1",
        "",
    ]
    for method in ("old_gnn", "old_sage", "cross_encoder", "final_sageqa"):
        comparison_lines.append(f"- {METHOD_LABELS[method]}: {comparisons['adaptive_minus_k1']['equal_dataset_macro'][method]:+.4f} macro F1; {comparisons['adaptive_minus_k1']['pooled_per_example_mean'][method]:+.4f} pooled F1.")
    (output_dir / "comparison_summary.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")

    protocol = """# Retrieval evaluation protocol

Retrieval precision, recall, and F1 require a defined, non-empty reference support set. The canonical TEST retrieval cohort therefore includes all and only examples for which ground-truth support is defined and non-empty. Examples with no gold support are excluded because there is no reference support set against which retrieval precision or recall can be defined; they are not classified as retrieval failures and this is not a removal of difficult examples.

The matched cohort contains 3,509 of the 4,249 frozen TEST predictions. The other 740 examples remain valid for downstream answer evaluation where the applicable answer protocol permits it. Every retrieval method in the manuscript comparison is evaluated on exactly the same 3,509 support-bearing example IDs.

Dataset-level values are unweighted means over eligible examples within each dataset. The primary equal-dataset macro is the unweighted mean of the ten dataset-level values. The secondary pooled per-example mean is the unweighted mean over all 3,509 eligible examples; these denominators are reported separately.

The prior 4,249-example Cross-Encoder evaluation is preserved as a diagnostic artifact. It assigned zero retrieval scores to the 740 rows with no gold support and is not used for the manuscript retrieval comparison.
"""
    (output_dir / "evaluation_protocol.md").write_text(protocol, encoding="utf-8")

    output_names = ("canonical_metrics.json", "per_dataset.csv", "cohort_audit.csv", "main_table.tex", "ablation_table.tex", "comparison_summary.md", "evaluation_protocol.md")
    manifest = {
        "schema_version": "canonical_manuscript_retrieval_artifact_manifest_hard_pair_v2_v1",
        "status": "complete_frozen_artifact_only_export",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "self_excluded_from_hashes": True,
        "source_files": {name: {"sha256": value} for name, value in sorted(source_hashes.items())},
        "files": {name: {"size_bytes": (output_dir / name).stat().st_size, "sha256": sha256(output_dir / name)} for name in output_names},
        "protected_sources_modified": False,
        "inference_invoked": False,
        "predictions_regenerated": False,
        "gnn_lineage": "production_generator_d_v2_hard_pair_test_retrieval",
        "historical_canonical_export_preserved": "outputs/final_results/manuscript_retrieval_results",
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = export(args.output_dir)
    print(json.dumps({"status": result["status"], "cohort": result["cohort"], "output_dir": str(args.output_dir)}))
