"""Correctly finalize post-freeze support metrics for clean GNN-RAG TEST."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.prepare_production_gnn_rag_clean import (
    evidence_node,
    inference_sample,
    unit_node,
)
from evaluation.run_production_gnn_rag_final_test import (
    DATASETS,
    artifact_manifest,
    best_overlap,
    canonical_hash,
    checked_groups,
    ontology_gold,
    read_json,
    read_jsonl,
    sha256,
    summary_markdown,
    text_gold,
    utc_now,
    write_json,
    write_jsonl,
)


def aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    eligible = [row[key] for row in rows if row["evaluation_eligible"]]
    return {
        "evaluation_examples": len(eligible),
        "precision": statistics.fmean(row["precision"] for row in eligible) if eligible else 0.0,
        "recall": statistics.fmean(row["recall"] for row in eligible) if eligible else 0.0,
        "f1": statistics.fmean(row["f1"] for row in eligible) if eligible else 0.0,
        "exact_match_rate": statistics.fmean(float(row["exact_match"]) for row in eligible)
        if eligible
        else 0.0,
        "complete_gold_recall_rate": statistics.fmean(
            float(row["contains_complete_gold"]) for row in eligible
        )
        if eligible
        else 0.0,
        "any_overlap_rate": statistics.fmean(float(row["recall"] > 0.0) for row in eligible)
        if eligible
        else 0.0,
    }


def candidate_unit_maps(
    root: Path, data_root: Path, dataset: str, domain: str
) -> dict[str, dict[str, str]]:
    node = evidence_node if domain == "text" else unit_node
    result = {}
    for example_id, rows in checked_groups(data_root / dataset / "test_subgraph_retrieval.jsonl"):
        _, candidate_units = inference_sample(rows, domain, 5)
        result[example_id] = {node(unit): unit for unit in candidate_units}
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "production_generator_d_v1"
    output_dir = (
        root / "outputs" / "final_results" / "production_generator_d_v1_test_baselines" / "gnn_rag"
    )
    freeze = read_json(output_dir / "prediction_freeze_manifest.json")
    predictions_path = output_dir / "predictions_frozen.jsonl"
    if freeze.get("status") != "predictions_frozen_gold_unopened":
        raise ValueError("Prediction freeze status is invalid")
    if sha256(predictions_path) != freeze.get("prediction_freeze_sha256"):
        raise ValueError("Prediction freeze hash mismatch")
    records = read_jsonl(predictions_path)
    if len(records) != freeze.get("prediction_examples") or canonical_hash(records) != freeze.get(
        "prediction_payload_canonical_sha256"
    ):
        raise ValueError("Prediction freeze payload mismatch")

    evaluated = []
    metric_rows = []
    gold_lineage = {}
    for dataset, cfg in DATASETS.items():
        dataset_records = [row for row in records if row["dataset"] == dataset]
        ids = {row["example_id"] for row in dataset_records}
        unit_maps = candidate_unit_maps(root, data_root, dataset, cfg["domain"])
        if set(unit_maps) != ids:
            raise ValueError(f"Candidate-map ID mismatch for {dataset}")
        gold_path = root / cfg["gold"]
        golds = (
            text_gold(gold_path, dataset, ids)
            if cfg["domain"] == "text"
            else ontology_gold(gold_path, ids)
        )
        if set(golds) != ids:
            raise ValueError(f"Incomplete post-freeze gold join for {dataset}")
        dataset_eval = []
        for frozen in dataset_records:
            row = dict(frozen)
            references = [gold for gold in golds[row["example_id"]] if gold]
            row["evaluation_eligible"] = bool(references)
            row["evaluation_exclusion_reason"] = (
                None if references else "no_gold_support_or_explanation"
            )
            mapping = unit_maps[row["example_id"]]
            predicted_units = []
            for candidate in row["native_ranked_entities"]:
                unit = mapping.get(candidate["entity"])
                if unit is not None and unit not in predicted_units:
                    predicted_units.append(unit)
            row["native_ranked_entity_units"] = predicted_units
            row["native_ranked_entity_support_overlap"] = (
                best_overlap(predicted_units, references) if references else None
            )
            row["retrieved_path_support_overlap"] = (
                best_overlap(row["retrieved_evidence_units"], references) if references else None
            )
            evaluated.append(row)
            dataset_eval.append(row)
        diagnostics = next(item for item in freeze["datasets"] if item["dataset"] == dataset)
        metric_rows.append(
            {
                "dataset": dataset,
                "inference_examples": diagnostics["examples"],
                "inference_failures": diagnostics["inference_failures"],
                "examples_with_ranked_entities": diagnostics["examples_with_ranked_entities"],
                "examples_with_nonempty_reader_context": diagnostics[
                    "examples_with_nonempty_reader_context"
                ],
                "mean_ranked_entities": diagnostics["mean_ranked_entities"],
                "median_ranked_entities": diagnostics["median_ranked_entities"],
                "mean_retrieved_paths": diagnostics["mean_retrieved_paths"],
                "mean_mapped_evidence_units": diagnostics["mean_mapped_evidence_units"],
                "excluded_no_gold_support_or_explanation": sum(
                    not row["evaluation_eligible"] for row in dataset_eval
                ),
                "native_ranked_entity_support_overlap": aggregate(
                    dataset_eval, "native_ranked_entity_support_overlap"
                ),
                "retrieved_path_support_overlap": aggregate(
                    dataset_eval, "retrieved_path_support_overlap"
                ),
            }
        )
        gold_lineage[dataset] = {
            "path": cfg["gold"],
            "sha256": sha256(gold_path),
            "fields": ["context", "supporting_facts"]
            if cfg["domain"] == "text"
            else ["gold explanations"],
        }

    metrics = {
        "schema_version": "production_generator_d_v1_clean_gnn_rag_final_test_metrics_v1",
        "status": "complete",
        "split": "test",
        "prediction_freeze_sha256": freeze["prediction_freeze_sha256"],
        "native_retrieval_semantics": "non-seed entities ranked by ReaRev probability and truncated after cumulative probability exceeds eps=0.95; shortest paths from question seed nodes form reader context",
        "common_metric_semantics": "macro best-match support-set P/R/F1 over support-bearing examples, computed only after global prediction freeze",
        "datasets": metric_rows,
    }
    write_jsonl(output_dir / "per_example_retrieval.jsonl", evaluated, exclusive=False)
    write_json(output_dir / "metrics.json", metrics, exclusive=False)
    (output_dir / "summary.md").write_text(
        summary_markdown(metrics, freeze["prediction_freeze_sha256"]),
        encoding="utf-8",
        newline="\n",
    )
    lineage = read_json(output_dir / "lineage_metadata.json")
    lineage.update(
        {
            "completed_at_utc": utc_now(),
            "post_freeze_gold_sources": gold_lineage,
            "evaluation_code_state": {
                "base_git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "file": "evaluation/finalize_production_gnn_rag_final_test.py",
                "sha256": sha256(Path(__file__)),
            },
            "evaluation_correction": "Native entity precision maps every ranked evidence entity through the frozen candidate-unit identity map before gold overlap; predictions and paths were not changed.",
        }
    )
    write_json(output_dir / "lineage_metadata.json", lineage, exclusive=False)
    write_json(output_dir / "artifact_manifest.json", {"being_regenerated": True}, exclusive=False)
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[path.relative_to(output_dir).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_json(
        output_dir / "artifact_manifest.json",
        {"self_excluded_from_hashes": True, "files": files},
        exclusive=False,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "examples": len(evaluated),
                "prediction_freeze_sha256": freeze["prediction_freeze_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
