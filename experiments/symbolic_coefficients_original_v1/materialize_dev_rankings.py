"""Materialize selected-coefficient DEV rankings for adaptive-k fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.symbolic_coefficients_original_v1.original_symbolic_features import (
    ProofCoefficients,
    TextCoefficients,
    score_proof_features,
    score_text_features,
)
from experiments.symbolic_coefficients_original_v1.prepare_dev_cache import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256,
    validate_cache,
    write_json,
    write_jsonl,
)


K_VALUES = (1, 2, 3, 5)
METHOD = "final_sageqa"


def support_scores(predicted: Sequence[str], alternatives: Sequence[Sequence[str]]) -> dict[str, float]:
    predicted_set = set(predicted)
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for gold in alternatives:
        gold_set = set(gold)
        overlap = len(predicted_set & gold_set)
        precision = overlap / len(predicted_set) if predicted_set else 0.0
        recall = overlap / len(gold_set) if gold_set else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"precision": precision, "recall": recall, "f1": f1}
    return best


def aggregate_prefix(candidates: Sequence[Mapping[str, Any]], k: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates[:k]:
        for unit in candidate["subgraph_units"]:
            if unit not in seen:
                seen.add(unit)
                result.append(unit)
    return result


def materialize(cache: Path, cache_manifest: Path, coefficients_path: Path, freeze_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    validate_cache(cache, cache_manifest)
    selected = read_json(coefficients_path)
    freeze = read_json(freeze_manifest_path)
    if sha256(coefficients_path) != freeze.get("selected_coefficients_sha256"):
        raise ValueError("Selected coefficient hash does not match freeze manifest")
    if selected.get("status") != "frozen_after_dev_selection" or selected.get("split") != "dev":
        raise ValueError("Coefficients are not a frozen DEV selection")
    text_coefficients = TextCoefficients(**selected["text"])
    proof_coefficients = ProofCoefficients(**selected["ontology"])
    output_rows: list[dict[str, Any]] = []
    dataset_metrics: dict[str, dict[str, list[float]]] = {}
    for row in read_jsonl(cache):
        domain = str(row["domain"])
        if len(row.get("candidates", [])) < max(K_VALUES):
            raise ValueError(f"{row.get('example_id')}: fewer than five cached candidates")
        ranked = []
        for candidate in row["candidates"]:
            adjustment = (
                score_text_features(candidate["features"], text_coefficients)
                if domain == "text"
                else score_proof_features(candidate["features"], proof_coefficients)
            )
            ranked.append({
                "candidate_identity": candidate["candidate_identity"],
                "score": float(candidate["cross_encoder_logit"]),
                "adjusted_score": float(candidate["cross_encoder_logit"]) + adjustment,
                "subgraph_size": len(candidate["subgraph_units"]),
                "subgraph_units": list(candidate["subgraph_units"]),
            })
        ranked.sort(key=lambda candidate: (-candidate["adjusted_score"], candidate["candidate_identity"]))
        prefixes = []
        for k in K_VALUES:
            units = aggregate_prefix(ranked, k)
            scores = support_scores(units, row["gold_explanations"])
            prefixes.append({"k": k, "selected_k": min(k, len(ranked)), "retrieved_evidence_units": units, "retrieved_evidence_count": len(units), **scores})
            dataset_metrics.setdefault(row["dataset"], {}).setdefault(str(k), []).append(scores["f1"])
        output_rows.append({
            "schema_version": "sageqa_symbolic_coefficients_original_v1_dev_ranking",
            "split": "dev",
            "method": METHOD,
            "dataset": row["dataset"],
            "domain": domain,
            "score_mode": "sageqa_text_chain" if domain == "text" else "sageqa_proof",
            "example_id": row["example_id"],
            "gold_explanations": row["gold_explanations"],
            "ranked_candidates": ranked,
            "prefix_evaluation": prefixes,
        })
    ranking_path = output_dir / "per_example_rankings.jsonl"
    write_jsonl(ranking_path, output_rows)
    metrics = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_dev_metrics",
        "status": "complete",
        "split": "dev",
        "method": METHOD,
        "k_values": list(K_VALUES),
        "datasets": {
            dataset: {key: {"support_f1": sum(values) / len(values)} for key, values in sorted(by_k.items())}
            for dataset, by_k in sorted(dataset_metrics.items())
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    metadata = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_dev_ranking_metadata",
        "status": "frozen_dev_rankings_for_adaptive_k",
        "adaptive_k_ready": True,
        "split": "dev",
        "test_rows_read": 0,
        "test_gold_accessed": False,
        "candidate_generation_rerun": False,
        "cross_encoder_predictions_regenerated": False,
        "inputs_sha256": {
            "cache": sha256(cache),
            "cache_manifest": sha256(cache_manifest),
            "selected_coefficients": sha256(coefficients_path),
            "freeze_manifest": sha256(freeze_manifest_path),
        },
        "rankings_sha256": sha256(ranking_path),
        "examples": len(output_rows),
        "datasets": {
            dataset: {"dev_path": str(cache), "dev_sha256": sha256(cache), "final_reranking_score_mode": "selected_original_v1_symbolic_coefficients"}
            for dataset in sorted(dataset_metrics)
        },
    }
    write_json(output_dir / "checkpoint_metadata.json", metadata)
    manifest = {
        "schema_version": "sageqa_symbolic_coefficients_original_v1_dev_ranking_manifest",
        "split": "dev",
        "test_accessed": False,
        "files_sha256": {
            name: sha256(output_dir / name)
            for name in ("per_example_rankings.jsonl", "metrics.json", "checkpoint_metadata.json")
        },
        "code_sha256": {
            "materialize_dev_rankings.py": sha256(Path(__file__)),
            "original_symbolic_features.py": sha256(Path(__file__).with_name("original_symbolic_features.py")),
            "PROTOCOL.md": sha256(Path(__file__).with_name("PROTOCOL.md")),
        },
    }
    write_json(output_dir / "ranking_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(canonical_json(materialize(args.cache, args.cache_manifest, args.coefficients, args.freeze_manifest, args.output_dir)))
