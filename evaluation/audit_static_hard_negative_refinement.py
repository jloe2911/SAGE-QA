"""Bounded implementation/invariance gate for the frozen static refinement."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training import run_static_hard_negative_refinement as refinement
from training.train_gnn_subgraph_retriever import encode_example_graph, score_candidate_rows


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints/production_generator_d_v1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/final_model_development/production_generator_d_static_hard_v1_gate/implementation_invariance_gate.json",
    )
    args = parser.parse_args()
    source = inspect.getsource(refinement)
    prohibited = [
        token
        for token in ("load_jsonl(", "prepare_examples(", "score_candidate_rows(")
        if token in source
    ]
    if prohibited:
        raise AssertionError(f"Corpus/full-candidate training path found: {prohibited}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {}
    for dataset, slug, expected_eligible in refinement.DATASETS:
        path = args.data_root / dataset / "train_subgraph_retrieval.jsonl"
        for example_id, raw_rows in refinement.grouped_jsonl(path):
            example = refinement.prepare_static_pair(raw_rows)
            if example is not None:
                break
        checkpoint, tokenizer, model = refinement.load_v1(
            args.checkpoint_root / slug / "best_model.pt", device
        )
        model.eval()
        encoded = encode_example_graph(
            model, tokenizer, example, device, max_length=refinement.FROZEN["max_length"]
        )
        with torch.no_grad():
            production_logits = score_candidate_rows(
                model, encoded, example["candidate_rows"], device
            )["logits"]
            representations = refinement.frozen_pair_representations(
                model, tokenizer, example, device
            )
            static_logits = model.classifier(representations).squeeze(-1)
        max_difference = float((production_logits - static_logits).abs().max().cpu())
        if max_difference > 1e-6:
            raise AssertionError(
                f"Frozen representation/scoring mismatch for {dataset}: {max_difference}"
            )
        refinement.freeze_except_classifier(model)
        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if not trainable or any(not name.startswith("classifier.") for name in trainable):
            raise AssertionError(f"Unexpected trainable parameters: {trainable}")
        datasets[dataset] = {
            "example_id": example_id,
            "raw_candidate_metadata_rows_held_for_current_example": len(raw_rows),
            "prepared_and_scored_training_rows": len(example["candidate_rows"]),
            "gnn_forwards_in_static_path": 1,
            "production_static_logit_max_abs_difference": max_difference,
            "trainable_parameters": trainable,
            "eligible_examples_from_existing_full_train_audit": expected_eligible,
        }
    report = {
        "schema_version": "sageqa_static_hard_refinement_gate_v1",
        "status": "passed",
        "split": "train",
        "bounded_code_path_check_not_performance_experiment": True,
        "implementation_sha256": refinement.sha256(Path(refinement.__file__).resolve()),
        "full_train_corpus_materialized": False,
        "full_candidate_set_scored_as_training_rows": False,
        "selected_training_rows_per_eligible_example": 2,
        "pair_selector": "existing select_reserved_hard_pair",
        "expected_eligible_examples": sum(item[2] for item in refinement.DATASETS),
        "datasets": datasets,
        "dev_accessed": False,
        "test_accessed": False,
        "answer_generation_run": False,
    }
    refinement.write_json(args.output, report)
    print(json.dumps({"status": "passed", "datasets": len(datasets), "output": str(args.output)}))


if __name__ == "__main__":
    main()
