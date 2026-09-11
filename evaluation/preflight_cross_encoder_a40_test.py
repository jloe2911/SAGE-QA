"""Validate the immutable cross-encoder adaptive TEST bundle before A40 inference."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.run_production_dev_k_sensitivity import grouped_jsonl  # noqa: E402
from evaluation.run_production_test_retrieval import DATASETS, load_policy_bundle, sha256  # noqa: E402
from experiments.cross_encoder_reranking_dev_v1.run_experiment import (  # noqa: E402
    MAX_INFERENCE_CANDIDATES,
    MAX_LENGTH,
    safe_inference_row,
)


EXPECTED = {
    "HotpotQA": (1000, 509863, "e295a5217f8042ba777f70415461ac1d10f3b93863b6611aa7de962a9713abd8"),
    "2WikiMultiHopQA": (1000, 502407, "1cd172edecbf8053f0259362331ee6a43f3b44b4e25871022d72d4d6b78f5569"),
    "FamilyOWL_1hop": (462, 236544, "287011713b18fd251a368ea81c712b3726cf4bbe509d21b5b7d7bb4083b60ebe"),
    "FamilyOWL_2hop": (462, 236544, "2d49a62953cd98226cb44fb80ce9ca0ae8b8a226715dc0b84dcccff830664673"),
    "pizza_100_1hop": (119, 60720, "cef8af4df1e83e0ff62e5789948c6cf1ed130d9fc76b2520197eeeed3eb5e956"),
    "pizza_100_2hop": (125, 64000, "baa5aa0254bb8f1970259b9ab95ce82e9172e76aaf94e13aa89a1424cec0fb7f"),
    "pizza_250_1hop": (149, 75626, "d02d1b42a4a3d3f66a3504a7cd57bc39eae57f16261004b5ba27164578dc9e2f"),
    "pizza_250_2hop": (157, 80384, "7b7b794830f044e52843e79b803d9226b17ec499bf8f41183039276c4895c12c"),
    "OWL2Bench_1hop": (376, 192512, "cc8aa7dc07a43e73cfb07d194bbc0e5a0addba00c54c8ddfadd702e5eb8fb877"),
    "OWL2Bench_2hop": (399, 204288, "38c8233f8f63fd9a95404449526670c8ecad6faaa55b7a849bf154f7d7c7de23"),
}
EXPECTED_CHECKPOINT_SHA256 = "02b06c91fa422369a0b0adbab3b33bfcf80f335b7c8a5911084c3b82af117ba9"
EXPECTED_THRESHOLDS = {
    "cross_encoder": {"text": 0.85, "ontology": 0.79},
    "final_sageqa": {"text": 0.81, "ontology": 0.87},
}
PROHIBITED = {
    "gold_explanations",
    "gold_units",
    "gold_support_units",
    "raw_supporting_facts",
    "supporting_facts",
    "evidences",
    "answer",
    "answers",
    "label",
    "rank_target",
    "best_set_f1_to_gold",
    "exact_match_any_gold",
    "contains_any_gold_explanation",
}
CODE_FILES = (
    "evaluation/run_cross_encoder_test_retrieval.py",
    "evaluation/adaptive_support_aggregation.py",
    "evaluation/adaptive_support_aggregation_v2.py",
    "evaluation/run_production_test_retrieval.py",
    "experiments/cross_encoder_reranking_dev_v1/run_experiment.py",
    "training/train_gnn_subgraph_retriever.py",
    "data_processing/retrieval_contracts.py",
)


def gpu_name() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(args: argparse.Namespace) -> None:
    if set(EXPECTED) != {dataset for dataset, *_ in DATASETS} or len(DATASETS) != 10:
        raise ValueError("Expected exactly the frozen ten-dataset TEST suite")
    device_name = gpu_name()
    if args.require_a40 and "A40" not in device_name.upper():
        raise RuntimeError(f"A40 required, found {device_name!r}")

    checkpoint_dir = args.cross_encoder_dir / "checkpoint_final"
    training = json.loads(
        (args.cross_encoder_dir / "training_report.json").read_text(encoding="utf-8")
    )
    checkpoint_hashes = {}
    for name, expected in training["checkpoint_files_sha256"].items():
        path = checkpoint_dir / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checkpoint/tokenizer hash mismatch: {path}")
        checkpoint_hashes[name] = actual
    if checkpoint_hashes.get("model.safetensors") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Unexpected encoder checkpoint does not match the frozen DEV checkpoint")

    policy_records = {}
    for method, root in (
        ("cross_encoder", args.cross_encoder_policy_dir),
        ("final_sageqa", args.final_sageqa_policy_dir),
    ):
        _, config, hashes = load_policy_bundle(root, method)
        if config.get("thresholds") != EXPECTED_THRESHOLDS[method]:
            raise ValueError(f"Frozen threshold mismatch for {method}")
        if config.get("allowed_k") != [1, 2, 3, 5]:
            raise ValueError(f"Frozen allowed-k mismatch for {method}")
        policy_records[method] = {
            "directory": str(root),
            "thresholds": config["thresholds"],
            "artifact_sha256": hashes,
        }

    datasets = {}
    for dataset, data_dir, _, domain, score_mode, _ in DATASETS:
        path = args.data_root / data_dir / "test_subgraph_retrieval.jsonl"
        expected_examples, expected_rows, expected_hash = EXPECTED[dataset]
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen Generator-D TEST hash mismatch: {dataset}")
        examples = 0
        rows = 0
        for _, group in grouped_jsonl(path):
            examples += 1
            rows += len(group)
            for raw in group:
                projected = safe_inference_row(raw)
                if PROHIBITED & set(projected):
                    raise ValueError(f"TEST-gold firewall failure: {dataset}")
        if (examples, rows) != (expected_examples, expected_rows):
            raise ValueError(
                f"Frozen TEST count mismatch for {dataset}: {(examples, rows)}"
            )
        datasets[dataset] = {
            "domain": domain,
            "score_mode": score_mode,
            "test_candidate_path": str(path),
            "test_candidate_sha256": actual_hash,
            "expected_examples": examples,
            "candidate_rows": rows,
        }

    manifest: dict[str, Any] = {
        "schema_version": "cross_encoder_adaptive_a40_test_preflight_v1",
        "status": "passed_frozen_preflight",
        "gpu": device_name,
        "a40_required": args.require_a40,
        "checkpoint_files_sha256": checkpoint_hashes,
        "policies": policy_records,
        "datasets": datasets,
        "expected_total_examples": sum(item[0] for item in EXPECTED.values()),
        "allowed_k": [1, 2, 3, 5],
        "max_inference_candidates": MAX_INFERENCE_CANDIDATES,
        "max_length": MAX_LENGTH,
        "serialization": "unchanged serialize_candidate from frozen DEV implementation",
        "test_gold_firewall": {
            "passed": True,
            "projection": "safe_inference_row allowlist",
            "gold_source_files_opened": 0,
            "prediction_freeze_required_before_evaluate": True,
        },
        "code_sha256": {name: sha256(Path(name)) for name in CODE_FILES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, "gpu": device_name, "examples": manifest["expected_total_examples"]}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/production_generator_d_v1"))
    parser.add_argument(
        "--cross-encoder-dir",
        type=Path,
        default=Path("outputs/development_runs/question_candidate_cross_encoder_v1"),
    )
    parser.add_argument("--cross-encoder-policy-dir", type=Path, required=True)
    parser.add_argument("--final-sageqa-policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-a40", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
