"""Run the single precommitted SAGE-QA v2 training protocol sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("HotpotQA", "hotpotqa"),
    ("2WikiMultiHopQA", "2wiki"),
    ("FamilyOWL_1hop", "familyowl_1hop"),
    ("FamilyOWL_2hop", "familyowl_2hop"),
    ("pizza_100_1hop", "pizza_100_1hop"),
    ("pizza_100_2hop", "pizza_100_2hop"),
    ("pizza_250_1hop", "pizza_250_1hop"),
    ("pizza_250_2hop", "pizza_250_2hop"),
    ("OWL2Bench_1hop", "OWL2Bench_1hop"),
    ("OWL2Bench_2hop", "OWL2Bench_2hop"),
)
IMPLEMENTATION_FILES = (
    "training/train_gnn_subgraph_retriever.py",
    "training/run_sageqa_v2_final.py",
    "evaluation/audit_hard_pair_reservation.py",
    "tests/test_hard_pair_reservation.py",
)
EXPECTED_CHECKPOINT_CONFIG = {
    "model_name": "google/bert_uncased_L-2_H-128_A-2",
    "freeze_encoder": True,
    "training_objective": "within_question_pairwise_ranking",
    "ranking_margin": 0.2,
    "ranking_weight": 2.0,
    "bce_weight": 0.2,
    "listwise_weight": 0.0,
    "max_pairs": 512,
    "hard_pair_reservation": True,
    "score_mode": "neural",
    "size_penalty": 0.01,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def checked_split(data_root: Path, dataset: str, split: str) -> Path:
    if split not in {"train", "dev"}:
        raise ValueError("SAGE-QA v2 training may load only TRAIN and DEV")
    path = data_root / dataset / f"{split}_subgraph_retrieval.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def training_command(python: str, train_path: Path, dev_path: Path, save_dir: Path) -> list[str]:
    return [
        python,
        "training/train_gnn_subgraph_retriever.py",
        "--train-path",
        str(train_path),
        "--dev-path",
        str(dev_path),
        "--save-dir",
        str(save_dir),
        "--model-name",
        "google/bert_uncased_L-2_H-128_A-2",
        "--epochs",
        "3",
        "--max-length",
        "128",
        "--candidate-batch-size",
        "512",
        "--freeze-encoder",
        "--ranking-margin",
        "0.2",
        "--ranking-weight",
        "2.0",
        "--bce-weight",
        "0.2",
        "--listwise-weight",
        "0.0",
        "--max-pairs",
        "512",
        "--score-mode",
        "neural",
        "--size-penalty",
        "0.01",
        "--hard-pair-reservation",
    ]


def validate_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual = {key: checkpoint.get(key) for key in EXPECTED_CHECKPOINT_CONFIG}
    if actual != EXPECTED_CHECKPOINT_CONFIG:
        raise AssertionError(f"Frozen checkpoint configuration mismatch: {actual}")
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "configuration": actual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "checkpoints/production_generator_d_v2"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "outputs/final_results/production_generator_d_v2_training",
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=ROOT
        / "outputs/final_results/production_generator_d_v2_freeze/implementation_invariance_gate.json",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.run_root = args.run_root.resolve()
    args.gate = args.gate.resolve()

    if args.checkpoint_root.exists() or args.run_root.exists():
        raise FileExistsError("Refusing to resume or overwrite a SAGE-QA v2 final run")
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("global") != {
        "eligible_examples": 7604,
        "reserved_candidate_inclusion_rate": 1.0,
        "reserved_margin_pair_inclusion_rate": 1.0,
    }:
        raise RuntimeError("The predeclared implementation/invariance gate did not pass")

    implementation = {
        path: {"sha256": sha256(ROOT / path), "size_bytes": (ROOT / path).stat().st_size}
        for path in IMPLEMENTATION_FILES
    }
    data = {}
    commands = {}
    for dataset, slug in DATASETS:
        train_path = checked_split(args.data_root, dataset, "train")
        dev_path = checked_split(args.data_root, dataset, "dev")
        data[dataset] = {
            "train": {"path": str(train_path.relative_to(ROOT)), "sha256": sha256(train_path)},
            "dev": {"path": str(dev_path.relative_to(ROOT)), "sha256": sha256(dev_path)},
        }
        commands[dataset] = training_command(
            sys.executable, train_path, dev_path, args.checkpoint_root / slug
        )

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
    manifest = {
        "schema_version": "sageqa_v2_final_training_freeze_v1",
        "status": "implementation_and_protocol_frozen_before_training",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_status": status,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "only_training_mechanism_changed": "deterministic complete and hard-incomplete candidate reservation plus exactly-once ordered margin-pair reservation",
        "dataset_order": [dataset for dataset, _ in DATASETS],
        "implementation": implementation,
        "gate": {
            "path": str(args.gate.relative_to(ROOT)),
            "sha256": sha256(args.gate),
            "global": gate["global"],
        },
        "data": data,
        "commands": commands,
        "frozen_hyperparameters": EXPECTED_CHECKPOINT_CONFIG,
        "epochs": 3,
        "candidate_batch_size": 512,
        "max_length": 128,
        "test_data_accessed": False,
        "answer_generation_run": False,
    }
    freeze_path = args.run_root / "frozen_training_protocol.json"
    write_json(freeze_path, manifest)
    frozen_hashes = {path: values["sha256"] for path, values in implementation.items()}

    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["PYTHONUTF8"] = "1"
    completed = []
    for dataset, slug in DATASETS:
        if {path: sha256(ROOT / path) for path in IMPLEMENTATION_FILES} != frozen_hashes:
            raise RuntimeError("Frozen SAGE-QA v2 implementation changed after training began")
        save_dir = args.checkpoint_root / slug
        command = commands[dataset]
        log_path = args.run_root / "training_logs" / f"{slug}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"training {dataset}", flush=True)
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
            log.flush()
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        checkpoint = validate_checkpoint(save_dir / "best_model.pt")
        completed.append(
            {"dataset": dataset, "checkpoint": checkpoint, "log_sha256": sha256(log_path)}
        )
        print(f"completed {dataset}", flush=True)

    completion = {
        "schema_version": "sageqa_v2_final_training_completion_v1",
        "status": "all_ten_trainings_complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_training_protocol_sha256": sha256(freeze_path),
        "datasets": completed,
        "implementation_hashes_unchanged": {
            path: sha256(ROOT / path) == digest for path, digest in frozen_hashes.items()
        },
        "test_data_accessed": False,
        "answer_generation_run": False,
    }
    if not all(completion["implementation_hashes_unchanged"].values()):
        raise RuntimeError("Implementation changed during final training")
    write_json(args.run_root / "all_training_complete.json", completion)
    print(json.dumps({"complete": True, "datasets": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
