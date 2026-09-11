"""Train frozen three-epoch ReaRev checkpoints from clean TRAIN/DEV adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.prepare_production_gnn_rag_clean import DATASETS, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints/production_generator_d_v1_gnn_rag_clean"),
    )
    parser.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = (
        (root / args.output_root).resolve()
        if not args.output_root.is_absolute()
        else args.output_root.resolve()
    )
    gnn_root = root / "third_party" / "GNN-RAG" / "gnn"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    for dataset in args.datasets:
        dataset_dir = output_root / dataset
        adapter_dir = dataset_dir / "adapter"
        if (adapter_dir / "test.json").exists():
            raise RuntimeError(f"TEST seal violation: {adapter_dir / 'test.json'} exists")
        invariance = json.loads(
            (dataset_dir / "leakage_invariance.json").read_text(encoding="utf-8")
        )
        if not invariance.get("pass"):
            raise RuntimeError(f"Leakage-invariance gate did not pass for {dataset}")
        experiment = f"rearev_lstm_clean_{dataset.lower()}"
        final_ckpt = dataset_dir / f"{experiment}-final.ckpt"
        if final_ckpt.exists():
            raise FileExistsError(f"Refusing to overwrite clean checkpoint: {final_ckpt}")
        dev_metrics = dataset_dir / "dev_metrics.json"
        config = {
            "dataset": dataset,
            "model": "ReaRev",
            "lm": "lstm",
            "entity_dim": 50,
            "num_epoch": 3,
            "batch_size": 8,
            "eval_every": 1,
            "num_iter": 2,
            "num_ins": 2,
            "num_gnn": 3,
            "relation_word_emb": False,
            "train_dev_only": True,
            "test_loaded": False,
            "legacy_checkpoint_reused": False,
            "checkpoint_selection_rule": "final epoch 3, preserving the historical production runner's explicit final-checkpoint semantics",
            "auxiliary_checkpoint_rule": "highest DEV F1 after upstream warmup semantics; not selected for production",
            "code_commit": commit,
        }
        write_json(dataset_dir / "training_config.json", config)
        command = [
            sys.executable,
            str(gnn_root / "main.py"),
            "ReaRev",
            "--entity_dim",
            "50",
            "--num_epoch",
            "3",
            "--batch_size",
            "8",
            "--eval_every",
            "1",
            "--data_folder",
            str(adapter_dir) + "\\",
            "--lm",
            "lstm",
            "--num_iter",
            "2",
            "--num_ins",
            "2",
            "--num_gnn",
            "3",
            "--relation_word_emb",
            "false",
            "--checkpoint_dir",
            str(dataset_dir),
            "--experiment_name",
            experiment,
            "--name",
            f"sageqa-clean-{dataset.lower()}",
            "--train_dev_only",
            "true",
            "--dev_metrics_file",
            str(dev_metrics),
        ]
        write_json(dataset_dir / "training_command.json", command)
        log_path = dataset_dir / "training.log"
        print(f"Training clean ReaRev checkpoint: {dataset}", flush=True)
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
            log.flush()
            result = subprocess.run(
                command, cwd=gnn_root, stdout=log, stderr=subprocess.STDOUT, text=True
            )
        if result.returncode:
            raise RuntimeError(f"Training failed for {dataset}; see {log_path}")
        metrics = json.loads(dev_metrics.read_text(encoding="utf-8"))
        if [row["epoch"] for row in metrics] != [1, 2, 3]:
            raise RuntimeError(f"Incomplete DEV metrics for {dataset}: {metrics}")
        manifest = {
            "dataset": dataset,
            "selected_checkpoint": final_ckpt.name,
            "selected_checkpoint_sha256": sha256_file(final_ckpt),
            "selection_rule": config["checkpoint_selection_rule"],
            "train_manifest_sha256": sha256_file(dataset_dir / "train_manifest.json"),
            "dev_manifest_sha256": sha256_file(dataset_dir / "dev_manifest.json"),
            "relation_vocabulary_sha256": sha256_file(adapter_dir / "relations.txt"),
            "dev_metrics": metrics,
            "test_inference_occurred": False,
            "test_metrics_inspected": False,
            "legacy_checkpoint_reused": False,
            "code_commit": commit,
        }
        write_json(dataset_dir / "frozen_checkpoint_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
