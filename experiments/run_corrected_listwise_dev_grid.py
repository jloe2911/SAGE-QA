"""Run the predeclared corrected-listwise grid on production Generator-D TRAIN/DEV."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.2)
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


def slug(weight: float) -> str:
    return "w_" + str(weight).replace(".", "p")


def checked_data_path(data_root: Path, dataset: str, split: str) -> Path:
    if split not in {"train", "dev"}:
        raise ValueError("Only TRAIN and DEV are allowed")
    path = data_root / dataset / f"{split}_subgraph_retrieval.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="*", type=float, default=list(WEIGHTS), choices=WEIGHTS)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[name for name, _ in DATASETS],
        choices=[name for name, _ in DATASETS],
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/production_generator_d_v1")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints/development/production_generator_d_v1_listwise_corrected_dev",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "outputs/development_runs/production_generator_d_v1_listwise_corrected_dev",
    )
    args = parser.parse_args()

    selected = [(name, directory) for name, directory in DATASETS if name in args.datasets]
    for weight in args.weights:
        weight_slug = slug(weight)
        weight_checkpoint_root = args.checkpoint_root / weight_slug
        weight_run_root = args.run_root / weight_slug
        for dataset, checkpoint_dir in selected:
            train_path = checked_data_path(args.data_root, dataset, "train")
            dev_path = checked_data_path(args.data_root, dataset, "dev")
            save_dir = weight_checkpoint_root / checkpoint_dir
            checkpoint = save_dir / "best_model.pt"
            if checkpoint.is_file():
                print(f"reuse complete checkpoint: {checkpoint}", flush=True)
                continue
            command = [
                sys.executable,
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
                str(weight),
                "--max-pairs",
                "512",
                "--score-mode",
                "neural",
                "--size-penalty",
                "0.01",
            ]
            print(f"train {dataset} listwise_weight={weight}", flush=True)
            run_logged(command, weight_run_root / "training_logs" / f"{checkpoint_dir}.log")

        if len(selected) == len(DATASETS):
            metrics_path = weight_run_root / "metrics.json"
            if not metrics_path.is_file():
                print(f"evaluate all DEV datasets listwise_weight={weight}", flush=True)
                run_logged(
                    [
                        sys.executable,
                        "evaluation/run_production_dev_k_sensitivity.py",
                        "--data-root",
                        str(args.data_root),
                        "--checkpoint-root",
                        str(weight_checkpoint_root),
                        "--output-dir",
                        str(weight_run_root),
                        "--max-length",
                        "128",
                        "--candidate-batch-size",
                        "512",
                    ],
                    weight_run_root / "evaluation.log",
                )
            if weight == 0.0:
                gate_output = (
                    ROOT
                    / "outputs/diagnostics/production_generator_d_v1_listwise_corrected_dev/baseline_reproduction.json"
                )
                run_logged(
                    [
                        sys.executable,
                        "evaluation/verify_listwise_zero_baseline.py",
                        "--fresh-run",
                        str(weight_run_root),
                        "--fresh-checkpoints",
                        str(weight_checkpoint_root),
                        "--output",
                        str(gate_output),
                    ],
                    weight_run_root / "baseline_reproduction.log",
                )


if __name__ == "__main__":
    main()
