import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATASETS = {
    "FamilyOWL_1hop": {
        "train": "data/FamilyOWL_1hop/train_subgraph_retrieval.jsonl",
        "dev": "data/FamilyOWL_1hop/dev_subgraph_retrieval.jsonl",
        "test": "data/FamilyOWL_1hop/test_subgraph_retrieval.jsonl",
        "source_name": "FamilyOWL_1hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_1hop",
        "results_dir": "outputs/final_results/FamilyOWL_1hop",
    },
    "FamilyOWL_2hop": {
        "train": "data/FamilyOWL_2hop/train_subgraph_retrieval.jsonl",
        "dev": "data/FamilyOWL_2hop/dev_subgraph_retrieval.jsonl",
        "test": "data/FamilyOWL_2hop/test_subgraph_retrieval.jsonl",
        "source_name": "FamilyOWL_2hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_2hop",
        "results_dir": "outputs/final_results/FamilyOWL_2hop",
    },
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def path_exists(path: str | Path) -> bool:
    return Path(path).exists()


def quote_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_command(
    cmd: List[str],
    log_path: Path,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> int:
    ensure_dir(log_path.parent)

    printable = quote_cmd(cmd)
    print("\n" + "=" * 100)
    print(printable)
    print("=" * 100)

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(printable + "\n\n")
        log_file.flush()

        if dry_run:
            print(f"[DRY RUN] Would write log to {log_path}")
            return 0

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env or os.environ.copy(),
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        process.wait()
        return int(process.returncode)


def require_file(path: str | Path, description: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def write_run_config(args: argparse.Namespace, output_root: Path) -> None:
    ensure_dir(output_root)
    config_path = output_root / "run_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved run config to {config_path}")


def run_random_subgraph(
    dataset_name: str,
    source_name: str,
    test_path: Path,
    results_dir: Path,
    logs_dir: Path,
    input_format: str,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "evaluation/eval_subgraph_baselines.py",
        "--test-path",
        str(test_path),
        "--baseline",
        "random",
        "--input-format",
        input_format,
        "--output-dir",
        str(results_dir),
        "--source-name",
        source_name,
    ]

    log_path = logs_dir / dataset_name / "random_subgraph.log"
    code = run_command(cmd, log_path=log_path, dry_run=dry_run)
    if code != 0:
        raise RuntimeError(f"Random subgraph baseline failed for {dataset_name}")


def run_lexical_subgraph(
    dataset_name: str,
    source_name: str,
    test_path: Path,
    results_dir: Path,
    logs_dir: Path,
    input_format: str,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "evaluation/eval_subgraph_baselines.py",
        "--test-path",
        str(test_path),
        "--baseline",
        "lexical",
        "--input-format",
        input_format,
        "--output-dir",
        str(results_dir),
        "--source-name",
        source_name,
    ]

    log_path = logs_dir / dataset_name / "lexical_subgraph.log"
    code = run_command(cmd, log_path=log_path, dry_run=dry_run)
    if code != 0:
        raise RuntimeError(f"Lexical subgraph baseline failed for {dataset_name}")


def run_random_axiom(
    dataset_name: str,
    source_name: str,
    test_path: Path,
    results_dir: Path,
    logs_dir: Path,
    input_format: str,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "evaluation/eval_axiom_retriever_baseline.py",
        "--test-path",
        str(test_path),
        "--baseline",
        "random",
        "--input-format",
        input_format,
        "--support-size-mode",
        "min_gold",
        "--output-dir",
        str(results_dir),
        "--source-name",
        source_name,
    ]

    log_path = logs_dir / dataset_name / "random_axiom.log"
    code = run_command(cmd, log_path=log_path, dry_run=dry_run)
    if code != 0:
        raise RuntimeError(f"Random axiom baseline failed for {dataset_name}")


def run_lexical_axiom(
    dataset_name: str,
    source_name: str,
    test_path: Path,
    results_dir: Path,
    logs_dir: Path,
    input_format: str,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "evaluation/eval_axiom_retriever_baseline.py",
        "--test-path",
        str(test_path),
        "--baseline",
        "lexical",
        "--input-format",
        input_format,
        "--support-size-mode",
        "min_gold",
        "--output-dir",
        str(results_dir),
        "--source-name",
        source_name,
    ]

    log_path = logs_dir / dataset_name / "lexical_axiom.log"
    code = run_command(cmd, log_path=log_path, dry_run=dry_run)
    if code != 0:
        raise RuntimeError(f"Lexical axiom baseline failed for {dataset_name}")


def train_gnn(
    dataset_name: str,
    train_path: Path,
    dev_path: Path,
    checkpoint_dir: Path,
    logs_dir: Path,
    args: argparse.Namespace,
) -> None:
    best_model = checkpoint_dir / "best_model.pt"

    if best_model.exists() and not args.force_train:
        print(f"[SKIP] Existing GNN checkpoint for {dataset_name}: {best_model}")
        return

    cmd = [
        sys.executable,
        "training/train_gnn_subgraph_retriever.py",
        "--train-path",
        str(train_path),
        "--dev-path",
        str(dev_path),
        "--save-dir",
        str(checkpoint_dir),
        "--model-name",
        args.model_name,
        "--epochs",
        str(args.epochs),
        "--candidate-batch-size",
        str(args.candidate_batch_size),
        "--score-mode",
        args.training_score_mode,
        "--ranking-margin",
        str(args.ranking_margin),
        "--ranking-weight",
        str(args.ranking_weight),
        "--bce-weight",
        str(args.bce_weight),
        "--listwise-weight",
        str(args.listwise_weight),
        "--source-name",
        dataset_name,
    ]

    if args.freeze_encoder:
        cmd.append("--freeze-encoder")

    # Optional candidate subsampling flags.
    # These will only work if you added them to train_gnn_subgraph_retriever.py.
    if args.use_subsampling_flags:
        cmd.extend(
            [
                "--max-pos-per-example",
                str(args.max_pos_per_example),
                "--max-hard-neg-per-example",
                str(args.max_hard_neg_per_example),
                "--max-easy-neg-per-example",
                str(args.max_easy_neg_per_example),
            ]
        )

    log_path = logs_dir / dataset_name / "train_gnn.log"
    code = run_command(cmd, log_path=log_path, dry_run=args.dry_run)
    if code != 0:
        raise RuntimeError(f"GNN training failed for {dataset_name}")


def eval_gnn(
    dataset_name: str,
    source_name: str,
    test_path: Path,
    checkpoint_dir: Path,
    results_dir: Path,
    logs_dir: Path,
    score_mode: str,
    dry_run: bool,
) -> None:
    checkpoint = checkpoint_dir / "best_model.pt"
    require_file(checkpoint, f"GNN checkpoint for {dataset_name}")

    if score_mode == "neural":
        out_dir = results_dir / "gnn_neural"
        log_name = "eval_gnn_neural.log"

    elif score_mode == "nesyqa_compact":
        out_dir = results_dir / "gnn_nesyqa_compact"
        log_name = "eval_gnn_nesyqa_compact.log"

    else:
        out_dir = results_dir / f"gnn_{score_mode}"
        log_name = f"eval_gnn_{score_mode}.log"

    cmd = [
        sys.executable,
        "evaluation/eval_gnn_subgraph_retriever.py",
        "--test-path",
        str(test_path),
        "--checkpoint",
        str(checkpoint),
        "--score-mode",
        score_mode,
        "--save-details",
        "--details-dir",
        str(out_dir),
        "--source-name",
        source_name,
    ]

    log_path = logs_dir / dataset_name / log_name
    code = run_command(cmd, log_path=log_path, dry_run=dry_run)
    if code != 0:
        raise RuntimeError(
            f"GNN evaluation failed for {dataset_name}, score_mode={score_mode}"
        )


def collect_results(args: argparse.Namespace, logs_dir: Path) -> None:
    cmd = [
        sys.executable,
        "evaluation/collect_final_results.py",
        "--results-root",
        args.results_root,
        "--output-csv",
        str(Path(args.results_root) / "final_summary.csv"),
        "--output-latex",
        str(Path(args.results_root) / "final_summary.tex"),
    ]

    log_path = logs_dir / "collect_final_results.log"
    code = run_command(cmd, log_path=log_path, dry_run=args.dry_run)
    if code != 0:
        raise RuntimeError("Final result collection failed.")


def check_required_scripts(args: argparse.Namespace) -> None:
    required = [
        "evaluation/eval_subgraph_baselines.py",
        "evaluation/eval_axiom_retriever_baseline.py",
        "training/train_gnn_subgraph_retriever.py",
        "evaluation/eval_gnn_subgraph_retriever.py",
        "evaluation/collect_final_results.py",
    ]

    for script in required:
        require_file(script, "required script")


def resolve_dataset_config(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    """
    Uses per-dataset split files by default:
      <data_root>/FamilyOWL_1hop/train_subgraph_retrieval.jsonl
      <data_root>/FamilyOWL_1hop/dev_subgraph_retrieval.jsonl
      <data_root>/FamilyOWL_1hop/test_subgraph_retrieval.jsonl
      <data_root>/FamilyOWL_2hop/...
    """
    if args.data_root is None:
        return DEFAULT_DATASETS

    root = Path(args.data_root)

    return {
        "FamilyOWL_1hop": {
            "train": str(root / "FamilyOWL_1hop" / "train_subgraph_retrieval.jsonl"),
            "dev": str(root / "FamilyOWL_1hop" / "dev_subgraph_retrieval.jsonl"),
            "test": str(root / "FamilyOWL_1hop" / "test_subgraph_retrieval.jsonl"),
            "source_name": "FamilyOWL_1hop",
            "checkpoint_dir": str(
                Path(args.checkpoints_root) / "gnn_subgraph_ranker_1hop"
            ),
            "results_dir": str(Path(args.results_root) / "FamilyOWL_1hop"),
        },
        "FamilyOWL_2hop": {
            "train": str(root / "FamilyOWL_2hop" / "train_subgraph_retrieval.jsonl"),
            "dev": str(root / "FamilyOWL_2hop" / "dev_subgraph_retrieval.jsonl"),
            "test": str(root / "FamilyOWL_2hop" / "test_subgraph_retrieval.jsonl"),
            "source_name": "FamilyOWL_2hop",
            "checkpoint_dir": str(
                Path(args.checkpoints_root) / "gnn_subgraph_ranker_2hop"
            ),
            "results_dir": str(Path(args.results_root) / "FamilyOWL_2hop"),
        },
    }


def validate_dataset_files(
    dataset_config: Dict[str, Dict[str, str]], skip_training: bool
) -> None:
    for dataset_name, cfg in dataset_config.items():
        require_file(cfg["test"], f"{dataset_name} test file")

        if not skip_training:
            require_file(cfg["train"], f"{dataset_name} train file")
            require_file(cfg["dev"], f"{dataset_name} dev file")


def run_dataset(
    dataset_name: str, cfg: Dict[str, str], args: argparse.Namespace
) -> None:
    print("\n" + "#" * 100)
    print(f"# DATASET: {dataset_name}")
    print("#" * 100)

    train_path = Path(cfg["train"])
    dev_path = Path(cfg["dev"])
    test_path = Path(cfg["test"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    results_dir = Path(cfg["results_dir"])
    source_name = cfg["source_name"]
    logs_dir = Path(args.logs_dir)

    ensure_dir(results_dir)
    ensure_dir(checkpoint_dir)

    if not args.skip_random_subgraph:
        run_random_subgraph(
            dataset_name=dataset_name,
            source_name=source_name,
            test_path=test_path,
            results_dir=results_dir,
            logs_dir=logs_dir,
            input_format=args.input_format,
            dry_run=args.dry_run,
        )

    if not args.skip_lexical_subgraph:
        run_lexical_subgraph(
            dataset_name=dataset_name,
            source_name=source_name,
            test_path=test_path,
            results_dir=results_dir,
            logs_dir=logs_dir,
            input_format=args.input_format,
            dry_run=args.dry_run,
        )

    if not args.skip_random_axiom:
        run_random_axiom(
            dataset_name=dataset_name,
            source_name=source_name,
            test_path=test_path,
            results_dir=results_dir,
            logs_dir=logs_dir,
            input_format=args.input_format,
            dry_run=args.dry_run,
        )

    if not args.skip_lexical_axiom:
        run_lexical_axiom(
            dataset_name=dataset_name,
            source_name=source_name,
            test_path=test_path,
            results_dir=results_dir,
            logs_dir=logs_dir,
            input_format=args.input_format,
            dry_run=args.dry_run,
        )

    if not args.skip_gnn_training:
        train_gnn(
            dataset_name=dataset_name,
            train_path=train_path,
            dev_path=dev_path,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
            args=args,
        )
    else:
        print(f"[SKIP] GNN training for {dataset_name}")

    if not args.skip_gnn_eval:
        eval_gnn(
            dataset_name=dataset_name,
            source_name=source_name,
            test_path=test_path,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            logs_dir=logs_dir,
            score_mode="neural",
            dry_run=args.dry_run,
        )

        eval_gnn(
            dataset_name=dataset_name,
            source_name=source_name,
            test_path=test_path,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            logs_dir=logs_dir,
            score_mode="nesyqa_compact",
            dry_run=args.dry_run,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final NeSyQA experiments for 1-hop and 2-hop datasets."
    )

    # Paths
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results-root", default="outputs/final_results")
    parser.add_argument("--checkpoints-root", default="checkpoints")
    parser.add_argument("--logs-dir", default=None)

    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FamilyOWL_1hop", "FamilyOWL_2hop"],
        choices=["FamilyOWL_1hop", "FamilyOWL_2hop"],
    )

    # Baseline input format
    parser.add_argument(
        "--input-format",
        choices=["nl", "abs", "sparql", "hybrid"],
        default="hybrid",
        help="Input representation for lexical/random baseline scoring.",
    )

    # GNN training settings
    parser.add_argument("--model-name", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--candidate-batch-size", type=int, default=512)
    parser.add_argument("--training-score-mode", default="completeness_adjusted")
    parser.add_argument("--ranking-margin", type=float, default=0.2)
    parser.add_argument("--ranking-weight", type=float, default=2.0)
    parser.add_argument("--bce-weight", type=float, default=0.2)
    parser.add_argument("--listwise-weight", type=float, default=0.0)

    # Optional candidate subsampling flags.
    # Turn on only if training/train_gnn_subgraph_retriever.py supports these CLI args.
    parser.add_argument("--use-subsampling-flags", action="store_true")
    parser.add_argument("--max-pos-per-example", type=int, default=64)
    parser.add_argument("--max-hard-neg-per-example", type=int, default=128)
    parser.add_argument("--max-easy-neg-per-example", type=int, default=128)

    # Skips
    parser.add_argument("--skip-random-subgraph", action="store_true")
    parser.add_argument("--skip-lexical-subgraph", action="store_true")
    parser.add_argument("--skip-random-axiom", action="store_true")
    parser.add_argument("--skip-lexical-axiom", action="store_true")
    parser.add_argument("--skip-gnn-training", action="store_true")
    parser.add_argument("--skip-gnn-eval", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")

    # Control
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.logs_dir is None:
        args.logs_dir = str(Path(args.results_root) / "logs" / now_stamp())

    return args


def main() -> None:
    args = parse_args()

    print("Running final NeSyQA pipeline with arguments:")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))

    check_required_scripts(args)

    dataset_config = resolve_dataset_config(args)
    dataset_config = {
        name: cfg for name, cfg in dataset_config.items() if name in set(args.datasets)
    }

    validate_dataset_files(dataset_config, skip_training=args.skip_gnn_training)

    results_root = Path(args.results_root)
    logs_dir = Path(args.logs_dir)

    ensure_dir(results_root)
    ensure_dir(logs_dir)

    write_run_config(args, results_root)

    for dataset_name, cfg in dataset_config.items():
        run_dataset(dataset_name, cfg, args)

    if not args.skip_collect:
        collect_results(args, logs_dir=logs_dir)

    print("\n" + "=" * 100)
    print("FINAL PIPELINE DONE")
    print("=" * 100)
    print(f"Results root: {args.results_root}")
    print(f"Logs:         {args.logs_dir}")
    print(f"Summary CSV:  {Path(args.results_root) / 'final_summary.csv'}")
    print(f"Summary TeX:  {Path(args.results_root) / 'final_summary.tex'}")


if __name__ == "__main__":
    main()
