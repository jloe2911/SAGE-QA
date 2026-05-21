import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


DATASETS = {
    "hotpotqa": {
        "display": "HotpotQA",
        "type": "text",
        "data_dir": "data/HotpotQA",
        "output_dir": "outputs/full_results/HotpotQA",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_hotpotqa_full",
        "gold_file": "data/HotpotQA/hotpot_test_subset_gold.json",
        "export_script": "evaluation/export_hotpot_predictions.py",
        "gold_export_script": "data_processing/export_hotpot_gold_subset.py",
        "gold_source_arg": "--parquet",
        "gold_source": "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
    },
    "2wiki": {
        "display": "2WikiMultiHopQA",
        "type": "text",
        "data_dir": "data/2WikiMultiHopQA",
        "output_dir": "outputs/full_results/2WikiMultiHopQA",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_2wiki_full",
        "gold_file": "data/2WikiMultiHopQA/2wiki_test_subset_gold.json",
        "export_script": "evaluation/export_2wiki_predictions.py",
        "gold_export_script": "data_processing/export_2wiki_gold_subset.py",
        "gold_source_arg": "--parquet",
        # Use validation unless you intentionally built test_subgraph_retrieval.jsonl from test.
        "gold_source": "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
    },
    "familyowl_1hop": {
        "display": "FamilyOWL_1hop",
        "type": "owl",
        "data_dir": "data/FamilyOWL_1hop",
        "output_dir": "outputs/full_results/FamilyOWL_1hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_familyowl_1hop_full",
    },
    "familyowl_2hop": {
        "display": "FamilyOWL_2hop",
        "type": "owl",
        "data_dir": "data/FamilyOWL_2hop",
        "output_dir": "outputs/full_results/FamilyOWL_2hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_familyowl_2hop_full",
    },
}


METHODS = {
    "lexical_subgraph": {
        "display": "Lexical Subgraph",
        "needs_training": False,
        "details_subdir": "lexical_subgraph",
        "score_mode": None,
        "valid_for": ["text", "owl"],
    },
    "gnn_neural": {
        "display": "GNN Subgraph",
        "needs_training": True,
        "details_subdir": "gnn_neural",
        "score_mode": "neural",
        "valid_for": ["text", "owl"],
    },
    "nesyqa_text_chain": {
        "display": "NeSyQA Text-Chain",
        "needs_training": True,
        "details_subdir": "gnn_nesyqa_text_chain",
        "score_mode": "nesyqa_text_chain",
        "valid_for": ["text"],
    },
    "nesyqa_compact": {
        "display": "NeSyQA Compact",
        "needs_training": True,
        "details_subdir": "gnn_nesyqa_compact",
        "score_mode": "nesyqa_compact",
        "valid_for": ["owl"],
    },
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def child_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    print("\n" + "=" * 100, flush=True)
    log("RUN:")
    print(" ".join(cmd), flush=True)
    print("=" * 100, flush=True)

    if dry_run:
        return

    start = time.time()
    result = subprocess.run(cmd, text=True, env=child_env())
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )

    log(f"Finished in {elapsed / 60:.1f} min")


def capture_cmd(cmd: List[str], dry_run: bool = False) -> str:
    print("\n" + "=" * 100, flush=True)
    log("CAPTURE:")
    print(" ".join(cmd), flush=True)
    print("=" * 100, flush=True)

    if dry_run:
        return "{}"

    start = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True, env=child_env())
    elapsed = time.time() - start

    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )

    log(f"Finished capture in {elapsed / 60:.1f} min")
    return result.stdout.strip()


def ensure_file(path: str | Path, description: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def parse_metrics_from_stdout(stdout: str) -> Dict[str, float]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            obj = ast.literal_eval(line)
            return {k: float(v) for k, v in obj.items()}
    raise ValueError(f"Could not parse metrics dict from stdout:\n{stdout}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_metrics_json(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # OWL evaluator writes {"metrics": {...}, ...}; text metrics are direct dicts.
    if isinstance(obj, dict) and "metrics" in obj:
        obj = obj["metrics"]

    return {k: float(v) for k, v in obj.items() if isinstance(v, (int, float))}


def safe_model_name(model: str) -> str:
    return model.replace(".", "_").replace("-", "_")


def default_methods_for_dataset(dataset_type: str) -> List[str]:
    if dataset_type == "text":
        return ["lexical_subgraph", "gnn_neural", "nesyqa_text_chain"]
    if dataset_type == "owl":
        return ["lexical_subgraph", "gnn_neural", "nesyqa_compact"]
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def train_gnn_if_needed(
    dataset_key: str,
    cfg: Dict,
    model_name: str,
    epochs: int,
    candidate_batch_size: int,
    dry_run: bool,
    force_train: bool,
) -> None:
    checkpoint = Path(cfg["checkpoint_dir"]) / "best_model.pt"

    if checkpoint.exists() and not force_train:
        log(f"Checkpoint already exists, skipping training: {checkpoint}")
        return

    train_path = str(Path(cfg["data_dir"]) / "train_subgraph_retrieval.jsonl")
    dev_path = str(Path(cfg["data_dir"]) / "dev_subgraph_retrieval.jsonl")

    ensure_file(train_path, f"{dataset_key} train data")
    ensure_file(dev_path, f"{dataset_key} dev data")

    cmd = [
        sys.executable,
        "training/train_gnn_subgraph_retriever.py",
        "--train-path",
        train_path,
        "--dev-path",
        dev_path,
        "--save-dir",
        cfg["checkpoint_dir"],
        "--model-name",
        model_name,
        "--epochs",
        str(epochs),
        "--freeze-encoder",
        "--candidate-batch-size",
        str(candidate_batch_size),
        "--score-mode",
        "neural",
        "--ranking-margin",
        "0.2",
        "--ranking-weight",
        "2.0",
        "--bce-weight",
        "0.2",
        "--listwise-weight",
        "0.0",
    ]

    log(f"Training GNN retriever for {cfg['display']}")
    run_cmd(cmd, dry_run=dry_run)


def run_lexical_details(
    dataset_key: str, cfg: Dict, dry_run: bool, resume: bool
) -> Path:
    test_path = str(Path(cfg["data_dir"]) / "test_subgraph_retrieval.jsonl")
    out_dir = Path(cfg["output_dir"]) / "lexical_subgraph"
    details_path = out_dir / "test_details.json"

    ensure_file(test_path, f"{dataset_key} test data")

    if resume and details_path.exists():
        log(f"Reusing lexical details: {details_path}")
        return details_path

    # Generic baseline script works for both text and OWL if it reads the standardized JSONL.
    cmd = [
        sys.executable,
        "evaluation/eval_lexical_hotpot_subgraph.py",
        "--test-path",
        test_path,
        "--output-dir",
        str(out_dir),
    ]

    log(f"Evaluating lexical retrieval for {cfg['display']}")
    run_cmd(cmd, dry_run=dry_run)

    return details_path


def run_gnn_details(
    dataset_key: str,
    cfg: Dict,
    method_key: str,
    method_cfg: Dict,
    candidate_batch_size: int,
    dry_run: bool,
    resume: bool,
) -> Path:
    train_path = str(Path(cfg["data_dir"]) / "train_subgraph_retrieval.jsonl")
    dev_path = str(Path(cfg["data_dir"]) / "dev_subgraph_retrieval.jsonl")
    test_path = str(Path(cfg["data_dir"]) / "test_subgraph_retrieval.jsonl")
    checkpoint = str(Path(cfg["checkpoint_dir"]) / "best_model.pt")
    out_dir = Path(cfg["output_dir"]) / method_cfg["details_subdir"]
    details_path = out_dir / "test_details.json"

    ensure_file(train_path, f"{dataset_key} train data")
    ensure_file(dev_path, f"{dataset_key} dev data")
    ensure_file(test_path, f"{dataset_key} test data")
    ensure_file(checkpoint, f"{dataset_key} checkpoint")

    if resume and details_path.exists():
        log(f"Reusing {method_cfg['display']} details: {details_path}")
        return details_path

    cmd = [
        sys.executable,
        "evaluation/eval_gnn_subgraph_retriever.py",
        "--train-path",
        train_path,
        "--dev-path",
        dev_path,
        "--test-path",
        test_path,
        "--checkpoint",
        checkpoint,
        "--score-mode",
        method_cfg["score_mode"],
        "--candidate-batch-size",
        str(candidate_batch_size),
        "--save-details",
        "--details-dir",
        str(out_dir),
    ]

    log(f"Evaluating {method_cfg['display']} retrieval for {cfg['display']}")
    run_cmd(cmd, dry_run=dry_run)

    return details_path


def export_gold_subset(
    dataset_key: str, cfg: Dict, details_path: Path, dry_run: bool, resume: bool
) -> Path | None:
    if cfg["type"] != "text":
        return None

    gold_file = Path(cfg["gold_file"])

    if resume and gold_file.exists():
        log(f"Reusing gold subset: {gold_file}")
        return gold_file

    cmd = [
        sys.executable,
        cfg["gold_export_script"],
        cfg["gold_source_arg"],
        cfg["gold_source"],
        "--details",
        str(details_path),
        "--output",
        str(gold_file),
    ]

    log(f"Exporting gold subset for {cfg['display']} from {cfg['gold_source']}")
    run_cmd(cmd, dry_run=dry_run)

    return gold_file


def run_llm_generation(
    cfg: Dict,
    details_path: Path,
    output_jsonl: Path,
    top_k: int,
    reader_model: str,
    dry_run: bool,
    skip_llm_if_exists: bool,
    resume_llm: bool,
    max_llm_examples: int,
) -> None:
    if output_jsonl.exists() and skip_llm_if_exists:
        log(f"LLM answers already exist, skipping: {output_jsonl}")
        return

    if not os.getenv("OPENAI_API_KEY") and not dry_run:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set in this terminal. "
            "Set it before running GPT generation."
        )

    if cfg["type"] == "text":
        script = "generation/generate_hotpot_answers_with_llm.py"
    elif cfg["type"] == "owl":
        script = "generation/generate_owl_answers_with_llm.py"
    else:
        raise ValueError(f"Unknown dataset type: {cfg['type']}")

    cmd = [
        sys.executable,
        script,
        "--details",
        str(details_path),
        "--output",
        str(output_jsonl),
        "--top-k",
        str(top_k),
        "--backend",
        "openai",
        "--model",
        reader_model,
    ]

    if max_llm_examples > 0:
        cmd.extend(["--max-examples", str(max_llm_examples)])

    if resume_llm and cfg["type"] == "owl":
        cmd.append("--resume")

    log(
        f"Generating LLM answers: dataset={cfg['display']}, details={details_path}, "
        f"top_k={top_k}, model={reader_model}"
    )
    run_cmd(cmd, dry_run=dry_run)


def export_text_predictions(
    dataset_key: str,
    cfg: Dict,
    details_path: Path,
    llm_answers_path: Path,
    pred_path: Path,
    top_k: int,
    dry_run: bool,
    resume: bool,
) -> None:
    if cfg["type"] != "text":
        return

    if resume and pred_path.exists():
        log(f"Reusing predictions: {pred_path}")
        return

    cmd = [
        sys.executable,
        cfg["export_script"],
        "--details",
        str(details_path),
        "--llm-answers",
        str(llm_answers_path),
        "--output",
        str(pred_path),
        "--top-k",
        str(top_k),
    ]

    log(f"Exporting text predictions to {pred_path}")
    run_cmd(cmd, dry_run=dry_run)


def evaluate_text_predictions(
    pred_path: Path,
    gold_path: Path,
    metrics_path: Path,
    dry_run: bool,
    resume: bool,
) -> Dict[str, float]:
    if resume and metrics_path.exists():
        log(f"Reusing metrics: {metrics_path}")
        return load_metrics_json(metrics_path)

    cmd = [
        sys.executable,
        "evaluation/hotpot_official_eval.py",
        str(pred_path),
        str(gold_path),
    ]

    log(f"Evaluating text predictions: {pred_path}")
    stdout = capture_cmd(cmd, dry_run=dry_run)

    if dry_run:
        return {}

    metrics = parse_metrics_from_stdout(stdout)
    write_json(metrics_path, metrics)
    return metrics


def evaluate_owl_predictions(
    details_path: Path,
    llm_answers_path: Path,
    metrics_path: Path,
    top_k: int,
    dry_run: bool,
    resume: bool,
) -> Dict[str, float]:
    if resume and metrics_path.exists():
        log(f"Reusing OWL metrics: {metrics_path}")
        return load_metrics_json(metrics_path)

    cmd = [
        sys.executable,
        "evaluation/evaluate_owl_qa_predictions.py",
        "--details",
        str(details_path),
        "--llm-answers",
        str(llm_answers_path),
        "--top-k",
        str(top_k),
        "--output",
        str(metrics_path),
    ]

    log(f"Evaluating OWL predictions: {details_path}")
    stdout = capture_cmd(cmd, dry_run=dry_run)

    if dry_run:
        return {}

    # Preferred: read output file, because evaluator writes nested result.
    if metrics_path.exists():
        return load_metrics_json(metrics_path)

    # Fallback: parse JSON printed to stdout.
    obj = json.loads(stdout)
    if "metrics" in obj:
        return {
            k: float(v)
            for k, v in obj["metrics"].items()
            if isinstance(v, (int, float))
        }
    return {k: float(v) for k, v in obj.items() if isinstance(v, (int, float))}


def count_llm_answers(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"llm_rows": 0, "llm_empty": 0, "llm_errors": 0}

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return {
        "llm_rows": len(rows),
        "llm_empty": sum(
            1 for r in rows if not str(r.get("predicted_answer", "")).strip()
        ),
        "llm_errors": sum(1 for r in rows if r.get("error")),
    }


def load_retrieval_metrics_from_test_metrics(
    method_out_dir: Path, top_k: int
) -> Dict[str, float]:
    path = method_out_dir / "test_metrics.json"

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        m = json.load(f)

    return {
        f"exact@{top_k}": m.get(f"exact_hit@{top_k}"),
        f"contained@{top_k}": m.get(f"contains_gold_hit@{top_k}"),
        f"support_set_f1@{top_k}": m.get(f"best_set_f1@{top_k}"),
    }


def append_result_row(
    results: List[Dict],
    dataset_key: str,
    method_key: str,
    metrics: Dict[str, float],
    llm_stats: Dict[str, int],
    top_k: int,
    reader_model: str,
) -> None:
    row = {
        "Dataset": DATASETS[dataset_key]["display"],
        "Dataset_Type": DATASETS[dataset_key]["type"],
        "Method": METHODS[method_key]["display"],
        "TopK": top_k,
        "Reader": reader_model,
        "Answer_EM": metrics.get("em"),
        "Answer_F1": metrics.get("f1"),
        "Answer_Prec": metrics.get("prec"),
        "Answer_Recall": metrics.get("recall"),
        "Support_EM": metrics.get("sp_em"),
        "Support_F1": metrics.get("sp_f1"),
        "Support_Prec": metrics.get("sp_prec"),
        "Support_Recall": metrics.get("sp_recall"),
        "Joint_EM": metrics.get("joint_em"),
        "Joint_F1": metrics.get("joint_f1"),
        "Joint_Prec": metrics.get("joint_prec"),
        "Joint_Recall": metrics.get("joint_recall"),
        "Exact_at_k": metrics.get(f"exact@{top_k}"),
        "Contained_at_k": metrics.get(f"contained@{top_k}"),
        "Support_Set_F1_at_k": metrics.get(f"support_set_f1@{top_k}"),
        **llm_stats,
    }
    results.append(row)


def save_results_csv(results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No results to save.")
        return

    fieldnames = list(results[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSaved results CSV to: {path}")


def validate_method_for_dataset(dataset_key: str, method_key: str) -> bool:
    dataset_type = DATASETS[dataset_key]["type"]
    return dataset_type in METHODS[method_key]["valid_for"]


def run_experiment(args) -> None:
    selected_datasets = args.datasets.split(",")

    for ds in selected_datasets:
        if ds not in DATASETS:
            raise ValueError(f"Unknown dataset: {ds}. Choices: {list(DATASETS)}")

    if args.methods == "auto":
        methods_by_dataset = {
            ds: default_methods_for_dataset(DATASETS[ds]["type"])
            for ds in selected_datasets
        }
    else:
        requested = args.methods.split(",")
        for method in requested:
            if method not in METHODS:
                raise ValueError(f"Unknown method: {method}. Choices: {list(METHODS)}")

        methods_by_dataset = {}
        for ds in selected_datasets:
            valid = [m for m in requested if validate_method_for_dataset(ds, m)]
            skipped = [m for m in requested if m not in valid]
            if skipped:
                log(
                    f"Skipping invalid methods for {DATASETS[ds]['display']}: {skipped}"
                )
            methods_by_dataset[ds] = valid

    log(f"Selected datasets: {', '.join(selected_datasets)}")
    log(f"Reader model: {args.reader_model}; top_k={args.top_k}")
    log(f"Encoder model: {args.encoder_model}; epochs={args.epochs}")

    results: List[Dict] = []

    for dataset_key in selected_datasets:
        cfg = DATASETS[dataset_key]
        selected_methods = methods_by_dataset[dataset_key]

        print("\n" + "#" * 100, flush=True)
        log(f"DATASET: {cfg['display']} ({cfg['type']})")
        log(f"Methods: {', '.join(selected_methods)}")
        print("#" * 100, flush=True)

        if any(METHODS[m]["needs_training"] for m in selected_methods):
            train_gnn_if_needed(
                dataset_key=dataset_key,
                cfg=cfg,
                model_name=args.encoder_model,
                epochs=args.epochs,
                candidate_batch_size=args.candidate_batch_size,
                dry_run=args.dry_run,
                force_train=args.force_train,
            )

        details_by_method: Dict[str, Path] = {}

        for method_key in selected_methods:
            method_cfg = METHODS[method_key]

            print("\n" + "-" * 100, flush=True)
            log(f"METHOD: {method_cfg['display']}")
            print("-" * 100, flush=True)

            if method_key == "lexical_subgraph":
                details_path = run_lexical_details(
                    dataset_key, cfg, dry_run=args.dry_run, resume=args.resume
                )
            else:
                details_path = run_gnn_details(
                    dataset_key=dataset_key,
                    cfg=cfg,
                    method_key=method_key,
                    method_cfg=method_cfg,
                    candidate_batch_size=args.candidate_batch_size,
                    dry_run=args.dry_run,
                    resume=args.resume,
                )

            details_by_method[method_key] = details_path

            if not args.dry_run:
                ensure_file(
                    str(details_path), f"{dataset_key}/{method_key} test details"
                )

        gold_path = None
        if cfg["type"] == "text":
            first_details = details_by_method[selected_methods[0]]
            gold_path = export_gold_subset(
                dataset_key,
                cfg,
                first_details,
                dry_run=args.dry_run,
                resume=args.resume,
            )
            if not args.dry_run:
                ensure_file(gold_path, f"{dataset_key} gold file")

        for method_key in selected_methods:
            method_cfg = METHODS[method_key]
            details_path = details_by_method[method_key]
            method_out_dir = Path(cfg["output_dir"]) / method_cfg["details_subdir"]

            model_tag = safe_model_name(args.reader_model)
            llm_answers_path = (
                method_out_dir / f"llm_answers_top{args.top_k}_{model_tag}.jsonl"
            )
            pred_path = (
                method_out_dir
                / f"{dataset_key}_predictions_top{args.top_k}_{model_tag}.json"
            )
            metrics_path = method_out_dir / f"metrics_top{args.top_k}_{model_tag}.json"

            if not args.skip_llm:
                run_llm_generation(
                    cfg=cfg,
                    details_path=details_path,
                    output_jsonl=llm_answers_path,
                    top_k=args.top_k,
                    reader_model=args.reader_model,
                    dry_run=args.dry_run,
                    skip_llm_if_exists=args.skip_llm_if_exists,
                    resume_llm=args.resume_llm,
                    max_llm_examples=args.max_llm_examples,
                )

            if cfg["type"] == "text":
                export_text_predictions(
                    dataset_key=dataset_key,
                    cfg=cfg,
                    details_path=details_path,
                    llm_answers_path=llm_answers_path,
                    pred_path=pred_path,
                    top_k=args.top_k,
                    dry_run=args.dry_run,
                    resume=args.resume,
                )

                metrics = evaluate_text_predictions(
                    pred_path=pred_path,
                    gold_path=gold_path,
                    metrics_path=metrics_path,
                    dry_run=args.dry_run,
                    resume=args.resume,
                )
            else:
                metrics = evaluate_owl_predictions(
                    details_path=details_path,
                    llm_answers_path=llm_answers_path,
                    metrics_path=metrics_path,
                    top_k=args.top_k,
                    dry_run=args.dry_run,
                    resume=args.resume,
                )

            llm_stats = count_llm_answers(llm_answers_path) if not args.dry_run else {}

            retrieval_metrics = load_retrieval_metrics_from_test_metrics(
                method_out_dir=method_out_dir,
                top_k=args.top_k,
            )

            metrics = {**metrics, **retrieval_metrics}

            append_result_row(
                results=results,
                dataset_key=dataset_key,
                method_key=method_key,
                metrics=metrics,
                llm_stats=llm_stats,
                top_k=args.top_k,
                reader_model=args.reader_model,
            )

    if not args.dry_run:
        save_results_csv(results, Path(args.output_csv))
        write_json(Path(args.output_json), results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,2wiki,familyowl_1hop,familyowl_2hop",
        help="Comma-separated: hotpotqa,2wiki,familyowl_1hop,familyowl_2hop",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="auto",
        help=(
            "Comma-separated method list or 'auto'. "
            "Text auto: lexical_subgraph,gnn_neural,nesyqa_text_chain. "
            "OWL auto: lexical_subgraph,gnn_neural,nesyqa_compact."
        ),
    )

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--reader-model", type=str, default="gpt-4.1-mini")
    parser.add_argument(
        "--encoder-model", type=str, default="google/bert_uncased_L-2_H-128_A-2"
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--candidate-batch-size", type=int, default=512)

    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-llm-if-exists", action="store_true")
    parser.add_argument("--resume-llm", action="store_true")
    parser.add_argument("--max-llm-examples", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing details, gold subsets, predictions, and metrics.",
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/full_results/full_pipeline_results.csv",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/full_results/full_pipeline_results.json",
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
