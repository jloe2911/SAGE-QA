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

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.llm_client import load_local_env, parse_model_ref


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
    "hotpotqa_openai": {
        "display": "HotpotQA_openai_gpt41mini",
        "type": "text",
        "data_dir": "data/HotpotQA_openai_gpt41mini",
        "output_dir": "outputs/full_results/HotpotQA_openai_gpt41mini",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_hotpotqa_openai_gpt41mini_full",
        "gold_file": "data/HotpotQA_openai_gpt41mini/hotpot_test_subset_gold.json",
        "export_script": "evaluation/export_hotpot_predictions.py",
        "gold_export_script": "data_processing/export_hotpot_gold_subset.py",
        "gold_source_arg": "--parquet",
        "gold_source": "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
    },
    "2wiki_openai": {
        "display": "2WikiMultiHopQA_openai_gpt41mini",
        "type": "text",
        "data_dir": "data/2WikiMultiHopQA_openai_gpt41mini",
        "output_dir": "outputs/full_results/2WikiMultiHopQA_openai_gpt41mini",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_2wiki_openai_gpt41mini_full",
        "gold_file": "data/2WikiMultiHopQA_openai_gpt41mini/2wiki_test_subset_gold.json",
        "export_script": "evaluation/export_2wiki_predictions.py",
        "gold_export_script": "data_processing/export_2wiki_gold_subset.py",
        "gold_source_arg": "--parquet",
        # Use validation unless you intentionally built test_subgraph_retrieval.jsonl from test.
        "gold_source": "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
    },
    "hotpotqa_contextkg_v2": {
        "display": "HotpotQA_contextkg_v2",
        "type": "text",
        "data_dir": "data/HotpotQA_contextkg_v2",
        "output_dir": "outputs/full_results/HotpotQA_contextkg_v2",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_hotpotqa_contextkg_v2_full",
        "gold_file": "data/HotpotQA_contextkg_v2/hotpot_test_subset_gold.json",
        "export_script": "evaluation/export_hotpot_predictions.py",
        "gold_export_script": "data_processing/export_hotpot_gold_subset.py",
        "gold_source_arg": "--parquet",
        "gold_source": "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
    },
    "2wiki_contextkg_v2": {
        "display": "2WikiMultiHopQA_contextkg_v2",
        "type": "text",
        "data_dir": "data/2WikiMultiHopQA_contextkg_v2",
        "output_dir": "outputs/full_results/2WikiMultiHopQA_contextkg_v2",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_2wiki_contextkg_v2_full",
        "gold_file": "data/2WikiMultiHopQA_contextkg_v2/2wiki_test_subset_gold.json",
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
    "owl2bench_1hop": {
        "display": "OWL2Bench_1hop",
        "type": "owl",
        "data_dir": "data/OWL2Bench_1hop",
        "output_dir": "outputs/full_results/OWL2Bench_1hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_owl2bench_1hop_full",
    },
    "owl2bench_2hop": {
        "display": "OWL2Bench_2hop",
        "type": "owl",
        "data_dir": "data/OWL2Bench_2hop",
        "output_dir": "outputs/full_results/OWL2Bench_2hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_owl2bench_2hop_full",
    },
    "pizza_100_1hop": {
        "display": "Pizza_100_1hop",
        "type": "owl",
        "data_dir": "data/pizza_100_1hop",
        "output_dir": "outputs/full_results/pizza_100_1hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_pizza_100_1hop_full",
    },
    "pizza_100_2hop": {
        "display": "Pizza_100_2hop",
        "type": "owl",
        "data_dir": "data/pizza_100_2hop",
        "output_dir": "outputs/full_results/pizza_100_2hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_pizza_100_2hop_full",
    },
    "pizza_250_1hop": {
        "display": "Pizza_250_1hop",
        "type": "owl",
        "data_dir": "data/pizza_250_1hop",
        "output_dir": "outputs/full_results/pizza_250_1hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_pizza_250_1hop_full",
    },
    "pizza_250_2hop": {
        "display": "Pizza_250_2hop",
        "type": "owl",
        "data_dir": "data/pizza_250_2hop",
        "output_dir": "outputs/full_results/pizza_250_2hop",
        "checkpoint_dir": "checkpoints/gnn_subgraph_ranker_pizza_250_2hop_full",
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
    "sageqa_text_chain": {
        "display": "sageqa Text-Chain",
        "needs_training": True,
        "details_subdir": "gnn_sageqa_text_chain",
        "score_mode": "sageqa_text_chain",
        "valid_for": ["text"],
    },
    "gnn_rag": {
        "display": "GNN-RAG",
        "needs_training": False,
        "details_subdir": "gnn_rag",
        "score_mode": None,
        "valid_for": ["text", "owl"],
        "gnn_rag": True,
    },
    "sageqa_compact": {
        "display": "sageqa Compact",
        "needs_training": True,
        "details_subdir": "gnn_sageqa_compact",
        "score_mode": "sageqa_compact",
        "valid_for": ["owl"],
    },
    "sageqa_proof": {
        "display": "sageqa Proof",
        "needs_training": True,
        "details_subdir": "gnn_sageqa_proof",
        "score_mode": "sageqa_proof",
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if path.suffix == ".jsonl" and path.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {path}")


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


def jsonl_example_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                example_id = row.get("example_id")
                if example_id:
                    ids.add(str(example_id))

    return ids


def details_example_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    return {str(row["example_id"]) for row in rows if row.get("example_id")}


def load_metrics_json(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # OWL evaluator writes {"metrics": {...}, ...}; text metrics are direct dicts.
    if isinstance(obj, dict) and "metrics" in obj:
        obj = obj["metrics"]

    return {k: float(v) for k, v in obj.items() if isinstance(v, (int, float))}


def safe_model_name(model: str) -> str:
    return (
        model.replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def has_key_for_model(model: str) -> bool:
    ref = parse_model_ref(model)
    if ref.provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    if ref.provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def missing_key_message(model: str) -> str:
    ref = parse_model_ref(model)
    if ref.provider == "openrouter":
        return "Model provider is openrouter, but OPENROUTER_API_KEY is not set."
    if ref.provider == "openai":
        return "Model provider is openai, but OPENAI_API_KEY is not set."
    return "Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set."


def default_methods_for_dataset(dataset_type: str) -> List[str]:
    if dataset_type == "text":
        return ["lexical_subgraph", "gnn_neural", "sageqa_text_chain", "gnn_rag"]
    if dataset_type == "owl":
        return [
            "lexical_subgraph",
            "gnn_neural",
            "sageqa_proof",
            "gnn_rag",
        ]
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def train_gnn_if_needed(
    dataset_key: str,
    cfg: Dict,
    model_name: str,
    epochs: int,
    candidate_batch_size: int,
    dry_run: bool,
    force_train: bool,
    freeze_encoder: bool,
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
    if freeze_encoder:
        cmd.append("--freeze-encoder")

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
    if not dry_run:
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
    fallback_top_k: int,
    reader_model: str,
    dry_run: bool,
    skip_llm_if_exists: bool,
    resume_llm: bool,
    max_llm_examples: int,
    answer_only_paths: List[Path] | None = None,
) -> None:
    existing_answer_only = [
        str(path) for path in answer_only_paths or [] if path.exists()
    ]

    if output_jsonl.exists() and skip_llm_if_exists:
        needed_ids = details_example_ids(details_path)
        for path in answer_only_paths or []:
            needed_ids.update(jsonl_example_ids(path))

        if needed_ids and not needed_ids.issubset(jsonl_example_ids(output_jsonl)):
            log(
                f"LLM answers exist but are missing current examples; "
                f"continuing generation: {output_jsonl}"
            )
            resume_llm = True
        else:
            log(f"LLM answers already exist, skipping: {output_jsonl}")
            return

    if not has_key_for_model(reader_model) and not dry_run:
        raise EnvironmentError(missing_key_message(reader_model))

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

    if cfg["type"] == "text" and fallback_top_k and fallback_top_k > top_k:
        cmd.extend(["--fallback-top-k", str(fallback_top_k)])

    if max_llm_examples > 0:
        cmd.extend(["--max-examples", str(max_llm_examples)])

    if existing_answer_only:
        cmd.extend(["--answer-only", *existing_answer_only])

    if resume_llm and cfg["type"] == "owl":
        cmd.append("--resume")

    log(
        f"Generating LLM answers: dataset={cfg['display']}, details={details_path}, "
        f"top_k={top_k}, fallback_top_k={fallback_top_k}, model={reader_model}"
    )
    run_cmd(cmd, dry_run=dry_run)


def run_gnn_rag_generation(
    dataset_key: str,
    cfg: Dict,
    details_path: Path,
    output_jsonl: Path,
    top_k: int,
    reader_model: str,
    retriever_epochs: int,
    dry_run: bool,
    resume_llm: bool,
    max_llm_examples: int,
    text_max_candidates: int = 5,
    answer_only_paths: List[Path] | None = None,
) -> None:
    if not has_key_for_model(reader_model) and not dry_run:
        raise EnvironmentError(missing_key_message(reader_model))

    repo_root = Path.cwd()
    gnn_rag_repo = repo_root / "third_party" / "GNN-RAG"
    gnn_root = gnn_rag_repo / "gnn"
    llm_root = gnn_rag_repo / "llm"
    ensure_file(gnn_root / "main.py", "GNN-RAG retriever")
    ensure_file(
        llm_root / "src" / "qa_prediction" / "predict_answer.py", "GNN-RAG predictor"
    )

    adapter_dataset = f"sageqa-{dataset_key}"
    gnn_data_dir = gnn_root / "data" / adapter_dataset
    gnn_checkpoint_dir = gnn_root / "checkpoint" / adapter_dataset
    gnn_experiment = f"rearev_lstm_{adapter_dataset}"
    gnn_info_path = gnn_checkpoint_dir / f"{gnn_experiment}_test.info"
    llm_data_dir = llm_root / "data" / adapter_dataset
    llm_gnn_dir = llm_root / "results" / "gnn" / adapter_dataset / "rearev-lstm"
    predict_root = llm_root / "results" / "sageqa-GNN-RAG"
    raw_predictions = (
        predict_root
        / adapter_dataset
        / safe_model_name(reader_model)
        / "test"
        / "no_rule"
        / "False"
        / "predictions.jsonl"
    )

    if cfg["type"] == "text":
        prepare_cmd = [
            sys.executable,
            "data_processing/prepare_text_gnn_rag.py",
            "--data-dir",
            cfg["data_dir"],
            "--gnn-data-dir",
            str(gnn_data_dir),
            "--details-output",
            str(details_path),
            "--max-candidates",
            str(max(1, text_max_candidates)),
        ]
        log(f"Preparing text data for upstream GNN-RAG: dataset={cfg['display']}")
    else:
        prepare_cmd = [
            sys.executable,
            "data_processing/prepare_familyowl_gnn_rag.py",
            "--data-dir",
            cfg["data_dir"],
            "--gnn-data-dir",
            str(gnn_data_dir),
            "--details-output",
            str(details_path),
        ]
        log(f"Preparing FamilyOWL data for upstream GNN-RAG: dataset={cfg['display']}")

    run_cmd(prepare_cmd, dry_run=dry_run)

    if not dry_run:
        gnn_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        str(gnn_root / "main.py"),
        "ReaRev",
        "--entity_dim",
        "50",
        "--num_epoch",
        str(max(1, retriever_epochs)),
        "--batch_size",
        "8",
        "--eval_every",
        "1",
        "--data_folder",
        str(gnn_data_dir) + os.sep,
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
        str(gnn_checkpoint_dir),
        "--experiment_name",
        gnn_experiment,
        "--name",
        adapter_dataset,
    ]
    log(f"Training upstream GNN-RAG retriever: dataset={cfg['display']}")
    run_cmd(train_cmd, dry_run=dry_run)

    eval_cmd = [
        sys.executable,
        str(gnn_root / "main.py"),
        "ReaRev",
        "--entity_dim",
        "50",
        "--data_folder",
        str(gnn_data_dir) + os.sep,
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
        str(gnn_checkpoint_dir),
        "--experiment_name",
        gnn_experiment,
        "--load_experiment",
        f"{gnn_experiment}-final.ckpt",
        "--is_eval",
        "--name",
        adapter_dataset,
    ]
    log(f"Evaluating upstream GNN-RAG retriever: dataset={cfg['display']}")
    run_cmd(eval_cmd, dry_run=dry_run)

    if not dry_run:
        ensure_file(gnn_info_path, "GNN-RAG retriever test.info")
        llm_data_dir.mkdir(parents=True, exist_ok=True)
        llm_gnn_dir.mkdir(parents=True, exist_ok=True)
        for split in ("test",):
            (llm_data_dir / f"{split}.json").write_text(
                (gnn_data_dir / f"{split}.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (llm_gnn_dir / f"{split}.json").write_text(
                (gnn_data_dir / f"{split}.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        (llm_gnn_dir / "test.info").write_text(
            gnn_info_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    log(
        f"Running upstream GNN-RAG predictor: dataset={cfg['display']}, "
        f"model={reader_model}"
    )
    predict_cmd = [
        sys.executable,
        "src/qa_prediction/predict_answer.py",
        "--data_path",
        "data",
        "--d",
        adapter_dataset,
        "--split",
        "test",
        "--model_name",
        reader_model,
        "--prompt_path",
        "prompts/llama2_predict.txt",
        "--rule_path_g1",
        str(Path("results") / "gnn" / adapter_dataset / "rearev-lstm" / "test.info"),
        "--rule_path_g2",
        "None",
        "--predict_path",
        str(Path("results") / "sageqa-GNN-RAG"),
        "-n",
        "1",
    ]

    if not resume_llm:
        predict_cmd.append("--force")

    print("\n" + "=" * 100, flush=True)
    log("RUN:")
    print(" ".join(predict_cmd), flush=True)
    print("=" * 100, flush=True)

    if not dry_run:
        env = child_env()
        env["PYTHONPATH"] = str(llm_root / "src")
        env["HF_HOME"] = str(repo_root / ".cache" / "hf")
        env["HF_DATASETS_CACHE"] = str(repo_root / ".cache" / "hf" / "datasets")
        Path(env["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(predict_cmd, cwd=llm_root, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"GNN-RAG predictor failed for {dataset_key} with exit code {result.returncode}"
            )
        ensure_file(raw_predictions, "GNN-RAG raw predictions")

    convert_cmd = [
        sys.executable,
        "evaluation/convert_gnn_rag_predictions.py",
        "--input",
        str(raw_predictions),
        "--output",
        str(output_jsonl),
    ]
    log(f"Converting GNN-RAG predictions: {raw_predictions}")
    run_cmd(convert_cmd, dry_run=dry_run)


def export_text_predictions(
    dataset_key: str,
    cfg: Dict,
    details_path: Path,
    llm_answers_path: Path,
    pred_path: Path,
    top_k: int,
    support_top_k: int,
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
        str(support_top_k),
    ]

    log(
        f"Exporting text predictions to {pred_path} "
        f"(reader_top_k={top_k}, support_top_k={support_top_k})"
    )
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
    answer_only_paths: List[Path] | None = None,
) -> Dict[str, float]:
    if resume and metrics_path.exists():
        metrics = load_metrics_json(metrics_path)
        expected_answer_only = sum(
            len(jsonl_example_ids(path))
            for path in answer_only_paths or []
            if path.exists()
        )
        saved_answer_only = int(metrics.get("answer_only_examples", 0))

        if expected_answer_only and saved_answer_only != expected_answer_only:
            log(
                f"OWL metrics exist but answer-only count changed "
                f"({saved_answer_only} -> {expected_answer_only}); recomputing."
            )
        else:
            log(f"Reusing OWL metrics: {metrics_path}")
            return metrics

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

    existing_answer_only = [
        str(path) for path in answer_only_paths or [] if path.exists()
    ]
    if existing_answer_only:
        cmd.extend(["--answer-only", *existing_answer_only])

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


def count_prediction_answers(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"llm_rows": 0, "llm_empty": 0, "llm_errors": 0}

    with path.open("r", encoding="utf-8") as f:
        pred = json.load(f)

    answers = pred.get("answer", {}) if isinstance(pred, dict) else {}
    return {
        "llm_rows": len(answers),
        "llm_empty": sum(1 for answer in answers.values() if not str(answer).strip()),
        "llm_errors": 0,
    }


def load_retrieval_metrics(method_out_dir: Path, top_k: int) -> Dict[str, float]:
    metrics_path = method_out_dir / "test_metrics.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            m = json.load(f)

        exact = m.get(f"exact_hit@{top_k}", m.get(f"exact@{top_k}"))
        contained = m.get(
            f"contains_gold_hit@{top_k}",
            m.get(f"gold_contained@{top_k}", m.get(f"contains@{top_k}")),
        )
        support_f1 = m.get(f"set_f1@{top_k}", m.get(f"support_f1@{top_k}"))
        support_precision = m.get(
            f"precision@{top_k}", m.get(f"support_precision@{top_k}")
        )
        support_recall = m.get(f"recall@{top_k}", m.get(f"support_recall@{top_k}"))

        if exact is not None and contained is not None and support_f1 is not None:
            return {
                f"exact@{top_k}": exact,
                f"contained@{top_k}": contained,
                f"support_set_f1@{top_k}": support_f1,
                f"retrieval_precision@{top_k}": support_precision,
                f"retrieval_recall@{top_k}": support_recall,
                f"retrieval_f1@{top_k}": support_f1,
            }

    details_path = method_out_dir / "test_details.json"

    if not details_path.exists():
        return {}

    with details_path.open("r", encoding="utf-8") as f:
        details = json.load(f)

    if not details:
        return {}

    exact_scores = []
    contained_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for ex in details:
        top = ex.get("top5", [])[:top_k]
        gold_sets = ex.get("gold_explanations", []) or []
        gold_support = ex.get("gold_support_units", []) or []
        if not gold_sets and gold_support:
            gold_sets = [gold_support]

        union_units = []
        seen = set()
        for cand in top:
            for unit in cand.get("subgraph_units", []) or []:
                if unit not in seen:
                    seen.add(unit)
                    union_units.append(unit)

        union_set = set(union_units)
        best_union = {
            "exact": 0.0,
            "contained": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        for gold in gold_sets:
            gold_set = set(gold)
            if not gold_set:
                continue

            inter = len(union_set & gold_set)
            precision = inter / max(len(union_set), 1)
            recall = inter / len(gold_set)
            f1 = (
                0.0
                if precision + recall == 0
                else (2 * precision * recall / (precision + recall))
            )

            if union_set == gold_set:
                best_union["exact"] = 1.0
            if gold_set.issubset(union_set):
                best_union["contained"] = 1.0
            if f1 > best_union["f1"]:
                best_union["precision"] = precision
                best_union["recall"] = recall
                best_union["f1"] = f1

        exact_scores.append(best_union["exact"])
        contained_scores.append(best_union["contained"])
        f1_scores.append(best_union["f1"])
        precision_scores.append(best_union["precision"])
        recall_scores.append(best_union["recall"])

    return {
        f"exact@{top_k}": sum(exact_scores) / len(exact_scores),
        f"contained@{top_k}": sum(contained_scores) / len(contained_scores),
        f"support_set_f1@{top_k}": sum(f1_scores) / len(f1_scores),
        f"retrieval_precision@{top_k}": sum(precision_scores) / len(precision_scores),
        f"retrieval_recall@{top_k}": sum(recall_scores) / len(recall_scores),
        f"retrieval_f1@{top_k}": sum(f1_scores) / len(f1_scores),
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
        "Answer_Examples": metrics.get("answer_examples"),
        "Support_EM": metrics.get("sp_em"),
        "Support_F1": metrics.get("sp_f1"),
        "Support_Prec": metrics.get("sp_prec"),
        "Support_Recall": metrics.get("sp_recall"),
        "Support_Examples": metrics.get("support_examples"),
        "Answer_Only_Examples": metrics.get("answer_only_examples"),
        "Joint_EM": metrics.get("joint_em"),
        "Joint_F1": metrics.get("joint_f1"),
        "Joint_Prec": metrics.get("joint_prec"),
        "Joint_Recall": metrics.get("joint_recall"),
        "Retrieval_Precision_at_k": metrics.get(f"retrieval_precision@{top_k}"),
        "Retrieval_Recall_at_k": metrics.get(f"retrieval_recall@{top_k}"),
        "Retrieval_F1_at_k": metrics.get(f"retrieval_f1@{top_k}"),
        "Exact_Evidence_Set_at_k": metrics.get(f"exact@{top_k}"),
        "Complete_Evidence_Recall_at_k": metrics.get(f"contained@{top_k}"),
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
                freeze_encoder=not args.fine_tune_encoder,
            )

        details_by_method: Dict[str, Path] = {}

        for method_key in selected_methods:
            method_cfg = METHODS[method_key]

            print("\n" + "-" * 100, flush=True)
            log(f"METHOD: {method_cfg['display']}")
            print("-" * 100, flush=True)

            if method_cfg.get("gnn_rag"):
                log(
                    "GNN-RAG uses the upstream ReaRev retriever on adapted "
                    "KGQA-style data; skipping local sageqa retrieval details."
                )
                continue
            elif method_key == "lexical_subgraph":
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
            if details_by_method:
                first_details = next(iter(details_by_method.values()))
                gold_path = export_gold_subset(
                    dataset_key,
                    cfg,
                    first_details,
                    dry_run=args.dry_run,
                    resume=args.resume,
                )
            else:
                gold_path = cfg["gold_file"]
            if not args.dry_run:
                ensure_file(gold_path, f"{dataset_key} gold file")

        for method_key in selected_methods:
            method_cfg = METHODS[method_key]
            method_out_dir = Path(cfg["output_dir"]) / method_cfg["details_subdir"]

            model_tag = safe_model_name(args.reader_model)
            reader_top_k = args.reader_top_k or args.top_k
            reader_fallback_top_k = args.reader_fallback_top_k or 0
            support_top_k = args.support_top_k or args.top_k
            reader_tag = (
                f"_reader_top{reader_top_k}" if reader_top_k != args.top_k else ""
            )
            fallback_tag = (
                f"_fallback_top{reader_fallback_top_k}" if reader_fallback_top_k else ""
            )
            support_tag = (
                f"_support_top{support_top_k}" if support_top_k != args.top_k else ""
            )
            llm_answers_path = (
                method_out_dir
                / f"llm_answers_top{args.top_k}{reader_tag}{fallback_tag}_{model_tag}.jsonl"
            )
            pred_path = (
                method_out_dir
                / f"{dataset_key}_predictions_top{args.top_k}{reader_tag}{fallback_tag}{support_tag}_{model_tag}.json"
            )
            metrics_path = (
                method_out_dir
                / f"metrics_top{args.top_k}{reader_tag}{fallback_tag}{support_tag}_{model_tag}.json"
            )
            answer_only_paths: List[Path] = []
            if cfg["type"] == "owl":
                answer_only_path = (
                    Path(cfg["data_dir"]) / "test_answer_only_no_explanation.jsonl"
                )
                if answer_only_path.exists():
                    answer_only_paths.append(answer_only_path)

            if method_cfg.get("gnn_rag"):
                details_path = method_out_dir / "test_details.json"
                llm_answers_path = (
                    method_out_dir
                    / f"llm_answers_gnn_rag_top{args.top_k}{reader_tag}_{model_tag}.jsonl"
                )
                pred_path = (
                    method_out_dir
                    / f"{dataset_key}_predictions_gnn_rag_top{args.top_k}{reader_tag}{fallback_tag}{support_tag}_{model_tag}.json"
                )
                metrics_path = (
                    method_out_dir
                    / f"metrics_gnn_rag_top{args.top_k}{reader_tag}{fallback_tag}{support_tag}_{model_tag}.json"
                )

                if not args.skip_llm:
                    run_gnn_rag_generation(
                        dataset_key=dataset_key,
                        cfg=cfg,
                        details_path=details_path,
                        output_jsonl=llm_answers_path,
                        top_k=reader_top_k,
                        reader_model=args.reader_model,
                        retriever_epochs=args.epochs,
                        dry_run=args.dry_run,
                        resume_llm=args.resume_llm,
                        max_llm_examples=args.max_llm_examples,
                        text_max_candidates=args.gnn_rag_text_max_candidates,
                        answer_only_paths=answer_only_paths,
                    )

                if cfg["type"] == "text":
                    export_text_predictions(
                        dataset_key=dataset_key,
                        cfg=cfg,
                        details_path=details_path,
                        llm_answers_path=llm_answers_path,
                        pred_path=pred_path,
                        top_k=args.top_k,
                        support_top_k=support_top_k,
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
                        resume=False,
                        answer_only_paths=[],
                    )

                llm_stats = (
                    count_llm_answers(llm_answers_path) if not args.dry_run else {}
                )
                retrieval_metrics = load_retrieval_metrics(
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
                continue

            details_path = details_by_method[method_key]

            if not args.skip_llm:
                run_llm_generation(
                    cfg=cfg,
                    details_path=details_path,
                    output_jsonl=llm_answers_path,
                    top_k=reader_top_k,
                    fallback_top_k=reader_fallback_top_k,
                    reader_model=args.reader_model,
                    dry_run=args.dry_run,
                    skip_llm_if_exists=args.skip_llm_if_exists,
                    resume_llm=args.resume_llm,
                    max_llm_examples=args.max_llm_examples,
                    answer_only_paths=answer_only_paths,
                )

            if cfg["type"] == "text":
                export_text_predictions(
                    dataset_key=dataset_key,
                    cfg=cfg,
                    details_path=details_path,
                    llm_answers_path=llm_answers_path,
                    pred_path=pred_path,
                    top_k=args.top_k,
                    support_top_k=support_top_k,
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
                    answer_only_paths=answer_only_paths,
                )

            llm_stats = count_llm_answers(llm_answers_path) if not args.dry_run else {}

            retrieval_metrics = load_retrieval_metrics(
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
    load_local_env()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        default=(
            "hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,"
            "pizza_100_1hop,pizza_100_2hop,pizza_250_1hop,pizza_250_2hop"
        ),
        help=(
            "Comma-separated dataset keys. Choices: "
            "hotpotqa,2wiki,familyowl_1hop,familyowl_2hop,"
            "owl2bench_1hop,owl2bench_2hop,"
            "pizza_100_1hop,pizza_100_2hop,pizza_250_1hop,pizza_250_2hop"
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="auto",
        help=(
            "Comma-separated method list or 'auto'. "
            "Text auto: lexical_subgraph,gnn_neural,sageqa_text_chain,gnn_rag. "
            "OWL auto: lexical_subgraph,gnn_neural,sageqa_proof,gnn_rag."
        ),
    )

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--reader-top-k",
        type=int,
        default=0,
        help=(
            "Number of ranked candidates passed to the LLM reader. "
            "Default 0 means use --top-k. Use 1 when top-k union evidence is noisy."
        ),
    )
    parser.add_argument(
        "--reader-fallback-top-k",
        type=int,
        default=0,
        help=(
            "For text datasets, retry answer generation with this larger top-k "
            "only when the first reader call returns unknown/empty."
        ),
    )
    parser.add_argument(
        "--support-top-k",
        type=int,
        default=0,
        help=(
            "Number of ranked candidates to union for text support export. "
            "Default 0 means use --top-k. Use 1 to reduce noisy support "
            "while keeping reader generation at --top-k."
        ),
    )
    parser.add_argument("--reader-model", type=str, default="gpt-4.1-mini")
    parser.add_argument(
        "--encoder-model", type=str, default="google/bert_uncased_L-2_H-128_A-2"
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--candidate-batch-size", type=int, default=512)

    parser.add_argument(
        "--fine-tune-encoder",
        action="store_true",
        help="Fine-tune the text encoder during GNN training instead of freezing it.",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-llm-if-exists", action="store_true")
    parser.add_argument("--resume-llm", action="store_true")
    parser.add_argument("--max-llm-examples", type=int, default=0)
    parser.add_argument("--gnn-rag-text-max-candidates", type=int, default=5)
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
