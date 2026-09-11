"""One-shot DEV-only question-candidate cross-encoder ranking experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.retrieval_contracts import cap_inference_candidate_rows
from evaluation.run_production_dev_k_sensitivity import (
    best_evidence_scores,
    sha256,
)
from training.train_gnn_subgraph_retriever import (
    compute_adjusted_score,
    gold_explanations_from_row,
    ranking_target,
)


SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MODEL_PATH_ENV = "SAGEQA_DISTILBERT_PATH"
MAX_LENGTH = 256
MARGIN = 0.2
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PAIR_BATCH_SIZE = 16
PREDICT_BATCH_SIZE = 128
MAX_PAIRS_PER_QUESTION = 8
MAX_INFERENCE_CANDIDATES = 320

DATASETS = (
    ("HotpotQA", "text", "sageqa_text_chain"),
    ("2WikiMultiHopQA", "text", "sageqa_text_chain"),
    ("FamilyOWL_1hop", "ontology", "sageqa_proof"),
    ("FamilyOWL_2hop", "ontology", "sageqa_proof"),
    ("pizza_100_1hop", "ontology", "sageqa_proof"),
    ("pizza_100_2hop", "ontology", "sageqa_proof"),
    ("pizza_250_1hop", "ontology", "sageqa_proof"),
    ("pizza_250_2hop", "ontology", "sageqa_proof"),
    ("OWL2Bench_1hop", "ontology", "sageqa_proof"),
    ("OWL2Bench_2hop", "ontology", "sageqa_proof"),
)

DEFAULT_DATA_ROOT = ROOT / "data" / "production_generator_d_v1"
DEFAULT_OLD_RANKINGS = (
    ROOT
    / "outputs"
    / "development_runs"
    / "production_generator_d_v2_hard_pair_k_sensitivity_merged"
    / "per_example_rankings.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "development_runs" / "question_candidate_cross_encoder_v1"


def validate_frozen_model_snapshot(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{MODEL_PATH_ENV} is not a directory: {path}")
    required = (path / "config.json", path / "tokenizer_config.json")
    missing = [item.name for item in required if not item.is_file()]
    if not any((path / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
        missing.append("tokenizer.json or vocab.txt")
    if not any(
        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        missing.append("model.safetensors or pytorch_model.bin")
    if missing:
        raise FileNotFoundError(f"Incomplete frozen DistilBERT snapshot at {path}: {missing}")


def pretrained_model_source() -> tuple[str, dict[str, str]]:
    configured_path = os.environ.get(MODEL_PATH_ENV, "").strip()
    if configured_path:
        snapshot_path = Path(configured_path).expanduser().resolve()
        validate_frozen_model_snapshot(snapshot_path)
        return str(snapshot_path), {}
    return MODEL_NAME, {"revision": MODEL_REVISION}


def load_pretrained_components() -> tuple[Any, Any]:
    source, revision_kwargs = pretrained_model_source()
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, **revision_kwargs
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=1, local_files_only=True, **revision_kwargs
    )
    return tokenizer, model


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def grouped_jsonl(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    seen: set[str] = set()
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id:
                raise ValueError(f"{path}:{line_number}: missing example_id")
            if current_id is None:
                current_id = example_id
            if example_id != current_id:
                if example_id in seen:
                    raise ValueError(f"{path}: non-contiguous example {example_id!r}")
                seen.add(current_id)
                yield current_id, current
                current_id, current = example_id, []
            current.append(row)
    if current_id is not None:
        yield current_id, current


def parse_text_unit(unit: str) -> tuple[str, str, str] | None:
    parts = str(unit).split("::", 3)
    if len(parts) == 4 and parts[0] == "SENT":
        return parts[1], parts[2], parts[3]
    return None


def serialize_candidate(question: str, units: Sequence[str], domain: str) -> str:
    evidence: list[str] = []
    for position, unit in enumerate(units, 1):
        parsed = parse_text_unit(str(unit)) if domain == "text" else None
        if parsed is None:
            evidence.append(f"[{position}] {unit}")
        else:
            title, sentence_index, sentence = parsed
            evidence.append(
                f"[{position}] Title: {title}\nSentence index: {sentence_index}\nText: {sentence}"
            )
    return f"Question:\n{question}\n\nCandidate evidence:\n" + "\n".join(evidence)


def candidate_identity(row: Mapping[str, Any]) -> str:
    units = [str(unit) for unit in row.get("subgraph_units", []) or []]
    payload = json.dumps(units, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_row_order(example_id: str, row: Mapping[str, Any]) -> str:
    payload = f"{SEED}\0{example_id}\0{candidate_identity(row)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_training_pairs(
    example_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_pairs: int = MAX_PAIRS_PER_QUESTION,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    by_target: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[round(float(ranking_target(dict(row))), 8)].append(row)
    targets = sorted(by_target, reverse=True)
    if len(targets) < 2:
        return []
    for target in targets:
        by_target[target].sort(key=lambda row: stable_row_order(example_id, row))

    target_pairs: list[tuple[float, float]] = []
    for index in range(len(targets) - 1):
        target_pairs.append((targets[index], targets[index + 1]))
    extreme = (targets[0], targets[-1])
    if extreme not in target_pairs:
        target_pairs.append(extreme)
    for high_index, high in enumerate(targets):
        for low in targets[high_index + 1 :]:
            if (high, low) not in target_pairs:
                target_pairs.append((high, low))

    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    signatures: set[tuple[str, str]] = set()
    round_index = 0
    while len(selected) < max_pairs and round_index < max_pairs * 4:
        added = False
        for high_target, low_target in target_pairs:
            high_rows, low_rows = by_target[high_target], by_target[low_target]
            high = high_rows[round_index % len(high_rows)]
            low = low_rows[(round_index * 3 + len(selected)) % len(low_rows)]
            signature = (candidate_identity(high), candidate_identity(low))
            if signature in signatures:
                continue
            signatures.add(signature)
            selected.append((high, low))
            added = True
            if len(selected) == max_pairs:
                break
        if not added and round_index >= max(len(rows) for rows in by_target.values()):
            break
        round_index += 1
    return selected


def input_paths(data_root: Path, split: str) -> list[tuple[str, str, str, Path]]:
    if split not in {"train", "dev"}:
        raise ValueError("This experiment permits only TRAIN and DEV inputs")
    return [
        (dataset, domain, score_mode, data_root / dataset / f"{split}_subgraph_retrieval.jsonl")
        for dataset, domain, score_mode in DATASETS
    ]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    files = input_paths(args.data_root, "train") + input_paths(args.data_root, "dev")
    missing = [str(path) for _, _, _, path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")
    if not args.old_rankings.is_file():
        raise FileNotFoundError(args.old_rankings)
    if not (Path(__file__).parent / "PROTOCOL.md").is_file():
        raise FileNotFoundError("Predeclared PROTOCOL.md is missing")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer, model = load_pretrained_components()
    del tokenizer, model

    manifest = {
        "schema_version": "question_candidate_cross_encoder_dev_v1_preflight",
        "status": "predeclared_protocol_validated_before_training",
        "scope": {"splits": ["train", "dev"], "test_accessed": False},
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "configuration": configuration(),
        "protocol_sha256": sha256(Path(__file__).parent / "PROTOCOL.md"),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for *_, path in files},
        "old_rankings": {
            "path": str(args.old_rankings.relative_to(ROOT)),
            "sha256": sha256(args.old_rankings),
        },
    }
    dump_json(args.output_dir / "preflight_manifest.json", manifest)
    return manifest


def configuration() -> dict[str, Any]:
    return {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "loss": "mean(max(0, 0.2 - score_higher + score_lower))",
        "margin": MARGIN,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "pair_batch_size": PAIR_BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "max_pairs_per_question": MAX_PAIRS_PER_QUESTION,
        "max_inference_candidates": MAX_INFERENCE_CANDIDATES,
    }


def build_training_pairs(args: argparse.Namespace) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    pairs: list[tuple[str, str]] = []
    by_dataset: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for dataset, domain, _, path in input_paths(args.data_root, "train"):
        questions = rankable = dataset_pairs = 0
        for example_id, rows in grouped_jsonl(path):
            questions += 1
            selected = select_training_pairs(example_id, rows)
            if selected:
                rankable += 1
            question = str(rows[0].get("question") or rows[0].get("abs_question") or example_id)
            for higher, lower in selected:
                pairs.append(
                    (
                        serialize_candidate(question, higher.get("subgraph_units", []), domain),
                        serialize_candidate(question, lower.get("subgraph_units", []), domain),
                    )
                )
                dataset_pairs += 1
        by_dataset[dataset] = {
            "questions": questions,
            "rankable_questions": rankable,
            "pairs": dataset_pairs,
        }
    random.Random(SEED).shuffle(pairs)
    return pairs, {
        "pairs": len(pairs),
        "by_dataset": by_dataset,
        "construction_seconds": time.perf_counter() - started,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    pairs, pair_stats = build_training_pairs(args)
    if not pairs:
        raise ValueError("No rankable TRAIN pairs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_pretrained_components()
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    started = time.perf_counter()
    loss_sum = 0.0
    steps = math.ceil(len(pairs) / PAIR_BATCH_SIZE)
    for step, offset in enumerate(range(0, len(pairs), PAIR_BATCH_SIZE), 1):
        batch = pairs[offset : offset + PAIR_BATCH_SIZE]
        texts = [higher for higher, _ in batch] + [lower for _, lower in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            scores = model(**encoded).logits.squeeze(-1)
            split = len(batch)
            loss = torch.relu(MARGIN - scores[:split] + scores[split:]).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach().cpu())
        if step == 1 or step % 250 == 0 or step == steps:
            print(f"train step {step}/{steps} loss={float(loss):.6f}", flush=True)

    checkpoint_dir = args.output_dir / "checkpoint_final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    report = {
        "schema_version": "question_candidate_cross_encoder_dev_v1_training",
        "status": "complete",
        "configuration": configuration(),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "pair_statistics": pair_stats,
        "optimizer_steps": steps,
        "mean_training_loss": loss_sum / steps,
        "training_seconds": time.perf_counter() - started,
        "checkpoint_files_sha256": {
            path.name: sha256(path) for path in sorted(checkpoint_dir.iterdir()) if path.is_file()
        },
    }
    dump_json(args.output_dir / "training_report.json", report)
    return report


def safe_inference_row(row: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "example_id",
        "dataset",
        "source_name",
        "question",
        "abs_question",
        "sparql_query",
        "subgraph_units",
        "subgraph_size",
        "graph_context_units",
        "kg_evidence_units",
        "symbolic_features",
        "task_type",
        "answer_type",
    }
    return {key: row[key] for key in allowed if key in row}


@torch.inference_mode()
def score_texts(
    texts: Sequence[str], tokenizer: Any, model: Any, device: torch.device
) -> list[float]:
    scores: list[float] = []
    for offset in range(0, len(texts), PREDICT_BATCH_SIZE):
        encoded = tokenizer(
            list(texts[offset : offset + PREDICT_BATCH_SIZE]),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            batch_scores = model(**encoded).logits.squeeze(-1)
        scores.extend(float(value) for value in batch_scores.float().cpu())
    return scores


def predict(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = args.output_dir / "checkpoint_final"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir, local_files_only=True
    ).to(device)
    model.eval()
    started = time.perf_counter()
    output_path = args.output_dir / "dev_predictions_frozen.jsonl"

    def records() -> Iterable[dict[str, Any]]:
        for dataset, domain, score_mode, path in input_paths(args.data_root, "dev"):
            count = 0
            for example_id, source_rows in grouped_jsonl(path):
                # Admission is frozen using only gold-free generator fields.
                admitted_source = cap_inference_candidate_rows(
                    source_rows, max_candidates=MAX_INFERENCE_CANDIDATES
                )
                rows = [safe_inference_row(row) for row in admitted_source]
                question = str(rows[0].get("question") or rows[0].get("abs_question") or example_id)
                identities = [candidate_identity(row) for row in rows]
                if len(set(identities)) != len(identities):
                    raise ValueError(f"Duplicate candidate identity for {example_id}")
                texts = [
                    serialize_candidate(question, row.get("subgraph_units", []), domain)
                    for row in rows
                ]
                scores = score_texts(texts, tokenizer, model, device)
                symbolic_scores = [
                    compute_adjusted_score(row, score, score_mode=score_mode, size_penalty=0.01)
                    for row, score in zip(rows, scores)
                ]
                neural_order = sorted(range(len(rows)), key=lambda i: (-scores[i], identities[i]))
                symbolic_order = sorted(
                    range(len(rows)), key=lambda i: (-symbolic_scores[i], identities[i])
                )
                count += 1
                if count % 100 == 0:
                    print(f"predict {dataset}: {count}", flush=True)
                yield {
                    "dataset": dataset,
                    "domain": domain,
                    "example_id": example_id,
                    "candidate_count": len(rows),
                    "candidate_identity_definition": "sha256(canonical JSON ordered subgraph_units)",
                    "cross_encoder_order": [identities[i] for i in neural_order],
                    "cross_encoder_scores": [scores[i] for i in neural_order],
                    "cross_encoder_symbolic_order": [identities[i] for i in symbolic_order],
                    "cross_encoder_symbolic_scores": [symbolic_scores[i] for i in symbolic_order],
                    "score_mode": score_mode,
                }

    write_jsonl_atomic(output_path, records())
    freeze = {
        "schema_version": "question_candidate_cross_encoder_dev_v1_prediction_freeze",
        "status": "frozen_before_dev_gold_join",
        "split": "dev",
        "test_accessed": False,
        "predictions_path": str(output_path.relative_to(ROOT)),
        "predictions_sha256": sha256(output_path),
        "checkpoint_sha256": sha256(checkpoint_dir / "model.safetensors"),
        "prediction_seconds": time.perf_counter() - started,
    }
    dump_json(args.output_dir / "prediction_freeze.json", freeze)
    return freeze


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_id in predictions:
                raise ValueError(f"Duplicate prediction {example_id}")
            predictions[example_id] = row
    return predictions


def load_old_rankings(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            method = str(row.get("method"))
            if method in {"gnn_only", "sageqa_final"}:
                result[str(row["example_id"])][method] = row
    return dict(result)


def rank_distribution(ranks: Sequence[int]) -> dict[str, int]:
    return {
        "rank_1": sum(rank == 1 for rank in ranks),
        "rank_2_5": sum(2 <= rank <= 5 for rank in ranks),
        "rank_6_10": sum(6 <= rank <= 10 for rank in ranks),
        "rank_11_50": sum(11 <= rank <= 50 for rank in ranks),
        "rank_51_100": sum(51 <= rank <= 100 for rank in ranks),
        "rank_101_320": sum(101 <= rank <= 320 for rank in ranks),
    }


def summarize_method(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    eligible = [row for row in records if row["best_complete_rank"] is not None]
    ranks = [int(row["best_complete_rank"]) for row in eligible]
    return {
        "examples": count,
        "support": {
            key: sum(float(row[key]) for row in records) / count
            for key in ("precision", "recall", "f1")
        },
        "complete_candidate_available": len(eligible),
        "complete_support_at_1_all_examples": sum(bool(row["top1_complete"]) for row in records)
        / count,
        "complete_support_at_1_when_available": sum(bool(row["top1_complete"]) for row in eligible)
        / len(eligible)
        if eligible
        else None,
        "median_best_complete_rank": statistics.median(ranks) if ranks else None,
        "mrr_best_complete_candidate_when_available": (
            sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else None
        ),
        "best_complete_rank_distribution": rank_distribution(ranks),
    }


def summarize_equal_dataset_macro(
    datasets: Mapping[str, Mapping[str, Mapping[str, Any]]],
    method_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    return {
        method: {
            key: sum(
                float(dataset_methods[method]["support"][key])
                for dataset_methods in datasets.values()
            )
            / len(datasets)
            for key in ("precision", "recall", "f1")
        }
        for method in method_names
    }


def old_record_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    k1 = next(item for item in row["prefix_evaluation"] if int(item["k"]) == 1)
    diagnostic = row.get("ranking_diagnostic") or {}
    return {
        "precision": float(k1["precision"]),
        "recall": float(k1["recall"]),
        "f1": float(k1["f1"]),
        "top1_complete": bool(diagnostic.get("top1_complete", False)),
        "best_complete_rank": diagnostic.get("best_complete_rank"),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.output_dir / "dev_predictions_frozen.jsonl"
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if sha256(prediction_path) != freeze["predictions_sha256"]:
        raise ValueError("Frozen prediction hash mismatch")
    predictions = load_predictions(prediction_path)
    old = load_old_rankings(args.old_rankings)
    per_example: list[dict[str, Any]] = []

    for dataset, _, _, path in input_paths(args.data_root, "dev"):
        for example_id, source_rows in grouped_jsonl(path):
            prediction = predictions.pop(example_id)
            admitted = cap_inference_candidate_rows(
                source_rows, max_candidates=MAX_INFERENCE_CANDIDATES
            )
            by_identity = {candidate_identity(row): row for row in admitted}
            if len(by_identity) != len(admitted):
                raise ValueError(f"Duplicate source candidate identity {example_id}")
            gold_explanations = gold_explanations_from_row(dict(admitted[0]))
            methods: dict[str, dict[str, Any]] = {
                "existing_gnn_k1": old_record_metrics(old[example_id]["gnn_only"]),
                "existing_sage_final_k1": old_record_metrics(old[example_id]["sageqa_final"]),
            }
            for method, order_key in (
                ("cross_encoder_k1", "cross_encoder_order"),
                ("cross_encoder_symbolic_k1", "cross_encoder_symbolic_order"),
            ):
                order = prediction[order_key]
                if len(order) != len(admitted) or set(order) != set(by_identity):
                    raise ValueError(f"Frozen/source candidate mismatch for {example_id}/{method}")
                ranked = [by_identity[identity] for identity in order]
                support = best_evidence_scores(
                    ranked[0].get("subgraph_units", []), gold_explanations
                )
                best_rank = next(
                    (
                        rank
                        for rank, row in enumerate(ranked, 1)
                        if bool(row.get("contains_any_gold_explanation", False))
                    ),
                    None,
                )
                methods[method] = {
                    **support,
                    "top1_complete": best_rank == 1,
                    "best_complete_rank": best_rank,
                }
            per_example.append({"dataset": dataset, "example_id": example_id, "methods": methods})
    if predictions:
        raise ValueError(f"Unmatched frozen predictions: {len(predictions)}")

    method_names = (
        "existing_gnn_k1",
        "cross_encoder_k1",
        "existing_sage_final_k1",
        "cross_encoder_symbolic_k1",
    )
    datasets: dict[str, Any] = {}
    for dataset, _, _ in DATASETS:
        selected = [row for row in per_example if row["dataset"] == dataset]
        datasets[dataset] = {
            method: summarize_method([row["methods"][method] for row in selected])
            for method in method_names
        }
    overall = {
        method: summarize_method([row["methods"][method] for row in per_example])
        for method in method_names
    }
    equal_dataset_macro = summarize_equal_dataset_macro(datasets, method_names)

    eligible_pairs = [
        (
            int(row["methods"]["existing_gnn_k1"]["best_complete_rank"]),
            int(row["methods"]["cross_encoder_k1"]["best_complete_rank"]),
        )
        for row in per_example
        if row["methods"]["existing_gnn_k1"]["best_complete_rank"] is not None
        and row["methods"]["cross_encoder_k1"]["best_complete_rank"] is not None
    ]
    matched = {}
    for old_name, new_name, label in (
        ("existing_gnn_k1", "cross_encoder_k1", "neural"),
        ("existing_sage_final_k1", "cross_encoder_symbolic_k1", "symbolic"),
    ):
        changes = {
            dataset: datasets[dataset][new_name]["support"]["f1"]
            - datasets[dataset][old_name]["support"]["f1"]
            for dataset, _, _ in DATASETS
        }
        pooled_per_example_macro_delta = (
            overall[new_name]["support"]["f1"] - overall[old_name]["support"]["f1"]
        )
        equal_dataset_macro_delta = (
            equal_dataset_macro[new_name]["f1"] - equal_dataset_macro[old_name]["f1"]
        )
        datasets_improved = sum(delta > 0.0 for delta in changes.values())
        datasets_without_material_regression = sum(
            delta >= -0.02 for delta in changes.values()
        )
        matched[label] = {
            "old_method": old_name,
            "new_method": new_name,
            "criterion_f1_aggregation": "pooled_dev_unweighted_per_example_mean",
            "pooled_per_example_macro_f1_delta": pooled_per_example_macro_delta,
            "equal_dataset_macro_f1_delta_audit": equal_dataset_macro_delta,
            "per_dataset_f1_delta": changes,
            "datasets_improved": datasets_improved,
            "datasets_without_material_regression": datasets_without_material_regression,
            "equal_dataset_macro_full_criterion_passed_audit": (
                equal_dataset_macro_delta >= 0.03
                and datasets_improved >= 6
                and datasets_without_material_regression >= 8
            ),
            "passed": pooled_per_example_macro_delta >= 0.03
            and datasets_improved >= 6
            and datasets_without_material_regression >= 8,
        }
    decision_passed = matched["neural"]["passed"] and matched["symbolic"]["passed"]
    report = {
        "schema_version": "question_candidate_cross_encoder_dev_v1_evaluation",
        "status": "complete",
        "split": "dev",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "configuration": configuration(),
        "adaptive_aggregation": {
            "applied": False,
            "reason": "Frozen policies standardize GNN-score-scale features; cross-encoder logits are not scale-compatible and refitting is forbidden.",
        },
        "aggregation_definitions": {
            "overall": "Unweighted mean of per-example metrics over the pooled DEV cohort; datasets are therefore weighted by their example counts.",
            "equal_dataset_macro": "Unweighted mean of the ten dataset-level per-example means; reported as an aggregation audit and not substituted into the frozen decision rule.",
        },
        "overall": overall,
        "equal_dataset_macro": equal_dataset_macro,
        "datasets": datasets,
        "old_gnn_vs_cross_encoder_when_complete_exists": {
            "examples": len(eligible_pairs),
            "old_complete_ranked_1": sum(old_rank == 1 for old_rank, _ in eligible_pairs)
            / len(eligible_pairs),
            "new_complete_ranked_1": sum(new_rank == 1 for _, new_rank in eligible_pairs)
            / len(eligible_pairs),
            "old_median_rank": statistics.median(old_rank for old_rank, _ in eligible_pairs),
            "new_median_rank": statistics.median(new_rank for _, new_rank in eligible_pairs),
            "improved": sum(new_rank < old_rank for old_rank, new_rank in eligible_pairs),
            "worsened": sum(new_rank > old_rank for old_rank, new_rank in eligible_pairs),
            "unchanged": sum(new_rank == old_rank for old_rank, new_rank in eligible_pairs),
        },
        "decision_rule": matched,
        "success_criterion_passed": decision_passed,
        "required_next_action": "freeze_for_fresh_test"
        if decision_passed
        else "stop_failed_no_second_variation",
    }
    dump_json(args.output_dir / "evaluation_report.json", report)
    write_jsonl_atomic(args.output_dir / "per_example_evaluation.jsonl", per_example)
    return report


def render_summary(args: argparse.Namespace, report: Mapping[str, Any]) -> None:
    training_report = json.loads(
        (args.output_dir / "training_report.json").read_text(encoding="utf-8")
    )
    freeze = json.loads((args.output_dir / "prediction_freeze.json").read_text(encoding="utf-8"))
    overall = report["overall"]
    lines = [
        "# Question-candidate cross-encoder DEV result",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Loss: `mean(max(0, {MARGIN} - s_higher + s_lower))`",
        f"- Training time: {training_report['training_seconds']:.1f} seconds",
        f"- Prediction time: {freeze['prediction_seconds']:.1f} seconds",
        f"- Success criterion passed: **{report['success_criterion_passed']}**",
        f"- Required next action: `{report['required_next_action']}`",
        "- Frozen criterion aggregation: pooled DEV unweighted per-example mean (dataset-size weighted)",
        "- Equal-dataset macro audit: unweighted mean of the 10 dataset-level means",
        "- Equal-dataset macro full-criterion audit passed: "
        f"**{all(item['equal_dataset_macro_full_criterion_passed_audit'] for item in report['decision_rule'].values())}**",
        "",
        "## Pooled DEV per-example means",
        "",
        "| Method | P | R | F1 | Complete@1 (eligible) | Median best-complete rank | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in (
        "existing_gnn_k1",
        "cross_encoder_k1",
        "existing_sage_final_k1",
        "cross_encoder_symbolic_k1",
    ):
        row = overall[method]
        lines.append(
            f"| {method} | {row['support']['precision']:.6f} | {row['support']['recall']:.6f} | "
            f"{row['support']['f1']:.6f} | {row['complete_support_at_1_when_available']:.6f} | "
            f"{row['median_best_complete_rank']} | {row['mrr_best_complete_candidate_when_available']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Equal-dataset macro audit",
            "",
            "| Comparison | Existing F1 | New F1 | Delta | Threshold | Full criterion pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison in ("neural", "symbolic"):
        decision = report["decision_rule"][comparison]
        old_name = decision["old_method"]
        new_name = decision["new_method"]
        old_f1 = report["equal_dataset_macro"][old_name]["f1"]
        new_f1 = report["equal_dataset_macro"][new_name]["f1"]
        audit_passed = decision["equal_dataset_macro_full_criterion_passed_audit"]
        lines.append(
            f"| {old_name} vs {new_name} | {old_f1:.6f} | {new_f1:.6f} | "
            f"{new_f1 - old_f1:+.6f} | +0.030000 | {audit_passed} |"
        )
    lines.extend(
        [
            "",
            "## Per-dataset F1",
            "",
            "| Dataset | Existing GNN | Cross-encoder | Delta | Existing SAGE | Cross-encoder+symbolic | Delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, _, _ in DATASETS:
        values = report["datasets"][dataset]
        old_gnn = values["existing_gnn_k1"]["support"]["f1"]
        new_gnn = values["cross_encoder_k1"]["support"]["f1"]
        old_sage = values["existing_sage_final_k1"]["support"]["f1"]
        new_sage = values["cross_encoder_symbolic_k1"]["support"]["f1"]
        lines.append(
            f"| {dataset} | {old_gnn:.6f} | {new_gnn:.6f} | {new_gnn - old_gnn:+.6f} | "
            f"{old_sage:.6f} | {new_sage:.6f} | {new_sage - old_sage:+.6f} |"
        )
    lines.extend(
        [
            "",
            "Adaptive aggregation was not applied: its frozen standardized decision features are tied to the GNN score scale.",
            "",
        ]
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "train", "predict", "evaluate", "run"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--old-rankings", type=Path, default=DEFAULT_OLD_RANKINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.old_rankings = args.old_rankings.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.phase in {"preflight", "run"}:
        preflight(args)
    if args.phase in {"train", "run"}:
        train(args)
    if args.phase in {"predict", "run"}:
        predict(args)
    if args.phase in {"evaluate", "run"}:
        report = evaluate(args)
        render_summary(args, report)
        print(json.dumps({"success_criterion_passed": report["success_criterion_passed"]}))


if __name__ == "__main__":
    main()
