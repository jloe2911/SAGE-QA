"""Frozen one-time TEST pipeline for the DEV-selected original-v1 reranker.

The generate phase is gold-free and writes an atomic prediction freeze.  The
evaluate phase is deliberately separate and cannot open gold until that freeze
has passed its hash and completeness checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_processing.build_2wiki_subgraph_dataset import (  # noqa: E402
    flatten_context as flatten_2wiki_context,
    get_gold_support_units as get_2wiki_gold,
    normalize_2wiki_record,
)
from data_processing.build_hotpot_subgraph_dataset import (  # noqa: E402
    flatten_context as flatten_hotpot_context,
    get_gold_support_units as get_hotpot_gold,
    normalize_hotpot_record,
)
from data_processing.retrieval_contracts import cap_inference_candidate_rows  # noqa: E402
from evaluation.adaptive_support_aggregation import fixed_k_support_aggregate  # noqa: E402
from evaluation.adaptive_support_aggregation_v2 import (  # noqa: E402
    CONTINUOUS_FEATURES,
    FEATURE_ORDER,
    AdaptiveV2Policy,
    compute_decision_features,
)
from experiments.symbolic_coefficients_original_v1.original_symbolic_features import (  # noqa: E402
    ProofCoefficients,
    TextCoefficients,
    candidate_identity,
    proof_features,
    score_proof_features,
    score_text_features,
    text_features,
)


ALLOWED_K = (1, 2, 3, 5)
TRANSITIONS = ((1, 2), (2, 3), (3, 5))
MAX_INFERENCE_CANDIDATES = 320
MAX_LENGTH = 256
PREDICT_BATCH_SIZE = 128
EXPECTED_CHECKPOINT_SHA256 = "02b06c91fa422369a0b0adbab3b33bfcf80f335b7c8a5911084c3b82af117ba9"
EXPECTED_THRESHOLDS = {"text": 0.77, "ontology": 0.86}

# The exact extracted reference bundle inspected on 2026-09-25.  These are
# external inputs; this repository never copies or rewrites them.
EXPECTED_REFERENCE_SHA256 = {
    "adaptive_k_v1/adaptive_k_config.json": "72d58714f8124ce1e696ec92b524d735d33b7848e30a5879605eb95506d4928a",
    "adaptive_k_v1/artifact_manifest.json": "45f771a36c4b1dd1d9115ec909058a6a97dbe2e074ca35d8f746b2143606ce16",
    "adaptive_k_v1/chosen_thresholds.joblib": "3e86b36ade638e7a63aad707529c1a23b74979bad520600d7f4edd9a478809c5",
    "adaptive_k_v1/grouped_cv_metadata.json": "a6c6eae18bee23bbe0f798f7333614301be1d3c6500eabeaceda5c52b4ec0078",
    "adaptive_k_v1/lineage_checkpoint_metadata.json": "94a269a790bee373cb4d2ed30cb92974397b822132fb281af65dd902ef4c439b",
    "adaptive_k_v1/metrics.json": "0c9e628a833ac4f8b839ca516d365bc379fb9cba5ff2dee6cd8d6da7ac834d7f",
    "adaptive_k_v1/ontology_model.joblib": "8da4afacaf2900abe373bd912a72e6a39c8b4702300b932cb679b322faf9f65e",
    "adaptive_k_v1/ontology_oof_predictions.jsonl": "616cfe1dcf5f72c0bb72204a2f667dc0205c34b48bbd0a91dfa9932e5222ccc1",
    "adaptive_k_v1/ontology_scaler.joblib": "5759af6860abecebc5b3641e89085535e2e491480b8ccf3aec23a1612ef1a5d6",
    "adaptive_k_v1/ontology_threshold_sweep.json": "4cd120dc1b224fcb0adddeaf4e4309cd523e6d394b41a54a694463d8240de041",
    "adaptive_k_v1/per_example_adaptive_dev.jsonl": "f0522f097eae2828594e73fd19bf120ff67fdb5b1b71cc9837753cc3f86a52be",
    "adaptive_k_v1/summary.md": "e2329497c162b75972a36df0b402eaf21e557cd92aff1078cc5ebc9209605473",
    "adaptive_k_v1/text_model.joblib": "54e35760c1c6a4ef4c9887bbeee7e2a01b96837222313c6ebf31f6d3aec6f2e6",
    "adaptive_k_v1/text_oof_predictions.jsonl": "f99f8d52b778e8f198bdee76156013c4558ce2bdf0fc3dbcf306ded19872b493",
    "adaptive_k_v1/text_scaler.joblib": "b8106b6c6450aaf5c1c99cdd3e9d96cc9a47a9b55854d940384f89f90b39218d",
    "adaptive_k_v1/text_threshold_sweep.json": "6d4a31613755e8aca0f6f79321d594aad23f7dcc3e494a321a3293039fc8a22a",
    "dev_rankings_v1/checkpoint_metadata.json": "9ad888f488e37e75164164486b746e3a25ef8790d7ad6a7a7477a33e67ef77a6",
    "dev_rankings_v1/metrics.json": "a76e9b74af319c0557d98631ab11a27f7ae45369ffdade7ec259468770cd6119",
    "frozen_coefficients_v1/freeze_manifest.json": "7f29dd32afa80c12ddc80cb5a78d48168f83604adb8832ba4634edc18d7aae6a",
    "frozen_coefficients_v1/proof_coefficients.json": "350c80fa94893e402ba7341a2b384ae69db129f69f7161859cb70fc8da7b1585",
    "frozen_coefficients_v1/selected_coefficients.json": "9cc3d4b870721964d335c21605aeaf1134650aa53fc07397f467654a68e74f8b",
    "frozen_coefficients_v1/text_chain_coefficients.json": "1291f8c078c1841eb41ea9c388b8a163137f009141d92b79dab1972ba1d3baef",
    "original_vs_selected_dev.json": "0c08b16fe3793429aaa0e04ab35c062c224c08de9a629a65950b36179fb634c0",
}

DATASETS = (
    ("HotpotQA", "HotpotQA", "text", "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet", "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6", 1000, 509863, "e295a5217f8042ba777f70415461ac1d10f3b93863b6611aa7de962a9713abd8"),
    ("2WikiMultiHopQA", "2WikiMultiHopQA", "text", "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet", "408e2dbb28edc6c8b9ca3ba0c94d4fc7bf17ffb923766593a3a7f546ab4cba59", 1000, 502407, "1cd172edecbf8053f0259362331ee6a43f3b44b4e25871022d72d4d6b78f5569"),
    ("FamilyOWL_1hop", "FamilyOWL_1hop", "ontology", "data/raw/family/FamilyOWL_1hop.json", "4cdc9ab099da180e066968cdcc4eb60a3ae6d9cfbb0f4cc92c9a219c0cb8c8a9", 462, 236544, "287011713b18fd251a368ea81c712b3726cf4bbe509d21b5b7d7bb4083b60ebe"),
    ("FamilyOWL_2hop", "FamilyOWL_2hop", "ontology", "data/raw/family/FamilyOWL_2hop.json", "2ed7ef66c2a07ac1972dc241e2b2e8a8511179761c09cc0d8754262046f04e7d", 462, 236544, "2d49a62953cd98226cb44fb80ce9ca0ae8b8a226715dc0b84dcccff830664673"),
    ("pizza_100_1hop", "pizza_100_1hop", "ontology", "data/raw/pizza_100/Pizza100_1hop.json", "474b75ee3adcd0b71922a919b63a4e8448a6e98e232dd139cff52a8040593cf1", 119, 60720, "cef8af4df1e83e0ff62e5789948c6cf1ed130d9fc76b2520197eeeed3eb5e956"),
    ("pizza_100_2hop", "pizza_100_2hop", "ontology", "data/raw/pizza_100/Pizza100_2hop.json", "7efca8c953c43a3279a222cc3d640c46a5216b9e3de1dbe4ad5110435598a0ad", 125, 64000, "baa5aa0254bb8f1970259b9ab95ce82e9172e76aaf94e13aa89a1424cec0fb7f"),
    ("pizza_250_1hop", "pizza_250_1hop", "ontology", "data/raw/pizza_250/Pizza250_1hop.json", "c385e7c0a13fd666d192816bc972cf7039ee2807db4aef3b5ccefbfb47d903fb", 149, 75626, "d02d1b42a4a3d3f66a3504a7cd57bc39eae57f16261004b5ba27164578dc9e2f"),
    ("pizza_250_2hop", "pizza_250_2hop", "ontology", "data/raw/pizza_250/Pizza250_2hop.json", "918e6764cb90cdc7387300107df486e813895945c8194d2431f65b3c4ccb25ba", 157, 80384, "7b7b794830f044e52843e79b803d9226b17ec499bf8f41183039276c4895c12c"),
    ("OWL2Bench_1hop", "OWL2Bench_1hop", "ontology", "data/raw/owl2bench/OWL2Bench_1hop.json", "400650df9e59a46561bf05c3618eec9e0d31bc26a8c9aa1e13e207344a7b7e0a", 376, 192512, "cc8aa7dc07a43e73cfb07d194bbc0e5a0addba00c54c8ddfadd702e5eb8fb877"),
    ("OWL2Bench_2hop", "OWL2Bench_2hop", "ontology", "data/raw/owl2bench/OWL2Bench_2hop.json", "43f98a42ec21c8b076b8a3737e686ddedb6673eda401a99a2aaea62d434fd146", 399, 204288, "38c8233f8f63fd9a95404449526670c8ecad6faaa55b7a849bf154f7d7c7de23"),
)

SAFE_FIELDS = {
    "example_id", "dataset", "source_name", "question", "abs_question",
    "sparql_query", "subgraph_units", "subgraph_size", "graph_context_units",
    "kg_evidence_units", "symbolic_features", "task_type", "answer_type",
}
PROHIBITED_FIELDS = {
    "gold_explanations", "gold_units", "gold_support_units", "raw_supporting_facts",
    "supporting_facts", "evidences", "answer", "answers", "label", "rank_target",
    "best_set_f1_to_gold", "best_set_precision_to_gold", "best_set_recall_to_gold",
    "exact_match_any_gold", "contains_any_gold_explanation",
}
CODE_FILES = (
    "experiments/symbolic_coefficients_original_v1/test_pipeline.py",
    "experiments/symbolic_coefficients_original_v1/original_symbolic_features.py",
    "evaluation/adaptive_support_aggregation.py",
    "evaluation/adaptive_support_aggregation_v2.py",
    "data_processing/retrieval_contracts.py",
    "data_processing/build_hotpot_subgraph_dataset.py",
    "data_processing/build_2wiki_subgraph_dataset.py",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def grouped_jsonl(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    seen: set[str] = set()
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
                    raise ValueError(f"{path}: non-contiguous example {example_id}")
                seen.add(current_id)
                yield current_id, current
                current_id, current = example_id, []
            current.append(row)
    if current_id is not None:
        yield current_id, current


def safe_inference_row(row: Mapping[str, Any]) -> dict[str, Any]:
    projected = {key: row[key] for key in SAFE_FIELDS if key in row}
    forbidden = PROHIBITED_FIELDS.intersection(projected)
    if forbidden:
        raise ValueError(f"Gold-bearing fields reached inference: {sorted(forbidden)}")
    return projected


def parse_text_unit(unit: str) -> tuple[str, int, str] | None:
    parts = str(unit).split("::", 3)
    if len(parts) != 4 or parts[0] != "SENT":
        return None
    try:
        sentence_index = int(parts[2])
    except ValueError:
        return None
    return parts[1], sentence_index, parts[3]


def serialize_candidate(question: str, units: Sequence[str], domain: str) -> str:
    evidence = []
    for position, unit in enumerate(units, 1):
        parsed = parse_text_unit(str(unit)) if domain == "text" else None
        if parsed is None:
            evidence.append(f"[{position}] {unit}")
        else:
            title, sentence_index, sentence = parsed
            evidence.append(f"[{position}] Title: {title}\nSentence index: {sentence_index}\nText: {sentence}")
    return f"Question:\n{question}\n\nCandidate evidence:\n" + "\n".join(evidence)


@torch.inference_mode()
def score_texts(texts: Sequence[str], tokenizer: Any, model: Any, device: torch.device) -> list[float]:
    scores: list[float] = []
    for offset in range(0, len(texts), PREDICT_BATCH_SIZE):
        encoded = tokenizer(list(texts[offset:offset + PREDICT_BATCH_SIZE]), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(**encoded).logits.squeeze(-1)
        scores.extend(float(value) for value in logits.float().cpu())
    return scores


def _validated_thresholds(value: Any, source: Path) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"text", "ontology"}:
        raise ValueError(f"Invalid thresholds in {source}")
    result = {}
    for domain in ("text", "ontology"):
        raw = value[domain]
        if isinstance(raw, bool) or not isinstance(raw, Real) or not math.isfinite(float(raw)):
            raise ValueError(f"Invalid {domain} threshold in {source}")
        result[domain] = float(raw)
    return result


def validate_reference_bundle(reference_dir: Path) -> dict[str, str]:
    actual_files = {
        path.relative_to(reference_dir).as_posix(): sha256(path)
        for path in reference_dir.rglob("*") if path.is_file()
    }
    if actual_files != EXPECTED_REFERENCE_SHA256:
        missing = sorted(set(EXPECTED_REFERENCE_SHA256) - set(actual_files))
        extra = sorted(set(actual_files) - set(EXPECTED_REFERENCE_SHA256))
        changed = sorted(name for name in set(actual_files) & set(EXPECTED_REFERENCE_SHA256) if actual_files[name] != EXPECTED_REFERENCE_SHA256[name])
        raise ValueError(f"Reference artifact inventory mismatch; missing={missing}, extra={extra}, changed={changed}")

    freeze = read_json(reference_dir / "frozen_coefficients_v1/freeze_manifest.json")
    selected_path = reference_dir / "frozen_coefficients_v1/selected_coefficients.json"
    if (
        freeze.get("status") != "frozen_after_dev_selection"
        or freeze.get("test_accessed") is not False
        or freeze.get("selected_coefficients_sha256") != sha256(selected_path)
    ):
        raise ValueError("Invalid selected-coefficient freeze manifest")
    for name, expected in freeze.get("coefficient_files_sha256", {}).items():
        if sha256(reference_dir / "frozen_coefficients_v1" / name) != expected:
            raise ValueError(f"Frozen coefficient hash mismatch: {name}")

    policy_dir = reference_dir / "adaptive_k_v1"
    manifest = read_json(policy_dir / "artifact_manifest.json")
    for name, expected in manifest.get("files", {}).items():
        if sha256(policy_dir / name) != expected:
            raise ValueError(f"Adaptive policy manifest mismatch: {name}")
    config = read_json(policy_dir / "adaptive_k_config.json")
    if config.get("ranking_method") != "final_sageqa" or config.get("allowed_k") != list(ALLOWED_K):
        raise ValueError("Wrong adaptive policy method or allowed-k contract")
    if config.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ValueError("Unexpected DEV-selected thresholds")
    if config.get("feature_order") != list(FEATURE_ORDER) or config.get("continuous_features_standardized") != list(CONTINUOUS_FEATURES):
        raise ValueError("Adaptive feature schema mismatch")
    return actual_files


def load_selected_coefficients(reference_dir: Path) -> tuple[TextCoefficients, ProofCoefficients]:
    selected = read_json(reference_dir / "frozen_coefficients_v1/selected_coefficients.json")
    if selected.get("split") != "dev" or selected.get("status") != "frozen_after_dev_selection":
        raise ValueError("Coefficients are not a frozen DEV selection")
    return TextCoefficients(**selected["text"]), ProofCoefficients(**selected["ontology"])


def load_adaptive_policies(reference_dir: Path) -> dict[str, AdaptiveV2Policy]:
    policy_dir = reference_dir / "adaptive_k_v1"
    config = read_json(policy_dir / "adaptive_k_config.json")
    thresholds_raw = joblib.load(policy_dir / "chosen_thresholds.joblib")
    thresholds = _validated_thresholds(thresholds_raw, policy_dir / "chosen_thresholds.joblib")
    if thresholds != EXPECTED_THRESHOLDS or thresholds_raw != config["thresholds"]:
        raise ValueError("Saved thresholds disagree with the frozen config")
    policies = {}
    for domain in ("text", "ontology"):
        scaler = joblib.load(policy_dir / f"{domain}_scaler.joblib")
        model = joblib.load(policy_dir / f"{domain}_model.joblib")
        described = config["full_dev_model_parameters"][domain]
        if scaler.mean_.tolist() != described["scaler_mean"] or scaler.scale_.tolist() != described["scaler_scale"]:
            raise ValueError(f"{domain} scaler parameters disagree with config")
        if model.coef_[0].tolist() != described["coefficients_in_model_feature_order"] or float(model.intercept_[0]) != float(described["intercept"]):
            raise ValueError(f"{domain} model parameters disagree with config")
        policies[domain] = AdaptiveV2Policy(domain=domain, scaler=scaler, model=model, threshold=thresholds[domain])
    return policies


def rank_selected(rows: Sequence[Mapping[str, Any]], logits: Sequence[float], domain: str, text_coefficients: TextCoefficients, proof_coefficients: ProofCoefficients) -> list[dict[str, Any]]:
    ranked = []
    for row, logit in zip(rows, logits):
        features = text_features(row) if domain == "text" else proof_features(row)
        adjustment = score_text_features(features, text_coefficients) if domain == "text" else score_proof_features(features, proof_coefficients)
        item = dict(row)
        item.update({"candidate_identity": candidate_identity(row), "score": float(logit), "symbolic_adjustment": float(adjustment), "adjusted_score": float(logit) + float(adjustment)})
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["adjusted_score"], item["candidate_identity"]))
    for rank, item in enumerate(ranked, 1):
        item["rank"] = rank
    return ranked


def fixed_select(ranked: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = fixed_k_support_aggregate(ranked, k=1)
    return {"selected_k": 1, "retrieved_evidence_units": list(aggregate["support_units"]), "retrieved_evidence_count": int(aggregate["final_support_size"])}


def adaptive_select(ranked: Sequence[Mapping[str, Any]], policy: AdaptiveV2Policy) -> dict[str, Any]:
    candidates = list(ranked[:5])
    if not candidates:
        raise ValueError("Adaptive selection requires candidates")
    selected_k, decisions = 1, []
    for current_k, next_k in TRANSITIONS:
        if len(candidates) <= current_k or selected_k != current_k:
            break
        features = compute_decision_features(candidates, decision_rank=current_k)
        probability = policy.continue_probability(features)
        decision = "CONTINUE" if probability >= policy.threshold else "STOP"
        decisions.append({"current_k": current_k, "continue_to_k": next_k, "predicted_continue_probability": probability, "threshold": policy.threshold, "decision": decision})
        if decision == "STOP":
            break
        selected_k = next_k
    aggregate = fixed_k_support_aggregate(candidates, k=selected_k)
    return {"selected_k": selected_k, "retrieved_evidence_units": list(aggregate["support_units"]), "retrieved_evidence_count": int(aggregate["final_support_size"]), "decisions": decisions}


def validate_candidate_fixture(path: Path, expected_examples: int | None = None, expected_rows: int | None = None) -> tuple[int, int]:
    examples = rows = 0
    for _, group in grouped_jsonl(path):
        examples += 1
        rows += len(group)
        for raw in group:
            projected = safe_inference_row(raw)
            if PROHIBITED_FIELDS.intersection(projected):
                raise ValueError("Gold-bearing field survived inference projection")
    if expected_examples is not None and (examples, rows) != (expected_examples, expected_rows):
        raise ValueError(f"Candidate count mismatch: {(examples, rows)}")
    return examples, rows


def _gpu_name() -> str:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def preflight(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Preflight output must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_hashes = validate_reference_bundle(args.reference_dir)
    load_selected_coefficients(args.reference_dir)
    load_adaptive_policies(args.reference_dir)
    gpu = _gpu_name()
    if args.require_a40 and "A40" not in gpu.upper():
        raise RuntimeError(f"A40 required, found {gpu!r}")

    training = read_json(args.cross_encoder_dir / "training_report.json")
    checkpoint_dir = args.cross_encoder_dir / "checkpoint_final"
    checkpoint_hashes = {}
    for name, expected in training.get("checkpoint_files_sha256", {}).items():
        actual = sha256(checkpoint_dir / name)
        if actual != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        checkpoint_hashes[name] = actual
    if checkpoint_hashes.get("model.safetensors") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Unexpected Cross-Encoder checkpoint")

    datasets = {}
    for dataset, data_dir, domain, gold_relative, gold_hash, expected_examples, expected_rows, candidate_hash in DATASETS:
        path = args.data_root / data_dir / "test_subgraph_retrieval.jsonl"
        if sha256(path) != candidate_hash:
            raise ValueError(f"Frozen CORE-LLM-Bench v1.0.0 candidate hash mismatch: {dataset}")
        validate_candidate_fixture(path, expected_examples, expected_rows)
        gold_path = repo_root() / gold_relative
        if sha256(gold_path) != gold_hash:
            raise ValueError(f"Frozen v1.0.0 gold-source hash mismatch: {dataset}")
        datasets[dataset] = {"domain": domain, "candidate_path": str(path), "candidate_sha256": candidate_hash, "examples": expected_examples, "candidate_rows": expected_rows, "gold_source_path": gold_relative, "gold_source_sha256": gold_hash}

    manifest = {"schema_version": "sageqa_symbolic_coefficients_original_v1_test_preflight", "status": "passed_frozen_preflight", "gpu": gpu, "a40_required": args.require_a40, "reference_artifact_sha256": reference_hashes, "checkpoint_files_sha256": checkpoint_hashes, "datasets": datasets, "expected_total_examples": sum(row[5] for row in DATASETS), "max_inference_candidates": MAX_INFERENCE_CANDIDATES, "max_length": MAX_LENGTH, "gold_source_files_opened_for_content": 0, "prediction_freeze_required_before_evaluate": True, "code_sha256": {name: sha256(repo_root() / name) for name in CODE_FILES}}
    write_json(args.output_dir / "preflight_manifest.json", manifest)
    print(json.dumps({"phase": "preflight", "complete": True, "examples": manifest["expected_total_examples"]}))


def generate(args: argparse.Namespace) -> None:
    preflight_path = args.output_dir / "preflight_manifest.json"
    manifest = read_json(preflight_path)
    if manifest.get("status") != "passed_frozen_preflight":
        raise ValueError("Missing passed frozen preflight")
    current_code_hashes = {name: sha256(repo_root() / name) for name in CODE_FILES}
    if manifest.get("code_sha256") != current_code_hashes:
        raise ValueError("Inference code changed after preflight")
    predictions_path = args.output_dir / "test_predictions_frozen.jsonl"
    freeze_path = args.output_dir / "generation_freeze.json"
    if predictions_path.exists() or freeze_path.exists():
        raise FileExistsError("Refusing to overwrite a one-time TEST prediction freeze")
    reference_before = validate_reference_bundle(args.reference_dir)
    text_coefficients, proof_coefficients = load_selected_coefficients(args.reference_dir)
    policies = load_adaptive_policies(args.reference_dir)

    checkpoint_dir = args.cross_encoder_dir / "checkpoint_final"
    for name, expected in manifest.get("checkpoint_files_sha256", {}).items():
        if sha256(checkpoint_dir / name) != expected:
            raise ValueError(f"Cross-Encoder checkpoint changed after preflight: {name}")
    if manifest.get("checkpoint_files_sha256", {}).get("model.safetensors") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Unexpected Cross-Encoder checkpoint after preflight")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    started = time.perf_counter()
    counts: dict[str, int] = defaultdict(int)

    def records() -> Iterable[dict[str, Any]]:
        for dataset, data_dir, domain, _, _, expected_examples, _, candidate_hash in DATASETS:
            path = args.data_root / data_dir / "test_subgraph_retrieval.jsonl"
            if sha256(path) != candidate_hash:
                raise ValueError(f"Candidate input changed after preflight: {dataset}")
            for example_id, raw_rows in grouped_jsonl(path):
                admitted = cap_inference_candidate_rows(raw_rows, max_candidates=MAX_INFERENCE_CANDIDATES)
                rows = [safe_inference_row(row) for row in admitted]
                identities = [candidate_identity(row) for row in rows]
                if not rows or len(identities) != len(set(identities)):
                    raise ValueError(f"Empty or duplicate candidate identity: {example_id}")
                question = str(rows[0].get("question") or rows[0].get("abs_question") or example_id)
                texts = [serialize_candidate(question, row.get("subgraph_units", []), domain) for row in rows]
                logits = score_texts(texts, tokenizer, model, device)
                ranked = rank_selected(rows, logits, domain, text_coefficients, proof_coefficients)
                counts[dataset] += 1
                yield {"schema_version": "sageqa_symbolic_coefficients_original_v1_test_prediction", "split": "test", "dataset": dataset, "domain": domain, "example_id": example_id, "score_mode": "selected_original_v1_symbolic_coefficients", "k1": fixed_select(ranked), "adaptive": adaptive_select(ranked, policies[domain])}
            if counts[dataset] != expected_examples:
                raise ValueError(f"Prediction count mismatch: {dataset}")

    total = write_jsonl_atomic(predictions_path, records())
    if total != sum(row[5] for row in DATASETS):
        raise ValueError("Incomplete TEST prediction set")
    if validate_reference_bundle(args.reference_dir) != reference_before:
        raise RuntimeError("Reference artifact changed during generation")
    freeze = {"schema_version": "sageqa_symbolic_coefficients_original_v1_test_prediction_freeze", "status": "complete_frozen_before_gold_join", "split": "test", "predictions_path": str(predictions_path), "predictions_sha256": sha256(predictions_path), "prediction_examples": total, "datasets": dict(counts), "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "reference_artifact_sha256": reference_before, "selected_coefficients_sha256": EXPECTED_REFERENCE_SHA256["frozen_coefficients_v1/selected_coefficients.json"], "adaptive_policy_manifest_sha256": EXPECTED_REFERENCE_SHA256["adaptive_k_v1/artifact_manifest.json"], "preflight_manifest_sha256": sha256(preflight_path), "gold_accessed_before_freeze": False, "device": str(device), "prediction_seconds": time.perf_counter() - started}
    write_json(freeze_path, freeze)
    print(json.dumps({"phase": "generate", "complete": True, "predictions": total}))


def _local_name(uri: str) -> str:
    value = str(uri or "").strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    if "#" in value:
        return value.split("#")[-1]
    if "/" in value:
        return value.rstrip("/").split("/")[-1]
    return value


def _normalize_explanation_unit(unit: str) -> str:
    value = " ".join(str(unit).strip().split())
    if "â" in value:
        try:
            value = value.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    if not value or value.startswith("TAG:"):
        return ""
    match = re.match(r"^(\S+)\s+owl:inverseOf\s+(\S+)$", value)
    if match:
        left, right = sorted((match.group(1), match.group(2)))
        return f"InverseObjectProperties({left},{right})"
    match = re.match(r"^(\S+)\s+rdfs:subPropertyOf\s+(\S+)$", value)
    if match:
        return f"SubObjectPropertyOf({match.group(1)},{match.group(2)})"
    match = re.match(r"^PropertyChain\((\S+)\s+∘\s+(\S+)\)\s+⊑\s+(\S+)$", value)
    if match:
        return f"ObjectPropertyChain({match.group(1)},{match.group(2)}->{match.group(3)})"
    match = re.match(r"^SubObjectPropertyOf\(ObjectPropertyChain\((.+)\)\s+(<[^>]+>|\S+)\)$", value)
    if match:
        chain = [_local_name(token) for token in re.findall(r"<[^>]+>|\S+", match.group(1))]
        if chain:
            return f"ObjectPropertyChain({','.join(chain)}->{_local_name(match.group(2))})"
    for prefix, rendered in (
        ("Symmetric", "SymmetricObjectProperty"),
        ("Transitive", "TransitiveObjectProperty"),
        ("Functional", "FunctionalObjectProperty"),
    ):
        match = re.match(rf"^{prefix}:\s*(.+)$", value)
        if match:
            return f"{rendered}({match.group(1).strip()})"
    match = re.match(r"^domain\((.+)\)\s*=\s*(.+)$", value)
    if match:
        return f"{match.group(1).strip()} domain {match.group(2).strip()}"
    match = re.match(r"^range\((.+)\)\s*=\s*(.+)$", value)
    if match:
        return f"{match.group(1).strip()} range {match.group(2).strip()}"
    parts = value.split()
    if len(parts) == 3 and parts[1] == "SubPropertyOf":
        return f"SubObjectPropertyOf({parts[0]},{parts[2]})"
    if len(parts) == 3 and parts[1] == "Domain":
        return f"{parts[0]} domain {parts[2]}"
    if len(parts) == 3 and parts[1] == "Range":
        return f"{parts[0]} range {parts[2]}"
    return value


def _normalize_explanation(value: Iterable[Any]) -> list[str]:
    result, seen = [], set()
    for unit in value:
        normalized = _normalize_explanation_unit(str(unit))
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _legacy_gold_explanations(qa: Mapping[str, Any]) -> list[list[str]]:
    explanations = [normalized for value in qa.get("Explanations", []) or [] if (normalized := _normalize_explanation(value))]
    if not explanations and (minimum := _normalize_explanation(qa.get("Minimum Explanation", []) or [])):
        explanations = [minimum]
    return explanations


def text_gold(source: Path, dataset: str, example_ids: set[str]) -> dict[str, list[list[str]]]:
    frame = pd.read_parquet(source, columns=["id", "context", "supporting_facts"])
    normalize = normalize_2wiki_record if dataset == "2WikiMultiHopQA" else normalize_hotpot_record
    flatten = flatten_2wiki_context if dataset == "2WikiMultiHopQA" else flatten_hotpot_context
    gold_fn = get_2wiki_gold if dataset == "2WikiMultiHopQA" else get_hotpot_gold
    result = {}
    for raw in frame.to_dict(orient="records"):
        example_id = f"{dataset}__test__{raw.get('id') or ''}"
        if example_id not in example_ids:
            continue
        example = normalize(raw)
        lookup = {(row["title"], int(row["sent_idx"])): row["unit"] for row in flatten(example)}
        support = gold_fn(example, lookup)
        result[example_id] = [support] if support else []
    return result


def ontology_gold(source: Path, example_ids: set[str]) -> dict[str, list[list[str]]]:
    groups = read_json(source)
    result = {}
    for example_id in example_ids:
        parts = example_id.split("__", 3)
        if len(parts) < 3 or not parts[1].startswith("g") or not parts[2].startswith("q"):
            raise ValueError(f"Cannot parse v1.0.0 ontology example ID: {example_id}")
        qa = groups[int(parts[1][1:])]["QAs"][int(parts[2][1:])]
        result[example_id] = _legacy_gold_explanations(qa)
    return result


def best_scores(predicted: Sequence[str], alternatives: Sequence[Sequence[str]]) -> dict[str, float]:
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


def summarize(rows: Sequence[Mapping[str, Any]], adaptive: bool = False) -> dict[str, Any]:
    result = {metric: statistics.fmean(float(row[metric]) for row in rows) for metric in ("precision", "recall", "f1")}
    if adaptive:
        counts = Counter(int(row["selected_k"]) for row in rows)
        result.update({"mean_selected_k": statistics.fmean(int(row["selected_k"]) for row in rows), "median_selected_k": statistics.median(int(row["selected_k"]) for row in rows), "selected_k_distribution": {str(k): {"count": counts[k], "percentage": 100 * counts[k] / len(rows)} for k in ALLOWED_K}})
    return result


def evaluate(args: argparse.Namespace) -> None:
    predictions_path = args.output_dir / "test_predictions_frozen.jsonl"
    freeze_path = args.output_dir / "generation_freeze.json"
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError("Refusing a second TEST evaluation")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "complete_frozen_before_gold_join" or freeze.get("gold_accessed_before_freeze") is not False:
        raise ValueError("Predictions are not frozen before gold join")
    if sha256(predictions_path) != freeze.get("predictions_sha256") or int(freeze.get("prediction_examples", -1)) != sum(row[5] for row in DATASETS):
        raise ValueError("Prediction freeze hash or completeness mismatch")
    preflight_path = args.output_dir / "preflight_manifest.json"
    if sha256(preflight_path) != freeze.get("preflight_manifest_sha256"):
        raise ValueError("Preflight manifest changed after prediction freeze")
    if read_json(preflight_path).get("code_sha256") != {name: sha256(repo_root() / name) for name in CODE_FILES}:
        raise ValueError("Evaluation code changed after preflight")
    validate_reference_bundle(args.reference_dir)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_dataset[row["dataset"]].append(row)
    scored, dataset_results = [], []
    for dataset, _, domain, gold_relative, gold_hash, expected_examples, _, _ in DATASETS:
        rows = by_dataset[dataset]
        if len(rows) != expected_examples:
            raise ValueError(f"Frozen prediction count mismatch: {dataset}")
        source = repo_root() / gold_relative
        if sha256(source) != gold_hash:
            raise ValueError(f"Gold source changed: {dataset}")
        ids = {row["example_id"] for row in rows}
        gold = text_gold(source, dataset, ids) if domain == "text" else ontology_gold(source, ids)
        if set(gold) != ids:
            raise ValueError(f"Incomplete gold join: {dataset}")
        settings = {"k1": [], "adaptive": []}
        excluded = 0
        for row in rows:
            alternatives = gold[row["example_id"]]
            if not alternatives:
                excluded += 1
                continue
            record = {"dataset": dataset, "domain": domain, "example_id": row["example_id"], "settings": {}}
            for setting in settings:
                selection = row[setting]
                values = {"selected_k": int(selection["selected_k"]), **best_scores(selection["retrieved_evidence_units"], alternatives)}
                settings[setting].append(values)
                record["settings"][setting] = values
            scored.append(record)
        if not settings["k1"]:
            raise ValueError(f"No support-bearing examples: {dataset}")
        dataset_results.append({"dataset": dataset, "domain": domain, "prediction_examples": len(rows), "evaluation_examples": len(settings["k1"]), "excluded_no_gold_support": excluded, "k1": summarize(settings["k1"]), "adaptive": summarize(settings["adaptive"], adaptive=True)})
    metrics = {"schema_version": "sageqa_symbolic_coefficients_original_v1_test_retrieval", "status": "complete_one_time_frozen_test_evaluation", "split": "test", "benchmark": "CORE-LLM-Bench v1.0.0 plus original thesis text settings", "metric_aggregation": "unweighted mean over support-bearing examples", "datasets": dataset_results, "equal_dataset_macro": {setting: {metric: statistics.fmean(row[setting][metric] for row in dataset_results) for metric in ("precision", "recall", "f1")} for setting in ("k1", "adaptive")}, "prediction_freeze_sha256": sha256(freeze_path), "test_informed_tuning": False}
    per_example_path = args.output_dir / "per_example_test_retrieval.jsonl"
    write_jsonl_atomic(per_example_path, scored)
    write_json(metrics_path, metrics)
    files = (predictions_path, freeze_path, per_example_path, metrics_path, args.output_dir / "preflight_manifest.json")
    write_json(args.output_dir / "artifact_manifest.json", {"status": "complete_frozen", "files": {path.name: sha256(path) for path in files}})
    print(json.dumps({"phase": "evaluate", "complete": True, "equal_dataset_macro": metrics["equal_dataset_macro"]}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "generate", "evaluate"))
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=repo_root() / "data/production_generator_d_v1")
    parser.add_argument("--cross-encoder-dir", type=Path, default=repo_root() / "outputs/development_runs/question_candidate_cross_encoder_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-a40", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"preflight": preflight, "generate": generate, "evaluate": evaluate}[args.phase](args)


if __name__ == "__main__":
    main()
