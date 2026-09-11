"""One-time frozen TEST retrieval evaluation for production Generator D."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import pandas as pd
import torch
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import get_gold_explanations
from data_processing.build_2wiki_subgraph_dataset import (
    flatten_context as flatten_2wiki_context,
    get_gold_support_units as get_2wiki_gold,
    normalize_2wiki_record,
)
from data_processing.build_hotpot_subgraph_dataset import (
    flatten_context as flatten_hotpot_context,
    get_gold_support_units as get_hotpot_gold,
    normalize_hotpot_record,
)
from evaluation.adaptive_support_aggregation import fixed_k_support_aggregate
from evaluation.adaptive_support_aggregation_v2 import (
    CONTINUOUS_FEATURES,
    FEATURE_ORDER,
    AdaptiveV2Policy,
    compute_decision_features,
)
from evaluation.eval_gnn_subgraph_retriever import load_gnn_checkpoint
from models.gnn_subgraph_retriever import GNNSubgraphRetriever
from training.train_gnn_subgraph_retriever import (
    compute_adjusted_score,
    encode_example_graph,
    prepare_examples,
    score_candidate_rows,
)


ALLOWED_K = (1, 2, 3, 5)
TRANSITIONS = ((1, 2), (2, 3), (3, 5))
DATASETS = (
    (
        "HotpotQA",
        "HotpotQA",
        "hotpotqa",
        "text",
        "sageqa_text_chain",
        "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
    ),
    (
        "2WikiMultiHopQA",
        "2WikiMultiHopQA",
        "2wiki",
        "text",
        "sageqa_text_chain",
        "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
    ),
    (
        "FamilyOWL_1hop",
        "FamilyOWL_1hop",
        "familyowl_1hop",
        "ontology",
        "sageqa_proof",
        "data/raw/family/FamilyOWL_1hop.json",
    ),
    (
        "FamilyOWL_2hop",
        "FamilyOWL_2hop",
        "familyowl_2hop",
        "ontology",
        "sageqa_proof",
        "data/raw/family/FamilyOWL_2hop.json",
    ),
    (
        "pizza_100_1hop",
        "pizza_100_1hop",
        "pizza_100_1hop",
        "ontology",
        "sageqa_proof",
        "data/raw/pizza_100/pizza_100_1hop.json",
    ),
    (
        "pizza_100_2hop",
        "pizza_100_2hop",
        "pizza_100_2hop",
        "ontology",
        "sageqa_proof",
        "data/raw/pizza_100/pizza_100_2hop.json",
    ),
    (
        "pizza_250_1hop",
        "pizza_250_1hop",
        "pizza_250_1hop",
        "ontology",
        "sageqa_proof",
        "data/raw/pizza_250/pizza_250_1hop.json",
    ),
    (
        "pizza_250_2hop",
        "pizza_250_2hop",
        "pizza_250_2hop",
        "ontology",
        "sageqa_proof",
        "data/raw/pizza_250/pizza_250_2hop.json",
    ),
    (
        "OWL2Bench_1hop",
        "OWL2Bench_1hop",
        "OWL2Bench_1hop",
        "ontology",
        "sageqa_proof",
        "data/raw/owl2bench/OWL2Bench_1hop.json",
    ),
    (
        "OWL2Bench_2hop",
        "OWL2Bench_2hop",
        "OWL2Bench_2hop",
        "ontology",
        "sageqa_proof",
        "data/raw/owl2bench/OWL2Bench_2hop.json",
    ),
)
POLICY_DOMAINS = ("text", "ontology")
RANKING_METHODS = ("gnn_only", "sageqa_final")
FITTED_POLICY_FILES = tuple(
    ["chosen_thresholds.joblib"]
    + [f"{domain}_{kind}.joblib" for domain in POLICY_DOMAINS for kind in ("scaler", "model")]
)
REQUIRED_POLICY_FILES = ("adaptive_k_config.json", *FITTED_POLICY_FILES)
PROHIBITED_TEST_FIELDS = {
    "gold_explanations",
    "gold_units",
    "gold_support_units",
    "raw_supporting_facts",
    "label",
    "rank_target",
    "best_set_f1_to_gold",
    "best_set_precision_to_gold",
    "best_set_recall_to_gold",
    "exact_match_any_gold",
    "contains_any_gold_explanation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def grouped_jsonl(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if not example_id:
                raise ValueError(f"{path}:{line_number}: missing example_id")
            forbidden = PROHIBITED_TEST_FIELDS.intersection(row)
            if forbidden:
                raise ValueError(
                    f"TEST candidate row contains gold-derived fields: {sorted(forbidden)}"
                )
            if row.get("gold_available_during_candidate_generation") is not False:
                raise ValueError(f"TEST row does not assert gold-free generation: {example_id}")
            if row.get("gold_used_during_labeling") not in {False, None}:
                raise ValueError(f"TEST row was gold-labeled: {example_id}")
            if current_id is None:
                current_id = example_id
            if example_id != current_id:
                if example_id in seen:
                    raise ValueError(f"Non-contiguous example: {example_id}")
                seen.add(current_id)
                yield current_id, current
                current_id, current = example_id, []
            current.append(row)
    if current_id is not None:
        yield current_id, current


def verify_manifest(root: Path, required_files: Sequence[str] = ()) -> dict[str, str]:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("files", {})
    if not isinstance(recorded, dict):
        raise ValueError(f"Invalid frozen artifact manifest: {manifest_path}")
    missing = sorted(set(required_files) - set(recorded))
    if missing:
        raise ValueError(f"Frozen artifact manifest omits required files in {root}: {missing}")
    for name, expected in recorded.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Invalid artifact name in frozen manifest: {name!r}")
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen artifact manifest mismatch: {path}")
    return {
        "artifact_manifest.json": sha256(manifest_path),
        **{name: sha256(root / name) for name in recorded},
    }


def _validated_thresholds(value: Any, *, source: Path) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(POLICY_DOMAINS):
        raise ValueError(f"Thresholds must contain exactly text and ontology in {source}")
    thresholds: dict[str, float] = {}
    for domain in POLICY_DOMAINS:
        raw = value[domain]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(f"Invalid {domain} threshold in {source}: {raw!r}")
        threshold = float(raw)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid {domain} threshold in {source}: {raw!r}")
        thresholds[domain] = threshold
    return thresholds


def load_policy_bundle(
    root: Path, expected_method: str
) -> tuple[dict[str, AdaptiveV2Policy], dict[str, Any], dict[str, str]]:
    if expected_method not in RANKING_METHODS:
        raise ValueError(f"Unsupported requested ranking method: {expected_method!r}")
    hashes = verify_manifest(root, REQUIRED_POLICY_FILES)
    config = json.loads((root / "adaptive_k_config.json").read_text(encoding="utf-8"))
    if config.get("ranking_method") != expected_method:
        raise ValueError(f"Wrong policy ranking method in {root}")
    if tuple(config.get("allowed_k", [])) != ALLOWED_K:
        raise ValueError(f"Wrong allowed-k configuration in {root}")
    if tuple(config.get("domains", [])) != POLICY_DOMAINS:
        raise ValueError(f"Wrong pooled policy domains in {root}")
    if config.get("dataset_specific_thresholds") is not False:
        raise ValueError(f"Dataset-specific adaptive thresholds are forbidden in {root}")
    configured_transitions = tuple(
        (x["current_k"], x["continue_to_k"]) for x in config.get("transitions", [])
    )
    if configured_transitions != TRANSITIONS:
        raise ValueError(f"Wrong adaptive transitions in {root}")
    if config.get("feature_order") != list(FEATURE_ORDER):
        raise ValueError(f"Wrong adaptive feature order in {root}")
    if config.get("continuous_features_standardized") != list(CONTINUOUS_FEATURES):
        raise ValueError(f"Wrong adaptive scaler feature order in {root}")
    configured_thresholds = _validated_thresholds(
        config.get("thresholds"), source=root / "adaptive_k_config.json"
    )
    saved_thresholds_raw = joblib.load(root / "chosen_thresholds.joblib")
    saved_thresholds = _validated_thresholds(
        saved_thresholds_raw, source=root / "chosen_thresholds.joblib"
    )
    if (
        saved_thresholds_raw != config.get("thresholds")
        or saved_thresholds != configured_thresholds
    ):
        raise ValueError(f"Saved threshold artifact disagrees with config in {root}")
    fitted_hashes = config.get("fitted_artifact_sha256", {})
    if not isinstance(fitted_hashes, dict) or set(fitted_hashes) != set(FITTED_POLICY_FILES):
        raise ValueError(f"Incomplete fitted artifact hashes in {root}")
    for name, expected_hash in fitted_hashes.items():
        if not isinstance(expected_hash, str) or sha256(root / name) != expected_hash:
            raise ValueError(f"Saved fitted artifact hash mismatch: {root / name}")
    policies = {
        domain: AdaptiveV2Policy(
            domain=domain,
            scaler=joblib.load(root / f"{domain}_scaler.joblib"),
            model=joblib.load(root / f"{domain}_model.joblib"),
            threshold=saved_thresholds[domain],
        )
        for domain in POLICY_DOMAINS
    }
    for domain, policy in policies.items():
        described = config.get("full_dev_model_parameters", {}).get(domain, {})
        if described.get("domain") != domain:
            raise ValueError(f"Loaded policy domain disagrees with frozen config: {root} {domain}")
        if described.get("feature_order") != list(FEATURE_ORDER):
            raise ValueError(f"Model feature order disagrees with frozen config: {root} {domain}")
        if policy.scaler.n_features_in_ != len(CONTINUOUS_FEATURES):
            raise ValueError(f"Loaded scaler has wrong feature count: {root} {domain}")
        if policy.model.n_features_in_ != len(FEATURE_ORDER):
            raise ValueError(f"Loaded model has wrong feature count: {root} {domain}")
        if policy.scaler.mean_.tolist() != described.get("scaler_mean"):
            raise ValueError(f"Loaded scaler mean disagrees with frozen config: {root} {domain}")
        if policy.scaler.scale_.tolist() != described.get("scaler_scale"):
            raise ValueError(f"Loaded scaler scale disagrees with frozen config: {root} {domain}")
        if policy.model.coef_[0].tolist() != described.get("coefficients_in_model_feature_order"):
            raise ValueError(
                f"Loaded model coefficients disagree with frozen config: {root} {domain}"
            )
        described_by_feature = described.get("coefficient_by_feature")
        if described_by_feature != dict(zip(FEATURE_ORDER, policy.model.coef_[0].tolist())):
            raise ValueError(
                f"Loaded model coefficient map disagrees with frozen config: {root} {domain}"
            )
        if float(policy.model.intercept_[0]) != float(described.get("intercept")):
            raise ValueError(
                f"Loaded model intercept disagrees with frozen config: {root} {domain}"
            )
        if policy.model.classes_.astype(int).tolist() != described.get("classes"):
            raise ValueError(f"Loaded model classes disagree with frozen config: {root} {domain}")
        model_config = config.get("model", {})
        model_params = policy.model.get_params()
        for key in ("class_weight", "solver", "max_iter", "random_state"):
            if model_params.get(key) != model_config.get(key):
                raise ValueError(
                    f"Loaded model {key} disagrees with frozen config: {root} {domain}"
                )
    return policies, config, hashes


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = load_gnn_checkpoint(str(checkpoint_path), device)
    model_name = checkpoint.get("model_name", "google/bert_uncased_L-2_H-128_A-2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = GNNSubgraphRetriever(
        model_name=model_name,
        node_symbolic_dim=checkpoint.get("node_symbolic_dim", 8),
        subgraph_symbolic_dim=checkpoint.get("subgraph_symbolic_dim", 8),
        gnn_hidden_dim=checkpoint.get("gnn_hidden_dim", 128),
        gnn_layers=checkpoint.get("gnn_layers", 2),
        classifier_hidden_dim=checkpoint.get("classifier_hidden_dim", 128),
        dropout=0.1,
        freeze_encoder=checkpoint.get("freeze_encoder", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, tokenizer, model


def ranked_view(rows: Sequence[Mapping[str, Any]], score_mode: str) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        item = dict(row)
        item["adjusted_score"] = compute_adjusted_score(
            item, float(item["score"]), score_mode=score_mode, size_penalty=0.01
        )
        ranked.append(item)
    ranked.sort(key=lambda item: float(item["adjusted_score"]), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def adaptive_select(
    ranked: Sequence[Mapping[str, Any]], policy: AdaptiveV2Policy
) -> dict[str, Any]:
    candidates = list(ranked[:5])
    if not candidates:
        raise ValueError("Adaptive selection requires candidates")
    selected_k = 1
    decisions = []
    for current_k, next_k in TRANSITIONS:
        if len(candidates) <= current_k or selected_k != current_k:
            break
        features = compute_decision_features(candidates, decision_rank=current_k)
        probability = policy.continue_probability(features)
        decision = "CONTINUE" if probability >= policy.threshold else "STOP"
        decisions.append(
            {
                "current_k": current_k,
                "continue_to_k": next_k,
                "feature_decision_rank": current_k,
                "predicted_continue_probability": probability,
                "threshold": policy.threshold,
                "decision": decision,
            }
        )
        if decision == "STOP":
            break
        selected_k = next_k
    aggregate = fixed_k_support_aggregate(candidates, k=selected_k)
    return {
        "selected_k": selected_k,
        "actual_candidates_aggregated": int(aggregate["selected_k"]),
        "retrieved_evidence_units": list(aggregate["support_units"]),
        "retrieved_evidence_count": int(aggregate["final_support_size"]),
        "decisions": decisions,
    }


def fixed_select(ranked: Sequence[Mapping[str, Any]], k: int) -> dict[str, Any]:
    aggregate = fixed_k_support_aggregate(ranked, k=k)
    return {
        "selected_k": k,
        "actual_candidates_aggregated": int(aggregate["selected_k"]),
        "retrieved_evidence_units": list(aggregate["support_units"]),
        "retrieved_evidence_count": int(aggregate["final_support_size"]),
    }


def score_and_freeze(
    *,
    test_path: Path,
    checkpoint_path: Path,
    domain: str,
    final_mode: str,
    policies: Mapping[str, Mapping[str, AdaptiveV2Policy]],
    device: torch.device,
    candidate_batch_size: int,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint, tokenizer, model = load_model(checkpoint_path, device)
    predictions = []
    candidate_rows = 0
    for example_index, (example_id, raw_rows) in enumerate(grouped_jsonl(test_path), start=1):
        candidate_rows += len(raw_rows)
        prepared = prepare_examples(raw_rows, candidate_selection="inference")
        if len(prepared) != 1 or prepared[0]["example_id"] != example_id:
            raise ValueError(f"Unexpected inference preparation for {example_id}")
        example = prepared[0]
        encoded = encode_example_graph(model, tokenizer, example, device, max_length=max_length)
        scored = []
        with torch.no_grad():
            for start in range(0, len(example["candidate_rows"]), candidate_batch_size):
                batch = example["candidate_rows"][start : start + candidate_batch_size]
                probs = (
                    score_candidate_rows(model, encoded, batch, device)["probs"]
                    .detach()
                    .cpu()
                    .tolist()
                )
                for row, probability in zip(batch, probs):
                    item = dict(row)
                    item["score"] = float(probability)
                    scored.append(item)
        methods = {}
        for method, mode in (("gnn_only", "neural"), ("sageqa_final", final_mode)):
            ranked = ranked_view(scored, mode)
            methods[method] = {
                "score_mode": mode,
                "k1": fixed_select(ranked, 1),
                "k3": fixed_select(ranked, 3),
                "adaptive": adaptive_select(ranked, policies[method][domain]),
            }
        predictions.append({"example_id": example_id, "methods": methods})
        if example_index % 100 == 0:
            print(f"{test_path.parent.name}: froze {example_index} TEST examples", flush=True)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, {
        "candidate_rows": candidate_rows,
        "prediction_examples": len(predictions),
        "checkpoint_model_name": checkpoint.get("model_name"),
        "checkpoint_training_objective": checkpoint.get("training_objective"),
        "checkpoint_score_mode": checkpoint.get("score_mode"),
    }


def text_gold(source: Path, dataset: str, example_ids: set[str]) -> dict[str, list[list[str]]]:
    # Explicitly exclude answer and, critically for 2Wiki, evidences.
    frame = pd.read_parquet(source, columns=["id", "context", "supporting_facts"])
    normalize = normalize_2wiki_record if dataset == "2WikiMultiHopQA" else normalize_hotpot_record
    flatten = flatten_2wiki_context if dataset == "2WikiMultiHopQA" else flatten_hotpot_context
    gold_fn = get_2wiki_gold if dataset == "2WikiMultiHopQA" else get_hotpot_gold
    prefix = f"{dataset}__test__"
    result = {}
    for raw in frame.to_dict(orient="records"):
        raw_id = str(raw.get("id") or "")
        example_id = prefix + raw_id
        if example_id not in example_ids:
            continue
        example = normalize(raw)
        lookup = {(row["title"], int(row["sent_idx"])): row["unit"] for row in flatten(example)}
        support = gold_fn(example, lookup)
        result[example_id] = [support] if support else []
    return result


def ontology_gold(source: Path, example_ids: set[str]) -> dict[str, list[list[str]]]:
    groups = json.loads(source.read_text(encoding="utf-8"))
    result = {}
    for example_id in example_ids:
        parts = example_id.split("__", 3)
        if len(parts) < 3 or not parts[1].startswith("g") or not parts[2].startswith("q"):
            raise ValueError(f"Cannot parse ontology example ID: {example_id}")
        group_index, qa_index = int(parts[1][1:]), int(parts[2][1:])
        result[example_id] = get_gold_explanations(groups[group_index]["QAs"][qa_index])
    return result


def evidence_scores(predicted: Sequence[Any], gold: Sequence[Any]) -> dict[str, float]:
    predicted_set, gold_set = set(map(str, predicted)), set(map(str, gold))
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def best_scores(predicted: Sequence[Any], golds: Sequence[Sequence[Any]]) -> dict[str, float]:
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for gold in golds:
        current = evidence_scores(predicted, gold)
        if current["f1"] > best["f1"]:
            best = current
    return best


def summarize_setting(rows: Sequence[Mapping[str, Any]], adaptive: bool = False) -> dict[str, Any]:
    result = {
        metric: statistics.fmean(float(row[metric]) for row in rows)
        for metric in ("precision", "recall", "f1")
    }
    if adaptive:
        counts = Counter(int(row["selected_k"]) for row in rows)
        result.update(
            {
                "mean_selected_k": statistics.fmean(int(row["selected_k"]) for row in rows),
                "median_selected_k": statistics.median(int(row["selected_k"]) for row in rows),
                "selected_k_distribution": {
                    str(k): {
                        "count": counts[k],
                        "percentage": 100.0 * counts[k] / len(rows),
                    }
                    for k in ALLOWED_K
                },
            }
        )
    return result


def summary_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen production Generator D — final TEST retrieval",
        "",
        "Macro P/R/F1 over support-bearing TEST examples. Each prediction is a deduplicated evidence-unit union.",
        "",
        "| Dataset | Method | Setting | P | R | F1 | Mean k | Median k | k=1 | k=2 | k=3 | k=5 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in metrics["datasets"]:
        for method in ("gnn_only", "sageqa_final"):
            for setting in ("k1", "k3", "adaptive"):
                row = dataset["methods"][method][setting]
                if setting == "adaptive":
                    dist = row["selected_k_distribution"]
                    extras = (
                        f"{row['mean_selected_k']:.4f} | {row['median_selected_k']:.1f} | "
                        + " | ".join(f"{dist[str(k)]['percentage']:.2f}%" for k in ALLOWED_K)
                    )
                else:
                    extras = "— | — | — | — | — | —"
                lines.append(
                    f"| {dataset['dataset']} | {method} | {setting} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {extras} |"
                )
    lines.extend(
        [
            "",
            "## F1 comparisons",
            "",
            "| Dataset | SAGE k1 − GNN k1 | SAGE k3 − GNN k3 | SAGE adaptive − GNN adaptive | GNN adaptive − k1 | GNN adaptive − k3 | SAGE adaptive − k1 | SAGE adaptive − k3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in metrics["datasets"]:
        d = dataset["f1_differences"]
        lines.append(
            f"| {dataset['dataset']} | {d['sageqa_k1_minus_gnn_k1']:+.6f} | {d['sageqa_k3_minus_gnn_k3']:+.6f} | {d['sageqa_adaptive_minus_gnn_adaptive']:+.6f} | {d['gnn_adaptive_minus_k1']:+.6f} | {d['gnn_adaptive_minus_k3']:+.6f} | {d['sageqa_adaptive_minus_k1']:+.6f} | {d['sageqa_adaptive_minus_k3']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen protocol confirmation",
            "",
            "Candidate generation was not rerun; GNNs were not retrained; adaptive policies were loaded from their saved scaler/model/threshold artifacts and were not refitted; Text-Chain and Proof reranking were unchanged; no TEST result altered any method or configuration; answer generation was not run.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/production_generator_d_v1"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints/production_generator_d_v1")
    )
    parser.add_argument(
        "--sageqa-policy-dir",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_adaptive_k"),
    )
    parser.add_argument(
        "--gnn-policy-dir",
        type=Path,
        default=Path("outputs/development_runs/production_generator_d_v1_gnn_adaptive_k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_results/production_generator_d_v1_test_retrieval"),
    )
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite final result: {args.output_dir}")
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(Path(__file__).relative_to(root)), *sys.argv[1:]]
    commit = git_output("rev-parse", "HEAD")
    status_before = git_output("status", "--short").splitlines()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sage_policies, sage_config, sage_hashes = load_policy_bundle(
        args.sageqa_policy_dir, "sageqa_final"
    )
    gnn_policies, gnn_config, gnn_hashes = load_policy_bundle(args.gnn_policy_dir, "gnn_only")
    policies = {"sageqa_final": sage_policies, "gnn_only": gnn_policies}

    corpus_files = sorted(path for path in args.data_root.rglob("*") if path.is_file())
    corpus_manifest = [
        {
            "path": str(path.relative_to(args.data_root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in corpus_files
    ]
    corpus_manifest_hash = canonical_hash(corpus_manifest)
    reranker_path = root / "training/train_gnn_subgraph_retriever.py"
    evaluator_path = Path(__file__).resolve()
    input_paths = [*corpus_files, reranker_path]
    checkpoint_paths, gold_paths = [], []
    for _, data_dir, checkpoint_dir, _, _, gold_relative in DATASETS:
        test_path = args.data_root / data_dir / "test_subgraph_retrieval.jsonl"
        checkpoint_path = args.checkpoint_root / checkpoint_dir / "best_model.pt"
        gold_path = root / gold_relative
        if not test_path.is_file() or not checkpoint_path.is_file() or not gold_path.is_file():
            raise FileNotFoundError(f"Missing frozen input for {data_dir}")
        checkpoint_paths.append(checkpoint_path)
        gold_paths.append(gold_path)
    input_paths.extend(checkpoint_paths)
    input_paths.extend(gold_paths)
    input_paths.extend(
        path
        for policy_dir in (args.sageqa_policy_dir, args.gnn_policy_dir)
        for path in policy_dir.iterdir()
        if path.is_file()
    )
    hashes_before = {str(path): sha256(path) for path in input_paths}

    # Phase 1: score and freeze every selection. No gold source is opened here.
    frozen: dict[str, list[dict[str, Any]]] = {}
    dataset_lineage: dict[str, Any] = {}
    for dataset, data_dir, checkpoint_dir, domain, final_mode, gold_relative in DATASETS:
        test_path = args.data_root / data_dir / "test_subgraph_retrieval.jsonl"
        checkpoint_path = args.checkpoint_root / checkpoint_dir / "best_model.pt"
        rows, run_info = score_and_freeze(
            test_path=test_path,
            checkpoint_path=checkpoint_path,
            domain=domain,
            final_mode=final_mode,
            policies=policies,
            device=device,
            candidate_batch_size=args.candidate_batch_size,
            max_length=args.max_length,
        )
        frozen[dataset] = rows
        dataset_lineage[dataset] = {
            "domain": domain,
            "test_candidate_path": str(test_path),
            "test_candidate_sha256": sha256(test_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "gold_source_path": gold_relative,
            "gold_source_sha256": sha256(root / gold_relative),
            "sageqa_score_mode": final_mode,
            "gnn_score_mode": "neural",
            "inference_candidate_cap": 320,
            **run_info,
        }

    predictions_frozen_at = datetime.now(timezone.utc).isoformat()

    # Phase 2: only now open gold annotations and evaluate the frozen supports.
    per_example = []
    metric_datasets = []
    for dataset, _, _, domain, _, gold_relative in DATASETS:
        predictions = frozen[dataset]
        ids = {row["example_id"] for row in predictions}
        gold_path = root / gold_relative
        gold_by_id = (
            text_gold(gold_path, dataset, ids)
            if domain == "text"
            else ontology_gold(gold_path, ids)
        )
        if set(gold_by_id) != ids:
            missing = sorted(ids - set(gold_by_id))
            raise ValueError(f"Gold join incomplete for {dataset}: {missing[:3]}")
        setting_rows = {
            method: {setting: [] for setting in ("k1", "k3", "adaptive")}
            for method in ("gnn_only", "sageqa_final")
        }
        excluded = 0
        for prediction in predictions:
            golds = gold_by_id[prediction["example_id"]]
            eligible = bool(golds)
            if not eligible:
                excluded += 1
            record = {
                "dataset": dataset,
                "domain": domain,
                "split": "test",
                "example_id": prediction["example_id"],
                "evaluation_eligible": eligible,
                "exclusion_reason": None if eligible else "no_gold_support_explanation",
                "gold_explanations": golds,
                "methods": prediction["methods"],
            }
            for method in ("gnn_only", "sageqa_final"):
                for setting in ("k1", "k3", "adaptive"):
                    selected = record["methods"][method][setting]
                    scores = (
                        best_scores(selected["retrieved_evidence_units"], golds)
                        if eligible
                        else {"precision": None, "recall": None, "f1": None}
                    )
                    selected.update(scores)
                    if eligible:
                        setting_rows[method][setting].append(selected)
            per_example.append(record)
        evaluated = len(predictions) - excluded
        if not evaluated:
            raise ValueError(f"No support-bearing TEST examples for {dataset}")
        methods = {
            method: {
                setting: summarize_setting(
                    setting_rows[method][setting], adaptive=setting == "adaptive"
                )
                for setting in ("k1", "k3", "adaptive")
            }
            for method in ("gnn_only", "sageqa_final")
        }
        gnn, sage = methods["gnn_only"], methods["sageqa_final"]
        differences = {
            "sageqa_k1_minus_gnn_k1": sage["k1"]["f1"] - gnn["k1"]["f1"],
            "sageqa_k3_minus_gnn_k3": sage["k3"]["f1"] - gnn["k3"]["f1"],
            "sageqa_adaptive_minus_gnn_adaptive": sage["adaptive"]["f1"] - gnn["adaptive"]["f1"],
            "gnn_adaptive_minus_k1": gnn["adaptive"]["f1"] - gnn["k1"]["f1"],
            "gnn_adaptive_minus_k3": gnn["adaptive"]["f1"] - gnn["k3"]["f1"],
            "sageqa_adaptive_minus_k1": sage["adaptive"]["f1"] - sage["k1"]["f1"],
            "sageqa_adaptive_minus_k3": sage["adaptive"]["f1"] - sage["k3"]["f1"],
        }
        metric_datasets.append(
            {
                "dataset": dataset,
                "domain": domain,
                "prediction_examples": len(predictions),
                "evaluation_examples": evaluated,
                "excluded_no_gold_support": excluded,
                "methods": methods,
                "f1_differences": differences,
            }
        )

    hashes_after = {str(path): sha256(path) for path in input_paths}
    if hashes_before != hashes_after:
        raise RuntimeError(
            "A frozen corpus/checkpoint/policy/reranker input changed during evaluation"
        )
    metrics = {
        "schema_version": "production_generator_d_v1_final_test_retrieval_v1",
        "status": "complete_frozen_one_time_test_retrieval_evaluation",
        "split": "test",
        "metric_aggregation": "macro mean over support-bearing examples",
        "support_aggregation": "order-preserving deduplicated union of ranked candidate evidence units",
        "adaptive_allowed_k": list(ALLOWED_K),
        "datasets": metric_datasets,
    }
    completed_at = datetime.now(timezone.utc).isoformat()
    lineage = {
        "schema_version": "production_generator_d_v1_final_test_retrieval_lineage_v1",
        "status": metrics["status"],
        "started_from_code_commit": commit,
        "evaluator_sha256": sha256(evaluator_path),
        "git_status_before": status_before,
        "git_status_after": git_output("status", "--short").splitlines(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "evaluation_command_argv": command,
        "evaluation_command_display": subprocess.list2cmdline(command),
        "configuration": {
            "data_root": str(args.data_root),
            "checkpoint_root": str(args.checkpoint_root),
            "sageqa_policy_dir": str(args.sageqa_policy_dir),
            "gnn_policy_dir": str(args.gnn_policy_dir),
            "output_dir": str(args.output_dir),
            "candidate_batch_size": args.candidate_batch_size,
            "max_length": args.max_length,
            "fixed_k": [1, 3],
            "adaptive_allowed_k": list(ALLOWED_K),
            "adaptive_transitions": [
                {"current_k": a, "continue_to_k": b, "feature_decision_rank": a}
                for a, b in TRANSITIONS
            ],
            "sageqa_thresholds": sage_config["thresholds"],
            "gnn_thresholds": gnn_config["thresholds"],
            "size_penalty_argument": 0.01,
        },
        "generator_d_version": "generator_d_frozen_v1",
        "production_builder_version": "generator_d_production_v1",
        "corpus_manifest": corpus_manifest,
        "corpus_manifest_sha256": corpus_manifest_hash,
        "checkpoint_hashes": {
            dataset: values["checkpoint_sha256"] for dataset, values in dataset_lineage.items()
        },
        "adaptive_policy_artifact_hashes": {"sageqa_final": sage_hashes, "gnn_only": gnn_hashes},
        "reranker_implementation": {
            "path": str(reranker_path.relative_to(root)),
            "sha256": sha256(reranker_path),
            "text_score_mode": "sageqa_text_chain",
            "ontology_score_mode": "sageqa_proof",
        },
        "datasets": dataset_lineage,
        "gold_boundary": {
            "all_predictions_and_selected_supports_frozen_before_any_gold_source_opened": True,
            "predictions_frozen_at_utc": predictions_frozen_at,
            "gold_access_started_after_predictions_frozen": True,
            "two_wiki_columns_read_post_freeze": ["id", "context", "supporting_facts"],
            "two_wiki_evidences_read": False,
            "two_wiki_answer_read": False,
        },
        "frozen_protocol_confirmations": {
            "candidate_generation_rerun": False,
            "gnns_retrained": False,
            "adaptive_policies_refitted": False,
            "adaptive_thresholds_changed": False,
            "text_chain_changed": False,
            "proof_reranking_changed": False,
            "reranker_weights_unchanged": True,
            "test_result_changed_method_or_configuration": False,
            "answer_generation_run": False,
            "manuscript_updated": False,
            "commit_or_push_performed": False,
            "all_frozen_inputs_hash_unchanged_during_run": True,
        },
        "completed_at_utc": completed_at,
    }

    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.json"
    per_example_path = args.output_dir / "per_example_test_retrieval.jsonl"
    summary_path = args.output_dir / "summary.md"
    lineage_path = args.output_dir / "lineage_metadata.json"
    write_json(metrics_path, metrics)
    with per_example_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in per_example:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(summary_markdown(metrics), encoding="utf-8", newline="\n")
    write_json(lineage_path, lineage)
    files = (metrics_path, per_example_path, summary_path, lineage_path)
    write_json(
        args.output_dir / "artifact_manifest.json",
        {
            "schema_version": "production_generator_d_v1_final_test_retrieval_manifest_v1",
            "status": metrics["status"],
            "self_excluded_from_hashes": True,
            "files": {
                path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in files
            },
        },
    )
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "complete": True, "datasets": len(metric_datasets)}
        )
    )


if __name__ == "__main__":
    main()
