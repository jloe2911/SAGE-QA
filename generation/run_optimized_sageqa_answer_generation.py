"""Gold-free reader preflight and generation for optimized SAGE-QA TEST support.

Only the frozen ``k1`` and ``adaptive`` supports from the optimized symbolic-
coefficient experiment are admitted.  ``preflight`` verifies their prediction
freeze, constructs immutable reader inputs, and inventories exact historical
reader-input matches without opening retrieval metrics, scored retrieval rows,
or benchmark answer/support gold.  ``generate`` reuses those validated answers
before scheduling deterministic OWL proofs or GPT-4.1-mini calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation import run_production_test_answer_generation as production
from utils.llm_client import load_local_env
from utils.paths import generator_d_root, outputs_root, repo_path_arg, repo_root


ROOT = repo_root()
MODEL_NAME = production.MODEL_NAME
METHOD = "sageqa_optimized"
CONFIGURATIONS = ((METHOD, "k1"), (METHOD, "adaptive"))
SETTINGS = tuple(setting for _method, setting in CONFIGURATIONS)
EXPECTED_DATASET_COUNTS = {
    "HotpotQA": 1000,
    "2WikiMultiHopQA": 1000,
    "FamilyOWL_1hop": 462,
    "FamilyOWL_2hop": 462,
    "pizza_100_1hop": 119,
    "pizza_100_2hop": 125,
    "pizza_250_1hop": 149,
    "pizza_250_2hop": 157,
    "OWL2Bench_1hop": 376,
    "OWL2Bench_2hop": 399,
}
EXPECTED_EXAMPLES = sum(EXPECTED_DATASET_COUNTS.values())
DEFAULT_SELECTION_ROOT = Path(
    r"C:\Users\julie\github\PhD\SAGEQA_GPU_handoff\outputs\final_results"
    r"\symbolic_coefficients_original_v1_test"
)
DEFAULT_CANDIDATE_ROOT = generator_d_root()
DEFAULT_OUTPUT = outputs_root() / "final_results/symbolic_coefficients_original_v1_test_end_to_end"
DEFAULT_SUPPORT_GOLD = (
    outputs_root()
    / "final_results/production_generator_d_v2_hard_pair_test_retrieval/per_example_test_retrieval.jsonl"
)
DEFAULT_HISTORICAL_ROOTS = (
    outputs_root() / "final_results/final_manuscript_test_end_to_end",
    outputs_root() / "final_results/production_generator_d_v2_hard_pair_test_end_to_end",
    outputs_root() / "final_results/final_manuscript_baselines_test_end_to_end",
    outputs_root() / "final_results/gold_support_complete_oracle",
)
CANONICAL_READER_HASH_SOURCE = DEFAULT_HISTORICAL_ROOTS[1] / "generation_freeze.json"
READER_CACHE_FIELDS = (
    "dataset",
    "example_id",
    "question",
    "support_sha256",
    "model",
    "domain",
    "reader_type",
    "task_type",
    "sparql_query",
    "answer_type",
)
HISTORICAL_OUTPUT_FIELDS = (
    "predicted_answer",
    "explanation",
    "raw_response",
    "answer_source",
)


def _manifest_hash(manifest: Mapping[str, Any], name: str) -> str:
    files = manifest.get("files", {})
    if isinstance(files, dict):
        value = files.get(name)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("sha256"), str):
            return str(value["sha256"])
    if isinstance(files, list):
        for value in files:
            if Path(str(value.get("path", ""))).name == name:
                return str(value.get("sha256", ""))
    raise ValueError(f"Manifest has no SHA-256 for {name}")


def _reader_type(domain: str) -> str:
    return "text_reader_v1" if domain == "text" else "ontology_proof_first_reader_v1"


def reader_cache_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete established reader identity, excluding method labels."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else row
    support = row.get("selected_support")
    if not isinstance(support, list) or not all(isinstance(value, str) for value in support):
        raise ValueError(f"Invalid ordered support for {row.get('example_id')}")
    stored_hash = str(row.get("support_sha256") or row.get("frozen_support_sha256") or "")
    actual_hash = production.canonical_hash(support)
    if stored_hash and stored_hash != actual_hash:
        raise ValueError(f"Support hash mismatch for {row.get('example_id')}")
    domain = str(row.get("domain") or "")
    return {
        "dataset": str(row.get("dataset") or ""),
        "example_id": str(row.get("example_id") or ""),
        "question": str(row.get("question") or metadata.get("question") or ""),
        "support_sha256": actual_hash,
        "model": str(row.get("model") or MODEL_NAME),
        "domain": domain,
        "reader_type": _reader_type(domain),
        "task_type": metadata.get("task_type"),
        "sparql_query": metadata.get("sparql_query"),
        "answer_type": metadata.get("answer_type"),
    }


def reader_cache_key(row: Mapping[str, Any]) -> str:
    contract = reader_cache_contract(row)
    missing = [field for field in READER_CACHE_FIELDS[:7] if not contract[field]]
    if missing:
        raise ValueError(f"Incomplete reader contract for {row.get('example_id')}: {missing}")
    return production.canonical_hash(contract)


def _reader_configuration() -> dict[str, Any]:
    canonical = json.loads(CANONICAL_READER_HASH_SOURCE.read_text(encoding="utf-8"))
    expected = canonical.get("code_sha256", {})
    current = {
        "text_reader": production.sha256(Path(production.text_reader_module.__file__).resolve()),
        "ontology_reader": production.sha256(
            Path(production.ontology_reader_module.__file__).resolve()
        ),
    }
    for name, digest in current.items():
        if digest != expected.get(name):
            raise ValueError(f"Frozen {name} implementation changed: {digest}")
    return {
        "model": MODEL_NAME,
        "text": {
            "reader_type": "text_reader_v1",
            "temperature": 0.0,
            "implementation_sha256": current["text_reader"],
        },
        "ontology": {
            "reader_type": "ontology_proof_first_reader_v1",
            "fallback_temperature": 0.0,
            "fallback_max_tokens": 300,
            "implementation_sha256": current["ontology_reader"],
        },
        "cache_key_fields": list(READER_CACHE_FIELDS),
        "canonical_hash_source": str(CANONICAL_READER_HASH_SOURCE),
    }


def _validate_optimized_source(selection_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = selection_root / "test_predictions_frozen.jsonl"
    freeze_path = selection_root / "generation_freeze.json"
    manifest_path = selection_root / "artifact_manifest.json"
    preflight_path = selection_root / "preflight_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    prediction_hash = production.sha256(predictions)
    if manifest.get("status") != "complete_frozen":
        raise ValueError("Optimized artifact manifest is not complete_frozen")
    if prediction_hash != _manifest_hash(manifest, predictions.name):
        raise ValueError("Optimized predictions differ from artifact_manifest.json")
    if prediction_hash != freeze.get("predictions_sha256"):
        raise ValueError("Optimized predictions differ from generation_freeze.json")
    if production.sha256(freeze_path) != _manifest_hash(manifest, freeze_path.name):
        raise ValueError("Optimized generation freeze differs from artifact_manifest.json")
    if production.sha256(preflight_path) != _manifest_hash(manifest, preflight_path.name):
        raise ValueError("Optimized preflight differs from artifact_manifest.json")
    if freeze.get("status") != "complete_frozen_before_gold_join":
        raise ValueError("Optimized predictions were not frozen before the gold join")
    if freeze.get("gold_accessed_before_freeze") is not False:
        raise ValueError("Optimized freeze does not attest a gold-blind prediction phase")
    if source_preflight.get("gold_source_files_opened_for_content") != 0:
        raise ValueError("Optimized preflight does not attest zero gold-source reads")

    rows = production.load_jsonl(predictions)
    counts = Counter(str(row.get("dataset") or "") for row in rows)
    if len(rows) != EXPECTED_EXAMPLES or dict(counts) != EXPECTED_DATASET_COUNTS:
        raise ValueError(f"Unexpected optimized dataset population: {dict(counts)}")
    seen: set[str] = set()
    for row in rows:
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in seen:
            raise ValueError(f"Missing or duplicate optimized example_id: {example_id!r}")
        seen.add(example_id)
        if row.get("split") != "test" or row.get("dataset") not in EXPECTED_DATASET_COUNTS:
            raise ValueError(f"Invalid optimized row identity: {example_id}")
        for setting in SETTINGS:
            selected = row.get(setting)
            support = selected.get("retrieved_evidence_units") if isinstance(selected, dict) else None
            if not isinstance(support, list) or not all(isinstance(unit, str) for unit in support):
                raise ValueError(f"Invalid optimized support for {example_id}/{setting}")
    return rows, {
        "prediction_sha256": prediction_hash,
        "generation_freeze_sha256": production.sha256(freeze_path),
        "artifact_manifest_sha256": production.sha256(manifest_path),
        "source_preflight_sha256": production.sha256(preflight_path),
        "checkpoint_sha256": freeze.get("checkpoint_sha256"),
        "selected_coefficients_sha256": freeze.get("selected_coefficients_sha256"),
        "adaptive_policy_manifest_sha256": freeze.get("adaptive_policy_manifest_sha256"),
        "dataset_counts": dict(counts),
        "gold_bearing_source_files_opened": [],
    }


def _candidate_hashes_from_source_preflight(selection_root: Path) -> dict[str, str]:
    value = json.loads((selection_root / "preflight_manifest.json").read_text(encoding="utf-8"))
    return {
        dataset: str(details["candidate_sha256"])
        for dataset, details in value.get("datasets", {}).items()
    }


def _build_reader_inputs(
    rows: Sequence[Mapping[str, Any]], candidate_root: Path, expected_candidate_hashes: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata, lineage = production.load_generation_metadata(candidate_root, rows)
    for dataset, details in lineage.items():
        if details["candidate_sha256"] != expected_candidate_hashes.get(dataset):
            raise ValueError(f"Candidate hash differs from optimized preflight for {dataset}")
    result: list[dict[str, Any]] = []
    for row in rows:
        example_id, dataset = str(row["example_id"]), str(row["dataset"])
        item_metadata = metadata[example_id]
        domain = production.DATASET_INFO[dataset][1]
        for setting in SETTINGS:
            support = list(row[setting]["retrieved_evidence_units"])
            item = {
                "schema_version": "optimized_sageqa_reader_input_v1",
                "dataset": dataset,
                "domain": domain,
                "split": "test",
                "example_id": example_id,
                "method": METHOD,
                "setting": setting,
                "question": item_metadata["question"],
                "metadata": item_metadata,
                "selected_support": support,
                "support_sha256": production.canonical_hash(support),
                "model": MODEL_NAME,
            }
            item["reader_cache_contract"] = reader_cache_contract(item)
            item["reader_cache_key_sha256"] = reader_cache_key(item)
            result.append(item)
    return result, lineage


def _historical_input_metadata(root: Path, freeze: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    inputs_path = root / "frozen_reader_inputs.jsonl"
    expected_hash = str(freeze.get("reader_inputs_sha256") or "")
    if not inputs_path.is_file():
        return {}
    if not expected_hash or production.sha256(inputs_path) != expected_hash:
        raise ValueError(f"Historical reader-input freeze is invalid: {root}")
    return {
        production.prediction_key(row): row
        for row in production.load_jsonl(inputs_path)
    }


def _load_historical_cache(
    roots: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    cache: dict[str, dict[str, Any]] = {}
    lineage: list[dict[str, Any]] = []
    duplicate_matches = 0
    for root in roots:
        freeze_path, predictions_path = root / "generation_freeze.json", root / "predictions.jsonl"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        actual_hash = production.sha256(predictions_path)
        if freeze.get("status") != "complete_frozen" or freeze.get("model") != MODEL_NAME:
            raise ValueError(f"Incompatible historical generation freeze: {root}")
        if actual_hash != freeze.get("predictions_sha256"):
            raise ValueError(f"Historical predictions changed after freeze: {root}")
        rows = production.load_jsonl(predictions_path)
        if len(rows) != int(freeze.get("prediction_count", -1)):
            raise ValueError(f"Historical prediction count mismatch: {root}")
        frozen_inputs = _historical_input_metadata(root, freeze)
        admitted = 0
        for row in rows:
            if row.get("generation_status") != "complete" or row.get("model") != MODEL_NAME:
                raise ValueError(f"Incomplete historical prediction in {root}")
            source = frozen_inputs.get(production.prediction_key(row), row)
            merged = dict(row)
            if isinstance(source.get("metadata"), Mapping):
                merged["metadata"] = source["metadata"]
            key = reader_cache_key(merged)
            value = {
                **{field: row.get(field) for field in HISTORICAL_OUTPUT_FIELDS},
                "historical_source": str(predictions_path),
                "historical_prediction_key": list(production.prediction_key(row)),
            }
            if key in cache:
                duplicate_matches += 1
                continue
            cache[key] = value
            admitted += 1
        lineage.append(
            {
                "path": str(predictions_path),
                "predictions_sha256": actual_hash,
                "prediction_count": len(rows),
                "unique_reader_inputs_admitted": admitted,
            }
        )
    return cache, lineage, duplicate_matches


def _dependency_report() -> dict[str, Any]:
    load_local_env()
    openai_installed = importlib.util.find_spec("openai") is not None
    key_present = bool(os.getenv("OPENAI_API_KEY"))
    missing = []
    if not openai_installed:
        missing.append("Python package 'openai' (install requirements.txt)")
    if not key_present:
        missing.append("OPENAI_API_KEY (required only for the later generate phase)")
    return {
        "python_openai_installed": openai_installed,
        "openai_api_key_present": key_present,
        "missing": missing,
        "network_or_api_calls_made": 0,
    }


def preflight_phase(
    selection_root: Path = DEFAULT_SELECTION_ROOT,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
    historical_roots: Sequence[Path] = DEFAULT_HISTORICAL_ROOTS,
) -> dict[str, Any]:
    reader_configuration = _reader_configuration()
    optimized_rows, source_lineage = _validate_optimized_source(selection_root)
    source_hash_before = source_lineage["prediction_sha256"]
    reader_inputs, candidate_lineage = _build_reader_inputs(
        optimized_rows,
        candidate_root,
        _candidate_hashes_from_source_preflight(selection_root),
    )
    historical, historical_lineage, duplicate_history = _load_historical_cache(historical_roots)
    unique_inputs: dict[str, dict[str, Any]] = {}
    rows_by_key: Counter[str] = Counter()
    for row in reader_inputs:
        key = str(row["reader_cache_key_sha256"])
        unique_inputs.setdefault(key, row)
        rows_by_key[key] += 1

    reusable_keys = set(unique_inputs) & set(historical)
    unmatched = {key: row for key, row in unique_inputs.items() if key not in historical}
    deterministic_keys: set[str] = set()
    for key, row in unmatched.items():
        if row["domain"] != "ontology":
            continue
        proof = production.ontology_reader_module.infer_owl_boolean_answer(
            item=dict(row["metadata"]), support_units=list(row["selected_support"])
        )
        if proof is not None:
            deterministic_keys.add(key)
    api_keys = set(unmatched) - deterministic_keys
    reuse_plan = [
        {
            "reader_cache_key_sha256": key,
            "historical_source": historical[key]["historical_source"],
            "historical_prediction_key": historical[key]["historical_prediction_key"],
            "target_prediction_rows": rows_by_key[key],
        }
        for key in sorted(reusable_keys)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path, reuse_path = output_dir / "frozen_reader_inputs.jsonl", output_dir / "reuse_plan.jsonl"
    production.write_jsonl(inputs_path, reader_inputs)
    production.write_jsonl(reuse_path, reuse_plan)
    report = {
        "schema_version": "optimized_sageqa_answer_preflight_v1",
        "status": "complete_gold_free_no_api_calls",
        "configurations": [{"method": method, "setting": setting} for method, setting in CONFIGURATIONS],
        "dataset_counts": source_lineage["dataset_counts"],
        "examples": len(optimized_rows),
        "prediction_rows": len(reader_inputs),
        "unique_reader_inputs": len(unique_inputs),
        "reusable_historical_prediction_rows": sum(rows_by_key[key] for key in reusable_keys),
        "reusable_historical_unique_answers": len(reusable_keys),
        "new_local_deterministic_answers_required": len(deterministic_keys),
        "new_api_calls_required": len(api_keys),
        "new_prediction_rows_after_reuse": len(reader_inputs)
        - sum(rows_by_key[key] for key in reusable_keys),
        "duplicate_historical_cache_entries_ignored": duplicate_history,
        "source_lineage": source_lineage,
        "candidate_lineage": candidate_lineage,
        "historical_lineage": historical_lineage,
        "reader_configuration": reader_configuration,
        "frozen_reader_inputs_sha256": production.sha256(inputs_path),
        "reuse_plan_sha256": production.sha256(reuse_path),
        "dependencies": _dependency_report(),
        "test_gold_opened": False,
        "gold_bearing_retrieval_files_opened": [],
        "openai_calls_made": 0,
    }
    production.write_json(output_dir / "preflight_report.json", report)
    production.write_json(
        output_dir / "artifact_manifest.json",
        {
            "status": "preflight_complete_gold_free_no_api_calls",
            "files": {
                "frozen_reader_inputs.jsonl": production.sha256(inputs_path),
                "reuse_plan.jsonl": production.sha256(reuse_path),
                "preflight_report.json": production.sha256(output_dir / "preflight_report.json"),
            },
        },
    )
    if production.sha256(selection_root / "test_predictions_frozen.jsonl") != source_hash_before:
        raise ValueError("Optimized frozen predictions changed during preflight")
    return report


def _expected_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    return {production.prediction_key(row) for row in rows}


def _result_from_history(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value.get(field) for field in HISTORICAL_OUTPUT_FIELDS}


def generate_phase(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    historical_roots: Sequence[Path] = DEFAULT_HISTORICAL_ROOTS,
    workers: int = 8,
    resume: bool = False,
    text_reader: production.TextReader = production.default_text_reader,
    ontology_reader: production.OntologyReader = production.default_ontology_reader,
) -> dict[str, Any]:
    """Reuse frozen answers, then run only unmatched inputs. This may call OpenAI."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    report = json.loads((output_dir / "preflight_report.json").read_text(encoding="utf-8"))
    inputs_path = output_dir / "frozen_reader_inputs.jsonl"
    if production.sha256(inputs_path) != report.get("frozen_reader_inputs_sha256"):
        raise ValueError("Frozen optimized reader inputs changed after preflight")
    inputs = production.load_jsonl(inputs_path)
    expected = _expected_keys(inputs)
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists() and not resume:
        raise FileExistsError("predictions.jsonl exists; use --resume")
    existing, completed = production._validated_existing_predictions(predictions_path, expected)
    history, _lineage, _duplicates = _load_historical_cache(historical_roots)
    jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inputs:
        if production.prediction_key(row) not in completed:
            jobs[reader_cache_key(row)].append(dict(row))
    for row in existing:
        history.setdefault(reader_cache_key(row), _result_from_history(row))

    def persist(cache_key: str, result: Mapping[str, Any], reused_from: str | None = None) -> None:
        for source in jobs[cache_key]:
            job = {
                "method": source["method"],
                "setting": source["setting"],
                "example_id": source["example_id"],
                "dataset": source["dataset"],
                "domain": source["domain"],
                "metadata": source["metadata"],
                "support": source["selected_support"],
                "support_hash": source["support_sha256"],
            }
            prediction = production._prediction_row(job, result, MODEL_NAME)
            prediction["reader_cache_key_sha256"] = cache_key
            prediction["historical_reuse_source"] = reused_from
            production.append_jsonl(predictions_path, prediction)
            completed.add(production.prediction_key(source))

    for key in list(jobs):
        if key in history:
            persist(key, _result_from_history(history[key]), history[key].get("historical_source"))
    pending = [key for key in jobs if key not in history]
    errors_path = output_dir / "generation_errors.jsonl"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[Mapping[str, Any]], str] = {}
        for key in pending:
            row = jobs[key][0]
            futures[
                executor.submit(
                    production._run_reader_job,
                    row["domain"],
                    row["metadata"],
                    list(row["selected_support"]),
                    MODEL_NAME,
                    text_reader,
                    ontology_reader,
                )
            ] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                persist(key, future.result())
            except Exception as exc:
                production.append_jsonl(
                    errors_path, {"reader_cache_key_sha256": key, "error": repr(exc)}
                )
    if completed != expected:
        raise RuntimeError(f"Generation incomplete: {len(completed)}/{len(expected)}; use --resume")
    rows, keys = production._validated_existing_predictions(predictions_path, expected)
    if keys != expected or len(rows) != len(expected):
        raise ValueError("Persisted optimized prediction population is incomplete")
    freeze = {
        "schema_version": "optimized_sageqa_answer_generation_freeze_v1",
        "status": "complete_frozen",
        "model": MODEL_NAME,
        "prediction_count": len(rows),
        "predictions_sha256": production.sha256(predictions_path),
        "reader_inputs_sha256": report["frozen_reader_inputs_sha256"],
        "optimized_retrieval_sha256": report["source_lineage"]["prediction_sha256"],
        "configurations": report["configurations"],
        "gold_answers_opened": False,
    }
    production.write_json(output_dir / "generation_freeze.json", freeze)
    return freeze


def evaluate_phase(
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    source_root: Path = ROOT,
    support_gold: Path = DEFAULT_SUPPORT_GOLD,
) -> dict[str, Any]:
    """Evaluate only after the exact 8,498-row prediction freeze is verified."""
    inputs_path, predictions_path = output_dir / "frozen_reader_inputs.jsonl", output_dir / "predictions.jsonl"
    freeze = json.loads((output_dir / "generation_freeze.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "complete_frozen":
        raise ValueError("Complete optimized prediction freeze is missing")
    if production.sha256(predictions_path) != freeze.get("predictions_sha256"):
        raise ValueError("Optimized predictions changed after freeze")
    if production.sha256(inputs_path) != freeze.get("reader_inputs_sha256"):
        raise ValueError("Optimized reader inputs changed after freeze")
    inputs = production.load_jsonl(inputs_path)
    expected = _expected_keys(inputs)
    predictions, keys = production._validated_existing_predictions(predictions_path, expected)
    if len(predictions) != EXPECTED_EXAMPLES * len(CONFIGURATIONS) or keys != expected:
        raise ValueError("All optimized answer predictions must be frozen before gold access")

    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in inputs:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))
    gold = {
        dataset: production.load_gold_answers(
            dataset, source_root / production.DATASET_INFO[dataset][2], ids
        )
        for dataset, ids in ids_by_dataset.items()
    }
    gold_support = {
        str(row["example_id"]): list(row.get("gold_explanations") or [])
        for row in production.load_jsonl(support_gold)
    }
    scored: list[dict[str, Any]] = []
    for prediction in predictions:
        dataset, example_id = str(prediction["dataset"]), str(prediction["example_id"])
        answer = production._answer_scores(
            str(prediction["domain"]), str(prediction["predicted_answer"]), gold[dataset][example_id]
        )
        support_alternatives = gold_support.get(example_id, [])
        row: dict[str, Any] = {
            "dataset": dataset,
            "domain": prediction["domain"],
            "example_id": example_id,
            "method": prediction["method"],
            "setting": prediction["setting"],
            "predicted_answer": prediction["predicted_answer"],
            "answer_em": answer[0],
            "answer_f1": answer[1],
            "support_evaluable": bool(support_alternatives),
        }
        if support_alternatives:
            support = production.best_support_scores(
                list(prediction["selected_support"]), support_alternatives
            )
            joint_precision, joint_recall = answer[2] * support["prec"], answer[3] * support["recall"]
            row.update(
                {
                    "support_em": support["em"],
                    "support_f1": support["f1"],
                    "joint_em": answer[0] * support["em"],
                    "joint_f1": (
                        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
                        if joint_precision + joint_recall
                        else 0.0
                    ),
                }
            )
        scored.append(row)
    production.write_jsonl(output_dir / "per_example_end_to_end.jsonl", scored)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[(str(row["dataset"]), str(row["setting"]))].append(row)

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        support_rows = [row for row in rows if row["support_evaluable"]]
        return {
            "answer_examples": len(rows),
            "support_examples": len(support_rows),
            **{
                field: fmean(float(row[field]) for row in (rows if field.startswith("answer") else support_rows))
                for field in ("answer_em", "answer_f1", "support_em", "support_f1", "joint_em", "joint_f1")
            },
        }

    by_dataset = {
        dataset: {setting: aggregate(grouped[(dataset, setting)]) for setting in SETTINGS}
        for dataset in EXPECTED_DATASET_COUNTS
    }
    macro = {
        setting: {
            field: fmean(float(by_dataset[dataset][setting][field]) for dataset in EXPECTED_DATASET_COUNTS)
            for field in ("answer_em", "answer_f1", "support_em", "support_f1", "joint_em", "joint_f1")
        }
        for setting in SETTINGS
    }
    metrics = {
        "schema_version": "optimized_sageqa_end_to_end_metrics_v1",
        "status": "complete_frozen",
        "primary_equal_dataset_macro": macro,
        "by_dataset": by_dataset,
    }
    production.write_json(output_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    preflight.add_argument("--candidate-root", type=repo_path_arg, default=DEFAULT_CANDIDATE_ROOT)
    preflight.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT)
    generate = sub.add_parser("generate")
    generate.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT)
    evaluate.add_argument("--source-root", type=repo_path_arg, default=ROOT)
    evaluate.add_argument("--support-gold", type=repo_path_arg, default=DEFAULT_SUPPORT_GOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        result = preflight_phase(args.selection_root, args.candidate_root, args.output_dir)
    elif args.phase == "generate":
        result = generate_phase(args.output_dir, workers=args.workers, resume=args.resume)
    else:
        result = evaluate_phase(
            args.output_dir, source_root=args.source_root, support_gold=args.support_gold
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
