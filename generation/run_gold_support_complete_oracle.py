"""Versioned complete Gold Support TEST oracle.

The preflight phase constructs one answer-only reader input for every TEST
example with persisted non-empty gold support.  Its support is the
order-preserving deduplicated union of every persisted gold explanation
alternative.  Generation is impossible until that input bundle and its
preflight report have been materialized and hash locked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation import run_production_test_answer_generation as production


ROOT = Path(__file__).resolve().parents[1]
MODEL = production.MODEL_NAME
METHOD = "gold_support_complete"
SETTING = "oracle"
CONDITION = f"{METHOD}/{SETTING}"
CACHE_FIELDS = (
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
DEFAULT_SELECTIONS = ROOT / (
    "outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40/"
    "test_predictions_frozen.jsonl"
)
DEFAULT_SELECTION_FREEZE = DEFAULT_SELECTIONS.parent / "generation_freeze.json"
DEFAULT_SUPPORT = ROOT / (
    "outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/"
    "per_example_test_retrieval.jsonl"
)
DEFAULT_SUPPORT_MANIFEST = DEFAULT_SUPPORT.parent / "artifact_manifest.json"
DEFAULT_CANDIDATE_ROOT = ROOT / "data/production_generator_d_v1"
DEFAULT_FULL_CONTEXT_ROOT = ROOT / "outputs/final_results/final_manuscript_baselines_test_end_to_end"
DEFAULT_OUTPUT = ROOT / "outputs/final_results/gold_support_complete_oracle"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_hash(path: Path, filename: str) -> str:
    files = _read_json(path).get("files", {})
    if isinstance(files, dict):
        entry = files.get(filename)
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            return str(entry.get("sha256", ""))
    if isinstance(files, list):
        for entry in files:
            if Path(str(entry.get("path", ""))).name == filename:
                return str(entry.get("sha256", ""))
    raise ValueError(f"Manifest does not lock {filename}: {path}")


def _source_hashes(
    selections: Path, selection_freeze: Path, support: Path, support_manifest: Path
) -> dict[str, str]:
    selection_hash = production.sha256(selections)
    if selection_hash != str(_read_json(selection_freeze).get("predictions_sha256", "")):
        raise ValueError("Cross-Encoder/final-SAGE selection freeze mismatch")
    support_hash = production.sha256(support)
    if support_hash != _manifest_hash(support_manifest, support.name):
        raise ValueError("Persisted support source manifest mismatch")
    return {
        str(selections.relative_to(ROOT)): selection_hash,
        str(selection_freeze.relative_to(ROOT)): production.sha256(selection_freeze),
        str(support.relative_to(ROOT)): support_hash,
        str(support_manifest.relative_to(ROOT)): production.sha256(support_manifest),
    }


def _ordered_union(alternatives: Sequence[Sequence[Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for alternative in alternatives:
        if not isinstance(alternative, list) or not all(isinstance(unit, str) for unit in alternative):
            raise ValueError("Every persisted gold explanation must be a list of strings")
        for unit in alternative:
            if unit not in seen:
                seen.add(unit)
                result.append(unit)
    return result


def _reader_type(domain: str) -> str:
    return "text_reader_v1" if domain == "text" else "ontology_proof_first_reader_v1"


def _cache_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row["metadata"]
    contract = {
        "dataset": row["dataset"],
        "example_id": row["example_id"],
        "question": row["question"],
        "support_sha256": row["support_sha256"],
        "model": MODEL,
        "domain": row["domain"],
        "reader_type": _reader_type(str(row["domain"])),
        "task_type": metadata.get("task_type"),
        "sparql_query": metadata.get("sparql_query"),
        "answer_type": metadata.get("answer_type"),
    }
    if tuple(contract) != CACHE_FIELDS:
        raise AssertionError("Corrected cache contract field order changed")
    return contract


def _cache_key(row: Mapping[str, Any]) -> str:
    return production.canonical_hash(_cache_contract(row))


def _load_support(path: Path, expected_ids: set[str]) -> tuple[dict[str, dict[str, Any]], Counter[int]]:
    result: dict[str, dict[str, Any]] = {}
    distribution: Counter[int] = Counter()
    for row in production.load_jsonl(path):
        example_id = str(row.get("example_id", ""))
        if example_id not in expected_ids:
            continue
        alternatives = row.get("gold_explanations") or []
        if not isinstance(alternatives, list):
            raise ValueError(f"Invalid persisted alternatives: {example_id}")
        complete = _ordered_union(alternatives)
        first = list(alternatives[0]) if alternatives else []
        if complete:
            distribution[len(alternatives)] += 1
            result[example_id] = {
                "complete": complete,
                "first": first,
                "alternative_count": len(alternatives),
            }
    if len(result) != 3509:
        raise ValueError(f"Expected 3,509 support-bearing examples, found {len(result)}")
    return result, distribution


def _expected_prediction_keys(inputs: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    return {production.prediction_key(row) for row in inputs}


def preflight_phase(
    *,
    selections: Path = DEFAULT_SELECTIONS,
    selection_freeze: Path = DEFAULT_SELECTION_FREEZE,
    support: Path = DEFAULT_SUPPORT,
    support_manifest: Path = DEFAULT_SUPPORT_MANIFEST,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    hashes = _source_hashes(selections, selection_freeze, support, support_manifest)
    selected_rows = production.load_jsonl(selections)
    if len(selected_rows) != 4249:
        raise ValueError(f"Expected 4,249 frozen TEST selections, found {len(selected_rows)}")
    ids = {str(row["example_id"]) for row in selected_rows}
    if len(ids) != 4249:
        raise ValueError("Frozen TEST selections contain duplicate IDs")
    metadata, candidate_lineage = production.load_generation_metadata(candidate_root, selected_rows)
    supports, alternative_distribution = _load_support(support, ids)

    inputs: list[dict[str, Any]] = []
    per_dataset: Counter[str] = Counter()
    differs = 0
    for selection in selected_rows:
        example_id = str(selection["example_id"])
        if example_id not in supports:
            continue
        dataset = str(selection["dataset"])
        item_metadata = metadata[example_id]
        complete = supports[example_id]["complete"]
        differs += complete != supports[example_id]["first"]
        row = {
            "schema_version": "gold_support_complete_reader_input_v1",
            "dataset": dataset,
            "domain": production.DATASET_INFO[dataset][1],
            "example_id": example_id,
            "method": METHOD,
            "setting": SETTING,
            "question": item_metadata["question"],
            "metadata": item_metadata,
            "selected_support": complete,
            "support_sha256": production.canonical_hash(complete),
            "gold_alternative_count": supports[example_id]["alternative_count"],
        }
        row["reader_cache_contract"] = _cache_contract(row)
        row["reader_cache_key_sha256"] = _cache_key(row)
        inputs.append(row)
        per_dataset[dataset] += 1

    unique = {_cache_key(row) for row in inputs}
    deterministic = 0
    for row in inputs:
        if row["domain"] == "ontology":
            proof = production.ontology_reader_module.infer_owl_boolean_answer(
                item=dict(row["metadata"]), support_units=list(row["selected_support"])
            )
            deterministic += proof is not None
    if len(inputs) != 3509 or len(unique) != 3509:
        raise ValueError("Corrected oracle population/cache identity invariant failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "frozen_reader_inputs.jsonl"
    production.write_jsonl(inputs_path, inputs)
    report = {
        "schema_version": "gold_support_complete_preflight_v1",
        "status": "complete_no_api_calls",
        "condition": CONDITION,
        "per_dataset_row_counts": dict(per_dataset),
        "gold_alternative_count_distribution": {
            str(key): alternative_distribution[key] for key in sorted(alternative_distribution)
        },
        "examples_differing_from_first_alternative_oracle": differs,
        "reader_input_rows": len(inputs),
        "unique_reader_inputs": len(unique),
        "deterministic_proof_first_count": deterministic,
        "expected_openai_calls": len(unique) - deterministic,
        "excluded_undefined_or_empty_gold_support": 4249 - len(inputs),
        "cache_key_fields": list(CACHE_FIELDS),
        "support_construction": "order-preserving deduplicated union of all persisted gold_explanations alternatives",
        "frozen_reader_inputs_sha256": production.sha256(inputs_path),
        "source_sha256": hashes,
        "candidate_metadata_lineage": candidate_lineage,
        "openai_calls_made": 0,
    }
    production.write_json(output_dir / "preflight_report.json", report)
    production.write_json(output_dir / "artifact_manifest.json", {
        "schema_version": "gold_support_complete_artifact_manifest_v1",
        "status": "preflight_complete_no_api_calls",
        "historical_gold_support_preserved": True,
        "files": {
            "frozen_reader_inputs.jsonl": production.sha256(inputs_path),
            "preflight_report.json": production.sha256(output_dir / "preflight_report.json"),
        },
        "source_sha256": hashes,
    })
    return report


def generate_phase(
    *, output_dir: Path = DEFAULT_OUTPUT, workers: int = 8, resume: bool = False
) -> dict[str, Any]:
    report = _read_json(output_dir / "preflight_report.json")
    inputs_path = output_dir / "frozen_reader_inputs.jsonl"
    if report.get("status") != "complete_no_api_calls":
        raise ValueError("A displayed/complete preflight is required before generation")
    if production.sha256(inputs_path) != report.get("frozen_reader_inputs_sha256"):
        raise ValueError("Corrected reader-input bundle differs from preflight")
    inputs = production.load_jsonl(inputs_path)
    for row in inputs:
        if row.get("reader_cache_contract") != _cache_contract(row):
            raise ValueError("Persisted corrected cache contract mismatch")
        if row.get("reader_cache_key_sha256") != _cache_key(row):
            raise ValueError("Persisted corrected cache key mismatch")
    expected = _expected_prediction_keys(inputs)
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists() and not resume:
        raise FileExistsError("Corrected predictions exist; use --resume")
    existing, completed = production._validated_existing_predictions(predictions_path, expected)
    cache: dict[str, Mapping[str, Any]] = {}
    for row in existing:
        contract = row.get("reader_cache_contract")
        if not isinstance(contract, dict) or tuple(contract) != CACHE_FIELDS:
            raise ValueError("A resume prediction lacks the corrected ten-field cache contract")
        cache[production.canonical_hash(contract)] = row
    jobs = {_cache_key(row): dict(row) for row in inputs if production.prediction_key(row) not in completed}

    def persist(key: str, result: Mapping[str, Any]) -> None:
        source = jobs[key]
        job = {
            "method": METHOD, "setting": SETTING,
            "example_id": source["example_id"], "dataset": source["dataset"],
            "domain": source["domain"], "metadata": source["metadata"],
            "support": source["selected_support"], "support_hash": source["support_sha256"],
        }
        prediction = production._prediction_row(job, result, MODEL)
        prediction["schema_version"] = "gold_support_complete_prediction_v1"
        prediction["reader_cache_contract"] = source["reader_cache_contract"]
        prediction["reader_cache_key_sha256"] = key
        production.append_jsonl(predictions_path, prediction)
        completed.add(production.prediction_key(prediction))

    for key in list(jobs):
        if key in cache:
            persist(key, cache[key])
    pending = [key for key in jobs if key not in cache]
    errors_path = output_dir / "generation_errors.jsonl"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[Mapping[str, Any]], str] = {}
        for key in pending:
            row = jobs[key]
            futures[executor.submit(
                production._run_reader_job, row["domain"], row["metadata"],
                list(row["selected_support"]), MODEL,
                production.default_text_reader, production.default_ontology_reader,
            )] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                persist(key, future.result())
            except Exception as exc:
                production.append_jsonl(errors_path, {"reader_cache_key_sha256": key, "error": repr(exc)})
    if completed != expected:
        raise RuntimeError(f"Generation incomplete: {len(completed)}/{len(expected)}; rerun with --resume")
    rows, keys = production._validated_existing_predictions(predictions_path, expected)
    if keys != expected or len(rows) != 3509:
        raise ValueError("Corrected oracle prediction freeze population mismatch")
    freeze = {
        "schema_version": "gold_support_complete_generation_freeze_v1",
        "status": "complete_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": CONDITION,
        "model": MODEL,
        "prediction_count": len(rows),
        "predictions_sha256": production.sha256(predictions_path),
        "reader_inputs_sha256": report["frozen_reader_inputs_sha256"],
        "cache_key_fields": list(CACHE_FIELDS),
        "gold_answers_opened": False,
    }
    production.write_json(output_dir / "generation_freeze.json", freeze)
    return freeze


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "answer_em": fmean(float(row["answer_em"]) for row in rows),
        "answer_f1": fmean(float(row["answer_f1"]) for row in rows),
    }


def _validate_full_context(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    predictions = root / "predictions.jsonl"
    freeze_path = root / "generation_freeze.json"
    scored_path = root / "per_example_end_to_end.jsonl"
    manifest_path = root / "artifact_manifest.json"
    freeze = _read_json(freeze_path)
    if production.sha256(predictions) != freeze.get("predictions_sha256"):
        raise ValueError("Frozen Full Context prediction hash mismatch")
    for filename in ("predictions.jsonl", "per_example_end_to_end.jsonl"):
        if production.sha256(root / filename) != _manifest_hash(manifest_path, filename):
            raise ValueError(f"Frozen Full Context manifest mismatch: {filename}")
    rows = [
        row for row in production.load_jsonl(scored_path)
        if row.get("method") == "full_context" and row.get("setting") == "full"
    ]
    if len(rows) != 4249:
        raise ValueError("Primary Full Context population is not 4,249")
    return rows, {
        str(predictions.relative_to(ROOT)): production.sha256(predictions),
        str(freeze_path.relative_to(ROOT)): production.sha256(freeze_path),
        str(scored_path.relative_to(ROOT)): production.sha256(scored_path),
        str(manifest_path.relative_to(ROOT)): production.sha256(manifest_path),
    }


def evaluate_phase(
    *, output_dir: Path = DEFAULT_OUTPUT,
    full_context_root: Path = DEFAULT_FULL_CONTEXT_ROOT,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    inputs_path = output_dir / "frozen_reader_inputs.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    freeze_path = output_dir / "generation_freeze.json"
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "complete_frozen":
        raise ValueError("Complete corrected prediction freeze is required before gold access")
    if production.sha256(predictions_path) != freeze.get("predictions_sha256"):
        raise ValueError("Corrected predictions differ from generation freeze")
    if production.sha256(inputs_path) != freeze.get("reader_inputs_sha256"):
        raise ValueError("Corrected reader inputs differ from generation freeze")
    inputs = production.load_jsonl(inputs_path)
    predictions, keys = production._validated_existing_predictions(
        predictions_path, _expected_prediction_keys(inputs)
    )
    if len(predictions) != 3509 or len(keys) != 3509:
        raise ValueError("Corrected oracle evaluation population mismatch")
    input_by_key = {production.prediction_key(row): row for row in inputs}
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in inputs:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))
    answers = {
        dataset: production.load_gold_answers(
            dataset, source_root / production.DATASET_INFO[dataset][2], ids
        )
        for dataset, ids in ids_by_dataset.items()
    }
    scored: list[dict[str, Any]] = []
    for prediction in predictions:
        key = production.prediction_key(prediction)
        source = input_by_key[key]
        answer_em, answer_f1, _answer_precision, _answer_recall = production._answer_scores(
            str(prediction["domain"]), str(prediction["predicted_answer"]),
            answers[str(prediction["dataset"])][str(prediction["example_id"])],
        )
        scored.append({
            "schema_version": "gold_support_complete_evaluation_example_v1",
            "dataset": prediction["dataset"], "domain": prediction["domain"],
            "example_id": prediction["example_id"], "method": METHOD, "setting": SETTING,
            "predicted_answer": prediction["predicted_answer"],
            "answer_em": answer_em, "answer_f1": answer_f1,
            "reader_input_sha256": source["reader_cache_key_sha256"],
        })
    production.write_jsonl(output_dir / "per_example_answer.jsonl", scored)

    eligible_ids = {str(row["example_id"]) for row in inputs}
    full_all, full_hashes = _validate_full_context(full_context_root)
    full_matched = [row for row in full_all if str(row["example_id"]) in eligible_ids]
    if len(full_matched) != 3509:
        raise ValueError("Matched Full Context cohort is not 3,509")
    per_dataset: dict[str, Any] = {}
    for dataset in production.DATASET_INFO:
        oracle_rows = [row for row in scored if row["dataset"] == dataset]
        full_rows = [row for row in full_matched if row["dataset"] == dataset]
        if len(oracle_rows) != len(full_rows):
            raise ValueError(f"Matched cohort mismatch for {dataset}")
        per_dataset[dataset] = {
            CONDITION: _aggregate(oracle_rows),
            "full_context/full_matched_support_bearing": _aggregate(full_rows),
        }
    oracle_macro = {
        metric: fmean(per_dataset[d][CONDITION][metric] for d in per_dataset)
        for metric in ("answer_em", "answer_f1")
    }
    full_macro = {
        metric: fmean(per_dataset[d]["full_context/full_matched_support_bearing"][metric] for d in per_dataset)
        for metric in ("answer_em", "answer_f1")
    }
    metrics = {
        "schema_version": "gold_support_complete_metrics_v1",
        "status": "complete_frozen_evaluation_only",
        "condition": CONDITION,
        "reporting_scope": "answer_only",
        "cohort_examples": 3509,
        "per_dataset": per_dataset,
        "equal_dataset_macro": {
            CONDITION: oracle_macro,
            "full_context/full_matched_support_bearing": full_macro,
        },
        "full_context_primary_all_example_reference": {
            "examples": 4249,
            "retained_as_primary": True,
            "source": str((full_context_root / "metrics.json").relative_to(ROOT)),
            "source_sha256": production.sha256(full_context_root / "metrics.json"),
        },
        "full_context_source_sha256": full_hashes,
        "api_calls_during_evaluation": 0,
    }
    production.write_json(output_dir / "metrics.json", metrics)
    manifest_path = output_dir / "artifact_manifest.json"
    preflight = _read_json(output_dir / "preflight_report.json")
    files = [
        "frozen_reader_inputs.jsonl", "preflight_report.json", "predictions.jsonl",
        "generation_freeze.json", "per_example_answer.jsonl", "metrics.json",
    ]
    if (output_dir / "generation_errors.jsonl").is_file():
        files.append("generation_errors.jsonl")
    historical_error_count = 0
    if (output_dir / "generation_errors.jsonl").is_file():
        historical_error_count = len(production.load_jsonl(output_dir / "generation_errors.jsonl"))
    production.write_json(manifest_path, {
        "schema_version": "gold_support_complete_artifact_manifest_v1",
        "status": "complete_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": CONDITION,
        "historical_gold_support_preserved": True,
        "files": {name: {"size_bytes": (output_dir / name).stat().st_size, "sha256": production.sha256(output_dir / name)} for name in files},
        "source_sha256": {**preflight["source_sha256"], **full_hashes},
        "inference_invoked_only_for_corrected_oracle": True,
        "protected_main_predictions_modified": False,
        "generation_attempt_audit": {
            "historical_failed_attempts_logged": historical_error_count,
            "final_complete_prediction_count": len(predictions),
            "missing_predictions": 0,
            "note": "Logged connection failures precede the completed retry and are retained as provenance.",
        },
    })
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "generate", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase == "preflight":
        result = preflight_phase(output_dir=args.output_dir)
    elif args.phase == "generate":
        result = generate_phase(output_dir=args.output_dir, workers=args.workers, resume=args.resume)
    else:
        result = evaluate_phase(output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
