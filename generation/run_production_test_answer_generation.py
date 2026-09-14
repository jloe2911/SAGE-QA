"""Two-phase answer generation and end-to-end evaluation over frozen TEST support.

The generate phase is deliberately gold-answer blind.  It copies support only
from ``methods[method][setting][retrieved_evidence_units]`` in the frozen
retrieval artifact and obtains question/query metadata from the clean TEST
candidate corpus.  The evaluate phase requires a complete hash-locked
prediction file before it opens any original benchmark answer source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_owl_qa_predictions import answer_set_scores, best_support_scores
from evaluation.hotpot_official_eval import exact_match_score, f1_score
from generation import generate_hotpot_answers_with_llm as text_reader_module
from generation import generate_owl_answers_with_llm as ontology_reader_module
from utils.paths import (
    generator_d_root,
    hard_pair_answer_root,
    hard_pair_test_retrieval_root,
    repo_path_arg,
    repo_root,
)


ROOT = repo_root()
DEFAULT_RETRIEVAL = hard_pair_test_retrieval_root() / "per_example_test_retrieval.jsonl"
DEFAULT_CANDIDATE_ROOT = generator_d_root()
DEFAULT_OUTPUT_DIR = hard_pair_answer_root()
MODEL_NAME = "gpt-4.1-mini"
CONFIGURATIONS = (
    ("gnn_only", "k1"),
    ("gnn_only", "adaptive"),
    ("sageqa_final", "k1"),
    ("sageqa_final", "adaptive"),
)
DATASETS = (
    (
        "HotpotQA",
        "HotpotQA",
        "text",
        "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
    ),
    (
        "2WikiMultiHopQA",
        "2WikiMultiHopQA",
        "text",
        "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
    ),
    ("FamilyOWL_1hop", "FamilyOWL_1hop", "ontology", "data/raw/family/FamilyOWL_1hop.json"),
    ("FamilyOWL_2hop", "FamilyOWL_2hop", "ontology", "data/raw/family/FamilyOWL_2hop.json"),
    ("pizza_100_1hop", "pizza_100_1hop", "ontology", "data/raw/pizza_100/pizza_100_1hop.json"),
    ("pizza_100_2hop", "pizza_100_2hop", "ontology", "data/raw/pizza_100/pizza_100_2hop.json"),
    ("pizza_250_1hop", "pizza_250_1hop", "ontology", "data/raw/pizza_250/pizza_250_1hop.json"),
    ("pizza_250_2hop", "pizza_250_2hop", "ontology", "data/raw/pizza_250/pizza_250_2hop.json"),
    ("OWL2Bench_1hop", "OWL2Bench_1hop", "ontology", "data/raw/owl2bench/OWL2Bench_1hop.json"),
    ("OWL2Bench_2hop", "OWL2Bench_2hop", "ontology", "data/raw/owl2bench/OWL2Bench_2hop.json"),
)
DATASET_INFO = {name: (directory, domain, source) for name, directory, domain, source in DATASETS}
METADATA_FIELDS = (
    "example_id",
    "dataset",
    "split",
    "question",
    "hop",
    "answer_type",
    "sparql_query",
    "task_type",
    "source_name",
    "group_index",
    "qa_index",
)
FORBIDDEN_GENERATION_FIELDS = {
    "answer",
    "answers",
    "gold_answer",
    "gold_answers",
    "evidences",
    "supporting_facts",
    "gold_explanations",
    "gold_units",
    "gold_support_units",
}

TextReader = Callable[[str, list[str], str], Mapping[str, Any]]
OntologyReader = Callable[[Mapping[str, Any], list[str], str], Mapping[str, Any]]
GoldLoader = Callable[[str, Path, set[str]], dict[str, str]]
ReaderCacheKey = tuple[str, str, str, str, str, str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prediction_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["example_id"]), str(row["method"]), str(row["setting"])


def reader_cache_key(
    *,
    dataset: str,
    example_id: str,
    question: str,
    support_hash: str,
    model: str,
    domain: str,
) -> ReaderCacheKey:
    """Identify the actual reader input independently of method/setting labels."""
    reader_type = "text_reader_v1" if domain == "text" else "ontology_proof_first_reader_v1"
    return dataset, example_id, question, support_hash, model, domain, reader_type


def load_frozen_retrieval(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        example_id = str(row.get("example_id") or "")
        dataset = str(row.get("dataset") or "")
        if not example_id or example_id in seen:
            raise ValueError(f"Missing or duplicate frozen retrieval example_id: {example_id!r}")
        if dataset not in DATASET_INFO:
            raise ValueError(f"Unexpected dataset in frozen retrieval: {dataset!r}")
        seen.add(example_id)
        methods = row.get("methods")
        if not isinstance(methods, dict):
            raise ValueError(f"Missing methods for {example_id}")
        for method, setting in CONFIGURATIONS:
            selected = methods.get(method, {}).get(setting)
            if not isinstance(selected, dict):
                raise ValueError(f"Missing frozen configuration {method}/{setting}: {example_id}")
            support = selected.get("retrieved_evidence_units")
            if not isinstance(support, list) or not all(isinstance(unit, str) for unit in support):
                raise ValueError(f"Invalid frozen support for {method}/{setting}: {example_id}")
    return rows


def expected_prediction_keys(
    retrieval_rows: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, str]]:
    return {
        (str(row["example_id"]), method, setting)
        for row in retrieval_rows
        for method, setting in CONFIGURATIONS
    }


def _validate_candidate_metadata(row: Mapping[str, Any], expected_dataset: str) -> dict[str, Any]:
    forbidden = FORBIDDEN_GENERATION_FIELDS & set(row)
    if forbidden:
        raise ValueError(
            f"Candidate metadata row contains prohibited generation fields: {sorted(forbidden)}"
        )
    if str(row.get("dataset") or row.get("source_name") or expected_dataset) != expected_dataset:
        raise ValueError(f"Candidate dataset mismatch for {row.get('example_id')}")
    question = str(row.get("question") or "")
    if not question:
        raise ValueError(f"Candidate metadata has no question: {row.get('example_id')}")
    return {field: row.get(field) for field in METADATA_FIELDS if field in row}


def load_generation_metadata(
    candidate_root: Path,
    retrieval_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in retrieval_rows:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))

    metadata: dict[str, dict[str, Any]] = {}
    lineage: dict[str, Any] = {}
    for dataset, wanted in ids_by_dataset.items():
        directory, _domain, _source = DATASET_INFO[dataset]
        path = candidate_root / directory / "test_subgraph_retrieval.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Clean frozen TEST candidate corpus is missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(metadata.keys() & wanted) == len(wanted):
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                example_id = str(row.get("example_id") or "")
                if example_id in wanted and example_id not in metadata:
                    metadata[example_id] = _validate_candidate_metadata(row, dataset)
        missing = wanted - set(metadata)
        if missing:
            raise ValueError(
                f"Candidate metadata join incomplete for {dataset}: {sorted(missing)[:3]}"
            )
        lineage[dataset] = {
            "candidate_path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "candidate_sha256": sha256(path),
            "examples_joined": len(wanted),
            "fields_read": list(METADATA_FIELDS),
            "gold_fields_read": [],
        }
    return metadata, lineage


def default_text_reader(question: str, support: list[str], model: str) -> Mapping[str, Any]:
    prompt = text_reader_module.build_prompt(question, support)
    raw = text_reader_module.call_openai(prompt=prompt, model=model, temperature=0.0)
    parsed = text_reader_module.extract_json_object(raw)
    return {
        "predicted_answer": text_reader_module.normalize_generated_answer(
            question, str(parsed.get("answer", "")).strip()
        ),
        "explanation": str(parsed.get("explanation", "")).strip(),
        "raw_response": raw,
        "answer_source": "gpt-4.1-mini_reader",
    }


def default_ontology_reader(
    metadata: Mapping[str, Any], support: list[str], model: str
) -> Mapping[str, Any]:
    item = dict(metadata)
    proof = ontology_reader_module.infer_owl_boolean_answer(item=item, support_units=support)
    if proof is not None:
        raw = json.dumps(proof, ensure_ascii=False)
        return {
            "predicted_answer": proof["answer"],
            "explanation": proof["explanation"],
            "raw_response": raw,
            "answer_source": "deterministic_owl_proof",
        }
    # Keep the frozen support order: the prompt receives the exact list copied
    # from the retrieval artifact, with no reconstruction or reranking.
    prompt = ontology_reader_module.build_prompt(str(metadata["question"]), support)
    raw = ontology_reader_module.call_openai(prompt=prompt, model=model)
    parsed = ontology_reader_module.parse_json_response(raw)
    return {
        "predicted_answer": ontology_reader_module.normalize_generated_answer(
            str(metadata["question"]), parsed["answer"]
        ),
        "explanation": parsed["explanation"],
        "raw_response": raw,
        "answer_source": "gpt-4.1-mini_fallback",
    }


def _run_reader_job(
    domain: str,
    metadata: Mapping[str, Any],
    support: list[str],
    model: str,
    text_reader: TextReader,
    ontology_reader: OntologyReader,
) -> Mapping[str, Any]:
    """Run one gold-free reader input; workers never receive retrieval/gold rows."""
    if FORBIDDEN_GENERATION_FIELDS & set(metadata):
        raise ValueError("Prohibited gold field reached a generation worker")
    if domain == "text":
        return text_reader(str(metadata["question"]), support, model)
    return ontology_reader(metadata, support, model)


def _prediction_row(
    job: Mapping[str, Any], result: Mapping[str, Any], model: str
) -> dict[str, Any]:
    metadata = job["metadata"]
    return {
        "schema_version": "production_test_answer_prediction_v1",
        "generation_status": "complete",
        "method": job["method"],
        "setting": job["setting"],
        "example_id": job["example_id"],
        "dataset": job["dataset"],
        "domain": job["domain"],
        "split": "test",
        "question": metadata["question"],
        "hop": metadata.get("hop"),
        "answer_type": metadata.get("answer_type"),
        "sparql_query": metadata.get("sparql_query"),
        "task_type": metadata.get("task_type"),
        "selected_support": list(job["support"]),
        "frozen_support_sha256": job["support_hash"],
        "predicted_answer": str(result.get("predicted_answer", "")),
        "explanation": str(result.get("explanation", "")),
        "answer_source": str(result.get("answer_source", "")),
        "raw_response": result.get("raw_response"),
        "error": None,
        "model": model,
        "generated_at_utc": utc_now(),
    }


def _validated_existing_predictions(
    path: Path, expected: set[tuple[str, str, str]]
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    if not path.exists():
        return [], set()
    rows = load_jsonl(path)
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        key = prediction_key(row)
        if key in keys:
            raise ValueError(f"Duplicate completed prediction: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected prediction configuration: {key}")
        if row.get("generation_status") != "complete":
            raise ValueError(f"Non-complete row found in predictions.jsonl: {key}")
        if canonical_hash(row.get("selected_support")) != row.get("frozen_support_sha256"):
            raise ValueError(f"Stored support hash mismatch: {key}")
        keys.add(key)
    return rows, keys


def _code_hashes() -> dict[str, str]:
    paths = {
        "production_runner": Path(__file__).resolve(),
        "text_reader": Path(text_reader_module.__file__).resolve(),
        "ontology_reader": Path(ontology_reader_module.__file__).resolve(),
        "ontology_evaluator": ROOT / "evaluation/evaluate_owl_qa_predictions.py",
        "hotpot_evaluator": ROOT / "evaluation/hotpot_official_eval.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def _manifest(output_dir: Path, *, status: str) -> dict[str, Any]:
    names = (
        "predictions.jsonl",
        "generation_errors.jsonl",
        "generation_freeze.json",
        "metrics.json",
        "per_example_end_to_end.jsonl",
        "summary.md",
        "lineage_metadata.json",
    )
    files = []
    for name in names:
        path = output_dir / name
        if path.is_file():
            files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "schema_version": "production_test_end_to_end_manifest_v1",
        "status": status,
        "created_at_utc": utc_now(),
        "files": files,
    }


def generate_phase(
    retrieval_path: Path,
    candidate_root: Path,
    output_dir: Path,
    *,
    model: str = MODEL_NAME,
    resume: bool = False,
    workers: int = 8,
    text_reader: TextReader = default_text_reader,
    ontology_reader: OntologyReader = default_ontology_reader,
) -> dict[str, Any]:
    if model != MODEL_NAME:
        raise ValueError(f"Production model is frozen to {MODEL_NAME!r}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    retrieval_path = retrieval_path.resolve()
    retrieval_hash_before = sha256(retrieval_path)
    retrieval_rows = load_frozen_retrieval(retrieval_path)
    expected = expected_prediction_keys(retrieval_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    freeze_path = output_dir / "generation_freeze.json"
    if predictions_path.exists() and not resume:
        raise FileExistsError(
            "predictions.jsonl exists; use --resume (completed rows are immutable)"
        )

    existing_rows, completed = _validated_existing_predictions(predictions_path, expected)
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if completed != expected or sha256(predictions_path) != freeze.get("predictions_sha256"):
            raise ValueError("Frozen predictions are incomplete or have changed")
        return freeze

    metadata, candidate_lineage = load_generation_metadata(candidate_root, retrieval_rows)
    errors_path = output_dir / "generation_errors.jsonl"
    jobs_by_reader_input: dict[ReaderCacheKey, list[dict[str, Any]]] = defaultdict(list)
    for retrieval_row in retrieval_rows:
        example_id = str(retrieval_row["example_id"])
        dataset = str(retrieval_row["dataset"])
        domain = DATASET_INFO[dataset][1]
        item_metadata = metadata[example_id]
        for method, setting in CONFIGURATIONS:
            key = (example_id, method, setting)
            if key in completed:
                continue
            # This is the only support-selection operation in this runner.
            frozen_value = retrieval_row["methods"][method][setting]["retrieved_evidence_units"]
            selected_support = list(frozen_value)
            support_hash = canonical_hash(selected_support)
            if selected_support != frozen_value:
                raise AssertionError(f"Frozen support copy changed for {key}")
            cache_key = reader_cache_key(
                dataset=dataset,
                example_id=example_id,
                question=str(item_metadata["question"]),
                support_hash=support_hash,
                model=model,
                domain=domain,
            )
            jobs_by_reader_input[cache_key].append(
                {
                    "key": key,
                    "example_id": example_id,
                    "dataset": dataset,
                    "domain": domain,
                    "method": method,
                    "setting": setting,
                    "metadata": item_metadata,
                    "support": selected_support,
                    "support_hash": support_hash,
                }
            )

    # A partial resume may already contain one member of an equivalent-input
    # group. Reuse its exact reader output instead of issuing another API call.
    cached_results: dict[ReaderCacheKey, Mapping[str, Any]] = {}
    for row in existing_rows:
        cache_key = reader_cache_key(
            dataset=str(row["dataset"]),
            example_id=str(row["example_id"]),
            question=str(row["question"]),
            support_hash=str(row["frozen_support_sha256"]),
            model=str(row["model"]),
            domain=str(row["domain"]),
        )
        cached_results[cache_key] = {
            "predicted_answer": row.get("predicted_answer", ""),
            "explanation": row.get("explanation", ""),
            "answer_source": row.get("answer_source", ""),
            "raw_response": row.get("raw_response"),
        }

    def persist_result(cache_key: ReaderCacheKey, result: Mapping[str, Any]) -> None:
        for job in jobs_by_reader_input[cache_key]:
            support = job["support"]
            if canonical_hash(support) != job["support_hash"]:
                raise AssertionError(f"Reader mutated frozen support for {job['key']}")
            row = _prediction_row(job, result, model)
            append_jsonl(predictions_path, row)
            completed.add(job["key"])
            existing_rows.append(row)

    for cache_key in list(jobs_by_reader_input):
        if cache_key in cached_results:
            persist_result(cache_key, cached_results[cache_key])

    pending_keys = [key for key in jobs_by_reader_input if key not in cached_results]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[Mapping[str, Any]], ReaderCacheKey] = {}
        for cache_key in pending_keys:
            job = jobs_by_reader_input[cache_key][0]
            future = executor.submit(
                _run_reader_job,
                job["domain"],
                job["metadata"],
                job["support"],
                model,
                text_reader,
                ontology_reader,
            )
            futures[future] = cache_key
        for future in as_completed(futures):
            cache_key = futures[future]
            try:
                persist_result(cache_key, future.result())
            except Exception as exc:
                for job in jobs_by_reader_input[cache_key]:
                    append_jsonl(
                        errors_path,
                        {
                            "example_id": job["example_id"],
                            "dataset": job["dataset"],
                            "method": job["method"],
                            "setting": job["setting"],
                            "frozen_support_sha256": job["support_hash"],
                            "error": repr(exc),
                            "failed_at_utc": utc_now(),
                        },
                    )

    if sha256(retrieval_path) != retrieval_hash_before:
        raise ValueError("Frozen retrieval artifact changed during generation")
    if completed != expected:
        missing = sorted(expected - completed)
        write_json(
            output_dir / "artifact_manifest.json",
            _manifest(output_dir, status="generation_incomplete"),
        )
        raise RuntimeError(
            f"Generation incomplete: {len(completed)}/{len(expected)} completed; "
            f"resume to retry {len(missing)} failed/missing calls"
        )

    # Re-read before freezing so the persisted bytes, uniqueness, and count are verified.
    persisted_rows, persisted_keys = _validated_existing_predictions(predictions_path, expected)
    if persisted_keys != expected or len(persisted_rows) != len(expected):
        raise ValueError("Persisted predictions failed the completion gate")
    freeze = {
        "schema_version": "production_test_answer_generation_freeze_v1",
        "status": "complete_frozen",
        "frozen_at_utc": utc_now(),
        "model": model,
        "configurations": [{"method": m, "setting": s} for m, s in CONFIGURATIONS],
        "prediction_count": len(persisted_rows),
        "predictions_sha256": sha256(predictions_path),
        "retrieval_path": str(retrieval_path),
        "retrieval_sha256": retrieval_hash_before,
        "support_lineage": ('record["methods"][method][setting]["retrieved_evidence_units"]'),
        "support_hash_algorithm": "sha256(canonical JSON list)",
        "code_sha256": _code_hashes(),
    }
    write_json(freeze_path, freeze)
    lineage = {
        "schema_version": "production_test_end_to_end_lineage_v1",
        "generation": freeze,
        "candidate_metadata": candidate_lineage,
        "gold_boundary": {
            "generation_opened_original_benchmark_answer_sources": False,
            "generation_rows_contain_gold_answers": False,
            "evaluation_requires_complete_prediction_freeze": True,
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(output_dir / "lineage_metadata.json", lineage)
    write_json(
        output_dir / "artifact_manifest.json",
        _manifest(output_dir, status="generation_complete_frozen"),
    )
    return freeze


def load_gold_answers(dataset: str, source: Path, example_ids: set[str]) -> dict[str, str]:
    """Open only the answer field(s) required by the evaluate phase."""
    _directory, domain, _relative = DATASET_INFO[dataset]
    if domain == "text":
        import pandas as pd

        frame = pd.read_parquet(source, columns=["id", "answer"])
        prefix = f"{dataset}__test__"
        answers = {
            prefix + str(row["id"]): str(row["answer"])
            for row in frame.to_dict(orient="records")
            if prefix + str(row["id"]) in example_ids
        }
    else:
        groups = json.loads(source.read_text(encoding="utf-8"))
        answers = {}
        for example_id in example_ids:
            parts = example_id.split("__", 3)
            if len(parts) < 3 or not parts[1].startswith("g") or not parts[2].startswith("q"):
                raise ValueError(f"Cannot parse ontology example ID: {example_id}")
            group_index, qa_index = int(parts[1][1:]), int(parts[2][1:])
            qa = groups[group_index]["QAs"][qa_index]
            if "Answer" not in qa:
                raise KeyError(f"Ontology gold Answer field missing: {example_id}")
            answers[example_id] = str(qa["Answer"])
    if set(answers) != example_ids:
        raise ValueError(
            f"Gold answer join incomplete for {dataset}: {sorted(example_ids - set(answers))[:3]}"
        )
    return answers


def _answer_scores(domain: str, prediction: str, gold: str) -> tuple[float, float, float, float]:
    if domain == "ontology":
        return answer_set_scores(prediction, gold)
    f1, precision, recall = f1_score(prediction, gold)
    return float(exact_match_score(prediction, gold)), f1, precision, recall


def _score_example(
    prediction: Mapping[str, Any], retrieval: Mapping[str, Any], gold_answer: str
) -> dict[str, Any]:
    support = list(
        retrieval["methods"][prediction["method"]][prediction["setting"]][
            "retrieved_evidence_units"
        ]
    )
    if prediction["selected_support"] != support:
        raise ValueError(
            f"Prediction support differs from frozen retrieval: {prediction_key(prediction)}"
        )
    ans_em, ans_f1, ans_precision, ans_recall = _answer_scores(
        str(prediction["domain"]), str(prediction["predicted_answer"]), gold_answer
    )
    gold_explanations = retrieval.get("gold_explanations") or []
    support_result = best_support_scores(support, gold_explanations) if gold_explanations else None
    if support_result is None:
        support_em = support_f1 = support_precision = support_recall = None
        joint_em = joint_f1 = joint_precision = joint_recall = None
    else:
        support_em = float(support_result["em"])
        support_f1 = float(support_result["f1"])
        support_precision = float(support_result["prec"])
        support_recall = float(support_result["recall"])
        joint_precision = ans_precision * support_precision
        joint_recall = ans_recall * support_recall
        joint_f1 = (
            2.0 * joint_precision * joint_recall / (joint_precision + joint_recall)
            if joint_precision + joint_recall
            else 0.0
        )
        joint_em = ans_em * support_em
    return {
        "schema_version": "production_test_end_to_end_example_v1",
        "dataset": prediction["dataset"],
        "domain": prediction["domain"],
        "method": prediction["method"],
        "setting": prediction["setting"],
        "example_id": prediction["example_id"],
        "question": prediction["question"],
        "gold_answer": gold_answer,
        "predicted_answer": prediction["predicted_answer"],
        "answer_em": ans_em,
        "answer_f1": ans_f1,
        "answer_precision": ans_precision,
        "answer_recall": ans_recall,
        "support_evaluable": support_result is not None,
        "support_em": support_em,
        "support_f1": support_f1,
        "support_precision": support_precision,
        "support_recall": support_recall,
        "joint_em": joint_em,
        "joint_f1": joint_f1,
        "joint_precision": joint_precision,
        "joint_recall": joint_recall,
        "selected_support": support,
        "best_gold_support": support_result["best_gold"] if support_result else [],
        "answer_source": prediction["answer_source"],
        "error": prediction.get("error"),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support_rows = [row for row in rows if row["support_evaluable"]]
    result = {
        "answer_examples": len(rows),
        "support_examples": len(support_rows),
        "answer_em": fmean(float(row["answer_em"]) for row in rows),
        "answer_f1": fmean(float(row["answer_f1"]) for row in rows),
        "support_em": None,
        "support_f1": None,
        "joint_em": None,
        "joint_f1": None,
    }
    if support_rows:
        for name in ("support_em", "support_f1", "joint_em", "joint_f1"):
            result[name] = fmean(float(row[name]) for row in support_rows)
    return result


def _summary(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen production TEST answer generation and end-to-end evaluation",
        "",
        "Predictions were frozen before original benchmark answers were opened. Answer metrics use all generated examples; support and joint metrics use only examples with frozen gold explanations.",
        "",
        "| Dataset | Method | Setting | A-EM | A-F1 | S-EM | S-F1 | J-EM | J-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in (item[0] for item in DATASETS):
        for method, setting in CONFIGURATIONS:
            row = metrics["by_dataset"][dataset][method][setting]
            fmt = lambda value: "N/A" if value is None else f"{value:.6f}"
            lines.append(
                f"| {dataset} | {method} | {setting} | {fmt(row['answer_em'])} | "
                f"{fmt(row['answer_f1'])} | {fmt(row['support_em'])} | {fmt(row['support_f1'])} | "
                f"{fmt(row['joint_em'])} | {fmt(row['joint_f1'])} |"
            )
    lines.extend(["", "## Evaluation populations", ""])
    for dataset in (item[0] for item in DATASETS):
        first = metrics["by_dataset"][dataset][CONFIGURATIONS[0][0]][CONFIGURATIONS[0][1]]
        lines.append(
            f"- {dataset}: answer n={first['answer_examples']}; support/joint n={first['support_examples']}."
        )
    return "\n".join(lines) + "\n"


def evaluate_phase(
    retrieval_path: Path,
    output_dir: Path,
    *,
    source_root: Path = ROOT,
    gold_loader: GoldLoader = load_gold_answers,
) -> dict[str, Any]:
    retrieval_path = retrieval_path.resolve()
    predictions_path = output_dir / "predictions.jsonl"
    freeze_path = output_dir / "generation_freeze.json"
    if not freeze_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Phase 1 must complete and freeze predictions before evaluation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "complete_frozen":
        raise ValueError("Generation freeze is not complete")
    if sha256(predictions_path) != freeze.get("predictions_sha256"):
        raise ValueError("predictions.jsonl changed after the generation freeze")
    if sha256(retrieval_path) != freeze.get("retrieval_sha256"):
        raise ValueError("Frozen retrieval artifact changed after answer generation")

    retrieval_rows = load_frozen_retrieval(retrieval_path)
    expected = expected_prediction_keys(retrieval_rows)
    predictions, keys = _validated_existing_predictions(predictions_path, expected)
    if keys != expected or len(predictions) != freeze.get("prediction_count"):
        raise ValueError("Prediction population is incomplete")
    retrieval_by_id = {str(row["example_id"]): row for row in retrieval_rows}

    # The first original gold-answer source access in the whole pipeline is here,
    # after every prediction byte has been hash-verified against the freeze.
    gold_by_dataset: dict[str, dict[str, str]] = {}
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in retrieval_rows:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))
    gold_access_started_at = utc_now()
    for dataset, ids in ids_by_dataset.items():
        source = source_root / DATASET_INFO[dataset][2]
        gold_by_dataset[dataset] = gold_loader(dataset, source, ids)

    per_example = [
        _score_example(
            prediction,
            retrieval_by_id[str(prediction["example_id"])],
            gold_by_dataset[str(prediction["dataset"])][str(prediction["example_id"])],
        )
        for prediction in predictions
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_example:
        grouped[(row["dataset"], row["method"], row["setting"])].append(row)
    by_dataset: dict[str, Any] = {}
    for dataset in (item[0] for item in DATASETS):
        if dataset not in ids_by_dataset:
            raise ValueError(f"Frozen retrieval is missing required dataset: {dataset}")
        by_dataset[dataset] = {}
        for method, setting in CONFIGURATIONS:
            rows = grouped[(dataset, method, setting)]
            if not rows:
                raise ValueError(f"No evaluation rows for {dataset}/{method}/{setting}")
            by_dataset[dataset].setdefault(method, {})[setting] = _aggregate(rows)
    metrics = {
        "schema_version": "production_test_end_to_end_metrics_v1",
        "status": "complete_frozen",
        "model": freeze["model"],
        "by_dataset": by_dataset,
    }
    write_jsonl(output_dir / "per_example_end_to_end.jsonl", per_example)
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "summary.md").write_text(_summary(metrics), encoding="utf-8", newline="\n")

    lineage_path = output_dir / "lineage_metadata.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["evaluation"] = {
        "status": "complete_frozen",
        "completed_at_utc": utc_now(),
        "gold_access_started_at_utc": gold_access_started_at,
        "predictions_sha256_verified_before_gold_access": freeze["predictions_sha256"],
        "answer_sources": {
            dataset: {
                "path": DATASET_INFO[dataset][2],
                "sha256": sha256(source_root / DATASET_INFO[dataset][2]),
                "answer_field": "answer" if DATASET_INFO[dataset][1] == "text" else "Answer",
            }
            for dataset in ids_by_dataset
        },
        "text_answer_semantics": "evaluation.hotpot_official_eval",
        "ontology_answer_semantics": "evaluation.evaluate_owl_qa_predictions.answer_set_scores",
        "support_source": "frozen retrieval retrieved_evidence_units and gold_explanations",
    }
    if sha256(predictions_path) != freeze["predictions_sha256"]:
        raise ValueError("predictions.jsonl changed during evaluation")
    write_json(lineage_path, lineage)
    write_json(
        output_dir / "artifact_manifest.json", _manifest(output_dir, status="complete_frozen")
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    generate = subparsers.add_parser("generate", help="Phase 1: gold-answer-blind generation")
    generate.add_argument("--retrieval", type=repo_path_arg, default=DEFAULT_RETRIEVAL)
    generate.add_argument("--candidate-root", type=repo_path_arg, default=DEFAULT_CANDIDATE_ROOT)
    generate.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT_DIR)
    generate.add_argument("--model", default=MODEL_NAME)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--workers", type=int, default=8)
    evaluate = subparsers.add_parser("evaluate", help="Phase 2: post-freeze gold evaluation")
    evaluate.add_argument("--retrieval", type=repo_path_arg, default=DEFAULT_RETRIEVAL)
    evaluate.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT_DIR)
    evaluate.add_argument("--source-root", type=repo_path_arg, default=ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "generate":
        generate_phase(
            args.retrieval,
            args.candidate_root,
            args.output_dir,
            model=args.model,
            resume=args.resume,
            workers=args.workers,
        )
    else:
        evaluate_phase(args.retrieval, args.output_dir, source_root=args.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
