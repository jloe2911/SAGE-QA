"""Run the frozen four-condition DEV reader diagnostic in two phases.

Generation reads only the prepared, hash-locked reader inputs. Evaluation is
refused until the complete prediction file has been frozen and hash-verified.
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
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_owl_qa_predictions import answer_set_scores
from evaluation.hotpot_official_eval import exact_match_score, f1_score
from generation import generate_hotpot_answers_with_llm as text_reader
from generation import generate_owl_answers_with_llm as ontology_reader
from utils.llm_client import openai_compatible_client


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "outputs/development_diagnostics/final_bottleneck_analysis/reader_inputs.jsonl"
)
DEFAULT_PREPARED_MANIFEST = (
    ROOT / "outputs/development_diagnostics/final_bottleneck_analysis/artifact_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/development_diagnostics/final_bottleneck_reader_diagnostic"
MODEL = "gpt-4.1-mini"
CONDITIONS = ("FULL_CONTEXT", "GOLD_SUPPORT", "ORACLE_GENERATOR_D", "SAGE_ACTUAL")
DATASETS = (
    ("HotpotQA", "text", "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet"),
    ("2WikiMultiHopQA", "text", "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet"),
    ("FamilyOWL_1hop", "ontology", "data/raw/family/FamilyOWL_1hop.json"),
    ("FamilyOWL_2hop", "ontology", "data/raw/family/FamilyOWL_2hop.json"),
    ("pizza_100_1hop", "ontology", "data/raw/pizza_100/pizza_100_1hop.json"),
    ("pizza_100_2hop", "ontology", "data/raw/pizza_100/pizza_100_2hop.json"),
    ("pizza_250_1hop", "ontology", "data/raw/pizza_250/pizza_250_1hop.json"),
    ("pizza_250_2hop", "ontology", "data/raw/pizza_250/pizza_250_2hop.json"),
    ("OWL2Bench_1hop", "ontology", "data/raw/owl2bench/OWL2Bench_1hop.json"),
    ("OWL2Bench_2hop", "ontology", "data/raw/owl2bench/OWL2Bench_2hop.json"),
)
DATASET_INFO = {name: (domain, source) for name, domain, source in DATASETS}


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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
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


def prediction_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["example_id"]), str(row["condition"])


def expected_messages(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["domain"] == "text":
        system = "You are a careful question answering system. Use only the given evidence."
        prompt = text_reader.build_prompt(str(row["question"]), list(row["selected_support"]))
    else:
        system = "You answer questions using only provided evidence and return valid JSON."
        prompt = ontology_reader.build_prompt(str(row["question"]), list(row["selected_support"]))
    return {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    }


def validate_inputs(path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locked = manifest["outputs"]["reader_inputs.jsonl"]["sha256"]
    if sha256(path) != locked:
        raise ValueError("Prepared reader input file differs from its artifact-manifest hash")
    rows = load_jsonl(path)
    expected_keys = set()
    for row in rows:
        key = prediction_key(row)
        if key in expected_keys:
            raise ValueError(f"Duplicate prepared condition row: {key}")
        expected_keys.add(key)
        dataset = str(row.get("dataset"))
        domain = str(row.get("domain"))
        if dataset not in DATASET_INFO or DATASET_INFO[dataset][0] != domain:
            raise ValueError(f"Dataset/domain mismatch: {dataset}/{domain}")
        if row.get("split") != "dev" or row.get("condition") not in CONDITIONS:
            raise ValueError(f"Non-DEV or unknown condition row: {key}")
        if row.get("api_call_executed") is not False:
            raise ValueError(f"Prepared input execution marker changed: {key}")
        if canonical_hash(row.get("selected_support")) != row.get("support_sha256"):
            raise ValueError(f"Prepared support hash mismatch: {key}")
        if canonical_hash(row.get("reader_input")) != row.get("reader_input_sha256"):
            raise ValueError(f"Prepared reader-input hash mismatch: {key}")
        if row.get("reader_input") != expected_messages(row):
            raise ValueError(f"Serialized prompt differs from frozen production builder: {key}")
    by_example: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_example[str(row["example_id"])].add(str(row["condition"]))
    if any(value != set(CONDITIONS) for value in by_example.values()):
        raise ValueError("Prepared examples do not all have exactly four matched conditions")
    return rows


def call_serialized_reader(row: Mapping[str, Any], model: str) -> dict[str, Any]:
    if model != MODEL:
        raise ValueError(f"Diagnostic model is frozen to {MODEL!r}")
    if row["domain"] == "ontology":
        proof = ontology_reader.infer_owl_boolean_answer(
            {"example_id": row["example_id"], "question": row["question"]},
            list(row["selected_support"]),
        )
        if proof is not None:
            return {
                "predicted_answer": proof["answer"],
                "explanation": proof["explanation"],
                "raw_response": json.dumps(proof, ensure_ascii=False),
                "answer_source": "deterministic_owl_proof",
            }
    # The requested model is an OpenAI model. Pin the provider explicitly so a
    # configured third-party compatibility key cannot silently reroute inputs.
    client, model_name, _provider = openai_compatible_client(f"openai:{model}")
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": row["reader_input"]["messages"],
        "temperature": 0.0,
    }
    if row["domain"] == "ontology":
        kwargs["max_tokens"] = 300
    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""
    if row["domain"] == "text":
        parsed = text_reader.extract_json_object(raw)
        answer = text_reader.normalize_generated_answer(
            str(row["question"]), str(parsed.get("answer", "")).strip()
        )
        explanation = str(parsed.get("explanation", "")).strip()
        source = "gpt-4.1-mini_reader"
    else:
        parsed = ontology_reader.parse_json_response(raw)
        answer = ontology_reader.normalize_generated_answer(str(row["question"]), parsed["answer"])
        explanation = parsed["explanation"]
        source = "gpt-4.1-mini_fallback"
    return {
        "predicted_answer": answer,
        "explanation": explanation,
        "raw_response": raw,
        "answer_source": source,
    }


def prediction_row(
    prepared: Mapping[str, Any], result: Mapping[str, Any], model: str
) -> dict[str, Any]:
    return {
        "schema_version": "dev_reader_diagnostic_prediction_v1",
        "generation_status": "complete",
        "dataset": prepared["dataset"],
        "domain": prepared["domain"],
        "split": "dev",
        "example_id": prepared["example_id"],
        "condition": prepared["condition"],
        "question": prepared["question"],
        "taxonomy": prepared.get("taxonomy"),
        "selected_support": prepared["selected_support"],
        "support_sha256": prepared["support_sha256"],
        "reader_input_sha256": prepared["reader_input_sha256"],
        "predicted_answer": str(result.get("predicted_answer", "")),
        "explanation": str(result.get("explanation", "")),
        "raw_response": result.get("raw_response"),
        "answer_source": result.get("answer_source"),
        "model": model,
        "generated_at_utc": utc_now(),
        "error": None,
    }


def validate_predictions(
    path: Path, prepared: Sequence[Mapping[str, Any]], model: str
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    expected = {prediction_key(row): row for row in prepared}
    rows = load_jsonl(path)
    seen = set()
    for row in rows:
        key = prediction_key(row)
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unexpected prediction: {key}")
        source = expected[key]
        if row.get("generation_status") != "complete" or row.get("model") != model:
            raise ValueError(f"Invalid completed prediction: {key}")
        for field in (
            "dataset",
            "domain",
            "split",
            "question",
            "selected_support",
            "support_sha256",
            "reader_input_sha256",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(f"Prediction/prepared-input mismatch in {field}: {key}")
        seen.add(key)
    return rows, seen


def code_hashes() -> dict[str, str]:
    return {
        "diagnostic_runner": sha256(Path(__file__).resolve()),
        "production_test_runner": sha256(
            ROOT / "generation/run_production_test_answer_generation.py"
        ),
        "text_reader": sha256(Path(text_reader.__file__).resolve()),
        "ontology_reader": sha256(Path(ontology_reader.__file__).resolve()),
        "text_evaluator": sha256(ROOT / "evaluation/hotpot_official_eval.py"),
        "ontology_evaluator": sha256(ROOT / "evaluation/evaluate_owl_qa_predictions.py"),
    }


def generate_phase(
    input_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    model: str,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    if model != MODEL or workers < 1:
        raise ValueError(f"Use model={MODEL!r} and workers >= 1")
    input_hash_before = sha256(input_path)
    prepared = validate_inputs(input_path, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    freeze_path = output_dir / "generation_freeze.json"
    if predictions_path.exists() and not resume:
        raise FileExistsError("predictions.jsonl exists; use --resume")
    existing, completed = validate_predictions(predictions_path, prepared, model)
    expected = {prediction_key(row) for row in prepared}
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if completed != expected or sha256(predictions_path) != freeze.get("predictions_sha256"):
            raise ValueError("Frozen predictions are incomplete or changed")
        return freeze

    prepared_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        if prediction_key(row) not in completed:
            prepared_by_hash[str(row["reader_input_sha256"])].append(row)
    cache: dict[str, Mapping[str, Any]] = {}
    for row in existing:
        cache[str(row["reader_input_sha256"])] = {
            "predicted_answer": row["predicted_answer"],
            "explanation": row["explanation"],
            "raw_response": row["raw_response"],
            "answer_source": row["answer_source"],
        }

    def persist(input_hash: str, result: Mapping[str, Any]) -> None:
        for prepared_row in prepared_by_hash[input_hash]:
            key = prediction_key(prepared_row)
            if key not in completed:
                append_jsonl(predictions_path, prediction_row(prepared_row, result, model))
                completed.add(key)

    for input_hash in list(prepared_by_hash):
        if input_hash in cache:
            persist(input_hash, cache[input_hash])
    pending = [value for value in prepared_by_hash if value not in cache]
    errors_path = output_dir / "generation_errors.jsonl"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[dict[str, Any]], str] = {
            executor.submit(
                call_serialized_reader, prepared_by_hash[input_hash][0], model
            ): input_hash
            for input_hash in pending
        }
        for future in as_completed(futures):
            input_hash = futures[future]
            try:
                persist(input_hash, future.result())
            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {
                        "reader_input_sha256": input_hash,
                        "error": repr(exc),
                        "failed_at_utc": utc_now(),
                    },
                )
    if sha256(input_path) != input_hash_before:
        raise ValueError("Prepared reader inputs changed during generation")
    if completed != expected:
        write_json(
            output_dir / "artifact_manifest.json",
            artifact_manifest(output_dir, "generation_incomplete"),
        )
        raise RuntimeError(
            f"Generation incomplete: {len(completed)}/{len(expected)} rows; use --resume"
        )
    persisted, keys = validate_predictions(predictions_path, prepared, model)
    if keys != expected or len(persisted) != len(expected):
        raise ValueError("Persisted prediction completion gate failed")
    freeze = {
        "schema_version": "dev_reader_diagnostic_generation_freeze_v1",
        "status": "complete_frozen",
        "frozen_at_utc": utc_now(),
        "model": model,
        "workers": workers,
        "conditions": list(CONDITIONS),
        "prediction_count": len(persisted),
        "unique_reader_input_count": len({row["reader_input_sha256"] for row in prepared}),
        "predictions_sha256": sha256(predictions_path),
        "prepared_reader_inputs_path": str(input_path.resolve()),
        "prepared_reader_inputs_sha256": input_hash_before,
        "prepared_manifest_sha256": sha256(manifest_path),
        "exact_cache_key": "reader_input_sha256",
        "api_provider": "openai",
        "code_sha256": code_hashes(),
        "gold_answer_sources_opened": False,
    }
    write_json(freeze_path, freeze)
    write_json(
        output_dir / "lineage_metadata.json",
        {
            "schema_version": "dev_reader_diagnostic_lineage_v1",
            "generation": freeze,
            "gold_boundary": {
                "generation_opened_gold_answer_sources": False,
                "evaluation_requires_complete_prediction_freeze": True,
            },
            "environment": {"python": sys.version, "platform": platform.platform()},
        },
    )
    write_json(
        output_dir / "artifact_manifest.json",
        artifact_manifest(output_dir, "generation_complete_frozen"),
    )
    return freeze


def load_gold(dataset: str, source: Path, ids: set[str]) -> dict[str, str]:
    domain, _ = DATASET_INFO[dataset]
    if domain == "text":
        import pandas as pd

        frame = pd.read_parquet(source, columns=["id", "answer"])
        prefix = f"{dataset}__dev__"
        answers = {
            prefix + str(row["id"]): str(row["answer"])
            for row in frame.to_dict(orient="records")
            if prefix + str(row["id"]) in ids
        }
    else:
        groups = json.loads(source.read_text(encoding="utf-8"))
        answers = {}
        for example_id in ids:
            parts = example_id.split("__", 3)
            group_index, qa_index = int(parts[1][1:]), int(parts[2][1:])
            answers[example_id] = str(groups[group_index]["QAs"][qa_index]["Answer"])
    if set(answers) != ids:
        raise ValueError(f"Gold join incomplete for {dataset}: {sorted(ids - set(answers))[:3]}")
    return answers


def score(domain: str, prediction: str, gold: str) -> tuple[float, float, float, float]:
    if domain == "ontology":
        return answer_set_scores(prediction, gold)
    f1, precision, recall = f1_score(prediction, gold)
    return float(exact_match_score(prediction, gold)), float(f1), float(precision), float(recall)


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0, "answer_em": None, "answer_f1": None}
    return {
        "examples": len(rows),
        "answer_em": fmean(float(row["answer_em"]) for row in rows),
        "answer_f1": fmean(float(row["answer_f1"]) for row in rows),
    }


def delta(high: Mapping[str, Any], low: Mapping[str, Any]) -> dict[str, float]:
    return {
        "answer_em": float(high["answer_em"]) - float(low["answer_em"]),
        "answer_f1": float(high["answer_f1"]) - float(low["answer_f1"]),
    }


def classify(macro: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    # Predeclared rule: average the macro EM and F1 loss at each causal transition.
    losses = {
        "A. ranking/selection": fmean(
            [
                max(0.0, macro["ORACLE_GENERATOR_D"][m] - macro["SAGE_ACTUAL"][m])
                for m in ("answer_em", "answer_f1")
            ]
        ),
        "B. candidate composition": fmean(
            [
                max(0.0, macro["GOLD_SUPPORT"][m] - macro["ORACLE_GENERATOR_D"][m])
                for m in ("answer_em", "answer_f1")
            ]
        ),
        "C. reader/context formulation": fmean(
            [
                max(0.0, macro["GOLD_SUPPORT"][m] - macro["FULL_CONTEXT"][m])
                for m in ("answer_em", "answer_f1")
            ]
        ),
    }
    ordered = sorted(losses.items(), key=lambda item: item[1], reverse=True)
    dominant = (
        ordered[0][0]
        if ordered[0][1] >= 0.02 and ordered[0][1] - ordered[1][1] >= 0.01
        else "D. mixed"
    )
    return {
        "dominant_downstream_gap": dominant,
        "transition_loss_scores": losses,
        "rule": "Unique dominant if largest mean macro EM/F1 loss is >=0.02 and exceeds second-largest by >=0.01; otherwise mixed.",
    }


def evaluate_phase(
    input_path: Path, manifest_path: Path, output_dir: Path, *, source_root: Path
) -> dict[str, Any]:
    prepared = validate_inputs(input_path, manifest_path)
    predictions_path = output_dir / "predictions.jsonl"
    freeze_path = output_dir / "generation_freeze.json"
    if not predictions_path.is_file() or not freeze_path.is_file():
        raise FileNotFoundError("Generation must complete and freeze before evaluation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "complete_frozen" or sha256(predictions_path) != freeze.get(
        "predictions_sha256"
    ):
        raise ValueError("Prediction freeze is absent or invalid")
    if sha256(input_path) != freeze.get("prepared_reader_inputs_sha256"):
        raise ValueError("Prepared reader inputs changed after generation")
    predictions, keys = validate_predictions(predictions_path, prepared, MODEL)
    expected = {prediction_key(row) for row in prepared}
    if keys != expected or len(predictions) != freeze["prediction_count"]:
        raise ValueError("Prediction population is incomplete")

    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))
    gold_started = utc_now()
    gold = {
        dataset: load_gold(dataset, source_root / DATASET_INFO[dataset][1], ids)
        for dataset, ids in ids_by_dataset.items()
    }
    per_example = []
    for row in predictions:
        gold_answer = gold[row["dataset"]][row["example_id"]]
        em, f1, precision, recall = score(row["domain"], row["predicted_answer"], gold_answer)
        per_example.append(
            {
                "schema_version": "dev_reader_diagnostic_example_v1",
                "dataset": row["dataset"],
                "domain": row["domain"],
                "split": "dev",
                "example_id": row["example_id"],
                "condition": row["condition"],
                "question": row["question"],
                "gold_answer": gold_answer,
                "predicted_answer": row["predicted_answer"],
                "answer_em": em,
                "answer_f1": f1,
                "answer_precision": precision,
                "answer_recall": recall,
                "answer_source": row["answer_source"],
                "reader_input_sha256": row["reader_input_sha256"],
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_example:
        grouped[(row["dataset"], row["condition"])].append(row)
    by_dataset = {
        dataset: {condition: aggregate(grouped[(dataset, condition)]) for condition in CONDITIONS}
        for dataset, _, _ in DATASETS
    }
    macro = {
        condition: {
            metric: fmean(by_dataset[d][condition][metric] for d, _, _ in DATASETS)
            for metric in ("answer_em", "answer_f1")
        }
        for condition in CONDITIONS
    }
    delta_names = (
        ("GOLD_SUPPORT - FULL_CONTEXT", "GOLD_SUPPORT", "FULL_CONTEXT"),
        ("ORACLE_GENERATOR_D - GOLD_SUPPORT", "ORACLE_GENERATOR_D", "GOLD_SUPPORT"),
        ("SAGE_ACTUAL - ORACLE_GENERATOR_D", "SAGE_ACTUAL", "ORACLE_GENERATOR_D"),
    )
    deltas = {
        name: {
            "by_dataset": {d: delta(by_dataset[d][a], by_dataset[d][b]) for d, _, _ in DATASETS},
            "macro": delta(macro[a], macro[b]),
        }
        for name, a, b in delta_names
    }
    ontology = {}
    for dataset, domain, _ in DATASETS:
        if domain != "ontology":
            continue
        ontology[dataset] = {}
        for condition in CONDITIONS:
            rows = grouped[(dataset, condition)]
            proofs = [row for row in rows if row["answer_source"] == "deterministic_owl_proof"]
            fallback = [row for row in rows if row["answer_source"] == "gpt-4.1-mini_fallback"]
            ontology[dataset][condition] = {
                "examples": len(rows),
                "deterministic_proof_resolved": len(proofs),
                "deterministic_proof_resolution_rate": len(proofs) / len(rows),
                "gpt_fallback": aggregate(fallback),
            }
    metrics = {
        "schema_version": "dev_reader_diagnostic_metrics_v1",
        "status": "complete_frozen",
        "model": MODEL,
        "by_dataset": by_dataset,
        "macro_average_across_datasets": macro,
        "deltas": deltas,
        "ontology_proof_and_fallback": ontology,
        "classification": classify(macro),
    }
    write_jsonl(output_dir / "per_example_evaluation.jsonl", per_example)
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "summary.md").write_text(summary(metrics), encoding="utf-8", newline="\n")
    lineage_path = output_dir / "lineage_metadata.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["evaluation"] = {
        "status": "complete_frozen",
        "gold_access_started_at_utc": gold_started,
        "predictions_sha256_verified_before_gold_access": freeze["predictions_sha256"],
        "answer_sources": {
            d: {"path": source, "sha256": sha256(source_root / source)} for d, _, source in DATASETS
        },
    }
    if sha256(predictions_path) != freeze["predictions_sha256"]:
        raise ValueError("Predictions changed during evaluation")
    write_json(lineage_path, lineage)
    write_json(
        output_dir / "artifact_manifest.json", artifact_manifest(output_dir, "complete_frozen")
    )
    return metrics


def fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def summary(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Matched four-condition DEV reader diagnostic",
        "",
        "All predictions were completed and hash-frozen before gold answers were opened.",
        "",
        "## Answer EM/F1",
        "",
        "| Dataset | Condition | n | EM | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, _, _ in DATASETS:
        for condition in CONDITIONS:
            row = metrics["by_dataset"][dataset][condition]
            lines.append(
                f"| {dataset} | {condition} | {row['examples']} | {fmt(row['answer_em'])} | {fmt(row['answer_f1'])} |"
            )
    lines += ["", "| Macro | Condition | EM | F1 |", "|---|---|---:|---:|"]
    for condition in CONDITIONS:
        row = metrics["macro_average_across_datasets"][condition]
        lines.append(
            f"| 10-dataset macro | {condition} | {fmt(row['answer_em'])} | {fmt(row['answer_f1'])} |"
        )
    lines += ["", "## Matched deltas", "", "| Scope | Delta | EM | F1 |", "|---|---|---:|---:|"]
    for name, values in metrics["deltas"].items():
        for dataset, row in values["by_dataset"].items():
            lines.append(
                f"| {dataset} | {name} | {fmt(row['answer_em'])} | {fmt(row['answer_f1'])} |"
            )
        row = values["macro"]
        lines.append(
            f"| 10-dataset macro | {name} | {fmt(row['answer_em'])} | {fmt(row['answer_f1'])} |"
        )
    lines += [
        "",
        "## Ontology deterministic proof and GPT fallback",
        "",
        "| Dataset | Condition | Proof resolved | Resolution rate | GPT fallback n | GPT fallback EM | GPT fallback F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in metrics["ontology_proof_and_fallback"].items():
        for condition, row in values.items():
            fallback = row["gpt_fallback"]
            lines.append(
                f"| {dataset} | {condition} | {row['deterministic_proof_resolved']} | {fmt(row['deterministic_proof_resolution_rate'])} | {fallback['examples']} | {fmt(fallback['answer_em'])} | {fmt(fallback['answer_f1'])} |"
            )
    classification = metrics["classification"]
    lines += [
        "",
        "## Dominant downstream gap",
        "",
        f"**{classification['dominant_downstream_gap']}**",
        "",
        classification["rule"],
        "",
    ]
    return "\n".join(lines)


def artifact_manifest(output_dir: Path, status: str) -> dict[str, Any]:
    names = (
        "predictions.jsonl",
        "generation_errors.jsonl",
        "generation_freeze.json",
        "metrics.json",
        "per_example_evaluation.jsonl",
        "summary.md",
        "lineage_metadata.json",
    )
    return {
        "schema_version": "dev_reader_diagnostic_artifact_manifest_v1",
        "status": status,
        "created_at_utc": utc_now(),
        "files": [
            {
                "path": name,
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256(output_dir / name),
            }
            for name in names
            if (output_dir / name).is_file()
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    generate.add_argument("--prepared-manifest", type=Path, default=DEFAULT_PREPARED_MANIFEST)
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    generate.add_argument("--model", default=MODEL)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    evaluate.add_argument("--prepared-manifest", type=Path, default=DEFAULT_PREPARED_MANIFEST)
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    evaluate.add_argument("--source-root", type=Path, default=ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "generate":
        result = generate_phase(
            args.input,
            args.prepared_manifest,
            args.output_dir,
            model=args.model,
            workers=args.workers,
            resume=args.resume,
        )
    else:
        result = evaluate_phase(
            args.input, args.prepared_manifest, args.output_dir, source_root=args.source_root
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
