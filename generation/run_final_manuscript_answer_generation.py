"""Final manuscript TEST reader preflight, generation, and evaluation.

This runner consumes the frozen Cross-Encoder / final SAGE-QA retrieval
selection file.  ``preflight`` materializes a hash-locked reader-input bundle
and reports the exact call population without invoking OpenAI.  ``generate``
uses only that bundle and the already frozen production reader functions.
``evaluate`` refuses to open answer gold until the complete prediction file is
frozen. Gold Support is answer-only and is restricted to examples with a
defined, non-empty gold support set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation import run_production_test_answer_generation as production


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = production.MODEL_NAME
CONFIGURATIONS = (
    ("cross_encoder", "k1"),
    ("final_sageqa", "k1"),
    ("final_sageqa", "adaptive"),
    ("gold_support", "oracle"),
)
RETRIEVAL_CONFIGURATIONS = CONFIGURATIONS[:-1]
DEFAULT_SELECTIONS = ROOT / (
    "outputs/final_results/question_candidate_cross_encoder_v1_adaptive_test_a40/"
    "test_predictions_frozen.jsonl"
)
DEFAULT_SELECTION_FREEZE = DEFAULT_SELECTIONS.parent / "generation_freeze.json"
DEFAULT_SELECTION_MANIFEST = DEFAULT_SELECTIONS.parent / "artifact_manifest.json"
DEFAULT_SUPPORT_GOLD = ROOT / (
    "outputs/final_results/production_generator_d_v2_hard_pair_test_retrieval/"
    "per_example_test_retrieval.jsonl"
)
DEFAULT_SUPPORT_MANIFEST = DEFAULT_SUPPORT_GOLD.parent / "artifact_manifest.json"
DEFAULT_CANDIDATE_ROOT = ROOT / "data/production_generator_d_v1"
DEFAULT_OUTPUT = ROOT / "outputs/final_results/final_manuscript_test_end_to_end"
OLD_METRICS = ROOT / "outputs/final_results/production_generator_d_v2_hard_pair_test_end_to_end/metrics.json"
BASELINE_METRICS = ROOT / "outputs/final_results/final_manuscript_baselines_test_end_to_end/metrics.json"


def _manifest_hash(manifest_path: Path, filename: str) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if isinstance(files, dict):
        value = files.get(filename)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("sha256"), str):
            return str(value["sha256"])
    if isinstance(files, list):
        for value in files:
            if Path(str(value.get("path", ""))).name == filename:
                return str(value["sha256"])
    raise ValueError(f"No hash for {filename} in {manifest_path}")


def _validate_sources(
    selections: Path, selection_freeze: Path, selection_manifest: Path,
    support_gold: Path, support_manifest: Path,
) -> dict[str, str]:
    actual_selection = production.sha256(selections)
    freeze = json.loads(selection_freeze.read_text(encoding="utf-8"))
    expected = str(freeze.get("predictions_sha256", ""))
    if actual_selection != expected:
        raise ValueError("Frozen selection hash differs from generation_freeze.json")
    if actual_selection != _manifest_hash(selection_manifest, selections.name):
        raise ValueError("Frozen selection hash differs from artifact_manifest.json")
    actual_support = production.sha256(support_gold)
    if actual_support != _manifest_hash(support_manifest, support_gold.name):
        raise ValueError("Support-gold artifact hash differs from artifact_manifest.json")
    return {
        "frozen_selections_sha256": actual_selection,
        "selection_freeze_sha256": production.sha256(selection_freeze),
        "selection_manifest_sha256": production.sha256(selection_manifest),
        "support_gold_sha256": actual_support,
        "support_manifest_sha256": production.sha256(support_manifest),
    }


def _load_selection_rows(path: Path) -> list[dict[str, Any]]:
    rows = production.load_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        eid = str(row.get("example_id", ""))
        if not eid or eid in seen:
            raise ValueError(f"Missing or duplicate selection example_id: {eid!r}")
        seen.add(eid)
        for method, setting in RETRIEVAL_CONFIGURATIONS:
            selected = row.get("methods", {}).get(method, {}).get(setting, {})
            units = selected.get("retrieved_evidence_units")
            if not isinstance(units, list) or not all(isinstance(x, str) for x in units):
                raise ValueError(f"Invalid frozen selection for {eid}/{method}/{setting}")
    return rows


def _load_support_gold(path: Path, expected_ids: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in production.load_jsonl(path):
        eid = str(row.get("example_id", ""))
        if eid not in expected_ids:
            continue
        alternatives = row.get("gold_explanations") or []
        # The evaluator permits alternative gold proof sets. For the single
        # oracle input, preserve the benchmark's persisted native order and
        # use its first alternative; no answer or prediction is consulted.
        units = list(alternatives[0]) if alternatives else []
        if not all(isinstance(x, str) for x in units):
            raise ValueError(f"Invalid gold support: {eid}")
        result[eid] = units
    if set(result) != expected_ids:
        raise ValueError("Support-gold join does not match frozen TEST selections")
    return result


def _input_key(row: Mapping[str, Any]) -> production.ReaderCacheKey:
    return production.reader_cache_key(
        dataset=str(row["dataset"]), example_id=str(row["example_id"]),
        question=str(row["question"]), support_hash=str(row["support_sha256"]),
        model=MODEL_NAME, domain=str(row["domain"]),
    )


def _audit_existing_denominators() -> dict[str, Any]:
    audited: dict[str, Any] = {}
    for label, path in (("old_gnn_and_sage", OLD_METRICS), ("lexical_gnn_rag_full", BASELINE_METRICS)):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        conditions: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for methods in metrics["by_dataset"].values():
            for method, value in methods.items():
                if "answer_examples" in value:
                    conditions[(method, "full" if method == "full_context" else "k1")].append(value)
                else:
                    for setting, row in value.items():
                        conditions[(method, setting)].append(row)
        summary: dict[str, Any] = {}
        for (method, setting), rows in conditions.items():
            answer_n = sum(int(row["answer_examples"]) for row in rows)
            support_values = [row.get("support_examples") for row in rows]
            support_n = None if all(value is None for value in support_values) else sum(
                int(value or 0) for value in support_values
            )
            if answer_n != 4249 or (support_n is not None and support_n != 3509):
                raise ValueError(f"Incompatible old denominator in {path}: {method}/{setting}")
            summary[f"{method}/{setting}"] = {"answer_examples": answer_n, "support_examples": support_n}
        audited[label] = {"path": str(path.relative_to(ROOT)), "sha256": production.sha256(path), "conditions": summary}
    return audited


def preflight_phase(
    selections: Path = DEFAULT_SELECTIONS,
    selection_freeze: Path = DEFAULT_SELECTION_FREEZE,
    selection_manifest: Path = DEFAULT_SELECTION_MANIFEST,
    support_gold: Path = DEFAULT_SUPPORT_GOLD,
    support_manifest: Path = DEFAULT_SUPPORT_MANIFEST,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    hashes = _validate_sources(
        selections, selection_freeze, selection_manifest, support_gold, support_manifest
    )
    selections_hash_before = production.sha256(selections)
    support_hash_before = production.sha256(support_gold)
    selected_rows = _load_selection_rows(selections)
    if len(selected_rows) != 4249:
        raise ValueError(f"Expected 4,249 TEST examples, found {len(selected_rows)}")
    expected_ids = {str(row["example_id"]) for row in selected_rows}
    metadata, candidate_lineage = production.load_generation_metadata(candidate_root, selected_rows)
    gold_support = _load_support_gold(support_gold, expected_ids)

    reader_inputs: list[dict[str, Any]] = []
    for row in selected_rows:
        eid, dataset = str(row["example_id"]), str(row["dataset"])
        domain = production.DATASET_INFO[dataset][1]
        supports = {
            (method, setting): list(row["methods"][method][setting]["retrieved_evidence_units"])
            for method, setting in RETRIEVAL_CONFIGURATIONS
        }
        if gold_support[eid]:
            supports[("gold_support", "oracle")] = list(gold_support[eid])
        for method, setting in CONFIGURATIONS:
            if (method, setting) not in supports:
                continue
            units = supports[(method, setting)]
            reader_inputs.append({
                "schema_version": "final_manuscript_reader_input_v1",
                "dataset": dataset, "domain": domain, "example_id": eid,
                "method": method, "setting": setting,
                "question": metadata[eid]["question"],
                "metadata": metadata[eid], "selected_support": units,
                "support_sha256": production.canonical_hash(units),
                "support_defined": bool(gold_support[eid]),
            })

    unique: dict[production.ReaderCacheKey, dict[str, Any]] = {}
    for row in reader_inputs:
        unique.setdefault(_input_key(row), row)
    deterministic = 0
    for row in unique.values():
        if row["domain"] == "ontology":
            proof = production.ontology_reader_module.infer_owl_boolean_answer(
                item=dict(row["metadata"]), support_units=list(row["selected_support"])
            )
            deterministic += proof is not None

    support_count = sum(bool(value) for value in gold_support.values())
    if support_count != 3509:
        raise ValueError(f"Expected 3,509 support-bearing examples, found {support_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "frozen_reader_inputs.jsonl"
    production.write_jsonl(inputs_path, reader_inputs)
    report = {
        "schema_version": "final_manuscript_preflight_v1", "status": "complete_no_api_calls",
        "answer_evaluation_examples": 4249, "support_joint_evaluation_examples": support_count,
        "old_metrics_denominator_recomputation_required": False,
        "old_metrics_audit_basis": (
            "Existing frozen evaluator uses every prediction for answer metrics and only "
            "rows with non-empty gold_explanations for support and joint metrics."
        ),
        "old_metrics_denominator_audit": _audit_existing_denominators(),
        "new_conditions": len(CONFIGURATIONS), "raw_prediction_rows": len(reader_inputs),
        "unique_reader_inputs": len(unique), "deterministic_proof_first_inputs": deterministic,
        "expected_openai_calls": len(unique) - deterministic,
        "gold_support_answer_examples": support_count,
        "gold_support_undefined_support_examples_excluded": 4249 - support_count,
        "frozen_reader_inputs_path": str(inputs_path.relative_to(ROOT)),
        "frozen_reader_inputs_sha256": production.sha256(inputs_path),
        "source_paths": {
            "frozen_selections": str(selections.relative_to(ROOT)),
            "support_gold": str(support_gold.relative_to(ROOT)),
        },
        "source_hashes": hashes, "candidate_metadata": candidate_lineage,
        "openai_calls_made": 0,
    }
    production.write_json(output_dir / "preflight_report.json", report)
    production.write_json(output_dir / "artifact_manifest.json", {
        "status": "preflight_complete_no_api_calls",
        "files": {
            "frozen_reader_inputs.jsonl": production.sha256(inputs_path),
            "preflight_report.json": production.sha256(output_dir / "preflight_report.json"),
        },
    })
    if production.sha256(selections) != selections_hash_before or production.sha256(support_gold) != support_hash_before:
        raise ValueError("A frozen source changed during preflight")
    return report


def _expected_keys(inputs: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    return {production.prediction_key(row) for row in inputs}


def generate_phase(output_dir: Path = DEFAULT_OUTPUT, *, workers: int = 8, resume: bool = False) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    report_path, inputs_path = output_dir / "preflight_report.json", output_dir / "frozen_reader_inputs.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if production.sha256(inputs_path) != report["frozen_reader_inputs_sha256"]:
        raise ValueError("Frozen reader inputs do not match preflight")
    inputs = production.load_jsonl(inputs_path)
    expected = _expected_keys(inputs)
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists() and not resume:
        raise FileExistsError("predictions.jsonl exists; use --resume")
    existing, completed = production._validated_existing_predictions(predictions_path, expected)
    jobs: dict[production.ReaderCacheKey, list[dict[str, Any]]] = defaultdict(list)
    for row in inputs:
        key = production.prediction_key(row)
        if key not in completed:
            jobs[_input_key(row)].append(dict(row))
    cache: dict[production.ReaderCacheKey, Mapping[str, Any]] = {}
    for row in existing:
        cache[production.reader_cache_key(
            dataset=str(row["dataset"]), example_id=str(row["example_id"]),
            question=str(row["question"]), support_hash=str(row["frozen_support_sha256"]),
            model=str(row["model"]), domain=str(row["domain"]),
        )] = row

    def persist(cache_key: production.ReaderCacheKey, result: Mapping[str, Any]) -> None:
        for source in jobs[cache_key]:
            job = {
                "method": source["method"], "setting": source["setting"],
                "example_id": source["example_id"], "dataset": source["dataset"],
                "domain": source["domain"], "metadata": source["metadata"],
                "support": source["selected_support"], "support_hash": source["support_sha256"],
            }
            production.append_jsonl(predictions_path, production._prediction_row(job, result, MODEL_NAME))
            completed.add(production.prediction_key(source))

    for key in list(jobs):
        if key in cache:
            persist(key, cache[key])
    pending = [key for key in jobs if key not in cache]
    errors_path = output_dir / "generation_errors.jsonl"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[Mapping[str, Any]], production.ReaderCacheKey] = {}
        for key in pending:
            row = jobs[key][0]
            futures[executor.submit(
                production._run_reader_job, row["domain"], row["metadata"],
                list(row["selected_support"]), MODEL_NAME,
                production.default_text_reader, production.default_ontology_reader,
            )] = key
        for future in as_completed(futures):
            key = futures[future]
            try:
                persist(key, future.result())
            except Exception as exc:
                production.append_jsonl(errors_path, {"reader_cache_key": list(key), "error": repr(exc)})
    if completed != expected:
        raise RuntimeError(f"Generation incomplete: {len(completed)}/{len(expected)}; rerun with --resume")
    rows, keys = production._validated_existing_predictions(predictions_path, expected)
    if keys != expected:
        raise ValueError("Persisted prediction population is incomplete")
    freeze = {
        "schema_version": "final_manuscript_generation_freeze_v1", "status": "complete_frozen",
        "model": MODEL_NAME, "prediction_count": len(rows),
        "predictions_sha256": production.sha256(predictions_path),
        "reader_inputs_sha256": report["frozen_reader_inputs_sha256"],
        "configurations": [{"method": m, "setting": s} for m, s in CONFIGURATIONS],
        "gold_answers_opened": False,
    }
    production.write_json(output_dir / "generation_freeze.json", freeze)
    return freeze


def _aggregate(rows: Sequence[Mapping[str, Any]], answer_only: bool = False) -> dict[str, Any]:
    support_rows = [row for row in rows if row["support_evaluable"]]
    result = {
        "answer_examples": len(rows),
        "answer_em": fmean(float(row["answer_em"]) for row in rows),
        "answer_f1": fmean(float(row["answer_f1"]) for row in rows),
    }
    if not answer_only:
        result.update({
            "support_examples": len(support_rows),
            **{name: fmean(float(row[name]) for row in support_rows)
               for name in ("support_em", "support_f1", "joint_em", "joint_f1")},
        })
    return result


def evaluate_phase(output_dir: Path = DEFAULT_OUTPUT, *, source_root: Path = ROOT) -> dict[str, Any]:
    inputs_path, predictions_path = output_dir / "frozen_reader_inputs.jsonl", output_dir / "predictions.jsonl"
    freeze = json.loads((output_dir / "generation_freeze.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "complete_frozen" or production.sha256(predictions_path) != freeze.get("predictions_sha256"):
        raise ValueError("Complete prediction freeze is missing or invalid")
    if production.sha256(inputs_path) != freeze.get("reader_inputs_sha256"):
        raise ValueError("Reader-input freeze is invalid")
    inputs = production.load_jsonl(inputs_path)
    predictions, keys = production._validated_existing_predictions(predictions_path, _expected_keys(inputs))
    if len(predictions) != 16256 or len(keys) != 16256:
        raise ValueError("Prediction population must contain 16,256 rows before gold access")
    by_input = {production.prediction_key(row): row for row in inputs}
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in inputs:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))
    gold: dict[str, dict[str, str]] = {}
    for dataset, ids in ids_by_dataset.items():
        gold[dataset] = production.load_gold_answers(
            dataset, source_root / production.DATASET_INFO[dataset][2], ids
        )
    support_by_id = {
        str(item["example_id"]): list(item.get("gold_explanations") or [])
        for item in production.load_jsonl(DEFAULT_SUPPORT_GOLD)
    }
    scored: list[dict[str, Any]] = []
    for pred in predictions:
        source = by_input[production.prediction_key(pred)]
        ans = production._answer_scores(
            str(pred["domain"]), str(pred["predicted_answer"]),
            gold[str(pred["dataset"])][str(pred["example_id"])],
        )
        row: dict[str, Any] = {
            "dataset": pred["dataset"], "domain": pred["domain"],
            "example_id": pred["example_id"], "method": pred["method"], "setting": pred["setting"],
            "predicted_answer": pred["predicted_answer"], "answer_em": ans[0], "answer_f1": ans[1],
            "support_evaluable": bool(source["support_defined"]),
        }
        if list(pred["selected_support"]) != list(source["selected_support"]):
            raise ValueError(
                f"Prediction support differs from frozen reader input: {production.prediction_key(pred)}"
            )
        if pred["method"] != "gold_support" and source["support_defined"]:
            # Support alternatives are opened only now, after the prediction freeze.
            support_score = production.best_support_scores(
                list(source["selected_support"]), support_by_id[str(pred["example_id"])]
            )
            row.update({"support_em": support_score["em"], "support_f1": support_score["f1"]})
            jp, jr = ans[2] * support_score["prec"], ans[3] * support_score["recall"]
            row.update({"joint_em": ans[0] * support_score["em"],
                        "joint_f1": 2 * jp * jr / (jp + jr) if jp + jr else 0.0})
        scored.append(row)
    production.write_jsonl(output_dir / "per_example_end_to_end.jsonl", scored)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[(str(row["dataset"]), str(row["method"]), str(row["setting"]))].append(row)
    by_dataset: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for (dataset, method, setting), rows in grouped.items():
        by_dataset[dataset][method][setting] = _aggregate(rows, answer_only=method == "gold_support")
    equal_dataset_macro: dict[str, Any] = {}
    pooled_per_example: dict[str, Any] = {}
    for method, setting in CONFIGURATIONS:
        dataset_rows = [by_dataset[name][method][setting] for name, *_ in production.DATASETS]
        fields = ("answer_em", "answer_f1") if method == "gold_support" else (
            "answer_em", "answer_f1", "support_em", "support_f1", "joint_em", "joint_f1"
        )
        equal_dataset_macro[f"{method}/{setting}"] = {
            field: fmean(float(row[field]) for row in dataset_rows) for field in fields
        }
        condition_rows = [row for row in scored if row["method"] == method and row["setting"] == setting]
        pooled_per_example[f"{method}/{setting}"] = _aggregate(
            condition_rows, answer_only=method == "gold_support"
        )
    metrics = {"schema_version": "final_manuscript_end_to_end_metrics_v1", "status": "complete_frozen",
               "denominators": {
                   "retrieval_answer": 4249, "retrieval_support_joint": 3509,
                   "gold_support_answer_only": 3509,
               },
               "primary_equal_dataset_macro": equal_dataset_macro,
               "secondary_pooled_per_example": pooled_per_example,
               "by_dataset": by_dataset}
    production.write_json(output_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    gen = sub.add_parser("generate")
    gen.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    gen.add_argument("--workers", type=int, default=8)
    gen.add_argument("--resume", action="store_true")
    eva = sub.add_parser("evaluate")
    eva.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    eva.add_argument("--source-root", type=Path, default=ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        print(json.dumps(preflight_phase(output_dir=args.output_dir), indent=2))
    elif args.phase == "generate":
        print(json.dumps(generate_phase(args.output_dir, workers=args.workers, resume=args.resume), indent=2))
    else:
        print(json.dumps(evaluate_phase(args.output_dir, source_root=args.source_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
