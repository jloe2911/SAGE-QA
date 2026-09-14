"""Freeze, generate, and evaluate the three remaining manuscript baselines.

The phases are intentionally separate:

``freeze-inputs``
    Reuse the frozen Lexical ranking and frozen clean GNN-RAG ranking, select
    exactly k=1, construct the full clean context, and hash-lock all reader
    inputs.  No answer reader is called.
``generate``
    Read only the input freeze and call the existing GPT-4.1-mini readers.
    Original benchmark files and evaluation gold are not opened.
``evaluate``
    Require a complete prediction freeze before opening answer/support gold.

This module writes only beneath its dedicated output directory.  Existing
GNN/SAGE retrieval and end-to-end artifacts are validated but never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import parse_owl_context
from data_processing.build_2wiki_subgraph_dataset import (
    flatten_context as flatten_2wiki_context,
    normalize_2wiki_record,
)
from data_processing.build_hotpot_subgraph_dataset import (
    flatten_context as flatten_hotpot_context,
    normalize_hotpot_record,
)
from data_processing.prepare_familyowl_gnn_rag import unit_node
from data_processing.prepare_text_gnn_rag import evidence_node
from evaluation.evaluate_owl_qa_predictions import best_support_scores
from generation import run_production_test_answer_generation as reader_pipeline
from utils.paths import (
    baseline_answer_root,
    baseline_retrieval_root,
    generator_d_root,
    hard_pair_answer_root,
    hard_pair_test_retrieval_root,
    repo_path_arg,
    repo_root,
)


ROOT = repo_root()
MODEL_NAME = reader_pipeline.MODEL_NAME
CONFIGURATIONS = (
    ("lexical_subgraph", "k1"),
    ("gnn_rag", "k1"),
    ("full_context", "full"),
)
SUPPORT_CONFIGURATIONS = CONFIGURATIONS[:2]
DEFAULT_LEXICAL = baseline_retrieval_root() / "lexical_subgraph/per_example_retrieval.jsonl"
DEFAULT_GNN_RAG = baseline_retrieval_root() / "gnn_rag/predictions_frozen.jsonl"
DEFAULT_GNN_RAG_FREEZE = DEFAULT_GNN_RAG.parent / "prediction_freeze_manifest.json"
DEFAULT_CANDIDATE_ROOT = generator_d_root()
DEFAULT_SUPPORT_GOLD = hard_pair_test_retrieval_root() / "per_example_test_retrieval.jsonl"
DEFAULT_OUTPUT_DIR = baseline_answer_root()
PROTECTED_MANIFESTS = (
    hard_pair_test_retrieval_root() / "artifact_manifest.json",
    hard_pair_answer_root() / "artifact_manifest.json",
)
FORBIDDEN_INPUT_FIELDS = reader_pipeline.FORBIDDEN_GENERATION_FIELDS

FullContextLoader = Callable[
    [Path, Mapping[str, Mapping[str, Any]]], tuple[dict[str, list[str]], dict[str, Any]]
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_entry(manifest_path: Path, relative: str) -> Mapping[str, Any]:
    manifest = _load_json(manifest_path)
    files = manifest.get("files", {})
    entry = files.get(relative)
    if entry is None and isinstance(files, list):
        entry = next((item for item in files if item.get("path") == relative), None)
    if not isinstance(entry, dict):
        raise ValueError(f"Manifest {manifest_path} does not lock {relative}")
    return entry


def _validate_manifest_file(manifest_path: Path, relative: str) -> None:
    entry = _manifest_entry(manifest_path, relative)
    target = manifest_path.parent / relative
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.stat().st_size != int(entry["size_bytes"]):
        raise ValueError(f"Manifest size mismatch: {target}")
    if reader_pipeline.sha256(target) != entry["sha256"]:
        raise ValueError(f"Manifest hash mismatch: {target}")


def protected_snapshot(paths: Sequence[Path] = PROTECTED_MANIFESTS) -> dict[str, str]:
    """Validate and snapshot every manifest-covered frozen GNN/SAGE file."""
    snapshot: dict[str, str] = {}
    for manifest_path in paths:
        manifest_path = manifest_path.resolve()
        manifest = _load_json(manifest_path)
        entries = manifest.get("files", {})
        if isinstance(entries, list):
            entries = {item["path"]: item for item in entries}
        if not isinstance(entries, dict):
            raise ValueError(f"Invalid protected manifest: {manifest_path}")
        for relative, entry in entries.items():
            target = manifest_path.parent / relative
            if not target.is_file():
                raise FileNotFoundError(f"Protected artifact missing: {target}")
            actual = reader_pipeline.sha256(target)
            if actual != entry["sha256"] or target.stat().st_size != int(entry["size_bytes"]):
                raise ValueError(f"Protected artifact no longer matches its manifest: {target}")
            snapshot[str(target)] = actual
        snapshot[str(manifest_path)] = reader_pipeline.sha256(manifest_path)
    return snapshot


def assert_protected_unchanged(before: Mapping[str, str]) -> None:
    for raw_path, expected in before.items():
        path = Path(raw_path)
        if not path.is_file() or reader_pipeline.sha256(path) != expected:
            raise ValueError(f"Frozen GNN/SAGE artifact changed: {path}")


def _assert_output_is_separate(output_dir: Path, protected: Sequence[Path]) -> None:
    resolved = output_dir.resolve()
    for manifest in protected:
        protected_root = manifest.resolve().parent
        if resolved == protected_root or protected_root in resolved.parents:
            raise ValueError(f"Output directory overlaps frozen GNN/SAGE artifacts: {resolved}")


def _candidate_metadata_and_unit_maps(
    candidate_root: Path, ids_by_dataset: Mapping[str, set[str]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    unit_maps: dict[str, dict[str, str]] = defaultdict(dict)
    lineage: dict[str, Any] = {}
    for dataset, wanted in ids_by_dataset.items():
        directory, domain, _source = reader_pipeline.DATASET_INFO[dataset]
        path = candidate_root / directory / "test_subgraph_retrieval.jsonl"
        node_fn = evidence_node if domain == "text" else unit_node
        seen: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                forbidden = FORBIDDEN_INPUT_FIELDS & set(row)
                if forbidden:
                    raise ValueError(
                        f"Gold field in clean candidate row {path}:{line_number}: {sorted(forbidden)}"
                    )
                example_id = str(row.get("example_id") or "")
                if example_id not in wanted:
                    continue
                if example_id not in metadata:
                    metadata[example_id] = reader_pipeline._validate_candidate_metadata(
                        row, dataset
                    )
                for unit in row.get("subgraph_units", []) or []:
                    unit = str(unit)
                    unit_maps[example_id].setdefault(node_fn(unit), unit)
                seen.add(example_id)
        missing = wanted - seen
        if missing:
            raise ValueError(
                f"Clean candidate join incomplete for {dataset}: {sorted(missing)[:3]}"
            )
        lineage[dataset] = {
            "path": str(path),
            "sha256": reader_pipeline.sha256(path),
            "gold_fields_read": [],
        }
    return metadata, dict(unit_maps), lineage


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _skip_json_value(text: str, index: int) -> int:
    """Skip one JSON value without decoding it (used to bypass ontology QAs/gold)."""
    index = _skip_ws(text, index)
    if index >= len(text):
        raise ValueError("Unexpected end of JSON")
    if text[index] == '"':
        _value, end = json.JSONDecoder().raw_decode(text, index)
        return end
    if text[index] in "[{":
        opening = text[index]
        closing = "]" if opening == "[" else "}"
        depth = 0
        in_string = False
        escaped = False
        for pos in range(index, len(text)):
            char = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return pos + 1
        raise ValueError("Unterminated JSON container")
    end = index
    while end < len(text) and text[end] not in ",]}":
        end += 1
    return end


def extract_top_level_string_field(path: Path, field: str) -> list[str]:
    """Extract one top-level object field from a JSON array, skipping all others."""
    text = path.read_text(encoding="utf-8")
    index = _skip_ws(text, 0)
    if index >= len(text) or text[index] != "[":
        raise ValueError(f"Expected JSON array: {path}")
    index += 1
    values: list[str] = []
    decoder = json.JSONDecoder()
    while True:
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "]":
            return values
        if index >= len(text) or text[index] != "{":
            raise ValueError(f"Expected object in {path} at byte {index}")
        index += 1
        found: str | None = None
        while True:
            index = _skip_ws(text, index)
            if text[index] == "}":
                index += 1
                break
            key, index = decoder.raw_decode(text, index)
            index = _skip_ws(text, index)
            if text[index] != ":":
                raise ValueError(f"Expected colon in {path} at byte {index}")
            index = _skip_ws(text, index + 1)
            if key == field:
                value, index = decoder.raw_decode(text, index)
                if not isinstance(value, str):
                    raise ValueError(f"{field} is not a string in {path}")
                found = value
            else:
                index = _skip_json_value(text, index)
            index = _skip_ws(text, index)
            if text[index] == ",":
                index += 1
                continue
            if text[index] != "}":
                raise ValueError(f"Expected object delimiter in {path} at byte {index}")
        if found is None:
            raise KeyError(f"Missing {field!r} in array item {len(values)} of {path}")
        values.append(found)
        index = _skip_ws(text, index)
        if text[index] == ",":
            index += 1
        elif text[index] != "]":
            raise ValueError(f"Expected array delimiter in {path} at byte {index}")


def load_full_contexts(
    source_root: Path, metadata: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Load only context-visible fields; answer/support fields are never decoded."""
    import pandas as pd

    by_dataset: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for example_id, item in metadata.items():
        by_dataset[str(item.get("dataset") or item.get("source_name"))].append((example_id, item))
    contexts: dict[str, list[str]] = {}
    lineage: dict[str, Any] = {}
    for dataset, items in by_dataset.items():
        _directory, domain, relative = reader_pipeline.DATASET_INFO[dataset]
        source = source_root / relative
        if domain == "text":
            frame = pd.read_parquet(source, columns=["id", "context"])
            prefix = f"{dataset}__test__"
            wanted = {example_id for example_id, _ in items}
            normalizer = (
                normalize_2wiki_record if dataset == "2WikiMultiHopQA" else normalize_hotpot_record
            )
            flattener = (
                flatten_2wiki_context if dataset == "2WikiMultiHopQA" else flatten_hotpot_context
            )
            for raw in frame.to_dict(orient="records"):
                example_id = prefix + str(raw["id"])
                if example_id in wanted:
                    contexts[example_id] = [row["unit"] for row in flattener(normalizer(raw))]
            fields = ["id", "context"]
            policy = "Parquet column projection excludes answer and support columns"
        else:
            owl_contexts = extract_top_level_string_field(source, "OWL Context")
            for example_id, item in items:
                group_index = int(item["group_index"])
                contexts[example_id] = parse_owl_context(owl_contexts[group_index])
            fields = ["OWL Context"]
            policy = "Selective JSON scanner decodes only top-level OWL Context; QAs are skipped"
        lineage[dataset] = {
            "path": str(source),
            "sha256": reader_pipeline.sha256(source),
            "fields_decoded": fields,
            "gold_fields_decoded": [],
            "projection_policy": policy,
        }
    missing = set(metadata) - set(contexts)
    if missing:
        raise ValueError(f"Full-context join incomplete: {sorted(missing)[:3]}")
    return contexts, lineage


def _ordered_rows(path: Path) -> list[dict[str, Any]]:
    rows = reader_pipeline.load_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in seen:
            raise ValueError(f"Missing or duplicate example_id in {path}: {example_id!r}")
        seen.add(example_id)
    return rows


def freeze_inputs_phase(
    lexical_path: Path,
    gnn_rag_path: Path,
    gnn_rag_freeze_path: Path,
    candidate_root: Path,
    source_root: Path,
    output_dir: Path,
    *,
    full_context_loader: FullContextLoader = load_full_contexts,
    protected_manifests: Sequence[Path] = PROTECTED_MANIFESTS,
) -> dict[str, Any]:
    _assert_output_is_separate(output_dir, protected_manifests)
    before = protected_snapshot(protected_manifests)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    lexical_manifest = lexical_path.parent / "artifact_manifest.json"
    _validate_manifest_file(lexical_manifest, lexical_path.name)
    gnn_freeze = _load_json(gnn_rag_freeze_path)
    if gnn_freeze.get("status") != "predictions_frozen_gold_unopened":
        raise ValueError("GNN-RAG prediction freeze is not gold-unopened")
    if reader_pipeline.sha256(gnn_rag_path) != gnn_freeze.get("prediction_freeze_sha256"):
        raise ValueError("GNN-RAG frozen prediction hash mismatch")

    lexical_rows = _ordered_rows(lexical_path)
    gnn_rows = _ordered_rows(gnn_rag_path)
    lexical_by_id = {str(row["example_id"]): row for row in lexical_rows}
    gnn_by_id = {str(row["example_id"]): row for row in gnn_rows}
    if set(lexical_by_id) != set(gnn_by_id):
        raise ValueError("Lexical and GNN-RAG TEST populations differ")
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in gnn_rows:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))
    metadata, unit_maps, candidate_lineage = _candidate_metadata_and_unit_maps(
        candidate_root, ids_by_dataset
    )
    full_contexts, context_lineage = full_context_loader(source_root, metadata)

    frozen_rows: list[dict[str, Any]] = []
    for gnn_row in gnn_rows:
        example_id = str(gnn_row["example_id"])
        dataset = str(gnn_row["dataset"])
        domain = reader_pipeline.DATASET_INFO[dataset][1]
        lexical = lexical_by_id[example_id]
        selected = lexical.get("settings", {}).get("k1")
        if not isinstance(selected, dict) or int(selected.get("requested_k", -1)) != 1:
            raise ValueError(f"Lexical k=1 selection missing: {example_id}")
        if int(selected.get("actual_candidates_aggregated", -1)) not in {0, 1}:
            raise ValueError(f"Lexical k=1 aggregated more than one candidate: {example_id}")
        lexical_support = list(selected.get("retrieved_evidence_units", []))

        ranked = list(gnn_row.get("native_ranked_entities", []) or [])
        if ranked and int(ranked[0].get("rank", 1)) != 1:
            raise ValueError(f"GNN-RAG first frozen entity is not rank 1: {example_id}")
        gnn_support: list[str] = []
        if ranked:
            mapped = unit_maps[example_id].get(str(ranked[0]["entity"]))
            if mapped is None:
                raise ValueError(
                    f"GNN-RAG rank-1 entity has no clean candidate identity: {example_id}"
                )
            gnn_support = [mapped]

        values = {
            ("lexical_subgraph", "k1"): lexical_support,
            ("gnn_rag", "k1"): gnn_support,
            ("full_context", "full"): list(full_contexts[example_id]),
        }
        for method, setting in CONFIGURATIONS:
            support = values[(method, setting)]
            frozen_rows.append(
                {
                    "schema_version": "final_manuscript_reader_input_v1",
                    "dataset": dataset,
                    "domain": domain,
                    "split": "test",
                    "example_id": example_id,
                    "method": method,
                    "setting": setting,
                    "question": metadata[example_id]["question"],
                    "metadata": metadata[example_id],
                    "selected_support": support,
                    "support_sha256": reader_pipeline.canonical_hash(support),
                    "selection_contract": (
                        "existing lexical settings.k1"
                        if method == "lexical_subgraph"
                        else "rank 1 from clean GNN-RAG prediction freeze"
                        if method == "gnn_rag"
                        else "full clean TEST context"
                    ),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "frozen_reader_inputs.jsonl"
    reader_pipeline.write_jsonl(inputs_path, frozen_rows)
    freeze = {
        "schema_version": "final_manuscript_input_freeze_v1",
        "status": "inputs_frozen_gold_answers_unopened",
        "frozen_at_utc": reader_pipeline.utc_now(),
        "configurations": [{"method": m, "setting": s} for m, s in CONFIGURATIONS],
        "examples": len(gnn_rows),
        "reader_inputs": len(frozen_rows),
        "reader_inputs_sha256": reader_pipeline.sha256(inputs_path),
        "lexical_retrieval_sha256": reader_pipeline.sha256(lexical_path),
        "gnn_rag_prediction_freeze_sha256": reader_pipeline.sha256(gnn_rag_path),
        "candidate_lineage": candidate_lineage,
        "full_context_lineage": context_lineage,
        "gold_answers_opened": False,
        "adaptive_k_used": False,
        "retrieval_recomputed": False,
    }
    reader_pipeline.write_json(output_dir / "input_freeze.json", freeze)
    assert_protected_unchanged(before)
    return freeze


def _validate_input_freeze(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = _load_json(output_dir / "input_freeze.json")
    inputs_path = output_dir / "frozen_reader_inputs.jsonl"
    if freeze.get("status") != "inputs_frozen_gold_answers_unopened":
        raise ValueError("Reader input freeze is invalid")
    if reader_pipeline.sha256(inputs_path) != freeze.get("reader_inputs_sha256"):
        raise ValueError("Frozen reader inputs changed")
    rows = reader_pipeline.load_jsonl(inputs_path)
    expected_configs = set(CONFIGURATIONS)
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        config = (str(row["method"]), str(row["setting"]))
        key = (str(row["example_id"]), *config)
        if config not in expected_configs or key in keys:
            raise ValueError(f"Unexpected or duplicate reader input: {key}")
        if reader_pipeline.canonical_hash(row["selected_support"]) != row["support_sha256"]:
            raise ValueError(f"Reader input support hash mismatch: {key}")
        if FORBIDDEN_INPUT_FIELDS & set(row.get("metadata", {})):
            raise ValueError(f"Gold field reached frozen reader input: {key}")
        keys.add(key)
    if len(rows) != int(freeze["reader_inputs"]):
        raise ValueError("Frozen reader input count mismatch")
    return freeze, rows


def _prediction_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["example_id"]), str(row["method"]), str(row["setting"])


def generate_phase(
    output_dir: Path,
    *,
    model: str = MODEL_NAME,
    resume: bool = False,
    workers: int = 8,
    text_reader: reader_pipeline.TextReader = reader_pipeline.default_text_reader,
    ontology_reader: reader_pipeline.OntologyReader = reader_pipeline.default_ontology_reader,
    protected_manifests: Sequence[Path] = PROTECTED_MANIFESTS,
) -> dict[str, Any]:
    if model != MODEL_NAME:
        raise ValueError(f"Production model is frozen to {MODEL_NAME}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    before = protected_snapshot(protected_manifests)
    input_freeze, inputs = _validate_input_freeze(output_dir)
    expected = {_prediction_key(row) for row in inputs}
    predictions_path = output_dir / "predictions.jsonl"
    generation_freeze_path = output_dir / "generation_freeze.json"
    if predictions_path.exists() and not resume:
        raise FileExistsError("predictions.jsonl exists; use --resume")
    existing = reader_pipeline.load_jsonl(predictions_path) if predictions_path.exists() else []
    completed: set[tuple[str, str, str]] = set()
    for row in existing:
        key = _prediction_key(row)
        if key not in expected or key in completed:
            raise ValueError(f"Unexpected or duplicate prediction: {key}")
        if reader_pipeline.canonical_hash(row["selected_support"]) != row["support_sha256"]:
            raise ValueError(f"Existing prediction support changed: {key}")
        completed.add(key)
    if generation_freeze_path.exists():
        freeze = _load_json(generation_freeze_path)
        if completed != expected or reader_pipeline.sha256(predictions_path) != freeze.get(
            "predictions_sha256"
        ):
            raise ValueError("Frozen predictions changed or are incomplete")
        assert_protected_unchanged(before)
        return freeze

    jobs = [row for row in inputs if _prediction_key(row) not in completed]
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                reader_pipeline._run_reader_job,
                row["domain"],
                row["metadata"],
                list(row["selected_support"]),
                model,
                text_reader,
                ontology_reader,
            ): row
            for row in jobs
        }
        for future in as_completed(futures):
            row = futures[future]
            key = _prediction_key(row)
            try:
                result = future.result()
                prediction = {
                    "schema_version": "final_manuscript_prediction_v1",
                    "generation_status": "complete",
                    "dataset": row["dataset"],
                    "domain": row["domain"],
                    "split": "test",
                    "example_id": row["example_id"],
                    "method": row["method"],
                    "setting": row["setting"],
                    "question": row["question"],
                    "selected_support": row["selected_support"],
                    "support_sha256": row["support_sha256"],
                    "predicted_answer": str(result.get("predicted_answer", "")),
                    "explanation": str(result.get("explanation", "")),
                    "raw_response": result.get("raw_response"),
                    "answer_source": str(result.get("answer_source", "")),
                    "model": model,
                    "generated_at_utc": reader_pipeline.utc_now(),
                }
                reader_pipeline.append_jsonl(predictions_path, prediction)
                completed.add(key)
            except Exception as exc:  # pragma: no cover - production failure path
                errors.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
    if errors or completed != expected:
        if errors:
            reader_pipeline.write_jsonl(output_dir / "generation_errors.jsonl", errors)
        raise RuntimeError(f"Generation incomplete: {len(completed)}/{len(expected)}")
    persisted = reader_pipeline.load_jsonl(predictions_path)
    if {_prediction_key(row) for row in persisted} != expected or len(persisted) != len(expected):
        raise ValueError("Persisted prediction completion gate failed")
    freeze = {
        "schema_version": "final_manuscript_generation_freeze_v1",
        "status": "complete_frozen",
        "frozen_at_utc": reader_pipeline.utc_now(),
        "model": model,
        "configurations": [{"method": m, "setting": s} for m, s in CONFIGURATIONS],
        "prediction_count": len(persisted),
        "predictions_sha256": reader_pipeline.sha256(predictions_path),
        "reader_inputs_sha256": input_freeze["reader_inputs_sha256"],
        "gold_answers_opened": False,
    }
    reader_pipeline.write_json(generation_freeze_path, freeze)
    reader_pipeline.write_json(
        output_dir / "lineage_metadata.json",
        {
            "schema_version": "final_manuscript_baselines_lineage_v1",
            "input_freeze": input_freeze,
            "generation_freeze": freeze,
            "gold_boundary": {
                "generation_reads_only_frozen_reader_inputs": True,
                "generation_opened_original_benchmark_sources": False,
                "evaluation_requires_prediction_hash_freeze": True,
            },
            "environment": {"python": sys.version, "platform": platform.platform()},
        },
    )
    assert_protected_unchanged(before)
    return freeze


def _aggregate(rows: Sequence[Mapping[str, Any]], include_support: bool) -> dict[str, Any]:
    result = {
        "answer_examples": len(rows),
        "answer_em": fmean(float(row["answer_em"]) for row in rows),
        "answer_f1": fmean(float(row["answer_f1"]) for row in rows),
    }
    if include_support:
        evaluable = [row for row in rows if row["support_evaluable"]]
        result.update(
            {
                "support_examples": len(evaluable),
                "support_em": fmean(float(row["support_em"]) for row in evaluable)
                if evaluable
                else None,
                "support_f1": fmean(float(row["support_f1"]) for row in evaluable)
                if evaluable
                else None,
                "joint_em": fmean(float(row["joint_em"]) for row in evaluable)
                if evaluable
                else None,
                "joint_f1": fmean(float(row["joint_f1"]) for row in evaluable)
                if evaluable
                else None,
            }
        )
    return result


def _summary(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Final manuscript TEST baselines",
        "",
        "All reader inputs and predictions were hash-frozen before answer/support gold was opened.",
        "",
        "| Dataset | Baseline | Setting | Answer EM | Answer F1 | Support EM | Support F1 | Joint EM | Joint F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, methods in metrics["by_dataset"].items():
        for method, setting in CONFIGURATIONS:
            row = methods[method]
            value = lambda name: (
                "N/A" if name not in row or row[name] is None else f"{row[name]:.6f}"
            )
            lines.append(
                f"| {dataset} | {method} | {setting} | {value('answer_em')} | "
                f"{value('answer_f1')} | {value('support_em')} | {value('support_f1')} | "
                f"{value('joint_em')} | {value('joint_f1')} |"
            )
    lines.extend(
        [
            "",
            "Full Context is answer-only by protocol; support and joint metrics are neither computed nor stored.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifact_manifest(output_dir: Path, status: str) -> None:
    names = (
        "frozen_reader_inputs.jsonl",
        "input_freeze.json",
        "predictions.jsonl",
        "generation_errors.jsonl",
        "generation_freeze.json",
        "per_example_end_to_end.jsonl",
        "metrics.json",
        "summary.md",
        "lineage_metadata.json",
    )
    files = []
    for name in names:
        path = output_dir / name
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": reader_pipeline.sha256(path),
                }
            )
    reader_pipeline.write_json(
        output_dir / "artifact_manifest.json",
        {
            "schema_version": "final_manuscript_baselines_artifact_manifest_v1",
            "status": status,
            "self_excluded_from_hashes": True,
            "files": files,
        },
    )


def _load_support_gold(path: Path, wanted: set[str]) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("example_id") or "")
            if example_id in wanted:
                result[example_id] = [
                    list(map(str, gold)) for gold in row.get("gold_explanations", []) or []
                ]
    if set(result) != wanted:
        raise ValueError(f"Support-gold join incomplete: {sorted(wanted - set(result))[:3]}")
    return result


def evaluate_phase(
    output_dir: Path,
    support_gold_path: Path,
    *,
    source_root: Path = ROOT,
    gold_loader: reader_pipeline.GoldLoader = reader_pipeline.load_gold_answers,
    support_gold_loader: Callable[
        [Path, set[str]], dict[str, list[list[str]]]
    ] = _load_support_gold,
    protected_manifests: Sequence[Path] = PROTECTED_MANIFESTS,
) -> dict[str, Any]:
    before = protected_snapshot(protected_manifests)
    input_freeze, inputs = _validate_input_freeze(output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    generation_freeze = _load_json(output_dir / "generation_freeze.json")
    if generation_freeze.get("status") != "complete_frozen":
        raise ValueError("Generation is not completely frozen")
    if reader_pipeline.sha256(predictions_path) != generation_freeze.get("predictions_sha256"):
        raise ValueError("predictions.jsonl changed after the generation freeze")
    predictions = reader_pipeline.load_jsonl(predictions_path)
    expected = {_prediction_key(row) for row in inputs}
    if (
        len(predictions) != generation_freeze.get("prediction_count")
        or {_prediction_key(row) for row in predictions} != expected
    ):
        raise ValueError("Prediction population is incomplete")
    inputs_by_key = {_prediction_key(row): row for row in inputs}
    ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in inputs:
        ids_by_dataset[str(row["dataset"])].add(str(row["example_id"]))

    # This is the first answer/support gold access, after the byte-level freeze gate.
    gold_access_started_at = reader_pipeline.utc_now()
    answers: dict[str, dict[str, str]] = {}
    for dataset, ids in ids_by_dataset.items():
        source = source_root / reader_pipeline.DATASET_INFO[dataset][2]
        answers[dataset] = gold_loader(dataset, source, ids)
    all_ids = set().union(*ids_by_dataset.values())
    support_gold = support_gold_loader(support_gold_path, all_ids)

    details: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        key = _prediction_key(prediction)
        frozen_input = inputs_by_key[key]
        if prediction["selected_support"] != frozen_input["selected_support"]:
            raise ValueError(f"Prediction support differs from frozen input: {key}")
        dataset = str(prediction["dataset"])
        domain = str(prediction["domain"])
        gold_answer = answers[dataset][str(prediction["example_id"])]
        ans_em, ans_f1, ans_precision, ans_recall = reader_pipeline._answer_scores(
            domain, str(prediction["predicted_answer"]), gold_answer
        )
        row: dict[str, Any] = {
            "schema_version": "final_manuscript_evaluation_example_v1",
            "dataset": dataset,
            "domain": domain,
            "example_id": prediction["example_id"],
            "method": prediction["method"],
            "setting": prediction["setting"],
            "gold_answer": gold_answer,
            "predicted_answer": prediction["predicted_answer"],
            "answer_em": ans_em,
            "answer_f1": ans_f1,
        }
        include_support = prediction["method"] != "full_context"
        if include_support:
            score = (
                best_support_scores(
                    list(prediction["selected_support"]),
                    support_gold[str(prediction["example_id"])],
                )
                if support_gold[str(prediction["example_id"])]
                else None
            )
            row["support_evaluable"] = score is not None
            if score is not None:
                support_precision = float(score["prec"])
                support_recall = float(score["recall"])
                joint_precision = ans_precision * support_precision
                joint_recall = ans_recall * support_recall
                row.update(
                    {
                        "support_em": float(score["em"]),
                        "support_f1": float(score["f1"]),
                        "joint_em": ans_em * float(score["em"]),
                        "joint_f1": (
                            2 * joint_precision * joint_recall / (joint_precision + joint_recall)
                            if joint_precision + joint_recall
                            else 0.0
                        ),
                    }
                )
            else:
                row.update(
                    {"support_em": None, "support_f1": None, "joint_em": None, "joint_f1": None}
                )
        details.append(row)
        grouped[(dataset, str(prediction["method"]), str(prediction["setting"]))].append(row)

    by_dataset: dict[str, Any] = {}
    for dataset in ids_by_dataset:
        by_dataset[dataset] = {}
        for method, setting in CONFIGURATIONS:
            rows = grouped[(dataset, method, setting)]
            by_dataset[dataset][method] = _aggregate(rows, method != "full_context")
    metrics = {
        "schema_version": "final_manuscript_baselines_metrics_v1",
        "status": "complete_frozen",
        "model": generation_freeze["model"],
        "configurations": [{"method": m, "setting": s} for m, s in CONFIGURATIONS],
        "by_dataset": by_dataset,
    }
    reader_pipeline.write_jsonl(output_dir / "per_example_end_to_end.jsonl", details)
    reader_pipeline.write_json(output_dir / "metrics.json", metrics)
    (output_dir / "summary.md").write_text(_summary(metrics), encoding="utf-8", newline="\n")
    lineage_path = output_dir / "lineage_metadata.json"
    lineage = _load_json(lineage_path)
    lineage["evaluation"] = {
        "status": "complete_frozen",
        "gold_access_started_at_utc": gold_access_started_at,
        "predictions_sha256_verified_before_gold_access": generation_freeze["predictions_sha256"],
        "support_and_joint_metrics": ["lexical_subgraph", "gnn_rag"],
        "answer_only_metrics": ["full_context"],
        "support_gold_path": str(support_gold_path),
        "support_gold_sha256": reader_pipeline.sha256(support_gold_path),
    }
    reader_pipeline.write_json(lineage_path, lineage)
    _write_artifact_manifest(output_dir, "complete_frozen")
    assert_protected_unchanged(before)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    freeze = subparsers.add_parser("freeze-inputs")
    freeze.add_argument("--lexical", type=repo_path_arg, default=DEFAULT_LEXICAL)
    freeze.add_argument("--gnn-rag", type=repo_path_arg, default=DEFAULT_GNN_RAG)
    freeze.add_argument("--gnn-rag-freeze", type=repo_path_arg, default=DEFAULT_GNN_RAG_FREEZE)
    freeze.add_argument("--candidate-root", type=repo_path_arg, default=DEFAULT_CANDIDATE_ROOT)
    freeze.add_argument("--source-root", type=repo_path_arg, default=ROOT)
    freeze.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT_DIR)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT_DIR)
    generate.add_argument("--model", default=MODEL_NAME)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--resume", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--output-dir", type=repo_path_arg, default=DEFAULT_OUTPUT_DIR)
    evaluate.add_argument("--support-gold", type=repo_path_arg, default=DEFAULT_SUPPORT_GOLD)
    evaluate.add_argument("--source-root", type=repo_path_arg, default=ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze-inputs":
        freeze_inputs_phase(
            args.lexical,
            args.gnn_rag,
            args.gnn_rag_freeze,
            args.candidate_root,
            args.source_root,
            args.output_dir,
        )
    elif args.phase == "generate":
        generate_phase(args.output_dir, model=args.model, resume=args.resume, workers=args.workers)
    else:
        evaluate_phase(args.output_dir, args.support_gold, source_root=args.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
