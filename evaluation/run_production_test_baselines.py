"""Frozen clean TEST retrieval for Lexical Subgraph and GNN-RAG only.

Retrieval is completely frozen before any gold source is opened.  The GNN-RAG
adapter written below is an inference representation of the existing Generator-D
candidate corpus; it deliberately omits the gold/answer injection performed by
the historical training adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import pandas as pd

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
from data_processing.prepare_familyowl_gnn_rag import (
    clean as owl_clean,
    extract_question_entities,
    relation_for_unit,
    unit_node,
)
from data_processing.prepare_text_gnn_rag import (
    clean as text_clean,
    collect_kg_units,
    evidence_node,
    kg_entity_node,
    kg_parts,
    kg_relation,
    question_node,
    sentence_parts,
    text_mentions_entity,
)
from evaluation.eval_lexical_hotpot_subgraph import lexical_score


DATASETS = (
    (
        "HotpotQA",
        "hotpotqa",
        "text",
        "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
    ),
    (
        "2WikiMultiHopQA",
        "2wiki",
        "text",
        "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
    ),
    ("FamilyOWL_1hop", "familyowl_1hop", "ontology", "data/raw/family/FamilyOWL_1hop.json"),
    ("FamilyOWL_2hop", "familyowl_2hop", "ontology", "data/raw/family/FamilyOWL_2hop.json"),
    ("pizza_100_1hop", "pizza_100_1hop", "ontology", "data/raw/pizza_100/pizza_100_1hop.json"),
    ("pizza_100_2hop", "pizza_100_2hop", "ontology", "data/raw/pizza_100/pizza_100_2hop.json"),
    ("pizza_250_1hop", "pizza_250_1hop", "ontology", "data/raw/pizza_250/pizza_250_1hop.json"),
    ("pizza_250_2hop", "pizza_250_2hop", "ontology", "data/raw/pizza_250/pizza_250_2hop.json"),
    ("OWL2Bench_1hop", "owl2bench_1hop", "ontology", "data/raw/owl2bench/OWL2Bench_1hop.json"),
    ("OWL2Bench_2hop", "owl2bench_2hop", "ontology", "data/raw/owl2bench/OWL2Bench_2hop.json"),
)
PROHIBITED_TEST_FIELDS = {
    "answer",
    "answers",
    "evidences",
    "supporting_facts",
    "raw_supporting_facts",
    "gold_explanations",
    "gold_units",
    "gold_support_units",
    "label",
    "rank_target",
    "best_set_f1_to_gold",
    "best_set_precision_to_gold",
    "best_set_recall_to_gold",
    "exact_match_any_gold",
    "contains_any_gold_explanation",
}
EXPECTED_CORPUS_MANIFEST_SHA256 = "fb77a4f30019d297ef6dbed22dec83bc86bedeaed5281fa51d2f1e7a8c33b076"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def grouped_clean_rows(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
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
                    f"TEST candidate row contains prohibited fields: {sorted(forbidden)}"
                )
            if row.get("gold_available_during_candidate_generation") is not False:
                raise ValueError(f"TEST row lacks gold-free generation assertion: {example_id}")
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


def deduplicated_union(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for unit in candidate.get("subgraph_units", []) or []:
            unit = str(unit)
            if unit not in seen:
                seen.add(unit)
                result.append(unit)
    return result


def freeze_lexical(test_path: Path, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen = []
    candidate_rows = 0
    for example_index, (example_id, rows) in enumerate(grouped_clean_rows(test_path), start=1):
        candidate_rows += len(rows)
        ranked = sorted(enumerate(rows), key=lambda pair: (-lexical_score(pair[1]), pair[0]))
        ranked_rows = [row for _, row in ranked]
        settings = {}
        for k in (1, 3):
            selected = ranked_rows[:k]
            settings[f"k{k}"] = {
                "requested_k": k,
                "actual_candidates_aggregated": len(selected),
                "retrieved_evidence_units": deduplicated_union(selected),
                "ranked_candidates": [
                    {
                        "rank": rank,
                        "lexical_score": lexical_score(row),
                        "generation_rank": row.get("generation_rank"),
                        "subgraph_units": list(row.get("subgraph_units", []) or []),
                    }
                    for rank, row in enumerate(selected, start=1)
                ],
            }
        frozen.append(
            {
                "dataset": dataset,
                "split": "test",
                "example_id": example_id,
                "question": str(rows[0].get("question", "")),
                "settings": settings,
            }
        )
        if example_index % 100 == 0:
            print(f"Lexical {dataset}: froze {example_index} examples", flush=True)
    return frozen, {"candidate_rows": candidate_rows, "examples": len(frozen)}


def text_native_sample(
    rows: list[dict[str, Any]], max_candidates: int
) -> tuple[dict[str, Any], dict[str, str]]:
    seed = rows[0]
    example_id = str(seed["example_id"])
    q_node = question_node(example_id)
    candidate_units = deduplicated_union(rows[:max_candidates])
    kg_units = collect_kg_units(rows)
    tuples: list[list[str]] = []
    entities: set[str] = {q_node}
    unit_map: dict[str, str] = {}

    for unit in candidate_units:
        title, idx, _ = sentence_parts(unit)
        title_node = f"TITLE::{title}"
        ev_node = evidence_node(unit)
        unit_map[ev_node] = unit
        entities.update((title_node, ev_node))
        tuples.extend(
            (
                [q_node, "mentions", title_node],
                [title_node, f"sentence_{idx}", ev_node],
                [q_node, "retrieved_evidence", ev_node],
            )
        )

    for unit in kg_units:
        parts = kg_parts(unit)
        if parts is None:
            continue
        subject, predicate, obj = parts
        subj_node, obj_node = kg_entity_node(subject), kg_entity_node(obj)
        relation = kg_relation(predicate)
        entities.update((subj_node, obj_node))
        tuples.extend(
            (
                [subj_node, relation, obj_node],
                [q_node, "kg_context", subj_node],
                [q_node, "kg_context", obj_node],
            )
        )
        for sent_unit in candidate_units:
            title, _, sentence = sentence_parts(sent_unit)
            ev_node = evidence_node(sent_unit)
            haystack = f"{title} {sentence}"
            if text_mentions_entity(haystack, subject):
                tuples.extend(
                    (
                        [ev_node, "mentions_kg_entity", subj_node],
                        [subj_node, "mentioned_in_sentence", ev_node],
                    )
                )
            if text_mentions_entity(haystack, obj):
                tuples.extend(
                    (
                        [ev_node, "mentions_kg_entity", obj_node],
                        [obj_node, "mentioned_in_sentence", ev_node],
                    )
                )

    sample = {
        "id": example_id,
        "question": text_clean(seed.get("question", "")),
        "entities": [q_node],
        "q_entity": [q_node],
        "choices": [],
        "graph": tuples,
        # The upstream loader requires the key, but inference labels may be empty.
        # ReaRev predictions do not consume answer_dist; it is used only for loss/
        # metrics, which are deliberately ignored for this native retrieval run.
        "answers": [],
        "subgraph": {"entities": sorted(entities), "tuples": tuples},
    }
    return sample, unit_map


def ontology_native_sample(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    seed = rows[0]
    example_id = str(seed["example_id"])
    candidate_units = deduplicated_union(rows)
    q_entities = extract_question_entities(seed)
    if not q_entities:
        q_entities = (
            [unit_node(candidate_units[0])]
            if candidate_units
            else [f"QUESTION::{hashlib.sha256(example_id.encode()).hexdigest()[:24]}"]
        )
    entities: set[str] = set(q_entities)
    tuples: list[list[str]] = []
    unit_map: dict[str, str] = {}
    for unit in candidate_units:
        node = unit_node(unit)
        unit_map[node] = unit
        entities.add(node)
        relation = relation_for_unit(unit)
        for entity in q_entities:
            if entity != node:
                tuples.append([entity, relation, node])
    sample = {
        "id": example_id,
        "question": owl_clean(seed.get("question", "")),
        "entities": q_entities,
        "q_entity": q_entities,
        "choices": [],
        "graph": tuples,
        "answers": [],
        "subgraph": {"entities": sorted(entities), "tuples": tuples},
    }
    return sample, unit_map


def prepare_clean_native_test(
    test_path: Path,
    native_dir: Path,
    source_native_dir: Path,
    domain: str,
    text_max_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    native_dir.mkdir(parents=True)
    for name in ("dev.json", "relations.txt", "vocab.txt"):
        shutil.copy2(source_native_dir / name, native_dir / name)
    old_entities = (source_native_dir / "entities.txt").read_text(encoding="utf-8").splitlines()
    known_entities = set(old_entities)
    relations = set((native_dir / "relations.txt").read_text(encoding="utf-8").splitlines())
    samples: list[dict[str, Any]] = []
    unit_maps: list[dict[str, str]] = []
    missing_relations: set[str] = set()
    with (native_dir / "test.json").open("x", encoding="utf-8", newline="\n") as handle:
        for example_index, (_, rows) in enumerate(grouped_clean_rows(test_path), start=1):
            sample, unit_map = (
                text_native_sample(rows, text_max_candidates)
                if domain == "text"
                else ontology_native_sample(rows)
            )
            for _, relation, _ in sample["graph"]:
                if relation not in relations:
                    missing_relations.add(relation)
            known_entities.update(sample["subgraph"]["entities"])
            samples.append(sample)
            unit_maps.append(unit_map)
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if example_index % 100 == 0:
                print(
                    f"GNN-RAG adapter {test_path.parent.name}: froze {example_index} examples",
                    flush=True,
                )
    if missing_relations:
        raise ValueError(
            f"Clean TEST uses relations absent from frozen GNN-RAG vocabulary: {sorted(missing_relations)[:10]}"
        )
    appended = sorted(known_entities - set(old_entities))
    (native_dir / "entities.txt").write_text(
        "\n".join([*old_entities, *appended]) + "\n", encoding="utf-8", newline="\n"
    )
    return (
        samples,
        unit_maps,
        {
            "examples": len(samples),
            "text_max_candidates": text_max_candidates if domain == "text" else None,
            "historical_entity_count": len(old_entities),
            "appended_inference_only_entities": len(appended),
            "relation_vocabulary_unchanged": True,
            "gold_or_answer_fields_copied_to_native_test": False,
        },
    )


def run_native_retriever(
    root: Path, native_dir: Path, checkpoint_source: Path, run_dir: Path, adapter_key: str
) -> tuple[Path, list[str]]:
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_name = checkpoint_source.name
    shutil.copy2(checkpoint_source, checkpoint_dir / checkpoint_name)
    experiment = f"clean_test_rearev_lstm_sageqa-{adapter_key}"
    gnn_root = root / "third_party/GNN-RAG/gnn"
    command = [
        sys.executable,
        str(gnn_root / "main.py"),
        "ReaRev",
        "--entity_dim",
        "50",
        "--data_folder",
        str(native_dir.resolve()) + "\\",
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
        str(checkpoint_dir.resolve()),
        "--experiment_name",
        experiment,
        "--load_experiment",
        checkpoint_name,
        "--is_eval",
        "--name",
        f"sageqa-{adapter_key}",
    ]
    log_path = run_dir / "native_retriever.log"
    with log_path.open("x", encoding="utf-8", newline="\n") as log_handle:
        log_handle.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log_handle.flush()
        result = subprocess.run(
            command,
            cwd=gnn_root,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"GNN-RAG failed for {adapter_key}; see {log_path}")
    info_path = checkpoint_dir / f"{experiment}_test.info"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing native GNN-RAG retrieval output: {info_path}")
    return info_path, command


def path_string(path: Sequence[tuple[str, str, str]]) -> str:
    if not path:
        return ""
    result = f"{path[0][0]} -> {path[0][1]} -> {path[0][2]}"
    for _, relation, tail in path[1:]:
        result += f" -> {relation} -> {tail}"
    return result.strip()


def native_paths(
    sample: Mapping[str, Any], candidates: Sequence[Sequence[Any]]
) -> tuple[list[list[list[str]]], list[str]]:
    graph = nx.Graph()
    for head, relation, tail in sample["graph"]:
        graph.add_edge(head, tail, relation=str(relation).strip())
    result: list[list[tuple[str, str, str]]] = []
    candidate_entities = [str(item[0]) for item in candidates]
    for head in sample["q_entity"]:
        if head not in graph:
            continue
        for tail in candidate_entities:
            if tail not in graph:
                continue
            try:
                for nodes in nx.all_shortest_paths(graph, head, tail):
                    result.append(
                        [
                            (
                                nodes[index],
                                graph[nodes[index]][nodes[index + 1]]["relation"],
                                nodes[index + 1],
                            )
                            for index in range(len(nodes) - 1)
                        ]
                    )
            except nx.NetworkXException:
                pass
    return [[list(edge) for edge in path] for path in result], [
        path_string(path) for path in result
    ]


def freeze_native_output(
    dataset: str,
    samples: Sequence[Mapping[str, Any]],
    unit_maps: Sequence[Mapping[str, str]],
    info_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info_rows = [
        json.loads(line)
        for line in info_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(info_rows) != len(samples):
        raise ValueError(
            f"GNN-RAG output count mismatch for {dataset}: {len(info_rows)} != {len(samples)}"
        )
    records = []
    for sample, unit_map, info in zip(samples, unit_maps, info_rows):
        if str(info.get("question", "")) != str(sample["question"]):
            raise ValueError(f"GNN-RAG output order/question mismatch: {sample['id']}")
        candidates = info.get("cand", []) or []
        paths, strings = native_paths(sample, candidates)
        retrieved_units: list[str] = []
        seen_units: set[str] = set()
        for path in paths:
            for head, _, tail in path:
                for node in (head, tail):
                    if node in unit_map and unit_map[node] not in seen_units:
                        seen_units.add(unit_map[node])
                        retrieved_units.append(unit_map[node])
        action_trace = []
        index = 0
        while str(index) in info:
            action_trace.append(info[str(index)])
            index += 1
        records.append(
            {
                "dataset": dataset,
                "split": "test",
                "example_id": sample["id"],
                "question": sample["question"],
                "native_ranked_entities": [
                    {"entity": item[0], "probability": item[1]} for item in candidates
                ],
                "native_relation_action_trace": action_trace,
                "retrieved_paths": paths,
                "retrieved_context": strings,
                "retrieved_evidence_units": retrieved_units,
                "native_graph_nodes": len(sample["subgraph"]["entities"]),
                "native_graph_edges": len(sample["graph"]),
            }
        )

    def mean(values: Sequence[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    diagnostics = {
        "examples": len(records),
        "examples_with_ranked_entities": sum(
            bool(row["native_ranked_entities"]) for row in records
        ),
        "examples_with_nonempty_path_context": sum(
            bool(row["retrieved_context"]) for row in records
        ),
        "examples_with_mapped_evidence_units": sum(
            bool(row["retrieved_evidence_units"]) for row in records
        ),
        "mean_ranked_entities": mean([len(row["native_ranked_entities"]) for row in records]),
        "median_ranked_entities": statistics.median(
            [len(row["native_ranked_entities"]) for row in records]
        )
        if records
        else 0.0,
        "mean_retrieved_paths": mean([len(row["retrieved_paths"]) for row in records]),
        "mean_mapped_evidence_units": mean(
            [len(row["retrieved_evidence_units"]) for row in records]
        ),
        "mean_native_graph_nodes": mean([row["native_graph_nodes"] for row in records]),
        "mean_native_graph_edges": mean([row["native_graph_edges"] for row in records]),
    }
    return records, diagnostics


def text_gold(source: Path, dataset: str, example_ids: set[str]) -> dict[str, list[list[str]]]:
    # Gold is opened only after both baselines are frozen. Answers and 2Wiki evidences are never read.
    frame = pd.read_parquet(source, columns=["id", "context", "supporting_facts"])
    normalize = normalize_2wiki_record if dataset == "2WikiMultiHopQA" else normalize_hotpot_record
    flatten = flatten_2wiki_context if dataset == "2WikiMultiHopQA" else flatten_hotpot_context
    gold_fn = get_2wiki_gold if dataset == "2WikiMultiHopQA" else get_hotpot_gold
    prefix = f"{dataset}__test__"
    result = {}
    for raw in frame.to_dict(orient="records"):
        example_id = prefix + str(raw.get("id") or "")
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
        group_index, qa_index = int(parts[1][1:]), int(parts[2][1:])
        result[example_id] = get_gold_explanations(groups[group_index]["QAs"][qa_index])
    return result


def best_scores(predicted: Sequence[Any], golds: Sequence[Sequence[Any]]) -> dict[str, float]:
    predicted_set = set(map(str, predicted))
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for gold in golds:
        gold_set = set(map(str, gold))
        overlap = len(predicted_set & gold_set)
        precision = overlap / len(predicted_set) if predicted_set else 0.0
        recall = overlap / len(gold_set) if gold_set else 0.0
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        if f1 > best["f1"]:
            best = {"precision": precision, "recall": recall, "f1": f1}
    return best


def evaluate_lexical(
    records: list[dict[str, Any]], golds: Mapping[str, Sequence[Sequence[Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_setting = {"k1": [], "k3": []}
    excluded = 0
    for record in records:
        references = golds[record["example_id"]]
        eligible = bool(references)
        record["evaluation_eligible"] = eligible
        record["exclusion_reason"] = None if eligible else "no_gold_support_explanation"
        if not eligible:
            excluded += 1
        for setting in ("k1", "k3"):
            scores = (
                best_scores(record["settings"][setting]["retrieved_evidence_units"], references)
                if eligible
                else {"precision": None, "recall": None, "f1": None}
            )
            record["settings"][setting].update(scores)
            if eligible:
                by_setting[setting].append(scores)
    metrics = {}
    for setting, rows in by_setting.items():
        metrics[setting] = {
            metric: statistics.fmean(row[metric] for row in rows)
            for metric in ("precision", "recall", "f1")
        }
    return {
        "prediction_examples": len(records),
        "evaluation_examples": len(records) - excluded,
        "excluded_no_gold_support": excluded,
        "settings": metrics,
    }, records


def lexical_summary(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Lexical Subgraph — clean frozen TEST retrieval",
        "",
        "Macro P/R/F1 over support-bearing TEST examples. Retrieval was frozen before gold was opened.",
        "",
        "| Dataset | Setting | Examples | Evaluated | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["datasets"]:
        for setting in ("k1", "k3"):
            score = row["settings"][setting]
            lines.append(
                f"| {row['dataset']} | {setting} | {row['prediction_examples']} | {row['evaluation_examples']} | {score['precision']:.6f} | {score['recall']:.6f} | {score['f1']:.6f} |"
            )
    lines.extend(
        [
            "",
            "No adaptive-k policy was introduced. Exact retrieved supports are in `per_example_retrieval.jsonl`.",
            "",
        ]
    )
    return "\n".join(lines)


def gnn_summary(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# GNN-RAG — clean frozen TEST native retrieval diagnostics",
        "",
        "GNN-RAG retains its native ranked-entity and shortest-path context semantics. These diagnostics are not forced into the support-set P/R/F1 table.",
        "",
        "| Dataset | Examples | Ranked entity context | Nonempty paths | Mapped evidence | Mean entities | Mean paths |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["datasets"]:
        lines.append(
            f"| {row['dataset']} | {row['examples']} | {row['examples_with_ranked_entities']} | {row['examples_with_nonempty_path_context']} | {row['examples_with_mapped_evidence_units']} | {row['mean_ranked_entities']:.4f} | {row['mean_retrieved_paths']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Exact ranked entities, relation-action traces, paths, and downstream reader context are in `per_example_retrieval.jsonl`.",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_manifest(directory: Path, required_names: Sequence[str]) -> None:
    write_json(
        directory / "artifact_manifest.json",
        {
            "self_excluded_from_hashes": True,
            "files": {
                name: {
                    "size_bytes": (directory / name).stat().st_size,
                    "sha256": sha256(directory / name),
                }
                for name in required_names
            },
        },
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def validate_artifact_manifest(directory: Path) -> None:
    manifest_path = directory / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completion manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    for name, expected in manifest.get("files", {}).items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Manifest member is missing: {path}")
        if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Manifest member changed: {path}")


def common_run_state(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    corpus_files = [path for path in sorted(args.data_root.rglob("*")) if path.is_file()]
    corpus_manifest = [
        {
            "path": path.relative_to(args.data_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in corpus_files
    ]
    corpus_manifest_sha = canonical_hash(corpus_manifest)
    split_file_manifest = [
        row for row in corpus_manifest if row["path"].endswith("_subgraph_retrieval.jsonl")
    ]
    split_file_manifest_blob = "".join(
        f"{row['path']}\t{row['sha256'].lower()}\n" for row in split_file_manifest
    ).encode("utf-8")
    split_file_manifest_sha = hashlib.sha256(split_file_manifest_blob).hexdigest()
    if split_file_manifest_sha != EXPECTED_CORPUS_MANIFEST_SHA256:
        raise ValueError(f"Frozen split-file corpus manifest mismatch: {split_file_manifest_sha}")

    protected_checkpoint_root = root / "checkpoints/production_generator_d_v1"
    completed_eval_dir = root / "outputs/final_results/production_generator_d_v1_test_retrieval"
    return {
        "corpus_files": corpus_files,
        "corpus_manifest": corpus_manifest,
        "corpus_manifest_sha256": corpus_manifest_sha,
        "split_file_manifest": split_file_manifest,
        "split_file_manifest_sha256": split_file_manifest_sha,
        "protected_checkpoint_root": protected_checkpoint_root,
        "protected_before": file_manifest(protected_checkpoint_root),
        "completed_eval_dir": completed_eval_dir,
        "completed_eval_before": file_manifest(completed_eval_dir)
        if completed_eval_dir.exists()
        else None,
    }


def assert_protected_unchanged(args: argparse.Namespace, state: Mapping[str, Any]) -> None:
    corpus_after = [
        {
            "path": path.relative_to(args.data_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in state["corpus_files"]
    ]
    if state["corpus_manifest"] != corpus_after:
        raise RuntimeError("Frozen production corpus changed during baseline evaluation")
    if state["protected_before"] != file_manifest(state["protected_checkpoint_root"]):
        raise RuntimeError(
            "Completed GNN/SAGE-QA checkpoint artifacts changed during baseline evaluation"
        )
    completed_eval_dir = state["completed_eval_dir"]
    completed_after = file_manifest(completed_eval_dir) if completed_eval_dir.exists() else None
    if state["completed_eval_before"] != completed_after:
        raise RuntimeError(
            "Completed GNN/SAGE-QA final TEST artifacts changed during baseline evaluation"
        )


def base_lineage(
    args: argparse.Namespace,
    root: Path,
    state: Mapping[str, Any],
    started_at: str,
    retrieval_frozen_at: str,
) -> dict[str, Any]:
    return {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval_frozen_at_utc": retrieval_frozen_at,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_status_before": subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).splitlines(),
        "python": platform.python_version(),
        "evaluation_command_argv": [
            sys.executable,
            str(Path(__file__).relative_to(root)),
            *sys.argv[1:],
        ],
        "evaluation_command_display": subprocess.list2cmdline(
            [sys.executable, str(Path(__file__).relative_to(root)), *sys.argv[1:]]
        ),
        "corpus_root": str(args.data_root),
        "corpus_manifest": state["corpus_manifest"],
        "corpus_manifest_sha256": state["corpus_manifest_sha256"],
        "split_file_manifest": state["split_file_manifest"],
        "split_file_manifest_sha256": state["split_file_manifest_sha256"],
        "generator_d_version": "generator_d_frozen_v1",
        "production_builder_version": "generator_d_production_v1",
        "protocol": {
            "retrieval_frozen_before_gold_access": True,
            "two_wiki_evidences_read": False,
            "two_wiki_answers_read": False,
            "ontology_answers_read_during_retrieval": False,
            "answer_generation_run": False,
            "generator_d_modified_or_rerun": False,
            "gnns_retrained": False,
            "adaptive_k_refitted_or_used": False,
            "test_tuning_performed": False,
            "manuscript_modified": False,
            "commit_or_push_performed": False,
            "completed_gnn_subgraph_or_sageqa_test_evaluation_rerun": False,
            "protected_checkpoint_manifest_unchanged": True,
            "completed_evaluation_manifest_unchanged": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/production_generator_d_v1"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_results/production_generator_d_v1_test_baselines"),
    )
    parser.add_argument("--text-max-candidates", type=int, default=5)
    parser.add_argument("--stage", choices=("lexical", "gnn", "finalize"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.text_max_candidates != 5:
        raise ValueError("The frozen manuscript GNN-RAG text candidate cap is 5")
    root = Path(__file__).resolve().parents[1]
    args.data_root = (
        (root / args.data_root).resolve()
        if not args.data_root.is_absolute()
        else args.data_root.resolve()
    )
    output_dir = (
        (root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    lexical_dir, gnn_dir = output_dir / "lexical_subgraph", output_dir / "gnn_rag"
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    if args.stage == "finalize":
        validate_artifact_manifest(lexical_dir)
        validate_artifact_manifest(gnn_dir)
        lexical_metrics = read_json(lexical_dir / "metrics.json")
        gnn_metrics = read_json(gnn_dir / "native_retrieval_diagnostics.json")
        lexical_count = sum(row["prediction_examples"] for row in lexical_metrics["datasets"])
        gnn_count = sum(row["examples"] for row in gnn_metrics["datasets"])
        confirmation_path = output_dir / "run_confirmation.json"
        if confirmation_path.exists():
            raise FileExistsError(f"Refusing to overwrite completion marker: {confirmation_path}")
        write_json(
            confirmation_path,
            {
                "status": "complete",
                "audited_split_file_manifest_sha256": EXPECTED_CORPUS_MANIFEST_SHA256,
                "lexical_examples": lexical_count,
                "gnn_rag_examples": gnn_count,
                "both_retrieval_outputs_ready_for_answer_generation": True,
                "answer_generation_run": False,
                "completed_gnn_subgraph_or_sageqa_test_evaluation_rerun_or_modified_by_baselines": False,
            },
        )
        root_required = (
            "run_confirmation.json",
            "lexical_subgraph/artifact_manifest.json",
            "gnn_rag/artifact_manifest.json",
        )
        write_json(
            output_dir / "artifact_manifest.json",
            {
                "self_excluded_from_hashes": True,
                "files": {
                    name: {
                        "size_bytes": (output_dir / name).stat().st_size,
                        "sha256": sha256(output_dir / name),
                    }
                    for name in root_required
                },
            },
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "lexical_examples": lexical_count,
                    "gnn_rag_examples": gnn_count,
                }
            ),
            flush=True,
        )
        return

    target_dir = lexical_dir if args.stage == "lexical" else gnn_dir
    if (target_dir / "artifact_manifest.json").exists():
        validate_artifact_manifest(target_dir)
        print(
            json.dumps(
                {"status": "already_complete", "stage": args.stage, "output_dir": str(target_dir)}
            ),
            flush=True,
        )
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    state = common_run_state(args, root)

    if args.stage == "lexical":
        unexpected = [path for path in lexical_dir.iterdir()]
        if unexpected:
            raise FileExistsError(f"Incomplete lexical output requires inspection: {unexpected}")
        lexical_by_dataset: dict[str, list[dict[str, Any]]] = {}
        lexical_run_info: dict[str, Any] = {}
        for dataset, _, _, _ in DATASETS:
            test_path = args.data_root / dataset / "test_subgraph_retrieval.jsonl"
            records, info = freeze_lexical(test_path, dataset)
            lexical_by_dataset[dataset] = records
            lexical_run_info[dataset] = info
        retrieval_frozen_at = datetime.now(timezone.utc).isoformat()

        metric_rows, all_records, gold_lineage = [], [], {}
        for dataset, _, domain, gold_relative in DATASETS:
            records = lexical_by_dataset[dataset]
            ids = {row["example_id"] for row in records}
            gold_path = root / gold_relative
            golds = (
                text_gold(gold_path, dataset, ids)
                if domain == "text"
                else ontology_gold(gold_path, ids)
            )
            if set(golds) != ids:
                raise ValueError(
                    f"Incomplete post-freeze gold join for {dataset}: {len(ids - set(golds))} missing"
                )
            result, evaluated = evaluate_lexical(records, golds)
            result["dataset"] = dataset
            metric_rows.append(result)
            all_records.extend(evaluated)
            gold_lineage[dataset] = {"path": gold_relative, "sha256": sha256(gold_path)}
        metrics = {
            "schema_version": "clean_lexical_subgraph_final_test_v1",
            "status": "complete",
            "split": "test",
            "metric_aggregation": "macro mean over support-bearing examples",
            "support_aggregation": "order-preserving deduplicated union of ranked candidate support units",
            "datasets": metric_rows,
        }
        lineage = {
            **base_lineage(args, root, state, started_at, retrieval_frozen_at),
            "schema_version": "clean_lexical_subgraph_final_test_lineage_v1",
            "configuration": {
                "fixed_k": [1, 3],
                "adaptive_k": None,
                "scorer": "evaluation.eval_lexical_hotpot_subgraph.lexical_score",
                "compactness_bonus_weight": 0.02,
            },
            "post_freeze_gold_fields": {
                "HotpotQA": ["supporting_facts"],
                "2WikiMultiHopQA": ["supporting_facts"],
                "ontology": ["gold explanations"],
            },
            "two_wiki_evidences_read": False,
            "test_candidate_inputs": {
                dataset: {
                    "path": str(args.data_root / dataset / "test_subgraph_retrieval.jsonl"),
                    "sha256": sha256(args.data_root / dataset / "test_subgraph_retrieval.jsonl"),
                    **lexical_run_info[dataset],
                }
                for dataset, _, _, _ in DATASETS
            },
            "post_freeze_gold_sources": gold_lineage,
        }
        assert_protected_unchanged(args, state)
        write_json(lexical_dir / "metrics.json", metrics)
        write_jsonl(lexical_dir / "per_example_retrieval.jsonl", all_records)
        (lexical_dir / "summary.md").write_text(
            lexical_summary(metrics), encoding="utf-8", newline="\n"
        )
        write_json(lexical_dir / "lineage_metadata.json", lineage)
        artifact_manifest(
            lexical_dir,
            ("metrics.json", "per_example_retrieval.jsonl", "summary.md", "lineage_metadata.json"),
        )
        print(
            json.dumps({"status": "complete", "stage": "lexical", "examples": len(all_records)}),
            flush=True,
        )
        return

    # GNN-RAG is isolated from post-freeze gold evaluation. Each dataset gets a
    # hash-validated completion marker before the next native run begins.
    all_records: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    dataset_lineage: dict[str, Any] = {}
    for dataset, adapter_key, domain, _ in DATASETS:
        test_path = args.data_root / dataset / "test_subgraph_retrieval.jsonl"
        source_native_dir = root / "third_party/GNN-RAG/gnn/data" / f"sageqa-{adapter_key}"
        checkpoint_source = (
            root
            / "third_party/GNN-RAG/gnn/checkpoint"
            / f"sageqa-{adapter_key}"
            / f"rearev_lstm_sageqa-{adapter_key}-final.ckpt"
        )
        run_dir = gnn_dir / "native_runs" / dataset
        native_dir = run_dir / "data"
        marker_path = run_dir / "dataset_complete.json"
        if marker_path.is_file():
            marker = read_json(marker_path)
            for name, expected_hash in marker["sha256"].items():
                if sha256(run_dir / name) != expected_hash:
                    raise ValueError(
                        f"Completed GNN-RAG dataset artifact changed: {run_dir / name}"
                    )
            records = read_jsonl(run_dir / "frozen_retrieval.jsonl")
            diagnostics = read_json(run_dir / "diagnostics.json")
            lineage = read_json(run_dir / "lineage_metadata.json")
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(
                    f"Incomplete GNN-RAG dataset requires inspection before replacement: {run_dir}"
                )
            samples, unit_maps, adapter_info = prepare_clean_native_test(
                test_path, native_dir, source_native_dir, domain, args.text_max_candidates
            )
            info_path, native_command = run_native_retriever(
                root, native_dir, checkpoint_source, run_dir, adapter_key
            )
            records, diagnostics = freeze_native_output(dataset, samples, unit_maps, info_path)
            diagnostics["dataset"] = dataset
            lineage = {
                "domain": domain,
                "test_candidate_path": str(test_path),
                "test_candidate_sha256": sha256(test_path),
                "frozen_checkpoint_path": str(checkpoint_source.relative_to(root)),
                "frozen_checkpoint_sha256": sha256(checkpoint_source),
                "clean_native_test_path": str((native_dir / "test.json").relative_to(output_dir)),
                "clean_native_test_sha256": sha256(native_dir / "test.json"),
                "native_info_path": str(info_path.relative_to(output_dir)),
                "native_info_sha256": sha256(info_path),
                "adapter": {
                    **adapter_info,
                    "native_answers_array_empty": True,
                    "native_answer_field_present": False,
                },
                "native_command_argv": native_command,
                "native_command_display": subprocess.list2cmdline(native_command),
            }
            write_jsonl(run_dir / "frozen_retrieval.jsonl", records)
            write_json(run_dir / "diagnostics.json", diagnostics)
            write_json(run_dir / "lineage_metadata.json", lineage)
            dataset_files = (
                "data/test.json",
                str(info_path.relative_to(run_dir)).replace("\\", "/"),
                "frozen_retrieval.jsonl",
                "diagnostics.json",
                "lineage_metadata.json",
                "native_retriever.log",
            )
            write_json(
                marker_path,
                {
                    "status": "complete",
                    "dataset": dataset,
                    "sha256": {name: sha256(run_dir / name) for name in dataset_files},
                },
            )
        all_records.extend(records)
        diagnostics_rows.append(diagnostics)
        dataset_lineage[dataset] = lineage

    retrieval_frozen_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "schema_version": "clean_gnn_rag_native_final_test_v1",
        "status": "complete",
        "split": "test",
        "metric_scope": "native retrieval diagnostics; no forced support-set P/R/F1",
        "datasets": diagnostics_rows,
    }
    lineage_doc = {
        **base_lineage(args, root, state, started_at, retrieval_frozen_at),
        "schema_version": "clean_gnn_rag_native_final_test_lineage_v1",
        "configuration": {
            "model": "ReaRev",
            "entity_dim": 50,
            "lm": "lstm",
            "num_iter": 2,
            "num_ins": 2,
            "num_gnn": 3,
            "relation_word_emb": False,
            "eps": 0.95,
            "text_max_candidates": args.text_max_candidates,
            "checkpoint_selection": "existing final checkpoint; no training",
        },
        "native_semantics": "ranked non-seed entities at cumulative probability eps=0.95, then all upstream-style shortest paths from question entities to ranked entities",
        "support_set_metric_computed": False,
        "gold_or_answer_labels_in_native_inference_adapter": False,
        "datasets": dataset_lineage,
    }
    assert_protected_unchanged(args, state)
    write_json(gnn_dir / "native_retrieval_diagnostics.json", metrics)
    write_jsonl(gnn_dir / "per_example_retrieval.jsonl", all_records)
    (gnn_dir / "summary.md").write_text(gnn_summary(metrics), encoding="utf-8", newline="\n")
    write_json(gnn_dir / "lineage_metadata.json", lineage_doc)
    artifact_manifest(
        gnn_dir,
        (
            "native_retrieval_diagnostics.json",
            "per_example_retrieval.jsonl",
            "summary.md",
            "lineage_metadata.json",
        ),
    )
    print(
        json.dumps({"status": "complete", "stage": "gnn", "examples": len(all_records)}), flush=True
    )


if __name__ == "__main__":
    main()
