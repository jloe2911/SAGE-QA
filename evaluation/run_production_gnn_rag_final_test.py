"""One-time clean GNN-RAG/ReaRev TEST retrieval with a hard gold boundary.

Run ``predict`` first. It reads only the clean production TEST candidate corpus,
uses the frozen epoch-3 checkpoints and TRAIN/DEV dictionaries, and writes a
global prediction freeze. Run ``evaluate`` only after ``predict`` completes; it
validates that freeze before opening support/explanation gold.
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
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.prepare_production_gnn_rag_clean import (
    DATASETS as CLEAN_DATASETS,
    evidence_node,
    grouped_rows,
    inference_sample,
    unit_node,
)


DATASETS = OrderedDict(
    (
        (
            "HotpotQA",
            {
                "domain": "text",
                "gold": "data/raw/hotpot_qa/distractor/validation-00000-of-00001.parquet",
            },
        ),
        (
            "2WikiMultiHopQA",
            {
                "domain": "text",
                "gold": "data/raw/2WikiMultihopQA/data/validation-00000-of-00001.parquet",
            },
        ),
        ("FamilyOWL_1hop", {"domain": "ontology", "gold": "data/raw/family/FamilyOWL_1hop.json"}),
        ("FamilyOWL_2hop", {"domain": "ontology", "gold": "data/raw/family/FamilyOWL_2hop.json"}),
        (
            "pizza_100_1hop",
            {"domain": "ontology", "gold": "data/raw/pizza_100/pizza_100_1hop.json"},
        ),
        (
            "pizza_100_2hop",
            {"domain": "ontology", "gold": "data/raw/pizza_100/pizza_100_2hop.json"},
        ),
        (
            "pizza_250_1hop",
            {"domain": "ontology", "gold": "data/raw/pizza_250/pizza_250_1hop.json"},
        ),
        (
            "pizza_250_2hop",
            {"domain": "ontology", "gold": "data/raw/pizza_250/pizza_250_2hop.json"},
        ),
        (
            "OWL2Bench_1hop",
            {"domain": "ontology", "gold": "data/raw/owl2bench/OWL2Bench_1hop.json"},
        ),
        (
            "OWL2Bench_2hop",
            {"domain": "ontology", "gold": "data/raw/owl2bench/OWL2Bench_2hop.json"},
        ),
    )
)
PROHIBITED_TEST_FIELDS = {
    "answer",
    "answers",
    "supporting_facts",
    "raw_supporting_facts",
    "evidences",
    "gold_support_units",
    "gold_units",
    "gold_explanations",
    "Minimum Explanation",
    "minimum_explanation",
    "Explanations",
    "label",
    "rank_target",
    "best_set_f1_to_gold",
    "best_set_precision_to_gold",
    "best_set_recall_to_gold",
    "exact_match_any_gold",
    "contains_any_gold_explanation",
}
CODE_FILES = (
    "evaluation/run_production_gnn_rag_final_test.py",
    "data_processing/prepare_production_gnn_rag_clean.py",
    "data_processing/prepare_text_gnn_rag.py",
    "data_processing/prepare_familyowl_gnn_rag.py",
    "third_party/GNN-RAG/gnn/main.py",
    "third_party/GNN-RAG/gnn/parsing.py",
    "third_party/GNN-RAG/gnn/dataset_load.py",
    "third_party/GNN-RAG/gnn/evaluate.py",
    "third_party/GNN-RAG/gnn/train_model.py",
    "third_party/GNN-RAG/gnn/models/ReaRev/rearev.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checked_groups(path: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    for example_id, rows in grouped_rows(path):
        for row in rows:
            forbidden = PROHIBITED_TEST_FIELDS.intersection(row)
            if forbidden:
                raise ValueError(
                    f"{path}: prohibited TEST fields for {example_id}: {sorted(forbidden)}"
                )
            if row.get("gold_available_during_candidate_generation") is not False:
                raise ValueError(f"{path}: missing gold-free generation assertion for {example_id}")
            if row.get("gold_used_during_labeling") not in {None, False}:
                raise ValueError(f"{path}: TEST row was gold-labelled for {example_id}")
        yield example_id, rows


def code_state(root: Path) -> dict[str, Any]:
    files = [{"path": name, "sha256": sha256(root / name)} for name in CODE_FILES]
    state = {
        "base_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "files": files,
    }
    state["code_state_sha256"] = canonical_hash(state)
    return state


def checkpoint_lock(root: Path, checkpoint_root: Path) -> dict[str, Any]:
    report_path = checkpoint_root / "final_report.json"
    report = read_json(report_path)
    if (
        report.get("checkpoint_selection_rule")
        != "final epoch 3, preserving the historical production runner's explicit final-checkpoint semantics"
    ):
        raise ValueError("Unexpected clean checkpoint selection rule")
    rows = {row["dataset"]: row for row in report["datasets"]}
    if set(rows) != set(DATASETS):
        raise ValueError(f"Clean checkpoint report dataset mismatch: {set(rows) ^ set(DATASETS)}")
    result: dict[str, Any] = {}
    for dataset in DATASETS:
        row = rows[dataset]
        dataset_dir = checkpoint_root / dataset
        frozen = read_json(dataset_dir / "frozen_checkpoint_manifest.json")
        selected = dataset_dir / row["selected_checkpoint"]
        adapter = dataset_dir / "adapter"
        if row["selected_checkpoint"] != frozen["selected_checkpoint"]:
            raise ValueError(f"Selected checkpoint disagreement for {dataset}")
        if not row["selected_checkpoint"].endswith("-final.ckpt"):
            raise ValueError(
                f"Selected checkpoint is not the frozen final epoch-3 checkpoint: {dataset}"
            )
        if (
            sha256(selected) != row["checkpoint_sha256"]
            or sha256(selected) != frozen["selected_checkpoint_sha256"]
        ):
            raise ValueError(f"Checkpoint hash mismatch for {dataset}")
        relation_hash = sha256(adapter / "relations.txt")
        if (
            relation_hash != row["relation_vocabulary_sha256"]
            or relation_hash != frozen["relation_vocabulary_sha256"]
        ):
            raise ValueError(f"Relation vocabulary hash mismatch for {dataset}")
        result[dataset] = {
            "checkpoint_path": selected.relative_to(root).as_posix(),
            "checkpoint_sha256": sha256(selected),
            "selection": "final epoch 3 from final_report.json",
            "relation_vocabulary_path": (adapter / "relations.txt").relative_to(root).as_posix(),
            "relation_vocabulary_sha256": relation_hash,
            "word_vocabulary_sha256": sha256(adapter / "vocab.txt"),
            "entity_dictionary_sha256": sha256(adapter / "entities.txt"),
            "clean_train_adapter_sha256": sha256(adapter / "train.json"),
            "clean_dev_adapter_sha256": sha256(adapter / "dev.json"),
            "clean_train_manifest_sha256": sha256(dataset_dir / "train_manifest.json"),
            "clean_dev_manifest_sha256": sha256(dataset_dir / "dev_manifest.json"),
        }
    return {
        "final_report_path": report_path.relative_to(root).as_posix(),
        "final_report_sha256": sha256(report_path),
        "datasets": result,
    }


def test_cohort_lock(root: Path, data_root: Path) -> dict[str, Any]:
    rows = []
    for dataset in DATASETS:
        path = data_root / dataset / "test_subgraph_retrieval.jsonl"
        count = sum(1 for _ in checked_groups(path))
        rows.append(
            {
                "dataset": dataset,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "examples": count,
            }
        )
    return {"datasets": rows, "test_cohort_sha256": canonical_hash(rows)}


def prepare_native_test(
    source: Path,
    destination: Path,
    adapter: Path,
    domain: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=False)
    copied_hashes = {}
    for name in ("entities.txt", "relations.txt", "vocab.txt"):
        shutil.copy2(adapter / name, destination / name)
        copied_hashes[name] = sha256(destination / name)
        if copied_hashes[name] != sha256(adapter / name):
            raise RuntimeError(f"Dictionary copy changed bytes: {name}")
    samples: list[dict[str, Any]] = []
    unit_maps: list[dict[str, str]] = []
    frozen_entities = set((adapter / "entities.txt").read_text(encoding="utf-8").splitlines())
    frozen_relations = set((adapter / "relations.txt").read_text(encoding="utf-8").splitlines())
    unseen: set[str] = set()
    missing_relations: set[str] = set()
    with (destination / "test.json").open("x", encoding="utf-8", newline="\n") as handle:
        for index, (example_id, rows) in enumerate(checked_groups(source), start=1):
            sample, candidate_units = inference_sample(rows, domain, 5)
            if sample["id"] != example_id or sample.get("answers") != []:
                raise ValueError(f"Invalid clean inference sample for {example_id}")
            if any(field in sample for field in PROHIBITED_TEST_FIELDS - {"answers"}):
                raise ValueError(f"Sensitive field reached native inference adapter: {example_id}")
            unit_node_fn = evidence_node if domain == "text" else unit_node
            unit_map = {unit_node_fn(unit): unit for unit in candidate_units}
            unseen.update(set(sample["subgraph"]["entities"]) - frozen_entities)
            missing_relations.update(
                edge[1] for edge in sample["graph"] if edge[1] not in frozen_relations
            )
            samples.append(sample)
            unit_maps.append(unit_map)
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % 100 == 0:
                print(f"{source.parent.name}: prepared {index} clean TEST examples", flush=True)
    if missing_relations:
        raise ValueError(
            f"TEST graph uses relations absent from frozen mapping: {sorted(missing_relations)}"
        )
    return (
        samples,
        unit_maps,
        {
            "examples": len(samples),
            "dictionary_copy_sha256": copied_hashes,
            "frozen_entity_dictionary_unchanged": True,
            "unseen_nodes_using_transient_local_indices": len(unseen),
            "relation_mapping_unchanged": True,
            "word_vocabulary_unchanged": True,
            "test_entity_vocabulary_persisted": False,
            "answers_array_empty": True,
        },
    )


def run_native(
    root: Path,
    run_dir: Path,
    native_data: Path,
    checkpoint_source: Path,
    dataset: str,
) -> tuple[Path, list[str]]:
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_copy = checkpoint_dir / checkpoint_source.name
    shutil.copy2(checkpoint_source, checkpoint_copy)
    if sha256(checkpoint_copy) != sha256(checkpoint_source):
        raise RuntimeError(f"Checkpoint copy hash mismatch for {dataset}")
    experiment = f"production_generator_d_v1_final_test_{dataset.lower()}"
    gnn_root = root / "third_party" / "GNN-RAG" / "gnn"
    command = [
        sys.executable,
        str(gnn_root / "main.py"),
        "ReaRev",
        "--entity_dim",
        "50",
        "--data_folder",
        str(native_data.resolve()) + "\\",
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
        checkpoint_copy.name,
        "--is_eval",
        "--name",
        f"sageqa-clean-{dataset.lower()}",
        "--frozen_entity_dictionary",
        "true",
        "--test_only_inference",
        "true",
    ]
    log_path = run_dir / "native_retriever.stdout_stderr.log"
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command, cwd=gnn_root, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    if result.returncode:
        raise RuntimeError(f"ReaRev TEST inference failed for {dataset}; see {log_path}")
    info_path = checkpoint_dir / f"{experiment}_test.info"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing native ReaRev output for {dataset}: {info_path}")
    return info_path, command


def path_string(path: Sequence[Sequence[str]]) -> str:
    if not path:
        return ""
    text = f"{path[0][0]} -> {path[0][1]} -> {path[0][2]}"
    for _, relation, tail in path[1:]:
        text += f" -> {relation} -> {tail}"
    return text


def native_paths(
    sample: Mapping[str, Any], ranked_entities: Sequence[Mapping[str, Any]]
) -> tuple[list[list[list[str]]], list[str]]:
    graph = nx.Graph()
    for head, relation, tail in sample["graph"]:
        graph.add_edge(str(head), str(tail), relation=str(relation))
    result: list[list[list[str]]] = []
    for head in map(str, sample["q_entity"]):
        if head not in graph:
            continue
        for candidate in ranked_entities:
            tail = str(candidate["entity"])
            if tail not in graph:
                continue
            try:
                for nodes in nx.all_shortest_paths(graph, head, tail):
                    result.append(
                        [
                            [nodes[i], graph[nodes[i]][nodes[i + 1]]["relation"], nodes[i + 1]]
                            for i in range(len(nodes) - 1)
                        ]
                    )
            except nx.NetworkXException:
                continue
    return result, [path_string(path) for path in result]


def freeze_dataset(
    dataset: str,
    samples: Sequence[Mapping[str, Any]],
    unit_maps: Sequence[Mapping[str, str]],
    info_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info_rows = read_jsonl(info_path)
    if len(info_rows) != len(samples):
        raise ValueError(
            f"Native output count mismatch for {dataset}: {len(info_rows)} != {len(samples)}"
        )
    records = []
    for sample, unit_map, info in zip(samples, unit_maps, info_rows):
        if str(info.get("example_id")) != str(sample["id"]):
            raise ValueError(f"Native output order/ID mismatch for {dataset}: {sample['id']}")
        cumulative = 0.0
        ranked = []
        for rank, raw in enumerate(info.get("cand", []) or [], start=1):
            probability = float(raw[1])
            cumulative += probability
            ranked.append(
                {
                    "rank": rank,
                    "entity": str(raw[0]),
                    "probability": probability,
                    "cumulative_probability": cumulative,
                }
            )
        paths, contexts = native_paths(sample, ranked)
        retrieved_units: list[str] = []
        seen_units: set[str] = set()
        for path in paths:
            for head, _, tail in path:
                for node in (head, tail):
                    if node in unit_map and unit_map[node] not in seen_units:
                        seen_units.add(unit_map[node])
                        retrieved_units.append(unit_map[node])
        trace = []
        step = 0
        while str(step) in info:
            trace.append({"iteration": step, **info[str(step)]})
            step += 1
        records.append(
            {
                "dataset": dataset,
                "split": "test",
                "example_id": sample["id"],
                "native_ranked_entities": ranked,
                "native_relation_action_trace": trace,
                "retrieved_paths": paths,
                "retrieved_evidence_units": retrieved_units,
                "reader_context_representation": "newline-joined ReaRev shortest-path strings",
                "reader_context_paths": contexts,
                "reader_context_text": "\n".join(contexts),
                "native_graph_nodes": len(sample["subgraph"]["entities"]),
                "native_graph_edges": len(sample["graph"]),
            }
        )
    lengths = [len(row["native_ranked_entities"]) for row in records]
    path_lengths = [len(row["retrieved_paths"]) for row in records]
    unit_lengths = [len(row["retrieved_evidence_units"]) for row in records]
    diagnostics = {
        "examples": len(records),
        "inference_failures": 0,
        "examples_with_ranked_entities": sum(bool(value) for value in lengths),
        "examples_with_nonempty_reader_context": sum(bool(value) for value in path_lengths),
        "examples_with_mapped_evidence_units": sum(bool(value) for value in unit_lengths),
        "mean_ranked_entities": statistics.fmean(lengths) if lengths else 0.0,
        "median_ranked_entities": statistics.median(lengths) if lengths else 0.0,
        "mean_retrieved_paths": statistics.fmean(path_lengths) if path_lengths else 0.0,
        "mean_mapped_evidence_units": statistics.fmean(unit_lengths) if unit_lengths else 0.0,
    }
    return records, diagnostics


def prediction_phase(
    args: argparse.Namespace, root: Path, data_root: Path, checkpoint_root: Path, output_dir: Path
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite or mix final TEST output: {output_dir}")
    output_dir.mkdir(parents=True)
    started = utc_now()
    locks = checkpoint_lock(root, checkpoint_root)
    cohorts = test_cohort_lock(root, data_root)
    code = code_state(root)
    all_records: list[dict[str, Any]] = []
    dataset_rows = []
    for dataset, cfg in DATASETS.items():
        print(f"Starting frozen ReaRev TEST inference: {dataset}", flush=True)
        run_dir = output_dir / "native_runs" / dataset
        run_dir.mkdir(parents=True)
        samples, unit_maps, adapter_info = prepare_native_test(
            data_root / dataset / "test_subgraph_retrieval.jsonl",
            run_dir / "data",
            checkpoint_root / dataset / "adapter",
            cfg["domain"],
        )
        checkpoint = root / locks["datasets"][dataset]["checkpoint_path"]
        info_path, command = run_native(root, run_dir, run_dir / "data", checkpoint, dataset)
        records, diagnostics = freeze_dataset(dataset, samples, unit_maps, info_path)
        write_jsonl(run_dir / "frozen_retrieval.jsonl", records)
        dataset_lineage = {
            "dataset": dataset,
            "adapter": adapter_info,
            "checkpoint": locks["datasets"][dataset],
            "test_cohort": next(row for row in cohorts["datasets"] if row["dataset"] == dataset),
            "native_command_argv": command,
            "native_command_display": subprocess.list2cmdline(command),
            "native_info_sha256": sha256(info_path),
            "frozen_retrieval_sha256": sha256(run_dir / "frozen_retrieval.jsonl"),
            "diagnostics": diagnostics,
        }
        write_json(run_dir / "lineage_metadata.json", dataset_lineage)
        all_records.extend(records)
        dataset_rows.append(
            {
                "dataset": dataset,
                **diagnostics,
                "frozen_retrieval_sha256": dataset_lineage["frozen_retrieval_sha256"],
            }
        )
        print(
            f"Completed frozen ReaRev TEST inference: {dataset} ({len(records)} examples)",
            flush=True,
        )
    frozen_path = output_dir / "predictions_frozen.jsonl"
    write_jsonl(frozen_path, all_records)
    frozen_at = utc_now()
    freeze_manifest = {
        "schema_version": "production_generator_d_v1_clean_gnn_rag_prediction_freeze_v1",
        "status": "predictions_frozen_gold_unopened",
        "started_at_utc": started,
        "predictions_frozen_at_utc": frozen_at,
        "prediction_file": "predictions_frozen.jsonl",
        "prediction_examples": len(all_records),
        "prediction_freeze_sha256": sha256(frozen_path),
        "prediction_payload_canonical_sha256": canonical_hash(all_records),
        "datasets": dataset_rows,
        "test_cohort_sha256": cohorts["test_cohort_sha256"],
        "code_state_sha256": code["code_state_sha256"],
        "gold_sources_opened_before_global_freeze": False,
        "two_wiki_evidences_used": False,
        "answer_generation_run": False,
        "training_run": False,
        "vocabulary_changed": False,
        "relation_mapping_changed": False,
        "test_entity_vocabulary_constructed_or_persisted": False,
    }
    write_json(output_dir / "prediction_freeze_manifest.json", freeze_manifest)
    write_json(
        output_dir / "phase1_lineage.json",
        {
            "checkpoint_lock": locks,
            "test_cohort_lock": cohorts,
            "code_state": code,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "protocol": {
                "permitted_test_fields_only": True,
                "test_answers_or_gold_in_native_adapter": False,
                "two_wiki_evidences_read": False,
                "predictions_frozen_before_gold_access": True,
                "checkpoint_selection_changed": False,
                "relation_mapping_changed": False,
                "word_vocabulary_changed": False,
                "entity_dictionary_changed": False,
                "graph_topology_changed_after_adapter_construction": False,
                "answer_generation_run": False,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "predictions_frozen_gold_unopened",
                "examples": len(all_records),
                "prediction_freeze_sha256": freeze_manifest["prediction_freeze_sha256"],
            }
        ),
        flush=True,
    )


def text_gold(source: Path, dataset: str, example_ids: set[str]) -> dict[str, list[list[str]]]:
    import pandas as pd
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
    from data.build_subgraph_training_data import get_gold_explanations

    groups = read_json(source)
    result = {}
    for example_id in example_ids:
        parts = example_id.split("__", 3)
        group_index, qa_index = int(parts[1][1:]), int(parts[2][1:])
        result[example_id] = get_gold_explanations(groups[group_index]["QAs"][qa_index])
    return result


def best_overlap(predicted: Sequence[str], golds: Sequence[Sequence[str]]) -> dict[str, Any]:
    predicted_set = set(map(str, predicted))
    best = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "exact_match": False,
        "contains_complete_gold": False,
        "gold_size": 0,
    }
    for gold in golds:
        gold_set = set(map(str, gold))
        if not gold_set:
            continue
        overlap = len(predicted_set & gold_set)
        precision = overlap / len(predicted_set) if predicted_set else 0.0
        recall = overlap / len(gold_set)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        candidate = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": predicted_set == gold_set,
            "contains_complete_gold": gold_set <= predicted_set,
            "gold_size": len(gold_set),
        }
        if (candidate["f1"], candidate["recall"], candidate["precision"]) > (
            best["f1"],
            best["recall"],
            best["precision"],
        ):
            best = candidate
    return best


def aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    eligible = [row[key] for row in rows if row["evaluation_eligible"]]
    return {
        "evaluation_examples": len(eligible),
        "precision": statistics.fmean(row["precision"] for row in eligible) if eligible else 0.0,
        "recall": statistics.fmean(row["recall"] for row in eligible) if eligible else 0.0,
        "f1": statistics.fmean(row["f1"] for row in eligible) if eligible else 0.0,
        "exact_match_rate": statistics.fmean(float(row["exact_match"]) for row in eligible)
        if eligible
        else 0.0,
        "complete_gold_recall_rate": statistics.fmean(
            float(row["contains_complete_gold"]) for row in eligible
        )
        if eligible
        else 0.0,
        "any_overlap_rate": statistics.fmean(float(row["recall"] > 0.0) for row in eligible)
        if eligible
        else 0.0,
    }


def summary_markdown(metrics: Mapping[str, Any], freeze_hash: str) -> str:
    lines = [
        "# Frozen clean GNN-RAG/ReaRev TEST retrieval",
        "",
        f"Prediction freeze SHA-256: `{freeze_hash}`",
        "",
        "All ten prediction cohorts were frozen before any TEST support/explanation gold was opened. No answer generation was run.",
        "",
        "| Dataset | Inference | Failures | Ranked entities | Reader context | Native entity F1 | Path-support F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["datasets"]:
        native = row["native_ranked_entity_support_overlap"]
        common = row["retrieved_path_support_overlap"]
        lines.append(
            f"| {row['dataset']} | {row['inference_examples']} | {row['inference_failures']} | "
            f"{row['examples_with_ranked_entities']} | {row['examples_with_nonempty_reader_context']} | "
            f"{native['f1']:.6f} | {common['f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Saved reader context",
            "",
            "Each example preserves ordered ReaRev entities with native probabilities/ranks, the relation-action trace, all derived shortest paths, mapped source evidence units, `reader_context_paths`, and the exact newline-joined `reader_context_text` that can later be passed to a reader.",
            "",
            "The upstream evaluator's answer-label F1/Hits/EM are not reported because the gold-blind native TEST adapter intentionally contains an empty `answers` array. Post-freeze entity-support and path-support overlap are reported instead.",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_manifest(output_dir: Path) -> None:
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files[path.relative_to(output_dir).as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    write_json(
        output_dir / "artifact_manifest.json", {"self_excluded_from_hashes": True, "files": files}
    )


def evaluation_phase(args: argparse.Namespace, root: Path, output_dir: Path) -> None:
    freeze_path = output_dir / "prediction_freeze_manifest.json"
    predictions_path = output_dir / "predictions_frozen.jsonl"
    if not freeze_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(
            "Global prediction freeze is incomplete; refusing to open TEST gold"
        )
    freeze = read_json(freeze_path)
    if freeze.get("status") != "predictions_frozen_gold_unopened" or sha256(
        predictions_path
    ) != freeze.get("prediction_freeze_sha256"):
        raise ValueError("Prediction freeze validation failed; refusing to open TEST gold")
    records = read_jsonl(predictions_path)
    if len(records) != freeze.get("prediction_examples") or canonical_hash(records) != freeze.get(
        "prediction_payload_canonical_sha256"
    ):
        raise ValueError("Prediction payload validation failed; refusing to open TEST gold")
    if set(row["dataset"] for row in records) != set(DATASETS):
        raise ValueError("Prediction freeze does not contain all ten datasets")
    phase1 = read_json(output_dir / "phase1_lineage.json")
    if code_state(root)["code_state_sha256"] != phase1["code_state"]["code_state_sha256"]:
        raise ValueError(
            "Inference code state changed after prediction freeze; refusing evaluation"
        )

    evaluated = []
    metric_rows = []
    gold_lineage = {}
    for dataset, cfg in DATASETS.items():
        dataset_records = [row for row in records if row["dataset"] == dataset]
        ids = {row["example_id"] for row in dataset_records}
        gold_path = root / cfg["gold"]
        golds = (
            text_gold(gold_path, dataset, ids)
            if cfg["domain"] == "text"
            else ontology_gold(gold_path, ids)
        )
        if set(golds) != ids:
            raise ValueError(
                f"Incomplete post-freeze gold join for {dataset}: {len(ids - set(golds))} missing"
            )
        for row in dataset_records:
            references = [gold for gold in golds[row["example_id"]] if gold]
            row = dict(row)
            row["evaluation_eligible"] = bool(references)
            row["evaluation_exclusion_reason"] = (
                None if references else "no_gold_support_or_explanation"
            )
            entity_units = []
            node_to_unit = {
                (evidence_node(unit) if cfg["domain"] == "text" else unit_node(unit)): unit
                for gold in references
                for unit in gold
            }
            for candidate in row["native_ranked_entities"]:
                unit = node_to_unit.get(candidate["entity"])
                if unit is not None and unit not in entity_units:
                    entity_units.append(unit)
            row["native_ranked_entity_support_overlap"] = (
                best_overlap(entity_units, references) if references else None
            )
            row["retrieved_path_support_overlap"] = (
                best_overlap(row["retrieved_evidence_units"], references) if references else None
            )
            evaluated.append(row)
        diagnostics = next(item for item in freeze["datasets"] if item["dataset"] == dataset)
        dataset_eval = [row for row in evaluated if row["dataset"] == dataset]
        metric_rows.append(
            {
                "dataset": dataset,
                "inference_examples": diagnostics["examples"],
                "inference_failures": diagnostics["inference_failures"],
                "examples_with_ranked_entities": diagnostics["examples_with_ranked_entities"],
                "examples_with_nonempty_reader_context": diagnostics[
                    "examples_with_nonempty_reader_context"
                ],
                "mean_ranked_entities": diagnostics["mean_ranked_entities"],
                "median_ranked_entities": diagnostics["median_ranked_entities"],
                "mean_retrieved_paths": diagnostics["mean_retrieved_paths"],
                "mean_mapped_evidence_units": diagnostics["mean_mapped_evidence_units"],
                "excluded_no_gold_support_or_explanation": sum(
                    not row["evaluation_eligible"] for row in dataset_eval
                ),
                "native_ranked_entity_support_overlap": aggregate(
                    dataset_eval, "native_ranked_entity_support_overlap"
                ),
                "retrieved_path_support_overlap": aggregate(
                    dataset_eval, "retrieved_path_support_overlap"
                ),
            }
        )
        gold_lineage[dataset] = {
            "path": cfg["gold"],
            "sha256": sha256(gold_path),
            "fields": ["context", "supporting_facts"]
            if cfg["domain"] == "text"
            else ["gold explanations"],
        }

    metrics = {
        "schema_version": "production_generator_d_v1_clean_gnn_rag_final_test_metrics_v1",
        "status": "complete",
        "split": "test",
        "prediction_freeze_sha256": freeze["prediction_freeze_sha256"],
        "native_retrieval_semantics": "non-seed entities ranked by ReaRev probability and truncated after cumulative probability exceeds eps=0.95; shortest paths from question seed nodes form reader context",
        "common_metric_semantics": "macro best-match support-set P/R/F1 over support-bearing examples, computed only after global prediction freeze",
        "datasets": metric_rows,
    }
    write_jsonl(output_dir / "per_example_retrieval.jsonl", evaluated)
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "summary.md").write_text(
        summary_markdown(metrics, freeze["prediction_freeze_sha256"]),
        encoding="utf-8",
        newline="\n",
    )
    lineage = {
        "schema_version": "production_generator_d_v1_clean_gnn_rag_final_test_lineage_v1",
        "completed_at_utc": utc_now(),
        "prediction_freeze_manifest": freeze,
        "checkpoint_lock": phase1["checkpoint_lock"],
        "test_cohort_lock": phase1["test_cohort_lock"],
        "code_state": phase1["code_state"],
        "post_freeze_gold_sources": gold_lineage,
        "protocol_confirmations": {
            "all_predictions_all_datasets_frozen_before_test_gold_opened": True,
            "prediction_freeze_revalidated_before_gold_join": True,
            "two_wiki_evidences_used": False,
            "training_run": False,
            "checkpoint_decision_changed": False,
            "word_vocabulary_changed": False,
            "entity_dictionary_changed": False,
            "relation_mapping_changed": False,
            "graph_topology_changed": False,
            "answer_generation_run": False,
            "stopped_after_test_retrieval_evaluation": True,
        },
        "saved_context_contract": {
            "paths": "retrieved_paths: ordered edge triples [head, relation, tail]",
            "reader_context_paths": "ordered GNN-RAG path strings",
            "reader_context_text": "exact newline join of reader_context_paths",
            "ranked_entities": "native_ranked_entities: rank, probability, cumulative_probability",
        },
    }
    write_json(output_dir / "lineage_metadata.json", lineage)
    artifact_manifest(output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "examples": len(evaluated),
                "prediction_freeze_sha256": freeze["prediction_freeze_sha256"],
            }
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "evaluate"))
    parser.add_argument("--data-root", type=Path, default=Path("data/production_generator_d_v1"))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/production_generator_d_v1_gnn_rag_clean"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final_results/production_generator_d_v1_test_baselines/gnn_rag"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = (
        (root / args.data_root).resolve()
        if not args.data_root.is_absolute()
        else args.data_root.resolve()
    )
    checkpoint_root = (
        (root / args.checkpoint_root).resolve()
        if not args.checkpoint_root.is_absolute()
        else args.checkpoint_root.resolve()
    )
    output_dir = (
        (root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    if tuple(DATASETS) != tuple(CLEAN_DATASETS):
        raise ValueError("Dataset order disagrees with the frozen clean adapter protocol")
    if args.stage == "predict":
        prediction_phase(args, root, data_root, checkpoint_root, output_dir)
    else:
        evaluation_phase(args, root, output_dir)


if __name__ == "__main__":
    main()
