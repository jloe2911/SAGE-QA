"""Build leakage-invariant TRAIN/DEV adapters for the clean GNN-RAG baseline.

Only inference-time fields construct graphs. Gold supervision is attached after
the graph and ordered candidate membership have been frozen. TEST is never read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.prepare_familyowl_gnn_rag import (
    clean as owl_clean,
    extract_question_entities,
    relation_for_unit,
    unit_node,
)
from data_processing.prepare_text_gnn_rag import (
    TOKEN_RE as TEXT_TOKEN_RE,
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


DATASETS = OrderedDict(
    (
        ("HotpotQA", {"directory": "HotpotQA", "domain": "text"}),
        ("2WikiMultiHopQA", {"directory": "2WikiMultiHopQA", "domain": "text"}),
        ("FamilyOWL_1hop", {"directory": "FamilyOWL_1hop", "domain": "ontology"}),
        ("FamilyOWL_2hop", {"directory": "FamilyOWL_2hop", "domain": "ontology"}),
        ("pizza_100_1hop", {"directory": "pizza_100_1hop", "domain": "ontology"}),
        ("pizza_100_2hop", {"directory": "pizza_100_2hop", "domain": "ontology"}),
        ("pizza_250_1hop", {"directory": "pizza_250_1hop", "domain": "ontology"}),
        ("pizza_250_2hop", {"directory": "pizza_250_2hop", "domain": "ontology"}),
        ("OWL2Bench_1hop", {"directory": "OWL2Bench_1hop", "domain": "ontology"}),
        ("OWL2Bench_2hop", {"directory": "OWL2Bench_2hop", "domain": "ontology"}),
    )
)

TEXT_SCHEMA_RELATIONS = {
    "contains_sentence",
    "kg_context",
    "mentioned_in_sentence",
    "mentions_kg_entity",
    "mentions_page",
    "retrieved_evidence",
}
ONTOLOGY_SCHEMA_RELATIONS = {"related_to"}
SENSITIVE_FIELDS = (
    "answer",
    "supporting_facts",
    "raw_supporting_facts",
    "gold_support_units",
    "evidences",
    "gold_explanations",
    "minimum_explanation",
    "Minimum Explanation",
    "Explanations",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{value}\n" for value in sorted(set(values))), encoding="utf-8", newline="\n"
    )


def grouped_rows(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
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


def deduplicated_units(rows: Sequence[Mapping[str, Any]], row_cap: int | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    selected = rows if row_cap is None else rows[:row_cap]
    for row in selected:
        for raw in row.get("subgraph_units", []) or []:
            unit = str(raw)
            if unit and unit not in seen:
                seen.add(unit)
                result.append(unit)
    return result


def text_inference_sample(
    rows: Sequence[dict[str, Any]], max_candidates: int
) -> tuple[dict[str, Any], list[str]]:
    seed = rows[0]
    example_id = str(seed["example_id"])
    q_node = question_node(example_id)
    candidate_units = deduplicated_units(rows, max_candidates)
    tuples: list[list[str]] = []
    entities: set[str] = {q_node}

    for unit in candidate_units:
        title, _, _ = sentence_parts(unit)
        title_node = f"TITLE::{title}"
        ev_node = evidence_node(unit)
        entities.update((title_node, ev_node))
        tuples.extend(
            (
                [q_node, "mentions_page", title_node],
                [title_node, "contains_sentence", ev_node],
                [q_node, "retrieved_evidence", ev_node],
            )
        )

    for unit in collect_kg_units(list(rows)):
        parts = kg_parts(unit)
        if parts is None:
            continue
        subject, predicate, obj = parts
        subj_node, obj_node = kg_entity_node(subject), kg_entity_node(obj)
        entities.update((subj_node, obj_node))
        tuples.extend(
            (
                [subj_node, kg_relation(predicate), obj_node],
                [q_node, "kg_context", subj_node],
                [q_node, "kg_context", obj_node],
            )
        )
        for sentence_unit in candidate_units:
            title, _, sentence = sentence_parts(sentence_unit)
            ev_node = evidence_node(sentence_unit)
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
        "answers": [],
        "subgraph": {"entities": sorted(entities), "tuples": tuples},
    }
    return sample, candidate_units


def ontology_inference_sample(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    seed = rows[0]
    example_id = str(seed["example_id"])
    candidate_units = deduplicated_units(rows, None)
    q_entities = extract_question_entities({"sparql_query": seed.get("sparql_query", "")})
    if not q_entities:
        q_entities = [f"QUESTION::{hashlib.sha256(example_id.encode('utf-8')).hexdigest()[:24]}"]
    entities: set[str] = set(q_entities)
    tuples: list[list[str]] = []
    for unit in candidate_units:
        node = unit_node(unit)
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
    return sample, candidate_units


def inference_sample(
    rows: Sequence[dict[str, Any]], domain: str, text_max_candidates: int
) -> tuple[dict[str, Any], list[str]]:
    if domain == "text":
        return text_inference_sample(rows, text_max_candidates)
    return ontology_inference_sample(rows)


def supervision_units(rows: Sequence[Mapping[str, Any]], domain: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    keys = ("gold_support_units",) if domain == "text" else ("gold_units", "gold_explanations")
    for row in rows:
        for key in keys:
            raw_values = row.get(key, []) or []
            if key == "gold_explanations":
                values = [unit for explanation in raw_values for unit in (explanation or [])]
            else:
                values = raw_values
            for raw in values:
                unit = str(raw)
                if unit and unit not in seen:
                    seen.add(unit)
                    result.append(unit)
    return result


def attach_supervision(
    sample: dict[str, Any],
    candidate_units: Sequence[str],
    rows: Sequence[dict[str, Any]],
    domain: str,
) -> int:
    candidate_set = set(candidate_units)
    targets = [unit for unit in supervision_units(rows, domain) if unit in candidate_set]
    node = evidence_node if domain == "text" else unit_node
    sample["answers"] = [{"kb_id": node(unit), "text": node(unit)} for unit in targets]
    return len(targets)


def structure_signature(
    sample: Mapping[str, Any], candidate_units: Sequence[str]
) -> dict[str, Any]:
    return {
        "nodes": sample["subgraph"]["entities"],
        "edges": sample["subgraph"]["tuples"],
        "relations": [edge[1] for edge in sample["subgraph"]["tuples"]],
        "candidate_membership": list(candidate_units),
        "candidate_ordering": list(candidate_units),
        "graph_topology": [[edge[0], edge[2]] for edge in sample["subgraph"]["tuples"]],
    }


def invariance_check(
    rows: Sequence[dict[str, Any]], domain: str, text_max_candidates: int
) -> dict[str, Any]:
    baseline, candidates = inference_sample(rows, domain, text_max_candidates)
    baseline_signature = structure_signature(baseline, candidates)
    mutated = copy.deepcopy(list(rows))
    for row in mutated:
        for field in SENSITIVE_FIELDS:
            row.pop(field, None)
        row.update(
            {
                "answer": "MUTATED ANSWER",
                "supporting_facts": [["MUTATED", 999]],
                "gold_support_units": ["SENT::MUTATED::999::MUTATED"],
                "evidences": [["MUTATED", "MUTATED", "MUTATED"]],
                "gold_explanations": [["MUTATED gold explanation"]],
                "Explanations": [["MUTATED ontology explanation"]],
            }
        )
    changed, changed_candidates = inference_sample(mutated, domain, text_max_candidates)
    changed_signature = structure_signature(changed, changed_candidates)
    return {
        "example_id": str(rows[0]["example_id"]),
        "pass": baseline_signature == changed_signature,
        "baseline_hash": canonical_hash(baseline_signature),
        "mutated_hash": canonical_hash(changed_signature),
        "fields_mutated_or_removed": list(SENSITIVE_FIELDS),
    }


def build_split(
    source: Path,
    output: Path,
    domain: str,
    text_max_candidates: int,
    invariance_limit: int,
) -> tuple[dict[str, Any], set[str], set[str], set[str], list[dict[str, Any]]]:
    entities: set[str] = set()
    relations: set[str] = set()
    words: set[str] = {"__unk__"}
    invariance: list[dict[str, Any]] = []
    example_count = candidate_row_count = supervised_examples = supervised_targets = 0
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for _, rows in grouped_rows(source):
            sample, candidates = inference_sample(rows, domain, text_max_candidates)
            target_count = attach_supervision(sample, candidates, rows, domain)
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            example_count += 1
            candidate_row_count += len(rows)
            supervised_examples += int(target_count > 0)
            supervised_targets += target_count
            entities.update(sample["subgraph"]["entities"])
            for head, relation, tail in sample["subgraph"]["tuples"]:
                entities.update((head, tail))
                relations.add(relation)
            words.update(token.lower() for token in TEXT_TOKEN_RE.findall(sample["question"]))
            if len(invariance) < invariance_limit:
                invariance.append(invariance_check(rows, domain, text_max_candidates))
    manifest = {
        "source_path": source.as_posix(),
        "source_sha256": sha256_file(source),
        "adapter_path": output.as_posix(),
        "adapter_sha256": sha256_file(output),
        "examples": example_count,
        "candidate_rows": candidate_row_count,
        "supervised_examples_with_reachable_target": supervised_examples,
        "supervised_targets": supervised_targets,
    }
    return manifest, entities, relations, words, invariance


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def prepare_dataset(
    root: Path,
    source_root: Path,
    output_root: Path,
    name: str,
    cfg: Mapping[str, str],
    text_max_candidates: int,
) -> dict[str, Any]:
    dataset_dir = output_root / name
    adapter_dir = dataset_dir / "adapter"
    if dataset_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing clean output: {dataset_dir}")
    adapter_dir.mkdir(parents=True)
    source_dir = source_root / cfg["directory"]
    split_results: dict[str, Any] = {}
    all_entities: set[str] = set()
    train_relations: set[str] = set()
    train_words: set[str] = set()
    checks: list[dict[str, Any]] = []
    dev_relations: set[str] = set()

    for split in ("train", "dev"):
        result, entities, relations, words, split_checks = build_split(
            source_dir / f"{split}_subgraph_retrieval.jsonl",
            adapter_dir / f"{split}.json",
            cfg["domain"],
            text_max_candidates,
            invariance_limit=3,
        )
        split_results[split] = result
        all_entities.update(entities)
        checks.extend({"split": split, **item} for item in split_checks)
        if split == "train":
            train_relations.update(relations)
            train_words.update(words)
        else:
            dev_relations.update(relations)

    declared_schema = (
        TEXT_SCHEMA_RELATIONS if cfg["domain"] == "text" else ONTOLOGY_SCHEMA_RELATIONS
    )
    relation_vocab = train_relations | declared_schema
    missing_dev_relations = sorted(dev_relations - relation_vocab)
    if missing_dev_relations:
        raise ValueError(
            f"{name}: DEV relations absent from TRAIN plus schema: {missing_dev_relations[:20]}"
        )
    if not all(item["pass"] for item in checks):
        raise RuntimeError(f"{name}: leakage-invariance gate failed")

    write_lines(adapter_dir / "entities.txt", all_entities)
    write_lines(adapter_dir / "relations.txt", relation_vocab)
    write_lines(
        adapter_dir / "vocab.txt",
        train_words
        | {token.lower() for rel in relation_vocab for token in TEXT_TOKEN_RE.findall(rel)},
    )
    relation_hash = sha256_file(adapter_dir / "relations.txt")
    config = {
        "adapter_contract": "production_generator_d_v1_gnn_rag_clean_v1",
        "dataset": name,
        "domain": cfg["domain"],
        "source_root": source_root.as_posix(),
        "splits_read": ["train", "dev"],
        "test_read": False,
        "text_max_candidate_rows": text_max_candidates if cfg["domain"] == "text" else None,
        "candidate_order": "source row order followed by within-row subgraph_units order; stable deduplication",
        "relation_vocabulary_source": "TRAIN graph relations plus declared graph schema; DEV compatibility check only",
        "declared_graph_schema_relations": sorted(declared_schema),
        "mentions_page_aliased_to_mentions": False,
        "supervision_attachment": "after frozen inference graph; exact existing unit nodes only; absent targets remain absent",
        "forbidden_graph_inputs": list(SENSITIVE_FIELDS),
        "code_commit": git_commit(root),
    }
    write_json(dataset_dir / "adapter_config.json", config)
    write_json(dataset_dir / "train_manifest.json", split_results["train"])
    write_json(dataset_dir / "dev_manifest.json", split_results["dev"])
    write_json(dataset_dir / "leakage_invariance.json", {"pass": True, "checks": checks})
    summary = {
        "dataset": name,
        "train_count": split_results["train"]["examples"],
        "dev_count": split_results["dev"]["examples"],
        "relation_vocabulary_size": len(relation_vocab),
        "relation_vocabulary_sha256": relation_hash,
        "leakage_invariance": "PASS",
    }
    write_json(dataset_dir / "adapter_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/production_generator_d_v1"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints/production_generator_d_v1_gnn_rag_clean"),
    )
    parser.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--text-max-candidates", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_root = (
        (root / args.source_root).resolve()
        if not args.source_root.is_absolute()
        else args.source_root.resolve()
    )
    output_root = (
        (root / args.output_root).resolve()
        if not args.output_root.is_absolute()
        else args.output_root.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.datasets:
        print(f"Preparing clean GNN-RAG adapter: {name}", flush=True)
        summaries.append(
            prepare_dataset(
                root,
                source_root,
                output_root,
                name,
                DATASETS[name],
                max(1, args.text_max_candidates),
            )
        )
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    write_json(output_root / "adapter_run_summary.json", summaries)


if __name__ == "__main__":
    main()
