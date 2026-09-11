import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in sorted(set(values)):
            f.write(str(value) + "\n")


def clean(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"<[^#>\s]+#([^>]+)>", r"\1", text)
    text = text.replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def answer_node(answer: Any) -> str:
    return "ANSWER::" + clean(answer).upper()


def is_boolean_answer(answer: Any) -> bool:
    return clean(answer).upper() in {"TRUE", "FALSE", "YES", "NO"}


def unit_node(unit: str) -> str:
    return "EVIDENCE::" + clean(unit)


def relation_for_unit(unit: str) -> str:
    unit = clean(unit)
    if " " in unit:
        parts = unit.split()
        if len(parts) >= 3:
            return parts[1]
    match = re.match(r"([A-Za-z0-9_]+)\(", unit)
    if match:
        return match.group(1)
    return "related_to"


def extract_question_entities(row: Dict[str, Any]) -> List[str]:
    entities = []
    for token in TOKEN_RE.findall(str(row.get("sparql_query", ""))):
        if "_" in token and not token.startswith("http"):
            entities.append(token)
    seen = set()
    out = []
    for entity in entities:
        entity = clean(entity)
        if entity and entity not in seen:
            seen.add(entity)
            out.append(entity)
    return out[:4]


def group_examples(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in iter_jsonl(path):
        grouped[str(row["example_id"])].append(row)
    return dict(grouped)


def answer_only_rows(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    return {str(row["example_id"]): [row] for row in iter_jsonl(path)}


def build_sample(
    example_id: str, rows: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seed = rows[0]
    extracted_q_entities = extract_question_entities(seed)
    entities = set(extracted_q_entities)
    tuples = []
    candidate_units = []
    seen_units = set()
    target_units = []
    seen_targets = set()

    for row in rows:
        for unit in row.get("subgraph_units", []) or []:
            unit = clean(unit)
            if unit and unit not in seen_units:
                seen_units.add(unit)
                candidate_units.append(unit)
        for unit in row.get("answer_context_units", []) or []:
            unit = clean(unit)
            if unit and unit not in seen_units:
                seen_units.add(unit)
                candidate_units.append(unit)
        for unit in row.get("gold_units", []) or []:
            unit = clean(unit)
            if unit and unit not in seen_targets:
                seen_targets.add(unit)
                target_units.append(unit)

    if not target_units:
        for row in rows:
            for unit in row.get("answer_context_units", []) or []:
                unit = clean(unit)
                if unit and unit not in seen_targets:
                    seen_targets.add(unit)
                    target_units.append(unit)

    target_nodes = [unit_node(unit) for unit in target_units]
    if not target_nodes:
        target_nodes = [answer_node(seed.get("answer", ""))]

    q_entities = extracted_q_entities
    if not q_entities and candidate_units:
        q_entities = [unit_node(candidate_units[0])]
    elif not q_entities:
        q_entities = target_nodes[:1]

    entities.update(q_entities)

    for entity in q_entities:
        entities.add(entity)

    for unit in candidate_units:
        node = unit_node(unit)
        entities.add(node)
        rel = relation_for_unit(unit)
        for entity in q_entities:
            if entity != node:
                tuples.append([entity, rel, node])

    sample = {
        "id": example_id,
        "question": clean(seed.get("question", "")),
        "entities": q_entities,
        "q_entity": q_entities,
        "answer": clean(seed.get("answer", "")),
        "choices": ["TRUE", "FALSE"] if is_boolean_answer(seed.get("answer", "")) else [],
        "graph": tuples,
        "answers": [{"kb_id": node, "text": node} for node in target_nodes],
        "subgraph": {
            "entities": sorted(entities),
            "tuples": tuples,
        },
    }

    details = {
        "example_id": example_id,
        "question": sample["question"],
        "answer": clean(seed.get("answer", "")),
        "dataset": seed.get("source_name", seed.get("dataset", "")),
        "hop": seed.get("hop", ""),
        "answer_type": seed.get("answer_type", ""),
        "gold_explanations": seed.get("gold_explanations", []),
        "gold_units": seed.get("gold_units", []),
        "evaluation_scope": seed.get("evaluation_scope", "support"),
        "answer_context_units": seed.get("answer_context_units", []),
    }
    return sample, details


def build_vocab(
    samples_by_split: Dict[str, List[Dict[str, Any]]],
) -> Tuple[set[str], set[str], set[str]]:
    entities = set()
    relations = set()
    words = {"__unk__"}
    for samples in samples_by_split.values():
        for sample in samples:
            words.update(token.lower() for token in TOKEN_RE.findall(sample["question"]))
            entities.update(sample["entities"])
            entities.update(sample["subgraph"]["entities"])
            for answer in sample["answers"]:
                entities.add(answer["kb_id"])
            for head, rel, tail in sample["subgraph"]["tuples"]:
                entities.add(head)
                entities.add(tail)
                relations.add(rel)
                words.update(token.lower() for token in TOKEN_RE.findall(rel))
    return entities, relations, words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gnn-data-dir", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    args = parser.parse_args()

    samples_by_split: Dict[str, List[Dict[str, Any]]] = {}
    details_rows = []
    for split in ("train", "dev", "test"):
        grouped = group_examples(args.data_dir / f"{split}_subgraph_retrieval.jsonl")
        grouped.update(
            answer_only_rows(args.data_dir / f"{split}_answer_only_no_explanation.jsonl")
        )
        samples = []
        for example_id, rows in grouped.items():
            sample, details = build_sample(example_id, rows)
            samples.append(sample)
            if split == "test":
                details_rows.append(details)
        samples_by_split[split] = samples

    for split, samples in samples_by_split.items():
        write_jsonl(args.gnn_data_dir / f"{split}.json", samples)

    entities, relations, words = build_vocab(samples_by_split)
    write_lines(args.gnn_data_dir / "entities.txt", entities)
    write_lines(args.gnn_data_dir / "relations.txt", relations)
    write_lines(args.gnn_data_dir / "vocab.txt", words)

    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    with args.details_output.open("w", encoding="utf-8") as f:
        json.dump(details_rows, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "train": len(samples_by_split["train"]),
                "dev": len(samples_by_split["dev"]),
                "test": len(samples_by_split["test"]),
                "entities": len(entities),
                "relations": len(relations),
                "gnn_data_dir": str(args.gnn_data_dir),
                "details": str(args.details_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
