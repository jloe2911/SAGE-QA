import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
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
    text = str(text or "").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def sentence_parts(unit: str) -> Tuple[str, str, str]:
    parts = str(unit).split("::", 3)
    if len(parts) == 4 and parts[0] == "SENT":
        return clean(parts[1]), clean(parts[2]), clean(parts[3])
    return "Evidence", "0", clean(unit)


def kg_parts(unit: str) -> Tuple[str, str, str] | None:
    parts = str(unit).split("::", 3)
    if len(parts) == 4 and parts[0] == "KG":
        return clean(parts[1]), clean(parts[2]), clean(parts[3])
    return None


def evidence_node(unit: str) -> str:
    title, idx, sent = sentence_parts(unit)
    return f"EVIDENCE::{title}::{idx}::{sent}"


def kg_entity_node(entity: str) -> str:
    return f"KG_ENTITY::{clean(entity)}"


def kg_relation(rel: str) -> str:
    rel = clean(rel)
    rel = re.sub(r"\s+", "_", rel)
    return rel or "related_to"


def question_node(example_id: str) -> str:
    return "QUESTION::" + re.sub(r"[^A-Za-z0-9_]+", "_", example_id)[-96:]


def add_unique(values: List[str], seen: set, items: Iterable[str]) -> None:
    for item in items:
        item = clean(item)
        if item and item not in seen:
            seen.add(item)
            values.append(item)


def collect_kg_units(rows: List[Dict[str, Any]]) -> List[str]:
    kg_units: List[str] = []
    seen = set()
    for row in rows:
        for unit in row.get("graph_context_units", []) or []:
            if kg_parts(unit) is not None and unit not in seen:
                seen.add(unit)
                kg_units.append(unit)
    return kg_units


def text_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(str(text)) if len(tok) > 1}


def text_mentions_entity(text: str, entity: str) -> bool:
    entity = clean(entity)
    text = clean(text)
    if not entity or not text:
        return False
    entity_norm = entity.lower()
    text_norm = text.lower()
    if entity_norm in text_norm:
        return True
    entity_tokens = text_tokens(entity)
    if not entity_tokens:
        return False
    return len(entity_tokens & text_tokens(text)) / len(entity_tokens) >= 0.6


def flush_example(
    rows: List[Dict[str, Any]],
    max_candidates: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seed = rows[0]
    example_id = str(seed["example_id"])
    q_node = question_node(example_id)

    candidate_units: List[str] = []
    seen_units = set()
    for row in rows[:max_candidates]:
        add_unique(candidate_units, seen_units, row.get("subgraph_units", []) or [])
        add_unique(
            candidate_units,
            seen_units,
            row.get("gold_support_units", []) or [],
        )

    gold_units: List[str] = []
    seen_gold = set()
    for row in rows:
        add_unique(
            gold_units,
            seen_gold,
            row.get("gold_support_units", []) or [],
        )
        if gold_units:
            break

    if not candidate_units:
        candidate_units = gold_units[:]

    kg_units = collect_kg_units(rows)
    target_units = gold_units or candidate_units[:1]
    target_nodes = [evidence_node(unit) for unit in target_units]
    if not target_nodes:
        target_nodes = [q_node]

    tuples = []
    entities = {q_node, *target_nodes}
    title_nodes = OrderedDict()

    for unit in candidate_units:
        title, idx, sent = sentence_parts(unit)
        title_node = f"TITLE::{title}"
        ev_node = evidence_node(unit)
        title_nodes[title_node] = None
        entities.update([title_node, ev_node])
        tuples.append([q_node, "mentions", title_node])
        tuples.append([title_node, f"sentence_{idx}", ev_node])

        # Add a lexical bridge from question to every evidence sentence so
        # upstream path extraction has a direct reasoning path if title matching is weak.
        tuples.append([q_node, "retrieved_evidence", ev_node])

    kg_entity_nodes = OrderedDict()
    for unit in kg_units:
        parts = kg_parts(unit)
        if parts is None:
            continue
        subject, predicate, obj = parts
        subj_node = kg_entity_node(subject)
        obj_node = kg_entity_node(obj)
        rel = kg_relation(predicate)
        kg_entity_nodes[subj_node] = None
        kg_entity_nodes[obj_node] = None
        entities.update([subj_node, obj_node])
        tuples.append([subj_node, rel, obj_node])
        tuples.append([q_node, "kg_context", subj_node])
        tuples.append([q_node, "kg_context", obj_node])

        for sent_unit in candidate_units:
            title, idx, sent = sentence_parts(sent_unit)
            ev_node = evidence_node(sent_unit)
            haystack = f"{title} {sent}"
            if text_mentions_entity(haystack, subject):
                tuples.append([ev_node, "mentions_kg_entity", subj_node])
                tuples.append([subj_node, "mentioned_in_sentence", ev_node])
            if text_mentions_entity(haystack, obj):
                tuples.append([ev_node, "mentions_kg_entity", obj_node])
                tuples.append([obj_node, "mentioned_in_sentence", ev_node])

    sample = {
        "id": example_id,
        "question": clean(seed.get("question", "")),
        "entities": [q_node],
        "q_entity": [q_node],
        "answer": clean(seed.get("answer", "")),
        "choices": [],
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
        "answer": sample["answer"],
        "dataset": seed.get("dataset", seed.get("source_dataset", "")),
        "hop": seed.get("hop", ""),
        "answer_type": seed.get("answer_type", ""),
        "gold_support_units": gold_units,
        "kg_units": kg_units,
    }
    return sample, details


def build_split(
    path: Path, max_candidates: int, keep_details: bool
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    samples = []
    details = []
    current_id = None
    rows: List[Dict[str, Any]] = []

    for row in iter_jsonl(path):
        example_id = str(row["example_id"])
        if current_id is None:
            current_id = example_id
        if example_id != current_id:
            sample, detail = flush_example(rows, max_candidates=max_candidates)
            samples.append(sample)
            if keep_details:
                details.append(detail)
            rows = []
            current_id = example_id
        rows.append(row)

    if rows:
        sample, detail = flush_example(rows, max_candidates=max_candidates)
        samples.append(sample)
        if keep_details:
            details.append(detail)

    return samples, details


def build_vocab(
    samples_by_split: Dict[str, List[Dict[str, Any]]],
) -> Tuple[set[str], set[str], set[str]]:
    entities = set()
    relations = set()
    words = {"__unk__"}
    for samples in samples_by_split.values():
        for sample in samples:
            words.update(
                token.lower() for token in TOKEN_RE.findall(sample["question"])
            )
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
    parser.add_argument("--max-candidates", type=int, default=5)
    args = parser.parse_args()

    samples_by_split: Dict[str, List[Dict[str, Any]]] = {}
    details_rows = []
    for split in ("train", "dev", "test"):
        samples, details = build_split(
            args.data_dir / f"{split}_subgraph_retrieval.jsonl",
            max_candidates=max(1, args.max_candidates),
            keep_details=(split == "test"),
        )
        samples_by_split[split] = samples
        details_rows.extend(details)

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
                "max_candidates": max(1, args.max_candidates),
                "gnn_data_dir": str(args.gnn_data_dir),
                "details": str(args.details_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
