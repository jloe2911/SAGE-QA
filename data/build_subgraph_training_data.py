import argparse
import itertools
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.symbolic_composer import extract_query_signature, parse_axiom


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"


def local_name(uri: str) -> str:
    uri = str(uri or "").strip()
    if uri.startswith("<") and uri.endswith(">"):
        uri = uri[1:-1]
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.rstrip("/").split("/")[-1]
    return uri


def split_tag(tag: str) -> Tuple[str, str]:
    if tag.startswith("{") and "}" in tag:
        ns, name = tag[1:].split("}", 1)
        return ns, name
    return "", tag


def normalize_explanation_unit(unit: str) -> str:
    unit = " ".join(str(unit).strip().split())

    if not unit or unit.startswith("TAG:"):
        return ""

    m = re.match(r"^Symmetric:\s*(.+)$", unit)
    if m:
        return f"SymmetricObjectProperty({m.group(1).strip()})"

    m = re.match(r"^Transitive:\s*(.+)$", unit)
    if m:
        return f"TransitiveObjectProperty({m.group(1).strip()})"

    m = re.match(r"^Functional:\s*(.+)$", unit)
    if m:
        return f"FunctionalObjectProperty({m.group(1).strip()})"

    m = re.match(r"^domain\((.+)\)\s*=\s*(.+)$", unit)
    if m:
        return f"{m.group(1).strip()} domain {m.group(2).strip()}"

    m = re.match(r"^range\((.+)\)\s*=\s*(.+)$", unit)
    if m:
        return f"{m.group(1).strip()} range {m.group(2).strip()}"

    parts = unit.split()
    if len(parts) == 3 and parts[1] == "SubPropertyOf":
        return f"SubObjectPropertyOf({parts[0]},{parts[2]})"
    if len(parts) == 3 and parts[1] == "Domain":
        return f"{parts[0]} domain {parts[2]}"
    if len(parts) == 3 and parts[1] == "Range":
        return f"{parts[0]} range {parts[2]}"

    return unit


def normalize_explanation(explanation: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for unit in explanation:
        normalized = normalize_explanation_unit(unit)
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def parse_owl_context(owl_context: str) -> List[str]:
    root = ET.fromstring(owl_context)
    axioms = []
    seen = set()

    def add(axiom: str) -> None:
        axiom = " ".join(str(axiom).strip().split())
        if axiom and axiom not in seen:
            axioms.append(axiom)
            seen.add(axiom)

    for desc in root.findall(f".//{RDF}Description"):
        subject_uri = desc.attrib.get(f"{RDF}about")
        if not subject_uri:
            continue

        subject = local_name(subject_uri)

        for child in list(desc):
            _, child_name = split_tag(child.tag)
            resource = child.attrib.get(f"{RDF}resource")
            if not resource:
                continue

            obj = local_name(resource)

            if child.tag == f"{RDF}type":
                if obj == "SymmetricProperty":
                    add(f"SymmetricObjectProperty({subject})")
                elif obj == "TransitiveProperty":
                    add(f"TransitiveObjectProperty({subject})")
                elif obj == "FunctionalProperty":
                    add(f"FunctionalObjectProperty({subject})")
                elif obj not in {"NamedIndividual", "ObjectProperty", "Ontology"}:
                    add(f"{subject} rdf:type {obj}")
            elif child.tag == f"{RDFS}subPropertyOf":
                add(f"SubObjectPropertyOf({subject},{obj})")
            elif child.tag == f"{OWL}inverseOf":
                add(f"InverseObjectProperties({subject},{obj})")
            elif child.tag == f"{OWL}equivalentProperty":
                add(f"EquivalentObjectProperties({subject},{obj})")
            elif child.tag == f"{RDFS}domain":
                add(f"{subject} domain {obj}")
            elif child.tag == f"{RDFS}range":
                add(f"{subject} range {obj}")
            elif child_name not in {
                "comment",
                "first",
                "rest",
                "propertyChainAxiom",
            }:
                add(f"{subject} {child_name} {obj}")

    return axioms


def unit_signature(unit: str) -> Tuple[Set[str], Set[str]]:
    parsed = parse_axiom(unit)
    entities = set(parsed.entities())
    properties = set(parsed.properties())

    if parsed.axiom_type == "unknown":
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_:-]*", unit))
        entities |= {t for t in tokens if t and t[0].islower()}
        properties |= {t for t in tokens if t in unit}

    return entities, properties


def edge_between_units(left: str, right: str) -> bool:
    left_entities, left_properties = unit_signature(left)
    right_entities, right_properties = unit_signature(right)
    return bool(
        (left_entities & right_entities) or (left_properties & right_properties)
    )


def is_connected(units: Tuple[str, ...]) -> bool:
    if len(units) <= 1:
        return True

    visited = {0}
    stack = [0]

    while stack:
        i = stack.pop()
        for j in range(len(units)):
            if j not in visited and edge_between_units(units[i], units[j]):
                visited.add(j)
                stack.append(j)

    return len(visited) == len(units)


def set_scores(pred: List[str], gold_explanations: List[List[str]]) -> Dict:
    pred_set = set(pred)
    best = {
        "best_jaccard_to_gold": 0.0,
        "best_set_precision_to_gold": 0.0,
        "best_set_recall_to_gold": 0.0,
        "best_set_f1_to_gold": 0.0,
        "exact_match_any_gold": False,
        "contains_any_gold_explanation": False,
        "contained_in_any_gold_explanation": False,
        "best_matching_gold_explanation": [],
        "best_matching_gold_index": -1,
    }

    for i, gold in enumerate(gold_explanations):
        gold_set = set(gold)
        if not gold_set:
            continue

        inter = len(pred_set & gold_set)
        precision = inter / max(len(pred_set), 1)
        recall = inter / max(len(gold_set), 1)
        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        jaccard = inter / max(len(pred_set | gold_set), 1)
        exact = pred_set == gold_set
        contains = gold_set.issubset(pred_set)
        contained = pred_set.issubset(gold_set)

        if f1 > best["best_set_f1_to_gold"]:
            best.update(
                {
                    "best_jaccard_to_gold": jaccard,
                    "best_set_precision_to_gold": precision,
                    "best_set_recall_to_gold": recall,
                    "best_set_f1_to_gold": f1,
                    "best_matching_gold_explanation": gold,
                    "best_matching_gold_index": i,
                }
            )

        best["exact_match_any_gold"] = best["exact_match_any_gold"] or exact
        best["contains_any_gold_explanation"] = (
            best["contains_any_gold_explanation"] or contains
        )
        best["contained_in_any_gold_explanation"] = (
            best["contained_in_any_gold_explanation"] or contained
        )

    return best


def relevant_context_axioms(
    axioms: List[str],
    question: str,
    sparql_query: str,
    max_context_units: int,
) -> List[str]:
    signature = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities = set(signature.get("query_entities", []))
    query_properties = set(signature.get("query_properties", []))

    scored = []
    for axiom in axioms:
        entities, properties = unit_signature(axiom)
        score = 2 * len(entities & query_entities) + len(properties & query_properties)
        if score > 0:
            scored.append((score, axiom))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [axiom for _, axiom in scored[:max_context_units]]


def infer_split(group_index: int, train_ratio: float, dev_ratio: float) -> str:
    bucket = group_index % 100
    if bucket < int(train_ratio * 100):
        return "train"
    if bucket < int((train_ratio + dev_ratio) * 100):
        return "dev"
    return "test"


def build_rows_for_qa(
    source_name: str,
    group_index: int,
    qa_index: int,
    item: Dict,
    qa: Dict,
    split: str,
    max_subgraph_size: int,
    min_subgraph_size: int,
    max_context_units: int,
    max_negative_per_example: int,
) -> List[Dict]:
    sparql_query = str(qa.get("SPARQL Query") or "")
    question = (
        qa.get("NL Question")
        or qa.get("ABS Question")
        or qa.get("Task ID")
        or sparql_query
    )
    question = str(question)

    gold_explanations = [
        normalized
        for explanation in qa.get("Explanations", [])
        if (normalized := normalize_explanation(explanation))
    ]

    if not gold_explanations:
        minimum = normalize_explanation(qa.get("Minimum Explanation", []))
        if minimum:
            gold_explanations = [minimum]

    if not gold_explanations:
        return []

    context_axioms = parse_owl_context(item["OWL Context"])
    candidate_units = []
    seen = set()

    for explanation in gold_explanations:
        for unit in explanation:
            if unit not in seen:
                candidate_units.append(unit)
                seen.add(unit)

    for unit in relevant_context_axioms(
        context_axioms,
        question=question,
        sparql_query=sparql_query,
        max_context_units=max_context_units,
    ):
        if unit not in seen:
            candidate_units.append(unit)
            seen.add(unit)

    candidate_subgraphs = set()

    for gold in gold_explanations:
        if min_subgraph_size <= len(gold) <= max_subgraph_size:
            candidate_subgraphs.add(tuple(gold))

    max_size = min(max_subgraph_size, len(candidate_units))
    for size in range(min_subgraph_size, max_size + 1):
        for combo in itertools.combinations(candidate_units, size):
            if is_connected(combo):
                candidate_subgraphs.add(combo)

    positives = []
    negatives = []

    for combo in candidate_subgraphs:
        subgraph_units = list(combo)
        scores = set_scores(subgraph_units, gold_explanations)
        label = int(
            scores["exact_match_any_gold"] or scores["contains_any_gold_explanation"]
        )

        row = {
            "example_id": (
                f"{source_name}__g{group_index}__q{qa_index}__"
                f"{qa.get('Task ID', '')}__{question}__{sparql_query}"
            ),
            "split": split,
            "question": question,
            "sparql_query": sparql_query,
            "task_type": item.get("Task Type", ""),
            "answer_type": item.get("Answer Type", ""),
            "answer": qa.get("Answer"),
            "source_name": source_name,
            "group_index": group_index,
            "qa_index": qa_index,
            "subgraph_node_ids": [
                candidate_units.index(unit)
                for unit in subgraph_units
                if unit in candidate_units
            ],
            "subgraph_units": subgraph_units,
            "subgraph_size": len(subgraph_units),
            "gold_explanations": gold_explanations,
            "gold_units": gold_explanations[0],
            **scores,
            "label": label,
        }

        if label:
            positives.append(row)
        else:
            negatives.append(row)

    negatives.sort(key=lambda r: r["best_set_f1_to_gold"], reverse=True)
    if max_negative_per_example >= 0:
        negatives = negatives[:max_negative_per_example]

    rows = positives + negatives
    random.shuffle(rows)
    return rows


def empty_splits() -> Dict[str, List[Dict]]:
    return {"train": [], "dev": [], "test": []}


def build_dataset(args: argparse.Namespace) -> Dict[str, Dict[str, List[Dict]]]:
    output = {}

    for input_path in args.input_json:
        source_name = Path(input_path).stem
        print(f"[LOAD] {source_name}: {input_path}")
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        source_output = empty_splits()

        for group_index, item in enumerate(data):
            if args.max_groups and group_index >= args.max_groups:
                break

            split = infer_split(
                group_index,
                train_ratio=args.train_ratio,
                dev_ratio=args.dev_ratio,
            )

            for qa_index, qa in enumerate(item.get("QAs", [])):
                rows = build_rows_for_qa(
                    source_name=source_name,
                    group_index=group_index,
                    qa_index=qa_index,
                    item=item,
                    qa=qa,
                    split=split,
                    max_subgraph_size=args.max_subgraph_size,
                    min_subgraph_size=args.min_subgraph_size,
                    max_context_units=args.max_context_units,
                    max_negative_per_example=args.max_negative_per_example,
                )
                source_output[split].extend(rows)

            if (group_index + 1) % 100 == 0:
                print(f"  processed {group_index + 1}/{len(data)} groups")

        output[source_name] = source_output

    return output


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[WRITE] {path}: {len(rows)} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final support-subgraph retrieval JSONL files."
    )
    parser.add_argument(
        "--input-json",
        nargs="+",
        default=["FamilyOWL_1hop.json", "FamilyOWL_2hop.json"],
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--min-subgraph-size", type=int, default=1)
    parser.add_argument("--max-subgraph-size", type=int, default=3)
    parser.add_argument("--max-context-units", type=int, default=40)
    parser.add_argument("--max-negative-per-example", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument(
        "--combined",
        action="store_true",
        help=(
            "Also write combined split files directly under --output-dir. "
            "Per-source files are always written."
        ),
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Optional smoke-test limit per input JSON. Use 0 for all groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_source = build_dataset(args)
    output_dir = Path(args.output_dir)

    combined = empty_splits()

    for source_name, splits in by_source.items():
        for split_name, rows in splits.items():
            write_jsonl(
                output_dir / source_name / f"{split_name}_subgraph_retrieval.jsonl",
                rows,
            )
            combined[split_name].extend(rows)

    if args.combined:
        for split_name, rows in combined.items():
            write_jsonl(output_dir / f"{split_name}_subgraph_retrieval.jsonl", rows)


if __name__ == "__main__":
    main()
