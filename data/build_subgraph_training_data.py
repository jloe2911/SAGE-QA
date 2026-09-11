import argparse
import itertools
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from rdflib import Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL as OWL_NS
from rdflib.namespace import RDF as RDF_NS
from rdflib.namespace import RDFS as RDFS_NS

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.symbolic_composer import extract_query_signature, parse_axiom
from data_processing.retrieval_contracts import (
    BUILDER_VERSION,
    git_provenance,
    sha256_file,
    write_json,
)
from data_processing.evidence_graph_candidates import (
    GENERATOR_D_CONFIG,
    GENERATOR_D_VERSION,
    build_ontology_evidence_graph,
    frozen_generator_d_config,
    frozen_generator_d_lineage,
    generate_generator_d_candidates,
)


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

    # Some OWL2Bench JSON rows contain UTF-8 text decoded once as cp1252
    # (for example ``âˆ˜``/``âŠ‘`` instead of ``∘``/``⊑``).
    if "â" in unit:
        try:
            unit = unit.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    if not unit or unit.startswith("TAG:"):
        return ""

    m = re.match(r"^(\S+)\s+owl:inverseOf\s+(\S+)$", unit)
    if m:
        left, right = sorted((m.group(1), m.group(2)))
        return f"InverseObjectProperties({left},{right})"

    m = re.match(r"^(\S+)\s+rdfs:subPropertyOf\s+(\S+)$", unit)
    if m:
        return f"SubObjectPropertyOf({m.group(1)},{m.group(2)})"

    m = re.match(r"^PropertyChain\((\S+)\s+∘\s+(\S+)\)\s+⊑\s+(\S+)$", unit)
    if m:
        return f"ObjectPropertyChain({m.group(1)},{m.group(2)}->{m.group(3)})"

    m = re.match(
        r"^SubObjectPropertyOf\(ObjectPropertyChain\((.+)\)\s+(<[^>]+>|\S+)\)$",
        unit,
    )
    if m:
        chain = [local_name(token) for token in re.findall(r"<[^>]+>|\S+", m.group(1))]
        super_property = local_name(m.group(2))
        if chain:
            return f"ObjectPropertyChain({','.join(chain)}->{super_property})"

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


def get_gold_explanations(qa: Dict) -> List[List[str]]:
    gold_explanations = [
        normalized
        for explanation in qa.get("Explanations", []) or []
        if (normalized := normalize_explanation(explanation))
    ]

    if not gold_explanations:
        minimum = normalize_explanation(qa.get("Minimum Explanation", []) or [])
        if minimum:
            gold_explanations = [minimum]

    return gold_explanations


def parse_rdf_graph(owl_context: str) -> Graph:
    text = str(owl_context or "").lstrip("\ufeff").strip()
    if not text:
        return Graph()

    preferred_formats = (
        ["xml", "turtle"]
        if text.startswith("<?xml") or "<rdf:RDF" in text[:500]
        else ["turtle", "xml"]
    )
    errors = []

    for rdf_format in preferred_formats:
        graph = Graph()
        try:
            graph.parse(data=text, format=rdf_format)
            return graph
        except Exception as exc:
            errors.append(f"{rdf_format}: {exc}")

    raise ValueError(
        f"Could not parse OWL Context as Turtle or RDF/XML. Parser errors: {'; '.join(errors)}"
    )


def parse_owl_context(owl_context: str) -> List[str]:
    graph = parse_rdf_graph(owl_context)
    axioms = []
    seen = set()

    def add(axiom: str) -> None:
        axiom = " ".join(str(axiom).strip().split())
        if axiom and axiom not in seen:
            axioms.append(axiom)
            seen.add(axiom)

    for subject_ref, predicate_ref, object_ref in graph:
        if not isinstance(subject_ref, URIRef) or not isinstance(object_ref, URIRef):
            continue

        subject = local_name(str(subject_ref))
        predicate = predicate_ref
        obj = local_name(str(object_ref))

        if predicate == RDF_NS.type:
            if obj in {"SymmetricProperty", "SymmetricObjectProperty"}:
                add(f"SymmetricObjectProperty({subject})")
            elif obj in {"TransitiveProperty", "TransitiveObjectProperty"}:
                add(f"TransitiveObjectProperty({subject})")
            elif obj in {"FunctionalProperty", "FunctionalObjectProperty"}:
                add(f"FunctionalObjectProperty({subject})")
            elif obj not in {
                "NamedIndividual",
                "ObjectProperty",
                "Ontology",
                "Class",
            }:
                add(f"{subject} rdf:type {obj}")
        elif predicate == RDFS_NS.subPropertyOf:
            add(f"SubObjectPropertyOf({subject},{obj})")
        elif predicate == RDFS_NS.subClassOf:
            add(f"{subject} SubClassOf {obj}")
        elif predicate == OWL_NS.inverseOf:
            add(f"InverseObjectProperties({subject},{obj})")
        elif predicate == OWL_NS.equivalentProperty:
            add(f"EquivalentObjectProperties({subject},{obj})")
        elif predicate == RDFS_NS.domain:
            add(f"{subject} domain {obj}")
        elif predicate == RDFS_NS.range:
            add(f"{subject} range {obj}")
        else:
            pred = local_name(str(predicate_ref))
            if pred not in {"comment", "first", "rest", "propertyChainAxiom"}:
                add(f"{subject} {pred} {obj}")

    for super_property, chain_head in graph.subject_objects(OWL_NS.propertyChainAxiom):
        if not isinstance(super_property, URIRef):
            continue
        chain = [
            local_name(str(member))
            for member in Collection(graph, chain_head)
            if isinstance(member, URIRef)
        ]
        if chain:
            add(f"ObjectPropertyChain({','.join(chain)}->{local_name(str(super_property))})")

    return axioms


def unit_signature(unit: str) -> Tuple[Set[str], Set[str]]:
    if str(unit).startswith("KG::"):
        parts = str(unit).split("::", 3)
        if len(parts) == 4:
            return {parts[1], parts[3]}, {parts[2]}
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
        (left_entities & right_entities)
        or (left_properties & right_properties)
        or (left_entities & right_properties)
        or (left_properties & right_entities)
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


def build_unit_adjacency(candidate_units: List[str]) -> List[Set[int]]:
    adjacency = [set() for _ in candidate_units]

    for i, j in itertools.combinations(range(len(candidate_units)), 2):
        if edge_between_units(candidate_units[i], candidate_units[j]):
            adjacency[i].add(j)
            adjacency[j].add(i)

    return adjacency


def score_unit_for_query(
    unit: str,
    query_entities: Set[str],
    query_properties: Set[str],
    degree: int = 0,
) -> float:
    entities, properties = unit_signature(unit)
    parsed = parse_axiom(unit)

    score = 0.0
    score += 2.0 * len(entities & query_entities)
    score += 1.5 * len(properties & query_properties)
    score += 0.15 if parsed.axiom_type == "fact" else 0.0
    score += 0.10 if parsed.axiom_type == "rule" else 0.0
    score += min(degree, 8) * 0.01
    return score


def score_subgraph_indices(
    indices: Tuple[int, ...],
    candidate_units: List[str],
    adjacency: List[Set[int]],
    unit_scores: List[float],
) -> float:
    units = [candidate_units[i] for i in indices]
    parsed = [parse_axiom(unit) for unit in units]
    kinds = {p.axiom_type for p in parsed}

    internal_edges = 0
    for pos, i in enumerate(indices):
        for j in indices[pos + 1 :]:
            internal_edges += int(j in adjacency[i])

    fact_rule_mix = 1.0 if ("fact" in kinds and "rule" in kinds) else 0.0
    avg_relevance = sum(unit_scores[i] for i in indices) / max(len(indices), 1)

    return (
        avg_relevance
        + 0.05 * internal_edges
        + 0.20 * fact_rule_mix
        - 0.015 * max(0, len(indices) - 3)
    )


def beam_connected_subgraphs(
    candidate_units: List[str],
    question: str,
    sparql_query: str,
    min_subgraph_size: int,
    max_subgraph_size: int,
    beam_width: int,
    max_candidate_subgraphs: int,
) -> Set[Tuple[str, ...]]:
    """
    Generate a bounded set of connected candidate supports.

    This replaces exhaustive n-choose-k enumeration for larger support sizes.
    The beam expands only through graph-neighboring axioms, so 4+ axiom chains
    are possible without materializing every combination.
    """
    if not candidate_units or max_subgraph_size < min_subgraph_size:
        return set()

    signature = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities = set(signature.get("query_entities", []))
    query_properties = set(signature.get("query_properties", []))

    adjacency = build_unit_adjacency(candidate_units)
    unit_scores = [
        score_unit_for_query(
            unit,
            query_entities=query_entities,
            query_properties=query_properties,
            degree=len(adjacency[i]),
        )
        for i, unit in enumerate(candidate_units)
    ]

    max_size = min(max_subgraph_size, len(candidate_units))
    beam_width = max(1, beam_width)
    max_candidate_subgraphs = max(1, max_candidate_subgraphs)

    seed_indices = sorted(
        range(len(candidate_units)),
        key=lambda i: (-unit_scores[i], candidate_units[i]),
    )

    frontier = [(i,) for i in seed_indices[: max(beam_width, max_candidate_subgraphs)]]
    scored_candidates = []
    seen = set(frontier)

    for size in range(1, max_size + 1):
        if size >= min_subgraph_size:
            for combo in frontier:
                scored_candidates.append(
                    (
                        score_subgraph_indices(
                            combo,
                            candidate_units=candidate_units,
                            adjacency=adjacency,
                            unit_scores=unit_scores,
                        ),
                        combo,
                    )
                )

        if size == max_size:
            break

        expansions = {}
        for combo in frontier:
            combo_set = set(combo)
            neighbors = set()
            for idx in combo:
                neighbors.update(adjacency[idx])

            for nxt in neighbors - combo_set:
                expanded = tuple(sorted((*combo, nxt)))
                if expanded in seen:
                    continue
                seen.add(expanded)
                expansions[expanded] = score_subgraph_indices(
                    expanded,
                    candidate_units=candidate_units,
                    adjacency=adjacency,
                    unit_scores=unit_scores,
                )

        if not expansions:
            break

        frontier = [
            combo
            for combo, _ in sorted(
                expansions.items(),
                key=lambda item: (-item[1], item[0]),
            )[:beam_width]
        ]

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    return {
        tuple(candidate_units[i] for i in combo)
        for _, combo in scored_candidates[:max_candidate_subgraphs]
    }


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
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
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
        best["contains_any_gold_explanation"] = best["contains_any_gold_explanation"] or contains
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

    if max_context_units <= 0:
        return []

    signatures = [unit_signature(axiom) for axiom in axioms]
    tokens_by_index = [entities | properties for entities, properties in signatures]
    token_index: Dict[str, Set[int]] = defaultdict(set)
    direct_scores = []
    for index, (entities, properties) in enumerate(signatures):
        for token in tokens_by_index[index]:
            token_index[token].add(index)
        score = 2 * len(entities & query_entities) + len(properties & query_properties)
        if score > 0:
            direct_scores.append((score, index))

    direct_scores.sort(key=lambda item: (-item[0], axioms[item[1]]))
    selected: List[int] = []
    selected_set: Set[int] = set()
    frontier = []
    for _, index in direct_scores:
        if index not in selected_set and len(selected) < max_context_units:
            selected.append(index)
            selected_set.add(index)
            frontier.append(index)

    # Deterministic query-guided local expansion. This uses only formal-query
    # seeds and context connectivity, and admits bridge/schema axioms that do
    # not themselves mention the query entity or queried property.
    while frontier and len(selected) < max_context_units:
        neighbor_overlap: Dict[int, int] = defaultdict(int)
        for index in frontier:
            for token in tokens_by_index[index]:
                for neighbor in token_index[token]:
                    if neighbor not in selected_set:
                        neighbor_overlap[neighbor] += 1
        if not neighbor_overlap:
            break
        ranked = sorted(
            neighbor_overlap,
            key=lambda index: (
                -neighbor_overlap[index],
                -len(tokens_by_index[index] & (query_entities | query_properties)),
                axioms[index],
            ),
        )
        frontier = []
        for index in ranked:
            if len(selected) >= max_context_units:
                break
            selected.append(index)
            selected_set.add(index)
            frontier.append(index)

    return [axioms[index] for index in selected]


def answer_type_for(item: Dict, qa: Dict | None = None) -> str:
    qa = qa or {}
    return str(qa.get("Answer Type") or item.get("Answer Type") or "").strip()


def build_split_map(groups: List[Dict], train_ratio: float, dev_ratio: float) -> Dict[int, str]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in [0, 1], got {train_ratio}")
    if not 0.0 <= dev_ratio <= 1.0:
        raise ValueError(f"dev_ratio must be in [0, 1], got {dev_ratio}")
    if train_ratio + dev_ratio >= 1.0:
        raise ValueError(
            "train_ratio + dev_ratio must leave a non-empty test split "
            f"(got {train_ratio + dev_ratio})"
        )

    buckets: Dict[str, List[int]] = defaultdict(list)
    for group_index, item in enumerate(groups):
        buckets[answer_type_for(item)].append(group_index)

    rng = random.Random(RANDOM_SEED)

    split_by_group = {}
    for answer_type, indices in buckets.items():
        rng.shuffle(indices)

        train_end = int(len(indices) * train_ratio)
        dev_end = train_end + int(len(indices) * dev_ratio)

        for rank, group_index in enumerate(indices):
            if rank < train_end:
                split_by_group[group_index] = "train"
            elif rank < dev_end:
                split_by_group[group_index] = "dev"
            else:
                split_by_group[group_index] = "test"

        print(
            "[SPLIT] "
            f"answer_type={answer_type or 'UNKNOWN'} "
            f"train={train_end} dev={dev_end - train_end} "
            f"test={len(indices) - dev_end}"
        )

    return split_by_group


def generate_ontology_candidates(
    *,
    question: str,
    sparql_query: str,
    owl_context: str,
    max_subgraph_size: int,
    min_subgraph_size: int,
    max_context_units: int,
    candidate_beam_width: int,
    max_candidate_subgraphs: int,
) -> Dict:
    """Freeze ontology candidates without accepting answer or explanation fields."""
    # Legacy beam/size/budget arguments remain in the public signature for row
    # schema compatibility; Generator D always uses its frozen configuration.
    context_axioms = parse_owl_context(owl_context)
    signature = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities = set(signature.get("query_entities", []))
    query_properties = set(signature.get("query_properties", []))
    evidence_graph = build_ontology_evidence_graph(
        context_axioms,
        unit_signature=unit_signature,
        query_entities=query_entities,
        query_properties=query_properties,
        score_unit=lambda unit, entities, properties, degree: score_unit_for_query(
            unit,
            query_entities=entities,
            query_properties=properties,
            degree=degree,
        ),
    )
    result = generate_generator_d_candidates(evidence_graph)
    candidate_units = [unit.unit_id for unit in evidence_graph.units]
    adjacency = [set(neighbors) for neighbors in evidence_graph.adjacency]
    unit_scores = list(evidence_graph.query_scores)
    frozen_candidate_subgraphs = [tuple(candidate) for candidate in result.candidates]

    return {
        "candidate_units": candidate_units,
        "candidate_subgraphs": frozen_candidate_subgraphs,
        "adjacency": adjacency,
        "unit_scores": unit_scores,
        "candidate_generator": GENERATOR_D_VERSION,
        "candidate_generation_config": frozen_generator_d_config(),
        "query_anchor_unit_ids": list(result.query_anchor_unit_ids),
        "gold_available_during_candidate_generation": False,
    }


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
    candidate_beam_width: int,
    max_candidate_subgraphs: int,
) -> List[Dict]:
    sparql_query = str(qa.get("SPARQL Query") or "")
    question = str(
        qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or sparql_query
    )
    generated = generate_ontology_candidates(
        question=question,
        sparql_query=sparql_query,
        owl_context=item["OWL Context"],
        max_subgraph_size=max_subgraph_size,
        min_subgraph_size=min_subgraph_size,
        max_context_units=max_context_units,
        candidate_beam_width=candidate_beam_width,
        max_candidate_subgraphs=max_candidate_subgraphs,
    )
    candidate_units = generated["candidate_units"]
    frozen_candidate_subgraphs = generated["candidate_subgraphs"]
    adjacency = generated.get("adjacency", [])
    unit_scores = generated.get("unit_scores", [])
    if not candidate_units:
        return []

    # Labeling starts only after candidates freeze. Test rows remain gold-free
    # until final evaluation of frozen predictions.
    attach_gold = split != "test"
    gold_explanations = get_gold_explanations(qa) if attach_gold else []
    if attach_gold and not gold_explanations:
        return []
    candidate_set = set(candidate_units)
    gold_context_coverage = max(
        (
            len(set(explanation) & candidate_set) / max(len(set(explanation)), 1)
            for explanation in gold_explanations
        ),
        default=0.0,
    )

    rows = []

    for generation_rank, combo in enumerate(frozen_candidate_subgraphs):
        subgraph_units = list(combo)
        scores = set_scores(subgraph_units, gold_explanations)
        label = int(scores["exact_match_any_gold"] or scores["contains_any_gold_explanation"])

        row = {
            "example_id": (
                f"{source_name}__g{group_index}__q{qa_index}__"
                f"{qa.get('Task ID', '')}__{question}__{sparql_query}"
            ),
            "split": split,
            "question": question,
            "sparql_query": sparql_query,
            "task_type": item.get("Task Type", ""),
            "answer_type": answer_type_for(item, qa),
            "source_name": source_name,
            "group_index": group_index,
            "qa_index": qa_index,
            "subgraph_node_ids": [
                candidate_units.index(unit) for unit in subgraph_units if unit in candidate_units
            ],
            "subgraph_units": subgraph_units,
            "subgraph_size": len(subgraph_units),
            "generation_rank": generation_rank,
            "candidate_pre_rank_score": score_subgraph_indices(
                tuple(candidate_units.index(unit) for unit in subgraph_units),
                candidate_units=candidate_units,
                adjacency=adjacency,
                unit_scores=unit_scores,
            ),
            "gold_available_during_candidate_generation": False,
            "gold_used_during_labeling": attach_gold,
            "candidate_generator": generated["candidate_generator"],
            "candidate_generation_config": generated["candidate_generation_config"],
            "builder_version": BUILDER_VERSION,
        }
        if attach_gold:
            row.update(
                {
                    "gold_explanations": gold_explanations,
                    "gold_units": gold_explanations[0],
                    "gold_context_coverage": gold_context_coverage,
                    **scores,
                    "label": label,
                    "rank_target": float(scores["best_set_f1_to_gold"]),
                }
            )

        rows.append(row)

    # Candidate budgets are enforced by the gold-free beam above. The legacy
    # max_negative_per_example option is not used to prune by labels/F1.
    return rows


def build_answer_only_row(
    source_name: str,
    group_index: int,
    qa_index: int,
    item: Dict,
    qa: Dict,
    split: str,
    max_context_units: int,
) -> Dict:
    sparql_query = str(qa.get("SPARQL Query") or "")
    question = qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or sparql_query
    question = str(question)

    context_axioms = parse_owl_context(item["OWL Context"])
    answer_context_units = relevant_context_axioms(
        context_axioms,
        question=question,
        sparql_query=sparql_query,
        max_context_units=max_context_units,
    )

    return {
        "example_id": (
            f"{source_name}__g{group_index}__q{qa_index}__"
            f"{qa.get('Task ID', '')}__{question}__{sparql_query}"
        ),
        "split": split,
        "question": question,
        "sparql_query": sparql_query,
        "task_type": item.get("Task Type", ""),
        "answer_type": answer_type_for(item, qa),
        "answer": qa.get("Answer"),
        "source_name": source_name,
        "group_index": group_index,
        "qa_index": qa_index,
        "gold_explanations": [],
        "gold_units": [],
        "answer_context_units": answer_context_units,
        "evaluation_scope": "answer_only",
        "has_gold_support": False,
    }


def empty_splits() -> Dict[str, List[Dict]]:
    return {"train": [], "dev": [], "test": []}


def assert_disjoint_output_ids(splits: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
    ids = {split: {str(row["example_id"]) for row in rows} for split, rows in splits.items()}
    overlaps = {
        "train_dev": sorted(ids["train"] & ids["dev"]),
        "train_test": sorted(ids["train"] & ids["test"]),
        "dev_test": sorted(ids["dev"] & ids["test"]),
    }
    for pair, overlap in overlaps.items():
        if overlap:
            raise AssertionError(f"{pair.replace('_', '/')} ID overlap: {overlap[:5]}")
    return overlaps


def build_dataset(args: argparse.Namespace) -> Dict[str, Dict[str, List[Dict]]]:
    output = {}
    answer_only = {}

    for input_path in args.input_json:
        source_name = Path(input_path).stem
        print(f"[LOAD] {source_name}: {input_path}")
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        if args.max_groups:
            data = data[: args.max_groups]
        split_by_group = build_split_map(
            groups=data,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
        )
        source_output = empty_splits()
        source_answer_only = empty_splits()

        for group_index, item in enumerate(data):
            split = split_by_group[group_index]

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
                    candidate_beam_width=args.candidate_beam_width,
                    max_candidate_subgraphs=args.max_candidate_subgraphs,
                )
                source_output[split].extend(rows)

                # This classification is post-generation and does not affect
                # the frozen candidate set.
                if split != "test" and not get_gold_explanations(qa):
                    row = build_answer_only_row(
                        source_name=source_name,
                        group_index=group_index,
                        qa_index=qa_index,
                        item=item,
                        qa=qa,
                        split=split,
                        max_context_units=args.max_context_units,
                    )
                    source_answer_only[split].append(row)

            if (group_index + 1) % 100 == 0:
                print(f"  processed {group_index + 1}/{len(data)} groups")

        output[source_name] = source_output
        answer_only[source_name] = source_answer_only
        assert_disjoint_output_ids(source_output)

    return {"support": output, "answer_only": answer_only}


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
    parser.add_argument(
        "--min-subgraph-size",
        type=int,
        default=1,
        help="Legacy compatibility option; frozen Generator D always starts at size 1.",
    )
    parser.add_argument(
        "--max-subgraph-size",
        type=int,
        default=6,
        help="Legacy compatibility option; frozen Generator D always uses size 6.",
    )
    parser.add_argument(
        "--max-context-units",
        type=int,
        default=40,
        help="Legacy compatibility option; Generator D adapts the full OWL context.",
    )
    parser.add_argument("--max-negative-per-example", type=int, default=200)
    parser.add_argument(
        "--candidate-beam-width",
        type=int,
        default=96,
        help="Legacy compatibility option; Generator D does not use the old beam.",
    )
    parser.add_argument(
        "--max-candidate-subgraphs",
        type=int,
        default=512,
        help="Legacy compatibility option; frozen Generator D always caps at 512.",
    )
    parser.add_argument(
        "--train-ratio",
        "--train-size",
        dest="train_ratio",
        type=float,
        default=0.65,
    )
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
    built = build_dataset(args)
    by_source = built["support"]
    answer_only_by_source = built["answer_only"]
    output_dir = Path(args.output_dir)

    combined = empty_splits()
    combined_answer_only = empty_splits()

    for source_name, splits in by_source.items():
        for split_name, rows in splits.items():
            write_jsonl(
                output_dir / source_name / f"{split_name}_subgraph_retrieval.jsonl",
                rows,
            )
            combined[split_name].extend(rows)
        source_path = next(Path(path) for path in args.input_json if Path(path).stem == source_name)
        split_ids = {
            split_name: list(dict.fromkeys(row["example_id"] for row in rows))
            for split_name, rows in splits.items()
        }
        split_overlaps = assert_disjoint_output_ids(splits)
        write_json(
            output_dir / source_name / "metadata.json",
            {
                "schema_version": "gold_free_ontology_retrieval_v1",
                "builder_version": BUILDER_VERSION,
                "dataset": source_name,
                "source_file": str(source_path),
                "raw_source_sha256": sha256_file(source_path),
                "git": git_provenance(Path(__file__).resolve().parents[1]),
                "seed": RANDOM_SEED,
                "example_ids": split_ids,
                "split_overlaps": split_overlaps,
                "candidate_generation_config": frozen_generator_d_config(),
                "candidate_generation_lineage": frozen_generator_d_lineage(),
                "gold_available_during_candidate_generation": False,
                "gold_used_during_labeling": {"train": True, "dev": True, "test": False},
                "candidate_composer": GENERATOR_D_VERSION,
            },
        )

    for source_name, splits in answer_only_by_source.items():
        for split_name, rows in splits.items():
            write_jsonl(
                output_dir / source_name / f"{split_name}_answer_only_no_explanation.jsonl",
                rows,
            )
            combined_answer_only[split_name].extend(rows)

    if args.combined:
        for split_name, rows in combined.items():
            write_jsonl(output_dir / f"{split_name}_subgraph_retrieval.jsonl", rows)
        for split_name, rows in combined_answer_only.items():
            write_jsonl(output_dir / f"{split_name}_answer_only_no_explanation.jsonl", rows)


if __name__ == "__main__":
    main()
