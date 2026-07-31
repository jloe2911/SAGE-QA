import argparse
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from rdflib import Graph, URIRef
from rdflib.namespace import OWL as OWL_NS
from rdflib.namespace import RDF as RDF_NS
from rdflib.namespace import RDFS as RDFS_NS

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
    if len(parts) == 3 and parts[1] in {"SubPropertyOf", "rdfs:subPropertyOf"}:
        return f"SubObjectPropertyOf({parts[0]},{parts[2]})"
    if len(parts) == 3 and parts[1] in {"inverseOf", "owl:inverseOf"}:
        first, second = sorted((parts[0], parts[2]))
        return f"InverseObjectProperties({first},{second})"
    if len(parts) == 3 and parts[1] in {
        "equivalentProperty",
        "owl:equivalentProperty",
    }:
        first, second = sorted((parts[0], parts[2]))
        return f"EquivalentObjectProperties({first},{second})"
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
        "Could not parse OWL Context as Turtle or RDF/XML. "
        f"Parser errors: {'; '.join(errors)}"
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
        elif predicate == OWL_NS.inverseOf:
            first, second = sorted((subject, obj))
            add(f"InverseObjectProperties({first},{second})")
        elif predicate == OWL_NS.equivalentProperty:
            first, second = sorted((subject, obj))
            add(f"EquivalentObjectProperties({first},{second})")
        elif predicate == RDFS_NS.domain:
            add(f"{subject} domain {obj}")
        elif predicate == RDFS_NS.range:
            add(f"{subject} range {obj}")
        else:
            pred = local_name(str(predicate_ref))
            if pred not in {"comment", "first", "rest", "propertyChainAxiom"}:
                add(f"{subject} {pred} {obj}")

    return axioms


_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "from",
    "in",
    "is",
    "of",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
}


def lexical_tokens(text: str) -> Set[str]:
    normalized = str(text or "").replace("_", " ").replace("::", " ")
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", normalized)
        if token.casefold() not in _QUERY_STOPWORDS and token.casefold() != "kg"
    }


def parse_kg_unit(unit: str) -> Tuple[str, str, str] | None:
    parts = str(unit).split("::", 3)
    if len(parts) == 4 and parts[0] == "KG":
        return parts[1].strip(), parts[2].strip(), parts[3].strip()
    return None


def unit_signature(unit: str) -> Tuple[Set[str], Set[str]]:
    kg_triple = parse_kg_unit(unit)
    if kg_triple is not None:
        subject, predicate, obj = kg_triple
        return {subject.casefold(), obj.casefold()}, {predicate.casefold()}

    schema_match = re.match(r"^(\S+)\s+(domain|range)\s+(\S+)$", unit)
    if schema_match:
        return {schema_match.group(3)}, {schema_match.group(1)}

    parsed = parse_axiom(unit)
    entities = set(parsed.entities())
    properties = set(parsed.properties())

    if parsed.axiom_type == "unknown":
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_:-]*", unit))
        entities |= {t for t in tokens if t and t[0].islower()}
        properties |= {t for t in tokens if t in unit}

    return entities, properties


def signature_values_overlap(left: Set[str], right: Set[str]) -> bool:
    normalized_left = {value.casefold() for value in left}
    normalized_right = {value.casefold() for value in right}
    return bool(normalized_left & normalized_right)


def unit_question_overlap(unit: str, question: str) -> float:
    question_tokens = lexical_tokens(question)
    if not question_tokens:
        return 0.0
    return len(question_tokens & lexical_tokens(unit)) / len(question_tokens)


def edge_between_units(left: str, right: str) -> bool:
    left_entities, left_properties = unit_signature(left)
    right_entities, right_properties = unit_signature(right)
    if parse_kg_unit(left) is None and parse_kg_unit(right) is None:
        return bool(
            (left_entities & right_entities) or (left_properties & right_properties)
        )
    return bool(
        signature_values_overlap(left_entities, right_entities)
        or signature_values_overlap(left_properties, right_properties)
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
    question: str = "",
) -> float:
    entities, properties = unit_signature(unit)
    parsed = parse_axiom(unit)
    kg_triple = parse_kg_unit(unit)

    score = 0.0
    if kg_triple is not None:
        score += 2.0 if signature_values_overlap(entities, query_entities) else 0.0
        score += 1.5 if signature_values_overlap(properties, query_properties) else 0.0
        score += 3.0 * unit_question_overlap(unit, question)
    else:
        # Natural-language overlap is shared by FamilyOWL and text benchmarks;
        # SPARQL-derived signatures are intentionally empty in NL-only runs.
        score += 2.0 * len(entities & query_entities)
        score += 1.5 * len(properties & query_properties)
        score += 3.0 * unit_question_overlap(unit, question)
    score += 0.15 if parsed.axiom_type == "fact" or kg_triple is not None else 0.0
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
    kinds = {
        "fact" if parse_kg_unit(unit) is not None else parsed_unit.axiom_type
        for unit, parsed_unit in zip(units, parsed)
    }

    fact_rule_mix = 1.0 if ("fact" in kinds and "rule" in kinds) else 0.0
    avg_relevance = sum(unit_scores[i] for i in indices) / max(len(indices), 1)

    return avg_relevance + 0.20 * fact_rule_mix - 0.04 * max(0, len(indices) - 3)


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
            question=question,
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

    # Reserve shortest reasoning chains between the most query-aligned units.
    # This is inference-safe (question/SPARQL + graph structure only) and
    # prevents beam pruning from losing a low-scoring bridge rule.
    path_candidate_indices = set()
    # NL-only class/rule units often share just one informative token with the
    # question (for example, ``Man``). Keep them eligible as path endpoints so
    # a low-scoring inverse/subproperty rule between a fact and the queried
    # class is not pruned from the beam.
    path_endpoints = [index for index in seed_indices if unit_scores[index] >= 0.5][:16]
    for start, goal in itertools.combinations(path_endpoints, 2):
        queue = [start]
        parents = {start: None}
        for current in queue:
            if current == goal:
                break
            for neighbor in sorted(adjacency[current]):
                if neighbor in parents:
                    continue
                parents[neighbor] = current
                queue.append(neighbor)

        if goal not in parents:
            continue
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parents[current]
        combo = tuple(sorted(path))
        if min_subgraph_size <= len(combo) <= max_size:
            path_candidate_indices.add(combo)

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

        ranked_expansions = sorted(
            expansions.items(),
            key=lambda item: (-item[1], item[0]),
        )

        # Keep the frontier diverse enough to retain low-scoring bridge rules.
        # Pure global top-k pruning favors immediately relevant facts and can
        # discard the inverse/subproperty rule needed to reach an exact proof.
        next_frontier = []
        next_seen = set()

        expansion_patterns = defaultdict(list)
        for combo, score in ranked_expansions:
            fact_count = sum(
                parse_axiom(candidate_units[index]).axiom_type == "fact"
                for index in combo
            )
            expansion_patterns[fact_count].append((combo, score))
        pattern_quota = max(1, (beam_width // 2) // max(len(expansion_patterns), 1))
        for fact_count in sorted(expansion_patterns):
            for combo, _ in expansion_patterns[fact_count][:pattern_quota]:
                if combo not in next_seen:
                    next_frontier.append(combo)
                    next_seen.add(combo)

        for anchor in seed_indices:
            anchored = next(
                (combo for combo, _ in ranked_expansions if anchor in combo),
                None,
            )
            if anchored is not None and anchored not in next_seen:
                next_frontier.append(anchored)
                next_seen.add(anchored)
                if len(next_frontier) >= beam_width:
                    break

        if len(next_frontier) < beam_width:
            for combo, _ in ranked_expansions:
                if combo in next_seen:
                    continue
                next_frontier.append(combo)
                next_seen.add(combo)
                if len(next_frontier) >= beam_width:
                    break

        frontier = next_frontier

    # Preserve candidates across proof sizes. A single global top-k allows
    # dense large subgraphs to crowd every compact proof out of the candidate
    # set, even when the compact proof is exact.
    by_pattern = defaultdict(list)
    for scored_candidate in scored_candidates:
        combo = scored_candidate[1]
        fact_count = sum(
            parse_axiom(candidate_units[index]).axiom_type == "fact" for index in combo
        )
        by_pattern[(len(combo), fact_count)].append(scored_candidate)

    selected = sorted(
        (
            score_subgraph_indices(
                combo,
                candidate_units=candidate_units,
                adjacency=adjacency,
                unit_scores=unit_scores,
            ),
            combo,
        )
        for combo in path_candidate_indices
    )
    selected.sort(key=lambda item: (-item[0], item[1]))
    selected = selected[:max_candidate_subgraphs]
    selected_combos = {combo for _, combo in selected}
    pattern_quota = max(1, max_candidate_subgraphs // max(len(by_pattern), 1))
    for pattern in sorted(by_pattern):
        for scored_candidate in sorted(
            by_pattern[pattern], key=lambda item: (-item[0], item[1])
        )[:pattern_quota]:
            selected.append(scored_candidate)
            selected_combos.add(scored_candidate[1])

    if len(selected) < max_candidate_subgraphs:
        for scored_candidate in sorted(
            scored_candidates, key=lambda item: (-item[0], item[1])
        ):
            if scored_candidate[1] in selected_combos:
                continue
            selected.append(scored_candidate)
            selected_combos.add(scored_candidate[1])
            if len(selected) >= max_candidate_subgraphs:
                break

    beam_candidates = {
        tuple(candidate_units[i] for i in combo)
        for _, combo in selected[:max_candidate_subgraphs]
    }
    return beam_candidates


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


def materialize_retrieval_rows(
    candidate_units: List[str],
    candidate_subgraphs: Iterable[Iterable[str]],
    gold_explanations: List[List[str]],
    base_row: Dict,
    max_negative_per_example: int = -1,
    shuffle_rows: bool = False,
) -> List[Dict]:
    """Create the common GNN retrieval rows from candidate KG subgraphs."""
    unit_to_index = {unit: index for index, unit in enumerate(candidate_units)}
    positives = []
    negatives = []

    for candidate in sorted(
        (tuple(subgraph) for subgraph in candidate_subgraphs),
        key=lambda subgraph: (len(subgraph), subgraph),
    ):
        subgraph_units = list(candidate)
        scores = set_scores(subgraph_units, gold_explanations)
        label = int(
            scores["exact_match_any_gold"] or scores["contains_any_gold_explanation"]
        )
        row = {
            **base_row,
            "subgraph_node_ids": [
                unit_to_index[unit] for unit in subgraph_units if unit in unit_to_index
            ],
            "subgraph_units": subgraph_units,
            "subgraph_size": len(subgraph_units),
            **scores,
            "label": label,
            "rank_target": float(scores["best_set_f1_to_gold"]),
        }
        (positives if label else negatives).append(row)

    negatives.sort(key=lambda row: row["best_set_f1_to_gold"], reverse=True)
    if max_negative_per_example >= 0:
        negatives = negatives[:max_negative_per_example]

    rows = positives + negatives
    if shuffle_rows:
        random.shuffle(rows)
    return rows


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
        score = (
            2 * len(entities & query_entities)
            + len(properties & query_properties)
            + 3.0 * unit_question_overlap(axiom, question)
        )
        if score > 0:
            scored.append((score, axiom))

    scored.sort(key=lambda item: (-item[0], item[1]))
    seed_budget = max(1, max_context_units // 2)
    selected = [axiom for _, axiom in scored[:seed_budget]]
    seen = set(selected)

    bridge_rules = sorted(
        axiom
        for axiom in axioms
        if axiom not in seen
        and parse_axiom(axiom).axiom_type == "rule"
        and any(edge_between_units(axiom, seed) for seed in selected)
    )
    for axiom in bridge_rules:
        if len(selected) >= max_context_units:
            break
        selected.append(axiom)
        seen.add(axiom)

    # Add graph neighbors of query-aligned seeds. This recovers intermediate
    # rules/facts without consulting a gold explanation.
    while len(selected) < max_context_units:
        neighbors = []
        for axiom in axioms:
            if axiom in seen:
                continue
            links = sum(edge_between_units(axiom, chosen) for chosen in selected)
            if links:
                is_rule = int(parse_axiom(axiom).axiom_type == "rule")
                neighbors.append((links, is_rule, axiom))
        if not neighbors:
            break
        neighbors.sort(key=lambda item: (-item[0], -item[1], item[2]))
        _, _, chosen = neighbors[0]
        selected.append(chosen)
        seen.add(chosen)

    return selected


def answer_type_for(item: Dict, qa: Dict | None = None) -> str:
    qa = qa or {}
    return str(qa.get("Answer Type") or item.get("Answer Type") or "").strip()


def build_split_map(
    groups: List[Dict], train_ratio: float, dev_ratio: float
) -> Dict[int, str]:
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
    reference_sparql_query = str(qa.get("SPARQL Query") or "")
    question = (
        qa.get("NL Question")
        or qa.get("ABS Question")
        or qa.get("Task ID")
        or reference_sparql_query
    )
    question = str(question)

    gold_explanations = get_gold_explanations(qa)

    if not gold_explanations:
        return []

    context_axioms = parse_owl_context(item["OWL Context"])
    # The ontology is already a KG. Candidate units must come only from that
    # inference-time context; gold explanations remain supervision.
    candidate_units = relevant_context_axioms(
        context_axioms,
        question=question,
        sparql_query="",
        max_context_units=max_context_units,
    )
    if not candidate_units:
        return []

    effective_max_subgraph_size = (
        len(candidate_units) if max_subgraph_size <= 0 else max_subgraph_size
    )

    candidate_subgraphs = beam_connected_subgraphs(
        candidate_units=candidate_units,
        question=question,
        sparql_query="",
        min_subgraph_size=min_subgraph_size,
        max_subgraph_size=effective_max_subgraph_size,
        beam_width=candidate_beam_width,
        max_candidate_subgraphs=max_candidate_subgraphs,
    )
    candidate_set = set(candidate_units)
    gold_context_coverage = max(
        (
            len(set(explanation) & candidate_set) / max(len(set(explanation)), 1)
            for explanation in gold_explanations
        ),
        default=0.0,
    )

    return materialize_retrieval_rows(
        candidate_units=candidate_units,
        candidate_subgraphs=candidate_subgraphs,
        gold_explanations=gold_explanations,
        base_row={
            "example_id": (
                f"{source_name}__g{group_index}__q{qa_index}__"
                f"{qa.get('Task ID', '')}__{question}"
            ),
            "split": split,
            "question": question,
            # Keep the annotation for traceability, but never expose it to
            # retrieval/model features.
            "sparql_query": "",
            "reference_sparql_query": reference_sparql_query,
            "task_type": item.get("Task Type", ""),
            "answer_type": answer_type_for(item, qa),
            "answer": qa.get("Answer"),
            "evidence_unit_type": "ontology_axiom",
            "source_name": source_name,
            "group_index": group_index,
            "qa_index": qa_index,
            "gold_explanations": gold_explanations,
            "gold_units": gold_explanations[0],
            "gold_context_coverage": gold_context_coverage,
        },
        max_negative_per_example=max_negative_per_example,
        shuffle_rows=True,
    )


def build_answer_only_row(
    source_name: str,
    group_index: int,
    qa_index: int,
    item: Dict,
    qa: Dict,
    split: str,
    max_context_units: int,
) -> Dict:
    reference_sparql_query = str(qa.get("SPARQL Query") or "")
    question = (
        qa.get("NL Question")
        or qa.get("ABS Question")
        or qa.get("Task ID")
        or reference_sparql_query
    )
    question = str(question)

    context_axioms = parse_owl_context(item["OWL Context"])
    answer_context_units = relevant_context_axioms(
        context_axioms,
        question=question,
        sparql_query="",
        max_context_units=max_context_units,
    )

    return {
        "example_id": (
            f"{source_name}__g{group_index}__q{qa_index}__"
            f"{qa.get('Task ID', '')}__{question}"
        ),
        "split": split,
        "question": question,
        "sparql_query": "",
        "reference_sparql_query": reference_sparql_query,
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


def load_selection_manifest(path: Path | None) -> Dict[str, Set[str]] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected = {}
    for split in ("train", "dev", "test"):
        ids = manifest.get("splits", {}).get(split, {}).get("selected_example_ids")
        if not isinstance(ids, list):
            raise ValueError(
                f"Selection manifest {path} has no selected IDs for {split}"
            )
        selected[split] = {str(example_id) for example_id in ids}
    return selected


def build_dataset(args: argparse.Namespace) -> Dict[str, Dict[str, List[Dict]]]:
    output = {}
    answer_only = {}
    manifest_selection = load_selection_manifest(args.selection_manifest)

    for input_path in args.input_json:
        source_name = Path(input_path).stem
        print(f"[LOAD] {source_name}: {input_path}")
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        split_by_group = build_split_map(
            groups=data,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
        )
        source_selection = None
        selected_pairs = None
        if manifest_selection is not None:
            source_selection = {
                split: {
                    example_id
                    for example_id in ids
                    if example_id.startswith(f"{source_name}__")
                }
                for split, ids in manifest_selection.items()
            }
            selected_ids = set().union(*source_selection.values())
            selected_pairs = {}
            for selected_split, ids in source_selection.items():
                for example_id in ids:
                    match = re.search(r"__g(\d+)__q(\d+)__", example_id)
                    if match is None:
                        raise ValueError(
                            f"Could not parse group/QA indices from {example_id!r}"
                        )
                    pair = (int(match.group(1)), int(match.group(2)))
                    previous = selected_pairs.setdefault(pair, selected_split)
                    if previous != selected_split:
                        raise ValueError(
                            f"Group/QA pair {pair} is assigned to both "
                            f"{previous} and {selected_split}"
                        )
            group_indices = sorted({group_index for group_index, _ in selected_pairs})
            if len(group_indices) == 0 and selected_ids:
                raise ValueError(
                    f"Could not parse group indices from selection manifest for "
                    f"{source_name}"
                )
        elif args.group_indices:
            group_indices = list(dict.fromkeys(args.group_indices))
            invalid = [index for index in group_indices if not 0 <= index < len(data)]
            if invalid:
                raise ValueError(
                    f"Group indices out of range for {source_name}: {invalid}"
                )
        else:
            group_indices = list(range(len(data)))
        if args.max_groups:
            group_indices = group_indices[: args.max_groups]

        source_output = empty_splits()
        source_answer_only = empty_splits()

        for processed_count, group_index in enumerate(group_indices, start=1):
            item = data[group_index]
            split = split_by_group[group_index]

            for qa_index, qa in enumerate(item.get("QAs", [])):
                question = str(
                    qa.get("NL Question")
                    or qa.get("ABS Question")
                    or qa.get("Task ID")
                    or qa.get("SPARQL Query")
                    or ""
                )
                example_id = (
                    f"{source_name}__g{group_index}__q{qa_index}__"
                    f"{qa.get('Task ID', '')}__{question}"
                )
                if selected_pairs is not None:
                    expected_split = selected_pairs.get((group_index, qa_index))
                    if expected_split is None:
                        continue
                    if expected_split != split:
                        raise ValueError(
                            f"Manifest assigns {example_id!r} to {expected_split}, "
                            f"but the raw-data split map assigns it to {split}"
                        )
                has_gold_support = bool(get_gold_explanations(qa))

                if not has_gold_support:
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

            if processed_count % 100 == 0:
                print(f"  processed {processed_count}/{len(group_indices)} groups")

        output[source_name] = source_output
        answer_only[source_name] = source_answer_only
        if selected_pairs is not None:
            built_pairs = {
                (int(row["group_index"]), int(row["qa_index"]))
                for split_rows in source_output.values()
                for row in split_rows
            }
            missing = set(selected_pairs) - built_pairs
            if missing:
                raise ValueError(
                    f"Failed to rebuild {len(missing)} selected examples for "
                    f"{source_name}, including group/QA pairs: {sorted(missing)[:3]}"
                )

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
    parser.add_argument("--min-subgraph-size", type=int, default=1)
    parser.add_argument(
        "--max-subgraph-size",
        type=int,
        default=7,
        help=(
            "Maximum support size to search. FamilyOWL gold explanations have "
            "at most 7 units. Use 0 only for an intentional unbounded search."
        ),
    )
    parser.add_argument("--max-context-units", type=int, default=40)
    parser.add_argument("--max-negative-per-example", type=int, default=200)
    parser.add_argument(
        "--candidate-beam-width",
        type=int,
        default=96,
        help=(
            "Beam width for connected support generation. Larger values explore "
            "more 4+ axiom chains but keep enumeration bounded."
        ),
    )
    parser.add_argument(
        "--max-candidate-subgraphs",
        type=int,
        default=320,
        help="Maximum non-gold candidate subgraphs generated per QA example.",
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
    parser.add_argument(
        "--group-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional fixed raw group indices for reproducible diagnostics. "
            "Splits are still computed from the full dataset."
        ),
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help=(
            "A gnn_dev_sample_v1 manifest whose exact example IDs override "
            "--group-indices/--max-groups."
        ),
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

    for source_name, splits in answer_only_by_source.items():
        for split_name, rows in splits.items():
            write_jsonl(
                output_dir
                / source_name
                / f"{split_name}_answer_only_no_explanation.jsonl",
                rows,
            )
            combined_answer_only[split_name].extend(rows)

    input_by_source = {Path(path).stem: str(Path(path)) for path in args.input_json}
    for source_name, splits in by_source.items():
        answer_splits = answer_only_by_source[source_name]
        metadata = {
            "schema_version": "unified_kg_reasoning_v3",
            "evidence_unit_type": "ontology_axiom",
            "retrieval_inference_inputs": ["question", "ontology_context"],
            "reference_only_fields": ["reference_sparql_query", "gold_explanations"],
            "source_json": input_by_source.get(source_name, ""),
            "candidate_composer": "beam_connected_subgraphs",
            "row_materializer": "materialize_retrieval_rows",
            "min_subgraph_size": args.min_subgraph_size,
            "max_subgraph_size": args.max_subgraph_size,
            "max_context_units": args.max_context_units,
            "max_negative_per_example": args.max_negative_per_example,
            "candidate_beam_width": args.candidate_beam_width,
            "max_candidate_subgraphs": args.max_candidate_subgraphs,
            "train_ratio": args.train_ratio,
            "dev_ratio": args.dev_ratio,
            "max_groups": args.max_groups,
            "group_indices": args.group_indices,
            "selection_manifest": (
                str(args.selection_manifest) if args.selection_manifest else None
            ),
            "train_rows": len(splits["train"]),
            "dev_rows": len(splits["dev"]),
            "test_rows": len(splits["test"]),
            "train_examples": len({row["example_id"] for row in splits["train"]}),
            "dev_examples": len({row["example_id"] for row in splits["dev"]}),
            "test_examples": len({row["example_id"] for row in splits["test"]}),
            "train_answer_only": len(answer_splits["train"]),
            "dev_answer_only": len(answer_splits["dev"]),
            "test_answer_only": len(answer_splits["test"]),
        }
        metadata_path = output_dir / source_name / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        print(f"[WRITE] {metadata_path}")

    if args.combined:
        for split_name, rows in combined.items():
            write_jsonl(output_dir / f"{split_name}_subgraph_retrieval.jsonl", rows)
        for split_name, rows in combined_answer_only.items():
            write_jsonl(
                output_dir / f"{split_name}_answer_only_no_explanation.jsonl", rows
            )


if __name__ == "__main__":
    main()
