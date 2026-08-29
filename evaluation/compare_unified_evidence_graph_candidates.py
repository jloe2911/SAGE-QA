"""Compare the current clean generator with one unified evidence-graph search.

This is a development-only architectural experiment.  Candidate construction
receives only question/context-visible inputs; benchmark supports are joined
after both candidate sets are frozen.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.build_subgraph_training_data import (  # noqa: E402
    extract_query_signature,
    generate_ontology_candidates,
    get_gold_explanations,
    parse_owl_context,
    score_unit_for_query,
    unit_signature,
)
from data_processing.build_2wiki_subgraph_dataset import (  # noqa: E402
    flatten_context as flatten_2wiki,
    generate_candidates as generate_2wiki,
    get_gold_support_units as gold_2wiki,
)
from data_processing.build_hotpot_subgraph_dataset import (  # noqa: E402
    flatten_context as flatten_hotpot,
    generate_candidates as generate_hotpot,
    get_gold_support_units as gold_hotpot,
)
from data_processing.evidence_graph_candidates import (  # noqa: E402
    EvidenceGraph,
    EvidenceUnit,
    ProgressiveExpansionConfig,
    progressive_connected_supports,
)
from data_processing.retrieval_contracts import (  # noqa: E402
    BUILDER_VERSION,
    clean_text_retrieval_input,
    git_provenance,
    sha256_file,
    write_json,
)
from data_processing.text_kg_constructor import KGConstructionConfig  # noqa: E402
from evaluation.validate_gold_free_candidate_pools import (  # noqa: E402
    ONTO,
    ONTO_BASE,
    ROOT,
    TEXT,
    TEXT_BASE,
    onto_rows,
    text_rows,
)

SEED = 42
TEXT_SAMPLE_SIZE = 200
UNIFIED_V1_CONFIG = ProgressiveExpansionConfig(
    max_support_size=6,
    max_candidates=512,
    max_seeds=32,
    max_first_hops_per_seed=6,
    max_neighbors_per_state=12,
    per_branch_width=2,
    protect_query_anchors=False,
)
PROTECTED_CONFIG = ProgressiveExpansionConfig(
    max_support_size=6,
    max_candidates=512,
    max_seeds=32,
    max_first_hops_per_seed=6,
    max_neighbors_per_state=12,
    per_branch_width=2,
    protect_query_anchors=True,
)
DISPLAY_NAMES = {
    "2WikiMultiHopQA": "2Wiki",
    "HotpotQA": "HotpotQA",
    "Family_1hop": "Family 1-hop",
    "Family_2hop": "Family 2-hop",
    "Pizza_100_1hop": "Pizza100 1-hop",
    "Pizza_100_2hop": "Pizza100 2-hop",
    "Pizza_250_1hop": "Pizza250 1-hop",
    "Pizza_250_2hop": "Pizza250 2-hop",
    "OWL2Bench_1hop": "OWL2Bench 1-hop",
    "OWL2Bench_2hop": "OWL2Bench 2-hop",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CAPITALIZED_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9'-]*)(?:\s+[A-Z][A-Za-z0-9'-]*)*")
STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "other",
    "over",
    "same",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "were",
    "whose",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(str(text))
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def _canonical(candidates: Iterable[Iterable[str]]) -> list[tuple[str, ...]]:
    return sorted(
        {tuple(sorted(set(candidate))) for candidate in candidates if candidate},
        key=lambda candidate: (len(candidate), candidate),
    )


def _digest(candidates: Sequence[Sequence[str]]) -> str:
    payload = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_graph(
    units: Sequence[EvidenceUnit],
    adjacency: Sequence[set[int]],
    scores: Sequence[float],
    anchors: Sequence[bool],
) -> EvidenceGraph:
    return EvidenceGraph(
        units=tuple(units),
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        query_scores=tuple(float(score) for score in scores),
        query_anchors=tuple(bool(anchor) for anchor in anchors),
    )


def build_text_evidence_graph(
    sentence_records: Sequence[Mapping[str, Any]], question: str
) -> EvidenceGraph:
    """One text adapter used unchanged for both 2Wiki and HotpotQA."""
    records_by_unit = {str(record["unit"]): record for record in sentence_records}
    ordered = [records_by_unit[key] for key in sorted(records_by_unit)]
    units = [
        EvidenceUnit(
            unit_id=str(record["unit"]),
            content=str(record.get("sentence", "")),
            metadata={
                "title": str(record.get("title", "")),
                "sentence_index": int(record.get("sent_idx", -1)),
            },
        )
        for record in ordered
    ]
    adjacency = [set() for _ in units]
    question_tokens = _tokens(question)
    title_tokens = [_tokens(str(unit.metadata["title"])) for unit in units]
    sentence_tokens = [_tokens(unit.content) for unit in units]
    named_entities = [
        {
            " ".join(token.lower() for token in TOKEN_RE.findall(match.group(0)))
            for match in CAPITALIZED_RE.finditer(unit.content)
            if TOKEN_RE.findall(match.group(0))
        }
        for unit in units
    ]

    def connect(left: int, right: int) -> None:
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)

    by_title: dict[str, list[int]] = collections.defaultdict(list)
    token_index: dict[str, list[int]] = collections.defaultdict(list)
    entity_index: dict[str, list[int]] = collections.defaultdict(list)
    bridge_index: dict[str, list[int]] = collections.defaultdict(list)
    for index, unit in enumerate(units):
        by_title[str(unit.metadata["title"]).casefold()].append(index)
        for token in title_tokens[index]:
            token_index[f"title:{token}"].append(index)
        for entity in named_entities[index]:
            entity_index[entity].append(index)
        for token in sentence_tokens[index]:
            if len(token) >= 4:
                bridge_index[token].append(index)

    # Local document continuity plus cross-document title/entity bridges.
    for indices in by_title.values():
        indices.sort(key=lambda index: (int(units[index].metadata["sentence_index"]), index))
        for left, right in zip(indices, indices[1:]):
            connect(left, right)
    for index, tokens in enumerate(sentence_tokens):
        for token in tokens:
            for title_index in token_index.get(f"title:{token}", ()):
                connect(index, title_index)
    for indices in entity_index.values():
        if 1 < len(indices) <= 12:
            for position, left in enumerate(indices):
                for right in indices[position + 1 :]:
                    connect(left, right)
    for indices in bridge_index.values():
        if 1 < len(indices) <= 8:
            for position, left in enumerate(indices):
                for right in indices[position + 1 :]:
                    connect(left, right)

    scores = []
    anchors = []
    denominator = max(len(question_tokens), 1)
    for index in range(len(units)):
        content_overlap = (
            len(question_tokens & (sentence_tokens[index] | title_tokens[index])) / denominator
        )
        title_overlap = len(question_tokens & title_tokens[index]) / denominator
        scores.append(content_overlap + 0.25 * title_overlap + 0.01 * min(len(adjacency[index]), 8))
        anchors.append(bool(question_tokens & (sentence_tokens[index] | title_tokens[index])))
    return _make_graph(units, adjacency, scores, anchors)


def build_ontology_evidence_graph(
    context_axioms: Sequence[str], question: str, sparql_query: str
) -> EvidenceGraph:
    """One ontology adapter used unchanged for all eight ontology cohorts."""
    ordered_axioms = sorted(set(context_axioms))
    units = [EvidenceUnit(unit_id=axiom, content=axiom) for axiom in ordered_axioms]
    adjacency = [set() for _ in units]
    signatures = [unit_signature(unit.unit_id) for unit in units]
    inverted: dict[str, list[int]] = collections.defaultdict(list)
    for index, (entities, properties) in enumerate(signatures):
        for term in entities | properties:
            inverted[str(term).casefold()].append(index)
    for indices in inverted.values():
        # Very common vocabulary terms produce uninformative cliques.  Their
        # specific schema/fact terms remain available through other keys.
        if len(indices) > 64:
            continue
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)

    signature = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities = set(signature.get("query_entities", []))
    query_properties = set(signature.get("query_properties", []))
    scores = [
        score_unit_for_query(
            unit.unit_id,
            query_entities=query_entities,
            query_properties=query_properties,
            degree=len(adjacency[index]),
        )
        for index, unit in enumerate(units)
    ]
    anchors = [
        bool((entities & query_entities) or (properties & query_properties))
        for entities, properties in signatures
    ]
    return _make_graph(units, adjacency, scores, anchors)


def _maximal_masks(masks: Iterable[int]) -> tuple[int, ...]:
    unique = sorted(set(masks) - {0}, key=lambda mask: (-mask.bit_count(), mask))
    kept: list[int] = []
    for mask in unique:
        if not any(mask | other == other for other in kept):
            kept.append(mask)
    return tuple(kept)


def _minimum_cover(masks: Sequence[int], full_mask: int, limit: int = 3) -> int | None:
    useful = _maximal_masks(masks)
    if not useful or not full_mask:
        return None
    reached = {0}
    frontier = {0}
    for depth in range(1, limit + 1):
        frontier = {state | mask for state in frontier for mask in useful} - reached
        if full_mask in frontier:
            return depth
        reached.update(frontier)
        if not frontier:
            break
    return None


def post_gold_diagnostics(
    candidates: Sequence[Sequence[str]],
    gold_explanations: Sequence[Sequence[str]],
    *,
    reached_units: Iterable[str],
    max_support_size: int,
) -> dict[str, Any]:
    candidate_sets = [set(candidate) for candidate in candidates]
    gold_sets = [tuple(dict.fromkeys(gold)) for gold in gold_explanations if gold]
    if not gold_sets:
        raise ValueError("At least one gold support is required after generation")
    union = set().union(*candidate_sets) if candidate_sets else set()
    reached = set(reached_units)
    minimums = []
    recalls = []
    for gold in gold_sets:
        bits = {unit: 1 << index for index, unit in enumerate(gold)}
        masks = [sum(bits.get(unit, 0) for unit in candidate) for candidate in candidate_sets]
        minimums.append(_minimum_cover(masks, (1 << len(bits)) - 1))
        recalls.append(len(union & set(gold)) / len(set(gold)))
    possible = [minimum for minimum in minimums if minimum is not None]
    minimum = min(possible, default=None)
    complete = any(set(gold) <= candidate for gold in gold_sets for candidate in candidate_sets)
    union_complete = any(set(gold) <= union for gold in gold_sets)

    if complete:
        failure = None
    else:
        reachable = [gold for gold in gold_sets if set(gold) <= reached]
        if not reachable:
            failure = "1_required_evidence_unit_never_reached_by_graph_expansion"
        elif all(len(set(gold)) > max_support_size for gold in reachable):
            failure = "3_support_exceeds_configured_maximum_size"
        else:
            failure = "2_required_units_reached_but_not_composed_into_coverable_support"
    return {
        "oracle_support_minimum_candidates": minimum,
        "oracle_support_coverage_at_1": complete,
        "oracle_support_coverage_at_2": minimum is not None and minimum <= 2,
        "oracle_support_coverage_at_3": minimum is not None and minimum <= 3,
        "all_candidate_union_coverage": union_complete,
        "evidence_unit_recall": max(recalls, default=0.0),
        "failure_class": failure,
    }


def _method_summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    total = len(rows)
    counts = [int(row[f"{prefix}_candidate_count"]) for row in rows]
    sizes = [size for row in rows for size in row[f"{prefix}_candidate_sizes"]]

    def rate(field: str) -> float:
        return sum(bool(row[f"{prefix}_{field}"]) for row in rows) / max(total, 1)

    failures = collections.Counter(
        row[f"{prefix}_failure_class"] for row in rows if row.get(f"{prefix}_failure_class")
    )
    return {
        "oracle_support_coverage_at_1": rate("oracle_support_coverage_at_1"),
        "oracle_support_coverage_at_2": rate("oracle_support_coverage_at_2"),
        "oracle_support_coverage_at_3": rate("oracle_support_coverage_at_3"),
        "all_candidate_union_coverage": rate("all_candidate_union_coverage"),
        "evidence_unit_recall": statistics.fmean(
            float(row[f"{prefix}_evidence_unit_recall"]) for row in rows
        )
        if rows
        else 0.0,
        "average_candidate_count": statistics.fmean(counts) if counts else 0.0,
        "candidate_count_range": [min(counts, default=0), max(counts, default=0)],
        "candidate_size_distribution": {
            str(size): count for size, count in sorted(collections.Counter(sizes).items())
        },
        "zero_candidate_rate": sum(count == 0 for count in counts) / max(total, 1),
        "runtime_seconds": sum(float(row[f"{prefix}_runtime_seconds"]) for row in rows),
        "runtime_seconds_per_example": sum(float(row[f"{prefix}_runtime_seconds"]) for row in rows)
        / max(total, 1),
        "failure_classes": dict(sorted(failures.items())),
    }


def _detail(
    example_id: str,
    current: Sequence[Sequence[str]],
    unified: Sequence[Sequence[str]],
    protected: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
    current_reached: Iterable[str],
    unified_reached: Iterable[str],
    protected_reached: Iterable[str],
    current_max_size: int,
    current_seconds: float,
    unified_seconds: float,
    protected_seconds: float,
    seeds: Sequence[str],
    generated_states: int,
    protected_seeds: Sequence[str],
    protected_generated_states: int,
    protected_component_count: int,
) -> dict[str, Any]:
    current_diag = post_gold_diagnostics(
        current, gold, reached_units=current_reached, max_support_size=current_max_size
    )
    unified_diag = post_gold_diagnostics(
        unified,
        gold,
        reached_units=unified_reached,
        max_support_size=UNIFIED_V1_CONFIG.max_support_size,
    )
    protected_diag = post_gold_diagnostics(
        protected,
        gold,
        reached_units=protected_reached,
        max_support_size=PROTECTED_CONFIG.max_support_size,
    )
    row: dict[str, Any] = {
        "example_id": example_id,
        "current_candidate_count": len(current),
        "current_candidate_sizes": [len(candidate) for candidate in current],
        "current_candidate_digest": _digest(current),
        "current_runtime_seconds": current_seconds,
        "unified_candidate_count": len(unified),
        "unified_candidate_sizes": [len(candidate) for candidate in unified],
        "unified_candidate_digest": _digest(unified),
        "unified_runtime_seconds": unified_seconds,
        "unified_seed_count": len(seeds),
        "unified_generated_state_count": generated_states,
        "protected_candidate_count": len(protected),
        "protected_candidate_sizes": [len(candidate) for candidate in protected],
        "protected_candidate_digest": _digest(protected),
        "protected_runtime_seconds": protected_seconds,
        "protected_seed_count": len(protected_seeds),
        "protected_generated_state_count": protected_generated_states,
        "protected_component_count": protected_component_count,
    }
    row.update({f"current_{key}": value for key, value in current_diag.items()})
    row.update({f"unified_{key}": value for key, value in unified_diag.items()})
    row.update({f"protected_{key}": value for key, value in protected_diag.items()})
    return row


def evaluate_text(
    name: str, sample_size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = text_rows(name, sample_size, seed)
    cfg = TEXT_BASE[name]
    generator = generate_2wiki if name == "2WikiMultiHopQA" else generate_hotpot
    flattener = flatten_2wiki if name == "2WikiMultiHopQA" else flatten_hotpot
    gold_fn = gold_2wiki if name == "2WikiMultiHopQA" else gold_hotpot
    kg = KGConstructionConfig(backend="context_only", max_triples=64)
    rows: list[dict[str, Any]] = []
    current_invariance_mismatches = []
    unified_invariance_mismatches = []
    protected_invariance_mismatches = []
    for index, example in enumerate(examples):
        clean = clean_text_retrieval_input(example)
        started = time.perf_counter()
        generated = generator(
            retrieval_example=clean,
            split_name="unified_comparison_development",
            max_sentences_per_example=cfg["atomic_budget"],
            max_subgraph_size=cfg["max_size"],
            max_candidates_per_question=cfg["candidate_cap"],
            kg_config=kg,
            llm_kg_constructor=None,
            kg_cache=None,
            kg_cache_path=None,
            kg_cache_lock=None,
            seed=seed + index,
        )
        current_seconds = time.perf_counter() - started
        current = _canonical(generated["candidates"])

        sentence_records = flattener(clean)
        started = time.perf_counter()
        graph = build_text_evidence_graph(sentence_records, str(clean.get("question", "")))
        result = progressive_connected_supports(graph, UNIFIED_V1_CONFIG)
        unified_seconds = time.perf_counter() - started
        unified = _canonical(result.candidates)
        started = time.perf_counter()
        protected_result = progressive_connected_supports(graph, PROTECTED_CONFIG)
        protected_seconds = time.perf_counter() - started
        protected = _canonical(protected_result.candidates)

        # Gold boundary: candidates, scores, seeds, and caps are frozen above.
        gold = [gold_fn(example, generated["sent_lookup"])]
        rows.append(
            _detail(
                generated["example_id"],
                current,
                unified,
                protected,
                gold,
                generated["sentence_pool"],
                result.explored_unit_ids,
                protected_result.explored_unit_ids,
                int(cfg["max_size"]),
                current_seconds,
                unified_seconds,
                protected_seconds,
                result.seed_unit_ids,
                result.generated_state_count,
                protected_result.protected_anchor_unit_ids,
                protected_result.generated_state_count,
                protected_result.protected_component_count,
            )
        )

        deleted = {
            key: value
            for key, value in example.items()
            if key not in {"answer", "supporting_facts", "evidences"}
        }
        deleted_clean = clean_text_retrieval_input(deleted)
        deleted_current = generator(
            retrieval_example=deleted_clean,
            split_name="unified_comparison_development",
            max_sentences_per_example=cfg["atomic_budget"],
            max_subgraph_size=cfg["max_size"],
            max_candidates_per_question=cfg["candidate_cap"],
            kg_config=kg,
            llm_kg_constructor=None,
            kg_cache=None,
            kg_cache_path=None,
            kg_cache_lock=None,
            seed=seed + index,
        )
        if any(
            generated[field] != deleted_current[field]
            for field in ("sentence_pool", "graph_context_units", "candidates")
        ):
            current_invariance_mismatches.append(generated["example_id"])
        deleted_graph = build_text_evidence_graph(
            flattener(deleted_clean), str(deleted_clean.get("question", ""))
        )
        deleted_result = progressive_connected_supports(deleted_graph, UNIFIED_V1_CONFIG)
        if result.candidates != deleted_result.candidates:
            unified_invariance_mismatches.append(generated["example_id"])
        deleted_protected_result = progressive_connected_supports(deleted_graph, PROTECTED_CONFIG)
        if protected_result.candidates != deleted_protected_result.candidates:
            protected_invariance_mismatches.append(generated["example_id"])
    return rows, {
        "examples": len(rows),
        "A_current_mismatches": current_invariance_mismatches,
        "B_unified_mismatches": unified_invariance_mismatches,
        "C_protected_mismatches": protected_invariance_mismatches,
    }


def evaluate_ontology(name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = ONTO_BASE
    rows: list[dict[str, Any]] = []
    current_invariance_mismatches = []
    unified_invariance_mismatches = []
    protected_invariance_mismatches = []
    for group_index, qa_index, item, qa in onto_rows(name):
        question = str(qa.get("NL Question") or qa.get("ABS Question") or qa.get("Task ID") or "")
        sparql = str(qa.get("SPARQL Query") or "")
        owl_context = str(item.get("OWL Context") or "")
        started = time.perf_counter()
        generated = generate_ontology_candidates(
            question=question,
            sparql_query=sparql,
            owl_context=owl_context,
            max_subgraph_size=cfg["max_size"],
            min_subgraph_size=1,
            max_context_units=cfg["atomic_budget"],
            candidate_beam_width=cfg["beam_width"],
            max_candidate_subgraphs=cfg["candidate_cap"],
        )
        current_seconds = time.perf_counter() - started
        current = _canonical(generated["candidate_subgraphs"])

        context_axioms = parse_owl_context(owl_context)
        started = time.perf_counter()
        graph = build_ontology_evidence_graph(context_axioms, question, sparql)
        result = progressive_connected_supports(graph, UNIFIED_V1_CONFIG)
        unified_seconds = time.perf_counter() - started
        unified = _canonical(result.candidates)
        started = time.perf_counter()
        protected_result = progressive_connected_supports(graph, PROTECTED_CONFIG)
        protected_seconds = time.perf_counter() - started
        protected = _canonical(protected_result.candidates)

        # Gold boundary: ontology explanations are read only here.
        gold = get_gold_explanations(qa)
        example_id = f"{name}__g{group_index}__q{qa_index}"
        rows.append(
            _detail(
                example_id,
                current,
                unified,
                protected,
                gold,
                generated["candidate_units"],
                result.explored_unit_ids,
                protected_result.explored_unit_ids,
                int(cfg["max_size"]),
                current_seconds,
                unified_seconds,
                protected_seconds,
                result.seed_unit_ids,
                result.generated_state_count,
                protected_result.protected_anchor_unit_ids,
                protected_result.generated_state_count,
                protected_result.protected_component_count,
            )
        )

        # The adapter signature accepts no QA annotations.  Reconstructing it
        # after deleting every annotation must therefore be byte-identical.
        deleted_current = generate_ontology_candidates(
            question=question,
            sparql_query=sparql,
            owl_context=owl_context,
            max_subgraph_size=cfg["max_size"],
            min_subgraph_size=1,
            max_context_units=cfg["atomic_budget"],
            candidate_beam_width=cfg["beam_width"],
            max_candidate_subgraphs=cfg["candidate_cap"],
        )
        if any(
            generated[field] != deleted_current[field]
            for field in ("candidate_units", "candidate_subgraphs", "unit_scores")
        ):
            current_invariance_mismatches.append(example_id)
        deleted_result = progressive_connected_supports(
            build_ontology_evidence_graph(context_axioms, question, sparql),
            UNIFIED_V1_CONFIG,
        )
        if result.candidates != deleted_result.candidates:
            unified_invariance_mismatches.append(example_id)
        deleted_protected_result = progressive_connected_supports(
            build_ontology_evidence_graph(context_axioms, question, sparql),
            PROTECTED_CONFIG,
        )
        if protected_result.candidates != deleted_protected_result.candidates:
            protected_invariance_mismatches.append(example_id)
    return rows, {
        "examples": len(rows),
        "A_current_mismatches": current_invariance_mismatches,
        "B_unified_mismatches": unified_invariance_mismatches,
        "C_protected_mismatches": protected_invariance_mismatches,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(output_dir: Path, text_sample_size: int, seed: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output_dir}")
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "protected_query_anchor_comparison_v1",
        "development_only": True,
        "seed": seed,
        "builder_version": BUILDER_VERSION,
        "git": git_provenance(ROOT),
        "common_representation": "EvidenceUnit -> EvidenceGraph -> progressive connected supports",
        "unified_v1_candidate_algorithm": "per-seed/per-first-hop lanes with support-size reservations",
        "protected_candidate_algorithm": "v1 plus query-anchor component admission and one reserved strongest connected path",
        "unified_v1_config": vars(UNIFIED_V1_CONFIG),
        "protected_config": vars(PROTECTED_CONFIG),
        "same_unified_config_all_datasets": True,
        "global_atomic_prefilter_for_unified_method": False,
        "gold_available_during_candidate_generation": False,
        "gold_join_boundary": "after both methods freeze candidates per example",
        "failure_class_policy": {
            "A_current": "current atomic pool is treated as the reached set",
            "B_unified": "all units visited by progressive graph expansion are treated as the reached set",
            "C_protected": "all units visited by protected progressive graph expansion are treated as the reached set",
            "priority": "complete; otherwise class 1 if no reference is fully reached; class 3 if every reached reference exceeds max size; otherwise class 2",
        },
        "2wiki_evidences_used": False,
        "parameter_grid_run": False,
        "test_data_used": False,
        "training_run": False,
        "downstream_components_modified": False,
        "datasets": {},
        "invariance": {},
    }
    for name in TEXT:
        rows, invariance = evaluate_text(name, text_sample_size, seed)
        display = DISPLAY_NAMES[name]
        report["datasets"][display] = {
            "development_examples": len(rows),
            "source_file": str(TEXT[name].relative_to(ROOT)),
            "source_sha256": sha256_file(TEXT[name]),
            "cohort_policy": f"existing deterministic seed-{seed} sample of official development",
            "A_current_clean_generator": _method_summary(rows, "current"),
            "B_unified_evidence_graph_v1": _method_summary(rows, "unified"),
            "C_protected_query_anchors": _method_summary(rows, "protected"),
        }
        report["invariance"][display] = invariance
        _write_jsonl(output_dir / name / "details.jsonl", rows)
    for name in ONTO:
        rows, invariance = evaluate_ontology(name)
        display = DISPLAY_NAMES[name]
        report["datasets"][display] = {
            "development_examples": len(rows),
            "source_file": str(ONTO[name].relative_to(ROOT)),
            "source_sha256": sha256_file(ONTO[name]),
            "cohort_policy": "all labeled QAs in deterministic group-stratified development split",
            "A_current_clean_generator": _method_summary(rows, "current"),
            "B_unified_evidence_graph_v1": _method_summary(rows, "unified"),
            "C_protected_query_anchors": _method_summary(rows, "protected"),
        }
        report["invariance"][display] = invariance
        _write_jsonl(output_dir / name / "details.jsonl", rows)
    report["all_invariance_checks_passed"] = all(
        not value["A_current_mismatches"]
        and not value["B_unified_mismatches"]
        and not value["C_protected_mismatches"]
        for value in report["invariance"].values()
    )
    write_json(output_dir / "comparison.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-sample-size", type=int, default=TEXT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/development_runs/protected_query_anchor_comparison_v1",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.output_dir, args.text_sample_size, args.seed), indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
