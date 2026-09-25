"""Frozen v1.0 symbolic features with configurable scoring coefficients.

This module deliberately copies the original feature semantics instead of
importing the mutable training pipeline.  The defaults are the historical
SAGE-QA values and must not be changed.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


TEXT_DATASETS = frozenset({"HotpotQA", "2WikiMultiHopQA"})
ONTOLOGY_DATASETS = frozenset(
    {
        "FamilyOWL_1hop", "FamilyOWL_2hop", "pizza_100_1hop",
        "pizza_100_2hop", "pizza_250_1hop", "pizza_250_2hop",
        "OWL2Bench_1hop", "OWL2Bench_2hop",
    }
)


@dataclass(frozen=True)
class TextCoefficients:
    question_title: float = 0.030
    question_overlap: float = 0.020
    cross_page: float = 0.020
    bridge_overlap: float = 0.025
    kg_connectivity: float = 0.030
    comparison: float = 0.020
    size_three_penalty: float = 0.002
    size_four_penalty: float = 0.004
    size_beyond_four_penalty: float = 0.010
    duplicate_page_penalty: float = 0.010


@dataclass(frozen=True)
class ProofCoefficients:
    proof_bonus: float = 0.180
    query_coverage_bonus: float = 0.030
    schema_mix_bonus: float = 0.020
    compact_weight: float = 0.350
    extra_unit_penalty: float = 0.010
    compact_property_bonus: float = 0.025
    compact_entity_bonus: float = 0.015
    compact_fact_rule_bonus: float = 0.020
    compact_size_penalty: float = 0.006
    compact_extra_schema_penalty: float = 0.004


ORIGINAL_TEXT_COEFFICIENTS = TextCoefficients()
ORIGINAL_PROOF_COEFFICIENTS = ProofCoefficients()
_ARTICLES = {"a", "an", "the"}


def coefficient_dict(value: TextCoefficients | ProofCoefficients) -> dict[str, float]:
    return {key: float(item) for key, item in asdict(value).items()}


def candidate_identity(row: Mapping[str, Any]) -> str:
    units = [str(unit) for unit in row.get("subgraph_units", []) or []]
    payload = json.dumps(units, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    value = str(text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if token not in _ARTICLES and len(token) > 1}


def _parse_sentence(unit: str) -> tuple[str, int, str]:
    parts = str(unit).split("::", 3)
    if len(parts) == 4 and parts[0] == "SENT":
        try:
            index = int(parts[2])
        except Exception:
            index = -1
        return parts[1], index, parts[3]
    return "", -1, str(unit)


def _question_title_coverage(question: str, units: Sequence[str]) -> float:
    question_tokens = _tokens(question)
    if not question_tokens:
        return 0.0
    title_tokens: set[str] = set()
    for unit in units:
        title_tokens |= _tokens(_parse_sentence(unit)[0])
    return len(question_tokens & title_tokens) / len(question_tokens)


def _candidate_question_overlap(question: str, units: Sequence[str]) -> float:
    question_tokens = _tokens(question)
    return (len(question_tokens & _tokens(" ".join(units))) / len(question_tokens)) if question_tokens else 0.0


def _cross_page_score(units: Sequence[str]) -> float:
    count = len({_parse_sentence(unit)[0] for unit in units if _parse_sentence(unit)[0]})
    return 1.0 if count >= 3 else 0.8 if count == 2 else 0.2 if count == 1 else 0.0


def _bridge_overlap_score(units: Sequence[str]) -> float:
    parsed = [_parse_sentence(unit) for unit in units]
    if len(parsed) < 2:
        return 0.0
    titles = [title for title, _, _ in parsed if title]
    sentences = [sentence for _, _, sentence in parsed]
    hits = possible = 0
    for left, title in enumerate(titles):
        title_tokens = _tokens(title)
        if not title_tokens:
            continue
        for right, sentence in enumerate(sentences):
            if left != right:
                possible += 1
                hits += bool(title_tokens & _tokens(sentence))
    return hits / possible if possible else 0.0


def _comparison_score(question: str, units: Sequence[str]) -> float:
    markers = {"same", "both", "older", "younger", "larger", "smaller", "earlier", "later", "more", "less", "which", "who", "are", "did", "do"}
    if not (set(_normalize(question).split()) & markers):
        return 0.0
    count = len({_parse_sentence(unit)[0] for unit in units if _parse_sentence(unit)[0]})
    return 1.0 if count >= 4 else 0.85 if count == 3 else 0.65 if count == 2 else 0.0


def _parse_kg(unit: str) -> tuple[str, str, str] | None:
    parts = str(unit).split("::", 3)
    return (parts[1], parts[2], parts[3]) if len(parts) == 4 and parts[0] == "KG" else None


def _mentions(text: str, entity: str) -> bool:
    entity_norm, text_norm = _normalize(entity), _normalize(text)
    if not entity_norm or not text_norm:
        return False
    if entity_norm in text_norm:
        return True
    entity_tokens = _tokens(entity)
    return bool(entity_tokens) and len(entity_tokens & _tokens(text)) / len(entity_tokens) >= 0.6


def _kg_connectivity(question: str, units: Sequence[str], kg_units: Sequence[str]) -> float:
    parsed = [value for unit in kg_units if (value := _parse_kg(unit))]
    if not units or not parsed:
        return 0.0
    question_tokens = _tokens(question)
    relevant = sum(bool(_tokens(f"{s} {p} {o}") & question_tokens) for s, p, o in parsed)
    connected = 0
    for unit in units:
        title, _, sentence = _parse_sentence(unit)
        connected += any(_mentions(f"{title} {sentence}", s) or _mentions(f"{title} {sentence}", o) for s, _, o in parsed)
    return 0.70 * connected / len(units) + 0.30 * relevant / len(parsed)


def text_features(row: Mapping[str, Any]) -> dict[str, float]:
    units = [str(unit) for unit in row.get("subgraph_units", []) or []]
    if not units:
        return {key: 0.0 for key in (
            "question_title", "question_overlap", "cross_page", "bridge_overlap",
            "kg_connectivity", "comparison", "is_size_three", "is_size_four",
            "size_beyond_four", "duplicate_page",
        )}
    question = str(row.get("question", ""))
    titles = [_parse_sentence(unit)[0] for unit in units if _parse_sentence(unit)[0]]
    return {
        "question_title": _question_title_coverage(question, units),
        "question_overlap": _candidate_question_overlap(question, units),
        "cross_page": _cross_page_score(units),
        "bridge_overlap": _bridge_overlap_score(units),
        "kg_connectivity": _kg_connectivity(question, units, row.get("graph_context_units", []) or []),
        "comparison": _comparison_score(question, units),
        "is_size_three": float(len(units) == 3),
        "is_size_four": float(len(units) == 4),
        "size_beyond_four": float(max(0, len(units) - 4)),
        "duplicate_page": float(len(units) >= 2 and len(set(titles)) <= 1),
    }


def score_text_features(features: Mapping[str, float], coefficients: TextCoefficients = ORIGINAL_TEXT_COEFFICIENTS) -> float:
    return (
        coefficients.question_title * features["question_title"]
        + coefficients.question_overlap * features["question_overlap"]
        + coefficients.cross_page * features["cross_page"]
        + coefficients.bridge_overlap * features["bridge_overlap"]
        + coefficients.kg_connectivity * features["kg_connectivity"]
        + coefficients.comparison * features["comparison"]
        - coefficients.size_three_penalty * features["is_size_three"]
        - coefficients.size_four_penalty * features["is_size_four"]
        - coefficients.size_beyond_four_penalty * features["size_beyond_four"]
        - coefficients.duplicate_page_penalty * features["duplicate_page"]
    )


def _uri_fragments(text: str) -> list[str]:
    return re.findall(r"#([^>]+)>", str(text)) if text else []


def _schema_unit(unit: str) -> bool:
    markers = ("SymmetricObjectProperty", "TransitiveObjectProperty", "SubObjectPropertyOf", "InverseObjectProperties", "PropertyChain", "FunctionalObjectProperty", "domain", "range", "subPropertyOf", "inverseOf")
    return any(marker in str(unit) for marker in markers)


def proof_features(row: Mapping[str, Any], *, proof_success: bool | None = None) -> dict[str, float]:
    units = [str(unit) for unit in row.get("subgraph_units", []) or []]
    query = str(row.get("sparql_query", "") or row.get("question", ""))
    query_tokens = {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query) if len(token) > 2 and not token.lower().startswith("http")}
    unit_text_lower = " ".join(units).lower()
    fragments = _uri_fragments(query)
    properties = {value for value in fragments if value.startswith("has") or value.startswith("is")}
    entities = {value for value in fragments if value not in properties}
    schemas = sum(_schema_unit(unit) for unit in units)
    mix = float(bool(units) and any(_schema_unit(unit) for unit in units) and any(not _schema_unit(unit) for unit in units))
    if proof_success is None:
        try:
            from generation.generate_owl_answers_with_llm import infer_owl_boolean_answer
            proof_success = infer_owl_boolean_answer(dict(row), units) is not None
        except Exception:
            proof_success = False
    return {
        "proof_success": float(proof_success),
        "query_coverage": sum(token in unit_text_lower for token in query_tokens) / max(len(query_tokens), 1),
        "fact_rule_mix": mix,
        "query_property_match": float(bool(properties) and any(value in " ".join(units) for value in properties)),
        "query_entity_coverage": sum(value in " ".join(units) for value in entities) / max(len(entities), 1) if entities else 0.0,
        "oversize": float(max(0, int(row.get("subgraph_size", len(units))) - 2)),
        "extra_schema": float(max(0, schemas - 1)),
    }


def score_proof_features(features: Mapping[str, float], coefficients: ProofCoefficients = ORIGINAL_PROOF_COEFFICIENTS) -> float:
    compact = (
        coefficients.compact_property_bonus * features["query_property_match"]
        + coefficients.compact_entity_bonus * features["query_entity_coverage"]
        + coefficients.compact_fact_rule_bonus * features["fact_rule_mix"]
        - coefficients.compact_size_penalty * features["oversize"]
        - coefficients.compact_extra_schema_penalty * features["extra_schema"]
    )
    return (
        coefficients.proof_bonus * features["proof_success"]
        + coefficients.query_coverage_bonus * features["query_coverage"]
        + coefficients.schema_mix_bonus * features["fact_rule_mix"]
        + coefficients.compact_weight * compact
        - coefficients.extra_unit_penalty * features["oversize"]
    )


def symbolic_adjustment(row: Mapping[str, Any], coefficients: TextCoefficients | ProofCoefficients | None = None, *, proof_success: bool | None = None) -> float:
    dataset = str(row.get("dataset") or row.get("source_name") or "")
    if dataset in TEXT_DATASETS:
        selected = coefficients or ORIGINAL_TEXT_COEFFICIENTS
        if not isinstance(selected, TextCoefficients):
            raise TypeError("Text rows require TextCoefficients")
        return score_text_features(text_features(row), selected)
    selected = coefficients or ORIGINAL_PROOF_COEFFICIENTS
    if not isinstance(selected, ProofCoefficients):
        raise TypeError("Ontology rows require ProofCoefficients")
    return score_proof_features(proof_features(row, proof_success=proof_success), selected)
