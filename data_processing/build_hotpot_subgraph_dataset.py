import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import random
import re
import string
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing.text_kg_constructor import (
    KGConstructionConfig,
    LLMKGConstructor,
    construct_text_kg,
)
from data_processing.retrieval_contracts import (
    BUILDER_VERSION,
    assert_disjoint_split_ids,
    assert_gold_free_mapping,
    clean_text_retrieval_input,
    git_provenance,
    select_disjoint_cohorts,
    split_manifest,
    validate_clean_cache_row,
    validate_clean_kg_backend,
    write_json,
)
from data_processing.evidence_graph_candidates import (
    GENERATOR_D_CONFIG,
    GENERATOR_D_VERSION,
    build_text_evidence_graph,
    frozen_generator_d_config,
    frozen_generator_d_lineage,
    generate_generator_d_candidates,
)


# ============================================================
# Basic text utilities
# ============================================================

_ARTICLES = {"a", "an", "the"}


def normalize_text(text: str) -> str:
    """
    Lightweight normalization for lexical scoring.
    """
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return [t for t in text.split() if t and t not in _ARTICLES]


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def safe_title(title: str) -> str:
    """
    Keep page title readable but stable inside the SENT::... format.
    """
    title = str(title).replace("\n", " ").replace("\t", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def clean_sentence(sent: str) -> str:
    sent = str(sent).replace("\n", " ").replace("\t", " ")
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent


def make_sentence_unit(title: str, sent_idx: int, sent: str) -> str:
    """
    Main HotpotQA evidence unit format.

    Example:
    SENT::Albert Einstein::0::Albert Einstein was a German-born physicist.
    """
    return f"SENT::{safe_title(title)}::{int(sent_idx)}::{clean_sentence(sent)}"


def parse_sentence_unit(unit: str) -> Tuple[str, int, str]:
    """
    Returns title, sentence index, sentence text.
    """
    parts = str(unit).split("::", 3)
    if len(parts) != 4 or parts[0] != "SENT":
        return "", -1, str(unit)
    title = parts[1]
    try:
        idx = int(parts[2])
    except ValueError:
        idx = -1
    sent = parts[3]
    return title, idx, sent


def to_python_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def make_kg_triple_unit(subject: str, predicate: str, obj: str) -> str:
    """
    Structured KG evidence triple used as a graph-context bridge node.
    """
    return f"KG::{clean_sentence(subject)}::{clean_sentence(predicate)}::{clean_sentence(obj)}"


def evidence_triple_units(evidences: List[List[str]]) -> List[str]:
    units: List[str] = []
    seen = set()
    for ev in evidences:
        if not isinstance(ev, list) or len(ev) < 3:
            continue
        unit = make_kg_triple_unit(ev[0], ev[1], ev[2])
        if unit not in seen:
            seen.add(unit)
            units.append(unit)
    return units


def construct_wiki_bridge_triples(
    sentence_records: List[Dict[str, Any]],
    max_triples: int,
) -> List[List[str]]:
    """
    Deterministic context-only KG fallback for Wikipedia text.

    It promotes page-title co-mentions into typed bridge triples so vanilla
    HotpotQA can expose structural nodes to the GNN without REBEL/Wikidata
    enrichment or gold-answer grounding.
    """
    if max_triples <= 0:
        return []

    titles = []
    seen_titles = set()
    for rec in sentence_records:
        title = rec.get("title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            titles.append(title)

    triples: List[List[str]] = []
    seen = set()

    def add(subject: str, predicate: str, obj: str) -> None:
        if len(triples) >= max_triples:
            return
        subject = clean_sentence(subject)
        predicate = clean_sentence(predicate)
        obj = clean_sentence(obj)
        if not subject or not predicate or not obj or subject == obj:
            return
        key = (subject, predicate, obj)
        if key not in seen:
            seen.add(key)
            triples.append([subject, predicate, obj])

    title_by_norm = {normalize_text(title): title for title in titles}
    for rec in sentence_records:
        source_title = rec.get("title", "")
        sentence = rec.get("sentence", "")

        for target_norm, target_title in title_by_norm.items():
            if not target_norm or target_title == source_title:
                continue
            if phrase_in_text(target_title, sentence):
                add(source_title, "mentions_page", target_title)

        if len(triples) >= max_triples:
            break

    return triples


def phrase_in_text(phrase: str, text: str) -> bool:
    phrase_norm = normalize_text(phrase)
    text_norm = normalize_text(text)
    if len(phrase_norm) < 3 or not text_norm:
        return False
    return f" {phrase_norm} " in f" {text_norm} "


# ============================================================
# HotpotQA parsing
# ============================================================


def normalize_hotpot_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes Hugging Face HotpotQA parquet rows into the format expected
    by the builder.

    Expected output:
      {
        "_id": ...,
        "question": ...,
        "answer": ...,
        "type": ...,
        "level": ...,
        "context": [[title, [sentences...]], ...],
        "supporting_facts": [[title, sent_idx], ...]
      }
    """
    out = dict(row)

    # Some HF versions use "id" instead of "_id".
    if "_id" not in out and "id" in out:
        out["_id"] = out["id"]

    # Normalize context.
    context = out.get("context")

    # Common HF parquet format:
    # context = {"title": [...], "sentences": [[...], [...]]}
    if isinstance(context, dict):
        titles = first_present(context, "title", "titles", default=[])
        sentences = first_present(context, "sentences", "sentence", default=[])

        normalized_context = []
        for title, sents in zip(titles, sentences):
            normalized_context.append([title, list(sents)])

        out["context"] = normalized_context

    # Sometimes context is already list-like but numpy arrays appear inside.
    elif isinstance(context, list):
        normalized_context = []
        for item in context:
            if isinstance(item, dict):
                title = item.get("title", "")
                sents = item.get("sentences", item.get("sentence", []))
                normalized_context.append([title, list(sents)])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                title, sents = item
                normalized_context.append([title, list(sents)])
        out["context"] = normalized_context

    # Normalize supporting_facts.
    sf = out.get("supporting_facts")

    # Common HF parquet format:
    # supporting_facts = {"title": [...], "sent_id": [...]}
    if isinstance(sf, dict):
        titles = first_present(sf, "title", "titles", default=[])
        sent_ids = first_present(
            sf,
            "sent_id",
            "sent_ids",
            "sentence_id",
            "sent_idx",
            default=[],
        )

        normalized_sf = []
        for title, idx in zip(titles, sent_ids):
            normalized_sf.append([title, int(idx)])

        out["supporting_facts"] = normalized_sf

    elif isinstance(sf, list):
        normalized_sf = []
        for item in sf:
            if isinstance(item, dict):
                title = item.get("title", "")
                idx = (
                    item.get("sent_id")
                    if "sent_id" in item
                    else item.get("sentence_id", item.get("sent_idx", 0))
                )
                normalized_sf.append([title, int(idx)])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                title, idx = item
                normalized_sf.append([title, int(idx)])
        out["supporting_facts"] = normalized_sf

    out.pop("evidences", None)
    out.pop("evidence", None)
    out.pop("kg_triples", None)
    out.pop("triples", None)

    return out


def first_present(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Return the first non-None value for keys without truth-testing arrays.
    """
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def load_hotpot_file(path: str) -> List[Dict[str, Any]]:
    """
    Loads HotpotQA from either:
      - .json
      - .jsonl
      - .parquet
      - a directory containing parquet files

    Hugging Face downloads usually use parquet.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"HotpotQA path not found: {p}")

    # Directory: load all parquet files inside it.
    if p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under directory: {p}")

        all_rows = []
        for pq in parquet_files:
            all_rows.extend(load_hotpot_file(str(pq)))
        return all_rows

    suffix = p.suffix.lower()

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list, got {type(data)} from {p}")
        return [normalize_hotpot_record(r) for r in data]

    if suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(normalize_hotpot_record(json.loads(line)))
        return rows

    if suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(p)
        records = df.to_dict(orient="records")

        # Normalize HF-style rows if needed.
        return [normalize_hotpot_record(r) for r in records]

    raise ValueError(f"Unsupported HotpotQA file type: {p}")


def flatten_context(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts HotpotQA context into sentence records.

    HotpotQA context format:
      "context": [
          ["Page Title", ["sent0", "sent1", ...]],
          ...
      ]

    Returns list of:
      {
        "title": ...,
        "sent_idx": ...,
        "sentence": ...,
        "unit": ...
      }
    """
    records: List[Dict[str, Any]] = []

    context = example.get("context", [])
    for page in context:
        if not isinstance(page, list) or len(page) != 2:
            continue

        title, sentences = page
        title = safe_title(title)

        if not isinstance(sentences, list):
            continue

        for idx, sent in enumerate(sentences):
            sent_clean = clean_sentence(sent)
            if not sent_clean:
                continue

            unit = make_sentence_unit(title, idx, sent_clean)
            records.append(
                {
                    "title": title,
                    "sent_idx": idx,
                    "sentence": sent_clean,
                    "unit": unit,
                }
            )

    return records


def get_gold_support_units(
    example: Dict[str, Any], sent_lookup: Dict[Tuple[str, int], str]
) -> List[str]:
    """
    Extract gold supporting facts from HotpotQA.

    HotpotQA supporting_facts format:
      [["Page Title", sentence_index], ...]

    Returns sentence units.
    """
    gold_units: List[str] = []

    for item in example.get("supporting_facts", []):
        if not isinstance(item, list) or len(item) != 2:
            continue

        title, idx = item
        title = safe_title(title)

        try:
            idx = int(idx)
        except Exception:
            continue

        unit = sent_lookup.get((title, idx))
        if unit is not None:
            gold_units.append(unit)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in gold_units:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    return deduped


def get_raw_supporting_facts(example: Dict[str, Any]) -> List[List[Any]]:
    facts: List[List[Any]] = []
    for item in example.get("supporting_facts", []):
        if not isinstance(item, list) or len(item) != 2:
            continue
        title, idx = item
        try:
            idx = int(idx)
        except Exception:
            continue
        facts.append([safe_title(title), idx])
    return facts


# ============================================================
# Candidate generation
# ============================================================


def lexical_overlap_score(query: str, text: str) -> float:
    q = token_set(query)
    t = token_set(text)
    if not q:
        return 0.0
    return len(q & t) / max(len(q), 1)


def answer_overlap_score(answer: str, text: str) -> float:
    answer = str(answer or "")
    if not answer:
        return 0.0

    answer_norm = normalize_text(answer)
    text_norm = normalize_text(text)

    if not answer_norm:
        return 0.0

    # Exact substring match is useful for HotpotQA answer grounding.
    if answer_norm in text_norm:
        return 1.0

    a = token_set(answer_norm)
    t = token_set(text_norm)
    if not a:
        return 0.0
    return len(a & t) / max(len(a), 1)


def title_overlap_score(question: str, units: List[str]) -> float:
    q = token_set(question)
    if not q:
        return 0.0

    title_tokens: Set[str] = set()
    for u in units:
        title, _, _ = parse_sentence_unit(u)
        title_tokens |= token_set(title)

    return len(q & title_tokens) / max(len(q), 1)


def select_sentence_pool(
    example: Dict[str, Any],
    sentence_records: List[Dict[str, Any]],
    max_sentences_per_example: int,
) -> List[str]:
    """
    Selects a limited pool of sentence units for candidate generation.

    Uses only question/context-visible lexical signals.
    """
    question = example.get("question", "")

    scored = []
    for rec in sentence_records:
        unit = rec["unit"]
        sent = rec["sentence"]
        title = rec["title"]

        q_score = lexical_overlap_score(question, sent + " " + title)
        score = q_score

        scored.append((score, unit))

    scored.sort(key=lambda x: x[0], reverse=True)

    pool: List[str] = []
    seen = set()

    # Include top lexical sentences.
    for _, u in scored:
        if len(pool) >= max_sentences_per_example:
            break
        if u not in seen:
            seen.add(u)
            pool.append(u)

    return pool[:max_sentences_per_example]


def support_f1(candidate: Iterable[str], gold: Iterable[str]) -> float:
    c = set(candidate)
    g = set(gold)
    if not c and not g:
        return 1.0
    if not c or not g:
        return 0.0
    return 2.0 * len(c & g) / (len(c) + len(g))


def support_jaccard(candidate: Iterable[str], gold: Iterable[str]) -> float:
    c = set(candidate)
    g = set(gold)
    if not c and not g:
        return 1.0
    if not c or not g:
        return 0.0
    return len(c & g) / len(c | g)


def label_candidate(
    candidate: List[str], gold_reference_sets: List[List[str]]
) -> Tuple[int, float, float, bool, bool]:
    """
    Returns:
      label,
      best_f1,
      best_jaccard,
      exact_match_any_gold,
      contains_any_gold_explanation

    label = 1 iff candidate contains a complete gold explanation.
    """
    cset = set(candidate)

    best_f1 = 0.0
    best_j = 0.0
    exact = False
    contains = False

    for gold in gold_reference_sets:
        gset = set(gold)

        f1 = support_f1(cset, gset)
        jac = support_jaccard(cset, gset)

        best_f1 = max(best_f1, f1)
        best_j = max(best_j, jac)

        if cset == gset:
            exact = True

        if gset and gset.issubset(cset):
            contains = True

    label = 1 if contains else 0
    return label, best_f1, best_j, exact, contains


def generate_candidate_subgraphs(
    example: Dict[str, Any],
    sentence_pool: List[str],
    max_subgraph_size: int,
    max_candidates_per_question: int,
    seed: int,
) -> List[List[str]]:
    """
    Generates candidate sentence subgraphs.

    Candidate = list of sentence units.

    We include:
      - all singletons
      - combinations up to max_subgraph_size
      - cap total number of candidates
    """
    rng = random.Random(seed)

    candidates: List[Tuple[str, ...]] = []
    seen = set()

    def add_candidate(units: Iterable[str]):
        cand = tuple(sorted(set(units)))
        if not cand:
            return
        if len(cand) > max_subgraph_size:
            return
        if cand not in seen:
            seen.add(cand)
            candidates.append(cand)

    # Add singletons.
    for u in sentence_pool:
        add_candidate([u])

    # Add pairs/triples.
    for size in range(2, max_subgraph_size + 1):
        combos = list(itertools.combinations(sentence_pool, size))

        # If too many, sample but keep deterministic.
        if len(combos) > max_candidates_per_question * 4:
            combos = rng.sample(combos, max_candidates_per_question * 4)

        for combo in combos:
            add_candidate(combo)
            if len(candidates) >= max_candidates_per_question:
                break

        if len(candidates) >= max_candidates_per_question:
            break

    # Deterministic gold-free cap in stable generation order.
    if len(candidates) > max_candidates_per_question:
        candidates = candidates[:max_candidates_per_question]

    return [list(c) for c in candidates]


# ============================================================
# Symbolic/text features
# ============================================================


def make_hotpot_symbolic_features(
    question: str,
    candidate_units: List[str],
    max_subgraph_size: int,
) -> List[float]:
    """
    8-dimensional gold-free symbolic/text feature vector.

    Keep dimension compatible with existing GNN.
    """
    candidate_text_parts = []
    titles = []
    sent_positions = []

    for u in candidate_units:
        title, idx, sent = parse_sentence_unit(u)
        candidate_text_parts.append(title)
        candidate_text_parts.append(sent)
        if title:
            titles.append(title)
        if idx >= 0:
            sent_positions.append(idx)

    candidate_text = " ".join(candidate_text_parts)

    q_overlap = lexical_overlap_score(question, candidate_text)
    title_question_coverage = title_overlap_score(question, candidate_units)

    unique_titles = len(set(titles))
    size = len(candidate_units)

    unique_page_ratio = unique_titles / max(size, 1)
    cross_page = 1.0 if unique_titles >= 2 else 0.0
    multi_sentence = 1.0 if size >= 2 else 0.0
    size_norm = min(size / max(max_subgraph_size, 1), 1.0)

    if sent_positions:
        # Lower sentence indices are often more salient.
        avg_pos = sum(sent_positions) / len(sent_positions)
        avg_sentence_position = min(avg_pos / 10.0, 1.0)
    else:
        avg_sentence_position = 0.0

    title_overlap = title_overlap_score(question, candidate_units)

    return [
        float(q_overlap),
        float(title_question_coverage),
        float(unique_page_ratio),
        float(cross_page),
        float(multi_sentence),
        float(size_norm),
        float(avg_sentence_position),
        float(title_overlap),
    ]


# ============================================================
# Row construction
# ============================================================


def candidate_pre_rank_score(question: str, candidate_units: List[str]) -> float:
    text = " ".join(candidate_units)
    cross_page = len({parse_sentence_unit(unit)[0] for unit in candidate_units}) > 1
    return (
        lexical_overlap_score(question, text)
        + 0.2 * title_overlap_score(question, candidate_units)
        + 0.05 * float(cross_page)
        - 0.005 * len(candidate_units)
    )


def generate_candidates(
    retrieval_example: Dict[str, Any],
    split_name: str,
    max_sentences_per_example: int,
    max_subgraph_size: int,
    max_candidates_per_question: int,
    kg_config: KGConstructionConfig,
    llm_kg_constructor: LLMKGConstructor | None,
    kg_cache: Dict[str, Dict[str, Any]] | None,
    kg_cache_path: Path | None,
    kg_cache_lock: Any,
    seed: int,
) -> Dict[str, Any]:
    assert_gold_free_mapping(retrieval_example, stage="HotpotQA candidate generation")
    validate_clean_kg_backend(kg_config.backend)
    ex_id = str(retrieval_example.get("id") or f"no_id_{seed}")
    example_id = f"HotpotQA__{split_name}__{ex_id}"
    question = str(retrieval_example.get("question", ""))
    sentence_records = flatten_context(retrieval_example)
    sent_lookup = {(rec["title"], int(rec["sent_idx"])): rec["unit"] for rec in sentence_records}
    cached_kg = kg_cache.get(example_id) if kg_cache is not None else None
    if cached_kg:
        validate_clean_cache_row(cached_kg, expected_signature=kg_config.cache_signature)
        kg_triples = cached_kg.get("kg_triples", [])
        kg_construction_method = str(cached_kg.get("construction_method", "cache"))
    else:
        kg_triples, kg_construction_method = construct_text_kg(
            sentence_records=sentence_records,
            question=question,
            config=kg_config,
            llm_constructor=llm_kg_constructor,
        )
        if llm_kg_constructor is not None and kg_cache is not None:
            cache_row = {
                "example_id": example_id,
                "kg_triples": kg_triples,
                "construction_method": kg_construction_method,
                "cache_signature": kg_config.cache_signature,
                "gold_available_during_candidate_generation": False,
                "input_fields": ["id", "question", "context"],
            }
            if kg_cache_lock is not None:
                with kg_cache_lock:
                    kg_cache[example_id] = cache_row
                    append_kg_cache_row(kg_cache_path, cache_row)
            else:
                kg_cache[example_id] = cache_row
                append_kg_cache_row(kg_cache_path, cache_row)
    graph_context_units = evidence_triple_units(kg_triples)

    # Legacy size/budget arguments remain in the public function signature for
    # compatibility; Generator D always uses its immutable production config.
    evidence_graph = build_text_evidence_graph(sentence_records, question)
    generator_result = generate_generator_d_candidates(evidence_graph)
    sentence_pool = [unit.unit_id for unit in evidence_graph.units]
    candidates = [list(candidate) for candidate in generator_result.candidates]

    return {
        "example_id": example_id,
        "raw_id": ex_id,
        "split_name": split_name,
        "question": question,
        "sent_lookup": sent_lookup,
        "sentence_pool": sentence_pool,
        "candidates": candidates,
        "graph_context_units": graph_context_units,
        "kg_construction_method": kg_construction_method,
        "kg_cache_signature": kg_config.cache_signature,
        "candidate_generator": GENERATOR_D_VERSION,
        "candidate_generation_config": frozen_generator_d_config(),
        "query_anchor_unit_ids": list(generator_result.query_anchor_unit_ids),
        "gold_available_during_candidate_generation": False,
    }


def label_candidates(
    generated: Dict[str, Any],
    *,
    supporting_facts: Any | None,
    max_subgraph_size: int,
) -> List[Dict[str, Any]]:
    attach_gold = supporting_facts is not None
    annotation = {"supporting_facts": supporting_facts or []}
    gold_units = get_gold_support_units(annotation, generated["sent_lookup"]) if attach_gold else []
    raw_supporting_facts = get_raw_supporting_facts(annotation) if attach_gold else []
    gold_reference_sets = [gold_units] if gold_units else []
    example_id = generated["example_id"]
    ex_id = generated["raw_id"]
    split_name = generated["split_name"]
    question = generated["question"]
    candidates = generated["candidates"]

    rows: List[Dict[str, Any]] = []

    for cand_idx, candidate_units in enumerate(candidates):
        label, best_f1, best_jaccard, exact, contains = label_candidate(
            candidate=candidate_units,
            gold_reference_sets=gold_reference_sets,
        )

        symbolic_features = make_hotpot_symbolic_features(
            question=question,
            candidate_units=candidate_units,
            max_subgraph_size=max_subgraph_size,
        )

        row_id = f"HotpotQA__{split_name}__{ex_id}__cand{cand_idx}"

        row = {
            "row_id": row_id,
            "example_id": example_id,
            "dataset": "HotpotQA",
            "hop": "2hop",
            "answer_type": "OPEN",
            "question": question,
            # Candidate support.
            "subgraph_units": candidate_units,
            "subgraph_size": len(candidate_units),
            # Graph context nodes are available to the GNN message-passing
            # graph, but are not scored as predicted support sentences.
            "graph_context_units": generated["graph_context_units"],
            # Model features.
            "symbolic_features": symbolic_features,
            "candidate_pre_rank_score": candidate_pre_rank_score(question, candidate_units),
            "generation_rank": cand_idx,
            "sentence_pool_size": len(generated["sentence_pool"]),
            "kg_construction_method": generated["kg_construction_method"],
            "kg_cache_signature": generated["kg_cache_signature"],
            "candidate_generator": generated["candidate_generator"],
            "candidate_generation_config": generated["candidate_generation_config"],
            "gold_available_during_candidate_generation": False,
            "gold_used_during_labeling": bool(attach_gold and gold_reference_sets),
            "builder_version": BUILDER_VERSION,
        }
        if attach_gold:
            row.update(
                {
                    "raw_supporting_facts": raw_supporting_facts,
                    "gold_support_units": gold_units,
                    "label": int(label),
                    "rank_target": float(best_f1),
                    "best_set_f1_to_gold": float(best_f1),
                    "best_jaccard_to_gold": float(best_jaccard),
                    "exact_match_any_gold": bool(exact),
                    "contains_any_gold_explanation": bool(contains),
                }
            )

        rows.append(row)

    return rows


def build_rows_for_example(
    example: Dict[str, Any],
    split_name: str,
    max_sentences_per_example: int,
    max_subgraph_size: int,
    max_candidates_per_question: int,
    kg_config: KGConstructionConfig,
    llm_kg_constructor: LLMKGConstructor | None,
    kg_cache: Dict[str, Dict[str, Any]] | None,
    kg_cache_path: Path | None,
    kg_cache_lock: Any,
    seed: int,
) -> List[Dict[str, Any]]:
    generated = generate_candidates(
        retrieval_example=clean_text_retrieval_input(example),
        split_name=split_name,
        max_sentences_per_example=max_sentences_per_example,
        max_subgraph_size=max_subgraph_size,
        max_candidates_per_question=max_candidates_per_question,
        kg_config=kg_config,
        llm_kg_constructor=llm_kg_constructor,
        kg_cache=kg_cache,
        kg_cache_path=kg_cache_path,
        kg_cache_lock=kg_cache_lock,
        seed=seed,
    )
    return label_candidates(
        generated,
        supporting_facts=(None if split_name == "test" else example.get("supporting_facts", [])),
        max_subgraph_size=GENERATOR_D_CONFIG.max_support_size,
    )


# ============================================================
# Dataset writing
# ============================================================


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_kg_cache(path: Path | None) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if path is None or not path.exists():
        return cache
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                example_id = row.get("example_id")
                if example_id:
                    cache[str(example_id)] = row
    return cache


def append_kg_cache_row(path: Path | None, row: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_dev_from_train(
    examples: List[Dict[str, Any]],
    dev_ratio: float,
    seed: int,
    max_train_examples: int = 0,
    max_dev_examples: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    if max_train_examples and max_train_examples > 0:
        # Keep enough examples for train + dev if possible.
        total = max_train_examples + (
            max_dev_examples if max_dev_examples > 0 else int(max_train_examples * dev_ratio)
        )
        shuffled = shuffled[: min(total, len(shuffled))]

    n_dev = int(round(len(shuffled) * dev_ratio))

    if max_dev_examples and max_dev_examples > 0:
        n_dev = min(n_dev, max_dev_examples)

    dev = shuffled[:n_dev]
    train = shuffled[n_dev:]

    if max_train_examples and max_train_examples > 0:
        train = train[:max_train_examples]

    return train, dev


def limit_examples(
    examples: List[Dict[str, Any]], max_examples: int, seed: int
) -> List[Dict[str, Any]]:
    if not max_examples or max_examples <= 0:
        return examples
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled[:max_examples]


def build_split_rows(
    examples: List[Dict[str, Any]],
    split_name: str,
    max_sentences_per_example: int,
    max_subgraph_size: int,
    max_candidates_per_question: int,
    kg_config: KGConstructionConfig,
    llm_kg_constructor: LLMKGConstructor | None,
    kg_construction_workers: int,
    kg_cache_path: Path | None,
    seed: int,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    start_time = time.time()
    uses_llm = llm_kg_constructor is not None
    kg_cache = load_kg_cache(kg_cache_path) if uses_llm else None
    kg_cache_lock = threading.Lock() if uses_llm else None
    print(
        f"  {split_name}: starting {len(examples)} examples "
        f"(kg_backend={kg_config.backend}, "
        f"workers={kg_construction_workers if uses_llm else 1})",
        flush=True,
    )
    if uses_llm and kg_cache_path is not None:
        print(
            f"  {split_name}: KG cache has {len(kg_cache or {})} examples at {kg_cache_path}",
            flush=True,
        )

    def print_progress(done_count: int, total_count: int) -> None:
        elapsed = max(time.time() - start_time, 1e-6)
        rate = done_count / elapsed
        remaining = max(total_count - done_count, 0)
        eta_seconds = remaining / rate if rate > 0 else 0.0
        print(
            f"  {split_name}: built {done_count}/{total_count} examples "
            f"({rate:.2f} ex/s, elapsed {elapsed / 60:.1f} min, "
            f"ETA {eta_seconds / 60:.1f} min)",
            flush=True,
        )

    if llm_kg_constructor is not None and kg_construction_workers > 1:
        indexed_results: List[Tuple[int, List[Dict[str, Any]]]] = []

        def build_one(
            idx_ex: Tuple[int, Dict[str, Any]],
        ) -> Tuple[int, List[Dict[str, Any]]]:
            idx, ex = idx_ex
            local_constructor = LLMKGConstructor(kg_config)
            return idx, build_rows_for_example(
                example=ex,
                split_name=split_name,
                max_sentences_per_example=max_sentences_per_example,
                max_subgraph_size=max_subgraph_size,
                max_candidates_per_question=max_candidates_per_question,
                kg_config=kg_config,
                llm_kg_constructor=local_constructor,
                kg_cache=kg_cache,
                kg_cache_path=kg_cache_path,
                kg_cache_lock=kg_cache_lock,
                seed=seed + idx,
            )

        with ThreadPoolExecutor(max_workers=kg_construction_workers) as pool:
            futures = [pool.submit(build_one, (idx, ex)) for idx, ex in enumerate(examples)]
            for done_count, future in enumerate(as_completed(futures), start=1):
                indexed_results.append(future.result())
                if done_count % 5 == 0 or done_count == len(futures):
                    print_progress(done_count, len(futures))

        for _, rows in sorted(indexed_results, key=lambda item: item[0]):
            all_rows.extend(rows)
        return all_rows

    for idx, ex in enumerate(examples):
        rows = build_rows_for_example(
            example=ex,
            split_name=split_name,
            max_sentences_per_example=max_sentences_per_example,
            max_subgraph_size=max_subgraph_size,
            max_candidates_per_question=max_candidates_per_question,
            kg_config=kg_config,
            llm_kg_constructor=llm_kg_constructor,
            kg_cache=kg_cache,
            kg_cache_path=kg_cache_path,
            kg_cache_lock=kg_cache_lock,
            seed=seed + idx,
        )
        all_rows.extend(rows)
        progress_every = 5 if uses_llm else 25
        if (idx + 1) % progress_every == 0 or idx + 1 == len(examples):
            print_progress(idx + 1, len(examples))

    return all_rows


def summarize_rows(name: str, rows: List[Dict[str, Any]]) -> None:
    example_ids = {r["example_id"] for r in rows}
    positives = sum(int(r.get("label", 0)) for r in rows)
    total = len(rows)
    print(f"{name}:")
    print(f"  rows:      {total}")
    print(f"  examples:  {len(example_ids)}")
    print(f"  positives: {positives}")
    print(f"  negatives: {total - positives}")
    if total:
        print(f"  pos rate:  {positives / total:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-file", type=str, nargs="+", required=True)
    parser.add_argument("--dev-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="data/HotpotQA")

    parser.add_argument("--dev-ratio-from-train", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-train-examples", type=int, default=1000)
    parser.add_argument("--max-dev-examples", type=int, default=200)
    parser.add_argument("--max-test-examples", type=int, default=300)

    parser.add_argument(
        "--max-sentences-per-example",
        type=int,
        default=30,
        help="Legacy compatibility option; Generator D adapts the full context.",
    )
    parser.add_argument(
        "--max-subgraph-size",
        type=int,
        default=6,
        help="Legacy compatibility option; frozen Generator D always uses size 6.",
    )
    parser.add_argument(
        "--max-candidates-per-question",
        type=int,
        default=512,
        help="Legacy compatibility option; frozen Generator D always caps at 512.",
    )
    parser.add_argument(
        "--max-kg-bridge-triples",
        type=int,
        default=64,
        help=("Maximum context-derived KG triples to attach per example."),
    )
    parser.add_argument(
        "--kg-construction-backend",
        choices=[
            "context_only",
            "deterministic",
            "llm",
            "llm_with_title_bridges",
            "none",
        ],
        default="context_only",
        help=(
            "How to build KG triples strictly from question/context. context_only "
            "uses deterministic title bridges; llm extracts from context text. "
            "llm_with_title_bridges combines context-only LLM triples with "
            "deterministic title co-mention bridges."
        ),
    )
    parser.add_argument("--kg-construction-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--kg-max-context-sentences", type=int, default=20)
    parser.add_argument("--kg-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--kg-request-timeout", type=float, default=90.0)
    parser.add_argument("--kg-max-retries", type=int, default=4)
    parser.add_argument("--kg-retry-initial-sleep", type=float, default=5.0)
    parser.add_argument(
        "--kg-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-split LLM KG cache JSONL files. Defaults to "
            "<output-dir>/kg_cache for LLM KG backends."
        ),
    )
    parser.add_argument(
        "--kg-construction-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel LLM KG extraction workers. Use 1 for serial "
            "execution; 4-8 can speed up large builds if rate limits allow."
        ),
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kg_cache_dir = args.kg_cache_dir or (out_dir / "kg_cache")
    kg_backend = (
        "context_only" if args.kg_construction_backend == "none" else args.kg_construction_backend
    )
    validate_clean_kg_backend(kg_backend)
    kg_config = KGConstructionConfig(
        backend=kg_backend,
        model=args.kg_construction_model,
        max_triples=args.max_kg_bridge_triples,
        max_context_sentences=args.kg_max_context_sentences,
        sleep_seconds=args.kg_sleep_seconds,
        request_timeout=args.kg_request_timeout,
        max_retries=args.kg_max_retries,
        retry_initial_sleep=args.kg_retry_initial_sleep,
    )
    if args.kg_construction_backend == "none":
        kg_config.max_triples = 0
    llm_kg_constructor = (
        LLMKGConstructor(kg_config)
        if args.kg_construction_backend in {"llm", "llm_with_title_bridges"}
        else None
    )

    print("Loading HotpotQA...")
    train_raw_all = []
    for train_file in args.train_file:
        train_raw_all.extend(load_hotpot_file(train_file))
    dev_raw_all = load_hotpot_file(args.dev_file)
    test_files = args.test_file if args.test_file is not None else [args.dev_file]
    test_raw_all = []
    for test_file in test_files:
        test_raw_all.extend(load_hotpot_file(test_file))

    print(f"Raw train examples: {len(train_raw_all)}")
    print(f"Raw dev examples:   {len(dev_raw_all)}")
    print(f"Raw test examples:  {len(test_raw_all)}")

    # Train and dev are sampled from their respective source files. If
    # --test-file is omitted, the official dev file is also used as test.
    train_examples = limit_examples(
        examples=train_raw_all,
        max_examples=args.max_train_examples,
        seed=args.seed,
    )
    dev_examples, test_examples = select_disjoint_cohorts(
        dev_raw_all,
        test_raw_all,
        max_dev_examples=args.max_dev_examples,
        max_test_examples=args.max_test_examples,
        seed=args.seed,
    )

    print(f"Selected train examples: {len(train_examples)}")
    print(f"Selected dev examples:   {len(dev_examples)}")
    print(f"Selected test examples:  {len(test_examples)}")

    print("\nBuilding train rows...")
    train_rows = build_split_rows(
        examples=train_examples,
        split_name="train",
        max_sentences_per_example=args.max_sentences_per_example,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        kg_config=kg_config,
        llm_kg_constructor=llm_kg_constructor,
        kg_construction_workers=max(1, args.kg_construction_workers),
        kg_cache_path=(kg_cache_dir / "train_kg_cache.jsonl")
        if llm_kg_constructor is not None
        else None,
        seed=args.seed,
    )

    print("Building dev rows...")
    dev_rows = build_split_rows(
        examples=dev_examples,
        split_name="dev",
        max_sentences_per_example=args.max_sentences_per_example,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        kg_config=kg_config,
        llm_kg_constructor=llm_kg_constructor,
        kg_construction_workers=max(1, args.kg_construction_workers),
        kg_cache_path=(kg_cache_dir / "dev_kg_cache.jsonl")
        if llm_kg_constructor is not None
        else None,
        seed=args.seed + 100000,
    )

    print("Building test rows...")
    test_rows = build_split_rows(
        examples=test_examples,
        split_name="test",
        max_sentences_per_example=args.max_sentences_per_example,
        max_subgraph_size=args.max_subgraph_size,
        max_candidates_per_question=args.max_candidates_per_question,
        kg_config=kg_config,
        llm_kg_constructor=llm_kg_constructor,
        kg_construction_workers=max(1, args.kg_construction_workers),
        kg_cache_path=(kg_cache_dir / "test_kg_cache.jsonl")
        if llm_kg_constructor is not None
        else None,
        seed=args.seed + 200000,
    )

    summarize_rows("TRAIN", train_rows)
    summarize_rows("DEV", dev_rows)
    summarize_rows("TEST", test_rows)

    write_jsonl(out_dir / "train_subgraph_retrieval.jsonl", train_rows)
    write_jsonl(out_dir / "dev_subgraph_retrieval.jsonl", dev_rows)
    write_jsonl(out_dir / "test_subgraph_retrieval.jsonl", test_rows)

    manifest = split_manifest(
        dataset="HotpotQA",
        source_paths={
            "train": [Path(path) for path in args.train_file],
            "dev": [Path(args.dev_file)],
            "test": [Path(path) for path in test_files],
        },
        split_examples={
            "train": train_examples,
            "dev": dev_examples,
            "test": test_examples,
        },
        seed=args.seed,
    )
    assert_disjoint_split_ids({"train": train_examples, "dev": dev_examples, "test": test_examples})
    write_json(out_dir / "split_manifest.json", manifest)

    metadata = {
        "dataset": "HotpotQA",
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "test_file": args.test_file,
        "effective_test_file": test_files,
        "output_dir": str(out_dir),
        "seed": args.seed,
        "max_train_examples": args.max_train_examples,
        "max_dev_examples": args.max_dev_examples,
        "max_test_examples": args.max_test_examples,
        "max_sentences_per_example": args.max_sentences_per_example,
        "max_subgraph_size": args.max_subgraph_size,
        "max_candidates_per_question": args.max_candidates_per_question,
        "max_kg_bridge_triples": args.max_kg_bridge_triples,
        "kg_construction_backend": kg_backend,
        "kg_construction_model": args.kg_construction_model,
        "kg_max_context_sentences": args.kg_max_context_sentences,
        "kg_construction_workers": args.kg_construction_workers,
        "kg_request_timeout": args.kg_request_timeout,
        "kg_max_retries": args.kg_max_retries,
        "kg_retry_initial_sleep": args.kg_retry_initial_sleep,
        "kg_cache_dir": str(kg_cache_dir) if llm_kg_constructor is not None else None,
        "schema_version": "gold_free_text_retrieval_v1",
        "builder_version": BUILDER_VERSION,
        "git": git_provenance(Path(__file__).resolve().parents[1]),
        "input_contract": {
            "candidate_generation_fields": ["id", "question", "context"],
            "forbidden_candidate_generation_fields": [
                "answer",
                "supporting_facts",
                "evidences",
            ],
        },
        "candidate_generation_config": {
            **frozen_generator_d_config(),
            "candidate_generator": GENERATOR_D_VERSION,
        },
        "candidate_generation_lineage": frozen_generator_d_lineage(),
        "kg_cache_provenance": {
            "signature": kg_config.cache_signature,
            "contaminated_methods_rejected": [
                "provided",
                "provided_plus_llm",
                "llm_with_provided",
            ],
        },
        "gold_available_during_candidate_generation": False,
        "gold_used_during_labeling": any(
            bool(row.get("gold_used_during_labeling")) for row in train_rows + dev_rows
        ),
        "test_gold_attached": False,
        "split_manifest": "split_manifest.json",
        "support_label_fields": ["raw_supporting_facts", "gold_support_units"],
        "candidate_field": "subgraph_units",
        "graph_context_field": "graph_context_units",
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "train_examples": len({r["example_id"] for r in train_rows}),
        "dev_examples": len({r["example_id"] for r in dev_rows}),
        "test_examples": len({r["example_id"] for r in test_rows}),
    }

    write_json(out_dir / "metadata.json", metadata)

    print(f"\nSaved HotpotQA subgraph retrieval data to: {out_dir}")


if __name__ == "__main__":
    main()
