"""
Type-aware graph-based supporting-fact retrieval over the deterministic-first
KG produced by text_kg_constructor.construct_text_kg(backend="deterministic").

Question types are routed to distinct retrieval shapes:
  - compositional:      single anchor, single linear relation chain.
  - bridge_comparison:  two anchors, identical relation chain per branch,
                        joined by an equality comparator.
  - comparison:         two anchors, usually a single hop per branch (needs
                        a type-specific relation-synonym table), joined by
                        an ordering comparator (earlier/later/older/younger).
  - inference:          family-relation chains our typed extractor doesn't
                        cover by design (gender can't be inferred safely
                        from regex) -- routed straight to the fallback
                        ladder instead of attempting typed traversal.

This is a heuristic surface-cue parser, not a full NLU system -- it covers
the closed ~33-relation vocabulary characterized earlier this session, not
open-domain question understanding.
"""

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

from data_processing.text_kg_constructor import normalize_text

Triple = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Question-type classification (lightweight surface heuristic)
# ---------------------------------------------------------------------------

_KINSHIP_TERMS = (
    "grandfather", "grandmother", "grandson", "granddaughter",
    "father-in-law", "mother-in-law", "son-in-law", "daughter-in-law",
    "brother-in-law", "sister-in-law", "nephew", "niece", "great-grandfather",
    "great-grandmother",
)

_DUAL_EQUALITY_CUES = ("same country", "same nationality", "both films", "do both", "both directors")
_DUAL_ORDER_CUES = ("older", "younger", "earlier", "later", "first,", "came out first",
                    "born first", "released first", "more recently", "which film has")


def classify_question_type(question: str) -> str:
    q = question.lower()
    if any(term in q for term in _KINSHIP_TERMS):
        return "inference"
    has_dual_anchor_cue = " or " in q or (" and " in q and "both" in q)
    if has_dual_anchor_cue and any(c in q for c in _DUAL_EQUALITY_CUES):
        return "bridge_comparison"
    if has_dual_anchor_cue and any(c in q for c in _DUAL_ORDER_CUES):
        return "comparison"
    return "compositional"


# ---------------------------------------------------------------------------
# Relation-chain parsing (closed vocabulary; comparison-type synonym table)
# ---------------------------------------------------------------------------

_BRIDGE_RELATION_CUES: List[Tuple[Tuple[str, ...], str]] = [
    (("directed by", "director of film", "director of the film", "who directed",
      "film's director", "the director"), "directed_by"),
    (("composer of", "composed by", "who composed", "the composer"), "composer"),
    (("performer of", "performed by", "who performed", "sung by", "the performer"), "performer"),
    (("producer of", "produced by", "who produced", "the producer"), "produced_by"),
    (("publisher of", "published by", "the publisher"), "publisher"),
    (("editor of", "edited by", "the editor"), "editor"),
    (("creator of", "created by", "the creator"), "created_by"),
    (("founder of", "founded by", "who founded", "the founder"), "founded_by"),
    (("presenter of", "presented by", "hosted by", "the presenter"), "presenter"),
    (("spouse of", "wife of", "husband of", "married", "'s wife", "'s husband"), "spouse_of"),
    # "'s mother"/"'s father" unambiguously name which parent from the
    # QUESTION's own phrasing -- the extractor excludes mother/father
    # entirely because prose like "son of X" can't tell which parent X is
    # without gender info, but that ambiguity doesn't apply here. These
    # never match a typed edge (none is ever extracted) but still let the
    # traversal bridge to the right entity before attempting the next hop,
    # instead of leaving the anchor unchanged and misreading the wrong
    # person's own attribute as the answer.
    (("'s mother", "the mother of", "mother of"), "mother"),
    (("'s father", "the father of", "father of"), "father"),
]

# "directors of films X and Y" (plural, dual-anchor phrasing) vs. "director
# of film X" (singular, single-anchor) -- checked separately from the
# substring cues above since the plural forms don't share a common
# substring with the singular ones.
_DIRECTOR_RE = re.compile(r"directors? of (?:the )?films?\b")

# comparison-type paraphrase gaps confirmed this session: the question rarely
# names these relations directly, it paraphrases the comparison dimension.
_COMPARISON_SYNONYMS: List[Tuple[Tuple[str, ...], str]] = [
    (("established",), "inception"),
    (("came out", "released", "release"), "publication_date"),
    (("scope of profession", "profession", "occupation"), "occupation"),
]


def parse_relation_chain(question: str) -> List[str]:
    q = question.lower()
    chain: List[str] = []

    if _DIRECTOR_RE.search(q):
        chain.append("directed_by")
    else:
        for cues, relation in _BRIDGE_RELATION_CUES:
            if any(c in q for c in cues):
                chain.append(relation)
                break

    if "cause of death" in q:
        chain.append("cause_of_death")
    elif "date of death" in q or (re.search(r"\bwhen\b", q) and ("die" in q or "death" in q)):
        chain.append("death_date")
    elif "date of birth" in q or (re.search(r"\bwhen\b", q) and ("born" in q or "birth" in q)):
        chain.append("birth_date")
    elif "where" in q and "born" in q:
        chain.append("born_in")
    elif "where" in q and ("die" in q or "death" in q):
        chain.append("died_in")
    elif "country" in q and ("originate" in q or "origin" in q):
        chain.append("country_of_origin")
    elif "nationality" in q or ("country" in q and ("citizen" in q or "from" in q)):
        chain.append("country_of_citizenship")
    elif any(c in q for cues, _ in _COMPARISON_SYNONYMS for c in cues):
        for cues, relation in _COMPARISON_SYNONYMS:
            if any(c in q for c in cues):
                chain.append(relation)
                break
    elif "older" in q or "younger" in q or re.search(r"born (first|last|earlier|later)", q):
        chain.append("birth_date")
    elif re.search(r"died (first|last|earlier|later)", q):
        chain.append("death_date")

    return chain


# ---------------------------------------------------------------------------
# Anchor-entity extraction: which real context page titles does the
# question literally name?
# ---------------------------------------------------------------------------

def _normalize_for_anchor(text: str) -> str:
    # Strip apostrophes (not just punctuation) BEFORE stripping other
    # punctuation, so a possessive like "Kerry's" becomes "Kerry s" (two
    # words, preserving the word boundary) instead of "Kerrys" (one word,
    # which would silently break substring anchor matching).
    text = text.replace("'", " ")
    return normalize_text(text)


def extract_anchor_entities(question: str, context_titles: List[str]) -> List[str]:
    q_norm = _normalize_for_anchor(question)
    matches = []
    for title in context_titles:
        title_norm = _normalize_for_anchor(title)
        if len(title_norm) < 3:
            continue
        if f" {title_norm} " in f" {q_norm} ":
            matches.append(title)

    # Drop a match subsumed by a longer match (e.g. "Haute Living" is a
    # literal substring of "Terre Haute Living", a different real page in
    # the same context -- keep only the longer, more specific title).
    matches.sort(key=len, reverse=True)
    kept: List[str] = []
    for m in matches:
        m_norm = _normalize_for_anchor(m)
        if any(f" {m_norm} " in f" {_normalize_for_anchor(k)} " for k in kept):
            continue
        kept.append(m)
    return kept


# ---------------------------------------------------------------------------
# Typed-edge traversal + fallback ladder
# ---------------------------------------------------------------------------

def _strip_disambiguator(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(title)).strip()


def _index_by_subject(kg_triples: List[Triple]) -> Tuple[Dict[str, List[Triple]], Dict[str, List[Triple]]]:
    """Two indices: exact normalized subject, and disambiguator-stripped
    normalized subject. A typed-pattern hop often resolves to the bare
    name used in prose ("David Howard"), while the KG node carrying that
    person's own facts is keyed by the disambiguated page title ("David
    Howard (director)") -- confirmed as the dominant real bridge-mismatch
    class earlier this session. Stripping must happen on the RAW subject
    before normalize_text, since normalize_text already deletes the
    parentheses that mark the disambiguator.
    """
    idx: Dict[str, List[Triple]] = {}
    stripped_idx: Dict[str, List[Triple]] = {}
    for s, r, o in kg_triples:
        idx.setdefault(normalize_text(s), []).append((s, r, o))
        stripped_idx.setdefault(normalize_text(_strip_disambiguator(s)), []).append((s, r, o))
    return idx, stripped_idx


def _lookup_edges(
    idx: Dict[str, List[Triple]],
    stripped_idx: Dict[str, List[Triple]],
    frontier: str,
) -> List[Triple]:
    key = normalize_text(frontier)
    if key in idx:
        return idx[key]
    return stripped_idx.get(normalize_text(_strip_disambiguator(frontier)), [])


def traverse_chain(
    kg_triples: List[Triple],
    anchor: str,
    relation_chain: List[str],
) -> Dict[str, Any]:
    """Walk relation_chain from anchor over typed KG edges. Stops at the
    first hop with no matching typed edge -- the caller applies the
    fallback ladder from there.
    """
    idx, stripped_idx = _index_by_subject(kg_triples)
    evidence: List[Triple] = []
    frontier = anchor

    for hop_idx, relation in enumerate(relation_chain):
        edges = _lookup_edges(idx, stripped_idx, frontier)
        match = next((e for e in edges if e[1] == relation), None)
        if match is None:
            return {
                "value": frontier,
                "evidence": evidence,
                "status": "typed_miss",
                "missing_relation": relation,
                "missing_hop_index": hop_idx,
            }
        evidence.append(match)
        frontier = match[2]

    return {"value": frontier, "evidence": evidence, "status": "ok", "missing_relation": None}


def apply_fallback(
    kg_triples: List[Triple],
    sentence_records: List[Dict[str, Any]],
    frontier: str,
) -> Dict[str, Any]:
    """Tier 3a: generic mentions_page bridge (connectivity without a typed
    label). Tier 3b: the frontier node's own source sentence(s) as the last
    resort when no graph edge -- typed or generic -- exists at all.
    """
    idx, stripped_idx = _index_by_subject(kg_triples)
    edges = _lookup_edges(idx, stripped_idx, frontier)
    bridge = next((e for e in edges if e[1] == "mentions_page"), None)
    if bridge is not None:
        return {"tier": "bridge_fallback", "value": bridge[2], "evidence": [bridge]}

    frontier_key = normalize_text(frontier)
    frontier_stripped_key = normalize_text(_strip_disambiguator(frontier))
    sentences = [
        r for r in sentence_records
        if normalize_text(r.get("title", "")) == frontier_key
        or normalize_text(_strip_disambiguator(r.get("title", ""))) == frontier_stripped_key
    ]
    return {"tier": "sentence_fallback", "value": frontier, "evidence": sentences}


def resolve_with_fallback(
    kg_triples: List[Triple],
    sentence_records: List[Dict[str, Any]],
    anchor: str,
    relation_chain: List[str],
) -> Dict[str, Any]:
    """Walk relation_chain hop by hop, applying the fallback ladder to
    whichever individual hop misses its typed edge and then CONTINUING the
    remaining hops from the recovered frontier -- rather than treating the
    fallback as a one-shot rescue applied only after the whole chain has
    already failed. A chain where hop 1 misses but recovers via a bridge
    edge still needs hop 2 attempted on the new frontier; the naive
    single-shot version returns silently wrong values on multi-hop chains
    whenever the miss isn't on the final hop.
    """
    idx, stripped_idx = _index_by_subject(kg_triples)
    evidence: List[Triple] = []
    fallback_steps: List[Dict[str, Any]] = []
    frontier = anchor

    for relation in relation_chain:
        edges = _lookup_edges(idx, stripped_idx, frontier)
        match = next((e for e in edges if e[1] == relation), None)
        if match is not None:
            evidence.append(match)
            frontier = match[2]
            continue

        fb = apply_fallback(kg_triples, sentence_records, frontier)
        fallback_steps.append(fb)
        if fb["tier"] == "bridge_fallback":
            evidence.append(fb["evidence"][0])
        frontier = fb["value"]
        if fb["tier"] == "sentence_fallback":
            break  # no further node to hop from once we hit sentence text

    return {
        "value": frontier,
        "evidence": evidence,
        "status": "ok" if not fallback_steps else "partial_fallback",
        "fallback_steps": fallback_steps,
        "fallback": fallback_steps[-1] if fallback_steps else None,
    }


# ---------------------------------------------------------------------------
# Date parsing for ordering comparators (birth_date / death_date /
# inception / publication_date all carry mixed "Month Day, Year" / bare
# "Year" formats).
# ---------------------------------------------------------------------------

_DATE_FORMATS = ["%B %d, %Y", "%B %d,%Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y"]


def parse_date_key(value: str) -> Optional[Tuple[int, int, int]]:
    value = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.datetime.strptime(value, fmt)
            return (dt.year, dt.month, dt.day)
        except ValueError:
            continue
    match = re.search(r"\d{4}", value)
    if match:
        return (int(match.group()), 1, 1)
    return None


# ---------------------------------------------------------------------------
# Per-type retrieval entry points
# ---------------------------------------------------------------------------

def retrieve_compositional(
    question: str,
    kg_triples: List[Triple],
    sentence_records: List[Dict[str, Any]],
    context_titles: List[str],
) -> Dict[str, Any]:
    chain = parse_relation_chain(question)
    anchors = extract_anchor_entities(question, context_titles)
    anchor = anchors[0] if anchors else None
    if anchor is None:
        return {"status": "no_anchor", "chain": chain}

    result = resolve_with_fallback(kg_triples, sentence_records, anchor, chain)
    return {
        "type": "compositional",
        "chain": chain,
        "anchor": anchor,
        "answer": result["value"],
        "evidence": result["evidence"],
        "status": result["status"],
        "fallback": result.get("fallback"),
    }


def _retrieve_dual_branch(
    question: str,
    kg_triples: List[Triple],
    sentence_records: List[Dict[str, Any]],
    context_titles: List[str],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    chain = parse_relation_chain(question)
    anchors = extract_anchor_entities(question, context_titles)
    branch_results = [
        resolve_with_fallback(kg_triples, sentence_records, anchor, chain) for anchor in anchors
    ]
    return chain, anchors, branch_results


_ORDERING_CUE_RE = re.compile(
    r"\b(first|last|earlier|later|older|younger|more recently)\b"
)
_LATEST_WINS_CUE_RE = re.compile(r"\b(last|later|more recently|younger)\b")


def _apply_dual_comparator(
    question: str,
    anchors: List[str],
    branches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Comparator style is a property of the QUESTION's phrasing, not of
    which question type routed here: 'bridge_comparison' questions can ask
    either "same country" (equality) or "director died first" (ordering),
    and vice versa for 'comparison' -- confirmed by a real example
    (Ghost Fever / Faceless) that is bridge_comparison but ordering-style.
    Detect the comparator from the question text itself rather than
    hardcoding it per calling function.
    """
    q = question.lower()
    if _ORDERING_CUE_RE.search(q):
        wants_earliest = not _LATEST_WINS_CUE_RE.search(q)
        keyed = [(parse_date_key(b["value"]), anchor) for anchor, b in zip(anchors, branches)]
        comparable = [(k, a) for k, a in keyed if k is not None]
        # A "winner" requires an actual comparison: if only one (or zero)
        # branch resolved to a real date, there's nothing to compare against
        # -- returning the sole value as the "winner" would be a confident
        # answer built on a missing fact, not a real comparison result.
        if len(comparable) < 2:
            return {"status": "insufficient_comparable_values",
                    "branch_values": [b["value"] for b in branches]}
        comparable.sort(key=lambda item: item[0], reverse=not wants_earliest)
        return {"answer": comparable[0][1], "comparator": "ordering"}

    values = [normalize_text(b["value"]) for b in branches]
    same = len(set(values)) == 1
    return {"answer": "yes" if same else "no", "comparator": "equality"}


def retrieve_bridge_comparison(
    question: str,
    kg_triples: List[Triple],
    sentence_records: List[Dict[str, Any]],
    context_titles: List[str],
) -> Dict[str, Any]:
    chain, anchors, branches = _retrieve_dual_branch(question, kg_triples, sentence_records, context_titles)
    if len(anchors) < 2:
        return {"status": "insufficient_anchors", "chain": chain, "anchors": anchors}

    comparator_result = _apply_dual_comparator(question, anchors, branches)
    return {
        "type": "bridge_comparison",
        "chain": chain,
        "anchors": anchors,
        "branch_values": [b["value"] for b in branches],
        "branch_evidence": [b["evidence"] for b in branches],
        **comparator_result,
    }


def retrieve_comparison(
    question: str,
    kg_triples: List[Triple],
    sentence_records: List[Dict[str, Any]],
    context_titles: List[str],
) -> Dict[str, Any]:
    chain, anchors, branches = _retrieve_dual_branch(question, kg_triples, sentence_records, context_titles)
    if len(anchors) < 2:
        return {"status": "insufficient_anchors", "chain": chain, "anchors": anchors}

    comparator_result = _apply_dual_comparator(question, anchors, branches)
    return {
        "type": "comparison",
        "chain": chain,
        "anchors": anchors,
        "branch_values": [b["value"] for b in branches],
        "branch_evidence": [b["evidence"] for b in branches],
        **comparator_result,
    }


def retrieve_inference(
    question: str,
    kg_triples: List[Triple],
    sentence_records: List[Dict[str, Any]],
    context_titles: List[str],
) -> Dict[str, Any]:
    """Family-relation chains aren't in the typed vocabulary by design --
    skip straight to the fallback ladder rather than wasting a typed
    lookup attempt we already know will miss.
    """
    anchors = extract_anchor_entities(question, context_titles)
    anchor = anchors[0] if anchors else None
    if anchor is None:
        return {"status": "no_anchor"}

    hops: List[Dict[str, Any]] = []
    frontier = anchor
    for _ in range(2):  # grandparent/mother-in-law style chains are 2 hops
        fb = apply_fallback(kg_triples, sentence_records, frontier)
        hops.append(fb)
        frontier = fb["value"]
        if fb["tier"] == "sentence_fallback":
            break  # no further node to hop from once we hit sentence text

    return {
        "type": "inference",
        "anchor": anchor,
        "hops": hops,
        "answer_hint": frontier,
    }
